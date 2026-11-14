"""Few-shot personalisation across all 12 OhioT1DM subjects.

    python scripts/run_ohio_personalization_sweep.py --root OhioT1DM
    python scripts/run_ohio_personalization_sweep.py --smoke

Trains one population checkpoint on `split.fit` (cached to disk, so the sweep
can be re-run without retraining), then adapts it independently to every subject
under the causal prefix protocol:

    |<- adapt prefix (first K h of TRAIN) ->|<- embargo ->|<- evaluate ->|
                                                          rest of TRAIN + TEST

WHY THE COMPARISON IS PAIRED, AND WHAT THE STATISTICS ARE FOR
--------------------------------------------------------------
Every arm's "before" is the population model scored on the SAME evaluation
windows as its "after". That pairing is what licenses a causal claim about
adaptation; comparing a personalised model against a population model scored on
a different slice would compare two numbers, not two methods.

With 12 subjects, a mean improvement is easy to produce by luck, so each mode
gets a paired t-test AND a Wilcoxon signed-rank test on the per-subject deltas
(n = 12 is small enough that the normality assumption behind the t-test is worth
not relying on alone), plus Holm-Bonferroni adjustment across the modes tested
at each prefix length -- because testing five modes and reporting the best raw
p-value is how a null result becomes a finding.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B  # noqa: E402
from causalcgm_mt.benchmark import ArmSpec, Column  # noqa: E402
from causalcgm_mt.extension import run_ohio_personalization  # noqa: E402
from causalcgm_mt.models import criterion_from_state  # noqa: E402
from scripts._cli import OHIO_HINT, missing_input  # noqa: E402


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_stats(before: list[float], after: list[float], better: str = "min") -> dict:
    """Paired t-test + Wilcoxon on per-subject deltas, with an effect size.

    `better="min"` for RMSE-like metrics (a negative delta is an improvement),
    `"max"` for PR-AUC-like ones. `improved` normalises the sign so a reader
    does not have to remember which way each metric runs.

    Subjects with a non-finite value on either side are dropped pairwise and
    counted, rather than propagating a nan through the whole test.
    """
    from scipy import stats

    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    n_dropped = int((~ok).sum())
    b, a = b[ok], a[ok]
    d = a - b
    out = {"n": int(d.size), "n_dropped": n_dropped,
           "mean_before": float(b.mean()) if d.size else float("nan"),
           "mean_after": float(a.mean()) if d.size else float("nan"),
           "mean_delta": float(d.mean()) if d.size else float("nan"),
           "sd_delta": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
           "n_improved": int(((d < 0) if better == "min" else (d > 0)).sum()),
           "t": float("nan"), "p_t": float("nan"),
           "wilcoxon_p": float("nan"), "cohens_dz": float("nan"),
           "ci95_low": float("nan"), "ci95_high": float("nan")}
    if d.size < 2 or float(np.std(d, ddof=1)) == 0.0:
        # No between-subject variance means there is no paired test to run. Two
        # cases land here and they are not the same thing, so they are named:
        # the zero-shot control (every delta exactly zero), and the pathological
        # case where every subject moved by an identical amount -- for which
        # scipy would emit a catastrophic-cancellation warning and a p-value of
        # 0 that means nothing.
        if d.size < 2:
            out["note"] = "too few subjects for a paired test"
        elif np.allclose(d, 0.0):
            out["note"] = "identical before/after (zero-shot control)"
        else:
            out["note"] = ("delta is constant across subjects; a paired test is "
                           "degenerate")
        return out

    t, p = stats.ttest_rel(a, b)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(d.size)
    crit = float(stats.t.ppf(0.975, d.size - 1))
    out.update({
        "t": float(t), "p_t": float(p),
        "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "ci95_low": float(d.mean() - crit * se),
        "ci95_high": float(d.mean() + crit * se),
    })
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError as exc:            # all-zero differences
        out["wilcoxon_note"] = str(exc)
    return out


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjustment. Controls FWER without assuming independence.

    Applied across the MODES compared at one prefix length, which is the family
    of hypotheses actually being tested together. Reported alongside the raw
    p-values rather than replacing them.
    """
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, v in pvalues.items():
        adj.setdefault(k, float("nan"))
    return adj


# --------------------------------------------------------------------------- #
# Base checkpoint
# --------------------------------------------------------------------------- #
def base_checkpoint(data: B.CohortData, args) -> tuple[B.TrainedArm, ArmSpec]:
    """Train the population model once, or reload it from `--out`."""
    import torch

    from causalcgm_mt.models import build_multimodal_model, load_state_dict_compat
    from causalcgm_mt.ohio_loader import aux_validity_indices
    from causalcgm_mt.train import pick_device

    def _spec_matches(saved: dict | None, want: ArmSpec) -> bool:
        """Does a stored spec describe THIS arm?

        Exact dict equality would treat a checkpoint as stale whenever a new
        `ArmSpec` field is added, even one whose default is "inherit" and which
        therefore describes the same arm the checkpoint was trained as -- an
        hour of retraining bought by a field that changed nothing. A key absent
        from the stored spec is compared against the current DEFAULT, so a field
        that is genuinely set to something else still misses.
        """
        if not isinstance(saved, dict):
            return False
        current, default = want.as_dict(), ArmSpec(want.name, want.family).as_dict()
        for key, value in current.items():
            if saved.get(key, default.get(key)) != value:
                return False
        return not (set(saved) - set(current))

    spec = ArmSpec("population base", "tcn_mm", horizons=tuple(args.horizons),
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   patience=args.patience, seed=args.seed,
                   fit_subsample=args.fit_subsample)
    path = os.path.join(args.out, "base_model.pt")

    if os.path.exists(path) and not args.retrain:
        ck = torch.load(path, weights_only=False, map_location="cpu")
        # The spec alone is not enough to decide a checkpoint is reusable: it
        # describes the ARM, not the data. A change to the windowing pipeline
        # (the CGM snap, an auxiliary mask) leaves every spec field identical
        # while changing what the model was trained on, so the version has to be
        # part of the key or the sweep silently adapts a model fitted to
        # pre-fix windows.
        if ck.get("pipeline_version") != B.WINDOW_PIPELINE_VERSION:
            print(f"{path} was trained by pipeline version "
                  f"{ck.get('pipeline_version', '<pre-versioning>')}, this is "
                  f"{B.WINDOW_PIPELINE_VERSION}; retraining")
        elif not ck.get("criterion_state_dict"):
            # The Kendall log-variances are part of the population solution, and
            # this checkpoint predates their being saved. Rebuilding the
            # criterion would reset them to a uniform task balance and
            # pos_weight to 1.0, so a RESUMED sweep would adapt under a
            # different loss than a fresh one -- the same command producing two
            # different tables depending on whether a cache file happened to
            # exist. Retrain rather than resume; see
            # `models.criterion_from_state`.
            print(f"{path} carries no criterion state (written before the task "
                  f"weights were persisted); retraining, because adapting "
                  f"under a uniform task balance is a different experiment")
        elif _spec_matches(ck.get("spec"), spec):
            model, _fresh = build_multimodal_model(
                ck["in_channels"], ck["aux_channels"],
                B._arm_config(spec, data).model, ck["pos_weight"],
                aux_valid_index=ck.get("aux_valid_index",
                                       aux_validity_indices(spec.modalities)))
            load_state_dict_compat(model, ck["state_dict"])
            # The criterion the model was TRAINED with -- its learned task
            # weights and its fold class prior -- not a fresh one.
            criterion = criterion_from_state(
                ck["criterion_state_dict"], B._arm_config(spec, data).model,
                context=path)
            device = pick_device()
            print(f"loaded population checkpoint from {path} "
                  f"(best {spec.monitor} = {ck.get('best_monitor', float('nan')):.4f}, "
                  f"task weights restored)")
            return B.TrainedArm(model=model.to(device).eval(),
                                criterion=criterion.to(device), multimodal=True,
                                epochs_run=ck.get("epochs_run", 0),
                                best_epoch=ck.get("best_epoch", -1),
                                best_monitor=ck.get("best_monitor", float("nan"))), spec
        print(f"{path} was trained with a different spec; retraining")

    print("=== training the population base model ===")
    trained = B.train_neural_arm(data, spec)
    os.makedirs(args.out, exist_ok=True)
    prev = float(data.ws.y_cls_multi[data.split.fit].mean())
    torch.save({"state_dict": {k: v.cpu() for k, v in trained.model.state_dict().items()},
                # The multi-task criterion is FITTED STATE, not a hyperparameter:
                # `log_var_reg`/`log_var_cls` are the learned Kendall weights and
                # `pos_weight` is this fit split's class prior. Saving the model
                # without them made a resumed sweep adapt under a uniform task
                # balance and an unweighted BCE, so the same command produced
                # different numbers depending on whether the cache existed.
                "criterion_state_dict": {k: v.detach().cpu() for k, v
                                         in trained.criterion.state_dict().items()},
                "spec": spec.as_dict(),
                "pipeline_version": B.WINDOW_PIPELINE_VERSION,
                "in_channels": int(data.ws.x.shape[-1]),
                "aux_channels": int(data.ws.x_aux.shape[-1]),
                # Not part of state_dict, and rebuilding with the wrong set
                # would silently mask the fusion branch on reload.
                "aux_valid_index": list(aux_validity_indices(spec.modalities)),
                "pos_weight": float((1 - prev) / max(prev, 1e-6)),
                "best_monitor": trained.best_monitor,
                "best_epoch": trained.best_epoch,
                "epochs_run": trained.epochs_run}, path)
    print(f"wrote {path}")
    return trained, spec


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #
METRICS = (
    ("rmse", "min", "RMSE"),
    ("pr_auc", "max", "PR-AUC"),
    ("sensitivity_at_5pct_fpr", "max", "Se@5\\%"),
    ("fapd_at_5pct_fpr", "min", "FAPD@5\\%"),
    ("mean_lead_time_min", "max", "Lead (min)"),
)


PREFIX_MODE_DOC = {
    "opening": "first K h of the training file (the prefix-length and "
               "step-budget axes always use this)",
    "recent": "last K h of the training file that clear the embargo, i.e. "
              "ending exactly one embargo before the first scored window",
    "note": "the contrast holds subjects, evaluation windows, adapter, step "
            "budget and prefix LENGTH fixed; only the prefix's distance from "
            "the scored windows changes",
}


def mode_label(mode: str, rank: int | None) -> str:
    return f"lora (r={rank})" if mode == "lora" else mode


def run_mode(base: B.TrainedArm, data: B.CohortData, cfg, mode: str,
             rank: int | None, prefix_hours: float, args,
             prefix_mode: str = "opening") -> list:
    return run_ohio_personalization(
        base.model, data.ws, data.split, data.pre, cfg,
        criterion=base.criterion, method=mode, prefix_hours=prefix_hours,
        lora_rank=rank, steps=args.adapt_steps, lr=args.adapt_lr,
        min_eval_windows=args.min_eval_windows, eval_scope=args.eval_scope,
        prefix_mode=prefix_mode, verbose=args.verbose,
    )


def result_rows(res: list) -> list[dict]:
    """Serialise per-subject results. One shape, so every block matches."""
    return [
        {"patient": r.patient, "method": r.method,
         "prefix_mode": r.prefix_mode, "prefix_gap_min": r.prefix_gap_min,
         "n_adapt_windows": r.n_adapt_windows,
         "n_adapt_in_fit": r.n_adapt_in_fit, "n_adapt_in_val": r.n_adapt_in_val,
         "n_eval_windows": r.n_eval_windows,
         "n_eval_from_train": r.n_eval_from_train,
         "n_eval_from_test": r.n_eval_from_test,
         "eval_scope": r.eval_scope,
         "n_eval_in_fit": r.n_eval_in_fit,
         "n_eval_in_val": r.n_eval_in_val,
         "n_trainable": r.n_trainable,
         "prefix_hours": r.prefix_hours, "embargo_min": r.embargo_min,
         "before": {str(h): rep.as_dict() for h, rep in r.before_by_horizon.items()},
         "after": {str(h): rep.as_dict() for h, rep in r.after_by_horizon.items()}}
        for r in res
    ]


def pair_on_patient(a: list[dict], b: list[dict], metric: str, primary: int,
                    key: str) -> tuple[list[float], list[float], list[str]]:
    """Align two result blocks on subject id before comparing them.

    Two runs can drop different subjects -- `adapt_ohio_subject` returns None
    when a prefix is too short, and a 24 h recent prefix can be shorter than the
    opening one for a subject whose record ends abruptly. Zipping the two lists
    positionally would then silently pair subject i of one with subject j of the
    other, which is a comparison of two different cohorts wearing one table row.
    """
    ia = {r["patient"]: r for r in a}
    ib = {r["patient"]: r for r in b}
    shared = sorted(set(ia) & set(ib))
    return ([ia[p][key][str(primary)][metric] for p in shared],
            [ib[p][key][str(primary)][metric] for p in shared],
            shared)


def backfill_geometry(rows: list[dict], data: B.CohortData, prefix_hours: float,
                      prefix_mode: str, eval_scope: str) -> list[dict]:
    """Add the prefix-position fields to rows serialised before they existed.

    Rows reused from a finished sweep predate `prefix_mode` / `prefix_gap_min`,
    and the gap is a property of the SPLIT, not of the model -- so it is
    recovered from the geometry rather than by re-running an adaptation. Rows
    that already carry the fields are returned untouched.
    """
    from causalcgm_mt.ohio_loader import ohio_prefix_embargo_split

    out = []
    for r in rows:
        if np.isfinite(r.get("prefix_gap_min", float("nan"))):
            out.append(r)
            continue
        ps = ohio_prefix_embargo_split(
            data.ws, data.split, r["patient"], data.cfg,
            prefix_hours=float(r.get("prefix_hours", prefix_hours)),
            eval_scope=eval_scope, prefix_mode=prefix_mode)
        r = dict(r)
        r["prefix_mode"] = prefix_mode
        r["prefix_gap_min"] = float("nan") if ps is None else ps.prefix_gap_min
        r["n_adapt_in_fit"] = 0 if ps is None else ps.n_adapt_in_fit
        r["n_adapt_in_val"] = 0 if ps is None else ps.n_adapt_in_val
        out.append(r)
    return out


def prefix_position_contrast(base, data, cfg, jobs, args, primary: int, hp: str,
                             sweep: dict) -> dict:
    """Opening vs recent adaptation prefix at the headline prefix length.

    The drift control. Section 5.7's negative result is explained by the opening
    prefix being a stale sample of a subject whose dynamics have moved by the
    time the test file starts. That explanation makes a prediction: a prefix
    taken from the evaluation window's own vicinity should not fail the same
    way. Everything else -- subjects, evaluation windows, adapter, step budget,
    prefix LENGTH -- is held fixed, so the only variable is how far the prefix
    sits from what is being scored.
    """
    contrast: dict = {}
    if "recent" in args.prefix_modes:
        print(f"\n{'=' * 72}\nPREFIX POSITION at prefix = {args.headline_prefix:g} h"
              f"\n{'=' * 72}")
        for mode, rank, label in jobs:
            contrast[label] = {}
            for pm in args.prefix_modes:
                # The opening arm at the headline prefix is exactly the run the
                # main sweep already did. Recomputing it would burn half the
                # budget of this block to reproduce numbers that must agree --
                # and if they ever did not, the bug would be invisible.
                if pm == "opening" and hp in sweep and label in sweep[hp]:
                    print(f"\n--- {label} @ {pm} (reused from the prefix sweep) ---")
                    rows = backfill_geometry(sweep[hp][label], data,
                                             args.headline_prefix, pm,
                                             args.eval_scope)
                else:
                    print(f"\n--- {label} @ {pm} ---")
                    rows = result_rows(run_mode(base, data, cfg, mode, rank,
                                                args.headline_prefix, args,
                                                prefix_mode=pm))
                stats = {}
                for metric, better, _l in METRICS:
                    b = [r["before"][str(primary)][metric] for r in rows]
                    a = [r["after"][str(primary)][metric] for r in rows]
                    stats[metric] = paired_stats(b, a, better)
                gaps = [r["prefix_gap_min"] for r in rows
                        if np.isfinite(r.get("prefix_gap_min", float("nan")))]
                contrast[label][pm] = {
                    "rows": rows,
                    "stats": stats,
                    "n_subjects": len(rows),
                    # The variable under test, reported rather than asserted.
                    "median_gap_min": float(np.median(gaps)) if gaps else float("nan"),
                    "median_adapt_windows": (
                        float(np.median([r["n_adapt_windows"] for r in rows]))
                        if rows else float("nan")),
                }

            # Head-to-head. Both arms share a `before` (the zero-shot model on
            # identical windows), so comparing the two `after` values IS
            # comparing the two adaptation deltas, paired within subject.
            if "opening" in contrast[label] and "recent" in contrast[label]:
                head = {}
                for metric, better, _l in METRICS:
                    o, r_, shared = pair_on_patient(
                        contrast[label]["opening"]["rows"],
                        contrast[label]["recent"]["rows"], metric, primary, "after")
                    st = paired_stats(o, r_, better)
                    st["n_paired_subjects"] = len(shared)
                    head[metric] = st
                contrast[label]["recent_vs_opening"] = head

        # Holm across modes, within each prefix position and each metric --
        # the same family structure the prefix-length axis uses.
        for pm in args.prefix_modes:
            for metric, _b, _l in METRICS:
                raw = {label: contrast[label][pm]["stats"][metric]["p_t"]
                       for label in contrast if pm in contrast[label]}
                adj = holm_bonferroni(raw)
                for label in raw:
                    contrast[label][pm]["stats"][metric]["p_t_holm"] = adj.get(
                        label, float("nan"))
        for metric, _b, _l in METRICS:
            raw = {label: contrast[label]["recent_vs_opening"][metric]["p_t"]
                   for label in contrast if "recent_vs_opening" in contrast[label]}
            adj = holm_bonferroni(raw)
            for label in raw:
                contrast[label]["recent_vs_opening"][metric]["p_t_holm"] = adj.get(
                    label, float("nan"))

    return contrast


def report_contrast(contrast: dict, args, primary: int) -> None:
    """Console summary of the prefix-position axis.

    Shared by the full run and `--contrast-only`, so the two cannot drift
    into printing the same experiment two different ways.
    """
    if not contrast:
        return
    print(f"\n{'=' * 104}\nPREFIX POSITION at {args.headline_prefix:g} h, "
          f"PH = {primary} min: opening vs recent\n{'=' * 104}")
    hdr = (f"{'mode':<16}{'metric':<22}{'open D':>10}{'recent D':>10}"
           f"{'rec-open':>10}{'p Holm':>10}{'n':>4}   "
           f"{'median gap (open / recent)':<28}")
    print(hdr + "\n" + "-" * len(hdr))
    for label in contrast:
        hh = contrast[label].get("recent_vs_opening", {})
        g_o = contrast[label].get("opening", {}).get("median_gap_min", float("nan"))
        g_r = contrast[label].get("recent", {}).get("median_gap_min", float("nan"))
        gap = f"{g_o / 1440:.1f} d / {g_r:.0f} min"
        for metric, _b, _l in METRICS[:3]:
            d_o = contrast[label]["opening"]["stats"][metric]["mean_delta"]
            d_r = contrast[label]["recent"]["stats"][metric]["mean_delta"]
            st = hh.get(metric, {})
            fmt = lambda v: ("--" if not np.isfinite(v) else f"{v:+.3f}")  # noqa: E731
            pf = ("--" if not np.isfinite(st.get("p_t_holm", float("nan")))
                  else f"{st['p_t_holm']:.4f}")
            print(f"{label:<16}{metric:<22}{fmt(d_o):>10}{fmt(d_r):>10}"
                  f"{fmt(st.get('mean_delta', float('nan'))):>10}{pf:>10}"
                  f"{st.get('n_paired_subjects', 0):>4}   {gap:<28}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="OhioT1DM")
    ap.add_argument("--out", default="runs/paper_results/personalization")
    ap.add_argument("--cache", default="runs/paper_results/ohio_ws.pkl")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60])
    ap.add_argument("--history", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--fit-subsample", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--modes", nargs="+",
                    default=["none", "bn_adapt", "lora", "head_only", "full"])
    ap.add_argument("--lora-ranks", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--prefix-hours", nargs="+", type=float,
                    default=[6.0, 12.0, 24.0, 48.0])
    ap.add_argument("--headline-prefix", type=float, default=24.0)
    ap.add_argument("--prefix-modes", nargs="*", default=["opening", "recent"],
                    choices=["opening", "recent"],
                    help="where the adaptation prefix is drawn from, run as a "
                         "contrast at the headline prefix. 'opening' is the "
                         "first K hours of the training file; 'recent' is the "
                         "last K hours that clear the embargo, i.e. immediately "
                         "before the scored windows. The contrast is the "
                         "control for the within-subject drift explanation of a "
                         "negative personalisation result: if the opening "
                         "prefix fails because dynamics move over weeks, a "
                         "prefix from the evaluation window's own vicinity "
                         "should not fail the same way. The prefix-length and "
                         "step-budget axes always use 'opening'.")
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--adapt-lr", type=float, default=5e-4)
    ap.add_argument("--step-budgets", nargs="*", type=int,
                    default=[10, 25, 50, 100, 200],
                    help="extra axis at the headline prefix: how many gradient "
                         "steps the adaptation gets. A fixed budget with no "
                         "per-subject early stopping is the most likely reason "
                         "for a personalised arm to LOSE to zero-shot, so it is "
                         "measured rather than assumed. Pass none to skip.")
    ap.add_argument("--step-budget-modes", nargs="*",
                    default=["lora", "head_only"],
                    help="which modes to run the step-budget axis for")
    ap.add_argument("--min-eval-windows", type=int, default=200)
    ap.add_argument("--eval-scope", choices=["test", "post_embargo"], default="test",
                    help="which rows the paired before/after is scored on. "
                         "'test' (default) is the official testing windows "
                         "only, which is the only set disjoint from the "
                         "population model's own fit and val rows. "
                         "'post_embargo' is the legacy training-tail + test "
                         "region, in which ~80%% of rows are in-sample for the "
                         "base model -- kept only to reproduce the published "
                         "numbers.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--contrast-only", action="store_true",
                    help="run ONLY the prefix-position contrast and merge it "
                         "into the existing personalization_sweep.json, leaving "
                         "every other block byte-identical. The opening arm is "
                         "read from the stored sweep rather than recomputed, so "
                         "this adds an axis to a finished run without putting "
                         "the published numbers back through a fresh fit.")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.patience, args.fit_subsample = 2, 1, 16
        args.adapt_steps = 8
        args.modes = ["none", "bn_adapt", "lora"]
        args.lora_ranks = [4]
        args.prefix_hours = [12.0, 24.0]
        args.headline_prefix = 24.0
        args.step_budgets = [4, 8]
        args.step_budget_modes = ["lora"]
        args.prefix_modes = ["opening", "recent"]

    with missing_input(OHIO_HINT):
        data = B.load_cohort(args.root, history_min=args.history,
                             horizons=tuple(args.horizons), val_frac=args.val_frac,
                             cache=args.cache)
    print(data.summary())
    base, base_spec = base_checkpoint(data, args)

    from causalcgm_mt.config import AdaptConfig
    cfg = dataclasses.replace(
        B._arm_config(base_spec, data),
        adapt=AdaptConfig(adapt_prefix_hours=args.headline_prefix,
                          steps=args.adapt_steps, lr=args.adapt_lr),
    )

    jobs = []
    for mode in args.modes:
        ranks = args.lora_ranks if mode == "lora" else [None]
        for r in ranks:
            jobs.append((mode, r, mode_label(mode, r)))

    primary = int(args.horizons[0])
    # Key into `sweep` for the headline prefix. Defined here because the
    # prefix-position block below reuses the headline opening run.
    hp = str(float(args.headline_prefix))

    if args.contrast_only:
        out_path = os.path.join(args.out, "personalization_sweep.json")
        if not os.path.exists(out_path):
            raise SystemExit(
                f"--contrast-only needs a finished sweep to merge into, and "
                f"{out_path} does not exist. Run the full sweep first.")
        import json
        with open(out_path) as f:
            payload = json.load(f)
        if payload.get("smoke") and not args.smoke:
            raise SystemExit(
                f"{out_path} is a SMOKE run. Merging a publication-budget "
                f"contrast into it would produce a file whose blocks came from "
                f"different budgets and whose `smoke` flag describes only some "
                f"of them. Re-run the full sweep first.")
        stored = payload.get("sweep", {})
        if hp not in stored:
            raise SystemExit(
                f"the stored sweep has no {hp} h prefix to reuse as the opening "
                f"arm (it has {sorted(stored)}); pass a --headline-prefix it does.")
        # Only the labels the stored run actually carries: a mode added to
        # --modes since that run has no opening arm to be contrasted against,
        # and inventing one by running it now would compare a fresh fit against
        # a months-old one under the same table heading.
        missing = [lbl for _m, _r, lbl in jobs if lbl not in stored[hp]]
        if missing:
            print(f"note: {missing} absent from the stored sweep; skipping them "
                  f"rather than recomputing an opening arm out of band")
        jobs = [j for j in jobs if j[2] in stored[hp]]

        contrast = prefix_position_contrast(base, data, cfg, jobs, args,
                                            primary, hp, stored)
        payload["prefix_mode_contrast"] = contrast
        payload.setdefault("protocol", {})["prefix_modes"] = PREFIX_MODE_DOC
        payload.setdefault("adapt", {})["prefix_modes"] = args.prefix_modes
        B.write_json(out_path, payload)
        print(f"\nmerged prefix_mode_contrast into {out_path}")
        report_contrast(contrast, args, primary)
        return

    sweep: dict = {}
    for prefix in args.prefix_hours:
        print(f"\n{'=' * 72}\nPREFIX = {prefix:g} h\n{'=' * 72}")
        sweep[str(prefix)] = {}
        for mode, rank, label in jobs:
            print(f"\n--- {label} @ {prefix:g} h ---")
            # The prefix-length axis is always the opening prefix: it asks "how
            # much history does adaptation need", which is only a coherent
            # question when the prefix's POSITION is held fixed. Position is the
            # separate axis below.
            res = run_mode(base, data, cfg, mode, rank, prefix, args,
                           prefix_mode="opening")
            sweep[str(prefix)][label] = result_rows(res)

    # ---- significance -------------------------------------------------------
    significance: dict = {}
    for prefix, by_mode in sweep.items():
        significance[prefix] = {}
        for metric, better, _lbl in METRICS:
            raw = {}
            for label, rows in by_mode.items():
                b = [r["before"][str(primary)][metric] for r in rows]
                a = [r["after"][str(primary)][metric] for r in rows]
                st = paired_stats(b, a, better)
                st["metric"] = metric
                significance[prefix].setdefault(label, {})[metric] = st
                raw[label] = st["p_t"]
            adj = holm_bonferroni(raw)
            for label in by_mode:
                significance[prefix][label][metric]["p_t_holm"] = adj.get(
                    label, float("nan"))

    # ---- step-budget axis ---------------------------------------------------
    # A fixed 200-step budget over a 24 h prefix is ~40 passes over ~288
    # windows. If a personalised arm loses to zero-shot, that is the first thing
    # to rule out -- so it is measured on its own axis rather than left as a
    # hypothesis in the discussion.
    budget: dict = {}
    if args.step_budgets:
        print(f"\n{'=' * 72}\nSTEP BUDGET at prefix = {args.headline_prefix:g} h"
              f"\n{'=' * 72}")
        for mode, rank, label in jobs:
            if mode not in args.step_budget_modes:
                continue
            for steps in args.step_budgets:
                key = f"{label} @ {steps} steps"
                print(f"\n--- {key} ---")
                res = run_ohio_personalization(
                    base.model, data.ws, data.split, data.pre, cfg,
                    criterion=base.criterion, method=mode,
                    prefix_hours=args.headline_prefix, lora_rank=rank,
                    steps=steps, lr=args.adapt_lr,
                    min_eval_windows=args.min_eval_windows,
                    eval_scope=args.eval_scope, verbose=args.verbose)
                b = [r.before_by_horizon[primary].rmse for r in res]
                a = [r.after_by_horizon[primary].rmse for r in res]
                bp = [r.before_by_horizon[primary].pr_auc for r in res]
                ap_ = [r.after_by_horizon[primary].pr_auc for r in res]
                budget.setdefault(label, {})[str(steps)] = {
                    "steps": steps,
                    "rmse": paired_stats(b, a, "min"),
                    "pr_auc": paired_stats(bp, ap_, "max"),
                    "n_adapt_windows": int(np.median([r.n_adapt_windows for r in res]))
                    if res else 0,
                }
                st = budget[label][str(steps)]["rmse"]
                print(f"    RMSE {st['mean_before']:.3f} -> {st['mean_after']:.3f} "
                      f"({st['mean_delta']:+.3f})")

    # ---- prefix-position axis: opening vs recent ----------------------------
    contrast = prefix_position_contrast(base, data, cfg, jobs, args,
                                        primary, hp, sweep)

    B.write_json(os.path.join(args.out, "personalization_sweep.json"), {
        "experiment": "ohio_personalization_sweep",
        "smoke": bool(args.smoke),
        "base_spec": base_spec.as_dict(),
        "base_best_monitor": base.best_monitor,
        "subjects": data.subjects,
        "protocol": {
            "prefix": "first K hours of each subject's official TRAINING file",
            "prefix_modes": PREFIX_MODE_DOC,
            "embargo": "max(configured, history + max horizon + one period)",
            "eval_scope": args.eval_scope,
            "evaluation": (
                "official TEST windows only -- disjoint from the population "
                "model's fit and val rows" if args.eval_scope == "test" else
                "LEGACY: remainder of TRAIN after the embargo + the entire "
                "official TEST file. ~80% of these rows are in-sample for the "
                "population model; the deltas are not out-of-sample."),
            "pairing": "before and after are scored on identical windows",
            "primary_horizon_min": primary,
        },
        "adapt": {"steps": args.adapt_steps, "lr": args.adapt_lr,
                  "lora_ranks": args.lora_ranks,
                  "prefix_hours": args.prefix_hours,
                  "prefix_modes": args.prefix_modes,
                  "headline_prefix_hours": float(args.headline_prefix)},
        "sweep": sweep,
        "significance": significance,
        "step_budget": budget,
        "prefix_mode_contrast": contrast,
    })

    # ---- reporting ----------------------------------------------------------
    if hp in significance:
        print(f"\n{'=' * 100}\nPAIRED before/after at prefix = {args.headline_prefix:g} h, "
              f"PH = {primary} min ({len(data.subjects)} subjects)\n{'=' * 100}")
        head = (f"{'mode':<16}{'metric':<26}{'before':>10}{'after':>10}"
                f"{'delta':>10}{'95% CI':>20}{'n+':>5}{'p (t)':>10}{'p Holm':>10}"
                f"{'p Wilcox':>10}")
        print(head + "\n" + "-" * len(head))
        for label in significance[hp]:
            for metric, _better, _l in METRICS:
                st = significance[hp][label][metric]
                ci = (f"[{st['ci95_low']:+.3f}, {st['ci95_high']:+.3f}]"
                      if np.isfinite(st["ci95_low"]) else "--")
                fmt = lambda v: ("--" if not np.isfinite(v) else f"{v:.4f}")  # noqa: E731
                print(f"{label:<16}{metric:<26}{st['mean_before']:>10.3f}"
                      f"{st['mean_after']:>10.3f}{st['mean_delta']:>+10.3f}{ci:>20}"
                      f"{st['n_improved']:>5}{fmt(st['p_t']):>10}"
                      f"{fmt(st['p_t_holm']):>10}{fmt(st['wilcoxon_p']):>10}")

    report_contrast(contrast, args, primary)

    print(f"\n{'=' * 72}\nSAMPLE EFFICIENCY: mean per-subject delta vs prefix length\n"
          f"{'=' * 72}")
    traj: dict = {}
    for metric, better, _l in METRICS[:2]:
        print(f"\n  {metric} (delta after - before; "
              f"{'lower' if better == 'min' else 'higher'} is better)")
        header = f"    {'mode':<16}" + "".join(
            f"{p:>12.0f}h" for p in args.prefix_hours)
        print(header)
        for _mode, _rank, label in jobs:
            cells = ""
            for p in args.prefix_hours:
                st = significance[str(p)][label][metric]
                traj.setdefault(metric, {}).setdefault(label, {})[str(p)] = st["mean_delta"]
                cells += (f"{st['mean_delta']:>+13.3f}"
                          if np.isfinite(st["mean_delta"]) else f"{'--':>13}")
            print(f"    {label:<16}{cells}")

    # ---- LaTeX --------------------------------------------------------------
    def sig_rows(prefix: str) -> list[dict]:
        rows = []
        for _mode, _rank, label in jobs:
            row = {"arm": label, "per_horizon": {str(primary): {"s": {}}}}
            for metric, _b, _l in METRICS:
                st = significance[prefix][label][metric]
                row["per_horizon"][str(primary)]["s"][metric] = st["mean_after"]
                row["per_horizon"][str(primary)]["s"][metric + "_delta"] = st["mean_delta"]
                row["per_horizon"][str(primary)]["s"][metric + "_p"] = st["p_t_holm"]
            rows.append(row)
        return rows

    # Three metrics in the printed table, not five: fifteen columns does not fit
    # on a page, and the JSON carries the rest for anyone who wants them.
    TABLE_METRICS = ("rmse", "pr_auc", "fapd_at_5pct_fpr")
    cols = []
    for metric, better, lbl in METRICS:
        if metric not in TABLE_METRICS:
            continue
        cols.append(Column(lbl, f"s.{metric}", "{:.3f}", better))
        cols.append(Column(f"$\\Delta$ {lbl}", f"s.{metric}_delta", "{:+.3f}", better))
        cols.append(Column("$p_{\\mathrm{Holm}}$", f"s.{metric}_p", "{:.3f}", None))
    B.write_latex(
        os.path.join(args.out, "table_personalization.tex"),
        B.latex_table(sig_rows(hp), cols, primary,
                      caption=(f"Few-shot personalisation on OhioT1DM, PH = "
                               f"{primary} min, {args.headline_prefix:g} h "
                               f"adaptation prefix, {len(data.subjects)} subjects. "
                               f"$\\Delta$ is the paired per-subject change from "
                               f"the zero-shot population model on identical "
                               f"evaluation windows; $p$ is a paired $t$-test "
                               f"with Holm-Bonferroni adjustment across modes."),
                      label="tab:personalization", row_label="Adaptation")
        if hp in significance else "% headline prefix not in --prefix-hours\n")

    if budget:
        print(f"\n{'=' * 72}\nSTEP BUDGET: mean paired delta at prefix = "
              f"{args.headline_prefix:g} h\n{'=' * 72}")
        header = f"    {'mode':<16}" + "".join(f"{s:>13}" for s in args.step_budgets)
        for metric in ("rmse", "pr_auc"):
            print(f"\n  delta {metric}\n{header}")
            for label, by_steps in budget.items():
                cells = "".join(
                    f"{by_steps[str(s)][metric]['mean_delta']:>+13.3f}"
                    if str(s) in by_steps else f"{'--':>13}"
                    for s in args.step_budgets)
                print(f"    {label:<16}{cells}")
        b_rows = [{"arm": label, "per_horizon": {str(primary): {"b": {
                      f"s{s}": by_steps.get(str(s), {}).get("rmse", {}).get("mean_delta")
                      for s in args.step_budgets}}}}
                  for label, by_steps in budget.items()]
        B.write_latex(
            os.path.join(args.out, "table_step_budget.tex"),
            B.latex_table(b_rows,
                          [Column(f"{s} steps", f"b.s{s}", "{:+.3f}", "min")
                           for s in args.step_budgets],
                          primary,
                          caption=(f"Adaptation budget: mean paired $\\Delta$RMSE "
                                   f"(mg/dL) against the zero-shot population "
                                   f"model as a function of the number of "
                                   f"gradient steps, at a "
                                   f"{args.headline_prefix:g} h prefix and "
                                   f"PH = {primary} min. Negative is an "
                                   f"improvement."),
                          label="tab:step_budget", row_label="Adaptation"))

    eff_rows = [{"arm": label,
                 "per_horizon": {str(primary): {"e": {
                     f"p{p:g}": traj.get("rmse", {}).get(label, {}).get(str(p))
                     for p in args.prefix_hours}}}}
                for _m, _r, label in jobs]
    eff_cols = [Column(f"{p:g} h", f"e.p{p:g}", "{:+.3f}", "min")
                for p in args.prefix_hours]
    B.write_latex(
        os.path.join(args.out, "table_sample_efficiency.tex"),
        B.latex_table(eff_rows, eff_cols, primary,
                      caption=(f"Sample efficiency: mean paired $\\Delta$RMSE "
                               f"(mg/dL) against the zero-shot population model "
                               f"as a function of adaptation prefix length, "
                               f"PH = {primary} min. Negative is an improvement."),
                      label="tab:sample_efficiency", row_label="Adaptation"))


if __name__ == "__main__":
    main()
