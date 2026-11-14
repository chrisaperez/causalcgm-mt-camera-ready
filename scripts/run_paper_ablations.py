"""The ablation studies.

    python scripts/run_paper_ablations.py --studies a b c d e
    python scripts/run_paper_ablations.py --studies a b c d e f  # include nadir-target control
    python scripts/run_paper_ablations.py --studies b --smoke

  A  Multi-task coupling      does joint training help or hurt forewarning?
  B  Hypoglycemia asymmetry   what does kappa buy, and what does it cost?
  C  Modality contribution    which wearable stream actually carries signal?
  D  Lookahead-bias leakage   how much of a typical published number is real?
  E  Soft-label band          does the boundary band do anything, and does it
                              matter whether it is estimated or pinned?
  F  Nadir target control     are classical baselines handicapped by level forecasting?

Each study writes its own JSON and LaTeX table under `--out`, plus a combined
`ablations_summary.json`. Every arm within a study is trained at an identical
budget (epochs, batch size, seed, subsample) so the comparison inside a table is
clean; budgets are NOT necessarily equal across studies, so numbers should be
read down a column, not across tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B  # noqa: E402
from causalcgm_mt.benchmark import ArmSpec, Column  # noqa: E402
from scripts._cli import OHIO_HINT, missing_input  # noqa: E402

# --------------------------------------------------------------------------- #
# Column sets, one per study
# --------------------------------------------------------------------------- #
COLS_A = (
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("ROC-AUC", "classification.roc_auc", "{:.4f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
    Column("FAPD@5\\%", "alarm.fapd_at_5pct_fpr", "{:.2f}", None),
    Column("Onset lead (min)", "alarm.mean_onset_lead_min", "{:.1f}", "max"),
    Column("RMSE", "regression.rmse", "{:.2f}", "min"),
    Column("Lag (min)", "regression.time_lag_min", "{:.1f}", "min"),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", "max"),
)
COLS_B = (
    Column("RMSE", "regression.rmse", "{:.2f}", "min"),
    Column("RMSE $<$70", "clinical.rmse_below_70", "{:.2f}", "min"),
    Column("Bias $<$70", "clinical.hypo_band_bias", "{:+.2f}", None),
    Column("D+E ($n$)", "clinical.zone_de_n", "{:.0f}", "min"),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", "max"),
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
)
COLS_C = (
    Column("RMSE", "regression.rmse", "{:.2f}", "min"),
    Column("MAE", "regression.mae", "{:.2f}", "min"),
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", "max"),
    Column("D+E ($n$)", "clinical.zone_de_n", "{:.0f}", "min"),
    Column("$n$ test", "n", "{:.0f}", None),
)
COLS_E = (
    Column("Band (mg/dL)", "soft_label.band_mgdl", "{:.2f}", None),
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("ROC-AUC", "classification.roc_auc", "{:.4f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
    Column("FAPD@5\\%", "alarm.fapd_at_5pct_fpr", "{:.2f}", None),
    Column("Onset lead (min)", "alarm.mean_onset_lead_min", "{:.1f}", "max"),
    Column("RMSE", "regression.rmse", "{:.2f}", "min"),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", "max"),
)
COLS_F = (
    # CLASSIFICATION ONLY, deliberately. A nadir-target arm's RMSE is measured
    # against g(t+PH) like every other arm's, so for that arm it is the error of
    # a forecast of a different quantity -- a number that is defined but does not
    # mean what the column heading says. The study asks a discrimination
    # question; the discrimination metrics are the honest answer to it.
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("ROC-AUC", "classification.roc_auc", "{:.4f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
    Column("FAPD@5\\%", "alarm.fapd_at_5pct_fpr", "{:.2f}", None),
    Column("Onset lead (min)", "alarm.mean_onset_lead_min", "{:.1f}", "max"),
)
COLS_D = (
    Column("RMSE", "regression.rmse", "{:.2f}", None),
    Column("MAE", "regression.mae", "{:.2f}", None),
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", None),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", None),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", None),
    Column("Lag (min)", "regression.time_lag_min", "{:.1f}", None),
    Column("$n$ test", "n", "{:.0f}", None),
)


# --------------------------------------------------------------------------- #
# Table titles and captions, as data
# --------------------------------------------------------------------------- #
# `rebuild_paper_tables.py` re-emits these tables from the same JSON and already
# imports `COLS_*` from this module rather than restating them, because a second
# copy of a column list drifts and the drift surfaces as two tables in one paper
# disagreeing about what a column means. The CAPTIONS were a second copy of
# exactly that kind, and they drifted the first time they were touched: study E's
# caption contained a raw `_`, which is a hard LaTeX error, and it had to be
# fixed identically in two files. They are data here for the same reason the
# columns are.
#
# The caption is LaTeX. Code identifiers need `\texttt` and escaped underscores.
TITLES = {
    "a_multitask": "Ablation A: multi-task coupling",
    "b_asymmetry": "Ablation B: asymmetric hypoglycemia cost",
    "d_leakage": "Ablation D: what lookahead bias buys",
    "e_soft_label": "Ablation E: the soft-label boundary band",
    "f_nadir_target": "Ablation F: is the endpoint baseline handicapped "
                      "by its risk mapping?",
}
CAPTIONS = {
    "a_multitask":
        "Single-task arms zero the other task's loss weight, so the trunk is "
        "trained by one objective; the untrained head's metrics are reported "
        "but not meaningful.",
    "b_asymmetry":
        "$\\kappa$ multiplies the regression loss when the reference is "
        "hypoglycemic and the prediction sits above it; $\\kappa = 1$ is a no-op.",
    "c_modality":
        "Scored only on the subjects whose device carries the arm's sensors, "
        "against a CGM-only control drawn from the same subjects. Not "
        "comparable across cohorts.",
    # Kept to one line: this table sits in the 8-page body, where a three-line
    # full-width caption costs more than it explains. The reading instruction is
    # restated in the prose that introduces the table.
    "d_leakage":
        "Arms are \\emph{not} scored on the same test set: changing the split is the "
        "flaw being measured.",
    "e_soft_label":
        "The band column is each arm's \\emph{realised} band. The estimated arm derives "
        "it from the fold's MARD against paired capillary reference BG; a band "
        "below the device specification falls back to the specification, and "
        "\\texttt{soft\\_label.band\\_source} in the JSON records which happened.",
    "f_nadir_target":
        "Classification metrics only. Persistence, Ridge-AR and ARIMA have no "
        "classification head and score hypoglycemia through "
        "$\\sigma((70 - \\hat{y})/\\tau)$, a strictly monotone map of their "
        "forecast; the label is the window \\emph{minimum}, which an endpoint forecast "
        "does not target. The nadir arms are the identical Ridge-AR fitted to "
        "$\\min g$ over $(t, t+\\mathrm{PH}]$ instead of $g(t+\\mathrm{PH})$ and "
        "mapped the same way, so the difference between the two rows is the "
        "target alone. LightGBM and every neural arm fit a classifier on the "
        "nadir label directly and are shown for reference, not as part of the "
        "contrast.",
}


# Ablation C, condensed. The per-cohort tables carry every metric and stay in
# the artefact directory, but four full tables of null results in the main text
# is four tables asking the reader to look for an effect the study is not
# powered to find. This collapses them to one, and reports each arm as a DELTA
# from its own cohort's CGM-only control -- the only comparison the design
# supports, since the two cohorts wore different devices and are not comparable
# to each other.
COLS_C_SUMMARY = (
    Column("$\\Delta$RMSE 30", "d.rmse30", "{:+.2f}", None),
    Column("$\\Delta$PR-AUC 30", "d.pr30", "{:+.3f}", None),
    Column("$\\Delta$RMSE 60", "d.rmse60", "{:+.2f}", None),
    Column("$\\Delta$PR-AUC 60", "d.pr60", "{:+.3f}", None),
    Column("$n$ subj", "d.n_subj", "{:.0f}", None),
)
C_SUMMARY_CAPTION = (
    "Ablation C, condensed: every auxiliary arm as a change from its \\emph{own} "
    "cohort's CGM-only control, which is the only comparison the design "
    "supports -- the 2018 and 2020 releases carry disjoint sensors, so the two "
    "blocks are not comparable to each other. Negative $\\Delta$RMSE and "
    "positive $\\Delta$PR-AUC are improvements. Each arm is one seed on six "
    "subjects, so the deltas are directions, not resolved effects: treatment "
    "inputs lower RMSE in both cohorts and at both horizons, and wearable "
    "streams do not do so consistently. Per-cohort tables with every metric "
    "are in the artefact directory."
)


def c_modality_summary_rows(c_study: dict, horizons) -> list[dict]:
    """One row per (cohort, arm), as a delta from that cohort's CGM-only control.

    The control is located by name rather than by position: `specs_c` puts it
    first today, and a table that silently subtracts the wrong row if that order
    ever changes is worse than one that fails.
    """
    h30, h60 = str(horizons[0]), str(horizons[-1])
    rows = []
    for cohort, arms in c_study.items():
        ctrl = next((a for a in arms if a["arm"] == "CGM only"), None)
        if ctrl is None:
            raise KeyError(f"cohort {cohort!r} has no 'CGM only' control row; "
                           f"got {[a['arm'] for a in arms]}")

        def g(row, h, block, key):
            return row["per_horizon"].get(h, {}).get(block, {}).get(key)

        def delta(row, h, block, key):
            a, b = g(row, h, block, key), g(ctrl, h, block, key)
            return None if a is None or b is None else a - b

        for a in arms:
            if a is ctrl:
                continue
            rows.append({
                "arm": f"{cohort}: {a['arm']}",
                "per_horizon": {h30: {"d": {
                    "rmse30": delta(a, h30, "regression", "rmse"),
                    "pr30": delta(a, h30, "classification", "pr_auc"),
                    "rmse60": delta(a, h60, "regression", "rmse"),
                    "pr60": delta(a, h60, "classification", "pr_auc"),
                    "n_subj": len(a.get("subjects", [])) or None,
                }}},
            })
    return rows


def c_modality_title(cohort: str) -> str:
    """Ablation C's title is per cohort, so it is built rather than stored."""
    return (f"Ablation C: modality contribution, {cohort} cohort "
            f"($n = 6$ subjects)")


def _study_keys(args) -> set[str]:
    """JSON keys the requested studies will write, so a merge knows what it is
    replacing rather than guessing from the letter."""
    names = {"a": "a_multitask", "b": "b_asymmetry", "c": "c_modality",
             "d": "d_leakage", "e": "e_soft_label", "f": "f_nadir_target"}
    return {names[k] for k in args.studies}


def _pred_dir(args, study: str) -> str | None:
    """Where one study's per-arm prediction dumps go, or None when disabled."""
    return (os.path.join(args.out, "predictions", study)
            if getattr(args, "save_predictions", False) else None)


def _base(args, **over) -> dict:
    d = dict(horizons=tuple(args.horizons), epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, patience=args.patience,
             seed=args.seed, fit_subsample=args.fit_subsample)
    d.update(over)
    return d


def _num(value, spec: str, dash: str = "--") -> str:
    """Format a number that may be absent, None, or not finite.

    `dict.get(key, default)` does not cover this: the key is PRESENT and its
    value is `None`. `SoftLabeler.as_dict` reports `band_mgdl = None` for a
    hard-label arm deliberately -- an arm with no ramp has no band, and printing
    the width it would have had puts a number in Ablation E's independent-variable
    column for the one row where it does not apply. Formatting that with `:6.2f`
    raises `TypeError`, which is what it did.

    Rendered as a right-aligned `--`, matching `benchmark._cells`: the log line
    and the LaTeX table are describing the same cell and should not disagree
    about how an inapplicable one looks.
    """
    width = int(re.match(r"(\d*)", spec).group(1) or 0)
    if value is None:
        return dash.rjust(width)
    try:
        if not math.isfinite(float(value)):
            return dash.rjust(width)
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value).rjust(width)


# --------------------------------------------------------------------------- #
# Ablation A: multi-task coupling
# --------------------------------------------------------------------------- #
def specs_a(args) -> list[ArmSpec]:
    """Single-task heads are obtained by zeroing the other task's lambda.

    With `lambda_cls = 0` the classification head receives no gradient and the
    trunk is trained by regression alone -- which is exactly the single-task
    model, without needing a second architecture that could differ in some other
    way. The unused head still exists and still outputs numbers; those numbers
    are meaningless and are reported as such.

    MODEL SELECTION FOLLOWS THE ARM'S OBJECTIVE. Early-stopping a
    regression-only arm on PR-AUC would select on a head that never received a
    gradient, i.e. on noise. The monitor therefore differs between arms, and
    that is a deliberate part of the design rather than an inconsistency.
    """
    return [
        ArmSpec("Regression only", "tcn_mm", loss_weighting="fixed",
                lambda_reg=1.0, lambda_cls=0.0, monitor="val/rmse",
                monitor_mode="min",
                notes="classification head receives no gradient; its metrics are "
                      "not meaningful", **_base(args)),
        ArmSpec("Classification only", "tcn_mm", loss_weighting="fixed",
                lambda_reg=0.0, lambda_cls=1.0,
                notes="regression head receives no gradient; its RMSE is not "
                      "meaningful", **_base(args)),
        ArmSpec("Joint, fixed $\\lambda=1$", "tcn_mm", loss_weighting="fixed",
                lambda_reg=1.0, lambda_cls=1.0, **_base(args)),
        ArmSpec("Joint, Kendall uncertainty", "tcn_mm",
                loss_weighting="uncertainty", **_base(args)),
    ]


# --------------------------------------------------------------------------- #
# Ablation B: hypoglycemia asymmetry sweep
# --------------------------------------------------------------------------- #
def specs_b(args) -> list[ArmSpec]:
    """kappa = 1.0 is an exact no-op (plain Huber) and anchors the sweep."""
    return [ArmSpec(f"$\\kappa$ = {k:g}", "tcn_mm", hypo_asymmetry=float(k),
                    **_base(args)) for k in args.kappas]


def specs_f(args) -> list[ArmSpec]:
    """Does an endpoint forecast miss nadirs, or does the risk MAPPING miss them?

    The three classical arms have no classification head. Their hypoglycemia
    score is `sigmoid((70 - yhat)/tau)`, a strictly monotone map of a forecast of
    `g(t + PH)` -- while the label is `min g` over `(t, t + PH]`. That is a real
    mismatch, and on its own the table cannot say whether it is the mapping that
    costs them or the fact that an endpoint is not a minimum.

    Fitting the SAME arm to the nadir separates the two. If the nadir arm's
    discrimination matches the level arm's, the endpoint forecast already
    carried the nadir information and the mapping was not a handicap; if it
    jumps, the level arms were being asked a question their target could not
    answer and the benchmark's classification column understated them.

    Persistence is included as the floor because its nadir variant is a
    genuinely different predictor -- `last_bg` is unchanged, but it is now being
    read as a forecast of the minimum rather than of the endpoint.
    """
    base = _base(args)
    return [
        ArmSpec("Ridge-AR, level target", "ridge_ar", target="level", **base),
        ArmSpec("Ridge-AR, nadir target", "ridge_ar", target="nadir",
                notes="identical design matrix and residual parameterisation; "
                      "only the regression target differs",
                **base),
        ArmSpec("LightGBM (native nadir classifier)", "lgbm", **base,
                notes="not part of the contrast -- fits a classifier on the "
                      "nadir label directly, so its risk is not a mapped "
                      "forecast"),
        ArmSpec("CausalCGM-MT (native nadir head)", "tcn_mm", **base,
                notes="reference row; multi-task classification head"),
    ]


# --------------------------------------------------------------------------- #
# Ablation C: modality contribution
# --------------------------------------------------------------------------- #
def specs_c(args) -> list[tuple[str, list[ArmSpec]]]:
    """Two cohort-scoped sub-tables, never one ranking across both.

    2018 wore a Basis Peak and 2020 an Empatica, and the sensor sets are
    disjoint: no subject has both heart rate and acceleration. An arm is
    therefore only measurable on the six subjects whose device carries its
    sensors, and each scope gets its own CGM-only control drawn from the SAME
    six subjects. At n=6 per group, between-patient variance in glycemic control
    swamps any sensor effect a cross-cohort ranking would imply, so this script
    does not print one.
    """
    def arm(name, mods, cohort):
        return ArmSpec(name, "tcn_mm",
                       unimodal=mods is not None and len(mods) == 0,
                       modalities=None if mods is None else tuple(mods),
                       cohorts=(cohort,), **_base(args))
    a2018 = [
        arm("CGM only", [], "2018"),
        arm("CGM + HR", ["heart_rate"], "2018"),
        arm("CGM + GSR", ["gsr"], "2018"),
        arm("CGM + steps", ["steps"], "2018"),
        arm("CGM + treatment", ["treatment"], "2018"),
        # `None` = every auxiliary channel, INCLUDING insulin, carbohydrate,
        # basal and finger sticks. It was labelled "all wearables", which it
        # is not: the treatment channels are the ones that move RMSE.
        arm("CGM + all auxiliary (incl. treatment)", None, "2018"),
    ]
    a2020 = [
        arm("CGM only", [], "2020"),
        arm("CGM + accel", ["accel"], "2020"),
        arm("CGM + GSR", ["gsr"], "2020"),
        arm("CGM + treatment", ["treatment"], "2020"),
        arm("CGM + all auxiliary (incl. treatment)", None, "2020"),
    ]
    return [("2018", a2018), ("2020", a2020)]


# --------------------------------------------------------------------------- #
# Ablation E: the soft-label boundary band
# --------------------------------------------------------------------------- #
def specs_e(args) -> list[ArmSpec]:
    """Hard labels, three pinned band widths, and the estimated band.

    WHY THIS STUDY DID NOT EXIST UNTIL NOW (audit finding F10). The estimated
    arm was unreachable. `band_mode="mard"` derived the band from a high-pass
    proxy for device noise, the released OhioT1DM CGM has been smoothed, and the
    proxy read 0.7% -- a +/-0.5 mg/dL band, half a display step, which makes the
    logistic ramp a hard label with extra arithmetic. The labeller warned and
    continued, so every driver pinned `band_mode="fixed"` and the treatment arm
    of this table would have been a copy of a control.

    Two changes make it a real sweep: the estimate now comes from the paired
    finger-stick reference that ships in the same files (a MARD by its actual
    definition, ~10.5% on this cohort, +/-7.35 mg/dL), and a band that still
    lands below the device specification falls back to the specification and
    SAYS SO in `soft_label.band_source`. The band column here is the arm's
    realised band, read back out of the results, not the requested one.

    The width sweep brackets the estimate on both sides so the table separates
    two questions that a two-arm version would confound: whether soft labels
    help at all, and whether it matters that the width was estimated rather
    than chosen.
    """
    hard = ArmSpec("Hard labels", "tcn_mm", soft_labels=False,
                   notes="control: 1[min BG < 70], no ramp", **_base(args))
    fixed = [ArmSpec(f"Fixed $\\pm${w:g} mg/dL", "tcn_mm", soft_labels=True,
                     band_mode="fixed", soft_band_mgdl=float(w), **_base(args))
             for w in args.bands]
    est = ArmSpec("Estimated (MARD)", "tcn_mm", soft_labels=True,
                  band_mode="mard", band_mard_multiplier=1.0,
                  notes="band = k * MARD * threshold, MARD from paired reference BG",
                  **_base(args))
    return [hard, *fixed, est]


# --------------------------------------------------------------------------- #
# Ablation D: lookahead-bias leakage
# --------------------------------------------------------------------------- #
def _leak_cohort(rung, args, strict: B.CohortData) -> B.CohortData:
    """Assemble the (windows, split, preprocessor) triple for one rung.

    The preprocessor is REFITTED on whatever that rung calls its fit rows, which
    is the faithful control. A naive practitioner fits their scaler on their own
    (leaky) training split; handing the leaky arm the strict arm's preprocessor
    would quietly repair part of the flaw and understate the inflation being
    measured.
    """
    from causalcgm_mt.hupa_loader import HupaConfig
    from causalcgm_mt.leakage_control import build_naive_windows, naive_random_split
    from causalcgm_mt.ohio_loader import OhioConfig, OhioPreprocessor, discover_records

    if rung.windows == "strict":
        ws = strict.ws
    else:
        # Version-keyed for the same reason `benchmark.load_cohort`'s cache is:
        # the naive builder calls `synchronize_ohio_record`, so a change to the
        # CGM snap or an auxiliary mask alters its window CONTENT while leaving
        # every shape identical. A bare pickle -- which this was -- silently
        # reuses pre-fix windows and the leakage ladder then compares a fixed
        # strict arm against a stale leaky one.
        cache = os.path.join(args.out, "naive_ws.pkl")
        ws = None
        if os.path.exists(cache):
            import pickle
            with open(cache, "rb") as f:
                payload = pickle.load(f)
            if (isinstance(payload, dict)
                    and payload.get("pipeline_version") == B.WINDOW_PIPELINE_VERSION
                    and tuple(payload.get("horizons", ())) == tuple(args.horizons)):
                ws = payload["ws"]
                print(f"loaded {len(ws)} cached NAIVE windows from {cache}")
            else:
                got = (payload.get("pipeline_version", "<none>")
                       if isinstance(payload, dict) else "<unversioned>")
                print(f"{cache} was built by pipeline version {got}, this is "
                      f"{B.WINDOW_PIPELINE_VERSION}; rebuilding")
        if ws is None:
            ws = build_naive_windows(discover_records(args.root), strict.cfg,
                                     HupaConfig(), OhioConfig(),
                                     horizons_min=tuple(args.horizons))
            print(f"built {len(ws)} NAIVE (leaky) windows")
            import pickle
            os.makedirs(args.out, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"ws": ws, "horizons": tuple(args.horizons),
                             "pipeline_version": B.WINDOW_PIPELINE_VERSION}, f)

    split = (strict.split if rung.split == "official"
             else naive_random_split(ws, seed=args.seed))
    pre = OhioPreprocessor.fit(ws, split, strict.cfg)
    return B.CohortData(ws=ws, split=split, pre=pre, cfg=strict.cfg,
                        horizons=strict.horizons)


def run_study_d(args, strict: B.CohortData) -> list[dict]:
    from causalcgm_mt.leakage_control import LEAK_LADDER

    results = []
    for rung in LEAK_LADDER:
        print(f"\n--- D: {rung.label} ---\n    {rung.description}")
        data = _leak_cohort(rung, args, strict)
        spec = ArmSpec(rung.label, "tcn_mm", **_base(args))
        res, _ = B.run_arm(spec, data, alpha=args.alpha,
                           predictions_dir=_pred_dir(args, "d_leakage"))
        res["leak_arm"] = dataclasses.asdict(rung)
        results.append(res)
    return results


def leakage_inflation(results: list[dict], horizon: int) -> dict:
    """How many points each flaw adds to a reported number.

    Stated as an inflation relative to the strict arm, and NOT as a paired
    comparison: the arms are scored on different test sets, necessarily, because
    changing the split is the flaw. That is the honest framing -- the question
    is "how much better does a published number look", not "how much worse is
    this particular model".
    """
    if not results:
        return {}
    base = results[0]["per_horizon"][str(horizon)]
    out = {}
    for r in results[1:]:
        cur = r["per_horizon"][str(horizon)]
        b_rmse = base["regression"]["rmse"]
        c_rmse = cur["regression"]["rmse"]
        b_pr, c_pr = base["classification"]["pr_auc"], cur["classification"]["pr_auc"]
        out[r["arm"]] = {
            "rmse_strict": b_rmse, "rmse_leaky": c_rmse,
            "rmse_absolute_improvement": b_rmse - c_rmse,
            "rmse_relative_improvement_pct": 100.0 * (b_rmse - c_rmse) / b_rmse,
            "pr_auc_strict": b_pr, "pr_auc_leaky": c_pr,
            "pr_auc_absolute_gain": c_pr - b_pr,
            "pr_auc_relative_gain_pct": 100.0 * (c_pr - b_pr) / max(b_pr, 1e-9),
        }
    return out


# --------------------------------------------------------------------------- #
def _emit(args, key, title, rows, columns, caption_extra="",
          single_horizon: bool = False):
    """Write one table per horizon, or a single table when the rows already
    carry every horizon in their own columns (the condensed Ablation C)."""
    if single_horizon:
        h = args.horizons[0]
        B.write_latex(
            os.path.join(args.out, f"table_ablation_{key}.tex"),
            B.latex_table(rows, columns, h, caption=caption_extra,
                          label=f"tab:ablation_{key}", row_label="Arm"))
        print(f"\n--- {title} (condensed) ---")
        print(B.text_table(rows, columns, h, width=30))
        return rows
    tex = []
    for h in args.horizons:
        tex.append(B.latex_table(
            rows, columns, h,
            caption=f"{title} (PH = {h} min). {caption_extra}",
            label=f"tab:ablation_{key}_ph{h}", row_label="Arm"))
        print(f"\n--- {title}, PH = {h} min ---")
        print(B.text_table(rows, columns, h, width=30))
    B.write_latex(os.path.join(args.out, f"table_ablation_{key}.tex"),
                  "\n\n".join(tex))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="OhioT1DM")
    ap.add_argument("--out", default="runs/paper_results/ablations")
    ap.add_argument("--cache", default="runs/paper_results/ohio_ws.pkl")
    ap.add_argument("--studies", nargs="+", default=["a", "b", "c", "d", "e"],
                    choices=["a", "b", "c", "d", "e", "f"],
                    help="which ablations to run. F (the nadir-target control) "
                         "is not in the default set because it answers a "
                         "reader's question about the benchmark table rather "
                         "than contributing a row to it; pass it explicitly.")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60])
    ap.add_argument("--history", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fit-subsample", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--kappas", nargs="+", type=float,
                    default=[1.0, 2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--bands", nargs="+", type=float, default=[5.0, 10.0, 20.0],
                    help="pinned soft-label band widths (mg/dL) for study E; "
                         "the estimated arm is added on top of these")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-predictions", action="store_true", default=True,
                    help="dump each arm's test predictions to "
                         "<out>/predictions/<study>/ for the figure scripts")
    ap.add_argument("--no-save-predictions", dest="save_predictions",
                    action="store_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-merge", action="store_true",
                    help="overwrite the summary instead of merging into it. By "
                         "default a partial re-run (say --studies d) keeps the "
                         "studies already in the file, because clobbering three "
                         "completed studies to re-run one is the more expensive "
                         "mistake -- the budget of each study is recorded "
                         "alongside it so a mixed file is still auditable.")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.patience, args.fit_subsample = 2, 1, 16
        args.kappas = [1.0, 3.0]
        args.bands = [10.0]

    # The reference pairs are only needed by study E's estimated arm, and
    # collecting them re-parses the raw XML (~7 s). Loaded on demand rather than
    # unconditionally, so the other four studies are not slowed by a table they
    # do not use.
    with missing_input(OHIO_HINT):
        data = B.load_cohort(args.root, history_min=args.history,
                             horizons=tuple(args.horizons), val_frac=args.val_frac,
                             cache=args.cache,
                             with_reference_pairs="e" in args.studies)
    print(data.summary())

    budget = {"epochs": args.epochs, "patience": args.patience,
              "fit_subsample": args.fit_subsample, "batch_size": args.batch_size,
              "lr": args.lr, "seed": args.seed,
              "horizons_min": list(args.horizons)}
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {"experiment": "ohio_ablations", "smoke": bool(args.smoke),
               "budget": budget, "generated": stamp, "studies": {},
               "study_meta": {}}

    out_path = os.path.join(args.out, "ablations_summary.json")
    if not args.no_merge and os.path.exists(out_path):
        with open(out_path) as f:
            prior = json.load(f)
        kept = [k for k in prior.get("studies", {}) if k not in _study_keys(args)]
        for k in kept:
            payload["studies"][k] = prior["studies"][k]
            payload["study_meta"][k] = prior.get("study_meta", {}).get(
                k, {"budget": prior.get("budget"),
                    "generated": prior.get("generated")})
            if payload["study_meta"][k].get("budget") != budget:
                print(f"NOTE: keeping study {k!r} from an earlier run with a "
                      f"DIFFERENT budget {payload['study_meta'][k].get('budget')}; "
                      f"read it down its own column, not across studies.")
        if kept:
            print(f"merging into {out_path}: keeping {kept}, replacing "
                  f"{sorted(_study_keys(args) & set(prior.get('studies', {})))}")

    if "a" in args.studies:
        print("\n" + "=" * 72 + "\nABLATION A: multi-task coupling\n" + "=" * 72)
        rows = [B.run_arm(s, data, alpha=args.alpha,
                          predictions_dir=_pred_dir(args, "a_multitask"))[0]
                for s in specs_a(args)]
        _emit(args, "a_multitask", TITLES["a_multitask"], rows, COLS_A,
              CAPTIONS["a_multitask"])
        payload["study_meta"]["a_multitask"] = {"budget": budget, "generated": stamp}
        payload["studies"]["a_multitask"] = rows

    if "b" in args.studies:
        print("\n" + "=" * 72 + "\nABLATION B: hypoglycemia asymmetry\n" + "=" * 72)
        rows = [B.run_arm(s, data, alpha=args.alpha,
                          predictions_dir=_pred_dir(args, "b_asymmetry"))[0]
                for s in specs_b(args)]
        _emit(args, "b_asymmetry", TITLES["b_asymmetry"], rows, COLS_B,
              CAPTIONS["b_asymmetry"])
        payload["study_meta"]["b_asymmetry"] = {"budget": budget, "generated": stamp}
        payload["studies"]["b_asymmetry"] = rows

    if "c" in args.studies:
        print("\n" + "=" * 72 + "\nABLATION C: modality contribution\n" + "=" * 72)
        payload["study_meta"]["c_modality"] = {"budget": budget, "generated": stamp}
        payload["studies"]["c_modality"] = {}
        for cohort, specs in specs_c(args):
            rows = [B.run_arm(s, data, alpha=args.alpha,
                               predictions_dir=_pred_dir(args, f"c_modality_{cohort}"))[0]
                    for s in specs]
            _emit(args, f"c_modality_{cohort}", c_modality_title(cohort),
                  rows, COLS_C, CAPTIONS["c_modality"])
            payload["studies"]["c_modality"][cohort] = rows
        # One condensed table for the main text, from the same rows.
        _emit(args, "c_modality_summary", "Ablation C: wearable contribution",
              c_modality_summary_rows(payload["studies"]["c_modality"],
                                      args.horizons),
              COLS_C_SUMMARY, C_SUMMARY_CAPTION, single_horizon=True)

    if "f" in args.studies:
        print("\n" + "=" * 72 +
              "\nABLATION F: nadir vs level target for the mapped arms\n" + "=" * 72)
        rows = [B.run_arm(s, data, alpha=args.alpha,
                          predictions_dir=_pred_dir(args, "f_nadir_target"))[0]
                for s in specs_f(args)]
        _emit(args, "f_nadir_target", TITLES["f_nadir_target"], rows, COLS_F,
              CAPTIONS["f_nadir_target"])
        payload["study_meta"]["f_nadir_target"] = {"budget": budget, "generated": stamp}
        payload["studies"]["f_nadir_target"] = rows

    if "e" in args.studies:
        print("\n" + "=" * 72 + "\nABLATION E: soft-label boundary band\n" + "=" * 72)
        rows = [B.run_arm(s, data, alpha=args.alpha,
                          predictions_dir=_pred_dir(args, "e_soft_label"))[0]
                for s in specs_e(args)]
        _emit(args, "e_soft_label", TITLES["e_soft_label"], rows, COLS_E,
              CAPTIONS["e_soft_label"])
        payload["study_meta"]["e_soft_label"] = {"budget": budget, "generated": stamp}
        payload["studies"]["e_soft_label"] = rows
        for r in rows:
            sl = r.get("soft_label", {})
            print(f"    {r['arm']:<28} band {_num(sl.get('band_mgdl'), '6.2f')} "
                  f"mg/dL  source={sl.get('band_source', '--')}  "
                  f"mard={_num(sl.get('mard_used'), '6.4f')}")

    if "d" in args.studies:
        print("\n" + "=" * 72 + "\nABLATION D: lookahead-bias leakage\n" + "=" * 72)
        rows = run_study_d(args, data)
        _emit(args, "d_leakage", TITLES["d_leakage"], rows, COLS_D,
              CAPTIONS["d_leakage"])
        infl = {str(h): leakage_inflation(rows, h) for h in args.horizons}
        payload["study_meta"]["d_leakage"] = {"budget": budget, "generated": stamp}
        payload["studies"]["d_leakage"] = {"arms": rows, "inflation": infl}
        for h, block in infl.items():
            print(f"\n  leakage inflation at PH = {h} min (vs the strict pipeline):")
            for name, v in block.items():
                print(f"    {name:<44} RMSE {v['rmse_strict']:.2f} -> "
                      f"{v['rmse_leaky']:.2f} ({v['rmse_relative_improvement_pct']:+.1f}%)"
                      f" | PR-AUC {v['pr_auc_strict']:.4f} -> {v['pr_auc_leaky']:.4f} "
                      f"({v['pr_auc_absolute_gain']:+.4f})")

    B.write_json(out_path, payload)


if __name__ == "__main__":
    main()
