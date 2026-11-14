"""Master SOTA comparison table on OhioT1DM (2018 + 2020, all 12 subjects).

    python scripts/run_paper_benchmarks.py --root OhioT1DM
    python scripts/run_paper_benchmarks.py --smoke          # 2-epoch plumbing check
    python scripts/run_paper_benchmarks.py --arms Persistence "CausalCGM-MT (fusion)"

Every arm is fitted on `split.fit`, early-stopped on `split.val`, conformal-
calibrated and alarm-thresholded on the disjoint, embargoed `split.calib`, and
scored on `split.test` -- the official `*-ws-testing.xml`
windows, unresampled, so the numbers stay comparable to the published Ohio
benchmark. That is enforced by the harness (`causalcgm_mt/benchmark.py`), not by
this file, which only decides WHICH arms to enumerate.

WHY THE PERSONALISED ROW IS SCORED WHERE IT IS
----------------------------------------------
`run_ohio_personalization` scores on the prefix protocol's evaluation region
(training tail + test file), which is right for a paired before/after claim and
wrong for a table whose other rows are scored on the test file alone. The LoRA
row here adapts identically -- same 24 h prefix, same embargo, same causality
assertion -- and is then scored on `split.test` only, so it sits in the same
column space as everything above it. `run_ohio_personalization_sweep.py` reports
the paired version.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B  # noqa: E402
from scripts._cli import OHIO_HINT, missing_input  # noqa: E402


def build_specs(args) -> list[B.ArmSpec]:
    """The arms of the master table, in the order they are reported."""
    hz = tuple(args.horizons)
    common = dict(horizons=hz, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, patience=args.patience, seed=args.seed,
                  fit_subsample=args.fit_subsample)
    return [
        B.ArmSpec("Persistence", "persistence", horizons=hz,
                  notes="BG(t+PH) = BG(t); the floor every model must clear"),
        B.ArmSpec("Ridge-AR", "ridge_ar", horizons=hz,
                  notes="direct multi-step linear AR on the 24-lag history"),
        B.ArmSpec(f"ARIMA{args.arima_order}", "arima", horizons=hz,
                  notes="statsmodels conditional-MLE fit per subject, applied "
                        "causally per window"),
        B.ArmSpec("LightGBM", "lgbm", horizons=hz,
                  notes="43 causal lag features, one model pair per horizon"),
        B.ArmSpec("Dilated LSTM (CGM)", "lstm", backbone="dilated_lstm",
                  unimodal=True, **common),
        B.ArmSpec("CausalCGM-MT (CGM)", "tcn", unimodal=True, **common),
        B.ArmSpec("CausalCGM-MT (fusion)", "tcn_mm", unimodal=False, **common),
        B.ArmSpec(f"CausalCGM-MT + LoRA ($r = {args.lora_rank}$)", "lora",
                  base_arm="CausalCGM-MT (fusion)", adapt_method="lora",
                  lora_rank=args.lora_rank,
                  adapt_prefix_hours=args.prefix_hours,
                  adapt_steps=args.adapt_steps, **common),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="OhioT1DM")
    ap.add_argument("--out", default="runs/paper_results/benchmarks")
    ap.add_argument("--cache", default="runs/paper_results/ohio_ws.pkl")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60])
    ap.add_argument("--history", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--fit-subsample", type=int, default=1,
                    help="keep every k-th FIT window; val/test are untouched, so "
                         "the reported metrics stay at native cadence")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage")
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--prefix-hours", type=float, default=24.0)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--arima-order", default="(4,1,0)",
                    help="label only; the fitted order is set in benchmark.py")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="run only these arm names (default: all)")
    ap.add_argument("--save-predictions", action="store_true", default=True,
                    help="dump each arm's test predictions to <out>/predictions/ "
                         "so the figure scripts plot the same numbers the tables "
                         "report, instead of retraining and hoping they agree")
    ap.add_argument("--no-save-predictions", dest="save_predictions",
                    action="store_false")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 1/16 of the fit split: checks the plumbing, "
                         "not the science")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.patience, args.fit_subsample = 2, 1, 16
        args.adapt_steps = 10

    with missing_input(OHIO_HINT):
        data = B.load_cohort(args.root, history_min=args.history,
                             horizons=tuple(args.horizons), val_frac=args.val_frac,
                             cache=args.cache)
    print(data.summary())
    print(f"subjects: {', '.join(data.subjects)}")

    specs = build_specs(args)
    if args.arms:
        wanted = set(args.arms)
        unknown = wanted - {s.name for s in specs}
        if unknown:
            ap.error(f"unknown arm(s) {sorted(unknown)}; available: "
                     f"{[s.name for s in specs]}")
        specs = [s for s in specs if s.name in wanted]

    results, trained_by_name, spec_by_name = [], {}, {s.name: s for s in specs}
    for spec in specs:
        print(f"\n=== {spec.name} ({spec.family}) ===")
        base = trained_by_name.get(spec.base_arm)
        if spec.family == "lora" and base is None:
            print(f"  skipped: base arm {spec.base_arm!r} was not run in this "
                  f"invocation, so there is no checkpoint to adapt")
            continue
        res, trained = B.run_arm(
            spec, data, alpha=args.alpha, base=base,
            base_spec=spec_by_name.get(spec.base_arm),
            predictions_dir=(os.path.join(args.out, "predictions")
                             if args.save_predictions else None))
        results.append(res)
        if trained is not None:
            trained_by_name[spec.name] = trained

    payload = {
        "experiment": "ohio_sota_benchmark",
        # Basename only: an absolute path would put the local username into a
        # results file that ships in the public artefact bundle.
        "dataset": {"root": os.path.basename(os.path.normpath(args.root)),
                    "subjects": data.subjects,
                    "history_min": args.history, "horizons_min": list(args.horizons),
                    "val_frac": args.val_frac,
                    "n_fit": int(data.split.fit.size), "n_val": int(data.split.val.size),
                    "n_test": int(data.split.test.size),
                    "n_embargoed": int(data.split.embargoed.size)},
        "protocol": {
            "split": "official OhioT1DM train/test files; chronological inner "
                     "validation with a history+horizon embargo",
            "fitted_on_fit_rows_only": list(data.pre.fitted_on_patients),
            "early_stopping": "split.val",
            "conformal_calibration": "split.calib (disjoint from and embargoed "
                                     "against split.val and split.test)",
            "alarm_thresholds": "fitted on split.calib at target FPR, applied "
                                "unchanged to split.test",
            "scored_on": "split.test (official testing windows, unresampled)",
            "alpha": args.alpha,
        },
        "smoke": bool(args.smoke),
        "arms": results,
    }
    B.write_json(os.path.join(args.out, "results_summary.json"), payload)

    tex = []
    for h in args.horizons:
        tex.append(B.latex_table(
            results, B.SOTA_COLUMNS, h,
            caption=B.sota_caption(h, len(data.subjects),
                                   int(data.split.test.size), args.alpha),
            label=f"tab:sota_ph{h}"))
        print(f"\n=== PH = {h} min ===")
        print(B.text_table(results, B.SOTA_COLUMNS[:8], h))
        print()
        print(B.text_table(results, B.SOTA_COLUMNS[8:], h))
    for h, one in zip(args.horizons, tex):
        # Mirrors rebuild_paper_tables.rebuild_benchmarks: the manuscript
        # \inputs the per-horizon files so the PH = 60 table can sit in an
        # appendix, while the combined file stays for package_release.py.
        B.write_latex(os.path.join(args.out, f"table_sota_h{h}.tex"), one)
    B.write_latex(os.path.join(args.out, "table_sota_comparison.tex"),
                  "\n\n".join(tex))


if __name__ == "__main__":
    main()
