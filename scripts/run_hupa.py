"""HUPA-UCM multimodal run, with the unimodal ablation as the control.

    python scripts/run_hupa.py --synthetic --epochs 10 --folds 2
    python scripts/run_hupa.py --dir data/HUPA --epochs 40

The ablation is the point. A fusion model that beats nothing is not evidence
that wearables help -- it is evidence that a bigger model was trained. So both
arms run on IDENTICAL windows, splits, scaler policy, soft labels and metrics;
the only difference is whether the aux tensor is present and which model
consumes it. `run_fold` detects multimodality from the window set precisely so
that no other variable can drift between arms.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config
from causalcgm_mt.data import WindowSet
from causalcgm_mt.hupa_loader import (
    HupaConfig, HupaSchema, build_multimodal_windows, load_hupa_files, to_canonical,
)
from causalcgm_mt.simulate import simulate_hupa_cohort
from causalcgm_mt.train import run_fold
from scripts._cli import HUPA_HINT, missing_input


def unimodal_view(mm) -> WindowSet:
    """Drop the aux stream, keeping every window identical.

    Deriving the control arm from the treatment arm's window set -- rather than
    rebuilding it -- guarantees the two arms score the same rows. Rebuilding
    would silently differ, because `min_aux_observed_frac` drops windows that a
    CGM-only pipeline would have kept.
    """
    return WindowSet(
        x=mm.x, y_reg=mm.y_reg, y_cls_hard=mm.y_cls_hard, y_min=mm.y_min,
        patient=mm.patient, t_index=mm.t_index, t_time=mm.t_time, last_bg=mm.last_bg,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="directory of HUPA CSVs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--patients", type=int, default=10)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", type=str, default="runs/hupa")
    args = ap.parse_args()

    if args.dir:
        with missing_input(HUPA_HINT):
            df = to_canonical(load_hupa_files(args.dir, HupaSchema()), HupaSchema())
    elif args.synthetic:
        df = to_canonical(simulate_hupa_cohort(n_patients=args.patients, days=args.days))
        print("WARNING: synthetic HUPA-shaped data. Exercises synchronisation; "
              "not clinical evidence.")
    else:
        ap.error("pass --dir or --synthetic")

    cfg = Config(out_dir=args.out)
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, max_epochs=args.epochs,
                                       n_folds=args.folds, num_workers=0))

    hcfg = HupaConfig()
    mm = build_multimodal_windows(df, cfg.data, hcfg)
    print(f"{len(mm)} windows | {len(set(mm.patient))} patients | "
          f"hypo prevalence {mm.y_cls_hard.mean():.2%} | "
          f"mean aux coverage {mm.aux_frac.mean():.1%}")

    uni = unimodal_view(mm)
    pats = sorted({str(p) for p in mm.patient})
    n_test = max(1, len(pats) // args.folds)

    results: dict[str, list] = {"unimodal": [], "multimodal": []}
    for k in range(args.folds):
        test_pat = np.array(pats[k * n_test : (k + 1) * n_test], dtype=object)
        train_pat = np.array([p for p in pats if p not in set(test_pat)], dtype=object)
        if len(test_pat) == 0:
            continue
        for name, wsx in (("unimodal", uni), ("multimodal", mm)):
            print(f"\n=== fold {k} / {name} ===")
            results[name].append(run_fold(wsx, cfg, train_pat, test_pat, k))

    print("\n" + "=" * 68)
    print(f"{'metric':>18} {'unimodal':>12} {'multimodal':>12} {'delta':>10}")
    print("-" * 68)
    for m, better in (("rmse", "lower"), ("mae", "lower"), ("mard", "lower"),
                      ("time_lag_min", "lower"), ("roc_auc", "higher"),
                      ("pr_auc", "higher"), ("sens_at_5pct_fpr", "higher")):
        u = np.nanmean([getattr(r.report, m) for r in results["unimodal"]])
        v = np.nanmean([getattr(r.report, m) for r in results["multimodal"]])
        d = v - u
        good = (d < 0) if better == "lower" else (d > 0)
        flag = "+" if good and abs(d) > 1e-9 else ("-" if abs(d) > 1e-9 else " ")
        print(f"{m:>18} {u:>12.4f} {v:>12.4f} {d:>+10.4f} {flag}")
    print("=" * 68)
    print("Folds are few and patients are synthetic: treat any delta as a smoke\n"
          "signal, not a result. The paired comparison only becomes meaningful\n"
          "with real HUPA-UCM data and enough folds to estimate variance.")


if __name__ == "__main__":
    main()
