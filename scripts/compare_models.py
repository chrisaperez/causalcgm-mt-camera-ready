"""TCN vs LightGBM under identical splits, with conformal intervals on both.

    python scripts/compare_models.py --synthetic --epochs 15 --folds 2

Two things are being measured:

  1. Does the TCN beat a well-tuned tabular baseline? Gradient boosting on lag
     features is genuinely competitive at a 30-minute horizon, so this is a real
     test rather than a formality.

  2. Are the conformal intervals valid, and are they valid WHERE IT MATTERS?
     Marginal coverage is the easy part. The report also stratifies by clinical
     range, because 90% marginal coverage is fully compatible with 60% coverage
     in the hypoglycemic range -- and an interval that fails exactly when the
     patient is dropping is worse than no interval at all.

Both models see byte-identical fit / validation / test patient partitions: the
LightGBM fold runner reuses `cfg.train.seed` and `inner_val_frac` to reproduce
the same inner split as `train.run_fold`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LightGBM must be importable alongside torch; see scripts/fix_openmp.sh.
from causalcgm_mt.baselines.lgbm_baseline import run_lgbm_fold  # noqa: E402
from causalcgm_mt.config import Config  # noqa: E402
from causalcgm_mt.conformal import (  # noqa: E402
    MondrianConformal, NormalizedConformal, SplitConformal,
    rate_of_change_difficulty,
)
from causalcgm_mt.data import build_windows, split_by_patient  # noqa: E402
from causalcgm_mt.simulate import simulate_cohort  # noqa: E402
from causalcgm_mt.train import run_fold  # noqa: E402


def tcn_predictions(fold_result, ws, idx, cfg):
    """Re-run the trained TCN over an index set to get aligned predictions."""
    from torch.utils.data import DataLoader

    from causalcgm_mt.models import build_model, load_state_dict_compat
    from causalcgm_mt.train import WindowDataset, pick_device, predict

    model, _ = build_model(ws.x.shape[-1], cfg.model)
    load_state_dict_compat(model, fold_result.state_dict)
    device = pick_device()
    model = model.to(device)

    dl = DataLoader(
        WindowDataset(
            fold_result.scaler.transform(ws.x[idx]),
            ws.y_reg[idx],
            fold_result.labeler(ws.y_min[idx]),
            ws.last_bg[idx],
        ),
        batch_size=512, shuffle=False,
    )
    return predict(model, dl, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="runs/compare")
    args = ap.parse_args()

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    elif args.synthetic:
        df = simulate_cohort(n_patients=args.patients, days=args.days)
        print("WARNING: synthetic data. Validates plumbing, not clinical performance.")
    else:
        ap.error("pass --csv or --synthetic")

    cfg = Config(out_dir=args.out)
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, max_epochs=args.epochs,
                                       n_folds=args.folds, num_workers=0))

    ws = build_windows(df, cfg.data)
    print(f"{len(ws)} windows | {len(set(ws.patient))} patients | "
          f"hypo prevalence {ws.y_cls_hard.mean():.2%}\n")

    pats = sorted({str(p) for p in ws.patient})
    n_test = max(1, len(pats) // args.folds)

    rows: dict[str, list] = {"TCN": [], "LightGBM": []}
    cov_rows = []

    for k in range(args.folds):
        test_pat = np.array(pats[k * n_test : (k + 1) * n_test], dtype=object)
        train_pat = np.array([p for p in pats if p not in set(test_pat)], dtype=object)
        if test_pat.size == 0:
            continue

        print(f"===== fold {k}: train {len(train_pat)} / test {len(test_pat)} patients =====")

        fr = run_fold(ws, cfg, train_pat, test_pat, k)
        rows["TCN"].append(fr.report)

        _, lgbm_rep = run_lgbm_fold(ws, cfg, train_pat, test_pat, k)
        print(f"[fold {k}] LightGBM\n{lgbm_rep}")
        rows["LightGBM"].append(lgbm_rep)

        # ---- conformal calibration -----------------------------------------
        # Calibrate on the inner VALIDATION patients: disjoint from both the
        # fit patients (so the model did not train on them) and the test
        # patients (so no test information reaches qhat).
        rng = np.random.default_rng(cfg.train.seed + k)
        perm = rng.permutation(train_pat)
        n_val = max(1, int(round(len(perm) * cfg.train.inner_val_frac)))
        val_pat = perm[:n_val]

        i_cal = split_by_patient(ws, val_pat)
        i_test = split_by_patient(ws, test_pat)

        cal_pred, _ = tcn_predictions(fr, ws, i_cal, cfg)
        test_pred, _ = tcn_predictions(fr, ws, i_test, cfg)

        conf = SplitConformal.calibrate(
            ws.y_reg[i_cal], cal_pred, args.alpha, patient_calib=ws.patient[i_cal])
        print(f"\n{conf}")
        rep = conf.evaluate(ws.y_reg[i_test], test_pred, ws.patient[i_test])
        print(rep)

        # Locally adaptive variant: difficulty from the input window only.
        sig_cal = rate_of_change_difficulty(ws, i_cal)
        sig_test = rate_of_change_difficulty(ws, i_test)
        nconf = NormalizedConformal.calibrate(
            ws.y_reg[i_cal], cal_pred, sig_cal, args.alpha,
            patient_calib=ws.patient[i_cal])
        nrep = nconf.evaluate(ws.y_reg[i_test], test_pred, sig_test, ws.patient[i_test])
        print("\n[normalised]")
        print(nrep)

        # Group-conditional variant: a separate qhat per clinical glycemic zone.
        # Same calibration rows, same test rows, so the three arms differ only in
        # how the interval is constructed -- which is what makes the comparison
        # readable.
        mconf = MondrianConformal.calibrate(
            ws.y_reg[i_cal], cal_pred, args.alpha, patient_calib=ws.patient[i_cal])
        print(f"\n{mconf}")
        mrep = mconf.evaluate(ws.y_reg[i_test], test_pred, ws.patient[i_test])
        print(mrep)
        cov_rows.append((rep, nrep, mrep))
        print()

    # ---- model comparison ---------------------------------------------------
    print("=" * 72)
    print(f"{'metric':>18} {'TCN':>12} {'LightGBM':>12} {'winner':>10}")
    print("-" * 72)
    for m, better in (("rmse", "lower"), ("mae", "lower"), ("mard", "lower"),
                      ("time_lag_min", "lower"), ("roc_auc", "higher"),
                      ("pr_auc", "higher"), ("sens_at_5pct_fpr", "higher")):
        a = np.nanmean([getattr(r, m) for r in rows["TCN"]])
        b = np.nanmean([getattr(r, m) for r in rows["LightGBM"]])
        win = ("TCN" if (a < b) == (better == "lower") else "LightGBM") if a != b else "tie"
        print(f"{m:>18} {a:>12.4f} {b:>12.4f} {win:>10}")

    print("\n" + "=" * 72)
    print(f"Conformal validity @ alpha={args.alpha} (target {1 - args.alpha:.0%})")
    print("-" * 72)
    print(f"  {'variant':<16} {'marginal':>9} {'hypo<70':>9} {'severe<54':>10} "
          f"{'width':>8} {'worst pt':>9}")
    for label, i in (("constant width", 0), ("normalised", 1), ("mondrian", 2)):
        cov = np.mean([c[i].empirical_coverage for c in cov_rows])
        wid = np.mean([c[i].mean_width for c in cov_rows])
        worst = np.nanmin([c[i].worst_patient_coverage for c in cov_rows])
        # Conditional coverage is the number that decides whether the interval is
        # usable for hypoglycemia alerting; the marginal one never was.
        hyp = np.mean([c[i].stratum(0).coverage for c in cov_rows
                       if c[i].stratum(0) is not None])
        sev = np.mean([c[i].coverage_by_range["severe hypo (<54)"][0]
                       for c in cov_rows
                       if "severe hypo (<54)" in c[i].coverage_by_range])
        print(f"  {label:<16} {cov:>8.2%} {hyp:>9.2%} {sev:>10.2%} "
              f"{wid:>7.2f}  {worst:>8.2%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
