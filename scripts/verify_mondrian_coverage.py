"""Verify Mondrian conformal coverage on a real held-out test split.

    python scripts/verify_mondrian_coverage.py --root OhioT1DM
    python scripts/verify_mondrian_coverage.py --synthetic     # no data needed

WHAT THIS DEMONSTRATES, AND WHY IT IS NOT A UNIT TEST
-----------------------------------------------------
The unit tests establish that the construction is correct on data whose
generating process is known. This script establishes that the FAILURE it fixes
is real on this pipeline's actual windows and actual model: marginal split
conformal reaches its nominal coverage overall while covering the hypoglycemic
range at ~60-70%, and Mondrian repairs that at a small cost in mean width.

PROTOCOL (the part that makes the numbers mean anything)
--------------------------------------------------------
    fit    -> model                        (LightGBM baseline; fast, and the
                                            conformal wrapper is model-agnostic
                                            by construction)
    val    -> qhat                          (calibration; disjoint from fit, and
                                            chronologically embargoed from it)
    test   -> coverage                      (the official OhioT1DM test files,
                                            unresampled)

No test label touches the model or the calibration at any point: `qhat` is a
quantile of |y - yhat| on the VALIDATION rows and is applied to the test rows as
a fixed offset. That is the whole leak surface, and it is three lines long.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.baselines.lgbm_baseline import LGBMBaseline
from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.conformal import (
    N_STRATA, MondrianConformal, SplitConformal, clinical_stratum,
)


def _ohio_windows(root: str, cfg: DataConfig):
    from causalcgm_mt.hupa_loader import HupaConfig
    from causalcgm_mt.ohio_loader import (
        OhioConfig, build_ohio_windows, discover_records, official_temporal_split,
    )
    ws = build_ohio_windows(discover_records(root), cfg, HupaConfig(), OhioConfig(),
                            horizons_min=(cfg.horizon_min,))
    return ws, official_temporal_split(ws, cfg, val_frac=0.2)


def _synthetic_windows(cfg: DataConfig):
    """Fallback cohort so the check runs without the licensed dataset.

    Patient-disjoint rather than temporal: with no official files to respect,
    holding out whole subjects is the harder and more informative split.
    """
    from causalcgm_mt.data import build_windows, split_by_patient
    from causalcgm_mt.simulate import simulate_cohort

    ws = build_windows(simulate_cohort(n_patients=18, days=12.0, seed=7), cfg)
    pats = sorted({str(p) for p in ws.patient})

    class _Split:
        # Four disjoint patient groups, mirroring the four regions of
        # `OhioSplit`: selection and calibration must not share rows, or qhat
        # is a quantile of residuals from the rows the model was stopped on.
        fit = split_by_patient(ws, pats[:9])
        val = split_by_patient(ws, pats[9:12])
        calib = split_by_patient(ws, pats[12:15])
        test = split_by_patient(ws, pats[15:])
    return ws, _Split()


def _print_report(name: str, rep) -> None:
    print(f"\n--- {name} ---")
    print(f"  marginal coverage : {rep.empirical_coverage:.2%}   "
          f"(target {rep.target_coverage:.0%})   "
          f"mean width {rep.mean_width:.1f} mg/dL")
    for k in range(N_STRATA):
        c = rep.stratum(k)
        if c is None:
            continue
        flag = "  <-- UNDERCOVERS" if c.coverage < rep.target_coverage - 0.02 else ""
        print(f"    {c.name:<22} {c.coverage:7.2%}  width {c.mean_width:6.1f}  "
              f"(n={c.n}){flag}")
    for label, (cov, n) in rep.coverage_by_range.items():
        if label.startswith(("severe hypo", "hypo")):
            print(f"      [{label}] {cov:.2%} (n={n})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="OhioT1DM")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the simulated cohort instead of OhioT1DM")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    cfg = DataConfig(history_min=120, horizon_min=args.horizon,
                     stride_steps=args.stride)
    use_synth = args.synthetic or not os.path.isdir(args.root)
    if use_synth and not args.synthetic:
        print(f"{args.root} not found; falling back to the simulated cohort.")

    ws, split = (_synthetic_windows(cfg) if use_synth
                 else _ohio_windows(args.root, cfg))
    print(f"windows: fit {split.fit.size}  val {split.val.size}  "
          f"calib {split.calib.size}  test {split.test.size}  |  hypo prevalence "
          f"{ws.y_cls_hard[split.test].mean():.2%}")
    if split.calib.size == 0:
        print("no calibration rows in this split; conformal needs a region "
              "disjoint from the early-stopping set. Rebuild with "
              "official_temporal_split(calib_frac=0.5).")
        return 2

    model = LGBMBaseline(cfg=Config(data=cfg), n_estimators=300).fit(
        ws, split.fit, split.val)
    # Early stopping read `split.val`; conformal reads `split.calib`. Using one
    # region for both makes qhat a quantile of residuals from the rows the
    # booster was stopped to fit best, which undercovers -- and would make this
    # script report a coverage gap that is an artefact of its own protocol.
    pred_cal, _ = model.predict(ws, split.calib)
    pred_test, _ = model.predict(ws, split.test)
    y_cal, y_test = ws.y_reg[split.calib], ws.y_reg[split.test]
    print(f"model: test RMSE {np.sqrt(np.mean((y_test - pred_test) ** 2)):.2f} mg/dL")

    marginal = SplitConformal.calibrate(y_cal, pred_cal, args.alpha,
                                        patient_calib=ws.patient[split.calib])
    mondrian = MondrianConformal.calibrate(y_cal, pred_cal, args.alpha,
                                           patient_calib=ws.patient[split.calib])
    print(f"\ncalibrated on {len(marginal.calibrated_on)} patients, "
          f"n={marginal.n_calib} (n_eff {marginal.n_effective:.0f})")
    print(mondrian)

    pat = ws.patient[split.test]
    _print_report("MARGINAL split conformal", marginal.evaluate(y_test, pred_test, pat))
    _print_report("MONDRIAN, naive predicted stratum (ablation)",
                  mondrian.evaluate(y_test, pred_test, pat, expand=False))
    rep_m = mondrian.evaluate(y_test, pred_test, pat)
    _print_report("MONDRIAN, label-conditional hull (default)", rep_m)
    _print_report("MONDRIAN, oracle true stratum (upper bound)",
                  mondrian.evaluate(y_test, pred_test, pat,
                                    stratum=clinical_stratum(y_test)))

    # The hull spans the gap between two disjoint pieces whenever one stratum's
    # qhat is much larger than another's, so quoting hull width alone overstates
    # what the method actually claims. The exact region is smaller.
    measure = mondrian.set_measure(pred_test)
    print(f"\nexact label-conditional region: mean measure {measure.mean():.1f} "
          f"mg/dL vs hull width {rep_m.mean_width:.1f} mg/dL")
    print(f"conditionally valid (every stratum within 2 points of "
          f"{1 - args.alpha:.0%}): {rep_m.is_conditionally_valid()}")

    # The theorem the construction rests on, checked on this split rather than
    # only in the unit tests.
    lo_h, hi_h = mondrian.predict_interval(pred_test)
    lo_o, hi_o = mondrian.predict_interval(pred_test, stratum=clinical_stratum(y_test))
    cov_h = (y_test >= lo_h) & (y_test <= hi_h)
    cov_o = (y_test >= lo_o) & (y_test <= hi_o)
    print(f"hull covers every point the oracle covers: {bool((cov_h | ~cov_o).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
