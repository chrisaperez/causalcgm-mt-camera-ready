"""Stability of severe-hypoglycemia conformal coverage across calibration draws.

    python scripts/conformal_calibration_bootstrap.py \
        --seeds-dir runs/paper_results/benchmarks_seeds --arm "CausalCGM-MT (fusion)"

A single calibration split gives one realisation of each interval's coverage.
In the < 54 mg/dL cell that realisation rests on ~100 autocorrelated test
windows and on a quantile fitted from autocorrelated calibration windows, so it
is not a stable estimate. This script re-calibrates split, normalised and
Mondrian conformal on block-bootstrap resamples of the calibration region and
re-scores the FIXED test windows each time.

Blocks are (subject, calendar day) on the calibration split: resampling whole
subject-days keeps the within-day autocorrelation inside a block, which a
row-level bootstrap would destroy and thereby understate the variance. The
model, its predictions and the test rows never change -- only which calibration
evidence the quantiles are fitted on.

Every construction is the repository's own (`causalcgm_mt.conformal`), called
exactly as `benchmark._conformal_block` calls it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.conformal import (  # noqa: E402
    MondrianConformal, NormalizedConformal, SplitConformal,
)

BANDS = {"severe_lt54": (0.0, 54.0), "hypo_lt70": (0.0, 70.0), "all": (0.0, np.inf)}


def _cov(y, lo, hi, band):
    a, b = BANDS[band]
    m = (y >= a) & (y < b)
    return float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()) if m.any() else float("nan")


def _width(lo, hi):
    return float(np.mean(hi - lo))


def one_draw(z, k, alpha, rows):
    yc, pc = z["calib_y_reg"][rows, k], z["calib_bg_hat"][rows, k]
    sc, pat_c = z["calib_sigma"][rows], z["calib_patient"][rows]
    yt, pt, st = z["y_reg"][:, k], z["bg_hat"][:, k], z["test_sigma"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SplitConformal.calibrate(yc, pc, alpha, patient_calib=pat_c)
        lo, hi = s.predict_interval(pt)
        out["split"] = (lo, hi)
        n = NormalizedConformal.calibrate(yc, pc, sc, alpha, patient_calib=pat_c)
        lo, hi = n.predict_interval(pt, st)
        out["normalized"] = (lo, hi)
        m = MondrianConformal.calibrate(yc, pc, alpha, patient_calib=pat_c)
        lo, hi = m.predict_interval(pt)
        out["mondrian"] = (lo, hi)
    return {name: {**{f"cov_{b}": _cov(yt.astype(float), lo, hi, b) for b in BANDS},
                   "width": _width(lo, hi)}
            for name, (lo, hi) in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", default="runs/paper_results/benchmarks_seeds")
    ap.add_argument("--arm", default="CausalCGM-MT (fusion)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--horizon-index", type=int, default=0, help="0 = 30 min")
    ap.add_argument("--out", default="runs/paper_results/conformal_bootstrap.json")
    args = ap.parse_args()

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in args.arm)
    files = sorted(glob.glob(os.path.join(args.seeds_dir, "seed*", "predictions", f"{safe}.npz")))
    if not files:
        raise SystemExit(f"no prediction files for {args.arm!r} under {args.seeds_dir}")

    rng = np.random.default_rng(20261006)
    report: dict = {"arm": args.arm, "alpha": args.alpha, "n_boot": args.n_boot,
                    "block": "(calibration subject, calendar day)",
                    "horizon_index": args.horizon_index, "seeds": {}}
    for f in files:
        seed = os.path.basename(os.path.dirname(os.path.dirname(f)))
        z = dict(np.load(f, allow_pickle=True))
        if "calib_y_reg" not in z:
            raise SystemExit(f"{f} has no calibration arrays; re-run the benchmark "
                             f"with the patched save_predictions.")
        k = args.horizon_index
        ok = np.isfinite(z["calib_bg_hat"][:, k])
        day = z["calib_t_time"].astype("datetime64[ns]").astype("datetime64[D]")
        block_id = np.array([f"{p}|{d}" for p, d in zip(z["calib_patient"], day)])
        blocks = [np.flatnonzero((block_id == b) & ok) for b in np.unique(block_id[ok])]

        point = one_draw(z, k, args.alpha, np.flatnonzero(ok))
        draws = {name: {key: [] for key in point[name]} for name in point}
        for _ in range(args.n_boot):
            pick = rng.integers(0, len(blocks), size=len(blocks))
            rows = np.concatenate([blocks[i] for i in pick])
            res = one_draw(z, k, args.alpha, rows)
            for name in res:
                for key, v in res[name].items():
                    draws[name][key].append(v)

        y_test = z["y_reg"][:, k].astype(float)
        seed_rep = {"n_calib_blocks": len(blocks),
                    "n_test_severe": int((y_test < 54).sum()),
                    "n_test_hypo": int((y_test < 70).sum())}
        for name in point:
            seed_rep[name] = {}
            for key in point[name]:
                a = np.asarray(draws[name][key], dtype=float)
                seed_rep[name][key] = {
                    "point": point[name][key],
                    "boot_median": float(np.nanmedian(a)),
                    "boot_p2_5": float(np.nanpercentile(a, 2.5)),
                    "boot_p97_5": float(np.nanpercentile(a, 97.5)),
                    **({"p_below_nominal": float(np.nanmean(a < 1 - args.alpha))}
                       if key.startswith("cov_") else {}),
                }
        report["seeds"][seed] = seed_rep
        sv = {n: seed_rep[n]["cov_severe_lt54"] for n in point}
        print(f"{seed}: <54 coverage (point [boot 95%]) " + " | ".join(
            f"{n} {v['point']:.3f} [{v['boot_p2_5']:.3f}, {v['boot_p97_5']:.3f}]"
            for n, v in sv.items()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
