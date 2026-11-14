"""Preprocess OhioT1DM (2018 + 2020) into PyTorch-ready tensors.

    python scripts/prepare_ohio.py --root OhioT1DM --out runs/ohio
    python scripts/prepare_ohio.py --root OhioT1DM --horizons 30 60 --history 120
    python scripts/prepare_ohio.py --root OhioT1DM --held-out 2018_559 2020_540

The default protocol is the official one: train on the `*-ws-training.xml`
files, evaluate on `*-ws-testing.xml`, with the inner validation split taken as
a chronological tail of the training files and an embargo at the boundary.
`--held-out` switches to the unseen-subject protocol instead; see
`ohio_loader.held_out_patient_split` for why the two are not interchangeable.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import DataConfig
from causalcgm_mt.hupa_loader import HupaConfig
from causalcgm_mt.ohio_loader import (
    OhioConfig,
    build_ohio_windows,
    describe,
    collect_reference_pairs,
    discover_records,
    held_out_patient_split,
    make_ohio_loaders,
    official_temporal_split,
    save_ohio_tensors,
)
from scripts._cli import OHIO_HINT, missing_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="OhioT1DM")
    ap.add_argument("--out", default="runs/ohio")
    ap.add_argument("--subsets", nargs="+", default=["2018", "2020"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60],
                    help="prediction horizons in minutes")
    ap.add_argument("--primary", type=int, default=30,
                    help="horizon mirrored into y_reg/y_cls_hard")
    ap.add_argument("--history", type=int, default=120, help="history length, minutes")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--held-out", nargs="*", default=None,
                    help="patient ids (e.g. 2018_559) to hold out entirely")
    ap.add_argument("--carbs-from-bwz", action="store_true",
                    help="also take carbs from bolus-wizard entries (deduplicated)")
    ap.add_argument("--band-mode", choices=["fixed", "mard"], default="fixed",
                    help="soft-label band width. 'mard' derives it from the "
                         "fold's CGM error, measured against the paired "
                         "finger-stick reference in the same files (~10.5%% "
                         "MARD, a +/-7.4 mg/dL band), and floors it at the "
                         "device specification if the cohort has no reference. "
                         "'fixed' remains the default because it is what the "
                         "published tables were produced under, not because "
                         "'mard' degenerates -- Ablation E sweeps the two.")
    ap.add_argument("--soft-band", type=float, default=10.0,
                    help="band half-width in mg/dL when --band-mode=fixed")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = DataConfig(history_min=args.history, horizon_min=args.primary,
                     stride_steps=args.stride, band_mode=args.band_mode,
                     soft_band_mgdl=args.soft_band)
    ocfg = OhioConfig(carbs_from_bwz=args.carbs_from_bwz)

    with missing_input(OHIO_HINT):
        records = discover_records(args.root, args.subsets)
    ws = build_ohio_windows(records, cfg, HupaConfig(), ocfg, horizons_min=args.horizons)
    print(describe(ws))

    if args.held_out:
        split = held_out_patient_split(ws, args.held_out, cfg, val_frac=args.val_frac)
        print(f"Protocol: held-out subjects {sorted(args.held_out)}")
    else:
        split = official_temporal_split(ws, cfg, val_frac=args.val_frac)
        print("Protocol: official OhioT1DM train/test split")
    print(split.summary(ws))

    loaders, pre = make_ohio_loaders(
        ws, split, cfg, num_workers=0,
        reference_pairs=collect_reference_pairs(records, cfg))
    lab = pre.labeler
    print(f"MARD (fit only): {pre.mard_proxy:.4f} "
          f"[{pre.noise.source if pre.noise else 'n/a'}] | "
          f"band: {lab.band:.2f} mg/dL [{lab.band_source}] | "
          f"pos_weight: { {k: round(v, 2) for k, v in pre.pos_weight.items()} }")
    for name, dl in loaders.items():
        print(f"  loader[{name}]: {0 if dl is None else len(dl)} batches")

    if not args.no_save:
        save_ohio_tensors(ws, split, cfg, args.out, pre=pre)


if __name__ == "__main__":
    main()
