"""End-to-end driver: build windows -> patient-wise CV -> personalisation.

  python scripts/run_experiment.py --synthetic --epochs 3
  python scripts/run_experiment.py --csv data/mycohort.csv --horizon 30 --out runs/mycohort
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config
from causalcgm_mt.data import build_windows
from causalcgm_mt.extension import run_personalization
from causalcgm_mt.models import build_model, load_state_dict_compat
from causalcgm_mt.simulate import simulate_cohort
from causalcgm_mt.train import cross_validate
from scripts._cli import CSV_HINT, missing_input


def criterion_for_fold(criterion_state: dict, cfg: Config):
    """The fold's OWN criterion, with its fitted state restored.

    Thin wrapper over `models.criterion_from_state`, kept as a named function
    because this driver's error is worth phrasing in terms of the fold. The
    policy itself lives in one place now: every driver that rebuilds a criterion
    (this one, `run_metabonet_benchmarks.py`, `run_ohio_personalization_sweep.py`)
    goes through the same helper, so the hazard cannot be handled three ways.
    """
    from causalcgm_mt.models import criterion_from_state

    return criterion_from_state(criterion_state, cfg.model,
                                context="this cross-validation fold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="long-format CSV: patient_id, timestamp, glucose")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None, help="PH in minutes")
    ap.add_argument("--adapt", type=str, default="lora",
                    choices=["lora", "bn_adapt", "head_only", "full", "none"])
    ap.add_argument("--out", type=str, default="runs/default")
    args = ap.parse_args()

    if args.csv:
        # OhioT1DM does not come through here. This driver cross-validates by
        # PATIENT, whereas Ohio ships a fixed temporal split within patient --
        # running it here would silently substitute a different protocol for
        # the published one. Fail with the right command instead of emitting a
        # UnicodeDecodeError from read_csv on a .pt file.
        if args.csv.endswith(".pt") or "ohio" in os.path.basename(args.csv).lower():
            ap.error(
                f"{args.csv} looks like an OhioT1DM artefact, not a long-format CSV.\n"
                f"  Ohio uses its own official train/test split; train it with:\n"
                f"      python scripts/train_ohio.py --data {args.csv} --epochs 50\n"
                f"  This driver cross-validates by patient and expects a CSV with "
                f"columns patient_id, timestamp, glucose."
            )
        with missing_input(CSV_HINT):
            df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    elif args.synthetic:
        df = simulate_cohort(n_patients=args.patients, days=args.days)
        print("WARNING: synthetic data. Numbers are for pipeline validation only.")
    else:
        ap.error("pass --csv or --synthetic")

    cfg = Config(out_dir=args.out)
    over = {}
    if args.epochs:  over["train"] = dataclasses.replace(cfg.train, max_epochs=args.epochs)
    if args.folds:   over["train"] = dataclasses.replace(over.get("train", cfg.train), n_folds=args.folds)
    if args.horizon: over["data"] = dataclasses.replace(cfg.data, horizon_min=args.horizon)
    if args.adapt != "none":
        over["adapt"] = dataclasses.replace(cfg.adapt, method=args.adapt)
    cfg = dataclasses.replace(cfg, **over)

    print(f"Windowing: H={cfg.data.history_steps} steps, PH={cfg.data.horizon_steps} steps")
    ws = build_windows(df, cfg.data)
    print(f"  {len(ws)} windows | {len(set(ws.patient))} patients "
          f"| hypo prevalence {ws.y_cls_hard.mean():.2%}")

    results = cross_validate(ws, cfg)

    if args.adapt != "none":
        print(f"\n=== Personalisation ({cfg.adapt.method}) on fold 0 test patients ===")
        r0 = results[0]
        model, _fresh = build_model(ws.x.shape[-1], cfg.model)
        load_state_dict_compat(model, r0.state_dict)
        criterion = criterion_for_fold(r0.criterion_state_dict, cfg)
        run_personalization(
            model, ws, r0.test_patients, r0.scaler, r0.labeler, criterion, cfg
        )


if __name__ == "__main__":
    main()
