"""Train a demo checkpoint on the synthetic cohort, for the dashboard.

The cohort spec (n_patients / days / seed) is written into the checkpoint so the
dashboard can regenerate byte-identical data and therefore show the model the
exact windows it was scored on. Without that, the dashboard would silently be
evaluating on a different cohort than the one reported.

Synthetic data only -- see simulate.py. Numbers here validate plumbing.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config
from causalcgm_mt.data import build_windows
from causalcgm_mt.simulate import simulate_cohort
from causalcgm_mt.train import run_fold

COHORT = dict(n_patients=14, days=8.0, seed=0)
OUT = "runs/demo"


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    df = simulate_cohort(**COHORT)
    cfg = Config(out_dir=OUT)
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, max_epochs=epochs))

    ws = build_windows(df, cfg.data)
    print(f"{len(ws)} windows | {len(set(ws.patient))} patients | "
          f"hypo prevalence {ws.y_cls_hard.mean():.2%}")

    pats = sorted({str(p) for p in ws.patient})
    train_pat, test_pat = pats[:10], pats[10:]

    r = run_fold(ws, cfg, train_pat, test_pat, fold=0)

    os.makedirs(OUT, exist_ok=True)
    torch.save(
        {
            # `export.REQUIRED_CHECKPOINT_KEYS` plus this driver's own extras.
            # `multimodal` is declared rather than inferred so
            # `load_and_export` dispatches on the checkpoint's own statement of
            # what it is, not on a guess from the key set.
            "state_dict": r.state_dict,
            "criterion_state_dict": r.criterion_state_dict,
            "scaler": r.scaler,
            "aux_scaler": r.aux_scaler,
            "labeler": r.labeler,
            "multimodal": False,
            "aux_channels": 0,
            "test_patients": r.test_patients,
            "train_patients": train_pat,
            "in_channels": int(ws.x.shape[-1]),
            "cohort": COHORT,
            "config": dataclasses.asdict(cfg),
            "report": r.report.as_dict(),
        },
        os.path.join(OUT, "demo.pt"),
    )
    print(f"\nsaved {OUT}/demo.pt")


if __name__ == "__main__":
    main()
