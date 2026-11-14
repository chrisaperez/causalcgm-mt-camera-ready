"""Export a trained checkpoint to ONNX, gated on parity + causality.

    python scripts/export_onnx.py [checkpoint] [output.onnx]

Exits non-zero if the exported graph disagrees with PyTorch or fails the
causality probe, so this is safe to run in CI ahead of a release.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.export import load_and_export


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a trained checkpoint to ONNX, gated on parity + causality."
    )
    ap.add_argument(
        "checkpoint",
        nargs="?",
        default="runs/demo/demo.pt",
        help="input PyTorch checkpoint path (default: runs/demo/demo.pt)",
    )
    ap.add_argument(
        "output",
        nargs="?",
        default="runs/demo/causalcgm_mt.onnx",
        help="output ONNX graph path (default: runs/demo/causalcgm_mt.onnx)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint at {args.checkpoint}. Run: python scripts/train_demo.py 30")
        return 1

    try:
        print(load_and_export(args.checkpoint, args.output))
    except RuntimeError as e:
        print(f"EXPORT REJECTED\n{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

