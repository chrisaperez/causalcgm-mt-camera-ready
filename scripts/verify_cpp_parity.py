"""Cross-language parity: C++ runner vs the Python pipeline.

The ONNX parity check in export.py compares two *graphs*. It cannot see the
hand-written C++ reimplementation of `causal_features` in the runner, and a
divergence there is textbook train/serve skew: the deployed model silently
receives different features from the ones it was trained on, while every
graph-level test still passes.

This script closes that gap by driving the runner's --verify mode with a set of
histories and diffing both the feature tensor and the final outputs against the
Python implementation, element by element.

    python scripts/verify_cpp_parity.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config
from causalcgm_mt.data import _causal_gap_fill, causal_features
from causalcgm_mt.export import DeployableCausalCGMMT
from causalcgm_mt.models import build_model, load_state_dict_compat

RUNNER = "cpp/build/causalcgm_mt_runner"
ONNX = "runs/demo/causalcgm_mt.onnx"
CKPT = "runs/demo/demo.pt"

# Feature values are O(100) mg/dL, so tolerances are absolute in each output's
# own units rather than a single shared epsilon.
ATOL_FEATURE = 1e-3   # mg/dL
ATOL_BG = 1e-2        # mg/dL -- far below CGM resolution
ATOL_RISK = 1e-4      # probability


def run_cpp(history: np.ndarray, observed: np.ndarray | None = None,
            onnx_path: str = ONNX, max_gap_steps: int = 3
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive the runner's --verify mode.

    `observed` is optional and, when given, is appended after the history so the
    runner builds the mask channel from it. Without it the parity claim only
    covered fully-observed windows -- both sides hard-coded a mask of ones, so a
    divergence in how partial observation is encoded was unobservable.
    """
    values = list(history) if observed is None else [*history, *observed]
    payload = "\n".join(f"{v:.9g}" for v in values)
    proc = subprocess.run(
        [RUNNER, "--verify", onnx_path, str(max_gap_steps)],
        input=payload, capture_output=True, text=True, check=True,
    )
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "features", proc.stdout

    feats, bg, risk = [], None, None
    for ln in lines[1:]:
        if ln.startswith("bg_hat"):
            bg = np.array([float(t) for t in ln.split()[1:]], dtype=np.float64)
        elif ln.startswith("risk"):
            risk = np.array([float(t) for t in ln.split()[1:]], dtype=np.float64)
        else:
            feats.append([float(t) for t in ln.split()])
    return np.array(feats, dtype=np.float32), bg, risk


def main() -> int:
    for p in (RUNNER, ONNX, CKPT):
        if not os.path.exists(p):
            print(f"missing {p}")
            return 1

    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = Config.from_dict(ck["config"])
    model, _ = build_model(ck["in_channels"], cfg.model)
    load_state_dict_compat(model, ck["state_dict"])
    net = DeployableCausalCGMMT(model.eval(), ck["scaler"]).eval()

    H = cfg.data.history_steps
    rng = np.random.default_rng(0)

    cases = {
        "rising ramp": np.linspace(90, 180, H),
        "falling into hypo": np.linspace(160, 58, H),
        "flat": np.full(H, 120.0),
        "noisy oscillation": 130 + 30 * np.sin(np.arange(H) * 0.4) + rng.normal(0, 5, H),
        "severe hypo": np.linspace(75, 42, H),
        "hyper excursion": np.linspace(180, 330, H),
    }

    # Observation masks, so the parity claim covers PARTIALLY observed windows
    # too. A real window routinely carries imputed steps; a check that only ever
    # feeds ones is a check over a subset of the input domain.
    masks = {
        "": None,                                          # fully observed
        " [sparse mask]": np.concatenate(
            [np.ones(H - 8), np.zeros(3), np.ones(5)]).astype(np.float32),
        # An INTERIOR dropout, which is the shape a real window has: the raw
        # values inside a gap are meaningless and the runner has to reconstruct
        # them the way `data._causal_gap_fill` does before featurising. Without
        # this case both sides were handed an already-dense history and the
        # parity claim never covered the fill at all.
        " [interior 10-min dropout]": np.concatenate(
            [np.ones(H - 14), np.zeros(2), np.ones(12)]).astype(np.float32),
        " [two interior dropouts]": np.concatenate(
            [np.ones(6), np.zeros(2), np.ones(6), np.zeros(3), np.ones(H - 17)]
        ).astype(np.float32),
    }

    worst_f = worst_bg = worst_r = 0.0
    failures = 0

    print(f"{'case':<38} {'d_features':>12} {'d_bg (mg/dL)':>14} {'d_risk':>12}")
    print("-" * 80)

    # The gap budget the runner must reproduce. It is a property of the trained
    # pipeline, not of the device, so it is read from the checkpoint's config
    # rather than assumed.
    max_gap_steps = max(
        0, cfg.data.max_interpolation_gap_min // cfg.data.sample_period_min)
    print(f"gap budget: {max_gap_steps} steps "
          f"({cfg.data.max_interpolation_gap_min} min)\n")

    for name, hist in cases.items():
        hist = hist.astype(np.float32)
        for suffix, obs in masks.items():
            observed = np.ones(H, dtype=np.float32) if obs is None else obs
            # RAW values at unobserved steps are garbage on a real device, so
            # make them garbage here too. If either side quietly used them
            # instead of the interpolant, the diff below explodes.
            raw = hist.copy()
            raw[observed == 0.0] = -999.0

            c_feat, c_bg, c_risk = run_cpp(
                raw, None if obs is None else observed,
                max_gap_steps=max_gap_steps)

            # Python's own path: fill the window's short interior gaps, THEN
            # featurise. This is what `build_windows` does before the model ever
            # sees a window, and it is the step the runner used to be missing.
            filled, _imputed = _causal_gap_fill(
                raw.astype(np.float64), observed.astype(bool), max_gap_steps)
            hist_for_torch = filled.astype(np.float32)
            py_feat = causal_features(hist_for_torch, observed)
            with torch.no_grad():
                p_bg, p_risk = net(
                    torch.from_numpy(py_feat[None]),
                    torch.tensor([hist_for_torch[-1]])
                )

            d_f = float(np.max(np.abs(py_feat - c_feat)))
            d_bg = float(np.max(np.abs(np.atleast_1d(p_bg.numpy()) - c_bg)))
            d_r = float(np.max(np.abs(np.atleast_1d(p_risk.numpy()) - c_risk)))

            ok = d_f <= ATOL_FEATURE and d_bg <= ATOL_BG and d_r <= ATOL_RISK
            failures += (not ok)
            worst_f = max(worst_f, d_f)
            worst_bg, worst_r = max(worst_bg, d_bg), max(worst_r, d_r)

            print(f"{name + suffix:<38} {d_f:>12.3e} {d_bg:>14.3e} {d_r:>12.3e}"
                  f"  {'ok' if ok else 'FAIL'}")

    # A window the training pipeline would have DROPPED must be refused rather
    # than forecast from. `build_windows` requires every history step to be
    # observed or imputed, so a trailing dropout (unfillable -- filling it needs
    # an extrapolation) and an over-budget interior gap are both invalid inputs.
    print()
    for label, mask in (
        ("trailing dropout (unfillable)",
         np.concatenate([np.ones(H - 3), np.zeros(3)]).astype(np.float32)),
        ("interior gap longer than the budget",
         np.concatenate([np.ones(5), np.zeros(max_gap_steps + 1),
                         np.ones(H - 5 - max_gap_steps - 1)]).astype(np.float32)),
    ):
        probe = np.full(H, 120.0, dtype=np.float32)
        probe[mask == 0.0] = -999.0
        rc = subprocess.run(
            [RUNNER, "--verify", ONNX, str(max_gap_steps)],
            input="\n".join(f"{v:.9g}" for v in [*probe, *mask]),
            capture_output=True, text=True).returncode
        ok = rc != 0
        failures += (not ok)
        print(f"{'refuses ' + label:<38} {'exit ' + str(rc):>12} "
              f"{'':>14} {'':>12}  {'ok' if ok else 'FAIL'}")

    print("-" * 80)
    print(f"{'worst':<38} {worst_f:>12.3e} {worst_bg:>14.3e} {worst_r:>12.3e}")
    print(f"{'tolerances':<38} {ATOL_FEATURE:>12.0e} {ATOL_BG:>14.0e} {ATOL_RISK:>12.0e}")

    if failures:
        print(f"\nFAIL: {failures} case(s) diverged. The C++ runner does not "
              "match the Python preprocessing (`_causal_gap_fill` then "
              "`causal_features`) -- this is train/serve skew.")
        return 2

    print("\nPASS: C++ feature construction and inference match Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
