"""CausalCGM-MT.

NOTE ON IMPORT ORDER (macOS OpenMP)
-----------------------------------
LightGBM and PyTorch each link their own copy of `libomp.dylib`. On macOS,
loading LightGBM *after* PyTorch reliably segfaults inside LightGBM's dataset
construction -- not an exception, a hard crash that takes the interpreter with
it. Loading LightGBM first is stable.

Neither `KMP_DUPLICATE_LIB_OK=TRUE` nor `OMP_NUM_THREADS=1` fixes it (both were
tried and both still crash), so the import order is enforced here rather than
left to whichever module a caller happens to touch first. Without this, the test
suite crashes or passes depending purely on file collection order, which is the
worst possible failure mode.

The preload is optional and silent: LightGBM is only needed for the baseline, so
its absence must not break the rest of the package.
"""

import sys as _sys

if "torch" not in _sys.modules:
    try:  # pragma: no cover - environment dependent
        import lightgbm as _lightgbm  # noqa: F401
    except Exception:  # lightgbm not installed, or not importable here
        pass

from .config import Config, DataConfig, ModelConfig, TrainConfig, AdaptConfig

__all__ = ["Config", "DataConfig", "ModelConfig", "TrainConfig", "AdaptConfig"]
