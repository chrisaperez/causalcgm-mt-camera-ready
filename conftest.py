"""Pytest session setup: detect the macOS duplicate-OpenMP condition early.

The problem
-----------
LightGBM (via Homebrew's `/opt/homebrew/opt/libomp`) and PyTorch (via its own
bundled `torch/lib/libomp.dylib`) can load TWO different OpenMP runtimes into
one process. That is undefined behaviour, and in practice it segfaults -- a hard
interpreter crash with no traceback, in whichever library initialises its thread
pool second. It is symmetric: with LightGBM first, torch's conv crashes; with
torch first, LightGBM's dataset construction crashes.

What does NOT fix it (all tried on this machine):
  * KMP_DUPLICATE_LIB_OK=TRUE
  * OMP_NUM_THREADS=1
  * import order -- it only moves the crash to the other library

The actual fix is to have exactly ONE libomp in the process; see
`scripts/fix_openmp.sh`, which points torch's bundled copy at the same Homebrew
libomp LightGBM uses.

This file only DETECTS the bad condition and reports it clearly. A mysterious
segfault that appears or disappears with test collection order costs far more
time than an explicit message.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _duplicate_libomp() -> list[str]:
    """Return distinct real paths of libomp copies that torch and lightgbm use.

    Resolves symlinks, so the `fix_openmp.sh` symlink correctly reports one.
    """
    found: set[str] = set()
    try:
        import torch
        p = Path(torch.__file__).parent / "lib" / "libomp.dylib"
        if p.exists():
            found.add(str(p.resolve()))
    except Exception:
        pass

    for candidate in ("/opt/homebrew/opt/libomp/lib/libomp.dylib",
                      "/opt/local/lib/libomp/libomp.dylib"):
        try:
            import lightgbm  # noqa: F401
        except Exception:
            return []
        if Path(candidate).exists():
            found.add(str(Path(candidate).resolve()))
            break
    return sorted(found)


def pytest_configure(config):  # noqa: D103
    if not sys.platform.startswith("darwin"):
        return
    libs = _duplicate_libomp()
    if len(libs) > 1:
        print(
            "\n" + "=" * 72 +
            "\nWARNING: multiple OpenMP runtimes detected. The suite will very "
            "likely\nsegfault when LightGBM and PyTorch are used in one process:\n"
            + "".join(f"  {p}\n" for p in libs) +
            "Fix with:  ./scripts/fix_openmp.sh\n" + "=" * 72 + "\n",
            file=sys.stderr,
        )
