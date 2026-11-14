#!/usr/bin/env bash
# Unify the OpenMP runtime so LightGBM and PyTorch can coexist (macOS).
#
# LightGBM links Homebrew's libomp; PyTorch bundles its own. Two OpenMP runtimes
# in one process is undefined behaviour and segfaults in practice. Neither
# KMP_DUPLICATE_LIB_OK nor OMP_NUM_THREADS avoids it, and import order only
# moves the crash between the two libraries.
#
# This points torch's bundled copy at the Homebrew one, leaving a single runtime.
# The original is kept as libomp.dylib.orig; rerun with --revert to restore it.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_OMP="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib","libomp.dylib"))')"
BREW_OMP=/opt/homebrew/opt/libomp/lib/libomp.dylib

if [ "${1:-}" = "--revert" ]; then
  if [ -f "$TORCH_OMP.orig" ]; then
    mv -f "$TORCH_OMP.orig" "$TORCH_OMP"
    echo "reverted $TORCH_OMP"
  else
    echo "no backup found; reinstall torch to restore"
  fi
  exit 0
fi

[ "$(uname -s)" = "Darwin" ] || { echo "macOS only; nothing to do."; exit 0; }
[ -f "$BREW_OMP" ] || { echo "Homebrew libomp not found at $BREW_OMP. Run: brew install libomp"; exit 1; }

if [ -L "$TORCH_OMP" ]; then
  echo "already unified: $TORCH_OMP -> $(readlink "$TORCH_OMP")"
  exit 0
fi

cp -n "$TORCH_OMP" "$TORCH_OMP.orig"
ln -sf "$BREW_OMP" "$TORCH_OMP"
echo "unified: $TORCH_OMP -> $BREW_OMP  (backup at $TORCH_OMP.orig)"

.venv/bin/python - <<'PY'
import torch, torch.nn as nn, numpy as np, lightgbm as lgb
nn.Conv1d(3,5,3)(torch.randn(2,3,20))
lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(np.random.rand(500,10), np.random.rand(500))
print("verified: torch and lightgbm coexist in one process")
PY
