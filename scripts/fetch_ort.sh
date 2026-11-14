#!/usr/bin/env bash
# Fetch the ONNX Runtime C++ SDK matching the venv's Python onnxruntime.
#
# The pip wheel ships the shared library but no headers. Version matching is
# required: causalcgm_mt/export.py validates PyTorch<->ONNX parity using the
# Python build, so a different C++ build would make that gate vacuous.
set -euo pipefail
cd "$(dirname "$0")/.."

VER=$(.venv/bin/python -c "import onnxruntime;print(onnxruntime.__version__)")
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PLAT=osx-arm64 ;;
  Darwin-x86_64) PLAT=osx-x86_64 ;;
  Linux-x86_64) PLAT=linux-x64 ;;
  Linux-aarch64) PLAT=linux-aarch64 ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

NAME="onnxruntime-${PLAT}-${VER}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${NAME}.tgz"

mkdir -p cpp/vendor
if [ -d "cpp/vendor/${NAME}" ]; then
  echo "already present: cpp/vendor/${NAME}"
  exit 0
fi

echo "fetching ${URL}"
curl -fsSL -o "cpp/vendor/${NAME}.tgz" "$URL"
tar xzf "cpp/vendor/${NAME}.tgz" -C cpp/vendor
rm "cpp/vendor/${NAME}.tgz"
echo "unpacked cpp/vendor/${NAME}  (version $(cat "cpp/vendor/${NAME}/VERSION_NUMBER"))"
