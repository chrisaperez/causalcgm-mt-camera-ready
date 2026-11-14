# Running CausalCGM-MT in a container

The image pins Python 3.10.13, every dependency in `requirements.txt`, and the
ONNX Runtime C++ SDK at the version the Python wheel resolves to. It is the
environment the pipeline is meant to be reproduced in.

> **Status:** the `Dockerfile` in this repository has not been built end to end.
> It was written against the verified host pipeline and its `COPY` sources,
> build order and entrypoint are checked, but no image has been produced from
> it. Treat the first build as part of the review, not as a settled result.

## Build

```bash
docker build -t causalcgm-mt .
```

The build downloads the ONNX Runtime SDK and compiles the C++ runner, so it
needs network access and takes several minutes. TeX Live is roughly two thirds
of the image; skip it if you do not need to compile the manuscript:

```bash
docker build --build-arg WITH_LATEX=0 -t causalcgm-mt .
```

## The data is not in the image

OhioT1DM is licensed and is deliberately excluded from both the image and the
build context (see `.dockerignore`). Request it from
<https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html>, lay it out as
`OhioT1DM/2018/{train,test}` and `OhioT1DM/2020/{train,test}`, and mount it
read-only:

```bash
docker run --rm \
  -v "$PWD/OhioT1DM:/app/OhioT1DM:ro" \
  -v "$PWD/runs:/app/runs" \
  causalcgm-mt
```

`runs/` is mounted read-write because that is where every artefact lands; the
container writes nothing outside it and `manuscript/`.

## What the default command does

The entrypoint is `scripts/reproduce_all_paper_results.sh`, so a bare
`docker run` executes the full pipeline: data check → benchmarks → multi-seed
sweep → seed analysis → ablations → personalisation → tables → figures → doc
gates. Arguments pass straight through. The MetaboNet appendix and the C++
latency sweep are separate (see the README, "Reproducing the paper").

```bash
# Check the whole path in ~10 minutes at a two-epoch budget.
docker run --rm -v "$PWD/OhioT1DM:/app/OhioT1DM:ro" -v "$PWD/runs:/app/runs" \
  causalcgm-mt --smoke

# Re-render tables and figures from JSONs that already exist. No data needed.
docker run --rm -v "$PWD/runs:/app/runs" causalcgm-mt --from tables

# One stage.
docker run --rm -v "$PWD/runs:/app/runs" causalcgm-mt --only figures
```

Smoke artefacts are labelled `"smoke": true` in every JSON, and
`scripts/package_release.py` refuses to package them.

## Other entrypoints

```bash
# Test suite.
docker run --rm --entrypoint .venv/bin/python causalcgm-mt -m pytest tests/ -q

# C++/Python parity gate.
docker run --rm -v "$PWD/runs:/app/runs" \
  --entrypoint .venv/bin/python causalcgm-mt scripts/verify_cpp_parity.py

# Compile the manuscript (needs WITH_LATEX=1).
docker run --rm -v "$PWD/manuscript:/app/manuscript" \
  --entrypoint bash causalcgm-mt -c \
  "cd manuscript && pdflatex -interaction=nonstopmode main && bibtex main \
   && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main"

# An interactive shell.
docker run --rm -it -v "$PWD/OhioT1DM:/app/OhioT1DM:ro" -v "$PWD/runs:/app/runs" \
  --entrypoint bash causalcgm-mt
```

## OpenMP

On macOS this repository ships `scripts/fix_openmp.sh`, which points PyTorch's
bundled `libomp.dylib` at Homebrew's so LightGBM and PyTorch share one OpenMP
runtime. **That script is not run in the container and must not be** — it is
written against Homebrew paths and `.dylib` files that do not exist on Linux.

The Linux hazard is the same in kind: PyTorch's x86 wheels can bring Intel's
`libiomp5` in alongside LightGBM's `libgomp`, and two OpenMP runtimes in one
process is undefined behaviour. The image installs `libgomp1` explicitly and
pins every thread pool to one thread (`OMP_NUM_THREADS=1` and friends), which
removes the contention rather than suppressing its symptom. That is also the
configuration the paper's latency numbers were measured under, so the timed and
the tested configurations are the same one.

`KMP_DUPLICATE_LIB_OK` is deliberately **not** set. It silences the abort
without fixing the cause, and on macOS it was measured to move the crash rather
than remove it. A container that fails loudly beats one that returns quietly
wrong numbers.

The build ends with a check that imports LightGBM and PyTorch into one process
and runs a convolution and a boosting round, so an OpenMP clash fails the build
rather than the first long run.

## Reproducibility caveats

- **Not bit-exact across architectures.** `linux/amd64` and `linux/arm64`
  produce different floating-point orderings in BLAS and cuDNN-free CPU kernels.
  Metrics agree to the precision the tables print; the last digits of a
  prediction dump may not. Pass `--platform linux/amd64` if you need to match a
  specific published run.
- **`requirements.txt` bounds every dependency but fixes only some.** The
  scientific core carries a floor and a next-major ceiling (`torch>=2.2,<3.0`),
  so a rebuild picks up patch releases but never a major this code has not been
  run against; packages whose *output* matters -- the ONNX toolchain, LightGBM,
  the Parkes backend -- are pinned exactly. `requirements-lock.txt` fixes the
  direct dependencies at the versions the suite was last verified against, and
  the build additionally writes the fully resolved set, transitive dependencies
  included, to `/app/requirements.lock.txt`; copy that out with `docker cp` if
  you need to pin an image exactly.
- **The window cache is version-keyed.** A cache built by an older pipeline is
  rejected and rebuilt rather than silently reused, so mounting an old `runs/`
  is safe.
