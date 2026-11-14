# CausalCGM-MT: a pinned environment that reproduces every published artefact.
#
#   docker build -t causalcgm-mt .
#   docker run --rm -v "$PWD/OhioT1DM:/app/OhioT1DM:ro" \
#              -v "$PWD/runs:/app/runs" causalcgm-mt
#
# See DOCKER.md for the full set of invocations.
#
# WHY A SINGLE PINNED STAGE AND NOT A MULTI-STAGE BUILD
# -----------------------------------------------------
# A multi-stage build exists to keep compilers out of the shipped image. This
# image is not a deployment target -- it is the environment a reader runs the
# pipeline in, and that pipeline compiles the C++ runner and rebuilds the ONNX
# graph as part of what it verifies. Discarding the toolchain would mean the
# released artefact could not be rebuilt inside the image that produced it.
# The C++ build is still isolated in its own layer so it caches independently.

FROM python:3.10.13-slim-bookworm

# The venv's Python is 3.10.13; pinning the base to the same patch release keeps
# a pip resolution from differing between host and container over a stdlib bump.

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------- #
# 1. System packages
# ---------------------------------------------------------------------------- #
# libgomp1 is the OpenMP runtime. It is installed EXPLICITLY rather than left to
# arrive as a transitive dependency, because which OpenMP ends up in the process
# is the single most fragile thing about this stack -- see the note in section 3.
#
# texlive-* is what lets `pdflatex main` work inside the container. It is the
# bulk of the image; DOCKER.md documents `--build-arg WITH_LATEX=0` to skip it.
ARG WITH_LATEX=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    git \
    libgomp1 \
    && if [ "$WITH_LATEX" = "1" ]; then apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    biber \
    ; fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------- #
# 2. Python dependencies, in their own layer
# ---------------------------------------------------------------------------- #
# requirements.txt is copied alone so that editing source does not invalidate
# the dependency layer -- the slowest one to rebuild by an order of magnitude.
#
# The venv lives at /app/.venv because every script in this repository invokes
# `.venv/bin/python` by convention; putting it anywhere else would mean the
# containerised commands differ from the documented host ones, which is the
# opposite of what this image is for.
COPY requirements.txt ./
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && .venv/bin/pip check

# Record exactly what resolved, so an image can be audited after the fact and a
# future rebuild can be pinned against it.
RUN .venv/bin/pip freeze > /app/requirements.lock.txt

# ---------------------------------------------------------------------------- #
# 3. OpenMP alignment
# ---------------------------------------------------------------------------- #
# On macOS this repository ships scripts/fix_openmp.sh, which symlinks torch's
# bundled libomp.dylib at Homebrew's so that LightGBM and PyTorch share one
# runtime. THAT SCRIPT IS NOT RUN HERE, and running it would be wrong: it is
# written against Homebrew paths and .dylib files that do not exist on Linux.
#
# The Linux hazard is the same in kind and different in detail. PyTorch's x86
# wheels can pull Intel's libiomp5 in alongside LightGBM's libgomp, and two
# OpenMP runtimes in one process is undefined behaviour whichever pair they are.
# Rather than patch shared libraries, the image pins every thread pool to one
# thread. That removes the contention outright, and it is also what the paper's
# latency numbers are measured under, so the timed configuration and the tested
# configuration are the same configuration.
#
# KMP_DUPLICATE_LIB_OK is deliberately NOT set. It suppresses the abort without
# fixing the cause, which on macOS was measured to move the crash rather than
# remove it; a container that segfaults loudly is better than one that corrupts
# results quietly.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------------------- #
# 4. C++ ONNX Runtime SDK and the runner
# ---------------------------------------------------------------------------- #
# fetch_ort.sh reads the version from the venv's onnxruntime and downloads the
# matching C++ SDK. The version match is the point: export.py certifies
# PyTorch<->ONNX parity using the PYTHON runtime, so a C++ build at a different
# version would be certifying something the runner never executes. The script
# already resolves linux-x64 and linux-aarch64, so this works on both.
COPY scripts/fetch_ort.sh scripts/
RUN ./scripts/fetch_ort.sh

COPY cpp/ cpp/
RUN cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
    && cmake --build cpp/build -j"$(nproc)" \
    && ls -la cpp/build

# ---------------------------------------------------------------------------- #
# 5. The repository
# ---------------------------------------------------------------------------- #
# Last, so that a source edit rebuilds only this layer.
COPY causalcgm_mt/ causalcgm_mt/
COPY tests/ tests/
COPY manuscript/ manuscript/
COPY scripts/ scripts/
# requirements-lock.txt rides in the SOURCE layer, not the dependency one:
# the image installs from requirements.txt, and putting the lock file beside
# it at line 67 would invalidate the slowest layer in the build every time a
# pin was recorded. It is copied because `pytest tests/ -q` inside the
# container checks that the lock file and requirements.txt still agree.
COPY conftest.py app.py README.md requirements-lock.txt ./

RUN chmod +x scripts/*.sh

# OhioT1DM is licensed and is NOT baked into the image. Mount it read-only:
#   -v "$PWD/OhioT1DM:/app/OhioT1DM:ro"
# The runner's data stage fails with instructions if it is absent.
VOLUME ["/app/OhioT1DM", "/app/runs"]

# A fast self-check that the environment is sound: imports resolve, LightGBM and
# PyTorch coexist in one process (the OpenMP question), and the C++ runner links.
RUN .venv/bin/python -c "\
    import lightgbm, torch, onnxruntime, numpy; \
    print('lightgbm', lightgbm.__version__); \
    print('torch   ', torch.__version__); \
    print('ort     ', onnxruntime.__version__); \
    import torch.nn as nn; \
    _ = nn.Conv1d(1, 1, 3)(torch.zeros(1, 1, 8)); \
    _ = lightgbm.train({'objective':'regression','verbose':-1}, \
    lightgbm.Dataset(numpy.random.rand(64, 4), label=numpy.random.rand(64)), \
    num_boost_round=1); \
    print('OpenMP  : lightgbm and torch coexist in one process')"

ENTRYPOINT ["./scripts/reproduce_all_paper_results.sh"]
CMD []
