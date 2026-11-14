# CausalCGM-MT

**Multitask continuous-glucose forecasting with an auditable anti-leakage
pipeline, conformal uncertainty, post-training personalisation, and a C++ ONNX
Runtime edge runner.**

This repository is the code release for the ML4H 2026 Findings Track paper
*Multitask CGM Forecasting with Strict Temporal Causality and Verified
Deployment*. It contains the library, the experiment drivers, the test suite,
the generated tables and macros the manuscript is built from, and the
aggregate result files they are generated from. It does **not** contain any
patient data (see [What is and is not included](#what-is-and-is-not-included)).

CausalCGM-MT regresses blood glucose at a horizon `t + PH` and, from the same
shared temporal backbone, forewarns hypoglycemia over the interval `(t, t + PH]`.

It is built around one uncomfortable property of glucose forecasting: a leaked
pipeline does not look broken — it looks *good*. So every mechanism here
(windowing, feature construction, patient splitting, statistic fitting,
evaluation) is arranged so lookahead bias trips an assertion instead of quietly
improving a number. Those guarantees are
[enforced and tested](#how-the-pipeline-forecloses-lookahead-bias), and one
ablation deliberately breaks them so the value of leakage is *measured* rather
than assumed.

Measured results are not restated in this README. They live in the paper and in
the generated artefacts listed under
[What is and is not included](#what-is-and-is-not-included); every inline number
in the paper resolves through a macro in [`manuscript/numbers.tex`](manuscript/numbers.tex).

MIT licensed ([LICENSE](LICENSE)) · Python 3.10 · no build step — scripts put
the repo root on `sys.path` and import `causalcgm_mt` from the tree.

## What it does

- **Multitask forecasting.** One causal TCN (or dilated-LSTM) backbone with
  multi-horizon heads at 30 and 60 minutes, trained on a regression target and a
  hypoglycemia target at once, balanced by Kendall uncertainty weighting rather
  than a hand-tuned loss ratio.
- **Multimodal fusion.** A dual-branch model over CGM plus insulin, carbohydrate,
  heart rate, steps, skin and air temperature, galvanic skin response and
  exercise, joined by causal cross-attention. Fusion is gated and residual, so a
  missing modality degrades to the CGM-only model instead of poisoning it.
- **Multi-study corpus.** Streamed ingestion of MetaboNet — public type-1
  studies consolidated into one parquet — with composite subject keys, because
  `id` alone collides across studies, and a contamination guard that drops the
  two studies which *are* this repository's own benchmarks (OhioT1DM and
  HUPA-UCM).
- **Calibrated uncertainty.** Split, normalised, per-patient and Mondrian
  (group-conditional) conformal prediction, calibrated on validation and never on
  test, with coverage reported per glycemic stratum rather than marginally.
- **Clinical and robustness evaluation.** Clarke and Parkes error grids,
  alarm-fatigue metrics (sensitivity, false alarms per day, detection lead time),
  and six ablation studies — one of which is a deliberately leaky control arm.
- **Post-training personalisation.** LoRA, BatchNorm adaptation, head-only and
  full fine-tuning against a single subject, fitted on a causal wall-clock prefix
  with an embargo before the evaluation region.
- **Edge deployment.** ONNX export gated on PyTorch↔ONNX parity and a causality
  probe, plus a C++ ONNX Runtime runner whose feature construction is verified
  against the Python implementation.

## Contents

- [Installation](#installation)
- [Running the tests](#running-the-tests)
- [Quickstart: the smoke test](#quickstart-the-smoke-test)
- [Data access](#data-access)
- [What is and is not included](#what-is-and-is-not-included)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [How the pipeline forecloses lookahead bias](#how-the-pipeline-forecloses-lookahead-bias)
- [Design choices worth knowing about](#design-choices-worth-knowing-about)
- [Post-training personalisation](#post-training-personalisation)
- [ONNX export and the C++ runner](#onnx-export-and-the-c-runner)
- [The harness, and why it is a library](#the-harness-and-why-it-is-a-library)
- [Licence](#licence)

## Installation

### Requirements

Python 3.10 (the Docker image pins 3.10.13), and a C++17 toolchain and CMake
only if you want the edge runner. No GPU is required; every command in this
README runs on CPU.

Development and testing are on macOS and Linux, and the container image is
Linux, so the POSIX `.venv/bin/...` paths used below are the supported ones.
Windows is untested; [`DOCKER.md`](DOCKER.md) is the portable route.

### Set up the environment

```bash
git clone https://github.com/chrisaperez/causalcgm-mt-camera-ready
cd causalcgm-mt-camera-ready
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) bounds every dependency between a floor
and the next major version, and fixes exactly those packages whose *output* this
repository depends on (the ONNX toolchain, the LightGBM baseline, the Parkes grid
backend). [`requirements-lock.txt`](requirements-lock.txt) is the exact set the
suite was last verified against.

### macOS: unify the OpenMP runtime

LightGBM links Homebrew's `libomp` and PyTorch bundles its own. Two OpenMP
runtimes in one process is undefined behaviour and segfaults in practice — a hard
interpreter crash with no traceback, in whichever library initialises its thread
pool second. Neither `KMP_DUPLICATE_LIB_OK` nor `OMP_NUM_THREADS` nor import
order avoids it. Run this once:

```bash
./scripts/fix_openmp.sh
```

It points PyTorch's bundled copy at the Homebrew one, keeps a backup, and then
proves the two libraries coexist. `--revert` undoes it. If a test run ever dies
with no output on macOS, this is why; `conftest.py` detects the condition and
warns.

## Running the tests

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m pytest tests/ -q
```

Some tests skip, and each skip prints its reason (`pytest -rs` lists them).
The skips are gates on inputs this repository cannot ship, not disabled tests:

| gate | how to enable |
| ---- | ------------- |
| the licensed OhioT1DM data is absent (`OhioT1DM/`) | request the data; see [Data access](#data-access) |
| the MetaboNet parquet is absent (`metabonet_public.parquet`) | download it; see [Data access](#data-access) |
| the C++ runner or an exported ONNX graph is absent | `./scripts/fetch_ort.sh`, then build `cpp/` and export a graph (see [the C++ runner](#onnx-export-and-the-c-runner)) |
| inverse guard that runs only when the Parkes backend is *absent* | never; it skips by design when `methcomp` is installed |

Two further tests skip, with a reason, on a public clone:
`test_the_full_tier_actually_adds_the_restricted_files` needs the per-window
`predictions/*.npz` dumps, which are DUA-restricted and not shipped, and
`test_claims_ledger_maps_to_assertions` needs the author's private claim map
(`ledger/claims.csv`). `test_the_runner_refuses_to_start_without_the_dataset`
sets `PY` itself, so it does not need a `.venv`.

The causality tests are mutation tests in the literal sense: they corrupt the
*future* of a record and assert that every window anchored before the
corruption is byte-identical (`tests/test_causality.py`,
`tests/test_benchmark_harness.py`, `tests/test_ohio_loader.py`), and the
structural checker is tested against deliberately damaged documents
(`tests/test_doc_structure.py`).

## Quickstart: the smoke test

Everything below runs on a synthetic cohort and needs **no licensed data**:

```bash
.venv/bin/python scripts/run_experiment.py --synthetic --epochs 3 --folds 2 --adapt lora
```

That exercises the whole path — windowing, causal feature construction,
patient-wise cross-validation, clinical and alarm metrics, then LoRA
personalisation of the fold model onto each held-out subject. It takes about a
minute on a laptop CPU and prints, abridged:

```
WARNING: synthetic data. Numbers are for pipeline validation only.
Windowing: H=24 steps, PH=6 steps
  15998 windows | 12 patients | hypo prevalence 3.38%
[fold 0] epoch 000 train_loss=...  val_rmse=...  val_pr_auc=...
[fold 0] TEST
n=... (pos=..., prev=...)
  REG  RMSE ... (persist ...) | MAE ... | MARD ... | lag ... min
  CLS  ROC-AUC ... | PR-AUC ... | Sens@5%FPR ... | F1 ...
  ALRM Sens/FAfPD @5%FPR .../....day | lead ... min (detect ...% of ... events)

=== CV summary ===
                rmse: ...
              pr_auc: ...

=== Personalisation (lora) on fold 0 test patients ===
[lora] P003: RMSE ... -> ... | PR-AUC ... -> ... (... trainable params, ... windows)
Mean delta over ... patients: RMSE ..., PR-AUC ...
```

If you see the CV summary and the personalisation block, the install is sound.
The metric values themselves are meaningless — the generator is a crude
oscillator with meal spikes and insulin dips, not a physiological model, and it
exists so the repository is runnable end to end without a data use agreement.

Two more checks that need no data:

```bash
.venv/bin/python scripts/check_doc_structure.py       # documents + result logs
.venv/bin/python scripts/run_hupa.py --synthetic --epochs 2 --folds 2
```

### Running on your own data

The generic driver takes a long-format CSV with columns
`patient_id, timestamp, glucose` and cross-validates by patient:

```bash
.venv/bin/python scripts/run_experiment.py --csv data/mycohort.csv --horizon 30 --out runs/mycohort
```

## Data access

No patient data is included in this repository, and none is downloaded
automatically.

**OhioT1DM** (the benchmark cohort: 12 subjects, 2018 and 2020 releases) is
distributed by its maintainers under a **data use agreement** (originally
issued through Ohio University) and **may not be redistributed**. Request it at
[https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html),
accept the agreement, and lay the release out like this (or point `--root`,
`ROOT` or `OHIOT1DM_ROOT` at the directory):

```
OhioT1DM/
├── 2018/
│   ├── train/    559 563 570 575 588 591  (*-ws-training.xml)
│   └── test/     559 563 570 575 588 591  (*-ws-testing.xml)
└── 2020/
    ├── train/    540 544 552 567 584 596  (*-ws-training.xml)
    └── test/     540 544 552 567 584 596  (*-ws-testing.xml)
```

That is 24 XML files, two per subject. A subject present in only one half is
rejected rather than silently contributing training windows that are never
evaluated. `OhioT1DM/` is gitignored and excluded from the Docker build context.
OhioT1DM has its own drivers because its split is temporal and *within* patient
— every subject appears in both `train/` and `test/` — which is not a
patient-wise fold.

**MetaboNet** (the corpus-scale appendix) is a public consolidated corpus of
type-1 diabetes studies; access requires signing in at
[https://metabo-net.org/](https://metabo-net.org/), and the terms of each
constituent study are the requester's to read. The loader expects one parquet
named `metabonet_public.parquet` at the repo root (override with
`--parquet-path`). MetaboNet **contains OhioT1DM and HUPA-UCM**, which are this
repository's own benchmarks, so `MetaboNetConfig.exclude_sources` drops both by
default and `assert_no_benchmark_contamination` refuses them explicitly. The
parquet therefore also carries DUA-restricted rows; it is gitignored and must
never be committed.

**HUPA-UCM** is read from a directory of per-subject CSVs
(`scripts/run_hupa.py --dir`). This repository does not record a canonical
download URL for it.

### What happens if a dataset is missing

Every driver that reads data this repository cannot ship exits with status **2** —
"a precondition failed, nothing was run" — and prints the acquisition pointer plus
a command that needs no data at all, instead of a traceback:

```console
$ .venv/bin/python scripts/prepare_ohio.py --root OhioT1DM

No OhioT1DM XML found under OhioT1DM. Expected OhioT1DM/<2018|2020>/<train|test>/*-ws-*.xml -- 24 files, two per subject, for the full 12-subject release.

OhioT1DM is licensed and cannot be downloaded automatically. Request it at
    https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
and place the release directories at <root>/2018 and <root>/2020.

To check the pipeline itself without any licensed data:
    .venv/bin/python scripts/run_experiment.py --synthetic --epochs 3 --folds 2
```

The convention is written down once, in [`scripts/_cli.py`](scripts/_cli.py):
`0` finished, `1` started and failed (partial artefacts may exist and be worth
resuming from), `2` never started.

## What is and is not included

**Included** (aggregate results and everything generated from them; the
allow-list is the `public` tier of
[`scripts/package_release.py`](scripts/package_release.py) and the `runs/`
rules in `.gitignore`):

| path | what it is |
| ---- | ---------- |
| `runs/paper_results/benchmarks/results_summary.json` and `table_sota_*.tex` | OhioT1DM benchmark, single training seed |
| `runs/paper_results/benchmarks_seeds/multiseed_summary.json` and `table_*multiseed*.tex` | multi-seed benchmark: mean, SD and subject-cluster bootstrap CIs |
| `runs/paper_results/ablations/ablations_summary.json` and `table_*.tex` | ablations A–F |
| `runs/paper_results/personalization/personalization_sweep.json` and `table_*.tex` | 12-subject personalisation sweep |
| `runs/paper_results/conformal_bootstrap.json` | calibration-bootstrap stability of conformal coverage |
| `runs/paper_results/fapd_latching/fapd_latching.json` and `table_fapd_latching.tex` | alarm-policy sensitivity |
| `runs/paper_results/edge/latency_sweep.json` | C++ runner latency percentiles, with the host it was measured on |
| `runs/metabonet_results/benchmarks/table_metabonet_comparison.tex` | the MetaboNet appendix table (the only MetaboNet artefact shipped) |
| `manuscript/numbers.tex` | every inline number of the paper, as generated macros |
| `manuscript/figures/` | the paper's figures (PDF and PNG) |

**Not included**, deliberately:

- **Any patient data.** No OhioT1DM, MetaboNet or HUPA-UCM records, and no
  `metabonet_public.parquet` (it contains OhioT1DM and HUPA-UCM rows).
- **Per-window prediction dumps** (`runs/paper_results/**/predictions/*.npz`).
  They carry CGM values, subject identifiers and timestamps, i.e. OhioT1DM
  itself. Figure 3 is drawn from them, so it ships as a PDF only.
- **Per-seed result directories** of the multi-seed sweep
  (`runs/paper_results/benchmarks_seeds/seed*/`); their aggregate,
  `multiseed_summary.json`, is shipped.
- **MetaboNet run outputs** other than the appendix table: the run summary
  (`metabonet_summary.json`) and per-fold checkpoints (`_folds/`) list the
  MetaboNet test subjects per fold and are not shipped. Figure 7 is drawn from
  them, so it ships as a PDF only.
- **Model checkpoints.** The population checkpoint
  `runs/paper_results/personalization/base_model.pt` is **not included**: it was
  trained on OhioT1DM. The personalisation stage of the reproduction script
  retrains it from the data.
- **ONNX graphs.** `runs/demo/causalcgm_mt.onnx` and
  `runs/demo/causalcgm_mt_fusion.onnx` (the graph timed in `latency_sweep.json`)
  are **not shipped**. See [ONNX export and the C++ runner](#onnx-export-and-the-c-runner)
  for how to build one.
- **Window caches** (`ohio_ws.pkl`, `ohio_windows.pt`, `naive_ws.pkl`) and logs:
  regenerable, large, and derived from the licensed data.

## Reproducing the paper

There are two levels, and they need different things.

### 1. Tables and inline numbers, from the shipped result files

Every table is `\input` from `runs/`, and every inline number in the prose is a
macro in `manuscript/numbers.tex`, rendered from the result JSON by one script:

```bash
.venv/bin/python scripts/rebuild_paper_tables.py
```

It refits nothing; it is a pure function of the JSON. Two tests hold it to
that: `tests/test_manuscript.py::test_numbers_file_is_current` re-renders
`numbers.tex` and requires it to equal the committed file, and
`test_rebuild_tables_is_idempotent_and_changes_no_numbers` requires two renders
to be byte-identical.

**Caveat (MetaboNet appendix).** The MetaboNet macros (`\NumMetabo...`) are
rendered from `runs/metabonet_results/benchmarks/metabonet_summary.json`,
which is not shipped. Without it the script skips those macros, so in a public
clone it regenerates every other macro and table, but re-rendering into
`manuscript/numbers.tex` would drop the MetaboNet ones. To compare without
overwriting, render to a scratch file:

```bash
.venv/bin/python scripts/rebuild_paper_tables.py --numbers /tmp/numbers.tex
diff /tmp/numbers.tex manuscript/numbers.tex     # differs only in \NumMetabo... lines
```

Figures: `scripts/make_paper_figures.py --figures 1 2 4 5 6` redraws figures
1, 2, 4, 5 and 6 from shipped files (1 and 2 are schematics). Figure 3 needs the
OhioT1DM prediction dumps and figure 7 the MetaboNet summary; neither is shipped.

### 2. Everything, from raw data

Re-running the experiments needs **OhioT1DM under its data use agreement** and,
for the corpus-scale appendix only, **MetaboNet** (see [Data access](#data-access)).
The one-command path for everything except MetaboNet and the latency sweep:

```bash
./scripts/reproduce_all_paper_results.sh --root /path/to/OhioT1DM   # hours, CPU
# or: OHIOT1DM_ROOT=/path/to/OhioT1DM ./scripts/reproduce_all_paper_results.sh
./scripts/reproduce_all_paper_results.sh --from tables              # resume at a stage
./scripts/reproduce_all_paper_results.sh --only seeds --seeds "42 123"
```

Stages, in order (each is a pure function of the previous ones' outputs):

| stage | script(s) | writes |
| ----- | --------- | ------ |
| `data` | — (checks the OhioT1DM layout) | — |
| `benchmarks` | `run_paper_benchmarks.py` | `runs/paper_results/benchmarks/` |
| `seeds` | `run_paper_benchmarks.py --seed S --out .../benchmarks_seeds/seedS`, for each of `SEEDS` (default `42 123 456 789 1337`) | `runs/paper_results/benchmarks_seeds/seed*/` |
| `seed_analysis` | `aggregate_seed_sweep.py`, `conformal_calibration_bootstrap.py`, `fapd_latching_sensitivity.py` | `multiseed_summary.json`, `conformal_bootstrap.json`, `fapd_latching/` |
| `ablations` | `run_paper_ablations.py --studies a b c d e f` | `runs/paper_results/ablations/` |
| `personalization` | `run_ohio_personalization_sweep.py` | `runs/paper_results/personalization/` |
| `tables` | `rebuild_paper_tables.py` | tables and `manuscript/numbers.tex` |
| `figures` | `make_paper_figures.py` | `manuscript/figures/` |
| `gates` | `check_doc_structure.py` | — |

The order is load-bearing: rendering before the drivers re-renders the
*previous* run's numbers into a manuscript that then looks freshly built. The
script aborts on the first failure, names the stage, keeps partial artefacts
and prints the `--from` command to resume. `--smoke` runs the same path at a
two-epoch budget for a plumbing check; its artefacts are labelled
`"smoke": true`, the renderers refuse them, and the multi-seed stages are
skipped. Never cite a smoke artefact. The personalisation driver reuses an
existing `base_model.pt` only when its spec and window-pipeline version match;
pass `--retrain` to it to force retraining.

Two result trees are produced outside that script on purpose:

- **MetaboNet appendix** (`runs/metabonet_results/`), isolated from the paper
  tree:

  ```bash
  .venv/bin/python scripts/run_metabonet_benchmarks.py --parquet-path /path/to/metabonet_public.parquet \
      --horizon 30 --folds 5 --max-windows-per-subject 400 --memory-budget-gb 12
  ```

  The run's settings (folds, epochs, seed, window cap, excluded sources) are
  recorded in the `provenance`/`config` blocks of its summary JSON; check them
  against the paper's appendix before comparing numbers.

- **Edge latency** (`runs/paper_results/edge/latency_sweep.json`), which is
  hardware-specific: export a graph and build the runner (see
  [ONNX export and the C++ runner](#onnx-export-and-the-c-runner)), then

  ```bash
  .venv/bin/python scripts/run_cpp_latency_sweep.py --model runs/demo/causalcgm_mt_fusion.onnx
  ```

Then `rebuild_paper_tables.py` and `make_paper_figures.py` pick up both. The
manuscript itself (`manuscript/main.tex`) is compiled with
`pdflatex main && bibtex main && pdflatex main && pdflatex main` in
`manuscript/`; no script edits its prose. A containerised version of the whole
path is in [DOCKER.md](DOCKER.md).

## Repository layout

```
causalcgm_mt/
  config.py      typed run configuration (all causality knobs in one place)
  data.py        regularisation, causal windowing, soft labels, scaler, splits
  models.py      causal TCN / dilated LSTM backbone, multi-horizon heads, multi-task loss
  train.py       patient-wise CV (generic / HUPA-UCM), AMP, early stopping
  evaluate.py    RMSE / MAE / MARD / time lag / ROC-AUC / PR-AUC / Sens@FPR
  clinical.py    Clarke error grid (validated); Parkes via a verified external backend
  extension.py   LoRA, BN-adaptation, head-only personalisation
  hupa_loader.py HUPA-UCM multimodal ingestion, synchronisation, windowing
  ohio_loader.py OhioT1DM XML parsing, synchronisation, official split, ablation groups
  metabonet_loader.py  MetaboNet lazy/streaming parquet ingestion (Polars),
                 disk-backed resumable window store, composite subject keys,
                 shipped is_test protocol, benchmark-contamination guard
  conformal.py   split / normalised / per-patient / Mondrian conformal prediction
  benchmark.py   shared experiment harness: arms as data, one fit-score-serialise path
  leakage_control.py  a DELIBERATELY BROKEN pipeline, the control arm of Ablation D
  baselines/     LightGBM tabular baseline on lag features
  export.py      ONNX export, gated on PyTorch parity + causality probe
  simulate.py    synthetic cohort (pipeline validation only, never results)
app.py           Streamlit diagnostic dashboard
cpp/             C++ ONNX Runtime inference runner + benchmark
scripts/         run_experiment (generic CSV / HUPA-UCM), run_hupa, prepare_ohio,
                 train_ohio, run_ablation, train_demo, compare_models, export_onnx,
                 verify_cpp_parity, fetch_ort, fix_openmp, verify_mondrian_coverage,
                 run_paper_benchmarks, run_paper_ablations,
                 run_ohio_personalization_sweep, run_cpp_latency_sweep,
                 run_metabonet_benchmarks (MetaboNet, isolated outputs),
                 rebuild_paper_tables, make_paper_figures, check_doc_structure,
                 reproduce_all_paper_results (the one-command pipeline),
                 package_release (artefact archive, DUA-aware),
                 _cli (shared exit-code convention for missing inputs)
manuscript/      main.tex, references.bib, generated numbers.tex, figures/
requirements.txt bounded dependency ranges; requirements-lock.txt is the exact
                 set the suite was last verified against
Dockerfile       pinned environment; see DOCKER.md
.dockerignore    keeps OhioT1DM, the parquet and the caches out of the build context
tests/           causality, multimodal, OhioT1DM, MetaboNet, clinical/ablation/
                 multi-horizon, conformal + baseline, experiment harness,
                 manuscript, audit regressions, soft-label band, document
                 structure + figures, release packaging + reproduction runner,
                 release readiness (bounded dependencies, usable synthetic
                 cohort, missing-input exit codes)
```

## How the pipeline forecloses lookahead bias

The guarantee is structural, not a matter of care. Five independent mechanisms:

**1. Window geometry is a closed interval.** For anchor `t`, inputs read
`[t-H+1, t]` and targets read `[t+1, t+PH]`. The two index ranges are disjoint
by construction in `build_windows`, and targets are written to a separate array
that no feature code touches. There is no code path from `y_*` back into `x`.

**2. Interpolation is per-window, not per-series.** The single most common leak
in CGM pipelines is `df.interpolate()` on the full record before windowing: a
gap ending *after* `t` drags future observations backwards into the history.
Here `_causal_gap_fill` runs on the history slice only, so both endpoints of any
interpolated gap are themselves inside `[t-H+1, t]`. Targets are never imputed —
a window with any unobserved target step is dropped, since imputing a label is
leakage by definition.

**3. Features are prefix functions.** In `causal_features`, channel value at
index `i` depends only on `hist[:i+1]` (rolling statistics use backward windows,
never centred ones). This is stricter than needed — the model sees the whole
window anyway — but it means the identical code path serves streaming inference.

**4. Every fitted statistic is fit on training patients only.** `run_fold`
executes a fixed order: split patients → fit scaler → estimate the fold's CGM
noise → derive the soft-label band → compute `pos_weight` → *then* transform train / val
/ test with those frozen objects. `CausalScaler` records `fitted_on`, so an audit
can confirm which patients contributed statistics. Nothing is refit downstream,
including during personalisation.

**5. Splits are patient-wise, and the architecture is causal.** `patient_folds`
groups by patient, never by row — overlapping windows from one patient share
almost all of their content, so row-wise splitting inflates scores dramatically.
`CausalConv1d` pads left-only and trims the tail; the LSTM alternative is
unidirectional, because a bidirectional pass reads future timesteps.

Each of these is asserted in `tests/test_causality.py`, and each test fails if
its guarantee is removed. The decisive one is
`test_future_perturbation_changes_targets_but_not_inputs`: it corrupts the tail
of every patient's record by +120 mg/dL and asserts that windows anchored before
the corruption have byte-identical inputs. Any forward read in any stage of the
pipeline breaks it.

## Design choices worth knowing about

**Residual regression.** The head predicts a change from the last observed BG,
not the absolute level. Absolute-level regression lets a model score well by
echoing its input — good RMSE, useless clinically. `time_lag_minutes` exists to
catch exactly this: pure persistence scores a lag equal to the full horizon, and
the metric is calibrated against that in a unit test.

**Soft labels near the threshold.** A hard `1[BG < 70]` treats 69.9 vs 70.1 as
maximally different, though CGM measurement error (~10% MARD) makes the true
class genuinely ambiguous there. `SoftLabeler` applies a logistic ramp centred
on the threshold, with band width `k · MARD · 70`. Far from the boundary the
label is effectively hard. Set `soft_labels=False` for the ablation baseline.

The MARD is estimated in three tiers, and the labeller records which one ran:

| tier                 | when                                           | on OhioT1DM                                   |
| -------------------- | ---------------------------------------------- | --------------------------------------------- |
| `paired_reference` | the cohort ships reference BG                  | **10.5% MARD → ±7.35 mg/dL band**     |
| `high_pass`        | CGM stream only                                | a lower bound only; the signal is smoothed    |
| `device_floor`     | the estimate is below the device specification | 9% → ±6.3 mg/dL                             |

The band is then clamped into `[band_min_mgdl, band_max_mgdl]` (5–20 mg/dL), and
`SoftLabeler.band_source` says whether it was measured, floored or clamped. That
provenance is what makes `band_mode` a sweepable arm (Ablation E in the paper),
and it is written into every results JSON under `soft_label`.

The `paired_reference` figures above (10.5% MARD, a ±7.35 mg/dL band) are not
from a results file: they are what `estimate_reference_mard` measures on the
released OhioT1DM reference pairs, and
`tests/test_doc_figures.py::test_the_readme_band_is_what_the_estimator_actually_produces`
recomputes both from the raw records and fails if this README disagrees (it
skips when `OhioT1DM/` is absent). The `device_floor` row is the sensor
specification, not a measurement.

**Uncertainty-weighted loss.** Huber (mg/dL²) and BCE (nats) are on incompatible
scales, so the default learns per-task log-variances (Kendall et al. 2018)
rather than using a hand-tuned λ. The effective weights are logged, because a
silently collapsing task weight is how a multi-task model degenerates into a
single-task one.

**Metrics.** Hypoglycemia is a few percent of samples, so ROC-AUC is optimistic;
PR-AUC (reported against the prevalence floor) and sensitivity at 5% FPR are the
discriminating numbers. Alarm fatigue, not AUC, decides whether a forewarning
system stays switched on.

## Post-training personalisation

Population models are evaluated on unseen patients — honest, but pessimistic
relative to deployment, where a new user has already worn the sensor for a day.
`extension.py` adapts a copy of the generic model per patient under a causal
protocol:

```
|<-- adapt prefix (first K hours) -->|<-- embargo -->|<-- evaluation -->|
```

The prefix is a strict wall-clock prefix, never a random sample. The embargo is
forced to at least `history + horizon`, so no evaluation window's history can
overlap an adaptation window's target — without it, adjacent windows share
almost all their content and "adaptation" becomes memorisation of the test
region. The population scaler and soft-label band are **not** refit.

Mechanisms, increasing in capacity:

| method        | trainable params          | notes                                                                                                                                                                                                                        |
| ------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `none`      | 0                         | zero-shot population model. A real experimental condition, not the absence of one — its before/after must be bit-identical.                                                                                                 |
| `bn_adapt`  | 0                         | recomputes BatchNorm statistics only; addresses covariate shift with no overfitting risk. Best for very short prefixes. Only the BN layers run in training mode, so dropout does not inflate the statistics being estimated. |
| `lora`      | adapters only             | rank-`r` adapters on backbone convs *and* the cross-attention `q/k/v/proj`; `B` initialised to zero so `ΔW = 0` and the first forward pass is bit-identical to the base model.                                    |
| `head_only` | heads only                | middle baseline for the ablation.                                                                                                                                                                                            |
| `full`      | all                       | included for comparison; will overfit a one-day prefix.                                                                                                                                                                      |

Gradient methods are anchored with L2-SP (distance from the *generic weights*,
not from zero — the correct prior when the population model is already good).
Comparison is paired: before/after on identical evaluation windows.

Two entry points, sharing the adaptation mechanics so an improvement measured on
one arm is not confounded by a different implementation on the other:
`run_personalization` for the cross-validated unseen-subject protocol, and
`run_ohio_personalization` for the within-subject OhioT1DM protocol
(`scripts/run_ohio_personalization_sweep.py`; results in the paper and in
`runs/paper_results/personalization/`).

The `none` arm is the live check that the harness adds no drift: scored twice
on the same windows, its mean RMSE change in the shipped sweep is exactly
0.0000 (`significance["24.0"]["none"]["rmse"]["mean_delta"]` in
`personalization_sweep.json`, asserted by `tests/test_doc_figures.py`).

## ONNX export and the C++ runner

No ONNX graph is shipped. To build one and run it from C++
([`cpp/causalcgm_mt_runner.cpp`](cpp/causalcgm_mt_runner.cpp)):

```bash
./scripts/fetch_ort.sh                          # C++ SDK matching the venv's onnxruntime
.venv/bin/python scripts/train_demo.py 30       # synthetic demo checkpoint -> runs/demo/demo.pt
.venv/bin/python scripts/export_onnx.py         # -> runs/demo/causalcgm_mt.onnx; refuses a graph that fails its gates
cmake -B cpp/build -S cpp && cmake --build cpp/build -j4
./cpp/build/causalcgm_mt_runner runs/demo/causalcgm_mt.onnx 2000 200 1 1
.venv/bin/python scripts/verify_cpp_parity.py
```

The demo graph is CGM-only and trained on synthetic data; it exercises the
plumbing. The graph the paper times is a fusion model. A fusion checkpoint in
the exportable schema is written by `scripts/train_ohio.py` (needs OhioT1DM):

```bash
.venv/bin/python scripts/train_ohio.py --root /path/to/OhioT1DM --out runs/ohio   # writes runs/ohio/model_<tag>.pt
.venv/bin/python scripts/export_onnx.py runs/ohio/model_<tag>.pt runs/demo/causalcgm_mt_fusion.onnx
```

The personalisation sweep's `base_model.pt` is not in that schema (it carries no
`scaler`/`config`), so `export_onnx.py` refuses it by design.

The exported graph folds in the population scaler and the output sigmoid, so the
C++ caller's contract is raw window in, mg/dL and probability out — no
preprocessing constants to drift out of sync with training. Every driver writes
one checkpoint schema (`export.REQUIRED_CHECKPOINT_KEYS`: `state_dict`,
`in_channels`, `config`, `scaler`, plus `aux_scaler` and `aux_channels` when
`multimodal` is set), checked by `assert_checkpoint_schema` before anything is
unpacked; `load_and_export` dispatches on the checkpoint's own `multimodal`
flag, so a fusion model exports as a fusion model.

**Three gates, because each catches something the others cannot:**

1. *Graph parity* (`export.py`) — max abs difference vs PyTorch, per output in
   its own units, with separate tolerances for `bg_hat` (mg/dL) and `risk`
   (probability). Export raises rather than writing a disagreeing model.
2. *Causality probe* (`export.py`) — re-runs the perturbation test against the
   **exported** graph. `CausalConv1d` achieves causality by padding both sides
   and slicing the tail off in `forward`; if the exporter folds that slice away,
   the result is a fast model that reads the future, and a latency benchmark
   would never notice.
3. *Cross-language parity* (`verify_cpp_parity.py`) — the runner reimplements
   `causal_features` in C++, which no graph-level check can see. This diffs the
   feature tensor *and* the outputs across hand-built clinical scenarios (ramps,
   hypo descent, hyper excursion). **Scope:** this gate covers the CGM-only
   graph. The runner builds the CGM feature channels and has no auxiliary
   feature path, so a fusion graph runs through it with `aux_present = 0` — the
   CGM-only degradation path — and its auxiliary preprocessing is not
   cross-checked in C++.

Latency measurements, and the host they were taken on, are in
`runs/paper_results/edge/latency_sweep.json` and the paper's edge-latency
figure. They are from a laptop CPU, not target hardware.

## The harness, and why it is a library

[benchmark.py](causalcgm_mt/benchmark.py) holds the pieces all three drivers need:
load the cohort, fit an arm, score it, serialise it. The drivers only enumerate
`ArmSpec`s. Writing that four times would guarantee the paper's tables were
computed four slightly different ways, and the differences would be invisible —
a benchmark table and an ablation table that disagree about the same arm's RMSE
is the kind of thing a reader finds and an author cannot explain.

`ArmSpec` is frozen and fully serialisable, and is written into the results JSON
verbatim, so any number in the paper traces back to the exact hyperparameters
that produced it without re-reading a driver script.

**Every arm goes through one scoring path.** Persistence, Ridge-AR, ARIMA,
LightGBM, the dilated LSTM, both CausalCGM-MT variants and the LoRA-adapted model
all emit `(bg_hat, risk)` on `split.val` and `split.test` and are scored by the
same `score_arm`. A baseline compared under different conditions is not a
baseline.

Regression-only baselines need a hypoglycemia score, and they get
`sigmoid((70 − ŷ)/τ)` — a strictly monotone map of the forecast, so their
ROC-AUC, PR-AUC and sensitivity-at-fixed-FPR are *exactly* those of the forecast
and nothing is invented by the wrapper. What that score cannot do is the thing
the multi-task head exists for: the label is `min BG over (t, t+PH] < 70`, and a
level forecast at exactly `t+PH` is blind to a nadir at `t+20` that recovers by
`t+30`. That gap is a real property of regression-only baselines and belongs in
the table.

`--fit-subsample k` shortens a run by keeping every k-th **fit** window. Val and
test are untouched, so reported metrics stay at native cadence and remain
comparable across arms trained at different budgets — and since adjacent
stride-1 windows share 23 of 24 history steps, dropping two of every three costs
far less information than it costs wall clock.

### About ARIMA

The classical arm is a real `statsmodels` conditional-MLE fit, one ARIMA(p, d, 0)
per subject on that subject's own training series, applied causally per window.
The moving-average order is fixed at zero deliberately: an MA term is a function
of the innovation history, which is only recoverable by filtering the series from
its start — applying that to a 24-sample window would either need the full
series (reaching across the official train/test boundary, i.e. leakage) or
silently start the filter from zero innovations. With `q = 0` the h-step
forecast is an exact closed-form function of the window's own lags, which is the
only version of "ARIMA" that is well defined under this protocol. The recursion
is then vectorised over all windows; calling `statsmodels` once per window would
produce the same numbers at far greater cost.

`Ridge-AR` is reported alongside as the direct multi-step linear autoregression:
one penalised least-squares fit per horizon, targeting the residual from the
last observed value rather than the level.

## Licence

MIT; see [LICENSE](LICENSE). The licence covers the code and the generated
artefacts in this repository. It grants nothing over OhioT1DM, MetaboNet or
HUPA-UCM, which remain under their own terms.

A citation entry will be added once the paper's proceedings record exists.
