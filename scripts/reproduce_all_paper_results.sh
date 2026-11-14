#!/usr/bin/env bash
#
# One command from a clean checkout to every artefact the manuscript cites.
#
#   ./scripts/reproduce_all_paper_results.sh              # full run, hours
#   ./scripts/reproduce_all_paper_results.sh --smoke      # same path, ~10 min
#   ./scripts/reproduce_all_paper_results.sh --from tables    # resume at a stage
#   ./scripts/reproduce_all_paper_results.sh --only figures
#   ./scripts/reproduce_all_paper_results.sh --root /path/to/OhioT1DM
#   OHIOT1DM_ROOT=/path/to/OhioT1DM ./scripts/reproduce_all_paper_results.sh
#
# WHAT THIS IS FOR
# ----------------
# The pipeline is already reproducible stage by stage; what it lacked was a
# single entry point that runs the stages in the ONE order that is correct.
# That order is not arbitrary: every table and figure is a pure function of the
# result JSONs, so `rebuild_paper_tables` and `make_paper_figures` must follow
# the drivers, and running them first silently re-renders the PREVIOUS run's
# numbers into a manuscript that then looks freshly built.
#
# WHAT IT DELIBERATELY DOES NOT DO
# --------------------------------
# It does not touch `manuscript/main.tex`, and it does not run pdflatex. The
# prose is written by hand and reviewed by hand; this script only regenerates
# what is machine-generated. See DOCKER.md for the containerised invocation.
#
# Two result trees the manuscript reads are NOT produced here, by design:
#   * the MetaboNet corpus-scale appendix (runs/metabonet_results/), produced by
#     scripts/run_metabonet_benchmarks.py, which is isolated from this tree;
#   * the C++ latency sweep (runs/paper_results/edge/latency_sweep.json), which
#     is hardware-specific: scripts/train_demo.py -> scripts/export_onnx.py ->
#     cmake build of cpp/ -> scripts/run_cpp_latency_sweep.py.
# The README ("Reproducing the paper") lists both paths.
#
# DATA LOCATION
#   --root DIR, or the ROOT or OHIOT1DM_ROOT environment variable (ROOT wins if
#   both are set), default ./OhioT1DM. Nothing in this script assumes where the
#   licensed data lives on your machine.
#
# EXIT CODES
#   0  every requested stage succeeded
#   1  a stage failed (the log names which, and the partial artefacts are kept)
#   2  a precondition failed (missing data, missing venv) -- nothing was run

set -Eeuo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PY="${PY:-.venv/bin/python}"
ROOT="${ROOT:-${OHIOT1DM_ROOT:-OhioT1DM}}"
LOG_DIR="${LOG_DIR:-runs/paper_results/logs}"
STUDIES="${STUDIES:-a b c d e f}"
# Training seeds of the multi-seed benchmark sweep (stage `seeds`).
SEEDS="${SEEDS:-42 123 456 789 1337}"
SEEDS_DIR="${SEEDS_DIR:-runs/paper_results/benchmarks_seeds}"
SMOKE=""
FROM=""
ONLY=""

STAGES=(data benchmarks seeds seed_analysis ablations personalization tables figures gates)

usage() {
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Stages, in order: ${STAGES[*]}"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)  SMOKE="--smoke"; shift ;;
        --from)   FROM="${2:?--from needs a stage}"; shift 2 ;;
        --only)   ONLY="${2:?--only needs a stage}"; shift 2 ;;
        --studies) STUDIES="${2:?--studies needs a list}"; shift 2 ;;
        --root)   ROOT="${2:?--root needs a directory}"; shift 2 ;;
        --seeds)  SEEDS="${2:?--seeds needs a list}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_HDR=$'\033[1;36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_HDR=""; C_DIM=""; C_OFF=""
fi

RUN_START=$(date +%s)
STAMP="$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/reproduce_${STAMP}.log"

log()  { printf '%s[%s]%s %s\n' "$C_DIM" "$(date +%H:%M:%S)" "$C_OFF" "$*" | tee -a "$RUN_LOG"; }
head_() { printf '\n%s=== %s ===%s\n' "$C_HDR" "$*" "$C_OFF" | tee -a "$RUN_LOG"; }
die()  { printf '%sFATAL%s %s\n' "$C_ERR" "$C_OFF" "$*" | tee -a "$RUN_LOG" >&2; exit 2; }

# Elapsed seconds -> "1h 04m 09s", so a multi-hour log is readable at a glance.
fmt_elapsed() {
    local s=$1
    if   [ "$s" -ge 3600 ]; then printf '%dh %02dm %02ds' $((s/3600)) $((s%3600/60)) $((s%60))
    elif [ "$s" -ge 60 ];   then printf '%dm %02ds' $((s/60)) $((s%60))
    else                         printf '%ds' "$s"; fi
}

# `set -e` aborts on the first failure; this reports WHERE before the shell
# unwinds, which a bare non-zero exit does not.
CURRENT_STAGE="startup"
on_err() {
    local code=$?
    printf '\n%sFAILED%s during stage %s (exit %d)\n' "$C_ERR" "$C_OFF" "$CURRENT_STAGE" "$code" \
        | tee -a "$RUN_LOG" >&2
    printf 'Partial artefacts were kept. Full log: %s\n' "$RUN_LOG" | tee -a "$RUN_LOG" >&2
    printf 'Resume with: %s --from %s\n' "$0" "$CURRENT_STAGE" | tee -a "$RUN_LOG" >&2
    exit 1
}
trap on_err ERR

# --------------------------------------------------------------------------- #
# Stage selection
# --------------------------------------------------------------------------- #
in_stages() { local n="$1"; shift; for s in "$@"; do [ "$s" = "$n" ] && return 0; done; return 1; }
in_stages "${FROM:-data}" "${STAGES[@]}" || die "--from: unknown stage '$FROM'"
[ -z "$ONLY" ] || in_stages "$ONLY" "${STAGES[@]}" || die "--only: unknown stage '$ONLY'"

SELECTED=()
reached=0
for s in "${STAGES[@]}"; do
    if [ -n "$ONLY" ]; then
        [ "$s" = "$ONLY" ] && SELECTED+=("$s")
    else
        [ -z "$FROM" ] || [ "$s" = "$FROM" ] && reached=1
        [ "$reached" = 1 ] && SELECTED+=("$s")
    fi
done
# `--only` on a later stage still needs the data check to have been satisfied
# by an earlier run; it is the caller's business, so we do not force it.

declare -a NAMES=() TIMES=()
run_stage() {
    local name="$1"; shift
    # Sub-steps (e.g. "seeds:42") keep the enclosing stage as the resume point.
    in_stages "$name" "${STAGES[@]}" && CURRENT_STAGE="$name"
    head_ "$name"
    log "\$ $*"
    local t0 rc
    t0=$(date +%s)
    # Tee rather than redirect: a multi-hour driver should stay watchable.
    set +e
    ( "$@" ) 2>&1 | tee -a "$RUN_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    [ "$rc" -eq 0 ] || return "$rc"
    local dt=$(( $(date +%s) - t0 ))
    NAMES+=("$name"); TIMES+=("$dt")
    log "${C_OK}ok${C_OFF} $name in $(fmt_elapsed "$dt")"
}

# --------------------------------------------------------------------------- #
head_ "CausalCGM-MT: reproducing the paper's artefacts"
log "repo      $REPO"
log "python    $PY"
log "data root $ROOT"
log "studies   $STUDIES"
log "seeds     $SEEDS"
log "stages    ${SELECTED[*]:-<none>}"
log "log       $RUN_LOG"
[ -n "$SMOKE" ] && log "${C_ERR}SMOKE MODE${C_OFF}: two-epoch budget. Results are labelled \"smoke\": true and are NOT publishable."

[ -x "$PY" ] || [ -n "$(command -v "$PY" || true)" ] \
    || die "interpreter not found: $PY  (python -m venv .venv && .venv/bin/pip install -r requirements.txt)"

for stage in "${SELECTED[@]}"; do
case "$stage" in

data)
    CURRENT_STAGE=data
    head_ "data"
    [ -d "$ROOT" ] || die "OhioT1DM not found at '$ROOT'. It is licensed and cannot be
       downloaded automatically: request it from
       https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html and place the
       release directories at $ROOT/2018 and $ROOT/2020, or set ROOT=..."
    missing=0
    for sub in 2018 2020; do
        for split in train test; do
            d="$ROOT/$sub/$split"
            if [ ! -d "$d" ]; then
                log "  MISSING $d"; missing=1
            else
                n=$(find "$d" -name '*.xml' | wc -l | tr -d ' ')
                log "  $d: $n XML files"
                [ "$n" -gt 0 ] || { log "  ...but no XML in it"; missing=1; }
            fi
        done
    done
    [ "$missing" -eq 0 ] || die "OhioT1DM layout is incomplete; see the paths above."
    # 12 subjects x 2 files. A short count means a partial copy, which would
    # otherwise surface hours later as a quietly smaller cohort.
    total=$(find "$ROOT" -name '*.xml' | wc -l | tr -d ' ')
    log "  total $total XML files (expected 24 for the full release)"
    [ "$total" -eq 24 ] || log "  ${C_ERR}NOTE${C_OFF}: not the full 12-subject cohort; every table will be scoped to what is present."
    log "${C_OK}ok${C_OFF} data"
    ;;

benchmarks)
    run_stage benchmarks "$PY" scripts/run_paper_benchmarks.py --root "$ROOT" $SMOKE
    ;;

seeds)
    CURRENT_STAGE=seeds
    if [ -n "$SMOKE" ]; then
        # aggregate_seed_sweep / fapd_latching_sensitivity refuse smoke inputs,
        # and a smoke sweep would overwrite real per-seed results. Skip.
        log "skip seeds: the multi-seed sweep does not run under --smoke"
        continue
    fi
    # Multi-seed sweep: the full benchmark, every arm, once per training seed.
    # Each seed writes its own results_summary.json (+ predictions/, which are
    # DUA-restricted and never shipped) under $SEEDS_DIR/seed<S>.
    # shellcheck disable=SC2086
    for seed in $SEEDS; do
        run_stage "seeds:$seed" "$PY" scripts/run_paper_benchmarks.py --root "$ROOT" \
            --seed "$seed" --out "$SEEDS_DIR/seed$seed" $SMOKE
    done
    ;;

seed_analysis)
    CURRENT_STAGE=seed_analysis
    if [ -n "$SMOKE" ]; then
        log "skip seed_analysis: its scripts refuse smoke artefacts"
        continue
    fi
    # Pure post-processing of the per-seed outputs: no model is refitted.
    #   aggregate_seed_sweep            -> multiseed_summary.json (+ tables)
    #   conformal_calibration_bootstrap -> conformal_bootstrap.json
    #   fapd_latching_sensitivity       -> fapd_latching/fapd_latching.json
    run_stage seed_aggregate "$PY" scripts/aggregate_seed_sweep.py --seeds-dir "$SEEDS_DIR"
    run_stage conformal_bootstrap "$PY" scripts/conformal_calibration_bootstrap.py \
        --seeds-dir "$SEEDS_DIR" --out runs/paper_results/conformal_bootstrap.json
    run_stage fapd_latching "$PY" scripts/fapd_latching_sensitivity.py \
        --seeds-dir "$SEEDS_DIR" --out runs/paper_results/fapd_latching
    ;;

ablations)
    # Word-split on purpose: STUDIES is a list of letters.
    # shellcheck disable=SC2086
    run_stage ablations "$PY" scripts/run_paper_ablations.py --root "$ROOT" --studies $STUDIES $SMOKE
    ;;

personalization)
    run_stage personalization "$PY" scripts/run_ohio_personalization_sweep.py --root "$ROOT" $SMOKE
    ;;

tables)
    # Pure function of the JSONs: refits nothing, so it is safe to re-run and
    # must come after every driver that writes one.
    #
    # The renderer REFUSES artefacts marked "smoke": true, because rendering one
    # produces a table indistinguishable from a real result. In smoke mode that
    # refusal would stop the very run that is exercising the plumbing, so this
    # is the one caller allowed to override it -- and the banner above already
    # says the output is not publishable.
    run_stage tables "$PY" scripts/rebuild_paper_tables.py \
        ${SMOKE:+--allow-smoke}
    ;;

figures)
    run_stage figures "$PY" scripts/make_paper_figures.py
    ;;

gates)
    run_stage gates "$PY" scripts/check_doc_structure.py
    ;;

esac
done

# --------------------------------------------------------------------------- #
head_ "summary"
for i in "${!NAMES[@]}"; do
    printf '  %-16s %s\n' "${NAMES[$i]}" "$(fmt_elapsed "${TIMES[$i]}")" | tee -a "$RUN_LOG"
done
printf '  %-16s %s\n' "TOTAL" "$(fmt_elapsed $(( $(date +%s) - RUN_START )))" | tee -a "$RUN_LOG"
log "log written to $RUN_LOG"

if [ -n "$SMOKE" ]; then
    printf '\n%sThese are smoke artefacts.%s Every JSON is labelled "smoke": true.\n' "$C_ERR" "$C_OFF" | tee -a "$RUN_LOG"
    printf 'Re-run without --smoke before citing anything.\n' | tee -a "$RUN_LOG"
else
    printf '\n%sDone.%s Next: package a release with\n' "$C_OK" "$C_OFF" | tee -a "$RUN_LOG"
    printf '  %s scripts/package_release.py --out dist/\n' "$PY" | tee -a "$RUN_LOG"
fi
