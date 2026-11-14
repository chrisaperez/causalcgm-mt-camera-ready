"""MetaboNet benchmark driver -- isolated from the paper's OhioT1DM artefacts.

    python scripts/run_metabonet_benchmarks.py --smoke
    python scripts/run_metabonet_benchmarks.py --horizon 30 --folds 5
    python scripts/run_metabonet_benchmarks.py --arms cgm_only fusion --max-subjects 60

    # the whole corpus: cap the windows, budget the memory, cache and resume
    python scripts/run_metabonet_benchmarks.py \
        --max-windows-per-subject 400 --memory-budget-gb 12 \
        --window-cache runs/metabonet_results/window_cache

SCALE
-----
1,873 subjects and 154,353,245 rows, so nothing here loads the corpus. Ingestion
streams one subject at a time through `pl.scan_parquet`, and windowing writes
per-subject shards to a disk-backed store whose big arrays are memmapped at the
end -- peak RSS is set by the largest single subject, not by the corpus.

Windowing at stride 1 with no cap would produce ~134M windows, ~356 GB of
tensors, which is not a RAM problem streaming can solve. `--max-windows-per-subject`
is therefore part of the protocol, and both the store size and the per-fold fit
rows are projected and checked against `--memory-budget-gb` before the work is
done, so a run that cannot finish says so in its first minute instead of being
killed by the OS an hour in.

Everything is resumable: the window store is keyed by a fingerprint of the
window geometry, and each (arm, fold) result is checkpointed under `_folds/` as
it completes.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This runs the repository's model arms on MetaboNet and writes every artefact
under `runs/metabonet_results/`. It is deliberately NOT part of
`reproduce_all_paper_results.sh`, does not write into `runs/paper_results/`, and
the only thing it produces that `scripts/package_release.py` ships is the one
appendix table, named explicitly -- that script ships an explicit allow-list of
paths rather than a glob, so isolation here is structural rather than a
convention to remember.

WHY THE PROTOCOL IS PATIENT-WISE CV, AND WHY THAT MATTERS
---------------------------------------------------------
`train.run_fold` splits by PATIENT: whole subjects move to the test side, and it
answers "can this forecast a stranger". MetaboNet also ships its own `is_test`
column, which for 164 subjects is a WITHIN-subject temporal holdout -- a
different question, answered by `metabonet_loader.official_metabonet_split`.
The two are not interchangeable, so this driver runs one of them (CV) and says
so in every artefact it writes, rather than mixing rows from both into a table
that reads like one experiment. That is the same rule `run_experiment.py`
enforces when it refuses an OhioT1DM artefact.

Under the CV protocol a subject's `is_test` tag is irrelevant -- the whole
subject is on one side of the fold either way -- so both sides of a temporal
subject's record are used, and the shipped tag is recorded but not acted on.

THE CONTAMINATION GUARD, AND WHY IT RUNS THREE TIMES
-----------------------------------------------------
MetaboNet consolidates fourteen studies, two of which are this repository's
benchmarks: OhioT1DM (all 12 subjects, including 31,734 CGM samples inside the
OFFICIAL TEST spans, every row labelled `is_test=False`) and HUPA-UCM. A model
fitted on those rows cannot be reported against the Ohio or HUPA protocols, and
no downstream metric reveals the problem -- the model simply looks good.

`assert_no_benchmark_contamination` is therefore called at three points, because
each catches something the others cannot:

  1. on the SELECTED subjects, before any windowing -- fails in seconds rather
     than after an hour of fitting;
  2. on the BUILT window set, which is what would actually be trained on -- the
     selection and the build are separate steps and a future edit could
     reintroduce a subject between them;
  3. on EVERY TEST FOLD, which is the claim a reader of the results table
     actually relies on.

`--allow-benchmark-source` waives the guard per source, explicitly, and the
waiver is written into the results JSON so a contaminated run can never be
mistaken for a clean one.

ARMS
----
Three, specified as `ArmSpec` exactly as the paper drivers do, so an arm's
hyperparameters are serialised verbatim into the results and a number traces
back to what produced it:

    cgm_only   unimodal causal TCN -- the baseline every other arm must beat
    fusion     dual-branch multimodal TCN over MetaboNet's 24 aux channels
    lora       post-training personalisation of the `cgm_only` fold model

`lora` adapts the UNIMODAL base on purpose: `extension.adapt_patient` runs the
generic held-out-subject protocol on the CGM path only (`multimodal=False` at
its forward and BN-recalibration calls). Adapting the fusion model would need
the Ohio-specific `adapt_ohio_subject`, which is bound to `OhioPreprocessor` and
`OhioSplit` and does not apply to a patient-wise fold. Running the unimodal arm
and labelling it honestly is better than running something else under the name.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import platform
import sys
import time
from types import SimpleNamespace
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B
from causalcgm_mt.benchmark import ArmSpec, Column, horizon_block
from causalcgm_mt.config import DataConfig
from causalcgm_mt.data import WindowSet, split_by_patient
from causalcgm_mt.metabonet_loader import (
    METABONET_AUX_CHANNELS, METABONET_WINDOW_VERSION, MetaboNetConfig,
    MetaboNetWindowSet, assert_no_benchmark_contamination,
    build_metabonet_windows_streaming, configure_logging, memory_note,
    metabonet_folds, scan_subjects, split_subject_key, window_bytes,
)
from scripts._cli import NO_DATA_HINT

log = logging.getLogger("causalcgm_mt.metabonet.runner")

DEFAULT_PARQUET = "metabonet_public.parquet"
DEFAULT_OUT = "runs/metabonet_results"

# The one tree this driver may not write into, as an absolute path resolved
# from THIS FILE. It used to be `os.path.abspath("runs/paper_results")`, which
# resolves against the current working directory: run from anywhere but the
# repository root -- a cron job, a Docker WORKDIR, `cd runs && python ../...`
# -- and the guard compared against a directory that does not exist, so every
# `--out` passed, the real paper tree included.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER_TREE = (REPO_ROOT / "runs" / "paper_results").resolve()


def is_inside_paper_tree(out: str | os.PathLike) -> bool:
    """True when `out` is the paper artefact tree, or anything under it.

    Compared component-wise on RESOLVED paths, not by string prefix: a prefix
    test has no separator boundary, so it refused a perfectly legitimate
    `runs/paper_results_metabonet` while a symlink pointing into the real tree
    walked straight past it. Resolving also means a relative `--out` is judged
    against the repository, wherever the process happens to be running.
    """
    target = pathlib.Path(out).expanduser().resolve()
    # TWO anchors, because the hazard has two shapes and only one of them is
    # about disk. `<repo>/runs/paper_results` is the tree `rebuild_paper_tables`,
    # `package_release` and `tests/test_manuscript.RESULT_ROOTS` actually read,
    # so it is the invariant. The cwd-relative one is kept because an operator
    # who types `--out runs/paper_results` from anywhere means THE paper tree
    # whatever it happens to resolve to, and refusing costs nothing: this
    # function may only ever grow the refusal set, never shrink it.
    anchors = (PAPER_TREE,
               (pathlib.Path.cwd() / "runs" / "paper_results").resolve())
    return any(target == a or a in target.parents for a in anchors)


# The table this driver renders. A subset of `benchmark.SOTA_COLUMNS`: the
# conformal columns are dropped because the CV protocol exposes no calibration
# split that is disjoint from both fit and test in the way `score_arm`'s
# conformal block requires. A column that cannot be computed is left out rather
# than filled with a number that means something else.
METABONET_COLUMNS: tuple[Column, ...] = tuple(
    c for c in B.SOTA_COLUMNS if not c.path.startswith("conformal.")
)


# The comparison table is the ONE artefact of this driver that the manuscript
# reads: `manuscript/main.tex` \inputs it straight out of this tree, because
# the corpus-scale validation is not a stage of the paper pipeline and is
# forbidden to write into `runs/paper_results/`.
#
# That makes it the one file a smoke run must not touch. It used to be
# overwritten unconditionally, with the only warning folded into the table's own
# caption -- so a two-epoch run replaced the appendix table while the
# `\NumMetabo*` macros beside it stayed at whatever the last publication run
# emitted, and the paper compiled clean describing two different experiments.
# `rebuild_paper_tables.metabonet_numbers` refuses a smoke summary by emitting
# NO macros, so the two halves fail in opposite directions and cannot converge
# on their own.
MANUSCRIPT_TABLE_NAME: str = "table_metabonet_comparison.tex"

# Deliberately NOT `.smoke.tex`: a sidecar that still ends in `.tex` can be
# picked up by a future `table_*.tex` glob or `\input`ed by a typo, and this
# repository's own rule is that globs are the wrong primitive for a safety
# boundary. A suffix past the extension can be neither.
SMOKE_TABLE_SUFFIX: str = ".smoke"


def comparison_table_path(bench_dir: str, smoke: bool,
                          allow_smoke_table: bool = False) -> str:
    r"""Where this run is allowed to write the comparison table.

    A publication-budget run writes the manuscript's `\input` target. A smoke
    run is diverted to a sidecar beside it, so the table stays readable for
    whoever asked for the smoke run while the published one is left alone.
    `--allow-smoke-table` overrides, for the operator who genuinely means it.
    """
    path = os.path.join(bench_dir, MANUSCRIPT_TABLE_NAME)
    if smoke and not allow_smoke_table:
        return path + SMOKE_TABLE_SUFFIX
    return path


def _nanmean(values) -> float:
    """Mean ignoring nan, returning nan for an all-nan input without warning.

    An all-nan metric is a real outcome here -- PR-AUC is undefined on a fold
    whose evaluation region holds no positive window -- so nan is the answer,
    and numpy's RuntimeWarning about it is noise in a log that is meant to be
    read.
    """
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def build_arms(horizon: int, epochs: int, seed: int, lora_rank: int,
               prefix_hours: float, adapt_steps: int) -> list[ArmSpec]:
    """The specs, as data. `horizons` is a 1-tuple by construction -- see below."""
    common = dict(horizons=(horizon,), epochs=epochs, seed=seed)
    return [
        ArmSpec(name="cgm_only", family="tcn", unimodal=True,
                notes="unimodal causal TCN on CGM features only", **common),
        ArmSpec(name="fusion", family="tcn_mm", unimodal=False,
                notes="dual-branch fusion over MetaboNet's aux channels", **common),
        ArmSpec(name="lora", family="lora", unimodal=True, base_arm="cgm_only",
                adapt_method="lora", lora_rank=lora_rank,
                adapt_prefix_hours=prefix_hours, adapt_steps=adapt_steps,
                notes="per-subject LoRA on the cgm_only fold model "
                      "(adapt_patient is CGM-only; see module docstring)",
                **common),
    ]


def check_single_horizon(specs: list[ArmSpec]) -> None:
    """`run_fold` trains ONE horizon: it reads `ws.y_reg` / `ws.y_min`, which
    mirror the primary horizon only.

    A multi-horizon spec would build a `2K`-head model through `arm_config`'s
    `n_horizons` and then feed it single-horizon targets. Rather than let that
    fail deep inside the loss, or silently train head 0 and leave head 1 at its
    initialisation, it is refused here with the reason.
    """
    bad = [s.name for s in specs if len(s.horizons) != 1]
    if bad:
        raise SystemExit(
            f"arms {bad} request multiple horizons, but this driver trains "
            f"through train.run_fold, which is single-horizon. Run it once per "
            f"horizon (--horizon 30, then --horizon 60); the multi-horizon "
            f"path is scripts/train_ohio.py's.")


# --------------------------------------------------------------------------- #
# Window-set views
# --------------------------------------------------------------------------- #
def unimodal_view(ws: MetaboNetWindowSet) -> WindowSet:
    """The same rows and targets, with the auxiliary branch removed entirely.

    `run_fold` detects multimodality with `hasattr(ws, "x_aux")`, so a unimodal
    arm needs a window set that genuinely lacks it -- not one whose aux tensor
    is zeroed. Zeroing would still build the fusion model, still run its
    attention and gate, and still spend parameters on a branch that carries no
    information: a different model reported under the baseline's name.

    Every array is passed through by reference, so the two views are the same
    rows in the same order and their metrics are directly comparable.
    """
    return WindowSet(
        x=ws.x, y_reg=ws.y_reg, y_cls_hard=ws.y_cls_hard, y_min=ws.y_min,
        patient=ws.patient, t_index=ws.t_index, t_time=ws.t_time,
        last_bg=ws.last_bg,
    )


def arm_view(ws: MetaboNetWindowSet, spec: ArmSpec):
    return unimodal_view(ws) if spec.unimodal else ws


# --------------------------------------------------------------------------- #
# Re-using one fold's model
# --------------------------------------------------------------------------- #
def rebuild_model(view, res, cfg):
    """Reconstruct a fold's trained model (and its criterion) from `FoldResult`.

    The criterion's learned log-variances and fold class prior are restored
    rather than re-initialised: they are fitted population quantities, and
    personalising under a different task balance than the model was trained
    with would be invisible in the adapted model's own metrics.

    There is deliberately no `pos_weight` fallback parameter. It used to default
    to 1.0, so a `FoldResult` with no criterion state silently dropped the
    rare-event class reweighting (~18x at 5% prevalence) instead of failing.
    `criterion_from_state` raises on that state, and the value is restored from
    the checkpoint's own buffer in every reachable case.
    """
    from causalcgm_mt.data import uses_unmasked_channels
    from causalcgm_mt.models import (build_model, build_multimodal_model,
                                     criterion_from_state,
                                     load_state_dict_compat)

    multimodal = hasattr(view, "x_aux")
    if multimodal:
        names = tuple(getattr(view, "aux_channels", METABONET_AUX_CHANNELS))
        mask_idx = tuple(i for i, c in enumerate(names) if c.endswith("_observed"))
        model, _fresh = build_multimodal_model(
            view.x.shape[-1], view.x_aux.shape[-1], cfg.model,
            aux_valid_index=mask_idx,
            always_valid=uses_unmasked_channels(names))
    else:
        model, _fresh = build_model(view.x.shape[-1], cfg.model)

    load_state_dict_compat(model, res.state_dict)
    criterion = criterion_from_state(res.criterion_state_dict, cfg.model,
                                     context=f"fold {res.fold}")
    return model, criterion, multimodal


def predict_rows(view, idx: np.ndarray, res, cfg, batch_size: int = 512):
    """`(bg_hat, risk)` for `idx`, using the fold's OWN frozen preprocessing.

    Nothing is refit here: the scaler, the auxiliary scaler and the soft-label
    band all come from `FoldResult`, where they were fitted on that fold's fit
    patients alone.
    """
    import torch
    from torch.utils.data import DataLoader

    from causalcgm_mt.data import aux_presence
    from causalcgm_mt.train import WindowDataset, pick_device, predict

    model, _criterion, multimodal = rebuild_model(view, res, cfg)
    device = pick_device()
    model = model.to(device)

    ds = WindowDataset(
        res.scaler.transform(view.x[idx]),
        view.y_reg[idx],
        res.labeler(view.y_min[idx]),
        view.last_bg[idx],
        x_aux=(res.aux_scaler.transform(view.x_aux[idx]) if multimodal else None),
        aux_present=(aux_presence(view.x_aux[idx],
                                  getattr(view, "aux_channels", METABONET_AUX_CHANNELS))
                     if multimodal else None),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        return predict(model, dl, device, multimodal)


# --------------------------------------------------------------------------- #
# One arm on one fold
# --------------------------------------------------------------------------- #
from causalcgm_mt.data import aux_presence  # noqa: E402


def run_arm_fold(spec: ArmSpec, ws: MetaboNetWindowSet, data_cfg: DataConfig,
                 train_pat: np.ndarray, test_pat: np.ndarray, fold: int,
                 predictions_dir: str | None, base_cache: dict) -> dict:
    """Fit (or reuse) one arm on one fold and return its serialisable result."""
    from causalcgm_mt.train import run_fold

    view = arm_view(ws, spec)
    cfg = B.arm_config(spec, data_cfg)
    horizon = int(spec.horizons[0])

    t0 = time.time()
    i_test = split_by_patient(view, test_pat)

    common = {
        "arm": spec.name,
        "family": spec.family,
        "fold": fold,
        "spec": spec.as_dict(),
        "protocol": "patient_wise_cv",
        "horizon_min": horizon,
        "test_subjects": sorted({str(p) for p in ws.patient[i_test]}),
        "test_sources": sorted({split_subject_key(str(p))[0]
                                for p in ws.patient[i_test]}),
        "n_test_subjects": len(set(map(str, test_pat))),
        "n_test": int(i_test.size),
    }

    if spec.family == "lora":
        # Personalisation adapts the BASE arm's fold model. Refitting a second
        # population model here would make the before/after comparison
        # unpaired -- the whole point of the arm is a paired delta.
        base_key = (spec.base_arm, fold)
        if base_key not in base_cache:
            raise SystemExit(
                f"arm 'lora' needs arm '{spec.base_arm}' on fold {fold} trained "
                f"in THIS process: it adapts the model, and a checkpoint "
                f"carries only that arm's metrics, not its weights. Either "
                f"include '{spec.base_arm}' in --arms, or re-run with "
                f"--no-resume so the base is refitted.")
        res, base_spec = base_cache[base_key]
        adapt_cfg = B.arm_adapt_config(spec, B.arm_config(base_spec, data_cfg))
        block = run_personalization_arm(spec, view, res, adapt_cfg, test_pat)
        # NO `per_horizon` row, deliberately. Every adapted model is scored on
        # its own subject's post-embargo evaluation region -- the prefix it
        # adapted on, and the embargo after it, are excluded -- so the LoRA
        # numbers do not live on the same rows as `cgm_only` and `fusion`, and
        # putting them in the same table would invite exactly the comparison
        # they do not support. The paired before/after IS the result here, which
        # is how the OhioT1DM personalisation table reports it too.
        return {
            **common,
            "scored_on": "per_subject_post_embargo_eval_regions",
            "paired_against": spec.base_arm,
            "seconds": round(time.time() - t0, 1),
            "per_horizon": {},
            **block,
        }

    res = run_fold(view, cfg, train_pat, test_pat, fold)
    base_cache[(spec.name, fold)] = (res, spec)

    bg_hat, risk = predict_rows(view, i_test, res, cfg)

    # Alarm thresholds come from the fold's inner-VALIDATION subjects -- held
    # out of fitting, disjoint from the test subjects -- and are applied to
    # test unchanged. This protocol has no separate calibration region, and
    # fitting thresholds on the test subjects' labels would be an oracle.
    from causalcgm_mt.evaluate import calibration_thresholds, evaluate
    i_val = split_by_patient(view, np.asarray(res.val_patients))
    _, risk_val = predict_rows(view, i_val, res, cfg)
    thresholds = calibration_thresholds(view.y_cls_hard[i_val], risk_val)
    rep = evaluate(
        view.y_reg[i_test], bg_hat, view.y_cls_hard[i_test], risk,
        view.last_bg[i_test], view.patient[i_test], data_cfg.sample_period_min,
        t_time=view.t_time[i_test], horizon_min=horizon,
        alarm_thresholds=thresholds,
    )

    if predictions_dir:
        save_fold_predictions(predictions_dir, spec, ws, i_test, fold,
                              bg_hat, risk)

    # `run_fold` already scored these rows with `evaluate`. Rescoring them from
    # the restored state dict must agree: a mismatch means the model was not
    # reconstructed faithfully, which would silently corrupt the prediction
    # dumps and the LoRA arm's base. Checked rather than assumed.
    drift = abs(float(res.report.rmse) - float(rep.rmse))
    if drift > 1e-3:
        raise SystemExit(
            f"{spec.name} fold {fold}: rescoring the restored model gives RMSE "
            f"{rep.rmse:.6f} but training reported {res.report.rmse:.6f} "
            f"(drift {drift:.2e}). The checkpoint or its scalers were not "
            f"restored faithfully; the dumps would not match the table.")

    return {
        **common,
        "scored_on": "fold_test_rows",
        "threshold_source": "inner_validation_subjects",
        "fusion_active_frac_test": (
            float(aux_presence(view.x_aux[i_test],
                               getattr(view, "aux_channels", METABONET_AUX_CHANNELS)).mean())
            if hasattr(view, "x_aux") else None),
        "n_fit_subjects": len(res.scaler.fitted_on),
        "n_val_subjects": len(res.val_patients),
        "scaler_fitted_on": list(res.scaler.fitted_on),
        "soft_label": res.labeler.as_dict(),
        "seconds": round(time.time() - t0, 1),
        "train_loop_rmse": float(res.report.rmse),
        "epochs_run": len(res.history),
        "history": res.history,
        "per_horizon": {str(horizon): horizon_block(rep, view.y_reg[i_test], bg_hat)},
    }


def run_personalization_arm(spec: ArmSpec, view, res, cfg, test_pat) -> dict:
    """Adapt the base fold model per held-out subject; return the paired deltas.

    `run_personalization` scores before/after on the SAME evaluation windows per
    subject, which is what licenses the causal claim about adaptation. The
    aggregate deltas are reported here; the arm's table row is scored separately
    on the fold's full test rows by the caller, so it stays comparable to the
    other arms.
    """
    from causalcgm_mt.extension import run_personalization

    model, criterion, _mm = rebuild_model(view, res, cfg)
    results = run_personalization(
        model, view, [str(p) for p in test_pat], res.scaler, res.labeler,
        criterion, cfg)

    if not results:
        # Loud, because a silently empty personalisation arm is indistinguishable
        # in a results file from one that ran and found no effect.
        print("  WARNING: no subject had enough data to adapt. `adapt_patient` "
              "needs an adaptation prefix of at least AdaptConfig.batch_size "
              "windows and 50 evaluation windows after the embargo; at "
              f"prefix_hours={spec.adapt_prefix_hours} and this stride, no "
              "subject cleared that.", file=sys.stderr)
        return {"personalization": {
            "n_subjects": 0,
            "note": "no subject had enough prefix and evaluation data to adapt; "
                    "this arm produced no result, it did not produce a null one",
        }}
    d_rmse = [r.after.rmse - r.before.rmse for r in results]
    d_pr = [r.after.pr_auc - r.before.pr_auc for r in results]
    return {"personalization": {
        "n_subjects": len(results),
        "method": spec.adapt_method,
        "lora_rank": spec.lora_rank,
        "prefix_hours": spec.adapt_prefix_hours,
        "n_trainable": int(results[0].n_trainable),
        "mean_delta_rmse": _nanmean(d_rmse),
        "mean_delta_pr_auc": _nanmean(d_pr),
        "n_improved_rmse": int(sum(1 for d in d_rmse if d < 0)),
        "per_subject": [
            {"subject": r.patient, "n_adapt_windows": r.n_adapt_windows,
             "rmse_before": r.before.rmse, "rmse_after": r.after.rmse,
             "pr_auc_before": r.before.pr_auc, "pr_auc_after": r.after.pr_auc}
            for r in results],
    }}


def save_fold_predictions(directory: str, spec: ArmSpec, ws, i_test, fold,
                          bg_hat, risk) -> str:
    """Per-window dump, in the same format `benchmark.save_predictions` writes.

    Reused rather than reimplemented so a MetaboNet dump and an Ohio dump can be
    read by the same figure code. The adapters carry only what that function
    touches.
    """
    path = os.path.join(directory, f"{spec.name}_fold{fold}.npz")
    data = SimpleNamespace(ws=ws, horizons=ws.horizons_min)
    split = SimpleNamespace(test=i_test)
    pred = B.Predictions(bg=bg_hat, risk=risk)
    return B.save_predictions(path, spec, data, split, pred)


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #
def load_cohort(args) -> tuple[MetaboNetWindowSet, DataConfig, dict]:
    """Scan, select, guard, load and window. Returns (ws, data_cfg, provenance)."""
    mcfg = MetaboNetConfig(
        exclude_sources=tuple(s for s in MetaboNetConfig().exclude_sources
                              if s not in set(args.allow_benchmark_source)),
        include_sources=tuple(args.sources) if args.sources else None,
    )
    data_cfg = DataConfig(history_min=args.history, horizon_min=args.horizon,
                          stride_steps=args.stride)
    mcfg = dataclasses.replace(
        mcfg, max_windows_per_subject=args.max_windows_per_subject,
        memory_budget_gb=args.memory_budget_gb)

    index = scan_subjects(args.parquet_path, mcfg=mcfg)
    print(index.summary())

    keys = index.keys()
    if args.max_subjects and len(keys) > args.max_subjects:
        # Deterministic subsample, stratified by source so a small run still
        # spans the corpus rather than the first study alphabetically.
        rng = np.random.default_rng(args.seed)
        by_source: dict[str, list[str]] = {}
        for k in keys:
            by_source.setdefault(split_subject_key(k)[0], []).append(k)
        picked: list[str] = []
        per = max(1, args.max_subjects // max(len(by_source), 1))
        for src in sorted(by_source):
            pool = sorted(by_source[src])
            take = min(per, len(pool))
            picked += [pool[i] for i in
                       rng.choice(len(pool), size=take, replace=False)]
        keys = sorted(picked)[: args.max_subjects]
        print(f"\nsubsampled to {len(keys)} subjects across "
              f"{len({split_subject_key(k)[0] for k in keys})} sources")

    # GUARD 1: before anything expensive happens.
    assert_no_benchmark_contamination(keys, allow=args.allow_benchmark_source)

    subject_rows = dict(zip(index.frame.key, index.frame.rows.astype(int)))
    selected_rows = sum(subject_rows.get(k, 0) for k in keys)
    log.info("selected %d subjects, %s rows; windowing streams one subject at a "
             "time (%s)", len(keys), f"{selected_rows:,}", memory_note())

    ws = build_metabonet_windows_streaming(
        args.parquet_path, keys, data_cfg, mcfg, horizons_min=(args.horizon,),
        cache_dir=args.window_cache, resume=not args.no_resume,
        subject_rows=subject_rows)

    # GUARD 2: on what would actually be trained on.
    assert_no_benchmark_contamination(ws.subjects(), allow=args.allow_benchmark_source)

    if ws.x_aux.shape[-1] != len(ws.aux_channels):
        raise SystemExit(
            f"aux tensor has {ws.x_aux.shape[-1]} channels but the layout names "
            f"{len(ws.aux_channels)}; the fusion arm's masks would be misread.")

    # What TRAINING will cost, as distinct from what the store costs. run_fold
    # materialises the fit rows through the scaler, so the peak is a multiple of
    # one fold's share of the tensors -- a store that fits on disk can still be
    # a fold that does not fit in RAM, and that is the shape of the OOM this
    # driver is built to avoid.
    per_window = window_bytes(data_cfg, ws.x_aux.shape[-1], len(ws.horizons_min))
    fit_share = 1.0 - 1.0 / max(args.folds, 2)
    fold_gb = len(ws) * per_window * fit_share / 2**30
    log.info("%s windows | %d subjects | %d aux channels | hypo %.2f%% | "
             "%.1f GB of tensors, ~%.1f GB per fold's fit rows (%s)",
             f"{len(ws):,}", len(ws.subjects()), ws.x_aux.shape[-1],
             100 * float(ws.y_cls_hard.mean()),
             len(ws) * per_window / 2**30, fold_gb, memory_note())
    if args.memory_budget_gb and fold_gb > args.memory_budget_gb:
        raise SystemExit(
            f"one fold's fit rows are ~{fold_gb:.1f} GB, over the "
            f"{args.memory_budget_gb:.1f} GB budget. run_fold scales them into "
            f"a second copy, so expect roughly twice that resident.\n"
            f"  Reduce --max-windows-per-subject, raise --stride, or use fewer "
            f"subjects. The window cache at {args.window_cache} is reusable.")

    provenance = {
        # BASENAME, not abspath: this summary is packaged in the public
        # tier and an absolute path here publishes the operator's home
        # directory and local project name. The filename is the part that
        # identifies the corpus; where it sat on one machine is not
        # provenance anyone downstream can use.
        "parquet": os.path.basename(args.parquet_path),
        "excluded_sources": list(mcfg.exclude_sources),
        "allowed_benchmark_sources": list(args.allow_benchmark_source),
        "sources": sorted({split_subject_key(k)[0] for k in ws.subjects()}),
        "n_subjects": len(ws.subjects()),
        "n_windows": int(len(ws)),
        "aux_channels": list(ws.aux_channels),
        "split_class_counts": index.frame.split_class.value_counts().to_dict(),
        "max_windows_per_subject": args.max_windows_per_subject,
        "stride_steps": args.stride,
        "window_cache": args.window_cache,
        "bytes_per_window": per_window,
        "shipped_is_test_used": False,
        "window_pipeline_version": B.WINDOW_PIPELINE_VERSION,
    }
    return ws, data_cfg, provenance


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(rows: list[dict], horizon: int) -> list[dict]:
    """Mean over folds per arm, in the same shape a single row has.

    Folds are averaged, not pooled: pooling every fold's windows into one metric
    would weight a fold by how many windows its subjects happen to contribute,
    so a handful of long records would dominate the mean. `n_folds` is carried
    so a reader can see how many numbers went into each.
    """
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    out = []
    for arm, group in by_arm.items():
        # An arm scored on a different row set (the LoRA arm, on per-subject
        # post-embargo regions) has no `per_horizon` block and is summarised
        # separately -- averaging it into a table whose other rows are the
        # fold's full test set would compare numbers computed on different data.
        blocks = [g["per_horizon"][str(horizon)] for g in group
                  if str(horizon) in g.get("per_horizon", {})]
        if not blocks:
            out.append({
                "arm": arm,
                "family": group[0]["family"],
                "spec": group[0]["spec"],
                "n_folds": len(group),
                "scored_on": group[0].get("scored_on"),
                "paired_against": group[0].get("paired_against"),
                "seconds": round(sum(g["seconds"] for g in group), 1),
                "per_horizon": {},
                **_merge_personalization(group),
            })
            continue
        merged: dict = {}
        for section in ("regression", "classification", "alarm", "clinical"):
            merged[section] = {}
            for key in blocks[0][section]:
                vals = [b[section][key] for b in blocks]
                if all(isinstance(v, (int, float)) for v in vals):
                    # An all-nan column means the metric could not be computed
                    # on any fold -- nan is the answer, not an error, so the
                    # warning numpy raises about it is noise.
                    merged[section][key] = _nanmean(vals)
                else:
                    merged[section][key] = vals[0]
        merged["n"] = int(sum(b["n"] for b in blocks))
        merged["n_positive"] = int(sum(b["n_positive"] for b in blocks))

        row = {
            "arm": arm,
            "family": group[0]["family"],
            "spec": group[0]["spec"],
            "n_folds": len(group),
            "n_test_total": int(sum(g["n_test"] for g in group)),
            "seconds": round(sum(g["seconds"] for g in group), 1),
            "per_horizon": {str(horizon): merged},
        }
        row.update(_merge_personalization(group))
        out.append(row)
    return sorted(out, key=lambda r: r["arm"])


def _merge_personalization(group: list[dict]) -> dict:
    """Pool the paired per-subject deltas across folds.

    Subjects are the unit, not folds: each subject is adapted and scored
    independently, and every subject appears in exactly one fold, so pooling
    subjects is the same population either way and keeps the count honest.
    """
    pers = [g["personalization"] for g in group if "personalization" in g]
    if not pers:
        return {}
    subjects = [s for p in pers for s in p.get("per_subject", [])]
    d_rmse = [s["rmse_after"] - s["rmse_before"] for s in subjects]
    d_pr = [
        s["pr_auc_after"] - s["pr_auc_before"]
        for s in subjects
        if s["pr_auc_after"] is not None and s["pr_auc_before"] is not None
    ]
    merged = {
        "n_subjects": len(subjects),
        "n_folds_attempted": len(pers),
        "notes": sorted({p["note"] for p in pers if "note" in p}),
    }
    if subjects:
        # No empty-list guard needed: `_nanmean` returns nan for an empty or
        # all-nan input, which is the honest answer when every subject's
        # PR-AUC was undefined.
        merged.update({
            "method": pers[0].get("method"),
            "lora_rank": pers[0].get("lora_rank"),
            "prefix_hours": pers[0].get("prefix_hours"),
            "n_trainable": pers[0].get("n_trainable"),
            "mean_delta_rmse": _nanmean(d_rmse),
            "mean_delta_pr_auc": _nanmean(d_pr),
            "n_improved_rmse": int(sum(1 for d in d_rmse if d < 0)),
            "per_subject": subjects,
        })
    return {"personalization": merged}


def caption(horizon: int, n_subjects: int, n_folds: int, sources: list[str],
            smoke: bool) -> str:
    r"""The MetaboNet table's caption. Says what protocol produced it."""
    smoke_note = (r"\textbf{SMOKE RUN -- two-epoch budget, not a result.} "
                  if smoke else "")
    return (
        f"{smoke_note}MetaboNet benchmark at PH~=~{horizon}~min, "
        f"{n_folds}-fold patient-wise cross validation over {n_subjects} "
        f"subjects from {len(sources)} source studies "
        f"({', '.join(sources)}). Whole subjects are held out, so the protocol "
        f"measures generalisation to an unseen individual; MetaboNet's own "
        f"\\texttt{{is\\_test}} column is a within-subject temporal split and is "
        f"not used here. OhioT1DM and HUPA-UCM are excluded from the corpus, "
        f"since both are benchmarks reported elsewhere in this work. Values are "
        f"means over folds."
    )


def run_fingerprint(args, keys: Sequence[str]) -> str:
    """Identity of a run, for deciding whether a checkpoint may be reused.

    Everything that changes what an arm computes goes in. A checkpoint whose
    fingerprint differs is ignored rather than adopted: resuming a run with a
    different horizon, seed, fold count or subject set would assemble one table
    out of two experiments, and the table would not say so.
    """
    payload = {
        "horizon": args.horizon, "history": args.history, "stride": args.stride,
        "folds": args.folds, "epochs": args.epochs, "seed": args.seed,
        "arms": sorted(args.arms), "group_by_source": bool(args.group_by_source),
        "max_windows_per_subject": args.max_windows_per_subject,
        # The cap alone does not say WHICH windows it keeps. The window store
        # is rebuilt when this changes (its own fingerprint carries it), but a
        # fold checkpoint written before the change would still be adopted --
        # and a table assembled half from the old sample and half from the new
        # would not say so. Version 3 moved every subject's sample from the
        # opening ~33 h of its record to two blocks spanning it.
        "metabonet_window_version": METABONET_WINDOW_VERSION,
        "lora_rank": args.lora_rank, "prefix_hours": args.prefix_hours,
        "adapt_steps": args.adapt_steps,
        "subjects": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:16],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def checkpoint_path(fold_dir: str, arm: str, fold: int) -> str:
    return os.path.join(fold_dir, f"{arm}_fold{fold}.json")


def load_checkpoints(fold_dir: str, fingerprint: str) -> dict[tuple[str, int], dict]:
    """Completed (arm, fold) results from a previous run of the same experiment."""
    out: dict[tuple[str, int], dict] = {}
    if not os.path.isdir(fold_dir):
        return out
    for name in sorted(os.listdir(fold_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(fold_dir, name)) as f:
                row = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("ignoring unreadable checkpoint %s (%s)", name, exc)
            continue
        if row.get("run_fingerprint") != fingerprint:
            log.debug("ignoring checkpoint %s from a different run", name)
            continue
        out[(row["arm"], int(row["fold"]))] = row
    return out


def save_checkpoint(fold_dir: str, row: dict) -> None:
    """Write one (arm, fold) result as soon as it exists.

    A twelve-hour run that dies on the last fold should not lose the first
    eleven. Written to a temporary name and renamed, so an interruption cannot
    leave a half-written file that the next run would read as complete.
    """
    os.makedirs(fold_dir, exist_ok=True)
    path = checkpoint_path(fold_dir, row["arm"], int(row["fold"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(B._clean(row), f, indent=1, cls=B._NumpyEncoder)
    os.replace(tmp, path)


def personalization_report(summary: list[dict]) -> str:
    """The paired before/after block, reported the way the Ohio table reports it.

    A mean delta alone hides the shape of the result: `subjects improved` is
    what separates "adaptation helps a little on average" from "adaptation helps
    two subjects and hurts ten", and those are different findings.
    """
    lines = []
    for row in summary:
        p = row.get("personalization")
        if not p:
            continue
        lines.append(f"personalisation -- arm '{row['arm']}' "
                     f"(paired against '{row.get('paired_against')}', scored on "
                     f"{row.get('scored_on')})")
        if not p.get("n_subjects"):
            for note in p.get("notes", []) or ["no subject was adapted"]:
                lines.append(f"  {note}")
            continue
        lines.append(
            f"  {p['n_subjects']} subjects | {p['method']} r={p['lora_rank']} | "
            f"{p['n_trainable']} trainable params | prefix {p['prefix_hours']} h")
        lines.append(
            f"  mean dRMSE {p['mean_delta_rmse']:+.3f} mg/dL | "
            f"mean dPR-AUC {p['mean_delta_pr_auc']:+.4f} | "
            f"improved {p['n_improved_rmse']}/{p['n_subjects']} subjects "
            f"(positive dRMSE is WORSE)")
    return "\n".join(lines) if lines else "personalisation: no arm requested"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run CausalCGM-MT arms on MetaboNet, isolated from the "
                    "paper's OhioT1DM artefacts.")
    ap.add_argument("--parquet-path", default=DEFAULT_PARQUET,
                    help=f"MetaboNet parquet (default: {DEFAULT_PARQUET})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output root (default: {DEFAULT_OUT})")
    ap.add_argument("--horizon", type=int, default=30,
                    help="prediction horizon in minutes")
    ap.add_argument("--history", type=int, default=120,
                    help="history length in minutes")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--arms", nargs="+", default=["cgm_only", "fusion", "lora"],
                    help="subset of: cgm_only fusion lora")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="subsample subjects, stratified by source (0 = all)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="restrict to these source_file values")
    ap.add_argument("--group-by-source", action="store_true",
                    help="keep every subject of a source in the same fold -- "
                         "the stricter protocol, since subjects within a source "
                         "share a device and a site")
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--prefix-hours", type=float, default=24.0)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--allow-benchmark-source", nargs="*", default=[],
                    metavar="SOURCE",
                    help="waive the contamination guard for these sources "
                         "(OhioT1DM, HUPA-UCM). Recorded in the results JSON.")
    ap.add_argument("--no-predictions", action="store_true",
                    help="skip the per-window .npz dumps")

    # ---- scale, memory and resumability ------------------------------------
    ap.add_argument("--max-windows-per-subject", type=int, default=400,
                    metavar="N",  # >= a 24 h prefix + embargo + 50 eval windows
                    help="thin each subject's windows to at most N, kept as "
                         "contiguous blocks spread across the record -- never "
                         "every k-th, which would leave each window its own "
                         "recording segment and empty the alarm metrics "
                         "(0 = no cap). The corpus yields ~134M windows at "
                         "stride 1, which is ~356 GB of tensors; a cap is part "
                         "of the protocol, not a convenience.")
    ap.add_argument("--memory-budget-gb", type=float, default=12.0,
                    help="refuse a run whose window store or per-fold fit rows "
                         "are projected past this (0 = no check)")
    ap.add_argument("--window-cache", default=None, metavar="DIR",
                    help="directory for the window store. Reused across runs "
                         "and resumable; without it a temporary one is built "
                         "and the work is not reusable.")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild the window cache and rerun completed folds "
                         "instead of resuming them")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None,
                    help="also write the log here (default: <out>/logs/run.log)")
    ap.add_argument("--smoke", action="store_true",
                    help="two epochs, two folds, few subjects -- checks the "
                         "whole path in minutes. Marked in the JSON.")
    ap.add_argument("--allow-smoke-table", action="store_true",
                    help="let a --smoke run overwrite the comparison table the "
                         "manuscript \\inputs. Off by default: a smoke table "
                         "under publication macros is audit finding C1.")
    args = ap.parse_args(argv)

    if args.smoke:
        # The window cap is NOT reduced here. Smoke mode bounds the run by
        # subjects and epochs; cutting the cap as well would drop it below what
        # the lora arm's prefix needs, and the documented smoke command would
        # fail its own guard.
        # Same shape as the paper drivers' smoke mode: every stage runs, at a
        # budget that cannot be mistaken for a result.
        args.epochs = min(args.epochs, 2)
        args.folds = min(args.folds, 2)
        args.max_subjects = args.max_subjects or 8
        args.adapt_steps = min(args.adapt_steps, 20)
        # NOT the stride. Widening it starves the LoRA prefix (adapt_patient
        # needs AdaptConfig.batch_size windows) and turns the lead-time and lag
        # metrics into nan, because both assume anchors at the native cadence.
        # A smoke run that silently drops half the metric block is not checking
        # the path it claims to check.
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # Before ANY directory is created. This check used to sit after
    # `configure_logging` and `makedirs(bench_dir)`, so a refused run still left
    # `<paper tree>/<out>/logs/` and `.../benchmarks/` behind -- empty, but
    # inside the one tree this driver is not allowed to touch, and the test that
    # exercises the refusal was quietly seeding them on every run.
    if is_inside_paper_tree(args.out):
        print(f"refusing to write into {PAPER_TREE}: this driver's results "
              f"are not paper artefacts.", file=sys.stderr)
        return 2

    log_file = args.log_file or os.path.join(args.out, "logs", "run.log")
    configure_logging(args.log_level, log_file, force=True)
    log.info("run start: %s", " ".join(argv if argv is not None else sys.argv[1:]))
    log.info("host memory: %s", memory_note())

    if not os.path.exists(args.parquet_path):
        # Same exit code and the same "here is what needs no data" pointer the
        # other drivers give through `scripts/_cli.missing_input`; this one
        # checks the path itself because it must refuse BEFORE `--out` is
        # validated against the paper tree.
        print(f"MetaboNet parquet not found: {args.parquet_path}\n"
              f"Pass --parquet-path, or place the file at {DEFAULT_PARQUET}.\n\n"
              f"{NO_DATA_HINT}",
              file=sys.stderr)
        return 2

    bench_dir = os.path.join(args.out, "benchmarks")
    pred_dir = None if args.no_predictions else os.path.join(bench_dir, "predictions")
    os.makedirs(bench_dir, exist_ok=True)

    ws, data_cfg, provenance = load_cohort(args)

    all_specs = {s.name: s for s in build_arms(
        args.horizon, args.epochs, args.seed, args.lora_rank,
        args.prefix_hours, args.adapt_steps)}
    unknown = [a for a in args.arms if a not in all_specs]
    if unknown:
        print(f"unknown arm(s) {unknown}; known: {sorted(all_specs)}",
              file=sys.stderr)
        return 2
    specs = [all_specs[a] for a in args.arms]
    check_single_horizon(specs)

    if any(s.family == "lora" for s in specs) and "cgm_only" not in args.arms:
        print("arm 'lora' adapts the 'cgm_only' fold model; add it to --arms.",
              file=sys.stderr)
        return 2

    # The window cap and the personalisation protocol interact, and the failure
    # is silent: `adapt_patient` needs a wall-clock prefix of at least
    # AdaptConfig.batch_size windows and 50 evaluation windows after the
    # embargo, all drawn from ONE subject's kept windows. Under a cap those are
    # a handful of contiguous blocks, and if the cap is smaller than the
    # protocol needs, every subject is skipped and the arm reports nothing
    # rather than reporting a null result. Checked here, in the units the
    # operator set.
    if any(s.family == "lora" for s in specs) and args.max_windows_per_subject:
        period = max(args.stride, 1) * DataConfig().sample_period_min
        prefix_w = int(args.prefix_hours * 60 / period)
        embargo_w = (args.history + args.horizon) // period
        need = prefix_w + embargo_w + 50
        if args.max_windows_per_subject < need:
            print(
                f"--max-windows-per-subject {args.max_windows_per_subject} is "
                f"too small for the lora arm: a {args.prefix_hours:g} h prefix "
                f"is {prefix_w} windows, the embargo is {embargo_w}, and "
                f"adapt_patient needs 50 evaluation windows after it -- "
                f"{need} in total, from one subject.\n"
                f"  Raise the cap to at least {need}, shorten --prefix-hours, "
                f"or drop 'lora' from --arms. Left as is, every subject would "
                f"be skipped and the arm would report nothing.",
                file=sys.stderr)
            return 2

    folds = metabonet_folds(ws, n_folds=args.folds, seed=args.seed,
                            group_by_source=args.group_by_source)

    # `run_fold` carves an inner validation split out of the TRAINING subjects
    # (`inner_val_frac`, at least one subject), so a fold with a single training
    # subject leaves zero to fit on and dies inside `CausalScaler.fit` with
    # "No training windows to fit the scaler on" -- an error that says nothing
    # about the fold arithmetic that caused it. Checked here, with the fix.
    thin = [(k, int(tr.size)) for k, (tr, _te) in enumerate(folds) if tr.size < 2]
    if thin:
        n_sub = len(ws.subjects())
        print(
            f"folds {[k for k, _ in thin]} have fewer than 2 training subjects "
            f"({[n for _, n in thin]}), which leaves none to fit on once "
            f"run_fold holds out its inner validation subject.\n"
            f"  {n_sub} subjects over {args.folds} folds gives "
            f"~{n_sub // max(args.folds, 1)} test subjects per fold; use fewer "
            f"folds (--folds {max(2, min(args.folds, n_sub // 2))}) or more "
            f"subjects (--max-subjects).", file=sys.stderr)
        return 2

    fold_dir = os.path.join(bench_dir, "_folds")
    fingerprint = run_fingerprint(args, ws.subjects())
    done = {} if args.no_resume else load_checkpoints(fold_dir, fingerprint)
    if done:
        log.info("resuming: %d (arm, fold) results already complete: %s",
                 len(done), sorted(f"{a}/f{k}" for a, k in done))

    rows: list[dict] = []
    base_cache: dict = {}
    for fold, (train_pat, test_pat) in enumerate(folds):
        if test_pat.size == 0 or train_pat.size == 0:
            log.warning("[fold %d] empty side, skipping", fold)
            continue
        # GUARD 3: the claim a reader of the table relies on.
        assert_no_benchmark_contamination(
            [str(p) for p in test_pat], allow=args.allow_benchmark_source)

        i_train = split_by_patient(ws, train_pat)
        i_test = split_by_patient(ws, test_pat)
        log.info("=" * 72)
        log.info("[fold %d] %d train / %d test subjects | %s train / %s test "
                 "windows | test hypo %.2f%% | %s", fold, train_pat.size,
                 test_pat.size, f"{i_train.size:,}", f"{i_test.size:,}",
                 100 * float(ws.y_cls_hard[i_test].mean()), memory_note())
        log.info("  run_fold will hold out %.0f%% of the training SUBJECTS as "
                 "the inner validation fold (patient-wise, never row-wise)",
                 100 * 0.2)

        for spec in specs:
            key = (spec.name, fold)
            if key in done:
                log.info("[fold %d] %s: reusing checkpoint (%s windows scored)",
                         fold, spec.name, f"{done[key].get('n_test', 0):,}")
                rows.append(done[key])
                continue
            log.info("--- %s (fold %d) --- %s", spec.name, fold, memory_note())
            row = run_arm_fold(spec, ws, data_cfg, train_pat, test_pat,
                               fold, pred_dir, base_cache)
            row["run_fingerprint"] = fingerprint
            save_checkpoint(fold_dir, row)
            rows.append(row)
            log.info("[fold %d] %s done in %.1f min (%s)", fold, spec.name,
                     row["seconds"] / 60, memory_note())

    if not rows:
        print("no arm produced a result", file=sys.stderr)
        return 1

    summary = aggregate(rows, args.horizon)
    sources = provenance["sources"]
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "smoke": bool(args.smoke),
        "protocol": "patient_wise_cv",
        "protocol_note": (
            "Whole subjects are held out. MetaboNet's shipped is_test column is "
            "a within-subject temporal split and answers a different question; "
            "it is not used by this driver."),
        "horizon_min": args.horizon,
        "n_folds": len(folds),
        "group_by_source": bool(args.group_by_source),
        "seed": args.seed,
        "epochs": args.epochs,
        "run_fingerprint": fingerprint,
        "resumable_checkpoints": os.path.relpath(fold_dir, args.out),
        "provenance": provenance,
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "config": dataclasses.asdict(data_cfg),
        "arms": [s.as_dict() for s in specs],
        "per_fold": rows,
        "summary": summary,
    }
    B.write_json(os.path.join(bench_dir, "metabonet_summary.json"), payload)

    # Only arms scored on the fold's test rows go in the comparison table. The
    # LoRA arm is scored on per-subject post-embargo regions, so a row of
    # dashes next to `cgm_only` would invite a comparison the numbers do not
    # support; its paired result is printed and serialised separately.
    comparable = [r for r in summary if r["per_horizon"]]
    table = B.text_table(comparable, METABONET_COLUMNS, args.horizon)
    print(f"\n{table}\n")
    print(personalization_report(summary))
    table_path = comparison_table_path(bench_dir, args.smoke,
                                       args.allow_smoke_table)
    published = os.path.join(bench_dir, MANUSCRIPT_TABLE_NAME)
    if table_path != published:
        log.warning(
            "smoke run: writing the comparison table to %s and LEAVING %s "
            "alone. The manuscript \\inputs the latter, and a smoke table "
            "under publication-budget macros is how the appendix ends up "
            "describing two different experiments. Pass --allow-smoke-table "
            "to overwrite it anyway.",
            os.path.basename(table_path), os.path.basename(published))
    # Stamped with this run's fingerprint, so `rebuild_paper_tables` can check
    # the table against the `\NumMetabo*` macros it renders beside it instead
    # of trusting that they came from the same experiment.
    B.write_latex(
        table_path,
        B.latex_table(comparable, METABONET_COLUMNS, args.horizon,
                      caption(args.horizon, provenance["n_subjects"],
                              len(folds), sources, args.smoke),
                      label="tab:metabonet_comparison"),
        provenance={"fingerprint": fingerprint, "smoke": bool(args.smoke),
                    "generated": payload["generated"]})
    with open(os.path.join(bench_dir, "table_metabonet_comparison.txt"), "w") as f:
        f.write(table + "\n\n" + personalization_report(summary) + "\n")

    mins = (time.time() - t0) / 60
    print(f"wrote {bench_dir}/  ({mins:.1f} min)")
    if args.smoke:
        print('\nSMOKE RUN: results are marked "smoke": true and are not '
              'results. Re-run without --smoke for real numbers.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
