"""Training loop: patient-wise cross validation, mixed precision, early stopping.

Clean PyTorch rather than Lightning, deliberately: the fold/scaler/soft-label
ordering is the part of this project most likely to be reviewed line by line for
leakage, and a framework that hides the loop makes that review harder.

The ORDER of operations inside `run_fold` is the load-bearing part:

    1. split patients               (no window data touched)
    2. fit scaler on TRAIN only
    3. estimate MARD on TRAIN only  -> soft-label band
    4. compute pos_weight on TRAIN only
    5. transform train / val / test with the frozen train-fitted objects

Any reordering that moves a `fit` after a merge of train and eval data is a
leak. The unit tests in tests/ assert this ordering holds.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .data import (
    SCALED_FEATURE_CLIP,
    CausalScaler,
    anchor_series_by_patient,
    SoftLabeler,
    WindowSet,
    estimate_mard_proxy,  # noqa: F401  (re-exported for callers of this module)
    estimate_signal_noise,
    patient_folds,
    split_by_patient,
)
from .data import aux_presence, uses_unmasked_channels
from .evaluate import EvalReport, evaluate
from .models import build_model

_log = logging.getLogger("causalcgm_mt.train")


# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Thin wrapper. Scaling has already happened -- the dataset performs no
    fitting of any kind, which keeps the leak surface confined to `run_fold`.

    `x_aux` is optional so the same dataset, loop and evaluation path serve both
    the univariate and the multimodal model. Keeping one code path means the
    unimodal-vs-multimodal ablation differs only in the model and the tensors it
    receives -- not in preprocessing, splitting or scoring, which is what makes
    the comparison interpretable.
    """

    def __init__(
        self,
        x: np.ndarray,
        y_reg: np.ndarray,
        y_soft: np.ndarray,
        last_bg: np.ndarray,
        x_aux: np.ndarray | None = None,
        aux_present: np.ndarray | None = None,
    ):
        self.x = torch.from_numpy(x)
        self.y_reg = torch.from_numpy(y_reg)
        self.y_soft = torch.from_numpy(y_soft)
        self.last_bg = torch.from_numpy(last_bg)
        self.multimodal = x_aux is not None
        # Empty placeholders keep the collated batch shape constant, so the
        # training loop does not need two branches.
        self.x_aux = torch.from_numpy(x_aux) if self.multimodal else torch.zeros(len(x), 0, 0)
        self.aux_present = (
            torch.from_numpy(aux_present) if aux_present is not None
            else torch.ones(len(x), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        return (self.x[i], self.y_reg[i], self.y_soft[i], self.last_bg[i],
                self.x_aux[i], self.aux_present[i])


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
def _forward(model, batch, device, multimodal: bool):
    """One forward pass, unimodal or multimodal. Single definition so the two
    paths cannot drift apart."""
    x, y_reg, y_soft, last, x_aux, aux_present = batch
    x, last = x.to(device, non_blocking=True), last.to(device, non_blocking=True)
    if multimodal:
        bg_hat, logit = model(
            x, x_aux.to(device, non_blocking=True), last,
            aux_present.to(device, non_blocking=True),
        )
    else:
        bg_hat, logit = model(x, last)
    return bg_hat, logit, y_reg.to(device), y_soft.to(device)


@torch.no_grad()
def predict(model, loader, device, multimodal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    bg, sc = [], []
    for batch in loader:
        bg_hat, logit, _, _ = _forward(model, batch, device, multimodal)
        bg.append(bg_hat.float().cpu().numpy())
        sc.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(bg), np.concatenate(sc)


def _monitor_value(rep: EvalReport, key: str) -> float:
    return float(getattr(rep, key.split("/")[-1]))


def _require_finite_monitor(value: float, monitor: str, epoch: int,
                            reports: Mapping[object, EvalReport]) -> None:
    """Refuse to continue on a non-finite model-selection monitor.

    `nan > -inf` is False, so a nan monitor never beats the initial `best`: no
    epoch is ever recorded, `best_state` stays None, early stopping fires on
    the patience counter, the final `load_state_dict` is skipped -- and the
    model that gets reported is whatever the LAST epoch happened to produce.
    The results file records `best_epoch: -1` and `best_monitor: nan`, which
    reads as "no improvement" rather than "model selection never ran".

    The usual cause is benign and specific: `monitor` defaults to
    `val/pr_auc`, and `EvalReport.pr_auc` is nan whenever the validation slice
    is single-class -- routine for a small patient-wise fold with no
    hypoglycemic window in it. That is a reason to widen the fold or change the
    monitor, not to silently ship the last epoch, so it stops the run and says
    which slice was degenerate.
    """
    if np.isfinite(value):
        return
    key = monitor.split("/")[-1]
    detail = "; ".join(
        f"{label}: {key}={_monitor_value(r, monitor):.4g} "
        f"(n={r.n}, n_positive={r.n_positive})"
        for label, r in reports.items()) or "no reports"
    raise RuntimeError(
        f"model-selection monitor {monitor!r} is not finite at epoch {epoch}, "
        f"so no epoch can ever be selected and the run would report its LAST "
        f"epoch as if it were its best.\n"
        f"  validation slice -- {detail}\n"
        f"  A single-class validation slice makes pr_auc nan. Widen the "
        f"validation split, or set the monitor to a metric defined here "
        f"(e.g. monitor='val/rmse' with monitor_mode='min')."
    )


@dataclass
class FoldResult:
    fold: int
    report: EvalReport
    state_dict: dict
    scaler: CausalScaler
    labeler: SoftLabeler
    test_patients: list[str]
    # Everything below is needed to REUSE this fold's model outside `run_fold`
    # -- to dump its predictions, or to personalise from it. Without the aux
    # scaler a multimodal fold's model cannot be fed correctly scaled inputs by
    # anyone else, and without the val patients a caller cannot tell which rows
    # were model selection rather than training. Both default so existing
    # constructions and readers are unaffected.
    aux_scaler: CausalScaler | None = None
    val_patients: list[str] = field(default_factory=list)
    # The multi-task loss's learned log-variances are a FITTED population
    # quantity (Kendall weighting), not a hyperparameter. Personalising from
    # this fold with a freshly constructed criterion would adapt under a
    # different task balance than the one the model was trained with, which is
    # invisible in the adapted model's own metrics.
    criterion_state_dict: dict = field(default_factory=dict)
    # Per-epoch training loss and validation scores, for a run that has to be
    # diagnosed after the fact from its log rather than watched.
    history: list = field(default_factory=list)


def run_fold(
    ws: WindowSet,
    cfg: Config,
    train_pat: np.ndarray,
    test_pat: np.ndarray,
    fold: int,
) -> FoldResult:
    set_seed(cfg.train.seed + fold)
    device = pick_device()

    # --- 1. inner train/val split, still strictly patient-wise ---------------
    rng = np.random.default_rng(cfg.train.seed + fold)
    perm = rng.permutation(train_pat)
    n_val = max(1, int(round(len(perm) * cfg.train.inner_val_frac)))
    val_pat, fit_pat = perm[:n_val], perm[n_val:]

    i_fit = split_by_patient(ws, fit_pat)
    i_val = split_by_patient(ws, val_pat)
    i_test = split_by_patient(ws, test_pat)

    # Multimodal is detected from the window set, so a single call site serves
    # both models and the ablation cannot accidentally change anything else.
    multimodal = hasattr(ws, "x_aux")

    # --- 2..4. fit every statistic on the FIT patients only ------------------
    scaler = CausalScaler.fit(ws, fit_pat)

    aux_scaler = None
    aux_mask_index: tuple[int, ...] = ()
    aux_names: tuple[str, ...] = ()
    if multimodal:
        from .hupa_loader import AUX_CHANNELS
        # The channel layout is read off the WINDOW SET when it carries one, and
        # only falls back to HUPA's. A loader that extends the layout
        # (`ohio_loader`, `metabonet_loader`) appends its channels at the end,
        # so indices 0-10 keep their meaning -- but the mask channels it appends
        # are past index 10, and a hardcoded `AUX_SKIP_SCALING == (7, 8, 9)`
        # would let the scaler rescale those extra 0/1 flags by an IQR and stop
        # them meaning "observed", while the model would never learn they gate a
        # modality. Deriving both index sets from the names keeps one rule.
        aux_names = tuple(getattr(ws, "aux_channels", AUX_CHANNELS))
        aux_mask_index = tuple(i for i, c in enumerate(aux_names)
                               if c.endswith("_observed"))
        # Same train-only policy as the CGM scaler; masks excluded.
        # Same policy as `OhioPreprocessor.fit`: an auxiliary tensor mixes
        # device streams spanning decades, so a plain-IQR spread can be fitted
        # to a near-zero bulk and re-expand the tail. The CGM scaler above does
        # not take these knobs -- see `CausalScaler.fit_array`.
        aux_scaler = CausalScaler.fit_array(
            ws.x_aux, ws.patient, fit_pat, aux_mask_index,
            tail_aware=True, clip=SCALED_FEATURE_CLIP,
        )

    # No paired reference BG on this path (the generic/synthetic cohorts carry
    # no finger sticks), so the high-pass tier is all there is. `SoftLabeler`
    # applies the device-specification floor on top and records which tier the
    # band came from -- see `data.NoiseEstimate`.
    noise = estimate_signal_noise(
        anchor_series_by_patient(ws, i_fit), cfg.data.sample_period_min
    )
    labeler = SoftLabeler.fit(ws.y_min[i_fit], cfg.data, noise=noise)

    prev = float(ws.y_cls_hard[i_fit].mean())
    pos_weight = float((1 - prev) / max(prev, 1e-6))

    # --- 5. apply the frozen objects downstream ------------------------------
    def make_loader(idx, shuffle):
        ds = WindowDataset(
            scaler.transform(ws.x[idx]),
            ws.y_reg[idx],
            labeler(ws.y_min[idx]),
            ws.last_bg[idx],
            x_aux=aux_scaler.transform(ws.x_aux[idx]) if multimodal else None,
            # Masks AND mask-free treatment channels, on raw values -- see
            # `data.aux_presence`. `aux_frac > 0` alone shut the fusion branch
            # on 99.6% of MetaboNet windows despite insulin being recorded.
            aux_present=aux_presence(ws.x_aux[idx], aux_names) if multimodal else None,
        )
        return DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.train.num_workers,
            # `and len(ds) > batch_size`: dropping the last partial batch of a
            # split SMALLER than one batch drops EVERY batch. The epoch loop
            # then runs zero optimiser steps without raising, and the fold
            # reports its randomly-initialised model with a plausible RMSE --
            # measured at 14.871 mg/dL against a persistence floor of 14.881,
            # which reads as a weak result rather than an absent one. Every
            # sibling loader factory already carries this guard
            # (`benchmark._loader`, `ohio_loader.make_ohio_loaders`,
            # `scripts/train_ohio.py`); this was the sole omission.
            drop_last=shuffle and len(ds) > cfg.train.batch_size,
        )

    dl_fit = make_loader(i_fit, True)
    dl_val = make_loader(i_val, False)
    dl_test = make_loader(i_test, False)

    if multimodal:
        from .models import build_multimodal_model
        # EVERY observation mask, not just heart rate. A step counts as
        # auxiliary-valid when any wearable stream reported at it. Gating on one
        # channel is how the OhioT1DM fusion arm ended up running CGM-only for
        # the whole 2020 cohort (see `ohio_loader.aux_validity_indices`); the
        # same failure is available here for any subject whose HR stream is
        # absent while their step or sleep stream is not. `aux_mask_index` is
        # derived from this window set's own channel names above.
        model, criterion = build_multimodal_model(
            ws.x.shape[-1], ws.x_aux.shape[-1], cfg.model, pos_weight,
            aux_valid_index=aux_mask_index,
            always_valid=uses_unmasked_channels(aux_names),
        )
    else:
        model, criterion = build_model(ws.x.shape[-1], cfg.model, pos_weight)
    model, criterion = model.to(device), criterion.to(device)

    # Model AND criterion, hoisted so the optimiser and the clip cannot drift
    # apart. They did: the optimiser took both, the clip took only the model,
    # so the Kendall log-variances stepped unclipped. Their gradient is
    # `0.5 - exp(-s) * L`, largest early in training when `L` is large --
    # exactly when `grad_clip` is protecting everything else -- and one
    # oversized step there rescales the whole multi-task balance, which is the
    # collapsing task weight `MultiTaskLoss` warns about.
    trainable = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(
        trainable,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=max(1, cfg.train.max_epochs * len(dl_fit))
    )
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_state, patience = -np.inf, None, 0
    sign = 1.0 if cfg.train.monitor_mode == "max" else -1.0

    history: list[dict] = []
    for epoch in range(cfg.train.max_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t_epoch = time.time()
        for batch in dl_fit:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                bg_hat, logit, y_reg, y_soft = _forward(model, batch, device, multimodal)
                loss, _ = criterion(bg_hat, logit, y_reg, y_soft)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()
            # Detached and accumulated as a float: keeping the tensor would hold
            # the whole graph for the epoch.
            epoch_loss += float(loss.detach())
            n_batches += 1

        if n_batches == 0:
            # Unreachable while `drop_last` is guarded above and the fit split
            # is non-empty (an empty one is refused earlier, by the scaler).
            # Pinned as an error rather than left to be re-derived: this is the
            # exact state the fold used to return a model from, and
            # `epoch_loss / max(n_batches, 1)` turns it into a train_loss of
            # 0.0000 that reads like a converged run.
            raise RuntimeError(
                f"[fold {fold}] the fit loader yielded no batches at epoch "
                f"{epoch} ({i_fit.size} windows, batch_size="
                f"{cfg.train.batch_size}); no optimiser step can run and the "
                f"fold would report an untrained model as its result.")

        train_loss = epoch_loss / max(n_batches, 1)
        bg_hat, score = predict(model, dl_val, device, multimodal)
        rep = evaluate(
            ws.y_reg[i_val], bg_hat, ws.y_cls_hard[i_val], score,
            ws.last_bg[i_val], ws.patient[i_val], cfg.data.sample_period_min,
            t_time=ws.t_time[i_val], horizon_min=cfg.data.horizon_min,
        )
        val = sign * _monitor_value(rep, cfg.train.monitor)
        _require_finite_monitor(val, cfg.train.monitor, epoch,
                                {f"h{cfg.data.horizon_min}": rep})
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_rmse": float(rep.rmse), "val_pr_auc": float(rep.pr_auc),
                        cfg.train.monitor: float(val * sign),
                        "lr": float(sched.get_last_lr()[0]),
                        "seconds": round(time.time() - t_epoch, 1)})
        # The training loss belongs next to the monitor: a monitor that stalls
        # while the loss is still falling is early stopping doing its job, and
        # one that stalls while the loss is flat is a run that never started.
        # Reporting only the monitor makes those two look identical.
        msg = (f"[fold {fold}] epoch {epoch:03d} train_loss={train_loss:.4f} "
               f"val_rmse={rep.rmse:7.3f} val_pr_auc={rep.pr_auc:.4f} "
               f"{cfg.train.monitor}={val * sign:.4f} "
               f"({history[-1]['seconds']:.0f}s)")
        print(msg, flush=True)
        _log.info(msg)

        if val > best_val:
            best_val, patience = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[fold {fold}] early stop at epoch {epoch}", flush=True)
                _log.info("[fold %d] early stop at epoch %d (best %s=%.4f)",
                          fold, epoch, cfg.train.monitor, best_val * sign)
                break

    if best_state is None:
        # Unreachable while at least one epoch runs, because the monitor is
        # checked finite above and any finite value beats -inf. Left as an
        # error rather than a silent skip: scoring an unselected model is what
        # the nan-monitor path used to do.
        raise RuntimeError(
            f"[fold {fold}] no epoch was ever selected "
            f"(max_epochs={cfg.train.max_epochs}); there is no trained model "
            f"to score.")
    model.load_state_dict(best_state)

    bg_hat, score = predict(model, dl_test, device, multimodal)
    report = evaluate(
        ws.y_reg[i_test], bg_hat, ws.y_cls_hard[i_test], score,
        ws.last_bg[i_test], ws.patient[i_test], cfg.data.sample_period_min,
        t_time=ws.t_time[i_test], horizon_min=cfg.data.horizon_min,
    )
    print(f"[fold {fold}] TEST\n{report}")

    return FoldResult(
        fold=fold,
        report=report,
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        scaler=scaler,
        labeler=labeler,
        test_patients=[str(p) for p in test_pat],
        aux_scaler=aux_scaler,
        val_patients=[str(p) for p in val_pat],
        criterion_state_dict={k: v.detach().cpu()
                              for k, v in criterion.state_dict().items()},
        history=history,
    )


def cross_validate(ws: WindowSet, cfg: Config) -> list[FoldResult]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())

    folds = patient_folds(ws.patient, cfg.train.n_folds, cfg.train.seed)
    results = []
    for k, (train_pat, test_pat) in enumerate(folds):
        r = run_fold(ws, cfg, train_pat, test_pat, k)
        multimodal = hasattr(ws, "x_aux")
        # `export.REQUIRED_CHECKPOINT_KEYS`, in full. Two of these were dropped
        # here for no reason: `config`, without which `export.load_and_export`
        # raised a bare `KeyError` and the fold could not be exported at all;
        # and `aux_scaler` / `criterion_state_dict`, which `FoldResult` carries
        # precisely so a multimodal fold can be reloaded and personalised from
        # (see the field comments on `FoldResult`) and which were then discarded
        # at the one point where they would have survived the process.
        torch.save(
            {
                "state_dict": r.state_dict,
                "criterion_state_dict": r.criterion_state_dict,
                "config": json.loads(cfg.to_json()),
                "scaler": r.scaler,
                "aux_scaler": r.aux_scaler,
                "labeler": r.labeler,
                "test_patients": r.test_patients,
                "val_patients": r.val_patients,
                "multimodal": multimodal,
                "in_channels": int(ws.x.shape[-1]),
                "aux_channels": int(ws.x_aux.shape[-1]) if multimodal else 0,
            },
            os.path.join(cfg.out_dir, f"fold{k}.pt"),
        )
        results.append(r)

    summary = {
        m: {
            "mean": float(np.nanmean([getattr(r.report, m) for r in results])),
            "std": float(np.nanstd([getattr(r.report, m) for r in results])),
        }
        for m in ("rmse", "mae", "mard", "time_lag_min", "roc_auc", "pr_auc", "sens_at_5pct_fpr")
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CV summary ===")
    for m, v in summary.items():
        print(f"  {m:>18}: {v['mean']:.4f} +/- {v['std']:.4f}")
    return results
