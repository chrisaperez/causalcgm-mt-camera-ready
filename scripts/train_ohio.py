"""Train and evaluate on OhioT1DM under the official split.

    python scripts/train_ohio.py --data runs/ohio/ohio_windows.pt --epochs 50
    python scripts/train_ohio.py --root OhioT1DM --horizon 60 --epochs 50
    python scripts/train_ohio.py --data runs/ohio/ohio_windows.pt --unimodal
    python scripts/train_ohio.py --data runs/ohio/ohio_windows.pt \
        --horizons 30 60 --hypo-asymmetry 3.0     # joint multi-horizon trunk
    python scripts/train_ohio.py --data runs/ohio/ohio_windows.pt \
        --arm cgm_hr --cohort 2018                # modality ablation arm

Reports both Clarke and (when `methcomp` is installed) Parkes error grids
alongside RMSE/PR-AUC, and the hypoglycemia direction bias -- see
`causalcgm_mt/clinical.py`. `--hypo-asymmetry` (kappa) controls the asymmetric
regression cost that penalises predicting ABOVE the reference when the
reference is hypoglycemic; kappa=1.0 is an exact no-op.

WHY THIS IS NOT `run_experiment.py`
-----------------------------------
`run_experiment.py` drives `cross_validate`, which splits patients into folds:
train on some people, test on others. OhioT1DM ships a fixed split that is
temporal and WITHIN patient -- every subject appears in both `train/` and
`test/`. Feeding Ohio windows to `cross_validate` would silently replace the
published protocol with a different (and much harder) one, and the resulting
numbers would not be comparable to any Ohio result in the literature.

So this driver keeps the split it was given and trains once. The training loop
mirrors `train.run_fold` step for step -- same order of operations, same
early-stopping monitor, same evaluation call -- but takes row indices from
`OhioSplit` instead of drawing patient folds. The pieces that are not
split-dependent (`_forward`, `predict`, `evaluate`, the models) are imported,
not reimplemented.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.clinical import clinical_report
from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.evaluate import evaluate
from causalcgm_mt.hupa_loader import HupaConfig
from causalcgm_mt.models import build_model, build_multimodal_model
from causalcgm_mt.ohio_loader import (
    DEFAULT_ABLATION,
    OhioConfig,
    aux_validity_indices,
    apply_modality_mask,
    resolve_arm,
    build_ohio_windows,
    collect_reference_pairs,
    describe,
    discover_records,
    load_ohio_tensors,
    make_dataset,
    official_temporal_split,
)
from causalcgm_mt.train import (_forward, _monitor_value,
                                _require_finite_monitor, pick_device, predict,
                                set_seed)
from scripts._cli import ARTIFACT_HINT, OHIO_HINT, missing_input


# --------------------------------------------------------------------------- #
def loaders_from_artifact(path: str, batch_size: int, horizons: list, unimodal: bool,
                          num_workers: int, seed: int, groups: list | None = None,
                          cohorts: list | None = None):
    """Rebuild loaders from a `save_ohio_tensors` artefact.

    The tensors were already scaled by a preprocessor fitted on the fit split
    only, so nothing is fitted here -- reloading cannot re-derive a statistic
    from val or test.
    """
    from torch.utils.data import DataLoader

    art = load_ohio_tensors(path)
    available = list(art["meta"]["horizons_min"])
    missing = [h for h in horizons if h not in available]
    if missing:
        raise SystemExit(f"Artefact has horizons {available} min; {missing} not among "
                         f"them. Re-run prepare_ohio.py with "
                         f"--horizons {' '.join(str(h) for h in horizons)}.")
    # Column indices into the artefact's multi-horizon tensors, in the order the
    # caller asked for. The head order follows this list, so it is recorded in
    # the checkpoint -- a silent reordering between train and serve would map
    # the 60-minute head onto the 30-minute target with no error anywhere.
    ks = [available.index(h) for h in horizons]

    g = torch.Generator().manual_seed(seed)
    out, meta = {}, {}
    for name in ("fit", "val", "test"):
        s = art["splits"][name]
        arrays = {
            "x": s["x"].numpy(),
            "x_aux": s["x_aux"].numpy(),
            # Select the requested horizon out of the multi-horizon tensors, so
            # --horizon picks a target rather than requiring a re-preprocess.
            "y_reg": s["y_reg_multi"][:, ks].numpy().squeeze(-1) if len(ks) == 1
                     else s["y_reg_multi"][:, ks].numpy(),
            "y_soft": s["y_soft_multi"][:, ks].numpy().squeeze(-1) if len(ks) == 1
                      else s["y_soft_multi"][:, ks].numpy(),
            "last_bg": s["last_bg"].numpy(),
            "aux_present": s["aux_present"].numpy(),
        }
        y_cls = s["y_cls_multi"][:, ks].numpy()   # (N, K), always 2-D
        patient = np.asarray(s["patient"])
        subset = np.asarray(s["subset"])
        # Anchor timestamps travel with the rows so the alarm metrics can break
        # episodes at sensor dropouts instead of assuming contiguity.
        t_time = np.asarray(s["t_time"]).astype("datetime64[ns]")

        # Cohort restriction comes FIRST, so every array below is already
        # confined to the subjects whose device actually carries this arm's
        # sensors. Masking channels without restricting cohorts would train on
        # windows where the arm's entire signal is zero.
        if cohorts is not None:
            keep = np.flatnonzero(np.isin(subset, np.asarray(cohorts)))
            arrays = {kk: vv[keep] for kk, vv in arrays.items()}
            y_cls, patient, t_time = y_cls[keep], patient[keep], t_time[keep]

        if unimodal:
            arrays["x_aux"] = arrays["x_aux"][:, :, :0]
        elif groups is not None:
            arrays["x_aux"] = apply_modality_mask(arrays["x_aux"], groups)

        # "compat" emits the 6-tuple `train._forward` unpacks; the multi-horizon
        # path needs the y_reg/y_soft tensors to carry K columns, which the same
        # dataset does correctly because it stores whatever shape it is given.
        ds = make_dataset(arrays, horizons, style="compat")
        shuffle = name == "fit"
        out[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers,
                               drop_last=shuffle and len(ds) > batch_size,
                               generator=g if shuffle else None)
        meta[name] = {
            "y_reg": arrays["y_reg"],
            "y_cls": y_cls,
            "last_bg": arrays["last_bg"],
            "patient": patient,
            "t_time": t_time,
        }

    prev = float(meta["fit"]["y_cls"].mean())
    # The fitted preprocessor travels back to the caller so the checkpoint can
    # carry its scalers. Without them the checkpoint has no deployable form:
    # `export.load_and_export` folds the scaler into the graph as a constant,
    # so a weights-only checkpoint cannot be exported at all.
    return (out, meta, float((1 - prev) / max(prev, 1e-6)), art["meta"],
            art["preprocessor"])


def loaders_from_xml(root: str, cfg: DataConfig, batch_size: int, horizon: int,
                     unimodal: bool, num_workers: int, seed: int, val_frac: float):
    """Build straight from the XML, for when no artefact has been prepared."""
    from causalcgm_mt.ohio_loader import OhioPreprocessor
    from torch.utils.data import DataLoader

    records = discover_records(root)
    ws = build_ohio_windows(records, cfg, HupaConfig(), OhioConfig(),
                            horizons_min=(horizon,))
    print(describe(ws))
    split = official_temporal_split(ws, cfg, val_frac=val_frac)
    print(split.summary(ws))

    pre = OhioPreprocessor.fit(
        ws, split, cfg, reference_pairs=collect_reference_pairs(records, cfg))
    print(f"soft-label band: {pre.labeler.band:.2f} mg/dL "
          f"[{pre.labeler.band_source}], MARD {pre.mard_proxy:.4f} "
          f"[{pre.noise.source if pre.noise else 'n/a'}]")
    g = torch.Generator().manual_seed(seed)
    out, meta = {}, {}
    for name, idx in (("fit", split.fit), ("val", split.val), ("test", split.test)):
        arrays = pre.transform(ws, idx)
        if unimodal:
            arrays["x_aux"] = arrays["x_aux"][:, :, :0]
        ds = make_dataset(arrays, [horizon], style="compat")
        shuffle = name == "fit"
        out[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers,
                               drop_last=shuffle and len(ds) > batch_size,
                               generator=g if shuffle else None)
        meta[name] = {"y_reg": ws.y_reg[idx], "y_cls": ws.y_cls_hard[idx][:, None],
                      "last_bg": ws.last_bg[idx],
                      "patient": np.asarray([str(p) for p in ws.patient[idx]]),
                      "t_time": ws.t_time[idx]}
    prev = float(ws.y_cls_hard[split.fit].mean())
    return out, meta, float((1 - prev) / max(prev, 1e-6)), {
        "aux_valid_index": list(aux_validity_indices()),
        "sample_period_min": cfg.sample_period_min,
    }, pre


# --------------------------------------------------------------------------- #
# Keys written to the JSON report. The alarm-fatigue block is included because
# it is the part a clinical reader reads: sensitivity at a fixed FPR without
# the false alarms per day it costs is only half an operating point.
_REPORT_KEYS = (
    "rmse", "mae", "mard", "time_lag_min", "roc_auc", "pr_auc",
    "sens_at_5pct_fpr", "sensitivity_at_5pct_fpr", "sensitivity_at_10pct_fpr",
    "fapd_at_5pct_fpr", "fapd_at_10pct_fpr", "mean_lead_time_min",
    "median_lead_time_min", "lead_time_detection_rate", "monitored_days",
)


def _col(a: np.ndarray, i: int) -> np.ndarray:
    """Column i of an (N, K) array, or the array itself when it is 1-D."""
    return a if a.ndim == 1 else a[:, i]


def _per_horizon_reports(meta_split: dict, bg_hat: np.ndarray, score: np.ndarray,
                         period: int, horizons: list) -> dict:
    """Score each horizon separately.

    A joint model must be reported per horizon, never pooled: pooling averages a
    30-minute RMSE with a 60-minute one, which are not the same quantity, and
    hides the case where the shared trunk improved one horizon by degrading the
    other.
    """
    return {
        h: evaluate(_col(meta_split["y_reg"], i), _col(bg_hat, i),
                    _col(meta_split["y_cls"], i), _col(score, i),
                    meta_split["last_bg"], meta_split["patient"], period,
                    t_time=meta_split.get("t_time"), horizon_min=h)
        for i, h in enumerate(horizons)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="artefact from scripts/prepare_ohio.py")
    src.add_argument("--root", help="OhioT1DM directory, to build windows now")
    ap.add_argument("--horizon", type=int, default=30,
                    help="PH in minutes (single-horizon model)")
    ap.add_argument("--horizons", nargs="+", type=int, default=None,
                    help="train ONE model jointly over these horizons, e.g. "
                         "--horizons 30 60. Shares the trunk and the fusion "
                         "policy; only the readout heads are horizon-specific.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--history", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    # `fixed` is the default because it is what every published table in this
    # repository was produced under, not because `mard` is unusable -- it stopped
    # being unusable when the estimator gained a paired-reference tier and a
    # device-specification floor (see README, "Closing the soft-label band").
    # Ablation E is the study that sweeps this.
    ap.add_argument("--band-mode", choices=["fixed", "mard"], default="fixed",
                    help="soft-label band: 'fixed' uses --soft-band verbatim; "
                         "'mard' derives it from the fold's estimated CGM error "
                         "and records the provenance in the run config")
    ap.add_argument("--soft-band", type=float, default=10.0,
                    help="band half-width in mg/dL when --band-mode=fixed")
    ap.add_argument("--unimodal", action="store_true",
                    help="CGM-only control arm; identical windows and split")
    ap.add_argument("--arm", default=None,
                    help=f"named ablation arm: {', '.join(a for a, _ in DEFAULT_ABLATION)}")
    ap.add_argument("--modalities", nargs="*", default=None,
                    help="explicit modality groups, e.g. --modalities heart_rate gsr")
    ap.add_argument("--cohort", nargs="*", default=None, choices=["2018", "2020"],
                    help="restrict to these release years. Required to give a "
                         "single-cohort arm a control drawn from the SAME subjects.")
    ap.add_argument("--hypo-asymmetry", type=float, default=3.0,
                    help="kappa: multiplier on the regression loss when the "
                         "reference is hypoglycemic and the prediction sits "
                         "ABOVE it. 1.0 disables the mechanism (plain Huber).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="runs/ohio")
    args = ap.parse_args()

    horizons = args.horizons or [args.horizon]
    if args.horizons and args.root:
        ap.error("--horizons (joint training) needs --data; build the artefact "
                 "with prepare_ohio.py --horizons ... first.")
    K = len(horizons)

    cfg = Config(out_dir=args.out)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, horizon_min=args.horizon,
                                 history_min=args.history, stride_steps=args.stride,
                                 band_mode=args.band_mode,
                                 soft_band_mgdl=args.soft_band),
        model=dataclasses.replace(cfg.model, n_horizons=K,
                                  hypo_asymmetry=args.hypo_asymmetry),
        train=dataclasses.replace(cfg.train, max_epochs=args.epochs, lr=args.lr,
                                  batch_size=args.batch_size, seed=args.seed,
                                  early_stop_patience=args.patience,
                                  num_workers=args.num_workers),
    )

    groups, cohorts = None, None
    if args.cohort:
        cohorts = sorted(args.cohort)
    if args.arm or args.modalities is not None:
        if args.arm:
            named = dict(DEFAULT_ABLATION)
            if args.arm not in named:
                ap.error(f"unknown arm {args.arm!r}; choose from {sorted(named)}")
            groups = list(named[args.arm])
        else:
            groups = list(args.modalities)
        _, arm_cohorts = resolve_arm(groups)
        # An explicit --cohort narrows further; it must never WIDEN past what
        # the arm's sensors support, or the arm trains on windows that are
        # entirely mask.
        cohorts = sorted(set(arm_cohorts) & set(cohorts)) if cohorts else arm_cohorts
        if not cohorts:
            ap.error(f"arm {groups} mixes sensors that no single cohort carries "
                     f"(2018=Basis Peak, 2020=Empatica). No subject has all of them.")
        print(f"ARM {args.arm or groups}: groups={groups or ['(none)']} "
              f"restricted to cohort(s) {cohorts}")
        if cohorts != ["2018", "2020"]:
            print("  NOTE: this arm is scored on ONE cohort's 6 subjects. It is "
                  "comparable only to other arms restricted to the same cohort.")

    # Both branches read a path this repository cannot ship: `--data` an
    # artefact from prepare_ohio.py, `--root` the licensed XML release itself.
    with missing_input(ARTIFACT_HINT if args.data else OHIO_HINT):
        if args.data:
            dl, meta, pos_weight, art_meta, pre = loaders_from_artifact(
                args.data, args.batch_size, horizons, args.unimodal,
                args.num_workers, args.seed, groups=groups, cohorts=cohorts)
        else:
            dl, meta, pos_weight, art_meta, pre = loaders_from_xml(
                args.root, cfg.data, args.batch_size, args.horizon, args.unimodal,
                args.num_workers, args.seed, args.val_frac)

    period = int(art_meta.get("sample_period_min", 5))
    set_seed(cfg.train.seed)
    device = pick_device()
    x0, *_ , xa0, _ = next(iter(dl["fit"]))
    multimodal = not args.unimodal and xa0.shape[-1] > 0
    print(f"device={device} | multimodal={multimodal} | "
          f"PH={horizons} min ({'joint trunk' if K > 1 else 'single'}) | "
          f"pos_weight={pos_weight:.2f} | hypo kappa={args.hypo_asymmetry}")

    if multimodal:
        model, criterion = build_multimodal_model(
            x0.shape[-1], xa0.shape[-1], cfg.model, pos_weight,
            aux_valid_index=art_meta.get("aux_valid_index", aux_validity_indices()),
        )
    else:
        model, criterion = build_model(x0.shape[-1], cfg.model, pos_weight)
    model, criterion = model.to(device), criterion.to(device)

    # Model AND criterion, hoisted so the optimiser and the clip cannot drift
    # apart. They did: the optimiser took both, the clip took only the model,
    # so the Kendall log-variances stepped unclipped. Their gradient is
    # `0.5 - exp(-s) * L`, largest early in training when `L` is large --
    # exactly when `grad_clip` is protecting everything else -- and one
    # oversized step there rescales the whole multi-task balance, which is the
    # collapsing task weight `MultiTaskLoss` warns about.
    trainable = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(trainable,
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=max(1, cfg.train.max_epochs * len(dl["fit"])))
    use_amp = cfg.train.amp and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, patience = -np.inf, None, 0
    sign = 1.0 if cfg.train.monitor_mode == "max" else -1.0

    for epoch in range(cfg.train.max_epochs):
        model.train()
        for batch in dl["fit"]:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                bg_hat, logit, y_reg, y_soft = _forward(model, batch, device, multimodal)
                loss, _ = criterion(bg_hat, logit, y_reg, y_soft)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            amp_scaler.step(opt)
            amp_scaler.update()
            sched.step()

        bg_hat, score = predict(model, dl["val"], device, multimodal)
        m = meta["val"]
        # Model selection on the MEAN of the monitor across horizons. Selecting
        # on one horizon would let the joint model quietly sacrifice the other,
        # which is precisely the failure mode a shared trunk invites.
        reps = _per_horizon_reports(m, bg_hat, score, period, horizons)
        # `nanmean`, matching `benchmark.train_neural_arm`: a plain mean lets a
        # single nan horizon poison a monitor the other horizon defines
        # perfectly well. All-nan still yields nan, which is checked below.
        with np.errstate(invalid="ignore"):
            per_h = [_monitor_value(r, cfg.train.monitor) for r in reps.values()]
            val = sign * float(np.nan if all(not np.isfinite(v) for v in per_h)
                               else np.nanmean(per_h))
        _require_finite_monitor(val, cfg.train.monitor, epoch,
                                {f"h{h}": r for h, r in reps.items()})
        detail = " ".join(f"h{h}={_monitor_value(r, cfg.train.monitor):.4f}"
                          for h, r in reps.items())
        print(f"epoch {epoch:03d} {cfg.train.monitor}(mean)={val * sign:.4f} | {detail}")

        # Model selection on the forewarning metric, since that is the
        # clinically load-bearing task -- same choice as `train.run_fold`.
        if val > best:
            best, patience = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(
            f"no epoch was ever selected (max_epochs={cfg.train.max_epochs}); "
            f"there is no trained model to score.")
    model.load_state_dict(best_state)

    bg_hat, score = predict(model, dl["test"], device, multimodal)
    m = meta["test"]
    reports = _per_horizon_reports(m, bg_hat, score, period, horizons)
    clinicals = {h: clinical_report(_col(m["y_reg"], i), _col(bg_hat, i))
                 for i, h in enumerate(horizons)}
    print(f"\n=== OFFICIAL OhioT1DM TEST "
          f"({'multimodal' if multimodal else 'CGM-only'}, "
          f"{'joint' if K > 1 else 'single'} trunk, kappa={args.hypo_asymmetry}) ===")
    for h in horizons:
        print(f"-- PH = {h} min --\n{reports[h]}\n{clinicals[h]}")
    report = reports[horizons[0]]
    clin = clinicals[horizons[0]]

    os.makedirs(args.out, exist_ok=True)
    tag = (f"ph{'-'.join(str(h) for h in horizons)}_"
           f"{args.arm or ('uni' if not multimodal else 'mm')}")
    if args.cohort:
        tag += "_" + "".join(sorted(args.cohort))
    # `export.REQUIRED_CHECKPOINT_KEYS`, in full. This driver used to write
    # weights + config and no scaler, so `load_and_export` raised a bare
    # `KeyError: 'scaler'` and the ONLY exportable checkpoint in the repository
    # was the synthetic demo's -- which is how the shipped ONNX graph came to be
    # a synthetic, unimodal, single-horizon model while the paper quoted its
    # latency as the fusion model's (audit finding F6).
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "criterion_state_dict": {k: v.detach().cpu() for k, v
                                         in criterion.state_dict().items()},
                "config": dataclasses.asdict(cfg),
                "horizons_min": horizons, "multimodal": multimodal,
                "in_channels": int(x0.shape[-1]),
                "aux_channels": int(xa0.shape[-1]),
                # The fitted preprocessing. `scaler` is folded into the exported
                # graph as a constant affine op; `aux_scaler` likewise for the
                # fusion branch. `per_patient` is NOT foldable (its statistics
                # are per subject) but ships so a deployment can apply the same
                # transform the model was trained under -- see
                # `DeployableMultiModalCausalCGMMT`.
                "scaler": pre.cgm_scaler,
                "aux_scaler": pre.aux_scaler if multimodal else None,
                "per_patient": pre.per_patient,
                "labeler": pre.labeler,
                "aux_valid_index": list(art_meta.get(
                    "aux_valid_index", aux_validity_indices())),
                "pos_weight": pos_weight},
               os.path.join(args.out, f"model_{tag}.pt"))
    with open(os.path.join(args.out, f"report_{tag}.json"), "w") as f:
        json.dump({
            "arm": args.arm or (groups if groups is not None else "all"),
            "cohorts": cohorts or ["2018", "2020"],
            "horizons_min": horizons,
            "joint_trunk": K > 1,
            "hypo_asymmetry": args.hypo_asymmetry,
            # Flat keys mirror the PRIMARY (first) horizon so existing readers,
            # including run_ablation.py's summary table, keep working.
            "statistical": {k: float(getattr(report, k)) for k in _REPORT_KEYS},
            "clinical": clin.as_dict(),
            "per_horizon": {
                str(h): {
                    "statistical": {k: float(getattr(reports[h], k)) for k in _REPORT_KEYS},
                    "clinical": clinicals[h].as_dict(),
                } for h in horizons
            },
        }, f, indent=2)
    print(f"Saved {os.path.join(args.out, f'model_{tag}.pt')}")


if __name__ == "__main__":
    main()
