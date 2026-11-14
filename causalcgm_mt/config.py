"""Typed configuration for the CausalCGM-MT replication.

Everything that could plausibly change between experiments lives here, so that a
run is fully described by a single serialisable object. This matters for
reproducibility, but it also matters for *auditability*: the causality-critical
knobs (window geometry, scaler fitting policy, soft-label band) are all visible
in one place rather than scattered across the pipeline.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Literal, Sequence


# --------------------------------------------------------------------------- #
# Data / windowing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    # Native CGM cadence. Dexcom/Libre-style devices sample every 5 minutes.
    sample_period_min: int = 5

    # History length fed to the model, in minutes. 120 min @ 5 min = 24 steps.
    history_min: int = 120

    # Prediction horizon (PH) in minutes. The regression target is BG(t + PH).
    horizon_min: int = 30

    # Column names in the raw long-format frame.
    patient_col: str = "patient_id"
    time_col: str = "timestamp"
    glucose_col: str = "glucose"
    # UNSUPPORTED on the generic path, and refused rather than ignored.
    #
    # `regularize_series` snapped these onto the grid and `build_windows` then
    # read `glucose_col` and nothing else, so they were computed, binned and
    # discarded: setting the knob cost CPU and changed nothing, while its name
    # reads as "this is how I add insulin to the generic pipeline".
    #
    # Exogenous streams are carried by the MULTIMODAL loaders (`hupa_loader`,
    # `ohio_loader`, `metabonet_loader`), which build a separate `x_aux` tensor
    # with its own channel layout, scaler and availability masks. The field is
    # kept so an existing config still constructs, but a non-empty value is now
    # an error at windowing time.
    exog_cols: Sequence[str] = ()

    # ---- Missing data policy ------------------------------------------------
    # Gaps strictly shorter than this are linearly interpolated *within the
    # history window only* (a backward-looking operation). Longer gaps are left
    # as NaN and the affected window is dropped.
    max_interpolation_gap_min: int = 15

    # A window is discarded if more than this fraction of its history steps were
    # imputed rather than observed.
    max_imputed_frac: float = 0.20

    # Targets are never imputed. A window whose target step(s) are missing is
    # dropped outright -- imputing a label is label leakage by construction.

    # ---- Hypoglycemia definition -------------------------------------------
    hypo_threshold_mgdl: float = 70.0

    # The classification label is positive if BG dips below the threshold
    # *anywhere* in (t, t + PH]. Set to False to use only the endpoint t + PH.
    classify_over_window: bool = True

    # ---- Statistically guided soft labels -----------------------------------
    # See data.SoftLabeler. `band_mode="mard"` derives the boundary band width
    # from the training-fold MARD estimate of CGM measurement error; "fixed"
    # uses `soft_band_mgdl` directly. Both are sweepable arms -- Ablation E in
    # `scripts/run_paper_ablations.py` -- rather than one real setting and one
    # that degenerates.
    soft_labels: bool = True
    band_mode: Literal["mard", "fixed"] = "mard"
    soft_band_mgdl: float = 10.0        # used when band_mode == "fixed"
    band_mard_multiplier: float = 1.0   # band = k * MARD * threshold

    # ---- Robust-estimator policy for band_mode == "mard" --------------------
    # The estimated MARD is floored here before the band formula is applied.
    #
    # WHY A FLOOR IS THE HONEST FALLBACK. The estimate can legitimately come out
    # near zero, and on real data it does: the released OhioT1DM stream has been
    # through the manufacturer's smoothing filter, so its high-frequency content
    # is ~0.7% and a band derived from it is +/-0.5 mg/dL -- half a display
    # step, which makes the "soft" labels hard. That is a fact about what was
    # released, not about the sensor: no CGM is more accurate than its own
    # specification. So a sub-spec estimate falls back to the SPEC and is pushed
    # through the same `band = k * MARD * T` formula, keeping `k` meaningful,
    # and `SoftLabeler.band_source` records that it was floored rather than
    # measured. The default is the Medtronic Guardian-class figure the OhioT1DM
    # cohorts were recorded with; set it to your own device's published MARD.
    band_mard_floor: float = 0.09

    # The window inside which a soft label does something a hard label does not.
    # Below `band_min_mgdl` the logistic ramp saturates within one reportable
    # CGM unit; above `band_max_mgdl` it reaches from the hypo threshold well
    # into euglycemia and starts labelling safe windows as ambiguous. The band
    # is clamped into this window and `SoftLabeler.band_source` records it.
    band_min_mgdl: float = 5.0
    band_max_mgdl: float = 20.0
    # ---- Soft-label clamp ---------------------------------------------------
    # The clamp is ASYMMETRIC, and deliberately so.
    #
    # `binary_cross_entropy_with_logits` is already numerically stable at
    # y in {0, 1} (it uses the log-sum-exp form), so the floor was never needed
    # for the stability its old comment claimed. What it did instead was
    # interact multiplicatively with `pos_weight = (1 - prev)/prev ~ 18 at 5%
    # prevalence: every clear-NEGATIVE window carried a positive-class term of
    # weight 18.2 x 0.02 = 0.364, which in aggregate is 37% of the positive mass
    # contributed by the genuine positives. That shifts the optimal constant
    # predictor from 0.500 to 0.578 -- a systematic upward miscalibration of the
    # risk head, invisible in every rank-based metric the paper reports (ROC-AUC,
    # PR-AUC, sensitivity at fixed FPR, FAPD at an ROC-derived threshold) and
    # very much visible to anything that reads `risk` as a probability.
    #
    # The ceiling has no such interaction: a clear positive contributes
    # (1 - 0.98) to the unweighted negative term, i.e. 0.05 x 0.02 = 0.001 of
    # cohort mass against the negatives' 0.95. It is kept.
    soft_label_floor: float = 0.0
    soft_label_ceiling: float = 0.98

    # ---- Windowing ----------------------------------------------------------
    stride_steps: int = 1
    # Drop windows that straddle a recording break longer than this.
    max_series_gap_min: int = 30

    @property
    def history_steps(self) -> int:
        return self.history_min // self.sample_period_min

    @property
    def horizon_steps(self) -> int:
        return self.horizon_min // self.sample_period_min


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    backbone: Literal["tcn", "dilated_lstm"] = "tcn"
    hidden_channels: int = 64
    n_blocks: int = 4              # dilations 1, 2, 4, 8 -> RF = 2*(2^4-1)*(k-1)+1
    kernel_size: int = 3
    dropout: float = 0.1
    head_hidden: int = 64
    # Multi-task loss weighting. "uncertainty" = Kendall et al. homoscedastic
    # learned weights; "fixed" uses the static lambdas below.
    loss_weighting: Literal["uncertainty", "fixed"] = "uncertainty"
    lambda_reg: float = 1.0
    lambda_cls: float = 1.0
    huber_delta: float = 10.0      # mg/dL, in *unscaled* target units

    # ---- Multi-horizon -------------------------------------------------------
    # Number of prediction horizons the heads cover. K=1 reproduces the original
    # single-horizon model exactly, including tensor shapes, so existing
    # checkpoints and call sites keep working.
    n_horizons: int = 1

    # ---- Asymmetric hypoglycemia cost ---------------------------------------
    # Measured on the official Ohio test set: mean signed error below 70 mg/dL
    # was +8.8 mg/dL. The model reads lows as HIGHER than they are, which is the
    # one direction a hypoglycemia warning system must not fail in, and it is
    # invisible in RMSE because hypo windows are ~5% of the data.
    #
    # `hypo_asymmetry` (kappa) is the multiplier applied to the regression loss
    # when the reference is hypoglycemic AND the prediction sits ABOVE it. 1.0
    # disables the mechanism and recovers plain Huber.
    hypo_asymmetry: float = 3.0
    # Centre and width of the smooth ramp that decides "how hypo is this
    # reference". A hard indicator at 70 puts a discontinuity exactly where CGM
    # noise straddles the boundary; the ramp is the same reasoning as the
    # soft labels in data.SoftLabeler.
    hypo_cost_threshold_mgdl: float = 70.0
    hypo_cost_tau_mgdl: float = 10.0
    # Optionally extend the asymmetry to the whole range at reduced strength, so
    # over-prediction is mildly discouraged everywhere rather than only below
    # the threshold. 0.0 = confine the cost to the hypoglycemic ramp.
    global_asymmetry: float = 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    early_stop_patience: int = 10
    # Model selection metric on the validation fold. Deliberately the
    # forewarning metric, since that is the clinically load-bearing task.
    monitor: str = "val/pr_auc"
    monitor_mode: Literal["max", "min"] = "max"
    # Zero, deliberately. `WindowDataset` holds the fold's tensors already
    # materialised in memory, so each worker adds a full pickle of them PER
    # EPOCH -- `run_fold` sets no `persistent_workers`, so the pool is torn down
    # and respawned every epoch -- and buys no parallelism in return, because
    # there is no per-item decode to overlap. `benchmark._loader` and
    # `scripts/train_ohio.py` already use 0 for exactly this dataset shape.
    num_workers: int = 0
    seed: int = 1337

    # Patient-wise cross validation.
    n_folds: int = 5
    # Fraction of *training* patients held out as the inner validation fold.
    inner_val_frac: float = 0.2


@dataclass(frozen=True)
class AdaptConfig:
    """Post-training personalisation (see extension.py)."""
    method: Literal["lora", "bn_adapt", "head_only", "full"] = "lora"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    # How much of the new patient's stream may be used to adapt. This is a
    # *causal prefix*: the first N hours of their record, never a random sample.
    adapt_prefix_hours: float = 24.0
    # Gap between the adaptation prefix and the evaluation region, to prevent
    # window overlap from bleeding adaptation data into test targets.
    embargo_min: int = 60
    lr: float = 5e-4
    steps: int = 200
    batch_size: int = 64
    # Elastic-weight-style anchor to the generic model, guards against
    # catastrophic forgetting on very short prefixes.
    l2_sp: float = 1e-3

    # ---- LoRA injection scope ------------------------------------------------
    # Whether LoRA also wraps the cross-attention projections, and which ones.
    # These exist as real fields rather than as `getattr` defaults inside
    # `extension.inject_lora`, because a knob that is only ever read through a
    # default is not a knob: `AdaptConfig` is frozen, so
    # `dataclasses.replace(cfg, lora_attention=False)` would raise TypeError and
    # the "trunk vs trunk+fusion" ablation the docstring promises would be
    # unreachable.
    lora_attention: bool = True
    lora_attention_targets: tuple[str, ...] = ("q", "k", "v", "proj")


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    out_dir: str = "runs/default"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent, default=list)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            adapt=AdaptConfig(**d.get("adapt", {})),
            out_dir=d.get("out_dir", "runs/default"),
        )
