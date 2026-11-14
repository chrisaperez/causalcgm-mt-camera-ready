"""CausalCGM-MT: shared temporal backbone, two heads, multi-task loss.

The architectural claim being replicated is that hypoglycemia forewarning and
BG regression share most of their useful representation, and that training them
jointly regularises both -- in particular that the classification head pulls the
shared trunk toward resolving the low-glucose region, which a pure MSE
regression objective under-weights because low excursions are rare.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# Causal temporal convolution
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Conv1d):
    """Dilated 1-D convolution with left-only padding.

    A standard `padding=k//2` conv is centred and therefore reads `k//2` steps
    into the future at every position -- an architectural lookahead leak that is
    invisible in the metrics but real. Here we pad only on the left and trim the
    tail, so output ``i`` depends strictly on inputs ``<= i``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        self._left_pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=self._left_pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        out = super().forward(x)
        return out[..., : -self._left_pad] if self._left_pad else out


class TemporalBlock(nn.Module):
    """Residual dilated block: two causal convs + weight norm + GELU + dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        # BatchNorm is used deliberately: it is the adaptation surface exploited
        # by the test-time BN personalisation in extension.py.
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return self.act(h + self.down(x))


class TCNBackbone(nn.Module):
    """Stack of exponentially dilated causal blocks."""

    def __init__(self, in_ch: int, cfg: ModelConfig):
        super().__init__()
        chans = [cfg.hidden_channels] * cfg.n_blocks
        blocks, prev = [], in_ch
        for i, c in enumerate(chans):
            blocks.append(TemporalBlock(prev, c, cfg.kernel_size, 2**i, cfg.dropout))
            prev = c
        self.net = nn.Sequential(*blocks)
        self.out_channels = prev

    @property
    def receptive_field(self) -> int:
        # 2 convs per block, dilation doubling: RF = 1 + 2*(k-1)*(2^B - 1)
        b = len(self.net)
        k = self.net[0].conv1.kernel_size[0]
        return 1 + 2 * (k - 1) * (2**b - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, C', T)
        return self.net(x.transpose(1, 2))


class DilatedLSTMBackbone(nn.Module):
    """Alternative recurrent trunk. Unidirectional -- bidirectional would be a
    lookahead leak, since the backward pass reads future timesteps."""

    def __init__(self, in_ch: int, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            in_ch, cfg.hidden_channels, num_layers=cfg.n_blocks,
            batch_first=True, dropout=cfg.dropout, bidirectional=False,
        )
        self.out_channels = cfg.hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return h.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    def __init__(self, in_ch: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _stack_heads(
    reg_heads: nn.ModuleList,
    cls_heads: nn.ModuleList,
    z: torch.Tensor,
    last_bg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every horizon head over the shared representation.

    Defined once and used by both models so the unimodal and multimodal paths
    cannot drift apart in output shape -- a difference there would surface as a
    silent broadcast in the loss rather than as an error.

    Every head predicts a RESIDUAL from the last observed BG, at every horizon.
    The residual parameterisation is what stops a model scoring well by echoing
    its input, and it matters more at 60 minutes than at 30: the further out the
    horizon, the more attractive echoing becomes.
    """
    if len(reg_heads) == 1:
        return last_bg + reg_heads[0](z), cls_heads[0](z)
    bg = torch.stack([last_bg + head(z) for head in reg_heads], dim=-1)   # (B, K)
    logit = torch.stack([head(z) for head in cls_heads], dim=-1)          # (B, K)
    return bg, logit


class CausalCGMMT(nn.Module):
    """Shared trunk -> {regression, hypoglycemia} heads.

    The regression head predicts a *residual from the last observed BG* rather
    than the absolute level. This is a meaningful choice: absolute-level
    regression lets a model score well by echoing its input, which inflates RMSE
    performance while producing large time lag. Residual parameterisation makes
    the model commit to a direction of change.
    """

    def __init__(self, in_channels: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = (
            TCNBackbone(in_channels, cfg) if cfg.backbone == "tcn"
            else DilatedLSTMBackbone(in_channels, cfg)
        )
        c = self.backbone.out_channels
        # Pool the last step plus a global average: the last step carries the
        # current state, the average carries recent variability.
        self.n_horizons = max(1, int(cfg.n_horizons))
        self.reg_heads = nn.ModuleList(
            [Head(2 * c, cfg.head_hidden, cfg.dropout) for _ in range(self.n_horizons)])
        self.cls_heads = nn.ModuleList(
            [Head(2 * c, cfg.head_hidden, cfg.dropout) for _ in range(self.n_horizons)])

    # Single-horizon aliases. Personalisation (`extension.configure_trainable`)
    # and any external checkpoint code refer to `reg_head`/`cls_head`; keeping
    # them as properties means the K=1 model stays a drop-in for everything that
    # already existed, instead of that code silently selecting no parameters.
    @property
    def reg_head(self) -> nn.Module:
        return self.reg_heads[0]

    @property
    def cls_head(self) -> nn.Module:
        return self.cls_heads[0]

    def forward(self, x: torch.Tensor, last_bg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C) scaled features. last_bg: (B,) UNSCALED mg/dL.

        Returns (B,) tensors when K == 1 and (B, K) when K > 1, so the
        single-horizon call sites are shape-identical to before the refactor.
        """
        h = self.backbone(x)                      # (B, C', T)
        z = torch.cat([h[..., -1], h.mean(-1)], dim=-1)
        return _stack_heads(self.reg_heads, self.cls_heads, z, last_bg)


# --------------------------------------------------------------------------- #
# Asymmetric hypoglycemia-aware regression cost
# --------------------------------------------------------------------------- #
def hypo_cost_weights(
    bg_hat: torch.Tensor,
    y_reg: torch.Tensor,
    kappa: float,
    threshold: float,
    tau: float,
    global_asymmetry: float = 0.0,
) -> torch.Tensor:
    """Per-sample multiplier for the regression loss.

    WHICH DIRECTION IS PENALISED, AND WHY
    -------------------------------------
    The audit measured a mean signed error of +8.8 mg/dL below 70 mg/dL: the
    model predicts HIGHER than the truth during hypoglycemia. In Clarke terms
    that is the road to zone D -- reference is hypo, prediction says normal, no
    warning fires. So the penalised direction is ``bg_hat > y_reg`` when
    ``y_reg`` is low. The model is pushed to read ambiguous lows as lower,
    trading false alarms (zone B/C, an unnecessary snack) for missed lows
    (zone D, a severe event). That trade is deliberate and it is the whole point
    of the mechanism -- it will make RMSE slightly WORSE, and anyone comparing
    runs on RMSE alone should expect that.

    SHAPE OF THE WEIGHT

        w = 1 + (kappa - 1) * ramp(y) * 1[bg_hat > y] + global_asymmetry * 1[bg_hat > y]

        ramp(y) = sigmoid((threshold - y) / tau)

    The ramp is smooth rather than a hard ``y < 70`` indicator, for the same
    reason the classification labels are soft: CGM error is on the order of
    10 mg/dL, so a hard cut puts a discontinuity precisely where the true class
    is least certain, and the network ends up fitting device noise at the
    boundary. With tau = 10 the weight is half-strength at 70, ~kappa by 50, and
    ~1 by 90.

    The result is a weighted (asymmetric) Huber -- structurally the same idea as
    pinball/quantile loss, but with the asymmetry gated on the *reference* being
    clinically dangerous rather than applied uniformly. Uniform quantile
    regression would shift the whole prediction curve down, degrading
    euglycemic and hyperglycemic accuracy to buy hypo sensitivity; gating on the
    ramp confines the cost to the region where the asymmetry is clinically
    justified.

    Returns a tensor broadcastable to ``bg_hat``; all-ones when kappa == 1 and
    global_asymmetry == 0, so the mechanism is exactly disabled at its default
    off-setting rather than approximately so.
    """
    over = (bg_hat > y_reg).to(bg_hat.dtype)          # 1 where the low is understated
    if kappa == 1.0 and global_asymmetry == 0.0:
        return torch.ones_like(bg_hat)
    ramp = torch.sigmoid((threshold - y_reg) / max(tau, 1e-6))
    return 1.0 + (kappa - 1.0) * ramp * over + global_asymmetry * over


def asymmetric_huber(
    bg_hat: torch.Tensor,
    y_reg: torch.Tensor,
    delta: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Huber with per-sample weights, reduced by the WEIGHTED mean.

    Normalising by ``weights.sum()`` rather than by ``n`` keeps the loss on the
    same scale as unweighted Huber regardless of kappa. Without it, raising
    kappa raises the total loss magnitude, which silently rescales the learned
    homoscedastic task weights and changes the regression/classification balance
    as a side effect of a safety knob -- two things moving when the operator
    turned one.
    """
    per = F.huber_loss(bg_hat, y_reg, delta=delta, reduction="none")
    return (per * weights).sum() / weights.sum().clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Multi-task loss
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """Huber regression + soft-target BCE, with learned homoscedastic weights.

    Weighting. The two losses live on incompatible scales (mg/dL^2 vs nats), so
    a hand-tuned lambda is fragile. With ``loss_weighting="uncertainty"`` we
    learn log-variances per task (Kendall, Gal & Cipolla 2018):

        L = sum_i [ exp(-s_i) * L_i + s_i / 2 ]

    The ``+s_i/2`` term prevents the trivial solution of driving all weights to
    zero. We expose the effective weights for logging, because a silently
    collapsing task weight is the most common way a multi-task model degenerates
    into a single-task one.

    Soft targets. BCE-with-logits accepts targets in [0, 1], so the soft label
    is consumed directly; the loss reduces to the standard cross entropy when
    targets are hard. Positive-class reweighting uses a ``pos_weight`` computed
    from the TRAINING fold only.
    """

    def __init__(self, cfg: ModelConfig, pos_weight: float | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_horizons = max(1, int(getattr(cfg, "n_horizons", 1)))
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight is not None else torch.tensor(1.0),
        )
        if cfg.loss_weighting == "uncertainty":
            # One log-variance PER TASK PER HORIZON: 2K tasks, not 2. The
            # horizons have genuinely different irreducible noise -- 60-minute
            # BG is far less predictable than 30-minute -- so forcing them to
            # share a weight lets the easier horizon dominate the gradient and
            # the harder one degrade toward predicting the mean. Kendall's
            # formulation extends to 2K terms unchanged; nothing about it
            # assumes two tasks.
            self.log_var_reg = nn.Parameter(torch.zeros(self.n_horizons))
            self.log_var_cls = nn.Parameter(torch.zeros(self.n_horizons))

    @staticmethod
    def _as_2d(t: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B, 1); (B, K) unchanged. Keeps one code path for both."""
        return t.unsqueeze(-1) if t.dim() == 1 else t

    def forward(
        self,
        bg_hat: torch.Tensor,
        logit: torch.Tensor,
        y_reg: torch.Tensor,
        y_cls_soft: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Accepts (B,) tensors (K=1) or (B, K) tensors, in any consistent mix.

        Shapes are normalised rather than asserted equal, because a (B,) target
        broadcasting against a (B, K) prediction would silently compute K copies
        of the wrong loss instead of failing.
        """
        bg2, lg2 = self._as_2d(bg_hat), self._as_2d(logit)
        yr2, yc2 = self._as_2d(y_reg), self._as_2d(y_cls_soft)
        K = bg2.shape[-1]
        if not (lg2.shape[-1] == yr2.shape[-1] == yc2.shape[-1] == K):
            raise ValueError(
                f"horizon count mismatch: bg_hat K={K}, logit K={lg2.shape[-1]}, "
                f"y_reg K={yr2.shape[-1]}, y_cls K={yc2.shape[-1]}"
            )
        if K != self.n_horizons:
            raise ValueError(
                f"loss configured for {self.n_horizons} horizon(s) but received "
                f"K={K}. Set ModelConfig.n_horizons to match the data."
            )

        # Asymmetric hypoglycemia cost, applied per horizon. The weights depend
        # on the REFERENCE value, so each horizon gets its own ramp -- a window
        # can be euglycemic at 30 min and hypoglycemic at 60.
        w = hypo_cost_weights(
            bg2, yr2,
            kappa=float(getattr(self.cfg, "hypo_asymmetry", 1.0)),
            threshold=float(getattr(self.cfg, "hypo_cost_threshold_mgdl", 70.0)),
            tau=float(getattr(self.cfg, "hypo_cost_tau_mgdl", 10.0)),
            global_asymmetry=float(getattr(self.cfg, "global_asymmetry", 0.0)),
        )

        l_reg = torch.stack([
            asymmetric_huber(bg2[:, k], yr2[:, k], self.cfg.huber_delta, w[:, k])
            for k in range(K)
        ])
        l_cls = torch.stack([
            F.binary_cross_entropy_with_logits(
                lg2[:, k], yc2[:, k], pos_weight=self.pos_weight)
            for k in range(K)
        ])

        if self.cfg.loss_weighting == "uncertainty":
            total = (
                (torch.exp(-self.log_var_reg) * l_reg + 0.5 * self.log_var_reg).sum()
                + (torch.exp(-self.log_var_cls) * l_cls + 0.5 * self.log_var_cls).sum()
            )
            w_reg_v = torch.exp(-self.log_var_reg).detach()
            w_cls_v = torch.exp(-self.log_var_cls).detach()
        else:
            total = (self.cfg.lambda_reg * l_reg.sum()
                     + self.cfg.lambda_cls * l_cls.sum())
            w_reg_v = torch.full((K,), float(self.cfg.lambda_reg))
            w_cls_v = torch.full((K,), float(self.cfg.lambda_cls))

        w_reg = float(w_reg_v[0])
        w_cls = float(w_cls_v[0])
        l_reg_s, l_cls_s = l_reg.sum(), l_cls.sum()
        # Scalar keys are the SUM over horizons, so existing loggers and the
        # early-stopping monitor keep reading a single comparable number;
        # per-horizon detail is added alongside under h{k} suffixes.
        stats = {
            "loss": float(total.detach()),
            "loss_reg": float(l_reg_s.detach()),
            "loss_cls": float(l_cls_s.detach()),
            "w_reg": w_reg,
            "w_cls": w_cls,
            "hypo_cost_mean": float(w.mean().detach()),
        }
        stats.update({f"loss_reg_h{k}": float(l_reg[k].detach()) for k in range(K)})
        stats.update({f"loss_cls_h{k}": float(l_cls[k].detach()) for k in range(K)})
        stats.update({f"w_reg_h{k}": float(w_reg_v[k]) for k in range(K)})
        stats.update({f"w_cls_h{k}": float(w_cls_v[k]) for k in range(K)})
        return total, stats


# --------------------------------------------------------------------------- #
# Multimodal fusion (HUPA-UCM: CGM + wearables + treatment)
# --------------------------------------------------------------------------- #
class CausalCrossAttention(nn.Module):
    """Single-head cross-attention from CGM queries to auxiliary keys/values,
    masked so position i attends only to auxiliary positions j <= i.

    Why attention rather than plain concatenation. The auxiliary streams are
    informative at variable, state-dependent lags: a bolus matters 20-90 minutes
    later depending on carbohydrates on board, and an exercise bout depresses
    glucose for hours afterwards. Channel concatenation forces the convolution
    stack to learn those lags through a fixed receptive field; attention lets
    each CGM timestep select the auxiliary history that is relevant to it.

    Why the mask is not optional. Unmasked attention is bidirectional and would
    let the representation at step i read auxiliary values at i+1..T. Inside a
    single window every such position is still at or before the anchor, so it
    would not leak beyond t -- but it WOULD break the strict prefix property the
    rest of the pipeline maintains, and would silently invalidate any streaming
    deployment, where positions after i do not exist yet. The mask is applied
    with a boolean `masked_fill` to -inf before softmax, which is exact rather
    than approximate.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        cgm: torch.Tensor,          # (B, T, D) queries
        aux: torch.Tensor,          # (B, T, D) keys / values
        aux_valid: torch.Tensor,    # (B, T) 1 where the aux step is observed
    ) -> torch.Tensor:
        B, T, _ = cgm.shape
        q, k, v = self.q(cgm), self.k(aux), self.v(aux)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, T, T)

        # Causal mask: keep j <= i.
        causal = torch.ones(T, T, dtype=torch.bool, device=cgm.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        # Availability mask: never attend to an unobserved auxiliary step, so
        # the model cannot mistake an imputed zero for a real measurement.
        avail = aux_valid.bool().unsqueeze(1).expand(B, T, T)
        scores = scores.masked_fill(~avail, float("-inf"))

        # A row can be fully masked (step 0 with no observed aux). softmax over
        # all -inf is NaN, so those rows are zeroed explicitly instead.
        dead = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(dead, 0.0)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out = self.proj(torch.matmul(attn, v))
        return out.masked_fill(dead, 0.0)


class MultiModalCausalCGMMT(nn.Module):
    """Dual-branch causal TCN with gated, causally-masked late fusion.

        CGM   -> TCN_cgm ->\
                             cross-attn (causal) -> gate -> heads
        aux   -> TCN_aux ->/

    Design decisions worth defending:

    * Separate trunks, not early concatenation. The streams have different
      noise characteristics, different missingness patterns and wildly different
      scales; a shared first layer forces one normalisation regime on all of
      them. Separate trunks also mean the CGM branch stays usable on its own.

    * Gated fusion. The gate is computed from the CGM branch and modulated by
      auxiliary availability, so the model can learn to ignore the aux branch
      when it is absent. Without a gate, missing wearables inject a constant
      zero vector that the heads must learn to subtract.

    * Residual, not replacement. Fusion adds to the CGM representation. If the
      auxiliary branch is useless, the model degrades to the univariate one
      rather than to noise -- so the multimodal model is bounded below by the
      unimodal baseline instead of gambling against it.

    * Modality dropout during training. The whole aux branch is dropped for a
      random subset of samples, which prevents the network from becoming
      dependent on wearables that are routinely missing in deployment (Fitbit
      off the wrist, sync gaps). This is a robustness mechanism, not
      regularisation for its own sake.
    """

    def __init__(
        self,
        cgm_channels: int,
        aux_channels: int,
        cfg: ModelConfig,
        aux_valid_index: int | Sequence[int] | None = None,
        modality_dropout: float = 0.15,
        always_valid: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.modality_dropout = modality_dropout
        # True when the channels in use include a mask-free treatment stream
        # (insulin, carbs, basal). Those carry a real value at EVERY step of a
        # present window, so per-step validity must not be read from the
        # wearable masks alone: doing so let a window whose wearable was off
        # hide its insulin record from the attention entirely. Absence is then
        # handled by `aux_present` (see `data.aux_presence`).
        self.always_valid = bool(always_valid)
        # Which aux channel(s) indicate "this step was really observed". A step
        # counts as auxiliary-valid when ANY of them is set.
        #
        # WHY THIS ACCEPTS A SET AND NOT ONE CHANNEL
        # ------------------------------------------
        # It used to be a single index, defaulted to the heart-rate mask on the
        # grounds that heart rate is the densest wearable stream. On OhioT1DM
        # that silently disabled the entire fusion branch for half the cohort:
        # the 2020 subjects wore an Empatica, which reports no heart rate at
        # all, so `hr_observed` is identically zero across all 71,946 of their
        # windows, `aux_valid` is zero at every timestep, the cross-attention is
        # fully masked, and the model degrades to the CGM-only path -- while
        # still reporting itself as multimodal. The same bug made every
        # modality-ablation arm that excluded heart rate produce bit-identical
        # numbers, because they were all secretly the same CGM-only model.
        #
        # Passing the mask channels of the streams an arm actually uses makes
        # validity a property of that arm rather than of one hard-coded sensor.
        # An empty set (or None) means "no mask channels", i.e. every step is
        # valid -- correct for always-on streams like insulin-on-board, which
        # have no observation mask because a zero is a real zero.
        self.aux_valid_index: tuple[int, ...] = (
            () if aux_valid_index is None
            else (int(aux_valid_index),) if isinstance(aux_valid_index, (int, np.integer))
            else tuple(int(i) for i in aux_valid_index)
        )

        self.cgm_backbone = (
            TCNBackbone(cgm_channels, cfg) if cfg.backbone == "tcn"
            else DilatedLSTMBackbone(cgm_channels, cfg)
        )
        # The auxiliary trunk is deliberately narrower: it carries a lower
        # information density than CGM and matching its capacity invites the
        # fusion to overfit sensor idiosyncrasies.
        aux_cfg = dataclasses.replace(cfg, hidden_channels=max(16, cfg.hidden_channels // 2))
        self.aux_backbone = (
            TCNBackbone(aux_channels, aux_cfg) if cfg.backbone == "tcn"
            else DilatedLSTMBackbone(aux_channels, aux_cfg)
        )

        d = self.cgm_backbone.out_channels
        self.aux_proj = nn.Linear(self.aux_backbone.out_channels, d)
        self.cross_attn = CausalCrossAttention(d, cfg.dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        # Normalise the FUSION TERM ONLY, never the summed representation.
        #
        # An earlier version applied LayerNorm to (h_cgm + fused). That silently
        # broke the "bounded below by unimodal" guarantee this class claims:
        # LayerNorm rescales each timestep across channels, destroying the
        # magnitude information the regression head needs to predict a residual
        # in mg/dL. Measured cost with the auxiliary stream carrying NOTHING:
        # PR-AUC 0.963 -> 0.927 and time lag 20 -> 30 min (i.e. back to
        # persistence). Normalising only the fusion term keeps the CGM path an
        # exact identity.
        #
        # elementwise_affine=False is required for the guarantee to be exact:
        # with a learned bias, LayerNorm(0) returns that bias rather than 0, so
        # an absent modality would still perturb the CGM representation.
        self.norm = nn.LayerNorm(d, elementwise_affine=False)

        # One head PAIR per horizon over the shared fused representation. The
        # trunk and the fusion policy are shared across horizons by
        # construction; only the readout is horizon-specific. That is the point
        # of the refactor: 30- and 60-minute forecasting are the same physiology
        # read at different lead times, so learning two separate trunks spends
        # twice the parameters to relearn the same dynamics -- and halves the
        # effective data each trunk sees for the rare hypo events.
        self.n_horizons = max(1, int(cfg.n_horizons))
        self.reg_heads = nn.ModuleList(
            [Head(2 * d, cfg.head_hidden, cfg.dropout) for _ in range(self.n_horizons)])
        self.cls_heads = nn.ModuleList(
            [Head(2 * d, cfg.head_hidden, cfg.dropout) for _ in range(self.n_horizons)])

    @property
    def reg_head(self) -> nn.Module:
        return self.reg_heads[0]

    @property
    def cls_head(self) -> nn.Module:
        return self.cls_heads[0]

    def forward(
        self,
        x: torch.Tensor,          # (B, T, C_cgm) scaled CGM features
        x_aux: torch.Tensor,      # (B, T, C_aux) scaled auxiliary features
        last_bg: torch.Tensor,    # (B,) UNSCALED mg/dL
        aux_present: torch.Tensor | None = None,   # (B,) 1 if aux usable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        h_cgm = self.cgm_backbone(x).transpose(1, 2)                # (B, T, D)
        h_aux = self.aux_proj(self.aux_backbone(x_aux).transpose(1, 2))

        # Per-step auxiliary validity: any configured mask channel set at that
        # step. With no channels configured every step is valid.
        if self.aux_valid_index and not self.always_valid:
            obs = x_aux[..., list(self.aux_valid_index)]          # (B, T, K)
            aux_valid = (obs > 0.5).any(dim=-1).to(x.dtype)
        else:
            aux_valid = torch.ones(B, T, device=x.device, dtype=x.dtype)

        if aux_present is None:
            aux_present = torch.ones(B, device=x.device)

        # Modality dropout: drop the aux branch for a random subset of the
        # batch, so the model keeps working when wearables are missing.
        if self.training and self.modality_dropout > 0:
            keep = (torch.rand(B, device=x.device) >= self.modality_dropout).float()
            aux_present = aux_present * keep

        aux_valid = aux_valid * aux_present.unsqueeze(1)

        fused = self.cross_attn(h_cgm, h_aux, aux_valid)
        g = self.gate(torch.cat([h_cgm, fused], dim=-1))
        # Residual + gate: aux modulates the CGM representation, never replaces
        # it, and is scaled to zero for samples with no auxiliary data at all.
        # With aux_present = 0 this is EXACTLY h_cgm -- the property that makes
        # the multimodal model bounded below by the unimodal one.
        h = h_cgm + g * self.norm(fused) * aux_present.view(B, 1, 1)

        # Pool the anchor step plus the window mean. Both read only positions
        # <= T-1, i.e. at or before the anchor.
        z = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        return _stack_heads(self.reg_heads, self.cls_heads, z, last_bg)


# --------------------------------------------------------------------------- #
# Checkpoint schema compatibility
# --------------------------------------------------------------------------- #
# Head layout before the multi-horizon refactor:  reg_head.*   / cls_head.*
# Head layout after it:                           reg_heads.0.* / cls_heads.0.*
#
# `CausalCGMMT.reg_head` is a property, and a property does NOT alias state_dict
# keys -- so every checkpoint written before the refactor fails to load into the
# current class with "Missing key(s) ... Unexpected key(s) ...". That broke the
# ONNX export path, the C++ parity gate and the dashboard at once, while the
# refactor's own comments claimed existing checkpoints kept working.
#
# The remap is one-directional and lossless: a legacy single-horizon checkpoint
# is exactly a K=1 checkpoint under the new names.
_LEGACY_HEAD_PREFIXES: tuple[tuple[str, str], ...] = (
    ("reg_head.", "reg_heads.0."),
    ("cls_head.", "cls_heads.0."),
)


def is_legacy_state_dict(state_dict: dict) -> bool:
    """True when `state_dict` uses the pre-multi-horizon head names."""
    return any(k.startswith(old) for k in state_dict for old, _ in _LEGACY_HEAD_PREFIXES)


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """Rename pre-refactor head keys onto the current schema.

    A no-op for checkpoints already in the current layout, so call sites can
    apply it unconditionally rather than branching on provenance. Key order is
    preserved; nothing else in the dict is touched.
    """
    if not is_legacy_state_dict(state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        for old, new in _LEGACY_HEAD_PREFIXES:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_state_dict_compat(model: nn.Module, state_dict: dict, strict: bool = True):
    """`model.load_state_dict`, tolerating pre-refactor head names."""
    return model.load_state_dict(remap_legacy_state_dict(state_dict), strict=strict)


def build_multimodal_model(
    cgm_channels: int,
    aux_channels: int,
    cfg: ModelConfig,
    pos_weight: float | None = None,
    aux_valid_index: int | Sequence[int] | None = None,
    modality_dropout: float = 0.15,
    always_valid: bool = False,
):
    model = MultiModalCausalCGMMT(
        cgm_channels, aux_channels, cfg,
        aux_valid_index=aux_valid_index, modality_dropout=modality_dropout,
        always_valid=always_valid,
    )
    return model, MultiTaskLoss(cfg, pos_weight)


def criterion_from_state(state: dict, cfg: ModelConfig, context: str = "") -> MultiTaskLoss:
    """Rebuild a `MultiTaskLoss` with its FITTED state restored, or refuse.

    `MultiTaskLoss` carries two fitted POPULATION quantities that a fresh
    construction silently discards:

      * `log_var_reg` / `log_var_cls` -- the Kendall homoscedastic task weights,
        `nn.Parameter`s learned jointly with the model. A fresh criterion resets
        them to zeros, i.e. an equal regression/classification weighting the
        model was never trained under.
      * `pos_weight` -- a registered buffer holding the training fold's class
        prior, ~18 at 5% hypoglycemia prevalence. A fresh criterion resets it to
        1.0, dropping the rare-event reweighting entirely.

    Both live in `state_dict()`, so persisting and restoring them costs a few
    hundred bytes. Neither is visible in an adapted model's own metrics, which
    is why this raises rather than warning: a personalisation run that adapts
    under a uniform task balance and an unweighted BCE is not the same
    experiment as one that does not, and the difference shows up only as a
    slightly different number in a table.

    `context` is prepended to the error so the caller says WHICH artefact is
    short, rather than leaving the reader to guess between three drivers.
    """
    if not state:
        where = f"{context}: " if context else ""
        raise RuntimeError(
            f"{where}no multi-task criterion state was recorded, so rebuilding "
            f"it would reset the Kendall log-variances to a uniform task "
            f"balance and pos_weight to 1.0 -- adapting or scoring under a "
            f"different loss than the model was trained with. Re-run training, "
            f"or re-save the checkpoint with "
            f"`criterion_state_dict={{k: v.detach().cpu() for k, v in "
            f"criterion.state_dict().items()}}`.")
    criterion = MultiTaskLoss(cfg)
    criterion.load_state_dict(state)
    return criterion


def build_model(in_channels: int, cfg: ModelConfig, pos_weight: float | None = None):
    model = CausalCGMMT(in_channels, cfg)
    if isinstance(model.backbone, TCNBackbone):
        rf = model.backbone.receptive_field
        if rf < 12:
            import warnings
            warnings.warn(f"TCN receptive field is only {rf} steps; increase n_blocks.")
    return model, MultiTaskLoss(cfg, pos_weight)
