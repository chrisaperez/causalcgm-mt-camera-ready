"""Extension: post-training personalisation to an unseen patient.

Problem being solved
--------------------
The cross-validated model is trained across a population and evaluated on
patients it has never seen. That is the honest generalisation number, but it is
also pessimistic relative to deployment: in practice, a new user wears the
sensor for a day before you need good predictions, and that day is legitimately
available. Glucose dynamics are strongly individual (insulin sensitivity,
basal patterns, meal timing), so a small amount of subject-specific adaptation
should recover a disproportionate share of the gap to a fully personal model.

Causal protocol (this is the part that makes the result meaningful)
-------------------------------------------------------------------
For each held-out patient:

    |<-- adapt prefix (first K hours) -->|<-- embargo -->|<-- evaluation -->|

  * The adaptation set is a strict PREFIX in wall-clock time, never a random
    sample of the patient's record. Random sampling would let the model adapt
    on data recorded after the windows it is scored on.
  * An embargo of at least (history + horizon) separates the two regions, so no
    evaluation window's history overlaps any adaptation window's target. Without
    it, adjacent windows share ~95% of their content and "adaptation" quietly
    becomes memorisation of the test region.
  * The population scaler and soft-label band are NOT refit. They were fit on
    the training population; refitting them on the new patient would import that
    patient's distribution into the transform.

Three adaptation mechanisms are provided, in increasing capacity:

  bn_adapt   Recompute BatchNorm running statistics on the prefix, all weights
             frozen. Zero trainable parameters, zero overfitting risk, and it
             addresses the dominant failure mode (covariate shift in the input
             distribution). This is the right default for very short prefixes.

  lora       Low-rank adapters on the backbone's convolutions. Trains
             r*(in+out) parameters per layer instead of the full kernel, which
             keeps the effective capacity far below the size of the prefix and
             makes overfitting tractable. Merges back into the base weights for
             inference, so there is no latency cost.

  head_only  Fine-tune the two heads, freeze the trunk. Useful as a middle
             baseline in the ablation.

All three are anchored to the generic weights with an L2-SP penalty, which
penalises distance from the starting point rather than from zero -- the correct
regulariser when the prior is "the population model is already good". For LoRA
that anchor has a closed form: since W = W0 + dW by construction, ||W - W0||^2
IS ||dW||^2, so the penalty is computed on the effective low-rank update rather
than on the (frozen, and therefore trivially unmoved) base weights. See
`lora_delta_w`.

TWO ENTRY POINTS
----------------
`run_personalization` takes a generic `WindowSet` and a list of held-out
patients: the cross-validated, unseen-subject protocol.

`run_ohio_personalization` takes an `OhioSplit` and adapts WITHIN each subject
under the official OhioT1DM protocol, on the multimodal (x_cgm, x_aux, mask)
tensors and the multi-horizon Kendall loss. The two differ in the split they
consume, not in the adaptation mechanics -- LoRA injection, BN recalibration and
the L2-SP anchor are shared, so an improvement measured on one arm is not
confounded by a different implementation on the other.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptConfig, Config
from .data import CausalScaler, SoftLabeler, WindowSet
from .evaluate import EvalReport, evaluate
from .models import CausalCGMMT, MultiModalCausalCGMMT, MultiTaskLoss
from .train import WindowDataset, _forward, pick_device, predict, set_seed

from torch.utils.data import DataLoader

# Personalisation methods, in increasing capacity. "none" is the zero-shot
# population arm: it is a real experimental condition, not the absence of one,
# and naming it here keeps the ablation table honest about what it compares to.
AdaptMethod = Literal["none", "bn_adapt", "lora", "head_only", "full"]
ADAPT_METHODS: tuple[str, ...] = ("none", "bn_adapt", "lora", "head_only", "full")


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRAConv1d(nn.Module):
    """Wraps a frozen Conv1d with a trainable low-rank update.

    y = W0 x + (alpha / r) * B(A x)

    ``A`` is the down-projection (initialised random), ``B`` the up-projection
    (initialised to zero, so the adapted model starts exactly equal to the base
    model -- adaptation can only improve from a correct starting point).

    The down-projection is a pointwise (1x1) conv and the up-projection carries
    the temporal kernel, preserving the base layer's causal padding behaviour by
    delegating to a clone of it.
    """

    def __init__(self, base: nn.Conv1d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        # Clone the base conv's temporal geometry (kernel, dilation, padding) so
        # the update has the same receptive field and causality as W0.
        self.lora_b = _clone_geometry(base, rank)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        if getattr(self.lora_b, "bias", None) is not None:
            nn.init.zeros_(self.lora_b.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.drop(self.lora_a(x)))


def _clone_geometry(base: nn.Conv1d, rank: int) -> nn.Module:
    """Build an up-projection with the same class and temporal geometry as
    ``base`` (this preserves CausalConv1d's left-padding-and-trim behaviour)."""
    cls = type(base)
    try:
        m = cls(rank, base.out_channels, base.kernel_size[0], dilation=base.dilation[0])
    except TypeError:  # plain nn.Conv1d signature
        m = nn.Conv1d(
            rank, base.out_channels, base.kernel_size[0],
            padding=base.padding, dilation=base.dilation[0],
        )
    if getattr(m, "bias", None) is not None:
        nn.init.zeros_(m.bias)
    return m


class LoRALinear(nn.Module):
    """Wraps a frozen Linear with a trainable low-rank update.

        y = W0 x + (alpha / r) * B(A x)

    Same contract as `LoRAConv1d`: ``B`` starts at zero, so the adapted model is
    bit-identical to the base model before the first optimiser step.

    This exists so LoRA can reach the CROSS-ATTENTION projections. In the
    multimodal model the attention layer is where the CGM representation decides
    how much to read from the wearable branch, and that weighting is exactly
    what differs between patients: how strongly heart rate or electrodermal
    activity predicts a given person's excursions is individual. Adapting only
    the convolutional trunk leaves that fusion policy fixed at the population
    average.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.drop(self.lora_a(x)))


def _lora_trunks(model: nn.Module) -> list[nn.Module]:
    """The feature-extractor subtrees LoRA may touch.

    The unimodal model exposes `backbone`; the multimodal model exposes
    `cgm_backbone` and `aux_backbone` and no `backbone` at all. Looking only for
    `backbone` raised AttributeError on every multimodal model, which meant
    `--adapt lora` was unreachable for the fusion arm.
    """
    trunks = [getattr(model, n) for n in ("backbone", "cgm_backbone", "aux_backbone")
              if getattr(model, n, None) is not None]
    if not trunks:
        raise RuntimeError(
            f"{type(model).__name__} exposes no recognised backbone "
            f"(looked for backbone / cgm_backbone / aux_backbone)."
        )
    return trunks


def inject_lora(model: nn.Module, cfg: AdaptConfig) -> int:
    """Replace trunk convolutions -- and optionally the attention projections --
    with LoRA-wrapped versions, in place.

    Heads are left alone: they are already small, and letting both the trunk and
    the heads move on a short prefix is where personalisation starts to overfit.
    Returns the number of layers wrapped.
    """
    # Snapshot the targets BEFORE mutating. Iterating `.modules()` while
    # replacing children would descend into the LoRA wrappers just created and
    # wrap their frozen `base` again, recursing forever.
    targets = [
        (module, name, child)
        for trunk in _lora_trunks(model)
        for module in trunk.modules()
        if not isinstance(module, (LoRAConv1d, LoRALinear))
        for name, child in module.named_children()
        # Skip 1x1 downsample convs -- a rank-r update to a rank-limited
        # pointwise map adds little and costs parameters.
        if isinstance(child, nn.Conv1d) and child.kernel_size[0] > 1
    ]
    for module, name, child in targets:
        setattr(module, name, LoRAConv1d(child, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout))

    n = len(targets)
    if cfg.lora_attention:
        n += inject_lora_attention(model, cfg)
    return n


def inject_lora_attention(model: nn.Module, cfg: AdaptConfig) -> int:
    """Wrap the cross-attention q/k/v (and output) projections.

    Deliberately separate from the trunk pass so the two can be ablated
    independently -- "does adapting the fusion policy help beyond adapting the
    feature extractor?" is a question worth being able to answer.

    `v` and `proj` are included alongside `q`/`k` because adapting only the
    query/key pair changes WHERE the model looks without changing WHAT it reads;
    for a per-patient sensor-reliability shift, the value path is the one that
    needs to move.
    """
    attn = getattr(model, "cross_attn", None)
    if attn is None:
        return 0
    names = tuple(cfg.lora_attention_targets)
    n = 0
    for name in names:
        child = getattr(attn, name, None)
        if isinstance(child, nn.Linear):
            setattr(attn, name, LoRALinear(child, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout))
            n += 1
    return n


# --------------------------------------------------------------------------- #
# L2-SP anchoring
# --------------------------------------------------------------------------- #
def lora_delta_w(module: LoRAConv1d | LoRALinear) -> torch.Tensor:
    """The EFFECTIVE weight update a LoRA wrapper applies, as a dense tensor.

    Materialised rather than approximated because it is what L2-SP is defined
    on. With W = W0 + dW and B zero-initialised, ||W - W0||^2 = ||dW||^2 -- so
    penalising `dW` is the exact L2-SP term, whereas penalising the raw adapter
    parameters ||A||^2 + ||B||^2 is a different (and rank-dependent) quantity
    that would change meaning when `lora_rank` changes and make an r=4 vs r=8
    ablation non-comparable.

    Cheap despite being dense: r is 4 or 8, so the contraction is a handful of
    small matmuls per layer.
    """
    if isinstance(module, LoRALinear):
        return module.scaling * (module.lora_b.weight @ module.lora_a.weight)
    if isinstance(module, LoRAConv1d):
        # A is a 1x1 conv: (r, in, 1). B carries the temporal kernel: (out, r, k).
        a = module.lora_a.weight[..., 0]                     # (r, in)
        b = module.lora_b.weight                             # (out, r, k)
        return module.scaling * torch.einsum("ork,ri->oik", b, a)
    raise TypeError(f"not a LoRA module: {type(module).__name__}")


def lora_delta_norm_sq(model: nn.Module) -> torch.Tensor:
    """Sum of ||dW||^2 over every LoRA adapter in the model."""
    terms = [(lora_delta_w(m) ** 2).sum() for m in model.modules()
             if isinstance(m, (LoRAConv1d, LoRALinear))]
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).sum()


def l2_sp_penalty(
    model: nn.Module,
    anchor: dict[str, torch.Tensor],
    method: str,
    device: torch.device,
) -> torch.Tensor:
    """Distance from the population solution, in the parameterisation in use.

    One function so the LoRA and full/head arms cannot drift into penalising
    different things -- which is exactly what had happened: the module docstring
    promised all three methods were anchored, and LoRA was silently
    unregularised because its base weights are frozen and therefore always at
    distance zero.
    """
    if method == "lora":
        return lora_delta_norm_sq(model)
    terms = [((p - anchor[n].to(device)) ** 2).sum()
             for n, p in model.named_parameters()
             if p.requires_grad and n in anchor]
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).sum()


# --------------------------------------------------------------------------- #
# Parameter selection per method
# --------------------------------------------------------------------------- #
def configure_trainable(model: CausalCGMMT, cfg: AdaptConfig) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)

    if cfg.method == "none":
        return []  # the zero-shot control: nothing is trained, by definition

    if cfg.method == "bn_adapt":
        return []  # statistics only, no gradient step

    if cfg.method == "lora":
        # Injection is idempotent: a model that has already been wrapped (by a
        # caller that wanted to inspect the adapter count first) must not be
        # wrapped twice, and must not look like a failed match either.
        already = sum(1 for m in model.modules() if isinstance(m, (LoRAConv1d, LoRALinear)))
        n = already or inject_lora(model, cfg)
        if n == 0:
            raise RuntimeError("LoRA injection matched no layers.")
        params = [p for m in model.modules() if isinstance(m, (LoRAConv1d, LoRALinear))
                  for p in list(m.lora_a.parameters()) + list(m.lora_b.parameters())]
    elif cfg.method == "head_only":
        # EVERY horizon's heads, not just the primary one. `model.reg_head` is a
        # compatibility property resolving to `reg_heads[0]`, so using it here
        # trained 2 of 4 heads on a K=2 model and left the 60-minute pair frozen
        # -- an arm reported as "head-only fine-tuning" whose second horizon was
        # a bit-identical no-op for every subject.
        params = [p for lst in (model.reg_heads, model.cls_heads)
                  for head in lst for p in head.parameters()]
    elif cfg.method == "full":
        params = list(model.parameters())
    else:
        raise ValueError(f"unknown method {cfg.method}")

    for p in params:
        p.requires_grad_(True)
    return params


# --------------------------------------------------------------------------- #
# Causal prefix / embargo split for one patient
# --------------------------------------------------------------------------- #
def prefix_embargo_split(
    ws: WindowSet,
    patient: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adapt_idx, eval_idx) for one patient under the causal protocol.

    The embargo is the max of the configured value and (history + horizon), so
    it is impossible for an evaluation window's history to overlap an adaptation
    window's target regardless of how the config is set.
    """
    idx = np.flatnonzero(ws.patient == patient)
    if idx.size == 0:
        return idx, idx
    order = idx[np.argsort(ws.t_time[idx])]
    t = ws.t_time[order].astype("datetime64[m]")
    t0 = t[0]

    prefix_end = t0 + np.timedelta64(int(cfg.adapt.adapt_prefix_hours * 60), "m")
    min_embargo = cfg.data.history_min + cfg.data.horizon_min
    embargo = max(cfg.adapt.embargo_min, min_embargo)
    eval_start = prefix_end + np.timedelta64(embargo, "m")

    adapt_idx = order[t <= prefix_end]
    eval_idx = order[t >= eval_start]
    return adapt_idx, eval_idx


# --------------------------------------------------------------------------- #
# Adaptation
# --------------------------------------------------------------------------- #
@dataclass
class AdaptResult:
    patient: str
    before: EvalReport
    after: EvalReport
    n_adapt_windows: int
    n_trainable: int
    method: str


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multimodal: bool = False,
) -> None:
    """Reset and re-estimate BatchNorm running statistics on the prefix.

    Momentum=None makes each BN layer accumulate a true cumulative average over
    the passes, rather than an exponentially-weighted one dominated by the last
    batch.

    Only the BatchNorm layers are put in training mode. Ordinary dropout and
    modality dropout are held at their inference behaviour, because the running
    statistics have to describe the activation distribution the model will
    actually see at inference: estimating them under dropout measures a
    distribution that never occurs in deployment, and the effect is not
    symmetric -- dropout inflates the activation variance, so the recalibrated
    model would under-normalise every input it is later shown.
    """
    saved_modality_dropout = getattr(model, "modality_dropout", None)

    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
    if saved_modality_dropout is not None:
        model.modality_dropout = 0.0

    try:
        for batch in loader:
            _forward(model, batch, device, multimodal)
    finally:
        if saved_modality_dropout is not None:
            model.modality_dropout = saved_modality_dropout
        model.eval()


def adapt_patient(
    base_model: CausalCGMMT,
    ws: WindowSet,
    patient: str,
    scaler: CausalScaler,
    labeler: SoftLabeler,
    criterion: MultiTaskLoss,
    cfg: Config,
) -> AdaptResult | None:
    """Personalise a copy of the generic model to one held-out patient."""
    set_seed(cfg.train.seed)
    device = pick_device()

    adapt_idx, eval_idx = prefix_embargo_split(ws, patient, cfg)
    if adapt_idx.size < cfg.adapt.batch_size or eval_idx.size < 50:
        return None  # not enough data to make a defensible claim either way

    def loader(idx, shuffle):
        return DataLoader(
            WindowDataset(
                scaler.transform(ws.x[idx]),   # population scaler, NOT refit
                ws.y_reg[idx],
                labeler(ws.y_min[idx]),        # population band, NOT refit
                ws.last_bg[idx],
            ),
            batch_size=cfg.adapt.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    dl_adapt = loader(adapt_idx, True)
    dl_eval = loader(eval_idx, False)

    def score(m) -> EvalReport:
        bg_hat, s = predict(m, dl_eval, device)
        return evaluate(
            ws.y_reg[eval_idx], bg_hat, ws.y_cls_hard[eval_idx], s,
            ws.last_bg[eval_idx], ws.patient[eval_idx], cfg.data.sample_period_min,
        )

    model = copy.deepcopy(base_model).to(device)
    # The criterion carries the Kendall log-variances as PARAMETERS, so it has
    # to move too. `adapt_ohio_subject` already did this; the omission here was
    # hidden behind the NameError above, and surfaced as a device mismatch on
    # MPS/CUDA the moment that was fixed.
    criterion = criterion.to(device)
    before = score(model)

    acfg = cfg.adapt
    if acfg.method == "none":
        # The zero-shot arm, and on THIS protocol the only paired baseline
        # there is -- `run_personalization` reports before/after per patient,
        # so "none" is what says how much of a delta is adaptation rather than
        # the harness. `adapt_ohio_subject` has always had it; here the call
        # fell through `configure_trainable` to its `unknown method` raise, so
        # the arm `ADAPT_METHODS` advertises was unreachable. Scored once and
        # reported twice on purpose: the "after" of a no-op must be identical
        # to its "before".
        return AdaptResult(
            patient=str(patient),
            before=before,
            after=before,
            n_adapt_windows=int(adapt_idx.size),
            n_trainable=0,
            method=acfg.method,
        )

    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = configure_trainable(model, acfg)
    model = model.to(device)

    if acfg.method == "bn_adapt":
        recalibrate_bn(model, loader(adapt_idx, False), device, multimodal=False)
    else:
        opt = torch.optim.AdamW(params, lr=acfg.lr, weight_decay=0.0)
        model.train()
        # Freeze BN statistics during gradient adaptation: with a small prefix,
        # letting both weights and running stats move is unstable.
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

        # The Kendall log-variances encode how the tasks trade off across the
        # whole population; one patient's prefix cannot re-estimate that. Frozen
        # structurally rather than by omission from the optimiser, because the
        # criterion is shared across patients by `run_personalization` and
        # leaving it accumulating gradients would hand the caller a mutated
        # object whose state depends on patient order.
        crit_grad = [(p, p.requires_grad) for p in criterion.parameters()]
        for p, _ in crit_grad:
            p.requires_grad_(False)

        try:
            step, it = 0, iter(dl_adapt)
            while step < acfg.steps:
                try:
                    batch = next(it)
                    x, y_reg, y_soft, last = batch[0], batch[1], batch[2], batch[3]
                except StopIteration:
                    it = iter(dl_adapt)
                    continue
                x, y_reg, y_soft, last = (t.to(device) for t in (x, y_reg, y_soft, last))
                opt.zero_grad(set_to_none=True)
                bg_hat, logit = model(x, last)
                loss, _ = criterion(bg_hat, logit, y_reg, y_soft)

                # L2-SP: penalise drift from the generic solution. Applies to
                # LoRA too -- see `l2_sp_penalty` for why that is not a no-op.
                if acfg.l2_sp > 0:
                    loss = loss + acfg.l2_sp * l2_sp_penalty(
                        model, anchor, acfg.method, device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params if p.requires_grad], 1.0)
                opt.step()
                step += 1
        finally:
            for p, was in crit_grad:
                p.requires_grad_(was)

    after = score(model)
    return AdaptResult(
        patient=str(patient),
        before=before,
        after=after,
        n_adapt_windows=int(adapt_idx.size),
        n_trainable=int(sum(p.numel() for p in params)),
        method=acfg.method,
    )


def run_personalization(
    base_model: CausalCGMMT,
    ws: WindowSet,
    test_patients: list[str],
    scaler: CausalScaler,
    labeler: SoftLabeler,
    criterion: MultiTaskLoss,
    cfg: Config,
) -> list[AdaptResult]:
    """Adapt independently per patient and report paired before/after deltas.

    Paired comparison on the *same* evaluation windows is what licenses the
    causal claim about adaptation; comparing a personalised model against a
    generic model scored on a different slice would not.
    """
    out = []
    for pid in test_patients:
        r = adapt_patient(base_model, ws, pid, scaler, labeler, criterion, cfg)
        if r is None:
            continue
        out.append(r)
        print(
            f"[{cfg.adapt.method}] {pid}: "
            f"RMSE {r.before.rmse:.3f} -> {r.after.rmse:.3f} | "
            f"PR-AUC {r.before.pr_auc:.4f} -> {r.after.pr_auc:.4f} "
            f"({r.n_trainable} trainable params, {r.n_adapt_windows} windows)"
        )

    if out:
        d_rmse = np.mean([r.after.rmse - r.before.rmse for r in out])
        d_pr = np.nanmean([r.after.pr_auc - r.before.pr_auc for r in out])
        print(f"\nMean delta over {len(out)} patients: RMSE {d_rmse:+.3f}, PR-AUC {d_pr:+.4f}")
    return out


# --------------------------------------------------------------------------- #
# OhioT1DM: within-subject few-shot personalisation
# --------------------------------------------------------------------------- #
# The official Ohio protocol is temporal and WITHIN patient -- every subject
# appears in both `train/` and `test/`. That makes personalisation the natural
# question to ask of it, and it also makes the causal protocol load-bearing in a
# way it is not for the held-out-subject arm: the adaptation data and the
# evaluation data come from the same person's continuous record, so the only
# thing separating them is time.
@dataclass
class OhioAdaptResult:
    """Paired before/after for one subject, on IDENTICAL evaluation windows.

    `before` is the zero-shot population model and `after` the personalised
    one, both scored on `eval` rows of the same `OhioPrefixSplit`. Pairing is
    what licenses the causal claim: a personalised model scored on a different
    slice than the generic one is not a comparison, it is two numbers.

    `before` / `after` mirror the PRIMARY horizon so callers that expect a
    single `EvalReport` keep working; `*_by_horizon` carries every horizon,
    because a joint model must be reported per horizon (pooling a 30-minute
    RMSE with a 60-minute one hides the case where the shared trunk improved
    one by degrading the other).
    """
    patient: str
    method: str
    before: EvalReport
    after: EvalReport
    before_by_horizon: dict[int, EvalReport] = field(default_factory=dict)
    after_by_horizon: dict[int, EvalReport] = field(default_factory=dict)
    n_adapt_windows: int = 0
    n_eval_windows: int = 0
    n_embargoed_windows: int = 0
    n_eval_from_train: int = 0
    n_eval_from_test: int = 0
    # Which rows were scored, and how many of them the POPULATION model had
    # already fitted or selected on. Zero under the default "test" scope.
    eval_scope: str = "test"
    n_eval_in_fit: int = 0
    n_eval_in_val: int = 0
    n_eval_in_calib: int = 0
    n_trainable: int = 0
    prefix_hours: float = float("nan")
    embargo_min: int = 0
    # Where the prefix was drawn from ("opening" / "recent") and how far its end
    # sits from the first scored window. `prefix_gap_min` is the variable the
    # opening/recent contrast turns on, so it travels with the result rather
    # than being re-derived from timestamps by whoever reads the artefact.
    prefix_mode: str = "opening"
    prefix_gap_min: float = float("nan")
    n_adapt_in_fit: int = 0
    n_adapt_in_val: int = 0
    n_adapt_in_calib: int = 0
    # Row indices of the evaluation region, so a caller can re-score the adapted
    # model on a SUBSET of it (e.g. the official test file alone, for a table
    # whose other rows are scored there) without re-deriving the split.
    eval_idx: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    # The `OhioPrefixSplit` this result was actually produced under. Carried
    # rather than left to be re-derived, because re-deriving it needs FOUR
    # arguments to agree (`prefix_hours`, `embargo_min`, `eval_scope`,
    # `prefix_mode`) and a caller that passes only the first silently gets the
    # function's defaults for the rest. `benchmark.adapt_and_predict` builds its
    # conformal exclusion mask from `adapt | embargoed`, so a split that differs
    # from the one adapted on admits rows the model was fitted on -- shrinking
    # qhat in the undercovering direction, which is the failure this mask exists
    # to prevent.
    prefix_split: object | None = None
    # The adapted model itself, only when `keep_model=True`. Left out by default
    # because a sweep over 12 subjects x 5 modes x 4 prefix lengths would
    # otherwise hold 240 model copies alive at once.
    model: object | None = None

    def delta(self, field_name: str, horizon: int | None = None) -> float:
        """after - before for one metric. Negative is better for RMSE."""
        b, a = ((self.before, self.after) if horizon is None else
                (self.before_by_horizon[horizon], self.after_by_horizon[horizon]))
        return float(getattr(a, field_name)) - float(getattr(b, field_name))

    def __str__(self) -> str:
        return (
            f"[{self.method}/{self.prefix_mode}] {self.patient}: "
            f"RMSE {self.before.rmse:.3f} -> {self.after.rmse:.3f} | "
            f"PR-AUC {self.before.pr_auc:.4f} -> {self.after.pr_auc:.4f} | "
            f"FAPD@5% {self.before.fapd_at_5pct_fpr:.2f} -> "
            f"{self.after.fapd_at_5pct_fpr:.2f} "
            f"({self.n_trainable} trainable params, {self.n_adapt_windows} prefix / "
            f"{self.n_eval_windows} eval windows)"
        )


def _ohio_horizon_columns(ws, cfg: Config, model: nn.Module) -> tuple[list[int], list[int]]:
    """(horizons in minutes, their column indices) for the heads this model has.

    Derived from the model rather than from the config, because the head count
    is what the loss and the target tensors must agree with. A K=1 model built
    from a two-horizon artefact must be fed the PRIMARY horizon only; feeding it
    both would broadcast silently in the loss rather than raise.
    """
    available = [int(h) for h in ws.horizons_min]
    k_model = int(getattr(model, "n_horizons", 1))
    if k_model == len(available):
        return available, list(range(len(available)))
    if k_model == 1:
        primary = int(cfg.data.horizon_min)
        if primary not in available:
            raise ValueError(
                f"cfg.data.horizon_min={primary} is not among the windowed "
                f"horizons {available}.")
        return [primary], [available.index(primary)]
    raise ValueError(
        f"model has {k_model} horizon head(s) but the window set carries "
        f"{len(available)} horizons {available}; they must match, or the model "
        f"must be single-horizon.")


def _ohio_arrays(ws, pre, idx: np.ndarray, cols: list[int], unimodal: bool) -> dict:
    """Frozen-preprocessor tensors for a set of rows, at the chosen horizons.

    NOTHING is fitted here. `pre` was fitted on the population fit split and is
    applied unchanged -- refitting the scaler or the soft-label band on the new
    subject's prefix would import that subject's distribution into the
    transform, which is the personalisation equivalent of scaling on test data.
    """
    arrays = pre.transform(ws, idx)
    squeeze = len(cols) == 1
    for name in ("y_reg", "y_soft"):
        multi = np.ascontiguousarray(arrays[f"{name}_multi"][:, cols], dtype=np.float32)
        arrays[name] = multi[:, 0].copy() if squeeze else multi
    if unimodal:
        arrays["x_aux"] = arrays["x_aux"][:, :, :0]
    return arrays


def _is_multimodal(model: nn.Module) -> bool:
    return isinstance(model, MultiModalCausalCGMMT) or hasattr(model, "cgm_backbone")


def adapt_ohio_subject(
    base_model: nn.Module,
    ws,
    split,
    patient: str,
    pre,
    criterion: MultiTaskLoss,
    cfg: Config,
    method: str | None = None,
    prefix_hours: float | None = None,
    embargo_min: int | None = None,
    min_adapt_windows: int | None = None,
    min_eval_windows: int = 50,
    lora_rank: int | None = None,
    steps: int | None = None,
    lr: float | None = None,
    keep_model: bool = False,
    eval_scope: str = "test",
    prefix_mode: str = "opening",
) -> OhioAdaptResult | None:
    """Personalise a copy of the population model to one OhioT1DM subject.

    `prefix_mode` selects where the adaptation prefix is taken from -- the
    opening hours of the subject's training file, or the most recent hours that
    still clear the embargo. See `ohio_prefix_embargo_split`.

    Returns None when the subject cannot support a defensible claim either way
    -- too short a prefix to adapt on, or too few evaluation windows to measure
    the result. Returning None rather than a noisy number is deliberate: a
    per-subject delta computed on 12 windows would enter the cohort mean with
    the same weight as one computed on 3000.
    """
    from .ohio_loader import assert_prefix_embargo_causal, ohio_prefix_embargo_split

    method = str(method or cfg.adapt.method)
    if method not in ADAPT_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {ADAPT_METHODS}")

    set_seed(cfg.train.seed)
    device = pick_device()

    ps = ohio_prefix_embargo_split(
        ws, split, patient, cfg.data,
        prefix_hours=(cfg.adapt.adapt_prefix_hours if prefix_hours is None
                      else prefix_hours),
        embargo_min=cfg.adapt.embargo_min if embargo_min is None else embargo_min,
        eval_scope=eval_scope,
        prefix_mode=prefix_mode,
    )
    if ps is None:
        return None
    # Verified on every run, not only under test: a personalisation result whose
    # prefix leaked into its evaluation region is meaningless, not merely weak.
    assert_prefix_embargo_causal(ws, ps, cfg.data)

    need_adapt = (cfg.adapt.batch_size if min_adapt_windows is None
                  else int(min_adapt_windows))
    if ps.adapt.size < need_adapt or ps.eval.size < min_eval_windows:
        return None

    multimodal = _is_multimodal(base_model)
    horizons, cols = _ohio_horizon_columns(ws, cfg, base_model)
    period = int(cfg.data.sample_period_min)

    from .ohio_loader import make_dataset

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = make_dataset(_ohio_arrays(ws, pre, idx, cols, not multimodal),
                          horizons, style="compat")
        return DataLoader(ds, batch_size=cfg.adapt.batch_size, shuffle=shuffle,
                          drop_last=False)

    dl_adapt_shuffled = loader(ps.adapt, True)
    dl_adapt_ordered = loader(ps.adapt, False)
    dl_eval = loader(ps.eval, False)

    ev = ps.eval

    def score(m: nn.Module) -> dict[int, EvalReport]:
        bg_hat, risk = predict(m, dl_eval, device, multimodal)
        out = {}
        for i, h in enumerate(horizons):
            c = cols[i]
            bg_i = bg_hat if bg_hat.ndim == 1 else bg_hat[:, i]
            s_i = risk if risk.ndim == 1 else risk[:, i]
            out[h] = evaluate(
                ws.y_reg_multi[ev, c], bg_i, ws.y_cls_multi[ev, c], s_i,
                ws.last_bg[ev], ws.patient[ev], period,
                t_time=ws.t_time[ev], horizon_min=h,
            )
        return out

    model = copy.deepcopy(base_model).to(device)
    criterion = criterion.to(device)
    before = score(model)

    if method == "none":
        # The zero-shot arm. Scored once and reported twice on purpose: the
        # "after" of a no-op must be bit-identical to its "before", which makes
        # this arm a live check that the harness itself adds no drift.
        return OhioAdaptResult(
            patient=ps.patient, method=method,
            before=before[horizons[0]], after=before[horizons[0]],
            before_by_horizon=before, after_by_horizon=dict(before),
            n_adapt_windows=int(ps.adapt.size), n_eval_windows=int(ps.eval.size),
            n_embargoed_windows=int(ps.embargoed.size),
            n_eval_from_train=ps.n_eval_from_train,
            n_eval_from_test=ps.n_eval_from_test,
            eval_scope=ps.eval_scope, n_eval_in_fit=ps.n_eval_in_fit,
            n_eval_in_val=ps.n_eval_in_val, n_eval_in_calib=ps.n_eval_in_calib,
            n_trainable=0, prefix_hours=ps.prefix_hours, embargo_min=ps.embargo_min,
            prefix_mode=ps.prefix_mode, prefix_gap_min=ps.prefix_gap_min,
            n_adapt_in_fit=ps.n_adapt_in_fit, n_adapt_in_val=ps.n_adapt_in_val,
            n_adapt_in_calib=ps.n_adapt_in_calib,
            eval_idx=ev, prefix_split=ps, model=model if keep_model else None,
        )

    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    acfg = dataclasses.replace(
        cfg.adapt, method=method,
        **({} if lora_rank is None else {"lora_rank": int(lora_rank)}),
        **({} if steps is None else {"steps": int(steps)}),
        **({} if lr is None else {"lr": float(lr)}),
    )
    params = configure_trainable(model, acfg)
    model = model.to(device)

    if method == "bn_adapt":
        recalibrate_bn(model, dl_adapt_ordered, device, multimodal=multimodal)
    else:
        # `acfg`, not `cfg.adapt`: the lr/steps/rank overrides passed to this
        # function are folded into `acfg` above, and reading the un-overridden
        # config here silently discarded the `lr=` argument.
        opt = torch.optim.AdamW(params, lr=acfg.lr, weight_decay=0.0)
        model.train()
        # Freeze BN statistics during gradient adaptation: with a short prefix,
        # letting both the weights and the running stats move is unstable.
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

        # The Kendall loss's learned log-variances are POPULATION quantities:
        # they encode how the regression and forewarning tasks trade off across
        # the whole cohort, and one subject's day of data cannot support
        # re-estimating that. Frozen structurally rather than by omission from
        # the optimiser -- the criterion is shared across subjects, so leaving
        # it accumulating gradients would hand the caller a mutated object.
        crit_grad = [(p, p.requires_grad) for p in criterion.parameters()]
        for p, _ in crit_grad:
            p.requires_grad_(False)

        try:
            step, it = 0, iter(dl_adapt_shuffled)
            while step < acfg.steps:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dl_adapt_shuffled)
                    continue
                opt.zero_grad(set_to_none=True)
                bg_hat, logit, y_reg, y_soft = _forward(model, batch, device, multimodal)
                loss, _ = criterion(bg_hat, logit, y_reg, y_soft)
                if acfg.l2_sp > 0:
                    loss = loss + acfg.l2_sp * l2_sp_penalty(
                        model, anchor, method, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 1.0)
                opt.step()
                step += 1
        finally:
            for p, was in crit_grad:
                p.requires_grad_(was)

    after = score(model)
    return OhioAdaptResult(
        patient=ps.patient, method=method,
        before=before[horizons[0]], after=after[horizons[0]],
        before_by_horizon=before, after_by_horizon=after,
        n_adapt_windows=int(ps.adapt.size), n_eval_windows=int(ps.eval.size),
        n_embargoed_windows=int(ps.embargoed.size),
        n_eval_from_train=ps.n_eval_from_train,
        n_eval_from_test=ps.n_eval_from_test,
        eval_scope=ps.eval_scope, n_eval_in_fit=ps.n_eval_in_fit,
        n_eval_in_val=ps.n_eval_in_val, n_eval_in_calib=ps.n_eval_in_calib,
        n_trainable=int(sum(p.numel() for p in params)),
        prefix_hours=ps.prefix_hours, embargo_min=ps.embargo_min,
        prefix_mode=ps.prefix_mode, prefix_gap_min=ps.prefix_gap_min,
        n_adapt_in_fit=ps.n_adapt_in_fit, n_adapt_in_val=ps.n_adapt_in_val,
        n_adapt_in_calib=ps.n_adapt_in_calib,
        eval_idx=ev, prefix_split=ps, model=model if keep_model else None,
    )


def run_ohio_personalization(
    base_model: nn.Module,
    ws,
    split,
    pre,
    cfg: Config,
    criterion: MultiTaskLoss | None = None,
    subjects: Sequence[str] | None = None,
    method: str | None = None,
    prefix_hours: float | None = None,
    embargo_min: int | None = None,
    min_adapt_windows: int | None = None,
    min_eval_windows: int = 50,
    lora_rank: int | None = None,
    steps: int | None = None,
    lr: float | None = None,
    keep_model: bool = False,
    eval_scope: str = "test",
    prefix_mode: str = "opening",
    verbose: bool = True,
) -> list[OhioAdaptResult]:
    """Few-shot personalisation across OhioT1DM subjects under the causal protocol.

        |<- adapt prefix (K h of TRAIN) ->|<- embargo ->|<- evaluate ->|

    For each subject the prefix is `cfg.adapt.adapt_prefix_hours` of their
    official TRAINING file, the embargo is at least (history + horizon), and the
    evaluation region is the official test file -- unresampled, so the numbers
    stay comparable to the published benchmark.

    `prefix_mode` selects which K hours: "opening" takes the start of the
    training file (the just-fitted-device story) and "recent" takes the last K
    hours that clear the embargo (the adapt-on-yesterday story). The two are the
    controlled contrast for the drift explanation of a negative personalisation
    result -- same subjects, same evaluation windows, same adapter, differing
    only in how far the prefix sits from the windows being scored.

    `criterion` defaults to a `MultiTaskLoss` built from `cfg.model` and the
    population class prior recorded in `pre.pos_weight`. Passing the criterion
    the model was TRAINED with is preferable when it is available: its learned
    task weights are part of the population solution, and rebuilding it resets
    them to the uniform initialisation.

    Returns one `OhioAdaptResult` per subject that had enough data, each a
    paired before/after on identical evaluation windows.
    """
    mode = str(method or cfg.adapt.method)
    if subjects is None:
        subjects = sorted({str(p) for p in ws.patient})

    if criterion is None:
        primary = int(cfg.data.horizon_min)
        pw = float(getattr(pre, "pos_weight", {}).get(primary, 1.0))
        criterion = MultiTaskLoss(cfg.model, pw)

    out: list[OhioAdaptResult] = []
    for pid in subjects:
        r = adapt_ohio_subject(
            base_model, ws, split, pid, pre, criterion, cfg, method=mode,
            prefix_hours=prefix_hours, embargo_min=embargo_min,
            min_adapt_windows=min_adapt_windows, min_eval_windows=min_eval_windows,
            lora_rank=lora_rank, steps=steps, lr=lr, keep_model=keep_model,
            eval_scope=eval_scope, prefix_mode=prefix_mode,
        )
        if r is None:
            if verbose:
                print(f"[{mode}/{prefix_mode}] {pid}: skipped (insufficient "
                      f"prefix or evaluation windows)")
            continue
        out.append(r)
        if verbose:
            print(str(r))

    if out and verbose:
        d_rmse = float(np.mean([r.delta("rmse") for r in out]))
        d_pr = float(np.nanmean([r.delta("pr_auc") for r in out]))
        d_fapd = float(np.nanmean([r.delta("fapd_at_5pct_fpr") for r in out]))
        print(f"\nMean delta over {len(out)} subject(s) [{mode}/{prefix_mode}]: "
              f"RMSE {d_rmse:+.3f}, PR-AUC {d_pr:+.4f}, "
              f"FAPD@5%FPR {d_fapd:+.2f}/day")
    return out
