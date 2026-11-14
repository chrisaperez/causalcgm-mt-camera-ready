"""Audit of the asymmetric hypoglycemia cost, the multi-horizon trunk, and the
Parkes backend delegation.

Each test names the specific failure it guards against. Three of these
correspond to bugs found while building this: a weighted mean that cancelled the
asymmetry on homogeneous batches, a zone-index normalisation that shifted every
Parkes zone by one, and a verification anchor that was itself wrong.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.clinical import (
    clinical_report, parkes_available, parkes_zones, verify_parkes_backend,
)
from causalcgm_mt.config import ModelConfig
from causalcgm_mt.models import (
    MultiTaskLoss, asymmetric_huber, build_model, build_multimodal_model,
    hypo_cost_weights,
)

AUX_C = 24


# --------------------------------------------------------------------------- #
# 1. Asymmetric hypoglycemia cost
# --------------------------------------------------------------------------- #
def test_penalises_understating_lows_not_overstating_them():
    """The measured defect was +8.8 mg/dL bias below 70: the model reads lows as
    HIGHER than they are, which is the road to Clarke zone D. That direction --
    and only that direction -- must be up-weighted."""
    y = torch.tensor([55.0, 55.0])
    p = torch.tensor([70.0, 40.0])          # understates the low, overstates it
    w = hypo_cost_weights(p, y, kappa=3.0, threshold=70.0, tau=10.0)
    assert w[0] > 2.0, "understating a low must be penalised"
    assert w[1] == pytest.approx(1.0), "overstating a low must NOT be penalised"


def test_cost_is_confined_to_the_hypoglycemic_ramp():
    """Uniform asymmetry would shift the whole prediction curve down, buying
    hypo sensitivity with euglycemic and hyperglycemic accuracy."""
    w_hypo = hypo_cost_weights(torch.tensor([70.0]), torch.tensor([50.0]), 3.0, 70.0, 10.0)
    w_eu = hypo_cost_weights(torch.tensor([160.0]), torch.tensor([140.0]), 3.0, 70.0, 10.0)
    assert w_hypo.item() > 2.5
    assert w_eu.item() < 1.05


def test_ramp_is_smooth_not_a_cliff_at_70():
    """A hard `y < 70` indicator puts a discontinuity exactly where CGM noise
    straddles the boundary, so the network fits device noise there."""
    ys = torch.tensor([50.0, 65.0, 70.0, 75.0, 90.0])
    w = hypo_cost_weights(ys + 20.0, ys, 3.0, 70.0, 10.0)
    assert torch.all(w[:-1] >= w[1:]), "weight must decrease monotonically with y"
    assert w[2].item() == pytest.approx(2.0, abs=0.05), "half strength at threshold"
    assert (w.diff().abs() < 1.0).all(), "no cliff"


def test_kappa_one_is_an_exact_no_op():
    """The safety mechanism must be exactly disabled at its off-setting, not
    approximately -- otherwise the ablation's control arm is not a control."""
    y = torch.rand(64) * 300 + 40
    p = y + torch.randn(64) * 20
    w = hypo_cost_weights(p, y, kappa=1.0, threshold=70.0, tau=10.0)
    assert (w == 1.0).all()
    plain = torch.nn.functional.huber_loss(p, y, delta=10.0)
    assert asymmetric_huber(p, y, 10.0, w).item() == pytest.approx(plain.item(), rel=1e-6)


def test_weighting_shifts_gradient_share_toward_hypo_samples():
    """The weighted MEAN means kappa acts RELATIVELY. On a homogeneous batch it
    cancels entirely -- which made an early single-sample check falsely suggest
    the mechanism was inert. Test it the way it is actually used: mixed batch."""
    y = torch.tensor([55.0, 120.0, 160.0, 100.0])
    p0 = torch.tensor([70.0, 130.0, 150.0, 112.0])
    cfg = ModelConfig(n_horizons=1, loss_weighting="fixed")

    def hypo_share(kappa):
        crit = MultiTaskLoss(dataclasses.replace(cfg, hypo_asymmetry=kappa), 1.0)
        p = p0.clone().requires_grad_(True)
        loss, _ = crit(p, torch.zeros(4), y, torch.zeros(4))
        loss.backward()
        return (p.grad[0] / p.grad.abs().sum()).item()

    assert hypo_share(5.0) > hypo_share(1.0) * 1.5


def test_loss_scale_is_stable_across_kappa():
    """Raising kappa must not inflate the total loss, or it silently rescales
    the learned homoscedastic task weights -- two things moving when the
    operator turned one knob."""
    y = torch.rand(256) * 250 + 45
    p = y + torch.randn(256) * 15
    cfg = ModelConfig(n_horizons=1, loss_weighting="fixed")
    vals = []
    for kappa in (1.0, 3.0, 10.0):
        crit = MultiTaskLoss(dataclasses.replace(cfg, hypo_asymmetry=kappa), 1.0)
        vals.append(crit(p, torch.zeros(256), y, torch.zeros(256))[0].item())
    assert max(vals) / min(vals) < 1.6, f"loss magnitude drifted with kappa: {vals}"


# --------------------------------------------------------------------------- #
# 2. Multi-horizon shared trunk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", ["uni", "mm"])
def test_single_horizon_shapes_are_unchanged(build):
    """K=1 must be a byte-for-byte drop-in, or every existing call site,
    checkpoint and test silently changes meaning."""
    cfg = ModelConfig(n_horizons=1)
    if build == "uni":
        m, crit = build_model(5, cfg, 1.0)
        bg, lg = m(torch.randn(8, 24, 5), torch.full((8,), 120.0))
    else:
        m, crit = build_multimodal_model(5, AUX_C, cfg, 1.0, aux_valid_index=7)
        bg, lg = m(torch.randn(8, 24, 5), torch.randn(8, 24, AUX_C),
                   torch.full((8,), 120.0), torch.ones(8))
    assert bg.shape == (8,) and lg.shape == (8,)
    loss, _ = crit(bg, lg, torch.full((8,), 100.0), torch.zeros(8))
    assert loss.dim() == 0


def test_multi_horizon_emits_one_column_per_horizon():
    cfg = ModelConfig(n_horizons=3)
    m, crit = build_multimodal_model(5, AUX_C, cfg, 1.0, aux_valid_index=7)
    bg, lg = m(torch.randn(8, 24, 5), torch.randn(8, 24, AUX_C),
               torch.full((8,), 120.0), torch.ones(8))
    assert bg.shape == (8, 3) and lg.shape == (8, 3)
    loss, stats = crit(bg, lg, torch.full((8, 3), 100.0), torch.zeros(8, 3))
    assert loss.dim() == 0
    assert {"loss_reg_h0", "loss_reg_h2", "w_cls_h1"} <= set(stats)


def test_heads_are_independent_but_the_trunk_is_shared():
    """The point of the refactor: one trunk, K readouts. If the heads were tied
    the horizons would be forced to identical predictions; if the trunk were
    duplicated there would be no sharing to gain from."""
    cfg = ModelConfig(n_horizons=2)
    m, _ = build_multimodal_model(5, AUX_C, cfg, 1.0, aux_valid_index=7)
    assert len(m.reg_heads) == 2 and len(m.cls_heads) == 2
    p0 = dict(m.reg_heads[0].named_parameters())
    p1 = dict(m.reg_heads[1].named_parameters())
    assert not torch.equal(p0["net.0.weight"], p1["net.0.weight"])
    # exactly one CGM trunk and one aux trunk, regardless of K
    assert sum(1 for n, _ in m.named_children() if n.endswith("backbone")) == 2


def test_kendall_weights_are_per_horizon():
    """60-minute BG is far less predictable than 30-minute. A shared weight lets
    the easier horizon dominate and the harder one collapse toward the mean."""
    crit = MultiTaskLoss(ModelConfig(n_horizons=2), 1.0)
    assert crit.log_var_reg.shape == (2,)
    assert crit.log_var_cls.shape == (2,)


def test_horizon_count_mismatch_raises_instead_of_broadcasting():
    """A (B,) target against a (B,K) prediction would broadcast into K copies of
    the wrong loss and train happily."""
    crit = MultiTaskLoss(ModelConfig(n_horizons=2), 1.0)
    with pytest.raises(ValueError, match="horizon count mismatch"):
        crit(torch.randn(8, 2), torch.randn(8, 2), torch.randn(8), torch.zeros(8, 2))


def test_loss_rejects_a_model_configured_for_a_different_k():
    crit = MultiTaskLoss(ModelConfig(n_horizons=3), 1.0)
    with pytest.raises(ValueError, match="configured for 3 horizon"):
        crit(torch.randn(8, 2), torch.randn(8, 2), torch.randn(8, 2), torch.zeros(8, 2))


def test_single_horizon_aliases_still_resolve():
    """`extension.configure_trainable` selects `model.reg_head` for head-only
    adaptation. Without the alias it would select nothing and adapt an empty
    parameter list -- a silent no-op, not an error."""
    m, _ = build_multimodal_model(5, AUX_C, ModelConfig(n_horizons=2), 1.0,
                                  aux_valid_index=7)
    assert m.reg_head is m.reg_heads[0]
    assert m.cls_head is m.cls_heads[0]


def test_every_horizon_head_receives_gradient():
    """A head that never gets gradient is a horizon that is not being trained."""
    cfg = ModelConfig(n_horizons=2, loss_weighting="fixed")
    m, crit = build_multimodal_model(5, AUX_C, cfg, 1.0, aux_valid_index=7)
    bg, lg = m(torch.randn(4, 24, 5), torch.randn(4, 24, AUX_C),
               torch.full((4,), 120.0), torch.ones(4))
    loss, _ = crit(bg, lg, torch.full((4, 2), 90.0), torch.zeros(4, 2))
    loss.backward()
    for k in range(2):
        g = m.reg_heads[k].net[0].weight.grad
        assert g is not None and g.abs().sum() > 0, f"reg head {k} got no gradient"


# --------------------------------------------------------------------------- #
# 3. Parkes backend delegation
# --------------------------------------------------------------------------- #
needs_parkes = pytest.mark.skipif(not parkes_available(),
                                  reason="no Parkes backend installed")


def test_missing_backend_raises_with_instructions_not_a_guess():
    if parkes_available():
        pytest.skip("backend installed")
    with pytest.raises(ImportError, match="pip install"):
        parkes_zones(np.array([100.0]), np.array([100.0]))


@needs_parkes
def test_backend_passes_verification():
    info = verify_parkes_backend()
    assert info["backend"]


@needs_parkes
def test_zone_labels_are_read_not_inferred():
    """The numeric convention (0- vs 1-based) is undocumented. Inferring it from
    the observed range shifted every zone by one whenever a batch contained no
    zone-A points -- which made a correct backend look broken."""
    exact = np.array([40.0, 100.0, 300.0])
    assert (parkes_zones(exact, exact) == 0).all()
    # A batch with no zone-A points must not be re-indexed.
    z = parkes_zones(np.array([100.0, 100.0]), np.array([260.0, 300.0]))
    assert (z > 0).all()


@needs_parkes
def test_grids_agree_grossly_and_are_reported_together():
    rng = np.random.default_rng(1)
    y = rng.uniform(45, 350, 2000)
    p = y + rng.normal(0, 18, 2000)
    rep = clinical_report(y, p)
    assert rep.parkes_ab_pct is not None
    assert abs(rep.parkes_ab_pct - rep.zone_ab_pct) < 8.0
    assert "Parkes" in str(rep)


def test_report_degrades_to_clarke_only_without_a_backend():
    """An absent backend must not fail the run, and must not silently report an
    unvalidated number either."""
    rng = np.random.default_rng(2)
    y = rng.uniform(45, 350, 500)
    rep = clinical_report(y, y + rng.normal(0, 15, 500), with_parkes=False)
    assert rep.parkes_ab_pct is None
    assert "Parkes" not in str(rep)
    assert rep.zone_ab_pct > 0
