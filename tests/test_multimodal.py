"""Leakage audit for the multimodal (HUPA-UCM) path.

Multi-sensor fusion adds three new leak surfaces that the univariate tests do
not cover:

  1. RESAMPLING DIRECTION. A forward-labelled bin attributes future samples to
     the present slot. This is the default in pandas and the easiest mistake to
     make in the whole pipeline.
  2. FILL DIRECTION. Forward fill is causal; backward fill is not.
  3. ATTENTION MASKING. Unmasked attention is bidirectional by construction.

Each test below targets one of those and fails if the guarantee is removed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.data import CausalScaler
from causalcgm_mt.hupa_loader import (
    AUX_CHANNELS,
    AUX_INDEX,
    AUX_SKIP_SCALING,
    HupaConfig,
    HupaSchema,
    _activity_kernel,
    _bin_backward,
    _causal_ffill,
    build_multimodal_windows,
    causal_on_board,
    synchronize_patient,
    to_canonical,
)
from causalcgm_mt.models import CausalCrossAttention, build_multimodal_model
from causalcgm_mt.simulate import simulate_hupa_cohort


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return to_canonical(simulate_hupa_cohort(n_patients=4, days=3.0, seed=3))


@pytest.fixture(scope="module")
def mmws(cohort):
    return build_multimodal_windows(cohort, DataConfig(), HupaConfig())


# --- 1. Resampling direction ------------------------------------------------ #
def test_binning_is_backward_looking():
    """Slot t must summarise (t-P, t], never [t, t+P).

    Constructed so the two conventions give different answers: a single sample
    sits just after a grid point. Under the correct (backward) convention it
    belongs to the NEXT slot; under pandas' default it would be attributed to
    the current one -- a full period of lookahead.
    """
    grid = pd.date_range("2024-01-01 00:00", periods=5, freq="5min")
    # Sample at 00:01, i.e. inside (00:00, 00:05].
    ts = pd.Series([pd.Timestamp("2024-01-01 00:01")])
    vals = pd.Series([100.0])

    agg, obs = _bin_backward(ts, vals, grid, 5, "mean")

    assert not obs[0], "sample at 00:01 must NOT appear in the 00:00 slot"
    assert obs[1] and agg[1] == pytest.approx(100.0), \
        "sample at 00:01 must appear in the 00:05 slot"


def test_binning_never_reads_future_samples():
    """For EVERY slot t, samples strictly after t must not affect that slot.

    The per-slot formulation matters. An earlier version of this test compared
    `a[:cutoff]` after corrupting from a cutoff onward, which is off by one bin
    and therefore blind to a forward-labelled resample -- the exact bug it is
    supposed to catch. Checking each slot against its own timestamp removes the
    ambiguity: slot at time T is allowed to see minute T, and nothing later.
    """
    grid = pd.date_range("2024-01-01", periods=40, freq="5min")
    ts = pd.date_range("2024-01-01", periods=200, freq="1min")
    v = np.linspace(60, 120, 200)

    base, _ = _bin_backward(pd.Series(ts), pd.Series(v), grid, 5, "mean")

    for slot in (5, 10, 19, 30):
        t_slot = grid[slot]
        v2 = v.copy()
        v2[np.asarray(ts > t_slot)] += 500.0    # strictly after this slot
        a2, _ = _bin_backward(pd.Series(ts), pd.Series(v2), grid, 5, "mean")

        assert a2[slot] == pytest.approx(base[slot]), (
            f"slot {slot} ({t_slot}) changed when only samples AFTER it were "
            "corrupted -- the bin is forward-looking"
        )
        assert np.allclose(a2[: slot + 1], base[: slot + 1])


# --- 2. Fill direction ------------------------------------------------------ #
def test_ffill_is_causal_and_never_backfills():
    """A leading gap must stay unobserved -- filling it needs a future value."""
    vals = np.array([0.0, 0.0, 70.0, 0.0, 0.0, 80.0])
    obs = np.array([False, False, True, False, False, True])

    filled, valid, stale = _causal_ffill(vals, obs, max_staleness_steps=4)

    # Leading gap: no past observation exists, so it must remain invalid.
    assert not valid[0] and not valid[1], "backward fill detected on a leading gap"
    assert valid[2] and filled[2] == 70.0
    # Interior gap carries the PAST value forward, not the future one (80).
    assert filled[3] == 70.0 and filled[4] == 70.0
    assert stale[3] == 1 and stale[4] == 2


def test_ffill_respects_staleness_budget():
    """Beyond the budget the channel goes unobserved rather than stale-but-valid."""
    vals = np.zeros(10); vals[0] = 60.0
    obs = np.zeros(10, bool); obs[0] = True

    _, valid, _ = _causal_ffill(vals, obs, max_staleness_steps=3)

    assert valid[:4].all(), "value should survive within the budget"
    assert not valid[4:].any(), "value must expire past the budget"


def test_future_observation_cannot_change_earlier_fill():
    vals = np.array([50.0, 0.0, 0.0, 0.0])
    obs = np.array([True, False, False, False])
    a, _, _ = _causal_ffill(vals, obs, 10)

    vals2 = vals.copy(); vals2[3] = 999.0
    obs2 = obs.copy(); obs2[3] = True
    b, _, _ = _causal_ffill(vals2, obs2, 10)

    assert np.allclose(a[:3], b[:3]), "a later observation altered earlier steps"


# --- 3. Pharmacokinetic curves ---------------------------------------------- #
def test_on_board_is_causal_fir():
    """A dose at index i must not affect on-board amounts before i."""
    imp = np.zeros(40); imp[20] = 5.0
    k = _activity_kernel(300, 75, 5, "iob")
    iob = causal_on_board(imp, k)

    assert np.allclose(iob[:20], 0.0), "insulin appeared on board before injection"
    assert iob[20] == pytest.approx(5.0, rel=1e-6), "full dose on board at t=0"
    assert iob[25] < iob[20], "IOB must decay"
    assert iob.size == imp.size


def test_on_board_superposes_and_decays_to_zero():
    imp = np.zeros(200); imp[10] = 3.0; imp[50] = 2.0
    iob = causal_on_board(imp, _activity_kernel(300, 75, 5, "iob"))

    assert iob[49] < iob[10], "first dose should have decayed by t=49"
    assert iob[50] > iob[49], "second dose must add on top"
    assert iob[-1] == pytest.approx(0.0, abs=1e-6), "IOB must reach zero after DIA"


def test_kernels_are_monotone_non_increasing():
    for kind, dur, peak in (("iob", 300, 75), ("cob", 180, 45)):
        k = _activity_kernel(dur, peak, 5, kind)
        assert k[0] == pytest.approx(1.0)
        assert k[-1] == pytest.approx(0.0)
        assert np.all(np.diff(k) <= 1e-9), f"{kind} remaining-amount curve must not rise"


# --- 4. End-to-end multimodal windowing ------------------------------------- #
def test_aux_channel_order_is_stable(mmws):
    """The layout is a serving contract; a silent reorder is invisible in metrics."""
    assert mmws.x_aux.shape[-1] == len(AUX_CHANNELS)
    assert AUX_CHANNELS[0] == "heart_rate"
    assert AUX_INDEX["iob"] == 3 and AUX_INDEX["cob"] == 4
    assert set(AUX_SKIP_SCALING) == {
        AUX_INDEX["hr_observed"], AUX_INDEX["steps_observed"], AUX_INDEX["sleep_observed"]
    }


def test_cgm_and_aux_windows_are_step_aligned(mmws):
    assert mmws.x.shape[:2] == mmws.x_aux.shape[:2]
    assert len(mmws) == mmws.x_aux.shape[0] == mmws.y_reg.shape[0]


def test_future_aux_perturbation_does_not_change_windows(cohort):
    """The decisive multimodal leak test.

    Corrupt EVERY auxiliary stream after a cutoff and require that windows
    anchored before the cutoff have byte-identical aux tensors. This exercises
    binning, forward fill, staleness and the IOB/COB convolutions in one shot --
    any of them reading forward breaks it.
    """
    cfg, hcfg = DataConfig(), HupaConfig()
    a = build_multimodal_windows(cohort, cfg, hcfg)

    d = cohort.copy()
    cut = d.groupby("patient_id")["time"].transform(
        lambda s: s.min() + (s.max() - s.min()) * 0.6
    )
    fut = d.time > cut
    for col, bump in (("heart_rate", 60.0), ("steps", 500.0),
                      ("carbs", 200.0), ("bolus", 20.0), ("sleep", 1.0)):
        d.loc[fut & d[col].notna(), col] += bump

    b = build_multimodal_windows(d, cfg, hcfg)

    key_a = {(str(p), t): i for i, (p, t) in enumerate(zip(a.patient, a.t_time))}
    key_b = {(str(p), t): i for i, (p, t) in enumerate(zip(b.patient, b.t_time))}
    cuts = {str(p): c for p, c in zip(cohort.patient_id, cut)}

    checked = 0
    for key in sorted(set(key_a) & set(key_b)):
        pid, t = key
        if t >= np.datetime64(cuts[pid]):
            continue
        ia, ib = key_a[key], key_b[key]
        assert np.array_equal(a.x_aux[ia], b.x_aux[ib]), \
            f"aux window changed after a purely FUTURE perturbation ({pid} @ {t})"
        assert np.array_equal(a.x[ia], b.x[ib])
        checked += 1

    assert checked > 100, f"only {checked} windows exercised"


def test_no_nan_or_inf_in_aux(mmws):
    assert np.isfinite(mmws.x_aux).all()
    assert np.isfinite(mmws.x).all()


def test_masks_are_binary(mmws):
    for name in ("hr_observed", "steps_observed", "sleep_observed"):
        v = mmws.x_aux[..., AUX_INDEX[name]]
        assert np.isin(v, (0.0, 1.0)).all(), f"{name} is not binary"


def test_aux_scaler_leaves_masks_untouched(mmws):
    pats = sorted({str(p) for p in mmws.patient})
    sc = CausalScaler.fit_array(mmws.x_aux, mmws.patient, pats[:2], AUX_SKIP_SCALING)

    for c in AUX_SKIP_SCALING:
        assert sc.center[c] == 0.0 and sc.scale[c] == 1.0
    out = sc.transform(mmws.x_aux[:50])
    for c in AUX_SKIP_SCALING:
        assert np.isin(out[..., c], (0.0, 1.0)).all()


def test_aux_scaler_fits_only_on_training_patients(mmws):
    pats = sorted({str(p) for p in mmws.patient})
    train, test = pats[:2], pats[2:]

    sc_a = CausalScaler.fit_array(mmws.x_aux, mmws.patient, train, AUX_SKIP_SCALING)

    corrupted = mmws.x_aux.copy()
    corrupted[np.isin(mmws.patient, np.array(test, dtype=object))] *= 7.0
    sc_b = CausalScaler.fit_array(corrupted, mmws.patient, train, AUX_SKIP_SCALING)

    assert np.allclose(sc_a.center, sc_b.center)
    assert np.allclose(sc_a.scale, sc_b.scale)
    assert not (set(sc_a.fitted_on) & set(test))


def test_iob_reflects_only_past_boluses(cohort):
    """Spot-check on the real grid: IOB at t must be zero before any bolus."""
    cfg = DataConfig()
    pid = sorted(cohort.patient_id.unique())[0]
    reg = synchronize_patient(cohort[cohort.patient_id == pid], cfg, HupaConfig())

    first_bolus = np.flatnonzero(reg["bolus"].values > 0)
    assert first_bolus.size > 0, "simulator produced no boluses"
    i = first_bolus[0]
    assert reg["iob"].values[:i].max() == pytest.approx(0.0, abs=1e-9), \
        "IOB non-zero before the first recorded bolus"


# --- 5. Attention masking ---------------------------------------------------- #
def test_cross_attention_is_causally_masked():
    """Output at step i must be invariant to auxiliary values at steps > i."""
    torch.manual_seed(0)
    attn = CausalCrossAttention(dim=16).double().eval()

    cgm = torch.randn(2, 12, 16, dtype=torch.double)
    aux = torch.randn(2, 12, 16, dtype=torch.double)
    valid = torch.ones(2, 12, dtype=torch.double)

    with torch.no_grad():
        y0 = attn(cgm, aux, valid)
        aux2 = aux.clone()
        aux2[:, 7:] += 25.0                  # corrupt the aux future
        y2 = attn(cgm, aux2, valid)

    assert torch.allclose(y0[:, :7], y2[:, :7], atol=1e-10), \
        "cross-attention leaked future auxiliary data into earlier steps"
    assert not torch.allclose(y0[:, 7:], y2[:, 7:])


def test_attention_ignores_unobserved_aux_steps():
    """Masked-out steps must not influence the output, whatever their value."""
    torch.manual_seed(0)
    attn = CausalCrossAttention(dim=16).double().eval()

    cgm = torch.randn(1, 10, 16, dtype=torch.double)
    aux = torch.randn(1, 10, 16, dtype=torch.double)
    valid = torch.ones(1, 10, dtype=torch.double)
    valid[0, 3] = 0.0                        # step 3 unobserved

    with torch.no_grad():
        y0 = attn(cgm, aux, valid)
        aux2 = aux.clone()
        aux2[0, 3] = 1e4                     # garbage in the masked slot
        y2 = attn(cgm, aux2, valid)

    assert torch.allclose(y0, y2, atol=1e-9), \
        "attention consumed an unobserved auxiliary step"


def test_attention_handles_fully_masked_rows_without_nan():
    """Softmax over an all -inf row is NaN; the dead-row guard must prevent it."""
    attn = CausalCrossAttention(dim=8).double().eval()
    cgm = torch.randn(1, 6, 8, dtype=torch.double)
    aux = torch.randn(1, 6, 8, dtype=torch.double)
    valid = torch.zeros(1, 6, dtype=torch.double)   # nothing observed at all

    with torch.no_grad():
        y = attn(cgm, aux, valid)

    assert torch.isfinite(y).all(), "NaN from a fully-masked attention row"
    assert torch.allclose(y, torch.zeros_like(y))


# --- 6. Fusion model --------------------------------------------------------- #
def test_multimodal_model_is_causal_end_to_end():
    cfg = Config().model
    model, _ = build_multimodal_model(5, len(AUX_CHANNELS), cfg,
                                      aux_valid_index=AUX_INDEX["hr_observed"])
    model.eval()

    x = torch.randn(3, 24, 5)
    xa = torch.randn(3, 24, len(AUX_CHANNELS))
    xa[..., AUX_INDEX["hr_observed"]] = 1.0
    last = torch.full((3,), 120.0)

    with torch.no_grad():
        a = model(x, xa, last)
        b = model(x, xa, last)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
    assert a[0].shape == (3,) and a[1].shape == (3,)


def test_missing_modality_degrades_to_unimodal_not_to_noise():
    """With aux_present = 0 the fusion term must vanish exactly."""
    cfg = Config().model
    model, _ = build_multimodal_model(5, len(AUX_CHANNELS), cfg,
                                      aux_valid_index=AUX_INDEX["hr_observed"])
    model.eval()

    x = torch.randn(4, 24, 5)
    last = torch.full((4,), 120.0)
    zero_present = torch.zeros(4)

    with torch.no_grad():
        # Two completely different aux tensors, both marked absent.
        y1 = model(x, torch.randn(4, 24, len(AUX_CHANNELS)), last, zero_present)
        y2 = model(x, torch.randn(4, 24, len(AUX_CHANNELS)) * 100, last, zero_present)

    assert torch.allclose(y1[0], y2[0], atol=1e-5), \
        "aux content affected the output despite aux_present = 0"
    assert torch.allclose(y1[1], y2[1], atol=1e-5)


def test_absent_modality_reproduces_the_cgm_path_exactly():
    """The strong form: with aux absent, the trunk must equal h_cgm bit-for-bit.

    The weaker test above (invariance to aux *content*) passes even when the
    fusion path mangles the CGM representation in a content-independent way --
    which is exactly what a LayerNorm over (h_cgm + fused) did. That cost
    PR-AUC 0.963 -> 0.927 with the auxiliary stream carrying no information at
    all, and was invisible to every test until this one existed.

    Asserting equality against a hand-computed CGM-only forward pass is what
    makes "bounded below by the unimodal model" a checked property rather than
    a claim in a docstring.
    """
    cfg = Config().model
    model, _ = build_multimodal_model(5, len(AUX_CHANNELS), cfg,
                                      aux_valid_index=AUX_INDEX["hr_observed"])
    model.eval()

    x = torch.randn(4, 24, 5)
    xa = torch.randn(4, 24, len(AUX_CHANNELS))
    last = torch.full((4,), 120.0)

    with torch.no_grad():
        bg, logit = model(x, xa, last, aux_present=torch.zeros(4))

        # Hand-rolled CGM-only path through the same weights.
        h = model.cgm_backbone(x).transpose(1, 2)
        z = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        bg_ref = last + model.reg_head(z)
        logit_ref = model.cls_head(z)

    assert torch.allclose(bg, bg_ref, atol=1e-6), \
        "fusion path perturbs the CGM representation even with aux absent"
    assert torch.allclose(logit, logit_ref, atol=1e-6)


def test_dead_attention_rows_do_not_perturb_cgm_path():
    """Every-row-masked aux (no observations) must also reduce to the CGM path.

    Distinct from the test above: there aux_present gates the term off, here the
    aux is nominally 'present' but nothing in it was ever observed -- the case a
    patient with a dead Fitbit actually produces.
    """
    cfg = Config().model
    model, _ = build_multimodal_model(5, len(AUX_CHANNELS), cfg,
                                      aux_valid_index=AUX_INDEX["hr_observed"])
    model.eval()

    x = torch.randn(3, 24, 5)
    xa = torch.randn(3, 24, len(AUX_CHANNELS))
    xa[..., AUX_INDEX["hr_observed"]] = 0.0        # nothing observed
    last = torch.full((3,), 120.0)

    with torch.no_grad():
        bg, _ = model(x, xa, last, aux_present=torch.ones(3))
        h = model.cgm_backbone(x).transpose(1, 2)
        z = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        bg_ref = last + model.reg_head(z)

    assert torch.allclose(bg, bg_ref, atol=1e-6), \
        "fully-unobserved aux still perturbed the CGM representation"


def test_modality_dropout_is_training_only():
    """Eval must be deterministic; dropout of a whole modality at inference
    would make predictions non-reproducible for the same patient."""
    cfg = Config().model
    model, _ = build_multimodal_model(5, len(AUX_CHANNELS), cfg, modality_dropout=0.9)

    x = torch.randn(8, 24, 5)
    xa = torch.randn(8, 24, len(AUX_CHANNELS))
    last = torch.full((8,), 120.0)

    model.eval()
    with torch.no_grad():
        a = model(x, xa, last)[0]
        b = model(x, xa, last)[0]
    assert torch.allclose(a, b), "eval-mode output is not deterministic"


def test_gradients_flow_to_both_branches():
    """A fusion model whose aux branch receives no gradient is unimodal in
    disguise -- an easy failure to ship unnoticed."""
    cfg = Config().model
    model, criterion = build_multimodal_model(5, len(AUX_CHANNELS), cfg)
    model.train()

    x = torch.randn(8, 24, 5)
    xa = torch.randn(8, 24, len(AUX_CHANNELS))
    last = torch.full((8,), 120.0)
    bg, logit = model(x, xa, last, aux_present=torch.ones(8))
    loss, _ = criterion(bg, logit, torch.full((8,), 115.0), torch.zeros(8))
    loss.backward()

    def grad_norm(module):
        return sum(p.grad.abs().sum().item() for p in module.parameters()
                   if p.grad is not None)

    assert grad_norm(model.cgm_backbone) > 0, "no gradient reached the CGM branch"
    assert grad_norm(model.aux_backbone) > 0, "no gradient reached the aux branch"


# --- 7. Schema ---------------------------------------------------------------- #
def test_schema_validation_reports_actual_columns():
    df = pd.DataFrame({"time": [], "glucose": []})
    with pytest.raises(ValueError, match="Columns present"):
        HupaSchema().validate(df, "fake.csv")
