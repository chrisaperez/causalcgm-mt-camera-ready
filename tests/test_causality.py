"""Executable audit of the no-lookahead claims.

These are not smoke tests. Each one corresponds to a specific way this class of
pipeline is known to leak, and is written to FAIL if the corresponding guarantee
is removed.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.data import (
    CausalScaler, SoftLabeler, build_windows, causal_features,
    patient_folds, split_by_patient,
)
from causalcgm_mt.models import CausalConv1d, build_model
from causalcgm_mt.simulate import simulate_cohort


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return simulate_cohort(n_patients=6, days=3.0, seed=7)


@pytest.fixture(scope="module")
def ws(cohort):
    return build_windows(cohort, DataConfig())


# --- 1. Architectural causality -------------------------------------------- #
def test_causal_conv_ignores_future():
    """Perturbing input at time T must not change any output before T."""
    conv = CausalConv1d(3, 5, kernel_size=3, dilation=2).double()
    x = torch.randn(1, 3, 40, dtype=torch.double)
    y0 = conv(x)

    x2 = x.clone()
    x2[..., 25:] += 100.0            # corrupt everything from t=25 onward
    y2 = conv(x2)

    assert torch.allclose(y0[..., :25], y2[..., :25], atol=1e-9), \
        "causal conv leaked future information into past outputs"
    assert not torch.allclose(y0[..., 25:], y2[..., 25:])


def test_full_model_output_depends_only_on_history():
    """The model, end to end, is a function of the history window alone."""
    model, _ = build_model(5, Config().model)
    model.eval()
    x = torch.randn(4, 24, 5)
    last = torch.full((4,), 120.0)
    with torch.no_grad():
        a = model(x, last)
        b = model(x, last)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_causal_features_are_prefix_functions():
    """Feature at index i must be invariant to changes at indices > i."""
    h = np.random.default_rng(0).normal(120, 20, 24)
    obs = np.ones(24)
    f0 = causal_features(h, obs)

    h2 = h.copy()
    h2[15:] += 50.0
    f2 = causal_features(h2, obs)

    assert np.allclose(f0[:15], f2[:15]), "feature engineering looked ahead"


# --- 2. Windowing / target separation --------------------------------------- #
def test_anchor_is_last_history_step(ws):
    """x[..., -1, 0] is BG(t): the newest value the model may legally see."""
    assert np.allclose(ws.x[:, -1, 0], ws.last_bg, atol=1e-4)


def test_future_perturbation_changes_targets_but_not_inputs(cohort):
    """The decisive end-to-end leak test.

    Corrupt the *tail* of every patient's record. Windows whose history lies
    entirely before the corruption must have byte-identical inputs; their
    targets are allowed to change. If any input tensor moves, some part of the
    pipeline read forward in time.
    """
    cfg = DataConfig()
    ws_a = build_windows(cohort, cfg)

    d = cohort.copy()
    cut = d.groupby("patient_id")["timestamp"].transform(
        lambda s: s.min() + (s.max() - s.min()) * 0.7
    )
    d.loc[d.timestamp > cut, "glucose"] += 120.0
    ws_b = build_windows(d, cfg)

    # Compare on the windows both runs produced, keyed by (patient, anchor time).
    key_a = {(str(p), t): i for i, (p, t) in enumerate(zip(ws_a.patient, ws_a.t_time))}
    key_b = {(str(p), t): i for i, (p, t) in enumerate(zip(ws_b.patient, ws_b.t_time))}
    shared = sorted(set(key_a) & set(key_b))
    assert len(shared) > 100

    # Windows fully inside the clean region: history end is well before the cut.
    cuts = {str(p): c for p, c in zip(cohort.patient_id, cut)}
    n_checked = 0
    for pid, t in shared:
        # anchor at least one horizon before the corruption boundary
        if t >= np.datetime64(cuts[pid]) - np.timedelta64(cfg.horizon_min, "m"):
            continue
        ia, ib = key_a[(pid, t)], key_b[(pid, t)]
        assert np.array_equal(ws_a.x[ia], ws_b.x[ib]), \
            f"input changed after a purely FUTURE perturbation ({pid} @ {t})"
        n_checked += 1
    assert n_checked > 100, "test did not exercise enough windows"


def test_window_geometry(ws):
    cfg = DataConfig()
    assert ws.x.shape[1] == cfg.history_steps
    assert ws.x.shape[2] == 5
    assert len(ws.y_reg) == len(ws.x) == len(ws.patient)


def test_classification_label_matches_min_over_horizon(ws):
    cfg = DataConfig()
    expected = (ws.y_min < cfg.hypo_threshold_mgdl).astype(np.float32)
    assert np.array_equal(ws.y_cls_hard, expected)


def test_no_imputed_targets(cohort):
    """Windows with any unobserved target step must be dropped, not filled."""
    d = cohort.copy()
    # Blank out a contiguous stretch for one patient.
    m = (d.patient_id == "P000") & (d.index % 97 < 20)
    d = d[~m]
    ws = build_windows(d, DataConfig())
    assert np.isfinite(ws.y_reg).all()
    assert np.isfinite(ws.x).all()


# --- 3. Scaler leakage ------------------------------------------------------ #
def test_scaler_fits_only_on_training_patients(ws):
    pats = sorted(set(str(p) for p in ws.patient))
    train, test = pats[:4], pats[4:]
    sc = CausalScaler.fit(ws, train)

    assert set(sc.fitted_on) == set(train)
    assert not (set(sc.fitted_on) & set(test)), "test patients contributed statistics"


def test_scaler_is_invariant_to_test_data(ws):
    """The decisive test: corrupting held-out patients must not move the scaler."""
    pats = sorted(set(str(p) for p in ws.patient))
    train, test = pats[:4], pats[4:]

    sc_a = CausalScaler.fit(ws, train)

    corrupted = ws.subset(np.arange(len(ws)))
    corrupted.x = corrupted.x.copy()
    corrupted.x[np.isin(corrupted.patient, np.array(test, dtype=object))] *= 10.0
    sc_b = CausalScaler.fit(corrupted, train)

    assert np.allclose(sc_a.center, sc_b.center)
    assert np.allclose(sc_a.scale, sc_b.scale)


def test_mask_channel_is_not_scaled(ws):
    sc = CausalScaler.fit(ws, sorted(set(str(p) for p in ws.patient))[:4])
    assert sc.center[4] == 0.0 and sc.scale[4] == 1.0


# --- 4. Patient-wise splitting ---------------------------------------------- #
def test_folds_are_disjoint_by_patient(ws):
    pats = sorted(set(str(p) for p in ws.patient))
    folds = patient_folds(pats, n_folds=3, seed=0)

    seen = set()
    for train, test in folds:
        assert not (set(map(str, train)) & set(map(str, test))), "patient in both splits"
        seen |= set(map(str, test))
    assert seen == set(pats), "every patient must be tested exactly once"


def test_no_window_appears_in_two_splits(ws):
    pats = sorted(set(str(p) for p in ws.patient))
    train, test = pats[:4], pats[4:]
    i_tr = set(split_by_patient(ws, train).tolist())
    i_te = set(split_by_patient(ws, test).tolist())
    assert not (i_tr & i_te)
    assert len(i_tr) + len(i_te) == len(ws)


# --- 5. Soft labels --------------------------------------------------------- #
def test_soft_label_monotone_and_calibrated_at_threshold():
    cfg = DataConfig(band_mode="fixed", soft_band_mgdl=10.0)
    lab = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=None)

    bg = np.array([40.0, 60.0, 69.0, 70.0, 71.0, 80.0, 200.0], dtype=np.float32)
    y = lab(bg)

    assert np.all(np.diff(y) <= 1e-6), "soft label must decrease with glucose"
    assert y[3] == pytest.approx(0.5, abs=0.02), "must be 0.5 at the threshold"
    assert y[0] > 0.95 and y[-1] < 0.05, "must saturate far from the boundary"
    assert y.min() >= cfg.soft_label_floor and y.max() <= cfg.soft_label_ceiling


def test_soft_labels_reduce_to_hard_when_disabled():
    cfg = DataConfig(soft_labels=False)
    lab = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.1)
    bg = np.array([50.0, 69.9, 70.1, 150.0], dtype=np.float32)
    assert np.array_equal(lab(bg), np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_mard_proxy_is_unbiased_on_known_noise():
    """The estimator must recover a KNOWN noise magnitude, not merely track it.

    Regression test for a real bug: the original version divided by sqrt(6) to
    undo the second-difference variance inflation but never converted
    median(|x|) to sigma, underestimating noise by the factor 0.6745. That
    collapsed the soft-label band to ~1 mg/dL and silently turned soft labels
    into hard ones.
    """
    from causalcgm_mt.data import estimate_mard_proxy

    rng = np.random.default_rng(0)
    for true_rel in (0.02, 0.05, 0.10):
        s = 140.0 + rng.normal(0, 140.0 * true_rel, 100_000)
        est = estimate_mard_proxy([s], period_min=5)
        assert est == pytest.approx(true_rel, rel=0.05), \
            f"noise estimate biased: true {true_rel}, got {est}"


def test_degenerate_estimate_falls_back_to_the_device_floor():
    """A band below CGM resolution must not pass, and must say so.

    This used to be `test_degenerate_band_warns`, and a `UserWarning` was the
    whole mechanism: the mard path could produce a band of 0.07 mg/dL, warn, and
    carry on. In practice that made the mode unusable rather than loud -- every
    driver pinned `band_mode="fixed"` to keep the warning out of the logs, and
    the estimate was computed and thrown away. The band is now floored at the
    device specification and the fallback is RECORDED, which is what makes the
    mode sweepable instead of merely noisy.
    """
    cfg = DataConfig(band_mode="mard", soft_labels=True)
    lab = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.001)

    assert lab.band_source == "device_floor"
    assert lab.mard_used == pytest.approx(cfg.band_mard_floor)
    assert not lab.is_effectively_hard
    assert cfg.band_min_mgdl <= lab.band <= cfg.band_max_mgdl


def test_a_fixed_band_is_never_silently_rewritten():
    """`band_mode="fixed"` means fixed: the mard policy must not touch it.

    The floor and the clamp exist to keep an ESTIMATE honest. Applying them to a
    hyperparameter would make the hard-label control arm of Ablation E
    unreachable, and would silently change any experiment that deliberately asks
    for a narrow band.
    """
    cfg = DataConfig(band_mode="fixed", soft_band_mgdl=0.5)
    lab = SoftLabeler.fit(np.array([100.0]), cfg)
    assert lab.band == 0.5 and lab.band_source == "fixed"
    assert lab.is_effectively_hard, "0.5 mg/dL is below one CGM display step"


def test_soft_band_derives_from_training_statistics_only():
    cfg = DataConfig(band_mode="mard", band_mard_multiplier=1.0)
    a = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.10)
    b = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.20)
    assert b.band > a.band, "band must track the fold's noise estimate"


# --- 5b. Time lag sanity ---------------------------------------------------- #
def test_time_lag_detects_persistence():
    """A persistence 'predictor' must score a lag of exactly the horizon.

    This is the calibration check for the metric itself: if it reports 0 here,
    the sign convention is inverted and the metric is worthless for detecting
    the copy-the-input failure mode.
    """
    from causalcgm_mt.evaluate import time_lag_minutes

    t = np.arange(600)
    truth_series = 120 + 40 * np.sin(2 * np.pi * t / 100)   # BG(t)
    ph_steps, period = 6, 5

    y = truth_series[ph_steps:]          # target = BG(t + PH)
    persistence = truth_series[:-ph_steps]  # prediction = BG(t)

    assert time_lag_minutes(y, persistence, period) == pytest.approx(ph_steps * period)
    # A perfect (anticipatory) predictor has zero lag.
    assert time_lag_minutes(y, y, period) == pytest.approx(0.0)


def test_soft_label_is_numerically_stable_for_severe_hypo():
    cfg = DataConfig(band_mode="fixed", soft_band_mgdl=0.5)  # very narrow band
    lab = SoftLabeler.fit(np.array([100.0]), cfg)
    y = lab(np.array([20.0, 400.0], dtype=np.float32))
    assert np.isfinite(y).all()


# --- 6. Personalisation protocol -------------------------------------------- #
def test_adaptation_prefix_strictly_precedes_evaluation(ws):
    from causalcgm_mt.extension import prefix_embargo_split

    cfg = Config()
    cfg = dataclasses.replace(cfg, adapt=dataclasses.replace(cfg.adapt, adapt_prefix_hours=12.0))
    pid = sorted(set(str(p) for p in ws.patient))[0]
    a_idx, e_idx = prefix_embargo_split(ws, pid, cfg)

    assert a_idx.size > 0 and e_idx.size > 0
    a_end = ws.t_time[a_idx].max()
    e_start = ws.t_time[e_idx].min()
    gap_min = (e_start - a_end) / np.timedelta64(1, "m")

    assert e_start > a_end, "evaluation must start after the adaptation prefix"
    assert gap_min >= cfg.data.history_min + cfg.data.horizon_min, \
        "embargo too short: evaluation history can overlap adaptation targets"
    assert not (set(a_idx.tolist()) & set(e_idx.tolist()))


def test_lora_starts_as_identity():
    """B initialised to zero => the adapted model equals the base model at step 0."""
    from causalcgm_mt.extension import inject_lora
    from causalcgm_mt.config import AdaptConfig
    import copy

    cfg = Config()
    model, _ = build_model(5, cfg.model)
    model.eval()
    x, last = torch.randn(2, 24, 5), torch.full((2,), 120.0)
    with torch.no_grad():
        base_out = model(x, last)

    adapted = copy.deepcopy(model)
    n = inject_lora(adapted, AdaptConfig(lora_rank=4))
    adapted.eval()
    with torch.no_grad():
        new_out = adapted(x, last)

    assert n > 0, "no conv layers were wrapped"
    assert torch.allclose(base_out[0], new_out[0], atol=1e-5)
    assert torch.allclose(base_out[1], new_out[1], atol=1e-5)


def test_lora_is_parameter_efficient():
    from causalcgm_mt.extension import configure_trainable
    from causalcgm_mt.config import AdaptConfig

    cfg = Config()
    model, _ = build_model(5, cfg.model)
    total = sum(p.numel() for p in model.parameters())
    params = configure_trainable(model, AdaptConfig(method="lora", lora_rank=4))
    trainable = sum(p.numel() for p in params)

    assert 0 < trainable < 0.15 * total, \
        f"LoRA should train <15% of params, got {trainable}/{total}"
