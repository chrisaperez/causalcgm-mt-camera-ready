"""Regression tests for the corrected evaluation protocol.

1. Fusion presence gate: mask-free treatment channels (insulin, carbs, basal)
   must open the gate. They could not, which shut the fusion branch on 99.6% of
   MetaboNet windows while insulin-on-board was recorded in 68% of them.
2. Alarm operating points are supplied from a calibration split and never
   fitted on the labels being scored.
3. Onset-referenced lead time and onset-only discrimination are stored, and an
   alert anchored at an already-hypoglycemic reading does not count as a
   forewarning.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causalcgm_mt.config import ModelConfig
from causalcgm_mt.data import TREATMENT_CHANNEL_NAMES, aux_presence, uses_unmasked_channels
from causalcgm_mt.evaluate import (
    calibration_thresholds, effective_lead_time_minutes, evaluate, onset_eligible,
)
from causalcgm_mt.metabonet_loader import METABONET_AUX_CHANNELS
from causalcgm_mt.models import build_multimodal_model
from causalcgm_mt.ohio_loader import OHIO_AUX_CHANNELS, modality_channel_names

T = 24


def _aux(names, n=4):
    return np.zeros((n, T, len(names)), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. presence gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("names", [METABONET_AUX_CHANNELS, OHIO_AUX_CHANNELS])
def test_treatment_alone_opens_the_gate(names):
    names = list(names)
    x = _aux(names)
    x[0, 5, names.index("iob")] = 1.3          # insulin on board, no wearable
    x[1, :, names.index("basal_rate")] = 0.8   # pump running, no wearable
    x[2, 3, names.index("hr_observed")] = 1.0 if "hr_observed" in names else 0.0
    # row 3: nothing at all
    p = aux_presence(x, names)
    assert p[0] == 1.0 and p[1] == 1.0
    assert p[3] == 0.0


def test_metabonet_shaped_cohort_is_not_gated_off():
    """The defect, reproduced: masks all zero, IOB populated on most windows."""
    names = list(METABONET_AUX_CHANNELS)
    rng = np.random.default_rng(0)
    x = _aux(names, n=1000)
    has_iob = rng.random(1000) < 0.68
    x[has_iob, :, names.index("iob")] = 1.0
    old_gate = (x[..., [i for i, c in enumerate(names) if c.endswith("_observed")]]
                .mean(axis=(1, 2)) > 0)
    assert old_gate.mean() == 0.0                       # what the old rule gave
    assert aux_presence(x, names).mean() == pytest.approx(has_iob.mean())


def test_presence_respects_the_arms_channels():
    """A wearable-only arm must not be gated open by insulin it never sees."""
    names = list(OHIO_AUX_CHANNELS)
    x = _aux(names, n=1)
    x[0, :, names.index("iob")] = 2.0
    assert aux_presence(x, names, use=modality_channel_names(["gsr"]))[0] == 0.0
    assert aux_presence(x, names, use=modality_channel_names(["treatment"]))[0] == 1.0
    assert uses_unmasked_channels(names, modality_channel_names(["treatment"]))
    assert not uses_unmasked_channels(names, modality_channel_names(["gsr"]))
    assert set(TREATMENT_CHANNEL_NAMES) & set(modality_channel_names(["treatment"]))


def test_fusion_branch_reads_treatment_when_wearables_are_off():
    """With treatment in use, the attention must see steps whose wearable masks
    are all zero; before the fix it was fully masked and fusion was a no-op."""
    torch.manual_seed(0)
    names = list(OHIO_AUX_CHANNELS)
    mask_idx = [i for i, c in enumerate(names) if c.endswith("_observed")]
    cfg = ModelConfig(n_horizons=1)
    x = torch.randn(3, T, 5)
    last = torch.full((3,), 120.0)
    x_aux = torch.zeros(3, T, len(names))
    x_aux[..., names.index("iob")] = torch.linspace(0.5, 2.0, T)   # masks all zero
    present = torch.ones(3)

    for always_valid, expect_effect in ((True, True), (False, False)):
        model, _ = build_multimodal_model(5, len(names), cfg, aux_valid_index=mask_idx,
                                          always_valid=always_valid)
        model.eval()
        with torch.no_grad():
            on, _ = model(x, x_aux, last, present)
            off, _ = model(x, x_aux, last, torch.zeros(3))
        moved = not torch.allclose(on, off)
        assert moved is expect_effect


# --------------------------------------------------------------------------- #
# 2. calibration thresholds
# --------------------------------------------------------------------------- #
def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    s = np.clip(y * 0.4 + rng.random(n) * 0.7, 0, 1)
    bg = np.full(n, 120.0)
    return y, s, bg


def test_no_operating_point_without_supplied_thresholds():
    y, s, bg = _toy()
    rep = evaluate(bg, bg, y, s, bg, np.array(["p"] * y.size), 5)
    assert np.isnan(rep.fapd_at_5pct_fpr)
    assert np.isnan(rep.sensitivity_at_5pct_fpr)
    assert rep.threshold_source == "none"


def test_thresholds_are_not_refit_on_test_labels():
    y_cal, s_cal, _ = _toy(seed=1)
    thr = calibration_thresholds(y_cal, s_cal)
    y, s, bg = _toy(seed=2)
    pat = np.array(["p"] * y.size)
    a = evaluate(bg, bg, y, s, bg, pat, 5, alarm_thresholds=thr)
    b = evaluate(bg, bg, 1 - y, s, bg, pat, 5, alarm_thresholds=thr)  # flipped labels
    assert a.threshold_at_5pct_fpr == b.threshold_at_5pct_fpr == thr[0.05]
    assert a.threshold_source == "calibration"
    fires = s >= thr[0.05]
    assert a.realized_fpr_at_5pct == pytest.approx(fires[y == 0].mean())


# --------------------------------------------------------------------------- #
# 3. onset metrics
# --------------------------------------------------------------------------- #
def test_alert_while_already_low_is_not_a_forewarning():
    # glucose: 100 x10, then low at index 10..14, back up
    g = np.array([100.0] * 10 + [60.0] * 5 + [100.0] * 5)
    s = np.zeros(g.size)
    s[10] = 1.0          # fires only once already low -> no onset forewarning
    rep = effective_lead_time_minutes(g, s, 0.5, horizon_steps=6)
    assert rep.n_onset_detected == 0
    s[7] = 1.0           # fires 3 steps (15 min) before onset, while >= 70
    rep = effective_lead_time_minutes(g, s, 0.5, horizon_steps=6)
    assert rep.n_onset_detected == 1 and rep.mean_onset_lead_min == 15.0


def test_onset_eligibility_excludes_recent_lows():
    g = np.array([100.0, 65.0, 100, 100, 100, 100, 100, 100, 100])
    seg = np.zeros(g.size, dtype=int)
    e = onset_eligible(g, seg, lookback_steps=6)
    assert not e[1]                  # low now
    assert not e[2] and not e[7]     # low within the last 30 min
    assert e[8] and e[0]


def test_evaluate_reports_onset_fields():
    rng = np.random.default_rng(3)
    n = 400
    bg = 90 + 40 * np.sin(np.arange(n) / 15.0)
    y = (np.roll(bg, -3) < 70).astype(int)
    s = 1 / (1 + np.exp((bg - 80) / 5)) + rng.random(n) * 0.05
    thr = calibration_thresholds(y, s)
    t = np.datetime64("2026-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")
    rep = evaluate(bg, bg, y, s, bg, np.array(["p"] * n), 5, t_time=t,
                   horizon_min=30, alarm_thresholds=thr)
    assert rep.n_onset_windows < n
    assert np.isfinite(rep.onset_detection_rate)


# --------------------------------------------------------------------------- #
# 4. alarm latching / persistence policy
# --------------------------------------------------------------------------- #
def test_latched_alarm_defaults_equal_alarm_episodes():
    from causalcgm_mt.clinical import alarm_episodes, latched_alarm
    rng = np.random.default_rng(5)
    m = rng.random(500) < 0.2
    seg = np.repeat(np.arange(5), 100)
    on, eps = latched_alarm(m, seg)
    assert eps == alarm_episodes(m, seg)
    assert (on == m).all()


def test_latching_merges_short_gaps_within_a_segment_only():
    from causalcgm_mt.clinical import latched_alarm
    m = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0], dtype=bool)
    seg = np.zeros(m.size, dtype=int)
    _, eps = latched_alarm(m, seg, merge_gap_steps=1)
    assert eps == [(0, 4), (7, 9)]                 # 1-step gap merged, 3-step not
    on, eps = latched_alarm(m, seg, merge_gap_steps=3)
    assert eps == [(0, 9)] and on[2] and on[4]     # gap steps are active
    seg2 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    _, eps = latched_alarm(m, seg2, merge_gap_steps=3)
    assert eps[0] == (0, 2)                        # never merges across segments


def test_persistence_filter_delays_and_drops_short_runs():
    from causalcgm_mt.clinical import latched_alarm
    m = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    on, eps = latched_alarm(m, None, min_duration_steps=3)
    assert eps == [(4, 5)]                         # only the 3-step run sounds, late
    assert on.sum() == 1
