"""Audit of the clinical-safety metrics, the modality ablation, and LoRA reach.

As elsewhere in this suite, each test names a specific failure mode. Several of
these correspond to bugs that were actually present: `basis_sleep` parsed under
a different attribute spelling, LoRA unreachable on the multimodal model, and a
Parkes grid that graded 0.5% of points acceptable where Clarke graded 99%.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.clinical import (
    ZONE_NAMES, alarm_episodes, clarke_zones, clinical_report, hypo_events,
    parkes_zones, runs_of_true, segment_ids,
)
from causalcgm_mt.config import AdaptConfig, ModelConfig
from causalcgm_mt.evaluate import (
    EvalReport, calibration_thresholds, effective_lead_time_minutes, evaluate,
    false_alarms_per_day,
    threshold_at_fpr,
)
from causalcgm_mt.extension import (
    LoRAConv1d, LoRALinear, configure_trainable, inject_lora, lora_delta_w,
    lora_delta_norm_sq, l2_sp_penalty,
)
from causalcgm_mt.models import build_model, build_multimodal_model
from causalcgm_mt.ohio_loader import (
    DEVICE_SCALED_CHANNELS, MODALITY_COHORTS, MODALITY_GROUPS,
    OHIO_AUX_CHANNELS, OHIO_AUX_INDEX, PerPatientNormalizer,
    apply_modality_mask, resolve_arm,
)


# --------------------------------------------------------------------------- #
# 1. Clarke error grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ref,pred,zone", [
    (100, 100, "A"), (100, 115, "A"), (60, 65, "A"),   # clinically identical
    (150, 200, "B"),                                    # off, but harmless
    (100, 220, "C"),                                    # prompts unwarranted treatment
    (60, 120, "D"), (300, 150, "D"),                    # FAILURE TO DETECT
    (60, 200, "E"), (300, 50, "E"),                     # opposite treatment
])
def test_clarke_canonical_points(ref, pred, zone):
    assert ZONE_NAMES[clarke_zones(np.array([ref]), np.array([pred]))[0]] == zone


def test_clarke_severity_wins_on_overlap():
    """The published rules overlap at zone corners. Scoring an overlapping point
    as the milder zone would understate exactly the risk being measured."""
    y = np.array([60.0]); p = np.array([190.0])
    assert ZONE_NAMES[clarke_zones(y, p)[0]] == "E"


def test_clarke_never_grades_a_d_point_as_the_milder_c():
    """The severity order has to hold for C vs D too, not just for E.

    The writes ran `zone[d] = 3` then `zone[c] = 2`, so a point matching both
    rules was recorded as C -- the reassuring direction, in the function whose
    whole purpose is to measure the dangerous one. Unreachable as the
    boundaries stand (the only C^D overlap is `y == 70, p >= 180`, where E
    overwrites both), which is exactly why nothing caught it: one boundary
    change makes it live and silent.

    Asserted over the whole assay grid against the D rule alone, so it pins the
    ordering without restating every rule in the file it is testing.
    """
    g = np.arange(1.0, 601.0, 1.0)
    y, p = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    zones = clarke_zones(y, p)

    d = (((y >= 240) & (p >= 70) & (p <= 180))
         | ((y <= 175.0 / 3.0) & (p <= 180) & (p >= 70))
         | (((y >= 175.0 / 3.0) & (y <= 70)) & (p >= (6.0 / 5.0) * y)))
    assert d.any(), "the D rule matched nothing; this test proves nothing"
    assert (zones[d] >= 3).all(), (
        "a failure-to-detect point was graded milder than D")

    # And nothing is graded A while a dangerous rule applies to it.
    assert not (zones[d] == 0).any()


def test_clarke_writes_zones_in_increasing_severity():
    """Structural, because the behavioural overlap above is currently empty:
    the source itself must not reintroduce the ordering that made a C-and-D
    point read as C."""
    import inspect

    src = inspect.getsource(clarke_zones)
    order = [src.index(f"zone[{m}] = {v}") for m, v in (("c", 2), ("d", 3), ("e", 4))]
    assert order == sorted(order), (
        "the zone writes are not in increasing severity, so an overlapping "
        "point takes the milder zone")


def test_clarke_clips_out_of_range_predictions():
    """An unclipped regressor can emit negative or absurd values; those must not
    fall off the edge of every rule and land in B by default."""
    z = clarke_zones(np.array([60.0, 60.0]), np.array([-50.0, 5000.0]))
    assert ZONE_NAMES[z[0]] == "A"    # both below 70 -> same treatment
    assert ZONE_NAMES[z[1]] == "E"    # truth hypo, prediction extreme hyper


def test_perfect_prediction_is_all_zone_a():
    y = np.linspace(40, 400, 500)
    assert (clarke_zones(y, y) == 0).all()


def test_hypo_bias_sign_is_reported_as_the_dangerous_direction():
    """A positive bias below 70 means the model reads lows as higher than they
    are. That is the direction that kills a warning system, and the sign is
    routinely read backwards -- so the report must say which it is."""
    y = np.full(100, 60.0)
    rep = clinical_report(y, y + 15.0)
    assert rep.hypo_band_bias > 0
    assert "dangerous" in str(rep)


def test_parkes_never_falls_back_to_a_guess():
    """A safety metric that is wrong in the reassuring direction is worse than
    no metric. The hand-transcribed grid was deleted, not left behind a flag:
    Parkes now either comes from a verified backend or raises. There is no path
    that returns an approximation.

    (Full backend coverage lives in test_multihorizon_and_cost.py.)"""
    from causalcgm_mt.clinical import parkes_available
    if parkes_available():
        exact = np.array([100.0, 55.0])
        assert (parkes_zones(exact, exact) == 0).all()
    else:
        with pytest.raises(ImportError, match="pip install"):
            parkes_zones(np.array([100.0]), np.array([100.0]))


# --------------------------------------------------------------------------- #
# 2. Modality ablation
# --------------------------------------------------------------------------- #
def test_masking_zeroes_values_and_masks_together():
    """Zeroing values but leaving the mask set would tell the model 'this WAS
    observed and its value is exactly zero' -- strictly worse than absence."""
    x = np.ones((4, 6, len(OHIO_AUX_CHANNELS)), dtype=np.float32)
    out = apply_modality_mask(x, ["heart_rate"])
    for c in MODALITY_GROUPS["heart_rate"]:
        assert (out[..., OHIO_AUX_INDEX[c]] == 1.0).all()
    assert (out[..., OHIO_AUX_INDEX["gsr"]] == 0.0).all()
    assert (out[..., OHIO_AUX_INDEX["gsr_observed"]] == 0.0).all()


def test_masking_preserves_tensor_width():
    """Arms must instantiate identically shaped models, or a difference between
    them could come from parameter count rather than from the sensor."""
    x = np.ones((2, 3, len(OHIO_AUX_CHANNELS)), dtype=np.float32)
    assert apply_modality_mask(x, ["gsr"]).shape == x.shape
    assert apply_modality_mask(x, []).shape == x.shape


def test_cohort_resolution_matches_the_released_sensor_sets():
    """2018 wore a Basis Peak, 2020 an Empatica. The sensor sets are disjoint,
    and an arm's cohort scope must follow from that, not from a guess."""
    assert resolve_arm(["heart_rate"])[1] == ["2018"]
    assert resolve_arm(["accel"])[1] == ["2020"]
    assert resolve_arm(["gsr"])[1] == ["2018", "2020"]


def test_unsatisfiable_arm_is_reported_as_empty():
    """No subject wore both devices, so heart rate + acceleration is not a
    trainable arm -- it must resolve to no cohort rather than to windows where
    half the signal is silently absent."""
    assert resolve_arm(["heart_rate", "accel"])[1] == []


def test_every_group_channel_exists():
    for group, chans in MODALITY_GROUPS.items():
        for c in chans:
            assert c in OHIO_AUX_INDEX, f"{group} references unknown channel {c}"
    assert set(MODALITY_COHORTS) == set(MODALITY_GROUPS)


# --------------------------------------------------------------------------- #
# 3. Per-patient device normalisation
# --------------------------------------------------------------------------- #
def test_per_patient_normalisation_equalises_device_scales():
    """Released GSR spans ~4 orders of magnitude across patients. A single
    global IQR maps most of them onto numerical zero."""
    n_ch = len(OHIO_AUX_CHANNELS)
    gsr = OHIO_AUX_INDEX["gsr"]
    x = np.zeros((200, 8, n_ch), dtype=np.float32)
    pat = np.array(["a"] * 100 + ["b"] * 100, dtype=object)
    rng = np.random.default_rng(0)
    x[:100, :, gsr] = rng.normal(1e-4, 2e-5, (100, 8))    # Basis Peak scale
    x[100:, :, gsr] = rng.normal(1e-2, 2e-3, (100, 8))    # Empatica scale

    norm = PerPatientNormalizer.fit(x, pat, np.arange(200))
    out = norm.transform(x, pat)
    sa = out[:100, :, gsr].std()
    sb = out[100:, :, gsr].std()
    assert 0.2 < sa / sb < 5.0, "post-normalisation scales should be comparable"


def test_per_patient_normalisation_is_identity_for_constant_channels():
    """A channel the patient never wore is constant; dividing by its ~zero IQR
    would amplify float noise into a strong spurious signal."""
    n_ch = len(OHIO_AUX_CHANNELS)
    x = np.zeros((10, 4, n_ch), dtype=np.float32)
    pat = np.array(["a"] * 10, dtype=object)
    out = PerPatientNormalizer.fit(x, pat, np.arange(10)).transform(x, pat)
    assert np.isfinite(out).all() and (out == 0.0).all()


def test_unseen_patient_passes_through_untransformed():
    n_ch = len(OHIO_AUX_CHANNELS)
    x = np.ones((6, 3, n_ch), dtype=np.float32)
    fit_pat = np.array(["a"] * 6, dtype=object)
    norm = PerPatientNormalizer.fit(x, fit_pat, np.arange(6))
    new = np.full((2, 3, n_ch), 7.0, dtype=np.float32)
    out = norm.transform(new, np.array(["zzz", "zzz"], dtype=object))
    np.testing.assert_array_equal(out, new)


def test_device_scaled_channels_are_all_real_channels():
    for c in DEVICE_SCALED_CHANNELS:
        assert c in OHIO_AUX_INDEX


# --------------------------------------------------------------------------- #
# 4. LoRA reach
# --------------------------------------------------------------------------- #
def _mm():
    return build_multimodal_model(5, len(OHIO_AUX_CHANNELS), ModelConfig(), 1.0,
                                  aux_valid_index=OHIO_AUX_INDEX["hr_observed"])[0]


def test_lora_reaches_the_multimodal_model():
    """`inject_lora` looked only for `model.backbone`, which the multimodal
    model does not have -- so `--adapt lora` raised AttributeError for the whole
    fusion arm."""
    m = _mm()
    assert inject_lora(m, AdaptConfig()) > 0


def test_lora_adapts_the_cross_attention_projections():
    """The attention layer is where the model decides how much to read from the
    wearable branch, and that weighting is exactly what differs per patient."""
    m = _mm()
    inject_lora(m, AdaptConfig())
    wrapped = {n for n, mod in m.named_modules() if isinstance(mod, LoRALinear)}
    assert {"cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.proj"} <= wrapped


def test_lora_still_works_on_the_unimodal_model():
    m = build_model(5, ModelConfig(), 1.0)[0]
    assert inject_lora(m, AdaptConfig()) > 0
    assert any(isinstance(x, LoRAConv1d) for x in m.modules())


def test_lora_starts_as_an_exact_identity():
    """B is zero-initialised, so adaptation begins from the base model. If this
    fails, every personalisation run starts by damaging the global model."""
    m = _mm().eval()
    x, xa = torch.randn(4, 24, 5), torch.randn(4, 24, len(OHIO_AUX_CHANNELS))
    last, ap = torch.full((4,), 120.0), torch.ones(4)
    with torch.no_grad():
        before = m(x, xa, last, ap)[0].clone()
    configure_trainable(m, AdaptConfig(method="lora"))
    m.eval()
    with torch.no_grad():
        after = m(x, xa, last, ap)[0]
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)


def test_lora_freezes_everything_else():
    m = _mm()
    params = configure_trainable(m, AdaptConfig(method="lora"))
    trainable = {id(p) for p in params}
    assert all(not p.requires_grad for p in m.parameters() if id(p) not in trainable)
    frac = sum(p.numel() for p in params) / sum(p.numel() for p in m.parameters())
    assert 0.0 < frac < 0.25, "LoRA should adapt a small slice, not the whole model"


def test_injection_is_idempotent():
    """A caller that inspects the adapter count before configuring must not end
    up with adapters wrapped twice, nor look like a failed match."""
    m = _mm()
    n = inject_lora(m, AdaptConfig())
    count = lambda: sum(1 for x in m.modules() if isinstance(x, (LoRAConv1d, LoRALinear)))
    assert count() == n
    configure_trainable(m, AdaptConfig(method="lora"))
    assert count() == n, "second injection wrapped the adapters again"
    # And no adapter may contain another adapter.
    for mod in m.modules():
        if isinstance(mod, (LoRAConv1d, LoRALinear)):
            assert not isinstance(mod.base, (LoRAConv1d, LoRALinear))


# --------------------------------------------------------------------------- #
# 5. Alarm-fatigue metrics
# --------------------------------------------------------------------------- #
# Each of these pins down a decision that changes the headline number by a
# factor, not by a rounding error: what counts as ONE alarm, what counts as ONE
# hypoglycemic event, and which alerts are early enough to have been warnings.
def test_consecutive_alarm_steps_are_one_episode():
    """A latched alarm is one beep, not one per 5-minute sample.

    288 samples = one day. Two spurious excursions, eight steps and four steps
    long, are TWO false alarms per day -- not twelve.
    """
    y = np.zeros(288)
    p = np.zeros(288)
    p[10:18] = 0.9        # one 40-minute excursion
    p[100:104] = 0.9      # one 20-minute excursion
    assert false_alarms_per_day(y, p, threshold=0.5) == pytest.approx(2.0)


def test_fapd_scales_with_the_monitoring_duration():
    """The denominator is time actually covered by the rows, so the same two
    episodes over two days is half the daily burden."""
    y = np.zeros(576)
    p = np.zeros(576)
    p[10:18] = 0.9
    p[400:404] = 0.9
    assert false_alarms_per_day(y, p, threshold=0.5) == pytest.approx(1.0)


def test_an_episode_containing_a_true_positive_is_not_a_false_alarm():
    """An alarm that fires early and stays on through a genuine event is the
    behaviour we want. Charging it as a false alarm too would penalise exactly
    the right answer."""
    y = np.zeros(288)
    y[14:20] = 1                       # the real event, inside the alarm run
    p = np.zeros(288)
    p[10:22] = 0.9                     # one episode, spans the event
    p[100:104] = 0.9                   # one genuinely spurious episode
    assert false_alarms_per_day(y, p, threshold=0.5) == pytest.approx(1.0)


def test_alarm_episodes_do_not_bridge_a_recording_gap():
    """A run of alarms either side of a sensor dropout is two alarms. Treating
    it as one halves the reported burden in exactly the situation -- gappy real
    CGM -- where the burden matters most."""
    y = np.zeros(20)
    p = np.zeros(20)
    p[8:12] = 0.9
    seg = np.zeros(20, dtype=np.int64)
    seg[10:] = 1                       # dropout between index 9 and 10

    one = false_alarms_per_day(y, p, 0.5, period_minutes=5)
    two = false_alarms_per_day(y, p, 0.5, period_minutes=5, segment=seg)
    assert two == pytest.approx(2.0 * one)


def test_segment_ids_break_on_gaps_and_on_patient_changes():
    t = np.array(["2021-01-01T00:00", "2021-01-01T00:05", "2021-01-01T02:00",
                  "2021-01-01T02:05", "2021-01-01T02:10"], dtype="datetime64[m]")
    pat = np.array(["a", "a", "a", "a", "b"], dtype=object)
    assert list(segment_ids(t, None, 5)) == [0, 0, 1, 1, 1]
    assert list(segment_ids(t, pat, 5)) == [0, 0, 1, 1, 2]
    assert list(segment_ids(None, pat, 5)) == [0, 0, 0, 0, 1]


def test_runs_of_true_handles_the_degenerate_shapes():
    assert runs_of_true(np.array([], dtype=bool)) == []
    assert runs_of_true(np.zeros(5, bool)) == []
    assert runs_of_true(np.ones(5, bool)) == [(0, 5)]
    assert runs_of_true(np.array([1, 0, 1, 1, 0, 1], bool)) == [(0, 1), (2, 4), (5, 6)]


def test_fapd_on_an_empty_slice_is_undefined_not_zero():
    """Zero days monitored is not zero false alarms per day."""
    assert np.isnan(false_alarms_per_day(np.array([]), np.array([]), 0.5))


def test_fapd_with_an_infinite_threshold_silences_the_alarm():
    """threshold_at_fpr returns +inf when no finite threshold meets the target;
    that is a real operating point (never fire), not an error."""
    y = np.zeros(288)
    p = np.full(288, 0.4)
    assert false_alarms_per_day(y, p, np.inf) == 0.0
    assert np.isnan(false_alarms_per_day(y, p, np.nan))


def test_fapd_ignores_non_finite_scores():
    """A missing prediction is not a warning; it must not latch the alarm."""
    y = np.zeros(288)
    p = np.zeros(288)
    p[10:14] = np.nan
    assert false_alarms_per_day(y, p, threshold=0.5) == 0.0


# --------------------------------------------------------------------------- #
def _ramp(n=60, nadir_at=40, floor=50.0):
    """A single clean descent to a nadir and back, in mg/dL."""
    bg = np.full(n, 140.0)
    bg[:nadir_at + 1] = np.linspace(140.0, floor, nadir_at + 1)
    bg[nadir_at:] = np.linspace(floor, 140.0, n - nadir_at)
    return bg


def test_lead_time_is_exact_on_a_known_ramp():
    """PH = 30 min = 6 steps. An alert at index nadir-4 is 20 minutes of warning.

    The valid-alert window is [nadir - 6, nadir - 1]: the label at anchor `a`
    covers (a, a+PH], so those are exactly the anchors whose prediction window
    contains the nadir.
    """
    bg = _ramp(nadir_at=40)
    score = np.zeros(bg.size)
    score[36] = 0.9                      # nadir - 4 -> 20 minutes of lead

    rep = effective_lead_time_minutes(bg, score, threshold=0.5, horizon_steps=6)
    assert rep.n_events == 1 and rep.n_detected == 1
    assert rep.mean_min == pytest.approx(20.0)
    assert rep.median_min == pytest.approx(20.0)
    assert rep.detection_rate == pytest.approx(1.0)


def test_lead_time_takes_the_earliest_alert_in_the_window():
    """A latched alarm gives credit for when it FIRST fired, not for its last
    step -- otherwise a model that alarms continuously would score a lead time
    of one step."""
    bg = _ramp(nadir_at=40)
    score = np.zeros(bg.size)
    score[35:40] = 0.9                   # fires at nadir - 5 and stays on
    rep = effective_lead_time_minutes(bg, score, 0.5, horizon_steps=6)
    assert rep.mean_min == pytest.approx(25.0)


def test_an_alert_at_the_nadir_is_not_a_warning():
    """An alert at the nadir itself has zero lead time and could not have
    prompted any action; counting it would inflate the detection rate with
    alarms that arrive after the fact."""
    bg = _ramp(nadir_at=40)
    score = np.zeros(bg.size)
    score[40] = 0.9
    rep = effective_lead_time_minutes(bg, score, 0.5, horizon_steps=6)
    assert rep.n_events == 1 and rep.n_detected == 0
    assert np.isnan(rep.mean_min)
    assert rep.detection_rate == 0.0


def test_an_alert_older_than_the_horizon_does_not_count():
    """PH = 30 min bounds how far ahead the model even claims to see. An alert
    7 steps before the nadir predicted a window that closed before it."""
    bg = _ramp(nadir_at=40)
    score = np.zeros(bg.size)
    score[33] = 0.9                      # nadir - 7, outside a 6-step horizon
    rep = effective_lead_time_minutes(bg, score, 0.5, horizon_steps=6)
    assert rep.n_detected == 0


def test_lead_time_to_onset_is_shorter_than_to_nadir():
    """Two different clinical questions: time until treatment is indicated
    versus time until the patient is at their worst. The onset comes first, so
    its lead time is the smaller of the two by construction."""
    bg = _ramp(nadir_at=40, floor=50.0)
    onset = int(np.flatnonzero(bg < 70.0)[0])
    score = np.zeros(bg.size)
    score[onset - 3] = 0.9
    score[36] = 0.9

    rep = effective_lead_time_minutes(bg, score, 0.5, horizon_steps=6)
    assert rep.mean_onset_lead_min == pytest.approx(15.0)
    assert rep.mean_onset_lead_min < rep.mean_min


def test_two_events_are_counted_separately_and_averaged():
    bg = np.concatenate([_ramp(60, 40, 50.0), np.full(20, 140.0),
                         _ramp(60, 30, 55.0)])
    score = np.zeros(bg.size)
    score[36] = 0.9                        # 20 min before the first nadir
    score[80 + 30 - 2] = 0.9               # 10 min before the second
    rep = effective_lead_time_minutes(bg, score, 0.5, horizon_steps=6)
    assert rep.n_events == 2 and rep.n_detected == 2
    assert rep.mean_min == pytest.approx(15.0)
    assert rep.median_min == pytest.approx(15.0)


def test_an_alert_across_a_recording_gap_cannot_warn_about_the_event():
    """An alarm on the far side of a sensor dropout warned nobody about this
    event: the device was not reporting in between."""
    bg = np.array([140.0] * 10 + [60.0, 55.0, 60.0] + [140.0] * 10)
    score = np.zeros(bg.size)
    score[8] = 0.9                          # 15 min before the nadir at index 11
    seg = np.zeros(bg.size, dtype=np.int64)
    seg[9:] = 1                             # dropout between the alert and the event

    plain = effective_lead_time_minutes(bg, score, 0.5, 6)
    assert plain.n_events == 1 and plain.n_detected == 1
    assert plain.mean_min == pytest.approx(15.0)

    gapped = effective_lead_time_minutes(bg, score, 0.5, 6, segment=seg)
    assert gapped.n_events == 1 and gapped.n_detected == 0


def test_lead_time_on_a_series_with_no_events_is_undefined():
    bg = np.full(50, 140.0)
    rep = effective_lead_time_minutes(bg, np.ones(50), 0.5, horizon_steps=6)
    assert rep.n_events == 0 and np.isnan(rep.mean_min)
    assert np.isnan(effective_lead_time_minutes(
        np.array([]), np.array([]), 0.5, 6).mean_min)


def test_lead_time_rejects_a_non_positive_horizon():
    with pytest.raises(ValueError, match="horizon_steps"):
        effective_lead_time_minutes(np.full(10, 60.0), np.ones(10), 0.5, 0)


def test_hypo_events_ignore_missing_readings():
    """A dropout is not evidence of hypoglycemia; imputing one would fabricate
    the very events the lead-time metric is scored against."""
    bg = np.array([120.0, np.nan, np.nan, 120.0, 60.0, 55.0, 120.0])
    ev = hypo_events(bg)
    assert len(ev) == 1
    assert (ev[0].start, ev[0].stop, ev[0].nadir) == (4, 6, 5)
    assert ev[0].nadir_bg == pytest.approx(55.0)


def test_hypo_events_can_require_a_minimum_duration():
    """The consensus definition counts an episode only after 15 minutes below
    threshold; the default is permissive, and asking for the rule must work."""
    bg = np.array([120.0, 65.0, 120.0, 60.0, 58.0, 59.0, 120.0])
    assert len(hypo_events(bg, min_steps=1)) == 2
    assert len(hypo_events(bg, min_steps=3)) == 1


def test_alarm_episodes_and_hypo_events_share_one_run_extractor():
    """One definition of 'contiguous run', used by both metrics. Two would
    drift, and the drift would be invisible in every number they produce."""
    mask = np.array([0, 1, 1, 0, 1], bool)
    assert alarm_episodes(mask) == runs_of_true(mask)


# --------------------------------------------------------------------------- #
def test_threshold_at_fpr_is_achievable_and_conservative():
    """The reported FAPD has to be measured at a threshold a device could
    actually be set to, so this returns a real order statistic and never
    overshoots the target FPR."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900, int), np.ones(100, int)])
    s = np.concatenate([rng.beta(2, 8, 900), rng.beta(8, 2, 100)])

    for target in (0.05, 0.10):
        t = threshold_at_fpr(y, s, target)
        realised = float(((s >= t) & (y == 0)).mean() * (1000 / 900))
        assert realised <= target + 1e-9, f"FPR {realised:.4f} exceeds {target}"

    assert np.isnan(threshold_at_fpr(np.zeros(10, int), np.arange(10.0), 0.05))


def test_eval_report_carries_the_alarm_block():
    """The metrics are only useful if they reach the report every driver prints."""
    rng = np.random.default_rng(1)
    n = 576
    bg = 120 + 65 * np.sin(np.arange(n) / 11.0)   # dips below 70 several times
    y_reg = np.roll(bg, -6)
    y_cls = (y_reg < 70).astype(int)
    score = 1.0 / (1.0 + np.exp((y_reg - 70) / 8.0)) + rng.normal(0, 0.05, n)
    t = np.arange(n).astype("timedelta64[m]") * 5 + np.datetime64("2021-01-01T00:00")

    # Operating points are supplied, never fitted inside `evaluate`. Fitting
    # them on these same rows is fine for a unit test of the alarm block.
    thr = calibration_thresholds(y_cls, score)
    rep = evaluate(y_reg, y_reg, y_cls, score, bg,
                   np.array(["p1"] * n, dtype=object), 5,
                   t_time=t, horizon_min=30, alarm_thresholds=thr)

    assert isinstance(rep, EvalReport)
    assert np.isfinite(rep.sensitivity_at_5pct_fpr)
    assert rep.sensitivity_at_5pct_fpr == rep.sens_at_5pct_fpr
    assert np.isfinite(rep.fapd_at_5pct_fpr) and rep.fapd_at_5pct_fpr >= 0.0
    assert rep.fapd_at_10pct_fpr >= rep.fapd_at_5pct_fpr, \
        "a looser FPR budget cannot produce FEWER alarms"
    assert rep.sensitivity_at_10pct_fpr >= rep.sensitivity_at_5pct_fpr
    assert rep.n_hypo_events > 0
    assert rep.monitored_days == pytest.approx(2.0)
    assert "ALRM" in str(rep)


def test_lead_time_stays_nan_without_a_horizon():
    """`evaluate` must not guess PH: a lead time computed against the wrong
    horizon is worse than an absent one."""
    n = 200
    bg = 120 + 65 * np.sin(np.arange(n) / 9.0)
    y_reg = np.roll(bg, -6)
    y_cls = (y_reg < 70).astype(int)
    score = 1.0 - y_reg / 300.0
    rep = evaluate(y_reg, y_reg, y_cls, score, bg,
                   np.array(["p1"] * n, dtype=object), 5,
                   alarm_thresholds=calibration_thresholds(y_cls, score))
    assert np.isfinite(rep.fapd_at_5pct_fpr), "FAPD needs no horizon"
    assert np.isnan(rep.mean_lead_time_min)
    assert rep.n_hypo_events == 0


def test_alarm_metrics_are_undefined_for_a_single_class_slice():
    """No negatives means no false-positive rate, so every operating point that
    is defined in terms of one is undefined too."""
    n = 100
    y = np.ones(n, int)
    rep = evaluate(np.full(n, 60.0), np.full(n, 60.0), y, np.full(n, 0.9),
                   np.full(n, 65.0), np.array(["p"] * n, dtype=object), 5,
                   horizon_min=30)
    assert np.isnan(rep.fapd_at_5pct_fpr)
    assert np.isnan(rep.sensitivity_at_5pct_fpr)


# --------------------------------------------------------------------------- #
# 6. L2-SP anchoring for LoRA
# --------------------------------------------------------------------------- #
def test_lora_delta_w_is_exactly_zero_at_initialisation():
    """B is zero-initialised, so dW = 0 and the adapted model IS the base model.

    Asserted on the weight update itself rather than only on a forward pass:
    a forward pass can agree to float tolerance while dW is merely small, and
    "merely small" would mean every personalisation run starts by damaging the
    population model.
    """
    m = _mm()
    inject_lora(m, AdaptConfig())
    adapters = [x for x in m.modules() if isinstance(x, (LoRAConv1d, LoRALinear))]
    assert adapters, "no adapters injected"
    for a in adapters:
        d = lora_delta_w(a)
        assert d.shape[0] == a.base.weight.shape[0]
        assert float(d.detach().abs().max()) == 0.0, f"{type(a).__name__} starts with dW != 0"
    assert float(lora_delta_norm_sq(m).detach()) == 0.0


def test_lora_l2_sp_penalty_is_zero_at_init_and_grows_with_the_update():
    """L2-SP anchors to the POPULATION weights, and for LoRA ||W - W0||^2 is
    exactly ||dW||^2. The penalty must therefore start at zero and rise as the
    adapter moves -- if it were computed on the frozen base weights it would be
    identically zero forever, which is the bug this replaced."""
    m = _mm()
    configure_trainable(m, AdaptConfig(method="lora"))
    anchor = {k: v.detach().clone() for k, v in m.state_dict().items()}
    dev = torch.device("cpu")

    assert float(l2_sp_penalty(m, anchor, "lora", dev).detach()) == 0.0
    for a in m.modules():
        if isinstance(a, (LoRAConv1d, LoRALinear)):
            with torch.no_grad():
                a.lora_b.weight.add_(0.01)
    assert float(l2_sp_penalty(m, anchor, "lora", dev).detach()) > 0.0


def test_lora_delta_w_matches_a_brute_force_forward_difference():
    """The closed form must equal what the layer actually adds. Checked against
    the layer's own forward on a one-hot input, so a transposed einsum cannot
    pass by symmetry."""
    base = torch.nn.Linear(6, 4, bias=False)
    lin = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        lin.lora_b.weight.normal_()
    x = torch.eye(6)
    observed = (lin(x) - lin.base(x)).T          # columns are dW @ e_i
    torch.testing.assert_close(observed, lora_delta_w(lin), atol=1e-6, rtol=1e-5)


def test_time_lag_is_quiet_on_a_constant_tail():
    """A flat stretch inside a segment makes one lagged slice constant, and
    `np.corrcoef` then divides by zero, warns, and returns nan.

    The result was always correct; the warnings were the problem -- a long run
    emitted thousands of them, which is how a real warning gets missed.
    """
    import warnings

    from causalcgm_mt.evaluate import time_lag_minutes

    y = np.concatenate([120 + 40 * np.sin(np.arange(60) / 7.0), np.full(20, 100.0)])
    p = np.concatenate([np.full(20, 100.0), 120 + 40 * np.sin(np.arange(60) / 7.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lag = time_lag_minutes(y, p, 5)
    assert np.isfinite(lag) and lag >= 0.0


def test_time_lag_still_returns_nan_for_a_wholly_constant_series():
    from causalcgm_mt.evaluate import time_lag_minutes
    assert np.isnan(time_lag_minutes(np.full(60, 100.0), np.arange(60.0), 5))
    assert np.isnan(time_lag_minutes(np.arange(60.0), np.full(60, 100.0), 5))
