"""Executable audit of the soft-label band mechanism (audit finding F10).

WHAT F10 WAS
------------
`band_mode="mard"` was unusable on real data. The estimator was a median
absolute second difference of the CGM stream, which measures the signal's
high-frequency content -- and the released OhioT1DM stream has been through the
manufacturer's smoothing filter, so there is almost none left. It read 0.7%
against a sensor whose published MARD is 9-14%, produced a +/-0.5 mg/dL band,
emitted a `UserWarning`, and carried on. Every driver therefore pinned
`band_mode="fixed"`, the estimate was computed and discarded, and the mode was
never ablated.

WHAT THESE TESTS PIN
--------------------
1. The band lands in a medically meaningful window (5-20 mg/dL) on BOTH the
   synthetic cohort and real OhioT1DM -- the property F10 was about.
2. The three estimator tiers each do their job, and each says which one ran.
   A floored band and a measured band are different claims about the data.
3. Neither mode warns, and neither falls back silently: `band_source` is on the
   object and in the results JSON, which is what makes the mode a sweepable
   ablation arm rather than a setting nobody dares turn on.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import DataConfig
from causalcgm_mt.data import (
    SMOOTHED_SIGNAL_RHO1,
    NoiseEstimate,
    SoftLabeler,
    estimate_mard_proxy,
    estimate_reference_mard,
    estimate_signal_noise,
)

REAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "OhioT1DM")
HAS_REAL = os.path.isdir(REAL_ROOT)

# The window the whole finding is about. Below it the logistic ramp saturates
# inside one reportable CGM unit and the "soft" labels are hard ones; above it
# the ramp reaches from 70 mg/dL well into euglycemia.
BAND_MIN_MGDL, BAND_MAX_MGDL = 5.0, 20.0


def _band(noise: NoiseEstimate, **cfg_over) -> SoftLabeler:
    cfg = DataConfig(band_mode="mard", **cfg_over)
    return SoftLabeler.fit(np.array([100.0]), cfg, noise=noise)


# --------------------------------------------------------------------------- #
# 1. The headline property: a usable band on both cohorts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ohio_reference():
    """Paired (CGM, finger-stick) readings over the real training files."""
    from causalcgm_mt.ohio_loader import collect_reference_pairs, discover_records

    return collect_reference_pairs(discover_records(REAL_ROOT))


@pytest.fixture(scope="module")
def synthetic_series() -> list[np.ndarray]:
    """The repository's own synthetic cohort, at the CGM cadence."""
    from causalcgm_mt.simulate import simulate_hupa_cohort

    df = simulate_hupa_cohort(n_patients=6, days=5.0, seed=0)
    out = []
    for _, g in df.groupby("patient_id", sort=True):
        v = g["glucose"].to_numpy(dtype=float)
        out.append(v[np.isfinite(v)])
    return out


@pytest.mark.skipif(not HAS_REAL, reason="OhioT1DM not present")
def test_mard_on_real_ohio_yields_a_medically_meaningful_band(ohio_reference):
    """The F10 acceptance test on real data.

    The band must sit inside 5-20 mg/dL, and on this cohort it must get there by
    MEASUREMENT -- OhioT1DM ships `finger_stick` events, so the paired reference
    that MARD is defined against is in the same XML files the CGM comes from.
    Landing in range via the device floor would still be a usable band, but it
    would mean the estimator never found the reference, so the source is
    asserted too.
    """
    est = estimate_reference_mard(ohio_reference.cgm, ohio_reference.reference,
                                  ohio_reference.patient)
    assert est.source == "paired_reference"
    assert est.n > 1000, "expected thousands of paired readings across 12 subjects"
    assert 0.05 < est.mard < 0.25, (
        f"a cohort MARD of {est.mard:.1%} is outside anything a released CGM "
        f"reports; the pairing or the screen is wrong")

    lab = _band(est)
    assert BAND_MIN_MGDL <= lab.band <= BAND_MAX_MGDL, (
        f"band {lab.band:.2f} mg/dL is outside the medically meaningful window "
        f"-- this is the F10 collapse")
    assert lab.band_source == "mard", (
        f"expected a measured band, got {lab.band_source!r}")
    assert not lab.is_effectively_hard


def test_mard_on_synthetic_yields_a_medically_meaningful_band(synthetic_series):
    """The same acceptance test on the synthetic cohort.

    The simulator's noise is a knob (`rng.normal(0, 2.5)`), not a device
    specification, and it is set well below any real sensor -- so the high-pass
    tier measures it correctly and the result is still too narrow to be a
    clinically meaningful ambiguity band. The device floor is what carries it
    into range, and it says so. That is the intended behaviour: a band narrower
    than the smallest error any real CGM makes is a property of the generator,
    not of the data.
    """
    est = estimate_signal_noise(synthetic_series, period_min=5)
    assert est.source == "high_pass"

    lab = _band(est)
    assert BAND_MIN_MGDL <= lab.band <= BAND_MAX_MGDL, (
        f"band {lab.band:.2f} mg/dL is outside the medically meaningful window")
    assert lab.band_source == "device_floor"
    assert lab.mard_used == pytest.approx(DataConfig().band_mard_floor)
    assert not lab.is_effectively_hard


# --------------------------------------------------------------------------- #
# 2. The estimator tiers
# --------------------------------------------------------------------------- #
def test_high_pass_estimator_is_unbiased_on_known_noise():
    """The IQR form must recover a KNOWN magnitude, as the median form did.

    Guards the swap itself: `IQR(d2) / (1.3490 * sqrt(6))` and
    `median(|d2|) / (0.6745 * sqrt(6))` are both sigma for Gaussian noise, and
    getting either constant wrong is invisible downstream -- the original bug
    (a missing 0.6745) understated cohort noise by a third for months.
    """
    rng = np.random.default_rng(0)
    for true_rel in (0.02, 0.05, 0.10):
        s = 140.0 + rng.normal(0, 140.0 * true_rel, 100_000)
        est = estimate_signal_noise([s], period_min=5)
        assert est.mard == pytest.approx(true_rel, rel=0.05)
        assert estimate_mard_proxy([s], period_min=5) == pytest.approx(true_rel, rel=0.05)


def test_high_pass_estimator_reports_when_the_signal_has_been_smoothed():
    """`rho1` is the identifiability diagnostic, and it must separate the cases.

    For (smooth trend + i.i.d. noise), the lag-1 autocorrelation of the first
    difference tends to -0.5 as the noise dominates. Once a low-pass filter has
    removed the noise there is nothing left to estimate, and rho1 rises. Without
    this, a 0.7% reading is indistinguishable from a genuinely quiet sensor.
    """
    rng = np.random.default_rng(1)
    trend = 140 + 30 * np.sin(np.linspace(0, 12, 4000))
    noisy = trend + rng.normal(0, 14.0, 4000)              # 10% of 140
    smoothed = np.round(np.convolve(noisy, np.ones(9) / 9, mode="same"))

    raw = estimate_signal_noise([noisy], period_min=5)
    filt = estimate_signal_noise([smoothed], period_min=5)

    assert raw.rho1 < SMOOTHED_SIGNAL_RHO1 and not raw.smoothed
    assert filt.rho1 > SMOOTHED_SIGNAL_RHO1 and filt.smoothed
    assert raw.mard == pytest.approx(0.10, rel=0.10)
    assert filt.mard < 0.02, "smoothing destroys the evidence, by construction"


@pytest.mark.skipif(not HAS_REAL, reason="OhioT1DM not present")
def test_the_released_ohio_signal_is_why_the_paired_tier_exists():
    """Documents the measurement that made F10 unfixable inside the old design.

    The released stream reads as smoothed on the same diagnostic as the
    synthetic control above, and its high-pass noise estimate is an order of
    magnitude below the sensor's published MARD. No amount of care in a
    high-pass estimator recovers a signal component that was filtered out before
    the file was written -- which is why the fix was to use the reference BG
    that ships alongside it, not to tune the proxy.
    """
    from causalcgm_mt.ohio_loader import parse_patient_xml

    series = []
    for sub, pid in (("2018", "559"), ("2018", "563"), ("2020", "540")):
        rec = parse_patient_xml(
            os.path.join(REAL_ROOT, sub, "train", f"{pid}-ws-training.xml"))
        v = rec.get("glucose_level")["value"].to_numpy(dtype=float)
        series.append(v[np.isfinite(v)])

    est = estimate_signal_noise(series, period_min=5)
    assert est.smoothed, "released OhioT1DM CGM is expected to read as smoothed"
    assert est.rho1 > 0.2, f"expected a strongly trend-dominated rho1, got {est.rho1}"
    assert est.mard < 0.02, (
        "if the high-pass tier ever recovers a realistic MARD from this stream, "
        "the released data changed and the two-tier design should be revisited")


def test_reference_screen_rejects_the_no_value_sentinel():
    """A reference of 0.0 is a sentinel, not a hypoglycemic reading.

    OhioT1DM subject 540 records 143 finger sticks as exactly 0.0. Dividing by
    them takes that subject's MEAN absolute relative deviation to 4022% and the
    pooled cohort mean to 642%. Both halves of the fix are load-bearing and are
    checked here: the plausibility screen, and the use of medians throughout.
    """
    cgm = np.concatenate([np.full(100, 110.0), np.full(20, 110.0)])
    ref = np.concatenate([np.full(100, 100.0), np.zeros(20)])   # 20 sentinels

    est = estimate_reference_mard(cgm, ref)
    assert est.n == 100, "sentinel rows must not reach the statistic"
    # median ARD is exactly 10%, scaled onto the MARD (mean) convention.
    assert est.mard == pytest.approx(0.10 * 1.1829, rel=1e-3)
    assert "screened" in est.detail


def test_reference_mard_is_not_dominated_by_the_best_instrumented_subject():
    """Per-subject medians first, then across subjects.

    Pair counts here range from 73 to 557 per subject. Pooling would let one
    subject's sensor set the whole cohort's label band.
    """
    rng = np.random.default_rng(2)
    ref = np.concatenate([np.full(2000, 100.0), np.full(50, 100.0)])
    cgm = np.concatenate([np.full(2000, 102.0), np.full(50, 130.0)])
    sub = np.array(["loud"] * 2000 + ["quiet"] * 50, dtype=object)
    _ = rng  # deterministic fixture; rng kept for symmetry with the suite

    pooled = estimate_reference_mard(cgm, ref)
    grouped = estimate_reference_mard(cgm, ref, sub)
    assert pooled.mard == pytest.approx(0.02 * 1.1829, rel=1e-3)
    # The two subjects' medians are 2% and 30%; their median is the midpoint.
    assert grouped.mard == pytest.approx(0.16 * 1.1829, rel=1e-3)


# --------------------------------------------------------------------------- #
# 3. The band policy: floor, clamp, and provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mard, expect_band, expect_source",
    [
        (0.0007, 6.3, "device_floor"),   # real Ohio via the old proxy
        (0.020, 6.3, "device_floor"),    # the synthetic cohort
        (0.105, 7.35, "mard"),           # real Ohio via paired reference
        (0.150, 10.5, "mard"),
        (0.400, 20.0, "clamped_high"),   # a corrupted estimate
    ],
)
def test_band_policy_is_a_total_function_of_the_estimate(mard, expect_band, expect_source):
    """Every estimate maps to a usable band, and every band names its origin."""
    lab = _band(NoiseEstimate(mard=mard, source="high_pass"))
    assert lab.band == pytest.approx(expect_band, abs=1e-6)
    assert lab.band_source == expect_source
    assert BAND_MIN_MGDL <= lab.band <= BAND_MAX_MGDL


def test_an_unavailable_estimate_still_produces_a_band():
    """No data is a reason to fall back to the prior, not to crash or to warn."""
    lab = _band(NoiseEstimate(mard=float("nan"), source="unavailable"))
    assert lab.band_source == "device_floor"
    assert BAND_MIN_MGDL <= lab.band <= BAND_MAX_MGDL


def test_neither_band_mode_warns():
    """The contract F10 asked for: no silent fallback AND no warning.

    A warning is not a mechanism. This one made the mard path something to be
    suppressed rather than something to be swept, and it is replaced by
    `band_source` -- which a driver can branch on and a results file can carry.
    """
    for cfg in (DataConfig(band_mode="mard"),
                DataConfig(band_mode="mard", band_mard_multiplier=4.0),
                DataConfig(band_mode="fixed", soft_band_mgdl=10.0),
                DataConfig(band_mode="fixed", soft_band_mgdl=0.5),
                DataConfig(soft_labels=False)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.0007)


def test_band_provenance_round_trips_to_json():
    """`as_dict` is what puts the band into the results file; keep it complete."""
    import json

    est = NoiseEstimate(mard=0.105, source="paired_reference", n=3100,
                        level_mgdl=154.0, detail="x")
    d = SoftLabeler.fit(np.array([100.0]), DataConfig(band_mode="mard"),
                        noise=est).as_dict()
    assert json.loads(json.dumps(d))["band_source"] == "mard"
    assert d["noise"]["source"] == "paired_reference" and d["noise"]["n"] == 3100
    assert set(d) >= {"band_mgdl", "band_source", "band_raw_mgdl", "mard_used",
                      "threshold_mgdl", "noise"}


# --------------------------------------------------------------------------- #
# 4. Sweepability -- the reason any of this was worth doing
# --------------------------------------------------------------------------- #
def test_the_two_band_modes_produce_genuinely_different_labels():
    """A sweep whose arms agree is not a sweep.

    The old mard path collapsed onto the hard-label arm, so an ablation over
    `band_mode` would have compared a control against a copy of itself. This is
    the check that the treatment arm is a treatment.
    """
    bg = np.array([55.0, 65.0, 69.0, 70.0, 72.0, 78.0, 90.0], dtype=np.float32)
    hard = SoftLabeler.fit(bg, DataConfig(soft_labels=False))(bg)
    mard = SoftLabeler.fit(bg, DataConfig(band_mode="mard"),
                           noise=NoiseEstimate(mard=0.105, source="paired_reference"))(bg)
    fixed = SoftLabeler.fit(bg, DataConfig(band_mode="fixed", soft_band_mgdl=10.0))(bg)

    assert np.abs(mard - hard).max() > 0.2, "mard arm must differ from the control"
    assert np.abs(mard - fixed).max() > 0.01, "the two soft arms must differ"
    for y in (mard, fixed):
        assert np.all(np.diff(y) <= 1e-6), "soft labels must decrease with glucose"


def test_arm_spec_carries_the_band_and_inherits_by_default():
    """`ArmSpec` is the sweep's unit; the knobs have to live on it.

    `None` means "inherit the cohort", so adding the fields cannot perturb any
    existing arm -- which is what lets Ablation E be added without re-running
    the benchmark tables.
    """
    from causalcgm_mt.benchmark import ArmSpec

    assert ArmSpec("base", "tcn_mm").soft_label_overrides == {}
    swept = ArmSpec("mard band", "tcn_mm", band_mode="mard")
    assert swept.soft_label_overrides == {"band_mode": "mard"}
    assert swept.as_dict()["band_mode"] == "mard"


@pytest.mark.skipif(not HAS_REAL, reason="OhioT1DM not present")
def test_reference_pairs_never_cross_the_fold_cut(ohio_reference):
    """The paired estimate is a fitted statistic and obeys the same contract.

    `collect_reference_pairs` matches a stick to the nearest CGM sample, which
    is correct for a two-sensor agreement statistic and would be a leak if the
    pairs were used unfiltered. `ReferencePairs.before` is where causality is
    enforced, so it is tested directly rather than trusted.
    """
    assert len(ohio_reference) > 0
    assert set(map(str, ohio_reference.split)) == {"train"}, \
        "test-file references must never be collected"

    pid = str(ohio_reference.patient[0])
    mine = ohio_reference.ts[ohio_reference.patient == pid]
    cut = np.sort(mine)[len(mine) // 2]

    kept = ohio_reference.before({pid: cut})
    assert len(kept) > 0
    assert set(map(str, kept.patient)) == {pid}, "patients without a cut are dropped"
    assert kept.ts.max() <= cut


# --------------------------------------------------------------------------- #
# 5. The reporting contract between `as_dict` and the drivers
# --------------------------------------------------------------------------- #
def test_every_arms_provenance_survives_the_driver_summary_line():
    """`run_paper_ablations` must be able to print every arm Ablation E runs.

    This is a regression test for a `TypeError` that reached a real run. The
    hard-label arm reports `band_mgdl = None` on purpose -- an arm with no ramp
    has no band -- and the driver formatted it with `:6.2f`, which raises. It
    survived review because `dict.get(key, default)` LOOKS like it handles the
    case: the key is present, so the default never fires, and the None goes
    straight to `format()`.

    The failure landed at the very end of the study, after every arm had been
    trained and the tables had been written, which is the worst place for a
    formatting bug to live. So the contract is tested here, at the point where
    the two sides meet, rather than only through a full run of the driver.
    """
    import sys as _sys

    _sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from run_paper_ablations import _num

    arms = {
        "hard": DataConfig(soft_labels=False, band_mode="fixed"),
        "fixed": DataConfig(band_mode="fixed", soft_band_mgdl=10.0),
        "mard": DataConfig(band_mode="mard"),
    }
    for name, cfg in arms.items():
        sl = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=0.1034).as_dict()
        line = (f"band {_num(sl.get('band_mgdl'), '6.2f')} mg/dL  "
                f"source={sl.get('band_source', '--')}  "
                f"mard={_num(sl.get('mard_used'), '6.4f')}")
        assert "None" not in line and "nan" not in line, \
            f"{name} arm renders a placeholder as text: {line!r}"

    # The hard arm is the one that broke: no band, and it must say so.
    hard = SoftLabeler.fit(np.array([100.0]), arms["hard"]).as_dict()
    assert hard["band_mgdl"] is None and hard["band_source"] == "disabled"
    assert _num(hard["band_mgdl"], "6.2f").strip() == "--"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.0, 7.35])
def test_the_driver_number_formatter_never_raises(value):
    """Any value the results JSON can hold must format, or the run dies at the
    end. `--` is the same placeholder `benchmark._cells` uses for the LaTeX
    table, so the log line and the table agree about an inapplicable cell."""
    import sys as _sys

    _sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from run_paper_ablations import _num

    out = _num(value, "6.2f")
    assert len(out) == 6
    assert out.strip() == "--" if (value is None or not np.isfinite(value)) \
        else float(out) == pytest.approx(value)
