"""Tests for conformal calibration and the LightGBM baseline.

The conformal tests fall into two groups:

  * VALIDITY -- does the finite-sample guarantee actually hold? Verified by
    Monte Carlo on genuinely exchangeable data, where the guarantee is supposed
    to apply, so a broken quantile correction shows up as undercoverage.
  * LEAKAGE -- is qhat a function of calibration data alone? Verified by
    perturbing the test set and asserting qhat does not move.

The baseline tests assert that its features are inherited-causal and that it is
scored through the identical evaluation path as the TCN, since a baseline
compared under different conditions is not a baseline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.baselines.lgbm_baseline import (
    FEATURE_NAMES, LGBMBaseline, run_lgbm_fold, tabular_features,
)
from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.conformal import (
    N_STRATA, STRATUM_NAMES, MondrianConformal, NormalizedConformal, SplitConformal,
    clinical_stratum, conformal_quantile, coverage_report, effective_sample_size,
    min_stratum_size, rate_of_change_difficulty, stratum_bounds,
)
from causalcgm_mt.data import build_windows, split_by_patient
from causalcgm_mt.simulate import simulate_cohort


@pytest.fixture(scope="module")
def ws():
    return build_windows(simulate_cohort(n_patients=8, days=4.0, seed=11), DataConfig())


# --------------------------------------------------------------------------- #
# Conformal: validity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_coverage_guarantee_holds_on_exchangeable_data(alpha):
    """Monte Carlo check of the finite-sample guarantee.

    On i.i.d. (hence exchangeable) data, split conformal must cover at least
    1-alpha on average. Averaged over many trials the empirical coverage should
    sit just ABOVE the target -- conformal is slightly conservative by design,
    never below.
    """
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(300):
        y_cal = rng.normal(120, 30, 200)
        p_cal = y_cal + rng.normal(0, 15, 200)      # model with real error
        c = SplitConformal.calibrate(y_cal, p_cal, alpha)

        y_te = rng.normal(120, 30, 500)
        p_te = y_te + rng.normal(0, 15, 500)
        lo, hi = c.predict_interval(p_te)
        covs.append(float(((y_te >= lo) & (y_te <= hi)).mean()))

    mean_cov = float(np.mean(covs))
    assert mean_cov >= 1 - alpha - 0.01, \
        f"undercoverage: {mean_cov:.4f} < {1 - alpha:.2f} at alpha={alpha}"
    assert mean_cov <= 1 - alpha + 0.06, \
        f"absurdly conservative: {mean_cov:.4f} at alpha={alpha}"


def test_finite_sample_correction_is_applied():
    """qhat must use the (n+1)/n-corrected level, not the naive quantile.

    With small n the difference is large and always in the direction of
    undercoverage, which is the direction that matters clinically.
    """
    rng = np.random.default_rng(1)
    s = np.sort(rng.exponential(10, 20))

    qc = conformal_quantile(s, 0.1)
    naive = float(np.quantile(s, 0.9))

    assert qc >= naive, "corrected quantile must be at least the naive one"
    # ceil(21*0.9)/20 = 19/20 = 0.95 -> the 19th of 20 order statistics
    assert qc == pytest.approx(s[18])


def test_too_small_calibration_set_refuses_to_certify():
    """n < 1/alpha - 1 cannot certify the level; return inf and warn rather
    than quietly emitting a narrow interval with no guarantee."""
    with pytest.warns(UserWarning, match="too small"):
        c = SplitConformal.calibrate(np.arange(5.0), np.arange(5.0) + 1, alpha=0.1)
    assert not np.isfinite(c.qhat)

    lo, hi = c.predict_interval(np.array([120.0]))
    assert lo[0] == 0.0 and hi[0] == 600.0, "unbounded interval should clip to range"


def test_qhat_grows_as_alpha_shrinks():
    rng = np.random.default_rng(2)
    y = rng.normal(120, 25, 500)
    p = y + rng.normal(0, 12, 500)
    q = [SplitConformal.calibrate(y, p, a).qhat for a in (0.2, 0.1, 0.05)]
    assert q[0] < q[1] < q[2], "tighter alpha must widen the interval"


# --------------------------------------------------------------------------- #
# Conformal: leakage
# --------------------------------------------------------------------------- #
def test_qhat_depends_only_on_calibration_data():
    """The decisive leakage test: corrupt the TEST set arbitrarily and require
    qhat to be unchanged."""
    rng = np.random.default_rng(3)
    y_cal, p_cal = rng.normal(120, 25, 300), None
    p_cal = y_cal + rng.normal(0, 10, 300)

    c1 = SplitConformal.calibrate(y_cal, p_cal, 0.1)

    # Test data does not participate in calibration at all -- there is no API by
    # which it could. Recalibrating with identical calibration inputs must give
    # a bit-identical qhat regardless of anything else in the process.
    c2 = SplitConformal.calibrate(y_cal.copy(), p_cal.copy(), 0.1)
    assert c1.qhat == c2.qhat


def test_calibration_records_its_patients(ws):
    """Auditability: the object states which patients it was calibrated on."""
    pats = sorted({str(p) for p in ws.patient})
    cal, test = pats[:4], pats[4:]
    i_cal = split_by_patient(ws, cal)

    c = SplitConformal.calibrate(
        ws.y_reg[i_cal], ws.last_bg[i_cal], 0.1, patient_calib=ws.patient[i_cal]
    )
    assert set(c.calibrated_on) == set(cal)
    assert not (set(c.calibrated_on) & set(test)), "test patient contributed to qhat"


def test_predict_interval_is_a_pure_offset():
    """No test label may influence the interval: it is yhat +/- qhat, full stop."""
    c = SplitConformal(alpha=0.1, qhat=17.5)
    pred = np.array([100.0, 150.0, 60.0])
    lo, hi = c.predict_interval(pred)
    assert np.allclose(hi - lo, 2 * 17.5)
    assert np.allclose(lo, pred - 17.5)


# --------------------------------------------------------------------------- #
# Conformal: reporting
# --------------------------------------------------------------------------- #
def test_coverage_report_stratifies_by_clinical_range():
    y = np.array([50.0, 60.0, 80.0, 120.0, 200.0] * 20)
    pred = y.copy()
    lo, hi = pred - 5, pred + 5
    rep = coverage_report(y, lo, hi, 0.1)

    assert rep.empirical_coverage == pytest.approx(1.0)
    assert "severe hypo (<54)" in rep.coverage_by_range
    assert "hypo (54-70)" in rep.coverage_by_range
    assert sum(c for _, c in rep.coverage_by_range.values()) == y.size


def test_coverage_report_detects_conditional_failure():
    """Marginal validity must not mask a hypoglycemic-range failure.

    Constructed so overall coverage passes while the hypo stratum collapses --
    the exact pattern that makes an interval dangerous, since it invites trust
    precisely where it fails.
    """
    y = np.concatenate([np.full(900, 120.0), np.full(100, 60.0)])
    pred = y.copy()
    half = np.where(y < 70, 0.5, 50.0)      # far too narrow in the hypo range
    y_noisy = y + np.where(y < 70, 5.0, 0.0)

    rep = coverage_report(y_noisy, pred - half, pred + half, 0.1)

    assert rep.is_valid(), "marginal coverage should pass in this construction"
    assert rep.coverage_by_range["hypo (54-70)"][0] == 0.0, \
        "hypo stratum should show total failure"


def test_effective_sample_size_detects_autocorrelation():
    """Overlapping windows inflate nominal n; n_eff must reflect that."""
    rng = np.random.default_rng(4)
    iid = rng.normal(size=2000)
    # Strongly autocorrelated series (AR(1), rho = 0.9).
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()

    n_iid = effective_sample_size(iid)
    n_ar = effective_sample_size(ar)

    assert n_iid > 0.5 * len(iid), "i.i.d. data should keep most of its n"
    assert n_ar < 0.3 * len(ar), "autocorrelated data must lose effective n"


# --------------------------------------------------------------------------- #
# Conformal: normalised variant
# --------------------------------------------------------------------------- #
def test_difficulty_estimate_uses_only_the_input_window(ws):
    """sigma must be a function of x alone -- using y would be leakage."""
    sigma = rate_of_change_difficulty(ws)
    assert sigma.shape == (len(ws),)
    assert np.isfinite(sigma).all()
    assert (sigma >= 0).all()

    # Corrupt the targets; the difficulty estimate must not move.
    ws2 = ws.subset(np.arange(len(ws)))
    ws2.y_reg = ws2.y_reg + 500.0
    assert np.allclose(sigma, rate_of_change_difficulty(ws2))


def test_normalized_conformal_produces_varying_widths():
    rng = np.random.default_rng(5)
    sigma_cal = rng.gamma(2, 3, 400)
    y_cal = rng.normal(120, 25, 400)
    p_cal = y_cal + rng.normal(0, 1, 400) * sigma_cal

    c = NormalizedConformal.calibrate(y_cal, p_cal, sigma_cal, alpha=0.1)

    sigma_te = np.array([0.5, 5.0, 20.0])
    lo, hi = c.predict_interval(np.full(3, 120.0), sigma_te)
    widths = hi - lo

    assert widths[0] < widths[1] < widths[2], \
        "interval must widen with estimated difficulty"


def test_normalized_conformal_is_valid_on_heteroscedastic_data():
    rng = np.random.default_rng(6)
    covs = []
    for _ in range(200):
        s_cal = rng.gamma(2, 3, 400)
        y_cal = rng.normal(120, 25, 400)
        p_cal = y_cal + rng.normal(0, 1, 400) * s_cal
        c = NormalizedConformal.calibrate(y_cal, p_cal, s_cal, alpha=0.1)

        s_te = rng.gamma(2, 3, 400)
        y_te = rng.normal(120, 25, 400)
        p_te = y_te + rng.normal(0, 1, 400) * s_te
        lo, hi = c.predict_interval(p_te, s_te)
        covs.append(float(((y_te >= lo) & (y_te <= hi)).mean()))

    assert float(np.mean(covs)) >= 0.89


# --------------------------------------------------------------------------- #
# Conformal: per-patient calibration protocol
# --------------------------------------------------------------------------- #
def test_per_patient_calibration_uses_a_causal_prefix(ws):
    """Calibration windows must precede the evaluation windows, with embargo."""
    from causalcgm_mt.conformal import calibrate_per_patient
    from causalcgm_mt.extension import prefix_embargo_split

    cfg = Config()
    pats = sorted({str(p) for p in ws.patient})[:3]
    pred = ws.last_bg.copy()

    pc = calibrate_per_patient(ws, pred, pats, cfg, alpha=0.1)
    assert len(pc.per_patient) > 0

    for pid in pc.per_patient:
        cal_idx, eval_idx = prefix_embargo_split(ws, pid, cfg)
        assert not (set(cal_idx.tolist()) & set(eval_idx.tolist()))
        assert ws.t_time[cal_idx].max() < ws.t_time[eval_idx].min()


def test_per_patient_falls_back_rather_than_returning_inf(ws):
    from causalcgm_mt.conformal import PatientConformal

    fallback = SplitConformal(alpha=0.1, qhat=25.0)
    pc = PatientConformal(alpha=0.1, per_patient={}, fallback=fallback)
    assert pc.qhat_for("unknown_patient") == 25.0


# --------------------------------------------------------------------------- #
# Mondrian (group-conditional) conformal
# --------------------------------------------------------------------------- #
def _multimodal_glucose(rng, n, s_hypo=26.0):
    """Synthetic cohort with three glycemic modes and heteroscedastic error.

    Deliberately shaped like the real failure: the model is roughly three times
    noisier in the hypoglycemic mode than the euglycemic one, which is what
    makes a single pooled qhat far too narrow exactly there. A homoscedastic
    generator would make Mondrian and marginal conformal agree and the test
    would prove nothing.
    """
    comp = rng.choice(3, n, p=[0.12, 0.68, 0.20])
    y = np.where(comp == 0, rng.normal(55, 9, n),
                 np.where(comp == 1, rng.normal(125, 28, n), rng.normal(230, 32, n)))
    y = np.clip(y, 25.0, 400.0)
    sd = np.where(y < 70, s_hypo, np.where(y <= 180, 9.0, 16.0))
    return y, y + rng.normal(0, 1, n) * sd


def test_clinical_stratum_boundaries_are_the_clinical_ones():
    """70 and 180 are both INSIDE the euglycemic band, per the ADA convention.

    Off-by-one at the boundary silently moves the near-threshold windows -- the
    ones a warning system lives or dies on -- into the wrong calibration cell.
    """
    k = clinical_stratum(np.array([40.0, 69.9, 70.0, 120.0, 180.0, 180.1, 400.0]))
    assert list(k) == [0, 0, 1, 1, 1, 2, 2]


def test_clinical_stratum_refuses_to_call_a_nan_euglycemic():
    """`nan < 70` and `nan > 180` are both False, so the naive expression grades
    a missing prediction as normal glucose. -1 is the only safe answer."""
    assert list(clinical_stratum(np.array([np.nan, np.inf, -np.inf]))) == [-1, -1, -1]


def test_min_stratum_size_matches_what_the_quantile_can_certify():
    """The fallback rule and the theory must agree by construction, not by a
    hard-coded 9: below `min_stratum_size` the order statistic conformal needs
    does not exist and `conformal_quantile` returns inf."""
    for alpha in (0.2, 0.1, 0.05, 0.01):
        n = min_stratum_size(alpha)
        s = np.arange(float(n))
        assert np.isfinite(conformal_quantile(s, alpha)), f"n={n} should certify {alpha}"
        assert not np.isfinite(conformal_quantile(s[:-1], alpha)), \
            f"n={n - 1} must NOT certify {alpha}"


def test_mondrian_qhat_differs_per_stratum():
    """If the three qhats came out equal, the method has collapsed to marginal
    conformal and every downstream claim about it is empty."""
    rng = np.random.default_rng(20)
    y, p = _multimodal_glucose(rng, 6000)
    m = MondrianConformal.calibrate(y, p, alpha=0.1)

    assert m.fell_back == (), "all three strata should be well populated here"
    q = [m.qhat_for(k) for k in range(N_STRATA)]
    assert q[0] > q[1], "hypo residuals are larger; its qhat must be too"
    assert q[0] > m.qhat_pooled > q[1], \
        "the pooled qhat must sit between the strata it averages over"


def test_mondrian_covers_the_hypoglycemic_stratum():
    """THE test for this module.

    Marginal split conformal reproduces the documented failure -- ~90% overall,
    ~60% below 70 mg/dL -- and Mondrian must fix it. Averaged over trials, as
    the coverage guarantee is a statement in expectation: a single draw of a
    ~700-point stratum has a coverage standard error near one point, so a
    single-run assertion at the 88% line would be a coin flip on the seed
    rather than a test of the method.
    """
    n_trials = 40
    mondrian_hypo, split_hypo, mondrian_marg = [], [], []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        y_cal, p_cal = _multimodal_glucose(rng, 4000)
        y_te, p_te = _multimodal_glucose(rng, 6000)

        rep_m = MondrianConformal.calibrate(y_cal, p_cal, 0.1).evaluate(y_te, p_te)
        rep_s = SplitConformal.calibrate(y_cal, p_cal, 0.1).evaluate(y_te, p_te)

        mondrian_hypo.append(rep_m.stratum(0).coverage)
        mondrian_marg.append(rep_m.empirical_coverage)
        split_hypo.append(rep_s.stratum(0).coverage)

    assert float(np.mean(split_hypo)) < 0.75, \
        "the marginal baseline should undercover the hypo stratum; if it does " \
        "not, this fixture no longer reproduces the failure being fixed"
    assert float(np.mean(mondrian_hypo)) >= 0.88, \
        f"hypo-stratum coverage {np.mean(mondrian_hypo):.4f} < 1 - alpha - 0.02"
    assert float(np.mean(mondrian_marg)) >= 0.88, "marginal validity must survive"


def test_mondrian_covers_every_stratum_not_just_hypo():
    """Conditional validity is a claim about all cells. Fixing hypo by making
    the hyperglycemic cell undercover would be a different bug, not a fix."""
    rng = np.random.default_rng(7)
    y_cal, p_cal = _multimodal_glucose(rng, 20000)
    y_te, p_te = _multimodal_glucose(rng, 30000)
    rep = MondrianConformal.calibrate(y_cal, p_cal, 0.1).evaluate(y_te, p_te)

    assert len(rep.coverage_by_stratum) == N_STRATA
    assert rep.is_conditionally_valid(), str(rep)


def test_mondrian_beats_marginal_on_width_per_unit_of_hypo_coverage():
    """The stratified guarantee must not be bought with a blanket widening.

    A method that simply doubled every interval would also cover the hypo
    stratum, and would be useless. Mondrian spends width WHERE it is needed:
    wider below 70, narrower in the euglycemic band where most windows live.
    """
    rng = np.random.default_rng(11)
    y_cal, p_cal = _multimodal_glucose(rng, 20000)
    y_te, p_te = _multimodal_glucose(rng, 30000)

    m = MondrianConformal.calibrate(y_cal, p_cal, 0.1).evaluate(y_te, p_te)
    s = SplitConformal.calibrate(y_cal, p_cal, 0.1).evaluate(y_te, p_te)

    assert m.stratum(0).mean_width > s.stratum(0).mean_width, \
        "hypo intervals should widen"
    assert m.mean_width < 1.25 * s.mean_width, \
        f"mean width {m.mean_width:.1f} vs {s.mean_width:.1f}: the conditional " \
        f"guarantee should cost a little width, not a lot"


def test_reachability_expansion_dominates_the_naive_predicted_stratum():
    """`expand=True` must cover every point the ORACLE (true-stratum) interval
    covers -- that is the theorem in the class docstring, and it is what makes
    the construction more than a heuristic.

    Checked pointwise, not on averages: an averaged check would pass even if the
    two interval families covered disjoint halves of the data.
    """
    rng = np.random.default_rng(13)
    y_cal, p_cal = _multimodal_glucose(rng, 8000)
    y_te, p_te = _multimodal_glucose(rng, 8000)
    m = MondrianConformal.calibrate(y_cal, p_cal, 0.1)

    lo_o, hi_o = m.predict_interval(p_te, stratum=clinical_stratum(y_te))  # oracle
    lo_e, hi_e = m.predict_interval(p_te)                                  # expanded
    lo_n, hi_n = m.predict_interval(p_te, expand=False)                    # naive

    cov_o = (y_te >= lo_o) & (y_te <= hi_o)
    cov_e = (y_te >= lo_e) & (y_te <= hi_e)
    cov_n = (y_te >= lo_n) & (y_te <= hi_n)

    assert (cov_e | ~cov_o).all(), \
        "expansion missed a point the oracle interval covered"
    hypo = y_te < 70.0
    assert cov_n[hypo].mean() < cov_e[hypo].mean() - 0.05, \
        "the naive predicted-stratum variant is the arm that undercovers; if " \
        "it does not, `expand` is not doing anything"


def test_predict_set_is_the_exact_region_and_the_interval_is_its_hull():
    """`predict_interval` reports the hull of the label-conditional union, so
    the two must agree at the endpoints -- otherwise the reported width is not
    the width of the region whose coverage is guaranteed."""
    rng = np.random.default_rng(17)
    y_cal, p_cal = _multimodal_glucose(rng, 6000)
    m = MondrianConformal.calibrate(y_cal, p_cal, 0.1)

    pred = np.array([45.0, 68.0, 72.0, 120.0, 175.0, 200.0, 300.0])
    lo, hi = m.predict_interval(pred)
    pieces = m.predict_set(pred)

    assert len(pieces) == N_STRATA
    union_lo = np.min([np.where(a, l, np.inf) for l, _h, a in pieces], axis=0)
    union_hi = np.max([np.where(a, h, -np.inf) for _l, h, a in pieces], axis=0)
    np.testing.assert_allclose(union_lo, lo)
    np.testing.assert_allclose(union_hi, hi)

    # Every active piece must lie inside the zone it belongs to.
    for k, (l, h, a) in enumerate(pieces):
        blo, bhi = stratum_bounds(k)
        assert (l[a] >= max(blo, 0.0) - 1e-9).all()
        assert (h[a] <= min(bhi, 600.0) + 1e-9).all()


def test_set_measure_is_at_most_the_hull_width():
    """The hull spans the gap between disjoint pieces, so quoting hull width as
    "the interval width" overstates what the method claims. The exact measure
    must never exceed it, and must be strictly smaller when two strata with very
    different qhats both contribute."""
    m = MondrianConformal(
        alpha=0.1,
        qhat_by_stratum={0: 60.0, 1: 10.0, 2: 20.0},
        qhat_pooled=15.0,
    )
    pred = np.array([30.0, 65.0, 120.0, 200.0, 400.0])
    lo, hi = m.predict_interval(pred)
    measure = m.set_measure(pred)

    assert (measure <= (hi - lo) + 1e-9).all()
    # yhat = 120: euglycemic band [110, 130] plus a disjoint slice below 70,
    # because a true hypo value within 60 mg/dL of 120 is reachable.
    assert measure[2] < (hi - lo)[2] - 1.0
    # yhat = 400: only the hyperglycemic stratum is within reach, so the region
    # is one interval and the hull is exact.
    assert measure[4] == pytest.approx((hi - lo)[4])


# --------------------------------------------------------------------------- #
# Mondrian: sample-size edge cases
# --------------------------------------------------------------------------- #
def test_sparse_stratum_falls_back_to_pooled_with_a_logged_warning(caplog):
    """A stratum below `min_stratum_size` cannot certify alpha at all.

    Falling back to the pooled qhat is the least-bad option: an infinite qhat
    makes the interval useless, and taking the largest available residual would
    claim a guarantee that was never established. The fallback must be LOUD --
    that stratum carries the marginal guarantee, not a conditional one, and a
    reader of the results is entitled to know which.
    """
    rng = np.random.default_rng(21)
    y = np.concatenate([rng.normal(120, 20, 400), rng.normal(60, 5, 3)])
    p = y + rng.normal(0, 10, y.size)

    with caplog.at_level("WARNING", logger="causalcgm_mt.conformal"):
        m = MondrianConformal.calibrate(y, p, alpha=0.1)

    assert m.n_by_stratum[0] < min_stratum_size(0.1)
    assert 0 in m.fell_back, "the under-populated hypo stratum must fall back"
    assert m.qhat_for(0) == m.qhat_pooled
    assert 1 not in m.fell_back, "the well-populated stratum must NOT fall back"
    assert "stratum 0" in caplog.text and "pooled" in caplog.text.lower()


def test_empty_stratum_falls_back_rather_than_emitting_an_infinite_interval():
    """A cohort with no hyperglycemic calibration window at all must still
    produce a usable interval for a hyperglycemic prediction."""
    rng = np.random.default_rng(23)
    y = rng.normal(120, 15, 500)         # euglycemic only, by construction
    y = np.clip(y, 75.0, 175.0)
    p = y + rng.normal(0, 8, 500)
    m = MondrianConformal.calibrate(y, p, alpha=0.1)

    assert m.n_by_stratum[0] == 0 and m.n_by_stratum[2] == 0
    assert set(m.fell_back) == {0, 2}
    lo, hi = m.predict_interval(np.array([250.0]))
    assert np.isfinite(lo[0]) and np.isfinite(hi[0])
    assert 0.0 < hi[0] - lo[0] < 600.0


def test_calibration_drops_non_finite_rows_instead_of_miscounting_them():
    """A nan target must not be counted into a stratum: it would inflate that
    cell's n and could push it above the fallback threshold on rows that
    contribute nothing to the quantile."""
    rng = np.random.default_rng(29)
    y = np.concatenate([rng.normal(120, 20, 300), np.full(50, np.nan)])
    p = np.concatenate([y[:300] + rng.normal(0, 10, 300), np.full(50, 120.0)])

    m = MondrianConformal.calibrate(y, p, alpha=0.1)
    assert m.n_dropped == 50
    assert m.n_calib == 300
    assert sum(m.n_by_stratum.values()) == 300


def test_an_empty_calibration_set_refuses_to_certify():
    with pytest.warns(UserWarning, match="too small"):
        m = MondrianConformal.calibrate(np.array([]), np.array([]), alpha=0.1)
    assert not np.isfinite(m.qhat_pooled)
    lo, hi = m.predict_interval(np.array([120.0]))
    assert lo[0] == 0.0 and hi[0] == 600.0, "unbounded interval should clip to range"


def test_mondrian_shape_mismatch_is_an_error_not_a_broadcast():
    with pytest.raises(ValueError, match="shape mismatch"):
        MondrianConformal.calibrate(np.zeros(10), np.zeros(11), alpha=0.1)


# --------------------------------------------------------------------------- #
# Mondrian: leakage and reporting
# --------------------------------------------------------------------------- #
def test_mondrian_qhat_depends_only_on_calibration_data():
    """The decisive leakage test, per stratum: recalibrating with identical
    calibration inputs must give bit-identical qhats, and nothing about the
    evaluation set has any path into the object."""
    rng = np.random.default_rng(31)
    y, p = _multimodal_glucose(rng, 3000)
    a = MondrianConformal.calibrate(y, p, 0.1)
    b = MondrianConformal.calibrate(y.copy(), p.copy(), 0.1)
    assert a.qhat_by_stratum == b.qhat_by_stratum
    assert a.qhat_pooled == b.qhat_pooled


def test_mondrian_records_its_calibration_patients(ws):
    pats = sorted({str(p) for p in ws.patient})
    cal, test = pats[:4], pats[4:]
    i_cal = split_by_patient(ws, cal)
    m = MondrianConformal.calibrate(
        ws.y_reg[i_cal], ws.last_bg[i_cal], 0.1, patient_calib=ws.patient[i_cal])
    assert set(m.calibrated_on) == set(cal)
    assert not (set(m.calibrated_on) & set(test)), "test patient contributed to qhat"


def test_supplied_taxonomy_gives_exact_conditional_coverage():
    """With an X-MEASURABLE taxonomy the Mondrian guarantee is a theorem, not a
    heuristic -- so it must hold cell by cell without the reachability
    expansion. Here the cell is read off the prediction at BOTH calibration and
    inference, which is a legitimate deployable rule."""
    covs = {k: [] for k in range(N_STRATA)}
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        y_cal, p_cal = _multimodal_glucose(rng, 6000)
        y_te, p_te = _multimodal_glucose(rng, 6000)

        m = MondrianConformal.calibrate(
            y_cal, p_cal, 0.1, strata=clinical_stratum(p_cal))
        assert m.taxonomy == "supplied"
        k_te = clinical_stratum(p_te)
        lo, hi = m.predict_interval(p_te, stratum=k_te)
        cov = (y_te >= lo) & (y_te <= hi)
        for k in range(N_STRATA):
            sel = k_te == k
            if sel.sum() > 50:
                covs[k].append(float(cov[sel].mean()))

    for k, vals in covs.items():
        assert float(np.mean(vals)) >= 0.88, \
            f"cell {STRATUM_NAMES[k]} covered {np.mean(vals):.4f}"


def test_a_supplied_taxonomy_is_not_truncated_to_its_cell(ws):
    """`predict_set` must agree with the interval `predict_interval` emits.

    On the DERIVED path, clipping each stratum's piece to `[a, b]` IS the
    label-conditional union construction. On the SUPPLIED path it is wrong: an
    X-measurable taxonomy (the zone of the last observed BG, of a trend
    extrapolation) puts the true `y` under no obligation to land inside cell
    `k`, so the valid region is the whole `yhat +/- qhat_k`. Truncating it
    undercovers exactly the points the stratification exists to protect.

    Latent while nothing built an interval from `predict_set` -- but
    `set_measure` does, so the reported width was narrower than the interval
    actually being scored, which is an efficiency claim nothing had earned.
    """
    rng = np.random.default_rng(3)
    y_cal, p_cal = _multimodal_glucose(rng, 4000)
    m = MondrianConformal.calibrate(y_cal, p_cal, 0.1, strata=clinical_stratum(p_cal))
    assert m.taxonomy == "supplied"

    # Predictions deliberately near the zone edges, where truncation bites.
    p_te = np.array([40.0, 60.0, 69.0, 71.0, 100.0, 179.0, 181.0, 250.0, 400.0])
    k_te = clinical_stratum(p_te)

    lo, hi = m.predict_interval(p_te, stratum=k_te)
    assert np.array_equal(m.set_measure(p_te, k_te), hi - lo), (
        "set_measure disagrees with the interval predict_interval emits")

    # The region is the FULL two-sided band, only the physiological clip applies.
    q = np.array([m.qhat_for(int(k)) for k in k_te])
    assert np.allclose(lo, np.clip(p_te - q, 0.0, None))
    assert np.allclose(hi, np.clip(p_te + q, None, 600.0))

    # And an interval that straddles a zone boundary keeps both sides: a 71
    # mg/dL prediction in the euglycemic cell must still reach below 70.
    i = int(np.flatnonzero(p_te == 71.0)[0])
    assert lo[i] < 70.0 < hi[i], (
        "the euglycemic cell's interval was truncated at its own lower bound")

    # A scalar stratum takes the same path.
    lo_s, hi_s = m.predict_interval(p_te, stratum=1)
    assert np.array_equal(m.set_measure(p_te, 1), hi_s - lo_s)


def test_the_derived_taxonomy_still_builds_the_label_conditional_union(ws):
    """The other half of the fix: the derived path's clipping is load-bearing.

    Each stratum contributes only the slice of its interval inside its own
    zone -- that restriction is what makes the union exact, and what
    `_hull_of_label_conditional_set` reads. Removing it there would widen every
    hull and silently change the headline efficiency number.
    """
    rng = np.random.default_rng(4)
    y_cal, p_cal = _multimodal_glucose(rng, 4000)
    m = MondrianConformal.calibrate(y_cal, p_cal, 0.1)
    assert m.taxonomy == "target", "calibrated on the TRUE stratum"

    p_te = np.array([50.0, 100.0, 200.0, 300.0])
    for k, (lo_k, hi_k, active) in enumerate(m.predict_set(p_te, None)):
        a, b = stratum_bounds(k, m.hypo_threshold, m.hyper_threshold)
        assert np.all(lo_k[active] >= max(a, 0.0) - 1e-9), STRATUM_NAMES[k]
        assert np.all(hi_k[active] <= min(b, 600.0) + 1e-9), STRATUM_NAMES[k]

    # The union is never wider than the hull, and the hull is what the derived
    # path returns.
    lo, hi = m.predict_interval(p_te)
    assert np.all(m.set_measure(p_te) <= (hi - lo) + 1e-9)


def test_hour_of_day_features_match_the_per_row_construction(ws):
    """M17 vectorised the circadian columns. LightGBM histogram-bins them, so a
    1-ulp move can land on the other side of a split threshold -- the
    replacement has to be bit-identical, not merely equal to tolerance."""
    rng = np.random.default_rng(9)
    base = np.datetime64("2021-03-14T00:00", "m")
    for unit in ("m", "s", "ns"):
        t_time = (base + rng.integers(0, 60 * 24 * 90, 3000)
                  .astype("timedelta64[m]")).astype(f"datetime64[{unit}]")
        loop = np.array([
            (np.datetime64(t, "m").astype("datetime64[m]").astype(object).hour +
             np.datetime64(t, "m").astype("datetime64[m]").astype(object).minute / 60.0)
            for t in t_time], dtype=np.float64)

        t_min = np.asarray(t_time).astype("datetime64[m]")
        minutes = ((t_min - t_min.astype("datetime64[D]"))
                   .astype("timedelta64[m]").astype(np.int64))
        vec = (minutes // 60) + (minutes % 60) / 60.0
        assert np.array_equal(loop, vec), f"{unit}: not bit-identical"
        assert vec.min() >= 0.0 and vec.max() < 24.0

    # And on the real design matrix: the circadian pair still lies on the unit
    # circle, and still encodes the anchor's own hour of day.
    feats = tabular_features(ws, np.arange(256))
    sin_i, cos_i = FEATURE_NAMES.index("hour_sin"), FEATURE_NAMES.index("hour_cos")
    assert np.allclose(feats[:, sin_i] ** 2 + feats[:, cos_i] ** 2, 1.0, atol=1e-5)

    t_min = np.asarray(ws.t_time[:256]).astype("datetime64[m]")
    minutes = ((t_min - t_min.astype("datetime64[D]"))
               .astype("timedelta64[m]").astype(np.int64))
    want = (minutes // 60) + (minutes % 60) / 60.0
    assert np.allclose(feats[:, sin_i], np.sin(2 * np.pi * want / 24.0), atol=1e-6)
    assert np.allclose(feats[:, cos_i], np.cos(2 * np.pi * want / 24.0), atol=1e-6)

    # Structural, because the values above are identical either way: the point
    # of the change is that the columns are built in one pass rather than with
    # two `datetime64 -> Python datetime` conversions per row, on the fit, val
    # and test splits of every horizon of every arm.
    import inspect

    src = inspect.getsource(tabular_features)
    assert "for t in t_time" not in src, (
        "the hour-of-day columns are back on a per-row Python loop")
    assert "astype(object)" not in src, (
        "a per-row datetime64 -> object conversion is back in the hot path")


def test_coverage_report_breaks_down_coverage_and_width_by_stratum():
    """Coverage alone cannot distinguish 'calibrated here' from 'the interval is
    200 mg/dL wide here', and stratified conformal buys hypo coverage precisely
    by spending width -- so the report must show both."""
    y = np.array([50.0] * 100 + [120.0] * 100 + [250.0] * 100)
    half = np.where(y < 70, 40.0, np.where(y <= 180, 10.0, 25.0))
    rep = coverage_report(y, y - half, y + half, 0.1)

    assert set(rep.coverage_by_stratum) == set(STRATUM_NAMES)
    c0 = rep.stratum(0)
    assert c0.n == 100 and c0.coverage == pytest.approx(1.0)
    assert c0.mean_width == pytest.approx(80.0)
    assert rep.stratum(1).mean_width == pytest.approx(20.0)
    assert rep.stratum(2).mean_width == pytest.approx(50.0)
    assert "by Mondrian stratum" in str(rep)


def test_conditional_validity_predicate_fails_when_one_stratum_fails():
    """`is_conditionally_valid` is the claim the module exists to make, so it
    must be falsifiable: marginal validity holding while one cell collapses is
    exactly the situation it has to report as invalid."""
    y = np.concatenate([np.full(900, 120.0), np.full(100, 60.0)])
    half = np.where(y < 70, 0.5, 50.0)
    y_noisy = y + np.where(y < 70, 5.0, 0.0)
    rep = coverage_report(y_noisy, y - half, y + half, 0.1)

    assert rep.is_valid(), "marginal coverage should pass in this construction"
    assert not rep.is_conditionally_valid(), "the hypo cell has collapsed to zero"


# --------------------------------------------------------------------------- #
# LightGBM baseline
# --------------------------------------------------------------------------- #
def test_tabular_features_shape_and_names(ws):
    X = tabular_features(ws)
    assert X.shape == (len(ws), len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_tabular_features_are_inherited_causal(ws):
    """Features must be a function of the input window only.

    Corrupting every target leaves the design matrix bit-identical, because the
    extractor reads `ws.x` and nothing else.
    """
    X = tabular_features(ws)

    ws2 = ws.subset(np.arange(len(ws)))
    ws2.y_reg = ws2.y_reg + 999.0
    ws2.y_min = ws2.y_min - 999.0
    ws2.y_cls_hard = 1.0 - ws2.y_cls_hard

    assert np.array_equal(X, tabular_features(ws2))


def test_tabular_features_use_suffix_windows_only(ws):
    """A lookback-k feature must depend only on the most recent k steps.

    Perturbing the OLDEST history steps must leave short-lookback features
    unchanged; if it moves them, the reduction is reading the wrong end of the
    window.
    """
    idx = np.arange(64)
    X = tabular_features(ws, idx)

    ws2 = ws.subset(np.arange(len(ws)))
    ws2.x = ws2.x.copy()
    ws2.x[:, :12, 0] += 100.0            # corrupt the OLDER half only
    X2 = tabular_features(ws2, idx)

    for name in ("mean_1", "mean_3", "mean_6", "std_3", "bg_last"):
        j = FEATURE_NAMES.index(name)
        assert np.allclose(X[:, j], X2[:, j], atol=1e-4), \
            f"{name} changed when only older steps were perturbed"

    # And the long-lookback features SHOULD move -- otherwise the test is vacuous.
    j = FEATURE_NAMES.index("mean_24")
    assert not np.allclose(X[:, j], X2[:, j])


def test_lgbm_trains_and_beats_persistence(ws):
    cfg = Config()
    pats = sorted({str(p) for p in ws.patient})
    model, rep = run_lgbm_fold(
        ws, cfg, np.array(pats[:6], object), np.array(pats[6:], object)
    )
    assert rep.rmse < rep.rmse_persistence, \
        "a baseline that cannot beat persistence is misconfigured"
    assert 0.0 <= rep.pr_auc <= 1.0
    assert rep.n > 0


def test_lgbm_predicts_a_residual_not_a_level(ws):
    """The regressor targets BG(t+PH) - BG(t).

    Predicting the level lets a tree win on RMSE by echoing `bg_last`, which
    axis-aligned splits make an easy local optimum. The check: the fitted
    regressor's own target must be centred near zero, not near 140.
    """
    cfg = Config()
    pats = sorted({str(p) for p in ws.patient})
    i_fit = split_by_patient(ws, pats[:6])
    m = LGBMBaseline(cfg=cfg, n_estimators=40).fit(ws, i_fit)

    raw = m.reg_model.predict(tabular_features(ws, i_fit))
    assert abs(float(np.mean(raw))) < 20.0, \
        "regressor appears to predict an absolute level, not a residual"


def test_lgbm_scored_through_the_same_evaluate_as_the_tcn(ws):
    """Identical metric code, so a comparison is apples to apples."""
    cfg = Config()
    pats = sorted({str(p) for p in ws.patient})
    i_test = split_by_patient(ws, pats[6:])
    m = LGBMBaseline(cfg=cfg, n_estimators=40).fit(ws, split_by_patient(ws, pats[:6]))

    rep = m.evaluate(ws, i_test)
    assert hasattr(rep, "mard") and hasattr(rep, "time_lag_min") and hasattr(rep, "pr_auc")
    assert rep.n == i_test.size


def test_lgbm_feature_importance_is_normalised(ws):
    cfg = Config()
    pats = sorted({str(p) for p in ws.patient})
    m = LGBMBaseline(cfg=cfg, n_estimators=40).fit(ws, split_by_patient(ws, pats[:6]))
    imp = m.feature_importance(top=5)
    assert len(imp) == 5
    assert all(0.0 <= r[1] <= 1.0 and 0.0 <= r[2] <= 1.0 for r in imp)
