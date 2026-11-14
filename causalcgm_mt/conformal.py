"""Split conformal prediction for the glucose regression head.

What split conformal gives you
------------------------------
Fit a model on a training split. On a *disjoint* calibration split compute
nonconformity scores s_i = |y_i - yhat_i|. Take

    qhat = Quantile_{ceil((n+1)(1-alpha))/n} ( {s_i} )

and emit intervals [yhat +/- qhat]. Under exchangeability of the calibration and
test points, P(y in interval) >= 1 - alpha, with no assumption whatsoever about
the model. The finite-sample correction (n+1)/n is not cosmetic: with n = 100 and
alpha = 0.1 the naive 90th percentile undercovers by roughly a point, and
undercoverage is the failure direction that matters clinically.

WHERE THIS ASSUMPTION BREAKS, WHICH IS THE POINT OF THIS MODULE
---------------------------------------------------------------
Exchangeability is a real assumption and CGM data violates it in two ways:

  1. TEMPORAL DEPENDENCE. Consecutive windows overlap by ~95% of their content,
     so calibration scores are massively autocorrelated. The effective sample
     size is far below n, which makes qhat noisier than n suggests. This does
     not bias coverage, but it widens the confidence band around it -- so
     `effective_n` is reported alongside every calibration.

  2. PATIENT SHIFT. Calibrating on patients A and B and deploying on patient C
     is NOT exchangeable: glucose dynamics are strongly individual. Marginal
     coverage across a population can hold at 90% while an individual patient
     sits at 70%. This is the failure mode that matters, and it is why
     `PatientConformal` (calibrate on each new patient's own causal prefix)
     exists alongside the pooled variant, and why `coverage_report` breaks
     coverage down per patient and per glucose range rather than reporting a
     single number.

Conditional coverage is NOT guaranteed by any of this. Marginal 90% coverage is
fully compatible with 60% coverage in the hypoglycemic range, because the model
is least accurate exactly where excursions are steepest. `coverage_report`
stratifies by clinical range for that reason -- an interval that is valid on
average and invalid when the patient is dropping is worse than useless, because
it invites trust precisely where it fails.

Normalised (locally adaptive) conformal is provided as `NormalizedConformal`,
which scales the interval by a per-point difficulty estimate. It trades constant
width for adaptive width and usually improves conditional coverage.

`MondrianConformal` attacks the same gap from the other direction: instead of a
continuous difficulty scale it calibrates a SEPARATE qhat inside each clinical
glycemic zone, which is the granularity a clinician actually reasons at.
Stratified calibration addresses conditional coverage by construction rather
than by hoping a difficulty proxy happens to correlate with the right thing.

How large the gap is depends on the model and the split, so this module does not
hard-code a number: run `scripts/verify_mondrian_coverage.py` to measure it on
the data in front of you. On the patient-disjoint synthetic arm the pooled
interval covers the hypoglycemic range at ~35% against a 90% target, and
stratification roughly doubles that. On the OhioT1DM official split with the
tabular baseline the per-stratum residual scales are close together, the pooled
qhat is already near the hypo one, and there is correspondingly little to fix --
which is the honest reading, not a disappointing one: the method should be
measured, not assumed.

Two things it does NOT fix, and which the verification script makes visible:
stratification cannot repair a violation of exchangeability BETWEEN patients
(see `PatientConformal`), and it cannot manufacture information -- where the
per-stratum residual scales genuinely differ by several-fold, the guarantee is
paid for in width.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .data import WindowSet

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Quantile with the finite-sample correction
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (n+1)-corrected empirical quantile used by split conformal.

    qhat is the k-th SMALLEST score with k = ceil((n + 1)(1 - alpha)). If k > n
    the calibration set is too small to certify the requested alpha at all --
    90% intervals need n >= 9 -- and we return +inf rather than silently
    returning the maximum score and pretending to a guarantee we cannot make.

    The order statistic is taken directly rather than via `np.quantile`. An
    earlier version used `np.quantile(s, k / n, method="higher")`, which is NOT
    the same thing: numpy interpolates on the virtual index q*(n-1), so for
    n = 20, alpha = 0.1 it returned s[19] (the maximum) where the theory calls
    for s[18]. That errs conservative, so coverage still held and a
    coverage-only test passed -- it just made every interval wider than
    necessary, most visibly at small n where calibration data is scarcest.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")

    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(s)[k - 1])


def effective_sample_size(scores: np.ndarray, max_lag: int = 50) -> float:
    """Autocorrelation-adjusted n: n_eff = n / (1 + 2 * sum_k rho_k).

    Overlapping windows make calibration scores strongly autocorrelated, so the
    nominal n overstates how much independent information qhat rests on. This is
    reported, never used to alter qhat -- coverage stays valid, it is the
    *stability* of qhat across resamples that degrades.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n < 10:
        return float(n)

    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom <= 0:
        return float(n)

    tot = 0.0
    for k in range(1, min(max_lag, n - 1)):
        rho = float(np.dot(s[:-k], s[k:]) / denom)
        if rho <= 0:            # standard initial-positive-sequence truncation
            break
        tot += rho
    return float(n / (1.0 + 2.0 * tot))


# --------------------------------------------------------------------------- #
# Mondrian (group-conditional) taxonomy
# --------------------------------------------------------------------------- #
# The three zones a clinician reasons in. They are coarser than CLINICAL_RANGES
# below on purpose: CLINICAL_RANGES is a REPORTING breakdown and can be as fine
# as you like, whereas a Mondrian taxonomy has to leave every cell with enough
# calibration points to support its own quantile. Splitting < 54 out as a fourth
# stratum sounds attractive and is the wrong trade -- on a typical Ohio fold it
# holds a few dozen windows, so its qhat would be noisier than the pooled one it
# replaced, and the guarantee it certifies would be the weakest of the four.
HYPO_THRESHOLD_MGDL: float = 70.0
HYPER_THRESHOLD_MGDL: float = 180.0

STRATUM_NAMES: tuple[str, ...] = (
    "hypoglycemic (<70)",
    "euglycemic (70-180)",
    "hyperglycemic (>180)",
)
N_STRATA: int = len(STRATUM_NAMES)


def clinical_stratum(
    y: np.ndarray,
    hypo_threshold: float = HYPO_THRESHOLD_MGDL,
    hyper_threshold: float = HYPER_THRESHOLD_MGDL,
) -> np.ndarray:
    """Map glucose values to Mondrian strata 0/1/2, or -1 when undefined.

        0  hypoglycemic   y <  70
        1  euglycemic     70 <= y <= 180
        2  hyperglycemic  y > 180

    Boundary convention matches the clinical one: 70 and 180 are both inside
    the euglycemic band. A non-finite value returns -1 rather than silently
    landing in the euglycemic stratum -- `nan < 70` and `nan > 180` are both
    False, so the naive expression would classify a missing prediction as
    normal glucose, which is the single most dangerous default available.
    """
    v = np.asarray(y, dtype=np.float64)
    k = np.full(v.shape, 1, dtype=np.int64)
    finite = np.isfinite(v)
    k[finite & (v < float(hypo_threshold))] = 0
    k[finite & (v > float(hyper_threshold))] = 2
    k[~finite] = -1
    return k


def stratum_bounds(
    k: int,
    hypo_threshold: float = HYPO_THRESHOLD_MGDL,
    hyper_threshold: float = HYPER_THRESHOLD_MGDL,
) -> tuple[float, float]:
    """Closed [lo, hi] envelope of stratum `k` on the glucose axis."""
    if k == 0:
        return (-np.inf, float(hypo_threshold))
    if k == 1:
        return (float(hypo_threshold), float(hyper_threshold))
    if k == 2:
        return (float(hyper_threshold), np.inf)
    raise ValueError(f"stratum must be 0, 1 or 2; got {k}")


def min_stratum_size(alpha: float) -> int:
    """Smallest calibration count that can certify level `alpha` at all.

    ceil((1 - alpha) / alpha): 9 points for alpha = 0.1, 19 for 0.05. Below it
    `conformal_quantile` returns +inf by construction, because the k-th order
    statistic it needs does not exist. Stated as its own function so the
    fallback rule in `MondrianConformal.calibrate` and the theory agree by
    definition rather than by a hard-coded 9.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return int(np.ceil((1.0 - alpha) / alpha))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StratumCoverage:
    """Coverage AND width inside one Mondrian stratum.

    Both, always. Coverage alone cannot distinguish "correctly calibrated here"
    from "the interval is 200 mg/dL wide here", and a stratified conformal
    method buys its hypoglycemic coverage precisely by spending width -- so a
    report that hid the width would make the method look free.
    """
    name: str
    index: int
    coverage: float
    mean_width: float
    median_width: float
    n: int


@dataclass
class CoverageReport:
    """Empirical validity, marginal and stratified.

    `covered` is the headline number; the stratified fields are what determine
    whether the intervals are clinically usable.
    """
    alpha: float
    target_coverage: float
    empirical_coverage: float
    mean_width: float
    median_width: float
    n: int
    coverage_by_range: dict[str, tuple[float, int]] = field(default_factory=dict)
    coverage_by_patient: dict[str, float] = field(default_factory=dict)
    worst_patient_coverage: float = float("nan")
    worst_patient: str = ""
    # Mondrian strata: the taxonomy `MondrianConformal` calibrates against, so
    # its guarantee can be checked against the same partition it certifies.
    coverage_by_stratum: dict[str, StratumCoverage] = field(default_factory=dict)

    def is_valid(self, tolerance: float = 0.02) -> bool:
        """Marginal validity only. Deliberately not the whole story."""
        return self.empirical_coverage >= self.target_coverage - tolerance

    def stratum(self, k: int) -> StratumCoverage | None:
        """The `StratumCoverage` for Mondrian stratum `k`, or None if empty."""
        return self.coverage_by_stratum.get(STRATUM_NAMES[k])

    def is_conditionally_valid(self, tolerance: float = 0.02,
                               min_n: int = 20) -> bool:
        """Every Mondrian stratum covers, not just the pooled average.

        This is the claim `MondrianConformal` exists to make, so it gets its own
        predicate rather than being left to the reader of `__str__`. Strata with
        fewer than `min_n` scored points are skipped: their empirical coverage
        is too noisy to falsify anything, and failing on them would make the
        check fire on sample size rather than on calibration.
        """
        cells = [c for c in self.coverage_by_stratum.values() if c.n >= min_n]
        if not cells:
            return False
        return all(c.coverage >= self.target_coverage - tolerance for c in cells)

    def is_patient_conditionally_valid(self, tolerance: float = 0.02) -> bool:
        """Every scored patient covers, not just the pooled average.

        `is_conditionally_valid` checks the Mondrian glycemic strata, which is
        the partition `MondrianConformal` certifies. It says nothing about the
        per-patient breakdown -- and per-patient coverage is the failure mode
        this module's header names as the one that matters, because a device is
        worn by one person. Returns False when no patient had enough scored
        points to judge, for the same reason the stratum predicate does.
        """
        if not self.coverage_by_patient:
            return False
        return all(c >= self.target_coverage - tolerance
                   for c in self.coverage_by_patient.values())

    def __str__(self) -> str:
        lines = [
            f"Conformal coverage @ alpha={self.alpha:.2f} "
            f"(target {self.target_coverage:.1%})",
            f"  empirical   : {self.empirical_coverage:.3%}  "
            f"({'VALID' if self.is_valid() else 'UNDERCOVERS'})",
            f"  width       : mean {self.mean_width:.2f} mg/dL, "
            f"median {self.median_width:.2f}",
            f"  n           : {self.n}",
        ]
        if self.coverage_by_range:
            lines.append("  by glucose range:")
            for k, (cov, cnt) in self.coverage_by_range.items():
                flag = "" if cov >= self.target_coverage - 0.02 else "   <-- undercovers"
                lines.append(f"    {k:<22} {cov:.3%}  (n={cnt}){flag}")
        if self.coverage_by_stratum:
            lines.append("  by Mondrian stratum (coverage / mean width):")
            for c in self.coverage_by_stratum.values():
                flag = "" if c.coverage >= self.target_coverage - 0.02 else "   <-- undercovers"
                lines.append(
                    f"    {c.name:<22} {c.coverage:.3%}  "
                    f"width {c.mean_width:6.2f} mg/dL  (n={c.n}){flag}"
                )
        if self.coverage_by_patient:
            lines.append(
                f"  worst patient: {self.worst_patient} at "
                f"{self.worst_patient_coverage:.3%}"
            )
        return "\n".join(lines)


# Clinical strata. Hypoglycemia is split out because that is where both the
# model error and the clinical cost are largest.
CLINICAL_RANGES: tuple[tuple[str, float, float], ...] = (
    ("severe hypo (<54)", -np.inf, 54.0),
    ("hypo (54-70)", 54.0, 70.0),
    ("near-threshold (70-90)", 70.0, 90.0),
    ("euglycemic (90-180)", 90.0, 180.0),
    ("hyper (>180)", 180.0, np.inf),
)


def coverage_report(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    patient: np.ndarray | None = None,
    hypo_threshold: float = HYPO_THRESHOLD_MGDL,
    hyper_threshold: float = HYPER_THRESHOLD_MGDL,
) -> CoverageReport:
    """Empirical validity check, stratified three ways.

    By CLINICAL_RANGES (fine-grained reporting), by Mondrian stratum (the
    partition `MondrianConformal` certifies, with mean width alongside so the
    cost of the stratified guarantee is visible), and by patient.

    The stratum breakdown is computed from the TRUE `y_true`, which is the only
    honest way to audit conditional coverage: assigning the stratum from the
    prediction would grade the method against its own guess and could report
    valid coverage for a point whose true value sits in a different zone
    entirely.
    """
    y = np.asarray(y_true, dtype=np.float64)
    lo, hi = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo

    by_range: dict[str, tuple[float, int]] = {}
    for name, a, b in CLINICAL_RANGES:
        m = (y >= a) & (y < b)
        if m.sum() > 0:
            by_range[name] = (float(covered[m].mean()), int(m.sum()))

    strata = clinical_stratum(y, hypo_threshold, hyper_threshold)
    by_stratum: dict[str, StratumCoverage] = {}
    for k, name in enumerate(STRATUM_NAMES):
        m = strata == k
        if not m.any():
            continue
        by_stratum[name] = StratumCoverage(
            name=name, index=k,
            coverage=float(covered[m].mean()),
            mean_width=float(np.mean(width[m])),
            median_width=float(np.median(width[m])),
            n=int(m.sum()),
        )

    by_patient: dict[str, float] = {}
    worst, worst_p = float("nan"), ""
    if patient is not None:
        for p in np.unique(patient):
            m = patient == p
            if m.sum() >= 20:      # too few points to be meaningful otherwise
                by_patient[str(p)] = float(covered[m].mean())
        if by_patient:
            worst_p = min(by_patient, key=by_patient.get)
            worst = by_patient[worst_p]

    return CoverageReport(
        alpha=alpha,
        target_coverage=1 - alpha,
        empirical_coverage=float(covered.mean()),
        mean_width=float(np.mean(width)),
        median_width=float(np.median(width)),
        n=int(y.size),
        coverage_by_range=by_range,
        coverage_by_patient=by_patient,
        worst_patient_coverage=worst,
        worst_patient=worst_p,
        coverage_by_stratum=by_stratum,
    )


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #
@dataclass
class SplitConformal:
    """Constant-width split conformal intervals.

    Calibration is a *fit*, and therefore obeys the same rule as every other
    fitted object in this codebase: it is estimated on calibration data only and
    applied downstream frozen. `calibrated_on` records which patients
    contributed, so an audit can confirm the test patients did not.
    """

    alpha: float
    qhat: float = float("nan")
    n_calib: int = 0
    n_effective: float = float("nan")
    calibrated_on: tuple[str, ...] = ()

    @staticmethod
    def calibrate(
        y_calib: np.ndarray,
        pred_calib: np.ndarray,
        alpha: float,
        patient_calib: np.ndarray | None = None,
    ) -> "SplitConformal":
        """Derive qhat from |y - yhat| on the calibration split ONLY.

        Nothing about the test distribution enters: the scores are functions of
        calibration labels and calibration predictions, and qhat is a quantile
        of those scores. At predict time the interval is a fixed offset applied
        to whatever the model outputs, so no test label is ever touched.
        """
        scores = np.abs(np.asarray(y_calib, np.float64) - np.asarray(pred_calib, np.float64))
        qhat = conformal_quantile(scores, alpha)

        if not np.isfinite(qhat):
            import warnings
            warnings.warn(
                f"Calibration set of {scores.size} points is too small to certify "
                f"alpha={alpha} (need n >= {int(np.ceil(1 / alpha)) - 1}). "
                "qhat is infinite; intervals will be unbounded rather than "
                "falsely narrow.",
                stacklevel=2,
            )

        return SplitConformal(
            alpha=alpha,
            qhat=qhat,
            n_calib=int(scores.size),
            n_effective=effective_sample_size(scores),
            calibrated_on=(
                tuple(sorted({str(p) for p in patient_calib}))
                if patient_calib is not None else ()
            ),
        )

    def predict_interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[yhat - qhat, yhat + qhat], clipped to the physiological range.

        Clipping is safe for coverage: true glucose cannot fall outside the
        sensor's own range, so trimming the interval there cannot exclude a
        realisable value. It only stops the interval claiming absurdities like
        -20 mg/dL.
        """
        p = np.asarray(pred, np.float64)
        return np.clip(p - self.qhat, 0.0, None), np.clip(p + self.qhat, None, 600.0)

    def evaluate(
        self,
        y_true: np.ndarray,
        pred: np.ndarray,
        patient: np.ndarray | None = None,
    ) -> CoverageReport:
        lo, hi = self.predict_interval(pred)
        return coverage_report(y_true, lo, hi, self.alpha, patient)

    def __str__(self) -> str:
        return (
            f"SplitConformal(alpha={self.alpha:.2f}, qhat={self.qhat:.2f} mg/dL, "
            f"n={self.n_calib}, n_eff={self.n_effective:.0f}, "
            f"patients={len(self.calibrated_on)})"
        )


# --------------------------------------------------------------------------- #
# Mondrian (group-conditional) conformal
# --------------------------------------------------------------------------- #
@dataclass
class MondrianConformal:
    """One qhat per clinical glycemic zone instead of one for the whole cohort.

    THE PROBLEM THIS SOLVES
    -----------------------
    Split conformal is marginally valid and conditionally silent. The pooled
    qhat is dominated by euglycemic windows -- ~80% of any CGM record, and the
    regime where the model is most accurate -- so the interval it produces can
    be far too narrow for the steep excursions that generate hypoglycemia
    labels. The interval is then most trustworthy where it matters least and
    least trustworthy where a clinician would act on it. `coverage_report`
    exists to surface that, and `scripts/verify_mondrian_coverage.py` measures
    how bad it is for a given model rather than assuming a figure.

    Mondrian (a.k.a. group-conditional or class-conditional) conformal splits
    the calibration set by a taxonomy and takes a separate quantile inside each
    cell. Coverage then holds *within each cell* rather than only on average --
    the guarantee that was wanted all along.

    THE EXCHANGEABILITY CAVEAT, STATED RATHER THAN BURIED
    -----------------------------------------------------
    Mondrian conformal is exactly valid when the taxonomy is a function of
    OBSERVABLE quantities. The clinical taxonomy here is a function of the
    TARGET, which is unknown at prediction time, so two regimes exist and they
    are not the same:

      * `strata=` supplied at BOTH calibration and prediction from something
        X-measurable (last observed BG, a trend estimate, a predicted zone
        fixed in advance): the finite-sample guarantee is exact, conditional on
        the cell. This is the theoretically clean path.

      * `strata=None` (the default, and what the clinical requirement asks
        for): the cell is read off the TRUE y during calibration and off the
        PREDICTION at inference. Where the two disagree -- necessarily, near a
        zone boundary -- the wrong qhat is applied. This is a heuristic, not a
        theorem, and it is the regime `expand=True` exists to defend.

    THE REACHABILITY EXPANSION, AND WHY IT RESTORES THE GUARANTEE
    -------------------------------------------------------------
    Applying the predicted stratum's qhat naively is not merely imperfect, it
    fails hardest exactly where it is needed: a true 60 mg/dL that the model
    reads as 100 gets the narrow euglycemic width, and the point is missed.

    The fix is the standard label-conditional construction. The exact
    conformal region for the taxonomy is the UNION over strata

        C(yhat) = U_k { y in region_k : |y - yhat| <= qhat_k }

    -- for each candidate zone, the interval you would have used had the truth
    been in that zone, restricted to that zone. `predict_interval` returns the
    convex hull of that union (downstream code, plotting and the C++ runtime all
    expect a single [lo, hi]); `predict_set` returns the union itself when the
    exact region is wanted.

    That construction is not a heuristic, and the reachability test is what
    makes it cheap: stratum k contributes iff dist(yhat, region_k) <= qhat_k.
    From which the guarantee follows directly --

        if the true y lies in stratum k and |y - yhat| <= qhat_k, then
        dist(yhat, region_k) <= |y - yhat| <= qhat_k, so stratum k is
        reachable, so the emitted half-width is at least qhat_k, so y IS
        COVERED.

    In words: the expanded interval covers y whenever the ORACLE interval --
    the one built from the true, unobservable stratum -- would have. Coverage
    conditional on the true clinical zone is therefore inherited from the
    per-stratum split-conformal guarantee, without ever observing the zone.
    The price is width, paid only near a boundary or where one stratum's qhat
    is much larger than another's, and `predict_set` is what makes that price
    visible instead of hidden inside a hull.

    Set `expand=False` for the strict predicted-stratum variant, which is the
    right ablation arm to report against: it is the version that undercovers.
    """

    alpha: float
    qhat_by_stratum: dict[int, float] = field(default_factory=dict)
    qhat_pooled: float = float("nan")
    n_by_stratum: dict[int, int] = field(default_factory=dict)
    n_effective_by_stratum: dict[int, float] = field(default_factory=dict)
    # Strata whose own calibration set was too small to certify alpha, and which
    # therefore use the pooled qhat. Recorded, not silent: a stratum on this
    # list carries the MARGINAL guarantee, not a conditional one, and a reader
    # of the results is entitled to know which.
    fell_back: tuple[int, ...] = ()
    n_calib: int = 0
    n_dropped: int = 0
    hypo_threshold: float = HYPO_THRESHOLD_MGDL
    hyper_threshold: float = HYPER_THRESHOLD_MGDL
    taxonomy: str = "target"          # "target" | "supplied"
    calibrated_on: tuple[str, ...] = ()

    # ---------------------------------------------------------------- fit ---
    @staticmethod
    def calibrate(
        y_calib: np.ndarray,
        pred_calib: np.ndarray,
        alpha: float,
        strata: np.ndarray | None = None,
        patient_calib: np.ndarray | None = None,
        hypo_threshold: float = HYPO_THRESHOLD_MGDL,
        hyper_threshold: float = HYPER_THRESHOLD_MGDL,
        min_stratum_n: int | None = None,
    ) -> "MondrianConformal":
        """Per-stratum qhat from |y - yhat| on the calibration split ONLY.

        `strata` overrides the clinical taxonomy with a caller-supplied cell
        assignment (values in 0..2); pass the same rule at prediction time. It
        is how the exact, X-measurable variant described in the class docstring
        is obtained.

        Sample-size policy. A stratum with fewer than
        `min_stratum_size(alpha)` points cannot certify the level at all -- the
        order statistic conformal needs does not exist -- so it falls back to
        the pooled qhat and says so in the log. Falling back is strictly better
        than the alternatives: an infinite qhat makes the interval useless, and
        using the largest available residual would claim a guarantee that was
        never established.

        Non-finite calibration rows (a dropped label, a diverged prediction)
        are removed before anything else. Keeping them would let `nan` propagate
        into a stratum count and quietly shrink that cell below the fallback
        threshold.
        """
        y = np.asarray(y_calib, dtype=np.float64)
        p = np.asarray(pred_calib, dtype=np.float64)
        if y.shape != p.shape:
            raise ValueError(f"shape mismatch: y {y.shape} vs pred {p.shape}")

        ok = np.isfinite(y) & np.isfinite(p)
        n_dropped = int((~ok).sum())
        if n_dropped:
            logger.warning(
                "MondrianConformal.calibrate: dropped %d of %d calibration rows "
                "with non-finite target or prediction.", n_dropped, y.size)

        if strata is None:
            k_all = clinical_stratum(y, hypo_threshold, hyper_threshold)
            taxonomy = "target"
        else:
            k_all = np.asarray(strata, dtype=np.int64)
            if k_all.shape != y.shape:
                raise ValueError(
                    f"strata {k_all.shape} must align with y {y.shape}")
            # The shape was checked and the VALUES were not. The per-stratum
            # loop below runs over `range(N_STRATA)`, so a row outside 0..2 --
            # an off-by-one, a -1 from a non-finite predictor, a four-cell
            # taxonomy someone tried -- contributed to the pooled quantile but
            # to no cell. Those rows silently received the MARGINAL guarantee
            # while `n_calib` still counted them and the object reported itself
            # `taxonomy="supplied"` with nothing in `fell_back`, so every
            # coverage number it emitted looked conditionally valid.
            bad = np.unique(k_all[(k_all < 0) | (k_all >= N_STRATA)])
            if bad.size:
                raise ValueError(
                    f"supplied strata must be in 0..{N_STRATA - 1}; got "
                    f"{bad.tolist()}. A taxonomy with more cells needs "
                    f"N_STRATA and STRATUM_NAMES extended.")
            taxonomy = "supplied"

        y, p, k = y[ok], p[ok], k_all[ok]
        scores = np.abs(y - p)
        pooled = conformal_quantile(scores, alpha)
        if not np.isfinite(pooled):
            import warnings
            warnings.warn(
                f"Pooled calibration set of {scores.size} points is too small to "
                f"certify alpha={alpha} (need n >= {min_stratum_size(alpha)}). "
                "Every stratum will inherit an infinite qhat rather than a "
                "falsely narrow one.",
                stacklevel=2,
            )

        need = int(min_stratum_n) if min_stratum_n is not None else min_stratum_size(alpha)
        qhat: dict[int, float] = {}
        n_by: dict[int, int] = {}
        neff: dict[int, float] = {}
        fell: list[int] = []

        for kk in range(N_STRATA):
            m = k == kk
            s_k = scores[m]
            n_by[kk] = int(s_k.size)
            neff[kk] = effective_sample_size(s_k)
            q_k = conformal_quantile(s_k, alpha) if s_k.size >= need else float("inf")
            if s_k.size < need or not np.isfinite(q_k):
                qhat[kk] = pooled
                fell.append(kk)
                logger.warning(
                    "MondrianConformal: stratum %d (%s) has %d calibration "
                    "point(s), below the %d needed to certify alpha=%.3g. "
                    "Falling back to the pooled qhat=%.3f -- this stratum "
                    "carries the MARGINAL guarantee, not a conditional one.",
                    kk, STRATUM_NAMES[kk], n_by[kk], need, alpha, pooled)
            else:
                qhat[kk] = float(q_k)

        return MondrianConformal(
            alpha=alpha,
            qhat_by_stratum=qhat,
            qhat_pooled=float(pooled),
            n_by_stratum=n_by,
            n_effective_by_stratum=neff,
            fell_back=tuple(fell),
            n_calib=int(scores.size),
            n_dropped=n_dropped,
            hypo_threshold=float(hypo_threshold),
            hyper_threshold=float(hyper_threshold),
            taxonomy=taxonomy,
            calibrated_on=(
                tuple(sorted({str(pp) for pp in np.asarray(patient_calib)[ok]}))
                if patient_calib is not None else ()
            ),
        )

    # ------------------------------------------------------------ predict ---
    def qhat_for(self, stratum: int) -> float:
        """The qhat applied in one stratum; the pooled value for an unknown
        cell (including -1, the code for a non-finite prediction)."""
        return float(self.qhat_by_stratum.get(int(stratum), self.qhat_pooled))

    def _q_vector(self) -> np.ndarray:
        return np.array([self.qhat_for(k) for k in range(N_STRATA)], dtype=np.float64)

    def _distance_to_stratum(self, pred: np.ndarray, k: int) -> np.ndarray:
        """Distance from each prediction to the nearest point of stratum `k`.

        Zero when the prediction already lies in that zone. Infinities in the
        bounds cancel through the max, so the open-ended strata need no special
        case.
        """
        a, b = stratum_bounds(k, self.hypo_threshold, self.hyper_threshold)
        return np.maximum(np.maximum(a - pred, pred - b), 0.0)

    def _hull_of_label_conditional_set(
        self, pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convex hull of the union region, plus a mask of points it covers.

        Each stratum contributes the slice of its own conformal interval that
        actually lies inside its zone -- so a prediction of 200 mg/dL whose only
        reachable stratum is the hyperglycemic one gets [180, 224.6] rather than
        the symmetric [175.4, 224.6]: the sub-180 tail belongs to a zone that
        was not reached, and keeping it would widen the interval with a region
        the construction never claimed.
        """
        pieces = self.predict_set(pred, None)
        lo = np.full(pred.shape, np.inf)
        hi = np.full(pred.shape, -np.inf)
        any_active = np.zeros(pred.shape, dtype=bool)
        for lo_k, hi_k, active in pieces:
            lo = np.where(active, np.minimum(lo, lo_k), lo)
            hi = np.where(active, np.maximum(hi, hi_k), hi)
            any_active |= active
        return lo, hi, any_active

    def predict_set(
        self,
        pred: np.ndarray,
        stratum: np.ndarray | int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The EXACT label-conditional region, one entry per stratum.

        Returns [(lower_k, upper_k, active_k)] where `active_k` marks the points
        for which zone k contributes at all. The union of the active pieces is
        the conformal region; `predict_interval` returns its convex hull.

        Worth having because the hull overstates the width whenever two zones
        contribute disjoint pieces, and reporting the hull as "the interval
        width" would understate the method's efficiency.
        """
        p = np.asarray(pred, dtype=np.float64)
        if stratum is not None:
            k_fixed = np.broadcast_to(np.asarray(stratum, dtype=np.int64), p.shape)
        out = []
        for k in range(N_STRATA):
            a, b = stratum_bounds(k, self.hypo_threshold, self.hyper_threshold)
            q_k = self.qhat_for(k)
            if stratum is None:
                # DERIVED taxonomy: the label-conditional union. Each stratum
                # contributes only the slice of its interval that lies inside
                # its own zone -- that restriction IS the construction, and
                # `_hull_of_label_conditional_set` depends on it.
                active = self._distance_to_stratum(p, k) <= q_k
                lo = np.clip(np.maximum(p - q_k, a), 0.0, None)
                hi = np.clip(np.minimum(p + q_k, b), None, 600.0)
            else:
                # SUPPLIED taxonomy: X-measurable, so the Mondrian guarantee is
                # already exact and the region is `yhat +/- qhat_k`, whole. The
                # true y is under no obligation to land inside cell k, and
                # truncating to [a, b] would undercover exactly the points the
                # stratification exists to protect. It also made `set_measure`
                # disagree with the interval `predict_interval` actually emits
                # on this path -- which does not clip -- so the reported width
                # was narrower than the interval being scored.
                active = k_fixed == k
                lo = np.clip(p - q_k, 0.0, None)
                hi = np.clip(p + q_k, None, 600.0)
            out.append((lo, hi, active))
        return out

    def predict_interval(
        self,
        pred: np.ndarray,
        stratum: np.ndarray | int | None = None,
        expand: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """[yhat - qhat_k, yhat + qhat_k], clipped to the physiological range.

        `stratum` may be an explicit per-point assignment (an X-measurable rule,
        e.g. the zone of the last observed BG or of a trend extrapolation) or a
        single int applied to every point. Left as None it is read off `pred`.

        `expand` selects between the two constructions on the derived-stratum
        path: True (the default there) emits the hull of the label-conditional
        region and inherits the oracle's coverage; False emits the strict
        `yhat +/- qhat_{stratum(yhat)}`, which is the undercovering ablation
        arm. When `stratum` is supplied the assignment is taken as given -- the
        Mondrian guarantee is already exact for an X-measurable taxonomy -- and
        `expand` is ignored.
        """
        p = np.asarray(pred, dtype=np.float64)
        derived = stratum is None
        if expand is None:
            expand = derived

        if derived and expand:
            lo, hi, any_active = self._hull_of_label_conditional_set(p)
            # No stratum reachable at all means every qhat is nan, i.e. this
            # object was never calibrated. Emit the pooled (also nan) interval
            # rather than the empty [inf, -inf], so the failure reads as
            # "uncalibrated" instead of "confidently empty".
            lo = np.where(any_active, lo, p - self.qhat_pooled)
            hi = np.where(any_active, hi, p + self.qhat_pooled)
            return np.clip(lo, 0.0, None), np.clip(hi, None, 600.0)

        k = (clinical_stratum(p, self.hypo_threshold, self.hyper_threshold)
             if derived else np.broadcast_to(np.asarray(stratum, dtype=np.int64), p.shape))
        q = np.array([self.qhat_for(int(v)) for v in np.ravel(k)],
                     dtype=np.float64).reshape(p.shape)
        return np.clip(p - q, 0.0, None), np.clip(p + q, None, 600.0)

    def set_measure(
        self,
        pred: np.ndarray,
        stratum: np.ndarray | int | None = None,
    ) -> np.ndarray:
        """Total length of the exact label-conditional region, per point.

        The honest efficiency number. The hull returned by `predict_interval`
        can be far wider than the region whose coverage is guaranteed: when the
        hypoglycemic qhat is several times the euglycemic one, a euglycemic
        prediction's region is a narrow band around yhat PLUS a separate slice
        below 70, and the hull spans the empty gap between them.

        The pieces are disjoint by construction (each is confined to its own
        zone, and the zones partition the axis), so the measure is just the sum
        of their lengths -- no inclusion-exclusion required.
        """
        p = np.asarray(pred, dtype=np.float64)
        total = np.zeros(p.shape, dtype=np.float64)
        for lo, hi, active in self.predict_set(p, stratum):
            total = total + np.where(active, np.maximum(hi - lo, 0.0), 0.0)
        return total

    def evaluate(
        self,
        y_true: np.ndarray,
        pred: np.ndarray,
        patient: np.ndarray | None = None,
        stratum: np.ndarray | int | None = None,
        expand: bool | None = None,
    ) -> CoverageReport:
        """Coverage on held-out points, broken down by TRUE stratum.

        Note the asymmetry, which is the whole audit: the interval is built
        from the PREDICTED stratum (all that is available at inference) and
        graded against the TRUE one. Grading against the predicted stratum
        would be circular and would report the method as valid exactly when it
        misassigns.
        """
        lo, hi = self.predict_interval(pred, stratum, expand)
        return coverage_report(y_true, lo, hi, self.alpha, patient,
                               self.hypo_threshold, self.hyper_threshold)

    def __str__(self) -> str:
        cells = " | ".join(
            f"{STRATUM_NAMES[k]}: qhat {self.qhat_for(k):.2f} "
            f"(n={self.n_by_stratum.get(k, 0)}"
            f"{', POOLED FALLBACK' if k in self.fell_back else ''})"
            for k in range(N_STRATA)
        )
        return (
            f"MondrianConformal(alpha={self.alpha:.2f}, taxonomy={self.taxonomy}, "
            f"n={self.n_calib}, pooled qhat={self.qhat_pooled:.2f})\n  {cells}"
        )


# --------------------------------------------------------------------------- #
# Locally adaptive (normalised) conformal
# --------------------------------------------------------------------------- #
@dataclass
class NormalizedConformal:
    """Width that adapts to per-point difficulty.

    Score is s_i = |y_i - yhat_i| / (sigma_i + eps) where sigma_i is a
    difficulty estimate available at prediction time. The interval becomes
    yhat +/- qhat * (sigma + eps), so it widens where the model is likely to
    struggle.

    The difficulty estimate must be computable from the INPUT WINDOW ALONE.
    Using anything derived from the target would leak. The default,
    `rate_of_change_difficulty`, uses the recent absolute rate of change of
    glucose over the history window -- a steep descent is exactly when the
    30-minute forecast is hardest and when hypoglycemia is most likely, so the
    interval widens where clinical stakes are highest.

    Constant-width conformal is marginally valid but conditionally poor; this
    variant usually buys back a substantial part of that gap at the cost of
    slightly wider average intervals.
    """

    alpha: float
    qhat: float = float("nan")
    eps: float = 1.0
    n_calib: int = 0
    n_effective: float = float("nan")
    calibrated_on: tuple[str, ...] = ()

    @staticmethod
    def calibrate(
        y_calib: np.ndarray,
        pred_calib: np.ndarray,
        sigma_calib: np.ndarray,
        alpha: float,
        eps: float = 1.0,
        patient_calib: np.ndarray | None = None,
    ) -> "NormalizedConformal":
        resid = np.abs(np.asarray(y_calib, np.float64) - np.asarray(pred_calib, np.float64))
        scores = resid / (np.asarray(sigma_calib, np.float64) + eps)
        return NormalizedConformal(
            alpha=alpha,
            qhat=conformal_quantile(scores, alpha),
            eps=eps,
            n_calib=int(scores.size),
            n_effective=effective_sample_size(scores),
            calibrated_on=(
                tuple(sorted({str(p) for p in patient_calib}))
                if patient_calib is not None else ()
            ),
        )

    def predict_interval(
        self, pred: np.ndarray, sigma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pred, np.float64)
        half = self.qhat * (np.asarray(sigma, np.float64) + self.eps)
        return np.clip(p - half, 0.0, None), np.clip(p + half, None, 600.0)

    def evaluate(
        self,
        y_true: np.ndarray,
        pred: np.ndarray,
        sigma: np.ndarray,
        patient: np.ndarray | None = None,
    ) -> CoverageReport:
        lo, hi = self.predict_interval(pred, sigma)
        return coverage_report(y_true, lo, hi, self.alpha, patient)


def rate_of_change_difficulty(ws: WindowSet, idx: np.ndarray | None = None) -> np.ndarray:
    """Per-window difficulty proxy from the INPUT WINDOW ONLY.

    Mean absolute first difference over the last 6 history steps (30 min),
    i.e. how fast glucose has recently been moving. Computed from channel 0 of
    the already-causal feature tensor, so it inherits the prefix guarantee and
    cannot see past the anchor.
    """
    x = ws.x if idx is None else ws.x[idx]
    g = x[:, :, 0]
    d = np.abs(np.diff(g[:, -7:], axis=1))
    return d.mean(axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# Per-patient calibration under the causal prefix protocol
# --------------------------------------------------------------------------- #
@dataclass
class PatientConformal:
    """Calibrate separately for each patient on their own causal prefix.

    This is the deployment-honest variant. Pooled calibration certifies coverage
    averaged over the calibration population; it says nothing about a specific
    new individual, and per-patient coverage genuinely varies. Calibrating on the
    first K hours of the new patient's own record targets coverage for THAT
    patient -- and it costs nothing extra operationally, since the personalised
    adaptation in `extension.py` already assumes exactly this prefix is
    available.

    The prefix/embargo split is reused from `extension.py` verbatim rather than
    reimplemented, so the calibration data cannot overlap the evaluation windows.
    """

    alpha: float
    per_patient: dict[str, SplitConformal] = field(default_factory=dict)
    fallback: SplitConformal | None = None

    def qhat_for(self, patient: str) -> float:
        c = self.per_patient.get(str(patient))
        if c is not None and np.isfinite(c.qhat):
            return c.qhat
        # A patient with too short a prefix falls back to pooled calibration
        # rather than to an unbounded interval.
        return self.fallback.qhat if self.fallback is not None else float("inf")

    def predict_interval(
        self, pred: np.ndarray, patient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.qhat_for(p) for p in patient], dtype=np.float64)
        p = np.asarray(pred, np.float64)
        return np.clip(p - q, 0.0, None), np.clip(p + q, None, 600.0)

    def evaluate(
        self, y_true: np.ndarray, pred: np.ndarray, patient: np.ndarray
    ) -> CoverageReport:
        lo, hi = self.predict_interval(pred, patient)
        return coverage_report(y_true, lo, hi, self.alpha, patient)


def calibrate_per_patient(
    ws: WindowSet,
    pred: np.ndarray,
    patients: list[str],
    cfg,
    alpha: float,
    pooled_fallback: SplitConformal | None = None,
) -> PatientConformal:
    """Build a `PatientConformal` using each patient's causal prefix.

    `pred` must be aligned to `ws` (one prediction per window). The prefix and
    embargo come from `extension.prefix_embargo_split`, so the windows used for
    calibration are strictly earlier than, and non-overlapping with, the windows
    that will be scored.
    """
    from .extension import prefix_embargo_split

    per: dict[str, SplitConformal] = {}
    for pid in patients:
        calib_idx, _ = prefix_embargo_split(ws, pid, cfg)
        if calib_idx.size < 20:
            continue
        per[str(pid)] = SplitConformal.calibrate(
            ws.y_reg[calib_idx], pred[calib_idx], alpha,
            patient_calib=ws.patient[calib_idx],
        )
    return PatientConformal(alpha=alpha, per_patient=per, fallback=pooled_fallback)
