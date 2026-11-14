"""Clinical safety evaluation: error grids and glycemic-zone breakdowns.

WHY STATISTICAL METRICS ARE NOT ENOUGH
--------------------------------------
RMSE weights a 30 mg/dL error at 250 exactly like a 30 mg/dL error at 60. The
first is clinically irrelevant; the second is the difference between "no action"
and "eat 15 g of glucose now". A model can lower cohort RMSE while getting worse
at precisely the readings that matter, because hypoglycemia is ~5% of windows
and contributes ~5% of the loss.

Error grids fix the weighting by asking a different question: if a clinician
acted on this prediction, would the action have been correct, benign, or
harmful? Zones A and B are clinically acceptable; C, D and E are not, and D
(failure to detect) and E (erroneous opposite treatment) are the ones that hurt
patients.

Report **Zone A+B percentage and the D/E counts** alongside RMSE. A model with
better RMSE and more zone-E predictions is worse.

Clarke is implemented and validated here. Parkes is deliberately NOT -- see
`parkes_zones` for why.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Clarke error grid
# --------------------------------------------------------------------------- #
def clarke_zones(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Assign each (reference, prediction) pair a Clarke zone 0..4 = A..E.

    Implements the standard piecewise rules from Clarke et al. (1987), the same
    formulation used by the widely circulated reference implementations:

      A: within 20% of the reference, or both values below 70 (both would lead
         to the same treatment)
      B: outside 20% but would not lead to inappropriate treatment
      C: would prompt treatment the reference does not warrant (overcorrection)
      D: dangerous FAILURE TO DETECT -- reference is hypo or hyper, prediction
         says normal. This is the zone a hypoglycemia warning system fails in.
      E: erroneous treatment -- predicts hyper when the truth is hypo, or the
         reverse. The most harmful category.

    Inputs are clipped to the assay range [1, 600] mg/dL; the grid is undefined
    outside it and unclipped regressor outputs would otherwise fall off the
    edge of every rule and silently land in B.
    """
    y = np.clip(np.asarray(reference, dtype=np.float64), 1.0, 600.0)
    p = np.clip(np.asarray(prediction, dtype=np.float64), 1.0, 600.0)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: reference {y.shape} vs prediction {p.shape}")

    zone = np.full(y.shape, 1, dtype=np.int64)  # default B

    a = ((y <= 70) & (p <= 70)) | (np.abs(p - y) <= 0.2 * y)
    e = ((y >= 180) & (p <= 70)) | ((y <= 70) & (p >= 180))
    c = (((y >= 70) & (y <= 290)) & (p >= y + 110)) | \
        (((y >= 130) & (y <= 180)) & (p <= (7.0 / 5.0) * y - 182))
    d = ((y >= 240) & (p >= 70) & (p <= 180)) | \
        ((y <= 175.0 / 3.0) & (p <= 180) & (p >= 70)) | \
        (((y >= 175.0 / 3.0) & (y <= 70)) & (p >= (6.0 / 5.0) * y))

    # Order matters: the rules overlap, and severity must win, so the writes
    # run in INCREASING severity and the harshest applicable zone lands last. A
    # point that is both "within 20%" and "erroneous treatment" cannot exist,
    # but C/D/E regions do overlap at their corners, and scoring such a point as
    # the milder zone would understate exactly the risk this function exists to
    # measure.
    #
    # It used to write D before C, so a C-and-D point was recorded as C -- the
    # reassuring direction. Unreachable as the boundaries stand (the only C^D
    # overlap is `y == 70, p >= 180`, where E overwrites both), which is why no
    # test caught it; one boundary change makes it live.
    zone[c] = 2
    zone[d] = 3
    zone[e] = 4
    zone[a & ~(c | d | e)] = 0
    return zone


# --------------------------------------------------------------------------- #
# Parkes (consensus) error grid -- delegated to a verified external backend
# --------------------------------------------------------------------------- #
# WHY THIS IS NOT IMPLEMENTED HERE
# -------------------------------
# Clarke is defined by RULES (ratios and thresholds), so it can be implemented
# from the paper's text and checked against canonical points -- which is what
# `clarke_zones` does. Parkes is defined by published BOUNDARY POLYGONS, so a
# from-scratch implementation is a vertex transcription exercise, and a single
# mistyped coordinate shifts a boundary by a few mg/dL and silently moves
# borderline points between zones.
#
# An earlier hand-transcribed version of this function graded 0.5% of real
# predictions as A+B where Clarke graded 99.2%. It was removed rather than
# shipped behind a "verify before publication" comment, because a safety metric
# that is wrong in the REASSURING direction is worse than no metric at all.
#
# So: delegate to a maintained package, and verify the delegation rather than
# trusting it. `verify_parkes_backend()` below is the check.
_PARKES_BACKENDS = (
    # (module path, callable name, docstring-worthy note)
    ("methcomp.glucose", "parkeszones",
     "methcomp -- pip install methcomp; "
     "parkeszones(type, reference, test, units, numeric=True)"),
)


def _load_parkes_backend():
    """Return (name, fn) for the first available backend, or (None, None)."""
    import importlib
    for mod_path, fn_name, _ in _PARKES_BACKENDS:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None:
            return f"{mod_path}.{fn_name}", fn
    return None, None


def parkes_available() -> bool:
    """True when a Parkes backend is installed. Callers should degrade to
    Clarke-only reporting rather than failing when this is False."""
    return _load_parkes_backend()[0] is not None


def parkes_zones(reference: np.ndarray, prediction: np.ndarray,
                 diabetes_type: int = 1) -> np.ndarray:
    """Parkes (consensus) zones 0..4 = A..E, via an external verified backend.

    Parkes grades by the clinical risk of the action a prediction would prompt,
    on a smooth consensus surface, rather than by Clarke's hard 20% rule. The
    two disagree most near the A/B border, which is exactly why reporting both
    is worth the dependency: agreement between them is evidence, and
    disagreement localises where the model sits on a clinical boundary.

    Raises ImportError with install instructions when no backend is present --
    deliberately, rather than falling back to a home-grown approximation.
    """
    name, fn = _load_parkes_backend()
    if fn is None:
        raise ImportError(
            "No Parkes error grid backend installed. Install one with:\n"
            "    pip install methcomp\n"
            "This project does not ship a hand-transcribed Parkes grid; see the "
            "comment above `_PARKES_BACKENDS` for why. Clarke (`clarke_zones`) "
            "is implemented and validated here and needs no dependency."
        )
    y = np.asarray(reference, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    # UNITS ARE MANDATORY and not defaulted by the backend. Passing them
    # explicitly (rather than relying on a default that does not exist) is what
    # keeps a 'mmol' misreading -- which would grade every mg/dL value as
    # catastrophically hyperglycemic -- from being possible here.
    #
    # CHARACTERS, not the numeric flag. The numeric convention (0-based vs
    # 1-based) is not documented as stable, and an earlier version of this
    # adapter tried to infer it from the observed range -- which silently
    # shifted every zone by one whenever a batch happened to contain no zone-A
    # points. Letters are unambiguous, so there is nothing left to infer.
    raw = np.asarray(fn(diabetes_type, y, p, "mgdl"), dtype=object)
    lookup = {name: i for i, name in enumerate(ZONE_NAMES)}
    try:
        return np.array([lookup[str(z).strip().upper()] for z in raw.ravel()],
                        dtype=np.int64).reshape(raw.shape)
    except KeyError as exc:
        raise RuntimeError(
            f"{name} returned an unrecognised zone label {exc}; expected one of "
            f"{ZONE_NAMES}."
        ) from exc


# Properties that follow from the DEFINITION of an error grid, so they can be
# asserted without transcribing a vertex table -- and without being circular
# (i.e. without asserting values read back from the backend being tested).
#
# Note what is deliberately NOT asserted: that a specific extreme error is zone
# E. An earlier version asserted (50, 500) -> E and "failed" the backend, but
# the backend was right: Parkes places that point in D, and reserves E for
# references below ~35 mg/dL. Clarke and Parkes genuinely differ there. The
# checks below assert only what every error grid must satisfy.
_EXACT_AGREEMENT_POINTS = (40.0, 70.0, 100.0, 180.0, 300.0, 400.0)


@functools.lru_cache(maxsize=None)
def verify_parkes_backend(tolerance_pct: float = 8.0) -> dict:
    """Check the installed backend before trusting a single number from it.

    Four independent checks, none of which requires a vertex table and none of
    which asserts a value read back from the backend under test:

      1. EXACT AGREEMENT is zone A. True in every error grid by construction.

      2. ASYMMETRY: swapping reference and prediction must be able to change the
         zone. Error grids are deliberately asymmetric -- reading a low as high
         is more dangerous than the reverse -- so a backend called with its
         arguments transposed would produce a symmetric map. Exact-agreement
         points cannot catch this, because they are symmetric.

      3. MONOTONIC SEVERITY: for a fixed reference, moving the prediction
         further away must never make the zone milder.

      4. GROSS AGREEMENT WITH CLARKE on a synthetic sweep. The grids disagree
         near boundaries, so this is a band, not equality -- but the scrapped
         hand-written implementation failed it by 98 points, which is what this
         check exists to catch.

    Returns a dict describing what was checked. Raises on failure.

    MEMOISED. The check runs a 4,000-point synthetic sweep through the backend
    -- 0.94 s -- and `clinical_report` called it on every invocation, which is
    ~27 s per benchmark run spent re-verifying a backend that cannot change
    within a process. `lru_cache` also caches the raised exception's absence
    only; a failure re-raises on every call, so an untrusted backend does not
    become trusted by being asked twice.
    """
    name, fn = _load_parkes_backend()
    if fn is None:
        raise ImportError("No Parkes backend installed; nothing to verify.")

    # 1. exact agreement
    exact = np.array(_EXACT_AGREEMENT_POINTS)
    z = parkes_zones(exact, exact)
    if not (z == 0).all():
        bad = [(v, ZONE_NAMES[zz]) for v, zz in zip(exact, z) if zz != 0]
        raise AssertionError(
            f"Parkes backend {name}: exact agreement is not zone A at {bad}. "
            f"Check the argument order and the units parameter."
        )

    # 2. asymmetry (catches transposed arguments)
    lo_ref = parkes_zones(np.array([20.0]), np.array([500.0]))[0]
    hi_ref = parkes_zones(np.array([500.0]), np.array([20.0]))[0]
    if lo_ref == hi_ref:
        raise AssertionError(
            f"Parkes backend {name} graded (ref=20, pred=500) and "
            f"(ref=500, pred=20) identically as {ZONE_NAMES[lo_ref]}. An error "
            f"grid is asymmetric by design; this suggests reference and "
            f"prediction are being passed in the wrong order."
        )

    # 3. monotone severity away from the diagonal
    ref = np.full(6, 100.0)
    preds = np.array([100.0, 130.0, 170.0, 220.0, 300.0, 450.0])
    zz = parkes_zones(ref, preds)
    if not np.all(np.diff(zz) >= 0):
        raise AssertionError(
            f"Parkes backend {name}: zone severity is not monotone as the "
            f"prediction moves away from a fixed reference: {list(zz)}"
        )

    # 4. gross agreement with Clarke
    rng = np.random.default_rng(0)
    y = rng.uniform(40.0, 400.0, 4000)
    p = y + rng.normal(0.0, 20.0, 4000)
    cz, pz = clarke_zones(y, p), parkes_zones(y, p)
    c_ab = float(((cz == 0) | (cz == 1)).mean() * 100.0)
    p_ab = float(((pz == 0) | (pz == 1)).mean() * 100.0)
    if abs(c_ab - p_ab) > tolerance_pct:
        raise AssertionError(
            f"Parkes backend {name} disagrees grossly with Clarke on a synthetic "
            f"sweep: Clarke A+B {c_ab:.2f}% vs Parkes A+B {p_ab:.2f}% "
            f"(tolerance {tolerance_pct}%). The grids differ near boundaries, "
            f"not by tens of percent -- treat this as a broken integration."
        )
    return {"backend": name, "exact_agreement_points": len(_EXACT_AGREEMENT_POINTS),
            "clarke_ab_pct": round(c_ab, 3), "parkes_ab_pct": round(p_ab, 3)}


# --------------------------------------------------------------------------- #
# Zone-resolved report
# --------------------------------------------------------------------------- #
ZONE_NAMES = ("A", "B", "C", "D", "E")

# Clinically meaningful glycemic bands (ADA / international consensus on CGM
# metrics). Level 2 hypoglycemia (<54) is the clinically significant one.
GLYCEMIC_BANDS = (
    ("severe_hypo", 0.0, 54.0),
    ("hypo", 54.0, 70.0),
    ("target", 70.0, 180.0),
    ("hyper", 180.0, 250.0),
    ("severe_hyper", 250.0, np.inf),
)


@dataclass
class ClinicalReport:
    """Zone percentages plus per-band accuracy.

    `zone_ab_pct` is the headline safety number; `zone_de_n` is the count that
    should never be traded away for RMSE.
    """
    zone_ab_pct: float
    zone_pct: dict            # "A".."E" -> percent
    zone_de_n: int
    parkes_ab_pct: float | None   # None when no verified backend is installed
    band_rmse: dict           # band -> RMSE within that reference band
    band_n: dict
    hypo_band_bias: float     # mean signed error where reference < 70

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        z = " ".join(f"{k} {self.zone_pct[k]:5.2f}%" for k in ZONE_NAMES)
        bands = " | ".join(
            f"{b} {self.band_rmse[b]:.1f}(n={self.band_n[b]})"
            for b, *_ in GLYCEMIC_BANDS if self.band_n[b] > 0
        )
        # A POSITIVE bias below 70 means the model predicts HIGHER than the
        # truth during hypoglycemia -- it understates the low, which is the
        # dangerous direction for a warning system. Spelling that out because
        # the sign alone is routinely read backwards.
        direction = ("understates lows (dangerous direction)" if self.hypo_band_bias > 0
                     else "overstates lows (conservative direction)")
        pk = ("" if self.parkes_ab_pct is None
              else f"       Parkes A+B {self.parkes_ab_pct:6.2f}%\n")
        return (
            f"  GRID Clarke A+B {self.zone_ab_pct:6.2f}% | {z} | D+E n={self.zone_de_n}\n"
            + pk +
            f"  BAND RMSE {bands}\n"
            f"       bias below 70 mg/dL: {self.hypo_band_bias:+.2f} mg/dL -- {direction}"
        )


def clinical_report(reference: np.ndarray, prediction: np.ndarray,
                    with_parkes: bool = True) -> ClinicalReport:
    """Full clinical-safety view of a set of glucose predictions.

    Parkes is included only when a backend is installed AND passes
    `verify_parkes_backend`. An unverified or absent backend degrades the report
    to Clarke-only rather than failing the run or, worse, reporting an
    unvalidated safety number.
    """
    y = np.asarray(reference, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)

    cz = clarke_zones(y, p)
    counts = {n: float((cz == i).mean() * 100.0) for i, n in enumerate(ZONE_NAMES)}

    parkes_ab = None
    if with_parkes and parkes_available():
        try:
            verify_parkes_backend()
            pz = parkes_zones(y, p)
            parkes_ab = float(((pz == 0) | (pz == 1)).mean() * 100.0)
        except Exception as exc:      # unverified backend -> Clarke only
            import warnings
            warnings.warn(f"Parkes backend present but not trusted: {exc}")

    band_rmse, band_n = {}, {}
    for name, lo, hi in GLYCEMIC_BANDS:
        m = (y >= lo) & (y < hi)
        band_n[name] = int(m.sum())
        band_rmse[name] = float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else float("nan")

    hypo = y < 70.0
    return ClinicalReport(
        zone_ab_pct=counts["A"] + counts["B"],
        zone_pct=counts,
        zone_de_n=int(((cz == 3) | (cz == 4)).sum()),
        parkes_ab_pct=parkes_ab,
        band_rmse=band_rmse,
        band_n=band_n,
        hypo_band_bias=float(np.mean(p[hypo] - y[hypo])) if hypo.any() else float("nan"),
    )


# --------------------------------------------------------------------------- #
# Alarm / event primitives for alarm-fatigue metrics
# --------------------------------------------------------------------------- #
# WHY THESE LIVE HERE AND NOT IN evaluate.py
# ------------------------------------------
# `evaluate.py` computes numbers; this module defines what a clinical EVENT is.
# Both the false-alarm rate and the warning lead time hinge on one decision --
# what counts as *one* alarm and *one* hypoglycemic episode -- and getting that
# decision wrong changes the headline number by an order of magnitude, not by a
# rounding error. At a 5-minute cadence a single 40-minute spurious excursion
# above the alarm threshold is EIGHT threshold crossings but ONE alarm: the
# device beeps, the patient looks, and nothing further happens until the alarm
# clears and re-arms. Counting the eight would report 8x the false-alarm burden
# a patient actually experiences.
#
# So episode extraction is defined once, here, and both metrics consume it.

HYPO_THRESHOLD_MGDL: float = 70.0
SEVERE_HYPO_THRESHOLD_MGDL: float = 54.0


def segment_ids(
    t_time: np.ndarray | None = None,
    patient: np.ndarray | None = None,
    period_min: int = 5,
    n: int | None = None,
) -> np.ndarray:
    """Label each row with the contiguous recording segment it belongs to.

    Rows are assumed to be sorted by (patient, time) -- the caller owns that,
    because only the caller knows which ordering is meaningful. A new segment
    starts whenever

      * the patient changes, or
      * the gap to the previous row exceeds one sampling period.

    Both matter. Alarm episodes must not bridge two patients (two people cannot
    share one beep), and they must not bridge a sensor dropout: a run of
    above-threshold scores either side of a four-hour gap is two alarms, and
    treating it as one silently halves the reported burden.

    With `t_time=None` and `patient=None` every row lands in one segment, which
    is the right default for a synthetic contiguous series.
    """
    if n is None:
        for a in (t_time, patient):
            if a is not None:
                n = int(np.asarray(a).shape[0])
                break
    if n is None:
        raise ValueError("segment_ids needs t_time, patient, or an explicit n")
    if n == 0:
        return np.empty(0, dtype=np.int64)

    brk = np.zeros(n, dtype=bool)
    if patient is not None:
        p = np.asarray(patient)
        if p.shape[0] != n:
            raise ValueError(f"patient has {p.shape[0]} rows, expected {n}")
        brk[1:] |= p[1:] != p[:-1]
    if t_time is not None:
        t = np.asarray(t_time).astype("datetime64[m]")
        if t.shape[0] != n:
            raise ValueError(f"t_time has {t.shape[0]} rows, expected {n}")
        gap = (t[1:] - t[:-1]).astype("timedelta64[m]").astype(np.int64)
        # Backwards steps (an unsorted caller) are treated as a break rather
        # than silently folded into the previous run.
        brk[1:] |= (gap != int(period_min))
    return np.cumsum(brk).astype(np.int64)


def runs_of_true(mask: np.ndarray, segment: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Maximal runs of True in `mask`, as half-open [start, stop) index pairs.

    A run never crosses a segment boundary, so `segment` is what keeps an
    episode from spanning a sensor dropout or two different patients.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    if segment is None:
        seg = np.zeros(m.size, dtype=np.int64)
    else:
        seg = np.asarray(segment, dtype=np.int64)
        if seg.size != m.size:
            raise ValueError(f"segment has {seg.size} rows, mask has {m.size}")

    # A run starts where mask is True and the previous row is either False, the
    # start of the array, or in a different segment.
    prev_on = np.empty(m.size, dtype=bool)
    prev_on[0] = False
    prev_on[1:] = m[:-1] & (seg[1:] == seg[:-1])
    starts = np.flatnonzero(m & ~prev_on)

    next_on = np.empty(m.size, dtype=bool)
    next_on[-1] = False
    next_on[:-1] = m[1:] & (seg[1:] == seg[:-1])
    stops = np.flatnonzero(m & ~next_on) + 1

    return list(zip(starts.tolist(), stops.tolist()))


@dataclass(frozen=True)
class HypoEvent:
    """One contiguous excursion below the hypoglycemia threshold.

    `nadir` is the index of the lowest reading in the episode -- the moment the
    patient is in most danger, and the reference point the warning lead time is
    measured against.
    """
    start: int          # first index below threshold (inclusive)
    stop: int           # first index at or above threshold (exclusive)
    nadir: int          # index of the minimum within [start, stop)
    nadir_bg: float
    segment: int = 0

    @property
    def n_steps(self) -> int:
        return self.stop - self.start


def hypo_events(
    bg: np.ndarray,
    threshold: float = HYPO_THRESHOLD_MGDL,
    segment: np.ndarray | None = None,
    min_steps: int = 1,
) -> list[HypoEvent]:
    """Extract hypoglycemic episodes from a glucose series.

    An episode is a maximal run of `bg < threshold` within one recording
    segment. `min_steps` drops single-sample dips: the international consensus
    on CGM metrics counts an episode only once the reading has stayed below
    threshold for 15 minutes (3 steps at the 5-minute cadence), and the default
    of 1 is deliberately permissive rather than silently applying that rule to
    a caller who did not ask for it.

    Non-finite readings are treated as "not below threshold" -- a missing value
    is not evidence of hypoglycemia, and imputing one here would fabricate the
    very events the lead-time metric is scored against.
    """
    g = np.asarray(bg, dtype=np.float64)
    if g.size == 0:
        return []
    below = np.isfinite(g) & (g < float(threshold))
    seg = (np.zeros(g.size, dtype=np.int64) if segment is None
           else np.asarray(segment, dtype=np.int64))

    events: list[HypoEvent] = []
    for a, b in runs_of_true(below, seg):
        if (b - a) < int(min_steps):
            continue
        nadir = a + int(np.argmin(g[a:b]))
        events.append(HypoEvent(start=a, stop=b, nadir=nadir,
                                nadir_bg=float(g[nadir]), segment=int(seg[a])))
    return events


def alarm_episodes(
    alarm: np.ndarray,
    segment: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Group consecutive above-threshold steps into single alarm events.

    This is the clinical alarm model: the device fires once when the score
    crosses the threshold and stays latched until the score falls back below
    it. Counting each 5-minute step separately would report the number of
    threshold-exceeding SAMPLES, which is not a quantity any patient
    experiences.
    """
    return runs_of_true(np.asarray(alarm, dtype=bool), segment)


def latched_alarm(
    alarm: np.ndarray,
    segment: np.ndarray | None = None,
    merge_gap_steps: int = 0,
    min_duration_steps: int = 1,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Apply a device alarm policy to a per-step above-threshold mask.

    Returns `(on, episodes)`: `on` marks the steps at which the device alarm is
    ACTIVE, and `episodes` lists each alarm as a half-open `[sound, stop)`
    interval, one per sounding.

    Two knobs, applied in this order, both within a recording segment:

    * `min_duration_steps` (persistence filter): a run of above-threshold steps
      sounds only once it has lasted this many steps, at index
      `start + min_duration_steps - 1`; shorter runs never sound. 1 = sound on
      the first step (the policy `alarm_episodes` implements).
    * `merge_gap_steps` (latching): a run that begins within this many steps of
      the previous sounded run's end does not sound again; the alarm stays
      latched through the gap, so the gap steps are ACTIVE. 0 = no latching
      (any dip below threshold ends the alarm).

    With both at their defaults the episodes are exactly `alarm_episodes`.
    """
    m = np.asarray(alarm, dtype=bool)
    seg = (np.zeros(m.size, dtype=np.int64) if segment is None
           else np.asarray(segment, dtype=np.int64))
    d = max(1, int(min_duration_steps))
    runs = [(a + d - 1, b) for a, b in runs_of_true(m, seg) if b - a >= d]
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if (merged and seg[a] == seg[merged[-1][0]]
                and a - merged[-1][1] <= int(merge_gap_steps)):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    on = np.zeros(m.size, dtype=bool)
    for a, b in merged:
        on[a:b] = True
    return on, merged
