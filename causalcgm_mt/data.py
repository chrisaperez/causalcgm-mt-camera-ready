"""Causal CGM windowing, missing-value handling, targets and soft labels.

Design rule that governs this entire module
-------------------------------------------
A window emitted at anchor index ``t`` may read raw values at indices
``[t - H + 1, ..., t]`` and *nothing else*. Every derived quantity that ends up
in ``x`` is a function of that closed interval alone. Every statistic used to
transform ``x`` (means, stds, medians, IQRs, MARD estimates, class priors) is
estimated on the training patients of the current fold only.

Targets deliberately live in a separate array and are never transformed by a
fitted object, so there is no path by which target information can re-enter the
inputs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig

# Bump whenever a change alters the CONTENT of a window tensor, so an artefact
# built by an older pipeline is rebuilt rather than silently reused. Geometry
# checks (horizons / history / stride) cannot catch a change that keeps every
# shape identical and only moves values -- which is exactly what the
# backward-snap and per-step sleep-mask fixes did.
#
#   1  original: nearest-neighbour CGM snap, whole-file sleep_observed
#   2  strictly backward CGM snap (t-P, t]; per-step sleep_observed;
#      prefix-sum causal features
#
# It lives here, in the module that defines what a window IS, so that every
# producer and consumer of a window artefact can key on it without importing the
# benchmark harness. `benchmark.WINDOW_PIPELINE_VERSION` re-exports it.
WINDOW_PIPELINE_VERSION = 2


# --------------------------------------------------------------------------- #
# 0. Strictly causal grid snapping
# --------------------------------------------------------------------------- #
def snap_to_grid_backward(
    ts: pd.Series,
    values: pd.DataFrame,
    grid: pd.DatetimeIndex,
    period_min: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assign each raw sample to the grid slot whose backward bin contains it.

    Slot ``t`` covers the half-open interval ``(t - P, t]`` -- the same rule every
    other stream in this pipeline uses -- and takes the LAST sample in that bin.
    Returns the binned frame (NaN where no sample fell in the bin) and the
    observation mask.

    WHY NOT ``merge_asof(direction="nearest")``
    -------------------------------------------
    A nearest-neighbour snap with a half-period tolerance lets slot ``t`` take a
    sample recorded AFTER ``t``. On OhioT1DM that is not a corner case: the CGM
    clock drifts, ``minute mod 5`` is spread across 0-4, and 40% of matched grid
    slots were taking their value from up to 120 s in the future. That is bounded
    by half a sampling period and so is unlikely to move a headline metric, but
    it contradicts the invariant every docstring in this pipeline states, and it
    is a real train/serve skew: a streaming deployment at wall-clock ``t`` has no
    such sample.

    WHY ``last`` AND NOT ``mean``
    -----------------------------
    CGM is an instantaneous measurement. Averaging a bin blurs the excursions the
    model exists to forecast -- a one-sample nadir of 55 mg/dL becomes 65 after a
    bin mean, which moves it across the label boundary -- and it makes the
    regression target a synthetic average rather than a value the device
    reported. Taking the last sample keeps the target measured.

    A sample is used at most once, so an empty bin stays empty: this snaps, it
    never forward-fills. Filling is the caller's decision, made per window by
    ``_causal_gap_fill``.

    On data already aligned to the grid this is identical to the nearest snap.
    """
    t = pd.to_datetime(pd.Series(np.asarray(ts)).reset_index(drop=True))
    v = pd.DataFrame(values).reset_index(drop=True)
    period = pd.Timedelta(minutes=period_min)

    out = pd.DataFrame(index=range(len(grid)), columns=v.columns, dtype="float64")
    if len(grid) == 0 or t.empty:
        return out, np.zeros(len(grid), dtype=bool)

    # Bin index of each sample: the first grid slot at or after its timestamp.
    # `searchsorted(..., "left")` gives exactly that, and samples landing after
    # the last slot or more than one period before the first are dropped.
    gv = pd.DatetimeIndex(grid).values
    idx = np.searchsorted(gv, t.values, side="left")
    keep = (idx < len(grid))
    # Guard the leading edge: a sample more than one period before grid[0] would
    # otherwise be credited to slot 0.
    keep &= (t.values > (gv[np.clip(idx, 0, len(grid) - 1)] - np.timedelta64(period)))

    idx, sub = idx[keep], v.loc[keep]
    if idx.size == 0:
        return out, np.zeros(len(grid), dtype=bool)

    # Later samples overwrite earlier ones in the same bin -> "last in bin".
    out.iloc[idx] = sub.to_numpy()
    observed = np.zeros(len(grid), dtype=bool)
    observed[idx] = True
    return out, observed


# --------------------------------------------------------------------------- #
# 1. Per-patient regularisation of the raw series
# --------------------------------------------------------------------------- #
def regularize_series(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> pd.DataFrame:
    """Put one patient's record on a strict uniform grid and mark observations.

    Returns a frame indexed 0..N-1 on the uniform grid with:
      - ``glucose``      : value, NaN where unobserved
      - ``observed``     : 1.0 where the value came from the device
      - ``segment``      : integer id, incremented across gaps longer than
                           ``max_series_gap_min`` (windows never cross segments)

    No imputation happens here; this is purely a resampling/alignment step, and
    resampling to a grid uses only the timestamps, not the values.
    """
    if cfg.exog_cols:
        # Refused, not ignored. These columns were binned onto the grid here and
        # then dropped by `build_windows`, which reads `glucose_col` alone -- a
        # knob that silently did nothing. See `DataConfig.exog_cols`.
        raise ValueError(
            f"DataConfig.exog_cols={tuple(cfg.exog_cols)!r} is not supported by "
            f"the generic windowing path: `build_windows` emits CGM features "
            f"only, so these columns would be binned and then discarded. Use a "
            f"multimodal loader (hupa_loader / ohio_loader / metabonet_loader), "
            f"which carries exogenous streams in a separate `x_aux` tensor.")

    d = df.sort_values(cfg.time_col).copy()
    d[cfg.time_col] = pd.to_datetime(d[cfg.time_col])

    # Collapse duplicate timestamps (device artefacts) by mean -- a same-instant
    # operation, so it cannot look ahead.
    value_cols = [cfg.glucose_col, *cfg.exog_cols]
    d = d.groupby(cfg.time_col, as_index=False)[list(value_cols)].mean()

    period = pd.Timedelta(minutes=cfg.sample_period_min)
    grid = pd.date_range(d[cfg.time_col].iloc[0], d[cfg.time_col].iloc[-1], freq=period)

    # Assign each raw sample to the backward bin (t - P, t] that contains it.
    # Strictly causal: slot t never sees a sample recorded after t.
    out = pd.DataFrame({cfg.time_col: grid})
    binned, _obs = snap_to_grid_backward(
        d[cfg.time_col], d[list(value_cols)], grid, cfg.sample_period_min)
    for c in value_cols:
        out[c] = binned[c].to_numpy()

    out["observed"] = out[cfg.glucose_col].notna().astype(np.float32)

    # Segment boundaries: runs of consecutive missing steps longer than the
    # allowed series gap split the record. Computed from the observation mask
    # only.
    max_gap_steps = max(1, cfg.max_series_gap_min // cfg.sample_period_min)
    miss = out["observed"].values == 0.0
    seg = np.zeros(len(out), dtype=np.int64)
    run = 0
    sid = 0
    for i, m in enumerate(miss):
        if m:
            run += 1
        else:
            if run > max_gap_steps:
                sid += 1
            run = 0
        seg[i] = sid
    out["segment"] = seg
    return out.reset_index(drop=True)


def _causal_gap_fill(
    values: np.ndarray,
    observed: np.ndarray,
    max_gap_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of short interior gaps.

    IMPORTANT: this is applied *per window, over the history slice only*. Within
    a history window ``[t-H+1, t]`` all endpoints of an interpolated gap are
    themselves inside the window, so the filled value is a function of data at
    or before ``t``. It is therefore causal with respect to the anchor. It is
    NOT valid to run this over a whole patient series before windowing, because
    a gap ending after ``t`` would pull future observations backwards -- that is
    exactly the leak this function's call site is structured to avoid.

    Returns the filled array and a mask of which entries were imputed.
    """
    v = values.astype(np.float64).copy()
    obs = observed.astype(bool)
    imputed = np.zeros_like(obs)

    idx = np.flatnonzero(obs)
    if idx.size == 0:
        return v, ~obs

    i = 0
    while i < idx.size - 1:
        a, b = idx[i], idx[i + 1]
        gap = b - a - 1
        if 0 < gap <= max_gap_steps:
            v[a + 1 : b] = np.interp(np.arange(a + 1, b), [a, b], [v[a], v[b]])
            imputed[a + 1 : b] = True
        i += 1

    # Leading/trailing unobserved steps are left NaN: filling them would require
    # either a future value (leading) or an extrapolation the model should not
    # be asked to trust.
    return v, imputed


# --------------------------------------------------------------------------- #
# 2. Causal feature construction
# --------------------------------------------------------------------------- #
ROLL_WINDOW: int = 6


def causal_features(hist: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Per-timestep features from the history window only.

    Every channel at position ``i`` depends solely on ``hist[:i+1]``, so the
    feature tensor is causal even *within* the window. This is stricter than
    necessary (the model sees the whole window anyway) but it means the same
    feature code can be reused for streaming inference without modification.

    Channels: value, first difference (ROC), rolling mean(6), rolling std(6),
    observation mask.

    Vectorised via prefix sums rather than a per-position Python loop. The loop
    cost 326 us per window, which is ~49 s over the OhioT1DM cohort every time
    the window cache is rebuilt; this is 0.48 us. The accumulation is float64
    even though the output is float32, because the naive E[x^2] - E[x]^2 form
    loses about 0.03 mg/dL to cancellation in float32 -- small, but it is the
    kind of difference that shows up as a C++/Python parity failure rather than
    as anything visible in a metric.
    """
    return causal_features_batch(np.asarray(hist)[None], np.asarray(observed)[None])[0]


def causal_features_batch(hist: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """`causal_features` over a stack of windows: (N, H) -> (N, H, 5).

    The batched form is what the windowing pass should use once it holds all its
    history slices; the single-window entry point above delegates to it so the
    two cannot drift.
    """
    h = np.asarray(hist, dtype=np.float64)
    if h.ndim == 1:
        h = h[None]
    obs = np.asarray(observed, dtype=np.float64).reshape(h.shape)
    n, H = h.shape
    w = ROLL_WINDOW

    diff = np.zeros_like(h)
    diff[:, 1:] = h[:, 1:] - h[:, :-1]

    # Backward window [i - w + 1, i], clipped at the start of the slice: the same
    # prefix property the loop had, expressed as a difference of cumulative sums.
    c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(h, axis=1)], axis=1)
    c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(h * h, axis=1)], axis=1)
    idx = np.arange(H)
    lo = np.maximum(0, idx - w + 1)
    cnt = (idx - lo + 1).astype(np.float64)

    s1 = c1[:, idx + 1] - c1[:, lo]
    s2 = c2[:, idx + 1] - c2[:, lo]
    roll_mean = s1 / cnt
    # Population variance (ddof=0), matching np.std's default and the C++ runner.
    var = np.maximum(s2 / cnt - roll_mean * roll_mean, 0.0)
    roll_std = np.sqrt(var)
    roll_std[:, 0] = 0.0            # a single-element window has no spread

    return np.stack([h, diff, roll_mean, roll_std, obs], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Soft labels near the hypoglycemia boundary
# --------------------------------------------------------------------------- #
# Robust-statistics constants, named rather than inlined because getting one of
# them wrong is invisible in every downstream metric (see README "Status", 3).
MEDIAN_ABS_TO_SIGMA = 0.6744897501960817   # median(|X|) = 0.6745 sigma, X ~ N
IQR_TO_SIGMA = 1.3489795003921634          # IQR(X)      = 1.3490 sigma, X ~ N
P1_P99_TO_SIGMA = 4.6526957480816816       # p99 - p1    = 4.6527 sigma, X ~ N
# (p99 - p1) / IQR under normality. A spread taken as the WIDER of the IQR and
# (p99 - p1) / TAIL_TO_IQR therefore equals the IQR on normal data and only
# widens when the middle 50% has collapsed relative to the channel's real
# dynamic range -- see `CausalScaler.fit_array(tail_aware=True)`. Defined once,
# here, because `ohio_loader.PerPatientNormalizer` applies the same policy and
# two copies of a constant like this drift.
TAIL_TO_IQR = P1_P99_TO_SIGMA / IQR_TO_SIGMA          # 3.4490485190687412

# The bound a scaled AUXILIARY feature may reach. Not applied to the CGM path:
# that tensor is built by `causal_features` from a physiologically bounded
# signal and peaks at a healthy 22.4, whereas the auxiliary tensor aggregates
# device streams spanning six decades. Clipping CGM at this bound would move it
# (and with it every CGM-only arm and the exported ONNX graph), so both knobs
# are opt-in and the CGM scaler is left bit-identical by construction.
SCALED_FEATURE_CLIP: float = 20.0
D2_VARIANCE_INFLATION = 6.0                # Var(x_t - 2x_{t-1} + x_{t-2}) = 6 s^2
# median(|X|) -> E|X| for a half-normal: sqrt(2/pi) / 0.6745. Converts a robust
# median absolute relative deviation onto the scale of a published MARD, which
# is by definition a MEAN absolute relative deviation.
MEDIAN_ARD_TO_MARD = 1.1829120487267799

# Lag-1 autocorrelation of the first difference, used as an identifiability
# diagnostic. For a signal that is (smooth trend + i.i.d. measurement noise),
# rho1(diff) -> -0.5 as the noise dominates and rises towards +1 as the trend
# does. A released CGM stream that has been through the manufacturer's smoothing
# filter has no high-frequency content left to measure, and reports rho1 well
# above this line -- OhioT1DM reads +0.43. Above the threshold, a high-pass
# noise estimate is a LOWER BOUND on device error, not a measurement of it.
WHITE_NOISE_RHO1 = -0.5
SMOOTHED_SIGNAL_RHO1 = -0.25

# Physiologically possible capillary reference readings, in mg/dL. OhioT1DM
# subject 540 records 143 finger sticks as exactly 0.0 -- a "no value" sentinel,
# not a measurement -- and dividing by it turns that subject's mean ARD into
# 4022%. The screen is what makes a paired estimate usable at all.
REFERENCE_BG_RANGE_MGDL = (20.0, 600.0)


@dataclass(frozen=True)
class NoiseEstimate:
    """What the fold's CGM error scale is, and how confidently we know it.

    ``mard`` is the number the soft-label band is derived from. ``source``
    records WHICH of the three estimators produced it, because the three are not
    equally trustworthy and a band that came from a device prior must not be
    reported as though it had been measured:

      ``paired_reference``  robust MARD against paired capillary reference BG.
                            This is MARD by its actual definition and is the
                            only tier that is a measurement rather than a proxy.
      ``high_pass``         robust noise scale from the signal's own
                            high-frequency content. Valid when the stream still
                            carries device noise; a lower bound once it has been
                            smoothed (see ``smoothed``).
      ``device_floor``      neither of the above was available or identifiable,
                            so a published device MARD is used as a prior.
      ``unavailable``       no data at all; the caller decides what to do.
    """
    mard: float
    source: str
    n: int = 0
    level_mgdl: float = float("nan")
    rho1: float = float("nan")
    smoothed: bool = False
    detail: str = ""

    @property
    def is_measured(self) -> bool:
        """True when ``mard`` came from data rather than from a prior."""
        return self.source in ("paired_reference", "high_pass")

    def as_dict(self) -> dict:
        return {
            "mard": float(self.mard), "source": self.source, "n": int(self.n),
            "level_mgdl": float(self.level_mgdl), "rho1": float(self.rho1),
            "smoothed": bool(self.smoothed), "detail": self.detail,
        }


def _first_difference_rho1(s: np.ndarray) -> float:
    """Lag-1 autocorrelation of the first difference. See ``SMOOTHED_SIGNAL_RHO1``."""
    d1 = np.diff(s)
    if d1.size < 3:
        return float("nan")
    x = d1 - d1.mean()
    v = float(x.var())
    if v <= 0:
        return float("nan")
    return float((x[:-1] * x[1:]).mean() / v)


def estimate_signal_noise(
    series_by_patient: Iterable[np.ndarray],
    period_min: int,
) -> NoiseEstimate:
    """Robust device-noise scale from the CGM stream alone.

    THE ESTIMATOR
    -------------
    Second-differencing is the high-pass filter: a linear ramp has zero second
    difference, so ``d2 = x_t - 2 x_{t-1} + x_{t-2}`` retains the jitter and
    discards the trend. The scale of ``d2`` is then read off with an
    INTERQUARTILE RANGE rather than a median of absolute values:

        sigma = IQR(d2) / (1.3490 * sqrt(6))

    Both forms are unbiased on Gaussian noise, and the suite pins that at 2%, 5%
    and 10%. The IQR is preferred for two reasons that matter on real data:
    it reads the signed distribution, so a trend that survives the high-pass
    shifts it rather than inflating it; and its breakdown behaviour is governed
    by the middle 50% of the sample rather than by a single order statistic of
    magnitudes, which is what a heavy-tailed sensor artefact moves first.

    The relative scale divides by a ROBUST LEVEL (the median of the series), not
    by the pointwise value as the previous proxy did. Dividing pointwise weights
    every hypoglycemic sample by ~2x relative to a hyperglycemic one purely
    because the denominator is smaller, which biases the estimate by the
    cohort's glycemic distribution rather than by its device.

    WHAT IT CANNOT DO
    -----------------
    It measures the high-frequency content that is PRESENT IN THE RELEASED
    SIGNAL. If the manufacturer's filter removed the device noise before the
    file was written, there is nothing here to recover and the estimate is a
    lower bound. ``rho1`` reports exactly that condition rather than leaving the
    caller to infer it from an implausibly small number: OhioT1DM reads +0.43
    against the -0.5 of a noise-dominated stream.
    """
    rels: list[float] = []
    rhos: list[float] = []
    total = 0
    levels: list[float] = []
    for s in series_by_patient:
        s = np.asarray(s, dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size < 8:
            continue
        d2 = s[2:] - 2 * s[1:-1] + s[:-2]
        q75, q25 = np.percentile(d2, [75, 25])
        sigma = (q75 - q25) / (IQR_TO_SIGMA * np.sqrt(D2_VARIANCE_INFLATION))
        level = float(np.median(s))
        if not np.isfinite(level) or level <= 1.0:
            continue
        rels.append(sigma / level)
        levels.append(level)
        rho = _first_difference_rho1(s)
        if np.isfinite(rho):
            rhos.append(rho)
        total += int(s.size)

    if not rels:
        return NoiseEstimate(mard=float("nan"), source="unavailable",
                             detail="no series long enough to difference")

    rho1 = float(np.median(rhos)) if rhos else float("nan")
    smoothed = bool(np.isfinite(rho1) and rho1 > SMOOTHED_SIGNAL_RHO1)
    return NoiseEstimate(
        mard=float(np.median(rels)),
        source="high_pass",
        n=total,
        level_mgdl=float(np.median(levels)),
        rho1=rho1,
        smoothed=smoothed,
        detail=(f"IQR high-pass over {len(rels)} series at {period_min}-min cadence; "
                f"rho1={rho1:.3f} "
                + ("(smoothed: this is a LOWER BOUND on device error)"
                   if smoothed else "(noise-dominated: estimate is identifiable)")),
    )


def anchor_series_by_patient(ws, idx: np.ndarray):
    """One time-ordered anchor series per patient: the noise estimator's input.

    `estimate_signal_noise` DIFFERENCES what it is given, so it needs a series.
    Both fitters used to hand it a generator over individual WINDOWS
    (`ws.x[i, :, 0]`, one H-step gap-filled history slice per call), which is
    the wrong object in two ways that never raised:

      * `rho1` -- the identifiability diagnostic deciding `NoiseEstimate.
        smoothed`, written verbatim into every results JSON -- was computed over
        24-step slices instead of the record. On OhioT1DM the per-window pass
        reports +0.523 where the series reports +0.435, and +0.43 is the figure
        `README.md` quotes;
      * the estimate was biased low by the pipeline's own imputation, since
        channel 0 carries `_causal_gap_fill`'s output and an interpolated run has
        identically zero second difference;
      * and `n` counted overlapping history steps, overstating the sample behind
        the estimate by a factor of H (2,320,608 against 96,692 anchors).

    `last_bg` is the anchor value of each window, so a time-ordered pass over a
    patient's anchors reconstructs the series the estimator was always meant to
    see -- and does it ~400x faster than one call per window.
    """
    idx = np.asarray(idx)
    pat = np.asarray([str(p) for p in ws.patient[idx]])
    for pid in sorted(set(pat.tolist())):
        rows = idx[pat == pid]
        rows = rows[np.argsort(ws.t_time[rows], kind="stable")]
        yield np.asarray(ws.last_bg[rows], dtype=np.float64)


def estimate_reference_mard(
    cgm_mgdl: np.ndarray,
    reference_mgdl: np.ndarray,
    subject: np.ndarray | None = None,
) -> NoiseEstimate:
    """MARD by its actual definition, against paired capillary reference BG.

    MARD is ``mean(|CGM - reference| / reference)`` over paired readings. Every
    other tier in this module is a proxy for it because a proxy is all a lone
    CGM stream supports -- but OhioT1DM ships ``finger_stick`` events, so on
    that cohort the real quantity is available and there is no reason to
    estimate around it.

    Three things make the difference between a usable number and a useless one:

    1. A PLAUSIBILITY SCREEN on the reference. Subject 540 records 143 finger
       sticks as exactly 0.0, and the resulting division sends that subject's
       mean ARD to 4022% and the pooled mean to 642%. Readings outside
       ``REFERENCE_BG_RANGE_MGDL`` are not measurements.
    2. A PER-SUBJECT MEDIAN, then the median across subjects. Pairs are wildly
       unbalanced (73 to 557 per subject here), so pooling would let the
       best-instrumented subject set the cohort's band.
    3. A DOCUMENTED MEDIAN-TO-MEAN CONVERSION. The robust statistic is a median
       absolute relative deviation; a published MARD is a mean. They differ by
       the half-normal factor ``MEDIAN_ARD_TO_MARD`` = 1.183, and reporting the
       median as though it were a MARD would understate the band by 15%.
    """
    c = np.asarray(cgm_mgdl, dtype=np.float64).ravel()
    r = np.asarray(reference_mgdl, dtype=np.float64).ravel()
    if c.size != r.size:
        raise ValueError("cgm and reference arrays must be the same length")
    lo, hi = REFERENCE_BG_RANGE_MGDL
    ok = (np.isfinite(c) & np.isfinite(r) & (r >= lo) & (r <= hi)
          & (c >= lo) & (c <= hi))
    n_screened = int((~ok).sum())
    c, r = c[ok], r[ok]
    if c.size == 0:
        return NoiseEstimate(mard=float("nan"), source="unavailable",
                             detail=f"no plausible reference pairs "
                                    f"({n_screened} screened out)")

    ard = np.abs(c - r) / r
    if subject is None:
        per_subject = [float(np.median(ard))]
    else:
        sub = np.asarray(subject, dtype=object).ravel()[ok]
        per_subject = [float(np.median(ard[sub == s])) for s in np.unique(sub)]
    median_ard = float(np.median(per_subject))
    return NoiseEstimate(
        mard=median_ard * MEDIAN_ARD_TO_MARD,
        source="paired_reference",
        n=int(c.size),
        level_mgdl=float(np.median(r)),
        detail=(f"{c.size} paired readings over {len(per_subject)} subject(s), "
                f"{n_screened} screened as implausible; median ARD "
                f"{median_ard * 100:.2f}% -> MARD {median_ard * MEDIAN_ARD_TO_MARD * 100:.2f}%"),
    )


def estimate_mard_proxy(series_by_patient: Iterable[np.ndarray], period_min: int) -> float:
    """Backwards-compatible scalar wrapper over `estimate_signal_noise`.

    Kept because the whole pipeline used to call it and its unbiasedness on
    known noise is a pinned regression test. New code should call
    `estimate_signal_noise` and read the provenance it returns; a bare float
    cannot say whether it was measured or guessed, which is the failure this
    module's rework was about.
    """
    est = estimate_signal_noise(series_by_patient, period_min)
    return 0.10 if not np.isfinite(est.mard) else float(est.mard)


@dataclass
class SoftLabeler:
    """Statistically guided soft labels for the hypoglycemia head.

    Motivation. A hard indicator ``1[BG < 70]`` treats 69.9 and 40.0 mg/dL as
    identical, and treats 69.9 vs 70.1 as maximally different. But CGM readings
    carry a measurement error on the order of 9-12% MARD, so near the threshold
    the *true* class is genuinely uncertain. Training on a hard label in that
    band forces the network to fit device noise.

    Formulation. Let ``m`` be the minimum glucose over the prediction window and
    ``T`` the threshold. The soft target is a logistic ramp centred on ``T``:

        y_soft = sigmoid((T - m) / s),   s = band / 4

    so that ``y_soft`` is ~0.5 at the threshold and saturates within roughly
    +/- band mg/dL. Outside the band the label is effectively hard, so
    far-from-boundary examples are unaffected.

    HOW THE BAND IS SET, AND WHY IT IS AN ABLATION ARM RATHER THAN A CONSTANT
    ------------------------------------------------------------------------
    ``band_mode="fixed"`` takes ``cfg.soft_band_mgdl`` verbatim: the band is a
    hyperparameter and the labeller does not touch it.

    ``band_mode="mard"`` derives it from the fold's estimated device error,
    ``band = k * MARD * T``, and then applies the policy in `DataConfig`:

      * the MARD is floored at ``cfg.band_mard_floor``, a published device
        specification, because no CGM is more accurate than its own spec and a
        released stream that has been smoothed will estimate lower than one;
      * the resulting band is clamped into
        ``[cfg.band_min_mgdl, cfg.band_max_mgdl]``, the window inside which a
        soft label is doing something a hard label does not -- below it the
        logistic saturates inside one reportable unit, above it the ramp
        reaches into euglycemia.

    Every one of those steps is RECORDED (`band_source`, `band_raw_mgdl`,
    `mard_used`, `noise`) rather than warned about. This class used to emit a
    `UserWarning` for a degenerate band, which meant the only way to run the arm
    was to not run it: drivers pinned ``band_mode="fixed"`` and the MARD path
    was computed and discarded. Provenance on the object makes the mode a
    sweepable arm -- see Ablation E in `scripts/run_paper_ablations.py` -- and
    keeps a floored band from being reported as a measured one.

    NOTE ON FIDELITY: soft labelling at the hypoglycemia boundary is due to Fang
    et al. (MT-HypoNet, iScience 29(4):115294, 2026), whose exact functional form
    this does NOT claim to reproduce -- it should be confirmed against their
    text before any fidelity claim is made. The knobs (`band_mode`,
    `soft_band_mgdl`, `band_mard_multiplier`, the floor/clamp policy, the
    label floor/ceiling) let you match a specific spec without touching the rest
    of the pipeline. Set ``soft_labels=False`` for the hard baseline.
    """

    threshold: float
    band: float
    floor: float = 0.02
    ceiling: float = 0.98
    enabled: bool = True
    # ---- provenance ---------------------------------------------------------
    band_source: str = "fixed"       # fixed | mard | device_floor | clamped_low
                                     # | clamped_high
    band_raw_mgdl: float = float("nan")   # before the floor and the clamp
    mard_used: float = float("nan")       # the MARD the band was derived from
    noise: NoiseEstimate | None = None    # the estimate, when one was supplied

    @staticmethod
    def fit(
        min_bg_train: np.ndarray,
        cfg: DataConfig,
        mard_train: float | None = None,
        noise: NoiseEstimate | None = None,
    ) -> "SoftLabeler":
        """Fit the band using TRAINING-FOLD statistics only.

        ``min_bg_train`` is passed so alternative band rules (e.g. a quantile of
        the training distribution around the threshold) can be implemented here
        without changing call sites.

        ``noise`` is the preferred input: it carries the estimate AND where it
        came from. ``mard_train`` remains accepted as a bare float for the call
        sites (and tests) that only have one; it is treated as a `high_pass`
        estimate, which is what it always was.
        """
        T = float(cfg.hypo_threshold_mgdl)
        common = dict(
            threshold=T,
            floor=float(cfg.soft_label_floor),
            ceiling=float(cfg.soft_label_ceiling),
            enabled=bool(cfg.soft_labels),
        )

        if noise is None and mard_train is not None:
            noise = NoiseEstimate(mard=float(mard_train), source="high_pass",
                                  detail="scalar MARD supplied by the caller")

        if cfg.band_mode == "fixed":
            # A fixed band is a hyperparameter, not an estimate: it is used
            # verbatim, including values the mard path would clamp away, because
            # a test that asks for a 0.5 mg/dL band is asking for exactly that.
            return SoftLabeler(band=float(cfg.soft_band_mgdl), band_source="fixed",
                               band_raw_mgdl=float(cfg.soft_band_mgdl),
                               noise=noise, **common)

        # ---- band_mode == "mard" --------------------------------------------
        mard = noise.mard if (noise is not None and np.isfinite(noise.mard)) else float("nan")
        source = "mard"
        if not np.isfinite(mard) or mard < cfg.band_mard_floor:
            # Fallback WITH SCALING: rather than silently substituting an
            # unrelated constant, fall back to the device's published error and
            # push it through the same band formula, so `k` still means what it
            # means in the estimated case.
            mard = float(cfg.band_mard_floor)
            source = "device_floor"

        raw = float(cfg.band_mard_multiplier) * mard * T
        band = float(np.clip(raw, cfg.band_min_mgdl, cfg.band_max_mgdl))
        if band != raw:
            source = "clamped_low" if raw < cfg.band_min_mgdl else "clamped_high"
        return SoftLabeler(band=band, band_source=source, band_raw_mgdl=raw,
                           mard_used=float(mard), noise=noise, **common)

    # ---- reporting ---------------------------------------------------------- #
    def as_dict(self) -> dict:
        """Serialisable provenance, for the results JSON.

        A band in a paper table is not interpretable without knowing whether it
        was measured, floored or clamped, and that information exists only here.
        """
        return {
            "enabled": bool(self.enabled),
            "threshold_mgdl": float(self.threshold),
            # `None`, not the configured width, when the ramp is off: a hard-label
            # arm has no band, and printing the band it would have had puts a
            # number in the ablation's independent-variable column for the one
            # row where it does not apply. `band_raw_mgdl` still records what was
            # configured, so nothing is lost.
            "band_mgdl": float(self.band) if self.enabled else None,
            "band_source": self.band_source if self.enabled else "disabled",
            "band_raw_mgdl": float(self.band_raw_mgdl),
            "mard_used": float(self.mard_used),
            "label_floor": float(self.floor),
            "label_ceiling": float(self.ceiling),
            "noise": self.noise.as_dict() if self.noise is not None else None,
        }

    @property
    def is_effectively_hard(self) -> bool:
        """True when the ramp saturates inside one reportable CGM unit.

        The band is in mg/dL and CGM reports integers, so a band below 2 mg/dL
        puts the whole transition inside a single display step: the labels are
        hard ones wearing a disguise, and a soft-vs-hard ablation on that
        setting measures nothing. Exposed as a property so a caller can assert
        on it -- `band_mode="mard"` can no longer reach this state, but
        `band_mode="fixed"` still can, deliberately.
        """
        return bool(self.enabled and self.band < 2.0)

    def __call__(self, min_bg: np.ndarray) -> np.ndarray:
        hard = (min_bg < self.threshold).astype(np.float32)
        if not self.enabled:
            return hard
        s = self.band / 4.0
        # Numerically stable logistic: the naive form overflows exp() for
        # severe hypoglycemia with a narrow band (z can reach several hundred).
        z = (self.threshold - np.asarray(min_bg, dtype=np.float64)) / s
        soft = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
        return np.clip(soft, self.floor, self.ceiling).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. The windowing pass
# --------------------------------------------------------------------------- #
@dataclass
class WindowSet:
    """Arrays are aligned along axis 0. ``patient`` supports grouped splitting."""
    x: np.ndarray            # (N, H, C) float32 -- model input
    y_reg: np.ndarray        # (N,) float32     -- BG at t + PH, mg/dL
    y_cls_hard: np.ndarray   # (N,) float32     -- 1[min BG in window < T]
    y_min: np.ndarray        # (N,) float32     -- min BG over the window
    patient: np.ndarray      # (N,) object      -- patient id
    t_index: np.ndarray      # (N,) int64       -- anchor index within segment
    t_time: np.ndarray       # (N,) datetime64  -- anchor timestamp
    last_bg: np.ndarray      # (N,) float32     -- BG at t, for persistence baseline

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(**{f.name: getattr(self, f.name)[idx] for f in dataclasses.fields(self)})


def build_windows(df: pd.DataFrame, cfg: DataConfig) -> WindowSet:
    """Generate all valid causal windows for a long-format multi-patient frame.

    For each anchor ``t``:
      inputs  <- indices [t-H+1 .. t]      (closed, backward-looking)
      targets <- indices [t+1 .. t+PH]     (never touched by feature code)

    A window is emitted only if:
      * history and target both lie inside one segment (no recording break),
      * after short-gap interpolation the imputed fraction is within budget,
      * the anchor value and every target step were actually observed.
    """
    H, P = cfg.history_steps, cfg.horizon_steps
    max_gap_steps = max(0, cfg.max_interpolation_gap_min // cfg.sample_period_min)

    xs, xs_obs, yr, ym, pids, tix, tts, lbg = [], [], [], [], [], [], [], []

    for pid, pdf in df.groupby(cfg.patient_col, sort=True):
        reg = regularize_series(pdf, cfg)
        g = reg[cfg.glucose_col].values.astype(np.float64)
        obs = reg["observed"].values.astype(bool)
        seg = reg["segment"].values
        ts = reg[cfg.time_col].values

        for _, sidx in pd.Series(np.arange(len(reg))).groupby(seg):
            si = sidx.values
            if si.size < H + P:
                continue
            gs, os_, tss = g[si], obs[si], ts[si]

            for a in range(H - 1, si.size - P, cfg.stride_steps):
                hs = slice(a - H + 1, a + 1)
                tgt = slice(a + 1, a + 1 + P)

                # --- targets: strictly no imputation ------------------------
                if not os_[tgt].all() or not os_[a]:
                    continue
                fut = gs[tgt]

                # --- history: causal short-gap fill -------------------------
                hv, himp = _causal_gap_fill(gs[hs], os_[hs], max_gap_steps)
                valid = os_[hs] | himp
                if not valid.all():
                    continue
                if himp.mean() > cfg.max_imputed_frac:
                    continue

                xs.append(hv); xs_obs.append(os_[hs].astype(np.float64))
                yr.append(fut[-1])                                   # BG(t+PH)
                ym.append(fut.min() if cfg.classify_over_window else fut[-1])
                pids.append(pid)
                tix.append(a)
                tts.append(tss[a])
                lbg.append(hv[-1])

    if not xs:
        raise ValueError("No valid windows produced; check cadence and gap settings.")

    y_min = np.asarray(ym, dtype=np.float32)
    return WindowSet(
        x=causal_features_batch(np.stack(xs), np.stack(xs_obs)),
        y_reg=np.asarray(yr, dtype=np.float32),
        y_cls_hard=(y_min < cfg.hypo_threshold_mgdl).astype(np.float32),
        y_min=y_min,
        patient=np.asarray(pids, dtype=object),
        t_index=np.asarray(tix, dtype=np.int64),
        t_time=np.asarray(tts),
        last_bg=np.asarray(lbg, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# 5. Scaling -- fit on train patients, apply everywhere
# --------------------------------------------------------------------------- #
# An interquartile range at or below this is not a divisor. The absolute floor
# catches an exactly-constant middle 50% (every event-sparse channel); the
# relative one catches an IQR that is negligible against the channel's own
# magnitude, which is the same degeneracy expressed in whatever unit the sensor
# happens to report in.
_ABS_IQR_FLOOR: float = 1e-9
_REL_IQR_FLOOR: float = 1e-6
# Below this many non-zero samples a quantile of them is noise, not a scale.
_MIN_SPARSE_EVENTS: int = 8


def _sparse_channel_scale(column: np.ndarray) -> float:
    """A usable scale for a channel whose interquartile range is degenerate.

    Read off the channel's OWN non-zero entries, in descending order of how much
    they say about its spread:

      1. their interquartile range      -- a real spread, when one exists;
      2. their median absolute deviation -- when the IQR is degenerate too
         (a handful of distinct event sizes);
      3. their typical magnitude         -- when every event has one value, so
         there is no spread at all but there is still a unit;
      4. 1.0, the identity               -- nothing to read, so leave the
         channel raw rather than invent a scale for it.

    Raw MAD, not MAD * 1.4826: `CausalScaler` divides by a raw IQR rather than
    by a Gaussian-calibrated sigma, so the substituted scale is on the same
    footing as the one it replaces.
    """
    col = np.asarray(column, dtype=np.float64)
    nz = col[np.isfinite(col) & (col != 0.0)]
    if nz.size < _MIN_SPARSE_EVENTS:
        return 1.0
    q75, q25 = np.percentile(nz, [75, 25])
    med = np.median(nz)
    for candidate in (q75 - q25, np.median(np.abs(nz - med)), np.median(np.abs(nz))):
        if float(candidate) > _ABS_IQR_FLOOR:
            return float(candidate)
    return 1.0


@dataclass
class CausalScaler:
    """Robust per-channel scaler.

    Deliberately *not* sklearn's: we need (a) an explicit ``fitted_on`` record so
    an audit can confirm which patients contributed statistics, and (b) channel
    exclusions (the binary observation mask must not be scaled).

    Uses median/IQR, which is far less sensitive to hypoglycemic and
    hyperglycemic excursions than mean/std.
    """
    center: np.ndarray
    scale: np.ndarray
    skip_channels: tuple[int, ...]
    fitted_on: tuple[str, ...]
    # Channels whose IQR was degenerate and whose scale was therefore
    # SUBSTITUTED by `_sparse_channel_scale`. Recorded because a substitution
    # that nothing reports is as invisible as the blow-up it replaced. Defaulted
    # so an artefact pickled before this field existed still loads; read it with
    # `getattr(scaler, "degenerate_channels", ())`.
    degenerate_channels: tuple[int, ...] = ()
    # Symmetric bound applied by `transform`, or None for the affine map alone.
    # Defaulted, and read through `getattr` in `transform`, so a scaler pickled
    # before this field existed still loads and still behaves as it did.
    clip: float | None = None

    @staticmethod
    def fit_array(
        x: np.ndarray,
        patient: np.ndarray,
        train_patients: Sequence[str],
        skip_channels: tuple[int, ...] = (),
        tail_aware: bool = False,
        clip: float | None = None,
    ) -> "CausalScaler":
        """Fit on an arbitrary (N, T, C) stream.

        Factored out of `fit` so the multimodal loader can scale each modality
        stream with the same train-only guarantee, rather than reimplementing
        (and potentially weakening) the policy per stream.
        """
        mask = np.isin(patient, np.asarray(list(train_patients), dtype=object))
        if not mask.any():
            raise ValueError("No training windows to fit the scaler on.")
        xt = x[mask].reshape(-1, x.shape[-1])
        med = np.median(xt, axis=0)
        q75, q25 = np.percentile(xt, [75, 25], axis=0)
        iqr = q75 - q25

        # TAIL-AWARE SPREAD (opt-in), for a tensor that has already been through
        # a per-patient normaliser.
        #
        # `OhioPreprocessor.transform` runs `PerPatientNormalizer` and then this
        # scaler over the same array. Bounding the first stage (H3) COMPRESSES
        # the pooled distribution -- every subject's spread grew by up to 1e4 --
        # so this scaler's GSR interquartile range fell to 0.0250 and it
        # re-expanded that bound by ~40x, putting 797.8 back into the fusion
        # branch. Neither stage is wrong alone; the defect is the composition.
        #
        # Taking the wider of the IQR and the 1-99 percentile span reads the
        # channel's actual dynamic range when the middle 50% sits on a near-zero
        # bulk. Measured on the real cohort's auxiliary tensor: gsr scale
        # 0.02507 -> 1.0181 and max |z| 797.8 -> 57.5, at which point the clip
        # below binds on 0.0056% of values rather than 1.99% -- so the rescale
        # is the fix and the clip is a guarantee, not a second lossy transform.
        #
        # Off by default. Enabling it for the CGM scaler would move it (channel
        # 1's scale 5.0 -> 8.70, max |z| 22.4 -> 12.88), taking every CGM-only
        # arm and the exported graph with it.
        if tail_aware:
            p99, p1 = np.percentile(xt, [99, 1], axis=0)
            iqr = np.maximum(iqr, (p99 - p1) / TAIL_TO_IQR)

        # A DEGENERATE IQR IS NOT A DIVIDE-BY-ZERO TO BE CLAMPED AWAY.
        #
        # This used to be `np.clip(q75 - q25, 1e-6, None)`, written to avoid a
        # divide-by-zero. On an event-sparse channel it converted one into a
        # MULTIPLICATION BY 1e6. Auxiliary streams here are overwhelmingly
        # sparse -- a bolus lands in ~3% of 5-minute bins, carbohydrates in ~1%
        # -- so any channel that is zero in more than 75% of (window, timestep)
        # pairs has q75 == q25 == 0, the clamp fires, and every non-zero entry
        # (every actual event, i.e. the channel's entire information content) is
        # scaled by a million.
        #
        # Measured on OhioT1DM before this fix: six of 24 auxiliary channels
        # (steps, bolus, carbs, finger_stick, exercise_intensity,
        # accel_magnitude) sat on the 1e-6 floor and entered the fusion branch
        # at magnitudes up to 5.9e8. Nothing showed it -- the observation masks
        # are excluded from scaling, so the availability gate kept reporting
        # itself open at 95% of timesteps.
        #
        # The magnitude proxy is built from statistics already computed, so the
        # test costs no extra pass over `xt`: a sparse channel has med == q75 ==
        # 0 and is caught by the absolute floor, while a dense channel is judged
        # against its own scale.
        magnitude = np.maximum(np.abs(med), np.abs(q75))
        floor = np.maximum(_ABS_IQR_FLOOR, _REL_IQR_FLOOR * magnitude)
        skip = {int(c) for c in skip_channels}
        degenerate: list[int] = []
        for c in np.flatnonzero(iqr <= floor):
            c = int(c)
            if c in skip:
                continue                     # overwritten unconditionally below
            degenerate.append(c)
            # A scale the channel's own non-zero entries actually set, so a
            # 0.5-unit bolus and an 8-unit bolus stay 16x apart afterwards.
            # `PerPatientNormalizer.fit` has always handled this case; this
            # brings the older and far more widely used scaler into line.
            iqr[c] = _sparse_channel_scale(xt[:, c])

        # Binary masks and indicator channels must pass through untouched:
        # rescaling a 0/1 flag by an IQR of 0 or 1 is either a no-op or a
        # divide-by-epsilon blowup, and either way destroys its meaning.
        for c in skip_channels:
            med[c], iqr[c] = 0.0, 1.0
        return CausalScaler(
            center=med.astype(np.float32),
            scale=iqr.astype(np.float32),
            skip_channels=tuple(skip_channels),
            fitted_on=tuple(sorted({str(p) for p in patient[mask]})),
            degenerate_channels=tuple(degenerate),
            clip=None if clip is None else float(clip),
        )

    @staticmethod
    def fit(ws: WindowSet, train_patients: Sequence[str], skip_channels: tuple[int, ...] = (4,)) -> "CausalScaler":
        return CausalScaler.fit_array(ws.x, ws.patient, train_patients, skip_channels)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """The frozen affine map, optionally bounded.

        The clip is symmetric and therefore order-preserving below its bound, so
        this stays a rescale rather than becoming a reshuffle. `getattr` because
        a scaler pickled before `clip` existed has no such attribute, and those
        artefacts must keep loading and behaving exactly as they did.
        """
        z = (x - self.center) / self.scale
        bound = getattr(self, "clip", None)
        if bound is not None:
            z = np.clip(z, -bound, bound)
        return z.astype(np.float32)


# --------------------------------------------------------------------------- #
# 6. Patient-wise splitting with a temporal embargo option
# --------------------------------------------------------------------------- #
def patient_folds(patients: Sequence[str], n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic grouped K-fold over *patients*, never over rows.

    Splitting by row would place windows from the same patient -- and often
    overlapping windows sharing most of their history -- on both sides of the
    split, inflating scores dramatically. Splitting by patient makes the
    evaluation an honest estimate of generalisation to an unseen individual.
    """
    uniq = np.array(sorted({str(p) for p in patients}), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq.size)
    buckets = np.array_split(order, n_folds)
    folds = []
    for b in buckets:
        test = uniq[b]
        train = np.array([p for p in uniq if p not in set(test)], dtype=object)
        folds.append((train, test))
    return folds


def split_by_patient(ws: WindowSet, patients: Sequence[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(ws.patient, np.asarray(list(patients), dtype=object)))


# --------------------------------------------------------------------------- #
# Auxiliary presence: the per-window gate on the fusion branch
# --------------------------------------------------------------------------- #
# Treatment streams carry no observation mask, because a zero is a real zero
# (no bolus in this bin) rather than a missing reading. The per-window presence
# flag used to be derived from observation masks ALONE, so these channels could
# never open the gate: on MetaboNet the fusion branch ran on 0.41% of windows
# (14 of 1,838 subjects) while insulin-on-board was non-zero in 68% of them,
# and on OhioT1DM 2020 it was shut whenever no finger stick or logged sleep fell
# in the window, even with the Empatica streams and pump data present.
TREATMENT_CHANNEL_NAMES: tuple[str, ...] = (
    "iob", "cob", "bolus", "carbs", "basal_rate", "basal_iob", "basal_dose",
)


def aux_presence(
    x_aux_raw: np.ndarray,
    channel_names: Sequence[str],
    use: Iterable[str] | None = None,
) -> np.ndarray:
    """(N,) float32: 1 where a window carries ANY usable auxiliary information.

    A window is present when, somewhere in its history,
      * an observation mask among the channels in use is set, or
      * a mask-free treatment channel in use is non-zero -- delivered basal or
        bolus insulin, insulin or carbs on board, logged carbohydrate.

    `x_aux_raw` must be UNSCALED: after scaling a treatment zero is no longer
    zero. `use` restricts the decision to the channels an arm actually
    receives (an arm that zeroes a modality must not be gated open by it);
    None means every channel. Everything is read from the window's own history,
    so the flag is causal.
    """
    names = list(channel_names)
    allowed = set(names) if use is None else set(use)
    mask_idx = [i for i, c in enumerate(names) if c.endswith("_observed") and c in allowed]
    treat_idx = [i for i, c in enumerate(names)
                 if c in TREATMENT_CHANNEL_NAMES and c in allowed]
    x = np.asarray(x_aux_raw)
    present = np.zeros(x.shape[0], dtype=bool)
    if mask_idx:
        present |= (x[..., mask_idx] > 0.5).any(axis=(1, 2))
    if treat_idx:
        present |= (np.abs(x[..., treat_idx]) > 0).any(axis=(1, 2))
    return present.astype(np.float32)


def uses_unmasked_channels(channel_names: Sequence[str],
                           use: Iterable[str] | None = None) -> bool:
    """Whether the channels in use include a mask-free (treatment) stream.

    When they do, every timestep of a present window carries real information,
    so the fusion branch must not take per-step validity from the wearable
    masks alone -- see `MultiModalCausalCGMMT(always_valid=...)`.
    """
    allowed = set(channel_names) if use is None else set(use)
    return any(c in TREATMENT_CHANNEL_NAMES and c in allowed for c in channel_names)
