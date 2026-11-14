"""HUPA-UCM multimodal ingestion, synchronisation and windowing.

The dataset couples CGM with Fitbit wearable streams (heart rate, steps, sleep)
and treatment logs (insulin boluses, carbohydrate intake). Those streams arrive
at incompatible cadences and with incompatible semantics, and each class needs a
DIFFERENT alignment rule. Applying one rule to all of them is the fastest way to
introduce a leak.

    stream           native cadence      semantics        alignment rule
    ---------------------------------------------------------------------
    CGM              ~5 min              instantaneous    snap to grid
    heart rate       1 s - 1 min         instantaneous    mean over (t-P, t]
    steps            1 min, cumulative   interval count   sum over (t-P, t]
    sleep            episodic intervals  state            forward fill + staleness
    bolus insulin    point events        impulse          sum into (t-P, t], then IOB
    carbohydrates    point events        impulse          sum into (t-P, t], then COB

The single rule that makes all of this safe:

    A grid slot labelled t summarises the half-open interval (t - P, t].

That is a BACKWARD-looking bin. Every aggregate at t is computed from samples at
or before t. The pandas incantation is `closed="right", label="right"`; the
default (`closed="left", label="left"`) labels the bin [t, t+P) as t, which
silently attributes up to one full period of FUTURE data to the present slot.
That default is the single most likely source of leakage in a multimodal CGM
pipeline, and it is the reason this module never calls a bare `.resample()`.

Similarly: forward fill is causal (it propagates the past forward), backward
fill is not (it drags the future backward). `_causal_ffill` is the only fill in
this module, and `assert_no_bfill` exists so a reader can grep for the
guarantee.

SCHEMA CAVEAT
-------------
HUPA-UCM's exact column names and file layout are not verified here. Everything
schema-dependent is isolated in `HupaSchema`, which validates against the real
files at load time and raises with the columns it actually found. Adapt the
schema; nothing downstream needs to change.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import (_causal_gap_fill, causal_features_batch,
                   snap_to_grid_backward)


# --------------------------------------------------------------------------- #
# Canonical aux channel layout
# --------------------------------------------------------------------------- #
# Fixed, explicit and named. A silent reordering between training and serving
# would be invisible in every metric, so the order is defined once here and
# asserted in the tests.
AUX_CHANNELS: tuple[str, ...] = (
    "heart_rate",        # 0  bpm, mean over the bin
    "steps",             # 1  count, summed over the bin
    "sleep",             # 2  0/1 asleep, forward-filled state
    "iob",               # 3  units, causal insulin-on-board
    "cob",               # 4  grams, causal carbs-on-board
    "bolus",             # 5  units delivered inside this bin
    "carbs",             # 6  grams ingested inside this bin
    "hr_observed",       # 7  mask
    "steps_observed",    # 8  mask
    "sleep_observed",    # 9  mask
    "hr_staleness",      # 10 minutes since last real HR sample (capped)
)
AUX_INDEX = {name: i for i, name in enumerate(AUX_CHANNELS)}

# Channels the scaler must leave alone: binary masks carry meaning only as 0/1.
AUX_SKIP_SCALING: tuple[int, ...] = (
    AUX_INDEX["hr_observed"],
    AUX_INDEX["steps_observed"],
    AUX_INDEX["sleep_observed"],
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HupaSchema:
    """Maps raw HUPA-UCM column names onto canonical names.

    Defaults follow the layout commonly reported for the released CSVs. They are
    a starting point, NOT a verified contract -- `validate` reports exactly what
    it found so a mismatch is a one-line fix rather than a debugging session.
    """
    time: str = "time"
    glucose: str = "glucose"
    heart_rate: str = "heart_rate"
    steps: str = "steps"
    calories: str = "calories"
    sleep: str = "sleep"
    bolus: str = "bolus_volume_delivered"
    basal: str = "basal_rate"
    carbs: str = "carb_input"

    # Streams whose absence is tolerable (the model handles missing modalities
    # via masks); glucose and time are not optional.
    optional: tuple[str, ...] = ("sleep", "calories", "basal")

    def required_columns(self) -> list[str]:
        return [
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("optional",) and f.name not in self.optional
        ]

    def validate(self, df: pd.DataFrame, path: str) -> list[str]:
        """Return the list of canonical streams actually present."""
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing required columns {missing}.\n"
                f"Columns present: {sorted(df.columns.tolist())}\n"
                f"Adapt HupaSchema(...) to this file's layout."
            )
        present = []
        for f in dataclasses.fields(self):
            if f.name == "optional":
                continue
            if getattr(self, f.name) in df.columns:
                present.append(f.name)
        return present


@dataclass(frozen=True)
class HupaConfig:
    """Alignment and pharmacokinetic parameters."""

    # ---- staleness budgets --------------------------------------------------
    # How long a forward-filled value stays usable. Beyond this the channel is
    # marked unobserved rather than carried indefinitely: a heart rate from six
    # hours ago is not evidence about the present, and letting it look like one
    # is a subtler failure than simply having no data.
    hr_max_staleness_min: int = 15
    steps_max_staleness_min: int = 15
    sleep_max_staleness_min: int = 60
    staleness_cap_min: int = 120        # clip the staleness feature itself

    # ---- insulin pharmacokinetics ------------------------------------------
    # Rapid-acting analogue (aspart/lispro). IOB decay follows a two-exponential
    # activity curve; `iob_dia_min` is duration of action, `iob_peak_min` the
    # activity peak. These are population defaults -- per-patient PK varies
    # substantially and is a candidate for the personalisation stage.
    iob_dia_min: int = 300
    iob_peak_min: int = 75

    # ---- carbohydrate absorption -------------------------------------------
    cob_dur_min: int = 180
    cob_peak_min: int = 45

    # ---- modality availability ---------------------------------------------
    # Drop a window if the aux stream is mostly absent across its history: a
    # window that is 90% mask carries no multimodal information and just adds
    # noise to the fusion branch.
    min_aux_observed_frac: float = 0.30


# --------------------------------------------------------------------------- #
# Causal primitives
# --------------------------------------------------------------------------- #
def _causal_ffill(
    values: np.ndarray,
    observed: np.ndarray,
    max_staleness_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward fill with a staleness budget. The ONLY fill used in this module.

    Forward fill is causal: the value written at t was observed at some t' <= t.
    Backward fill would write a value observed at t' > t, which is precisely
    lookahead. There is deliberately no bfill path here, not even an option.

    Returns (filled, still_observed, staleness_steps). `still_observed` is 0
    where the last real sample is older than the budget, so downstream code sees
    an honest mask rather than a stale value dressed up as fresh.
    """
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    staleness = np.full(n, np.inf, dtype=np.float64)

    last_val = np.nan
    last_age = np.inf
    for i in range(n):
        if observed[i]:
            last_val = values[i]
            last_age = 0.0
        elif np.isfinite(last_age):
            last_age += 1.0

        if np.isfinite(last_age) and last_age <= max_staleness_steps and np.isfinite(last_val):
            out[i] = last_val
            valid[i] = True
            staleness[i] = last_age
        # else: leave 0.0 / False -- the mask channel carries the information.

    return out, valid, staleness


def _bin_backward(
    ts: pd.Series,
    values: pd.Series,
    grid: pd.DatetimeIndex,
    period_min: int,
    how: Literal["mean", "sum"],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate irregular samples into BACKWARD-looking grid bins.

    Slot t covers (t - period, t]. Implemented with an explicit right-closed,
    right-labelled resample rather than pandas' default, which labels [t, t+P)
    as t and would attribute up to a full period of future data to slot t.

    Returns (aggregate, observed_mask).
    """
    s = pd.Series(np.asarray(values, dtype=np.float64), index=pd.DatetimeIndex(ts))
    s = s[np.isfinite(s.values)]
    if s.empty:
        return np.zeros(len(grid)), np.zeros(len(grid), dtype=bool)

    rule = f"{period_min}min"
    agg = s.resample(rule, closed="right", label="right").agg(how)
    cnt = s.resample(rule, closed="right", label="right").count()

    agg = agg.reindex(grid)
    cnt = cnt.reindex(grid).fillna(0)

    observed = (cnt.values > 0)
    out = np.where(observed, np.nan_to_num(agg.values), 0.0)
    return out.astype(np.float64), observed


def _activity_kernel(
    duration_min: int, peak_min: int, period_min: int, kind: Literal["iob", "cob"]
) -> np.ndarray:
    """Remaining-amount curve, sampled on the grid, normalised to start at 1.

    IOB uses the standard bi-exponential rapid-acting model; COB uses a simpler
    piecewise-linear absorption. The important structural property for this
    module is not the exact pharmacology but that the kernel is applied as a
    CAUSAL FIR filter: the amount on board at t is a function of doses at times
    <= t and of nothing else.
    """
    n = max(2, duration_min // period_min + 1)
    t = np.arange(n) * period_min

    if kind == "iob":
        # Two-compartment exponential (Hovorka-style tail), parameterised so the
        # activity peak lands near `peak_min`.
        tau = peak_min * (1 - peak_min / duration_min) / (1 - 2 * peak_min / duration_min)
        a = 2 * tau / duration_min
        S = 1 / (1 - a + (1 + a) * np.exp(-duration_min / tau))
        frac = 1 - S * (1 - a) * (
            (t**2 / (tau * duration_min * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1
        )
    else:
        # Trapezoidal absorption: linear ramp to peak, linear decay to zero.
        rate = np.clip(np.minimum(t / max(peak_min, 1), (duration_min - t) / max(duration_min - peak_min, 1)), 0, None)
        absorbed = np.cumsum(rate)
        frac = 1 - absorbed / max(absorbed[-1], 1e-9)

    frac = np.clip(frac, 0.0, 1.0)
    frac[0] = 1.0
    frac[-1] = 0.0
    return frac.astype(np.float64)


def causal_on_board(impulses: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """On-board amount from an impulse train, by causal FIR convolution.

        on_board[t] = sum_{d <= t} impulse[d] * kernel[t - d]

    `np.convolve(..., mode="full")` truncated to the original length is exactly
    this sum -- index t of the full convolution touches only impulse indices
    <= t. The truncation removes the tail that extends past the series, never
    anything that would have pulled a future dose backwards.
    """
    if impulses.size == 0:
        return impulses.astype(np.float64)
    return np.convolve(impulses.astype(np.float64), kernel, mode="full")[: impulses.size]


def assert_no_bfill(df: pd.DataFrame) -> None:
    """Check that no column carries the signature of a backward fill.

    This used to be `return None` with a docstring claiming it was
    "documentation-as-code" -- a function that asserted nothing and so could not
    contradict the future edit it was supposedly guarding against. It now checks
    the one thing that is actually checkable at this point in the pipeline.

    A backward fill propagates a value BACKWARDS across an unobserved run, so it
    leaves a masked channel non-zero where its own observation flag is clear.
    Forward fill cannot do that: `_causal_ffill` writes zero and clears the mask
    together once the staleness budget is exhausted. Any `value != 0` under
    `mask == 0` therefore means something wrote a value the mask does not
    account for.

    Raises AssertionError rather than warning: a bfill anywhere in this module
    invalidates every downstream number, so it should stop the run.
    """
    for name, mask in (("heart_rate", "hr_observed"),
                       ("sleep", "sleep_observed")):
        if name not in df.columns or mask not in df.columns:
            continue
        v = np.asarray(df[name].values, dtype=np.float64)
        m = np.asarray(df[mask].values, dtype=np.float64) > 0.5
        bad = int(np.count_nonzero((~m) & (v != 0.0)))
        if bad:
            raise AssertionError(
                f"{name}: {bad} step(s) carry a non-zero value while "
                f"{mask} is clear. `_causal_ffill` clears value and mask "
                f"together, so this is the signature of a backward fill.")


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def to_canonical(df: pd.DataFrame, schema: HupaSchema | None = None) -> pd.DataFrame:
    """Rename raw dataset columns to the canonical names used downstream.

    Everything after this point speaks canonical names only, so a schema change
    touches exactly one place.
    """
    schema = schema or HupaSchema()
    mapping = {
        getattr(schema, f.name): f.name
        for f in dataclasses.fields(schema)
        if f.name != "optional" and getattr(schema, f.name) in df.columns
    }
    out = df.rename(columns=mapping)
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"])
    return out


def load_hupa_files(
    directory: str,
    schema: HupaSchema | None = None,
    patient_from_filename: bool = True,
) -> pd.DataFrame:
    """Read every per-patient CSV in `directory` into one long frame.

    Patient identity comes from the filename by default, because concatenating
    subjects without a reliable id would silently break the patient-wise
    splitting that the whole evaluation rests on.
    """
    schema = schema or HupaSchema()
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {directory}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        schema.validate(df, p)
        df = df.rename(columns={
            getattr(schema, f.name): f.name
            for f in dataclasses.fields(schema)
            if f.name != "optional" and getattr(schema, f.name) in df.columns
        })
        df["patient_id"] = (
            os.path.splitext(os.path.basename(p))[0] if patient_from_filename
            else df.get("patient_id", os.path.basename(p))
        )
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} files, {len(out)} rows, "
          f"{out.patient_id.nunique()} patients")
    return out


# --------------------------------------------------------------------------- #
# Synchronisation
# --------------------------------------------------------------------------- #
def synchronize_patient(
    pdf: pd.DataFrame,
    cfg: DataConfig,
    hcfg: HupaConfig,
) -> pd.DataFrame:
    """Put one patient's multimodal record on the shared 5-minute CGM grid.

    Every column produced here obeys: value at t depends only on samples at
    times <= t. The grid is defined by the CGM stream, because CGM is the
    modality the targets come from and re-gridding it would change the
    prediction problem.
    """
    P = cfg.sample_period_min
    d = pdf.sort_values("time").copy()
    d["time"] = pd.to_datetime(d["time"])

    grid = pd.date_range(d["time"].iloc[0].ceil(f"{P}min"),
                         d["time"].iloc[-1].floor(f"{P}min"),
                         freq=f"{P}min")
    if len(grid) < 2:
        return pd.DataFrame()

    out = pd.DataFrame({"time": grid})

    # ---- CGM: strictly backward snap into (t - P, t] -----------------------
    # Instantaneous measurement, so the bin takes the LAST sample rather than a
    # mean; a mean would blur the very excursions we are trying to predict. The
    # bin is backward-looking for the same reason every other stream's is: a
    # nearest-neighbour snap lets slot t take a sample recorded after t.
    cgm = d.loc[d["glucose"].notna(), ["time", "glucose"]]
    binned, _ = snap_to_grid_backward(cgm["time"], cgm[["glucose"]], grid, P)
    out["glucose"] = binned["glucose"].to_numpy()
    out["glucose_observed"] = out["glucose"].notna().astype(np.float32)

    steps_grid = lambda m: max(0, int(round(m / P)))  # noqa: E731

    # ---- heart rate: mean over (t-P, t], then bounded forward fill ----------
    if "heart_rate" in d.columns:
        hr, hr_obs = _bin_backward(d["time"], d["heart_rate"], grid, P, "mean")
        hr_f, hr_valid, hr_stale = _causal_ffill(
            hr, hr_obs, steps_grid(hcfg.hr_max_staleness_min))
    else:
        hr_f = np.zeros(len(grid)); hr_valid = np.zeros(len(grid), bool)
        hr_stale = np.full(len(grid), np.inf)
    out["heart_rate"] = hr_f
    out["hr_observed"] = hr_valid.astype(np.float32)
    out["hr_staleness"] = np.clip(
        np.where(np.isfinite(hr_stale), hr_stale * P, hcfg.staleness_cap_min),
        0, hcfg.staleness_cap_min)

    # ---- steps: SUM over (t-P, t] ------------------------------------------
    # Steps are an interval count, not a level. Averaging them would understate
    # activity and forward-filling them would invent movement that never
    # happened, so an unobserved step bin is zero with the mask cleared.
    if "steps" in d.columns:
        st, st_obs = _bin_backward(d["time"], d["steps"], grid, P, "sum")
    else:
        st = np.zeros(len(grid)); st_obs = np.zeros(len(grid), bool)
    out["steps"] = st
    out["steps_observed"] = st_obs.astype(np.float32)

    # ---- sleep: state, forward filled with a longer budget -----------------
    if "sleep" in d.columns:
        sl_raw = pd.to_numeric(d["sleep"], errors="coerce")
        sl, sl_obs = _bin_backward(d["time"], sl_raw, grid, P, "mean")
        sl_f, sl_valid, _ = _causal_ffill(
            sl, sl_obs, steps_grid(hcfg.sleep_max_staleness_min))
    else:
        sl_f = np.zeros(len(grid)); sl_valid = np.zeros(len(grid), bool)
    out["sleep"] = (sl_f > 0.5).astype(np.float64)
    out["sleep_observed"] = sl_valid.astype(np.float32)

    # ---- treatment impulses -> on-board curves ------------------------------
    # Doses are point events: they are summed into the bin that contains them
    # and never filled, because "no bolus recorded" means zero insulin, not
    # unknown insulin.
    for name, dur, peak, kind in (
        ("bolus", hcfg.iob_dia_min, hcfg.iob_peak_min, "iob"),
        ("carbs", hcfg.cob_dur_min, hcfg.cob_peak_min, "cob"),
    ):
        if name in d.columns:
            imp, _ = _bin_backward(d["time"], d[name].fillna(0.0), grid, P, "sum")
        else:
            imp = np.zeros(len(grid))
        out[name] = imp
        out[kind] = causal_on_board(imp, _activity_kernel(dur, peak, P, kind))

    # ---- segmentation on CGM gaps ------------------------------------------
    max_gap_steps = max(1, cfg.max_series_gap_min // P)
    miss = out["glucose_observed"].values == 0.0
    seg, run, sid = np.zeros(len(out), np.int64), 0, 0
    for i, m in enumerate(miss):
        if m:
            run += 1
        else:
            if run > max_gap_steps:
                sid += 1
            run = 0
        seg[i] = sid
    out["segment"] = seg

    assert_no_bfill(out)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
@dataclass
class MMWindowSet:
    """Parallel CGM and auxiliary history windows over a shared time axis.

    `x` and `x_aux` are aligned window-for-window and step-for-step: index
    (n, i) refers to the same instant in both. That is what lets the fusion
    model attend across modalities without any resampling at train time.
    """
    x: np.ndarray            # (N, H, 5)             CGM features
    x_aux: np.ndarray        # (N, H, len(AUX))      auxiliary features
    y_reg: np.ndarray        # (N,)
    y_cls_hard: np.ndarray   # (N,)
    y_min: np.ndarray        # (N,)
    patient: np.ndarray      # (N,)
    t_index: np.ndarray      # (N,)
    t_time: np.ndarray       # (N,)
    last_bg: np.ndarray      # (N,)
    aux_frac: np.ndarray     # (N,) fraction of aux steps genuinely observed

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, idx: np.ndarray) -> "MMWindowSet":
        return MMWindowSet(**{f.name: getattr(self, f.name)[idx]
                              for f in dataclasses.fields(self)})


def build_multimodal_windows(
    df: pd.DataFrame,
    cfg: DataConfig,
    hcfg: HupaConfig | None = None,
) -> MMWindowSet:
    """Emit parallel CGM/aux windows under the same causal contract as
    `data.build_windows`.

    For anchor t:
        x[n]     <- CGM features over [t-H+1, t]
        x_aux[n] <- aux features over the SAME closed interval
        targets  <- glucose over [t+1, t+PH]

    The aux slice uses identical indices to the CGM slice. There is no separate
    aux horizon and no aux value drawn from beyond t -- notably, IOB and COB at
    step i are the on-board amounts as of i, computed from doses at or before i,
    so a bolus taken during the prediction horizon is invisible to the model.
    That is the clinically correct framing: at inference time that bolus has not
    happened yet.
    """
    hcfg = hcfg or HupaConfig()
    H, P = cfg.history_steps, cfg.horizon_steps
    max_gap_steps = max(0, cfg.max_interpolation_gap_min // cfg.sample_period_min)

    xs, xs_obs, xa, yr, ym, pids, tix, tts, lbg, afrac = ([] for _ in range(10))

    for pid, pdf in df.groupby("patient_id", sort=True):
        reg = synchronize_patient(pdf, cfg, hcfg)
        if reg.empty:
            continue

        g = reg["glucose"].values.astype(np.float64)
        obs = reg["glucose_observed"].values.astype(bool)
        aux = np.stack([reg[c].values.astype(np.float64) for c in AUX_CHANNELS], axis=-1)
        seg = reg["segment"].values
        ts = reg["time"].values

        # Fraction of aux channels that are genuinely observed, per step.
        obs_cols = [AUX_INDEX[c] for c in ("hr_observed", "steps_observed", "sleep_observed")]
        aux_obs_step = aux[:, obs_cols].mean(axis=1)

        for _, sidx in pd.Series(np.arange(len(reg))).groupby(seg):
            si = sidx.values
            if si.size < H + P:
                continue
            gs, os_, tss = g[si], obs[si], ts[si]
            aus, aos = aux[si], aux_obs_step[si]

            for a in range(H - 1, si.size - P, cfg.stride_steps):
                hs = slice(a - H + 1, a + 1)
                tgt = slice(a + 1, a + 1 + P)

                if not os_[tgt].all() or not os_[a]:
                    continue
                fut = gs[tgt]

                hv, himp = _causal_gap_fill(gs[hs], os_[hs], max_gap_steps)
                if not (os_[hs] | himp).all():
                    continue
                if himp.mean() > cfg.max_imputed_frac:
                    continue

                frac = float(aos[hs].mean())
                if frac < hcfg.min_aux_observed_frac:
                    continue

                xs.append(hv); xs_obs.append(os_[hs].astype(np.float64))
                xa.append(aus[hs])
                yr.append(fut[-1])
                ym.append(fut.min() if cfg.classify_over_window else fut[-1])
                pids.append(pid)
                tix.append(a)
                tts.append(tss[a])
                lbg.append(hv[-1])
                afrac.append(frac)

    if not xs:
        raise ValueError(
            "No valid multimodal windows. Check cadence, gap settings, and "
            "HupaConfig.min_aux_observed_frac (aux coverage may be too sparse)."
        )

    y_min = np.asarray(ym, dtype=np.float32)
    return MMWindowSet(
        x=causal_features_batch(np.stack(xs), np.stack(xs_obs)),
        x_aux=np.stack(xa).astype(np.float32),
        y_reg=np.asarray(yr, dtype=np.float32),
        y_cls_hard=(y_min < cfg.hypo_threshold_mgdl).astype(np.float32),
        y_min=y_min,
        patient=np.asarray(pids, dtype=object),
        t_index=np.asarray(tix, dtype=np.int64),
        t_time=np.asarray(tts),
        last_bg=np.asarray(lbg, dtype=np.float32),
        aux_frac=np.asarray(afrac, dtype=np.float32),
    )
