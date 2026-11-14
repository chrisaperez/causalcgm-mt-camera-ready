"""OhioT1DM (2018 + 2020) ingestion, synchronisation, windowing and splitting.

WHAT MAKES THIS DATASET DIFFERENT FROM HUPA
-------------------------------------------
`hupa_loader` splits patients into folds: train on some people, test on others.
OhioT1DM ships a *fixed, official* split that is temporal and WITHIN patient --
each subject appears in both `train/` and `test/`, with the testing file
covering roughly the last 10 days of their record. The benchmark is "predict
this person's future given their past", not "generalise to a stranger".

That changes what leakage means here, so this module does not reuse
`patient_folds` and deliberately does not hand its window set to
`train.run_fold` (which splits by patient and would silently turn the official
protocol into something else). The Ohio-specific guarantees are:

  1. Windows are built PER FILE. A window's history and its targets always come
     from the same XML file, so no window can straddle the train/test boundary.
     This is structural, not a check that can be forgotten.
  2. The inner validation split is a chronological TAIL of each patient's
     training file, never a random sample. Overlapping windows share up to
     H-1 history steps, so a random split puts near-duplicates on both sides and
     inflates validation scores enough to break model selection.
  3. An EMBARGO of (history + horizon) steps is removed at that boundary. A
     window anchored just before the cut has targets that reach past it, and a
     window anchored just after reads history from before it; without the
     embargo those two windows overlap in raw samples.
  4. Every fitted object -- scaler, soft-label band, class prior -- is estimated
     on the fit portion only, in the same order as `train.run_fold`.

ALIGNMENT
---------
All the causal primitives are imported from `hupa_loader` rather than
reimplemented, so the "slot t summarises (t-P, t]" rule, the forward-fill
staleness budget and the on-board FIR convolutions have exactly one definition
in this codebase. Read that module's header for the reasoning; it applies here
unchanged.

OHIO-SPECIFIC ALIGNMENT DECISIONS
---------------------------------
    field           shape in XML              handling
    ------------------------------------------------------------------
    glucose_level   5-min point events        snap to grid (the target stream)
    finger_stick    point events              bin + bounded ffill (+ mask)
    basal           step function, U/hr       ffill with NO staleness budget
    temp_basal      interval override, U/hr   overrides basal inside interval
    bolus           point OR interval (dual)  dose spread over its interval
    meal            point events, grams       impulse -> COB
    exercise        ts + duration + intensity intensity held over its duration
    basis_heart_rate/steps/sleep              as in hupa_loader

Basal is the one channel with no staleness budget, and that is correct: a pump
basal profile stays in force until the patient changes it, so a six-hour-old
basal setting IS the current setting. That is the opposite of a six-hour-old
heart rate, which is why the two get different treatment.

WHAT THIS MODULE DOES NOT USE
-----------------------------
`hypo_event` is parsed and returned for analysis but never used to build labels.
Labels come from CGM only, because `hypo_event` is a self-reported/annotated
field with inconsistent coverage across subjects -- using it would make the
positive class mean different things for different patients.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import (  # noqa: F401  (causal_features is re-exported for callers)
    SCALED_FEATURE_CLIP, TAIL_TO_IQR, WINDOW_PIPELINE_VERSION, CausalScaler,
    NoiseEstimate, SoftLabeler, aux_presence,
    _causal_gap_fill, anchor_series_by_patient, causal_features,
    causal_features_batch,
    estimate_reference_mard, estimate_signal_noise, snap_to_grid_backward)
from .hupa_loader import (
    AUX_CHANNELS,
    HupaConfig,
    MMWindowSet,
    _bin_backward,
    _causal_ffill,
    _activity_kernel,
    causal_on_board,
)


# --------------------------------------------------------------------------- #
# Channel layout
# --------------------------------------------------------------------------- #
# The first 11 entries are byte-for-byte `hupa_loader.AUX_CHANNELS`, in the same
# order. That is load-bearing: `AUX_INDEX["hr_observed"] == 7` is passed to the
# model as `aux_valid_index`, and `AUX_SKIP_SCALING == (7, 8, 9)` is used by
# `train.run_fold`. Appending Ohio's extra channels at the END keeps every one
# of those indices valid, so a model or scaler written for HUPA does not
# silently misinterpret an Ohio tensor.
OHIO_EXTRA_CHANNELS: tuple[str, ...] = (
    "basal_rate",            # 11 U/hr in force at t (temp_basal applied)
    "basal_iob",             # 12 units, on-board from basal delivery alone
    "finger_stick",          # 13 mg/dL, last capillary BG (bounded ffill)
    "finger_stick_observed", # 14 mask
    "exercise_intensity",    # 15 self-reported 1-10, held over its duration
    "gsr",                   # 16 electrodermal activity (device units)
    "gsr_observed",          # 17 mask
    "skin_temp",             # 18 degrees F as released
    "skin_temp_observed",    # 19 mask
    "air_temp",              # 20 degrees F as released (2018 only)
    "air_temp_observed",     # 21 mask
    "accel_magnitude",       # 22 g, Empatica accelerometer (2020 only)
    "accel_observed",        # 23 mask
)
OHIO_AUX_CHANNELS: tuple[str, ...] = AUX_CHANNELS + OHIO_EXTRA_CHANNELS
OHIO_AUX_INDEX = {name: i for i, name in enumerate(OHIO_AUX_CHANNELS)}

# Binary indicators the scaler must pass through untouched. Rescaling a 0/1 flag
# by an IQR that is either 0 or 1 is a divide-by-epsilon blowup or a no-op, and
# either way the flag stops meaning "observed".
OHIO_AUX_SKIP_SCALING: tuple[int, ...] = tuple(
    i for i, c in enumerate(OHIO_AUX_CHANNELS) if c.endswith("_observed")
)


# --------------------------------------------------------------------------- #
# Modality groups -- the unit of the ablation
# --------------------------------------------------------------------------- #
# Each group is a set of channels that stands or falls together, because they
# come from one sensor. An ablation arm is a SET of groups; every channel not in
# the selected groups is zeroed, INCLUDING its mask, so the model sees an
# honestly absent modality rather than a silently zero-valued present one.
MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "treatment": ("iob", "cob", "bolus", "carbs", "basal_rate", "basal_iob"),
    "heart_rate": ("heart_rate", "hr_observed", "hr_staleness"),
    "steps": ("steps", "steps_observed"),
    "sleep": ("sleep", "sleep_observed"),
    "gsr": ("gsr", "gsr_observed"),
    "skin_temp": ("skin_temp", "skin_temp_observed"),
    "air_temp": ("air_temp", "air_temp_observed"),
    "accel": ("accel_magnitude", "accel_observed"),
    "finger_stick": ("finger_stick", "finger_stick_observed"),
    "exercise": ("exercise_intensity",),
}

# Which cohort actually carries each group. Established by auditing the released
# files (see `modality_availability`), not assumed: the 2018 subjects wore a
# Basis Peak and the 2020 subjects an Empatica Embrace, and the two devices
# expose DISJOINT sensor sets. Any arm whose groups are cohort-specific is
# evaluated on that cohort's 6 subjects only -- comparing it against an arm
# scored on the other 6 measures the patients, not the sensor.
MODALITY_COHORTS: dict[str, tuple[str, ...]] = {
    "heart_rate": ("2018",),        # Basis Peak only; absent from every 2020 file
    "steps": ("2018",),             # Basis Peak only
    "air_temp": ("2018",),          # Basis Peak only
    "accel": ("2020",),             # Empatica only
    "gsr": ("2018", "2020"),        # both, but in DIFFERENT UNITS -- see below
    "skin_temp": ("2018", "2020"),
    "sleep": ("2018", "2020"),
    "treatment": ("2018", "2020"),
    "finger_stick": ("2018", "2020"),
    "exercise": ("2018", "2020"),
}

# Channels whose raw scale is a property of the DEVICE, not the patient.
# Released 2018 GSR sits around 1e-4 and 2020 GSR around 1.4e-2 -- two orders of
# magnitude apart for the same physiological quantity. A single cohort-wide
# median/IQR is dominated by the larger scale and crushes the other cohort's
# signal to numerical zero, so these are normalised PER PATIENT from their own
# training file before the global scaler ever sees them.
DEVICE_SCALED_CHANNELS: tuple[str, ...] = (
    "gsr", "skin_temp", "air_temp", "accel_magnitude", "heart_rate",
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
# OhioT1DM timestamps are day-first with a shifted (anonymised) year: the 2018
# release carries 2021 dates, the 2020 release 2027 dates. Only intervals matter
# downstream, so the absolute year is irrelevant -- but day-first parsing is
# NOT, because "07-12-2021" is a valid date under both conventions and a
# month-first read silently reorders the record.
_TS_FORMATS = ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M")


def _parse_timestamps(raw: Sequence[str]) -> pd.Series:
    """Day-first timestamp parsing with an explicit failure count.

    Tries the two documented layouts, then falls back to pandas with
    ``dayfirst=True``. Unparseable values become NaT and are dropped by the
    caller rather than silently coerced to epoch 0, which would place them
    decades before the record and create a spurious segment.
    """
    s = pd.Series(list(raw), dtype=object)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna()
    for fmt in _TS_FORMATS:
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        out[todo] = parsed
        todo = out.isna() & s.notna()
    if todo.any():
        out[todo] = pd.to_datetime(s[todo], errors="coerce", dayfirst=True)
    return out


def _to_float(raw: Sequence[str]) -> np.ndarray:
    """Numeric coercion that tolerates empty attributes and stray whitespace.

    Several Ohio fields carry ``value=""`` or ``type=" "`` for events the
    subject did not annotate. Those become NaN, which the aggregation layer
    treats as unobserved -- as opposed to 0.0, which would read as a genuine
    measurement of zero.
    """
    return pd.to_numeric(pd.Series(list(raw), dtype=object).astype(str).str.strip(),
                         errors="coerce").values.astype(np.float64)


@dataclass
class PatientRecord:
    """One XML file, parsed into tidy per-field frames.

    Kept as separate frames rather than one merged table because the fields have
    genuinely different shapes -- instants, step functions and intervals -- and
    merging them before alignment is what forces a single (wrong) alignment rule
    onto all of them.
    """
    patient_id: str          # e.g. "2018_559"; unique across both releases
    raw_id: str              # "559"
    subset: str              # "2018" | "2020"
    split: str               # "train" | "test"
    path: str
    attrib: dict             # patient-level XML attributes (weight, insulin_type)
    fields: dict             # field name -> DataFrame

    def get(self, name: str) -> pd.DataFrame:
        """Missing fields return an EMPTY frame with the right columns.

        2020 has no `exercise` or `sleep` for several subjects, and 2018 has no
        `acceleration`. Returning an empty frame rather than None means the
        synchronisation code has exactly one path instead of a presence check at
        every use site.
        """
        return self.fields.get(name, pd.DataFrame(columns=["ts", "value"]))


# Attribute names that carry a timestamp, and the canonical name each maps to.
#
# `basis_sleep` uses `tbegin`/`tend` while `sleep`, `bolus` and `temp_basal` use
# `ts_begin`/`ts_end` -- the same concept under two spellings, in the same file.
# Normalising here is not cosmetic: a parser that only knows `ts_begin` leaves
# `basis_sleep` with no `ts` column, every downstream interval lookup misses,
# and the device-derived sleep channel silently reads as "never asleep" for the
# entire cohort. That is a missing modality wearing the costume of a real one.
_TS_ATTR_MAP = {
    "ts": "ts", "ts_begin": "ts", "tbegin": "ts",
    "ts_end": "ts_end", "tend": "ts_end",
}


def parse_patient_xml(path: str) -> PatientRecord:
    """Parse one OhioT1DM XML file into per-field frames.

    Robust by construction: every field present in the file is parsed
    generically from its event attributes, so a field this module does not know
    about (``acceleration``, ``stressors``, ``illness``) is still available to a
    caller without a code change. Timestamps are normalised to ``ts`` /
    ``ts_end``; every other attribute is kept under its own name, numeric where
    it parses as numeric and string otherwise.
    """
    root = ET.parse(path).getroot()

    base = os.path.basename(path)
    raw_id = base.split("-")[0]
    subset = "2020" if os.sep + "2020" + os.sep in path else "2018"
    split = "test" if "testing" in base else "train"

    fields: dict[str, pd.DataFrame] = {}
    for node in root:
        events = list(node)
        if not events:
            continue
        keys = sorted({k for e in events for k in e.attrib})
        cols: dict[str, object] = {}
        for k in keys:
            vals = [e.attrib.get(k) for e in events]
            if k in _TS_ATTR_MAP:
                cols[_TS_ATTR_MAP[k]] = _parse_timestamps(vals)
            else:
                num = _to_float(vals)
                # A column is numeric only if something in it actually parsed;
                # `type=" "` and `competitive=""` stay as strings.
                cols[k] = num if np.isfinite(num).any() else pd.Series(vals, dtype=object)
        df = pd.DataFrame(cols)

        if "ts" in df.columns:
            bad = df["ts"].isna().sum()
            if bad:
                warnings.warn(f"{base}:{node.tag}: dropped {bad} events with unparseable ts")
            df = df[df["ts"].notna()]

        # Some 2018 `sleep` events have ts_begin AFTER ts_end -- the two are
        # transposed in the source file. Swapping is the only interpretation
        # that yields a non-negative duration; leaving it would produce an empty
        # interval and silently erase every sleep episode for those subjects.
        if "ts_end" in df.columns and "ts" in df.columns:
            flip = df["ts_end"].notna() & (df["ts_end"] < df["ts"])
            if flip.any():
                b = df.loc[flip, "ts"].values.copy()
                df.loc[flip, "ts"] = df.loc[flip, "ts_end"].values
                df.loc[flip, "ts_end"] = b

        fields[node.tag] = df.sort_values("ts").reset_index(drop=True) if "ts" in df else df

    return PatientRecord(
        patient_id=f"{subset}_{raw_id}",
        raw_id=raw_id,
        subset=subset,
        split=split,
        path=path,
        attrib=dict(root.attrib),
        fields=fields,
    )


def discover_records(root_dir: str, subsets: Sequence[str] = ("2018", "2020")) -> list[PatientRecord]:
    """Parse every XML under ``root_dir/<subset>/{train,test}``.

    Raises if a patient is missing one half of the official split, because a
    patient present only in `train/` would contribute training windows that
    never get evaluated -- a silent change to the benchmark rather than an
    error.
    """
    records: list[PatientRecord] = []
    for sub in subsets:
        for split_dir in ("train", "test"):
            pattern = os.path.join(root_dir, sub, split_dir, "*.xml")
            for p in sorted(glob.glob(pattern)):
                records.append(parse_patient_xml(p))
    if not records:
        # The acquisition pointer belongs HERE rather than in each of the five
        # drivers that reach this line, because this is the one place that knows
        # the layout it wanted. `scripts/_cli.missing_input` turns it into a
        # message and exit code 2 instead of a traceback.
        raise FileNotFoundError(
            f"No OhioT1DM XML found under {root_dir}. Expected "
            f"{root_dir}/<2018|2020>/<train|test>/*-ws-*.xml -- 24 files, "
            f"two per subject, for the full 12-subject release."
        )

    by_pid: dict[str, set[str]] = {}
    for r in records:
        by_pid.setdefault(r.patient_id, set()).add(r.split)
    incomplete = {k: sorted(v) for k, v in by_pid.items() if v != {"train", "test"}}
    if incomplete:
        raise ValueError(f"Patients missing half of the official split: {incomplete}")

    n_pat = len(by_pid)
    print(f"Parsed {len(records)} OhioT1DM files, {n_pat} patients "
          f"({', '.join(sorted(subsets))})")
    return records


# --------------------------------------------------------------------------- #
# Ohio-specific alignment parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OhioConfig:
    """Ohio-specific knobs. Shared PK/staleness parameters live in `HupaConfig`."""

    # A capillary reading stays informative longer than a heart rate, but not
    # indefinitely -- past this it is dropped and the mask cleared.
    finger_stick_max_staleness_min: int = 60

    # Basal deliberately has NO staleness budget: a pump profile is in force
    # until changed. This flag exists so the choice is visible and testable
    # rather than implied by the absence of a parameter.
    basal_unbounded_ffill: bool = True

    # Cap on a single temp_basal interval. A malformed ts_end (or a temp that
    # was cancelled without a closing event) can otherwise pin the basal rate
    # for days.
    max_temp_basal_min: int = 24 * 60

    # Cap on a single square/dual bolus delivery interval, same reasoning.
    max_bolus_interval_min: int = 8 * 60

    # Exercise intensity is held for the reported duration, then released.
    max_exercise_min: int = 6 * 60

    # `bolus` carries `bwz_carb_input` (grams entered into the bolus wizard) in
    # the 2018 release. It largely duplicates `meal`, so folding it in
    # double-counts carbohydrate for most events. Off by default; when enabled,
    # a bwz entry within `bwz_dedup_min` of a logged meal is discarded.
    carbs_from_bwz: bool = False
    bwz_dedup_min: int = 30

    # A sleep log covering less than this fraction of the record is treated as
    # ABSENT rather than as evidence of wakefulness. Several 2020 subjects have
    # 9 recorded episodes across two months; marking the other 97% of the record
    # "awake" would be a fabrication, and one the model would happily fit.
    min_sleep_coverage: float = 0.10

    # Windows whose auxiliary history is almost entirely mask are dropped.
    # Lower than the HUPA default: Ohio's wearable coverage is patchier, and at
    # the HUPA default several subjects lose most of their record.
    min_aux_observed_frac: float = 0.0


# --------------------------------------------------------------------------- #
# Interval -> grid helpers
# --------------------------------------------------------------------------- #
def _interval_level(
    grid: pd.DatetimeIndex,
    begins: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    max_len_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise ``[begin, end)`` intervals onto the grid as a level.

    Causality. A slot t is assigned an interval's value only when
    ``begin <= t < end``, i.e. the interval had already started at t. The end
    time is programmed at the start (a temp basal or square bolus is entered
    with its duration), so knowing at t that the interval is running uses no
    future information. Slots BEFORE ``begin`` are never touched, which is the
    property that would break if this were implemented as a backward fill.

    Later intervals win on overlap, matching pump behaviour where a new temp
    basal cancels the running one.
    """
    lvl = np.zeros(len(grid), dtype=np.float64)
    act = np.zeros(len(grid), dtype=bool)
    if len(grid) == 0:
        return lvl, act
    gv = grid.values.astype("datetime64[ns]")

    for b, e, v in zip(begins, ends, values):
        if not np.isfinite(v) or pd.isna(b):
            continue
        b = np.datetime64(pd.Timestamp(b), "ns")
        if pd.isna(e):
            e = b + np.timedelta64(max_len_min, "m")
        else:
            e = np.datetime64(pd.Timestamp(e), "ns")
        e = min(e, b + np.timedelta64(max_len_min, "m"))
        if e <= b:
            continue
        i0, i1 = np.searchsorted(gv, b, "left"), np.searchsorted(gv, e, "left")
        if i1 > i0:
            lvl[i0:i1] = v
            act[i0:i1] = True
    return lvl, act


def _interval_impulse(
    grid: pd.DatetimeIndex,
    begins: np.ndarray,
    ends: np.ndarray,
    amounts: np.ndarray,
    period_min: int,
    max_len_min: int,
) -> np.ndarray:
    """Spread each dose uniformly across the slots its delivery covers.

    Ohio boluses carry `ts_begin`/`ts_end`. For a `normal` bolus the two are
    equal and this reduces to dropping the whole dose into one slot. For
    `square`/`dual` boluses the insulin is genuinely delivered over minutes to
    hours, and crediting all of it at `ts_begin` would place insulin on board
    before the pump had delivered it -- an optimistic look-ahead that makes IOB
    lead reality by up to the extended duration.
    """
    imp = np.zeros(len(grid), dtype=np.float64)
    if len(grid) == 0:
        return imp
    gv = grid.values.astype("datetime64[ns]")
    step = np.timedelta64(period_min, "m")

    for b, e, a in zip(begins, ends, amounts):
        if not np.isfinite(a) or a == 0.0 or pd.isna(b):
            continue
        b = np.datetime64(pd.Timestamp(b), "ns")
        e = b if pd.isna(e) else np.datetime64(pd.Timestamp(e), "ns")
        e = min(max(e, b), b + np.timedelta64(max_len_min, "m"))

        # Slot k covers (k - P, k], so a dose at time b lands in the first slot
        # at or after b.
        i0 = int(np.searchsorted(gv, b, "left"))
        i1 = int(np.searchsorted(gv, e + step, "left"))
        i0, i1 = min(i0, len(grid)), min(max(i1, i0 + 1), len(grid))
        if i0 >= len(grid):
            continue  # dose after the record ends
        imp[i0:i1] += a / (i1 - i0)
    return imp


def _duration_level(
    grid: pd.DatetimeIndex,
    ts: np.ndarray,
    duration_min: np.ndarray,
    values: np.ndarray,
    max_len_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold a value for a reported duration starting at its timestamp."""
    ends = np.array(
        [pd.Timestamp(t) + pd.Timedelta(minutes=float(np.clip(d, 0, max_len_min)))
         if np.isfinite(d) else pd.NaT
         for t, d in zip(ts, duration_min)],
        dtype=object,
    )
    return _interval_level(grid, ts, ends, values, max_len_min)


# --------------------------------------------------------------------------- #
# Synchronisation
# --------------------------------------------------------------------------- #
def synchronize_ohio_record(
    rec: PatientRecord,
    cfg: DataConfig,
    hcfg: HupaConfig | None = None,
    ocfg: OhioConfig | None = None,
) -> pd.DataFrame:
    """Put one file's record on the shared 5-minute CGM grid.

    The grid is defined by the CGM stream, because CGM is where the targets come
    from -- re-gridding it would change the prediction problem rather than just
    its representation.

    Every column produced here satisfies: the value at t is a function of
    samples at times <= t.
    """
    hcfg = hcfg or HupaConfig()
    ocfg = ocfg or OhioConfig()
    P = cfg.sample_period_min
    steps = lambda m: max(0, int(round(m / P)))  # noqa: E731

    cgm = rec.get("glucose_level")
    cgm = cgm[np.isfinite(_to_float(cgm["value"]))] if len(cgm) else cgm
    if len(cgm) < 2:
        return pd.DataFrame()

    grid = pd.date_range(cgm["ts"].iloc[0].ceil(f"{P}min"),
                         cgm["ts"].iloc[-1].floor(f"{P}min"),
                         freq=f"{P}min")
    if len(grid) < 2:
        return pd.DataFrame()

    out = pd.DataFrame({"time": grid})

    # ---- CGM: strictly backward snap into (t - P, t] ------------------------
    # An instantaneous measurement, so the bin takes the LAST sample rather than
    # a mean, which would blur the excursions we are trying to predict. Backward
    # rather than nearest: Ohio's CGM clock drifts off the 5-minute grid, and a
    # nearest snap with a half-period tolerance was giving 40% of slots a value
    # recorded up to 120 s in the future.
    _cgm_df = pd.DataFrame({"time": cgm["ts"].values,
                            "glucose": _to_float(cgm["value"])}).dropna()
    binned, _ = snap_to_grid_backward(
        _cgm_df["time"], _cgm_df[["glucose"]], grid, P)
    out["glucose"] = binned["glucose"].to_numpy()
    out["glucose_observed"] = out["glucose"].notna().astype(np.float32)

    # ---- wearable streams (identical rules to hupa_loader) ------------------
    hr = rec.get("basis_heart_rate")
    if len(hr):
        v, o = _bin_backward(hr["ts"], pd.Series(_to_float(hr["value"])), grid, P, "mean")
        hr_f, hr_ok, hr_stale = _causal_ffill(v, o, steps(hcfg.hr_max_staleness_min))
    else:
        hr_f = np.zeros(len(grid)); hr_ok = np.zeros(len(grid), bool)
        hr_stale = np.full(len(grid), np.inf)
    out["heart_rate"] = hr_f
    out["hr_observed"] = hr_ok.astype(np.float32)
    out["hr_staleness"] = np.clip(
        np.where(np.isfinite(hr_stale), hr_stale * P, hcfg.staleness_cap_min),
        0, hcfg.staleness_cap_min)

    # Steps are an interval COUNT, not a level: summed over the bin, never
    # forward-filled. Filling them would invent movement that did not happen.
    st = rec.get("basis_steps")
    if len(st):
        s_v, s_o = _bin_backward(st["ts"], pd.Series(_to_float(st["value"])), grid, P, "sum")
    else:
        s_v = np.zeros(len(grid)); s_o = np.zeros(len(grid), bool)
    out["steps"] = s_v
    out["steps_observed"] = s_o.astype(np.float32)

    # Sleep: `basis_sleep` (device, `tbegin`/`tend`) and `sleep` (self-reported,
    # `ts_begin`/`ts_end`) are both interval-shaped; either marks the state.
    #
    # THE MASK IS PER STEP, AND THAT IS THE WHOLE POINT
    # --------------------------------------------------
    # It used to be a per-FILE constant: `sl.mean() >= min_sleep_coverage`
    # broadcast to every slot, so a subject with 10% recorded sleep got
    # `sleep_observed = 1` at all 12,080 of their slots and a subject with 3%
    # got 0 at all 13,108 of theirs. Two things were wrong with that.
    #
    # It was computed from the whole file, so the value at slot t depended on
    # data after t -- harmless in itself, since a within-file constant cannot
    # leak the target, but a direct contradiction of the invariant this
    # function's docstring states.
    #
    # The consequence was not harmless. `sleep_observed` is a member of
    # `aux_validity_indices()`, and the fusion gate opens when ANY configured
    # mask is set, so a constant 1 jammed the cross-attention gate open at every
    # timestep for those subjects regardless of whether a single sensor had
    # reported there. That is the `hr_observed` defect this module documents
    # fixing, inverted: instead of a gate stuck shut for half the cohort, a gate
    # stuck open for the other half.
    #
    # A step is now sleep-observed when it lies inside a recorded episode, or
    # within a staleness budget of one -- the same rule every other episodic
    # stream gets. A subject with no sleep record at all still gets 0
    # everywhere, so the model can tell "awake" from "we do not know".
    sl = np.zeros(len(grid)); sl_active = np.zeros(len(grid), bool)
    for tag in ("basis_sleep", "sleep"):
        d = rec.get(tag)
        if len(d) and "ts_end" in d.columns and "ts" in d.columns:
            v, a = _interval_level(grid, d["ts"].values, d["ts_end"].values,
                                   np.ones(len(d)), 24 * 60)
            sl = np.maximum(sl, v)
            sl_active |= a
    out["sleep"] = sl

    # Coverage gate stays: a log covering less than `min_sleep_coverage` of the
    # record is treated as ABSENT rather than as evidence of wakefulness, since
    # marking 97% of a record "awake" off nine episodes is a fabrication the
    # model would happily fit. It now gates the whole CHANNEL rather than
    # masquerading as a per-step observation.
    covered = bool(sl_active.any()) and sl.mean() >= ocfg.min_sleep_coverage
    if covered:
        _, sl_ok, _ = _causal_ffill(
            sl_active.astype(np.float64), sl_active,
            steps(hcfg.sleep_max_staleness_min))
        out["sleep_observed"] = sl_ok.astype(np.float32)
    else:
        out["sleep_observed"] = np.zeros(len(grid), dtype=np.float32)

    # ---- bolus insulin -> IOB ----------------------------------------------
    bo = rec.get("bolus")
    if len(bo):
        bolus_imp = _interval_impulse(
            grid, bo["ts"].values,
            bo["ts_end"].values if "ts_end" in bo.columns else np.full(len(bo), pd.NaT),
            _to_float(bo["dose"]) if "dose" in bo.columns else np.zeros(len(bo)),
            P, ocfg.max_bolus_interval_min,
        )
    else:
        bolus_imp = np.zeros(len(grid))
    iob_kernel = _activity_kernel(hcfg.iob_dia_min, hcfg.iob_peak_min, P, "iob")
    out["bolus"] = bolus_imp
    out["iob"] = causal_on_board(bolus_imp, iob_kernel)

    # ---- carbohydrate -> COB ------------------------------------------------
    ml = rec.get("meal")
    carb_ts = list(ml["ts"].values) if len(ml) else []
    carb_g = list(_to_float(ml["carbs"])) if len(ml) and "carbs" in ml.columns else []

    if ocfg.carbs_from_bwz and len(bo) and "bwz_carb_input" in bo.columns:
        bwz = _to_float(bo["bwz_carb_input"])
        meal_ts = pd.DatetimeIndex(carb_ts) if carb_ts else pd.DatetimeIndex([])
        tol = pd.Timedelta(minutes=ocfg.bwz_dedup_min)
        for t, g in zip(bo["ts"].values, bwz):
            if not np.isfinite(g) or g <= 0:
                continue
            # Drop wizard entries that duplicate a logged meal; otherwise the
            # same carbohydrate is counted twice into COB.
            if len(meal_ts) and (np.abs(meal_ts - pd.Timestamp(t)) <= tol).any():
                continue
            carb_ts.append(t); carb_g.append(g)

    if carb_ts:
        carb_imp, _ = _bin_backward(pd.Series(carb_ts), pd.Series(carb_g), grid, P, "sum")
    else:
        carb_imp = np.zeros(len(grid))
    out["carbs"] = carb_imp
    out["cob"] = causal_on_board(
        carb_imp, _activity_kernel(hcfg.cob_dur_min, hcfg.cob_peak_min, P, "cob"))

    # ---- basal insulin ------------------------------------------------------
    # A basal profile stays in force until the pump is reprogrammed, so this is
    # the one stream forward-filled with no staleness budget. Then temp_basal
    # overrides it inside its interval (value 0.0 = suspend, which is a real and
    # clinically important state, not a missing value).
    ba = rec.get("basal")
    if len(ba):
        b_v, b_o = _bin_backward(ba["ts"], pd.Series(_to_float(ba["value"])), grid, P, "mean")
        rate, _, _ = _causal_ffill(b_v, b_o, len(grid) if ocfg.basal_unbounded_ffill
                                   else steps(hcfg.hr_max_staleness_min))
    else:
        rate = np.zeros(len(grid))

    tb = rec.get("temp_basal")
    if len(tb) and "ts_end" in tb.columns:
        t_lvl, t_act = _interval_level(grid, tb["ts"].values, tb["ts_end"].values,
                                       _to_float(tb["value"]), ocfg.max_temp_basal_min)
        rate = np.where(t_act, t_lvl, rate)
    out["basal_rate"] = rate
    # U/hr held over a P-minute slot -> units delivered in that slot.
    out["basal_iob"] = causal_on_board(rate * (P / 60.0), iob_kernel)

    # ---- finger sticks ------------------------------------------------------
    fs = rec.get("finger_stick")
    if len(fs):
        f_v, f_o = _bin_backward(fs["ts"], pd.Series(_to_float(fs["value"])), grid, P, "mean")
        f_f, f_ok, _ = _causal_ffill(f_v, f_o, steps(ocfg.finger_stick_max_staleness_min))
    else:
        f_f = np.zeros(len(grid)); f_ok = np.zeros(len(grid), bool)
    out["finger_stick"] = f_f
    out["finger_stick_observed"] = f_ok.astype(np.float32)

    # ---- analog wearable streams -------------------------------------------
    # Mean over (t-P, t], then a bounded forward fill, exactly as for heart
    # rate: these are levels, not counts. Each carries its own mask, so a
    # cohort that never wore the sensor is explicitly absent rather than
    # implicitly zero -- otherwise the model learns that "GSR == 0" is a
    # property of the 2018 patients rather than a property of the device.
    for name, tag, budget in (
        ("gsr", "basis_gsr", hcfg.hr_max_staleness_min),
        ("skin_temp", "basis_skin_temperature", hcfg.hr_max_staleness_min),
        ("air_temp", "basis_air_temperature", hcfg.hr_max_staleness_min),
        ("accel_magnitude", "acceleration", hcfg.hr_max_staleness_min),
    ):
        d = rec.get(tag)
        if len(d) and "value" in d.columns:
            v, o = _bin_backward(d["ts"], pd.Series(_to_float(d["value"])), grid, P, "mean")
            f, ok, _ = _causal_ffill(v, o, steps(budget))
        else:
            f = np.zeros(len(grid)); ok = np.zeros(len(grid), bool)
        out[name] = f
        mask_name = "accel_observed" if name == "accel_magnitude" else f"{name}_observed"
        out[mask_name] = ok.astype(np.float32)

    # ---- exercise -----------------------------------------------------------
    ex = rec.get("exercise")
    if len(ex) and "duration" in ex.columns:
        e_lvl, _ = _duration_level(grid, ex["ts"].values, _to_float(ex["duration"]),
                                   _to_float(ex["intensity"]) if "intensity" in ex.columns
                                   else np.ones(len(ex)),
                                   ocfg.max_exercise_min)
    else:
        e_lvl = np.zeros(len(grid))
    out["exercise_intensity"] = e_lvl

    # ---- segmentation on CGM gaps ------------------------------------------
    # Windows never cross a segment boundary, so a multi-hour sensor dropout
    # cannot be interpolated into a plausible-looking history.
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

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
@dataclass
class OhioWindowSet(MMWindowSet):
    """`MMWindowSet` plus the official split tag and the multi-horizon targets.

    Subclassing keeps `x`, `x_aux`, `y_reg`, `y_cls_hard`, `y_min`, `patient`,
    `last_bg` and `aux_frac` at the same names the rest of the codebase already
    reads, so evaluation and model code work unchanged. `subset()` is inherited
    and rebuilds via `dataclasses.fields`, so it picks up the new arrays
    automatically.

    `y_reg` / `y_cls_hard` / `y_min` mirror the PRIMARY horizon (`cfg.horizon_min`)
    for drop-in compatibility; the `*_multi` arrays carry every horizon in
    `horizons_min`, column k corresponding to `horizons_min[k]`.
    """
    split: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    subset_tag: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    y_reg_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    y_min_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    y_cls_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    horizons_min: tuple[int, ...] = ()

    def subset(self, idx: np.ndarray) -> "OhioWindowSet":
        """Index every aligned array; `horizons_min` is metadata, not an axis."""
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v if f.name == "horizons_min" else v[idx]
        return OhioWindowSet(**kw)

    def where(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def build_ohio_windows(
    records: Iterable[PatientRecord],
    cfg: DataConfig,
    hcfg: HupaConfig | None = None,
    ocfg: OhioConfig | None = None,
    horizons_min: Sequence[int] = (30, 60),
) -> OhioWindowSet:
    """Emit causal windows for every parsed file.

    For an anchor t:
        x[n]     <- CGM features over [t-H+1, t]
        x_aux[n] <- auxiliary features over the SAME closed interval
        targets  <- glucose over [t+1, t+PH_max]

    A window is emitted only if:
      * history AND every horizon's targets lie inside one segment,
      * the anchor step and every target step were genuinely observed
        (targets are never imputed -- imputing a label is leakage by
        construction),
      * after causal short-gap interpolation the imputed fraction of the history
        is within `cfg.max_imputed_frac`.

    Windows are built per FILE, so the official train/test boundary cannot be
    crossed by any single window.
    """
    hcfg = hcfg or HupaConfig()
    ocfg = ocfg or OhioConfig()
    horizons = tuple(int(h) for h in sorted(set(horizons_min)))
    if cfg.horizon_min not in horizons:
        horizons = tuple(sorted(set(horizons) | {int(cfg.horizon_min)}))

    hz_steps = [h // cfg.sample_period_min for h in horizons]
    for h, s in zip(horizons, hz_steps):
        if s * cfg.sample_period_min != h:
            raise ValueError(f"Horizon {h} min is not a multiple of the "
                             f"{cfg.sample_period_min}-min sampling period.")
    primary = horizons.index(int(cfg.horizon_min))

    H, Pmax = cfg.history_steps, max(hz_steps)
    max_gap_steps = max(0, cfg.max_interpolation_gap_min // cfg.sample_period_min)
    obs_cols = [OHIO_AUX_INDEX[c] for c in
                ("hr_observed", "steps_observed", "sleep_observed", "finger_stick_observed")]

    xs, xs_obs, xa, yr, ym = [], [], [], [], []
    pids, tix, tts, lbg, afrac, splits, subs = [], [], [], [], [], [], []

    for rec in records:
        reg = synchronize_ohio_record(rec, cfg, hcfg, ocfg)
        if reg.empty:
            warnings.warn(f"{rec.path}: no usable CGM grid, skipped")
            continue

        g = reg["glucose"].values.astype(np.float64)
        obs = reg["glucose_observed"].values.astype(bool)
        aux = np.stack([reg[c].values.astype(np.float64) for c in OHIO_AUX_CHANNELS], axis=-1)
        aux_obs_step = aux[:, obs_cols].mean(axis=1)
        seg = reg["segment"].values
        ts = reg["time"].values

        for _, sidx in pd.Series(np.arange(len(reg))).groupby(seg):
            si = sidx.values
            if si.size < H + Pmax:
                continue
            gs, os_, tss = g[si], obs[si], ts[si]
            aus, aos = aux[si], aux_obs_step[si]

            for a in range(H - 1, si.size - Pmax, cfg.stride_steps):
                hs = slice(a - H + 1, a + 1)

                # Targets: no imputation, at any horizon.
                if not os_[a] or not os_[a + 1 : a + 1 + Pmax].all():
                    continue
                fut = gs[a + 1 : a + 1 + Pmax]

                # History: causal short-gap fill, confined to this window so
                # every interpolation endpoint is itself at or before t.
                hv, himp = _causal_gap_fill(gs[hs], os_[hs], max_gap_steps)
                if not (os_[hs] | himp).all():
                    continue
                if himp.mean() > cfg.max_imputed_frac:
                    continue

                frac = float(aos[hs].mean())
                if frac < ocfg.min_aux_observed_frac:
                    continue

                # Regression target: BG at exactly t+PH.
                # Classification target: the MINIMUM over (t, t+PH]. An early
                # warning that only fires when the endpoint happens to be low
                # would miss a nadir at t+20 that recovers by t+30 -- which is
                # the event the patient actually needs warning about.
                yr.append([fut[s - 1] for s in hz_steps])
                ym.append([fut[:s].min() if cfg.classify_over_window else fut[s - 1]
                           for s in hz_steps])

                xs.append(hv); xs_obs.append(os_[hs].astype(np.float64))
                xa.append(aus[hs])
                pids.append(rec.patient_id)
                tix.append(a)
                tts.append(tss[a])
                lbg.append(hv[-1])
                afrac.append(frac)
                splits.append(rec.split)
                subs.append(rec.subset)

    if not xs:
        raise ValueError(
            "No valid OhioT1DM windows. Check history/horizon settings against "
            "the 5-minute cadence, and DataConfig.max_series_gap_min."
        )

    y_reg_multi = np.asarray(yr, dtype=np.float32)
    y_min_multi = np.asarray(ym, dtype=np.float32)
    y_cls_multi = (y_min_multi < cfg.hypo_threshold_mgdl).astype(np.float32)

    return OhioWindowSet(
        x=causal_features_batch(np.stack(xs), np.stack(xs_obs)),
        x_aux=np.stack(xa).astype(np.float32),
        y_reg=y_reg_multi[:, primary].copy(),
        y_cls_hard=y_cls_multi[:, primary].copy(),
        y_min=y_min_multi[:, primary].copy(),
        patient=np.asarray(pids, dtype=object),
        t_index=np.asarray(tix, dtype=np.int64),
        t_time=np.asarray(tts),
        last_bg=np.asarray(lbg, dtype=np.float32),
        aux_frac=np.asarray(afrac, dtype=np.float32),
        split=np.asarray(splits, dtype=object),
        subset_tag=np.asarray(subs, dtype=object),
        y_reg_multi=y_reg_multi,
        y_min_multi=y_min_multi,
        y_cls_multi=y_cls_multi,
        horizons_min=horizons,
    )


# --------------------------------------------------------------------------- #
# Splitting -- the anti-leakage core
# --------------------------------------------------------------------------- #
@dataclass
class OhioSplit:
    """Row indices into an `OhioWindowSet`. Disjoint by construction.

    FOUR regions, not three. `calib` exists because model selection and
    conformal calibration are different statistical jobs that cannot share
    rows:

      fit    fits the weights
      val    selects the epoch (early stopping) and nothing else
      calib  derives the conformal qhat and nothing else
      test   the official test file, scored once

    Split conformal's finite-sample guarantee assumes the predictor is fixed
    INDEPENDENTLY of the calibration set. When `val` served both jobs, the
    checkpoint being calibrated had been chosen *because* it scored best on the
    very rows whose residuals set `qhat` -- so the residuals were optimistically
    biased, `qhat` came out too small, and the intervals undercovered. That is
    the failure direction `conformal.py`'s header calls the one that matters
    clinically, and nothing in the artefact showed it: `calibrated_on`,
    `n_calib` and `n_effective` all looked healthy.

    `calib` is defaulted so an `OhioSplit` built by older code still constructs;
    `_conformal_block` refuses an empty one rather than falling back to `val`,
    because a silent fall back is how the two jobs came to share rows.
    """
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray
    embargoed: np.ndarray          # dropped at every internal boundary
    val_cut: dict                  # patient -> boundary timestamp
    calib: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    calib_cut: dict = field(default_factory=dict)   # patient -> val/calib boundary

    def summary(self, ws: OhioWindowSet) -> str:
        def line(name, idx):
            if idx.size == 0:
                return f"  {name:>10}: empty"
            return (f"  {name:>10}: {idx.size:7d} windows | "
                    f"{len(set(ws.patient[idx])):2d} patients | "
                    f"hypo {ws.y_cls_hard[idx].mean():6.2%}")
        return "\n".join([line("fit", self.fit), line("val", self.val),
                          line("calib", self.calib), line("test", self.test),
                          f"  {'embargoed':>10}: {self.embargoed.size:7d} windows dropped"])


def official_temporal_split(
    ws: OhioWindowSet,
    cfg: DataConfig,
    val_frac: float = 0.2,
    embargo_steps: int | None = None,
    calib_frac: float = 0.5,
) -> OhioSplit:
    """The OhioT1DM protocol: official train/test files, chronological inner val.

    Four separate leak paths are closed here, and all four are easy to open by
    accident:

    1. TEST is exactly the official `*-ws-testing.xml` windows. It is never
       resampled, rebalanced or mixed with training windows, so results stay
       comparable to the published benchmark.

    2. The HELD-OUT TAIL is the chronological end of each patient's TRAINING
       file -- the last `val_frac` of their windows by anchor time, computed per
       patient so a subject with a shorter record still contributes. A random
       split would be catastrophic here rather than merely optimistic: with
       stride 1 and a 24-step history, adjacent windows share 23 of 24 history
       steps, so a random split places near-identical rows on both sides and
       validation stops measuring generalisation at all.

    3. THAT TAIL IS SPLIT AGAIN, into VAL (model selection) and CALIB (conformal
       calibration), because they are different statistical jobs. Split
       conformal is only valid when the predictor is fixed independently of the
       calibration set, and an early-stopped checkpoint is chosen BECAUSE it
       scored best on the selection rows -- so calibrating on those same rows
       biases every residual downward, shrinks `qhat`, and undercovers. `calib`
       is the LATER of the two: it sits adjacent to the test file, which is
       where exchangeability with the scored rows is strongest, and it leaves
       model selection on the rows further from the boundary where a
       drift-sensitive monitor is least distorted.

       `calib_frac` is the share of the held-out tail given to calibration
       (0.5 by default: half of `val_frac` each). `fit` is UNCHANGED by this
       split -- the boundary that determines it is still `cut_time - embargo` --
       so a model trained under the four-way split sees exactly the rows it saw
       under the three-way one.

    4. EMBARGO, at every internal boundary. A clean chronological cut still
       leaves overlap: the window anchored at the last fit step has targets
       reaching `horizon` steps past the cut, and the window anchored at the
       first val step reads `history` steps back across it. Dropping
       `history + max_horizon` steps' worth of windows on the EARLIER side of
       each boundary (fit|val, val|calib, calib|test) removes every window that
       shares a raw sample across it. The default is derived from the geometry
       rather than guessed.

       The calib|test embargo is new with `calib` and is load-bearing for the
       same reason as the rest: the Ohio testing file is the immediately
       following stretch of one continuous record, so without it the last
       calibration windows share raw samples with the first scored ones -- the
       exact rows a conformal guarantee is weakest on.

    The embargo is always taken from the earlier side: shrinking the training
    set is a cost, whereas shrinking validation would make model selection
    noisier on the one signal used to stop training, and the test file is fixed
    by the published protocol and must not be resampled at all.
    """
    if embargo_steps is None:
        max_hz = max(h // cfg.sample_period_min for h in ws.horizons_min) if ws.horizons_min \
            else cfg.horizon_steps
        embargo_steps = cfg.history_steps + max_hz

    if not 0.0 <= calib_frac < 1.0:
        raise ValueError(
            f"calib_frac must be in [0, 1); got {calib_frac}. It is the share "
            f"of the held-out tail given to conformal calibration, and 1.0 "
            f"would leave model selection with no rows at all.")

    test = ws.where("test")
    train = ws.where("train")

    fit_parts, val_parts, cal_parts, emb_parts = [], [], [], []
    cuts: dict = {}
    calib_cuts: dict = {}
    period = np.timedelta64(cfg.sample_period_min, "m")
    embargo_td = period * int(embargo_steps)

    # First scored anchor per patient, so the calibration region can be
    # embargoed off the test file the same way fit is embargoed off val. Only
    # patients who appear in BOTH files get one: under a held-out-subject
    # protocol the test patients are different people and share no timeline.
    test_start: dict = {}
    if test.size:
        t_pat = np.asarray([str(p) for p in ws.patient[test]])
        for pid in sorted(set(t_pat.tolist())):
            test_start[pid] = ws.t_time[test[t_pat == pid]].min()

    for pid in sorted({str(p) for p in ws.patient[train]}):
        idx = train[ws.patient[train] == pid]
        if idx.size == 0:
            continue
        order = idx[np.argsort(ws.t_time[idx], kind="stable")]
        t_sorted = ws.t_time[order]

        n_val = int(round(order.size * val_frac))
        if n_val < 1 or n_val >= order.size:
            # Too little data to hold out a meaningful tail: keep it all for
            # fitting and let the caller notice the empty val set, rather than
            # producing a one-window validation split that early stopping would
            # then trust.
            fit_parts.append(order)
            continue

        cut_time = t_sorted[order.size - n_val]
        cuts[pid] = cut_time

        held = ws.t_time[order] >= cut_time
        # Embargo: fit windows whose targets could reach across the cut. The
        # comparison is >=, not >, because a fit window anchored exactly at
        # `cut - embargo` has its final target step land on the first history
        # step of the earliest val window -- one shared raw sample, which
        # `assert_no_temporal_overlap` treats (correctly) as overlap.
        is_emb = (~held) & (ws.t_time[order] >= cut_time - embargo_td)
        fit_parts.append(order[(~held) & (~is_emb)])
        emb_parts.append(order[is_emb])

        # ---- carve the held-out tail into selection | embargo | calibration --
        tail = order[held]
        n_cal = int(round(tail.size * calib_frac))
        n_sel = tail.size - n_cal
        if n_cal < 1 or n_sel < 1:
            # The tail cannot pay for both jobs. Model selection is the one that
            # decides which weights get shipped, so it keeps the rows; the arm
            # then has no calibration data for this patient, which
            # `_conformal_block` reports rather than papering over.
            val_parts.append(tail)
            continue

        calib_cut = ws.t_time[tail][n_sel]
        calib_cuts[pid] = calib_cut

        is_cal = ws.t_time[tail] >= calib_cut
        # Same inclusive comparison, same reason, on the val side this time.
        is_cal_emb = (~is_cal) & (ws.t_time[tail] >= calib_cut - embargo_td)
        val_parts.append(tail[(~is_cal) & (~is_cal_emb)])
        emb_parts.append(tail[is_cal_emb])

        # ---- and embargo the calibration region off the test file -----------
        cal = tail[is_cal]
        ts = test_start.get(pid)
        if ts is not None and cal.size:
            reaches_test = ws.t_time[cal] >= ts - embargo_td
            emb_parts.append(cal[reaches_test])
            cal = cal[~reaches_test]
        cal_parts.append(cal)

    cat = lambda parts: (np.concatenate(parts) if parts else np.empty(0, np.int64)).astype(np.int64)  # noqa: E731
    split = OhioSplit(fit=np.sort(cat(fit_parts)), val=np.sort(cat(val_parts)),
                      test=np.sort(test.astype(np.int64)),
                      embargoed=np.sort(cat(emb_parts)), val_cut=cuts,
                      calib=np.sort(cat(cal_parts)), calib_cut=calib_cuts)

    assert_disjoint(split)
    assert_no_temporal_overlap(ws, split, cfg)
    return split


def assert_disjoint(split: OhioSplit) -> None:
    """Row-level disjointness. Cheap, and catches an index bug immediately."""
    for a, b in (("fit", "val"), ("fit", "calib"), ("fit", "test"),
                 ("val", "calib"), ("val", "test"), ("calib", "test")):
        inter = np.intersect1d(getattr(split, a), getattr(split, b))
        if inter.size:
            raise AssertionError(f"{a}/{b} splits share {inter.size} windows")


# The chronological order of the four regions. Every ADJACENT pair must be
# separated by the embargo, and so must every non-adjacent one (transitively) --
# but checking the adjacent pairs is not enough on its own, because a patient
# can be absent from the middle region entirely (too short a tail to carve a
# calibration slice), which would leave `fit` and `calib` adjacent with nothing
# between them. So every ordered pair is checked.
_SPLIT_ORDER: tuple[str, ...] = ("fit", "val", "calib", "test")


def assert_no_temporal_overlap(ws: OhioWindowSet, split: OhioSplit, cfg: DataConfig) -> None:
    """No window in an earlier region may share a RAW SAMPLE with a later one.

    Disjoint row indices are not enough: two different windows overlap in raw
    data whenever their [history, target] time spans intersect. This is the
    check that actually verifies the embargo did its job, so it runs on every
    split rather than only in the tests.

    All six ordered pairs over (fit, val, calib, test) are checked, not just
    fit/val. The pair that matters most for the paper's coverage claim is
    calib/test: `calib` derives the conformal `qhat` and `test` is what the
    interval is scored on, and in OhioT1DM the testing file is the immediately
    following stretch of one continuous record -- so without an embargo the last
    calibration windows and the first scored windows genuinely share raw
    samples. Checking only fit/val, as this did, could not see that.
    """
    P = np.timedelta64(cfg.sample_period_min, "m")
    hist = P * int(cfg.history_steps)
    max_hz = max(h // cfg.sample_period_min for h in ws.horizons_min) if ws.horizons_min \
        else cfg.horizon_steps
    fwd = P * int(max_hz)

    regions = {name: getattr(split, name) for name in _SPLIT_ORDER}
    for i, earlier in enumerate(_SPLIT_ORDER):
        for later in _SPLIT_ORDER[i + 1:]:
            a_idx, b_idx = regions[earlier], regions[later]
            if a_idx.size == 0 or b_idx.size == 0:
                continue
            shared = (set(str(p) for p in ws.patient[a_idx])
                      & set(str(p) for p in ws.patient[b_idx]))
            for pid in sorted(shared):
                a = a_idx[ws.patient[a_idx] == pid]
                b = b_idx[ws.patient[b_idx] == pid]
                if a.size == 0 or b.size == 0:
                    continue
                # Spans are [t - history, t + horizon]; overlap iff the latest
                # earlier-span end reaches the earliest later-span start.
                a_end = ws.t_time[a].max() + fwd
                b_start = ws.t_time[b].min() - hist
                if a_end >= b_start:
                    raise AssertionError(
                        f"Patient {pid}: {earlier} window spans reach {a_end} "
                        f"but {later} spans start at {b_start} -- embargo too "
                        f"short for the configured history/horizon."
                    )


def held_out_patient_split(
    ws: OhioWindowSet, held_out: Sequence[str], cfg: DataConfig, val_frac: float = 0.2,
) -> OhioSplit:
    """Alternative protocol: generalisation to UNSEEN subjects.

    The official split answers "can we forecast this person's future given their
    past". This one answers "can we forecast a stranger", by moving whole
    patients (both their files) into test. Both are legitimate; they measure
    different things, and reporting one while describing the other is the most
    common way Ohio results become incomparable.

    No embargo is needed between train and test here -- different people share
    no raw samples -- but the inner fit/val split is still chronological, for
    the same window-overlap reason as above.
    """
    ho = set(str(p) for p in held_out)
    unknown = ho - {str(p) for p in ws.patient}
    if unknown:
        raise ValueError(f"Unknown patient ids: {sorted(unknown)}")

    is_test = np.array([str(p) in ho for p in ws.patient])
    inner = ws.subset(np.flatnonzero(~is_test))
    # Re-tag every remaining window as "train" so the chronological splitter
    # treats the held-out patients' own test files as ordinary training data.
    inner.split = np.full(len(inner), "train", dtype=object)
    inner_split = official_temporal_split(inner, cfg, val_frac=val_frac)

    remap = np.flatnonzero(~is_test)
    return OhioSplit(
        fit=remap[inner_split.fit],
        val=remap[inner_split.val],
        test=np.flatnonzero(is_test),
        embargoed=remap[inner_split.embargoed],
        val_cut=inner_split.val_cut,
        calib=remap[inner_split.calib],
        calib_cut=inner_split.calib_cut,
    )


# --------------------------------------------------------------------------- #
# Per-subject adaptation prefix (few-shot personalisation protocol)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OhioPrefixSplit:
    """One subject's adaptation prefix, embargo and evaluation region.

    `prefix_mode` selects WHERE in the training file the prefix is taken from,
    and it is the only thing that differs between the two arms:

      "opening"   |<- adapt (first K h of TRAIN) ->|<- embargo ->|<- evaluate ->|

      "recent"    |<- unused TRAIN ->|<- adapt (last K h) ->|<-emb->|<- TEST ->|

    Row indices into the `OhioWindowSet`, each sorted by anchor time. `adapt`,
    `embargoed` and `eval` are pairwise disjoint by construction.

    WHY THE EVALUATION REGION KEEPS THE OFFICIAL TEST FILE WHOLE
    ------------------------------------------------------------
    Ohio results are only comparable to the literature when scored on the
    released `*-ws-testing.xml` windows. Personalisation must therefore take
    its prefix from the TRAINING file and leave the test file untouched --
    every window of it, unresampled. The tail of the training file is added to
    the evaluation region rather than discarded because it is legitimately
    available (it is strictly after the embargo) and it roughly doubles the
    number of hypoglycemic events the paired comparison rests on.

    WHAT `n_adapt_in_fit` / `n_adapt_in_val` ARE FOR
    ------------------------------------------------
    The opening prefix lands in `split.fit` (rows the population model took
    gradient steps on). The recent prefix does not: in the released sweep no
    recent-prefix window lies in `split.fit` or in `split.val` (the rows it was
    early-stopped on); `n_adapt_in_calib` counts any that reach the conformal
    calibration slice. The two prefixes therefore differ in more than temporal
    distance to the evaluation window -- they also differ in whether the
    population model saw the windows in training -- so the contrast changes
    recency and training exposure together and does not separate them. The
    counts are recorded so a reader does not have to infer this.
    """
    patient: str
    adapt: np.ndarray
    eval: np.ndarray
    embargoed: np.ndarray
    prefix_end: np.datetime64
    eval_start: np.datetime64
    prefix_hours: float
    embargo_min: int
    n_eval_from_train: int
    n_eval_from_test: int
    eval_scope: str = "test"
    n_eval_in_fit: int = 0
    n_eval_in_val: int = 0
    # Rows the CONFORMAL CALIBRATION region also holds. Contamination of the
    # same kind as `n_eval_in_fit`, and reported for the same reason: an
    # evaluation row that calibrated the interval it is scored under is not
    # held out, and a reader of the artefact should not have to re-derive that.
    n_eval_in_calib: int = 0
    prefix_mode: str = "opening"
    prefix_start: np.datetime64 | None = None
    n_adapt_in_fit: int = 0
    n_adapt_in_val: int = 0
    n_adapt_in_calib: int = 0
    # Anchor of the earliest row actually SCORED. Not derivable from
    # `eval_start`: under "opening" with the default scope, `eval_start` is
    # where the embargo ends, while the first scored window is the start of the
    # test file -- weeks later. The gap that matters is to the latter.
    eval_first_anchor: np.datetime64 | None = None

    def __str__(self) -> str:
        contaminated = (self.n_eval_in_fit + self.n_eval_in_val
                        + self.n_eval_in_calib)
        warn = f"  !! {contaminated} in-sample eval rows" if contaminated else ""
        return (
            f"  {self.patient}: adapt {self.adapt.size:5d} | embargo "
            f"{self.embargoed.size:4d} | eval {self.eval.size:5d} "
            f"({self.n_eval_from_train} train-tail + {self.n_eval_from_test} test, "
            f"scope={self.eval_scope}) | "
            f"prefix {self.prefix_hours:.1f} h ({self.prefix_mode}), "
            f"embargo {self.embargo_min} min{warn}"
        )

    @property
    def prefix_gap_min(self) -> float:
        """Minutes from the end of the adaptation prefix to the first SCORED window.

        This is the quantity the opening/recent contrast is about, and it is the
        one number that separates the two arms: under "recent" it is the embargo
        itself, under "opening" it is however many weeks separate the start of
        the training file from the test file.
        """
        if self.adapt.size == 0 or self.eval_first_anchor is None:
            return float("nan")
        delta = (np.datetime64(self.eval_first_anchor, "m")
                 - np.datetime64(self.prefix_end, "m"))
        return float(delta / np.timedelta64(1, "m"))


def required_embargo_min(ws: OhioWindowSet, cfg: DataConfig) -> int:
    """A conservative embargo that makes window overlap impossible, in minutes.

    history + max horizon + ONE sampling period, which is ONE PERIOD MORE than
    the tight bound. The geometry, in steps, under the window convention of
    `build_windows` (history `[a - H + 1, a]`, targets `[a + 1, a + P]`):

        anchors a1 < a2 share a raw sample
            iff  a2 - H + 1 <= a1 + P
            iff  a2 - a1    <= H + P - 1.

    So the touching case -- the last adaptation target landing exactly on the
    first evaluation history step -- is at a separation of `H + P - 1`, and
    `H + P` is already the smallest separation with no shared sample. This
    function returns `H + P + 1` periods regardless: one step of margin, kept
    deliberately because `assert_prefix_embargo_causal` and
    `official_temporal_split` both compare inclusively (`>=`), and a bound that
    is exactly tight leaves those comparisons no room to be off by one in the
    unsafe direction. Section 2 (invariant I5) states the tight bound `H + P`
    and notes that the implementation embargoes one step beyond it.
    """
    max_hz = (max(h // cfg.sample_period_min for h in ws.horizons_min)
              if ws.horizons_min else cfg.horizon_steps)
    return int(cfg.history_min + (max_hz + 1) * cfg.sample_period_min)


def ohio_prefix_embargo_split(
    ws: OhioWindowSet,
    split: OhioSplit,
    patient: str,
    cfg: DataConfig,
    prefix_hours: float = 24.0,
    embargo_min: int | None = None,
    eval_scope: str = "test",
    prefix_mode: str = "opening",
) -> OhioPrefixSplit | None:
    """Carve one subject's record into adapt / embargo / evaluate.

    The prefix is a strict PREFIX IN WALL-CLOCK TIME of the subject's official
    training file -- never a random sample of it. A random sample would let the
    model adapt on data recorded after the windows it is then scored on, which
    is the single most common way a "few-shot personalisation" result stops
    meaning anything.

    WHERE THE PREFIX IS TAKEN FROM -- `prefix_mode`
    -----------------------------------------------
    Both modes are strict wall-clock windows ending at least `embargo` before
    the first scored window; they differ only in how far before it they sit.

      "opening"  (default) the FIRST `prefix_hours` of the training file. This
                 is the deployment story of "the device has just been fitted and
                 has one day of history", and on OhioT1DM it puts the prefix
                 several weeks upstream of the test file.

      "recent"   the LAST `prefix_hours` of the training file that still clear
                 the embargo -- i.e. the prefix ENDS exactly `embargo` before
                 the earliest scored window. This is the deployment story of "the
                 device adapts on the day it just recorded", and it is the
                 control for the drift explanation: if the opening prefix fails
                 because a subject's dynamics move over weeks, a prefix taken
                 from the immediate temporal vicinity of the evaluation window
                 should not fail the same way. Requires `eval_scope="test"`,
                 because the legacy scope puts training-tail rows INSIDE the
                 evaluation region and a recent prefix would then be drawn from
                 after the windows it is scored on -- the exact failure the
                 wall-clock prefix exists to prevent.

    The two modes do not hold the population model's exposure fixed: the opening
    prefix lands in `split.fit`, whereas in the released sweep no recent-prefix
    window lands in `split.fit` or `split.val`. The contrast therefore changes
    temporal distance and training exposure together and does not isolate
    either. The counts go on the result as `n_adapt_in_fit` / `n_adapt_in_val` /
    `n_adapt_in_calib`.

    The training rows are taken as `fit + val + calib + embargoed` from the
    supplied `OhioSplit`, i.e. everything that came out of the training file
    including the windows the internal embargoes dropped. Those windows were
    excluded to protect *model selection* and *conformal calibration*, which is
    not what is happening here; excluding them again would silently shorten the
    prefix. `calib` in particular is the LAST stretch of the training file, so
    omitting it would truncate `prefix_mode="recent"` -- the arm whose whole
    definition is "the most recent hours that still clear the embargo".

    The embargo defaults to `required_embargo_min` and is raised to it if a
    smaller value is passed, so the causal separation cannot be configured
    away.

    WHICH ROWS ARE SCORED -- `eval_scope`
    -------------------------------------
    The prefix/embargo geometry separates the ADAPTATION data from the
    evaluation data. It says nothing about the POPULATION model, and that is a
    second boundary that has to be drawn separately.

    The evaluation region used to be "everything after the embargo", i.e. the
    tail of the training file plus the whole test file. But the base model was
    fitted on `split.fit` and selected on `split.val`, and those are precisely
    the training-file rows. Measured across all twelve subjects at a 24 h
    prefix, that made the evaluation region 63.8% base-model training rows,
    16.6% model-selection rows and only 19.4% genuinely held out -- so the
    zero-shot "before" anchoring every paired delta was an in-sample number, and
    any explanation of why adaptation hurts was confounded with "adaptation
    moves the model off a solution it has memorised for 80% of the rows".

      "test"          (default) score only the official `*-ws-testing.xml`
                      windows. Disjoint from `fit` and `val` by construction, so
                      the paired delta is measured entirely out of sample. This
                      is also the set every other row of the benchmark table is
                      scored on, so the numbers are comparable.

      "post_embargo"  the legacy behaviour: training tail + test file. Roughly
                      doubles the hypoglycemic events, and is only defensible
                      when the population model did NOT train on this subject's
                      training file -- a held-out-subject protocol, not the
                      official one. Kept so the published numbers can be
                      reproduced; `n_eval_in_fit` records the damage.

    `n_eval_in_fit` / `n_eval_in_val` are recorded on the result either way, so
    the contamination is a number in the artefact rather than something a reader
    has to re-derive.

    Returns None when the subject has no training windows at all (which is what
    a held-out-subject split produces), because there is then no prefix to
    adapt on and a zero-length one would silently become a no-op arm.
    """
    pid = str(patient)
    is_pid = np.asarray([str(p) == pid for p in ws.patient])
    if not is_pid.any():
        # An id that is simply absent is a caller mistake, not an empty split.
        # Returning None for both would let a typo in a subject list look like
        # "this subject had too little data" and silently shrink the cohort.
        raise ValueError(
            f"Unknown patient id {pid!r}; window set carries "
            f"{sorted({str(p) for p in ws.patient})}")

    train_pool = np.union1d(
        np.union1d(np.union1d(split.fit, split.val),
                   np.asarray(getattr(split, "calib", np.empty(0, np.int64)),
                              dtype=np.int64)),
        split.embargoed)
    train_rows = train_pool[is_pid[train_pool]]
    test_rows = split.test[is_pid[split.test]]
    if train_rows.size == 0:
        return None

    train_rows = train_rows[np.argsort(ws.t_time[train_rows], kind="stable")]
    test_rows = test_rows[np.argsort(ws.t_time[test_rows], kind="stable")]

    if prefix_mode not in ("opening", "recent"):
        raise ValueError(
            f"unknown prefix_mode {prefix_mode!r}; expected 'opening' or 'recent'")
    if prefix_mode == "recent" and eval_scope != "test":
        # Under "post_embargo" the training tail is part of the evaluation
        # region, so a prefix drawn from the end of the training file would be
        # recorded AFTER windows it is then scored on. That is not a weaker
        # protocol, it is an invalid one, so it is refused rather than warned
        # about.
        raise ValueError(
            "prefix_mode='recent' requires eval_scope='test'; under "
            f"{eval_scope!r} the training tail is inside the evaluation region "
            "and a recent prefix would be drawn from after the scored windows.")

    t = ws.t_time[train_rows].astype("datetime64[m]")
    need = required_embargo_min(ws, cfg)
    emb = max(int(embargo_min), need) if embargo_min is not None else need
    span = np.timedelta64(int(round(prefix_hours * 60)), "m")

    if prefix_mode == "opening":
        prefix_start = t[0]
        prefix_end = prefix_start + span
        eval_start = prefix_end + np.timedelta64(int(emb), "m")
        adapt = train_rows[t <= prefix_end]
        tail = train_rows[t >= eval_start]
        embargoed = train_rows[(t > prefix_end) & (t < eval_start)]
    else:
        # Anchor the prefix to the EVALUATION region and work backwards, so the
        # gap to the first scored window is exactly the embargo by construction
        # rather than by arithmetic that could drift from `required_embargo_min`.
        if test_rows.size == 0:
            return None
        first_eval = ws.t_time[test_rows[0]].astype("datetime64[m]")
        eval_start = first_eval
        prefix_end = first_eval - np.timedelta64(int(emb), "m")
        prefix_start = prefix_end - span
        adapt = train_rows[(t >= prefix_start) & (t <= prefix_end)]
        tail = np.empty(0, np.int64)
        # Everything from the end of the prefix to the test file. These rows are
        # dropped, which is what the embargo IS -- unlike the opening mode there
        # is no post-embargo training tail to recover, because the prefix has
        # been pushed up against the evaluation region.
        embargoed = train_rows[t > prefix_end]

    if eval_scope == "test":
        ev = test_rows
        n_from_train = 0
    elif eval_scope == "post_embargo":
        ev = np.concatenate([tail, test_rows]) if test_rows.size else tail
        n_from_train = int(tail.size)
    else:
        raise ValueError(
            f"unknown eval_scope {eval_scope!r}; expected 'test' or 'post_embargo'")

    ev = np.asarray(ev, dtype=np.int64)
    adapt = adapt.astype(np.int64)
    return OhioPrefixSplit(
        patient=pid,
        adapt=adapt,
        eval=ev,
        embargoed=embargoed.astype(np.int64),
        prefix_end=prefix_end,
        eval_start=eval_start,
        prefix_hours=float(prefix_hours),
        embargo_min=int(emb),
        n_eval_from_train=n_from_train,
        n_eval_from_test=int(test_rows.size),
        eval_scope=str(eval_scope),
        # Contamination of the evaluation region by the POPULATION model's own
        # fitting and selection rows. Zero under the default scope; recorded so
        # a reader of the artefact can see it rather than infer it.
        n_eval_in_fit=int(np.intersect1d(ev, split.fit).size),
        n_eval_in_val=int(np.intersect1d(ev, split.val).size),
        n_eval_in_calib=int(np.intersect1d(ev, split.calib).size),
        prefix_mode=str(prefix_mode),
        prefix_start=prefix_start,
        # Which split the PREFIX came from. Not contamination -- the prefix is
        # meant to be the subject's own past -- but the two modes draw from
        # different parts of the population model's training data, and that
        # difference belongs in the artefact.
        n_adapt_in_fit=int(np.intersect1d(adapt, split.fit).size),
        n_adapt_in_val=int(np.intersect1d(adapt, split.val).size),
        n_adapt_in_calib=int(np.intersect1d(adapt, split.calib).size),
        eval_first_anchor=(ws.t_time[ev].min().astype("datetime64[m]")
                           if ev.size else None),
    )


def assert_prefix_embargo_causal(
    ws: OhioWindowSet, ps: OhioPrefixSplit, cfg: DataConfig
) -> None:
    """No adaptation window may share a RAW SAMPLE with an evaluation window.

    Row disjointness is not enough, for the same reason it is not enough in
    `assert_no_temporal_overlap`: two windows overlap in raw data whenever their
    [t - history, t + horizon] spans intersect, and at stride 1 adjacent windows
    share 23 of 24 history steps. This runs on every split rather than only in
    the tests, because a personalisation result whose prefix leaked into its
    evaluation region is not a weaker result -- it is a meaningless one.
    """
    for a, b in (("adapt", "eval"), ("adapt", "embargoed"), ("embargoed", "eval")):
        inter = np.intersect1d(getattr(ps, a), getattr(ps, b))
        if inter.size:
            raise AssertionError(
                f"{ps.patient}: {a}/{b} share {inter.size} windows")

    if ps.adapt.size == 0 or ps.eval.size == 0:
        return

    P = np.timedelta64(cfg.sample_period_min, "m")
    max_hz = (max(h // cfg.sample_period_min for h in ws.horizons_min)
              if ws.horizons_min else cfg.horizon_steps)
    adapt_end = ws.t_time[ps.adapt].max().astype("datetime64[m]") + P * int(max_hz)
    eval_start = (ws.t_time[ps.eval].min().astype("datetime64[m]")
                  - P * int(cfg.history_steps))
    if adapt_end >= eval_start:
        raise AssertionError(
            f"Patient {ps.patient}: adaptation window spans reach {adapt_end} but "
            f"evaluation spans start at {eval_start} -- embargo of "
            f"{ps.embargo_min} min is too short for history "
            f"{cfg.history_min} min + horizon {max_hz * cfg.sample_period_min} min."
        )



# --------------------------------------------------------------------------- #
# Paired reference BG -- the input that makes MARD a measurement
# --------------------------------------------------------------------------- #
@dataclass
class ReferencePairs:
    """Capillary reference readings matched to the CGM value of the same instant.

    OhioT1DM ships `finger_stick` events alongside the CGM stream, which means
    the cohort carries the paired reference BG that MARD is *defined* against.
    The pipeline was estimating a high-pass proxy for device noise while that
    reference sat unused in the same XML file -- and the proxy is exactly the
    quantity the manufacturer's smoothing filter destroys, which is why it read
    0.7% on a sensor whose specification is 9-14%.

    Collected ONCE from the raw records and then filtered per fold, because the
    anti-leakage contract applies here like everywhere else: a pair recorded
    after the fit/val cut is evaluation data.

    Arrays are aligned along axis 0. `ts` is the CGM sample's timestamp, which
    is what the fold cut is compared against.
    """
    ts: np.ndarray            # (N,) datetime64
    cgm: np.ndarray           # (N,) float64, mg/dL
    reference: np.ndarray     # (N,) float64, mg/dL
    patient: np.ndarray       # (N,) object
    split: np.ndarray         # (N,) object -- "train" | "test"

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def select(self, mask: np.ndarray) -> "ReferencePairs":
        return ReferencePairs(**{f.name: getattr(self, f.name)[mask]
                                 for f in dataclasses.fields(self)})

    def before(self, cut_by_patient: dict) -> "ReferencePairs":
        """Pairs at or before each patient's cut. Patients with no cut are dropped.

        This is the causal filter. `cut_by_patient` maps a patient id to the
        last timestamp that patient contributes to the fit rows; anything later
        belongs to validation or test and must not reach a fitted statistic.
        """
        keep = np.zeros(len(self), dtype=bool)
        for pid, cut in cut_by_patient.items():
            m = (self.patient == pid) & (self.ts <= np.datetime64(cut))
            keep |= m
        return self.select(keep)


def collect_reference_pairs(
    records: Iterable[PatientRecord],
    cfg: DataConfig | None = None,
    max_offset_min: float = 3.0,
    splits: Sequence[str] = ("train",),
) -> ReferencePairs:
    """Match every finger stick to the CGM sample nearest in time.

    `max_offset_min` bounds the match. CGM moves at up to ~3 mg/dL/min, so a
    5-minute mismatch can contribute 15 mg/dL of pure timing error to an
    "accuracy" statistic; 3 minutes keeps the pairing inside one CGM period
    while still matching the great majority of sticks (2,600 of 3,900 here).

    `direction="nearest"` is deliberate and is NOT a causality violation: this
    is a measurement-error statistic about two sensors reading the same instant,
    not a feature. What keeps it causal is the FOLD filter -- see
    `ReferencePairs.before` -- which is applied where the estimate is consumed.

    Defaults to training files only, so a test-file reference reading cannot
    reach a fitted object even if a caller forgets to filter.
    """
    keep_splits = {str(s) for s in splits}
    tol = pd.Timedelta(minutes=float(max_offset_min))
    ts, cg, rf, pid, sp = [], [], [], [], []

    for rec in records:
        if keep_splits and rec.split not in keep_splits:
            continue
        cgm = rec.get("glucose_level")
        fs = rec.get("finger_stick")
        if not len(cgm) or not len(fs):
            continue
        c = pd.DataFrame({"ts": pd.to_datetime(cgm["ts"].values),
                          "cgm": _to_float(cgm["value"])}).dropna().sort_values("ts")
        f = pd.DataFrame({"ts": pd.to_datetime(fs["ts"].values),
                          "reference": _to_float(fs["value"])}).dropna().sort_values("ts")
        if c.empty or f.empty:
            continue
        m = pd.merge_asof(f, c, on="ts", direction="nearest", tolerance=tol).dropna()
        if m.empty:
            continue
        ts.append(m["ts"].to_numpy())
        cg.append(m["cgm"].to_numpy(dtype=np.float64))
        rf.append(m["reference"].to_numpy(dtype=np.float64))
        pid.append(np.full(len(m), rec.patient_id, dtype=object))
        sp.append(np.full(len(m), rec.split, dtype=object))

    if not ts:
        empty_t = np.empty(0, dtype="datetime64[ns]")
        return ReferencePairs(ts=empty_t, cgm=np.empty(0), reference=np.empty(0),
                              patient=np.empty(0, dtype=object),
                              split=np.empty(0, dtype=object))
    return ReferencePairs(ts=np.concatenate(ts), cgm=np.concatenate(cg),
                          reference=np.concatenate(rf), patient=np.concatenate(pid),
                          split=np.concatenate(sp))


def fit_row_cutoffs(ws: "OhioWindowSet", fit_idx: np.ndarray) -> dict:
    """Last fit-row anchor timestamp per patient -- the causal cut for a fold."""
    out: dict = {}
    if fit_idx.size == 0:
        return out
    pats = ws.patient[fit_idx]
    times = ws.t_time[fit_idx]
    for pid in {str(p) for p in pats}:
        m = pats == pid
        if m.any():
            out[pid] = times[m].max()
    return out

# --------------------------------------------------------------------------- #
# Fitted preprocessing -- train-only, in a fixed order
# --------------------------------------------------------------------------- #
# (p99 - p1) / IQR for a normal distribution. The two spreads therefore COINCIDE
# on well-behaved data -- measured at 1.3519 vs 1.3536 on 200k standard normal
# samples -- so the wider term is inert until the interquartile range collapses
# relative to the channel's actual dynamic range.
#
# Imported rather than redefined: `CausalScaler.fit_array(tail_aware=True)`
# applies the identical policy one stage later, and two copies of this constant
# would drift.
_TAIL_TO_IQR: float = TAIL_TO_IQR

# The per-patient normalised value is clipped here. It is a BACKSTOP, not the
# mechanism: with the tail-aware spread above it binds on 0.020% of the real
# cohort's device-scaled values, against 2.609% for clipping a bare-IQR scale to
# the same bound. Saturating a handful of electrode-artefact spikes is the
# intended trade -- the alternative is letting them set every BatchNorm
# statistic in the auxiliary branch.
_PER_PATIENT_CLIP: float = 20.0


@dataclass
class PerPatientNormalizer:
    """Robust per-patient, per-channel centring for device-scaled streams.

    Why per patient rather than per cohort. Released GSR medians range from
    6e-5 (2018/588) to 7e-2 (2020/540) -- four orders of magnitude, and the
    spread is as wide WITHIN the 2018 cohort as between cohorts, because it
    reflects electrode contact and skin properties as much as the device. A
    single global median/IQR maps most patients' entire dynamic range into a few
    ULPs of the scaled tensor, so the fusion branch sees a constant.

    Why this does not leak. Statistics come only from rows in the FIT split, so
    a patient's validation and test periods never contribute. This is the same
    contract as the global scaler, applied at finer granularity -- legitimate
    here precisely because Ohio's protocol is within-patient, so a patient's own
    past is available at inference time by construction.

    A patient with no observed samples for a channel falls back to the identity,
    leaving the (zero) values untouched and the mask to carry the information.
    """
    center: dict          # patient -> (C,) float32
    scale: dict           # patient -> (C,) float32
    channels: tuple[int, ...]

    @staticmethod
    def fit(x_aux: np.ndarray, patient: np.ndarray, idx: np.ndarray,
            channel_names: Sequence[str] = DEVICE_SCALED_CHANNELS) -> "PerPatientNormalizer":
        chans = tuple(OHIO_AUX_INDEX[c] for c in channel_names if c in OHIO_AUX_INDEX)
        center, scale = {}, {}
        for pid in sorted({str(p) for p in patient[idx]}):
            rows = idx[patient[idx] == pid]
            v = x_aux[rows][..., chans].reshape(-1, len(chans))
            med = np.median(v, axis=0)
            q75, q25 = np.percentile(v, [75, 25], axis=0)
            p99, p1 = np.percentile(v, [99, 1], axis=0)
            iqr = q75 - q25

            # A TAIL-AWARE SPREAD, because the IQR alone is fitted to the wrong
            # part of these distributions.
            #
            # `flat = iqr < 1e-12` caught a channel that is EXACTLY constant for
            # a subject. The failure that actually occurs is subtler: OhioT1DM
            # subject 2018_588's released GSR trace is 97% sensor zero (p25
            # 5.9e-05, p50 6.3e-05, p75 7.4e-05), so its interquartile range of
            # 1.5e-05 is measuring the quantisation noise of a disconnected
            # electrode, while the 1.4% of real skin contact reaches 23.6. The
            # IQR is not degenerate in any relative sense -- iqr/median = 0.24
            # -- it is simply the spread of the noise floor. Dividing by it put
            # (max - median)/IQR = 1.57e6 into the fusion branch, on four of
            # twelve subjects, and the aux tensor's global maximum was 1.767e6.
            #
            # Taking the wider of the IQR and the 1-99 percentile span (rescaled
            # to be IQR-equivalent under normality) reads the channel's actual
            # dynamic range when the middle 50% has collapsed, and is inert
            # otherwise. Global max |z| over the real cohort: 1.573e6 -> 152,
            # and -> 20 once `transform` applies its clip.
            spread = np.maximum(iqr, (p99 - p1) / _TAIL_TO_IQR)

            # Both degenerate means the channel is genuinely constant (usually:
            # never worn). Identity-transform it rather than dividing by an
            # epsilon, which would amplify float noise into a strong spurious
            # signal.
            flat = spread < 1e-12
            center[pid] = np.where(flat, 0.0, med).astype(np.float32)
            scale[pid] = np.where(flat, 1.0, spread).astype(np.float32)
        return PerPatientNormalizer(center=center, scale=scale, channels=chans)

    def transform(self, x_aux: np.ndarray, patient: np.ndarray) -> np.ndarray:
        out = x_aux.copy()
        chans = list(self.channels)
        for pid in {str(p) for p in patient}:
            if pid not in self.center:
                # Unseen patient (held-out-subject protocol): leave raw and let
                # the global scaler handle it, rather than inventing statistics.
                continue
            m = patient == pid
            # Clipped as a backstop to the tail-aware spread in `fit`: a single
            # electrode-artefact spike must not set the scale of every
            # activation downstream of it. Applied only where the per-patient
            # transform was applied, so an unseen subject's raw device units are
            # left for the global scaler rather than clipped in their own units.
            out[np.ix_(np.flatnonzero(m), np.arange(out.shape[1]), chans)] = np.clip(
                (x_aux[m][..., chans] - self.center[pid]) / self.scale[pid],
                -_PER_PATIENT_CLIP, _PER_PATIENT_CLIP)
        return out


@dataclass
class OhioPreprocessor:
    """Everything fitted from data, bundled so it can be frozen and serialised.

    The ORDER in `fit` is the load-bearing part, and it mirrors
    `train.run_fold` exactly:

        1. split                     (no window data touched)
        2. scalers      <- FIT rows only
        3. noise / MARD <- FIT rows only  -> soft-label band
        4. class prior  <- FIT rows only  -> pos_weight

    Any reordering that fits one of these after fit/val/test have been merged is
    a leak. `fitted_on_patients` records which patients contributed, so an audit
    can confirm it after the fact rather than trusting the call order.

    `reference_pairs`, when supplied, upgrades step 3 from a high-pass proxy to
    a real MARD against paired capillary BG. It is filtered to each patient's
    fit rows first (`ReferencePairs.before`), so it enters the fold under the
    same contract as everything else here.
    """
    cgm_scaler: CausalScaler
    aux_scaler: CausalScaler
    per_patient: PerPatientNormalizer
    labeler: SoftLabeler
    pos_weight: dict                # horizon (min) -> positive class weight
    mard_proxy: float               # the MARD the band was derived from
    fitted_on_patients: tuple[str, ...]
    fitted_on_rows: int
    horizons_min: tuple[int, ...]
    noise: NoiseEstimate | None = None   # provenance behind `mard_proxy`

    @staticmethod
    def fit(ws: OhioWindowSet, split: OhioSplit, cfg: DataConfig,
            reference_pairs: "ReferencePairs | None" = None) -> "OhioPreprocessor":
        if split.fit.size == 0:
            raise ValueError("Empty fit split; nothing to estimate statistics from.")

        # `CausalScaler.fit_array` selects rows by patient id, but here fit and
        # val share patients -- selecting by patient would pull val rows into
        # the statistics. Passing a fit-only view with synthetic ids keeps the
        # audited `fitted_on` field meaningful while restricting the rows.
        fit_pat = ws.patient[split.fit]
        cgm_scaler = CausalScaler.fit_array(
            ws.x[split.fit], fit_pat, sorted({str(p) for p in fit_pat}), skip_channels=(4,))

        # Per-patient device normalisation comes FIRST, so the global scaler
        # sees comparable units across patients. Fitting the global scaler on
        # raw device units would let one patient's electrode offset set the IQR
        # for the whole cohort.
        per_patient = PerPatientNormalizer.fit(ws.x_aux, ws.patient, split.fit)
        aux_fit = per_patient.transform(ws.x_aux[split.fit], fit_pat)
        # `tail_aware` + `clip`: the per-patient stage above bounds each
        # subject, but bounding it compresses the POOLED distribution, so a
        # plain-IQR global scaler re-expands that bound ~40x (GSR reached 797.8).
        # See `CausalScaler.fit_array`. The CGM scaler above deliberately does
        # not take these -- its tensor is well conditioned at max |z| 22.4.
        aux_scaler = CausalScaler.fit_array(
            aux_fit, fit_pat, sorted({str(p) for p in fit_pat}),
            skip_channels=OHIO_AUX_SKIP_SCALING,
            tail_aware=True, clip=SCALED_FEATURE_CLIP)

        # Tier 1: paired capillary reference, restricted to this fold's rows.
        # Tier 2: the signal's own high-frequency content, same restriction.
        # `SoftLabeler.fit` applies tier 3 (the device-specification floor) and
        # records which tier the band actually came from.
        noise = None
        if reference_pairs is not None and len(reference_pairs):
            pairs = reference_pairs.before(fit_row_cutoffs(ws, split.fit))
            if len(pairs):
                noise = estimate_reference_mard(pairs.cgm, pairs.reference,
                                                pairs.patient)
        if noise is None or not np.isfinite(noise.mard):
            noise = estimate_signal_noise(
                anchor_series_by_patient(ws, split.fit), cfg.sample_period_min)
        labeler = SoftLabeler.fit(ws.y_min[split.fit], cfg, noise=noise)
        mard = noise.mard if np.isfinite(noise.mard) else 0.10

        pos_weight = {}
        for k, h in enumerate(ws.horizons_min):
            prev = float(ws.y_cls_multi[split.fit, k].mean())
            pos_weight[int(h)] = float((1 - prev) / max(prev, 1e-6))

        return OhioPreprocessor(
            cgm_scaler=cgm_scaler, aux_scaler=aux_scaler, per_patient=per_patient,
            labeler=labeler,
            pos_weight=pos_weight, mard_proxy=float(mard), noise=noise,
            fitted_on_patients=tuple(sorted({str(p) for p in fit_pat})),
            fitted_on_rows=int(split.fit.size),
            horizons_min=tuple(int(h) for h in ws.horizons_min),
        )

    def transform(self, ws: OhioWindowSet, idx: np.ndarray) -> dict:
        """Apply the frozen objects. Nothing is fitted here, ever.

        Targets are returned untransformed and in mg/dL. Scaling them would put
        a fitted object on the label path, and the Huber delta in `ModelConfig`
        is specified in mg/dL, so unscaled targets are also what the loss
        expects.
        """
        return {
            "x": self.cgm_scaler.transform(ws.x[idx]),
            "x_aux": self.aux_scaler.transform(
                self.per_patient.transform(ws.x_aux[idx], ws.patient[idx])),
            "y_reg": ws.y_reg[idx],
            "y_reg_multi": ws.y_reg_multi[idx],
            "y_soft": self.labeler(ws.y_min[idx]),
            "y_soft_multi": np.stack(
                [self.labeler(ws.y_min_multi[idx, k]) for k in range(len(self.horizons_min))],
                axis=-1),
            "y_cls_hard": ws.y_cls_hard[idx],
            "y_cls_multi": ws.y_cls_multi[idx],
            "last_bg": ws.last_bg[idx],
            # From the masks AND the mask-free treatment channels, on raw
            # values -- see `data.aux_presence`. `ws.aux_frac` alone ignored
            # every Empatica stream and every insulin/carb channel.
            "aux_present": aux_presence(ws.x_aux[idx], OHIO_AUX_CHANNELS),
        }


# --------------------------------------------------------------------------- #
# PyTorch datasets and loaders
# --------------------------------------------------------------------------- #
_DATASET_CLASS = None


def _dataset_class():
    """Build the Dataset subclass on first use.

    Deferred so that importing this module does not import torch: the XML
    parsing, synchronisation and windowing paths are useful on their own (and
    are what the tests exercise most), and torch is a heavy and, on macOS,
    import-order-sensitive dependency -- see the note in `__init__`.
    """
    global _DATASET_CLASS
    if _DATASET_CLASS is not None:
        return _DATASET_CLASS

    import torch
    from torch.utils.data import Dataset

    _KEEP = ("x", "x_aux", "y_reg", "y_soft", "last_bg", "aux_present",
             "y_reg_multi", "y_soft_multi", "y_cls_multi")

    class _OhioWindowDataset(Dataset):
        """Multi-horizon multitask dataset.

        Batch layout. The first six entries match `train.WindowDataset` exactly,
        so `train._forward` and the existing multimodal model consume an Ohio
        batch unchanged; the multi-horizon tensors are appended after them:

            x            (H, C_cgm)   scaled CGM features
            y_reg        ()           BG at the primary horizon, mg/dL
            y_soft       ()           soft hypo target, primary horizon
            last_bg      ()           BG at t, mg/dL (persistence baseline)
            x_aux        (H, C_aux)   scaled auxiliary features
            aux_present  ()           1 if the aux stream is usable
            y_reg_multi  (K,)         BG at each horizon
            y_soft_multi (K,)         soft hypo target at each horizon
            y_cls_multi  (K,)         hard hypo label at each horizon

        No fitting happens here. Scaling and soft labelling were applied by a
        frozen `OhioPreprocessor` before construction, which keeps the entire
        leak surface inside `OhioPreprocessor.fit`.
        """

        def __init__(self, arrays: dict, horizons_min):
            self.horizons_min = tuple(int(h) for h in horizons_min)
            self._t = {k: torch.from_numpy(np.ascontiguousarray(v))
                       for k, v in arrays.items() if k in _KEEP}

        def __len__(self) -> int:
            return self._t["x"].shape[0]

        def __getitem__(self, i):
            t = self._t
            return (t["x"][i], t["y_reg"][i], t["y_soft"][i], t["last_bg"][i],
                    t["x_aux"][i], t["aux_present"][i],
                    t["y_reg_multi"][i], t["y_soft_multi"][i], t["y_cls_multi"][i])

    _DATASET_CLASS = _OhioWindowDataset
    return _DATASET_CLASS


def OhioWindowDataset(arrays: dict, horizons_min) -> "object":
    """Construct the multitask dataset (see `_dataset_class` for the layout)."""
    return _dataset_class()(arrays, horizons_min)


def make_dataset(arrays: dict, horizons_min: Sequence[int], style: str = "mtl"):
    """Wrap transformed arrays in a Dataset.

    Two styles, because there are two consumers and silently serving one shape
    to the other would surface as an unpacking error deep inside a training
    loop:

      "mtl"    -> `OhioWindowDataset`, the 9-tuple with the multi-horizon
                  tensors appended. For a model with per-horizon heads.
      "compat" -> `train.WindowDataset`, the exact 6-tuple the existing
                  `train._forward` unpacks, carrying the PRIMARY horizon only.
                  This is what lets the Ohio data train the current model with
                  no changes to `models.py` or the loss.
    """
    if style == "mtl":
        return OhioWindowDataset(arrays, horizons_min)
    if style == "compat":
        from .train import WindowDataset
        return WindowDataset(
            arrays["x"], arrays["y_reg"], arrays["y_soft"], arrays["last_bg"],
            x_aux=arrays["x_aux"], aux_present=arrays["aux_present"],
        )
    raise ValueError(f"Unknown dataset style {style!r}; expected 'mtl' or 'compat'.")


def make_ohio_loaders(
    ws: OhioWindowSet,
    split: OhioSplit,
    cfg: DataConfig,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 1337,
    pre: OhioPreprocessor | None = None,
    style: str = "mtl",
    reference_pairs: ReferencePairs | None = None,
) -> tuple[dict, OhioPreprocessor]:
    """Build the three DataLoaders plus the fitted preprocessor.

    Only the fit loader shuffles. Shuffling is safe *within* a split because the
    windows are already materialised -- the temporal structure that matters was
    enforced when the split was drawn, not by iteration order.

    Pass an existing `pre` to reuse a preprocessor fitted elsewhere (for example
    when scoring a saved model); the default fits a fresh one on `split.fit`.
    `reference_pairs` reaches `OhioPreprocessor.fit`, which is what upgrades
    `band_mode="mard"` from a high-pass proxy to a measured MARD.
    See `make_dataset` for `style`.
    """
    import torch
    from torch.utils.data import DataLoader

    pre = pre or OhioPreprocessor.fit(ws, split, cfg,
                                      reference_pairs=reference_pairs)
    g = torch.Generator().manual_seed(seed)

    loaders = {}
    for name, idx, shuffle in (("fit", split.fit, True),
                               ("val", split.val, False),
                               ("test", split.test, False)):
        if idx.size == 0:
            loaders[name] = None
            continue
        ds = make_dataset(pre.transform(ws, idx), ws.horizons_min, style)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            drop_last=shuffle and len(ds) > batch_size,
            generator=g if shuffle else None,
        )
    return loaders, pre


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_ohio_tensors(
    ws: OhioWindowSet,
    split: OhioSplit,
    cfg: DataConfig,
    out_dir: str,
    pre: OhioPreprocessor | None = None,
) -> str:
    """Write scaled tensors + metadata to `out_dir/ohio_windows.pt`.

    Saved SCALED, using a preprocessor fitted on the fit split only, so the
    artefact cannot be reloaded in a way that re-derives statistics from
    val/test. The preprocessor itself is saved alongside, so inference on new
    data applies exactly the transform the model was trained under, and the
    `fitted_on` audit trail survives with the file.
    """
    import torch

    pre = pre or OhioPreprocessor.fit(ws, split, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ohio_windows.pt")

    payload = {
        "splits": {},
        "meta": {
            # This artefact is materialised window CONTENT, so it carries the
            # same version key every other cache in the pipeline does. Without
            # it a dump built before the backward-snap fix reloads with every
            # shape and every meta field identical and silently wrong values --
            # the one failure this whole versioning scheme exists to stop.
            "pipeline_version": WINDOW_PIPELINE_VERSION,
            "horizons_min": list(ws.horizons_min),
            "cgm_channels": ["value", "diff", "roll_mean", "roll_std", "observed"],
            "aux_channels": list(OHIO_AUX_CHANNELS),
            "aux_skip_scaling": list(OHIO_AUX_SKIP_SCALING),
            # Every observation mask, not just heart rate: `hr_observed` is
            # identically zero for the 2020 Empatica cohort, and a model gated
            # on it alone runs with its fusion branch fully masked for half the
            # subjects. See `aux_validity_indices`.
            "aux_valid_index": list(aux_validity_indices()),
            "hypo_threshold_mgdl": cfg.hypo_threshold_mgdl,
            "sample_period_min": cfg.sample_period_min,
            "history_steps": cfg.history_steps,
            "data_config": dataclasses.asdict(cfg),
            "n_embargoed": int(split.embargoed.size),
            # `calib` travels with the artefact like every other region. A
            # consumer that computes conformal intervals needs a calibration
            # set disjoint from the early-stopping set, and re-deriving one
            # from a reloaded artefact is not possible: the split's chronology
            # lives in the row order, not in the tensors.
            "n_calib": int(split.calib.size),
        },
        "preprocessor": pre,
    }
    for name, idx in (("fit", split.fit), ("val", split.val),
                      ("calib", split.calib), ("test", split.test)):
        if idx.size == 0:
            continue
        arrays = pre.transform(ws, idx)
        arrays["patient"] = np.asarray([str(p) for p in ws.patient[idx]])
        arrays["t_time"] = ws.t_time[idx].astype("datetime64[ns]").astype(np.int64)
        arrays["subset"] = np.asarray([str(s) for s in ws.subset_tag[idx]])
        payload["splits"][name] = {
            k: (torch.from_numpy(v) if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" else v)
            for k, v in arrays.items()
        }

    torch.save(payload, path)
    print(f"Saved {path}")
    return path


def load_ohio_tensors(path: str, require_current_version: bool = True) -> dict:
    """Reload a `save_ohio_tensors` artefact.

    `weights_only=False` is required because the payload carries the fitted
    preprocessor, not just tensors. Only load artefacts you produced.

    RAISES on a dump from a different windowing pipeline rather than warning.
    Everything downstream of here consumes tensors -- there is no later stage
    that could notice the values are from a superseded snap, so a warning would
    be read once and then scrolled past while the numbers went into a table.
    Pass `require_current_version=False` to inspect an old dump deliberately.
    """
    import torch

    payload = torch.load(path, weights_only=False)
    got = payload.get("meta", {}).get("pipeline_version", "<pre-versioning>")
    if require_current_version and got != WINDOW_PIPELINE_VERSION:
        raise ValueError(
            f"{path} was written by window pipeline version {got}, this is "
            f"{WINDOW_PIPELINE_VERSION}. Its tensors have the right shapes and "
            f"the wrong values; regenerate with `save_ohio_tensors`, or pass "
            f"require_current_version=False to inspect it anyway.")
    return payload


def describe(ws: OhioWindowSet) -> str:
    """Cohort summary. Worth printing on every run: a sudden change in window
    count or hypo prevalence is the fastest signal that a config change did
    something unintended."""
    lines = [
        f"{len(ws)} windows | {len(set(map(str, ws.patient)))} patients | "
        f"horizons {ws.horizons_min} min",
    ]
    for name in ("train", "test"):
        idx = ws.where(name)
        if idx.size == 0:
            continue
        prev = ", ".join(f"{h}min {ws.y_cls_multi[idx, k].mean():.2%}"
                         for k, h in enumerate(ws.horizons_min))
        lines.append(f"  {name:>5}: {idx.size:7d} windows | hypo prevalence: {prev}")
    lines.append(f"  mean aux coverage: {ws.aux_frac.mean():.1%}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Modality ablation
# --------------------------------------------------------------------------- #
def modality_availability(ws: OhioWindowSet) -> pd.DataFrame:
    """Per-cohort, per-patient observed fraction for every modality group.

    Run this BEFORE designing an ablation. The 2018 subjects wore a Basis Peak
    and the 2020 subjects an Empatica Embrace, and the sensor sets do not
    overlap: heart rate and steps exist only in 2018, acceleration only in 2020.
    An arm built on a cohort-specific sensor is therefore scored on six
    subjects, and comparing it to an arm scored on the other six compares
    PATIENTS, not sensors. This table is what makes that visible instead of
    inferred.
    """
    rows = []
    for pid in sorted({str(p) for p in ws.patient}):
        m = ws.patient == pid
        row = {"patient": pid, "subset": str(ws.subset_tag[m][0]), "windows": int(m.sum())}
        for group, chans in MODALITY_GROUPS.items():
            masks = [OHIO_AUX_INDEX[c] for c in chans if c.endswith("_observed")]
            if masks:
                row[group] = float(ws.x_aux[m][..., masks].mean())
            else:
                # No mask channel: report the fraction of steps that are non-zero,
                # which for impulse/level streams is the honest analogue.
                idxs = [OHIO_AUX_INDEX[c] for c in chans]
                row[group] = float((np.abs(ws.x_aux[m][..., idxs]) > 0).any(axis=-1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_arm(groups: Sequence[str]) -> tuple[list[str], list[str]]:
    """Expand an ablation arm into (channel names, cohorts that can support it).

    The cohort list is the INTERSECTION over the arm's groups: an arm asking for
    both heart rate (2018 only) and acceleration (2020 only) is unsatisfiable,
    and is reported as such rather than quietly training on windows where one of
    the two is entirely mask.
    """
    unknown = [g for g in groups if g not in MODALITY_GROUPS]
    if unknown:
        raise ValueError(f"Unknown modality groups {unknown}. "
                         f"Available: {sorted(MODALITY_GROUPS)}")
    chans: list[str] = []
    cohorts = {"2018", "2020"}
    for g in groups:
        chans.extend(MODALITY_GROUPS[g])
        cohorts &= set(MODALITY_COHORTS.get(g, ("2018", "2020")))
    return chans, sorted(cohorts)


def apply_modality_mask(x_aux: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """Zero every auxiliary channel outside the selected groups.

    Values AND masks are zeroed together. Zeroing only the values would leave
    the mask asserting "this was observed" over a channel of zeros, which is
    strictly worse than absence: the model would read a confident measurement of
    nothing. Keeping the tensor's WIDTH fixed (rather than slicing columns out)
    means every arm instantiates an identically shaped model, so a difference
    between arms cannot come from a difference in parameter count.
    """
    keep = set()
    for g in groups:
        keep |= set(MODALITY_GROUPS[g])
    idx = [i for i, c in enumerate(OHIO_AUX_CHANNELS) if c not in keep]
    out = x_aux.copy()
    out[..., idx] = 0.0
    return out


def aux_validity_indices(groups: Sequence[str] | None = None) -> tuple[int, ...]:
    """Observation-mask channels that make a timestep auxiliary-valid.

    Pass the modality groups an arm actually uses (or None for all of them);
    the result goes to `MultiModalCausalCGMMT(aux_valid_index=...)`, where a step
    counts as valid when ANY of these channels is set.

    WHY THIS EXISTS
    ---------------
    The model used to read validity from `hr_observed` alone. The 2020 subjects
    wore an Empatica, which reports no heart rate, so that channel is
    identically zero for all 71,946 of their windows -- the cross-attention was
    fully masked and the "multimodal" model was silently the CGM-only one for
    half the cohort. It also made every modality-ablation arm that excluded
    heart rate collapse onto the same numbers, for the same reason.

    Groups with no observation mask (`treatment`, `exercise`) contribute no
    indices, which is correct: insulin-on-board has no mask because a zero is a
    real zero, not a missing reading. An arm consisting only of such groups
    therefore gets an EMPTY tuple, which the model reads as "every step is
    valid" rather than "no step is valid".
    """
    names = list(MODALITY_GROUPS) if groups is None else [str(g) for g in groups]
    unknown = [g for g in names if g not in MODALITY_GROUPS]
    if unknown:
        raise ValueError(f"unknown modality group(s) {unknown}; expected "
                         f"{sorted(MODALITY_GROUPS)}")
    idx = [OHIO_AUX_INDEX[c] for g in names for c in MODALITY_GROUPS[g]
           if c.endswith("_observed")]
    return tuple(sorted(set(idx)))


def modality_channel_names(groups: Sequence[str] | None = None) -> tuple[str, ...]:
    """The auxiliary channels an arm receives: every channel for None."""
    if groups is None:
        return tuple(OHIO_AUX_CHANNELS)
    keep: set[str] = set()
    for g in groups:
        keep |= set(MODALITY_GROUPS[str(g)])
    return tuple(c for c in OHIO_AUX_CHANNELS if c in keep)


def restrict_to_cohorts(ws: OhioWindowSet, cohorts: Sequence[str]) -> np.ndarray:
    """Row indices belonging to the given release years."""
    want = set(str(c) for c in cohorts)
    return np.flatnonzero(np.array([str(s) in want for s in ws.subset_tag]))


# Default ablation ladder. Each arm is (name, groups). The cohort restriction is
# derived, not specified, so adding an arm cannot accidentally mix cohorts.
DEFAULT_ABLATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cgm_only", ()),
    ("cgm_treatment", ("treatment",)),
    ("cgm_hr", ("heart_rate",)),                       # 2018 only
    ("cgm_gsr", ("gsr",)),                             # both cohorts
    ("cgm_accel", ("accel",)),                         # 2020 only
    ("cgm_skin_temp", ("skin_temp",)),
    ("cgm_sleep_steps", ("sleep", "steps")),           # 2018 only
    ("all_2018", ("treatment", "heart_rate", "steps", "sleep", "gsr",
                  "skin_temp", "air_temp", "finger_stick", "exercise")),
    ("all_2020", ("treatment", "accel", "gsr", "skin_temp", "sleep",
                  "finger_stick", "exercise")),
    ("all_shared", ("treatment", "gsr", "skin_temp", "sleep", "finger_stick")),
)
