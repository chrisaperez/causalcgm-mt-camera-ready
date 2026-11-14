"""MetaboNet ingestion, synchronisation, splitting and windowing.

MetaboNet is not a study -- it is a CONSOLIDATION of fourteen public type-1
diabetes datasets into one long-format parquet (154,842,077 rows, 37 columns,
1,907 subjects) on a shared 5-minute grid. That shape makes it convenient and
makes three specific things dangerous. Each is handled structurally here rather
than left to the caller's care.

1. IT CONTAINS THIS REPOSITORY'S BENCHMARKS.
   `source_file` takes fourteen values, two of which are `OhioT1DM` (all 12
   subjects, 190,966 rows) and `HUPA-UCM` (22 subjects, 297,866 rows) -- the
   cohorts every headline number in this repository is measured on. Worse, all
   12 Ohio subjects' rows are labelled `is_test = False`: 31,734 CGM samples
   lying inside the OFFICIAL Ohio *test* spans sit on MetaboNet's TRAIN side.
   Training on MetaboNet-train and reporting on the Ohio benchmark is therefore
   direct test-set training, and nothing in the file's own schema says so.
   `MetaboNetConfig.exclude_sources` drops those two sources BY DEFAULT and
   `assert_no_benchmark_contamination` refuses a split that carries them, so
   including them is an explicit, recorded act rather than an omission.

2. `id` IS NOT A SUBJECT KEY.
   241 of the 1,291 distinct `id` values appear under more than one
   `source_file` -- `id="1"` exists in AZT1D, CTR3, DCLP5, Flair and HUPA-UCM,
   and those are five different people. Grouping by `id` would merge strangers
   into one "patient" and scatter one patient's rows across folds, defeating
   patient-wise splitting in the one way it is designed to prevent. The subject
   key here is always the pair, formatted `"<source_file>:<id>"` by
   `subject_key`, and `patient` on the window set carries that composite.

3. `basal` IS UNITS PER BIN, NOT UNITS PER HOUR.
   Verified against the OhioT1DM XML this repository ships: MetaboNet reports
   subject 596's basal as 0.0375 where the source XML says 0.45 U/hr, and
   0.0375 x 12 = 0.45. So `basal` is the dose DELIVERED in the 5-minute slot,
   and `insulin == basal + bolus` exactly (checked on the same subject). Basal
   is consequently treated as an impulse train and put through the same causal
   IOB filter as bolus -- reading it as a rate, the way `ohio_loader` correctly
   reads its own `basal` field, would understate insulin on board by ~12x.
   `insulin` itself is never ingested: it is the sum, and using it alongside
   its own components would double count.

SPLIT PROTOCOL
--------------
MetaboNet ships `is_test`, and `subject_split_across_traintest` claims to say
whether a subject appears on both sides. Derived from the rows rather than
believed, the released file's 1,907 subjects fall into four classes
(`scan_subjects`):

    train_only   1,556 subjects, every row is_test=False -- includes all of
                 OhioT1DM
    test_only      182 subjects, every row is_test=True
    temporal       164 subjects whose test rows are STRICTLY LATER than their
                 train rows: a legitimate within-subject temporal split, the
                 same protocol as OhioT1DM's official one
    interleaved      5 subjects (CTR3 ids 5, 9, 10, 12, 13) whose train and
                 test spans OVERLAP IN TIME. Their own split leaks: windows
                 from the same hours land on both sides. They are quarantined
                 by default (`on_interleaved="drop"`).

The shipped flag does not identify that last class, in either direction. It is
True for 166 subjects; the rows say 169 straddle, so it MISSES three (CTR3 9,
10, 13), and two of the subjects it does flag (CTR3 5, 12) are interleaved
rather than temporal. It is a straddle hint, not a statement that a temporal
holdout of the subject is valid, so nothing here branches on it -- every class
above is derived from the timestamps of the subject's own rows.

`official_metabonet_split` honours the shipped split under this repository's
causality contract; `patient_wise_metabonet_split` offers the unseen-subject
protocol instead. They measure different things and are never mixed silently.

WHAT IS SHARED WITH THE OTHER LOADERS, AND WHY
----------------------------------------------
The causal primitives (`_causal_ffill`, `_bin_backward`, `_activity_kernel`,
`causal_on_board`, `snap_to_grid_backward`, `_causal_gap_fill`) are imported
from `hupa_loader` / `data`, never reimplemented. A second copy of a backward
bin is a second place for the `closed="right"` argument to be wrong, and the
whole point of the guarantee is that there is one implementation to audit. The
aux channel layout likewise EXTENDS `hupa_loader.AUX_CHANNELS` at the end,
exactly as `ohio_loader` does, so indices 0-10 keep their meaning and a tensor
built here is intelligible to code written for either of the others.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import (WINDOW_PIPELINE_VERSION, _causal_gap_fill,  # noqa: F401
                   causal_features, causal_features_batch,
                   snap_to_grid_backward)
from .hupa_loader import (AUX_CHANNELS, HupaConfig, MMWindowSet, _activity_kernel,
                          _bin_backward, _causal_ffill, assert_no_bfill,
                          causal_on_board)

# --------------------------------------------------------------------------- #
# Logging and memory accounting
# --------------------------------------------------------------------------- #
# A module logger rather than prints, so a caller decides the destination and
# level. `configure_logging` is a convenience for scripts; importing this module
# adds no handler and changes no global state.
log = logging.getLogger("causalcgm_mt.metabonet")

try:                                    # optional, and genuinely optional
    import psutil as _psutil
    _PROC = _psutil.Process()
except Exception:                       # pragma: no cover - environment dependent
    _psutil = None
    _PROC = None


def rss_gb() -> float:
    """Resident set size in GiB, by whatever means this machine offers.

    `psutil` is exact and cheap. Without it, `resource.getrusage` reports the
    process PEAK RSS (`ru_maxrss`), which is not the same quantity -- it never
    goes down -- so it is reported as an upper bound rather than silently
    substituted for the live figure. A wrong memory number is worse than an
    honest approximation, because the whole point of logging it is to decide
    whether the next step will fit.
    """
    if _PROC is not None:
        return _PROC.memory_info().rss / 2**30
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS/BSD bytes.
        return (peak / 2**20 if os.uname().sysname == "Linux" else peak / 2**30)
    except Exception:
        return float("nan")


def available_gb() -> float:
    """Physically available RAM in GiB, or nan when it cannot be determined."""
    if _psutil is not None:
        return _psutil.virtual_memory().available / 2**30
    return float("nan")


def memory_note() -> str:
    """`RAM 4.2 GB` / `RAM 4.2 GB, 9.1 GB free` -- for the tail of a log line."""
    rss, avail = rss_gb(), available_gb()
    kind = "RAM" if _PROC is not None else "peak RAM"
    out = f"{kind} {rss:.2f} GB" if np.isfinite(rss) else f"{kind} unknown"
    if np.isfinite(avail):
        out += f", {avail:.1f} GB free"
    return out


def configure_logging(level: str | int = "INFO", log_file: str | None = None,
                      force: bool = False) -> None:
    """Attach a flushed stream handler (and optionally a file) to this package.

    Scripts call this; the library never does. `force` replaces existing
    handlers, which is what a driver wants when it owns the process.
    """
    root = logging.getLogger("causalcgm_mt")
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)
    root.setLevel(level if isinstance(level, int) else str(level).upper())
    root.propagate = False


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
# The two consolidated sources that ARE this repository's benchmarks. Training
# on them through this loader and evaluating on the OhioT1DM or HUPA-UCM
# protocols is training on the test set; see the module docstring.
BENCHMARK_SOURCES: tuple[str, ...] = ("OhioT1DM", "HUPA-UCM")

# Version of THIS module's windowing and thinning logic, distinct from
# `data.WINDOW_PIPELINE_VERSION` (which covers window CONTENT and is shared with
# the OhioT1DM cache). It goes into the window-store fingerprint, so a change to
# how windows are selected -- not just to what a window holds -- invalidates a
# cached store instead of being silently mixed with shards built the old way.
# Bump it when `_subject_windows` or `_thin_blocks` changes behaviour.
#   1  initial streaming store
#   2  block thinning fills the cap (a 400 cap with a 288 block kept 288)
#   3  a cap under 1.5x the block length no longer collapses to ONE block at
#      index 0. The shipped default (cap 400, block 288) did exactly that, so
#      every store built under version 2 holds each subject's opening ~33 h and
#      nothing else -- results from one are not comparable with results from
#      this version and the cache must be rebuilt, not adopted.
METABONET_WINDOW_VERSION: int = 3

# Physiologically possible CGM readings, mg/dL. MetaboNet's T1D-UOM slice
# reports values as low as 1.8 and several sources reach 500+; a sensor does not
# measure 1.8 mg/dL in a living subject, so such a row is a sentinel or a unit
# error, and letting it through would corrupt both the regression target and the
# fold's noise estimate.
PLAUSIBLE_CGM_MGDL: tuple[float, float] = (20.0, 600.0)

# basal is delivered units per sampling bin; multiply by bins-per-hour for U/hr.
MINUTES_PER_HOUR = 60


def subject_key(source: str, id_: str) -> str:
    """The only admissible patient identity for MetaboNet.

    `id` alone collides across `source_file` (241 of 1,291 ids), so every split,
    scaler fit and fold in this module keys on the pair. The separator is `":"`,
    which appears in neither field in the released file.
    """
    return f"{source}:{id_}"


def split_subject_key(key: str) -> tuple[str, str]:
    source, _, id_ = str(key).partition(":")
    return source, id_


# --------------------------------------------------------------------------- #
# Channel layout
# --------------------------------------------------------------------------- #
# Entries 0-10 are byte-for-byte `hupa_loader.AUX_CHANNELS`, in the same order,
# for the same load-bearing reason `ohio_loader` does it: `AUX_INDEX[...]` and
# `AUX_SKIP_SCALING == (7, 8, 9)` are read by `train.run_fold` and passed to the
# model as `aux_valid_index`. MetaboNet's extra streams are appended at the END
# so none of those indices move.
#
# `sleep` (index 2) has no MetaboNet column at all. It is kept in the layout,
# permanently zero with its mask clear -- an honestly absent modality rather
# than a renumbered tensor. That is the graceful-degradation path in requirement
# 3, and `test_absent_sleep_is_masked_not_invented` pins it.
METABONET_EXTRA_CHANNELS: tuple[str, ...] = (
    "basal_dose",            # 11 units delivered in this bin (NOT U/hr)
    "basal_rate",            # 12 U/hr implied by the bin dose, for readability
    "basal_iob",             # 13 units on board from basal delivery alone
    "gsr",                   # 14 electrodermal activity, device units
    "gsr_observed",          # 15 mask
    "skin_temp",             # 16 as released
    "skin_temp_observed",    # 17 mask
    "air_temp",              # 18 as released
    "air_temp_observed",     # 19 mask
    "calories",              # 20 kcal burned in the bin (interval count)
    "calories_observed",     # 21 mask
    "workout_intensity",     # 22 held over the logged duration
    "workout_observed",      # 23 mask
)
METABONET_AUX_CHANNELS: tuple[str, ...] = AUX_CHANNELS + METABONET_EXTRA_CHANNELS
METABONET_AUX_INDEX = {name: i for i, name in enumerate(METABONET_AUX_CHANNELS)}

# Binary indicators the scaler must pass through untouched.
METABONET_AUX_SKIP_SCALING: tuple[int, ...] = tuple(
    i for i, c in enumerate(METABONET_AUX_CHANNELS) if c.endswith("_observed")
)

# Channels that stand or fall together because they come from one sensor. The
# unit of a modality ablation, mirroring `ohio_loader.MODALITY_GROUPS`.
MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "hr": ("heart_rate", "hr_observed", "hr_staleness"),
    "steps": ("steps", "steps_observed"),
    "sleep": ("sleep", "sleep_observed"),
    "treatment": ("iob", "cob", "bolus", "carbs",
                  "basal_dose", "basal_rate", "basal_iob"),
    "gsr": ("gsr", "gsr_observed"),
    "temp": ("skin_temp", "skin_temp_observed", "air_temp", "air_temp_observed"),
    "activity": ("calories", "calories_observed",
                 "workout_intensity", "workout_observed"),
}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetaboNetSchema:
    """Maps MetaboNet's released column names onto canonical names.

    Unlike `HupaSchema`, these defaults ARE verified: they were read from the
    parquet footer of the released file. `validate` still reports what it found,
    because a future release may add or rename columns and a schema drift should
    surface as one clear error rather than as an all-NaN channel.
    """
    time: str = "date"
    glucose: str = "CGM"
    patient: str = "id"
    source: str = "source_file"
    is_test: str = "is_test"
    split_flag: str = "subject_split_across_traintest"

    heart_rate: str = "heartrate"
    steps: str = "steps"
    bolus: str = "bolus"
    basal: str = "basal"
    carbs: str = "carbs"
    gsr: str = "galvanic_skin_response"
    skin_temp: str = "skin_temp"
    air_temp: str = "air_temp"
    calories: str = "calories_burned"
    workout_intensity: str = "workout_intensity"
    workout_duration: str = "workout_duration"

    # Without these five the file is not MetaboNet and nothing downstream is
    # meaningful. Everything else may be absent -- the mask channels carry it.
    required: tuple[str, ...] = ("time", "glucose", "patient", "source", "is_test")

    def required_columns(self) -> list[str]:
        return [getattr(self, name) for name in self.required]

    def optional_columns(self) -> list[str]:
        return [getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name not in self.required and f.name != "required"]

    def all_columns(self) -> list[str]:
        return self.required_columns() + self.optional_columns()

    def validate(self, columns: Sequence[str], path: str) -> list[str]:
        """Raise if a required column is missing; return the optional ones found."""
        have = set(map(str, columns))
        missing = [c for c in self.required_columns() if c not in have]
        if missing:
            raise ValueError(
                f"{path} is missing required MetaboNet column(s) {missing}. "
                f"Found: {sorted(have)}. Adjust MetaboNetSchema; nothing "
                f"downstream depends on the raw names."
            )
        return [c for c in self.optional_columns() if c in have]

    def rename_map(self, columns: Sequence[str]) -> dict[str, str]:
        have = set(map(str, columns))
        return {getattr(self, f.name): f.name
                for f in dataclasses.fields(self)
                if f.name != "required" and getattr(self, f.name) in have}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetaboNetConfig:
    """MetaboNet-specific ingestion policy.

    Alignment and pharmacokinetic parameters are NOT duplicated here: they live
    in `HupaConfig` and are composed, so a change to the IOB curve applies to
    every loader that uses it instead of to whichever ones were remembered.
    """
    hupa: HupaConfig = field(default_factory=HupaConfig)

    # ---- provenance policy --------------------------------------------------
    # Sources dropped before anything is built. Defaults to the two that ARE
    # this repository's benchmarks; see the module docstring. Pass an empty
    # tuple to include them, which is then recorded on the window set and the
    # split so the decision is visible in the artefacts rather than only in a
    # command line.
    exclude_sources: tuple[str, ...] = BENCHMARK_SOURCES
    include_sources: tuple[str, ...] | None = None      # None = everything else

    # What to do with a subject whose train and test rows interleave in time.
    # "drop"  -- quarantine the subject (default; its own split is unusable)
    # "train" -- keep it, forcing every row onto the train side
    # "raise" -- stop, for a caller who would rather fix the input
    on_interleaved: Literal["drop", "train", "raise"] = "drop"

    # ---- plausibility -------------------------------------------------------
    cgm_range_mgdl: tuple[float, float] = PLAUSIBLE_CGM_MGDL

    # ---- coverage floors ----------------------------------------------------
    # A subject with almost no CGM contributes windows that are mostly imputed
    # history and no usable targets; the fixed cost of carrying it through the
    # pipeline is larger than its information content.
    min_cgm_frac: float = 0.05
    min_rows_per_subject: int = 0        # 0 = derive from history + horizon

    # ---- ingestion ----------------------------------------------------------
    # How many subjects one lazy `collect` materialises. Peak ingestion memory
    # is this many subjects' rows, never the corpus. The released file's largest
    # subject is ~126k rows, so 32 is a few hundred MB.
    ingest_batch_subjects: int = 32

    # ...but a batch is closed early once it reaches this many rows. Subjects
    # differ by more than 10x in record length between studies (a Loop subject
    # is ~126k rows, an AZT1D one ~13k), so a fixed subject count means a batch
    # of Loop subjects is ten times the frame a batch of AZT1D ones is. Rows are
    # what actually cost memory, so rows are what is bounded.
    ingest_batch_rows: int = 1_500_000

    # Row ceiling for `load_metabonet_frame`, the eager convenience path. The
    # whole corpus is 154,842,077 rows (~24 GB in pandas), so an accidental
    # whole-file call is a process kill rather than an exception; this turns it
    # into an error that names the streaming alternative. 0 disables the check.
    max_eager_rows: int = 20_000_000

    # ---- window budget ------------------------------------------------------
    # The corpus yields ~134M windows at stride 1. At 2,784 B/window (a 24-step
    # history of 5 CGM features plus 24 aux channels, float32) that is ~370 GB
    # of tensors -- not a RAM problem to be solved by streaming, a quantity that
    # does not fit on the disk either. A cap is therefore part of the protocol,
    # not a convenience.
    #
    # Windows are thinned within the subject as CONTIGUOUS BLOCKS spread across
    # the record (`_thin_blocks`, at least two of them), never as every k-th and
    # never truncated to the first N. A record spans weeks, so keeping only its
    # opening would bias every arm towards whatever was happening when the
    # sensor was first fitted; keeping every k-th would bound memory just as
    # well and silently empty every alarm metric instead. 0 = no cap.
    max_windows_per_subject: int = 0

    # Windows are kept as contiguous blocks of this many, not as every k-th, so
    # the anchor cadence survives inside a block. 288 is 24 h at a 5-minute
    # cadence and stride 1: long enough to hold whole hypoglycemic episodes and
    # a personalisation prefix. See `_thin_blocks` for what breaks otherwise.
    thin_block_windows: int = 288

    # Refuse to build a window store projected to exceed this many GiB. Checked
    # against a running estimate, so a run that cannot finish stops early with
    # the knobs that would make it fit. 0 disables the check.
    memory_budget_gb: float = 0.0


# --------------------------------------------------------------------------- #
# Subject index -- the cheap first pass
# --------------------------------------------------------------------------- #
@dataclass
class SubjectIndex:
    """Per-subject provenance, derived from the rows rather than from the flag.

    Built by a projection scan over five columns (~13 s on the released file)
    so a caller can choose subjects without materialising 154M rows of 37
    columns. `frame` carries one row per (source, id) with:

        key, source, id, rows, n_train, n_test, t_start, t_end,
        test_start, split_class, flag_says_split
    """
    frame: pd.DataFrame

    CLASSES = ("train_only", "test_only", "temporal", "interleaved")

    def __len__(self) -> int:
        return len(self.frame)

    def keys(self, classes: Sequence[str] | None = None) -> list[str]:
        f = self.frame
        if classes is not None:
            f = f[f.split_class.isin(list(classes))]
        return sorted(f.key.tolist())

    def sources(self) -> list[str]:
        return sorted(self.frame.source.unique().tolist())

    @property
    def disagreements(self) -> pd.DataFrame:
        """Subjects where `subject_split_across_traintest` contradicts the rows.

        On the released file this is three CTR3 subjects (9, 10, 13) that
        straddle train and test while the flag says they do not. Note that
        agreement is not safety: the flag also marks CTR3 5 and 12 exactly as it
        marks the 164 clean temporal subjects, and those two interleave. Read
        `split_class` for that, never the flag.
        """
        f = self.frame
        derived = f.split_class.isin(("temporal", "interleaved"))
        return f[derived != f.flag_says_split]

    def summary(self) -> str:
        f = self.frame
        by = f.groupby(["source", "split_class"]).size().unstack(fill_value=0)
        return (f"{len(f)} subjects, {int(f['rows'].sum()):,} rows\n"
                f"{by.to_string()}")


def scan_subjects(
    path: str,
    schema: MetaboNetSchema | None = None,
    mcfg: MetaboNetConfig | None = None,
) -> SubjectIndex:
    """One lazy pass over the provenance columns only.

    Reads `id`, `source_file`, `is_test`, `subject_split_across_traintest` and
    the timestamp -- five columns of 37 -- through `pl.scan_parquet`, and
    derives each subject's split class from the timestamps of its own rows.

    Polars rather than a pyarrow row-group loop, and it is not a stylistic
    preference: the aggregation is expressed lazily and executed by the
    streaming engine, so the 154,842,077-row file is reduced to one row per
    (subject, is_test) group without ever materialising the column. On the
    released file that is 0.5 s and ~0.2 GB, against 13 s and a per-row-group
    pandas frame for the loop it replaces.

    The classification is deliberately not a lookup of the shipped flag. On the
    released file the flag misses three straddling subjects outright (CTR3 9,
    10, 13) and marks two interleaved ones (CTR3 5, 12) no differently from the
    164 clean temporal ones. A protocol built on it would put interleaved rows
    of one person on both sides of the split and call it a temporal holdout.
    """
    pl = _polars()
    schema = schema or MetaboNetSchema()
    mcfg = mcfg or MetaboNetConfig()

    t0 = time.time()
    lf = pl.scan_parquet(path)
    available = list(lf.collect_schema().names())
    schema.validate(available, path)

    have_flag = schema.split_flag in available
    keys = [schema.source, schema.patient, schema.is_test]
    aggs = [pl.len().alias("rows"),
            pl.col(schema.time).min().alias("t_start"),
            pl.col(schema.time).max().alias("t_end")]
    aggs.append(pl.col(schema.split_flag).max().alias("flag") if have_flag
                else pl.lit(False).alias("flag"))

    log.info("scanning subject index from %s (%s)", path, memory_note())
    raw = (lf.group_by(keys).agg(aggs)
             .collect(engine="streaming")
             .rename({schema.source: "source", schema.patient: "patient",
                      schema.is_test: "is_test"})
             .to_pandas())
    if raw.empty:
        raise ValueError(f"{path} contains no rows.")
    raw["is_test"] = raw["is_test"].astype(bool)
    raw["flag"] = raw["flag"].astype(bool)
    log.info("index scan: %d (subject, is_test) groups in %.1fs (%s)",
             len(raw), time.time() - t0, memory_note())
    rows = []
    for (src, pid), g in raw.groupby(["source", "patient"], observed=True):
        tr, te = g[~g.is_test], g[g.is_test]
        n_train, n_test = int(tr["rows"].sum()), int(te["rows"].sum())

        if n_test == 0:
            cls = "train_only"
        elif n_train == 0:
            cls = "test_only"
        elif tr["t_end"].max() <= te["t_start"].min():
            cls = "temporal"
        else:
            cls = "interleaved"

        rows.append({
            "key": subject_key(src, pid), "source": src, "id": pid,
            "rows": n_train + n_test, "n_train": n_train, "n_test": n_test,
            "t_start": g["t_start"].min(), "t_end": g["t_end"].max(),
            "test_start": (te["t_start"].min() if n_test else pd.NaT),
            "split_class": cls, "flag_says_split": bool(g["flag"].max()),
        })

    frame = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)

    n_all = len(frame)
    if mcfg.include_sources is not None:
        frame = frame[frame.source.isin(list(mcfg.include_sources))]
    if mcfg.exclude_sources:
        frame = frame[~frame.source.isin(list(mcfg.exclude_sources))]

    dropped = n_all - len(frame)
    log.info("subject index: %d subjects retained, %d dropped by source policy "
             "(excluded=%s), %s rows", len(frame), dropped,
             list(mcfg.exclude_sources) or "none", f"{int(frame['rows'].sum()):,}")
    for cls, n in frame.split_class.value_counts().items():
        log.info("  split_class %-12s %5d subjects", cls, n)
    return SubjectIndex(frame=frame.reset_index(drop=True))


def assert_no_benchmark_contamination(
    subjects: Iterable[str] | SubjectIndex,
    allow: Sequence[str] = (),
) -> None:
    """Refuse a subject set drawn from this repository's own benchmark cohorts.

    MetaboNet's OhioT1DM slice includes all twelve subjects' OFFICIAL TEST
    spans, every row of it labelled `is_test = False`. A model fitted on those
    rows and then reported on the Ohio benchmark has been trained on its test
    set, and no downstream metric can reveal that -- it looks like a good model.
    HUPA-UCM is present for the same reason and carries the same hazard for the
    multimodal arm.

    `allow` is the explicit opt-in, e.g. for a study that deliberately measures
    the size of the contamination. It has to be named per source, because the
    thing being waived is the guarantee that a number is comparable to the rest
    of the repository's tables.
    """
    keys = (subjects.keys() if isinstance(subjects, SubjectIndex)
            else [str(k) for k in subjects])
    allowed = set(allow)
    hit: dict[str, int] = {}
    for k in keys:
        src, _ = split_subject_key(k)
        if src in BENCHMARK_SOURCES and src not in allowed:
            hit[src] = hit.get(src, 0) + 1
    if hit:
        detail = ", ".join(f"{s} ({n} subjects)" for s, n in sorted(hit.items()))
        raise AssertionError(
            f"MetaboNet subject set contains this repository's benchmark "
            f"cohorts: {detail}. Those rows include the OhioT1DM official TEST "
            f"spans (31,734 CGM samples labelled is_test=False), so a model fit "
            f"on them cannot be reported against the Ohio or HUPA-UCM protocols. "
            f"Drop them (MetaboNetConfig.exclude_sources, the default) or waive "
            f"the check explicitly with allow={sorted(hit)}."
        )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _polars():
    """Import Polars with an error that says what to do.

    Required rather than optional: the whole ingestion path is built on lazy
    evaluation, and a pandas fallback would be a second implementation of the
    part of this module most likely to exhaust memory -- which is exactly the
    thing that must not have two versions.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MetaboNet ingestion needs Polars: pip install 'polars>=1.0' "
            "(it is in requirements.txt). The loader streams a 154M-row "
            "parquet through pl.scan_parquet; there is no eager fallback."
        ) from exc
    return pl


def _canonical_frame(pdf: pd.DataFrame, schema: MetaboNetSchema,
                     mcfg: MetaboNetConfig) -> pd.DataFrame:
    """Released names -> canonical names, composite key, plausibility screen."""
    d = pdf.rename(columns=schema.rename_map(pdf.columns))
    d["patient"] = [subject_key(s, p)
                    for s, p in zip(d["source"].astype(str),
                                    d["patient"].astype(str))]
    d["time"] = pd.to_datetime(d["time"])
    lo, hi = mcfg.cgm_range_mgdl
    g = pd.to_numeric(d["glucose"], errors="coerce")
    d["glucose"] = g.where((g >= lo) & (g <= hi))
    d["is_test"] = d["is_test"].astype(bool)
    return d.sort_values(["patient", "time"]).reset_index(drop=True)


def _row_bounded_batches(
    source: str,
    ids: Sequence[str],
    max_subjects: int,
    max_rows: int,
    subject_rows: dict[str, int] | None,
) -> Iterator[list[str]]:
    """Group one source's ids into batches bounded by subject count AND rows."""
    batch: list[str] = []
    rows = 0
    for sid in ids:
        n = int(subject_rows.get(subject_key(source, sid), 0)) if subject_rows else 0
        if batch and (len(batch) >= max_subjects
                      or (max_rows and rows + n > max_rows)):
            yield batch
            batch, rows = [], 0
        batch.append(sid)
        rows += n
    if batch:
        yield batch


def iter_subject_frames(
    path: str,
    keys: Sequence[str],
    schema: MetaboNetSchema | None = None,
    mcfg: MetaboNetConfig | None = None,
    batch_size: int | None = None,
    subject_rows: dict[str, int] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield `(subject_key, canonical frame)` one subject at a time.

    This is the memory contract of the whole loader: at most `batch_size`
    subjects' rows are resident at once, never the corpus. The released file is
    154,842,077 rows across 37 columns -- roughly 24 GB as a pandas frame -- so
    a function that returns all of it cannot run on a normal machine, and the
    process is killed by the OS rather than raising something a caller could
    catch.

    Two properties of the released file make the streaming exact rather than
    approximate:

      * Predicate pushdown reaches the row-group statistics, so filtering to a
        batch of subjects reads only the row groups that contain them. A single
        subject collects in well under a second against a 1.26 GB file.
      * Every subject's rows are contiguous (verified: 0 of 1,907 subjects span
        a discontinuous run of row groups, and none spans more than two), so a
        batch touches a bounded slice of the file rather than all of it.

    Subjects are yielded in the order given, and each frame is complete: a
    caller can window it and drop it before the next one is read.

    A batch closes at `ingest_batch_subjects` subjects OR `ingest_batch_rows`
    rows, whichever comes first, so the resident frame is bounded in the unit
    that costs memory. Without the row bound, a batch of 32 Loop subjects is
    ~4M rows where a batch of 32 AZT1D ones is ~430k.
    """
    pl = _polars()
    schema = schema or MetaboNetSchema()
    mcfg = mcfg or MetaboNetConfig()
    batch_size = batch_size or mcfg.ingest_batch_subjects

    lf = pl.scan_parquet(path)
    available = list(lf.collect_schema().names())
    schema.validate(available, path)
    cols = [c for c in schema.all_columns() if c in available]

    keys = [str(k) for k in keys]
    if not keys:
        raise ValueError(
            "No MetaboNet subjects selected. With the default "
            "MetaboNetConfig.exclude_sources this can only happen if the file "
            "contains nothing but the excluded benchmark sources.")

    # Batch by SOURCE so each collect's filter is a cheap (source, id-set) pair
    # rather than a disjunction over composite strings, which polars cannot push
    # into the row-group statistics.
    by_source: dict[str, list[str]] = {}
    for k in keys:
        src, sid = split_subject_key(k)
        by_source.setdefault(src, []).append(sid)

    seen = 0
    for src in sorted(by_source):
        for chunk in _row_bounded_batches(src, by_source[src], batch_size,
                                          mcfg.ingest_batch_rows, subject_rows):
            t0 = time.time()
            batch = (lf.filter((pl.col(schema.source) == src)
                               & (pl.col(schema.patient).is_in(chunk)))
                       .select(cols)
                       .collect(engine="streaming")
                       .to_pandas())
            log.debug("read %s: %d subjects, %s rows in %.2fs (%s)", src,
                      len(chunk), f"{len(batch):,}", time.time() - t0,
                      memory_note())
            if batch.empty:
                continue
            frame = _canonical_frame(batch, schema, mcfg)
            del batch
            for key, pdf in frame.groupby("patient", sort=True):
                seen += 1
                yield str(key), pdf.reset_index(drop=True)
            del frame

    if seen == 0:
        raise ValueError(f"No rows in {path} matched the selected subjects.")


def load_metabonet_frame(
    path: str,
    keys: Sequence[str] | None = None,
    schema: MetaboNetSchema | None = None,
    mcfg: MetaboNetConfig | None = None,
    index: SubjectIndex | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Materialise the selected subjects as ONE canonical long frame.

    Convenience over `iter_subject_frames`, for tests and for small selections.
    It is deliberately NOT how the corpus is loaded: the released file is
    154,842,077 rows over 37 columns, roughly 24 GB in pandas, and asking for
    all of it does not raise -- the kernel kills the process. Use
    `build_metabonet_windows_streaming`, which never holds more than one
    subject.

    `max_rows` (default `mcfg.max_eager_rows`) is the guard. It is checked from
    the SUBJECT INDEX before a single row is read, so an accidental
    whole-corpus call fails in half a second with the row count and the
    alternative, instead of after ten minutes of swapping.

    Rows whose CGM is outside `mcfg.cgm_range_mgdl` have their glucose set to
    NaN rather than being dropped: dropping the row would silently close a gap
    in the time series, which is how an implausible reading turns into an
    interpolated plausible one.
    """
    schema = schema or MetaboNetSchema()
    mcfg = mcfg or MetaboNetConfig()
    cap = mcfg.max_eager_rows if max_rows is None else max_rows

    idx = index
    if keys is None or cap:
        idx = idx if idx is not None else scan_subjects(path, schema, mcfg)
        if keys is None:
            keys = idx.keys()
        if cap:
            wanted = set(map(str, keys))
            n_rows = int(idx.frame.loc[idx.frame.key.isin(wanted), "rows"].sum())
            if n_rows > cap:
                raise MemoryError(
                    f"load_metabonet_frame would materialise {n_rows:,} rows "
                    f"for {len(wanted)} subjects, over the {cap:,}-row eager "
                    f"limit. This function holds every row in memory at once; "
                    f"at ~160 B/row that is ~{n_rows * 160 / 2**30:.1f} GB and "
                    f"the process would be killed rather than raise.\n"
                    f"  Stream instead: build_metabonet_windows_streaming(...), "
                    f"or iter_subject_frames(...) for one subject at a time.\n"
                    f"  To override deliberately, raise "
                    f"MetaboNetConfig.max_eager_rows or pass max_rows=0.")

    t0 = time.time()
    # Row counts, when the index is already in hand, so batches are bounded in
    # the unit that costs memory rather than by subject count.
    rows_by_key = (dict(zip(idx.frame.key, idx.frame.rows.astype(int)))
                   if idx is not None else None)
    parts = [pdf for _key, pdf in
             iter_subject_frames(path, keys, schema, mcfg,
                                 subject_rows=rows_by_key)]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    log.info("loaded %s rows for %d subjects in %.1fs (%s)",
             f"{len(df):,}", len(parts), time.time() - t0, memory_note())
    return df.sort_values(["patient", "time"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synchronisation
# --------------------------------------------------------------------------- #
def _subject_test_boundary(
    pdf: pd.DataFrame, mcfg: MetaboNetConfig
) -> tuple[pd.Timestamp | None, str]:
    """The instant a subject's official test region begins, and its class.

    Returns (boundary, class). `boundary` is None for a subject living entirely
    on one side. For an interleaved subject the boundary is undefined, and the
    class is returned so the caller can apply `on_interleaved` rather than
    inventing one.
    """
    te = pdf.loc[pdf["is_test"], "time"]
    tr = pdf.loc[~pdf["is_test"], "time"]
    if te.empty:
        return None, "train_only"
    if tr.empty:
        return None, "test_only"
    if tr.max() <= te.min():
        return te.min(), "temporal"
    return None, "interleaved"


def synchronize_subject(
    pdf: pd.DataFrame,
    cfg: DataConfig,
    mcfg: MetaboNetConfig | None = None,
) -> pd.DataFrame:
    """Put one subject's record on the shared CGM grid, causally.

    Every column produced here obeys the invariant the rest of the pipeline
    rests on: the value at `t` is a function of samples at times <= `t`. The
    grid is defined by the CGM stream because CGM is where the targets come
    from, and re-gridding it would change the prediction problem.

    Stream-by-stream rules, each chosen for the stream's SEMANTICS -- one rule
    applied to all of them is the fastest way to leak:

        CGM                instantaneous     snap to last sample in (t-P, t]
        heart rate         instantaneous     mean over (t-P, t], bounded ffill
        steps / calories   interval count    SUM over (t-P, t], never filled
        gsr / skin / air   instantaneous     mean over (t-P, t], bounded ffill
        bolus / basal      impulse (units)   sum into (t-P, t] -> causal IOB
        carbs              impulse (grams)   sum into (t-P, t] -> causal COB
        workout intensity  state             held over its logged duration

    `sleep` has no MetaboNet column; it stays zero with its mask clear.
    """
    mcfg = mcfg or MetaboNetConfig()
    hcfg = mcfg.hupa
    P = cfg.sample_period_min

    d = pdf.sort_values("time").copy()
    d["time"] = pd.to_datetime(d["time"])
    if d.empty:
        return pd.DataFrame()

    grid = pd.date_range(d["time"].iloc[0].ceil(f"{P}min"),
                         d["time"].iloc[-1].floor(f"{P}min"),
                         freq=f"{P}min")
    if len(grid) < 2:
        return pd.DataFrame()

    out = pd.DataFrame({"time": grid})
    steps_grid = lambda m: max(0, int(round(m / P)))  # noqa: E731

    # ---- CGM: strictly backward snap into (t - P, t] -----------------------
    cgm = d.loc[d["glucose"].notna(), ["time", "glucose"]]
    binned, _ = snap_to_grid_backward(cgm["time"], cgm[["glucose"]], grid, P)
    out["glucose"] = binned["glucose"].to_numpy()
    out["glucose_observed"] = out["glucose"].notna().astype(np.float32)

    # ---- instantaneous wearable channels: mean, then bounded ffill ----------
    for col, chan, mask_col, budget in (
        ("heart_rate", "heart_rate", "hr_observed", hcfg.hr_max_staleness_min),
        ("gsr", "gsr", "gsr_observed", hcfg.hr_max_staleness_min),
        ("skin_temp", "skin_temp", "skin_temp_observed", hcfg.hr_max_staleness_min),
        ("air_temp", "air_temp", "air_temp_observed", hcfg.hr_max_staleness_min),
    ):
        if col in d.columns:
            v, obs = _bin_backward(d["time"], d[col], grid, P, "mean")
            v, valid, stale = _causal_ffill(v, obs, steps_grid(budget))
        else:
            v = np.zeros(len(grid))
            valid = np.zeros(len(grid), bool)
            stale = np.full(len(grid), np.inf)
        out[chan] = v
        out[mask_col] = valid.astype(np.float32)
        if chan == "heart_rate":
            out["hr_staleness"] = np.clip(
                np.where(np.isfinite(stale), stale * P, hcfg.staleness_cap_min),
                0, hcfg.staleness_cap_min)

    # ---- interval counts: SUM, never filled ---------------------------------
    # Forward-filling a step or calorie count would invent movement that never
    # happened, so an unobserved bin is zero with the mask cleared.
    for col, chan, mask_col in (("steps", "steps", "steps_observed"),
                                ("calories", "calories", "calories_observed")):
        if col in d.columns:
            v, obs = _bin_backward(d["time"], d[col], grid, P, "sum")
        else:
            v, obs = np.zeros(len(grid)), np.zeros(len(grid), bool)
        out[chan] = v
        out[mask_col] = obs.astype(np.float32)

    # ---- sleep: absent from MetaboNet ---------------------------------------
    out["sleep"] = np.zeros(len(grid))
    out["sleep_observed"] = np.zeros(len(grid), dtype=np.float32)

    # ---- treatment impulses -> on-board curves ------------------------------
    # `basal` is units DELIVERED in the bin, not U/hr (verified against the
    # OhioT1DM XML: 0.0375 here == 0.45 U/hr there, and 0.0375 * 12 == 0.45).
    # So it is an impulse train like `bolus`, and it goes through the same
    # causal FIR filter. `insulin` is deliberately not read: it is exactly
    # basal + bolus, and ingesting it too would double count.
    iob_kernel = _activity_kernel(hcfg.iob_dia_min, hcfg.iob_peak_min, P, "iob")
    cob_kernel = _activity_kernel(hcfg.cob_dur_min, hcfg.cob_peak_min, P, "cob")

    for col, chan in (("bolus", "bolus"), ("carbs", "carbs"), ("basal", "basal_dose")):
        if col in d.columns:
            imp, _ = _bin_backward(d["time"], d[col].fillna(0.0), grid, P, "sum")
        else:
            imp = np.zeros(len(grid))
        out[chan] = imp

    out["iob"] = causal_on_board(out["bolus"].to_numpy(), iob_kernel)
    out["cob"] = causal_on_board(out["carbs"].to_numpy(), cob_kernel)
    out["basal_iob"] = causal_on_board(out["basal_dose"].to_numpy(), iob_kernel)
    out["basal_rate"] = out["basal_dose"].to_numpy() * (MINUTES_PER_HOUR / P)

    # ---- workout: a state, held over its own logged duration ----------------
    wi = np.zeros(len(grid))
    wo = np.zeros(len(grid), dtype=bool)
    if "workout_intensity" in d.columns:
        ev = d.loc[d["workout_intensity"].notna(), ["time", "workout_intensity"]]
        if "workout_duration" in d.columns:
            ev = ev.join(d["workout_duration"])
        gv = pd.DatetimeIndex(grid).values
        for _, row in ev.iterrows():
            dur = float(row.get("workout_duration", np.nan))
            dur = dur if np.isfinite(dur) and dur > 0 else P
            # The bout occupies (start, start + duration]; a slot is inside it
            # only once the slot's own instant has been reached, so nothing here
            # tells the model about a workout that has not started yet.
            start = np.datetime64(pd.Timestamp(row["time"]))
            end = start + np.timedelta64(int(round(dur)), "m")
            sel = (gv > start) & (gv <= end)
            wi[sel] = float(row["workout_intensity"])
            wo[sel] = True
    out["workout_intensity"] = wi
    out["workout_observed"] = wo.astype(np.float32)

    # ---- split tag on the grid ----------------------------------------------
    # Derived from the subject's own boundary instant, so a grid slot is "test"
    # exactly when its instant is at or after the first test sample. Binning the
    # flag would smear the boundary by up to a period, in the direction that
    # puts test data on the train side.
    boundary, cls = _subject_test_boundary(d, mcfg)
    if cls == "test_only":
        out["split"] = "test"
    elif boundary is None:
        out["split"] = "train"
    else:
        out["split"] = np.where(pd.DatetimeIndex(grid) >= boundary, "test", "train")

    # ---- segmentation ------------------------------------------------------
    # A segment breaks on a CGM recording gap AND on the train/test boundary, so
    # no window can straddle the official split: the same structural guarantee
    # `ohio_loader` gets for free by building windows per XML file.
    max_gap_steps = max(1, cfg.max_series_gap_min // P)
    miss = out["glucose_observed"].to_numpy() == 0.0
    tag = out["split"].to_numpy()
    seg = np.zeros(len(out), np.int64)
    run, sid = 0, 0
    for i, m in enumerate(miss):
        if i > 0 and tag[i] != tag[i - 1]:
            sid += 1
            run = 0
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
class MetaboNetWindowSet(MMWindowSet):
    """`MMWindowSet` plus MetaboNet's split tag, provenance and multi-horizon
    targets.

    Subclassing keeps `x`, `x_aux`, `y_reg`, `y_cls_hard`, `y_min`, `patient`,
    `last_bg` and `aux_frac` under the names the rest of the codebase already
    reads, so `train.run_fold`, `CausalScaler` and the evaluation code work
    unchanged. `patient` holds the COMPOSITE subject key; `source` carries the
    originating dataset so provenance survives into every artefact.

    `y_reg` / `y_cls_hard` / `y_min` mirror the primary horizon
    (`cfg.horizon_min`); the `*_multi` arrays carry every horizon in
    `horizons_min`, column k corresponding to `horizons_min[k]`.
    """
    split: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    source: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    y_reg_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    y_min_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    y_cls_multi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    horizons_min: tuple[int, ...] = ()
    aux_channels: tuple[str, ...] = METABONET_AUX_CHANNELS
    excluded_sources: tuple[str, ...] = ()

    def subset(self, idx: np.ndarray) -> "MetaboNetWindowSet":
        """Index every aligned array; the metadata tuples are not axes."""
        meta = ("horizons_min", "aux_channels", "excluded_sources")
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v if f.name in meta else v[idx]
        return MetaboNetWindowSet(**kw)

    def where(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def subjects(self) -> list[str]:
        return sorted({str(p) for p in self.patient})


def native_cadence_min(pdf: pd.DataFrame, time_col: str = "time") -> int:
    """The subject's own median sampling interval, in whole minutes; 0 if unknown.

    The grid is built at `cfg.sample_period_min` whatever a study actually
    sampled at, and the corpus consolidates fourteen of them. A 15-minute study
    lands two-thirds unobserved on a 5-minute grid: it clears `min_cgm_frac`,
    survives segmentation, and is then annihilated window by window, because
    filling the 2-step gaps puts `himp.mean()` at ~0.67 against a
    `max_imputed_frac` of 0.20. Every window is dropped and the subject is
    recorded as barren -- correct, but indistinguishable from "this subject had
    no usable stretches".
    """
    if time_col not in pdf.columns or len(pdf) < 3:
        return 0
    t = pd.to_datetime(pdf[time_col]).sort_values().to_numpy()
    dt = np.diff(t).astype("timedelta64[s]").astype(np.int64)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0
    return int(round(float(np.median(dt)) / 60.0))


def source_contributions(requested: Sequence[str], contributed: Sequence[str],
                         barren: Sequence[str]) -> dict:
    """Per-source ledger over composite subject keys, as `<source>:<id>`.

    `requested` is what the index offered; `contributed` produced windows;
    `barren` produced none and said so. Anything left is `unaccounted` -- never
    reached, because the pass was interrupted or a filter removed it earlier.
    A study that contributes nothing then shows up as a ZERO rather than as an
    absence, which is how two of the corpus's studies went unnoticed: they
    appeared in neither the contributing list nor the barren one, having never
    produced a shard to be recorded in either.
    """
    def src(key: str) -> str:
        return split_subject_key(str(key))[0]

    ledger: dict[str, dict] = {}
    for key in requested:
        ledger.setdefault(src(key), {"requested": 0, "contributed": 0,
                                     "barren": 0, "unaccounted": 0})
        ledger[src(key)]["requested"] += 1
    for key in contributed:
        ledger.setdefault(src(key), {"requested": 0, "contributed": 0,
                                     "barren": 0, "unaccounted": 0})
        ledger[src(key)]["contributed"] += 1
    for key in barren:
        ledger.setdefault(src(key), {"requested": 0, "contributed": 0,
                                     "barren": 0, "unaccounted": 0})
        ledger[src(key)]["barren"] += 1
    for stats in ledger.values():
        stats["unaccounted"] = max(
            0, stats["requested"] - stats["contributed"] - stats["barren"])
    return dict(sorted(ledger.items()))


def _log_source_ledger(requested: Sequence[str], store) -> dict:
    """Log the per-source ledger, and flag any study that produced nothing.

    A source with zero contributions appears in neither the shard list nor the
    barren list -- it never produced a shard to be recorded in either -- so it
    is invisible to any count taken from the finished window set. That is how
    two of the corpus's studies went unnoticed.
    """
    ledger = source_contributions(
        requested, [e["key"] for e in store.entries], store.barren)
    log.info("per-source contribution "
             "(requested / windowed / barren / never reached):")
    for source, stats in ledger.items():
        if stats["contributed"] == 0:
            note = "   <-- CONTRIBUTED NOTHING"
        elif stats["barren"] or stats["unaccounted"]:
            note = "   <-- partial"
        else:
            note = ""
        log.info("  %-14s %4d / %4d / %4d / %4d%s", source, stats["requested"],
                 stats["contributed"], stats["barren"], stats["unaccounted"],
                 note)
    empty = [s for s, v in ledger.items() if v["contributed"] == 0]
    if empty:
        log.warning(
            "%d source(s) contributed no windows at all: %s. A source count "
            "taken from the finished window set will not mention them.",
            len(empty), ", ".join(empty))
    return ledger


def _thin_blocks(n: int, cap: int, block: int) -> np.ndarray:
    """Indices of at most `cap` windows, kept as CONTIGUOUS BLOCKS in time.

    Thinning is unavoidable at this scale, but HOW it thins decides whether the
    result is still measurable. Keeping every k-th window bounds memory and
    quietly destroys two things that depend on the anchor cadence:

      * `clinical.segment_ids` starts a new recording segment whenever the gap
        between consecutive anchors exceeds one sampling period. Every-k-th
        thinning makes every window its own segment, so alarm episodes are all
        length 1, false-alarms-per-day is meaningless and lead time -- which
        needs anchors strictly before a nadir, inside one segment -- is `nan`.
        The metrics do not fail; they come back empty.
      * `extension.prefix_embargo_split` takes a wall-clock prefix of the
        subject's record. Windows spread thinly across weeks put almost nothing
        inside the first K hours, so no subject clears `adapt_patient`'s minimum
        and the personalisation arm silently adapts nobody.

    Keeping whole blocks preserves the native cadence INSIDE each block, so both
    survive: a block is a stretch of real recording, and the gaps between blocks
    read as exactly what they are -- separate segments, which every metric here
    already handles correctly. Blocks are spread evenly across the record rather
    than taken from its start, so the sample still spans the subject's weeks.

    ONE BLOCK IS NOT A SAMPLE
    -------------------------
    A single block is a truncation wearing this function's name: it is "the
    first `cap` windows" however it was arrived at. So whenever the budget can
    pay for two blocks this takes two, even when that makes each one shorter
    than `block` -- straddling the record matters more than the block length,
    and half of a 400 budget is still 16 h of contiguous recording. The shipped
    default sat exactly in that trap: `cap=400` with `block=288` rounded to one
    block at index 0, so every MetaboNet arm was fitted and scored on the
    opening ~33 h of a multi-week record. Nothing showed it, because the cadence
    INSIDE the block was correct -- segment ids, alarm episodes and lead time
    all came back healthy, just from the wrong 0.4% of each subject.
    """
    if not cap or n <= cap:
        return np.arange(n)
    want = max(1, min(int(block or cap), cap))
    # How many blocks of at least `want` the budget buys, and then how long each
    # may actually be. `cap // want` rather than `round(cap / want)`: rounding
    # 400/288 down to a single block is the degeneracy above. The budget is
    # still spent in full either way -- `block_eff` divides `cap` by the block
    # count, so blocks grow past `want` instead of leaving a remainder unspent
    # (a 400 cap with a 288-window block once kept 288, which is how the
    # personalisation prefix ended up 112 windows short of what it promised).
    n_blocks = max(1, cap // want)
    # ... and never just one, while the cap can pay for a useful pair. A cap too
    # small even for two half-blocks is a degenerate configuration in which one
    # block genuinely is all there is.
    if n_blocks == 1 and cap >= 2 * max(1, want // 2):
        n_blocks = 2
    block_eff = max(1, cap // n_blocks)
    # Blocks are placed by their START OFFSET -- the first flush with the
    # beginning of the record, the last flush with its end, evenly spaced in
    # between -- rather than on a fixed grid of `block_eff`. On a grid the final
    # block is partial whenever `n` is not a multiple of the block length, and
    # the budget goes unspent exactly as it did before version 2: a 400 budget
    # over a 417-window record kept 217. `n > cap >= n_blocks * block_eff`
    # bounds the spacing below by `block_eff`, so the blocks cannot overlap and
    # every one of them is full length.
    starts = np.unique(np.linspace(0, n - block_eff, n_blocks).round().astype(int))
    idx = np.concatenate([np.arange(b, b + block_eff) for b in starts])
    return idx[:cap]


def _subject_windows(
    pid: str,
    pdf: pd.DataFrame,
    cfg: DataConfig,
    mcfg: MetaboNetConfig,
    hz: tuple[int, ...],
) -> dict | None:
    """Every valid window for ONE subject, or None if it contributes none.

    This is the single definition of the window contract. Both builders -- the
    in-memory one and the streaming one -- call it, so a guarantee asserted
    against one holds for the other. For anchor `t`:

        x[n]     <- CGM features over [t-H+1, t]
        x_aux[n] <- aux features over the SAME closed interval
        targets  <- glucose over [t+1, t+PH]

    The two index ranges are disjoint by construction and the targets are
    written to arrays no feature code touches, so there is no path from `y_*`
    back into `x`. History gaps are filled per window by `_causal_gap_fill`
    (both endpoints of any interpolated run lie inside `[t-H+1, t]`); targets
    are never imputed -- a window with any unobserved target step is dropped,
    because imputing a label is leakage by definition.

    A window is emitted only if its history and every target step lie inside one
    segment, and segments break at the train/test boundary, so no window can
    straddle the official split.

    `mcfg.max_windows_per_subject` thins ANCHORS, in contiguous blocks
    (`_thin_blocks`), after validity is decided. Thinning cannot make an invalid
    window valid, and every window kept is one the full-stride pass would also
    have emitted -- so the cap changes how many windows a subject contributes,
    never what a window means.

    Two passes rather than one, for a memory reason and not a semantic one: the
    first decides which anchors are valid, the second cuts tensors for only the
    anchors the cap keeps. Building every window and then keeping `cap` of them
    peaked at roughly 1.2 GB on the corpus's longest subject (~126k windows, a
    (24, 24) float64 aux slice each, plus a full copy in `np.stack`) to retain
    400 windows -- about a megabyte of output. Peak is now O(cap), and
    `--memory-budget-gb` still cannot see inside a subject, so this is the only
    place that bound can be enforced.
    """
    hcfg = mcfg.hupa
    hz_steps = [h // cfg.sample_period_min for h in hz]
    max_hz = max(hz_steps)
    H = cfg.history_steps
    max_gap_steps = max(0, cfg.max_interpolation_gap_min // cfg.sample_period_min)
    min_rows = mcfg.min_rows_per_subject or (H + max_hz)

    if len(pdf) < min_rows:
        return None

    # A cadence coarser than the grid guarantees every window exceeds
    # `max_imputed_frac`, so the subject contributes nothing. Say so, once, with
    # the numbers -- otherwise it is indistinguishable from a subject that
    # simply had no usable stretches. See `native_cadence_min`.
    cadence = native_cadence_min(pdf)
    if cadence > cfg.sample_period_min:
        log.warning(
            "subject %s samples every ~%d min against a %d-min grid: after "
            "short-gap filling every window will exceed max_imputed_frac=%.2f, "
            "so this subject will contribute NO windows. Its study needs its "
            "own sample_period_min, or to be excluded explicitly.",
            pid, cadence, cfg.sample_period_min, cfg.max_imputed_frac)

    _, cls = _subject_test_boundary(pdf, mcfg)
    if cls == "interleaved":
        if mcfg.on_interleaved == "raise":
            raise ValueError(
                f"Subject {pid} has interleaved train/test rows: its own "
                f"official split overlaps in time, so no temporal holdout "
                f"of it is valid. Set MetaboNetConfig.on_interleaved to "
                f"'drop' or 'train'.")
        if mcfg.on_interleaved == "drop":
            return None
        pdf = pdf.assign(is_test=False)

    reg = synchronize_subject(pdf, cfg, mcfg)
    if reg.empty:
        return None
    if float(reg["glucose_observed"].mean()) < mcfg.min_cgm_frac:
        return None

    obs_cols = [METABONET_AUX_INDEX[c] for c in METABONET_AUX_CHANNELS
                if c.endswith("_observed")]
    g = reg["glucose"].to_numpy(dtype=np.float64)
    obs = reg["glucose_observed"].to_numpy().astype(bool)
    aux = np.stack([reg[c].to_numpy(dtype=np.float64)
                    for c in METABONET_AUX_CHANNELS], axis=-1)
    seg = reg["segment"].to_numpy()
    ts = reg["time"].to_numpy()
    tag = reg["split"].to_numpy()
    aux_obs_step = aux[:, obs_cols].mean(axis=1)
    source, _ = split_subject_key(pid)

    # ---- pass 1: WHICH anchors are valid, without cutting one tensor -------
    # Deciding validity costs a handful of H-length temporaries per anchor, all
    # of them transient. Materialising the window costs an (H, C_aux) float64
    # slice that would have to be HELD until the cap is applied -- see the
    # docstring for what that cost on the corpus's longest subject.
    seg_rows: list[np.ndarray] = []      # rows of each contributing segment
    seg_anchors: list[np.ndarray] = []   # its valid anchor positions, ascending

    for _, sidx in pd.Series(np.arange(len(reg))).groupby(seg):
        si = sidx.values
        if si.size < H + max_hz:
            continue
        gs, os_, aos = g[si], obs[si], aux_obs_step[si]

        valid: list[int] = []
        for a in range(H - 1, si.size - max_hz, cfg.stride_steps):
            hs = slice(a - H + 1, a + 1)
            tgt = slice(a + 1, a + 1 + max_hz)

            # --- targets: strictly no imputation ------------------------
            if not os_[a] or not os_[tgt].all():
                continue

            # --- history: causal short-gap fill only --------------------
            _, himp = _causal_gap_fill(gs[hs], os_[hs], max_gap_steps)
            if not (os_[hs] | himp).all():
                continue
            if himp.mean() > cfg.max_imputed_frac:
                continue

            frac = float(aos[hs].mean())
            if frac and frac < hcfg.min_aux_observed_frac:
                # A window that is almost all mask carries no multimodal
                # information. A subject with NO aux at all (frac == 0) is
                # kept: that is the unimodal degradation path, not a
                # coverage failure.
                continue

            valid.append(a)

        if valid:
            seg_rows.append(si)
            seg_anchors.append(np.asarray(valid, dtype=np.int64))

    if not seg_anchors:
        return None

    # ---- the cap, decided on anchors and paid before any tensor exists -----
    n_valid = int(sum(a.size for a in seg_anchors))
    cap = int(mcfg.max_windows_per_subject or 0)
    keep = _thin_blocks(n_valid, cap, int(mcfg.thin_block_windows))
    subsample_k = int(round(n_valid / max(len(keep), 1)))
    # Flat window index -> segment. Segments are visited below in the order they
    # were visited above, and anchors ascend within each, so a position in
    # `keep` means what it meant when the whole set was enumerated.
    bounds = np.concatenate([[0], np.cumsum([a.size for a in seg_anchors])])

    # ---- pass 2: cut tensors for the kept anchors ONLY ---------------------
    xs, xs_obs, xa, tix, tts, lbg, afrac, tags = ([] for _ in range(8))
    yr_multi, ym_multi = [], []

    for s_i, (si, anchors) in enumerate(zip(seg_rows, seg_anchors)):
        lo, hi = np.searchsorted(keep, [bounds[s_i], bounds[s_i + 1]])
        if lo == hi:
            continue
        gs, os_, tss = g[si], obs[si], ts[si]
        aus, aos, tgs = aux[si], aux_obs_step[si], tag[si]

        for a in anchors[keep[lo:hi] - bounds[s_i]]:
            a = int(a)
            hs = slice(a - H + 1, a + 1)
            tgt = slice(a + 1, a + 1 + max_hz)
            fut = gs[tgt]
            hv, _ = _causal_gap_fill(gs[hs], os_[hs], max_gap_steps)

            xs.append(hv)
            xs_obs.append(os_[hs].astype(np.float64))
            xa.append(aus[hs])
            yr_multi.append([fut[k - 1] for k in hz_steps])
            ym_multi.append([fut[:k].min() if cfg.classify_over_window
                             else fut[k - 1] for k in hz_steps])
            tix.append(a)
            tts.append(tss[a])
            lbg.append(hv[-1])
            afrac.append(float(aos[hs].mean()))
            tags.append(tgs[a])

    y_reg_multi = np.asarray(yr_multi, dtype=np.float32)
    y_min_multi = np.asarray(ym_multi, dtype=np.float32)
    n = int(y_reg_multi.shape[0])

    return {
        "n": n,
        "subsample_k": subsample_k,
        "n_before_cap": n_valid,
        "x": causal_features_batch(np.stack(xs), np.stack(xs_obs)),
        "x_aux": np.stack(xa).astype(np.float32),
        "y_reg_multi": y_reg_multi,
        "y_min_multi": y_min_multi,
        "y_cls_multi": (y_min_multi < cfg.hypo_threshold_mgdl).astype(np.float32),
        # One string OBJECT shared by every row, not n copies: an object array
        # of a repeated key costs 8 bytes a row, where a fixed-width unicode
        # array would cost the key's length in 4-byte code points.
        "patient": np.full(n, pid, dtype=object),
        "source": np.full(n, source, dtype=object),
        "t_index": np.asarray(tix, dtype=np.int64),
        "t_time": np.asarray(tts),
        "last_bg": np.asarray(lbg, dtype=np.float32),
        "aux_frac": np.asarray(afrac, dtype=np.float32),
        "split": np.asarray(tags, dtype=object),
    }


def _assemble(parts: Sequence[dict], cfg: DataConfig, mcfg: MetaboNetConfig,
              hz: tuple[int, ...]) -> MetaboNetWindowSet:
    """Concatenate per-subject window dicts into one window set."""
    cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
    y_reg_multi = cat("y_reg_multi")
    y_min_multi = cat("y_min_multi")
    y_cls_multi = cat("y_cls_multi")
    primary = hz.index(cfg.horizon_min) if cfg.horizon_min in hz else 0
    return MetaboNetWindowSet(
        x=cat("x"), x_aux=cat("x_aux"),
        y_reg=y_reg_multi[:, primary],
        y_cls_hard=y_cls_multi[:, primary],
        y_min=y_min_multi[:, primary],
        patient=cat("patient"), t_index=cat("t_index"), t_time=cat("t_time"),
        last_bg=cat("last_bg"), aux_frac=cat("aux_frac"),
        split=cat("split"), source=cat("source"),
        y_reg_multi=y_reg_multi, y_min_multi=y_min_multi,
        y_cls_multi=y_cls_multi, horizons_min=hz,
        aux_channels=METABONET_AUX_CHANNELS,
        excluded_sources=tuple(mcfg.exclude_sources),
    )


NO_WINDOWS_MESSAGE = (
    "No valid MetaboNet windows. Check the horizon against the subjects' "
    "record lengths, cfg.max_series_gap_min, and MetaboNetConfig.min_cgm_frac "
    "(CGM coverage runs from 33% to 100% depending on source_file).")


def build_metabonet_windows(
    df: pd.DataFrame,
    cfg: DataConfig,
    mcfg: MetaboNetConfig | None = None,
    horizons_min: Sequence[int] | None = None,
) -> MetaboNetWindowSet:
    """Windows for an already-materialised frame.

    Holds every subject's windows in memory at once, so it is for a selection
    that fits -- tests, a handful of subjects, an interactive look. The corpus
    goes through `build_metabonet_windows_streaming`, which shares this
    function's per-subject core (`_subject_windows`) and differs only in where
    the results are put.
    """
    mcfg = mcfg or MetaboNetConfig()
    hz = tuple(int(h) for h in sorted(set(horizons_min or (cfg.horizon_min,))))

    parts = []
    for pid, pdf in df.groupby("patient", sort=True):
        part = _subject_windows(str(pid), pdf, cfg, mcfg, hz)
        if part is not None:
            parts.append(part)

    if not parts:
        raise ValueError(NO_WINDOWS_MESSAGE)
    return _assemble(parts, cfg, mcfg, hz)


# --------------------------------------------------------------------------- #
# Disk-backed window store
# --------------------------------------------------------------------------- #
# Arrays large enough to be worth keeping out of RAM. Everything else (targets,
# timestamps, the object arrays of repeated subject keys) is a few tens of bytes
# a window and stays resident.
MEMMAP_ARRAYS: tuple[str, ...] = ("x", "x_aux")

# Arrays that are concatenated in memory, in the order `_assemble` expects.
SMALL_ARRAYS: tuple[str, ...] = (
    "y_reg_multi", "y_min_multi", "y_cls_multi", "patient", "source",
    "t_index", "t_time", "last_bg", "aux_frac", "split",
)


def window_bytes(cfg: DataConfig, n_aux: int, n_horizons: int) -> int:
    """Bytes one window costs, so a run can be priced before it is started."""
    H = cfg.history_steps
    return (H * 5 * 4          # x        (value, diff, roll mean, roll std, mask)
            + H * n_aux * 4    # x_aux
            + n_horizons * 3 * 4   # y_reg / y_min / y_cls, per horizon
            + 8 + 4 + 4 + 8    # t_time, last_bg, aux_frac, t_index
            + 3 * 8)           # patient / source / split object pointers


def _fingerprint(cfg: DataConfig, mcfg: MetaboNetConfig,
                 hz: tuple[int, ...]) -> str:
    """Identity of the windows a store holds.

    Everything that changes what a window CONTAINS goes in. A cache whose
    fingerprint differs is rebuilt rather than reused -- the same reason
    `WINDOW_PIPELINE_VERSION` gates the OhioT1DM pickle: silently reusing
    windows built under different geometry is how a reader reproduces numbers
    from a pipeline that no longer exists.
    """
    payload = {
        "pipeline_version": WINDOW_PIPELINE_VERSION,
        "metabonet_window_version": METABONET_WINDOW_VERSION,
        "aux_channels": list(METABONET_AUX_CHANNELS),
        "horizons": list(hz),
        "data": {k: v for k, v in dataclasses.asdict(cfg).items()},
        "mcfg": {
            "exclude_sources": list(mcfg.exclude_sources),
            "include_sources": (list(mcfg.include_sources)
                                if mcfg.include_sources else None),
            "on_interleaved": mcfg.on_interleaved,
            "cgm_range_mgdl": list(mcfg.cgm_range_mgdl),
            "min_cgm_frac": mcfg.min_cgm_frac,
            "min_rows_per_subject": mcfg.min_rows_per_subject,
            "max_windows_per_subject": mcfg.max_windows_per_subject,
            "thin_block_windows": mcfg.thin_block_windows,
            "hupa": dataclasses.asdict(mcfg.hupa),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _pid_alive(pid: int) -> bool:
    """True when `pid` is a live process this user can see."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # alive, owned by somebody else
        return True
    except OSError:
        return False
    return True


class WindowStore:
    """Per-subject window shards on disk, finalised into memmapped arrays.

    Two jobs, and the second is why it exists rather than a list:

    1. BOUNDED MEMORY. A subject's windows are written and dropped, so peak RSS
       during a build is one subject, not the corpus. At the end the big arrays
       are memmapped, so `ws.x` is a file the OS pages in on demand and only the
       rows a fold actually touches are ever resident.

    2. RESUMABILITY. Each shard is recorded in a manifest as it lands. An
       interrupted run -- a laptop lid, an OOM kill on a later subject, a
       Ctrl-C twelve hours in -- resumes from the next subject instead of
       repeating the ones already paid for. The manifest carries a fingerprint
       of the window geometry, so a store built under a different history,
       horizon or channel layout is rebuilt rather than silently mixed with new
       shards.
    """

    def __init__(self, directory: str, fingerprint: str):
        self.dir = directory
        self.fingerprint = fingerprint
        self.shard_dir = os.path.join(directory, "shards")
        self.manifest_path = os.path.join(directory, "manifest.json")
        self.lock_path = os.path.join(directory, "LOCK")
        self.entries: list[dict] = []
        # Subjects that produced NO windows. Recorded so a resume does not read
        # and re-window them every time: on the released corpus 35 of 1,873
        # contribute nothing, and re-deriving that costs a fifth of a minute on
        # every restart for an answer that cannot change under a fingerprint
        # that already covers every input to the decision.
        self.barren: list[str] = []
        self._locked = False
        os.makedirs(self.shard_dir, exist_ok=True)

    # ---- exclusion ---------------------------------------------------------
    def acquire(self) -> None:
        """Refuse a second builder on the same store.

        Two builds sharing a cache directory interleave their shards and
        overwrite each other's manifest, and the result looks like a valid store
        -- it is the manifest of whichever process wrote last, over a shard set
        neither of them produced. The failure is invisible at read time, which
        is what makes it worth a lock rather than a convention.

        Advisory, and stale-tolerant: a lock whose process is gone (a kill, a
        crash, a reboot) is reclaimed with a warning, because the alternative is
        a cache that can never be resumed without manual cleanup.
        """
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path) as f:
                    holder = json.load(f)
                pid = int(holder.get("pid", -1))
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pid, holder = -1, {}
            if pid == os.getpid():
                pass                      # ours already; re-acquiring is a no-op
            elif pid > 0 and _pid_alive(pid):
                raise RuntimeError(
                    f"window cache {self.dir} is being built by process {pid} "
                    f"(started {holder.get('started', 'unknown')}). Two builders "
                    f"on one cache interleave their shards and overwrite each "
                    f"other's manifest.\n"
                    f"  Wait for it, point --window-cache elsewhere, or remove "
                    f"{self.lock_path} if that process is gone.")
            else:
                log.warning("reclaiming lock from process %s, which is no "
                            "longer running, in %s", pid, self.dir)
        with open(self.lock_path, "w") as f:
            json.dump({"pid": os.getpid(),
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        self._locked = True

    def release(self) -> None:
        if self._locked:
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
            self._locked = False

    def __enter__(self) -> "WindowStore":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ---- manifest ----------------------------------------------------------
    def load_manifest(self) -> bool:
        """True when a compatible manifest was adopted for resuming."""
        if not os.path.exists(self.manifest_path):
            return False
        try:
            with open(self.manifest_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("window cache manifest unreadable (%s); rebuilding", exc)
            return False
        if m.get("fingerprint") != self.fingerprint:
            log.warning("window cache was built for fingerprint %s, this run is "
                        "%s (history, horizon, channel layout or window policy "
                        "changed); rebuilding",
                        m.get("fingerprint"), self.fingerprint)
            return False
        kept = [e for e in m.get("entries", [])
                if os.path.exists(os.path.join(self.shard_dir, e["shard"]))]
        if len(kept) != len(m.get("entries", [])):
            log.warning("%d shard(s) named in the manifest are missing; they "
                        "will be rebuilt", len(m.get("entries", [])) - len(kept))
        self.entries = kept
        self.barren = [str(k) for k in m.get("barren", [])]
        return True

    def save_manifest(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"fingerprint": self.fingerprint,
                       "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "entries": self.entries,
                       "barren": self.barren}, f, indent=1)
        # Atomic: a manifest half-written by an interrupted run would strand the
        # cache in a state neither resumable nor obviously broken.
        os.replace(tmp, self.manifest_path)

    @property
    def done_keys(self) -> set[str]:
        """Subjects that need no further work -- windowed, or known barren."""
        return {e["key"] for e in self.entries} | set(self.barren)

    @property
    def n_windows(self) -> int:
        return sum(int(e["n"]) for e in self.entries)

    # ---- shards ------------------------------------------------------------
    def append(self, key: str, part: dict) -> None:
        name = f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
        arrays = {k: v for k, v in part.items()
                  if isinstance(v, np.ndarray)}
        tmp = os.path.join(self.shard_dir, name + ".tmp")
        # Through a handle, not a path: `np.savez` appends `.npz` to a filename
        # that does not already end in it, so passing `<name>.npz.tmp` writes
        # `<name>.npz.tmp.npz` and the rename below fails on a file that was
        # written correctly under a name nobody asked for.
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, os.path.join(self.shard_dir, name))
        self.entries.append({"key": key, "shard": name, "n": int(part["n"]),
                             "subsample_k": int(part["subsample_k"]),
                             "n_before_cap": int(part["n_before_cap"])})

    def read_shard(self, entry: dict) -> dict:
        with np.load(os.path.join(self.shard_dir, entry["shard"]),
                     allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    # ---- finalisation ------------------------------------------------------
    def finalize(self, cfg: DataConfig, mcfg: MetaboNetConfig,
                 hz: tuple[int, ...]) -> MetaboNetWindowSet:
        """Stitch the shards into one window set, big arrays memmapped."""
        if not self.entries:
            raise ValueError(NO_WINDOWS_MESSAGE)
        entries = sorted(self.entries, key=lambda e: e["key"])
        total = sum(int(e["n"]) for e in entries)

        first = self.read_shard(entries[0])
        array_dir = os.path.join(self.dir, "arrays")
        os.makedirs(array_dir, exist_ok=True)

        big: dict[str, np.ndarray] = {}
        for name in MEMMAP_ARRAYS:
            shape = (total,) + first[name].shape[1:]
            big[name] = np.lib.format.open_memmap(
                os.path.join(array_dir, f"{name}.npy"), mode="w+",
                dtype=first[name].dtype, shape=shape)
        small: dict[str, list] = {name: [] for name in SMALL_ARRAYS}

        t0, off = time.time(), 0
        for i, e in enumerate(entries):
            part = self.read_shard(e)
            n = int(e["n"])
            for name in MEMMAP_ARRAYS:
                big[name][off:off + n] = part[name]
            for name in SMALL_ARRAYS:
                small[name].append(part[name])
            off += n
            if (i + 1) % 200 == 0:
                log.info("  stitched %d/%d subjects (%s)", i + 1, len(entries),
                         memory_note())
            del part
        # Hand the caller READ-ONLY views. The handles above are open "w+" over
        # the persistent, resumable cache directory, so one accidental in-place
        # write anywhere downstream -- an `out[...] =` that lost its `.copy()`,
        # a stray `np.clip(..., out=ws.x)` -- would rewrite the STORE, and every
        # later resume would read the corrupted arrays under a matching
        # fingerprint and no warning. Closing them (dropping the last reference
        # closes the mapping) and reopening `mmap_mode="r"` turns that class of
        # accident into a ValueError at the point of the write. A caller that
        # genuinely needs to mutate copies first, as they all already do.
        for name in MEMMAP_ARRAYS:
            big[name].flush()
        del big
        big = {name: np.load(os.path.join(array_dir, f"{name}.npy"),
                             mmap_mode="r")
               for name in MEMMAP_ARRAYS}

        merged = {name: np.concatenate(parts) for name, parts in small.items()}
        primary = hz.index(cfg.horizon_min) if cfg.horizon_min in hz else 0
        log.info("finalised %s windows from %d subjects in %.1fs; %s memmapped "
                 "under %s (%s)", f"{total:,}", len(entries), time.time() - t0,
                 ", ".join(MEMMAP_ARRAYS), array_dir, memory_note())

        return MetaboNetWindowSet(
            x=big["x"], x_aux=big["x_aux"],
            y_reg=merged["y_reg_multi"][:, primary],
            y_cls_hard=merged["y_cls_multi"][:, primary],
            y_min=merged["y_min_multi"][:, primary],
            patient=merged["patient"], t_index=merged["t_index"],
            t_time=merged["t_time"], last_bg=merged["last_bg"],
            aux_frac=merged["aux_frac"], split=merged["split"],
            source=merged["source"],
            y_reg_multi=merged["y_reg_multi"], y_min_multi=merged["y_min_multi"],
            y_cls_multi=merged["y_cls_multi"], horizons_min=hz,
            aux_channels=METABONET_AUX_CHANNELS,
            excluded_sources=tuple(mcfg.exclude_sources),
        )


def _project_windows(
    done_windows: int,
    uncapped_seen: int,
    rows_seen: int,
    remaining: Sequence[str],
    subject_rows: dict[str, int] | None,
    cap: int,
) -> tuple[int, str]:
    """Estimate the finished window count, per remaining subject.

    Two things make a naive extrapolation wrong here, and both bite:

      * Subjects arrive in SOURCE order, and studies differ in record length by
        more than an order of magnitude, so "windows per subject" measured over
        the first twenty is not a sample of the rest.
      * Under `max_windows_per_subject`, windows per ROW is not stable either: a
        13k-row subject and a 126k-row subject both yield exactly `cap`, so a
        rate measured on small subjects over-projects the large ones tenfold.

    So the rate is taken on the UNCAPPED count (which is proportional to rows),
    applied to each remaining subject's own row count from the index, and the
    cap is applied per subject afterwards -- which is where it actually applies.
    """
    if not rows_seen:
        return done_windows, "no rows processed yet"
    rate = uncapped_seen / rows_seen
    if not subject_rows:
        n_seen = max(len(remaining), 1)
        per_subject = done_windows / n_seen
        return int(done_windows + per_subject * len(remaining)), \
            f"{per_subject:,.0f} windows/subject (no row counts supplied)"

    total = done_windows
    for key in remaining:
        est = rate * int(subject_rows.get(key, 0))
        total += min(est, cap) if cap else est
    return int(total), (f"{rate:.3f} uncapped windows/row over {rows_seen:,} "
                        f"rows, capped per subject at "
                        f"{cap if cap else 'no cap'}")


def build_metabonet_windows_streaming(
    path: str,
    keys: Sequence[str],
    cfg: DataConfig,
    mcfg: MetaboNetConfig | None = None,
    horizons_min: Sequence[int] | None = None,
    cache_dir: str | None = None,
    resume: bool = True,
    schema: MetaboNetSchema | None = None,
    log_every: int = 25,
    subject_rows: dict[str, int] | None = None,
) -> MetaboNetWindowSet:
    """Window the corpus one subject at a time, through a disk-backed store.

    The memory contract, end to end: one subject's rows and one subject's
    windows are resident at a time; everything already built is on disk. Peak
    RSS is therefore set by the largest single subject (~126k rows, ~110k
    windows, ~0.3 GB on the released file) rather than by the corpus.

    Cost is reported before it is incurred and again as it accrues. The
    projection is per ROW, not per subject, and `subject_rows` (from
    `SubjectIndex`) is what makes it honest: subjects are processed in source
    order and the sources differ in record length by more than an order of
    magnitude, so extrapolating "GB per subject" from the first twenty -- all of
    them from whichever study sorts first -- misprices the run badly. Windows
    per row is stable across sources, and the exact remaining row count is
    already known from the index.

    A run that cannot fit `mcfg.memory_budget_gb` stops in the first minute,
    naming the knobs that would make it fit, instead of being killed by the OS
    an hour in with nothing written.

    `cache_dir` is both the working store and the checkpoint. With `resume`, a
    second invocation skips every subject already sharded, so an interrupted
    twelve-hour build continues from where it stopped.
    """
    mcfg = mcfg or MetaboNetConfig()
    hz = tuple(int(h) for h in sorted(set(horizons_min or (cfg.horizon_min,))))
    keys = [str(k) for k in keys]
    fp = _fingerprint(cfg, mcfg, hz)

    tmp_dir = None
    if cache_dir is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="metabonet_windows_")
        cache_dir = tmp_dir
        resume = False
    os.makedirs(cache_dir, exist_ok=True)

    store = WindowStore(cache_dir, fp)
    store.acquire()
    resumed = store.load_manifest() if resume else False
    done = store.done_keys if resumed else set()
    if resumed and done:
        log.info("resuming window cache %s: %d/%d subjects already done "
                 "(%s windows from %d subjects; %d known to contribute none)",
                 cache_dir, len(done), len(keys), f"{store.n_windows:,}",
                 len(store.entries), len(store.barren))
    elif not resumed:
        # A stale or foreign store must not be mixed with this run's shards.
        if os.path.exists(store.manifest_path):
            shutil.rmtree(store.shard_dir, ignore_errors=True)
            os.makedirs(store.shard_dir, exist_ok=True)
        store.entries = []
        store.barren = []

    todo = [k for k in keys if k not in done]
    per_window = window_bytes(cfg, len(METABONET_AUX_CHANNELS), len(hz))
    log.info("windowing %d subjects (%d already done) | history %d min, "
             "horizons %s, stride %d | %d B/window | cap %s | cache %s",
             len(todo), len(done), cfg.history_min, list(hz), cfg.stride_steps,
             per_window, mcfg.max_windows_per_subject or "none", cache_dir)

    # Exact remaining work, from the index, so the projection below divides by a
    # real row count rather than a subject count that means different things in
    # different studies.
    cap = int(mcfg.max_windows_per_subject or 0)

    t0 = time.time()
    n_done, n_skipped, budget_checked = len(done), 0, False
    rows_seen, uncapped_seen = 0, 0

    # A fully cached store has nothing to ingest. Falling through to the reader
    # would ask it for an empty subject set, which it correctly refuses -- so a
    # complete resume would fail where a partial one succeeds.
    if not todo:
        log.info("every selected subject is already in the cache; nothing to "
                 "window")
        _log_source_ledger(keys, store)
        ws = store.finalize(cfg, mcfg, hz)
        store.release()
        return ws

    for i, (key, pdf) in enumerate(
            iter_subject_frames(path, todo, schema, mcfg,
                                subject_rows=subject_rows), start=1):
        rows_seen += len(pdf)
        part = _subject_windows(key, pdf, cfg, mcfg, hz)
        del pdf
        if part is None:
            n_skipped += 1
            store.barren.append(key)
            log.debug("subject %d/%d: %s contributed no windows", i, len(todo), key)
            continue

        store.append(key, part)
        n = int(part["n"])
        uncapped_seen += int(part["n_before_cap"])
        thinned = (f", {part['subsample_k']}x block-thinned from "
                   f"{part['n_before_cap']:,}" if part["subsample_k"] > 1 else "")
        del part

        n_done += 1
        if i % log_every == 0 or i == len(todo) or i == 1:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            log.info("subject %d/%d: %s (%s windows%s) | total %s | "
                     "%.1f subj/s, ETA %.0f min | %s",
                     i, len(todo), key, f"{n:,}", thinned,
                     f"{store.n_windows:,}", rate, eta / 60, memory_note())
            store.save_manifest()

        # Price the run from real subjects rather than from an assumption, once
        # there is enough of a sample to extrapolate from.
        if not budget_checked and i >= min(20, len(todo)) and mcfg.memory_budget_gb:
            projected_windows, basis = _project_windows(
                store.n_windows, uncapped_seen, rows_seen, todo[i:],
                subject_rows, cap)
            projected = projected_windows * per_window / 2**30
            log.info("projected store size: %.1f GB (%s windows) for %d "
                     "subjects, from %s (budget %.1f GB)", projected,
                     f"{projected_windows:,}", len(keys), basis,
                     mcfg.memory_budget_gb)
            if projected > mcfg.memory_budget_gb:
                store.save_manifest()
                store.release()
                raise MemoryError(
                    f"projected window store is {projected:.1f} GB for "
                    f"{len(keys)} subjects, over the "
                    f"{mcfg.memory_budget_gb:.1f} GB budget.\n"
                    f"  Reduce it with MetaboNetConfig.max_windows_per_subject "
                    f"(block thinning within each subject), a larger "
                    f"DataConfig.stride_steps, or fewer subjects.\n"
                    f"  {len(store.entries)} subjects are already cached in "
                    f"{cache_dir} and will be reused on the next attempt.")
            budget_checked = True

    store.save_manifest()
    log.info("windowed %d subjects in %.1f min (%d contributed nothing); "
             "%s windows, %.1f GB of tensors (%s)",
             n_done, (time.time() - t0) / 60, n_skipped,
             f"{store.n_windows:,}", store.n_windows * per_window / 2**30,
             memory_note())

    _log_source_ledger(keys, store)
    ws = store.finalize(cfg, mcfg, hz)
    store.release()
    if tmp_dir is not None:
        # The memmaps live under this directory, so it cannot be removed while
        # the window set is alive -- deleting it here would pull the arrays out
        # from under the caller. Reported at INFO rather than swallowed, because
        # it is real disk the operator may want back before the OS reclaims it.
        log.info("window store is in a TEMPORARY directory and will not be "
                 "reused: %s (%.1f GB). Pass cache_dir= to keep and resume it.",
                 tmp_dir, store.n_windows * per_window / 2**30)
    return ws


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class MetaboNetSplit:
    """Row indices into a `MetaboNetWindowSet`. Disjoint by construction."""
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray
    embargoed: np.ndarray
    val_cut: dict                       # subject key -> inner val boundary
    protocol: Literal["official", "patient_wise"] = "official"
    sources: tuple[str, ...] = ()

    def summary(self, ws: MetaboNetWindowSet) -> str:
        def line(name, idx):
            if idx.size == 0:
                return f"  {name:>10}: empty"
            return (f"  {name:>10}: {idx.size:7d} windows | "
                    f"{len(set(ws.patient[idx])):4d} subjects | "
                    f"hypo {ws.y_cls_hard[idx].mean():6.2%}")
        return "\n".join([
            f"  protocol: {self.protocol}",
            line("fit", self.fit), line("val", self.val), line("test", self.test),
            f"  {'embargoed':>10}: {self.embargoed.size:7d} windows dropped",
        ])


def assert_disjoint(split: MetaboNetSplit) -> None:
    """Row-level disjointness. Cheap, and catches an index bug immediately."""
    for a, b in (("fit", "val"), ("fit", "test"), ("val", "test")):
        inter = np.intersect1d(getattr(split, a), getattr(split, b))
        if inter.size:
            raise AssertionError(f"{a}/{b} splits share {inter.size} windows")


def assert_no_temporal_overlap(
    ws: MetaboNetWindowSet, split: MetaboNetSplit, cfg: DataConfig
) -> None:
    """No fit window may share a RAW SAMPLE with a val or test window of the
    same subject.

    Disjoint row indices are not enough: two windows overlap in raw data
    whenever their `[t - history, t + horizon]` spans intersect. This is the
    check that verifies the embargo actually did its job, so it runs on every
    split rather than only under test.

    Unlike the OhioT1DM equivalent it also checks fit-vs-TEST, because
    MetaboNet's temporal subjects have no gap between their train and test
    regions -- the test side starts at the very next sample -- so the boundary
    needs an embargo of its own.
    """
    P = np.timedelta64(cfg.sample_period_min, "m")
    hist = P * int(cfg.history_steps)
    max_hz = (max(h // cfg.sample_period_min for h in ws.horizons_min)
              if ws.horizons_min else cfg.horizon_steps)
    fwd = P * int(max_hz)

    for later_name in ("val", "test"):
        later = getattr(split, later_name)
        if later.size == 0:
            continue
        for pid in sorted({str(p) for p in ws.patient[later]}):
            f = split.fit[ws.patient[split.fit] == pid]
            l = later[ws.patient[later] == pid]
            if f.size == 0 or l.size == 0:
                continue
            fit_end = ws.t_time[f].max() + fwd
            later_start = ws.t_time[l].min() - hist
            if fit_end >= later_start:
                raise AssertionError(
                    f"Subject {pid}: fit window spans reach {fit_end} but "
                    f"{later_name} spans start at {later_start} -- embargo too "
                    f"short for the configured history/horizon."
                )


def _chronological_inner_split(
    ws: MetaboNetWindowSet,
    train_rows: np.ndarray,
    cfg: DataConfig,
    val_frac: float,
    embargo_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-subject chronological tail as validation, with a fit-side embargo.

    A RANDOM inner split would be catastrophic rather than merely optimistic:
    at stride 1 with a 24-step history, adjacent windows share 23 of 24 history
    steps, so a random split scores near-duplicates against each other and
    validation stops measuring generalisation at all.

    The embargo is taken from the FIT side deliberately -- shrinking training is
    a cost, whereas shrinking validation would make model selection noisier on
    the one signal early stopping trusts.
    """
    fit_parts, val_parts, emb_parts = [], [], []
    cuts: dict = {}
    embargo_td = np.timedelta64(cfg.sample_period_min, "m") * int(embargo_steps)

    for pid in sorted({str(p) for p in ws.patient[train_rows]}):
        idx = train_rows[ws.patient[train_rows] == pid]
        if idx.size == 0:
            continue
        order = idx[np.argsort(ws.t_time[idx], kind="stable")]

        n_val = int(round(order.size * val_frac))
        if n_val < 1 or n_val >= order.size:
            # Too little data to hold out a meaningful tail: keep it all for
            # fitting and let the caller see the smaller val set, rather than
            # emitting a one-window validation split that early stopping would
            # then trust.
            fit_parts.append(order)
            continue

        cut_time = ws.t_time[order][order.size - n_val]
        cuts[pid] = cut_time

        is_val = ws.t_time[order] >= cut_time
        # `>=`, not `>`: a fit window anchored exactly at `cut - embargo` has its
        # last target step land on the first history step of the earliest val
        # window -- one shared raw sample, which the overlap assertion counts
        # (correctly) as overlap.
        is_emb = (~is_val) & (ws.t_time[order] >= cut_time - embargo_td)

        val_parts.append(order[is_val])
        emb_parts.append(order[is_emb])
        fit_parts.append(order[(~is_val) & (~is_emb)])

    cat = lambda parts: (np.concatenate(parts) if parts  # noqa: E731
                         else np.empty(0, np.int64)).astype(np.int64)
    return cat(fit_parts), cat(val_parts), cat(emb_parts), cuts


def _log_split(split: MetaboNetSplit, ws: MetaboNetWindowSet, cfg: DataConfig,
               embargo_steps: int, boundary: int, inner: int) -> None:
    """Partition statistics, at INFO, with the embargo accounted for.

    The embargo is the part a reader cannot check from the fit/val/test counts
    alone -- windows that were dropped leave no trace in what remains -- so how
    many went, at which boundary, and how long the embargo was are all reported
    rather than inferred.
    """
    total = len(ws)
    emb_min = embargo_steps * cfg.sample_period_min
    log.info("split (%s): fit %s | val %s | test %s | embargoed %s of %s windows",
             split.protocol, f"{split.fit.size:,}", f"{split.val.size:,}",
             f"{split.test.size:,}", f"{split.embargoed.size:,}", f"{total:,}")
    log.info("  embargo %d steps (%d min = history %d + max horizon) | "
             "%s dropped at the fit/val cut, %s at the train/test boundary",
             embargo_steps, emb_min, cfg.history_steps, f"{inner:,}",
             f"{boundary:,}")
    for name in ("fit", "val", "test"):
        idx = getattr(split, name)
        if idx.size == 0:
            log.info("  %-4s empty", name)
            continue
        log.info("  %-4s %s windows | %d subjects | hypo %.2f%%", name,
                 f"{idx.size:,}", len({str(p) for p in ws.patient[idx]}),
                 100 * float(ws.y_cls_hard[idx].mean()))


def official_metabonet_split(
    ws: MetaboNetWindowSet,
    cfg: DataConfig,
    val_frac: float = 0.2,
    embargo_steps: int | None = None,
) -> MetaboNetSplit:
    """MetaboNet's shipped `is_test` protocol, under this repo's contract.

    Three leak paths are closed here, and all three are easy to open by
    accident:

    1. TEST is exactly the windows tagged `test` by `is_test`. It is never
       resampled or rebalanced, so results stay comparable to anything else
       computed on the released split.

    2. VAL is the chronological TAIL of each subject's train-side windows, per
       subject, never a random sample (see `_chronological_inner_split`).

    3. EMBARGO, at BOTH boundaries. The fit/val cut gets the usual
       `history + max_horizon` embargo. The train/test boundary gets one too,
       which OhioT1DM does not need -- its train and test live in separate files
       with a gap between them, whereas a MetaboNet temporal subject's test
       region begins at the very next sample after its train region, so a train
       window anchored near the boundary has targets reaching into the first
       history steps of the earliest test window.

    Windows cannot straddle the boundary in the first place: `synchronize_subject`
    breaks a segment there, so this embargo removes overlap between *distinct*
    windows rather than repairing a window that spans the cut.
    """
    if embargo_steps is None:
        max_hz = (max(h // cfg.sample_period_min for h in ws.horizons_min)
                  if ws.horizons_min else cfg.horizon_steps)
        embargo_steps = cfg.history_steps + max_hz

    train_rows = ws.where("train")
    test_rows = ws.where("test")
    embargo_td = np.timedelta64(cfg.sample_period_min, "m") * int(embargo_steps)

    # ---- embargo at the train/test boundary, per subject -------------------
    boundary_emb: list[np.ndarray] = []
    keep_train: list[np.ndarray] = []
    test_start = {}
    for pid in sorted({str(p) for p in ws.patient[test_rows]}):
        t = test_rows[ws.patient[test_rows] == pid]
        if t.size:
            test_start[pid] = ws.t_time[t].min()

    for pid in sorted({str(p) for p in ws.patient[train_rows]}):
        idx = train_rows[ws.patient[train_rows] == pid]
        start = test_start.get(pid)
        if start is None:
            keep_train.append(idx)
            continue
        too_late = ws.t_time[idx] >= start - embargo_td
        boundary_emb.append(idx[too_late])
        keep_train.append(idx[~too_late])

    cat = lambda parts: (np.concatenate(parts) if parts  # noqa: E731
                         else np.empty(0, np.int64)).astype(np.int64)
    train_rows = cat(keep_train)

    fit, val, emb, cuts = _chronological_inner_split(
        ws, train_rows, cfg, val_frac, embargo_steps)

    split = MetaboNetSplit(
        fit=np.sort(fit), val=np.sort(val), test=np.sort(test_rows.astype(np.int64)),
        embargoed=np.sort(np.concatenate([emb, cat(boundary_emb)]).astype(np.int64)),
        val_cut=cuts, protocol="official",
        sources=tuple(sorted({str(s) for s in ws.source})),
    )
    assert_disjoint(split)
    assert_no_temporal_overlap(ws, split, cfg)
    _log_split(split, ws, cfg, embargo_steps,
               boundary=int(cat(boundary_emb).size), inner=int(emb.size))
    return split


def patient_wise_metabonet_split(
    ws: MetaboNetWindowSet,
    cfg: DataConfig,
    held_out: Sequence[str],
    val_frac: float = 0.2,
    embargo_steps: int | None = None,
) -> MetaboNetSplit:
    """The unseen-subject protocol: whole subjects move into test.

    The official split answers "can we forecast this person's future from their
    past". This one answers "can we forecast a stranger". Both are legitimate,
    they measure different things, and reporting one while describing the other
    is the most common way a CGM result becomes incomparable.

    No embargo is needed between train and test here -- different people share
    no raw samples -- but the inner fit/val split is still chronological, for
    the same window-overlap reason as everywhere else.

    `held_out` are composite subject keys. Passing a bare `id` is an error
    rather than a near-miss, because `id` is ambiguous across `source_file`.
    """
    known = {str(p) for p in ws.patient}
    ho = {str(k) for k in held_out}
    unknown = ho - known
    if unknown:
        bare = {k for k in unknown if ":" not in k}
        hint = ("  (ids must be composite '<source>:<id>' keys -- a bare id is "
                "ambiguous across source_file)" if bare else "")
        raise ValueError(f"Unknown MetaboNet subject keys: {sorted(unknown)}.{hint}")
    if not ho:
        raise ValueError("patient_wise_metabonet_split needs at least one held-out subject.")

    if embargo_steps is None:
        max_hz = (max(h // cfg.sample_period_min for h in ws.horizons_min)
                  if ws.horizons_min else cfg.horizon_steps)
        embargo_steps = cfg.history_steps + max_hz

    is_test = np.array([str(p) in ho for p in ws.patient])
    train_rows = np.flatnonzero(~is_test)

    fit, val, emb, cuts = _chronological_inner_split(
        ws, train_rows, cfg, val_frac, embargo_steps)

    split = MetaboNetSplit(
        fit=np.sort(fit), val=np.sort(val),
        test=np.sort(np.flatnonzero(is_test).astype(np.int64)),
        embargoed=np.sort(emb), val_cut=cuts, protocol="patient_wise",
        sources=tuple(sorted({str(s) for s in ws.source})),
    )
    assert_disjoint(split)
    assert_no_temporal_overlap(ws, split, cfg)
    _log_split(split, ws, cfg, embargo_steps, boundary=0, inner=int(emb.size))
    return split


def metabonet_folds(
    ws: MetaboNetWindowSet, n_folds: int, seed: int,
    group_by_source: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic grouped K-fold over composite SUBJECT KEYS, never rows.

    `group_by_source` additionally keeps every subject of one `source_file` in
    the same fold. That is the stricter protocol and the honest one for a
    consolidated corpus: subjects within a source share a device, a protocol and
    often a clinical site, so a model can score well across a within-source
    split by recognising the source rather than by forecasting glucose.
    """
    keys = np.array(sorted({str(p) for p in ws.patient}), dtype=object)
    rng = np.random.default_rng(seed)

    if group_by_source:
        by_source: dict[str, list[str]] = {}
        for k in keys:
            by_source.setdefault(split_subject_key(k)[0], []).append(k)
        groups = [np.array(v, dtype=object) for _, v in sorted(by_source.items())]
    else:
        groups = [np.array([k], dtype=object) for k in keys]

    order = rng.permutation(len(groups))
    buckets = np.array_split(order, n_folds)
    folds = []
    for b in buckets:
        test = np.concatenate([groups[i] for i in b]) if len(b) else np.empty(0, object)
        test_set = set(test.tolist())
        train = np.array([k for k in keys if k not in test_set], dtype=object)
        folds.append((train, test))
    return folds


# --------------------------------------------------------------------------- #
# One-call entry point
# --------------------------------------------------------------------------- #
def load_metabonet(
    path: str,
    cfg: DataConfig | None = None,
    mcfg: MetaboNetConfig | None = None,
    schema: MetaboNetSchema | None = None,
    keys: Sequence[str] | None = None,
    horizons_min: Sequence[int] | None = None,
    protocol: Literal["official", "patient_wise"] = "official",
    held_out: Sequence[str] | None = None,
    val_frac: float = 0.2,
    allow_benchmark_sources: Sequence[str] = (),
    cache_dir: str | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> tuple[MetaboNetWindowSet, MetaboNetSplit]:
    """Scan, load, window and split in the one order that is correct.

    Returns `(window_set, split)` ready for `train.run_fold` (which detects
    multimodality from `x_aux` and needs no MetaboNet-specific branch).

    Windowing streams: subjects are read and windowed one at a time into a
    disk-backed store, so this runs on the whole corpus without holding it. Pass
    `cache_dir` to keep the store between invocations, which also makes an
    interrupted build resumable.

    The contamination check runs BEFORE any windowing, so a run that would have
    trained on the OhioT1DM test spans fails in seconds rather than after an
    hour of fitting.
    """
    cfg = cfg or DataConfig()
    mcfg = mcfg or MetaboNetConfig()
    schema = schema or MetaboNetSchema()

    index = scan_subjects(path, schema, mcfg)
    selected = list(keys) if keys is not None else index.keys()
    assert_no_benchmark_contamination(selected, allow=allow_benchmark_sources)

    if verbose:
        print(index.summary())
        dis = index.disagreements
        if len(dis):
            print(f"\n{len(dis)} subject(s) whose subject_split_across_traintest "
                  f"flag contradicts their own rows:\n"
                  f"{dis[['key', 'split_class', 'flag_says_split']].to_string(index=False)}")

    ws = build_metabonet_windows_streaming(
        path, selected, cfg, mcfg, horizons_min=horizons_min,
        cache_dir=cache_dir, resume=resume, schema=schema,
        subject_rows=dict(zip(index.frame.key, index.frame.rows.astype(int))))

    if protocol == "official":
        split = official_metabonet_split(ws, cfg, val_frac=val_frac)
    else:
        if not held_out:
            raise ValueError("protocol='patient_wise' requires `held_out` subject keys.")
        split = patient_wise_metabonet_split(ws, cfg, held_out, val_frac=val_frac)

    if verbose:
        print(f"\n{len(ws)} windows over {len(ws.subjects())} subjects")
        print(split.summary(ws))
    return ws, split


def describe(ws: MetaboNetWindowSet) -> str:
    """Human-readable provenance of a built window set."""
    by_source: dict[str, int] = {}
    for s in ws.source:
        by_source[str(s)] = by_source.get(str(s), 0) + 1
    lines = [
        f"MetaboNet window set: {len(ws)} windows, {len(ws.subjects())} subjects",
        f"  horizons      : {list(ws.horizons_min)} min",
        f"  cgm channels  : {ws.x.shape[-1]}   aux channels: {ws.x_aux.shape[-1]}",
        f"  hypo prevalence: {ws.y_cls_hard.mean():.2%}",
        f"  excluded source_file: {list(ws.excluded_sources) or 'none'}",
        "  windows per source:",
    ]
    for s, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {s:<14} {n:8d}")
    return "\n".join(lines)
