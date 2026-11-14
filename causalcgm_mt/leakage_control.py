"""A DELIBERATELY BROKEN pipeline, kept as the control arm of Ablation D.

READ THIS BEFORE IMPORTING ANYTHING HERE
========================================
Nothing in this module is fit to train a model you intend to use. Every function
in it leaks future information into the past, on purpose. It exists so the paper
can answer a question that is otherwise unanswerable from the literature:

    How many points of RMSE and PR-AUC does a CGM forecasting result gain from
    lookahead bias alone, holding the model, the data and the hyperparameters
    fixed?

Papers in this area frequently report numbers produced with one or more of the
flaws below, usually without noticing. Readers cannot subtract the inflation
because nobody measures it. This module measures it, by running the same
architecture twice -- once through `causalcgm_mt/ohio_loader.py` and once through
here -- and reporting the gap.

THE THREE FLAWS, AND WHY EACH ONE IS A LEAK
-------------------------------------------
1. FULL-SERIES INTERPOLATION BEFORE WINDOWING (`interpolate_whole_series`).
   `df.interpolate()` over a patient's entire record fills a gap using the
   observations on BOTH sides of it. A window anchored inside such a gap
   therefore contains values computed from readings after the anchor -- and,
   because targets are drawn from the same filled series, some labels are
   interpolations of the very future they are supposed to predict. The strict
   pipeline confines interpolation to the history slice of each window, where
   both endpoints are at or before the anchor (`data._causal_gap_fill`).

2. CENTRED ROLLING STATISTICS (`centered_features`), COMPUTED ON THE WHOLE
   SERIES AND THEN SLICED. A centred rolling mean at position i averages
   `[i-w/2, i+w/2]`, so every feature step carries data from after itself. This
   is the single most common accidental leak in time-series work, because
   `pandas.rolling(center=True)` is one keyword away from the correct call and
   produces smoother, better-looking features.

   THE ORDER OF OPERATIONS IS THE WHOLE FLAW, and it is easy to get wrong even
   when deliberately writing broken code. Computing centred features on each
   window's own history slice leaks nothing past the anchor -- the centring has
   no future to reach into, because the slice ends at the anchor. The realistic
   mistake is `df['roll'] = df.glucose.rolling(6, center=True).mean()` on the
   full frame FOLLOWED BY windowing, which puts up to w/2 steps of the future
   into the last feature step of every window. An earlier version of this module
   did it per window and consequently measured almost no leakage at all -- an
   error in the reassuring direction, which is the one this whole module exists
   to guard against.

3. UNEMBARGOED RANDOM SPLIT (`naive_random_split`). At stride 1 with a 24-step
   history, adjacent windows share 23 of 24 history steps. A random row split
   puts near-duplicates of the same raw samples on both sides, so the test set
   is not held out in any meaningful sense -- it is a paraphrase of the training
   set. This one flaw alone is usually worth more than the other two combined.

WHAT IS HELD FIXED
------------------
The architecture, the loss, the optimiser, the seed, the horizons and the
feature CHANNEL LAYOUT are identical to the strict arm. Only the three flaws
above differ, so the measured gap is attributable to them and not to a
confound.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig
from .hupa_loader import HupaConfig
from .ohio_loader import (
    OHIO_AUX_CHANNELS, OhioConfig, OhioSplit, OhioWindowSet,
    synchronize_ohio_record,
)

LEAKY = True   # module-level flag; imported by tests that assert this is a control


# --------------------------------------------------------------------------- #
# Flaw 1: interpolate the whole series, in both directions
# --------------------------------------------------------------------------- #
def interpolate_whole_series(values: np.ndarray, observed: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """LEAKY. Fill every gap from both sides, over the entire record.

    `limit_direction="both"` also back-fills leading gaps, which is the second
    half of the same mistake: the first reading of a record gets a value derived
    from readings that had not happened yet.

    Returns (filled, imputed_mask). Contrast `data._causal_gap_fill`, which is
    the same idea confined to one window's history and bounded by
    `max_interpolation_gap_min`.
    """
    v = pd.Series(np.where(observed, values, np.nan), dtype="float64")
    filled = v.interpolate(method="linear", limit_direction="both").to_numpy()
    # A record with no observations at all stays NaN; the caller drops it.
    return filled, (~observed.astype(bool)) & np.isfinite(filled)


# --------------------------------------------------------------------------- #
# Flaw 2: centred rolling statistics
# --------------------------------------------------------------------------- #
def centered_features(series: np.ndarray, observed: np.ndarray,
                      window: int = 6) -> np.ndarray:
    """LEAKY drop-in for `data.causal_features`, identical channel layout.

    Channels: value, CENTRED first difference, CENTRED rolling mean, CENTRED
    rolling std, observation mask. The centred difference at i is
    (h[i+1] - h[i-1]) / 2, which reads one step into the future; the causal
    version is h[i] - h[i-1].

    CALL THIS ON THE WHOLE SERIES AND SLICE AFTERWARDS, which is what
    `build_naive_windows` does and what the realistic mistake looks like.
    Calling it on a window's own history slice leaks nothing past the anchor,
    because the slice ends there and the centring has no future to reach into --
    the argument is named `series` rather than `hist` to say so.

    The layout matches channel for channel so the same model, the same scaler
    and the same channel-skip lists apply to both arms -- if the tensors had
    different widths, the comparison would be measuring architecture, not
    leakage.
    """
    h = np.asarray(series, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.float32)

    diff = np.zeros_like(h)
    if h.size > 2:
        diff[1:-1] = (h[2:] - h[:-2]) / 2.0
        diff[0], diff[-1] = h[1] - h[0], h[-1] - h[-2]

    s = pd.Series(h.astype(np.float64))
    roll = s.rolling(window, center=True, min_periods=1)
    roll_mean = roll.mean().to_numpy().astype(np.float32)
    roll_std = np.nan_to_num(roll.std().to_numpy()).astype(np.float32)
    return np.stack([h, diff, roll_mean, roll_std, obs], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Flaw 3: random split with no embargo
# --------------------------------------------------------------------------- #
def naive_random_split(ws: OhioWindowSet, seed: int = 1337,
                       val_frac: float = 0.2, test_frac: float = 0.2) -> OhioSplit:
    """LEAKY. Shuffle every window and cut, ignoring time entirely.

    No embargo, no chronology, no respect for the official file boundary. Rows
    are disjoint -- which is exactly what makes this flaw so durable: a
    `assert len(set(train) & set(test)) == 0` check passes, and the split is
    still worthless, because disjoint ROWS are not disjoint DATA when the
    windows overlap by 96%.

    Deliberately applicable to the strict window set as well, so Ablation D can
    isolate the split flaw from the feature flaws.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ws))
    n_test = int(round(len(ws) * test_frac))
    n_val = int(round(len(ws) * val_frac))
    test, held = perm[:n_test], perm[n_test:n_test + n_val]
    fit = perm[n_test + n_val:]
    # The held-out region is halved into selection and calibration, mirroring
    # `official_temporal_split`'s four-way shape so the ablation compares two
    # SPLITS rather than a four-region pipeline against a three-region one. Both
    # halves are just as leaky as everything else here -- that is the point of
    # this module -- but the arm still has a calibration region, so the strict
    # and control arms differ in chronology and embargo alone.
    n_cal = held.size // 2
    val, calib = held[n_cal:], held[:n_cal]
    return OhioSplit(fit=np.sort(fit).astype(np.int64),
                     val=np.sort(val).astype(np.int64),
                     test=np.sort(test).astype(np.int64),
                     embargoed=np.empty(0, dtype=np.int64), val_cut={},
                     calib=np.sort(calib).astype(np.int64), calib_cut={})


# --------------------------------------------------------------------------- #
# The naive window builder
# --------------------------------------------------------------------------- #
def build_naive_windows(records, cfg: DataConfig, hcfg: HupaConfig | None = None,
                        ocfg: OhioConfig | None = None,
                        horizons_min=(30, 60)) -> OhioWindowSet:
    """LEAKY counterpart of `ohio_loader.build_ohio_windows`.

    Differences, all of them deliberate:

      * The training and testing FILES of a subject are concatenated into one
        series before windowing, so windows straddle the official boundary and
        a single window can read training history to predict a test target.
      * Gaps are filled over that whole concatenated series, in both directions
        and with no length limit, so a multi-hour dropout becomes a smooth ramp
        computed from its own future.
      * Features are centred (`centered_features`).
      * Segment boundaries are ignored, and a window is emitted whether or not
        its anchor and targets were observed -- imputed labels included, which
        is label leakage by construction.
      * The auxiliary-coverage filter (`OhioConfig.min_aux_observed_frac`) and
        the imputed-fraction cap (`DataConfig.max_imputed_frac`) are both
        dropped. These are quality filters rather than leaks, but they are the
        reason the strict builder emits fewer windows, and keeping them here
        would attribute part of the window-count difference to the wrong cause.

    Everything else -- channel layout, horizons, target definitions, the
    auxiliary tensor -- matches the strict builder exactly.
    """
    hcfg, ocfg = hcfg or HupaConfig(), ocfg or OhioConfig()
    horizons = tuple(int(h) for h in sorted(set(horizons_min)))
    if cfg.horizon_min not in horizons:
        horizons = tuple(sorted(set(horizons) | {int(cfg.horizon_min)}))
    hz_steps = [h // cfg.sample_period_min for h in horizons]
    primary = horizons.index(int(cfg.horizon_min))
    H, Pmax = cfg.history_steps, max(hz_steps)

    by_patient: dict[str, list] = {}
    for rec in records:
        reg = synchronize_ohio_record(rec, cfg, hcfg, ocfg)
        if reg.empty:
            continue
        reg = reg.copy()
        reg["__split"] = rec.split
        reg["__subset"] = rec.subset
        by_patient.setdefault(rec.patient_id, []).append(reg)

    xs, xa, yr, ym = [], [], [], []
    pids, tix, tts, lbg, afrac, splits, subs = [], [], [], [], [], [], []

    for pid, frames in by_patient.items():
        # FLAW: concatenate the official train and test files into one series.
        reg = pd.concat(frames, ignore_index=True).sort_values("time")
        reg = reg.reset_index(drop=True)

        g = reg["glucose"].to_numpy(dtype=np.float64)
        obs = reg["glucose_observed"].to_numpy(dtype=bool)
        # FLAW: fill every gap over the whole concatenated record.
        g, imputed = interpolate_whole_series(g, obs)
        if not np.isfinite(g).all():
            warnings.warn(f"{pid}: record is entirely unobserved, skipped")
            continue

        # FLAW: features are computed over the WHOLE concatenated record and
        # sliced afterwards, so the centred statistics at each window's last
        # step carry up to window/2 samples of that window's own future. Doing
        # this per window would leak nothing -- see `centered_features`.
        feats_full = centered_features(g, (~imputed).astype(np.float64))

        aux = np.stack([reg[c].to_numpy(dtype=np.float64) for c in OHIO_AUX_CHANNELS],
                       axis=-1)
        ts = reg["time"].to_numpy()
        split_tag = reg["__split"].to_numpy()
        subset_tag = reg["__subset"].to_numpy()

        for a in range(H - 1, len(reg) - Pmax, cfg.stride_steps):
            hs = slice(a - H + 1, a + 1)
            fut = g[a + 1: a + 1 + Pmax]
            # FLAW: no observed-target requirement -- an imputed label is used
            # exactly as if it had been measured.
            yr.append([fut[s - 1] for s in hz_steps])
            ym.append([fut[:s].min() if cfg.classify_over_window else fut[s - 1]
                       for s in hz_steps])
            xs.append(feats_full[hs])
            xa.append(aux[hs])
            pids.append(pid)
            tix.append(a)
            tts.append(ts[a])
            lbg.append(g[a])
            afrac.append(1.0)
            # The window is TAGGED by its anchor's file, but its history may
            # reach back into the other one -- which is the straddle this arm
            # exists to demonstrate.
            splits.append(str(split_tag[a]))
            subs.append(str(subset_tag[a]))

    if not xs:
        raise ValueError("no naive windows produced; check the record list")

    y_reg_multi = np.asarray(yr, dtype=np.float32)
    y_min_multi = np.asarray(ym, dtype=np.float32)
    y_cls_multi = (y_min_multi < cfg.hypo_threshold_mgdl).astype(np.float32)
    return OhioWindowSet(
        x=np.stack(xs).astype(np.float32),
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
        y_reg_multi=y_reg_multi, y_min_multi=y_min_multi, y_cls_multi=y_cls_multi,
        horizons_min=horizons,
    )


@dataclass(frozen=True)
class LeakArm:
    """Label + description for one rung of the leakage-attribution ladder."""
    key: str
    label: str
    windows: str        # "strict" | "naive"
    split: str          # "official" | "random"
    description: str


LEAK_LADDER: tuple[LeakArm, ...] = (
    LeakArm("strict", "Strict causal pipeline (ours)", "strict", "official",
            "closed-interval windows, per-window causal gap fill, backward-only "
            "rolling statistics, chronological split with a history+horizon "
            "embargo"),
    LeakArm("random_split", "+ unembargoed random split", "strict", "random",
            "identical causal windows; only the split is replaced by a random "
            "row shuffle, isolating the split flaw from the feature flaws"),
    LeakArm("naive", "+ full-series interpolation & centred features",
            "naive", "random",
            "all three flaws: whole-record two-sided interpolation across the "
            "official file boundary, centred rolling statistics, and the "
            "unembargoed random split"),
)
