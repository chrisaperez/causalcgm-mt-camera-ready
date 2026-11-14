"""Shared harness for the paper's benchmark, ablation and personalisation tables.

WHY A LIBRARY AND NOT THREE SCRIPTS
-----------------------------------
`run_paper_benchmarks.py`, `run_paper_ablations.py` and
`run_ohio_personalization_sweep.py` all need the same four things: load the Ohio
cohort under the official split, fit an arm, score it through one metric path,
and serialise the result. Writing that three times guarantees the three tables
in the paper are computed three slightly different ways, and the differences
would be invisible -- a benchmark table and an ablation table that disagree
about the same arm's RMSE is the kind of thing a reader finds and an author
cannot explain.

So the arms are DATA (`ArmSpec`), the pipeline is one function
(`run_arm`), and the scripts only decide which specs to enumerate.

THE ANTI-LEAKAGE CONTRACT, RESTATED FOR THIS LAYER
---------------------------------------------------
Everything fitted -- scaler, per-patient normaliser, soft-label band, class
prior, model weights, conformal quantile -- is estimated on rows that are
strictly earlier than, and non-overlapping with, the rows it is later applied
to:

    split.fit   -> preprocessor, model weights
    split.val   -> early stopping, and nothing else
    split.calib -> conformal calibration, and nothing else
    split.test  -> reported metrics, and nothing else, ever

`val` and `calib` are separate regions because they are separate statistical
jobs. Split conformal is valid only when the predictor is fixed independently of
the calibration set, and an early-stopped checkpoint is chosen BECAUSE it scored
best on the selection rows -- so calibrating there biases every residual
downward, shrinks `qhat`, and undercovers. Sharing one region between the two
was the defect; `official_temporal_split(calib_frac=...)` carves them apart and
embargoes the boundary.

`OhioPreprocessor.fit` enforces the first (and records `fitted_on_patients` so
an audit can confirm it); `official_temporal_split` enforces the embargo at
every internal boundary, and `assert_no_temporal_overlap` verifies all six
ordered pairs on every run. This module adds no fitting of its own outside those
two calls -- if it did, that would be the leak.

EVERY ARM IS SCORED ON THE SAME ROWS, THROUGH THE SAME CODE
-----------------------------------------------------------
Persistence, Ridge-AR, ARIMA, LightGBM, the LSTM, both CausalCGM-MT variants and
the LoRA-adapted model all produce `(bg_hat, risk)` on `split.calib` and
`split.test`, and all go through one `score_arm`. A baseline compared under
different conditions is not a baseline.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Sequence

import numpy as np

from .clinical import clinical_report, parkes_available
from .config import Config, DataConfig, ModelConfig, TrainConfig
from .conformal import (
    MondrianConformal, NormalizedConformal, SplitConformal,
    rate_of_change_difficulty,
)
from .data import WINDOW_PIPELINE_VERSION as _WINDOW_PIPELINE_VERSION
from .evaluate import EvalReport, calibration_thresholds, evaluate

HYPO_THRESHOLD_MGDL = 70.0

# Re-exported from `data`, which is where window content is defined. Every cache
# and checkpoint in the pipeline keys on this; see the note at its definition.
WINDOW_PIPELINE_VERSION = _WINDOW_PIPELINE_VERSION

# Families the harness knows how to fit. Declared as data so the scripts can
# validate `--arms` against it instead of failing halfway through a two-hour run.
FAMILIES: tuple[str, ...] = (
    "persistence", "ridge_ar", "arima", "lgbm", "lstm", "tcn", "tcn_mm", "lora",
)
NEURAL_FAMILIES: tuple[str, ...] = ("lstm", "tcn", "tcn_mm", "lora")
# Arms with no classification head, whose hypoglycemia score is a monotone
# map of their point forecast (`risk_from_forecast`). Everything else fits a
# classifier on the nadir label itself.
MAPPED_RISK_FAMILIES: tuple[str, ...] = ("persistence", "ridge_ar", "arima")


# --------------------------------------------------------------------------- #
# Arm specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmSpec:
    """One row of one table: everything needed to reproduce it.

    Frozen and fully serialisable on purpose -- the spec is written into the
    results JSON verbatim, so a number in the paper can be traced to the exact
    hyperparameters that produced it without re-reading the driver script.
    """
    name: str
    family: str
    horizons: tuple[int, ...] = (30, 60)
    unimodal: bool = False
    modalities: tuple[str, ...] | None = None     # None = every aux channel
    cohorts: tuple[str, ...] | None = None        # None = 2018 + 2020
    backbone: Literal["tcn", "dilated_lstm"] = "tcn"

    # ---- optimisation -------------------------------------------------------
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 6
    seed: int = 1337
    fit_subsample: int = 1        # keep every k-th FIT window; val/test untouched

    # ---- loss ---------------------------------------------------------------
    loss_weighting: Literal["uncertainty", "fixed"] = "uncertainty"
    lambda_reg: float = 1.0
    lambda_cls: float = 1.0
    hypo_asymmetry: float = 3.0

    # ---- soft-label band (see data.SoftLabeler) -----------------------------
    # `None` inherits the cohort's setting, which is what every arm outside
    # Ablation E does. Setting any of them makes this arm's LABELS differ from
    # the cohort's, so `run_arm` refits the preprocessor for it -- on the arm's
    # own fit rows, under the same contract -- rather than reusing a
    # preprocessor whose labeller was built for a different band.
    soft_labels: bool | None = None
    band_mode: Literal["mard", "fixed"] | None = None
    soft_band_mgdl: float | None = None
    band_mard_multiplier: float | None = None

    # ---- model selection ----------------------------------------------------
    monitor: str = "val/pr_auc"
    monitor_mode: Literal["max", "min"] = "max"

    # ---- what the regression target IS --------------------------------------
    # "level" predicts g(t+PH), the endpoint. "nadir" predicts
    # min g over (t, t+PH] -- the quantity the CLASSIFICATION label thresholds.
    # This exists to answer one question: the classical arms score hypoglycemia
    # through a monotone map of their forecast, so a level-forecasting arm is
    # being asked about a nadir it never optimised for. Training the same arm on
    # the nadir directly and mapping THAT measures how much of the gap is the
    # mapping and how much is endpoint forecasting itself.
    #
    # Only meaningful for the arms that go through `risk_from_forecast`
    # (persistence, ridge_ar, arima). LightGBM and every neural arm fit a
    # classifier on the nadir label directly and are unaffected.
    target: Literal["level", "nadir"] = "level"

    # ---- personalisation (family == "lora") ---------------------------------
    adapt_method: str = "lora"
    lora_rank: int = 4
    adapt_prefix_hours: float = 24.0
    adapt_steps: int = 200
    adapt_lr: float = 5e-4
    base_arm: str = ""            # name of the arm whose checkpoint is adapted

    notes: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected {FAMILIES}")
        if self.fit_subsample < 1:
            raise ValueError("fit_subsample must be >= 1")
        if self.target not in ("level", "nadir"):
            raise ValueError(f"unknown target {self.target!r}; "
                             f"expected 'level' or 'nadir'")
        if self.target == "nadir" and self.family not in MAPPED_RISK_FAMILIES:
            # A nadir target on an arm with its own classifier would change the
            # regression head's meaning while leaving the risk score untouched,
            # which is a different experiment wearing this one's name.
            raise ValueError(
                f"target='nadir' is only defined for {MAPPED_RISK_FAMILIES} -- "
                f"family {self.family!r} fits a classifier on the nadir label "
                f"directly and does not derive its risk from the forecast.")
        if not self.horizons:
            raise ValueError("at least one horizon is required")

    @property
    def is_neural(self) -> bool:
        return self.family in NEURAL_FAMILIES

    @property
    def soft_label_overrides(self) -> dict:
        """The `DataConfig` fields this arm changes, empty when it inherits."""
        return {k: v for k, v in (
            ("soft_labels", self.soft_labels),
            ("band_mode", self.band_mode),
            ("soft_band_mgdl", self.soft_band_mgdl),
            ("band_mard_multiplier", self.band_mard_multiplier),
        ) if v is not None}

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("horizons", "modalities", "cohorts"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d


# --------------------------------------------------------------------------- #
# Cohort loading (cached -- the XML parse is ~60 s)
# --------------------------------------------------------------------------- #
@dataclass
class CohortData:
    ws: object                 # OhioWindowSet
    split: object              # OhioSplit
    pre: object                # OhioPreprocessor
    cfg: DataConfig
    horizons: tuple[int, ...]
    reference_pairs: object = None   # ohio_loader.ReferencePairs | None

    @property
    def subjects(self) -> list[str]:
        return sorted({str(p) for p in self.ws.patient})

    def summary(self) -> str:
        return self.split.summary(self.ws)


def load_cohort(
    root: str = "OhioT1DM",
    history_min: int = 120,
    horizons: Sequence[int] = (30, 60),
    stride_steps: int = 1,
    val_frac: float = 0.2,
    cache: str | None = None,
    band_mode: Literal["mard", "fixed"] = "fixed",
    soft_band_mgdl: float = 10.0,
    with_reference_pairs: bool = True,
) -> CohortData:
    """Build (or reload) the Ohio windows, official split and fitted preprocessor.

    `band_mode` defaults to `"fixed"`, matching `scripts/train_ohio.py`, because
    that is the setting every published table in this repository was produced
    under and changing a default silently rewrites results. It is a PARAMETER
    rather than a literal so the alternative is reachable: Ablation E sweeps it,
    and `run_arm` refits the preprocessor for any arm that overrides it.

    `with_reference_pairs` parses the raw records for `finger_stick` events
    (~7 s) so `band_mode="mard"` can estimate a real MARD against paired
    capillary BG rather than a high-pass proxy. Turn it off for a run that will
    never touch the mard path.

    The cache holds the windows and the split ONLY. The preprocessor is refitted
    on load, cheaply, because it is the object with the leak surface and a
    stale pickle of it is the one thing here worth not trusting.
    """
    from .hupa_loader import HupaConfig
    from .ohio_loader import (
        OhioConfig, OhioPreprocessor, build_ohio_windows, collect_reference_pairs,
        discover_records, official_temporal_split,
    )

    hz = tuple(int(h) for h in sorted(set(horizons)))
    cfg = DataConfig(history_min=history_min, horizon_min=hz[0],
                     stride_steps=stride_steps, band_mode=band_mode,
                     soft_band_mgdl=soft_band_mgdl)

    ws = None
    if cache and os.path.exists(cache):
        import pickle
        with open(cache, "rb") as f:
            payload = pickle.load(f)
        if payload.get("pipeline_version") != WINDOW_PIPELINE_VERSION:
            print(f"{cache} was built by pipeline version "
                  f"{payload.get('pipeline_version', '<pre-versioning>')}, this is "
                  f"{WINDOW_PIPELINE_VERSION}; rebuilding")
        elif (tuple(payload["horizons"]) == hz
                and payload["history_min"] == history_min
                and payload["stride_steps"] == stride_steps):
            ws = payload["ws"]
            print(f"loaded {len(ws)} cached windows from {cache}")
        else:
            print(f"{cache} was built with different geometry; rebuilding")

    if ws is None:
        t0 = time.time()
        ws = build_ohio_windows(discover_records(root), cfg, HupaConfig(),
                                OhioConfig(), horizons_min=hz)
        print(f"built {len(ws)} windows from {root} in {time.time() - t0:.0f}s")
        if cache:
            import pickle
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"ws": ws, "horizons": hz, "history_min": history_min,
                             "stride_steps": stride_steps,
                             "pipeline_version": WINDOW_PIPELINE_VERSION}, f)

    pairs = None
    if with_reference_pairs:
        t0 = time.time()
        pairs = collect_reference_pairs(discover_records(root), cfg)
        print(f"collected {len(pairs)} paired reference readings in "
              f"{time.time() - t0:.0f}s")

    split = official_temporal_split(ws, cfg, val_frac=val_frac)
    pre = OhioPreprocessor.fit(ws, split, cfg, reference_pairs=pairs)
    return CohortData(ws=ws, split=split, pre=pre, cfg=cfg, horizons=hz,
                      reference_pairs=pairs)


def arm_cohort(data: CohortData, spec: ArmSpec) -> CohortData:
    """`data`, or a copy whose preprocessor was refitted for this arm's band.

    Only Ablation E's arms differ here, and only in the soft-label policy. The
    refit runs on the SAME fit rows as the cohort-level preprocessor, so the
    scaler, the per-patient normaliser and the class prior come out identical
    and the only thing that moves is `y_soft` -- which is the point of the
    ablation. Reusing `data.pre` instead would train every arm on the cohort
    default's labels while reporting them under the arm's name, i.e. it would
    report a sweep that never happened.

    Refitting rather than mutating also keeps `data.pre` frozen for every other
    arm in the same run: `OhioPreprocessor` is fitted once and applied, and an
    in-place band swap would leak across arms in whichever order they ran.
    """
    over = spec.soft_label_overrides
    if not over:
        return data

    from .ohio_loader import OhioPreprocessor

    cfg = dataclasses.replace(data.cfg, **over)
    pre = OhioPreprocessor.fit(data.ws, data.split, cfg,
                               reference_pairs=data.reference_pairs)
    return dataclasses.replace(data, cfg=cfg, pre=pre)


def restrict_split(data: CohortData, spec: ArmSpec) -> object:
    """A cohort-restricted copy of the split, for single-cohort ablation arms.

    2018 wore a Basis Peak and 2020 an Empatica; the sensor sets are disjoint,
    so an arm that needs heart rate is only measurable on 2018 subjects. Cutting
    the SPLIT rather than the window set keeps every row index valid against the
    same `ws`, which is what lets the restricted and unrestricted arms share one
    scoring path.
    """
    from .ohio_loader import OhioSplit

    if spec.cohorts is None:
        return data.split
    keep = np.isin(np.asarray([str(s) for s in data.ws.subset_tag]),
                   np.asarray([str(c) for c in spec.cohorts]))
    s = data.split
    sel = lambda idx: idx[keep[idx]]  # noqa: E731
    return OhioSplit(fit=sel(s.fit), val=sel(s.val), test=sel(s.test),
                     embargoed=sel(s.embargoed), val_cut=s.val_cut,
                     calib=sel(s.calib), calib_cut=s.calib_cut)


def subsample_rows(ws, idx: np.ndarray, k: int) -> np.ndarray:
    """Keep every k-th window per patient, in time order.

    Used to shorten training runs. Legitimate because adjacent stride-1 windows
    share 23 of 24 history steps, so dropping two of every three costs far less
    information than it costs wall-clock -- and it is applied to the FIT rows
    only, never to val or test, so the reported metrics stay at native cadence
    and remain comparable across arms with different budgets.
    """
    if k <= 1 or idx.size == 0:
        return idx
    # Stringified ONCE. Rebuilding it inside the loop ran a Python-level
    # `str()` over every fit row per patient -- ~1.2M calls per arm on the Ohio
    # cohort, quadratic in subject count on a MetaboNet-scale index.
    pat = np.asarray([str(p) for p in ws.patient[idx]])
    out = []
    for pid in sorted(set(pat.tolist())):
        rows = idx[pat == pid]
        rows = rows[np.argsort(ws.t_time[rows], kind="stable")]
        out.append(rows[::k])
    return np.sort(np.concatenate(out)).astype(np.int64)


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def _as_column(a: np.ndarray) -> np.ndarray:
    """(N,) -> (N, 1); (N, K) unchanged. One horizon is a column, never a row."""
    v = np.asarray(a, dtype=np.float64)
    return v.reshape(-1, 1) if v.ndim == 1 else v


@dataclass
class Predictions:
    """(N, K) forecasts and hypoglycemia risk scores, columns matching horizons."""
    bg: np.ndarray
    risk: np.ndarray

    def __post_init__(self) -> None:
        # A 1-D array is ONE horizon over N windows, i.e. (N, 1). `atleast_2d`
        # would make it (1, N) -- the transpose -- and every metric downstream
        # would then be computed over a single row without raising.
        self.bg = _as_column(self.bg)
        self.risk = _as_column(self.risk)
        if self.bg.shape != self.risk.shape:
            raise ValueError(f"bg {self.bg.shape} vs risk {self.risk.shape}")

    @property
    def n_horizons(self) -> int:
        return int(self.bg.shape[1])

    def col(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.bg[:, k], self.risk[:, k]


def risk_from_forecast(bg_hat: np.ndarray, threshold: float = HYPO_THRESHOLD_MGDL,
                       tau: float = 10.0) -> np.ndarray:
    """A hypoglycemia score for models that only forecast a level.

        risk = sigmoid((threshold - bg_hat) / tau)

    A strictly monotone map of the forecast, so ROC-AUC, PR-AUC and
    sensitivity-at-fixed-FPR for this score are EXACTLY those of the underlying
    regression -- no information is invented, and the ranking baselines get is
    the best one their forecast supports. FAPD is well defined too, because its
    operating point is derived from the ROC rather than from a calibrated
    probability.

    What this score cannot do is the thing the multi-task head exists for. The
    label is `min BG over (t, t+PH] < 70`, and a level forecast at exactly t+PH
    is blind to a nadir at t+20 that recovers by t+30. That gap is a real
    property of regression-only baselines, and it belongs in the table rather
    than being papered over with a better-calibrated wrapper.
    """
    z = (float(threshold) - np.asarray(bg_hat, dtype=np.float64)) / float(tau)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def persistence_predictions(ws, idx: np.ndarray, horizons: Sequence[int]) -> Predictions:
    """BG(t + PH) = BG(t). The floor every other model must clear.

    Included at every horizon because its failure mode is horizon-dependent and
    RMSE alone hides it: persistence achieves a respectable 30-minute RMSE while
    being a full horizon late, which `time_lag_min` is there to expose.
    """
    last = np.asarray(ws.last_bg[idx], dtype=np.float64)
    bg = np.repeat(last[:, None], len(horizons), axis=1)
    return Predictions(bg=bg, risk=risk_from_forecast(bg))


def _lag_matrix(ws, idx: np.ndarray) -> np.ndarray:
    """The raw glucose history of each window, oldest first: (N, H) in mg/dL.

    Channel 0 of the already-causal feature tensor, so it inherits the prefix
    guarantee -- these are lags at or before the anchor, by construction.
    """
    return np.asarray(ws.x[idx][:, :, 0], dtype=np.float64)


class _quiet_blas:
    """Suppress Apple Accelerate's spurious FP-flag warnings on large matmuls.

    On numpy 2.x + macOS Accelerate, `X.T @ X` for an (N, 24) design with
    N ~ 10^5 raises "divide by zero / overflow / invalid encountered in matmul"
    even when every input is finite and the result is exact: the vectorised
    kernel leaves IEEE status flags set from lanes that were masked out. Checked
    against a closed-form `np.linalg.solve` on this cohort -- coefficients agree
    to 5e-13 and every prediction is finite.

    So the warnings are suppressed here and the OUTPUT is checked instead
    (`_require_finite`), which catches a genuine numerical failure that
    silencing the warning would otherwise hide.
    """

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        self._e = np.errstate(divide="ignore", over="ignore", invalid="ignore")
        self._e.__enter__()
        return self

    def __exit__(self, *exc):
        self._e.__exit__(*exc)
        self._w.__exit__(*exc)
        return False


def _require_finite(a: np.ndarray, what: str) -> np.ndarray:
    if not np.isfinite(a).all():
        raise FloatingPointError(
            f"{what} produced {int((~np.isfinite(a)).sum())} non-finite values")
    return a


@dataclass
class RidgeAR:
    """Direct multi-step linear autoregression, one ridge fit per horizon.

    y(t+PH) - y(t) = w . [y(t-H+1) ... y(t)] + b

    Two choices worth defending. It targets the RESIDUAL from the last observed
    value, for the same reason every other model here does: predicting the level
    lets a linear model score well by echoing its input, which inflates RMSE
    performance while producing a time lag equal to the full horizon.

    And it is DIRECT multi-step (one model per horizon) rather than an iterated
    one-step recursion. Iterating compounds error and, more importantly, makes
    the 60-minute forecast a nonlinear function of the fitted parameters, which
    is not what a classical AR baseline is usually taken to mean.
    """
    alpha: float = 10.0
    target: str = "level"
    models: dict = field(default_factory=dict)      # horizon index -> Ridge

    def fit(self, ws, fit_idx: np.ndarray, horizons: Sequence[int],
            cols: Sequence[int]) -> "RidgeAR":
        from sklearn.linear_model import Ridge

        X = _require_finite(_lag_matrix(ws, fit_idx), "Ridge-AR design matrix")
        last = np.asarray(ws.last_bg[fit_idx], dtype=np.float64)
        # "nadir" fits the same design matrix to min g over (t, t+PH] instead of
        # g(t+PH). Still a residual from the last observed value, so the two
        # variants differ in the target alone and nothing else moves.
        tgt = ws.y_min_multi if self.target == "nadir" else ws.y_reg_multi
        with _quiet_blas():
            for k, c in enumerate(cols):
                y = np.asarray(tgt[fit_idx, c], dtype=np.float64) - last
                m = Ridge(alpha=self.alpha).fit(X, y)
                _require_finite(m.coef_, f"Ridge-AR coefficients (horizon {c})")
                self.models[k] = m
        return self

    def predict(self, ws, idx: np.ndarray) -> Predictions:
        X = _lag_matrix(ws, idx)
        last = np.asarray(ws.last_bg[idx], dtype=np.float64)
        with _quiet_blas():
            bg = np.stack([last + self.models[k].predict(X)
                           for k in sorted(self.models)], axis=1)
        _require_finite(bg, "Ridge-AR forecast")
        return Predictions(bg=np.clip(bg, 1.0, 600.0), risk=risk_from_forecast(bg))


@dataclass
class ARIMAForecaster:
    """ARIMA(p, d, 0) fitted per subject by statsmodels, applied causally per window.

    WHY q = 0, AND WHY THE FORECAST IS RECOMPUTED RATHER THAN CALLED
    ----------------------------------------------------------------
    A moving-average term is a function of the innovation history, which is only
    recoverable by filtering the series from its start. Applying that to a
    WINDOW -- 24 samples in isolation -- would either need the full series
    (which reaches across the official train/test boundary, i.e. leakage) or
    would silently start the filter from zero innovations. Restricting to the
    pure autoregressive form makes the h-step forecast an exact closed-form
    function of the window's own lags, which is the only version of "ARIMA" that
    is well defined under this evaluation protocol.

    Given that, the forecast is the AR recursion iterated h times on the
    d-differenced window and cumulated back -- vectorised over all windows at
    once. Calling `statsmodels` once per window would produce the same numbers
    at ~10^5 times the cost. The PARAMETERS still come from statsmodels'
    conditional MLE on the subject's own training series, so this is a real
    ARIMA fit and not a hand-rolled substitute.
    """
    order: tuple[int, int, int] = (4, 1, 0)
    params: dict = field(default_factory=dict)      # patient -> (const, ar coefs)

    def fit(self, ws, fit_idx: np.ndarray) -> "ARIMAForecaster":
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        p, d, _q = self.order
        pat = np.asarray([str(x) for x in ws.patient[fit_idx]])
        for pid in sorted(set(pat)):
            rows = fit_idx[pat == pid]
            rows = rows[np.argsort(ws.t_time[rows], kind="stable")]
            # One contiguous training series per subject: the anchor value of
            # each fit window in time order. Fit rows are chronologically before
            # val and test by construction of the official split.
            series = np.asarray(ws.last_bg[rows], dtype=np.float64)
            if series.size < 10 * max(p, 1):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(series, order=(p, d, 0),
                                trend="n" if d else "c").fit()
                ar = np.asarray(res.arparams, dtype=np.float64)
                # statsmodels' `trend="c"` models `y_t = c + u_t` with
                # `phi(L) u_t = eps`, so `params[0]` is the process MEAN, not
                # the AR intercept. `_forecast` runs the recursion on raw
                # LEVELS, `y_t = a + sum phi_i y_{t-i}`, whose intercept is
                # `a = c * (1 - sum phi)`. Feeding the mean in directly made the
                # forecast drift by ~c*sum(phi) per step: on an AR(2) with mean
                # 150 the six-step forecast came out at 434.95 where
                # statsmodels' own `forecast(6)` said 149.85.
                const = (0.0 if d else
                         float(res.params[0]) * float(1.0 - ar.sum()))
                self.params[pid] = (const, ar)
            except Exception as exc:       # non-convergence on a short record
                print(f"  ARIMA failed for {pid} ({exc}); falling back to persistence")
        return self

    def _forecast(self, hist: np.ndarray, steps: int, const: float,
                  ar: np.ndarray) -> np.ndarray:
        """Iterate the AR recursion `steps` ahead. hist: (N, H) oldest first."""
        p, d = len(ar), self.order[1]
        if d:
            work = np.diff(hist, axis=1)[:, -p:]          # (N, p) differences
            level = hist[:, -1].copy()
            for _ in range(steps):
                nxt = work[:, ::-1] @ ar                  # most recent lag first
                level = level + nxt
                work = np.concatenate([work[:, 1:], nxt[:, None]], axis=1)
            return level
        work = hist[:, -p:]
        for _ in range(steps):
            nxt = const + work[:, ::-1] @ ar
            work = np.concatenate([work[:, 1:], nxt[:, None]], axis=1)
        return work[:, -1]

    def predict(self, ws, idx: np.ndarray, horizon_steps: Sequence[int]) -> Predictions:
        hist = _lag_matrix(ws, idx)
        last = np.asarray(ws.last_bg[idx], dtype=np.float64)
        pat = np.asarray([str(x) for x in ws.patient[idx]])
        bg = np.repeat(last[:, None], len(horizon_steps), axis=1)
        for pid, (const, ar) in self.params.items():
            m = pat == pid
            if not m.any():
                continue
            for k, steps in enumerate(horizon_steps):
                bg[m, k] = self._forecast(hist[m], int(steps), const, ar)
        # Clip to the assay range: an unstable AR root can otherwise emit values
        # the error grid is undefined on, which would land silently in zone B.
        bg = np.clip(bg, 1.0, 600.0)
        return Predictions(bg=bg, risk=risk_from_forecast(bg))


# --------------------------------------------------------------------------- #
# Neural arms
# --------------------------------------------------------------------------- #
def arm_config(spec: ArmSpec, data_cfg: DataConfig) -> Config:
    """A full `Config` for one arm. Every knob the arm changes lives here.

    Takes a `DataConfig` rather than a cohort so the mapping from an `ArmSpec`
    to a `Config` is available to any runner, not only the Ohio one. An arm's
    spec has to mean the same thing wherever it is fitted -- two runners with
    two mappings is two different experiments sharing a name.
    """
    return Config(
        data=dataclasses.replace(data_cfg, horizon_min=int(spec.horizons[0])),
        model=ModelConfig(
            backbone=spec.backbone,
            n_horizons=len(spec.horizons),
            loss_weighting=spec.loss_weighting,
            lambda_reg=spec.lambda_reg,
            lambda_cls=spec.lambda_cls,
            hypo_asymmetry=spec.hypo_asymmetry,
        ),
        train=TrainConfig(max_epochs=spec.epochs, batch_size=spec.batch_size,
                          lr=spec.lr, seed=spec.seed,
                          early_stop_patience=spec.patience, num_workers=0,
                          monitor=spec.monitor, monitor_mode=spec.monitor_mode),
    )


def _arm_config(spec: ArmSpec, data: CohortData) -> Config:
    """`arm_config` for an Ohio cohort. Kept so existing call sites are unchanged."""
    return arm_config(spec, data.cfg)


def arm_adapt_config(spec: ArmSpec, cfg: Config) -> Config:
    """`cfg` with the arm's personalisation knobs applied.

    Split out of `adapt_and_predict` for the same reason as `arm_config`: the
    LoRA rank, prefix length, step count and learning rate a spec asks for must
    mean the same thing under every protocol that adapts, or the `lora` row of
    one table is not the `lora` row of another.
    """
    return dataclasses.replace(
        cfg,
        adapt=dataclasses.replace(
            Config().adapt, method=spec.adapt_method, lora_rank=spec.lora_rank,
            adapt_prefix_hours=spec.adapt_prefix_hours, steps=spec.adapt_steps,
            lr=spec.adapt_lr),
    )


def _arm_arrays(data: CohortData, spec: ArmSpec, idx: np.ndarray,
                cols: Sequence[int]) -> dict:
    """Frozen-preprocessor tensors for one arm's rows at its horizons.

    The modality mask is applied AFTER scaling, and it zeroes the value channels
    and their observation masks together -- zeroing values but leaving the mask
    set would tell the model "this WAS observed and its value is exactly zero",
    which is strictly worse than absence.
    """
    from .ohio_loader import apply_modality_mask

    from .data import aux_presence
    from .ohio_loader import modality_channel_names

    arrays = data.pre.transform(data.ws, idx)
    # Presence is a property of THIS arm's channels: an arm that zeroes a
    # modality must not be gated open by it, and one that keeps the treatment
    # channels must be gated open by them.
    arrays["aux_present"] = aux_presence(data.ws.x_aux[idx],
                                         modality_channel_names(None),
                                         use=modality_channel_names(spec.modalities))
    for name in ("y_reg", "y_soft"):
        multi = np.ascontiguousarray(arrays[f"{name}_multi"][:, cols], dtype=np.float32)
        arrays[name] = multi[:, 0].copy() if len(cols) == 1 else multi
    if spec.unimodal:
        arrays["x_aux"] = np.ascontiguousarray(arrays["x_aux"][:, :, :0])
    elif spec.modalities is not None:
        arrays["x_aux"] = apply_modality_mask(arrays["x_aux"], list(spec.modalities))
    return arrays


def _loader(data: CohortData, spec: ArmSpec, idx: np.ndarray, cols: Sequence[int],
            shuffle: bool, generator=None):
    from torch.utils.data import DataLoader

    from .ohio_loader import make_dataset

    ds = make_dataset(_arm_arrays(data, spec, idx, cols), list(spec.horizons),
                      style="compat")
    return DataLoader(ds, batch_size=spec.batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=shuffle and len(ds) > spec.batch_size,
                      generator=generator if shuffle else None)


@dataclass
class TrainedArm:
    model: object
    criterion: object
    multimodal: bool
    epochs_run: int
    best_epoch: int
    best_monitor: float
    history: list = field(default_factory=list)
    # Which auxiliary observation masks gated the fusion branch. Recorded
    # because a wrong set here is INVISIBLE in every metric: it silently turns
    # the multimodal model into the CGM-only one while it keeps reporting
    # itself as multimodal. See `ohio_loader.aux_validity_indices`.
    aux_valid_index: tuple[int, ...] = ()
    # Fraction of FIT windows on which the fusion branch is on (`aux_present`).
    aux_valid_frac: float = float("nan")
    aux_present_by_subject: dict = field(default_factory=dict)


def train_neural_arm(data: CohortData, spec: ArmSpec, split=None,
                     verbose: bool = True) -> TrainedArm:
    """Train one neural arm. Mirrors `train.run_fold` step for step.

    Same order of operations, same early-stopping monitor, same evaluation call
    -- only the row indices come from `OhioSplit` instead of from patient folds.
    The pieces that are not split-dependent (`_forward`, `predict`, the models,
    the loss) are imported rather than reimplemented, so an arm here and a run of
    `scripts/train_ohio.py` cannot silently diverge.
    """
    import torch

    from .models import build_model, build_multimodal_model
    from .data import uses_unmasked_channels
    from .ohio_loader import aux_validity_indices, modality_channel_names
    from .train import (_forward, _monitor_value, _require_finite_monitor,
                        pick_device, predict, set_seed)

    split = split if split is not None else data.split
    cols = [list(data.horizons).index(h) for h in spec.horizons]
    cfg = _arm_config(spec, data)

    set_seed(spec.seed)
    device = pick_device()
    g = torch.Generator().manual_seed(spec.seed)

    fit_idx = subsample_rows(data.ws, split.fit, spec.fit_subsample)
    dl_fit = _loader(data, spec, fit_idx, cols, True, g)
    dl_val = _loader(data, spec, split.val, cols, False)

    x0, *_rest, xa0, _ = next(iter(dl_fit))
    multimodal = (not spec.unimodal) and xa0.shape[-1] > 0

    prev = float(data.ws.y_cls_multi[fit_idx][:, cols].mean())
    pos_weight = float((1 - prev) / max(prev, 1e-6))

    if multimodal:
        # Validity comes from THIS ARM's mask channels. Gating on one hard-coded
        # sensor is what made every heart-rate-free ablation arm collapse onto
        # the same numbers -- see `ohio_loader.aux_validity_indices`.
        model, criterion = build_multimodal_model(
            x0.shape[-1], xa0.shape[-1], cfg.model, pos_weight,
            aux_valid_index=aux_validity_indices(spec.modalities),
            always_valid=uses_unmasked_channels(
                modality_channel_names(None),
                use=modality_channel_names(spec.modalities)))
    else:
        model, criterion = build_model(x0.shape[-1], cfg.model, pos_weight)
    model, criterion = model.to(device), criterion.to(device)

    # Model AND criterion, hoisted so the optimiser and the clip cannot drift
    # apart. They did: the optimiser took both, the clip took only the model,
    # so the Kendall log-variances stepped unclipped. Their gradient is
    # `0.5 - exp(-s) * L`, largest early in training when `L` is large --
    # exactly when `grad_clip` is protecting everything else -- and one
    # oversized step there rescales the whole multi-task balance, which is the
    # collapsing task weight `MultiTaskLoss` warns about.
    trainable = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(trainable,
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr,
        total_steps=max(1, cfg.train.max_epochs * max(1, len(dl_fit))))

    sign = 1.0 if spec.monitor_mode == "max" else -1.0
    best, best_state, best_epoch, patience = -np.inf, None, -1, 0
    history: list[dict] = []

    for epoch in range(cfg.train.max_epochs):
        model.train()
        for batch in dl_fit:
            opt.zero_grad(set_to_none=True)
            bg_hat, logit, y_reg, y_soft = _forward(model, batch, device, multimodal)
            loss, _ = criterion(bg_hat, logit, y_reg, y_soft)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            opt.step()
            sched.step()

        bg_hat, risk = predict(model, dl_val, device, multimodal)
        # Model selection on the MEAN of the monitor across horizons: selecting
        # on one would let a joint model quietly sacrifice the other, which is
        # precisely the failure a shared trunk invites.
        reps = score_horizons(data, split.val, Predictions(bg_hat, risk), spec, cols)
        # `nanmean` so one degenerate horizon does not poison a usable mean --
        # but an ALL-nan list still returns nan (with a RuntimeWarning), and
        # that is the case where selection silently stops happening.
        with np.errstate(invalid="ignore"):
            per_h = [_monitor_value(r, spec.monitor) for r in reps.values()]
            val = sign * float(np.nan if all(not np.isfinite(v) for v in per_h)
                               else np.nanmean(per_h))
        _require_finite_monitor(val, spec.monitor, epoch,
                                {f"h{h}": r for h, r in reps.items()})
        history.append({"epoch": epoch, "monitor": float(val * sign)})
        if verbose:
            detail = " ".join(f"h{h}={_monitor_value(r, spec.monitor):.4f}"
                              for h, r in reps.items())
            print(f"  [{spec.name}] epoch {epoch:03d} {spec.monitor}={val * sign:.4f} | {detail}")

        if val > best:
            best, best_epoch, patience = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if verbose:
                    print(f"  [{spec.name}] early stop at epoch {epoch}")
                break

    if best_state is None:
        # Unreachable while at least one epoch runs: the monitor is checked
        # finite above and any finite value beats -inf. An error rather than a
        # silent skip, because scoring an unselected model is exactly what the
        # nan-monitor path used to do.
        raise RuntimeError(
            f"[{spec.name}] no epoch was ever selected "
            f"(max_epochs={cfg.train.max_epochs}); there is no trained model "
            f"to score.")
    model.load_state_dict(best_state)

    # How often the fusion branch is actually ON, over EVERY fit row and per
    # subject. The previous probe read `fit_idx[:2048]` -- in practice the first
    # 2018 subject -- and measured the per-step wearable mask rather than the
    # per-window `aux_present` gate that multiplies the whole fusion term, so it
    # could not see a cohort or a channel group the gate was shutting out.
    valid_idx = tuple(aux_validity_indices(spec.modalities)) if multimodal else ()
    frac = float("nan")
    by_subject: dict[str, float] = {}
    if multimodal:
        from .data import aux_presence
        ap = aux_presence(data.ws.x_aux[fit_idx], modality_channel_names(None),
                          use=modality_channel_names(spec.modalities))
        frac = float(ap.mean())
        pat = np.asarray([str(p) for p in data.ws.patient[fit_idx]])
        by_subject = {p: float(ap[pat == p].mean()) for p in sorted(set(pat))}
        if frac < 1e-6:
            import warnings
            warnings.warn(
                f"arm {spec.name!r}: the fusion branch is gated off on every fit "
                f"window, so this arm is effectively CGM-only. Check its "
                f"modality set against the cohort's sensors.", stacklevel=2)

    return TrainedArm(model=model, criterion=criterion, multimodal=multimodal,
                      epochs_run=len(history), best_epoch=best_epoch,
                      best_monitor=float(best * sign), history=history,
                      aux_valid_index=valid_idx, aux_valid_frac=frac,
                      aux_present_by_subject=by_subject)


def neural_predictions(trained: TrainedArm, data: CohortData, spec: ArmSpec,
                       idx: np.ndarray) -> Predictions:
    from .train import pick_device, predict

    cols = [list(data.horizons).index(h) for h in spec.horizons]
    dl = _loader(data, spec, idx, cols, False)
    bg, risk = predict(trained.model, dl, pick_device(), trained.multimodal)
    return Predictions(bg, risk)


def lgbm_predictions(data: CohortData, spec: ArmSpec, split, verbose: bool = False):
    """One LightGBM pair (regressor + classifier) per horizon.

    Trees have no shared trunk, so "multi-horizon" here means independent models
    -- which is the honest tabular baseline, and also the reason the comparison
    against the joint-trunk network is interesting rather than rigged.
    """
    from .baselines.lgbm_baseline import LGBMBaseline

    cfg = _arm_config(spec, data)
    calib, test = [], []
    for h in spec.horizons:
        c = list(data.horizons).index(h)
        # `LGBMBaseline` reads ws.y_reg / ws.y_cls_hard, so point them at this
        # horizon's column. `dataclasses.replace` shares every other array by
        # reference -- `ws.subset(arange(N))` would fancy-index all of them and
        # transiently double a 400 MB window set for nothing.
        view = dataclasses.replace(
            data.ws,
            y_reg=np.ascontiguousarray(data.ws.y_reg_multi[:, c]),
            y_cls_hard=np.ascontiguousarray(data.ws.y_cls_multi[:, c]),
            y_min=np.ascontiguousarray(data.ws.y_min_multi[:, c]))
        m = LGBMBaseline(cfg=dataclasses.replace(
            cfg, data=dataclasses.replace(cfg.data, horizon_min=int(h))),
            n_estimators=400).fit(view, split.fit, split.val, verbose=verbose)
        # Early stopping reads `split.val`; conformal reads `split.calib`. The
        # two must not be the same rows -- a booster stopped at the round that
        # scored best on a set has optimistically small residuals there, and
        # calibrating on them undercovers.
        calib.append(m.predict(view, split.calib))
        test.append(m.predict(view, split.test))
    stack = lambda parts: Predictions(  # noqa: E731
        np.stack([p[0] for p in parts], axis=1),
        np.stack([p[1] for p in parts], axis=1))
    return stack(calib), stack(test)


# --------------------------------------------------------------------------- #
# Scoring -- one path, every arm
# --------------------------------------------------------------------------- #
def score_horizons(data: CohortData, idx: np.ndarray, pred: Predictions,
                   spec: ArmSpec, cols: Sequence[int],
                   alarm: bool = False,
                   thresholds: dict[int, dict[float, float]] | None = None,
                   ) -> dict[int, EvalReport]:
    """`EvalReport` per horizon on the given rows.

    `alarm=False` during training: the lead-time and FAPD block needs an anchor
    timestamp per row and an explicit horizon, costs a few hundred milliseconds,
    and is never the early-stopping monitor. It is computed once, at scoring
    time, where it is actually reported.
    """
    ws = data.ws
    out: dict[int, EvalReport] = {}
    for k, c in enumerate(cols):
        h = int(data.horizons[c])
        bg, risk = pred.col(k)
        out[h] = evaluate(
            ws.y_reg_multi[idx, c], bg, ws.y_cls_multi[idx, c], risk,
            ws.last_bg[idx], ws.patient[idx], data.cfg.sample_period_min,
            t_time=ws.t_time[idx] if alarm else None,
            horizon_min=h if alarm else None,
            alarm_thresholds=(thresholds or {}).get(h) if alarm else None,
        )
    return out


def calib_thresholds(data: CohortData, split, calib_pred: Predictions,
                     cols: Sequence[int]) -> dict[int, dict[float, float]]:
    """Per-horizon alarm thresholds fitted on `split.calib`, never on test.

    Rows a caller could not predict (the LoRA arm masks its adaptation prefix
    with nan) are dropped by `calibration_thresholds` before fitting.
    """
    out: dict[int, dict[float, float]] = {}
    for k, c in enumerate(cols):
        h = int(data.horizons[c])
        out[h] = calibration_thresholds(data.ws.y_cls_multi[split.calib, c],
                                        calib_pred.risk[:, k])
    return out


def _conformal_block(data: CohortData, split, calib_pred: Predictions,
                     test_pred: Predictions, k: int, c: int,
                     alpha: float, cal_mask: np.ndarray | None = None) -> dict:
    """Split and Mondrian conformal, calibrated on CALIB and scored on TEST.

    NOT on `val`. Split conformal's finite-sample guarantee assumes the
    predictor is fixed independently of the calibration set, and `val` is what
    early stopping selects the checkpoint on (`train_neural_arm`,
    `lgbm_predictions`). Calibrating there meant the residuals came from the
    rows the checkpoint had been chosen to fit best: `qhat` too small, intervals
    too narrow, coverage below nominal -- the failure direction `conformal.py`'s
    header calls the one that matters clinically. Nothing in the artefact showed
    it, because `calibrated_on`, `n_calib` and `n_effective` were all still
    correct descriptions of the wrong rows.

    `split.calib` is a disjoint, embargoed region carved from the same held-out
    tail (see `ohio_loader.official_temporal_split`). No test label participates
    either way: qhat is a quantile of |y - yhat| on calibration rows and is
    applied to test rows as a fixed offset (Mondrian, as a stratum lookup).

    `cal_mask` selects which calibration rows may be used. It exists for the
    personalised arm: a subject's adaptation prefix is drawn from their own
    training file, and any calibration row inside that prefix is training data
    for THAT subject's model. Its residual would be optimistically small, and
    including it would shrink qhat further. The default is every row, which is
    correct for every population-fitted arm.
    """
    ws = data.ws
    if split.calib.size == 0:
        # An empty calibration region is a SPLIT construction bug, not a data
        # condition, and falling back to `val` would silently reinstate exactly
        # the leak this function was changed to close.
        raise ValueError(
            "conformal calibration has no rows: split.calib is empty. It is "
            "carved from the held-out tail by "
            "`ohio_loader.official_temporal_split(calib_frac=...)`; a split "
            "built by older code, or with calib_frac=0, has none. Rebuild the "
            "split rather than calibrating on split.val -- val is the "
            "early-stopping set, and calibrating there undercovers.")
    m = (np.ones(split.calib.size, dtype=bool) if cal_mask is None
         else np.asarray(cal_mask, dtype=bool))
    cal_rows = split.calib[m]
    if cal_rows.size == 0:
        # Every conformal interval would come back unbounded, which is a valid
        # answer to a question nobody asked. A mask that admits nothing is a
        # caller bug, not a data condition -- fail here rather than emit a table
        # of `inf` widths that reads as a coverage result.
        raise ValueError(
            f"conformal calibration got 0 of {split.calib.size} calibration "
            f"rows. The cal_mask admitted nothing; check how it was derived.")
    y_cal = np.asarray(ws.y_reg_multi[cal_rows, c], dtype=np.float64)
    y_te = np.asarray(ws.y_reg_multi[split.test, c], dtype=np.float64)
    p_cal, p_te = calib_pred.bg[m, k], test_pred.bg[:, k]

    sc = SplitConformal.calibrate(y_cal, p_cal, alpha,
                                  patient_calib=ws.patient[cal_rows])
    mc = MondrianConformal.calibrate(y_cal, p_cal, alpha,
                                     patient_calib=ws.patient[cal_rows])
    rep_s = sc.evaluate(y_te, p_te, ws.patient[split.test])
    rep_m = mc.evaluate(y_te, p_te, ws.patient[split.test])

    # The locally adaptive variant, on the same rows and the same predictions,
    # so the three constructions differ only in how the interval is built. Its
    # difficulty estimate is a function of the INPUT WINDOW alone (recent rate
    # of change), so including it here adds no label to the calibration path.
    sig_cal = rate_of_change_difficulty(ws, cal_rows)
    sig_te = rate_of_change_difficulty(ws, split.test)
    nc = NormalizedConformal.calibrate(y_cal, p_cal, sig_cal, alpha,
                                       patient_calib=ws.patient[cal_rows])
    rep_n = nc.evaluate(y_te, p_te, sig_te, ws.patient[split.test])

    def block(rep, obj) -> dict:
        hypo = rep.stratum(0)
        return {
            "marginal_coverage": rep.empirical_coverage,
            "mean_width": rep.mean_width,
            "median_width": rep.median_width,
            "hypo_stratum_coverage": None if hypo is None else hypo.coverage,
            "hypo_stratum_width": None if hypo is None else hypo.mean_width,
            "hypo_stratum_n": 0 if hypo is None else hypo.n,
            # All three strata, not just the hypoglycemic one. `conditionally
            # _valid` is a claim about every cell, so a reader who sees it fail
            # needs to be able to see WHICH cell failed without re-running.
            "by_stratum": {
                c.name: {"coverage": c.coverage, "mean_width": c.mean_width,
                         "median_width": c.median_width, "n": c.n}
                for c in rep.coverage_by_stratum.values()},
            "by_clinical_range": {k: {"coverage": v[0], "n": v[1]}
                                  for k, v in rep.coverage_by_range.items()},
            "severe_hypo_coverage": rep.coverage_by_range.get(
                "severe hypo (<54)", (None, 0))[0],
            "worst_patient_coverage": rep.worst_patient_coverage,
            "worst_patient": rep.worst_patient,
            "conditionally_valid": bool(rep.is_conditionally_valid()),
            # Stratum-conditional validity says nothing about the failure mode
            # `conformal.py`'s own header calls decisive -- a cohort can hold
            # 90% while an individual sits at 70%. Reported as its own flag so a
            # reader is not left inferring it from a number two columns away.
            "patient_conditionally_valid": bool(rep.is_patient_conditionally_valid()),
            "n_calib": int(getattr(obj, "n_calib", 0)),
            # Overlapping windows make calibration scores strongly
            # autocorrelated, so `n_calib` overstates how much independent
            # information qhat rests on -- on this cohort by about 16x. This is
            # REPORTED, never used to reweight or to inflate qhat: the
            # finite-sample guarantee is an exact statement about an order
            # statistic of exchangeable scores, and weighting them would trade a
            # theorem for a heuristic. What it buys is calibration of the
            # READER's confidence in the coverage figure.
            "n_effective": float(getattr(obj, "n_effective", float("nan"))),
            "calibrated_on": list(getattr(obj, "calibrated_on", ())),
        }

    out = {"alpha": alpha, "split": block(rep_s, sc),
           "normalized": block(rep_n, nc), "mondrian": block(rep_m, mc)}
    out["normalized"]["qhat"] = float(nc.qhat)
    out["mondrian"]["qhat_by_stratum"] = {
        str(s): mc.qhat_for(s) for s in range(3)}
    out["mondrian"]["fell_back"] = list(mc.fell_back)
    out["mondrian"]["n_effective_by_stratum"] = {
        str(k): float(v) for k, v in mc.n_effective_by_stratum.items()}
    out["mondrian"]["n_by_stratum"] = {
        str(k): int(v) for k, v in mc.n_by_stratum.items()}
    out["mondrian"]["mean_set_measure"] = float(np.mean(mc.set_measure(p_te)))
    out["split"]["qhat"] = float(sc.qhat)
    out["n_calib_rows_available"] = int(split.calib.size)
    out["n_calib_rows_used"] = int(cal_rows.size)
    # Named so a reader of the artefact can confirm, without re-running,
    # that calibration and model selection came from disjoint regions.
    out["calibrated_on_split"] = "calib"
    out["n_selection_rows"] = int(split.val.size)
    return out


def horizon_block(rep: EvalReport, y_true: np.ndarray, bg_hat: np.ndarray) -> dict:
    """One horizon's result block: the shape every table column indexes into.

    Extracted from `score_arm` so a runner on another cohort emits the SAME
    structure under the SAME key names rather than a plausible-looking
    rearrangement of it. `Column.path` values like `regression.rmse` and
    `clinical.zone_ab_pct` are resolved against this dict, so two runners that
    built it separately would silently produce tables whose columns mean
    different things -- and a reader has no way to see that from the output.

    Conformal is deliberately NOT here: it needs a calibration split, which is
    a property of the protocol rather than of a horizon's predictions.
    """
    y_true = np.asarray(y_true, np.float64)
    bg_hat = np.asarray(bg_hat, np.float64)
    clin = clinical_report(y_true, bg_hat)
    # The hypoglycemic band as a clinician reads it -- everything below 70,
    # not `clinical_report`'s two sub-bands. Reported alongside them because
    # the kappa sweep trades overall RMSE against exactly this number, and
    # splitting it at 54 makes the trade harder to see, not easier.
    below = y_true < HYPO_THRESHOLD_MGDL
    rmse_lt70 = (float(np.sqrt(np.mean((bg_hat[below] - y_true[below]) ** 2)))
                 if below.any() else float("nan"))
    return {
        "regression": {
            "rmse": rep.rmse, "mae": rep.mae, "mard": rep.mard,
            "time_lag_min": rep.time_lag_min,
            "rmse_persistence": rep.rmse_persistence,
            "time_lag_persistence_min": rep.time_lag_persistence_min,
        },
        "classification": {
            "roc_auc": rep.roc_auc, "pr_auc": rep.pr_auc,
            "prevalence": rep.pr_auc_prevalence,
            "sensitivity_at_5pct_fpr": rep.sensitivity_at_5pct_fpr,
            "sensitivity_at_10pct_fpr": rep.sensitivity_at_10pct_fpr,
            "best_f1": rep.best_f1,
            "pr_auc_onset": rep.pr_auc_onset,
            "roc_auc_onset": rep.roc_auc_onset,
            "onset_prevalence": rep.onset_prevalence,
            "n_onset_windows": rep.n_onset_windows,
        },
        "alarm": {
            "fapd_at_5pct_fpr": rep.fapd_at_5pct_fpr,
            "fapd_at_10pct_fpr": rep.fapd_at_10pct_fpr,
            "mean_lead_time_min": rep.mean_lead_time_min,
            "median_lead_time_min": rep.median_lead_time_min,
            "lead_time_detection_rate": rep.lead_time_detection_rate,
            "n_hypo_events": rep.n_hypo_events,
            "monitored_days": rep.monitored_days,
            "mean_onset_lead_min": rep.mean_onset_lead_min,
            "median_onset_lead_min": rep.median_onset_lead_min,
            "onset_detection_rate": rep.onset_detection_rate,
            "threshold_source": rep.threshold_source,
            "threshold_at_5pct_fpr": rep.threshold_at_5pct_fpr,
            "threshold_at_10pct_fpr": rep.threshold_at_10pct_fpr,
            "realized_fpr_at_5pct": rep.realized_fpr_at_5pct,
            "realized_fpr_at_10pct": rep.realized_fpr_at_10pct,
        },
        "clinical": {
            "zone_pct": clin.zone_pct,
            "zone_ab_pct": clin.zone_ab_pct,
            "zone_de_n": clin.zone_de_n,
            "parkes_ab_pct": clin.parkes_ab_pct,
            "band_rmse": clin.band_rmse,
            "band_n": clin.band_n,
            "rmse_below_70": rmse_lt70,
            "n_below_70": int(below.sum()),
            "hypo_band_bias": clin.hypo_band_bias,
        },
        "n": rep.n,
        "n_positive": rep.n_positive,
    }


def score_arm(spec: ArmSpec, data: CohortData, split, calib_pred: Predictions,
              test_pred: Predictions, alpha: float = 0.1,
              extra: dict | None = None,
              cal_mask: np.ndarray | None = None) -> dict:
    """Every number the paper reports for one arm, on `split.test`.

    `calib_pred` is this arm's prediction on `split.calib` -- the conformal
    calibration region, which is disjoint from and embargoed against the
    `split.val` rows early stopping used. It feeds `_conformal_block` and
    nothing else.
    """
    ws = data.ws
    cols = [list(data.horizons).index(h) for h in spec.horizons]
    if split.calib.size == 0:
        # Same refusal `_conformal_block` makes, raised BEFORE thresholds are
        # fitted from the (empty) calibration region.
        raise ValueError(
            "conformal calibration has no rows: split.calib is empty. Alarm "
            "thresholds and conformal quantiles are both fitted there.")
    thresholds = calib_thresholds(data, split, calib_pred, cols)
    reports = score_horizons(data, split.test, test_pred, spec, cols, alarm=True,
                             thresholds=thresholds)

    per_horizon: dict[str, dict] = {}
    for k, c in enumerate(cols):
        h = int(data.horizons[c])
        per_horizon[str(h)] = {
            **horizon_block(reports[h], ws.y_reg_multi[split.test, c],
                            test_pred.bg[:, k]),
            "conformal": _conformal_block(data, split, calib_pred, test_pred,
                                          k, c, alpha, cal_mask),
        }

    return {
        "arm": spec.name,
        "family": spec.family,
        "spec": spec.as_dict(),
        "subjects": sorted({str(p) for p in ws.patient[split.test]}),
        "n_fit": int(split.fit.size), "n_val": int(split.val.size),
        "n_calib": int(split.calib.size), "n_test": int(split.test.size),
        "parkes_backend": bool(parkes_available()),
        # The band this arm's classification targets were built with, and where
        # it came from. Without it a table row cannot be read: a +/-6.3 mg/dL
        # band that was FLOORED from a device specification and one that was
        # MEASURED at 9% MARD are different claims, and Ablation E's whole
        # point is the difference between them.
        "soft_label": data.pre.labeler.as_dict(),
        "per_horizon": per_horizon,
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
# One arm, end to end
# --------------------------------------------------------------------------- #
def save_predictions(path: str, spec: ArmSpec, data: CohortData, split,
                     test_pred: Predictions,
                     calib_pred: Predictions | None = None) -> str:
    """Dump one arm's test predictions and references next to its metrics.

    Figures need the point cloud, not the summary: a Clarke grid is a scatter of
    (reference, prediction) pairs and cannot be reconstructed from an RMSE. The
    alternative -- retraining inside the plotting script -- would make every
    figure a fresh experiment that could silently disagree with the table above
    it, which is the failure this whole harness exists to prevent.

    Reference values are stored alongside, so a figure is a pure function of one
    file and cannot pick up a different split than the one that was scored.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [list(data.horizons).index(h) for h in spec.horizons]
    np.savez_compressed(
        path,
        arm=np.asarray(spec.name),
        horizons=np.asarray(spec.horizons, dtype=np.int64),
        bg_hat=test_pred.bg.astype(np.float32),
        risk=test_pred.risk.astype(np.float32),
        y_reg=np.asarray(data.ws.y_reg_multi[split.test][:, cols], np.float32),
        y_cls=np.asarray(data.ws.y_cls_multi[split.test][:, cols], np.float32),
        last_bg=np.asarray(data.ws.last_bg[split.test], np.float32),
        patient=np.asarray([str(p) for p in data.ws.patient[split.test]]),
        t_time=data.ws.t_time[split.test].astype("datetime64[ns]").astype(np.int64),
        # Calibration rows, so conformal and threshold analyses can be re-run
        # offline (e.g. a block bootstrap over calibration draws) without
        # retraining. `sigma_*` is the normalised-conformal difficulty.
        **({} if calib_pred is None else dict(
            calib_bg_hat=calib_pred.bg.astype(np.float32),
            calib_risk=calib_pred.risk.astype(np.float32),
            calib_y_reg=np.asarray(data.ws.y_reg_multi[split.calib][:, cols], np.float32),
            calib_y_cls=np.asarray(data.ws.y_cls_multi[split.calib][:, cols], np.float32),
            calib_patient=np.asarray([str(p) for p in data.ws.patient[split.calib]]),
            calib_t_time=data.ws.t_time[split.calib].astype("datetime64[ns]").astype(np.int64),
            calib_sigma=rate_of_change_difficulty(data.ws, split.calib).astype(np.float32),
            test_sigma=rate_of_change_difficulty(data.ws, split.test).astype(np.float32),
        )),
    )
    return path


def run_arm(spec: ArmSpec, data: CohortData, alpha: float = 0.1,
            base: TrainedArm | None = None, base_spec: ArmSpec | None = None,
            verbose: bool = True, predictions_dir: str | None = None
            ) -> tuple[dict, TrainedArm | None]:
    """Fit, predict and score one arm. Returns (result dict, trained model|None).

    The trained object is handed back so a driver can adapt it (the LoRA arm)
    or export it, without this function needing to know which.
    """
    data = arm_cohort(data, spec)
    split = restrict_split(data, spec)
    if split.fit.size == 0 or split.test.size == 0:
        raise ValueError(f"arm {spec.name!r} has an empty fit or test split "
                         f"(cohorts={spec.cohorts})")
    if split.calib.size == 0:
        raise ValueError(
            f"arm {spec.name!r} has an empty conformal calibration split "
            f"(cohorts={spec.cohorts}). `official_temporal_split` carves it "
            f"from the held-out tail; a cohort restriction that keeps no "
            f"calibration rows cannot support a coverage claim.")

    cols = [list(data.horizons).index(h) for h in spec.horizons]
    t0 = time.time()
    trained: TrainedArm | None = None
    extra: dict = {}

    if spec.family == "persistence":
        calib_pred = persistence_predictions(data.ws, split.calib, spec.horizons)
        test_pred = persistence_predictions(data.ws, split.test, spec.horizons)

    elif spec.family == "ridge_ar":
        ar = RidgeAR(target=spec.target).fit(data.ws, split.fit, spec.horizons, cols)
        calib_pred = ar.predict(data.ws, split.calib)
        test_pred = ar.predict(data.ws, split.test)

    elif spec.family == "arima":
        steps = [int(h) // data.cfg.sample_period_min for h in spec.horizons]
        am = ARIMAForecaster().fit(data.ws, split.fit)
        extra["arima_fitted_subjects"] = sorted(am.params)
        calib_pred = am.predict(data.ws, split.calib, steps)
        test_pred = am.predict(data.ws, split.test, steps)

    elif spec.family == "lgbm":
        calib_pred, test_pred = lgbm_predictions(data, spec, split)

    elif spec.family == "lora":
        if base is None or base_spec is None:
            raise ValueError("the lora arm needs the base arm it adapts "
                             "(pass base= and base_spec=)")
        calib_pred, test_pred, extra = adapt_and_predict(
            spec, base, base_spec, data, split, verbose=verbose)

    elif spec.is_neural:
        trained = train_neural_arm(data, spec, split, verbose=verbose)
        # `split.calib`, never `split.val`: `train_neural_arm` early-stopped on
        # val, so val residuals are biased low and would shrink qhat.
        calib_pred = neural_predictions(trained, data, spec, split.calib)
        test_pred = neural_predictions(trained, data, spec, split.test)
        extra["training"] = {"epochs_run": trained.epochs_run,
                             "best_epoch": trained.best_epoch,
                             "best_monitor": trained.best_monitor,
                             "multimodal": trained.multimodal,
                             "aux_valid_index": list(trained.aux_valid_index),
                             "aux_valid_frac": trained.aux_valid_frac,
                             "aux_present_by_subject": trained.aux_present_by_subject}
    else:                                    # unreachable: ArmSpec validates
        raise ValueError(f"unhandled family {spec.family!r}")

    result = score_arm(spec, data, split, calib_pred, test_pred, alpha, extra,
                       cal_mask=extra.pop("_cal_mask", None))
    result["wall_clock_s"] = round(time.time() - t0, 1)
    if predictions_dir:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in spec.name)
        result["predictions_file"] = save_predictions(
            os.path.join(predictions_dir, f"{safe}.npz"), spec, data, split,
            test_pred, calib_pred)
    if verbose:
        h0 = str(spec.horizons[0])
        r = result["per_horizon"][h0]
        print(f"  [{spec.name}] PH={h0}: RMSE {r['regression']['rmse']:.3f} | "
              f"PR-AUC {r['classification']['pr_auc']:.4f} | "
              f"A+B {r['clinical']['zone_ab_pct']:.2f}% | "
              f"FAPD@5% {r['alarm']['fapd_at_5pct_fpr']:.2f}/day | "
              f"{result['wall_clock_s']:.0f}s")
    return result, trained


# --------------------------------------------------------------------------- #
# The personalised arm, scored on the same test rows as everything else
# --------------------------------------------------------------------------- #
def adapt_and_predict(spec: ArmSpec, base: TrainedArm, base_spec: ArmSpec,
                      data: CohortData, split, verbose: bool = True
                      ) -> tuple[Predictions, Predictions, dict]:
    """Adapt the population model per subject, then predict on the SHARED rows.

    WHY NOT JUST REPORT `run_ohio_personalization`'s NUMBERS
    --------------------------------------------------------
    That function scores on the prefix protocol's evaluation region -- the tail
    of the training file plus the whole test file -- which is the right set for
    a paired before/after claim, and the WRONG set for a table whose other rows
    are scored on the official test file alone. Mixing the two would make the
    LoRA row incomparable to every other row in the same table while looking
    like it belonged there.

    So the adaptation is unchanged (same prefix, same embargo, same causality
    assertion) and only the SCORED rows differ: each subject's adapted model is
    run over that subject's `split.test` rows, which are a subset of its own
    evaluation region, and the pieces are reassembled in `split.test` order.

    Calibration rows are handled the same way, with one extra guard: a
    calibration row that falls inside a subject's adaptation prefix is training
    data for that subject's model, so it is excluded via the returned mask
    rather than quietly shrinking qhat.

    Note which region is predicted here: `split.calib`, not `split.val`. The
    base model was early-stopped on val, so val residuals are biased low for
    every subject before adaptation even begins -- calibrating there would stack
    a selection bias on top of the adaptation bias this mask exists to remove.
    """
    import torch

    from .extension import adapt_ohio_subject
    from .train import pick_device

    device = pick_device()
    K = len(spec.horizons)
    cols = [list(data.horizons).index(h) for h in spec.horizons]
    cfg = arm_adapt_config(spec, _arm_config(spec, data))

    cal_bg = np.full((split.calib.size, K), np.nan)
    cal_risk = np.full((split.calib.size, K), np.nan)
    test_bg = np.full((split.test.size, K), np.nan)
    test_risk = np.full((split.test.size, K), np.nan)
    cal_mask = np.zeros(split.calib.size, dtype=bool)

    pat_cal = np.asarray([str(p) for p in data.ws.patient[split.calib]])
    pat_test = np.asarray([str(p) for p in data.ws.patient[split.test]])
    subjects = sorted(set(pat_test))
    per_subject, skipped = {}, []

    for pid in subjects:
        r = adapt_ohio_subject(
            base.model, data.ws, split, pid, data.pre, base.criterion, cfg,
            method=spec.adapt_method, prefix_hours=spec.adapt_prefix_hours,
            lora_rank=spec.lora_rank, steps=spec.adapt_steps, lr=spec.adapt_lr,
            keep_model=True,
        )
        if r is None:
            # No prefix worth adapting on: fall back to the population model so
            # the row still covers every subject, and record which ones.
            skipped.append(pid)
            model = base.model
            ps = None
        else:
            # The split `adapt_ohio_subject` ACTUALLY used, carried on the
            # result. Re-deriving it here (as this used to) silently took
            # `ohio_prefix_embargo_split`'s own defaults for `embargo_min`,
            # `eval_scope` and `prefix_mode` instead of the ones the adaptation
            # ran under, so the exclusion mask could describe a different
            # prefix than the model was fitted on.
            model, ps = r.model, r.prefix_split
            per_subject[pid] = {
                "n_adapt_windows": r.n_adapt_windows,
                "n_trainable": r.n_trainable,
                "prefix_hours": r.prefix_hours,
                "embargo_min": r.embargo_min,
            }

        # Which validation rows this subject's adapted model may calibrate on.
        #
        # The exclusion is the ADAPTATION PREFIX and its embargo: a validation
        # row in there is training data for THAT subject's model, so its
        # residual would be optimistically small and would shrink qhat --
        # undercoverage, the failure direction that matters.
        #
        # Derived from `adapt | embargoed` rather than from `ps.eval`. Using
        # `ps.eval` was an accident of the old evaluation scope, where it
        # happened to contain the whole training tail and so acted as a proxy
        # for "outside the prefix". Under the out-of-sample scope `ps.eval` is
        # the test file alone and contains NO validation rows, so that proxy
        # silently emptied the calibration set and every conformal interval for
        # the LoRA arm came back unbounded.
        excluded = (set(ps.adapt.tolist()) | set(ps.embargoed.tolist())
                    if ps is not None else set())
        for rows_all, pat, bg_out, risk_out, is_cal in (
                (split.calib, pat_cal, cal_bg, cal_risk, True),
                (split.test, pat_test, test_bg, test_risk, False)):
            where = np.flatnonzero(pat == pid)
            if where.size == 0:
                continue
            sub_spec = dataclasses.replace(spec, family=base_spec.family,
                                           unimodal=base_spec.unimodal,
                                           modalities=base_spec.modalities)
            dl = _loader(data, sub_spec, rows_all[where], cols, False)
            from .train import predict as _predict
            bg, risk = _predict(model, dl, device, base.multimodal)
            bg, risk = _as_column(bg), _as_column(risk)
            bg_out[where], risk_out[where] = bg, risk
            if is_cal and ps is not None:
                cal_mask[where] = ~np.isin(rows_all[where], list(excluded))
            elif is_cal:
                cal_mask[where] = True          # unadapted: every row is valid
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        if verbose:
            print(f"    adapted {pid}: {per_subject.get(pid, {}).get('n_trainable', 0)} "
                  f"trainable params")

    extra = {
        "personalisation": {
            "method": spec.adapt_method, "lora_rank": spec.lora_rank,
            "prefix_hours": spec.adapt_prefix_hours, "steps": spec.adapt_steps,
            "base_arm": base_spec.name,
            "per_subject": per_subject,
            "subjects_without_prefix": skipped,
            "n_calib_rows_excluded_from_calibration": int((~cal_mask).sum()),
        },
        "_cal_mask": cal_mask,
    }
    return Predictions(cal_bg, cal_risk), Predictions(test_bg, test_risk), extra


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
class _NumpyEncoder(json.JSONEncoder):
    """numpy scalars and arrays are not JSON-serialisable; nan/inf are not
    valid JSON either. Both are mapped to something a reader can act on rather
    than left to blow up at write time, hours into a run."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not np.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _clean(obj):
    """Recursively replace non-finite floats with None so the JSON is valid."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer, np.ndarray, np.bool_)):
        return _clean(json.loads(json.dumps(obj, cls=_NumpyEncoder)))
    return obj


def write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(payload), f, indent=2, cls=_NumpyEncoder)
    print(f"wrote {path}")
    return path


def dig(d: dict, path: str, default=None):
    """Dotted lookup: dig(row, "regression.rmse")."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class Column:
    """One column of a results table: where the number lives and how to read it.

    `best` says which direction is better, and is what lets the table bold a
    winner. Left as None for columns where "better" is not defined -- interval
    width and false alarms per day are trade-offs against coverage and
    sensitivity, and bolding the smallest would assert a preference the data
    does not support.
    """
    label: str
    path: str
    fmt: str = "{:.3f}"
    best: Literal["min", "max"] | None = None


SOTA_COLUMNS: tuple[Column, ...] = (
    Column("RMSE", "regression.rmse", "{:.2f}", "min"),
    Column("MAE", "regression.mae", "{:.2f}", "min"),
    Column("MARD (\\%)", "regression.mard", "{:.2f}", "min"),
    Column("Lag (min)", "regression.time_lag_min", "{:.1f}", "min"),
    Column("A+B (\\%)", "clinical.zone_ab_pct", "{:.2f}", "max"),
    Column("D+E ($n$)", "clinical.zone_de_n", "{:.0f}", "min"),
    Column("Parkes A+B (\\%)", "clinical.parkes_ab_pct", "{:.2f}", "max"),
    Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"),
    Column("Se@5\\%", "classification.sensitivity_at_5pct_fpr", "{:.3f}", "max"),
    Column("Se@10\\%", "classification.sensitivity_at_10pct_fpr", "{:.3f}", "max"),
    Column("FAPD@5\\%", "alarm.fapd_at_5pct_fpr", "{:.2f}", None),
    # Lead time to ONSET (first reading < 70), counting only alerts anchored at
    # a reading that is not already hypoglycemic. The nadir-referenced lead this
    # column used to carry is saturated: persistence reaches its maximum.
    Column("Onset lead (min)", "alarm.mean_onset_lead_min", "{:.1f}", "max"),
    Column("Cov.", "conformal.split.marginal_coverage", "{:.3f}", None),
    Column("Hypo cov.", "conformal.mondrian.hypo_stratum_coverage", "{:.3f}", "max"),
    # Width is the split-conformal interval width, the same construction as the
    # "Cov." column beside it; "Hypo cov." is the Mondrian < 70 stratum, and the
    # caption (`sota_caption`) says so. It used to read the Mondrian width.
    Column("Width", "conformal.split.mean_width", "{:.1f}", None),
)


def sota_caption(horizon: int, n_subjects: int, n_test: int,
                 alpha: float) -> str:
    r"""The benchmark table's caption, in one place.

    It lived in `run_paper_benchmarks.py` and again in
    `rebuild_paper_tables.py`, which is the arrangement that let Ablation E's
    caption drift into a hard LaTeX error in one copy and not the other. The
    ablation captions were moved into their driver as data for that reason;
    this is the same fix for the benchmark table.

    The classification caveat is load-bearing rather than decorative: three arms
    have no classification head and are scored through a monotone map of a
    forecast of `g(t+PH)`, while the label is the window minimum, so their PR-AUC
    and sensitivity columns are lower bounds and Ablation F measures by how much.
    The caption states that in one clause and points at the appendix section that
    spells it out. It used to spell it out here, which cost six full-width lines
    on the most crowded page of an 8-page body; the caveat is no less binding for
    being stated once in prose instead of once per table.
    """
    return (f"OhioT1DM official test split, PH = {horizon} min "
            f"({n_subjects} subjects, $n = {n_test}$ windows). "
            f"Coverage and width are split-conformal at $\\alpha = {alpha}$, "
            f"calibrated on the disjoint calibration split; hypo coverage is "
            f"the Mondrian $<70$ mg/dL stratum. "
            f"Persistence, Ridge-AR and ARIMA have no classification head, so "
            f"their PR-AUC, Se@FPR and FAPD entries are \\emph{{lower bounds}} "
            f"(Appendix~\\ref{{app:benchmarks}}).")


_TEX_UNSAFE = (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
               ("_", r"\_"), ("#", r"\#"), ("{", r"\{"), ("}", r"\}"),
               ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"))


def _escape_latex(text: str) -> str:
    r"""Escape LaTeX specials OUTSIDE math mode, and pass math through verbatim.

    Arm names are deliberately authored with math in them -- `$\kappa$ = 3`,
    `Joint, fixed $\lambda=1$` -- because that is what the table should print.
    A blanket escaper turns those into the literal characters `$\kappa$`, which
    still compiles and still looks like a table, so the damage is invisible
    until someone reads the rendered PDF. It also has to keep escaping the
    genuinely unsafe ones: an arm named `CausalCGM-MT + LoRA r=4` is fine, and one
    with an `_` or a bare `&` in it would break the document.

    So the string is split on `$...$` spans: text segments are escaped, math
    segments are trusted. An unpaired `$` is treated as text and escaped, which
    is the safe reading -- a stray dollar sign that opens math mode swallows the
    rest of the row.
    """
    parts = text.split("$")
    if len(parts) % 2 == 0:          # unbalanced: no math here, escape it all
        parts, math_spans = [text], False
    else:
        math_spans = True

    out = []
    for i, seg in enumerate(parts):
        if math_spans and i % 2 == 1:
            out.append(f"${seg}$")   # math, verbatim
            continue
        for a, b in _TEX_UNSAFE:
            seg = seg.replace(a, b)
        if not math_spans:
            seg = seg.replace("$", r"\$")
        out.append(seg)
    return "".join(out)


_MISSING = object()


def _cells(rows: Sequence[dict], columns: Sequence[Column], horizon: int):
    """(values, formatted, winner index) per column.

    A path is looked up in the horizon's block first and then at the arm level,
    so a table can carry a column that is a property of the ARM rather than of a
    horizon -- the soft-label band, for instance, which is one number per arm and
    is the independent variable of Ablation E. Horizon-first keeps every existing
    column resolving exactly as before.
    """
    out = []
    for col in columns:
        vals = []
        for r in rows:
            v = dig(r["per_horizon"].get(str(horizon), {}), col.path, _MISSING)
            vals.append(dig(r, col.path) if v is _MISSING else v)
        num = [v if isinstance(v, (int, float)) and v is not None
               and np.isfinite(v) else None for v in vals]
        txt = [("--" if v is None else col.fmt.format(v)) for v in num]
        win = None
        if col.best is not None and any(v is not None for v in num):
            finite = [(i, v) for i, v in enumerate(num) if v is not None]
            win = (min if col.best == "min" else max)(finite, key=lambda t: t[1])[0]
        out.append((num, txt, win))
    return out


def latex_table(rows: Sequence[dict], columns: Sequence[Column], horizon: int,
                caption: str, label: str, row_label: str = "Model",
                name_key: str = "arm", bold_best: bool = True,
                wide: bool = True) -> str:
    r"""A booktabs table, one row per arm, best value per column in bold.

    `wide` emits `table*` rather than `table`, which is what a two-column
    manuscript needs for anything past about four numeric columns. The tabular
    is wrapped in a `\resizebox` that shrinks ONLY when the natural width
    exceeds the text block -- an unconditional resize would scale a narrow table
    up and leave its type larger than the body text.

    The result is a standalone float, so a manuscript `\input`s it and cannot
    drift from the JSON it was rendered from.
    """
    cells = _cells(rows, columns, horizon)
    env = "table*" if wide else "table"
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        # Shrink to fit only if needed; never magnify.
        "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\toprule",
        _escape_latex(row_label) + " & " + " & ".join(c.label for c in columns) + r" \\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        vals = []
        for _num, txt, win in cells:
            v = txt[i]
            vals.append(f"\\textbf{{{v}}}" if (bold_best and win == i) else v)
        lines.append(_escape_latex(str(r.get(name_key, "?"))) + " & "
                     + " & ".join(vals) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Provenance stamping for generated LaTeX
# --------------------------------------------------------------------------- #
# A generated table and the macro file whose prose surrounds it in the
# manuscript are written by two different processes, from two different result
# JSONs, under two different smoke guards. Nothing used to tie them together, so
# a table re-rendered by one run could sit directly beneath numbers emitted by
# another and the paper still compiled clean -- every macro resolved, every
# table was well formed, and the two halves described different experiments.
#
# The stamp is the tie. It is a LaTeX COMMENT, so it is invisible in the PDF and
# cannot perturb a float, and it carries the fingerprint of the run that
# produced the file plus whether that run was a smoke run. Written by
# `write_latex`, read by `read_latex_provenance`, compared by
# `scripts/rebuild_paper_tables.check_metabonet_coupling`.
LATEX_PROVENANCE_PREFIX: str = "% causalcgm-mt-provenance:"

_PROVENANCE_RE = re.compile(
    re.escape(LATEX_PROVENANCE_PREFIX)
    + r"\s+fingerprint=(?P<fingerprint>\S*)"
    r"\s+smoke=(?P<smoke>true|false)"
    r"(?:\s+generated=(?P<generated>\S*))?")


def latex_provenance_line(fingerprint: str, smoke: bool,
                          generated: str = "") -> str:
    """The one-line stamp identifying which run produced a generated table.

    Whitespace-free values only: the parser splits on spaces so that a stamp
    stays greppable and diffable, and every field it carries (a hex fingerprint,
    a bool, an ISO timestamp) is naturally free of them.
    """
    line = (f"{LATEX_PROVENANCE_PREFIX} fingerprint={str(fingerprint) or '-'} "
            f"smoke={'true' if smoke else 'false'}")
    if generated:
        line += f" generated={str(generated).replace(' ', 'T')}"
    return line


def read_latex_provenance(path: str) -> dict | None:
    """The stamp on a generated `.tex`, or None when it carries none.

    None means "unverifiable", not "fine": a table written before stamping
    existed cannot be matched against the macros beside it, and the caller is
    expected to treat that as a problem rather than as a pass.
    """
    try:
        with open(path) as f:
            head = f.read(8192)
    except OSError:
        return None
    m = _PROVENANCE_RE.search(head)
    if not m:
        return None
    return {"fingerprint": m.group("fingerprint"),
            "smoke": m.group("smoke") == "true",
            "generated": m.group("generated") or ""}


def write_latex(path: str, body: str, provenance: dict | None = None) -> str:
    """Write a generated table, optionally stamped with its run's provenance.

    `provenance` takes `fingerprint`, `smoke` and an optional `generated`. Pass
    it for anything the manuscript `\\input`s, so the table can be checked
    against the macro file rendered beside it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = body.rstrip() + "\n"
    if provenance is not None:
        text = latex_provenance_line(
            provenance.get("fingerprint", ""),
            bool(provenance.get("smoke", False)),
            provenance.get("generated", "")) + "\n" + text
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")
    return path


def text_table(rows: Sequence[dict], columns: Sequence[Column], horizon: int,
               name_key: str = "arm", width: int = 26) -> str:
    """The same table for stdout, so a long run is readable while it happens."""
    cells = _cells(rows, columns, horizon)
    head = f"{'arm':<{width}}" + "".join(
        f"{c.label.replace(chr(92) + '%', '%').replace(chr(92), ''):>13}" for c in columns)
    out = [head, "-" * len(head)]
    for i, r in enumerate(rows):
        line = f"{str(r.get(name_key, '?'))[:width]:<{width}}"
        for _num, txt, _win in cells:
            line += f"{txt[i]:>13}"
        out.append(line)
    return "\n".join(out)
