"""LightGBM tabular baseline: gradient-boosted trees on flattened lag features.

Why this baseline is worth the effort
-------------------------------------
Deep sequence models are routinely reported for CGM forecasting without a strong
tabular control, and gradient boosting on hand-built lag features is a genuinely
competitive comparator at a 30-minute horizon -- often within a mg/dL or two of
a TCN, at a fraction of the training cost and with far better interpretability.
If the TCN cannot beat this, that is a finding, not an embarrassment, and it is
much better discovered locally than in review.

Causality
---------
Feature construction here is causal *by construction rather than by care*: every
feature is a function of `ws.x[n, :, 0]`, the glucose history of window n, which
`data.build_windows` has already restricted to `[t-H+1, t]`. This module never
touches the raw series, never re-indexes by time, and never sees `y_reg` outside
`fit`. There is consequently no code path by which a future observation could
enter a feature -- the guarantee is inherited from the windowing stage and
cannot be weakened here.

Trees need no scaling, so `CausalScaler` is deliberately NOT applied: feeding
scaled inputs would add a fitted transform to the pipeline for no benefit, and
every fitted transform is a leak surface. The one fitted quantity is the
classifier's class weight, computed on the training split only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..data import WindowSet, split_by_patient
from ..evaluate import EvalReport, evaluate


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
# Lookbacks in STEPS at the 5-minute grid: 5, 15, 30, 60 minutes plus the full
# 2-hour window. Expressed in steps so the code stays correct if the cadence
# changes; `tabular_features` clamps anything longer than the history.
LOOKBACK_STEPS: tuple[int, ...] = (1, 3, 6, 12, 24)


def _feature_names() -> list[str]:
    names = ["bg_last", "bg_first"]
    for k in LOOKBACK_STEPS:
        names += [
            f"mean_{k}", f"std_{k}", f"min_{k}", f"max_{k}",
            f"delta_{k}", f"slope_{k}",
        ]
    names += [
        "roc_last",          # most recent first difference
        "accel_last",        # most recent second difference
        "range_full",
        "n_below_80", "n_below_70",
        "frac_falling",
        "time_since_max", "time_since_min",
        "imputed_frac",      # from the observation-mask channel
        "hour_sin", "hour_cos",
    ]
    return names


FEATURE_NAMES: list[str] = _feature_names()


def tabular_features(ws: WindowSet, idx: np.ndarray | None = None) -> np.ndarray:
    """Flatten causal history windows into a tabular design matrix.

    Vectorised over windows; every column is a reduction over a SUFFIX of the
    history (the most recent k steps), which is the natural tabular analogue of
    the causal rolling statistics the TCN sees.

    Returns (N, len(FEATURE_NAMES)) float32.
    """
    x = ws.x if idx is None else ws.x[idx]
    t_time = ws.t_time if idx is None else ws.t_time[idx]

    g = x[:, :, 0].astype(np.float64)          # glucose history
    obs = x[:, :, 4].astype(np.float64)        # observation mask
    n, H = g.shape

    cols: list[np.ndarray] = [g[:, -1], g[:, 0]]

    for k in LOOKBACK_STEPS:
        kk = min(k, H)
        w = g[:, -kk:]                          # most recent kk steps: causal
        cols.append(w.mean(axis=1))
        cols.append(w.std(axis=1) if kk > 1 else np.zeros(n))
        cols.append(w.min(axis=1))
        cols.append(w.max(axis=1))
        # Change over the lookback, and its per-step slope.
        cols.append(g[:, -1] - g[:, -kk])
        cols.append((g[:, -1] - g[:, -kk]) / max(kk - 1, 1))

    d1 = np.diff(g, axis=1)
    cols.append(d1[:, -1])
    cols.append(d1[:, -1] - d1[:, -2] if H >= 3 else np.zeros(n))
    cols.append(g.max(axis=1) - g.min(axis=1))
    cols.append((g < 80).sum(axis=1).astype(np.float64))
    cols.append((g < 70).sum(axis=1).astype(np.float64))
    cols.append((d1 < 0).mean(axis=1))
    # Steps since the window's extremes -- captures whether a peak or nadir is
    # recent (predictive) or stale.
    cols.append((H - 1 - g.argmax(axis=1)).astype(np.float64))
    cols.append((H - 1 - g.argmin(axis=1)).astype(np.float64))
    cols.append(1.0 - obs.mean(axis=1))

    # Time of day: circadian effects are real (dawn phenomenon, nocturnal
    # hypoglycemia). Encoded as sin/cos so midnight is adjacent to 23:59.
    # Minutes since midnight, vectorised. The per-row form built two
    # `datetime64 -> Python datetime` objects for every window, on the fit, val
    # and test splits of every horizon of every arm, and dominated LightGBM's
    # feature construction on a 1e5-row split. Truncating to [m] first and
    # subtracting the [D] floor gives the same hour-of-day in one pass.
    t_min = np.asarray(t_time).astype("datetime64[m]")
    minutes = (t_min - t_min.astype("datetime64[D]")).astype("timedelta64[m]").astype(np.int64)
    # `h + m/60`, not `minutes/60`: the two differ in the last ulp, and these
    # columns are histogram-binned by LightGBM, where a 1-ulp move can land on
    # the other side of a split threshold. Bit-identical to the loop it replaces.
    hours = (minutes // 60) + (minutes % 60) / 60.0
    cols.append(np.sin(2 * np.pi * hours / 24.0))
    cols.append(np.cos(2 * np.pi * hours / 24.0))

    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == len(FEATURE_NAMES), (
        f"feature/name mismatch: {out.shape[1]} columns vs "
        f"{len(FEATURE_NAMES)} names"
    )
    return out


def tabular_features_multimodal(mmws, idx: np.ndarray | None = None) -> np.ndarray:
    """Append auxiliary summaries for the HUPA-UCM multimodal window set.

    Same contract: every column reduces over the aux history window, which the
    loader has already restricted to `[t-H+1, t]`.
    """
    from ..hupa_loader import AUX_INDEX

    base = tabular_features(mmws, idx)
    a = mmws.x_aux if idx is None else mmws.x_aux[idx]

    extra = []
    for name in ("iob", "cob", "heart_rate", "steps"):
        c = a[:, :, AUX_INDEX[name]].astype(np.float64)
        extra += [c[:, -1], c.max(axis=1), c.mean(axis=1)]
    extra.append(a[:, :, AUX_INDEX["sleep"]].mean(axis=1))
    extra.append(a[:, :, AUX_INDEX["hr_observed"]].mean(axis=1))

    return np.concatenate([base, np.stack(extra, axis=1).astype(np.float32)], axis=1)


MULTIMODAL_FEATURE_NAMES: list[str] = FEATURE_NAMES + [
    f"{n}_{s}" for n in ("iob", "cob", "heart_rate", "steps") for s in ("last", "max", "mean")
] + ["sleep_frac", "hr_observed_frac"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class LGBMBaseline:
    """Paired LightGBM regressor + classifier, matching the two CausalCGM-MT heads.

    Two separate models rather than one multi-output model, because the tasks
    have different objectives and gradient boosting has no shared-representation
    mechanism to exploit anyway. This makes the comparison against the
    multi-task TCN honest in the right direction: if multi-task learning helps,
    the TCN should beat two independent trees.
    """

    cfg: Config
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 1337
    multimodal: bool = False

    reg_model: object | None = field(default=None, repr=False)
    cls_model: object | None = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list, repr=False)

    # ---- residual-on-persistence target ------------------------------------
    # The regressor predicts (BG(t+PH) - BG(t)) rather than the level, mirroring
    # the TCN's residual head. Trees especially benefit: predicting the level
    # lets the model win on RMSE by echoing bg_last, and axis-aligned splits
    # make that a very easy local optimum. Predicting the delta forces a
    # commitment to a direction of change and makes the comparison meaningful.
    def _featurize(self, ws, idx):
        if self.multimodal:
            return tabular_features_multimodal(ws, idx)
        return tabular_features(ws, idx)

    def fit(
        self,
        ws: WindowSet,
        fit_idx: np.ndarray,
        val_idx: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "LGBMBaseline":
        """Fit both heads on `fit_idx`, early-stopping on `val_idx`.

        `val_idx` must be a disjoint set of PATIENTS, not a random row split --
        the caller is responsible for that, exactly as in `train.run_fold`.
        Early stopping on a random row split would select rounds using windows
        that overlap the training windows.
        """
        import lightgbm as lgb

        self.feature_names = (
            MULTIMODAL_FEATURE_NAMES if self.multimodal else FEATURE_NAMES
        )

        X = self._featurize(ws, fit_idx)
        y_delta = ws.y_reg[fit_idx] - ws.last_bg[fit_idx]
        y_cls = ws.y_cls_hard[fit_idx]

        eval_reg = eval_cls = None
        if val_idx is not None and val_idx.size > 0:
            Xv = self._featurize(ws, val_idx)
            eval_reg = [(Xv, ws.y_reg[val_idx] - ws.last_bg[val_idx])]
            eval_cls = [(Xv, ws.y_cls_hard[val_idx])]

        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        # Huber matches the TCN's regression loss, so the two models are
        # optimising comparable objectives rather than being compared across a
        # squared-vs-robust mismatch.
        self.reg_model = lgb.LGBMRegressor(objective="huber", alpha=self.cfg.model.huber_delta, **common)

        # Class weight from the TRAINING split only, mirroring `pos_weight` in
        # the TCN loss.
        prev = float(y_cls.mean())
        pos_weight = (1 - prev) / max(prev, 1e-6)
        self.cls_model = lgb.LGBMClassifier(
            objective="binary", scale_pos_weight=pos_weight, **common
        )

        cb = [lgb.early_stopping(self.early_stopping_rounds, verbose=verbose)] if eval_reg else []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg_model.fit(X, y_delta, eval_set=eval_reg, callbacks=cb)
            self.cls_model.fit(X, y_cls, eval_set=eval_cls, callbacks=cb)
        return self

    def predict(self, ws: WindowSet, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (bg_hat in mg/dL, hypo probability)."""
        if self.reg_model is None or self.cls_model is None:
            raise RuntimeError("fit() must be called before predict()")
        X = self._featurize(ws, idx)
        # Undo the residual parameterisation to get an absolute level back.
        bg_hat = ws.last_bg[idx] + self.reg_model.predict(X)
        risk = self.cls_model.predict_proba(X)[:, 1]
        return bg_hat.astype(np.float64), risk.astype(np.float64)

    def evaluate(self, ws: WindowSet, idx: np.ndarray) -> EvalReport:
        """Score with the SAME `evaluate()` the TCN uses -- identical metrics,
        identical time-lag computation, identical persistence baseline."""
        bg_hat, risk = self.predict(ws, idx)
        return evaluate(
            ws.y_reg[idx], bg_hat, ws.y_cls_hard[idx], risk,
            ws.last_bg[idx], ws.patient[idx], self.cfg.data.sample_period_min,
            t_time=ws.t_time[idx], horizon_min=self.cfg.data.horizon_min,
        )

    def feature_importance(self, top: int = 15) -> list[tuple[str, float, float]]:
        """(name, regression gain, classification gain), most important first.

        Gain rather than split count: split count is dominated by
        high-cardinality features regardless of usefulness.
        """
        if self.reg_model is None:
            raise RuntimeError("fit() must be called first")
        gr = self.reg_model.booster_.feature_importance("gain")
        gc = self.cls_model.booster_.feature_importance("gain")
        gr = gr / max(gr.sum(), 1e-9)
        gc = gc / max(gc.sum(), 1e-9)
        rows = list(zip(self.feature_names, gr, gc))
        return sorted(rows, key=lambda r: -(r[1] + r[2]))[:top]


# --------------------------------------------------------------------------- #
# Fold runner mirroring train.run_fold
# --------------------------------------------------------------------------- #
def run_lgbm_fold(
    ws: WindowSet,
    cfg: Config,
    train_pat: np.ndarray,
    test_pat: np.ndarray,
    fold: int = 0,
    multimodal: bool = False,
) -> tuple[LGBMBaseline, EvalReport]:
    """Same patient-wise split protocol as `train.run_fold`.

    The inner validation split is drawn with the same seed and the same
    `inner_val_frac`, so the two models see byte-identical fit / validation /
    test partitions. Any difference in reported metrics is then attributable to
    the model rather than to the split.
    """
    rng = np.random.default_rng(cfg.train.seed + fold)
    perm = rng.permutation(train_pat)
    n_val = max(1, int(round(len(perm) * cfg.train.inner_val_frac)))
    val_pat, fit_pat = perm[:n_val], perm[n_val:]

    i_fit = split_by_patient(ws, fit_pat)
    i_val = split_by_patient(ws, val_pat)
    i_test = split_by_patient(ws, test_pat)

    model = LGBMBaseline(cfg=cfg, multimodal=multimodal).fit(ws, i_fit, i_val)
    report = model.evaluate(ws, i_test)
    return model, report
