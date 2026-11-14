"""Clinical evaluation metrics for BG prediction and hypoglycemia forewarning.

Two principles here:

1. Regression metrics on their own are misleading for CGM. A model that copies
   its last input achieves a respectable RMSE at PH=30 while being clinically
   useless, because its predictions arrive a full horizon late. We therefore
   always report the persistence baseline alongside, plus an explicit time-lag
   estimate.

2. Forewarning is a rare-event detection problem (hypoglycemia is a few percent
   of samples), so ROC-AUC is optimistic. PR-AUC and sensitivity at a fixed
   false-alarm rate are the metrics that actually discriminate between models.

3. Even a fixed false-positive RATE is not what a patient experiences. A 5%
   FPR sounds tolerable until it is expressed in the unit the patient lives in:
   at a 5-minute cadence, 5% of 288 samples per day is 14 false-positive
   SAMPLES -- but because a spurious excursion latches the alarm for several
   consecutive steps, that is a handful of BEEPS, not fourteen. Getting from
   one to the other requires grouping consecutive threshold crossings into
   episodes, which is what `false_alarms_per_day` does (via
   `clinical.alarm_episodes`). Alarm fatigue -- not AUC -- is what makes a
   forewarning system get switched off, so False Alarms Per Day (FAPD) is
   reported alongside sensitivity at the SAME operating point.

4. A warning is only useful if it arrives early enough to act on. 15 g of oral
   glucose takes ~15 minutes to raise BG, so a 5-minute warning is nearly
   worthless and a 25-minute one is actionable. `effective_lead_time_minutes`
   measures the gap between the first valid alert and the glucose nadir, which
   is the number that distinguishes a genuinely anticipatory model from one
   that announces hypoglycemia as it happens.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

from .clinical import (
    HYPO_THRESHOLD_MGDL,
    alarm_episodes,
    hypo_events,
    segment_ids,
)

MINUTES_PER_DAY: float = 24.0 * 60.0


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def mard(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute relative difference (%), the standard CGM accuracy metric.

    Reference values are clipped at 1 mg/dL to keep the ratio finite; in
    practice no physiological reading approaches that.
    """
    denom = np.clip(np.abs(y), 1.0, None)
    return float(np.mean(np.abs(y - p) / denom) * 100.0)


def time_lag_minutes(
    y_series: np.ndarray,
    p_series: np.ndarray,
    period_min: int,
    max_lag_steps: int = 12,
) -> float:
    """Lag that maximises cross-correlation between prediction and truth.

    Computed on a contiguous per-patient sequence of anchors. A model whose only
    strategy is persistence will show a lag close to the full prediction
    horizon; a genuinely anticipatory model shows a small lag. Reported in
    minutes, positive = prediction trails truth.
    """
    y = np.asarray(y_series, float)
    p = np.asarray(p_series, float)
    n = min(y.size, p.size)
    if n < max_lag_steps + 2:
        return float("nan")
    y, p = y[:n] - y[:n].mean(), p[:n] - p[:n].mean()
    if y.std() < 1e-8 or p.std() < 1e-8:
        return float("nan")

    best_lag, best_corr = 0, -np.inf
    for lag in range(0, max_lag_steps + 1):
        # Find L such that p[t+L] best matches y[t]: i.e. the prediction only
        # reproduces the truth once it has been shifted L steps LATER, meaning
        # it arrived L steps late. Pure persistence (p[t] = BG(t) against
        # y[t] = BG(t+PH)) therefore scores a lag of exactly PH.
        a, b = (y[: n - lag], p[lag:]) if lag else (y, p)
        # The whole-series guard above is not enough: a SLICE can be constant
        # even when the series is not (a flat stretch at the end of a segment,
        # for instance). numpy divides by that zero standard deviation, warns,
        # and returns nan -- harmless for the result, but it floods a long run's
        # log with thousands of RuntimeWarnings that hide the real ones.
        if a.size <= 2 or a.std() < 1e-8 or b.std() < 1e-8:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, lag
    return float(best_lag * period_min)


# --------------------------------------------------------------------------- #
# Classification / forewarning
# --------------------------------------------------------------------------- #
def sensitivity_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.05) -> float:
    """Recall at a fixed false-alarm rate.

    This is the operationally meaningful number: alarm fatigue, not AUC, is what
    determines whether a forewarning system gets switched off by its user.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(thr[min(i, thr.size - 1)])


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.05) -> float:
    """The most permissive score threshold whose false-positive rate does not
    EXCEED `target_fpr`.

    `sensitivity_at_fpr` reads the ROC curve by interpolation, which is the
    right thing for a curve summary but is not an operating point a device can
    be configured to: interpolation invents a threshold between two achievable
    ones. FAPD has to be measured at a threshold that actually exists, so this
    returns a concrete order statistic of the scores and errs on the
    conservative side (<= target, never >).

    Returns ``+inf`` when no finite threshold achieves the target -- i.e. the
    alarm can only be silenced completely. That is a meaningful answer (FAPD 0,
    sensitivity 0), not an error, so it is returned rather than raised.
    ``nan`` is returned when the labels are single-class, where FPR is
    undefined.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, _tpr, thr = roc_curve(y, s)
    ok = np.flatnonzero(fpr <= float(target_fpr))
    if ok.size == 0:                       # cannot happen: fpr[0] == 0
        return float("inf")
    return float(thr[ok[-1]])


def false_alarms_per_day(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    period_minutes: int = 5,
    segment: np.ndarray | None = None,
) -> float:
    """False alarm EPISODES per day of monitoring.

        FAPD = (number of false-positive alarm episodes) / (days monitored)

    Episode grouping is the whole point. A device latches its alarm while the
    score stays above threshold, so a 40-minute spurious excursion is one beep,
    not eight. Counting samples would report eight times the burden the patient
    experiences, and would make any model with smooth score dynamics look
    worse than one with jittery scores at identical discrimination.

    An alarm episode is counted as FALSE only when NO step inside it carries a
    positive label. An episode that fires 10 minutes early and stays on through
    a genuine event is a true alarm that happens to have started early -- the
    behaviour we are trying to encourage -- so charging it as a false alarm as
    well would penalise exactly the right answer. This is the one modelling
    decision here that a reader might make differently, so it is stated rather
    than buried.

    `segment` (see `clinical.segment_ids`) prevents an episode from bridging a
    sensor dropout or two different patients. Without it, a run of alarms
    either side of a four-hour gap counts once instead of twice.

    Days monitored is the time actually COVERED by the supplied rows
    (n * period), not the wall-clock span between the first and last of them.
    A cohort's windows are not contiguous, and charging a model for false
    alarms during hours when it was never asked for a prediction would make
    FAPD depend on how much data was dropped upstream.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_prob, dtype=np.float64)
    if y.size != s.size:
        raise ValueError(f"y_true has {y.size} rows, y_prob has {s.size}")
    if y.size == 0:
        return float("nan")

    days = y.size * float(period_minutes) / MINUTES_PER_DAY
    if days <= 0:
        return float("nan")
    if np.isnan(threshold):
        return float("nan")

    pos = np.asarray(y).astype(np.float64) > 0.5
    # `+inf` silences the alarm entirely (zero episodes) and `-inf` latches it
    # on; both fall out of this comparison without a special case. Non-finite
    # SCORES never alarm -- a missing prediction is not a warning.
    alarm = np.isfinite(s) & (s >= float(threshold))
    n_false = sum(1 for a, b in alarm_episodes(alarm, segment) if not pos[a:b].any())
    return float(n_false) / days


@dataclass(frozen=True)
class LeadTimeReport:
    """How early the alarm fired, relative to the glucose nadir.

    `mean_min` / `median_min` are computed over DETECTED events only -- an
    undetected event has no lead time, and scoring it as zero would conflate
    "warned late" with "never warned". `detection_rate` is therefore reported
    alongside and must be read with them: a model that warns about one event in
    ten, 25 minutes early, has an excellent mean lead time and is useless.
    """
    mean_min: float
    median_min: float
    n_events: int
    n_detected: int
    detection_rate: float
    mean_onset_lead_min: float
    median_onset_lead_min: float
    lead_times_min: tuple[float, ...] = ()
    # Events with a valid alert in the P anchors before ONSET, where the alert
    # anchor itself was not already hypoglycemic. This is the detection rate
    # that matters clinically: an alert that fires only once the patient is
    # below 70 announces the event rather than forewarning it.
    n_onset_detected: int = 0
    onset_detection_rate: float = float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["lead_times_min"] = list(self.lead_times_min)
        return d

    def __str__(self) -> str:
        return (
            f"lead time: mean {self.mean_min:.1f} min, median {self.median_min:.1f} min "
            f"over {self.n_detected}/{self.n_events} detected events "
            f"({self.detection_rate:.1%}); to onset: mean "
            f"{self.mean_onset_lead_min:.1f} min"
        )


def effective_lead_time_minutes(
    y_true_series: np.ndarray,
    y_prob_series: np.ndarray,
    threshold: float,
    horizon_steps: int,
    period_minutes: int = 5,
    hypo_threshold: float = HYPO_THRESHOLD_MGDL,
    segment: np.ndarray | None = None,
    min_event_steps: int = 1,
) -> LeadTimeReport:
    """Warning lead time for each true hypoglycemic event.

    `y_true_series` is the OBSERVED GLUCOSE trace aligned to the score series
    (mg/dL, one value per anchor), not the regression target: an event is
    defined by where glucose actually went, and `y(t) < hypo_threshold` is that
    definition. `y_prob_series` is the model's hypoglycemia score at the same
    anchors. Both must be sorted by time.

    WHICH ALERTS COUNT, AND WHY THIS WINDOW
    ---------------------------------------
    The classifier's label at anchor `a` is "BG dips below threshold somewhere
    in (a, a+PH]". So an alert at `a` is a legitimate warning about an event
    whose nadir sits at index `m` exactly when m in (a, a+PH], i.e. when

        a in [m - horizon_steps, m - 1]

    -- the `horizon_steps` anchors strictly before the nadir. That window is
    derived from the label definition rather than assumed, which is why it is
    closed on the left and open on the right: an "alert" at the nadir itself is
    an observation, not a warning, and admitting it would inflate the detection
    rate with alarms of zero lead time.

    The lead time is then `(m - a) * period_minutes` for the EARLIEST such
    alert, so it lies in [period_minutes, PH]. The search never crosses a
    recording-segment boundary: an alarm on the far side of a sensor dropout
    could not have warned anyone about this event.

    The onset-referenced lead time (`mean_onset_lead_min`) is reported
    alongside because the two answer different clinical questions -- time to
    the nadir is time until the patient is at their worst, time to the onset is
    time until treatment is indicated -- and the nadir-referenced number is the
    larger of the two by construction.
    """
    g = np.asarray(y_true_series, dtype=np.float64)
    s = np.asarray(y_prob_series, dtype=np.float64)
    if g.size != s.size:
        raise ValueError(f"y_true_series has {g.size} rows, y_prob_series has {s.size}")
    if int(horizon_steps) < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

    empty = LeadTimeReport(float("nan"), float("nan"), 0, 0, float("nan"),
                           float("nan"), float("nan"), ())
    if g.size == 0:
        return empty

    seg = (np.zeros(g.size, dtype=np.int64) if segment is None
           else np.asarray(segment, dtype=np.int64))
    events = hypo_events(g, threshold=hypo_threshold, segment=seg,
                         min_steps=min_event_steps)
    if not events:
        return empty

    alarm = np.isfinite(s) & (s >= float(threshold)) if np.isfinite(threshold) \
        else np.zeros(g.size, dtype=bool)

    def first_alert_before(ref: int, not_already_low: bool = False) -> int | None:
        """Earliest alert in [ref - horizon_steps, ref - 1], same segment.

        With `not_already_low`, an alert anchored at a reading already below
        `hypo_threshold` does not count: it detects an ongoing low rather than
        forewarning this one.
        """
        lo = max(0, ref - int(horizon_steps))
        for a in range(lo, ref):
            if alarm[a] and seg[a] == seg[ref]:
                if not_already_low and not (np.isfinite(g[a]) and g[a] >= hypo_threshold):
                    continue
                return a
        return None

    nadir_leads: list[float] = []
    onset_leads: list[float] = []
    for ev in events:
        a = first_alert_before(ev.nadir)
        if a is not None:
            nadir_leads.append((ev.nadir - a) * float(period_minutes))
        b = first_alert_before(ev.start, not_already_low=True)
        if b is not None:
            onset_leads.append((ev.start - b) * float(period_minutes))

    n_det = len(nadir_leads)
    return LeadTimeReport(
        mean_min=float(np.mean(nadir_leads)) if n_det else float("nan"),
        median_min=float(np.median(nadir_leads)) if n_det else float("nan"),
        n_events=len(events),
        n_detected=n_det,
        detection_rate=float(n_det) / len(events),
        mean_onset_lead_min=float(np.mean(onset_leads)) if onset_leads else float("nan"),
        median_onset_lead_min=float(np.median(onset_leads)) if onset_leads else float("nan"),
        lead_times_min=tuple(nadir_leads),
        n_onset_detected=len(onset_leads),
        onset_detection_rate=float(len(onset_leads)) / len(events),
    )


def onset_eligible(
    last_bg: np.ndarray,
    segment: np.ndarray,
    lookback_steps: int = 6,
    threshold: float = HYPO_THRESHOLD_MGDL,
) -> np.ndarray:
    """Anchors at which forewarning is a forecast rather than a detection.

    An anchor is eligible when its reading and every reading in the preceding
    `lookback_steps` anchors of the same segment are >= `threshold`. Rows must
    be sorted by (patient, time) with `segment` from `clinical.segment_ids`.
    Windows where the patient is already (or was just) hypoglycemic are where a
    persistence model scores its PR-AUC, so excluding them is what separates
    anticipation from echoing the current reading.
    """
    g = np.asarray(last_bg, dtype=np.float64)
    seg = np.asarray(segment)
    ok = np.isfinite(g) & (g >= threshold)
    out = ok.copy()
    for k in range(1, int(lookback_steps) + 1):
        prev_ok = np.ones_like(ok)
        prev_ok[k:] = ok[:-k] | (seg[k:] != seg[:-k])
        out &= prev_ok
    return out


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class EvalReport:
    """Statistical and clinical-alarm view of one horizon's predictions.

    The alarm fields at the bottom are the ones a clinician reads. They all
    default to nan so that a caller who does not supply the extra context they
    need (`t_time`, `horizon_min`) still gets a valid report rather than a
    TypeError -- adding a metric must not break an existing call site.
    """
    rmse: float
    mae: float
    mard: float
    time_lag_min: float
    rmse_persistence: float
    time_lag_persistence_min: float
    roc_auc: float
    pr_auc: float
    pr_auc_prevalence: float     # the no-skill PR-AUC floor = class prevalence
    sens_at_5pct_fpr: float
    best_f1: float
    best_f1_threshold: float
    n: int
    n_positive: int

    # ---- clinical alarm-fatigue metrics -------------------------------------
    # `sensitivity_at_5pct_fpr` is the same quantity as the legacy
    # `sens_at_5pct_fpr` above, under the name the alarm block uses; both are
    # kept so existing readers (scripts/run_ablation.py, the saved JSON
    # reports) do not silently lose a key.
    sensitivity_at_5pct_fpr: float = float("nan")
    sensitivity_at_10pct_fpr: float = float("nan")
    fapd_at_5pct_fpr: float = float("nan")
    fapd_at_10pct_fpr: float = float("nan")
    threshold_at_5pct_fpr: float = float("nan")
    threshold_at_10pct_fpr: float = float("nan")
    mean_lead_time_min: float = float("nan")
    median_lead_time_min: float = float("nan")
    lead_time_detection_rate: float = float("nan")
    n_hypo_events: int = 0
    monitored_days: float = float("nan")

    # ---- operating-point provenance -----------------------------------------
    # Every operating-point metric above (sensitivity_at_*, fapd_at_*, the lead
    # times, best_f1) is computed at thresholds supplied by the caller, fitted
    # on a CALIBRATION split. Nothing here selects a threshold from the labels
    # being scored. The FPR actually realised on this set at that threshold is
    # reported so threshold transfer is visible rather than assumed.
    threshold_source: str = "none"
    realized_fpr_at_5pct: float = float("nan")
    realized_fpr_at_10pct: float = float("nan")

    # ---- onset-referenced forewarning ---------------------------------------
    # Lead time to the FIRST reading below 70 (not the nadir), counting only
    # alerts anchored at a reading that was not already hypoglycemic.
    mean_onset_lead_min: float = float("nan")
    median_onset_lead_min: float = float("nan")
    onset_detection_rate: float = float("nan")
    # Discrimination restricted to anchors that are >= 70 mg/dL now and over the
    # preceding 30 minutes (`onset_eligible`): the forecasting task, with the
    # already-low windows that persistence detects trivially removed.
    pr_auc_onset: float = float("nan")
    roc_auc_onset: float = float("nan")
    onset_prevalence: float = float("nan")
    n_onset_windows: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"n={self.n} (pos={self.n_positive}, prev={self.pr_auc_prevalence:.3%})",
            f"  REG  RMSE {self.rmse:7.3f} (persist {self.rmse_persistence:7.3f}) | "
            f"MAE {self.mae:6.3f} | MARD {self.mard:5.2f}% | "
            f"lag {self.time_lag_min:5.1f} min (persist {self.time_lag_persistence_min:5.1f})",
            f"  CLS  ROC-AUC {self.roc_auc:.4f} | PR-AUC {self.pr_auc:.4f} | "
            f"Sens@5%FPR {self.sens_at_5pct_fpr:.4f} | F1 {self.best_f1:.4f}",
        ]
        if np.isfinite(self.fapd_at_5pct_fpr) or np.isfinite(self.mean_lead_time_min):
            lines.append(
                f"  ALRM Sens/FAPD @5%FPR {self.sensitivity_at_5pct_fpr:.3f}/"
                f"{self.fapd_at_5pct_fpr:.2f}/day | @10%FPR "
                f"{self.sensitivity_at_10pct_fpr:.3f}/{self.fapd_at_10pct_fpr:.2f}/day | "
                f"lead {self.mean_lead_time_min:.1f} min "
                f"(detect {self.lead_time_detection_rate:.1%} of "
                f"{self.n_hypo_events} events)"
            )
        return "\n".join(lines)


def evaluate(
    y_reg: np.ndarray,
    bg_hat: np.ndarray,
    y_cls: np.ndarray,
    score: np.ndarray,
    last_bg: np.ndarray,
    patient: np.ndarray,
    period_min: int,
    t_time: np.ndarray | None = None,
    horizon_min: int | None = None,
    hypo_threshold: float = HYPO_THRESHOLD_MGDL,
    alarm_thresholds: dict[float, float] | None = None,
) -> EvalReport:
    """Full report. ``y_cls`` must be the HARD label -- soft labels are a
    training device; scoring against them would make the metric unreadable.

    ``t_time`` (anchor timestamps, aligned to the rows) and ``horizon_min``
    are optional because the alarm metrics need context the statistical ones do
    not:

      * ``t_time`` lets alarm episodes be broken at sensor dropouts and patient
        boundaries. Without it the rows are assumed to be one contiguous
        recording per patient, which is right for a single-patient sequence and
        wrong -- in the direction of UNDER-counting alarms -- for a cohort with
        gaps. Supply it whenever you have it.
      * ``horizon_min`` is needed to know which alerts could legitimately have
        warned about a given event (see `effective_lead_time_minutes`). Without
        it the lead-time fields stay nan rather than being computed against a
        guessed horizon.

    The hypoglycemic events for the lead-time metric are taken from
    ``last_bg`` -- the OBSERVED glucose at each anchor -- not from ``y_reg``,
    which is glucose one horizon in the future and would place every event PH
    minutes too early.
    """
    y_cls = y_cls.astype(int)

    # ---- alarm/segment bookkeeping -----------------------------------------
    # Rows must be in time order within a patient for episode grouping to mean
    # anything; sorting here rather than trusting the caller keeps the metric
    # correct for a caller who assembled the rows from several splits.
    order = np.lexsort((np.asarray(t_time), np.asarray(patient, dtype=str))) \
        if t_time is not None else np.lexsort((np.arange(y_cls.size),
                                               np.asarray(patient, dtype=str)))
    seg = segment_ids(None if t_time is None else np.asarray(t_time)[order],
                      np.asarray(patient, dtype=str)[order], period_min,
                      n=int(y_cls.size))

    # Time lag is a cross-correlation, so it is only meaningful on a CONTIGUOUS
    # run of anchors: on a subsampled or shuffled evaluation set, one array step
    # is not one sampling period and the reported lag is scaled by whatever the
    # subsampling factor happens to be. That is not a hypothetical -- it made a
    # randomly-split control arm look like it forecast 20 minutes earlier than
    # the strict pipeline, which is exactly backwards.
    #
    # With `t_time` supplied the lag is therefore computed per contiguous
    # segment and the median taken over segments long enough to support it;
    # without it the rows are assumed contiguous per patient, as before.
    lags, lags_p = [], []
    if t_time is None:
        for pid in np.unique(patient):
            m = patient == pid
            lags.append(time_lag_minutes(y_reg[m], bg_hat[m], period_min))
            lags_p.append(time_lag_minutes(y_reg[m], last_bg[m], period_min))
    else:
        y_o, p_o, l_o = (np.asarray(a)[order] for a in (y_reg, bg_hat, last_bg))
        for s in np.unique(seg):
            m = seg == s
            if int(m.sum()) < 20:      # too short for a 12-lag cross-correlation
                continue
            lags.append(time_lag_minutes(y_o[m], p_o[m], period_min))
            lags_p.append(time_lag_minutes(y_o[m], l_o[m], period_min))

    def _median(v: list[float]) -> float:
        # nanmedian over an all-nan list warns and returns nan; short-circuit
        # instead. Every patient's lag is nan whenever the slice is too short or
        # constant, which is a normal condition (a single-class evaluation
        # slice), not something worth a RuntimeWarning on every call.
        a = np.asarray(v, dtype=np.float64)
        return float(np.nanmedian(a)) if a.size and not np.isnan(a).all() else float("nan")

    has_both = len(np.unique(y_cls)) > 1
    s_all = np.asarray(score, dtype=np.float64)

    # ---- operating points: supplied, never fitted here ----------------------
    # An earlier version chose each threshold with `threshold_at_fpr` on the
    # very labels it then scored -- an oracle operating point no deployed device
    # could reach. Thresholds now come from the caller, fitted on a calibration
    # split (`calibration_thresholds`), and are applied unchanged. Without them
    # every operating-point metric is nan rather than silently oracle.
    thr_at = {t: float("nan") for t in (0.05, 0.10)}
    if alarm_thresholds is not None:
        for t in thr_at:
            if t in alarm_thresholds:
                thr_at[t] = float(alarm_thresholds[t])
    neg, pos = y_cls == 0, y_cls == 1

    fapd, sens_at, fpr_at = {}, {}, {}
    for target, t_star in thr_at.items():
        if np.isnan(t_star):
            fapd[target] = sens_at[target] = fpr_at[target] = float("nan")
            continue
        fires = np.isfinite(s_all) & (s_all >= t_star)
        sens_at[target] = float(fires[pos].mean()) if pos.any() else float("nan")
        fpr_at[target] = float(fires[neg].mean()) if neg.any() else float("nan")
        # An INFINITE threshold is a real answer (the alarm never fires) and
        # goes through the normal path, which returns 0 alarms per day.
        fapd[target] = false_alarms_per_day(y_cls[order], s_all[order],
                                            t_star, period_min, seg)

    # F1 at the 5%-FPR calibration operating point -- not the best F1 over test
    # thresholds, which would be another oracle. The field name is kept so
    # existing readers do not silently lose a key.
    f1, thr = float("nan"), thr_at[0.05]
    if np.isfinite(thr) and has_both:
        fires = np.isfinite(s_all) & (s_all >= thr)
        tp = float((fires & pos).sum()); fp = float((fires & neg).sum())
        fn = float((~fires & pos).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

    # ---- onset-only discrimination ------------------------------------------
    elig_sorted = onset_eligible(np.asarray(last_bg, dtype=np.float64)[order], seg,
                                 threshold=hypo_threshold)
    elig = np.empty_like(elig_sorted)
    elig[order] = elig_sorted
    y_on, s_on = y_cls[elig], s_all[elig]
    on_both = y_on.size > 0 and len(np.unique(y_on)) > 1

    lead = LeadTimeReport(float("nan"), float("nan"), 0, 0, float("nan"),
                          float("nan"), float("nan"), ())
    if horizon_min is not None and np.isfinite(thr_at[0.05]):
        lead = effective_lead_time_minutes(
            np.asarray(last_bg, dtype=np.float64)[order],
            np.asarray(score, dtype=np.float64)[order],
            threshold=thr_at[0.05],
            horizon_steps=max(1, int(horizon_min) // int(period_min)),
            period_minutes=period_min,
            hypo_threshold=hypo_threshold,
            segment=seg,
        )

    return EvalReport(
        rmse=rmse(y_reg, bg_hat),
        mae=mae(y_reg, bg_hat),
        mard=mard(y_reg, bg_hat),
        time_lag_min=_median(lags),
        rmse_persistence=rmse(y_reg, last_bg),
        time_lag_persistence_min=_median(lags_p),
        roc_auc=float(roc_auc_score(y_cls, score)) if has_both else float("nan"),
        pr_auc=float(average_precision_score(y_cls, score)) if has_both else float("nan"),
        pr_auc_prevalence=float(y_cls.mean()),
        sens_at_5pct_fpr=sens_at[0.05],
        best_f1=f1,
        best_f1_threshold=thr,
        n=int(y_cls.size),
        n_positive=int(y_cls.sum()),
        sensitivity_at_5pct_fpr=sens_at[0.05],
        sensitivity_at_10pct_fpr=sens_at[0.10],
        fapd_at_5pct_fpr=fapd[0.05],
        fapd_at_10pct_fpr=fapd[0.10],
        threshold_at_5pct_fpr=thr_at[0.05],
        threshold_at_10pct_fpr=thr_at[0.10],
        mean_lead_time_min=lead.mean_min,
        median_lead_time_min=lead.median_min,
        lead_time_detection_rate=lead.detection_rate,
        n_hypo_events=lead.n_events,
        monitored_days=float(y_cls.size) * float(period_min) / MINUTES_PER_DAY,
        threshold_source="calibration" if alarm_thresholds is not None else "none",
        realized_fpr_at_5pct=fpr_at[0.05],
        realized_fpr_at_10pct=fpr_at[0.10],
        mean_onset_lead_min=lead.mean_onset_lead_min,
        median_onset_lead_min=lead.median_onset_lead_min,
        onset_detection_rate=lead.onset_detection_rate,
        pr_auc_onset=float(average_precision_score(y_on, s_on)) if on_both else float("nan"),
        roc_auc_onset=float(roc_auc_score(y_on, s_on)) if on_both else float("nan"),
        onset_prevalence=float(y_on.mean()) if y_on.size else float("nan"),
        n_onset_windows=int(y_on.size),
    )


def calibration_thresholds(
    y_cls_calib: np.ndarray,
    score_calib: np.ndarray,
    targets: tuple[float, ...] = (0.05, 0.10),
) -> dict[float, float]:
    """Operating thresholds fitted on a CALIBRATION split, one per target FPR.

    The only place a threshold is chosen from labels. Pass the result to
    `evaluate(..., alarm_thresholds=...)` on the test rows, where it is applied
    unchanged and the realised FPR is reported next to the target.
    """
    y = np.asarray(y_cls_calib).astype(int)
    s = np.asarray(score_calib, dtype=np.float64)
    ok = np.isfinite(s)
    return {float(t): threshold_at_fpr(y[ok], s[ok], t) for t in targets}
