"""Interactive diagnostic dashboard for CausalCGM-MT.

    streamlit run app.py

Purpose. Summary statistics hide the failure modes that matter. The synthetic
run reported RMSE ~24 mg/dL, which looks respectable, while the model was in
fact reproducing persistence -- visible only in the time-lag metric, and only
after that metric's sign bug was fixed. This tool makes those failures visible
directly:

  * the anchor line separates what the model may see from what it must predict,
    so any lookahead confusion is immediately obvious on screen;
  * the forecast is drawn against the persistence baseline, so a model that
    merely echoes its input shows up as two overlapping traces;
  * the soft-label ramp is drawn against the actual excursion depths, which is
    how you tell whether the band is doing anything at all on a given cohort.

Everything here is READ-ONLY with respect to the pipeline: the dashboard fits
nothing. It loads the scaler and soft-labeler that were fit on the training fold
and applies them unchanged. Refitting either on the patient being viewed would
reintroduce exactly the leak the pipeline is built to prevent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causalcgm_mt.config import Config
from causalcgm_mt.data import CausalScaler, SoftLabeler, build_windows  # noqa: F401
from causalcgm_mt.evaluate import evaluate
from causalcgm_mt.models import build_model, load_state_dict_compat
from causalcgm_mt.simulate import simulate_cohort

CKPT = "runs/demo/demo.pt"

st.set_page_config(page_title="CausalCGM-MT diagnostics", layout="wide", page_icon="🩸")

# Colour roles kept consistent across every panel.
C_TRUTH = "#2b6cb0"
C_HIST = "#1a365d"
C_PRED = "#dd6b20"
C_PERSIST = "#a0aec0"
C_HYPO = "#c53030"
C_BAND = "rgba(197,48,48,0.10)"


# --------------------------------------------------------------------------- #
# Loading (cached; the checkpoint carries the cohort spec so data is identical)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading checkpoint…")
def load_bundle(path: str):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    cfg = Config.from_dict(ck["config"])
    model, _ = build_model(ck["in_channels"], cfg.model)
    load_state_dict_compat(model, ck["state_dict"])
    model.eval()
    return ck, cfg, model


@st.cache_data(show_spinner="Generating cohort…")
def load_data(cohort: dict, _cfg: Config):
    df = simulate_cohort(**cohort)
    ws = build_windows(df, _cfg.data)
    return df, ws


@st.cache_data(show_spinner="Running inference…")
def run_inference(_model, _ws, _scaler, _cfg: Config, cohort_key: str):
    """One forward pass over every window. Batched, no gradients."""
    xs = _scaler.transform(_ws.x)
    bg, sc = [], []
    with torch.no_grad():
        for i in range(0, len(xs), 512):
            x = torch.from_numpy(xs[i : i + 512])
            last = torch.from_numpy(_ws.last_bg[i : i + 512])
            b, l = _model(x, last)
            bg.append(b.numpy())
            sc.append(torch.sigmoid(l).numpy())
    return np.concatenate(bg), np.concatenate(sc)


if not os.path.exists(CKPT):
    st.error(f"No checkpoint at `{CKPT}`.")
    st.code("python scripts/train_demo.py 30", language="bash")
    st.stop()

ck, cfg, model = load_bundle(CKPT)
df, ws = load_data(ck["cohort"], cfg)
scaler, labeler = ck["scaler"], ck["labeler"]
bg_hat, risk = run_inference(model, ws, scaler, cfg, str(ck["cohort"]))

PERIOD = cfg.data.sample_period_min
H_MIN = cfg.data.history_min
PH_MIN = cfg.data.horizon_min
THR = cfg.data.hypo_threshold_mgdl


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("CausalCGM-MT")
st.sidebar.caption("Synthetic cohort — pipeline diagnostics, not results.")

held_out = set(ck["test_patients"])
patients = sorted({str(p) for p in ws.patient})
patient = st.sidebar.selectbox(
    "Patient",
    patients,
    format_func=lambda p: f"{p}  {'· held out' if p in held_out else '· seen in training'}",
)
is_held_out = patient in held_out

if not is_held_out:
    st.sidebar.warning(
        "This patient was in the training fold. Predictions here are optimistic "
        "and must not be read as generalisation.",
        icon="⚠️",
    )

alert_thr = st.sidebar.slider("Alert threshold (risk score)", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(
    "Raising this trades sensitivity for fewer false alarms. Alarm fatigue, not "
    "AUC, decides whether a forewarning system stays switched on."
)

show_persist = st.sidebar.checkbox("Overlay persistence baseline", True)
context_hours = st.sidebar.slider("Context window (hours)", 2, 24, 8)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Geometry**  \nHistory *H* = {H_MIN} min ({cfg.data.history_steps} steps)  \n"
    f"Horizon *PH* = {PH_MIN} min ({cfg.data.horizon_steps} steps)  \n"
    f"Hypo threshold = {THR:.0f} mg/dL  \n"
    f"Soft band = ±{labeler.band:.1f} mg/dL {'(on)' if labeler.enabled else '(off)'}"
)
st.sidebar.caption(
    f"Scaler fit on {len(scaler.fitted_on)} training patients; not refit here."
)

# --------------------------------------------------------------------------- #
# Patient slice
# --------------------------------------------------------------------------- #
pm = ws.patient == patient
p_idx = np.flatnonzero(pm)
p_idx = p_idx[np.argsort(ws.t_time[p_idx])]

p_time = pd.to_datetime(ws.t_time[p_idx])
p_truth = ws.y_reg[p_idx]
p_pred = bg_hat[p_idx]
p_risk = risk[p_idx]
p_last = ws.last_bg[p_idx]
p_hard = ws.y_cls_hard[p_idx]
p_min = ws.y_min[p_idx]

raw = df[df.patient_id == patient].sort_values("timestamp")

st.title("CausalCGM-MT — window diagnostics")

anchor_i = st.slider(
    "Anchor time t  (scrub through the record)",
    0, len(p_idx) - 1, len(p_idx) // 2,
    format="",
)
t_anchor = p_time[anchor_i]
st.caption(
    f"Anchor **t = {t_anchor:%Y-%m-%d %H:%M}** · "
    f"model reads [t−{H_MIN}min, t] · predicts (t, t+{PH_MIN}min]"
)


# --------------------------------------------------------------------------- #
# Panel 1: the window in context
# --------------------------------------------------------------------------- #
t0 = t_anchor - pd.Timedelta(hours=context_hours / 2)
t1 = t_anchor + pd.Timedelta(hours=context_hours / 2)
ctx = raw[(raw.timestamp >= t0) & (raw.timestamp <= t1)]

hist_start = t_anchor - pd.Timedelta(minutes=H_MIN)
target_end = t_anchor + pd.Timedelta(minutes=PH_MIN)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
    subplot_titles=("Glucose (mg/dL)", "Hypoglycemia risk score"),
)

# Hypo region and threshold.
fig.add_hrect(y0=0, y1=THR, fillcolor=C_BAND, line_width=0, row=1, col=1)
fig.add_hline(y=THR, line=dict(color=C_HYPO, width=1, dash="dot"), row=1, col=1)
# Soft-label band: where the label is genuinely uncertain.
if labeler.enabled:
    fig.add_hrect(
        y0=THR - labeler.band, y1=THR + labeler.band,
        fillcolor="rgba(197,48,48,0.06)", line_width=0, row=1, col=1,
    )

fig.add_trace(
    go.Scatter(x=ctx.timestamp, y=ctx.glucose, name="CGM", mode="lines",
               line=dict(color=C_TRUTH, width=1.6)),
    row=1, col=1,
)

# The history window the model is actually allowed to read.
hist = ctx[(ctx.timestamp >= hist_start) & (ctx.timestamp <= t_anchor)]
fig.add_trace(
    go.Scatter(x=hist.timestamp, y=hist.glucose, name=f"History (H={H_MIN}m)",
               mode="lines", line=dict(color=C_HIST, width=4)),
    row=1, col=1,
)

# The forecast: a single point at t+PH, plus the ray from the anchor.
fig.add_trace(
    go.Scatter(
        x=[t_anchor, target_end], y=[p_last[anchor_i], p_pred[anchor_i]],
        name="Forecast", mode="lines+markers",
        line=dict(color=C_PRED, width=2.5, dash="dash"),
        marker=dict(size=[6, 13], color=C_PRED, symbol="diamond"),
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=[target_end], y=[p_truth[anchor_i]], name="Truth @ t+PH",
               mode="markers", marker=dict(size=13, color=C_TRUTH, symbol="circle-open",
                                           line=dict(width=3))),
    row=1, col=1,
)
if show_persist:
    fig.add_trace(
        go.Scatter(x=[t_anchor, target_end], y=[p_last[anchor_i]] * 2,
                   name="Persistence", mode="lines",
                   line=dict(color=C_PERSIST, width=2, dash="dot")),
        row=1, col=1,
    )

# Shade the prediction horizon and mark the anchor.
fig.add_vrect(x0=t_anchor, x1=target_end, fillcolor="rgba(221,107,32,0.08)",
              line_width=0, row=1, col=1)
fig.add_vline(x=t_anchor, line=dict(color="#000", width=2), row=1, col=1)
fig.add_vline(x=t_anchor, line=dict(color="#000", width=2), row=2, col=1)

# Risk trace.
fig.add_trace(
    go.Scatter(x=p_time, y=p_risk, name="Risk", mode="lines",
               line=dict(color=C_HYPO, width=1.5)),
    row=2, col=1,
)
fig.add_hline(y=alert_thr, line=dict(color="#000", width=1, dash="dash"), row=2, col=1)
# Ground-truth hypo episodes as a rug.
hypo_t = p_time[p_hard > 0]
if len(hypo_t):
    fig.add_trace(
        go.Scatter(x=hypo_t, y=np.full(len(hypo_t), -0.06), mode="markers",
                   name="True hypo", marker=dict(color=C_HYPO, size=5, symbol="line-ns-open")),
        row=2, col=1,
    )

fig.update_xaxes(range=[t0, t1], row=1, col=1)
fig.update_xaxes(range=[t0, t1], row=2, col=1)
fig.update_yaxes(range=[-0.12, 1.05], row=2, col=1)
fig.update_layout(height=620, hovermode="x unified", margin=dict(t=50, b=10),
                  legend=dict(orientation="h", y=1.10))
st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Panel 2: this window's numbers
# --------------------------------------------------------------------------- #
err = p_pred[anchor_i] - p_truth[anchor_i]
persist_err = p_last[anchor_i] - p_truth[anchor_i]
soft = float(labeler(np.array([p_min[anchor_i]]))[0])

c = st.columns(6)
c[0].metric("Truth @ t+PH", f"{p_truth[anchor_i]:.1f}")
c[1].metric("Forecast", f"{p_pred[anchor_i]:.1f}", f"{err:+.1f} err", delta_color="off")
c[2].metric("Persistence err", f"{persist_err:+.1f}",
            f"{abs(err) - abs(persist_err):+.1f} vs model", delta_color="inverse")
c[3].metric("Min BG in window", f"{p_min[anchor_i]:.1f}")
c[4].metric("Soft label", f"{soft:.3f}", f"hard = {p_hard[anchor_i]:.0f}", delta_color="off")
c[5].metric("Risk score", f"{p_risk[anchor_i]:.3f}",
            "ALERT" if p_risk[anchor_i] >= alert_thr else "clear",
            delta_color="inverse" if p_risk[anchor_i] >= alert_thr else "off")

if 0.15 < soft < 0.85:
    st.info(
        f"This window sits inside the soft-label band (min BG = {p_min[anchor_i]:.1f} "
        f"vs threshold {THR:.0f}). A hard label would force the network to commit to "
        f"{p_hard[anchor_i]:.0f} here, at the level where CGM measurement error makes "
        "the true class genuinely ambiguous.",
        icon="🎯",
    )

# --------------------------------------------------------------------------- #
# Panel 3: patient-level behaviour
# --------------------------------------------------------------------------- #
st.divider()
tab1, tab2, tab3 = st.tabs(["Forecast vs persistence", "Alert timeline", "Soft-label curve"])

with tab1:
    rep = evaluate(p_truth, p_pred, p_hard, p_risk, p_last,
                   ws.patient[p_idx], PERIOD,
                   t_time=ws.t_time[p_idx], horizon_min=cfg.data.horizon_min)
    m = st.columns(5)
    m[0].metric("RMSE", f"{rep.rmse:.2f}", f"{rep.rmse - rep.rmse_persistence:+.2f} vs persist",
                delta_color="inverse")
    m[1].metric("MARD", f"{rep.mard:.2f}%")
    m[2].metric("Time lag", f"{rep.time_lag_min:.0f} min",
                f"persist {rep.time_lag_persistence_min:.0f}", delta_color="off")
    m[3].metric("PR-AUC", f"{rep.pr_auc:.3f}", f"floor {rep.pr_auc_prevalence:.3f}",
                delta_color="off")
    m[4].metric("ROC-AUC", f"{rep.roc_auc:.3f}")

    if rep.time_lag_min >= PH_MIN:
        st.error(
            f"**Time lag equals the full horizon ({rep.time_lag_min:.0f} min).** The model "
            "is reproducing persistence: it is not anticipating anything, it is echoing "
            "its most recent input. RMSE looks acceptable regardless, which is exactly why "
            "this metric is reported.",
            icon="🚨",
        )

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=p_time, y=p_truth, name="Truth", mode="lines",
                            line=dict(color=C_TRUTH, width=1.4)))
    f2.add_trace(go.Scatter(x=p_time, y=p_pred, name="Forecast", mode="lines",
                            line=dict(color=C_PRED, width=1.4)))
    if show_persist:
        f2.add_trace(go.Scatter(x=p_time, y=p_last, name="Persistence", mode="lines",
                                line=dict(color=C_PERSIST, width=1, dash="dot")))
    f2.add_hline(y=THR, line=dict(color=C_HYPO, width=1, dash="dot"))
    f2.add_vline(x=t_anchor, line=dict(color="#000", width=1.5))
    f2.update_layout(height=380, hovermode="x unified", margin=dict(t=20),
                     yaxis_title="mg/dL", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f2, use_container_width=True)
    st.caption(
        "If the forecast trace is the persistence trace shifted right, the model has "
        "learned nothing beyond the identity — the visual signature of a lag equal to PH."
    )

with tab2:
    alerts = p_risk >= alert_thr
    tp = int((alerts & (p_hard > 0)).sum())
    fp = int((alerts & (p_hard == 0)).sum())
    fn = int((~alerts & (p_hard > 0)).sum())
    hours = max((p_time[-1] - p_time[0]).total_seconds() / 3600, 1e-9)

    a = st.columns(4)
    a[0].metric("True alerts", tp)
    a[1].metric("False alarms", fp)
    a[2].metric("Missed events", fn)
    a[3].metric("False alarms / day", f"{fp / hours * 24:.1f}")

    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=p_time, y=p_truth, name="Truth", mode="lines",
                            line=dict(color=C_TRUTH, width=1.2)))
    f3.add_hline(y=THR, line=dict(color=C_HYPO, width=1, dash="dot"))
    for label, mask, colour, symbol in [
        ("True alert", alerts & (p_hard > 0), "#2f855a", "triangle-up"),
        ("False alarm", alerts & (p_hard == 0), "#d69e2e", "x"),
        ("Missed", ~alerts & (p_hard > 0), C_HYPO, "triangle-down"),
    ]:
        if mask.any():
            f3.add_trace(go.Scatter(
                x=p_time[mask], y=p_truth[mask], name=label, mode="markers",
                marker=dict(color=colour, size=7, symbol=symbol)))
    f3.update_layout(height=400, hovermode="closest", margin=dict(t=20),
                     yaxis_title="mg/dL", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f3, use_container_width=True)
    st.caption(
        f"At threshold {alert_thr:.2f} this patient sees {fp / hours * 24:.1f} false "
        f"alarms per day. Scrub the threshold slider to find the operating point where "
        "that rate becomes tolerable, then read off what sensitivity costs."
    )

with tab3:
    grid = np.linspace(THR - 4 * labeler.band, THR + 4 * labeler.band, 400)
    soft_curve = labeler(grid.astype(np.float32))
    hard_curve = (grid < THR).astype(float)

    f4 = make_subplots(specs=[[{"secondary_y": True}]])
    f4.add_trace(go.Scatter(x=grid, y=hard_curve, name="Hard label", mode="lines",
                            line=dict(color=C_PERSIST, width=2, dash="dot")))
    f4.add_trace(go.Scatter(x=grid, y=soft_curve, name="Soft label", mode="lines",
                            line=dict(color=C_HYPO, width=3)))
    f4.add_trace(
        go.Histogram(x=p_min, nbinsx=60, name="min BG in window",
                     marker=dict(color=C_TRUTH), opacity=0.35),
        secondary_y=True,
    )
    f4.add_vline(x=THR, line=dict(color="#000", width=1.5, dash="dash"))
    f4.update_xaxes(title="min BG over (t, t+PH]  (mg/dL)",
                    range=[grid[0], grid[-1]])
    f4.update_yaxes(title="label", range=[-0.05, 1.05], secondary_y=False)
    f4.update_yaxes(title="count", secondary_y=True, showgrid=False)
    f4.update_layout(height=400, margin=dict(t=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f4, use_container_width=True)

    in_band = float(((p_min > THR - labeler.band) & (p_min < THR + labeler.band)).mean())
    st.caption(
        f"Band width ±{labeler.band:.1f} mg/dL, derived from the **training fold's** "
        f"noise estimate. {in_band:.1%} of this patient's windows fall inside it — the "
        "fraction of the training signal the soft label actually changes. If that number "
        "is near zero the soft-label strategy cannot matter on this cohort, whatever the "
        "ablation reports."
    )

st.divider()
st.caption(
    "Read-only diagnostic: the dashboard fits nothing. The scaler and soft-labeler are "
    "loaded from the training fold and applied unchanged, so nothing shown here can leak "
    "into the model. Synthetic cohort — validates plumbing, not clinical performance."
)
