"""Publication figures, built from the artefacts the experiment drivers wrote.

    python scripts/make_paper_figures.py                 # all seven
    python scripts/make_paper_figures.py --figures 3 4   # a subset

Every data figure is a pure function of a file under `runs/paper_results/`.
Nothing here trains, scores, or recomputes a metric: a plotting script that
re-derives its own numbers is a second experiment that can silently disagree
with the table it sits next to, and the disagreement is invisible because both
look plausible. Figures 3-6 therefore read prediction dumps and summary JSON,
and fail loudly with the command to regenerate them when a file is missing.

Figures 1 and 2 are schematics with no data behind them; they are drawn from
constants defined here and labelled as diagrams in the manuscript.

Figure 7 is the one exception to the "fail loudly" rule, and only for a missing
file: it reads `runs/metabonet_results/`, which is not produced by the paper
pipeline, so it SKIPS when that tree is absent and refuses it only when it is
present and marked as a smoke run.

Output: `manuscript/figures/figN_*.pdf` and `.png` at 300 DPI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RESULTS = "runs/paper_results"
# The corpus-scale validation is a SEPARATE results tree on purpose: it is not
# a stage of `reproduce_all_paper_results.sh` and its driver is forbidden to
# write into the paper tree. Read-only from here, exactly as
# `rebuild_paper_tables.py` reads it for the \NumMetabo* macros.
METABONET = "runs/metabonet_results"
OUTDIR = "manuscript/figures"

# Okabe-Ito: colourblind-safe, and distinguishable in greyscale print.
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442",
    "sky": "#56B4E9", "grey": "#666666", "light": "#DDDDDD",
}
ZONE_COLOR = {"A": CB["green"], "B": CB["sky"], "C": CB["yellow"],
              "D": CB["orange"], "E": CB["red"]}


# The manuscript's text block is 6.5 in wide (\textwidth = 469.75pt). Every
# figure is drawn at the width it is INCLUDED at, so a 7 pt label prints at 7 pt.
# They were all drawn 7.4 in wide and scaled down by \includegraphics -- to 54%
# in one case, which printed 7 pt labels at under 4 pt.
TEXTWIDTH_IN = 6.5


def set_style() -> None:
    """Academic typography. Times/Helvetica with a DejaVu fallback, because a
    missing font silently substitutes and changes every figure's metrics."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:                      # older matplotlib naming
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real fonts, not Type-3
    })


def save(fig, name: str) -> list[str]:
    os.makedirs(OUTDIR, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300)
        out.append(path)
    plt.close(fig)
    print(f"  wrote {' and '.join(os.path.basename(p) for p in out)}")
    return out


def _need(path: str, how: str):
    if not os.path.exists(path):
        raise SystemExit(
            f"missing artefact: {path}\nRegenerate it with:\n    {how}")
    return path


def load_json(path: str, how: str) -> dict:
    with open(_need(path, how)) as f:
        return json.load(f)


def load_predictions(study: str, arm_file: str, how: str) -> dict:
    if study:
        candidates = [
            os.path.join(RESULTS, "ablations", "predictions", study, arm_file),
            os.path.join(RESULTS, "predictions", study, arm_file),
        ]
    else:
        candidates = [
            os.path.join(RESULTS, "benchmarks", "predictions", arm_file),
            os.path.join(RESULTS, "predictions", arm_file),
        ]
    for p in candidates:
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            return {k: z[k] for k in z.files}
    _need(candidates[0], how)


# --------------------------------------------------------------------------- #
# Small drawing helpers for the schematic figures
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, text, fc, ec=None, fontsize=7.5, weight="normal",
        tc="black", radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc, edgecolor=ec or CB["grey"], linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=tc, zorder=3,
            linespacing=1.35)


def arrow(ax, xy_from, xy_to, style="-|>", color=None, lw=1.0, ls="-",
          rad=0.0, zorder=1, shrinkB=4.0):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=9,
        color=color or CB["grey"], linewidth=lw, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=shrinkB))


def blank_axes(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)


# --------------------------------------------------------------------------- #
# Figure 1: architecture
# --------------------------------------------------------------------------- #
def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    blank_axes(ax)

    W = 0.168
    xs = [0.012, 0.212, 0.412, 0.612, 0.818]          # column left edges

    # (a) input
    box(ax, xs[0], 0.62, W, 0.26,
        "(a)  Causal window\n" r"$[\,t-H+1,\ t\,]$" "\n\n"
        "CGM, IOB/COB,\nHR / GSR / accel\nbackward-only bins",
        "#EAF3FA", CB["blue"], fontsize=7)
    box(ax, xs[0], 0.26, W, 0.24,
        "Per-window prefix\ninterpolation\n"
        r"gaps $\leq 15$ min," "\n" r"both endpoints $\leq t$",
        "#F5F5F5", fontsize=7)
    arrow(ax, (xs[0] + W / 2, 0.62), (xs[0] + W / 2, 0.50))

    # (b) trunks
    box(ax, xs[1], 0.66, W, 0.22,
        "(b)  Causal TCN\nCGM branch\n" r"dilations $1,2,4,8$",
        "#E7F3EE", CB["green"], fontsize=7)
    box(ax, xs[1], 0.30, W, 0.22,
        "Causal TCN\nauxiliary branch\nnarrower trunk",
        "#E7F3EE", CB["green"], fontsize=7)
    arrow(ax, (xs[0] + W, 0.75), (xs[1], 0.77))
    arrow(ax, (xs[0] + W, 0.38), (xs[1], 0.41))

    # (c) fusion
    box(ax, xs[2], 0.36, W, 0.52,
        "(c)  Causal\ncross-attention\n\n"
        r"query $\leftarrow$ CGM" "\n" r"key/value $\leftarrow$ aux" "\n"
        r"mask: $j \leq i$; gate open if" "\n" "any obs. mask set or\ntreatment non-zero\n\n"
        r"$h = h_{\mathrm{cgm}}$" "\n" r"$+\ g\odot\mathrm{LN}(f)\,m$" "\n"
        "residual + gate:\naux modulates,\nnever replaces",
        "#FBF0E4", CB["orange"], fontsize=7)
    arrow(ax, (xs[1] + W, 0.77), (xs[2], 0.74))
    arrow(ax, (xs[1] + W, 0.41), (xs[2], 0.50))

    # (d) heads
    box(ax, xs[3], 0.70, W, 0.18,
        r"(d)  Heads @ $t{+}30$" "\n"
        r"$\hat{y}=\mathrm{BG}(t)+\Delta$" "\n"
        r"$p(\min \mathrm{BG} < 70)$", "#EFE8F4", CB["purple"], fontsize=7)
    box(ax, xs[3], 0.48, W, 0.18,
        r"Heads @ $t{+}60$" "\n"
        r"$\hat{y}=\mathrm{BG}(t)+\Delta$" "\n"
        r"$p(\min \mathrm{BG} < 70)$", "#EFE8F4", CB["purple"], fontsize=7)
    arrow(ax, (xs[2] + W, 0.68), (xs[3], 0.79))
    arrow(ax, (xs[2] + W, 0.56), (xs[3], 0.57))
    box(ax, xs[3], 0.18, W, 0.22,
        "Shared trunk and fusion;\nonly the readout is\nhorizon-specific\n\n"
        "Kendall " r"$\sum_i e^{-s_i}L_i + s_i/2$" "\n"
        r"+ asymmetric Huber $\kappa$",
        "#F5F5F5", fontsize=7)
    arrow(ax, (xs[3] + W / 2, 0.48), (xs[3] + W / 2, 0.40), ls=(0, (3, 2)))

    # (e) deployment
    box(ax, xs[4], 0.60, 0.17, 0.28,
        "(e)  Mondrian\nconformal calibration\n"
        r"$[\hat{y}\pm\hat{q}_k]$ per" "\nglycemic zone",
        "#FAE9E4", CB["red"], fontsize=7)
    box(ax, xs[4], 0.24, 0.17, 0.26,
        "C++ ONNX Runtime\nserving engine\n\n"
        "sub-millisecond\nper-window inference",
        "#FAE9E4", CB["red"], fontsize=7)
    arrow(ax, (xs[3] + W, 0.79), (xs[4], 0.76))
    arrow(ax, (xs[4] + 0.085, 0.60), (xs[4] + 0.085, 0.50), ls=(0, (3, 2)))

    ax.text(0.5, 0.075,
            "Every arrow is forward in time. No block reads a sample later than "
            r"$t$; the targets over $(t,\ t{+}PH\,]$ are never inputs.",
            ha="center", va="center", fontsize=7.4, color=CB["grey"])
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------- #
# Figure 2: anti-leakage protocol
# --------------------------------------------------------------------------- #
def figure2() -> None:
    """Schematic contrast. The geometry is drawn to scale in sample units, so a
    band that crosses the anchor line really does read the future."""
    fig, axes = plt.subplots(1, 2, figsize=(0.9 * TEXTWIDTH_IN, 2.8), sharey=True)
    n, anchor = 26, 15
    rows_y = {"roll": 3.0, "bin": 2.0, "interp": 1.0, "split": -0.15}

    for ax, leaky in zip(axes, (True, False)):
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-1.5, 4.9)
        ax.grid(False)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        colour = CB["red"] if leaky else CB["green"]
        ax.set_title("Standard pipeline\n(as commonly published)" if leaky
                     else "Structural causal pipeline\n(ours)",
                     fontsize=9, color=colour, pad=6)

        # sample timeline
        for i in range(n):
            ax.add_patch(plt.Rectangle(
                (i - 0.40, 4.15), 0.80, 0.45,
                facecolor=CB["light"] if i <= anchor else "#FFFFFF",
                edgecolor=CB["grey"], linewidth=0.4))
        ax.axvline(anchor + 0.5, color="black", lw=1.0, ls="--", zorder=0)
        ax.text(anchor + 0.5, 4.72, r"anchor $t$", ha="center", fontsize=7.5)
        ax.text(n - 0.5, 4.72, "future", ha="right", fontsize=7,
                color=CB["grey"], style="italic")

        spans = [
            ("roll", (anchor - 3, anchor + 3) if leaky else (anchor - 5, anchor),
             r"centred: reads $t{+}3$" if leaky else r"backward: $[\,t-5,\ t\,]$"),
            ("bin", (anchor, anchor + 3) if leaky else (anchor - 3, anchor),
             "forward-labelled bin" if leaky else "closed interval, label at right"),
            ("interp", (anchor - 2, anchor + 4) if leaky else (anchor - 4, anchor),
             "full-series, two-sided" if leaky else r"per-window, endpoints $\leq t$"),
        ]
        for key, (a, b), note in spans:
            y = rows_y[key]
            ax.add_patch(plt.Rectangle(
                (a - 0.40, y), (b - a) + 0.80, 0.45, facecolor=colour, alpha=0.30,
                edgecolor=colour, linewidth=0.8))
            if leaky and b > anchor:
                ax.annotate("", xy=(b + 0.40, y + 0.225),
                            xytext=(anchor + 0.5, y + 0.225),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.2))
            ax.text((a + b) / 2, y - 0.30, note, ha="center", va="top",
                    fontsize=6.8, style="italic", color=colour, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

        # split geometry
        y = rows_y["split"]
        if leaky:
            rng = np.random.default_rng(3)
            is_test = rng.permutation(n) < 6
            for i in range(n):
                ax.add_patch(plt.Rectangle(
                    (i - 0.40, y), 0.80, 0.45,
                    facecolor=colour if is_test[i] else CB["light"],
                    alpha=0.85 if is_test[i] else 1.0,
                    edgecolor=CB["grey"], linewidth=0.4))
            note = ("random split, no embargo:\n"
                    "test windows interleave training ones")
        else:
            emb_a, emb_b = 11, 18
            for i in range(n):
                fc = CB["light"] if i < emb_a else ("#FFFFFF" if i <= emb_b else colour)
                ax.add_patch(plt.Rectangle(
                    (i - 0.40, y), 0.80, 0.45, facecolor=fc,
                    alpha=0.55 if i > emb_b else 1.0,
                    edgecolor=CB["grey"], linewidth=0.4))
            ax.annotate("", xy=(emb_b + 0.40, y - 0.28), xytext=(emb_a - 0.40, y - 0.28),
                        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.9))
            ax.text((emb_a + emb_b) / 2, y - 0.46,
                    r"embargo $\geq H + PH + 1$ period", ha="center", va="top",
                    fontsize=6.8, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
            note = "chronological split with embargo:\nno shared raw sample"
        ax.text(n / 2 - 0.5, y - 0.92, note, ha="center", va="top",
                fontsize=6.8, style="italic", color=colour, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # Row labels once, on the shared left axis.
    axes[0].set_yticks([4.375] + [rows_y[k] + 0.225 for k in
                                  ("roll", "bin", "interp", "split")])
    axes[0].set_yticklabels(["samples", "rolling statistics", "resample / binning",
                             "gap interpolation", "train / test split"],
                            fontsize=7.5)
    axes[0].tick_params(axis="y", length=0)
    # sharey leaves tick marks on the right panel that read as stray minus signs
    axes[1].tick_params(axis="y", length=0, labelleft=False)

    # No suptitle: the caption carries it, and it cost body space.
    fig.tight_layout()
    save(fig, "fig2_anti_leakage_protocol")


# --------------------------------------------------------------------------- #
# Figure 3: Clarke error grids under the asymmetry sweep
# --------------------------------------------------------------------------- #
_ABL_HOW = ("python scripts/run_paper_ablations.py --root OhioT1DM --studies b")


def _clarke_background(ax, lim=400):
    """The published Clarke boundaries, drawn as guides.

    Only the lines are drawn from geometry; every POINT and every zone
    percentage comes from `clinical.clarke_zones`, so the figure cannot disagree
    with the table by drawing its own grid slightly differently.
    """
    ax.plot([0, lim], [0, lim], color=CB["grey"], lw=0.7, ls="--", zorder=1)
    for f in (1.2, 0.8):
        ax.plot([0, lim], [0, lim * f], color=CB["grey"], lw=0.5, ls=":", zorder=1)
    ax.axhline(70, color=CB["grey"], lw=0.5, zorder=1)
    ax.axvline(70, color=CB["grey"], lw=0.5, zorder=1)
    ax.axhline(180, color=CB["grey"], lw=0.5, zorder=1)
    ax.axvline(180, color=CB["grey"], lw=0.5, zorder=1)


def figure3(horizon: int = 30, kappas=(1, 3, 8), max_points: int = 6000) -> None:
    from causalcgm_mt.clinical import ZONE_NAMES, clarke_zones

    fig, axes = plt.subplots(1, len(kappas), figsize=(TEXTWIDTH_IN, 2.7), sharex=True,
                             sharey=True)
    rng = np.random.default_rng(0)

    for ax, k in zip(np.atleast_1d(axes), kappas):
        d = load_predictions("b_asymmetry", _kappa_file(k), _ABL_HOW)
        col = int(np.flatnonzero(d["horizons"] == horizon)[0])
        y, p = d["y_reg"][:, col].astype(float), d["bg_hat"][:, col].astype(float)
        zones = clarke_zones(y, p)

        # Subsample only what is DRAWN; every percentage below is computed on
        # the full test set, so the annotation is not a property of the sample.
        idx = (rng.choice(y.size, max_points, replace=False)
               if y.size > max_points else np.arange(y.size))
        order = np.argsort(-zones[idx])       # draw dangerous zones last, on top
        sel = idx[order]
        ax.scatter(y[sel], p[sel], s=1.6, linewidths=0,
                   c=[ZONE_COLOR[ZONE_NAMES[z]] for z in zones[sel]],
                   alpha=0.55, zorder=2, rasterized=True)
        _clarke_background(ax)

        pct = {n: float((zones == i).mean() * 100) for i, n in enumerate(ZONE_NAMES)}
        de = int(((zones == 3) | (zones == 4)).sum())
        ax.set_title(rf"$\kappa = {k}$", fontsize=9.5)
        ax.text(0.03, 0.97,
                f"A+B  {pct['A'] + pct['B']:.2f}%\nD+E  n = {de}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor=CB["light"], pad=2.5))
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_aspect("equal")
        ax.set_xlabel("reference BG (mg/dL)")

    np.atleast_1d(axes)[0].set_ylabel("predicted BG (mg/dL)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4,
                          color=ZONE_COLOR[z], label=f"zone {z}")
               for z in ZONE_NAMES]
    fig.suptitle(f"Clarke error grid on the official OhioT1DM test split, "
                 f"PH = {horizon} min", fontsize=10, y=1.04)
    fig.tight_layout()
    # After tight_layout, so the legend sits below the finished axes rather than
    # being laid out into them.
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.10))
    save(fig, "fig3_clarke_parkes_grids")


def _kappa_file(k) -> str:
    """`run_arm` sanitises the LaTeX arm name into a filename; mirror it here
    rather than guessing, so a rename in the driver surfaces as a missing-file
    error with a regeneration command instead of a silently empty figure."""
    name = rf"$\kappa$ = {k:g}"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) + ".npz"


# --------------------------------------------------------------------------- #
# Figure 4: conditional coverage
# --------------------------------------------------------------------------- #
_BENCH_HOW = "python scripts/run_paper_benchmarks.py --root OhioT1DM"

# The bands the figure reports. The first three come from `coverage_by_range`
# (fine-grained clinical reporting); the marginal number is the pooled one.
_BANDS = [
    ("marginal", "all windows"),
    ("euglycemic (90-180)", "euglycemic\n90-180"),
    ("hypo (54-70)", "mild hypo\n54-70"),
    ("severe hypo (<54)", "severe hypo\n<54"),
]
_VARIANTS = [("split", "constant width", CB["blue"]),
             ("normalized", "normalised", CB["orange"]),
             ("mondrian", "Mondrian", CB["green"])]


def figure4(arm: str = "CausalCGM-MT (fusion)", horizon: int = 30) -> None:
    res = load_json(os.path.join(RESULTS, "benchmarks", "results_summary.json"),
                    _BENCH_HOW)
    row = next((a for a in res["arms"] if a["arm"] == arm), None)
    if row is None:
        raise SystemExit(f"arm {arm!r} not in the benchmark results; have "
                         f"{[a['arm'] for a in res['arms']]}")
    conf = row["per_horizon"][str(horizon)]["conformal"]
    target = 1.0 - conf["alpha"]

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0),
                                  gridspec_kw={"width_ratios": [1.55, 1.0]})
    width = 0.26
    xs = np.arange(len(_BANDS))

    for j, (key, label, colour) in enumerate(_VARIANTS):
        blk = conf[key]
        vals, ns = [], []
        for band, _pretty in _BANDS:
            if band == "marginal":
                vals.append(blk["marginal_coverage"])
                ns.append(sum(v["n"] for v in blk["by_clinical_range"].values()))
            else:
                cell = blk["by_clinical_range"].get(band)
                vals.append(cell["coverage"] if cell else np.nan)
                ns.append(cell["n"] if cell else 0)
        pos = xs + (j - 1) * width
        ax.bar(pos, vals, width * 0.92, label=label, color=colour, alpha=0.9,
               edgecolor="white", linewidth=0.5)
        for x, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=colour)

    # The target is named in the legend: a text label on the line collides with
    # the bar-value labels of the severe band.
    ax.axhline(target, color="black", lw=1.0, ls="--", zorder=3,
               label=f"target {target:.0%}")
    ax.set_xticks(xs)
    ax.set_xticklabels([p for _b, p in _BANDS], fontsize=7.5)
    # Coverage lives in [0.7, 1]; starting the axis at zero wastes two thirds of
    # the panel on space no conformal interval can occupy and flattens the
    # differences the figure exists to show.
    ax.set_ylim(0.70, 1.05)
    ax.set_ylabel("empirical coverage")
    ax.legend(loc="upper center", fontsize=7.5, ncol=4, columnspacing=1.2,
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Coverage by clinical band", fontsize=9.5)
    counts = []
    for band, _pretty in _BANDS:
        if band == "marginal":
            counts.append(sum(v["n"] for v in
                              conf["split"]["by_clinical_range"].values()))
        else:
            counts.append(conf["split"]["by_clinical_range"].get(band, {}).get("n", 0))
    ax.set_xticklabels([f"{p}\n$n = {n:,}$" for (_b, p), n in zip(_BANDS, counts)],
                       fontsize=7.5)

    # Width is the other half of the trade: coverage bought with width is not
    # free, and a figure that showed only coverage would hide the cost.
    for j, (key, label, colour) in enumerate(_VARIANTS):
        blk = conf[key]
        hypo = blk.get("hypo_stratum_width")
        axw.bar([j - 0.17], [blk["mean_width"]], 0.32, color=colour, alpha=0.95,
                edgecolor="white", linewidth=0.5)
        if hypo:
            axw.bar([j + 0.17], [hypo], 0.32, color=colour, alpha=0.40,
                    edgecolor=colour, linewidth=0.8, hatch="//")
        axw.text(j - 0.17, blk["mean_width"] + 1.2, f"{blk['mean_width']:.0f}",
                 ha="center", fontsize=7, color=colour)
        if hypo:
            axw.text(j + 0.17, hypo + 1.2, f"{hypo:.0f}", ha="center",
                     fontsize=7, color=colour)
    axw.set_xticks(range(len(_VARIANTS)))
    axw.set_xticklabels([lbl for _k, lbl, _c in _VARIANTS], fontsize=7.5)
    axw.set_ylabel("mean interval width (mg/dL)")
    axw.set_title("Width: all windows (solid)\nvs hypoglycemic stratum (hatched)",
                  fontsize=9.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.suptitle(
        f"{arm}, official OhioT1DM test split, PH = {horizon} min "
        f"(calibrated on the disjoint calibration split)", fontsize=9.5, y=1.03)
    save(fig, "fig4_conformal_coverage")


# --------------------------------------------------------------------------- #
# Figure 5: asymmetry trade-off curves
# --------------------------------------------------------------------------- #
def figure5(horizon: int = 30) -> None:
    abl = load_json(os.path.join(RESULTS, "ablations", "ablations_summary.json"),
                    _ABL_HOW)
    rows = abl["studies"].get("b_asymmetry")
    if not rows:
        raise SystemExit("no b_asymmetry study in the ablation summary; run:\n"
                         f"    {_ABL_HOW}")

    def kappa_of(r) -> float:
        return float(r["spec"]["hypo_asymmetry"])

    rows = sorted(rows, key=kappa_of)
    k = np.array([kappa_of(r) for r in rows])
    g = lambda path: np.array(  # noqa: E731
        [dig(r["per_horizon"][str(horizon)], path, np.nan) for r in rows],
        dtype=float)

    # Drawn at the full text width it is included at (it was 0.8 textwidth
    # and 2.3 in tall, which printed its annotations at about 6 pt).
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.7))

    axa.plot(k, g("regression.rmse"), "o-", color=CB["blue"], lw=1.4, ms=4,
             label="RMSE, all windows")
    axa.plot(k, g("clinical.rmse_below_70"), "s-", color=CB["red"], lw=1.4, ms=4,
             label=r"RMSE, reference $<$ 70")
    axa.set_xlabel(r"asymmetry $\kappa$")
    axa.set_ylabel("RMSE (mg/dL)")
    axa.set_xticks(k)
    axa.set_xlim(k.min() - 0.55, k.max() + 0.55)
    axa.set_title("A.  What the asymmetry costs, and buys", fontsize=9.5)

    ax2 = axa.twinx()
    ax2.plot(k, g("clinical.hypo_band_bias"), "^--", color=CB["purple"], lw=1.2,
             ms=4, label=r"signed bias, reference $<$ 70")
    ax2.axhline(0.0, color=CB["grey"], lw=0.6, ls=":")
    ax2.set_ylabel("bias (mg/dL)", color=CB["purple"])
    ax2.tick_params(axis="y", colors=CB["purple"])
    ax2.grid(False)
    # One legend for both axes: a second legend box on the twin axis lands on
    # top of the very curve it labels.
    h1, l1 = axa.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # D+E counts sit to the right of their marker, except the last point, which
    # would run off the axes and goes below-left of it, clear of the right axis.
    de = g("clinical.zone_de_n")
    for i, (x, y, n) in enumerate(zip(k, g("clinical.hypo_band_bias"), de)):
        last = i == len(k) - 1
        ax2.annotate(f"D+E {int(n)}", (x, y), textcoords="offset points",
                     xytext=(-4, -12) if last else (5, 3),
                     ha="right" if last else "left", fontsize=7.5,
                     color=CB["purple"])

    axb.plot(k, g("classification.pr_auc"), "o-", color=CB["green"], lw=1.4, ms=4,
             label="PR-AUC")
    axb.plot(k, g("classification.sensitivity_at_5pct_fpr"), "s-",
             color=CB["orange"], lw=1.4, ms=4, label="Sensitivity @ 5% FPR")
    axb.set_xlabel(r"asymmetry $\kappa$")
    axb.set_ylabel("forewarning metric")
    axb.set_xticks(k)
    axb.set_title("B.  What it costs the forewarning task", fontsize=9.5)

    # One legend row under both panels. Inside the axes, a legend at this size
    # lands on the curves it labels. No suptitle: the caption carries it.
    h3, l3 = axb.get_legend_handles_labels()
    fig.tight_layout(w_pad=2.2, rect=(0, 0.12, 1, 1))
    fig.legend(h1 + h2 + h3, l1 + [l2[0] + " (right axis)"] + l3, loc="lower center",
               ncol=3, fontsize=8, frameon=False, columnspacing=1.4,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig5_ablation_tradeoffs")


def dig(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


# --------------------------------------------------------------------------- #
# Figure 6: edge latency
# --------------------------------------------------------------------------- #
_EDGE_HOW = "python scripts/run_cpp_latency_sweep.py"


def figure6() -> None:
    d = load_json(os.path.join(RESULTS, "edge", "latency_sweep.json"), _EDGE_HOW)
    runs, deadline = d["runs"], d["deadline_ms"]
    batches = sorted({r["batch"] for r in runs})
    threads = sorted({r["threads"] for r in runs})

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0))
    colours = [CB["blue"], CB["orange"], CB["green"], CB["purple"]]
    width = 0.8 / max(len(threads), 1)

    for j, th in enumerate(threads):
        xs, p50, p99, mx = [], [], [], []
        for i, b in enumerate(batches):
            r = next((r for r in runs if r["batch"] == b and r["threads"] == th), None)
            if r is None:
                continue
            xs.append(i + (j - (len(threads) - 1) / 2) * width)
            p50.append(r["latency_ms"]["p50"])
            p99.append(r["latency_ms"]["p99"])
            mx.append(r["latency_ms"]["max"])
        c = colours[j % len(colours)]
        axl.bar(xs, p50, width * 0.9, color=c, alpha=0.85,
                label=f"{th} thread" + ("s" if th > 1 else ""),
                edgecolor="white", linewidth=0.4)
        axl.vlines(xs, p50, p99, color=c, lw=1.2)
        axl.scatter(xs, mx, marker="_", s=48, color=c, zorder=3)

    axl.axhline(deadline, color=CB["red"], lw=1.2, ls="--")
    axl.text(len(batches) - 0.5, deadline * 1.15, f"{deadline:.0f} ms deadline",
             ha="right", color=CB["red"], fontsize=7.5)
    axl.set_yscale("log")
    axl.set_ylim(bottom=0.08, top=deadline * 3.2)  # Added bottom=0.08 to show sub-millisecond values cleanly

    axl.set_xticks(range(len(batches)))
    axl.set_xticklabels([str(b) for b in batches])
    axl.set_xlabel("batch size (windows per Run call)")
    axl.set_ylabel("latency per Run call (ms, log scale)")
    axl.set_title("A.  Bar = p50, whisker = p99, dash = max", fontsize=9.5)
    axl.legend(fontsize=7.5, loc="upper left", ncol=3, columnspacing=0.8)

    for j, th in enumerate(threads):
        pts = [(r["batch"], r["latency_ms"]["p50"] / r["batch"])
               for r in runs if r["threads"] == th]
        pts.sort()
        axr.plot([p[0] for p in pts], [p[1] * 1000 for p in pts], "o-",
                 color=colours[j % len(colours)], lw=1.4, ms=4,
                 label=f"{th} thread" + ("s" if th > 1 else ""))
    axr.set_xscale("log", base=2)
    axr.set_ylim(bottom=0)
    axr.set_xticks(batches)
    axr.set_xticklabels([str(b) for b in batches])
    axr.set_xlabel("batch size")
    axr.set_ylabel(r"amortised latency per window ($\mu$s)")
    axr.set_title("B.  Cost of one prediction", fontsize=9.5)
    axr.legend(fontsize=7.5)

    single = next((r for r in runs if r["batch"] == 1 and r["threads"] == 1), None)
    if single:
        axr.annotate(
            f"batch 1, 1 thread: p50 = {single['latency_ms']['p50']:.3f} ms\n"
            f"(one prediction, one core, no batching)",
            xy=(1, single["latency_ms"]["p50"] * 1000),
            xytext=(0.06, 0.06), textcoords="axes fraction", fontsize=7,
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=CB["grey"],
                            connectionstyle="arc3,rad=-0.25"))

    host = d.get("host", {}).get("machine", "")
    fig.suptitle(f"C++ ONNX Runtime engine, percentiles over "
                 f"{runs[0]['iterations']} timed calls" + (f" ({host})" if host else ""),
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_edge_latency")



# --------------------------------------------------------------------------- #
# Figure 7: corpus-scale external validation (appendix)
# --------------------------------------------------------------------------- #
_METABO_HOW = ("python scripts/run_metabonet_benchmarks.py "
               "--max-windows-per-subject 400 --memory-budget-gb 12")


def _metabonet_summary():
    """The appendix artefact, or `None` when the driver has not been run.

    ABSENT is not an error here, unlike every other figure in this file. The
    MetaboNet driver is not a stage of the reproduction pipeline, so a tree
    without it is a normal tree, and raising would make the paper's `figures`
    stage fail on the one artefact the pipeline never produces. A summary that
    is PRESENT and marked `"smoke": true` is a different matter and is refused:
    two-epoch numbers plotted at publication styling look exactly like results.
    """
    path = os.path.join(METABONET, "benchmarks", "metabonet_summary.json")
    if not os.path.exists(path):
        print(f"  skip: {path} not found; run:\n    {_METABO_HOW}")
        return None
    with open(path) as f:
        res = json.load(f)
    if res.get("smoke"):
        raise SystemExit(
            f"REFUSING {path}: it is a --smoke run and its numbers are not "
            f"reportable. Re-run at publication budget:\n    {_METABO_HOW}")
    return res


def _fold_series(per_fold, arm: str, horizon: int, section: str, key: str):
    """`{fold: value}` for one arm, so the two arms can be paired by fold."""
    out = {}
    for r in per_fold:
        if r["arm"] != arm:
            continue
        blk = r["per_horizon"].get(str(horizon), {}).get(section)
        if blk is not None and blk.get(key) is not None:
            out[r["fold"]] = float(blk[key])
    return out


def _nice_clip(a, plo: float = 0.5, phi: float = 99.0):
    """A plotting window read off the data, rounded outward to a round step.

    Hand-picked limits are a hardcoded number in a figure that is otherwise a
    pure function of an artefact: they happen to frame THIS run's distribution
    and would quietly amputate the next one. Percentiles keep the frame tied to
    the data; the rounding is only so the ticks are readable, and zero is always
    inside the window because it is the reference the panel is read against.
    """
    lo, hi = (float(v) for v in np.percentile(a, [plo, phi]))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = max(hi - lo, 1e-12)
    step = 10.0 ** np.floor(np.log10(span)) / 2.0
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def _delta_panel(ax, deltas, *, clip, bins, colour, xlabel, title,
                 worse_is, mean, median, q1, q3):
    """One per-subject delta histogram, with the tail stated rather than cut.

    Clipping a long tail off a histogram and saying nothing turns "a few
    subjects are destroyed by adaptation" into "no subject is", which is the
    half of this result that matters most for a device. The clipped mass is
    annotated with its count.
    """
    lo, hi = clip
    beyond = int(np.sum(deltas > hi) + np.sum(deltas < lo))
    counts, _, _ = ax.hist(np.clip(deltas, lo, hi), bins=bins, range=(lo, hi),
                           color=colour, alpha=0.80, edgecolor="white",
                           linewidth=0.35)
    ax.axvspan(q1, q3, color=colour, alpha=0.12, zorder=0)
    ax.axvline(0.0, color=CB["grey"], lw=1.0, ls="--")
    ax.axvline(median, color=CB["red"], lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("subjects")
    ax.set_xlim(lo, hi)
    # Headroom for the statistics box. Without it the box lands on the mode,
    # which is the one part of the distribution the reader came for.
    ax.set_ylim(0, counts.max() * 1.42)
    ax.set_title(title, fontsize=9.5)

    txt = (f"median {median:+.3g}\nIQR [{q1:+.3g}, {q3:+.3g}]\n"
           f"mean {mean:+.3g}\n{worse_is}")
    if beyond:
        txt += f"\n{beyond} beyond axis"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=CB["light"], linewidth=0.6))


def figure7(horizon: int = 30) -> None:
    res = _metabonet_summary()
    if res is None:
        return
    horizon = int(res.get("horizon_min", horizon))
    prov = res.get("provenance", {})
    per_fold = res.get("per_fold", [])

    cgm = _fold_series(per_fold, "cgm_only", horizon, "regression", "rmse")
    fus = _fold_series(per_fold, "fusion", horizon, "regression", "rmse")
    folds = sorted(set(cgm) & set(fus))
    if len(folds) < 2:
        raise SystemExit("figure 7 needs the cgm_only/fusion pair in at least "
                         f"two folds; run:\n    {_METABO_HOW}")

    pers = next((a.get("personalization", {}) for a in res.get("summary", [])
                 if a["arm"] == "lora"), {})
    subjects = pers.get("per_subject", []) if pers else []
    if not subjects:
        raise SystemExit("figure 7 needs the lora arm's per-subject deltas; "
                         f"run:\n    {_METABO_HOW}")

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.7))

    # -- A. the fusion gate, paired by fold ---------------------------------- #
    # Paired, because both arms are trained on identical splits: the unpaired
    # comparison throws away the pairing and lands the effect inside a
    # between-fold spread several times its own size.
    c = np.array([cgm[k] for k in folds])
    f = np.array([fus[k] for k in folds])
    d = f - c
    x = np.arange(len(folds))
    for xi, ci, fi in zip(x, c, f):
        axa.plot([xi - 0.16, xi + 0.16], [ci, fi], "-", color=CB["grey"],
                 lw=0.8, alpha=0.55, zorder=1)
    axa.scatter(x - 0.16, c, s=26, color=CB["blue"], zorder=3, label="unimodal (CGM)")
    axa.scatter(x + 0.16, f, s=26, color=CB["orange"], marker="s", zorder=3,
                label="fusion")
    axa.set_xticks(x)
    axa.set_xticklabels([f"{k + 1}" for k in folds])
    axa.set_xlabel("cross-validation fold")
    axa.set_ylabel("RMSE (mg/dL)")
    axa.set_xlim(-0.55, len(folds) - 0.45)
    # Room reserved BELOW the points for the paired-delta box and ABOVE for the
    # legend. Both were landing on the markers at the natural limits, and the
    # lowest fusion point -- the one fold where the gate moves at all -- was the
    # marker being covered.
    span = max(float(np.max(np.r_[c, f]) - np.min(np.r_[c, f])), 1e-6)
    axa.set_ylim(float(np.min(np.r_[c, f])) - 0.62 * span,
                 float(np.max(np.r_[c, f])) + 0.58 * span)
    # One row across the top of the reserved band. A boxed legend in the corner
    # sat on the highest fold's pair, which is the pair a reader checks first.
    axa.legend(fontsize=7.2, loc="upper center", ncol=2, handletextpad=0.4,
               columnspacing=1.1, borderaxespad=0.25, frameon=False)
    axa.set_title("A.  Fusion vs CGM-only, paired by fold", fontsize=9.5)
    axa.annotate(
        f"paired $\\Delta$ = {d.mean():+.3f} $\\pm$ {d.std(ddof=1):.3f} mg/dL\n"
        f"between-fold SD = {c.std(ddof=1):.3f} mg/dL",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=7, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor=CB["light"], linewidth=0.6))

    # -- B and C. what adaptation does to each held-out subject -------------- #
    d_rmse = np.array([s["rmse_after"] - s["rmse_before"] for s in subjects
                       if s.get("rmse_after") is not None
                       and s.get("rmse_before") is not None], dtype=float)
    # PR-AUC is undefined for a subject with no hypoglycemia in its evaluation
    # region, so this panel covers a subset of panel B's subjects and says so.
    d_prauc = np.array([s["pr_auc_after"] - s["pr_auc_before"] for s in subjects
                        if s.get("pr_auc_after") is not None
                        and s.get("pr_auc_before") is not None], dtype=float)

    q1, med, q3 = np.percentile(d_rmse, [25, 50, 75])
    worse = int(np.sum(d_rmse > 0))
    _delta_panel(
        axb, d_rmse, clip=_nice_clip(d_rmse), bins=44, colour=CB["purple"],
        xlabel=r"$\Delta$ RMSE after adaptation (mg/dL)",
        title=f"B.  Personalisation, {len(d_rmse):,} subjects",
        worse_is=f"{100.0 * worse / len(d_rmse):.1f}% worse",
        mean=float(d_rmse.mean()), median=float(med), q1=float(q1), q3=float(q3))

    if len(d_prauc):
        q1p, medp, q3p = np.percentile(d_prauc, [25, 50, 75])
        worse_p = int(np.sum(d_prauc < 0))
        _delta_panel(
            axc, d_prauc, clip=_nice_clip(d_prauc), bins=44, colour=CB["green"],
            xlabel=r"$\Delta$ PR-AUC after adaptation",
            title=f"C.  Forewarning, {len(d_prauc):,} subjects",
            worse_is=f"{100.0 * worse_p / len(d_prauc):.1f}% worse",
            mean=float(d_prauc.mean()), median=float(medp), q1=float(q1p),
            q3=float(q3p))
    else:
        blank_axes(axc)

    n_sub = prov.get("n_subjects")
    n_src = len(prov.get("sources", []))
    fig.suptitle(
        f"MetaboNet, {n_sub:,} subjects from {n_src} source studies, "
        f"{res.get('n_folds', len(folds))}-fold patient-wise CV, "
        f"PH = {horizon} min",
        fontsize=9.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig7_metabonet_corpus_scale")


# --------------------------------------------------------------------------- #
FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5,
           6: figure6, 7: figure7}


def main() -> None:
    global OUTDIR, RESULTS, METABONET

    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="*", type=int, default=sorted(FIGURES),
                    choices=sorted(FIGURES))
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--metabonet", default=METABONET,
                    help="corpus-scale validation results (figure 7); READ "
                         "ONLY, and skipped when absent")
    args = ap.parse_args()

    OUTDIR, RESULTS, METABONET = args.outdir, args.results, args.metabonet

    set_style()
    os.makedirs(OUTDIR, exist_ok=True)
    for n in args.figures:
        print(f"Figure {n}:")
        FIGURES[n]()
    print(f"\nAll figures in {OUTDIR}/")


if __name__ == "__main__":
    main()
