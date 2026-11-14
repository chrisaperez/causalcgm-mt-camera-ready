"""Aggregate the multi-seed OhioT1DM benchmark: mean +/- sd and bootstrap CIs.

    python scripts/aggregate_seed_sweep.py --seeds-dir runs/paper_results/benchmarks_seeds

Inputs are the per-seed outputs of `run_paper_benchmarks.py --seed S --out
<seeds-dir>/seedS`: `results_summary.json` plus `predictions/*.npz`.

Two kinds of uncertainty are reported, and they answer different questions:

* mean +/- sd ACROSS SEEDS -- how much the number moves when only the training
  randomness changes. Classical arms (persistence, Ridge-AR, ARIMA) are
  deterministic, and LightGBM's `random_state` is fixed, so their sd is 0.
* a SUBJECT-CLUSTER BOOTSTRAP CI of the seed-mean -- resample the 12 subjects
  with replacement, recompute each seed's metric on the resample, average over
  seeds. Twelve clusters make this CI approximate; it still captures the
  between-subject variance that dominates this benchmark, which a window-level
  bootstrap would understate by treating 28k autocorrelated windows as
  independent.

Paired differences against the fusion arm are bootstrapped the same way, on the
same resamples, so their CIs are paired.

Operating-point metrics (FAPD, sensitivity at the 5% target) use the threshold
each seed fitted on ITS calibration split, read from the results JSON; nothing
here selects a threshold from test labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.clinical import clarke_zones, segment_ids  # noqa: E402
from causalcgm_mt.evaluate import false_alarms_per_day, onset_eligible  # noqa: E402

REFERENCE_ARM = "CausalCGM-MT (fusion)"
HORIZONS = (30, 60)
# JSON paths of the per-seed metrics summarised as mean +/- sd.
SEED_METRICS = {
    "rmse": ("regression", "rmse"),
    "pr_auc": ("classification", "pr_auc"),
    "pr_auc_onset": ("classification", "pr_auc_onset"),
    "sens_5": ("classification", "sensitivity_at_5pct_fpr"),
    "realized_fpr_5": ("alarm", "realized_fpr_at_5pct"),
    "fapd_5": ("alarm", "fapd_at_5pct_fpr"),
    "onset_lead": ("alarm", "mean_onset_lead_min"),
    "onset_det": ("alarm", "onset_detection_rate"),
    "nadir_lead": ("alarm", "mean_lead_time_min"),
    "de": ("clinical", "zone_de_n"),
}
BOOT_METRICS = ("rmse", "pr_auc", "pr_auc_onset", "fapd_5", "de")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _get(d: dict, path: tuple[str, ...]):
    for p in path:
        d = d[p]
    return d


class SeedArm:
    """One arm's test predictions for one seed, pre-split by subject."""

    def __init__(self, npz: str, k: int, h: int, thr5: float):
        z = np.load(npz, allow_pickle=True)
        pat = z["patient"].astype(str)
        t = z["t_time"].astype("datetime64[ns]")
        order = np.lexsort((t.astype(np.int64), pat))
        self.pat = pat[order]
        y = z["y_reg"][:, k].astype(float)[order]
        p = z["bg_hat"][:, k].astype(float)[order]
        yc = z["y_cls"][:, k].astype(int)[order]
        s = z["risk"][:, k].astype(float)[order]
        last = z["last_bg"].astype(float)[order]
        seg = segment_ids(t[order], self.pat, 5, n=len(y))
        elig = onset_eligible(last, seg)
        zones = clarke_zones(y, p)
        self.subjects = sorted(set(self.pat))
        self.by = {}
        for sub in self.subjects:
            m = self.pat == sub
            days = m.sum() * 5.0 / 1440.0
            fapd = (false_alarms_per_day(yc[m], s[m], thr5, 5, seg[m])
                    if np.isfinite(thr5) else float("nan"))
            self.by[sub] = dict(
                sse=float(((y[m] - p[m]) ** 2).sum()), n=int(m.sum()),
                yc=yc[m], s=s[m], elig=elig[m],
                de=int((zones[m] >= 3).sum()),
                false_eps=fapd * days, days=days)

    def metrics(self, subs: list[str]) -> dict:
        parts = [self.by[x] for x in subs]
        yc = np.concatenate([q["yc"] for q in parts]); s = np.concatenate([q["s"] for q in parts])
        el = np.concatenate([q["elig"] for q in parts])
        n = sum(q["n"] for q in parts)
        pr = average_precision_score(yc, s) if 0 < yc.sum() < yc.size else np.nan
        yo, so = yc[el], s[el]
        pro = average_precision_score(yo, so) if 0 < yo.sum() < yo.size else np.nan
        return {"rmse": float(np.sqrt(sum(q["sse"] for q in parts) / n)),
                "pr_auc": float(pr), "pr_auc_onset": float(pro),
                "fapd_5": float(sum(q["false_eps"] for q in parts) / sum(q["days"] for q in parts)),
                "de": float(sum(q["de"] for q in parts))}


def aggregate(seeds_dir: str, n_boot: int) -> dict:
    seed_dirs = sorted(d for d in glob.glob(os.path.join(seeds_dir, "seed*"))
                       if os.path.exists(os.path.join(d, "results_summary.json")))
    if not seed_dirs:
        raise SystemExit(f"no seed results under {seeds_dir}")
    runs = {os.path.basename(d): json.load(open(os.path.join(d, "results_summary.json")))
            for d in seed_dirs}
    if any(r.get("smoke") for r in runs.values()):
        raise SystemExit("refusing to aggregate smoke artefacts")
    arms = [a["arm"] for a in next(iter(runs.values()))["arms"]]

    out: dict = {"seeds": sorted(runs), "n_boot": n_boot, "reference_arm": REFERENCE_ARM,
                 "bootstrap": "subject-cluster (12 subjects, resampled with replacement); "
                              "CI of the seed-mean",
                 "thresholds": "per seed, fitted on that seed's calibration split",
                 "arms": {}, "diffs_vs_reference": {}}
    rng = np.random.default_rng(20261006)

    for k, h in enumerate(HORIZONS):
        hk = str(h)
        # ---- mean +/- sd across seeds from the JSON -------------------------
        for arm in arms:
            vals = {m: [] for m in SEED_METRICS}
            for r in runs.values():
                a = next(x for x in r["arms"] if x["arm"] == arm)
                ph = a["per_horizon"][hk]
                for m, path in SEED_METRICS.items():
                    vals[m].append(float(_get(ph, path)))
                for name in ("split", "normalized", "mondrian"):
                    vals.setdefault(f"cov_lt54_{name}", []).append(
                        float(ph["conformal"][name]["severe_hypo_coverage"]))
            out["arms"].setdefault(arm, {})[hk] = {
                m: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))
                    if len(v) > 1 else 0.0, "per_seed": v}
                for m, v in vals.items()}

        # ---- subject-cluster bootstrap --------------------------------------
        loaded = {}
        for arm in arms:
            loaded[arm] = []
            for sd, r in zip(seed_dirs, runs.values()):
                a = next(x for x in r["arms"] if x["arm"] == arm)
                thr5 = float(a["per_horizon"][hk]["alarm"]["threshold_at_5pct_fpr"])
                loaded[arm].append(SeedArm(os.path.join(sd, "predictions", f"{_safe(arm)}.npz"),
                                           k, h, thr5))
        subjects = loaded[arms[0]][0].subjects
        draws = [list(rng.choice(subjects, size=len(subjects), replace=True))
                 for _ in range(n_boot)]

        def seed_mean(arm, subs):
            ms = [sa.metrics(subs) for sa in loaded[arm]]
            return {m: float(np.nanmean([x[m] for x in ms])) for m in BOOT_METRICS}

        boot = {arm: [seed_mean(arm, d) for d in draws] for arm in arms}
        point = {arm: seed_mean(arm, subjects) for arm in arms}
        for arm in arms:
            ci = out["arms"][arm][hk].setdefault("ci95", {})
            for m in BOOT_METRICS:
                a = np.array([b[m] for b in boot[arm]])
                ci[m] = [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
        for arm in arms:
            if arm == REFERENCE_ARM:
                continue
            d = {}
            for m in BOOT_METRICS:
                a = np.array([br[m] - bo[m] for br, bo in zip(boot[REFERENCE_ARM], boot[arm])])
                d[m] = {"diff": point[REFERENCE_ARM][m] - point[arm][m],
                        "ci95": [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]}
            out["diffs_vs_reference"].setdefault(arm, {})[hk] = d
        print(f"PH={h}: aggregated {len(arms)} arms over {len(runs)} seeds")
    return out


# --------------------------------------------------------------------------- #
# Rendering (also called by rebuild_paper_tables.py, so tables stay a pure
# function of the JSON)
# --------------------------------------------------------------------------- #
def _pm(cell: dict, fmt: str) -> str:
    return f"{fmt.format(cell['mean'])} $\\pm$ {fmt.format(cell['sd'])}"


def render_tables(summary: dict) -> dict[str, str]:
    n_seeds = len(summary["seeds"])
    tables = {}
    for hk in ("30", "60"):
        rows = []
        for arm, per in summary["arms"].items():
            c = per[hk]
            rows.append(" & ".join([
                arm.replace("$r = 4$", "r=4"),
                _pm(c["rmse"], "{:.2f}"), _pm(c["pr_auc"], "{:.3f}"),
                _pm(c["pr_auc_onset"], "{:.3f}"), _pm(c["sens_5"], "{:.3f}"),
                _pm(c["fapd_5"], "{:.2f}"), _pm(c["onset_lead"], "{:.1f}"),
                _pm(c["de"], "{:.0f}")]) + r" \\")
        tables[f"table_sota_multiseed_h{hk}.tex"] = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            f"\\caption{{OhioT1DM test split, $\\PH = {hk}$~min: mean $\\pm$ sd over "
            f"{n_seeds} training seeds. Alarm thresholds are fitted on the calibration "
            "split and applied unchanged to test. PR-AUC (onset) excludes anchors that "
            "are below 70~mg/dL now or in the preceding 30~min. Onset lead is minutes "
            "from the first valid alert to the first reading below 70. FAPD@5\\% counts "
            "false alarm episodes under the default alarm policy: an episode lasts while "
            "the score stays above threshold, with no merging across gaps and a minimum "
            "duration of one reading (5~min); Appendix~\\ref{sec:appendix_latching} "
            "varies both. Classical arms and LightGBM are deterministic (sd 0).}\n"
            f"\\label{{tab:sota_multiseed_ph{hk}}}\n"
            "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Arm & RMSE & PR-AUC & PR-AUC (onset) & Se@5\\% (cal.) & FAPD@5\\% "
            "& Onset lead & D+E \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
        drows = []
        for arm, per in summary["diffs_vs_reference"].items():
            d = per[hk]
            def cell(m, fmt):
                return (f"{fmt.format(d[m]['diff'])} "
                        f"[{fmt.format(d[m]['ci95'][0])}, {fmt.format(d[m]['ci95'][1])}]")
            drows.append(" & ".join([arm.replace("$r = 4$", "r=4"),
                                     cell("rmse", "{:+.2f}"), cell("pr_auc", "{:+.3f}"),
                                     cell("pr_auc_onset", "{:+.3f}"), cell("fapd_5", "{:+.2f}"),
                                     cell("de", "{:+.0f}")]) + r" \\")
        tables[f"table_multiseed_diffs_h{hk}.tex"] = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            f"\\caption{{Fusion minus each arm at $\\PH = {hk}$~min, seed-mean difference "
            "with a 95\\% subject-cluster bootstrap CI (12 subjects resampled, "
            f"{summary['n_boot']} draws). A CI spanning 0 is not a resolved difference.}}\n"
            f"\\label{{tab:multiseed_diffs_ph{hk}}}\n"
            "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "vs.\\ arm & $\\Delta$RMSE & $\\Delta$PR-AUC & $\\Delta$PR-AUC (onset) "
            "& $\\Delta$FAPD & $\\Delta$(D+E) \\\\\n\\midrule\n" + "\n".join(drows) +
            "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", default="runs/paper_results/benchmarks_seeds")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    summary = aggregate(args.seeds_dir, args.n_boot)
    path = os.path.join(args.seeds_dir, "multiseed_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    for name, body in render_tables(summary).items():
        with open(os.path.join(args.seeds_dir, name), "w") as fh:
            fh.write(body)
    print(f"wrote {path} and {len(render_tables(summary))} tables")


if __name__ == "__main__":
    main()
