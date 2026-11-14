"""Sensitivity of false-alarm burden to the device alarm policy.

    python scripts/fapd_latching_sensitivity.py --seeds-dir runs/paper_results/benchmarks_seeds

FAPD counts alarm EPISODES, so it depends on what ends an episode. The paper's
policy (`clinical.alarm_episodes`) ends an alarm the moment the score dips below
threshold, so one 5-minute dip splits one alarm into two soundings. This script
re-scores every arm under a grid of policies (`clinical.latched_alarm`):

* latching gap G minutes: an above-threshold run starting within G minutes of
  the previous alarm's end does not sound again; the alarm stays latched;
* minimum duration D minutes: a run sounds only once it has persisted D
  minutes, which suppresses blips and can cost lead time.

Held fixed: each seed's model, its test predictions, and its alarm threshold,
which was fitted on that seed's CALIBRATION split at a 5% FPR target and is read
from `results_summary.json`. No threshold is refitted per policy, so a policy is
compared on the same scores and the same operating threshold.

Per policy and arm: false alarms per day, true alarms per day, and onset
detection / lead time computed from the same "alarm active" series -- an event
(first reading < 70 mg/dL, `clinical.hypo_events`) is warned when the alarm is
active at some anchor in the P steps before onset whose reading is >= 70.

Uncertainty: mean +/- sd over seeds, and a 95% subject-cluster bootstrap CI
(12 subjects resampled, seed-mean) for each arm's difference from the fusion arm.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.clinical import hypo_events, latched_alarm, segment_ids  # noqa: E402

REFERENCE = "CausalCGM-MT (fusion)"
ARMS = ("Persistence", "Ridge-AR", "ARIMA(4,1,0)", "LightGBM", "Dilated LSTM (CGM)",
        "CausalCGM-MT (CGM)", "CausalCGM-MT (fusion)")
GAPS_MIN = (0, 5, 15, 30, 60)
DURATIONS_MIN = (5, 10, 15)
PERIOD = 5


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def policies():
    out = [(g, 5) for g in GAPS_MIN]
    out += [(0, d) for d in DURATIONS_MIN if d != 5]
    return out


def per_subject_stats(npz: str, k: int, horizon: int, thr: float) -> dict:
    """{policy: {subject: counts}} for one seed, arm and horizon."""
    z = np.load(npz, allow_pickle=True)
    pat = z["patient"].astype(str)
    t = z["t_time"].astype("datetime64[ns]")
    order = np.lexsort((t.astype(np.int64), pat))
    pat, t = pat[order], t[order]
    y = z["y_cls"][:, k].astype(int)[order]
    s = z["risk"][:, k].astype(float)[order]
    g = z["last_bg"].astype(float)[order]
    seg = segment_ids(t, pat, PERIOD, n=len(y))
    alarm = np.isfinite(s) & (s >= thr)
    P = horizon // PERIOD
    events = hypo_events(g, threshold=70.0, segment=seg, min_steps=1)

    out: dict = {}
    for gap, dur in policies():
        on, eps = latched_alarm(alarm, seg, merge_gap_steps=gap // PERIOD,
                                min_duration_steps=max(1, dur // PERIOD))
        stats = {sub: dict(false=0, true=0, rows=0, events=0, detected=0, lead_sum=0.0)
                 for sub in np.unique(pat)}
        for sub in stats:
            stats[sub]["rows"] = int((pat == sub).sum())
        for a, b in eps:
            key = "true" if y[a:b].any() else "false"
            stats[pat[a]][key] += 1
        for ev in events:
            sub = pat[ev.start]
            stats[sub]["events"] += 1
            lo = max(0, ev.start - P)
            for a in range(lo, ev.start):
                if on[a] and seg[a] == seg[ev.start] and g[a] >= 70.0:
                    stats[sub]["detected"] += 1
                    stats[sub]["lead_sum"] += (ev.start - a) * PERIOD
                    break
        out[(gap, dur)] = stats
    return out


def summarise(stats_by_seed: list[dict], subjects) -> dict:
    """Seed-mean metrics over the given (possibly resampled) subject list."""
    vals = {"fapd": [], "true_apd": [], "onset_det": [], "onset_lead": []}
    for st in stats_by_seed:
        rows = sum(st[x]["rows"] for x in subjects)
        days = rows * PERIOD / 1440.0
        ev = sum(st[x]["events"] for x in subjects)
        det = sum(st[x]["detected"] for x in subjects)
        vals["fapd"].append(sum(st[x]["false"] for x in subjects) / days)
        vals["true_apd"].append(sum(st[x]["true"] for x in subjects) / days)
        vals["onset_det"].append(det / ev if ev else np.nan)
        vals["onset_lead"].append(sum(st[x]["lead_sum"] for x in subjects) / det if det else np.nan)
    return {k: float(np.nanmean(v)) for k, v in vals.items()} | {
        f"{k}_sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", default="runs/paper_results/benchmarks_seeds")
    ap.add_argument("--out", default="runs/paper_results/fapd_latching")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.seeds_dir, "seed*")))
    runs = [json.load(open(os.path.join(d, "results_summary.json"))) for d in seed_dirs]
    if any(r.get("smoke") for r in runs):
        raise SystemExit("refusing smoke artefacts")
    rng = np.random.default_rng(20261006)
    report = {"seeds": [os.path.basename(d) for d in seed_dirs], "n_boot": args.n_boot,
              "threshold": "per seed, fitted on the calibration split at 5% FPR (not refitted)",
              "policies": [{"latch_gap_min": g, "min_duration_min": d} for g, d in policies()],
              "results": {}, "verification": {}}

    for k, h in enumerate((30, 60)):
        hk = str(h)
        stats = {}
        for arm in ARMS:
            stats[arm] = []
            for d, r in zip(seed_dirs, runs):
                a = next(x for x in r["arms"] if x["arm"] == arm)["per_horizon"][hk]["alarm"]
                st = per_subject_stats(os.path.join(d, "predictions", f"{_safe(arm)}.npz"),
                                       k, h, float(a["threshold_at_5pct_fpr"]))
                stats[arm].append(st)
                # The default policy must reproduce the harness exactly.
                base = summarise([st[(0, 5)]], sorted(st[(0, 5)]))
                ok = (abs(base["fapd"] - a["fapd_at_5pct_fpr"]) < 1e-9
                      and abs(base["onset_det"] - a["onset_detection_rate"]) < 1e-9
                      and abs(base["onset_lead"] - a["mean_onset_lead_min"]) < 1e-9)
                report["verification"][f"{arm}|{os.path.basename(d)}|{hk}"] = ok
                if not ok:
                    raise SystemExit(f"default policy does not reproduce the harness for "
                                     f"{arm} {d} PH{h}: {base} vs {a}")
        subjects = sorted(stats[ARMS[0]][0][(0, 5)])
        draws = [list(rng.choice(subjects, size=len(subjects), replace=True))
                 for _ in range(args.n_boot)]
        res = {}
        for pol in policies():
            key = f"G{pol[0]}_D{pol[1]}"
            res[key] = {}
            point = {arm: summarise([st[pol] for st in stats[arm]], subjects) for arm in ARMS}
            boot = {arm: [summarise([st[pol] for st in stats[arm]], dr) for dr in draws]
                    for arm in ARMS}
            for arm in ARMS:
                entry = dict(point[arm])
                if arm != REFERENCE:
                    for m in ("fapd", "onset_det"):
                        diffs = np.array([b0[m] - b1[m] for b0, b1 in
                                          zip(boot[REFERENCE], boot[arm])])
                        entry[f"diff_{m}_vs_fusion"] = point[REFERENCE][m] - point[arm][m]
                        entry[f"diff_{m}_vs_fusion_ci95"] = [
                            float(np.nanpercentile(diffs, 2.5)),
                            float(np.nanpercentile(diffs, 97.5))]
                res[key][arm] = entry
            ranks = sorted(ARMS, key=lambda a: point[a]["fapd"])
            res[key]["_fapd_rank"] = ranks
        report["results"][hk] = res
        print(f"PH={h}: verified default policy against harness for "
              f"{len(ARMS) * len(seed_dirs)} arm-seeds; scored {len(policies())} policies")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fapd_latching.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out, "table_fapd_latching.tex"), "w") as fh:
        fh.write(render(report))
    print(f"wrote {args.out}/fapd_latching.json and table_fapd_latching.tex")


def render(report: dict) -> str:
    short = {"Persistence": "Persist.", "Ridge-AR": "Ridge", "ARIMA(4,1,0)": "ARIMA",
             "LightGBM": "LGBM", "Dilated LSTM (CGM)": "LSTM",
             "CausalCGM-MT (CGM)": "TCN-CGM", "CausalCGM-MT (fusion)": "Fusion"}
    res = report["results"]["30"]
    head = " & ".join(["Policy (G, D)"] + [short[a] for a in ARMS]) + r" \\"
    rows = []
    for key in res:
        g, d = key[1:].split("_D")
        cells = [f"G={g}, D={d}"]
        for a in ARMS:
            e = res[key][a]
            cells.append(f"{e['fapd']:.2f} / {100 * e['onset_det']:.0f}\\%")
        rows.append(" & ".join(cells) + r" \\")
    return ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{False alarms per day / onset detection rate at $\\PH = 30$~min under "
            "alternative alarm policies, mean over " + str(len(report["seeds"])) +
            " seeds. G: latching gap (min) within which a new above-threshold run does not "
            "sound again; D: minimum duration (min) before an alarm sounds. G=0, D=5 is the "
            "policy used elsewhere in the paper. Thresholds are each seed's calibration-split "
            "5\\%-FPR thresholds, not refitted per policy.}\n"
            "\\label{tab:fapd_latching}\n\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
            "\\begin{tabular}{l" + "r" * len(ARMS) + "}\n\\toprule\n" + head +
            "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


if __name__ == "__main__":
    main()
