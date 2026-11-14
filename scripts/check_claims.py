"""Guard every comparative or qualitative claim in the paper against the result JSON.

    python scripts/check_claims.py            # exit 0 iff every claim holds
    python scripts/check_claims.py --root DIR # check a copy of the tree (tests)
    python scripts/check_claims.py --list     # print every assertion and its rule

Each assertion below states one rule ("95% CI of fusion-minus-LightGBM RMSE
includes 0", "PR-AUC is not monotone in kappa", ...) and evaluates it on the
result JSON under ``runs/`` (plus ``manuscript/metabonet_study_design.json``,
which is itself a JSON input to the closed-loop macros). Nothing here reads
``numbers.tex`` or manuscript text, and no expected result value is typed in:
every threshold is either the nominal level read from the JSON (1 - alpha), a
quantity computed from another JSON field (a seed-to-seed range, a fold SD), or
a structural constant of the rule itself (zero, "one tenth", a quarter) that is
written into the rule string so it can be reviewed.

The text-to-assertion map is ``ledger/claims.csv``. Claims that are false or
cannot be checked from the JSON are listed in ``ledger/claims_failed.md``; an
assertion is never weakened to make one pass. A mutation self-test lives in
``tests/test_check_claims.py``: every assertion has a JSON perturbation that
flips it, and only it.

Fail-closed: a missing file or key fails the assertion that needed it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "MS": "runs/paper_results/benchmarks_seeds/multiseed_summary.json",
    "BENCH": "runs/paper_results/benchmarks/results_summary.json",
    "CB": "runs/paper_results/conformal_bootstrap.json",
    "FL": "runs/paper_results/fapd_latching/fapd_latching.json",
    "AB": "runs/paper_results/ablations/ablations_summary.json",
    "PS": "runs/paper_results/personalization/personalization_sweep.json",
    "MB": "runs/metabonet_results/benchmarks/metabonet_summary.json",
    "SD": "manuscript/metabonet_study_design.json",
}

FUSION = "CausalCGM-MT (fusion)"
CGM_ONLY = "CausalCGM-MT (CGM)"
LSTM = "Dilated LSTM (CGM)"
LGBM = "LightGBM"
PERSIST = "Persistence"
# "The three neural arms" of Section 5 (the personalised LoRA arm is not one of
# them: its seed spread includes per-subject adaptation, not training noise).
NEURAL = (LSTM, CGM_ONLY, FUSION)
LEAK_SPLIT = "+ unembargoed random split"
LEAK_NAIVE = "+ full-series interpolation & centred features"
HORIZONS = ("30", "60")


class Data:
    """Lazy, cached JSON loader rooted at a tree (the repo or a temp copy)."""

    def __init__(self, root: str):
        self.root = root
        self._cache: dict[str, object] = {}

    def __getitem__(self, key: str):
        if key not in self._cache:
            with open(os.path.join(self.root, FILES[key])) as f:
                self._cache[key] = json.load(f)
        return self._cache[key]


@dataclass
class Assertion:
    aid: str
    rule: str
    fn: Callable[[Data], tuple[bool, str]]


ASSERTIONS: list[Assertion] = []


def claim(aid: str, rule: str):
    def deco(fn):
        if any(a.aid == aid for a in ASSERTIONS):
            raise RuntimeError(f"duplicate assertion id {aid}")
        ASSERTIONS.append(Assertion(aid, rule, fn))
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def ms_mean(d, arm, h, metric):
    return d["MS"]["arms"][arm][h][metric]["mean"]


def ms_diff(d, arm, h, metric):
    """Fusion minus `arm` (multiseed_summary.diffs_vs_reference), with its CI."""
    blk = d["MS"]["diffs_vs_reference"][arm][h][metric]
    return blk["diff"], blk["ci95"][0], blk["ci95"][1]


def ms_seed_range(d, arm, h, metric):
    ps = d["MS"]["arms"][arm][h][metric]["per_seed"]
    return max(ps) - min(ps)


def neural_seed_range(d, h, metric):
    """Largest per-seed range (max - min over seeds) of any of the three neural arms."""
    return max(ms_seed_range(d, a, h, metric) for a in NEURAL)


def spans_zero(lo, hi):
    return lo <= 0.0 <= hi


def leak(d, h, arm):
    return d["AB"]["studies"]["d_leakage"]["inflation"][h][arm]


def abl_rows(d, study):
    return d["AB"]["studies"][study]


def abl_arm(d, study, name):
    for r in abl_rows(d, study):
        if r["arm"] == name:
            return r
    raise KeyError(f"{study}: no arm {name!r}")


def ph(row, h):
    return row["per_horizon"][h]


def rmse(row, h):
    return ph(row, h)["regression"]["rmse"]


def prauc(row, h):
    return ph(row, h)["classification"]["pr_auc"]


def fapd(row, h):
    return ph(row, h)["alarm"]["fapd_at_5pct_fpr"]


def de(row, h):
    return ph(row, h)["clinical"]["zone_de_n"]


def kappa_rows(d):
    rows = abl_rows(d, "b_asymmetry")
    return sorted(rows, key=lambda r: float(r["spec"]["hypo_asymmetry"]))


def kappa_row(d, k):
    for r in kappa_rows(d):
        if float(r["spec"]["hypo_asymmetry"]) == k:
            return r
    raise KeyError(f"no kappa = {k}")


def strictly_increasing(xs):
    return all(b > a for a, b in zip(xs, xs[1:]))


def strictly_decreasing(xs):
    return all(b < a for a, b in zip(xs, xs[1:]))


def monotone(xs):
    return (all(b >= a for a, b in zip(xs, xs[1:]))
            or all(b <= a for a, b in zip(xs, xs[1:])))


# Ablation C: per cohort, the CGM-only control and the deltas of every other arm.
C_CONTROL = "CGM only"
C_TREAT = "CGM + treatment"
C_ALL = "CGM + all auxiliary (incl. treatment)"


def c_cohorts(d):
    return d["AB"]["studies"]["c_modality"]


def c_delta(d, cohort, arm, h):
    rows = {r["arm"]: r for r in c_cohorts(d)[cohort]}
    return rmse(rows[arm], h) - rmse(rows[C_CONTROL], h)


def c_wearable_arms(d, cohort):
    return [r["arm"] for r in c_cohorts(d)[cohort]
            if r["arm"] not in (C_CONTROL, C_TREAT, C_ALL)]


def nominal(d):
    return 1.0 - float(d["CB"]["alpha"])


def cb_seeds(d):
    return d["CB"]["seeds"]


def cb_severe(d, seed, method):
    return cb_seeds(d)[seed][method]["cov_severe_lt54"]


def bench_fusion_conformal(d):
    for a in d["BENCH"]["arms"]:
        if a["arm"] == FUSION:
            return a["per_horizon"]["30"]["conformal"]
    raise KeyError("no fusion arm in benchmarks/results_summary.json")


def mb_fold_pairs(d, section, metric):
    """Fusion minus CGM-only per fold (MetaboNet), for per_horizon[h][section][metric]."""
    h = str(d["MB"]["horizon_min"])
    by = {}
    for r in d["MB"]["per_fold"]:
        if r["arm"] in ("fusion", "cgm_only") and r.get("per_horizon"):
            by[(r["arm"], r["fold"])] = r["per_horizon"][h][section][metric]
    folds = sorted({f for (_a, f) in by})
    out = [by[("fusion", f)] - by[("cgm_only", f)] for f in folds]
    if not out:
        raise KeyError("no paired MetaboNet folds")
    return out


def mb_cgm_fold_values(d, section, metric):
    h = str(d["MB"]["horizon_min"])
    return [r["per_horizon"][h][section][metric] for r in d["MB"]["per_fold"]
            if r["arm"] == "cgm_only" and r.get("per_horizon")]


def mb_pers(d):
    for s in d["MB"]["summary"]:
        if s.get("personalization"):
            return s["personalization"]
    raise KeyError("no personalization block in metabonet_summary.json")


def mb_subjects_by_source(d):
    arm = d["MB"]["per_fold"][0]["arm"]
    out: dict[str, int] = {}
    for r in d["MB"]["per_fold"]:
        if r["arm"] != arm:
            continue
        for src, n in r["test_subjects_by_source"].items():
            out[src] = out.get(src, 0) + n
    return out


def ps_headline(d):
    """Prefix length the paper quotes: the one in adapt.headline_prefix_hours."""
    return str(float(d["PS"]["adapt"]["headline_prefix_hours"]))


def ps_sig(d, hours, mode, metric="rmse"):
    return d["PS"]["significance"][hours][mode][metric]


def ps_modes(d, hours):
    return [m for m in d["PS"]["significance"][hours] if m != "none"]


def ps_contrast(d):
    return {m: b for m, b in d["PS"]["prefix_mode_contrast"].items() if m != "none"}


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------------------- #
# Ablation D -- leakage (rules from ledger/leakage_definitions.md)
# r = rmse_relative_improvement_pct (positive = leaky RMSE lower),
# g = pr_auc_absolute_gain (positive = leaky PR-AUC higher), both vs strict.
# --------------------------------------------------------------------------- #
@claim("LEAK_NAIVE_30_DIRECTION",
       "PH30 naive (full ladder) arm: r > 0 (measured RMSE lower) and g > 0 (PR-AUC higher)")
def _(d):
    v = leak(d, "30", LEAK_NAIVE)
    r, g = v["rmse_relative_improvement_pct"], v["pr_auc_absolute_gain"]
    return r > 0 and g > 0, f"r={fmt(r)} g={fmt(g)}"


@claim("LEAK_NAIVE_60_DIRECTION",
       "PH60 naive arm: r > 0 and g > 0")
def _(d):
    v = leak(d, "60", LEAK_NAIVE)
    r, g = v["rmse_relative_improvement_pct"], v["pr_auc_absolute_gain"]
    return r > 0 and g > 0, f"r={fmt(r)} g={fmt(g)}"


@claim("LEAK_NAIVE_EXCEEDS_SPLIT_BOTH_H",
       "for H in {30,60}: r_naive > r_split and g_naive > g_split (full ladder vs split rung)")
def _(d):
    ev, ok = [], True
    for h in HORIZONS:
        n, s = leak(d, h, LEAK_NAIVE), leak(d, h, LEAK_SPLIT)
        ok &= (n["rmse_relative_improvement_pct"] > s["rmse_relative_improvement_pct"]
               and n["pr_auc_absolute_gain"] > s["pr_auc_absolute_gain"])
        ev.append(f"H{h}: r {fmt(n['rmse_relative_improvement_pct'])} vs "
                  f"{fmt(s['rmse_relative_improvement_pct'])}, g "
                  f"{fmt(n['pr_auc_absolute_gain'])} vs {fmt(s['pr_auc_absolute_gain'])}")
    return ok, "; ".join(ev)


@claim("LEAK_INCREMENT_EXCEEDS_SPLIT_BOTH_H",
       "for H in {30,60}: preprocessing increment beats the split effect on both metrics: "
       "(rmse_leaky[S]-rmse_leaky[N]) > (rmse_strict-rmse_leaky[S]) and "
       "(pr_leaky[N]-pr_leaky[S]) > (pr_leaky[S]-pr_strict)")
def _(d):
    ev, ok = [], True
    for h in HORIZONS:
        n, s = leak(d, h, LEAK_NAIVE), leak(d, h, LEAK_SPLIT)
        inc_r = s["rmse_leaky"] - n["rmse_leaky"]
        spl_r = s["rmse_strict"] - s["rmse_leaky"]
        inc_g = n["pr_auc_leaky"] - s["pr_auc_leaky"]
        spl_g = s["pr_auc_leaky"] - s["pr_auc_strict"]
        ok &= inc_r > spl_r and inc_g > spl_g
        ev.append(f"H{h}: dRMSE {fmt(inc_r)} vs {fmt(spl_r)}, dPR {fmt(inc_g)} vs {fmt(spl_g)}")
    return ok, "; ".join(ev)


@claim("LEAK_SPLIT_30_DIRECTION",
       "PH30 split rung: r < 0 (measured RMSE HIGHER) and g > 0 (PR-AUC rises)")
def _(d):
    v = leak(d, "30", LEAK_SPLIT)
    r, g = v["rmse_relative_improvement_pct"], v["pr_auc_absolute_gain"]
    return r < 0 and g > 0, f"r={fmt(r)} g={fmt(g)}"


@claim("LEAK_SPLIT_30_LITTLE",
       "'changes little' at PH30: |r_split| < 0.25*|r_naive| and |g_split| < 0.25*|g_naive|")
def _(d):
    s, n = leak(d, "30", LEAK_SPLIT), leak(d, "30", LEAK_NAIVE)
    rr = abs(s["rmse_relative_improvement_pct"]) / abs(n["rmse_relative_improvement_pct"])
    gr = abs(s["pr_auc_absolute_gain"]) / abs(n["pr_auc_absolute_gain"])
    return rr < 0.25 and gr < 0.25, f"split/naive ratio: RMSE {fmt(rr, 3)}, PR {fmt(gr, 3)}"


@claim("LEAK_SPLIT_60_NOT_NEGLIGIBLE",
       "PH60 split rung: r > 0 and g > 0, and 'not negligible': "
       "r_split >= 0.25*r_naive or g_split >= 0.25*g_naive")
def _(d):
    s, n = leak(d, "60", LEAK_SPLIT), leak(d, "60", LEAK_NAIVE)
    r, g = s["rmse_relative_improvement_pct"], s["pr_auc_absolute_gain"]
    big = (r >= 0.25 * n["rmse_relative_improvement_pct"]
           or g >= 0.25 * n["pr_auc_absolute_gain"])
    return r > 0 and g > 0 and big, f"r={fmt(r)} g={fmt(g)}"


# --------------------------------------------------------------------------- #
# Section 5 / Appendix B -- multi-seed benchmark
# --------------------------------------------------------------------------- #
@claim("MS_FUSION_BEATS_CGMONLY_RMSE",
       "for H in {30,60}: fusion-minus-CGM-only RMSE diff < 0 and its 95% CI upper bound < 0")
def _(d):
    ev, ok = [], True
    for h in HORIZONS:
        x, lo, hi = ms_diff(d, CGM_ONLY, h, "rmse")
        ok &= x < 0 and hi < 0
        ev.append(f"H{h}: {fmt(x)} [{fmt(lo)}, {fmt(hi)}]")
    return ok, "; ".join(ev)


@claim("MS_FUSION_VS_LGBM_UNRESOLVED",
       "for H in {30,60}: 95% CI of fusion-minus-LightGBM RMSE and of PR-AUC both include 0")
def _(d):
    ev, ok = [], True
    for h in HORIZONS:
        for m in ("rmse", "pr_auc"):
            x, lo, hi = ms_diff(d, LGBM, h, m)
            ok &= spans_zero(lo, hi)
            ev.append(f"H{h} {m}: {fmt(x)} [{fmt(lo)}, {fmt(hi)}]")
    return ok, "; ".join(ev)


@claim("MS_LGBM_ONSET_PRAUC",
       "PH30 onset-only PR-AUC: fusion-minus-LightGBM diff < 0 (LightGBM higher) and CI includes 0")
def _(d):
    x, lo, hi = ms_diff(d, LGBM, "30", "pr_auc_onset")
    return x < 0 and spans_zero(lo, hi), f"{fmt(x)} [{fmt(lo)}, {fmt(hi)}]"


@claim("MS_NEURAL_PRAUC_UNSEPARATED",
       "for H in {30,60}: 95% CIs of fusion-minus-CGM-only and fusion-minus-LSTM PR-AUC include 0")
def _(d):
    ev, ok = [], True
    for h in HORIZONS:
        for a in (CGM_ONLY, LSTM):
            x, lo, hi = ms_diff(d, a, h, "pr_auc")
            ok &= spans_zero(lo, hi)
            ev.append(f"H{h} {a}: {fmt(x)} [{fmt(lo)}, {fmt(hi)}]")
    return ok, "; ".join(ev)


@claim("MS_DE_FUSION_BELOW_LGBM",
       "PH30 Clarke D+E: fusion-minus-LightGBM diff < 0 and 95% CI upper bound < 0")
def _(d):
    x, lo, hi = ms_diff(d, LGBM, "30", "de")
    return x < 0 and hi < 0, f"{fmt(x, 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}]"


@claim("MS_DE_SAME_LOSS_UNRESOLVED",
       "PH30 Clarke D+E: 95% CIs of fusion-minus-CGM-only and fusion-minus-LSTM include 0")
def _(d):
    ev, ok = [], True
    for a in (CGM_ONLY, LSTM):
        x, lo, hi = ms_diff(d, a, "30", "de")
        ok &= spans_zero(lo, hi)
        ev.append(f"{a}: {fmt(x, 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}]")
    return ok, "; ".join(ev)


@claim("DE_GAP_IS_THE_LOSS",
       "kappa=1 fusion D+E (Ablation B, PH30) is closer to LightGBM's multi-seed mean D+E "
       "than to fusion's multi-seed mean D+E")
def _(d):
    k1 = de(kappa_row(d, 1.0), "30")
    lg, fu = ms_mean(d, LGBM, "30", "de"), ms_mean(d, FUSION, "30", "de")
    return abs(k1 - lg) < abs(k1 - fu), f"kappa1={k1} LightGBM={fmt(lg, 1)} fusion={fmt(fu, 1)}"


@claim("PERSIST_STRONG_CLASSIFIER",
       "single-seed benchmark PH30: persistence PR-AUC >= the lowest PR-AUC among the learned, "
       "non-personalised arms")
def _(d):
    arms = {a["arm"]: a["per_horizon"]["30"]["classification"]["pr_auc"]
            for a in d["BENCH"]["arms"]}
    p = arms.pop(PERSIST)
    learned = {k: v for k, v in arms.items() if "LoRA" not in k}
    lo = min(learned.values())
    return p >= lo, f"persistence={fmt(p)} min learned={fmt(lo)}"


@claim("PERSIST_ONSET_FLOOR",
       "multi-seed PH30: persistence has the largest drop from PR-AUC to onset-only PR-AUC of any arm")
def _(d):
    drops = {a: ms_mean(d, a, "30", "pr_auc") - ms_mean(d, a, "30", "pr_auc_onset")
             for a in d["MS"]["arms"]}
    top = max(drops, key=drops.get)
    return top == PERSIST, f"largest drop: {top} ({fmt(drops[top])})"


@claim("METABO_REPLICATES_SMALLER",
       "MetaboNet fold-mean fusion-minus-CGM-only RMSE < 0 and its magnitude < |OhioT1DM PH30 "
       "multi-seed fusion-minus-CGM-only RMSE diff|")
def _(d):
    mb = statistics.mean(mb_fold_pairs(d, "regression", "rmse"))
    oh = ms_diff(d, CGM_ONLY, "30", "rmse")[0]
    return mb < 0 and abs(mb) < abs(oh), f"MetaboNet {fmt(mb)} vs Ohio {fmt(oh)}"


# --------------------------------------------------------------------------- #
# Appendix C -- Ablation C (modality)
# --------------------------------------------------------------------------- #
@claim("ABLC_TREATMENT_LOWERS_RMSE",
       "both cohorts, H in {30,60}: RMSE(CGM + treatment) < RMSE(CGM only)")
def _(d):
    ev, ok = [], True
    for c in c_cohorts(d):
        for h in HORIZONS:
            x = c_delta(d, c, C_TREAT, h)
            ok &= x < 0
            ev.append(f"{c} H{h}: {fmt(x, 3)}")
    return ok, "; ".join(ev)


@claim("ABLC_ALL_AUX_NO_BETTER",
       "both cohorts, H in {30,60}: RMSE(CGM + all auxiliary) >= RMSE(CGM + treatment)")
def _(d):
    ev, ok = [], True
    for c in c_cohorts(d):
        for h in HORIZONS:
            x = c_delta(d, c, C_ALL, h) - c_delta(d, c, C_TREAT, h)
            ok &= x >= 0
            ev.append(f"{c} H{h}: {fmt(x, 3)}")
    return ok, "; ".join(ev)


@claim("ABLC_TREATMENT_OVER_WEARABLES",
       "both cohorts, H in {30,60}: treatment-only RMSE delta is below (better than) every "
       "wearable-only arm's delta")
def _(d):
    ev, ok = [], True
    for c in c_cohorts(d):
        for h in HORIZONS:
            t = c_delta(d, c, C_TREAT, h)
            w = min(c_delta(d, c, a, h) for a in c_wearable_arms(d, c))
            ok &= t < w
            ev.append(f"{c} H{h}: treat {fmt(t, 3)} vs best wearable {fmt(w, 3)}")
    return ok, "; ".join(ev)


@claim("ABLC_WEARABLE_30_SMALL",
       "PH30: every wearable-only |RMSE delta| < the smaller |treatment-only delta| of the two "
       "cohorts ('small' / 'null or worse')")
def _(d):
    w = max(abs(c_delta(d, c, a, "30")) for c in c_cohorts(d) for a in c_wearable_arms(d, c))
    t = min(abs(c_delta(d, c, C_TREAT, "30")) for c in c_cohorts(d))
    return w < t, f"max wearable |d|={fmt(w, 3)} min treatment |d|={fmt(t, 3)}"


@claim("ABLC_WEARABLE_30_SIGN_INCONSISTENT",
       "PH30: some wearable modality present in both cohorts has RMSE deltas of opposite sign")
def _(d):
    cs = list(c_cohorts(d))
    shared = set(c_wearable_arms(d, cs[0]))
    for c in cs[1:]:
        shared &= set(c_wearable_arms(d, c))
    ev, ok = [], False
    for a in sorted(shared):
        signs = {math.copysign(1, c_delta(d, c, a, "30")) for c in cs}
        ok |= len(signs) > 1
        ev.append(f"{a}: " + ", ".join(f"{c} {fmt(c_delta(d, c, a, '30'), 3)}" for c in cs))
    return ok, "; ".join(ev) or "no shared wearable modality"


@claim("ABLC_HR_GSR_WORSE_2018",
       "2018 cohort, H in {30,60}: CGM + HR and CGM + GSR both have RMSE above the CGM-only control")
def _(d):
    ev, ok = [], True
    for a in ("CGM + HR", "CGM + GSR"):
        for h in HORIZONS:
            x = c_delta(d, "2018", a, h)
            ok &= x > 0
            ev.append(f"{a} H{h}: {fmt(x, 3)}")
    return ok, "; ".join(ev)


@claim("ABLC_STEPS_60_EXCEPTION",
       "PH60: CGM + steps (2018) lowers RMSE vs control, and is the only wearable-only arm that does")
def _(d):
    lowering = [(c, a) for c in c_cohorts(d) for a in c_wearable_arms(d, c)
                if c_delta(d, c, a, "60") < 0]
    return lowering == [("2018", "CGM + steps")], f"wearable arms lowering PH60 RMSE: {lowering}"


# --------------------------------------------------------------------------- #
# Appendix C -- Ablation A (multi-task)
# --------------------------------------------------------------------------- #
A_REG, A_CLS = "Regression only", "Classification only"
A_FIX, A_KEN = "Joint, fixed $\\lambda=1$", "Joint, Kendall uncertainty"


@claim("ABLA_JOINT_COSTS_SPECIALISTS",
       "PH30: classification-only PR-AUC > Kendall joint PR-AUC and regression-only RMSE < Kendall RMSE")
def _(d):
    c, r, k = (abl_arm(d, "a_multitask", n) for n in (A_CLS, A_REG, A_KEN))
    ok = prauc(c, "30") > prauc(k, "30") and rmse(r, "30") < rmse(k, "30")
    return ok, (f"PR cls {fmt(prauc(c, '30'))} vs joint {fmt(prauc(k, '30'))}; "
                f"RMSE reg {fmt(rmse(r, '30'), 3)} vs joint {fmt(rmse(k, '30'), 3)}")


@claim("ABLA_COST_FEW_THOUSANDTHS",
       "PH30 coupling cost (classification-only minus Kendall PR-AUC) is 'a few thousandths': "
       "0.001 <= cost < 0.01")
def _(d):
    x = prauc(abl_arm(d, "a_multitask", A_CLS), "30") - prauc(abl_arm(d, "a_multitask", A_KEN), "30")
    return 0.001 <= x < 0.01, f"cost={fmt(x)}"


@claim("ABLA_KENDALL_VS_FIXED_DIRECTIONS",
       "H in {30,60}: Kendall PR-AUC > fixed-lambda PR-AUC and fixed-lambda RMSE < Kendall RMSE")
def _(d):
    k, f = abl_arm(d, "a_multitask", A_KEN), abl_arm(d, "a_multitask", A_FIX)
    ev, ok = [], True
    for h in HORIZONS:
        ok &= prauc(k, h) > prauc(f, h) and rmse(f, h) < rmse(k, h)
        ev.append(f"H{h}: PR {fmt(prauc(k, h))}/{fmt(prauc(f, h))} RMSE "
                  f"{fmt(rmse(k, h), 3)}/{fmt(rmse(f, h), 3)}")
    return ok, "; ".join(ev)


COMPARABLE_FACTOR = 1.25  # "comparable to the seed-to-seed range": at most 1.25x that range


@claim("ABLA_KENDALL_VS_FIXED_COMPARABLE_TO_SEED_RANGE",
       "H in {30,60}: |Kendall - fixed| PR-AUC and RMSE are each <= 1.25 x the largest per-seed range "
       "(max - min over seeds) of that metric among the three neural arms of the multi-seed benchmark")
def _(d):
    k, f = abl_arm(d, "a_multitask", A_KEN), abl_arm(d, "a_multitask", A_FIX)
    ev, ok = [], True
    for h in HORIZONS:
        dp, dr = abs(prauc(k, h) - prauc(f, h)), abs(rmse(k, h) - rmse(f, h))
        sp, sr = neural_seed_range(d, h, "pr_auc"), neural_seed_range(d, h, "rmse")
        ok &= dp <= COMPARABLE_FACTOR * sp and dr <= COMPARABLE_FACTOR * sr
        ev.append(f"H{h}: |dPR| {fmt(dp)} vs 1.25 x seed range {fmt(COMPARABLE_FACTOR * sp)}; "
                  f"|dRMSE| {fmt(dr, 3)} vs {fmt(COMPARABLE_FACTOR * sr, 3)}")
    return ok, "; ".join(ev)


# --------------------------------------------------------------------------- #
# Section 5 / Appendix C -- Ablation B (kappa)
# --------------------------------------------------------------------------- #
def _kappa_series(d, h, fn):
    return [fn(r, h) for r in kappa_rows(d)]


def _r70(r, h):
    return ph(r, h)["clinical"]["rmse_below_70"]


def _bias(r, h):
    return ph(r, h)["clinical"]["hypo_band_bias"]


@claim("ABLB_KAPPA3_TRADE",
       "PH30, kappa=3 vs kappa=1: overall RMSE higher, Clarke D+E lower, PR-AUC lower")
def _(d):
    a, b = kappa_row(d, 1.0), kappa_row(d, 3.0)
    ok = rmse(b, "30") > rmse(a, "30") and de(b, "30") < de(a, "30") and prauc(b, "30") < prauc(a, "30")
    return ok, (f"RMSE {fmt(rmse(a, '30'), 3)}->{fmt(rmse(b, '30'), 3)} D+E {de(a, '30')}->{de(b, '30')} "
                f"PR {fmt(prauc(a, '30'))}->{fmt(prauc(b, '30'))}")


@claim("ABLB_MONOTONE_30",
       "PH30, kappa ascending: D+E, RMSE<70 and hypo bias strictly decreasing; overall RMSE strictly increasing")
def _(d):
    s = {n: _kappa_series(d, "30", fn) for n, fn in
         (("de", de), ("r70", _r70), ("bias", _bias), ("rmse", rmse))}
    ok = (strictly_decreasing(s["de"]) and strictly_decreasing(s["r70"])
          and strictly_decreasing(s["bias"]) and strictly_increasing(s["rmse"]))
    return ok, "; ".join(f"{k}={[round(v, 3) for v in xs]}" for k, xs in s.items())


@claim("ABLB_PRAUC_NOT_MONOTONE_30",
       "PH30: PR-AUC is not monotone in kappa")
def _(d):
    xs = _kappa_series(d, "30", prauc)
    return not monotone(xs), f"PR-AUC={[round(v, 4) for v in xs]}"


@claim("ABLB_PRAUC_RANGE_COMPARABLE_TO_SEEDS",
       "PH30: range of PR-AUC across kappa arms is 'comparable' to seed variation: "
       "0.5 <= range / (largest per-seed PR-AUC range of a neural arm) <= 2")
def _(d):
    xs = _kappa_series(d, "30", prauc)
    ratio = (max(xs) - min(xs)) / neural_seed_range(d, "30", "pr_auc")
    return 0.5 <= ratio <= 2.0, f"ratio={fmt(ratio, 3)}"


@claim("ABLB_R70_NOT_MONOTONE_60",
       "PH60: RMSE below 70 mg/dL is not monotone in kappa")
def _(d):
    xs = _kappa_series(d, "60", _r70)
    return not monotone(xs), f"RMSE<70={[round(v, 3) for v in xs]}"


@claim("ABLB_SAME_SHAPE_60",
       "PH60, largest kappa vs kappa=1: RMSE<70 lower, hypo bias lower, overall RMSE higher, PR-AUC lower")
def _(d):
    rows = kappa_rows(d)
    a, b = rows[0], rows[-1]
    ok = (_r70(b, "60") < _r70(a, "60") and _bias(b, "60") < _bias(a, "60")
          and rmse(b, "60") > rmse(a, "60") and prauc(b, "60") < prauc(a, "60"))
    return ok, (f"r70 {fmt(_r70(a, '60'), 3)}->{fmt(_r70(b, '60'), 3)} bias {fmt(_bias(a, '60'), 3)}->"
                f"{fmt(_bias(b, '60'), 3)} RMSE {fmt(rmse(a, '60'), 3)}->{fmt(rmse(b, '60'), 3)} "
                f"PR {fmt(prauc(a, '60'))}->{fmt(prauc(b, '60'))}")


# --------------------------------------------------------------------------- #
# Appendix C -- Ablation E (soft-label band)
# --------------------------------------------------------------------------- #
def _band(r):
    sl = r.get("soft_label") or {}
    return sl.get("band_mgdl") if sl.get("enabled") else 0.0


@claim("ABLE_NO_DOMINANT_CONFIG",
       "H in {30,60}: no arm is best-or-tied on all of RMSE (min), PR-AUC (max), FAPD (min), D+E (min)")
def _(d):
    rows = abl_rows(d, "e_soft_label")
    ev, ok = [], True
    for h in HORIZONS:
        for r in rows:
            dom = (rmse(r, h) <= min(rmse(x, h) for x in rows)
                   and prauc(r, h) >= max(prauc(x, h) for x in rows)
                   and fapd(r, h) <= min(fapd(x, h) for x in rows)
                   and de(r, h) <= min(de(x, h) for x in rows))
            if dom:
                ok = False
                ev.append(f"H{h}: {r['arm']} dominates")
    return ok, "; ".join(ev) or "no dominant arm"


@claim("ABLE_WIDTH_DOES_NOT_ORDER",
       "H in {30,60}: PR-AUC over the pinned arms ordered by band width (hard = 0) is not monotone")
def _(d):
    rows = sorted([r for r in abl_rows(d, "e_soft_label")
                   if (r.get("soft_label") or {}).get("band_source") in ("disabled", "fixed")],
                  key=_band)
    ev, ok = [], True
    for h in HORIZONS:
        xs = [prauc(r, h) for r in rows]
        ok &= not monotone(xs)
        ev.append(f"H{h}: {[round(v, 4) for v in xs]}")
    return ok, "; ".join(ev)


@claim("ABLE_ESTIMATED_LIKE_NEIGHBOURS",
       "H in {30,60}: |PR-AUC(estimated band) - PR-AUC(pinned band on each side of it)| <= the largest "
       "per-seed PR-AUC range of a neural arm")
def _(d):
    rows = abl_rows(d, "e_soft_label")
    est = [r for r in rows if (r.get("soft_label") or {}).get("band_source") == "mard"][0]
    pinned = sorted([r for r in rows if (r.get("soft_label") or {}).get("band_source") == "fixed"],
                    key=_band)
    b = _band(est)
    below = max((r for r in pinned if _band(r) <= b), key=_band)
    above = min((r for r in pinned if _band(r) >= b), key=_band)
    ev, ok = [], True
    for h in HORIZONS:
        lim = neural_seed_range(d, h, "pr_auc")
        gaps = [abs(prauc(est, h) - prauc(n, h)) for n in (below, above)]
        ok &= max(gaps) <= lim
        ev.append(f"H{h}: gaps {[round(g, 4) for g in gaps]} vs {fmt(lim)}")
    return ok, f"band {fmt(b, 2)} between {_band(below)} and {_band(above)}; " + "; ".join(ev)


@claim("ABLE_HARD_HIGHEST_FAPD",
       "H in {30,60}: hard labels have strictly the highest FAPD (every soft band lower)")
def _(d):
    rows = abl_rows(d, "e_soft_label")
    hard = [r for r in rows if _band(r) == 0.0][0]
    ev, ok = [], True
    for h in HORIZONS:
        others = max(fapd(r, h) for r in rows if r is not hard)
        ok &= fapd(hard, h) > others
        ev.append(f"H{h}: hard {fmt(fapd(hard, h), 3)} vs max soft {fmt(others, 3)}")
    return ok, "; ".join(ev)


# --------------------------------------------------------------------------- #
# Appendix B -- Ablation F (nadir target) and alarm-policy sensitivity
# --------------------------------------------------------------------------- #
F_LEVEL, F_NADIR = "Ridge-AR, level target", "Ridge-AR, nadir target"
F_LGBM, F_MT = "LightGBM (native nadir classifier)", "CausalCGM-MT (native nadir head)"


def _roc(r, h):
    return ph(r, h)["classification"]["roc_auc"]


@claim("ABLF_NADIR_HELPS_RIDGE",
       "nadir-targeted Ridge-AR: PR-AUC higher at PH30 and PH60, FAPD lower at PH30, than level target")
def _(d):
    a, b = abl_arm(d, "f_nadir_target", F_LEVEL), abl_arm(d, "f_nadir_target", F_NADIR)
    ok = all(prauc(b, h) > prauc(a, h) for h in HORIZONS) and fapd(b, "30") < fapd(a, "30")
    return ok, (f"PR30 {fmt(prauc(a, '30'))}->{fmt(prauc(b, '30'))} PR60 {fmt(prauc(a, '60'))}->"
                f"{fmt(prauc(b, '60'))} FAPD30 {fmt(fapd(a, '30'), 3)}->{fmt(fapd(b, '30'), 3)}")


@claim("ABLF_ROC_BARELY_MOVES",
       "PH30: |ROC-AUC change| from level to nadir target < 0.1 * the PR-AUC gain")
def _(d):
    a, b = abl_arm(d, "f_nadir_target", F_LEVEL), abl_arm(d, "f_nadir_target", F_NADIR)
    dr, dp = abs(_roc(b, "30") - _roc(a, "30")), prauc(b, "30") - prauc(a, "30")
    return dr < 0.1 * dp, f"|dROC|={fmt(dr, 5)} dPR={fmt(dp)}"


@claim("ABLF_NADIR_RIDGE_OVERTAKES_LGBM",
       "PH30: nadir-targeted Ridge-AR PR-AUC > LightGBM native classifier PR-AUC")
def _(d):
    b, g = abl_arm(d, "f_nadir_target", F_NADIR), abl_arm(d, "f_nadir_target", F_LGBM)
    return prauc(b, "30") > prauc(g, "30"), f"{fmt(prauc(b, '30'))} vs {fmt(prauc(g, '30'))}"


@claim("ABLF_MT_LEAD_SMALL",
       "H in {30,60}: 0 < PR-AUC(CausalCGM-MT) - PR-AUC(nadir Ridge-AR) <= 2 * multi-seed SD of "
       "fusion PR-AUC at that horizon")
def _(d):
    m, b = abl_arm(d, "f_nadir_target", F_MT), abl_arm(d, "f_nadir_target", F_NADIR)
    ev, ok = [], True
    for h in HORIZONS:
        gap = prauc(m, h) - prauc(b, h)
        sd = d["MS"]["arms"][FUSION][h]["pr_auc"]["sd"]
        ok &= 0 < gap <= 2 * sd
        ev.append(f"H{h}: gap {fmt(gap)} vs 2*sd {fmt(2 * sd)}")
    return ok, "; ".join(ev)


@claim("LATCH_PERSIST_LOWEST_FAPD_SHORTEST_LEAD",
       "every policy, H in {30,60}: persistence has the lowest FAPD and the shortest onset lead of all arms")
def _(d):
    bad = []
    for h, pols in d["FL"]["results"].items():
        for pol, cell in pols.items():
            arms = {a: v for a, v in cell.items() if not a.startswith("_")}  # skip _fapd_rank
            fa = min(arms, key=lambda a: arms[a]["fapd"])
            ld = min(arms, key=lambda a: arms[a]["onset_lead"])
            if fa != PERSIST or ld != PERSIST:
                bad.append(f"H{h} {pol}: min FAPD {fa}, min lead {ld}")
    return not bad, "; ".join(bad) or "persistence lowest in every cell"


@claim("LATCH_FUSION_BELOW_LGBM",
       "PH30, every latching policy (gap > 0): fusion-minus-LightGBM FAPD < 0 with 95% CI upper bound < 0")
def _(d):
    ev, ok = [], True
    for pol, arms in d["FL"]["results"]["30"].items():
        if not pol.startswith("G") or pol.startswith("G0_"):
            continue
        blk = arms[LGBM]
        x, (lo, hi) = blk["diff_fapd_vs_fusion"], blk["diff_fapd_vs_fusion_ci95"]
        ok &= x < 0 and hi < 0
        ev.append(f"{pol}: {fmt(x, 3)} [{fmt(lo, 3)}, {fmt(hi, 3)}]")
    return ok and bool(ev), "; ".join(ev)


@claim("LATCH_D15_FUSION_DETECTS_FEWER",
       "PH30, 15-min minimum duration: fusion-minus-LightGBM onset detection < 0 with 95% CI upper bound < 0")
def _(d):
    blk = d["FL"]["results"]["30"]["G0_D15"][LGBM]
    x, (lo, hi) = blk["diff_onset_det_vs_fusion"], blk["diff_onset_det_vs_fusion_ci95"]
    return x < 0 and hi < 0, f"{fmt(x)} [{fmt(lo)}, {fmt(hi)}]"


@claim("LATCH_FUSION_FALLS_UNDER_60MIN",
       "PH30: fusion FAPD under 60-minute latching < fusion FAPD under the default policy")
def _(d):
    r = d["FL"]["results"]["30"]
    a, b = r["G0_D5"][FUSION]["fapd"], r["G60_D5"][FUSION]["fapd"]
    return b < a, f"{fmt(a, 3)} -> {fmt(b, 3)}"


# --------------------------------------------------------------------------- #
# Section 6 / Appendix D / Limitations -- conformal
# --------------------------------------------------------------------------- #
@claim("CONF_SPLIT_MARGINAL_VALID",
       "single-seed fusion PH30: split-conformal marginal coverage >= 1 - alpha")
def _(d):
    c = bench_fusion_conformal(d)
    x, a = c["split"]["marginal_coverage"], 1.0 - float(c["alpha"])
    return x >= a, f"{fmt(x)} vs {fmt(a, 2)}"


@claim("CONF_SPLIT_HYPO_OVERCOVERS",
       "single-seed fusion PH30: split hypoglycemic-band coverage > 1 - alpha and > split marginal coverage")
def _(d):
    c = bench_fusion_conformal(d)
    s, a = c["split"], 1.0 - float(c["alpha"])
    x = s["hypo_stratum_coverage"]
    return x > a and x > s["marginal_coverage"], f"hypo {fmt(x)} marginal {fmt(s['marginal_coverage'])}"


@claim("CONF_MONDRIAN_EQUALISES_AND_NARROWS",
       "single-seed fusion PH30: spread (max - min) of zone coverage is smaller for Mondrian than split, "
       "and Mondrian hypoglycemic width < split hypoglycemic width")
def _(d):
    c = bench_fusion_conformal(d)

    def spread(m):
        v = [z["coverage"] for z in c[m]["by_stratum"].values()]
        return max(v) - min(v)
    sm, ss = spread("mondrian"), spread("split")
    wm, ws = c["mondrian"]["hypo_stratum_width"], c["split"]["hypo_stratum_width"]
    return sm < ss and wm < ws, f"spread {fmt(sm)} vs {fmt(ss)}; hypo width {fmt(wm, 1)} vs {fmt(ws, 1)}"


@claim("CONF_MONDRIAN_NARROW_HYPO_COSTS_SEVERE",
       "single-seed fusion PH30: Mondrian hypo width < normalised hypo width, and Mondrian <54 coverage "
       "< 1 - alpha")
def _(d):
    c = bench_fusion_conformal(d)
    a = 1.0 - float(c["alpha"])
    wm, wn = c["mondrian"]["hypo_stratum_width"], c["normalized"]["hypo_stratum_width"]
    sv = c["mondrian"]["by_clinical_range"]["severe hypo (<54)"]["coverage"]
    return wm < wn and sv < a, f"width {fmt(wm, 1)} vs {fmt(wn, 1)}; Mondrian <54 {fmt(sv)}"


@claim("CONF_MONDRIAN_UNDERCOVERS_EVERY_SEED",
       "every seed (conformal_bootstrap): Mondrian <54 point coverage < 1 - alpha and "
       "p_below_nominal > 0.5")
def _(d):
    a = nominal(d)
    ev, ok = [], True
    for s in cb_seeds(d):
        v = cb_severe(d, s, "mondrian")
        ok &= v["point"] < a and v["p_below_nominal"] > 0.5
        ev.append(f"{s}: {fmt(v['point'], 3)} p_below={v['p_below_nominal']}")
    return ok, "; ".join(ev)


@claim("CONF_NORMALISED_NOT_CONSISTENT",
       "number of seeds on which normalised <54 point coverage > split's is < number of seeds")
def _(d):
    seeds = cb_seeds(d)
    n = sum(cb_severe(d, s, "normalized")["point"] > cb_severe(d, s, "split")["point"] for s in seeds)
    return n < len(seeds), f"normalised beats split on {n} of {len(seeds)}"


@claim("CONF_SEED_VARIATION_EXCEEDS_DRAW_VARIATION",
       "for split and normalised: range of <54 point coverage across seeds > the largest within-seed "
       "bootstrap 95% width (p97.5 - p2.5)")
def _(d):
    ev, ok = [], True
    for m in ("split", "normalized"):
        pts = [cb_severe(d, s, m)["point"] for s in cb_seeds(d)]
        within = max(cb_severe(d, s, m)["boot_p97_5"] - cb_severe(d, s, m)["boot_p2_5"]
                     for s in cb_seeds(d))
        ok &= (max(pts) - min(pts)) > within
        ev.append(f"{m}: between {fmt(max(pts) - min(pts))} vs within {fmt(within)}")
    return ok, "; ".join(ev)


@claim("CONF_NO_CONSTRUCTION_CERTIFIES_SEVERE",
       "for each of split, normalised, Mondrian: some seed has <54 point coverage < 1 - alpha")
def _(d):
    a = nominal(d)
    ev, ok = [], True
    for m in ("split", "normalized", "mondrian"):
        lo = min(cb_severe(d, s, m)["point"] for s in cb_seeds(d))
        ok &= lo < a
        ev.append(f"{m} min {fmt(lo, 3)}")
    return ok, "; ".join(ev)


@claim("CONF_SPLIT_NORMALISED_OVERLAP",
       "split and normalised <54 point-coverage ranges across seeds overlap ('do not separate')")
def _(d):
    sp = [cb_severe(d, s, "split")["point"] for s in cb_seeds(d)]
    no = [cb_severe(d, s, "normalized")["point"] for s in cb_seeds(d)]
    ok = max(min(sp), min(no)) <= min(max(sp), max(no))
    return ok, f"split [{fmt(min(sp), 3)}, {fmt(max(sp), 3)}] normalised [{fmt(min(no), 3)}, {fmt(max(no), 3)}]"


# --------------------------------------------------------------------------- #
# Appendix E -- MetaboNet
# --------------------------------------------------------------------------- #
@claim("METABO_FUSION_RMSE_ALL_FOLDS",
       "fusion-minus-CGM-only RMSE < 0 in every fold")
def _(d):
    xs = mb_fold_pairs(d, "regression", "rmse")
    return all(x < 0 for x in xs), f"{[round(x, 3) for x in xs]}"


@claim("METABO_FUSION_LOWERS_FAPD",
       "fold-mean fusion-minus-CGM-only FAPD < 0")
def _(d):
    xs = mb_fold_pairs(d, "alarm", "fapd_at_5pct_fpr")
    return statistics.mean(xs) < 0, f"mean {fmt(statistics.mean(xs), 3)} folds {[round(x, 3) for x in xs]}"


@claim("METABO_PRAUC_GAIN_NEGLIGIBLE",
       "fold-mean fusion-minus-CGM-only PR-AUC > 0 and smaller than the between-fold SD of CGM-only PR-AUC")
def _(d):
    m = statistics.mean(mb_fold_pairs(d, "classification", "pr_auc"))
    sd = statistics.stdev(mb_cgm_fold_values(d, "classification", "pr_auc"))
    return 0 < m < sd, f"mean {fmt(m)} vs fold SD {fmt(sd)}"


@claim("METABO_DE_UNRESOLVED",
       "fusion-minus-CGM-only Clarke D+E: |fold mean| < fold SD and folds of both signs")
def _(d):
    xs = mb_fold_pairs(d, "clinical", "zone_de_n")
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    ok = abs(m) < sd and any(x < 0 for x in xs) and any(x > 0 for x in xs)
    return ok, f"mean {fmt(m, 1)} sd {fmt(sd, 1)} folds {xs}"


@claim("METABO_PERS_DEGRADES",
       "MetaboNet LoRA: mean, median and lower quartile of per-subject RMSE change > 0, and fewer than "
       "half of subjects improve")
def _(d):
    p = mb_pers(d)
    xs = sorted(s["rmse_after"] - s["rmse_before"] for s in p["per_subject"]
                if s.get("rmse_after") is not None and s.get("rmse_before") is not None)
    q1, med, _q3 = statistics.quantiles(xs, n=4)
    ok = p["mean_delta_rmse"] > 0 and med > 0 and q1 > 0 and p["n_improved_rmse"] < p["n_subjects"] / 2
    return ok, f"mean {fmt(p['mean_delta_rmse'], 3)} median {fmt(med, 3)} q1 {fmt(q1, 3)} improved {p['n_improved_rmse']}/{p['n_subjects']}"


def _mb_pr_deltas(p):
    return [s["pr_auc_after"] - s["pr_auc_before"] for s in p["per_subject"]
            if s.get("pr_auc_after") is not None and s.get("pr_auc_before") is not None]


@claim("METABO_PERS_PRAUC_SMALLER_BASE",
       "MetaboNet LoRA, 'degrades too, on a smaller base': mean PR-AUC change < 0 and PR-AUC defined "
       "for fewer subjects than RMSE")
def _(d):
    p = mb_pers(d)
    pr = _mb_pr_deltas(p)
    return len(pr) < p["n_subjects"] and p["mean_delta_pr_auc"] < 0, \
        f"{len(pr)} of {p['n_subjects']} subjects; mean {fmt(p['mean_delta_pr_auc'])}"


# "The sign matches OhioT1DM's" is the conjunction of METABO_PERS_DEGRADES
# (MetaboNet mean > 0) and PERS_EVERY_MODE_WORSE (OhioT1DM LoRA r=4 > 0); a
# separate assertion could never fail on its own, so the claim maps to those two.


@claim("METABO_MOST_AID",
       "share of MetaboNet subjects from sources classified closed_loop == 'yes' > 50%")
def _(d):
    cls = {s["source"]: s["closed_loop"] for s in d["SD"]["studies"]}
    by = mb_subjects_by_source(d)
    n = int(d["MB"]["provenance"]["n_subjects"])
    yes = sum(v for k, v in by.items() if cls[k] == "yes")
    return sum(by.values()) == n and yes / n > 0.5, f"{yes}/{n} = {fmt(100 * yes / n, 1)}%"


@claim("METABO_LARGEST_SOURCE_AID",
       "the MetaboNet source contributing the most subjects is classified closed_loop == 'yes'")
def _(d):
    cls = {s["source"]: s["closed_loop"] for s in d["SD"]["studies"]}
    by = mb_subjects_by_source(d)
    top = max(by, key=by.get)
    return cls[top] == "yes", f"largest {top} ({by[top]}) class {cls[top]}"


# --------------------------------------------------------------------------- #
# Section 5 / Appendix E -- OhioT1DM personalisation
# --------------------------------------------------------------------------- #
@claim("PERS_EVERY_MODE_WORSE",
       "every prefix length, every adaptation mode: mean RMSE change > 0")
def _(d):
    bad = [(h, m, ps_sig(d, h, m)["mean_delta"]) for h in d["PS"]["significance"]
           for m in ps_modes(d, h) if not ps_sig(d, h, m)["mean_delta"] > 0]
    return not bad, f"non-positive: {bad}" if bad else "all positive"


@claim("PERS_HEADLINE_SIGNIFICANT",
       "headline prefix: every mode has Holm-adjusted p < 0.05 for its RMSE change")
def _(d):
    h = ps_headline(d)
    ps = {m: ps_sig(d, h, m)["p_t_holm"] for m in ps_modes(d, h)}
    return all(p is not None and p < 0.05 for p in ps.values()), \
        "; ".join(f"{m} {fmt(p, 4)}" for m, p in ps.items())


@claim("PERS_LORA_DAMAGE_GROWS_WITH_STEPS",
       "LoRA r=4 step budget: mean RMSE change strictly increasing in steps")
def _(d):
    b = d["PS"]["step_budget"]["lora (r=4)"]
    xs = [b[k]["rmse"]["mean_delta"] for k in sorted(b, key=int)]
    return strictly_increasing(xs), f"{[round(x, 3) for x in xs]}"


@claim("PERS_BN_STOPS_SHRINKING",
       "BatchNorm RMSE change > 0 at every prefix, and the 24h->48h reduction < 0.1 * the 12h->24h reduction")
def _(d):
    xs = {float(h): ps_sig(d, h, "bn_adapt")["mean_delta"] for h in d["PS"]["significance"]}
    pos = all(v > 0 for v in xs.values())
    late, early = xs[24.0] - xs[48.0], xs[12.0] - xs[24.0]
    return pos and late < 0.1 * early, f"{ {k: round(v, 3) for k, v in sorted(xs.items())} }"


@claim("PERS_HEAD_ONLY_IMPROVES_NOT_ZERO",
       "head-only RMSE change strictly decreasing with prefix length and > 0 at every length")
def _(d):
    hs = sorted(d["PS"]["significance"], key=float)
    xs = [ps_sig(d, h, "head_only")["mean_delta"] for h in hs]
    return strictly_decreasing(xs) and all(x > 0 for x in xs), f"{[round(x, 3) for x in xs]}"


@claim("POS_RECENT_STILL_DEGRADES",
       "recent prefix: every mode has mean RMSE change > 0 with Holm-adjusted p < 0.05")
def _(d):
    ev, ok = [], True
    for m, b in ps_contrast(d).items():
        st = b["recent"]["stats"]["rmse"]
        ok &= st["mean_delta"] > 0 and st["p_t_holm"] is not None and st["p_t_holm"] < 0.05
        ev.append(f"{m} {fmt(st['mean_delta'], 3)} p={fmt(st['p_t_holm'], 4)}")
    return ok, "; ".join(ev)


@claim("POS_RECENT_VS_OPENING_NULL",
       "'null throughout': every mode, RMSE and PR-AUC, recent-minus-opening Holm-adjusted p > 0.05")
def _(d):
    bad = [(m, met, b["recent_vs_opening"][met]["p_t_holm"]) for m, b in ps_contrast(d).items()
           for met in ("rmse", "pr_auc")
           if not (b["recent_vs_opening"][met]["p_t_holm"] is not None
                   and b["recent_vs_opening"][met]["p_t_holm"] > 0.05)]
    return not bad, f"not null: {bad}" if bad else "all Holm p > 0.05"


@claim("POS_NO_MODE_BETTER_FROM_RECENT",
       "'no mode does better from the recent prefix: every difference is positive': every non-'none' "
       "mode's recent-minus-opening RMSE mean change > 0 (recent_vs_opening = recent - opening)")
def _(d):
    xs = {m: b["recent_vs_opening"]["rmse"]["mean_delta"] for m, b in ps_contrast(d).items()}
    return all(x > 0 for x in xs.values()), "; ".join(f"{m} {fmt(x, 3)}" for m, x in xs.items())


@claim("POS_LORA4_CI_STRADDLES_ZERO",
       "LoRA r=4 recent-minus-opening RMSE 95% CI includes 0")
def _(d):
    st = ps_contrast(d)["lora (r=4)"]["recent_vs_opening"]["rmse"]
    return spans_zero(st["ci95_low"], st["ci95_high"]), f"[{fmt(st['ci95_low'], 3)}, {fmt(st['ci95_high'], 3)}]"


# "Head-only fine-tuning is nominally worse from the recent prefix" is one case of
# POS_NO_MODE_BETTER_FROM_RECENT (every mode > 0); a separate assertion could not fail alone.


@claim("POS_FULL_WORSE_FOREWARNING_RECENT",
       "full fine-tuning recent-minus-opening PR-AUC mean change < 0")
def _(d):
    x = ps_contrast(d)["full"]["recent_vs_opening"]["pr_auc"]["mean_delta"]
    return x < 0, fmt(x)


# --------------------------------------------------------------------------- #
# Provenance of quoted quantities: compute hours, prefix membership, step budget,
# subject counts, table column sources (audit blockers B1-B4, fix F1)
# --------------------------------------------------------------------------- #
def _import_by_path(name, rel):
    import importlib.util
    path = os.path.join(ROOT, rel)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _all_seconds(obj):
    """Every `seconds` field anywhere under `obj` (the superseded, over-counting rule)."""
    if isinstance(obj, dict):
        return sum(v if k == "seconds" and isinstance(v, (int, float)) else _all_seconds(v)
                   for k, v in obj.items())
    if isinstance(obj, list):
        return sum(_all_seconds(v) for v in obj)
    return 0.0


@claim("COMPUTE_HOURS_ARE_PER_FOLD_SUM",
       "MetaboNet compute hours (\\NumMetaboComputeHours) = sum of per_fold[].seconds (the generator's "
       "value rounds to the same 0.1 h), and differs from the sum of every `seconds` field, which "
       "also counts summary[] totals and per-epoch history already inside each fold")
def _(d):
    mod = _import_by_path("_rebuild_paper_tables_for_claims", "scripts/rebuild_paper_tables.py")
    res = d["MB"]
    per_fold = sum(f["seconds"] for f in res["per_fold"]) / 3600.0
    every = _all_seconds(res) / 3600.0
    macro = mod._wall_clock_hours(res)
    ok = round(macro, 1) == round(per_fold, 1) and round(every, 1) != round(per_fold, 1)
    return ok, f"macro {fmt(macro, 3)} h; per_fold {fmt(per_fold, 3)} h; all `seconds` fields {fmt(every, 3)} h"


@claim("TABLE_WIDTH_COLUMN_IS_SPLIT",
       "Tables 3-4: the 'Cov.' and 'Width' columns are both read from conformal.split.*, the caption "
       "says split-conformal and names the Mondrian < 70 stratum for 'Hypo cov.', and split mean_width "
       "exists for every arm that has a conformal block at both horizons")
def _(d):
    import importlib
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    B = importlib.import_module("causalcgm_mt.benchmark")
    cols = {c.label: c.path for c in B.SOTA_COLUMNS}
    cap = B.sota_caption(30, 12, 1, 0.1)
    paths_ok = (cols["Cov."].startswith("conformal.split.")
                and cols["Width"].startswith("conformal.split.")
                and cols["Hypo cov."].startswith("conformal.mondrian."))
    cap_ok = "split-conformal" in cap and "Mondrian $<70$" in cap
    missing = []
    for arm in d["BENCH"]["arms"]:
        for h in HORIZONS:
            conf = arm.get("per_horizon", {}).get(h, {}).get("conformal")
            if conf and "mean_width" not in conf.get("split", {}):
                missing.append((arm["arm"], h))
    return paths_ok and cap_ok and not missing, \
        f"Cov. {cols['Cov.']}; Width {cols['Width']}; Hypo cov. {cols['Hypo cov.']}; missing split width {missing}"


@claim("POS_RECENT_PREFIX_OUTSIDE_FIT_AND_VAL",
       "prefix-position control: for every mode and subject, no recent-prefix window lies in the "
       "population model's fit or model-selection rows (n_adapt_in_fit = n_adapt_in_val = 0), while every "
       "opening-prefix window lies in the fit rows")
def _(d):
    bad = []
    for m, b in d["PS"]["prefix_mode_contrast"].items():
        for r in b["recent"]["rows"]:
            if r["n_adapt_in_fit"] != 0 or r["n_adapt_in_val"] != 0:
                bad.append((m, r["patient"], "recent", r["n_adapt_in_fit"], r["n_adapt_in_val"]))
        for r in b["opening"]["rows"]:
            if r["n_adapt_in_fit"] != r["n_adapt_windows"]:
                bad.append((m, r["patient"], "opening", r["n_adapt_in_fit"], r["n_adapt_windows"]))
    n = sum(len(b["recent"]["rows"]) for b in d["PS"]["prefix_mode_contrast"].values())
    return n > 0 and not bad, f"{n} recent rows checked; violations: {bad}" if bad else f"{n} recent rows checked"


@claim("PERS_SHORT_HEAD_ONLY_NOMINALLY_LOWER_RMSE",
       "step budget, head-only: mean RMSE change < 0 at 10, 25 and 50 steps and > 0 at 100 and 200; "
       "the 10-step 95% CI excludes 0 with p_t < 0.05, and that cell has no Holm adjustment (unadjusted); "
       "LoRA r=4 and r=8: mean RMSE change > 0 at the largest budget")
def _(d):
    sb = d["PS"]["step_budget"]
    h = sb["head_only"]
    signs = (all(h[k]["rmse"]["mean_delta"] < 0 for k in ("10", "25", "50"))
             and all(h[k]["rmse"]["mean_delta"] > 0 for k in ("100", "200")))
    t = h["10"]["rmse"]
    ten = t["ci95_high"] < 0 and t["p_t"] < 0.05 and t.get("p_t_holm") is None
    top = str(max(int(k) for k in sb["lora (r=4)"]))
    lora = all(sb[m][top]["rmse"]["mean_delta"] > 0 for m in ("lora (r=4)", "lora (r=8)"))
    return signs and ten and lora, (f"head-only {[round(h[k]['rmse']['mean_delta'], 3) for k in ('10', '25', '50', '100', '200')]}; "
                                    f"10-step CI [{fmt(t['ci95_low'], 3)}, {fmt(t['ci95_high'], 3)}] p={fmt(t['p_t'], 3)} holm={t.get('p_t_holm')}")


@claim("PERS_STEP_BUDGET_PRAUC_NEVER_IMPROVES",
       "step budget: mean PR-AUC change < 0 at every step count for every adapter")
def _(d):
    bad = [(m, k, v["pr_auc"]["mean_delta"]) for m, by in d["PS"]["step_budget"].items()
           for k, v in by.items() if not v["pr_auc"]["mean_delta"] < 0]
    return not bad, f"non-negative: {bad}" if bad else "all negative"


@claim("PERS_PREFIX_SUBJECT_COUNTS",
       "prefix-length trends: the 12, 24 and 48 h cells use the same subjects for every mode, and the "
       "6 h cell uses fewer, so the six-hour figures must be quoted with their n")
def _(d):
    sig = d["PS"]["significance"]
    ns = {h: {ps_sig(d, h, m)["n"] for m in ps_modes(d, h)} for h in sig}
    same = all(len(v) == 1 for v in ns.values()) and len({next(iter(ns[h])) for h in ("12.0", "24.0", "48.0")}) == 1
    fewer = next(iter(ns["6.0"])) < next(iter(ns["12.0"]))
    return same and fewer, f"n by prefix: { {h: sorted(v) for h, v in ns.items()} }"


# --------------------------------------------------------------------------- #
def run(root: str = ROOT) -> list[tuple[str, bool, str, str]]:
    """Evaluate every assertion; returns (id, ok, rule, evidence). Errors fail closed."""
    d = Data(root)
    out = []
    for a in ASSERTIONS:
        try:
            ok, ev = a.fn(d)
            ok = bool(ok)
        except Exception as e:  # noqa: BLE001 -- a missing key is a failed claim
            ok, ev = False, f"ERROR {type(e).__name__}: {e}"
        out.append((a.aid, ok, a.rule, ev))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="tree holding runs/ and manuscript/")
    ap.add_argument("--list", action="store_true", help="print the assertions and exit")
    ap.add_argument("-q", "--quiet", action="store_true", help="print failures only")
    args = ap.parse_args(argv)
    if args.list:
        for a in ASSERTIONS:
            print(f"{a.aid}: {a.rule}")
        return 0
    res = run(args.root)
    fails = [r for r in res if not r[1]]
    for aid, ok, rule, ev in res:
        if ok and args.quiet:
            continue
        print(f"[{'PASS' if ok else 'FAIL'}] {aid}\n    rule: {rule}\n    json: {ev}")
    print(f"\n{len(res) - len(fails)}/{len(res)} claim assertions hold.")
    if fails:
        print("FAILED: " + ", ".join(r[0] for r in fails), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
