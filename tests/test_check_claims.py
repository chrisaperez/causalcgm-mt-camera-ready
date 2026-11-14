"""Mutation self-test for scripts/check_claims.py.

For every assertion the checker makes, a temp copy of the result JSON is
perturbed so that exactly that assertion flips (a true claim becomes false, or,
for an assertion that currently fails because the text is wrong, the JSON is
perturbed into the state the text describes) and every other assertion keeps
its status. A checker that cannot fail proves nothing; one whose assertions
fail together proves less than it claims.

The temp tree symlinks every unmutated input to the real file and writes the
mutated one, so the real runs/ are never touched.

Two assertions are logically entailed by others and cannot flip alone; their
mutations declare the co-failing assertion and the reason (``also``).
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_claims", os.path.join(ROOT, "scripts", "check_claims.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_claims"] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


CC = _load_checker()


def _status(root):
    return {aid: ok for aid, ok, _rule, _ev in CC.run(root)}


@pytest.fixture(scope="module")
def baseline():
    for rel in CC.FILES.values():
        if not os.path.exists(os.path.join(ROOT, rel)):
            pytest.skip(f"result artefact missing: {rel}")
    return _status(ROOT)


# --------------------------------------------------------------------------- #
# Mutation helpers (operate on the loaded JSON of one file)
# --------------------------------------------------------------------------- #
def _leak(o, h, arm):
    return o["studies"]["d_leakage"]["inflation"][h][arm]


def _arm(o, study, name, cohort=None):
    rows = o["studies"][study] if cohort is None else o["studies"][study][cohort]
    return next(r for r in rows if r["arm"] == name)


def _kappa(o, k):
    return next(r for r in o["studies"]["b_asymmetry"]
                if float(r["spec"]["hypo_asymmetry"]) == k)


def _set_rmse(r, h, v):
    r["per_horizon"][h]["regression"]["rmse"] = v


def _set_pr(r, h, v):
    r["per_horizon"][h]["classification"]["pr_auc"] = v


def _bench_conf(o):
    return next(a for a in o["arms"] if a["arm"] == CC.FUSION)["per_horizon"]["30"]["conformal"]


def _mb_fold(o, arm, fold):
    return next(r for r in o["per_fold"] if r["arm"] == arm and r["fold"] == fold)


def _mb_pers(o):
    return next(s for s in o["summary"] if s.get("personalization"))["personalization"]


def _drop_nested_seconds(o):
    """Remove every `seconds` field that is not a per_fold[] fold total."""
    def strip(x, top):
        if isinstance(x, dict):
            for k in [k for k in x if k == "seconds" and not top]:
                del x[k]
            for v in x.values():
                strip(v, False)
        elif isinstance(x, list):
            for v in x:
                strip(v, False)
    for f in o["per_fold"]:
        for k, v in f.items():
            if k != "seconds":
                strip(v, False)
    o["summary"] = [{k: v for k, v in r.items() if k != "seconds"} if isinstance(r, dict) else r
                    for r in o["summary"]]


def _ps(o, h, mode):
    return o["significance"][h][mode]["rmse"]


def _contrast(o, mode, metric):
    return o["prefix_mode_contrast"][mode]["recent_vs_opening"][metric]


def _kendall_doubled_fixed(o):
    """Fixed-lambda PR-AUC that doubles the PH30 Kendall-minus-fixed edge."""
    k = _arm(o, "a_multitask", "Joint, Kendall uncertainty")["per_horizon"]["30"]["classification"]["pr_auc"]
    f = _arm(o, "a_multitask", "Joint, fixed $\\lambda=1$")["per_horizon"]["30"]["classification"]["pr_auc"]
    return k - 2 * (k - f)


N, S = CC.LEAK_NAIVE, CC.LEAK_SPLIT
LGBM, LSTM, CGMO, FUS, PER = CC.LGBM, CC.LSTM, CC.CGM_ONLY, CC.FUSION, CC.PERSIST


def m(file, fn, also=(), why=""):
    return {"file": file, "fn": fn, "also": set(also), "why": why}


def _seq(*fns):
    def go(o):
        for f in fns:
            f(o)
    return go


def _assign(getter, key, value):
    return lambda o: getter(o).__setitem__(key, value)


MUTATIONS = {
    # ---- Ablation D -----------------------------------------------------
    "LEAK_NAIVE_30_DIRECTION": m(
        "AB", _assign(lambda o: _leak(o, "30", N), "rmse_relative_improvement_pct", -0.5),
        also={"LEAK_SPLIT_30_LITTLE"},
        why="any r_naive30 <= 0 that still exceeds r_split30 (< 0) is smaller in magnitude "
            "than 4*|r_split30|, so 'changes little' fails with it"),
    "LEAK_NAIVE_60_DIRECTION": m(
        "AB", _assign(lambda o: _leak(o, "60", N), "pr_auc_absolute_gain", -0.01),
        also={"LEAK_NAIVE_EXCEEDS_SPLIT_BOTH_H"},
        why="g_split60 > 0, so g_naive60 <= 0 cannot exceed it"),
    "LEAK_NAIVE_EXCEEDS_SPLIT_BOTH_H": m(
        "AB", _assign(lambda o: _leak(o, "60", N), "rmse_relative_improvement_pct", 4.0)),
    "LEAK_INCREMENT_EXCEEDS_SPLIT_BOTH_H": m(
        "AB", _assign(lambda o: _leak(o, "60", N), "rmse_leaky", 29.0)),
    "LEAK_SPLIT_30_DIRECTION": m(
        "AB", _assign(lambda o: _leak(o, "30", S), "rmse_relative_improvement_pct", 0.5)),
    "LEAK_SPLIT_30_LITTLE": m(
        "AB", _assign(lambda o: _leak(o, "30", S), "pr_auc_absolute_gain", 0.05)),
    "LEAK_SPLIT_60_NOT_NEGLIGIBLE": m(
        "AB", _seq(_assign(lambda o: _leak(o, "60", S), "rmse_relative_improvement_pct", 1.0),
                   _assign(lambda o: _leak(o, "60", S), "pr_auc_absolute_gain", 0.01))),
    # ---- multi-seed benchmark --------------------------------------------
    "MS_FUSION_BEATS_CGMONLY_RMSE": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][CGMO]["60"]["rmse"], "ci95", [-2.2, 0.1])),
    "MS_FUSION_VS_LGBM_UNRESOLVED": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][LGBM]["60"]["rmse"], "ci95", [-0.99, -0.01])),
    "MS_LGBM_ONSET_PRAUC": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][LGBM]["30"]["pr_auc_onset"], "diff", 0.01)),
    "MS_NEURAL_PRAUC_UNSEPARATED": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][LSTM]["60"]["pr_auc"], "ci95", [0.001, 0.02])),
    "MS_DE_FUSION_BELOW_LGBM": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][LGBM]["30"]["de"], "ci95", [-125.0, 5.0])),
    "MS_DE_SAME_LOSS_UNRESOLVED": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][LSTM]["30"]["de"], "ci95", [1.0, 59.0])),
    "DE_GAP_IS_THE_LOSS": m(
        "MS", _assign(lambda o: o["arms"][LGBM]["30"]["de"], "mean", 400.0)),
    "PERSIST_STRONG_CLASSIFIER": m(
        "BENCH", _assign(lambda o: next(a for a in o["arms"] if a["arm"] == PER)
                         ["per_horizon"]["30"]["classification"], "pr_auc", 0.5)),
    "PERSIST_ONSET_FLOOR": m(
        "MS", _assign(lambda o: o["arms"][PER]["30"]["pr_auc_onset"], "mean", 0.6)),
    "METABO_REPLICATES_SMALLER": m(
        "MS", _assign(lambda o: o["diffs_vs_reference"][CGMO]["30"]["rmse"], "diff", -0.2)),
    # ---- Ablation C --------------------------------------------------------
    "ABLC_TREATMENT_LOWERS_RMSE": m(
        "AB", _seq(lambda o: _set_rmse(_arm(o, "c_modality", CC.C_TREAT, "2020"), "60", 34.95),
                   lambda o: _set_rmse(_arm(o, "c_modality", CC.C_ALL, "2020"), "60", 35.0))),
    "ABLC_ALL_AUX_NO_BETTER": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", CC.C_ALL, "2020"), "60", 32.0)),
    "ABLC_TREATMENT_OVER_WEARABLES": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", "CGM + steps", "2018"), "60", 30.6)),
    "ABLC_WEARABLE_30_SMALL": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", "CGM + HR", "2018"), "30", 19.5)),
    "ABLC_WEARABLE_30_SIGN_INCONSISTENT": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", "CGM + GSR", "2020"), "30", 19.66)),
    "ABLC_HR_GSR_WORSE_2018": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", "CGM + HR", "2018"), "30", 18.9)),
    "ABLC_STEPS_60_EXCEPTION": m(
        "AB", lambda o: _set_rmse(_arm(o, "c_modality", "CGM + accel", "2020"), "60", 34.9)),
    # ---- Ablation A --------------------------------------------------------
    "ABLA_JOINT_COSTS_SPECIALISTS": m(
        "AB", lambda o: _set_rmse(_arm(o, "a_multitask", "Regression only"), "30", 18.6)),
    "ABLA_COST_FEW_THOUSANDTHS": m(
        "AB", lambda o: _set_pr(_arm(o, "a_multitask", "Classification only"), "30", 0.885)),
    "ABLA_KENDALL_VS_FIXED_DIRECTIONS": m(
        "AB", lambda o: _set_pr(_arm(o, "a_multitask", "Joint, fixed $\\lambda=1$"), "60", 0.78)),
    # double the PH30 Kendall edge (0.0223 -> 0.0446 > 1.25 x 0.0219)
    "ABLA_KENDALL_VS_FIXED_COMPARABLE_TO_SEED_RANGE": m(
        "AB", lambda o: _set_pr(_arm(o, "a_multitask", "Joint, fixed $\\lambda=1$"), "30",
                                _kendall_doubled_fixed(o))),
    # ---- Ablation B --------------------------------------------------------
    "ABLB_KAPPA3_TRADE": m("AB", lambda o: _set_pr(_kappa(o, 3.0), "30", 0.87)),
    "ABLB_MONOTONE_30": m(
        "AB", _assign(lambda o: _kappa(o, 5.0)["per_horizon"]["30"]["clinical"], "hypo_band_bias", 7.0)),
    "ABLB_PRAUC_NOT_MONOTONE_30": m(
        "AB", _seq(lambda o: _set_pr(_kappa(o, 2.0), "30", 0.866),
                   lambda o: _set_pr(_kappa(o, 8.0), "30", 0.85))),
    "ABLB_PRAUC_RANGE_COMPARABLE_TO_SEEDS": m("AB", lambda o: _set_pr(_kappa(o, 8.0), "30", 0.80)),
    "ABLB_R70_NOT_MONOTONE_60": m(
        "AB", _assign(lambda o: _kappa(o, 2.0)["per_horizon"]["60"]["clinical"], "rmse_below_70", 29.0)),
    "ABLB_SAME_SHAPE_60": m("AB", lambda o: _set_pr(_kappa(o, 8.0), "60", 0.78)),
    # ---- Ablation E --------------------------------------------------------
    "ABLE_NO_DOMINANT_CONFIG": m(
        "AB", _seq(lambda o: _set_pr(_arm(o, "e_soft_label", "Fixed $\\pm$20 mg/dL"), "30", 0.87),
                   _assign(lambda o: _arm(o, "e_soft_label", "Fixed $\\pm$20 mg/dL")
                           ["per_horizon"]["30"]["clinical"], "zone_de_n", 160))),
    "ABLE_WIDTH_DOES_NOT_ORDER": m(
        "AB", _seq(lambda o: _set_pr(_arm(o, "e_soft_label", "Fixed $\\pm$5 mg/dL"), "60", 0.765),
                   lambda o: _set_pr(_arm(o, "e_soft_label", "Fixed $\\pm$10 mg/dL"), "60", 0.768),
                   lambda o: _set_pr(_arm(o, "e_soft_label", "Fixed $\\pm$20 mg/dL"), "60", 0.77))),
    "ABLE_ESTIMATED_LIKE_NEIGHBOURS": m(
        "AB", lambda o: _set_pr(_arm(o, "e_soft_label", "Estimated (MARD)"), "30", 0.89)),
    "ABLE_HARD_HIGHEST_FAPD": m(
        "AB", _assign(lambda o: _arm(o, "e_soft_label", "Fixed $\\pm$20 mg/dL")
                      ["per_horizon"]["60"]["alarm"], "fapd_at_5pct_fpr", 3.8)),
    # ---- Ablation F / latching ------------------------------------------------
    "ABLF_NADIR_HELPS_RIDGE": m(
        "AB", _assign(lambda o: _arm(o, "f_nadir_target", "Ridge-AR, nadir target")
                      ["per_horizon"]["30"]["alarm"], "fapd_at_5pct_fpr", 3.1)),
    "ABLF_ROC_BARELY_MOVES": m(
        "AB", _assign(lambda o: _arm(o, "f_nadir_target", "Ridge-AR, nadir target")
                      ["per_horizon"]["30"]["classification"], "roc_auc", 0.95)),
    "ABLF_NADIR_RIDGE_OVERTAKES_LGBM": m(
        "AB", lambda o: _set_pr(_arm(o, "f_nadir_target", "LightGBM (native nadir classifier)"), "30", 0.87)),
    "ABLF_MT_LEAD_SMALL": m(
        "AB", lambda o: _set_pr(_arm(o, "f_nadir_target", "CausalCGM-MT (native nadir head)"), "60", 0.80)),
    "LATCH_PERSIST_LOWEST_FAPD_SHORTEST_LEAD": m(
        "FL", _assign(lambda o: o["results"]["60"]["G30_D5"][PER], "fapd", 10.0)),
    "LATCH_FUSION_BELOW_LGBM": m(
        "FL", _assign(lambda o: o["results"]["30"]["G15_D5"][LGBM], "diff_fapd_vs_fusion_ci95", [-0.686, 0.01])),
    "LATCH_D15_FUSION_DETECTS_FEWER": m(
        "FL", _assign(lambda o: o["results"]["30"]["G0_D15"][LGBM], "diff_onset_det_vs_fusion_ci95", [-0.08, 0.001])),
    "LATCH_FUSION_FALLS_UNDER_60MIN": m(
        "FL", _assign(lambda o: o["results"]["30"]["G60_D5"][FUS], "fapd", 3.0)),
    # ---- conformal -----------------------------------------------------------
    "CONF_SPLIT_MARGINAL_VALID": m(
        "BENCH", _assign(lambda o: _bench_conf(o)["split"], "marginal_coverage", 0.88)),
    "CONF_SPLIT_HYPO_OVERCOVERS": m(
        "BENCH", _assign(lambda o: _bench_conf(o)["split"], "hypo_stratum_coverage", 0.91)),
    "CONF_MONDRIAN_EQUALISES_AND_NARROWS": m(
        "BENCH", _assign(lambda o: _bench_conf(o)["mondrian"]["by_stratum"]["hypoglycemic (<70)"],
                         "coverage", 0.80)),
    "CONF_MONDRIAN_NARROW_HYPO_COSTS_SEVERE": m(
        "BENCH", _assign(lambda o: _bench_conf(o)["normalized"], "hypo_stratum_width", 45.0)),
    "CONF_MONDRIAN_UNDERCOVERS_EVERY_SEED": m(
        "CB", _assign(lambda o: o["seeds"]["seed456"]["mondrian"]["cov_severe_lt54"], "p_below_nominal", 0.3)),
    "CONF_NORMALISED_NOT_CONSISTENT": m(
        "CB", _seq(_assign(lambda o: o["seeds"]["seed123"]["normalized"]["cov_severe_lt54"], "point", 0.89),
                   _assign(lambda o: o["seeds"]["seed1337"]["normalized"]["cov_severe_lt54"], "point", 0.98))),
    "CONF_SEED_VARIATION_EXCEEDS_DRAW_VARIATION": m(
        "CB", _seq(_assign(lambda o: o["seeds"]["seed123"]["split"]["cov_severe_lt54"], "boot_p2_5", 0.80),
                   _assign(lambda o: o["seeds"]["seed123"]["split"]["cov_severe_lt54"], "boot_p97_5", 0.95))),
    "CONF_NO_CONSTRUCTION_CERTIFIES_SEVERE": m(
        "CB", _seq(_assign(lambda o: o["seeds"]["seed123"]["normalized"]["cov_severe_lt54"], "point", 0.92),
                   _assign(lambda o: o["seeds"]["seed1337"]["normalized"]["cov_severe_lt54"], "boot_p2_5", 0.90))),
    "CONF_SPLIT_NORMALISED_OVERLAP": m(
        "CB", lambda o: [o["seeds"][s]["normalized"]["cov_severe_lt54"].__setitem__(
            "point", 0.70 if s == "seed123" else 0.86) for s in o["seeds"]]),
    # ---- MetaboNet -------------------------------------------------------------
    "METABO_FUSION_RMSE_ALL_FOLDS": m(
        "MB", lambda o: _mb_fold(o, "fusion", 4)["per_horizon"]["30"]["regression"].__setitem__("rmse", 23.4)),
    "METABO_FUSION_LOWERS_FAPD": m(
        "MB", lambda o: [_mb_fold(o, "fusion", f)["per_horizon"]["30"]["alarm"].__setitem__(
            "fapd_at_5pct_fpr", _mb_fold(o, "cgm_only", f)["per_horizon"]["30"]["alarm"]["fapd_at_5pct_fpr"] + 1.0)
            for f in range(5)]),
    "METABO_PRAUC_GAIN_NEGLIGIBLE": m(
        "MB", lambda o: _mb_fold(o, "fusion", 0)["per_horizon"]["30"]["classification"].__setitem__(
            "pr_auc", _mb_fold(o, "fusion", 0)["per_horizon"]["30"]["classification"]["pr_auc"] + 0.1)),
    "METABO_DE_UNRESOLVED": m(
        "MB", lambda o: [_mb_fold(o, "fusion", f)["per_horizon"]["30"]["clinical"].__setitem__("zone_de_n", 1700)
                         for f in (0, 1)]),
    "METABO_PERS_DEGRADES": m(
        "MB", lambda o: [s.__setitem__("rmse_after", s["rmse_before"] - 5.0)
                         for s in _mb_pers(o)["per_subject"][:600]]),
    "METABO_PERS_PRAUC_SMALLER_BASE": m(
        "MB", lambda o: _mb_pers(o).__setitem__("mean_delta_pr_auc", 0.01)),
    "METABO_MOST_AID": m(
        "SD", lambda o: [s.__setitem__("closed_loop", "mixed") for s in o["studies"]
                         if s["closed_loop"] == "yes" and s["source"] != "Loop"]),
    "METABO_LARGEST_SOURCE_AID": m(
        "MB", lambda o: [(r["test_subjects_by_source"].__setitem__("Loop", r["test_subjects_by_source"]["Loop"] - 80),
                          r["test_subjects_by_source"].__setitem__("ReplaceBG", r["test_subjects_by_source"]["ReplaceBG"] + 80))
                         for r in o["per_fold"] if r["arm"] == o["per_fold"][0]["arm"]]),
    # ---- OhioT1DM personalisation ------------------------------------------------
    "PERS_EVERY_MODE_WORSE": m("PS", _assign(lambda o: _ps(o, "6.0", "lora (r=8)"), "mean_delta", -0.1)),
    "PERS_HEADLINE_SIGNIFICANT": m("PS", _assign(lambda o: _ps(o, "24.0", "head_only"), "p_t_holm", 0.2)),
    "PERS_LORA_DAMAGE_GROWS_WITH_STEPS": m(
        "PS", _assign(lambda o: o["step_budget"]["lora (r=4)"]["100"]["rmse"], "mean_delta", 3.0)),
    "PERS_BN_STOPS_SHRINKING": m("PS", _assign(lambda o: _ps(o, "48.0", "bn_adapt"), "mean_delta", 0.5)),
    "PERS_HEAD_ONLY_IMPROVES_NOT_ZERO": m("PS", _assign(lambda o: _ps(o, "12.0", "head_only"), "mean_delta", 2.0)),
    "POS_RECENT_STILL_DEGRADES": m(
        "PS", _assign(lambda o: o["prefix_mode_contrast"]["bn_adapt"]["recent"]["stats"]["rmse"], "p_t_holm", 0.2)),
    "POS_RECENT_VS_OPENING_NULL": m("PS", _assign(lambda o: _contrast(o, "full", "pr_auc"), "p_t_holm", 0.01)),
    "POS_NO_MODE_BETTER_FROM_RECENT": m(
        "PS", _assign(lambda o: _contrast(o, "bn_adapt", "rmse"), "mean_delta", -0.1)),
    "POS_LORA4_CI_STRADDLES_ZERO": m("PS", _assign(lambda o: _contrast(o, "lora (r=4)", "rmse"), "ci95_low", 0.1)),
    "POS_FULL_WORSE_FOREWARNING_RECENT": m("PS", _assign(lambda o: _contrast(o, "full", "pr_auc"), "mean_delta", 0.01)),
    # ---- provenance of quoted quantities (B1-B4, F1) ----------------------------
    # Dropping the summary[] totals and per-epoch history leaves only per-fold
    # seconds, so "the all-fields sum differs from the per-fold sum" stops holding.
    "COMPUTE_HOURS_ARE_PER_FOLD_SUM": m("MB", _drop_nested_seconds),
    "TABLE_WIDTH_COLUMN_IS_SPLIT": m(
        "BENCH", lambda o: _bench_conf(o)["split"].pop("mean_width")),
    "POS_RECENT_PREFIX_OUTSIDE_FIT_AND_VAL": m(
        "PS", lambda o: o["prefix_mode_contrast"]["full"]["recent"]["rows"][0].__setitem__("n_adapt_in_fit", 5)),
    "PERS_SHORT_HEAD_ONLY_NOMINALLY_LOWER_RMSE": m(
        "PS", _assign(lambda o: o["step_budget"]["head_only"]["25"]["rmse"], "mean_delta", 0.1)),
    "PERS_STEP_BUDGET_PRAUC_NEVER_IMPROVES": m(
        "PS", _assign(lambda o: o["step_budget"]["lora (r=8)"]["10"]["pr_auc"], "mean_delta", 0.01)),
    "PERS_PREFIX_SUBJECT_COUNTS": m(
        "PS", _assign(lambda o: _ps(o, "48.0", "bn_adapt"), "n", 11)),
}

_ORIG_CACHE: dict[str, object] = {}


def _original(key):
    if key not in _ORIG_CACHE:
        with open(os.path.join(ROOT, CC.FILES[key])) as f:
            _ORIG_CACHE[key] = json.load(f)
    return copy.deepcopy(_ORIG_CACHE[key])


def _mutated_tree(tmp_path, key, fn):
    """Temp tree: every input symlinked to the real file except `key`, which is mutated."""
    for k, rel in CC.FILES.items():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if k == key:
            obj = _original(k)
            fn(obj)
            dst.write_text(json.dumps(obj))
        else:
            os.symlink(os.path.join(ROOT, rel), dst)
    return str(tmp_path)


def test_every_assertion_has_a_mutation():
    ids = {a.aid for a in CC.ASSERTIONS}
    assert ids == set(MUTATIONS), (sorted(ids - set(MUTATIONS)), sorted(set(MUTATIONS) - ids))


@pytest.mark.parametrize("aid", sorted(MUTATIONS))
def test_mutation_flips_exactly_its_assertion(aid, baseline, tmp_path):
    spec = MUTATIONS[aid]
    root = _mutated_tree(tmp_path, spec["file"], spec["fn"])
    after = _status(root)
    flipped = {k for k in baseline if after[k] != baseline[k]}
    assert aid in flipped, f"{aid} did not change status under its mutation (baseline {baseline[aid]})"
    assert flipped == {aid} | spec["also"], (
        f"mutation for {aid} also flipped {sorted(flipped - {aid} - spec['also'])}; "
        f"failed to flip declared {sorted(spec['also'] - flipped)}")
    for k in spec["also"]:
        assert spec["why"], f"co-failure {k} declared without a reason"


def test_checker_exits_nonzero_on_a_false_claim(baseline, tmp_path):
    """The CLI, not just run(), fails when one claim is false."""
    aid = next(a for a in sorted(MUTATIONS) if baseline[a])
    spec = MUTATIONS[aid]
    root = _mutated_tree(tmp_path, spec["file"], spec["fn"])
    assert CC.main(["--root", root, "-q"]) == 1


def test_missing_input_fails_closed(tmp_path):
    for k, rel in CC.FILES.items():
        if k == "FL":
            continue  # leave the alarm-policy JSON out entirely
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.join(ROOT, rel), dst)
    res = {aid: ok for aid, ok, _r, _e in CC.run(str(tmp_path))}
    assert not any(ok for aid, ok in res.items() if aid.startswith("LATCH_"))


def test_checker_reads_only_json():
    assert all(rel.endswith(".json") for rel in CC.FILES.values())
    src = open(os.path.join(ROOT, "scripts", "check_claims.py")).read()
    assert "numbers.tex" not in src.replace("``numbers.tex``", "")


def test_failing_assertions_are_documented(baseline):
    """Every assertion that fails on the real tree is listed in ledger/claims_failed.md."""
    path = os.path.join(ROOT, "ledger", "claims_failed.md")
    text = open(path).read() if os.path.exists(path) else ""
    undocumented = [aid for aid, ok in baseline.items() if not ok and aid not in text]
    assert not undocumented, f"failing assertions missing from claims_failed.md: {undocumented}"


def test_claims_ledger_maps_to_assertions():
    path = os.path.join(ROOT, "ledger", "claims.csv")
    if not os.path.exists(path):
        pytest.skip("ledger/claims.csv is the author's private claim map and is not shipped")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {a.aid for a in CC.ASSERTIONS}
    used = set()
    for r in rows:
        for aid in filter(None, re.split(r"[;\s]+", r["assertion_ids"])):
            if aid in ("NONE",):
                continue
            assert aid in ids, f"claims.csv row {r['id']} names unknown assertion {aid}"
            used.add(aid)
    assert ids <= used, f"assertions not mapped to any claim: {sorted(ids - used)}"
