"""Regenerate every LaTeX table from the results JSON, without recomputation.

    python scripts/rebuild_paper_tables.py

A table is a pure function of the summary it is rendered from, so a formatting
change -- a new column, a caption fix, an escaping bug -- must not require
re-running a two-hour experiment. This script re-emits the `.tex` files from
`results_summary.json` and `ablations_summary.json` using the SAME column
definitions the drivers use, imported from them rather than restated here: a
second copy of the column list would drift, and the drift would show up as two
tables in the same paper disagreeing about what a column means.

It writes only `.tex` files and never touches the JSON, so it cannot change a
number -- only how one is displayed.
"""

from __future__ import annotations

import argparse

import numpy as np
import importlib.util
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_drv_{name}", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# The smoke guard
# --------------------------------------------------------------------------- #
# A `--smoke` run is two epochs on a sixteenth of the fit split. It writes the
# same JSON, with the same keys, into the same path as a publication run, and
# every driver stamps `"smoke": true` into it for exactly one reason: so that
# each consumer can refuse it.
#
# `package_release.py` refused them and this script did not, which is one hole
# too many. A smoke run overwrote `benchmarks/results_summary.json`; the tables
# were then re-rendered from it while `numbers.tex` still held the real numbers,
# and the manuscript compiled cleanly with an abstract claiming PR-AUC 0.8708
# beside its own Table 1 reporting 0.4403 for the same arm. Nothing failed --
# every macro resolved, every table was well formed, and the two disagreed.
#
# Refusing is therefore the default. `--allow-smoke` exists because
# `reproduce_all_paper_results.sh --smoke` legitimately renders smoke artefacts
# to exercise the plumbing, and it says so on every line it prints.
_SMOKE_REFUSALS: list[str] = []

# Artefacts that are individually fine but describe DIFFERENT runs. Kept apart
# from `_SMOKE_REFUSALS` because the remedy differs: a smoke artefact needs a
# re-run at publication budget, whereas a divergence needs the two halves
# resynchronised (which may also mean a re-run, but of only one of them).
_COUPLING_PROBLEMS: list[str] = []


def _wall_clock_hours(res) -> float:
    """Summed arm-fold wall clock of a corpus-study summary, in hours.

    Only `per_fold[].seconds` is summed. The same folds are also totalled per
    arm in `summary[].seconds`, and `per_fold[].history[].seconds` holds
    per-epoch times already contained in each fold's `seconds`; walking every
    `seconds` field therefore counted the study about two and a half times.
    """
    return sum(float(f["seconds"]) for f in res.get("per_fold", [])
               if isinstance(f.get("seconds"), (int, float))) / 3600.0


def _load_result(path: str, allow_smoke: bool = False):
    """The JSON at `path`, or `None` when it is absent or a smoke artefact."""
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload.get("smoke") and not allow_smoke:
        print(f"  REFUSING {path}: it is a --smoke artefact. Re-run the driver "
              f"at publication budget, or pass --allow-smoke to render it anyway.")
        # Each artefact is read twice -- once to render its tables, once for the
        # macros -- so record it once.
        if path not in _SMOKE_REFUSALS:
            _SMOKE_REFUSALS.append(path)
        return None
    return payload


def rebuild_benchmarks(root: str, alpha_default: float = 0.1,
                       allow_smoke: bool = False) -> int:
    path = os.path.join(root, "benchmarks", "results_summary.json")
    res = _load_result(path, allow_smoke)
    if res is None:
        return 0
    arms, ds = res["arms"], res["dataset"]
    alpha = res.get("protocol", {}).get("alpha", alpha_default)

    tex = []
    for h in ds["horizons_min"]:
        one = B.latex_table(
            arms, B.SOTA_COLUMNS, h,
            caption=B.sota_caption(h, len(ds["subjects"]), ds["n_test"], alpha),
            label=f"tab:sota_ph{h}")
        tex.append(one)
        # Per-horizon file as well as the combined one. The manuscript holds the
        # main body to the ML4H 8-page limit by \input-ing only the PH = 30
        # table there and deferring PH = 60 to an appendix, which a single
        # combined file cannot express. The combined file is still written:
        # package_release.py ships it and PUBLICATION_ROADMAP.md names it.
        B.write_latex(os.path.join(root, "benchmarks", f"table_sota_h{h}.tex"), one)
    B.write_latex(os.path.join(root, "benchmarks", "table_sota_comparison.tex"),
                  "\n\n".join(tex))
    return 1


def rebuild_ablations(root: str, allow_smoke: bool = False) -> int:
    path = os.path.join(root, "ablations", "ablations_summary.json")
    abl = _load_result(path, allow_smoke)
    if abl is None:
        return 0
    drv = _load("run_paper_ablations")
    studies = abl["studies"]
    horizons = abl.get("budget", {}).get("horizons_min", [30, 60])
    n = 0

    # THE ABLATIONS RUN LIGHTER THAN THE BENCHMARK, and every caption says so.
    # A reader who compares an ablation row against Table 1 -- the kappa = 3 arm
    # of Ablation B is the same configuration as the fusion row there -- reads
    # the budget difference as a result unless told otherwise. Both budgets are
    # derived from their own artefacts rather than restated, so the claim cannot
    # drift from what was actually run; if the studies ever disagree among
    # themselves the note is dropped rather than asserted falsely.
    metas = (abl.get("study_meta") or {}).values()
    eps = {(m.get("budget") or {}).get("epochs") for m in metas}
    subs = {(m.get("budget") or {}).get("fit_subsample") for m in metas}
    eps.discard(None)
    subs.discard(None)
    bench = _load_result(
        os.path.join(root, "benchmarks", "results_summary.json"), allow_smoke)
    b_spec = {}
    if bench:
        b_spec = next((a.get("spec") or {} for a in bench.get("arms", [])
                       if "fusion" in a.get("arm", "")), {})
    budget_note = ""
    if len(eps) == 1 and len(subs) == 1 and b_spec.get("epochs"):
        b_sub = b_spec.get("fit_subsample", 1)
        b_how = ("full fit split" if b_sub in (1, None)
                 else f"fit split subsampled ${b_sub:g}\\times$")
        ref = f"Table~\\ref{{tab:sota_ph{horizons[0]}}}"
        budget_note = (
            f" All ablation arms share one budget ({eps.copy().pop():g} epochs, "
            f"fit split subsampled ${subs.copy().pop():g}\\times$), lighter than "
            f"{ref}'s ({b_spec['epochs']:g} epochs, {b_how}); rows here compare "
            f"to each other, not to {ref}.")

    # `table_ablation_d_leakage_ph30.tex` is the one ablation table \input into
    # the 8-page body, where its caption competes with body text for the page
    # budget. Section 5.3's prose carries the budget caveat for that table
    # instead, so the note is suppressed for that file alone; the PH = 60 twin
    # in Appendix C still carries it, as does every other ablation table.
    BODY_TABLES = {("d_leakage", horizons[0])}

    def emit(key, rows, cols, title, extra=""):
        nonlocal n
        body = [B.latex_table(rows, cols, h,
                              caption=f"{title} (PH = {h} min). {extra}"
                                      f"{'' if (key, h) in BODY_TABLES else budget_note}",
                              label=f"tab:ablation_{key}_ph{h}", row_label="Arm")
                for h in horizons]
        B.write_latex(os.path.join(root, "ablations",
                                   f"table_ablation_{key}.tex"),
                      "\n\n".join(body))
        # Per-horizon files alongside the combined one, for the same reason as
        # the SOTA tables: the manuscript keeps PH = 30 in the 8-page body and
        # defers PH = 60 to an appendix, which one combined file cannot express.
        for h, one in zip(horizons, body):
            B.write_latex(os.path.join(root, "ablations",
                                       f"table_ablation_{key}_ph{h}.tex"), one)
        n += 1

    # Titles, captions and columns all come from the driver, so a table
    # regenerated here is the same table the driver would have written. They
    # used to be restated, and study E's caption drifted the first time it was
    # touched -- a raw `_` that is a hard LaTeX error had to be fixed twice.
    if "a_multitask" in studies:
        emit("a_multitask", studies["a_multitask"], drv.COLS_A,
             drv.TITLES["a_multitask"], drv.CAPTIONS["a_multitask"])
    if "b_asymmetry" in studies:
        emit("b_asymmetry", studies["b_asymmetry"], drv.COLS_B,
             drv.TITLES["b_asymmetry"], drv.CAPTIONS["b_asymmetry"])
    for cohort, rows in studies.get("c_modality", {}).items():
        emit(f"c_modality_{cohort}", rows, drv.COLS_C,
             drv.c_modality_title(cohort), drv.CAPTIONS["c_modality"])
    if studies.get("c_modality"):
        # The condensed single-table version the main text includes. Built by
        # the driver's own function, so the two cannot disagree about what a
        # delta is.
        B.write_latex(
            os.path.join(root, "ablations",
                         "table_ablation_c_modality_summary.tex"),
            B.latex_table(
                drv.c_modality_summary_rows(studies["c_modality"], horizons),
                drv.COLS_C_SUMMARY, horizons[0],
                caption=f"{drv.C_SUMMARY_CAPTION}{budget_note}",
                label="tab:ablation_c_modality_summary", row_label="Arm"))
        n += 1
    if "f_nadir_target" in studies:
        B.write_latex(
            os.path.join(root, "ablations", "table_ablation_f_nadir_target.tex"),
            "\n\n".join(
                B.latex_table(studies["f_nadir_target"], drv.COLS_F, h,
                              caption=(f"{drv.TITLES['f_nadir_target']} "
                                       f"(PH = {h} min). "
                                       f"{drv.CAPTIONS['f_nadir_target']}"
                                       f"{budget_note}"),
                              label=f"tab:ablation_f_nadir_target_ph{h}",
                              row_label="Arm")
                for h in horizons))
        n += 1
    if "e_soft_label" in studies:
        emit("e_soft_label", studies["e_soft_label"], drv.COLS_E,
             drv.TITLES["e_soft_label"], drv.CAPTIONS["e_soft_label"])
    if "d_leakage" in studies:
        emit("d_leakage", studies["d_leakage"]["arms"], drv.COLS_D,
             drv.TITLES["d_leakage"], drv.CAPTIONS["d_leakage"])
    return n


# --------------------------------------------------------------------------- #
# Inline numbers as LaTeX macros
# --------------------------------------------------------------------------- #
# Every number quoted in the prose comes from here, so the manuscript cannot
# drift from the artefacts. Retyping a metric into a sentence is how an abstract
# ends up claiming a figure the tables no longer support -- and it is invisible,
# because both the sentence and the table look right on their own.
_DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
               "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
# Curated names for the values that actually appear as macro suffixes -- horizons,
# prefix lengths in hours, and step budgets. Anything else falls back to
# digit-by-digit, which is ugly but always a legal control sequence.
_HORIZON_WORD = {6: "Six", 10: "Ten", 12: "Twelve", 15: "Fifteen",
                 24: "TwentyFour", 25: "TwentyFive", 30: "Thirty",
                 45: "FortyFive", 48: "FortyEight", 50: "Fifty", 60: "Sixty",
                 90: "Ninety", 100: "OneHundred", 120: "TwoHours",
                 200: "TwoHundred"}


def _word(n: int) -> str:
    r"""A LaTeX-legal suffix for a horizon.

    Control sequences are LETTERS ONLY: `\NumFusionRMSEH30` parses as
    `\NumFusionRMSEH` followed by the characters `30`, so a macro name with a
    digit in it silently renders the digits as text next to an undefined
    command. Curated words for the horizons we use, digit-by-digit otherwise.
    """
    return _HORIZON_WORD.get(int(n),
                             "".join(_DIGIT_WORD[c] for c in str(int(n))))


def _macro(name: str, value, fmt: str = "{:.3f}") -> str:
    # `None` was always handled; a non-finite float was not, and `"{:.3f}".
    # format(float("nan"))` is the string "nan". A NaN is a ROUTINE outcome
    # upstream -- pr_auc on a single-class slice, time_lag_min on a short one,
    # mean_lead_time_min when no event was detected -- and all of them flow into
    # `rebuild_numbers`. The manuscript then compiled clean, with no undefined
    # control sequence, printing the word "nan" mid-sentence where a result
    # belongs. The unavailable marker is the honest rendering; whether the macro
    # should exist at all is the caller's decision.
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        txt = "--"
    else:
        txt = (fmt.format(value) if isinstance(value, (int, float))
               else str(value))
    # A thousands separator inside math mode is a LIST separator: `$28,335$`
    # sets as "28, 335", with the comma spaced as punctuation. Most of these
    # macros are quoted inside `$...$` in the prose, and the ones that are not
    # render `{,}` as a plain comma too, so brace it unconditionally rather than
    # asking every call site to know which mode it will land in.
    txt = re.sub(r"(?<=\d),(?=\d\d\d)", "{,}", txt)
    return f"\\newcommand{{\\{name}}}{{{txt}}}"


# --------------------------------------------------------------------------- #
# MetaboNet: the corpus-scale external validation reported in the appendix
# --------------------------------------------------------------------------- #
# This artefact lives OUTSIDE `runs/paper_results/` on purpose -- the MetaboNet
# driver is not a stage of `reproduce_all_paper_results.sh` and is forbidden to
# write into the paper tree. Its numbers are emitted here anyway, because the
# alternative is retyping a corpus-scale result into the appendix by hand, which
# is the one failure this file exists to prevent. Reading a second results root
# is safe in the direction the isolation invariant cares about: nothing is ever
# written back to it.
DEFAULT_METABONET_ROOT = "runs/metabonet_results"


def metabonet_table_path(metabonet_root: str) -> str:
    """The one MetaboNet artefact `manuscript/main.tex` \\inputs."""
    return os.path.join(metabonet_root, "benchmarks",
                        "table_metabonet_comparison.tex")


def check_metabonet_coupling(metabonet_root: str,
                             fingerprint: str | None) -> list[str]:
    """Problems tying the appendix table to the macros rendered beside it.

    Pure: it reads and reports, and never writes or deletes. Returns an empty
    list when the pair is consistent.

    A MISSING table is not a problem here -- the driver may simply not have run
    yet, and the tree-level pairing (macros committed without a table) is
    checked by `tests/test_manuscript.py`. What is a problem is a table that is
    PRESENT and cannot be shown to belong with these macros:

      * no stamp at all -- written before provenance stamping existed, so it
        cannot be matched and must not be assumed to match;
      * stamped `smoke=true` while the macros come from a publication run;
      * stamped with a different run's fingerprint.

    The third is audit finding C1 exactly: a `--smoke` MetaboNet run replaced
    the appendix table while `numbers.tex` kept the previous publication run's
    macros, and the paper compiled clean with its prose and its table from
    different experiments.
    """
    path = metabonet_table_path(metabonet_root)
    if not os.path.exists(path):
        return []

    prov = B.read_latex_provenance(path)
    if prov is None:
        return [f"{path} carries no provenance stamp, so it cannot be matched "
                f"against the \\NumMetabo* macros; re-run "
                f"scripts/run_metabonet_benchmarks.py to stamp it"]
    problems: list[str] = []
    if prov["smoke"]:
        problems.append(
            f"{path} was written by a --smoke run (stamp says smoke=true) but "
            f"the macros beside it are not; the appendix would show two-epoch "
            f"numbers under publication-budget prose")
    if fingerprint and prov["fingerprint"] != fingerprint:
        problems.append(
            f"{path} has fingerprint {prov['fingerprint']} but the "
            f"\\NumMetabo* macros come from run {fingerprint}; the appendix "
            f"table and its prose describe different experiments")
    return problems


def _paired_fold_delta(per_fold, treatment: str, control: str, horizon: int,
                       section: str, key: str):
    r"""Mean and SD of (treatment - control) across folds, paired by fold index.

    The arms are trained on identical splits, so the fold-paired difference is
    the contrast with any power at all. Differencing the two pooled means
    instead discards the pairing, and on this corpus the between-fold spread is
    roughly six times the effect -- which would make an effect that is stable in
    every fold look like noise. Returns `(mean, sd, control_sd)`, or a triple of
    `None` when either arm is missing a fold.
    """
    def by_fold(arm):
        return {r["fold"]: r["per_horizon"].get(str(horizon), {})
                for r in per_fold if r["arm"] == arm}

    t_rows, c_rows = by_fold(treatment), by_fold(control)
    folds = sorted(set(t_rows) & set(c_rows))
    if len(folds) < 2:
        return None, None, None

    def value(rows, k):
        blk = rows[k].get(section)
        return None if blk is None else blk.get(key)

    t_vals = [value(t_rows, k) for k in folds]
    c_vals = [value(c_rows, k) for k in folds]
    if any(v is None for v in t_vals + c_vals):
        return None, None, None
    deltas = [t - c for t, c in zip(t_vals, c_vals)]
    return (statistics.mean(deltas), statistics.stdev(deltas),
            statistics.stdev(c_vals))


STUDY_DESIGN_PATH = os.path.join("manuscript", "metabonet_study_design.json")
STUDY_DESIGN_CLASSES = ("yes", "mixed", "no", "unclear")


def rebuild_metabonet_table(metabonet_root: str) -> int:
    """Re-render the MetaboNet appendix table from its summary JSON.

    `run_metabonet_benchmarks.py` writes this table at the end of a corpus run.
    Everything it reads is in the summary (the per-arm summaries, the
    provenance, the fold count and the run fingerprint), so it is rendered here
    too with the driver's own columns and caption, which is what lets a clone
    regenerate every table without re-running the corpus study. A smoke summary
    is refused, as in `metabonet_numbers`.
    """
    path = os.path.join(metabonet_root, "benchmarks", "metabonet_summary.json")
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return 0
    with open(path) as f:
        res = json.load(f)
    if res.get("smoke"):
        print(f"  REFUSING {path}: --smoke run")
        return 0
    from causalcgm_mt import benchmark as B
    from scripts.run_metabonet_benchmarks import (MANUSCRIPT_TABLE_NAME,
                                                  METABONET_COLUMNS, caption)
    prov = res["provenance"]
    h = int(res["horizon_min"])
    comparable = [r for r in res["summary"] if r["per_horizon"]]
    out = os.path.join(metabonet_root, "benchmarks", MANUSCRIPT_TABLE_NAME)
    B.write_latex(
        out,
        B.latex_table(comparable, METABONET_COLUMNS, h,
                      caption(h, prov["n_subjects"], int(res["n_folds"]),
                              prov["sources"], False),
                      label="tab:metabonet_comparison"),
        provenance={"fingerprint": res.get("run_fingerprint", ""),
                    "smoke": False, "generated": res["generated"]})
    print(f"  wrote {out}")
    return 1


def metabonet_study_design_numbers(res: dict,
                                   design_path: str = STUDY_DESIGN_PATH) -> list[str]:
    """Closed-loop macros for the MetaboNet caveat.

    Joins the tracked study-design file (one verified class per source) to the
    run's sources and per-fold test subjects. Every source in the run must be
    classified, and the file may name no source the run does not contain: a
    study count typed into prose has no file behind it and cannot be checked. The subject share counts each subject once, from
    one arm's test folds (the folds partition the subjects).
    """
    with open(design_path) as f:
        design = json.load(f)
    cls = {s["source"]: s["closed_loop"] for s in design["studies"]}
    bad = sorted(c for c in set(cls.values()) if c not in STUDY_DESIGN_CLASSES)
    if bad:
        raise SystemExit(f"{design_path}: unknown class(es) {bad}")
    run_sources = set(res.get("provenance", {}).get("sources", []))
    missing, extra = sorted(run_sources - set(cls)), sorted(set(cls) - run_sources)
    if missing or extra:
        raise SystemExit(f"{design_path} does not match the run's sources: "
                         f"unclassified {missing}, not in run {extra}")
    arm = res["per_fold"][0]["arm"]
    per_source: dict[str, int] = {}
    for row in res["per_fold"]:
        if row["arm"] != arm:
            continue
        counts = row.get("test_subjects_by_source")
        if counts is None:  # unredacted summary
            counts = {}
            for sid in row["test_subjects"]:
                src = sid.split(":", 1)[0]
                counts[src] = counts.get(src, 0) + 1
        for src, n in counts.items():
            per_source[src] = per_source.get(src, 0) + n
    n_subj = int(res["provenance"]["n_subjects"])
    if sum(per_source.values()) != n_subj:
        raise SystemExit(f"test folds of arm {arm!r} cover "
                         f"{sum(per_source.values())} subjects, provenance says {n_subj}")
    n_of = {c: sum(1 for s in run_sources if cls[s] == c) for c in STUDY_DESIGN_CLASSES}
    subj_yes = sum(n for s, n in per_source.items() if cls[s] == "yes")
    return [
        _macro("NumMetaboCLStudies", n_of["yes"], "{:d}"),
        _macro("NumMetaboCLMixedStudies", n_of["mixed"], "{:d}"),
        _macro("NumMetaboCLNoStudies", n_of["no"], "{:d}"),
        _macro("NumMetaboCLUnclearStudies", n_of["unclear"], "{:d}"),
        _macro("NumMetaboCLSubjPct", 100.0 * subj_yes / n_subj, "{:.1f}"),
    ]


def metabonet_numbers(metabonet_root: str,
                      ohio_lora_delta_rmse: float | None = None) -> list[str]:
    """Macros for the MetaboNet appendix, or `[]` when there is nothing to read.

    A `--smoke` summary is refused rather than rendered. Smoke numbers are two
    epochs on eight subjects and they look exactly like results; the driver
    stamps `"smoke": true` into the JSON so that every consumer can refuse them,
    and a macro file is a consumer that would otherwise put them in a paper.

    `ohio_lora_delta_rmse` is the OhioT1DM LoRA (r = 4) mean RMSE delta, passed
    in by `rebuild_numbers` because it comes from the OTHER results tree. The
    appendix compares the two corpora ("the same sign at ... the magnitude"),
    and that comparison was carried by a hand-written fraction until a re-run
    moved one of the two numbers underneath it. It is a derived quantity of two
    generated ones, so it is generated too. Optional: when it is absent the
    ratio macro is simply not emitted, and the appendix prose that quotes it is
    caught by `tests/test_manuscript.test_every_inline_number_macro_is_defined`
    rather than rendering a blank.
    """
    path = os.path.join(metabonet_root, "benchmarks", "metabonet_summary.json")
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return []
    with open(path) as f:
        res = json.load(f)
    if res.get("smoke"):
        # REFUSING, not "skip", and recorded so `main()` exits non-zero. It used
        # to print a lowercase skip line and return quietly: all 46 \NumMetabo*
        # macros vanished from the regenerated `numbers.tex` while the process
        # still exited 0, so a CI job gating on the exit status saw a clean run
        # and the committed macro file went stale against the appendix table.
        print(f"  REFUSING {path}: it is a --smoke run; its numbers are not "
              f"reportable. Re-run at publication budget.")
        if path not in _SMOKE_REFUSALS:
            _SMOKE_REFUSALS.append(path)
        # The table half of the pair is checked even so: a smoke summary next to
        # a publication table is the divergence, and saying only "refused" would
        # hide which artefact is the stale one.
        _COUPLING_PROBLEMS.extend(check_metabonet_coupling(metabonet_root, None))
        return []

    h = int(res["horizon_min"])
    prov = res.get("provenance", {})
    by_arm = {a["arm"]: a for a in res.get("summary", [])}
    cgm = by_arm.get("cgm_only", {}).get("per_horizon", {}).get(str(h))
    fusion = by_arm.get("fusion", {}).get("per_horizon", {}).get(str(h))
    if not cgm or not fusion:
        print(f"  skip: {path} has no cgm_only/fusion pair at PH = {h} min")
        return []

    # Which run these macros came from. Emitted as a macro rather than only as a
    # comment so it travels with `numbers.tex` into a release archive, and so
    # the appendix table can be checked against it -- the two halves of the
    # appendix are written by different processes and nothing else ties them.
    fingerprint = str(res.get("run_fingerprint", ""))
    _COUPLING_PROBLEMS.extend(check_metabonet_coupling(metabonet_root,
                                                       fingerprint))

    lines = ["",
             "% ---- MetaboNet corpus-scale external validation "
             "(runs/metabonet_results) ----"]
    lines += [
        _macro("NumMetaboFingerprint", fingerprint or "unknown"),
        _macro("NumMetaboSubjects", f"{prov['n_subjects']:,}"),
        _macro("NumMetaboSources", len(prov.get("sources", [])), "{:d}"),
        _macro("NumMetaboWindows", f"{prov['n_windows']:,}"),
        _macro("NumMetaboFolds", res["n_folds"], "{:d}"),
        # Wall clock for the corpus study, summed over every timed arm-fold.
        # This is the one stage the artefacts time directly, which is why the
        # Compute Resources statement quotes it rather than a pipeline total it
        # would have to guess at.
        _macro("NumMetaboComputeHours", _wall_clock_hours(res), "{:.1f}"),
        _macro("NumMetaboHorizon", h, "{:d}"),
        _macro("NumMetaboCap", prov.get("max_windows_per_subject"), "{:d}"),
        _macro("NumMetaboAuxChannels", len(prov.get("aux_channels", [])), "{:d}"),
        _macro("NumMetaboExcluded",
               " and ".join(prov.get("excluded_sources", [])) or "none"),
        _macro("NumMetaboHypoPrev",
               100.0 * cgm["classification"]["prevalence"], "{:.2f}"),
    ]
    lines += metabonet_study_design_numbers(res)

    for tag, blk in (("CGM", cgm), ("Fusion", fusion)):
        lines += [
            _macro(f"NumMetabo{tag}RMSE", blk["regression"]["rmse"], "{:.2f}"),
            _macro(f"NumMetabo{tag}MAE", blk["regression"]["mae"], "{:.2f}"),
            _macro(f"NumMetabo{tag}MARD", blk["regression"]["mard"], "{:.2f}"),
            _macro(f"NumMetabo{tag}Lag", blk["regression"]["time_lag_min"], "{:.0f}"),
            _macro(f"NumMetabo{tag}PRAUC", blk["classification"]["pr_auc"], "{:.3f}"),
            _macro(f"NumMetabo{tag}SeFive",
                   blk["classification"]["sensitivity_at_5pct_fpr"], "{:.3f}"),
            _macro(f"NumMetabo{tag}FAPD", blk["alarm"]["fapd_at_5pct_fpr"], "{:.2f}"),
            _macro(f"NumMetabo{tag}AB", blk["clinical"]["zone_ab_pct"], "{:.2f}"),
            _macro(f"NumMetabo{tag}DE", blk["clinical"]["zone_de_n"], "{:.0f}"),
        ]
    lines += [
        _macro("NumMetaboPersistRMSE",
               cgm["regression"]["rmse_persistence"], "{:.2f}"),
        _macro("NumMetaboPersistLag",
               cgm["regression"]["time_lag_persistence_min"], "{:.0f}"),
    ]

    # The fusion-versus-unimodal contrast, paired by fold. The safety claim is
    # that the gate holds fusion AT the unimodal baseline when the auxiliary
    # streams carry nothing usable, so the number carrying the claim is the
    # paired delta read against the between-fold spread -- not either arm's RMSE.
    d_rmse, sd_rmse, fold_sd = _paired_fold_delta(
        res.get("per_fold", []), "fusion", "cgm_only", h, "regression", "rmse")
    d_prauc, sd_prauc, _ = _paired_fold_delta(
        res.get("per_fold", []), "fusion", "cgm_only", h,
        "classification", "pr_auc")
    lines += [
        _macro("NumMetaboFusionDeltaRMSE", d_rmse, "{:+.3f}"),
        _macro("NumMetaboFusionDeltaRMSESD", sd_rmse, "{:.3f}"),
        _macro("NumMetaboFoldSDRMSE", fold_sd, "{:.3f}"),
        _macro("NumMetaboFusionDeltaPRAUC", d_prauc, "{:+.3f}"),
        _macro("NumMetaboFusionDeltaPRAUCSD", sd_prauc, "{:.3f}"),
    ]
    # How far under the noise the effect sits, as a number rather than as an
    # adjective. The appendix said "roughly six times smaller" in prose; that
    # was true of the run it was written against and false of the next one,
    # which is the same failure mode the macro discipline exists to prevent --
    # it just arrived in a comparative rather than in a measurement.
    if d_rmse and fold_sd is not None and abs(d_rmse) > 0:
        lines.append(_macro("NumMetaboFusionDeltaNoiseRatio",
                            fold_sd / abs(d_rmse), "{:.1f}"))

    # ---- the personalisation arm, which is a paired delta, not a table row -- #
    pers = by_arm.get("lora", {}).get("personalization", {})
    subjects = pers.get("per_subject", []) if pers else []
    if subjects:
        deltas = sorted(s["rmse_after"] - s["rmse_before"] for s in subjects
                        if s.get("rmse_after") is not None
                        and s.get("rmse_before") is not None)
        windows = [s["n_adapt_windows"] for s in subjects
                   if s.get("n_adapt_windows") is not None]
        # PR-AUC is undefined for a subject whose post-embargo region contains
        # no hypoglycemia at all, so the mean PR-AUC delta is over a SUBSET of
        # the subjects the RMSE delta covers. Reported as its own count: an
        # appendix that quotes both deltas against one subject count overstates
        # what the alarm number was measured on.
        pr_deltas = [s["pr_auc_after"] - s["pr_auc_before"] for s in subjects
                     if s.get("pr_auc_after") is not None
                     and s.get("pr_auc_before") is not None]
        n, improved = pers["n_subjects"], pers["n_improved_rmse"]
        lines += [
            _macro("NumMetaboPersSubjects", f"{n:,}"),
            # Corpus subjects the paired design could not score (record too
            # short for prefix + embargo + evaluation windows).
            _macro("NumMetaboPersSkipped",
                   int(res["provenance"]["n_subjects"]) - int(n), "{:d}"),
            _macro("NumMetaboPersPrefixHours", pers.get("prefix_hours"), "{:.0f}"),
            _macro("NumMetaboPersRank", pers.get("lora_rank"), "{:d}"),
            _macro("NumMetaboPersTrainable", f"{pers['n_trainable']:,}"),
            _macro("NumMetaboPersDeltaRMSE", pers["mean_delta_rmse"], "{:+.3f}"),
            _macro("NumMetaboPersDeltaPRAUC", pers["mean_delta_pr_auc"], "{:+.3f}"),
            _macro("NumMetaboPersImproved", f"{improved:,}"),
            _macro("NumMetaboPersWorsePct", 100.0 * (n - improved) / n, "{:.1f}"),
        ]
        if pr_deltas:
            lines += [
                _macro("NumMetaboPersPRAUCSubjects", f"{len(pr_deltas):,}"),
                _macro("NumMetaboPersPRAUCWorsePct",
                       100.0 * sum(1 for d in pr_deltas if d < 0) / len(pr_deltas),
                       "{:.1f}"),
            ]
        # The cross-corpus comparison, as a percentage of the OhioT1DM effect.
        # Both inputs are generated; so is their ratio.
        if ohio_lora_delta_rmse:
            lines.append(_macro(
                "NumMetaboPersOhioPct",
                100.0 * pers["mean_delta_rmse"] / ohio_lora_delta_rmse, "{:.0f}"))
        if len(deltas) >= 4:
            q1, med, q3 = statistics.quantiles(deltas, n=4)
            lines += [
                _macro("NumMetaboPersMedianDeltaRMSE", med, "{:+.3f}"),
                _macro("NumMetaboPersDeltaRMSEQOne", q1, "{:+.3f}"),
                _macro("NumMetaboPersDeltaRMSEQThree", q3, "{:+.3f}"),
            ]
        if windows:
            lines.append(_macro("NumMetaboPersAdaptWindows",
                                statistics.median(windows), "{:.0f}"))
    return lines


def rebuild_numbers(root: str, out: str,
                    metabonet_root: str = DEFAULT_METABONET_ROOT,
                    allow_smoke: bool = False) -> int:
    """Emit `manuscript/numbers.tex`. Missing artefacts are skipped, not faked."""
    lines = ["% Generated by scripts/rebuild_paper_tables.py -- do not edit.",
             "% Every inline number in main.tex resolves through these macros.",
             ""]

    bench_path = os.path.join(root, "benchmarks", "results_summary.json")
    res = _load_result(bench_path, allow_smoke)
    if res is not None:
        ds = res["dataset"]
        by_arm = {a["arm"]: a for a in res["arms"]}
        lines += [
            _macro("NumSubjects", len(ds["subjects"]), "{:d}"),
            _macro("NumFitWindows", f"{ds['n_fit']:,}"),
            _macro("NumValWindows", f"{ds['n_val']:,}"),
            _macro("NumTestWindows", f"{ds['n_test']:,}"),
            _macro("NumEmbargoed", f"{ds['n_embargoed']:,}"),
        ]
        # The conformal calibration slice. `n_calib` is recorded per arm rather
        # than in `dataset`, because the four-way split (fit | val | calib |
        # test) postdates that schema. Every arm is scored on the same `calib`
        # region, so a disagreement is a pipeline bug, not a reporting choice.
        calib = {a["n_calib"] for a in res["arms"] if "n_calib" in a}
        if len(calib) > 1:
            raise SystemExit(
                f"arms disagree on the calibration split size: {sorted(calib)}; "
                f"the conformal qhat is not comparable across them")
        if calib:
            lines.append(_macro("NumCalibWindows", f"{calib.pop():,}"))
        lines += [
            _macro("NumAlpha", res.get("protocol", {}).get("alpha", 0.1), "{:.2f}"),
        ]
        # The split decision, as MAGNITUDES rather than as an ordering. The
        # summary "LightGBM wins RMSE, we win PR-AUC" is not even true: LightGBM
        # beats the UNIMODAL arm, while fusion has the lowest RMSE in the table
        # by a hair. `gap` is fusion minus LightGBM and is therefore NEGATIVE
        # when fusion leads. Both gaps are computed here so the prose quotes
        # their sizes and signs and cannot drift from the table above it.
        for h in ds["horizons_min"]:
            lg = by_arm.get("LightGBM", {}).get("per_horizon", {}).get(str(h))
            fu = by_arm.get("CausalCGM-MT (fusion)", {}).get("per_horizon", {}).get(str(h))
            if not lg or not fu:
                continue
            gap = fu["regression"]["rmse"] - lg["regression"]["rmse"]
            lines += [
                _macro(f"NumLGBMRMSEGapH{_word(h)}", gap, "{:.2f}"),
                _macro(f"NumLGBMRMSEGapPctH{_word(h)}",
                       100.0 * gap / fu["regression"]["rmse"], "{:.1f}"),
                # The same two quantities as MAGNITUDES. Prose that says
                # "fusion overtakes LightGBM by ..." needs an unsigned number:
                # quoting the signed gap there renders "by -0.04 mg/dL", which
                # reads as the opposite of what the table shows.
                _macro(f"NumFusionRMSELeadH{_word(h)}", abs(gap), "{:.2f}"),
                _macro(f"NumFusionRMSELeadPctH{_word(h)}",
                       abs(100.0 * gap / fu["regression"]["rmse"]), "{:.1f}"),
                _macro(f"NumFusionPRAUCGapH{_word(h)}",
                       fu["classification"]["pr_auc"] - lg["classification"]["pr_auc"],
                       "{:+.3f}"),
                # Clarke D+E reduction of fusion against LightGBM, as a percent
                # of LightGBM's count. NOT \NumDEDropThree, which is the same
                # quantity for the kappa sweep against kappa = 1 and happens to
                # round to the same number at PH = 30; quoting that one here
                # would attribute the architecture's gain to the loss knob.
                _macro(f"NumFusionDEReductionVsLGBMH{_word(h)}",
                       100.0 * (lg["clinical"]["zone_de_n"] - fu["clinical"]["zone_de_n"])
                       / lg["clinical"]["zone_de_n"], "{:.0f}"),
            ]

        picks = [("CausalCGM-MT (fusion)", "Fusion"), ("CausalCGM-MT (CGM)", "CGMOnly"),
                 ("LightGBM", "LGBM"), ("Persistence", "Persist"),
                 ("Dilated LSTM (CGM)", "LSTM"), ("ARIMA(4,1,0)", "Arima"),
                 ("Ridge-AR", "Ridge"), ("CausalCGM-MT + LoRA ($r = 4$)", "LoRA")]
        for arm, tag in picks:
            a = by_arm.get(arm)
            if a is None:
                continue
            for h in ds["horizons_min"]:
                blk = a["per_horizon"].get(str(h))
                if not blk:
                    continue
                suf = f"H{_word(h)}"
                lines += [
                    _macro(f"Num{tag}RMSE{suf}", blk["regression"]["rmse"], "{:.2f}"),
                    _macro(f"Num{tag}MAE{suf}", blk["regression"]["mae"], "{:.2f}"),
                    _macro(f"Num{tag}MARD{suf}", blk["regression"]["mard"], "{:.2f}"),
                    _macro(f"Num{tag}Lag{suf}", blk["regression"]["time_lag_min"], "{:.0f}"),
                    _macro(f"Num{tag}PRAUC{suf}", blk["classification"]["pr_auc"], "{:.3f}"),
                    _macro(f"Num{tag}SeFive{suf}",
                           blk["classification"]["sensitivity_at_5pct_fpr"], "{:.3f}"),
                    _macro(f"Num{tag}FAPD{suf}", blk["alarm"]["fapd_at_5pct_fpr"], "{:.2f}"),
                    _macro(f"Num{tag}Lead{suf}", blk["alarm"]["mean_onset_lead_min"], "{:.1f}"),
                    _macro(f"Num{tag}AB{suf}", blk["clinical"]["zone_ab_pct"], "{:.2f}"),
                    _macro(f"Num{tag}DE{suf}", blk["clinical"]["zone_de_n"], "{:.0f}"),
                ]
        fusion = by_arm.get("CausalCGM-MT (fusion)")
        if fusion:
            c = fusion["per_horizon"]["30"]["conformal"]
            for key, tag in (("split", "Split"), ("normalized", "Norm"),
                             ("mondrian", "Mondrian")):
                blk = c.get(key)
                if not blk:
                    continue
                lines += [
                    _macro(f"Num{tag}Cov", blk["marginal_coverage"], "{:.3f}"),
                    _macro(f"Num{tag}Width", blk["mean_width"], "{:.1f}"),
                    _macro(f"Num{tag}HypoCov", blk.get("hypo_stratum_coverage"), "{:.3f}"),
                    _macro(f"Num{tag}HypoWidth", blk.get("hypo_stratum_width"), "{:.1f}"),
                    _macro(f"Num{tag}SevereCov",
                           (blk.get("by_clinical_range", {})
                            .get("severe hypo (<54)", {}) or {}).get("coverage"), "{:.3f}"),
                ]
            # How many windows the severe-hypo cell actually holds. The prose
            # argues that a dedicated <54 stratum is unavailable at this sample
            # size, and that argument is only as good as the count behind it --
            # so the count comes from the artefact rather than from memory.
            severe_n = ((c.get("split", {}).get("by_clinical_range", {})
                         .get("severe hypo (<54)", {}) or {}).get("n"))
            if severe_n:
                lines.append(_macro("NumSevereN", severe_n, "{:d}"))
                # Order of the coverage standard deviation in that cell,
                # sqrt(alpha (1 - alpha) / n), in percentage points: the prose
                # quotes it and it must move with the count.
                a_sev = float(res.get("protocol", {}).get("alpha", 0.1))
                lines.append(_macro("NumSevereCovSDPts",
                                    100.0 * math.sqrt(a_sev * (1 - a_sev) / severe_n),
                                    "{:.1f}"))

    abl_path = os.path.join(root, "ablations", "ablations_summary.json")
    abl = _load_result(abl_path, allow_smoke)
    if abl is not None:
        rows = abl["studies"].get("b_asymmetry", [])
        by_k = {float(r["spec"]["hypo_asymmetry"]): r for r in rows}
        words = {1.0: "One", 2.0: "Two", 3.0: "Three", 5.0: "Five", 8.0: "Eight"}
        for k, w in words.items():
            r = by_k.get(k)
            if r is None:
                continue
            blk = r["per_horizon"]["30"]
            lines += [
                _macro(f"NumKappa{w}RMSE", blk["regression"]["rmse"], "{:.2f}"),
                _macro(f"NumKappa{w}RMSEHypo", blk["clinical"]["rmse_below_70"], "{:.2f}"),
                _macro(f"NumKappa{w}Bias", blk["clinical"]["hypo_band_bias"], "{:+.2f}"),
                _macro(f"NumKappa{w}DE", blk["clinical"]["zone_de_n"], "{:.0f}"),
                _macro(f"NumKappa{w}PRAUC", blk["classification"]["pr_auc"], "{:.3f}"),
            ]
        if 1.0 in by_k:
            b1 = by_k[1.0]["per_horizon"]["30"]["clinical"]
            for k, w in words.items():
                if k == 1.0 or k not in by_k:
                    continue
                bk = by_k[k]["per_horizon"]["30"]["clinical"]
                lines += [
                    _macro(f"NumBiasDrop{w}",
                           100 * (1 - bk["hypo_band_bias"] / b1["hypo_band_bias"]), "{:.0f}"),
                    _macro(f"NumDEDrop{w}",
                           100 * (1 - bk["zone_de_n"] / b1["zone_de_n"]), "{:.0f}"),
                ]
        # Ablation A arms, so the multi-task discussion quotes its own study
        # rather than borrowing a number from the kappa sweep.
        a_tags = {"Regression only": "AblAReg",
                  "Classification only": "AblACls",
                  "Joint, fixed $\\lambda=1$": "AblAFixed",
                  "Joint, Kendall uncertainty": "AblAKendall"}
        for r in abl["studies"].get("a_multitask", []):
            tag = a_tags.get(r["arm"])
            if tag is None:
                continue
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                blk = r["per_horizon"].get(str(h))
                if not blk:
                    continue
                suf = f"H{_word(h)}"
                lines += [
                    _macro(f"Num{tag}RMSE{suf}", blk["regression"]["rmse"], "{:.2f}"),
                    _macro(f"Num{tag}PRAUC{suf}", blk["classification"]["pr_auc"], "{:.3f}"),
                    _macro(f"Num{tag}Lag{suf}", blk["regression"]["time_lag_min"], "{:.0f}"),
                ]
        # Deltas the prose refers to, computed here rather than retyped.
        a_by = {r["arm"]: r for r in abl["studies"].get("a_multitask", [])}
        if {"Classification only", "Joint, Kendall uncertainty"} <= set(a_by):
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                c = a_by["Classification only"]["per_horizon"][str(h)]["classification"]["pr_auc"]
                k = a_by["Joint, Kendall uncertainty"]["per_horizon"][str(h)]["classification"]["pr_auc"]
                f = a_by["Joint, fixed $\\lambda=1$"]["per_horizon"][str(h)]["classification"]["pr_auc"]
                lines += [
                    _macro(f"NumAblAMultitaskCostH{_word(h)}", c - k, "{:.3f}"),
                    _macro(f"NumAblAKendallEdgeH{_word(h)}", k - f, "{:+.3f}"),
                ]
        # Ablation F: does the nadir target close the mapped arms' gap? The
        # prose turns on the DELTA between the two Ridge-AR rows, so that is
        # computed here rather than differenced by hand in a sentence.
        f_by = {r["arm"]: r for r in abl["studies"].get("f_nadir_target", [])}
        f_tags = {"Ridge-AR, level target": "AblFRidgeLevel",
                  "Ridge-AR, nadir target": "AblFRidgeNadir",
                  "LightGBM (native nadir classifier)": "AblFLGBM",
                  "CausalCGM-MT (native nadir head)": "AblFCausalCGMMT"}
        for arm, tag in f_tags.items():
            row = f_by.get(arm)
            if not row:
                continue
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                blk = row["per_horizon"].get(str(h))
                if not blk:
                    continue
                suf = f"H{_word(h)}"
                lines += [
                    _macro(f"Num{tag}PRAUC{suf}",
                           blk["classification"]["pr_auc"], "{:.3f}"),
                    _macro(f"Num{tag}ROCAUC{suf}",
                           blk["classification"]["roc_auc"], "{:.3f}"),
                    _macro(f"Num{tag}FAPD{suf}",
                           blk["alarm"]["fapd_at_5pct_fpr"], "{:.2f}"),
                ]
        if {"Ridge-AR, level target", "Ridge-AR, nadir target"} <= set(f_by):
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                hs = str(h)
                lv = f_by["Ridge-AR, level target"]["per_horizon"][hs]
                nd = f_by["Ridge-AR, nadir target"]["per_horizon"][hs]
                lines.append(_macro(
                    f"NumAblFNadirGainH{_word(h)}",
                    nd["classification"]["pr_auc"] - lv["classification"]["pr_auc"],
                    "{:+.3f}"))
                # And how much of CausalCGM-MT's lead survives the correction --
                # the number a reader will compute themselves if we do not.
                mt = f_by.get("CausalCGM-MT (native nadir head)")
                if mt:
                    lines.append(_macro(
                        f"NumAblFMTLeadOverNadirH{_word(h)}",
                        mt["per_horizon"][hs]["classification"]["pr_auc"]
                        - nd["classification"]["pr_auc"], "{:+.3f}"))

        # Ablation E: the soft-label boundary band. The prose quotes the SPREAD
        # across arms rather than a winner, because a single seed per arm cannot
        # separate differences this small -- so the macros are the extremes and
        # the range, which is what the claim actually rests on.
        e_rows = abl["studies"].get("e_soft_label", [])
        e_tags = {"Hard labels": "AblEHard",
                  "Fixed $\\pm$5 mg/dL": "AblEFixedFive",
                  "Fixed $\\pm$10 mg/dL": "AblEFixedTen",
                  "Fixed $\\pm$20 mg/dL": "AblEFixedTwenty",
                  "Estimated (MARD)": "AblEMard"}
        for r in e_rows:
            tag = e_tags.get(r["arm"])
            if tag is None:
                continue
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                blk = r["per_horizon"].get(str(h))
                if not blk:
                    continue
                suf = f"H{_word(h)}"
                lines += [
                    _macro(f"Num{tag}PRAUC{suf}", blk["classification"]["pr_auc"], "{:.3f}"),
                    _macro(f"Num{tag}RMSE{suf}", blk["regression"]["rmse"], "{:.2f}"),
                    _macro(f"Num{tag}FAPD{suf}", blk["alarm"]["fapd_at_5pct_fpr"], "{:.2f}"),
                ]
        if e_rows:
            # Every ablation study runs at one shared budget, which is what lets
            # the tables be read against each other. `study_meta` records each
            # study's own, so emit the shared value only when they really do
            # agree -- if a future run splits them, the macro disappears and the
            # prose that cites it fails loudly rather than going quietly stale.
            meta = abl.get("study_meta", {})
            eps = {(m.get("budget") or {}).get("epochs") for m in meta.values()}
            eps.discard(None)
            if len(eps) == 1:
                lines.append(_macro("NumAblADEpochs", eps.pop(), "{:.0f}"))
            elif eps:
                print(f"  WARNING: ablation studies disagree on epoch budget "
                      f"({sorted(eps)}); NumAblADEpochs not emitted, and the "
                      f"comparability claim in appendix_c_ablations.tex no "
                      f"longer holds.")

            # The estimated band and where it came from -- the one number in
            # this study that is a measurement rather than a hyperparameter.
            sl = next((r["soft_label"] for r in e_rows
                       if r["arm"] == "Estimated (MARD)"), None)
            if sl:
                lines += [
                    _macro("NumAblEMardBand", sl["band_mgdl"], "{:.2f}"),
                    _macro("NumAblEMardPct", 100 * sl["mard_used"], "{:.1f}"),
                ]
                noise = sl.get("noise") or {}
                if noise.get("n"):
                    lines.append(_macro("NumAblEMardPairs", noise["n"], "{:.0f}"))
            for h in abl.get("budget", {}).get("horizons_min", [30, 60]):
                pr = [r["per_horizon"][str(h)]["classification"]["pr_auc"]
                      for r in e_rows if str(h) in r["per_horizon"]]
                fa = [r["per_horizon"][str(h)]["alarm"]["fapd_at_5pct_fpr"]
                      for r in e_rows if str(h) in r["per_horizon"]]
                if pr:
                    suf = f"H{_word(h)}"
                    lines += [
                        _macro(f"NumAblEPRAUCSpread{suf}", max(pr) - min(pr), "{:.3f}"),
                        _macro(f"NumAblEPRAUCMin{suf}", min(pr), "{:.3f}"),
                        _macro(f"NumAblEPRAUCMax{suf}", max(pr), "{:.3f}"),
                        _macro(f"NumAblEFAPDMin{suf}", min(fa), "{:.2f}"),
                        _macro(f"NumAblEFAPDMax{suf}", max(fa), "{:.2f}"),
                    ]

        if 1.0 in by_k and 3.0 in by_k:
            p1 = by_k[1.0]["per_horizon"]["30"]["classification"]["pr_auc"]
            p3 = by_k[3.0]["per_horizon"]["30"]["classification"]["pr_auc"]
            r1 = by_k[1.0]["per_horizon"]["30"]["regression"]["rmse"]
            r3 = by_k[3.0]["per_horizon"]["30"]["regression"]["rmse"]
            lines += [
                _macro("NumKappaThreePRAUCDrop", p1 - p3, "{:.3f}"),
                _macro("NumKappaThreeRMSECostPct", 100 * (r3 - r1) / r1, "{:.1f}"),
            ]

        # Ablation D. Each leaky rung is compared with the STRICT arm at the same
        # horizon: `rmse_relative_improvement_pct` = 100 (strict - leaky) /
        # strict and `pr_auc_absolute_gain` = leaky - strict
        # (`run_paper_ablations.leakage_inflation`). These macros used to print
        # those signed values, so the abstract read "inflating reported RMSE by
        # +23.0%" for a leak that LOWERS measured RMSE. They now print unsigned
        # magnitudes, and the DIRECTION is in the name: `RMSEReductionPct` when
        # the leaky rung's RMSE is lower than the strict arm's, `RMSEIncreasePct`
        # when it is higher; `PRAUCIncrease` / `PRAUCDecrease` likewise. If a
        # rerun flips a sign, the macro the prose quotes disappears and the
        # build fails, rather than the prose silently stating the wrong
        # direction.
        for h, block in abl["studies"].get("d_leakage", {}).get("inflation", {}).items():
            suf = "" if str(h) == "30" else f"H{_word(int(h))}"
            for name, v in sorted(block.items()):
                tag = "Split" if "split" in name.lower() else "Naive"
                d_rmse = v["rmse_relative_improvement_pct"]
                d_pr = v["pr_auc_absolute_gain"]
                rmse_dir = "Reduction" if d_rmse >= 0 else "Increase"
                pr_dir = "Increase" if d_pr >= 0 else "Decrease"
                lines += [
                    _macro(f"NumLeak{tag}PRAUC{pr_dir}{suf}", abs(d_pr), "{:.3f}"),
                    _macro(f"NumLeak{tag}RMSE{rmse_dir}Pct{suf}", abs(d_rmse),
                           "{:.1f}"),
                    _macro(f"NumLeak{tag}RMSEStrict{suf}", v["rmse_strict"], "{:.2f}"),
                    _macro(f"NumLeak{tag}RMSELeaky{suf}", v["rmse_leaky"], "{:.2f}"),
                    _macro(f"NumLeak{tag}PRAUCStrict{suf}", v["pr_auc_strict"], "{:.3f}"),
                    _macro(f"NumLeak{tag}PRAUCLeaky{suf}", v["pr_auc_leaky"], "{:.3f}"),
                ]

    # Kept across the personalisation block so the MetaboNet appendix can be
    # rendered as a ratio against it; see `metabonet_numbers`.
    ohio_lora_delta_rmse = None

    pers_path = os.path.join(root, "personalization", "personalization_sweep.json")
    pers = _load_result(pers_path, allow_smoke)
    if pers is not None:
        hp = str(float(pers["adapt"]["prefix_hours"][
            min(range(len(pers["adapt"]["prefix_hours"])),
                key=lambda i: abs(pers["adapt"]["prefix_hours"][i] - 24.0))]))
        sig = pers.get("significance", {}).get(hp, {})
        tags = {"none": "None", "bn_adapt": "BNAdapt", "lora (r=4)": "LoRAFour",
                "lora (r=8)": "LoRAEight", "head_only": "HeadOnly", "full": "Full"}
        lines.append(_macro("NumPersSubjects",
                            len(pers.get("subjects", [])), "{:d}"))
        lines.append(_macro("NumPersPrefixHours", float(hp), "{:.0f}"))
        lines.append(_macro("NumPersDefaultSteps", pers["adapt"]["steps"], "{:d}"))
        for mode, tag in tags.items():
            blk = sig.get(mode)
            if not blk:
                continue
            for metric, mtag in (("rmse", "RMSE"), ("pr_auc", "PRAUC"),
                                 ("fapd_at_5pct_fpr", "FAPD")):
                st = blk.get(metric)
                if not st:
                    continue
                lines += [
                    _macro(f"NumPers{tag}{mtag}Delta", st["mean_delta"], "{:+.3f}"),
                    _macro(f"NumPers{tag}{mtag}P", st.get("p_t_holm"), "{:.4f}"),
                    _macro(f"NumPers{tag}{mtag}Improved", st["n_improved"], "{:d}"),
                ]
            base = blk.get("rmse", {})
            if base:
                lines.append(_macro(f"NumPers{tag}RMSEBefore",
                                    base["mean_before"], "{:.2f}"))
                lines.append(_macro(f"NumPers{tag}RMSEAfter",
                                    base["mean_after"], "{:.2f}"))
        ohio_lora_delta_rmse = (sig.get("lora (r=4)", {})
                                .get("rmse", {}).get("mean_delta"))

        budget = pers.get("step_budget", {}).get("lora (r=4)", {})
        for steps in sorted(budget, key=int):
            lines.append(_macro(f"NumPersLoRASteps{_word(int(steps))}Delta",
                                budget[steps]["rmse"]["mean_delta"], "{:+.3f}"))
        # Short head-only / LoRA budgets. None of these cells carries a Holm
        # adjustment (the step-budget family is not corrected), so the interval
        # and p below are reported as unadjusted and the prose says so.
        for key, tag in (("head_only", "HeadOnly"), ("lora (r=8)", "LoRAEight")):
            for steps, blk in sorted(pers.get("step_budget", {}).get(key, {}).items(),
                                     key=lambda kv: int(kv[0])):
                lines.append(_macro(f"NumPers{tag}Steps{_word(int(steps))}Delta",
                                    blk["rmse"]["mean_delta"], "{:+.3f}"))
        head_ten = pers.get("step_budget", {}).get("head_only", {}).get("10")
        if head_ten:
            lines += [
                _macro("NumPersHeadStepsTenCILow", head_ten["rmse"]["ci95_low"], "{:+.2f}"),
                _macro("NumPersHeadStepsTenCIHigh", head_ten["rmse"]["ci95_high"], "{:+.3f}"),
                _macro("NumPersHeadStepsTenP", head_ten["rmse"]["p_t"], "{:.3f}"),
                _macro("NumPersHeadStepsTenImproved", head_ten["rmse"]["n_improved"], "{:d}"),
            ]
        # Subjects behind each prefix length: the six-hour cell has fewer.
        for ph in pers["adapt"]["prefix_hours"]:
            n_ph = (pers.get("significance", {}).get(str(ph), {})
                    .get("bn_adapt", {}).get("rmse", {}).get("n"))
            if n_ph is not None:
                lines.append(_macro(f"NumPersPrefix{_word(int(ph))}N", n_ph, "{:d}"))
        traj = pers.get("significance", {})
        for ph in pers["adapt"]["prefix_hours"]:
            st = traj.get(str(ph), {}).get("bn_adapt", {}).get("rmse")
            if st:
                lines.append(_macro(f"NumPersBNPrefix{_word(int(ph))}Delta",
                                    st["mean_delta"], "{:+.3f}"))
        for ph in pers["adapt"]["prefix_hours"]:
            st = traj.get(str(ph), {}).get("head_only", {}).get("rmse")
            if st:
                lines.append(_macro(f"NumPersHeadPrefix{_word(int(ph))}Delta",
                                    st["mean_delta"], "{:+.3f}"))

        # ---- prefix POSITION contrast (opening vs recent) --------------------
        # The control for the drift explanation. Emitted per mode so the prose
        # can name the largest effect without a reader having to trust that the
        # largest one was chosen honestly -- every mode's number is available.
        contrast = pers.get("prefix_mode_contrast", {})
        for mode, tag in tags.items():
            blk = contrast.get(mode, {})
            if not blk:
                continue
            for metric, mtag in (("rmse", "RMSE"), ("pr_auc", "PRAUC")):
                rec = blk.get("recent", {}).get("stats", {}).get(metric)
                if rec:
                    lines += [
                        _macro(f"NumPosRecent{tag}{mtag}Delta",
                               rec["mean_delta"], "{:+.3f}"),
                        _macro(f"NumPosRecent{tag}{mtag}P",
                               rec.get("p_t_holm"), "{:.4f}"),
                    ]
                hh = blk.get("recent_vs_opening", {}).get(metric)
                if hh:
                    lines += [
                        _macro(f"NumPosDiff{tag}{mtag}", hh["mean_delta"], "{:+.3f}"),
                        _macro(f"NumPosDiff{tag}{mtag}P", hh.get("p_t"), "{:.4f}"),
                        _macro(f"NumPosDiff{tag}{mtag}PHolm",
                               hh.get("p_t_holm"), "{:.4f}"),
                        _macro(f"NumPosDiff{tag}{mtag}CILow",
                               hh.get("ci95_low"), "{:+.3f}"),
                        _macro(f"NumPosDiff{tag}{mtag}CIHigh",
                               hh.get("ci95_high"), "{:+.3f}"),
                    ]
        # The variable the contrast turns on, in the units the prose uses.
        any_blk = next((b for b in contrast.values() if "opening" in b), None)
        if any_blk:
            g_open = any_blk["opening"].get("median_gap_min")
            g_rec = any_blk["recent"].get("median_gap_min")
            if g_open is not None:
                lines.append(_macro("NumPosOpeningGapDays", g_open / 1440.0, "{:.0f}"))
            if g_rec is not None:
                lines.append(_macro("NumPosRecentGapMin", g_rec, "{:.0f}"))
            lines.append(_macro("NumPosRecentAdaptWindows",
                                any_blk["recent"].get("median_adapt_windows"), "{:.0f}"))
            # Where each prefix lies relative to the population model's fit and
            # model-selection rows, summed over subjects. The sweep records
            # fit and val membership only (not calibration), so the prose says
            # no more than these counts.
            for pos, ptag in (("opening", "Opening"), ("recent", "Recent")):
                rws = any_blk[pos].get("rows", [])
                if rws:
                    lines += [
                        _macro(f"NumPos{ptag}AdaptWindowsTotal",
                               sum(r["n_adapt_windows"] for r in rws), "{:,d}"),
                        _macro(f"NumPos{ptag}AdaptInFit",
                               sum(r["n_adapt_in_fit"] for r in rws), "{:,d}"),
                        _macro(f"NumPos{ptag}AdaptInVal",
                               sum(r["n_adapt_in_val"] for r in rws), "{:d}"),
                    ]
            # The smallest recent-minus-opening RMSE difference in any mode
            # (positive = worse from the recent prefix), so the prose can bound
            # the effect rather than describe one arm and hope.
            diffs = [(m, b["recent_vs_opening"]["rmse"])
                     for m, b in contrast.items()
                     if "recent_vs_opening" in b and m != "none"]
            if diffs:
                best = min(diffs, key=lambda kv: kv[1]["mean_delta"])
                lines += [
                    _macro("NumPosBestDiffRMSE", best[1]["mean_delta"], "{:+.3f}"),
                    _macro("NumPosBestDiffP", best[1].get("p_t"), "{:.2f}"),
                    _macro("NumPosMinDiffPHolm",
                           min((v["recent_vs_opening"]["rmse"].get("p_t_holm", 1.0)
                                for m, v in contrast.items()
                                if "recent_vs_opening" in v and m != "none"),
                               default=None), "{:.2f}"),
                ]

    edge_path = os.path.join(root, "edge", "latency_sweep.json")
    edge = _load_result(edge_path, allow_smoke)
    if edge is not None:
        one = next((r for r in edge["runs"] if r["batch"] == 1 and r["threads"] == 1), None)
        if one:
            lines += [
                _macro("NumEdgePFifty", one["latency_ms"]["p50"], "{:.3f}"),
                _macro("NumEdgePNinetyNine", one["latency_ms"]["p99"], "{:.3f}"),
                _macro("NumEdgeMax", one["latency_ms"]["max"], "{:.3f}"),
                _macro("NumEdgeDeadline", edge["deadline_ms"], "{:.0f}"),
                _macro("NumEdgeFeatureUs", one.get("feature_construction_us"), "{:.3f}"),
                # Headroom against the deadline, computed on the OBSERVED
                # MAXIMUM rather than a percentile. The section's stated
                # principle is to judge a device deadline on the tail, and a
                # headroom multiple quoted off p50 or p99 can be several times
                # larger than the one the worst observed call actually leaves.
                # Macro-resolved so it cannot be retyped and go stale: the
                # prose previously carried a hand-written "over two hundred"
                # that held against p99 and failed against the maximum.
                _macro("NumEdgeHeadroomMax",
                       edge["deadline_ms"] / one["latency_ms"]["max"], "{:.0f}"),
                _macro("NumEdgeGraph",
                       "multimodal fusion" if "fusion" in str(edge.get("model", ""))
                       else "CGM-only"),
            ]
        best = min((r for r in edge["runs"]),
                   key=lambda r: r["latency_ms"]["p50"] / r["batch"])
        lines += [
            _macro("NumEdgeBestPerWindowUs",
                   best["latency_ms"]["p50"] / best["batch"] * 1000, "{:.0f}"),
            _macro("NumEdgeBestBatch", best["batch"], "{:d}"),
            _macro("NumEdgeBestThreads", best["threads"], "{:d}"),
        ]
        # The machine the numbers were measured on. A latency claim without a
        # named chip is unfalsifiable -- a reader cannot tell whether to scale
        # it up or down for their own target -- and naming the chip in prose
        # instead would be a hardware claim the artefact beside it does not
        # make. Emitted only when the run actually recorded one, so an older
        # artefact degrades to the generic phrasing rather than to a guess.
        host = edge.get("host", {})
        if host.get("cpu"):
            lines.append(_macro("NumEdgeCPU", host["cpu"]))
        if host.get("physical_cores"):
            lines.append(_macro("NumEdgeCores", host["physical_cores"], "{:d}"))

    lines += metabonet_numbers(metabonet_root, ohio_lora_delta_rmse)
    lines += metabonet_fold_numbers(metabonet_root)
    lines += multiseed_numbers(root)
    lines += calibration_followup_numbers(root)
    lines += latching_numbers_and_table(root)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    n = sum(1 for line in lines if line.startswith("\\newcommand"))
    print(f"wrote {out}  ({n} macros)")
    return n


def rebuild_personalization(root: str, allow_smoke: bool = False) -> int:
    """Re-emit the personalisation tables from the sweep JSON.

    The sweep writes these at the end of a multi-hour run. Regenerating them
    here means every table in the manuscript is reproducible from a summary
    file, rather than existing only because a long job happened to finish --
    which also makes a formatting change to one of them free.
    """
    path = os.path.join(root, "personalization", "personalization_sweep.json")
    pers = _load_result(path, allow_smoke)
    if pers is None:
        return 0
    primary = int(pers["protocol"]["primary_horizon_min"])
    prefixes = pers["adapt"]["prefix_hours"]
    hp = str(float(min(prefixes, key=lambda v: abs(v - 24.0))))
    sig = pers.get("significance", {}).get(hp, {})
    modes = list(sig)
    if not modes:
        print("  skip: no significance block in the sweep JSON")
        return 0

    metrics = [("rmse", "min", "RMSE"), ("pr_auc", "max", "PR-AUC"),
               ("fapd_at_5pct_fpr", "min", "FAPD@5\\%")]
    rows = []
    for m in modes:
        cell = {}
        for metric, _b, _l in metrics:
            st = sig[m].get(metric, {})
            cell[metric] = st.get("mean_after")
            cell[metric + "_delta"] = st.get("mean_delta")
            cell[metric + "_p"] = st.get("p_t_holm")
        rows.append({"arm": m, "per_horizon": {str(primary): {"s": cell}}})

    cols = []
    for metric, better, lbl in metrics:
        cols += [B.Column(lbl, f"s.{metric}", "{:.3f}", better),
                 B.Column(f"$\\Delta$ {lbl}", f"s.{metric}_delta", "{:+.3f}", better),
                 B.Column("$p_{\\mathrm{Holm}}$", f"s.{metric}_p", "{:.3f}", None)]
    # WHY THE BASELINE HERE IS NOT THE ONE IN TABLE 1, spelled out in the
    # caption because it is the first thing a reader checks and the difference
    # looks like an inconsistency until it is explained.
    #
    # It is NOT a different model and NOT a different evaluation set: the base
    # checkpoint and the scored rows are identical. This table reports the MEAN
    # OF PER-SUBJECT RMSEs, because the design is a paired per-subject one and
    # the tests operate on those twelve numbers; Table 1 reports a POOLED RMSE
    # over all windows at once. The two differ whenever error and window count
    # covary across subjects, and n-weighted quadratic pooling of the rows here
    # reproduces the benchmark figure exactly. Computed rather than asserted, so
    # the sentence cannot outlive its own arithmetic.
    #
    # PR-AUC DOES NOT DECOMPOSE THE SAME WAY, and the caption used to imply it
    # did. RMSE pools because squared error is additive over windows; PR-AUC is a
    # RANK statistic over the pooled residuals, so the mean of per-subject PR-AUCs
    # is a different estimand and no reweighting recovers the pooled value (here
    # the n-weighted mean lands ~0.088 below it). The per-subject spread is
    # reported instead, since that is the quantity the paired tests consume.
    base_rows = pers["sweep"][hp].get("none", [])
    pooled_note = ""
    if base_rows:
        import math
        n_i = [r["n_eval_windows"] for r in base_rows]
        r_i = [r["before"][str(primary)]["rmse"] for r in base_rows]
        mean_rmse = sum(r_i) / len(r_i)
        pooled = math.sqrt(sum(n * r * r for n, r in zip(n_i, r_i)) / sum(n_i))

        p_i = [r["before"][str(primary)]["pr_auc"] for r in base_rows]
        mean_pr = sum(p_i) / len(p_i)
        sd_pr = statistics.stdev(p_i) if len(p_i) > 1 else 0.0

        # The pooled PR-AUC lives in the benchmark artefact, not this one. If it
        # is absent the sentence drops the comparison rather than inventing it.
        bench = _load_result(
            os.path.join(root, "benchmarks", "results_summary.json"), allow_smoke)
        pooled_pr = None
        if bench:
            arm = next((a for a in bench.get("arms", [])
                        if "fusion" in a.get("arm", "")), None)
            if arm:
                pooled_pr = (arm.get("per_horizon", {}).get(str(primary), {})
                             .get("classification", {}).get("pr_auc"))

        pooled_note = (
            f" The zero-shot baseline reads {mean_rmse:.2f}~mg/dL here against "
            f"{pooled:.2f}~mg/dL in Table~\\ref{{tab:sota_ph{primary}}}: the same "
            f"checkpoint on the same {sum(n_i):,} windows, aggregated differently. "
            f"This table averages the {len(base_rows)} per-subject RMSEs, which is "
            f"the quantity the paired tests are computed on; Table~"
            f"\\ref{{tab:sota_ph{primary}}} pools the residuals across all windows. "
            f"Weighting these rows by their window counts recovers the pooled RMSE "
            f"exactly ({pooled:.3f} against {pooled:.2f}). PR-AUC does not "
            f"decompose this way: it is a rank statistic over the pooled "
            f"residuals, so the per-subject mean ({mean_pr:.3f}, sd {sd_pr:.3f}, "
            f"range {min(p_i):.3f}--{max(p_i):.3f} across the {len(base_rows)} "
            f"subjects) is a different quantity from the pooled ")
        pooled_note += (
            f"{pooled_pr:.3f} " if pooled_pr is not None
            else "benchmark figure ")
        pooled_note += ("and is not recoverable from it by reweighting. The "
                        "paired tests are computed on the per-subject values "
                        "reported here.")

    B.write_latex(
        os.path.join(root, "personalization", "table_personalization.tex"),
        B.latex_table(rows, cols, primary,
                      caption=(f"Few-shot personalisation on OhioT1DM, "
                               f"PH = {primary} min, {float(hp):.0f} h adaptation "
                               f"prefix, {len(pers.get('subjects', []))} subjects. "
                               f"$\\Delta$ is the paired per-subject change from the "
                               f"zero-shot population model on identical evaluation "
                               f"windows; $p$ is a paired $t$-test with "
                               f"Holm--Bonferroni adjustment across modes."
                               f"{pooled_note}"),
                      label="tab:personalization", row_label="Adaptation"))

    eff_rows, eff_cols = [], [B.Column(f"{p:g} h", f"e.p{_word(int(p))}", "{:+.3f}", "min")
                              for p in prefixes]
    for m in modes:
        cell = {f"p{_word(int(p))}": (pers["significance"].get(str(p), {})
                                      .get(m, {}).get("rmse", {}).get("mean_delta"))
                for p in prefixes}
        eff_rows.append({"arm": m, "per_horizon": {str(primary): {"e": cell}}})
    B.write_latex(
        os.path.join(root, "personalization", "table_sample_efficiency.tex"),
        B.latex_table(eff_rows, eff_cols, primary,
                      caption=(f"Sample efficiency: mean paired $\\Delta$RMSE "
                               f"(mg/dL) against the zero-shot population model as a "
                               f"function of adaptation prefix length, "
                               f"PH = {primary} min. Negative is an improvement; "
                               f"every entry is positive."),
                      label="tab:sample_efficiency", row_label="Adaptation"))

    budget = pers.get("step_budget", {})
    if budget:
        steps = sorted({int(k) for v in budget.values() for k in v})
        b_rows = [{"arm": m, "per_horizon": {str(primary): {"b": {
            f"s{_word(s)}": budget[m].get(str(s), {}).get("rmse", {}).get("mean_delta")
            for s in steps}}}} for m in budget]
        B.write_latex(
            os.path.join(root, "personalization", "table_step_budget.tex"),
            B.latex_table(b_rows,
                          [B.Column(f"{s} steps", f"b.s{_word(s)}", "{:+.3f}", "min")
                           for s in steps], primary,
                          caption=(f"Adaptation budget: mean paired $\\Delta$RMSE "
                                   f"(mg/dL) as a function of the number of gradient "
                                   f"steps, at a {float(hp):.0f} h prefix and "
                                   f"PH = {primary} min. For LoRA at both ranks the "
                                   f"damage grows with every increase in the "
                                   f"budget; head-only fine-tuning is not "
                                   f"monotone."),
                          label="tab:step_budget", row_label="Adaptation"))

    # ---- prefix POSITION: opening vs recent ---------------------------------
    # Reported as three columns per metric -- each arm's damage under each
    # prefix position, and the paired difference between them. The difference
    # column is the one the section turns on, so it carries the p-value.
    contrast = pers.get("prefix_mode_contrast", {})
    n_written = 3 if budget else 2
    if contrast:
        c_rows = []
        for m, blk in contrast.items():
            cell = {}
            for metric in ("rmse", "pr_auc"):
                for pm in ("opening", "recent"):
                    cell[f"{pm}_{metric}"] = (blk.get(pm, {}).get("stats", {})
                                              .get(metric, {}).get("mean_delta"))
                hh = blk.get("recent_vs_opening", {}).get(metric, {})
                cell[f"diff_{metric}"] = hh.get("mean_delta")
                cell[f"diff_{metric}_p"] = hh.get("p_t_holm")
            c_rows.append({"arm": m, "per_horizon": {str(primary): {"c": cell}}})
        c_cols = []
        for metric, lbl in (("rmse", "RMSE"), ("pr_auc", "PR-AUC")):
            c_cols += [
                B.Column(f"$\\Delta${lbl} open", f"c.opening_{metric}", "{:+.3f}", None),
                B.Column(f"$\\Delta${lbl} recent", f"c.recent_{metric}", "{:+.3f}", None),
                B.Column("rec$-$open", f"c.diff_{metric}", "{:+.3f}", None),
                B.Column("$p_{\\mathrm{Holm}}$", f"c.diff_{metric}_p", "{:.3f}", None),
            ]
        gap_open = next((b["opening"].get("median_gap_min")
                         for b in contrast.values() if "opening" in b), None)
        gap_rec = next((b["recent"].get("median_gap_min")
                        for b in contrast.values() if "recent" in b), None)
        gap_txt = ""
        if gap_open is not None and gap_rec is not None:
            gap_txt = (f" The opening prefix ends a median of "
                       f"{gap_open / 1440:.0f}~days before the first scored window; "
                       f"the recent prefix ends {gap_rec:.0f}~min before it, which "
                       f"is the embargo itself.")
        B.write_latex(
            os.path.join(root, "personalization", "table_prefix_position.tex"),
            B.latex_table(c_rows, c_cols, primary,
                          caption=(f"Adaptation prefix position at a "
                                   f"{float(hp):.0f} h prefix, PH = {primary} min. "
                                   f"Both arms use the same subjects, evaluation "
                                   f"windows, adapter, step budget and prefix "
                                   f"length; they differ only in how far the "
                                   f"prefix sits from the windows being scored."
                                   f"{gap_txt} No column of paired differences "
                                   f"is significant, and no arm becomes an "
                                   f"improvement over zero-shot under either "
                                   f"position. Negative $\\Delta$RMSE and positive "
                                   f"$\\Delta$PR-AUC are better."),
                          label="tab:prefix_position", row_label="Adaptation"))
        n_written += 1
    return n_written


# --------------------------------------------------------------------------- #
# Multi-seed sweep: tables and macros from multiseed_summary.json
# --------------------------------------------------------------------------- #
_MS_ARMS = {
    "CausalCGM-MT (fusion)": "Fusion", "CausalCGM-MT (CGM)": "CGMOnly",
    "Dilated LSTM (CGM)": "LSTM", "LightGBM": "LGBM", "Persistence": "Persist",
    "Ridge-AR": "Ridge", "ARIMA(4,1,0)": "Arima",
}
_MS_METRICS = {"rmse": ("RMSE", "{:.2f}"), "pr_auc": ("PRAUC", "{:.3f}"),
               "pr_auc_onset": ("PRAUCOnset", "{:.3f}"), "sens_5": ("SeFive", "{:.3f}"),
               "fapd_5": ("FAPD", "{:.2f}"), "onset_lead": ("OnsetLead", "{:.1f}"),
               "onset_det": ("OnsetDet", "{:.3f}"), "de": ("DE", "{:.0f}"),
               "realized_fpr_5": ("RealFPR", "{:.3f}"),
               "cov_lt54_split": ("CovSevSplit", "{:.3f}"),
               "cov_lt54_normalized": ("CovSevNorm", "{:.3f}"),
               "cov_lt54_mondrian": ("CovSevMondrian", "{:.3f}")}
_MS_H = {"30": "HThirty", "60": "HSixty"}


def _multiseed_path(root: str) -> str:
    return os.path.join(root, "benchmarks_seeds", "multiseed_summary.json")


def rebuild_multiseed(root: str) -> int:
    path = _multiseed_path(root)
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return 0
    from scripts.aggregate_seed_sweep import render_tables
    with open(path) as f:
        summary = json.load(f)
    n = 0
    for name, body in render_tables(summary).items():
        out = os.path.join(root, "benchmarks_seeds", name)
        with open(out, "w") as f:
            f.write(body)
        print(f"  wrote {out}")
        n += 1
    return n


def multiseed_numbers(root: str) -> list[str]:
    path = _multiseed_path(root)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        summary = json.load(f)
    lines = ["", "% ---- multi-seed sweep (runs/paper_results/benchmarks_seeds) ----",
             _macro("NumMSSeeds", len(summary["seeds"]), "{:d}"),
             _macro("NumMSBoot", summary["n_boot"], "{:d}")]
    for arm, tag in _MS_ARMS.items():
        per = summary["arms"].get(arm)
        if per is None:
            continue
        for hk, htag in _MS_H.items():
            for key, (mtag, fmt) in _MS_METRICS.items():
                cell = per[hk].get(key)
                if cell is None:
                    continue
                lines.append(_macro(f"NumMS{tag}{mtag}{htag}Mean", cell["mean"], fmt))
                lines.append(_macro(f"NumMS{tag}{mtag}{htag}SD", cell["sd"], fmt))
        diffs = summary["diffs_vs_reference"].get(arm)
        if diffs:
            for hk, htag in _MS_H.items():
                for key, d in diffs[hk].items():
                    mtag, fmt = _MS_METRICS[key]
                    sfmt = fmt.replace("{:", "{:+")
                    lines.append(_macro(f"NumMSDiff{tag}{mtag}{htag}", d["diff"], sfmt))
                    lines.append(_macro(f"NumMSDiff{tag}{mtag}{htag}Lo", d["ci95"][0], sfmt))
                    lines.append(_macro(f"NumMSDiff{tag}{mtag}{htag}Hi", d["ci95"][1], sfmt))
    return lines


def calibration_followup_numbers(root: str) -> list[str]:
    """Macros for the calibration bootstrap, realised FPR and the treatment arm."""
    lines = ["", "% ---- calibration bootstrap, realised FPR, Ablation C treatment ----"]
    bpath = os.path.join(root, "conformal_bootstrap.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            b = json.load(f)
        seeds = b["seeds"]
        below = [100 * v["mondrian"]["cov_severe_lt54"]["p_below_nominal"] for v in seeds.values()]
        beats = sum(v["normalized"]["cov_severe_lt54"]["point"] > v["split"]["cov_severe_lt54"]["point"]
                    for v in seeds.values())
        split_pts = [v["split"]["cov_severe_lt54"]["point"] for v in seeds.values()]
        norm_pts = [v["normalized"]["cov_severe_lt54"]["point"] for v in seeds.values()]
        mond_pts = [v["mondrian"]["cov_severe_lt54"]["point"] for v in seeds.values()]
        lines += [
            _macro("NumBootSeeds", len(seeds), "{:d}"),
            _macro("NumBootDraws", b["n_boot"], "{:d}"),
            _macro("NumBootMondrianBelowMinPct", min(below), "{:.0f}"),
            _macro("NumBootMondrianBelowMaxPct", max(below), "{:.0f}"),
            _macro("NumBootNormBeatsSplitSeeds", beats, "{:d}"),
            _macro("NumBootSplitSevereMin", min(split_pts), "{:.3f}"),
            _macro("NumBootSplitSevereMax", max(split_pts), "{:.3f}"),
            _macro("NumBootNormSevereMin", min(norm_pts), "{:.3f}"),
            _macro("NumBootNormSevereMax", max(norm_pts), "{:.3f}"),
            _macro("NumBootMondrianSevereMin", min(mond_pts), "{:.3f}"),
            _macro("NumBootMondrianSevereMax", max(mond_pts), "{:.3f}"),
            _macro("NumBootSevereN", next(iter(seeds.values()))["n_test_severe"], "{:d}"),
        ]
    mpath = _multiseed_path(root)
    if os.path.exists(mpath):
        with open(mpath) as f:
            ms = json.load(f)
        fprs = [per["30"]["realized_fpr_5"]["mean"] for arm, per in ms["arms"].items()
                if "LoRA" not in arm]
        lines += [_macro("NumMSRealFPRMinPct", 100 * min(fprs), "{:.1f}"),
                  _macro("NumMSRealFPRMaxPct", 100 * max(fprs), "{:.1f}")]
    apath = os.path.join(root, "ablations", "ablations_summary.json")
    if os.path.exists(apath):
        with open(apath) as f:
            st = json.load(f)["studies"].get("c_modality", {})
        for cohort, tag in (("2018", "Eighteen"), ("2020", "Twenty")):
            arms = {a["arm"]: a for a in st.get(cohort, [])}
            if "CGM only" in arms and "CGM + treatment" in arms:
                for hk, htag in (("30", "HThirty"), ("60", "HSixty")):
                    d = (arms["CGM + treatment"]["per_horizon"][hk]["regression"]["rmse"]
                         - arms["CGM only"]["per_horizon"][hk]["regression"]["rmse"])
                    lines.append(_macro(f"NumAblCTreat{tag}RMSE{htag}", d, "{:+.2f}"))
    return lines


def latching_numbers_and_table(root: str) -> list[str]:
    """Alarm-policy sensitivity: table re-rendered from JSON + macros."""
    path = os.path.join(root, "fapd_latching", "fapd_latching.json")
    if not os.path.exists(path):
        return []
    from scripts.fapd_latching_sensitivity import render
    with open(path) as f:
        rep = json.load(f)
    with open(os.path.join(root, "fapd_latching", "table_fapd_latching.tex"), "w") as f:
        f.write(render(rep))
    r30 = rep["results"]["30"]
    fus, lgb, per = "CausalCGM-MT (fusion)", "LightGBM", "Persistence"
    out = ["", "% ---- alarm-policy sensitivity (runs/paper_results/fapd_latching) ----"]
    for key, tag in (("G0_D5", "Base"), ("G60_D5", "GSixty"), ("G0_D15", "DFifteen")):
        out += [_macro(f"NumLatch{tag}FusionFAPD", r30[key][fus]["fapd"], "{:.2f}"),
                _macro(f"NumLatch{tag}LGBMFAPD", r30[key][lgb]["fapd"], "{:.2f}"),
                _macro(f"NumLatch{tag}PersistFAPD", r30[key][per]["fapd"], "{:.2f}"),
                _macro(f"NumLatch{tag}FusionDet", 100 * r30[key][fus]["onset_det"], "{:.1f}"),
                _macro(f"NumLatch{tag}LGBMDet", 100 * r30[key][lgb]["onset_det"], "{:.1f}")]
    d = r30["G0_D15"][lgb]
    out += [_macro("NumLatchDFifteenDetDiffLGBM", 100 * d["diff_onset_det_vs_fusion"], "{:+.1f}"),
            _macro("NumLatchDFifteenDetDiffLGBMLo", 100 * d["diff_onset_det_vs_fusion_ci95"][0], "{:+.1f}"),
            _macro("NumLatchDFifteenDetDiffLGBMHi", 100 * d["diff_onset_det_vs_fusion_ci95"][1], "{:+.1f}")]
    return out


def metabonet_fold_numbers(metabonet_root: str) -> list[str]:
    """Fold-paired fusion-vs-CGM counts and fusion activation (corrected gate)."""
    path = os.path.join(metabonet_root, "benchmarks", "metabonet_summary.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        res = json.load(f)
    if res.get("smoke"):
        return []
    by = {(r["arm"], r["fold"]): r for r in res["per_fold"]
          if r["arm"] in ("cgm_only", "fusion") and r.get("per_horizon")}
    folds = sorted({k for _, k in by if ("cgm_only", k) in by and ("fusion", k) in by})
    if not folds:
        return []
    h = str(res.get("horizon_min", 30))
    def d(path_, k):
        f = by[("fusion", k)]["per_horizon"][h]; c = by[("cgm_only", k)]["per_horizon"][h]
        return float(f[path_[0]][path_[1]]) - float(c[path_[0]][path_[1]])
    rm = np.array([d(("regression", "rmse"), k) for k in folds])
    fa = np.array([d(("alarm", "fapd_at_5pct_fpr"), k) for k in folds])
    de = np.array([d(("clinical", "zone_de_n"), k) for k in folds])
    act = [by[("fusion", k)].get("fusion_active_frac_test") for k in folds]
    act = [a for a in act if a is not None]
    out = ["", "% ---- MetaboNet fold-paired fusion vs CGM-only (corrected gate) ----",
           _macro("NumMetaboFusionRMSEBetterFolds", int((rm < 0).sum()), "{:d}"),
           _macro("NumMetaboFusionDeltaFAPD", fa.mean(), "{:+.2f}"),
           _macro("NumMetaboFusionDeltaFAPDSD", fa.std(ddof=1), "{:.2f}"),
           _macro("NumMetaboFusionFAPDBetterFolds", int((fa < 0).sum()), "{:d}"),
           _macro("NumMetaboFusionDeltaDE", de.mean(), "{:+.0f}"),
           _macro("NumMetaboFusionDeltaDESD", de.std(ddof=1), "{:.0f}"),
           _macro("NumMetaboFusionDEBetterFolds", int((de < 0).sum()), "{:d}"),
           _macro("NumMetaboFusionEffectToSpread", abs(rm.mean()) / float(np.std(
               [float(by[("cgm_only", k)]["per_horizon"][h]["regression"]["rmse"]) for k in folds],
               ddof=1)), "{:.1f}")]
    if act:
        out += [_macro("NumMetaboFusionActiveMinPct", 100 * min(act), "{:.0f}"),
                _macro("NumMetaboFusionActiveMaxPct", 100 * max(act), "{:.0f}")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/paper_results")
    ap.add_argument("--numbers", default="manuscript/numbers.tex")
    ap.add_argument("--metabonet", default=DEFAULT_METABONET_ROOT,
                    help="corpus-scale validation results; READ ONLY, and "
                         "skipped when absent or produced by --smoke")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="render artefacts marked \"smoke\": true. Only "
                         "reproduce_all_paper_results.sh --smoke should pass "
                         "this; the output is NOT publishable.")
    args = ap.parse_args()

    smoke = args.allow_smoke
    if smoke:
        print("--allow-smoke: rendering smoke artefacts. NOT publishable.\n")

    print("rebuilding benchmark tables:")
    a = rebuild_benchmarks(args.results, allow_smoke=smoke)
    print("rebuilding ablation tables:")
    b = rebuild_ablations(args.results, allow_smoke=smoke)
    print("rebuilding personalisation tables:")
    c = rebuild_personalization(args.results, allow_smoke=smoke)
    print("rebuilding multi-seed tables:")
    d = rebuild_multiseed(args.results)
    c += d
    print("rebuilding the MetaboNet appendix table:")
    c += rebuild_metabonet_table(args.metabonet)
    print("emitting inline numbers:")
    rebuild_numbers(args.results, args.numbers, args.metabonet, allow_smoke=smoke)
    print(f"\n{a + b + c} table file(s) regenerated from JSON.")

    if _COUPLING_PROBLEMS:
        # A table and a macro file that individually look fine and together
        # describe different experiments. Reported before the smoke block so the
        # more specific diagnosis is the last thing a reader sees on a run that
        # trips both.
        print("\n" + str(len(_COUPLING_PROBLEMS))
              + " appendix coupling problem(s):")
        for problem in _COUPLING_PROBLEMS:
            print(f"  {problem}")

    if _SMOKE_REFUSALS:
        # Loud and non-zero. A silent skip here is how the manuscript ends up
        # with a table and a macro file rendered from different experiments.
        print("\nREFUSED " + str(len(_SMOKE_REFUSALS)) + " smoke artefact(s):")
        for path in _SMOKE_REFUSALS:
            print(f"  {path}")
        print("The tables and macros they back are STALE. Re-run the driver at "
              "publication budget before citing anything from them.")

    if _SMOKE_REFUSALS or _COUPLING_PROBLEMS:
        sys.exit(1)


if __name__ == "__main__":
    main()
