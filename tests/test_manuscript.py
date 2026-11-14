"""Checks on the manuscript and its generated inputs.

The manuscript is not prose that happens to mention numbers -- every inline
figure resolves through a macro in `numbers.tex`, and every table is `\\input`
from `runs/paper_results/`. That is the property worth protecting: a paper whose
abstract claims a result its own tables no longer support is the single most
expensive error available here, and it is invisible, because both the sentence
and the table look correct in isolation.

These tests are static: they parse the `.tex` files and never run LaTeX, so they
work without a TeX distribution installed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUSCRIPT = os.path.join(REPO, "manuscript")
MAIN = os.path.join(MANUSCRIPT, "main.tex")
NUMBERS = os.path.join(MANUSCRIPT, "numbers.tex")
BIB = os.path.join(MANUSCRIPT, "references.bib")

# The corpus-scale validation lives OUTSIDE the paper tree on purpose (its
# driver is forbidden to write into `runs/paper_results/`), so its one appendix
# table is `\input` straight from where the driver put it. That is the seam the
# section 6 tests below defend.
METABONET_ROOT = os.path.join(REPO, "runs", "metabonet_results")
METABONET_BENCH = os.path.join(METABONET_ROOT, "benchmarks")
METABONET_SUMMARY = os.path.join(METABONET_BENCH, "metabonet_summary.json")
METABONET_TABLE = os.path.join(METABONET_BENCH, "table_metabonet_comparison.tex")
REBUILD = os.path.join(REPO, "scripts", "rebuild_paper_tables.py")

sys.path.insert(0, REPO)

pytestmark = pytest.mark.skipif(
    not os.path.exists(MAIN), reason="manuscript not present")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


SECTIONS = os.path.join(MANUSCRIPT, "sections")


def _body() -> str:
    r"""`main.tex` plus every file it \input s from `manuscript/sections/`.

    The manuscript is modular: the main body is held to the ML4H Proceedings
    8-page limit and the ablations, the 60-minute tables and the secondary
    validations live in `sections/*.tex`. A test that read only `main.tex`
    would stop seeing half the paper and would pass by not looking -- which is
    the failure mode these tests exist to catch. `numbers.tex` and the
    `../runs/` tables are deliberately NOT followed: the first is nothing but
    numbers and would defeat the typed-number test, and the second is generated
    data rather than prose.
    """
    out, seen = [], set()
    stack = [MAIN]
    while stack:
        path = os.path.normpath(stack.pop(0))
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        txt = _read(path)
        out.append(txt)
        for rel in re.findall(r"\\input\{([^}]+)\}", txt):
            if not rel.endswith(".tex"):
                rel += ".tex"
            cand = os.path.normpath(os.path.join(MANUSCRIPT, rel))
            if cand.startswith(SECTIONS + os.sep):
                stack.append(cand)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 1. Inline numbers
# --------------------------------------------------------------------------- #
def test_every_inline_number_macro_is_defined():
    """An undefined macro renders as nothing (or an error) in the PDF, which is
    how a paper ends up with a blank where a headline result should be."""
    used = set(re.findall(r"\\(Num[A-Za-z]+)", _body()))
    defined = set(re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", _read(NUMBERS)))
    assert used, "the manuscript should quote numbers through macros"
    assert not (used - defined), f"undefined in numbers.tex: {sorted(used - defined)}"


def test_macro_names_are_legal_latex_control_sequences():
    r"""LaTeX control words are LETTERS ONLY. `\NumFusionRMSEH30` parses as
    `\NumFusionRMSEH` followed by the characters `30`, so a digit in a macro
    name silently prints the digits beside an undefined command."""
    defined = re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}", _read(NUMBERS))
    assert defined
    bad = [m for m in defined if not m.isalpha()]
    assert not bad, f"macro names containing non-letters: {bad}"


def test_leakage_macros_are_unsigned_and_name_their_direction():
    r"""`\NumLeakNaiveRMSEPct` printed `+23.0`, and the abstract read
    "inflating reported RMSE by +23.0%" for a leak that LOWERS measured RMSE.
    Every Ablation D effect macro must print an unsigned magnitude, and its
    name must say which way the leaky arm moved relative to the strict one."""
    leak = dict(re.findall(r"\\newcommand\{\\(NumLeak[A-Za-z]+)\}\{([^}]*)\}",
                           _read(NUMBERS)))
    assert leak, "numbers.tex carries no Ablation D macros"
    signed = {k: v for k, v in leak.items() if v.strip()[:1] in "+-"}
    assert not signed, f"signed leakage macros: {signed}"
    effects = [k for k in leak if not re.search(r"(Strict|Leaky)(H[A-Za-z]+)?$", k)]
    undirected = [k for k in effects
                  if not re.search(r"(RMSE(Reduction|Increase)Pct|"
                                   r"PRAUC(Increase|Decrease))(H[A-Za-z]+)?$", k)]
    assert effects and not undirected, f"leakage macros without a direction: {undirected}"


def test_numbers_file_is_generated_not_hand_edited():
    head = _read(NUMBERS).splitlines()[0]
    assert "rebuild_paper_tables.py" in head and "do not edit" in head.lower()


def test_numbers_file_is_current(tmp_path):
    """`numbers.tex` must equal what the drivers would render RIGHT NOW.

    Idempotence -- two tmp runs agreeing -- says the renderer is deterministic.
    It says nothing about whether the COMMITTED file matches the result JSONs,
    which is the property the "no typed number" invariant actually rests on. A
    stale macro file passed every other check here, compiled cleanly, and
    shipped in `package_release`'s public tier.
    """
    out = str(tmp_path / "numbers.tex")
    r = subprocess.run([sys.executable, REBUILD, "--numbers", out],
                       capture_output=True, text=True, cwd=REPO, timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert _read(out) == _read(NUMBERS), (
        "manuscript/numbers.tex is stale relative to the result JSONs. "
        "Re-run scripts/rebuild_paper_tables.py.")


def test_no_bare_decimal_results_smuggled_into_the_prose():
    """Results must come through macros.

    A four-significant-figure decimal in the body text is almost certainly a
    retyped metric; the few legitimate constants (thresholds, alpha, version
    numbers) are short and are allowed explicitly.
    """
    body = _body()
    body = re.sub(r"\\Num[A-Za-z]+", "", body)          # macros are fine
    body = re.sub(r"%.*", "", body)                      # comments are fine
    allowed = {"0.75", "1.0", "0.5", "0.9", "0.02", "0.009"}
    suspicious = [m for m in re.findall(r"(?<![\w.])\d+\.\d{3,}(?![\w])", body)
                  if m not in allowed]
    assert not suspicious, (
        f"decimals in the prose that should be macros: {suspicious}")


def test_no_bare_percentages_in_the_prose():
    r"""Integers followed by \% slip past the decimal check above.

    The abstract carried a hard-coded `$2\%$` for the kappa=3 RMSE cost while
    `\NumKappaThreeRMSECostPct` was defined and used correctly two sections
    later. It was a defensible rounding of 1.8; after the pipeline was corrected
    the true value became 2.7 and the abstract was simply wrong -- which is the
    failure this whole macro discipline exists to prevent, arriving through the
    one hole in the guard.

    Percentages that are part of an operating point (5\% FPR, 10\% FPR) or a
    tolerance are legitimate constants and are allowed by name.
    """
    body = _body()
    body = re.sub(r"\\Num[A-Za-z]+", "", body)
    body = re.sub(r"%.*", "", body)                      # LaTeX comments
    allowed = {"5", "10", "20", "90", "95", "100"}       # operating points, tolerances
    suspicious = [m for m in re.findall(r"(?<![\w.\\])(\d+)\\%", body)
                  if m not in allowed]
    assert not suspicious, (
        f"bare percentages in the prose that should be macros: {suspicious}")


# --------------------------------------------------------------------------- #
# 2. Structure, tables and figures
# --------------------------------------------------------------------------- #
def test_document_has_the_required_sections():
    body = _body()
    # "Temporal Causality", not "Structural Causality": the paper renamed the
    # section to keep `causality` unambiguously the signal-processing sense
    # (a non-anticipative filtration) rather than the interventional one.
    for heading in ("Introduction", "Temporal Causality", "Architecture",
                    "Experimental Setup", "Results", "Limitations",
                    "Conclusion"):
        assert re.search(rf"\\section\{{[^}}]*{heading}", body), \
            f"missing section: {heading}"
    assert "\\begin{abstract}" in body and "\\end{abstract}" in body
    assert "\\bibliography{references}" in body


def test_every_input_and_graphic_exists():
    """`\\input` of a missing table fails the build; a missing graphic silently
    leaves a grey box in some engines."""
    body = _body()
    for rel in re.findall(r"\\input\{([^}]+)\}", body):
        path = os.path.join(MANUSCRIPT, rel)
        if not rel.endswith(".tex"):
            path += ".tex"
        assert os.path.exists(path), f"\\input target missing: {rel}"
    for rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        path = os.path.join(MANUSCRIPT, "figures", rel)
        assert os.path.exists(path), f"figure missing: {rel}"


def test_every_figure_the_script_can_draw_is_referenced():
    """Figure 7 is in this list on the same terms as the other six.

    It is drawn from `runs/metabonet_results/`, which is not a stage of the
    reproduction pipeline -- but the appendix already `\\input`s that tree's
    table, so the manuscript's dependence on it is established, and a figure
    generated but never included is a figure nobody checks.
    """
    body = _body()
    for n, stem in enumerate(
            ["fig1_architecture", "fig2_anti_leakage_protocol",
             "fig3_clarke_parkes_grids", "fig4_conformal_coverage",
             "fig5_ablation_tradeoffs", "fig6_edge_latency",
             "fig7_metabonet_corpus_scale"], start=1):
        assert stem in body, f"Figure {n} ({stem}) is never included"


def test_every_label_that_is_referenced_exists():
    body = _body()
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for tex in re.findall(r"\\input\{([^}]+)\}", body):
        path = os.path.join(MANUSCRIPT, tex if tex.endswith(".tex") else tex + ".tex")
        if os.path.exists(path):
            labels |= set(re.findall(r"\\label\{([^}]+)\}", _read(path)))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    assert not (refs - labels), f"dangling \\ref: {sorted(refs - labels)}"


# Study key in ablations_summary.json -> the table STEM the manuscript must
# `\input`. A stem rather than a filename because `rebuild_paper_tables.py`
# emits both a combined `table_ablation_<key>.tex` and a per-horizon
# `table_ablation_<key>_ph{30,60}.tex`; the manuscript inputs the per-horizon
# files so the PH = 30 table can sit in the 8-page body while PH = 60 goes to an
# appendix. Any of the three satisfies "this study reached the paper".
ABLATION_TABLE_INPUTS = {
    "a_multitask": "table_ablation_a_multitask",
    "b_asymmetry": "table_ablation_b_asymmetry",
    "d_leakage": "table_ablation_d_leakage",
    "e_soft_label": "table_ablation_e_soft_label",
}


def test_every_completed_ablation_reaches_the_manuscript():
    r"""A study in the JSON must have its table `\input` by `main.tex`.

    `rebuild_paper_tables.py` emits a table for every study PRESENT in the
    summary, and `main.tex` inputs a fixed list. Those two lists are maintained
    in different files, so a study that is run but never referenced produces a
    table nobody reads and a manuscript that silently omits a result its own
    artefacts contain -- which is the failure this whole harness exists to stop,
    pointed the other way.

    This is written to pass NOW and fail the moment Ablation E is run at
    publication budget, which is exactly when `main.tex` needs the `\input` and
    a paragraph around it. `c_modality` is excluded because it is a dict of
    per-cohort sub-tables rather than one table, and both of its members are
    already inputs.
    """
    path = os.path.join(REPO, "runs", "paper_results", "ablations",
                        "ablations_summary.json")
    if not os.path.exists(path):
        pytest.skip("no ablation summary; run scripts/run_paper_ablations.py")

    import json
    with open(path) as f:
        studies = set(json.load(f).get("studies", {}))
    body = _body()

    missing = sorted(
        table for key, table in ABLATION_TABLE_INPUTS.items()
        if key in studies and table not in body)
    assert not missing, (
        f"these studies have results but no \\input in main.tex: {missing}. "
        f"Add the table and the prose that reads it, or drop the study from "
        f"the summary.")


def test_no_ablation_table_outlives_the_study_that_produced_it():
    r"""The inverse coupling: a `.tex` in the results tree needs a study behind it.

    `rebuild_paper_tables.py` only ever writes; it never prunes. So a table
    generated from a study that is later dropped from the summary -- or written
    during a partial or aborted run -- stays on disk, holding numbers no
    artefact backs any more, in the one directory `main.tex` is allowed to
    `\input` from.

    This is not hypothetical. Building these checks left
    `table_ablation_e_soft_label.tex` in the tree carrying SMOKE numbers, from a
    moment when the summary transiently contained study E. Nothing noticed:
    the summary was correct, every other table was correct, and the orphan was
    one `\input` away from the manuscript.
    """
    root = os.path.join(REPO, "runs", "paper_results", "ablations")
    summary = os.path.join(root, "ablations_summary.json")
    if not os.path.exists(summary):
        pytest.skip("no ablation summary; run scripts/run_paper_ablations.py")

    import glob
    import json
    with open(summary) as f:
        studies = json.load(f).get("studies", {})

    # `c_modality` is a dict of per-cohort sub-tables, plus one condensed
    # cross-cohort view that the main text actually inputs; the rest are one
    # each. The condensed table is DERIVED from the same study rows -- it is
    # backed by an artefact, which is what this check is about, so it is
    # enumerated here rather than exempted from the rule.
    expected = set()
    for key, value in studies.items():
        if key == "c_modality" and isinstance(value, dict):
            expected |= {f"c_modality_{c}" for c in value}
            expected.add("c_modality_summary")
        else:
            expected.add(key)

    # A per-horizon file (`..._ph30.tex`) is the same study as its combined
    # parent, so strip the horizon suffix before comparing against the summary.
    on_disk = {re.sub(r"_ph\d+$", "",
                      os.path.basename(p)[len("table_ablation_"):-len(".tex")])
               for p in glob.glob(os.path.join(root, "table_ablation_*.tex"))}
    orphans = sorted(on_disk - expected)
    assert not orphans, (
        f"ablation tables with no study in the summary: {orphans}. They hold "
        f"numbers nothing backs any more -- re-run the study or delete them.")


# Roots a table may be `\\input` from. Both are written by a driver from a
# result JSON and never edited by hand -- which is the property this list
# defends. `runs/metabonet_results/` is separate on purpose: the corpus-scale
# validation is not a stage of the paper pipeline and is forbidden to write into
# the paper tree, so its one appendix table is `\\input` from where its driver
# put it rather than copied across the boundary.
RESULT_ROOTS = ("runs/paper_results/", "runs/metabonet_results/")


def test_tables_are_input_from_the_results_tree():
    """Tables must be the generated artefacts, not pasted copies that can drift
    from the JSON they came from."""
    inputs = re.findall(r"\\input\{([^}]+)\}", _body())
    tables = [i for i in inputs if "table_" in i]
    assert len(tables) >= 5, f"expected the generated tables to be inputs, got {tables}"
    stray = [t for t in tables if not any(r in t for r in RESULT_ROOTS)]
    assert not stray, f"tables not from a results tree: {stray}"


# --------------------------------------------------------------------------- #
# 3. Bibliography
# --------------------------------------------------------------------------- #
def test_every_citation_resolves_to_a_bib_entry():
    keys = set(re.findall(r"@\w+\{([^,]+),", _read(BIB)))
    cited = set()
    for cmd in re.findall(r"\\cite[a-z]*\{([^}]+)\}", _read(MAIN)):
        cited |= {k.strip() for k in cmd.split(",")}
    assert cited, "the manuscript should cite something"
    assert not (cited - keys), f"citations with no bib entry: {sorted(cited - keys)}"


def test_the_canonical_citations_are_present_and_used():
    """The four the submission brief names explicitly."""
    body, keys = _read(MAIN), set(re.findall(r"@\w+\{([^,]+),", _read(BIB)))
    for key in ("kendall2018multitask", "vovk2005algorithmic",
                "clarke1987evaluating", "marling2018ohiot1dm",
                "marling2020ohiot1dm"):
        assert key in keys, f"{key} missing from references.bib"
        assert key in body, f"{key} is in the bib but never cited"


def test_bib_entries_have_the_fields_a_style_needs():
    text = _read(BIB)
    entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S)
    assert len(entries) >= 20
    for kind, key, body in entries:
        assert re.search(r"\btitle\s*=", body), f"{key}: no title"
        assert re.search(r"\byear\s*=", body), f"{key}: no year"
        if kind.lower() not in ("misc",):
            assert re.search(r"\bauthor\s*=", body), f"{key}: no author"


def test_braces_balance_in_the_manuscript():
    for path in (MAIN, NUMBERS):
        body = re.sub(r"(?<!\\)%.*", "", _read(path))
        body = body.replace(r"\{", "").replace(r"\}", "")
        assert body.count("{") == body.count("}"), f"unbalanced braces in {path}"


# --------------------------------------------------------------------------- #
# 4. The generator is runnable and idempotent
# --------------------------------------------------------------------------- #
def test_rebuild_tables_is_idempotent_and_changes_no_numbers(tmp_path):
    """Regenerating must be a pure function of the JSON: running it twice has to
    produce byte-identical output, or the tables are not reproducible.

    `--metabonet` is pinned at an empty directory on purpose. The renderer now
    exits non-zero when the corpus-scale artefacts are a smoke run or diverge
    from the macros beside them, and that is a statement about the state of a
    tree this test does not own -- it would turn a purity check into a data
    check and make this fail for the wrong reason. The MetaboNet half is
    covered by `test_metabonet_table_and_macros_carry_a_matching_run_fingerprint`
    and `test_rebuild_exits_nonzero_when_the_metabonet_summary_is_smoke`.
    """
    script = os.path.join(REPO, "scripts", "rebuild_paper_tables.py")
    out = str(tmp_path / "numbers.tex")
    absent = tmp_path / "no_metabonet"
    absent.mkdir()
    runs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, script, "--numbers", out,
                            "--metabonet", str(absent)],
                           capture_output=True, text=True, cwd=REPO, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        runs.append(_read(out))
    assert runs[0] == runs[1]
    assert "\\newcommand" in runs[0]


# --------------------------------------------------------------------------- #
# 5. It actually builds
# --------------------------------------------------------------------------- #
def _have(tool: str) -> bool:
    from shutil import which
    return which(tool) is not None


@pytest.mark.skipif(not _have("pdflatex"), reason="no LaTeX toolchain")
def test_manuscript_compiles_without_undefined_references(tmp_path):
    """A full pdflatex/bibtex/pdflatex/pdflatex cycle in a scratch directory.

    Static checks catch a missing macro; only a real build catches a missing
    package, an unbalanced environment, or a `\\ref` that resolves to `??`.
    Undefined citations and references are checked in the LOG rather than by
    exit status, because LaTeX exits 0 with them.
    """
    import shutil

    work = tmp_path / "m"
    shutil.copytree(MANUSCRIPT, work)
    # `\input` reaches into the results trees with relative paths. Both are
    # copied because the appendix table comes from the MetaboNet tree, which the
    # paper pipeline deliberately does not write into.
    for root in ("paper_results", "metabonet_results"):
        src_root = os.path.join(REPO, "runs", root)
        if os.path.exists(src_root):
            shutil.copytree(src_root, work.parent / "runs" / root,
                            ignore=shutil.ignore_patterns("*.npz", "*.pkl",
                                                          "*.pt", "*.npy"))

    def run(cmd):
        return subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                              timeout=900)

    r = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    assert r.returncode == 0, r.stdout[-3000:]
    run(["bibtex", "main"])
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    r = run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    assert r.returncode == 0, r.stdout[-3000:]
    assert (work / "main.pdf").exists()

    log = (work / "main.log").read_text(errors="replace")
    for pattern in ("Undefined control sequence",
                    "Citation .* undefined",
                    "Reference .* undefined",
                    "There were undefined references"):
        assert not re.search(pattern, log), f"{pattern} in main.log"


@pytest.mark.skipif(not _have("pdflatex"), reason="no LaTeX toolchain")
def test_generated_tables_are_two_column_safe():
    """In a twocolumn document a wide table must be a `table*` float, or it
    silently overflows the column and prints past the page margin."""
    import glob
    found = 0
    for path in glob.glob(os.path.join(REPO, "runs", "paper_results", "*",
                                       "table_*.tex")):
        body = _read(path)
        found += 1
        assert "\\begin{table*}" in body, f"{os.path.basename(path)} is not wide"
        assert "\\resizebox" in body, f"{os.path.basename(path)} will not shrink to fit"
        assert body.count("\\begin{table*}") == body.count("\\end{table*}")
    assert found >= 5, "expected the generated tables to be present"


# --------------------------------------------------------------------------- #
# 6. The MetaboNet appendix: table and macros must describe the SAME run
# --------------------------------------------------------------------------- #
# Audit finding C1. The appendix has two halves that are produced by two
# different processes with two different smoke guards:
#
#   table_metabonet_comparison.tex   <- run_metabonet_benchmarks.py, and it used
#                                       to be overwritten unconditionally
#   \NumMetabo* in numbers.tex       <- rebuild_paper_tables.py, which refuses a
#                                       smoke summary by emitting NO macros
#
# So a `--smoke` MetaboNet run replaced the table the manuscript inputs while
# the macros beside it stayed at whatever the last publication run left behind,
# and the paper compiled clean with its prose and its table from different
# experiments. Every existing gate missed it: the orphan-table check covers only
# `runs/paper_results/ablations`, the idempotence check compares two tmp copies
# to each other rather than to the committed file, and `package_release` does
# not list the MetaboNet summary among the artefacts whose provenance it checks.
#
# The coupling is now a run FINGERPRINT stamped into both halves. These tests
# pin the three properties that make it hold.


def _minimal_smoke_summary() -> dict:
    """The smallest summary that is unambiguously a smoke artefact."""
    return {"smoke": True, "horizon_min": 30, "n_folds": 2,
            "run_fingerprint": "0000smoke0000", "provenance": {},
            "summary": [], "per_fold": []}


def test_a_smoke_metabonet_run_cannot_overwrite_the_manuscript_table(tmp_path):
    """A smoke run must divert to a sidecar, leaving the published table intact.

    This is the write that started the incident. The driver had no guard at all:
    it called `write_latex` on the manuscript's `\\input` target whatever budget
    it had just run at, and the only trace was a warning folded into the table's
    own caption -- which nothing reads.
    """
    import scripts.run_metabonet_benchmarks as R

    bench = tmp_path / "benchmarks"
    bench.mkdir()
    published = bench / "table_metabonet_comparison.tex"
    published.write_text("PUBLICATION BUDGET TABLE\n")

    smoke_target = R.comparison_table_path(str(bench), smoke=True)
    assert smoke_target != str(published), (
        "a smoke run resolved to the manuscript's own \\input target; it will "
        "overwrite a publication table with two-epoch numbers")

    with open(smoke_target, "w") as f:
        f.write("SMOKE TABLE\n")
    assert published.read_text() == "PUBLICATION BUDGET TABLE\n", (
        "the published table was clobbered by a smoke run")

    # ...and a publication run still writes exactly where the manuscript reads.
    assert R.comparison_table_path(str(bench), smoke=False) == str(published)

    # The escape hatch exists, but it has to be asked for by name.
    assert R.comparison_table_path(str(bench), smoke=True,
                                   allow_smoke_table=True) == str(published)


def test_metabonet_table_and_macros_carry_a_matching_run_fingerprint(tmp_path):
    """Both halves must be stamped with the run that produced them, and a
    mismatch has to be reported rather than rendered."""
    from causalcgm_mt import benchmark as B

    import scripts.rebuild_paper_tables as RPT

    if not os.path.exists(METABONET_SUMMARY):
        pytest.skip("no MetaboNet summary; run scripts/run_metabonet_benchmarks.py")

    # A real summary with the smoke flag cleared: the exact shape the renderer
    # consumes, without depending on a hand-built fixture drifting from it.
    payload = json.loads(_read(METABONET_SUMMARY))
    payload["smoke"] = False
    fingerprint = payload["run_fingerprint"]

    root = tmp_path / "metabonet_results"
    bench = root / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "metabonet_summary.json").write_text(json.dumps(payload))

    macros = "\n".join(RPT.metabonet_numbers(str(root)))
    found = re.search(r"\\newcommand\{\\NumMetaboFingerprint\}\{([^}]*)\}", macros)
    assert found, (
        "the \\NumMetabo* macros carry no run fingerprint, so nothing can "
        "check them against the table the manuscript inputs beside them")
    assert found.group(1) == fingerprint

    # The table half of the pair carries the same stamp, as a LaTeX comment so
    # it is invisible in the PDF and readable by the checker.
    table = bench / "table_metabonet_comparison.tex"
    B.write_latex(str(table), "\\begin{table}\\end{table}",
                  provenance={"fingerprint": fingerprint, "smoke": False})
    prov = B.read_latex_provenance(str(table))
    assert prov is not None, "write_latex did not stamp the table"
    assert prov["fingerprint"] == fingerprint
    assert prov["smoke"] is False
    assert "\\NumMetaboFingerprint" not in _read(str(table))

    # Matching pair -> no complaint.
    assert RPT.check_metabonet_coupling(str(root), fingerprint) == []

    # A table left behind by a DIFFERENT run must be caught.
    B.write_latex(str(table), "\\begin{table}\\end{table}",
                  provenance={"fingerprint": "deadbeefdeadbeef", "smoke": False})
    problems = RPT.check_metabonet_coupling(str(root), fingerprint)
    assert problems, (
        "a table from a different run than the macros went undetected -- this "
        "is exactly the divergence C1 describes")
    assert any("fingerprint" in p for p in problems)

    # A smoke table under a publication summary must be caught too.
    B.write_latex(str(table), "\\begin{table}\\end{table}",
                  provenance={"fingerprint": fingerprint, "smoke": True})
    assert RPT.check_metabonet_coupling(str(root), fingerprint), (
        "a smoke table went undetected under a publication-budget summary")


def test_rebuild_exits_nonzero_when_the_metabonet_summary_is_smoke(tmp_path):
    """The MetaboNet refusal must be as loud as every other smoke refusal.

    It was not: `metabonet_numbers` did its own `json.load` and its own smoke
    check, printed a lowercase `skip:` line, dropped all 46 `\\NumMetabo*`
    macros and returned -- and `main()` still exited 0, because the refusal was
    never recorded in `_SMOKE_REFUSALS`. A CI job gating on the exit status saw
    a clean run while the committed `numbers.tex` went stale.
    """
    bench = tmp_path / "metabonet_results" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "metabonet_summary.json").write_text(
        json.dumps(_minimal_smoke_summary()))

    r = subprocess.run(
        [sys.executable, REBUILD,
         "--numbers", str(tmp_path / "numbers.tex"),
         "--metabonet", str(tmp_path / "metabonet_results")],
        capture_output=True, text=True, cwd=REPO, timeout=900)

    assert r.returncode != 0, (
        "rebuild_paper_tables.py exited 0 after refusing a smoke MetaboNet "
        "summary and silently dropping every \\NumMetabo* macro:\n"
        + r.stdout[-2000:])
    assert "metabonet_summary.json" in (r.stdout + r.stderr)


def test_metabonet_appendix_table_and_macros_describe_the_same_run():
    """The tree invariant: the two halves of the appendix must agree.

    Skipped on a fresh clone, where `runs/` is gitignored and neither half
    exists. Whenever the macros ARE committed, the table they are read beside
    has to be present, non-smoke, and from the same run.
    """
    numbers = _read(NUMBERS)
    has_macros = "\\NumMetaboSubjects" in numbers
    has_table = os.path.exists(METABONET_TABLE)
    if not has_macros and not has_table:
        pytest.skip("no MetaboNet appendix in this tree")

    assert has_macros, (
        f"{METABONET_TABLE} exists but numbers.tex carries no \\NumMetabo* "
        f"macros; the appendix table has no prose behind it")
    assert has_table, (
        "numbers.tex carries \\NumMetabo* macros but the table main.tex "
        "\\inputs beside them is absent -- re-run "
        "scripts/run_metabonet_benchmarks.py at publication budget")

    import scripts.rebuild_paper_tables as RPT

    # Diagnose rather than just fail. "No fingerprint" has two very different
    # causes and two very different remedies, and a bare assertion sends the
    # reader to the wrong one.
    summary_is_smoke = False
    if os.path.exists(METABONET_SUMMARY):
        summary_is_smoke = bool(json.loads(_read(METABONET_SUMMARY)).get("smoke"))

    found = re.search(r"\\newcommand\{\\NumMetaboFingerprint\}\{([^}]*)\}", numbers)
    if not found:
        assert not summary_is_smoke, (
            "the MetaboNet appendix is DESYNCHRONISED (audit finding C1).\n"
            f"  {METABONET_SUMMARY}\n"
            "    is a --smoke artefact, so rebuild_paper_tables.py refuses it "
            "and emits no \\NumMetabo* macros at all,\n"
            f"  {NUMBERS}\n"
            "    still carries macros from an earlier publication run whose "
            "artefacts are no longer in the tree.\n"
            "  The paper would compile with its appendix prose and its "
            "appendix table describing different experiments.\n"
            "  Remedy: re-run scripts/run_metabonet_benchmarks.py at "
            "publication budget, then scripts/rebuild_paper_tables.py. "
            "There is no code-only fix -- the artefacts the macros describe "
            "do not exist on disk.")
        pytest.fail(
            "numbers.tex predates run-fingerprint stamping; re-run "
            "scripts/rebuild_paper_tables.py so the appendix halves can be "
            "coupled")

    problems = RPT.check_metabonet_coupling(METABONET_ROOT, found.group(1))
    assert not problems, (
        "the MetaboNet appendix table and the macros beside it come from "
        "different runs:\n  " + "\n  ".join(problems))
