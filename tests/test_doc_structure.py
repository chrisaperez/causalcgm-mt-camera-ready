"""The structural-integrity checks, run as part of the suite.

`scripts/check_doc_structure.py` is a standalone tool so it can be a pre-commit
hook; this file makes it a gate, and pins each detector against a synthetic file
carrying exactly the damage it exists to catch.

The mutation tests are the point. A structural checker that reports "0 findings"
is indistinguishable from one that does nothing at all, and this one is aimed at
a failure mode -- a replacement that swallows an adjacent block -- whose whole
character is that the file still looks fine afterwards.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import check_doc_structure as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


def _write(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# 1. The gate
# --------------------------------------------------------------------------- #
def test_the_repository_documents_are_structurally_intact():
    """Every tracked design document and result log passes every check.

    This caught real damage the first time it ran: `PUBLICATION_ROADMAP.md` was
    missing its `## Phase 1` heading and the whole of `### 1.1`, with four
    orphaned lines of 1.1's closing bullet stranded above `### 1.2`. Nothing
    else in the repository recorded that the section had ever existed.
    """
    findings = []
    for path in C.default_targets():
        findings.extend(C.check_path(path))
    assert not findings, "structural damage:\n" + "\n".join(str(f) for f in findings)


def test_the_checker_covers_the_documents_that_matter():
    """A check that silently stopped looking at a file would still report clean."""
    targets = {os.path.relpath(p, REPO) for p in C.default_targets()}
    assert {"README.md", "DOCKER.md"} <= targets
    assert "manuscript/main.tex" in targets


# --------------------------------------------------------------------------- #
# 2. Markdown detectors, each against the damage it is for
# --------------------------------------------------------------------------- #
def test_a_swallowed_numbered_section_is_detected(tmp_path):
    """The headline case: a section is deleted and its neighbours still parse."""
    intact = "# D\n\n## Phase 1: A\n\n### 1.1 One\n\ntext\n\n### 1.2 Two\n\ntext\n"
    damaged = "# D\n\n## Phase 1: A\n\n### 1.2 Two\n\ntext\n"
    assert not _codes(C.check_markdown(_write(tmp_path, "ok.md", intact)))
    # The parent survives, so this is not an orphan -- the only evidence that
    # 1.1 ever existed is that 1.2 is the first child.
    assert "MD-NUMGAP" in _codes(C.check_markdown(_write(tmp_path, "bad.md", damaged)))


def test_a_missing_parent_heading_is_detected(tmp_path):
    """Exactly the roadmap's damage: the children survive, the parent does not."""
    damaged = "# D\n\n### 1.2 Two\n\ntext\n\n### 1.3 Three\n\ntext\n"
    assert "MD-NUMORPHAN" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_an_orphaned_continuation_line_is_detected(tmp_path):
    """The residue a swallowed block leaves behind.

    An indented line after a blank line or a rule continues nothing. It renders
    as a code block, which is why it survives a read-through.
    """
    damaged = "# D\n\n---\n\n        tail of a bullet whose bullet is gone\n\n## Next\n"
    assert "MD-ORPHAN" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_indented_text_under_a_real_bullet_is_not_flagged(tmp_path):
    """The same shape, legitimately: this must NOT fire, or nobody will run it."""
    fine = "# D\n\n* a bullet\n  - nested\n        continued here\n\n## Next\n"
    assert "MD-ORPHAN" not in _codes(C.check_markdown(_write(tmp_path, "ok.md", fine)))


def test_a_duplicated_section_is_detected(tmp_path):
    """A replacement that lands twice, the other half of the same failure mode."""
    damaged = "# D\n\n## Phase 1: A\n\n### 1.1 One\n\nx\n\n### 1.1 One\n\nx\n"
    assert "MD-NUMDUP" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_a_skipped_heading_level_is_detected(tmp_path):
    damaged = "# D\n\ntext\n\n### Deep\n\ntext\n"
    assert "MD-LEVEL" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_an_unclosed_code_fence_is_detected(tmp_path):
    damaged = "# D\n\n```python\nx = 1\n"
    assert "MD-FENCE" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_headings_inside_a_code_fence_are_not_headings(tmp_path):
    """A shell comment is not an h1. Without this the checker is unusable on a
    README with 38 fenced blocks."""
    fine = "# D\n\n```bash\n# not a heading\n### also not one\n```\n\n## Real\n"
    assert not _codes(C.check_markdown(_write(tmp_path, "ok.md", fine)))


def test_a_ragged_table_row_is_detected(tmp_path):
    """A replacement that eats a pipe shifts one row's values under the wrong
    headers, and Markdown renders it without complaint."""
    damaged = "# D\n\n| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n| 1 | 2 |\n"
    assert "MD-TABLE" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_a_broken_relative_link_is_detected(tmp_path):
    damaged = "# D\n\nSee [it](no/such/file.py).\n"
    assert "MD-LINK" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


def test_a_conflict_marker_is_detected(tmp_path):
    damaged = "# D\n\n<<<<<<< HEAD\nx\n"
    assert "MD-TRUNC" in _codes(C.check_markdown(_write(tmp_path, "b.md", damaged)))


# --------------------------------------------------------------------------- #
# 3. LaTeX detectors -- "breaking section tags"
# --------------------------------------------------------------------------- #
def test_an_unclosed_latex_environment_is_detected(tmp_path):
    damaged = "\\begin{document}\n\\begin{table}\nx\n\\end{document}\n"
    assert "TEX-ENV" in _codes(C.check_latex(_write(tmp_path, "b.tex", damaged)))


def test_a_crossed_latex_environment_is_detected(tmp_path):
    """`\\end{table}` closing a `\\begin{figure}` -- what a swallowed block leaves."""
    damaged = ("\\begin{document}\n\\begin{figure}\nx\n\\end{table}\n"
               "\\end{document}\n")
    assert "TEX-ENV" in _codes(C.check_latex(_write(tmp_path, "b.tex", damaged)))


def test_balanced_latex_is_not_flagged(tmp_path):
    fine = "\\begin{document}\n\\begin{table}\nx\n\\end{table}\n\\end{document}\n"
    assert not _codes(C.check_latex(_write(tmp_path, "ok.tex", fine)))


def test_unbalanced_latex_braces_are_detected(tmp_path):
    damaged = "\\begin{document}\n\\textbf{unclosed\n\\end{document}\n"
    assert "TEX-BRACE" in _codes(C.check_latex(_write(tmp_path, "b.tex", damaged)))


def test_graphicspath_is_honoured(tmp_path):
    """Without this the checker calls every figure in a correct document missing."""
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "f.pdf").write_text("x")
    doc = ("\\graphicspath{{figures/}}\n\\begin{document}\n"
           "\\includegraphics[width=1cm]{f.pdf}\n\\end{document}\n")
    assert "TEX-GRAPHIC" not in _codes(C.check_latex(_write(tmp_path, "ok.tex", doc)))

    doc_bad = doc.replace("{f.pdf}", "{missing.pdf}")
    assert "TEX-GRAPHIC" in _codes(C.check_latex(_write(tmp_path, "b.tex", doc_bad)))


# --------------------------------------------------------------------------- #
# 4. JSON result logs
# --------------------------------------------------------------------------- #
def test_a_truncated_json_log_is_detected(tmp_path):
    """A run killed mid-write leaves a file that is still 99% correct."""
    damaged = '{"experiment": "x", "arms": [{"arm": "a"'
    assert "JSON-PARSE" in _codes(C.check_json(_write(tmp_path, "b.json", damaged)))


def test_a_nan_literal_is_rejected(tmp_path):
    """`json.load` accepts `NaN`; nothing else does, including every reader a
    reader would use on the released artefact."""
    p = _write(tmp_path, "results_summary.json",
               '{"experiment": "x", "arms": [], "dataset": {}, '
               '"protocol": {}, "v": NaN}')
    assert "JSON-PARSE" in _codes(C.check_json(p))


def test_missing_consumer_keys_are_detected(tmp_path):
    """`rebuild_paper_tables.py` indexes these by literal name at the end of a
    multi-hour pipeline."""
    p = _write(tmp_path, "results_summary.json", json.dumps({"experiment": "x"}))
    assert "JSON-KEYS" in _codes(C.check_json(p))


def test_a_partial_study_merge_is_detected(tmp_path):
    """A study with no recorded budget cannot be read: the numbers in it were
    produced at an unknown epoch count and nothing says which."""
    p = _write(tmp_path, "ablations_summary.json", json.dumps({
        "experiment": "x", "budget": {}, "generated": "t",
        "studies": {"a_multitask": [], "e_soft_label": []},
        "study_meta": {"a_multitask": {}},
    }))
    assert "JSON-MERGE" in _codes(C.check_json(p))


# --------------------------------------------------------------------------- #
# 5. The tool itself
# --------------------------------------------------------------------------- #
def test_the_script_exits_non_zero_on_damage(tmp_path, monkeypatch, capsys):
    bad = _write(tmp_path, "b.md", "# D\n\n### 1.2 Two\n\nx\n")
    monkeypatch.setattr(sys, "argv", ["check_doc_structure.py", bad])
    assert C.main() == 1
    assert "MD-NUMORPHAN" in capsys.readouterr().out


def test_the_script_exits_zero_when_clean(tmp_path, monkeypatch):
    ok = _write(tmp_path, "ok.md", "# D\n\n## Phase 1: A\n\n### 1.1 One\n\nx\n")
    monkeypatch.setattr(sys, "argv", ["check_doc_structure.py", ok])
    assert C.main() == 0
