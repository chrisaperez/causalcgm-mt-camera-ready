"""Structural integrity checks for the design documents and result logs.

    python scripts/check_doc_structure.py            # check everything, exit 1 on damage
    python scripts/check_doc_structure.py --list     # what would be checked
    python scripts/check_doc_structure.py README.md  # check specific files

WHY THIS EXISTS
---------------
The documents in this repository are edited by large mechanical string
replacements, and that class of edit fails in a characteristic way: it does not
corrupt a character here and there, it swallows a whole adjacent block. The
replacement lands, the file still parses as Markdown, every heading that remains
is well formed -- and a section is simply gone, leaving the tail of its last
bullet stranded at the top level of the document.

That is invisible to a spell checker, to a diff read at the hunk level, and to a
human reading for content, because nothing on the page is wrong: the wrongness
is what is absent. It is not invisible to a numbering check.

`PUBLICATION_ROADMAP.md` carried exactly this damage while these checks were
being written -- `## Phase 1` and its `### 1.1` were missing, with four orphaned
lines of 1.1's last bullet left behind above `### 1.2`.

WHAT IS CHECKED
---------------
Markdown  fenced blocks balance; heading levels never skip; numbered sections
          are contiguous and sit at a level consistent with their number; no
          orphaned continuation lines; tables keep a constant column count;
          no truncation or conflict markers; relative links resolve.
LaTeX     every `\\begin{X}` has a matching `\\end{X}`, correctly nested; braces
          balance; `\\input` and `\\includegraphics` targets exist.
JSON      parses under a STRICT reader (no NaN/Infinity, which `json.load`
          accepts and every other language's parser rejects); known result files
          carry the keys the table builders index into.

Exit status is 0 when clean and 1 when any finding is reported, so it works as a
pre-commit hook as well as through `tests/test_doc_structure.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Documents whose structure is load-bearing. A design document is a contract
# about what was decided; a numbering gap in it is a decision that has gone
# missing, not a typo.
DESIGN_DOCS = ("README.md", "DOCKER.md")
LATEX_DOCS = ("manuscript/main.tex",)
JSON_LOGS = (
    "runs/paper_results/benchmarks/results_summary.json",
    "runs/paper_results/ablations/ablations_summary.json",
    "runs/paper_results/personalization/personalization_sweep.json",
    "runs/paper_results/edge/latency_sweep.json",
    # Outside the paper tree by design, but the manuscript appendix now reads
    # macros out of it, so it is a load-bearing artefact like the four above.
    "runs/metabonet_results/benchmarks/metabonet_summary.json",
)

# Text that only appears when generation was interrupted or a merge went wrong.
TRUNCATION_MARKERS = (
    "<<<<<<<", ">>>>>>>", "=======\n=======",
    "[truncated]", "...(truncated)", "<truncated>", "\ufffd",
)

# The result files are indexed by `rebuild_paper_tables.py` and the figure
# scripts through fixed paths. A missing top-level key there is a crash in a
# script that runs at the very end of a multi-hour pipeline.
# Keys taken from the CONSUMERS -- `rebuild_paper_tables.py` and
# `make_paper_figures.py` index these by literal name, so an absent one is an
# unhandled KeyError at the end of a multi-hour pipeline rather than a warning.
JSON_REQUIRED_KEYS = {
    "results_summary.json": ("experiment", "arms", "dataset", "protocol"),
    "ablations_summary.json": ("experiment", "budget", "generated", "studies",
                               "study_meta"),
    "personalization_sweep.json": ("experiment", "protocol", "sweep",
                                   "subjects", "significance"),
    "latency_sweep.json": ("experiment", "runs", "model", "deadline_ms"),
    # `rebuild_paper_tables.metabonet_numbers` indexes all five by literal name,
    # and `smoke` is the one that decides whether the file may be rendered at
    # all -- an absent flag would read as False and let a two-epoch run into the
    # appendix.
    "metabonet_summary.json": ("smoke", "horizon_min", "n_folds", "provenance",
                               "summary", "per_fold"),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{loc}: [{self.code}] {self.message}"


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# `1.2 Wire LoRA ...`, and also `Phase 2: Full Empirical ...` -- a numbered
# document rarely numbers its top level bare, and a checker that only sees the
# bare form reports every child as an orphan instead of reporting the one parent
# that is actually missing.
_NUMBERED = re.compile(
    r"^(?:[A-Za-z]+\s+)?(\d+)(?:\.(\d+))?[:.]?(?:\s|$)")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _strip_fenced(lines: list[str]) -> list[bool]:
    """Per-line flag: is this line inside a fenced code block?

    Everything else in this module has to know, because a Markdown checker that
    reads `# comment` inside a shell block as a heading reports nothing but
    false positives on a repository with 38 fenced blocks in one file.
    """
    inside, out, fence = False, [], ""
    for ln in lines:
        s = ln.lstrip()
        if not inside and (s.startswith("```") or s.startswith("~~~")):
            inside, fence = True, s[:3]
            out.append(True)
            continue
        if inside and s.startswith(fence):
            inside = False
            out.append(True)
            continue
        out.append(inside)
    return out


def check_markdown(path: str) -> list[Finding]:
    rel = os.path.relpath(path, REPO)
    with open(path) as f:
        lines = f.read().split("\n")
    inside = _strip_fenced(lines)
    out: list[Finding] = []

    # ---- fences balance --------------------------------------------------- #
    if inside and inside[-1]:
        out.append(Finding(rel, len(lines), "MD-FENCE",
                           "file ends inside a fenced code block"))

    # ---- truncation / conflict markers ------------------------------------ #
    for i, ln in enumerate(lines, 1):
        for marker in TRUNCATION_MARKERS:
            if marker in ln:
                out.append(Finding(rel, i, "MD-TRUNC",
                                   f"truncation/conflict marker {marker!r}"))

    # ---- headings --------------------------------------------------------- #
    heads: list[tuple[int, int, str]] = []       # (line, level, text)
    for i, (ln, skip) in enumerate(zip(lines, inside), 1):
        if skip:
            continue
        m = _HEADING.match(ln)
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))

    prev_level = 0
    for line_no, level, text in heads:
        if prev_level and level > prev_level + 1:
            out.append(Finding(rel, line_no, "MD-LEVEL",
                               f"heading jumps from h{prev_level} to h{level} "
                               f"({text!r}); a level was removed with its section"))
        prev_level = level

    out += _check_section_numbering(rel, heads)
    out += _check_orphaned_continuations(rel, lines, inside)
    out += _check_tables(rel, lines, inside)
    out += _check_links(rel, path, lines, inside)
    return out


def _check_section_numbering(rel: str, heads: list[tuple[int, int, str]]) -> list[Finding]:
    """Numbered headings must be contiguous, unique, and correctly nested.

    This is the check that finds a swallowed block. A run of `1.1, 1.2, 1.3`
    that becomes `1.2, 1.3` is not a formatting problem -- decision 1.1 is gone,
    and nothing else in the file records that it ever existed.

    The expected range is derived from the document rather than configured, so a
    document that renumbers or grows validates without anyone editing this file.
    """
    out: list[Finding] = []
    major_at: dict[int, int] = {}                 # major -> heading level
    minors: dict[int, list[tuple[int, int]]] = {}  # major -> [(minor, line)]
    majors: list[tuple[int, int]] = []            # (major, line)
    seen: set[tuple[int, int | None]] = set()

    for line_no, level, text in heads:
        m = _NUMBERED.match(text)
        if not m:
            continue
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) else None
        if (major, minor) in seen:
            label = f"{major}.{minor}" if minor else str(major)
            out.append(Finding(rel, line_no, "MD-NUMDUP",
                               f"section {label} appears more than once; a "
                               f"replacement probably landed twice"))
            continue
        seen.add((major, minor))

        if minor is None:
            majors.append((major, line_no))
            major_at[major] = level
        else:
            minors.setdefault(major, []).append((minor, line_no))
            parent = major_at.get(major)
            if parent is None:
                out.append(Finding(rel, line_no, "MD-NUMORPHAN",
                                   f"section {major}.{minor} has no {major}. "
                                   f"parent heading above it"))
            elif level != parent + 1:
                out.append(Finding(rel, line_no, "MD-NUMLEVEL",
                                   f"section {major}.{minor} is at h{level} but "
                                   f"its parent {major}. is at h{parent}"))

    for name, seq in [("major section", [n for n, _ in majors])] + [
            (f"section {maj}.x", [n for n, _ in sorted(v)]) for maj, v in sorted(minors.items())]:
        if not seq:
            continue
        # Numbering starts at 1. Checking only the gaps INSIDE the observed run
        # misses the easiest block to lose: swallow `1.1` and a lone surviving
        # `1.2` is a run of length one, internally contiguous, and silent.
        if min(seq) > 1:
            out.append(Finding(rel, 0, "MD-NUMGAP",
                               f"{name} numbering starts at {min(seq)}: "
                               f"{list(range(1, min(seq)))} missing from the top"))
        missing = sorted(set(range(min(seq), max(seq) + 1)) - set(seq))
        if missing:
            out.append(Finding(rel, 0, "MD-NUMGAP",
                               f"{name} numbering has gaps: missing {missing} "
                               f"from the run {min(seq)}-{max(seq)}"))
        if seq != sorted(seq):
            out.append(Finding(rel, 0, "MD-NUMORDER",
                               f"{name} numbering is out of order: {seq}"))
    return out


def _check_orphaned_continuations(rel: str, lines: list[str],
                                  inside: list[bool]) -> list[Finding]:
    """An indented line that continues nothing is the residue of a deleted block.

    A bullet's continuation is indented under the bullet. Remove the bullet --
    and the heading above it -- and the continuation stays behind, indented,
    attached to nothing. It renders as a code block, so it looks deliberate.

    Only flags a run that begins after a blank line or a horizontal rule, which
    is where a swallowed section leaves its tail; a genuinely indented block
    inside a list has a list item above it.
    """
    out: list[Finding] = []
    prev_meaningful = ""
    for i, (ln, skip) in enumerate(zip(lines, inside), 1):
        if skip:
            prev_meaningful = ln
            continue
        indented = ln[:1] == " " and ln.strip() and (len(ln) - len(ln.lstrip())) >= 4
        if indented and (prev_meaningful.strip() in ("", "---", "***", "___")):
            out.append(Finding(rel, i, "MD-ORPHAN",
                               f"indented line continues nothing: {ln.strip()[:60]!r}"))
        if ln.strip():
            prev_meaningful = ln
        elif prev_meaningful.strip() not in ("---", "***", "___"):
            prev_meaningful = ln
    return out


def _check_tables(rel: str, lines: list[str], inside: list[bool]) -> list[Finding]:
    """Every row of one table must have the same number of cells.

    A replacement that eats a `|` merges two columns for one row, and Markdown
    renders it without complaint -- the row simply shows the wrong values under
    the wrong headers, which is worse than a broken table.
    """
    out: list[Finding] = []
    run: list[tuple[int, int]] = []          # (line, n_cells)

    def flush():
        if len(run) >= 2:
            widths = {n for _, n in run}
            if len(widths) > 1:
                head = run[0][1]
                for line_no, n in run:
                    if n != head:
                        out.append(Finding(rel, line_no, "MD-TABLE",
                                           f"table row has {n} cells, header has {head}"))
        run.clear()

    for i, (ln, skip) in enumerate(zip(lines, inside), 1):
        s = ln.strip()
        if not skip and s.startswith("|") and s.endswith("|"):
            run.append((i, len(s.split("|")) - 2))
        else:
            flush()
    flush()
    return out


def _check_links(rel: str, path: str, lines: list[str],
                 inside: list[bool]) -> list[Finding]:
    """Relative links must resolve to something in the repository."""
    out: list[Finding] = []
    base = os.path.dirname(path)
    for i, (ln, skip) in enumerate(zip(lines, inside), 1):
        if skip:
            continue
        for target in _LINK.findall(ln):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = os.path.normpath(os.path.join(base, target.split("#")[0]))
            if not os.path.exists(resolved):
                out.append(Finding(rel, i, "MD-LINK",
                                   f"link target does not exist: {target}"))
    return out


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
_BEGIN = re.compile(r"\\begin\{([^}]+)\}")
_END = re.compile(r"\\end\{([^}]+)\}")
_INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def check_latex(path: str) -> list[Finding]:
    rel = os.path.relpath(path, REPO)
    with open(path) as f:
        raw = f.read()
    lines = raw.split("\n")
    out: list[Finding] = []

    for i, ln in enumerate(lines, 1):
        for marker in TRUNCATION_MARKERS:
            if marker in ln:
                out.append(Finding(rel, i, "TEX-TRUNC",
                                   f"truncation/conflict marker {marker!r}"))

    # ---- environment nesting ---------------------------------------------- #
    stack: list[tuple[str, int]] = []
    for i, ln in enumerate(lines, 1):
        code = ln.split("%")[0] if not ln.lstrip().startswith("%") else ""
        for name in _BEGIN.findall(code):
            stack.append((name, i))
        for name in _END.findall(code):
            if not stack:
                out.append(Finding(rel, i, "TEX-ENV",
                                   rf"\end{{{name}}} with no open environment"))
            elif stack[-1][0] != name:
                open_name, open_line = stack[-1]
                out.append(Finding(rel, i, "TEX-ENV",
                                   rf"\end{{{name}}} closes \begin{{{open_name}}} "
                                   f"opened at line {open_line}"))
                stack.pop()
            else:
                stack.pop()
    for name, line_no in stack:
        out.append(Finding(rel, line_no, "TEX-ENV",
                           rf"\begin{{{name}}} is never closed"))

    # ---- brace balance ----------------------------------------------------- #
    depth = 0
    for i, ln in enumerate(lines, 1):
        code = re.sub(r"(?<!\\)%.*$", "", ln)
        code = re.sub(r"\\[{}]", "", code)
        depth += code.count("{") - code.count("}")
        if depth < 0:
            out.append(Finding(rel, i, "TEX-BRACE", "unbalanced closing brace"))
            depth = 0
    if depth:
        out.append(Finding(rel, len(lines), "TEX-BRACE",
                           f"{depth} unclosed brace(s) at end of file"))

    # ---- referenced files -------------------------------------------------- #
    # `\graphicspath{{figures/}}{{alt/}}` adds search roots for
    # `\includegraphics`, so a checker that only looks beside main.tex reports
    # every figure in a correctly configured document as missing.
    base = os.path.dirname(path)
    graphic_roots = [base] + [
        os.path.join(base, d)
        for spec in re.findall(r"\\graphicspath\{(.+?)\}\s*$", raw, re.M)
        for d in re.findall(r"\{([^{}]*)\}", spec)
    ]
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("%"):
            continue
        for target in _INPUT.findall(ln):
            cands = [target, target + ".tex"]
            if not any(os.path.exists(os.path.join(base, c)) for c in cands):
                out.append(Finding(rel, i, "TEX-INPUT",
                                   rf"\input target missing: {target}"))
        for target in _GRAPHIC.findall(ln):
            cands = [target] + [target + e for e in (".pdf", ".png", ".jpg", ".eps")]
            if not any(os.path.exists(os.path.join(root, c))
                       for root in graphic_roots for c in cands):
                out.append(Finding(rel, i, "TEX-GRAPHIC",
                                   f"graphic missing: {target}"))
    return out


# --------------------------------------------------------------------------- #
# JSON result logs
# --------------------------------------------------------------------------- #
def _reject_nonfinite(_: str):
    raise ValueError("non-finite literal (NaN/Infinity) is not valid JSON")


def check_json(path: str) -> list[Finding]:
    rel = os.path.relpath(path, REPO)
    out: list[Finding] = []
    try:
        with open(path) as f:
            payload = json.load(
                f, parse_constant=_reject_nonfinite)
    except ValueError as e:
        return [Finding(rel, 0, "JSON-PARSE", str(e))]

    if not isinstance(payload, dict):
        return [Finding(rel, 0, "JSON-SHAPE", "top level is not an object")]

    for name, keys in JSON_REQUIRED_KEYS.items():
        if os.path.basename(path) != name:
            continue
        missing = [k for k in keys if k not in payload]
        if missing:
            out.append(Finding(rel, 0, "JSON-KEYS",
                               f"missing required top-level keys: {missing}"))

    # Ablation summaries carry a per-study budget. A study present in `studies`
    # but absent from `study_meta` is a partial merge, and its numbers were then
    # produced at an unknown budget -- unreadable, and silently so.
    if "studies" in payload and "study_meta" in payload:
        undocumented = sorted(set(payload["studies"]) - set(payload["study_meta"]))
        if undocumented:
            out.append(Finding(rel, 0, "JSON-MERGE",
                               f"studies with no recorded budget: {undocumented}"))
    return out


# --------------------------------------------------------------------------- #
def check_path(path: str) -> list[Finding]:
    if path.endswith(".md"):
        return check_markdown(path)
    if path.endswith(".tex"):
        return check_latex(path)
    if path.endswith(".json"):
        return check_json(path)
    return []


def default_targets() -> list[str]:
    """Every tracked document that exists. Absent artefacts are not failures --
    the result logs are rebuilt by hours of compute and a fresh clone has none."""
    out = []
    for rel in (*DESIGN_DOCS, *LATEX_DOCS, *JSON_LOGS):
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="files to check (default: all tracked docs)")
    ap.add_argument("--list", action="store_true", help="print the targets and exit")
    args = ap.parse_args()

    targets = [os.path.abspath(p) for p in args.paths] or default_targets()
    if args.list:
        for t in targets:
            print(os.path.relpath(t, REPO))
        return 0

    findings: list[Finding] = []
    for t in targets:
        if not os.path.exists(t):
            findings.append(Finding(os.path.relpath(t, REPO), 0, "IO",
                                    "file does not exist"))
            continue
        findings.extend(check_path(t))

    for f in findings:
        print(f)
    print(f"\n{len(targets)} file(s) checked, {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
