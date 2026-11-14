"""manuscript/metabonet_study_design.json classifies exactly the run's sources,
and the closed-loop macros are generated from it, never typed."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(ROOT, "manuscript", "metabonet_study_design.json")
SUMMARY = os.path.join(ROOT, "runs", "metabonet_results", "benchmarks",
                       "metabonet_summary.json")
NUMBERS = os.path.join(ROOT, "manuscript", "numbers.tex")


def _rebuild():
    spec = importlib.util.spec_from_file_location(
        "rebuild_paper_tables", os.path.join(ROOT, "scripts", "rebuild_paper_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _design():
    with open(DESIGN) as f:
        return json.load(f)


def test_every_entry_is_classified_with_evidence():
    seen = set()
    for s in _design()["studies"]:
        assert s["source"] not in seen, f"duplicate entry {s['source']}"
        seen.add(s["source"])
        assert s["closed_loop"] in ("yes", "mixed", "no", "unclear")
        assert s["evidence_url"].startswith("https://")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["retrieved"])


@pytest.mark.skipif(not os.path.exists(SUMMARY), reason="summary not present")
def test_design_file_covers_exactly_the_runs_sources():
    with open(SUMMARY) as f:
        run_sources = set(json.load(f)["provenance"]["sources"])
    design_sources = {s["source"] for s in _design()["studies"]}
    assert design_sources == run_sources


@pytest.mark.skipif(not os.path.exists(SUMMARY), reason="summary not present")
def test_macros_match_the_design_file_and_refuse_a_gap(tmp_path):
    R = _rebuild()
    with open(SUMMARY) as f:
        res = json.load(f)
    lines = R.metabonet_study_design_numbers(res, DESIGN)
    with open(NUMBERS) as f:
        committed = f.read()
    for line in lines:
        assert line in committed, f"numbers.tex is stale: {line}"
    n_yes = sum(1 for s in _design()["studies"] if s["closed_loop"] == "yes")
    assert f"\\newcommand{{\\NumMetaboCLStudies}}{{{n_yes}}}" in lines

    # An unclassified source must stop the build, not drop out of the count.
    gap = _design()
    gap["studies"] = gap["studies"][1:]
    p = tmp_path / "gap.json"
    p.write_text(json.dumps(gap))
    with pytest.raises(SystemExit):
        R.metabonet_study_design_numbers(copy.deepcopy(res), str(p))

    # Reclassifying a study moves the generated count.
    flip = _design()
    for s in flip["studies"]:
        if s["closed_loop"] == "yes":
            s["closed_loop"] = "no"
            break
    p2 = tmp_path / "flip.json"
    p2.write_text(json.dumps(flip))
    flipped = R.metabonet_study_design_numbers(copy.deepcopy(res), str(p2))
    assert f"\\newcommand{{\\NumMetaboCLStudies}}{{{n_yes - 1}}}" in flipped
