"""Every number quoted in the prose must match the artefact it came from.

`test_doc_structure.py` checks that the documents are structurally intact --
that no section was swallowed and no table went ragged. This file checks the
other half: that the numbers *inside* them are still true.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "runs", "paper_results")
README = os.path.join(REPO, "README.md")
NUMBERS = os.path.join(REPO, "manuscript", "numbers.tex")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load(*parts) -> dict:
    path = os.path.join(RESULTS, *parts)
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, REPO)} not present; run its driver")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def personalization() -> dict:
    return _load("personalization", "personalization_sweep.json")


def _markdown_table(doc: str, header_contains: str) -> dict:
    """The first markdown table whose header row contains `header_contains`,
    as {first cell: [remaining cells]}.
    """
    rows: dict = {}
    in_table = False
    for line in doc.split("\n"):
        s = line.strip()
        is_row = s.startswith("|") and s.endswith("|")
        if not in_table:
            if is_row and header_contains in s:
                in_table = True
            continue
        if not is_row:
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):      # separator row
            continue
        rows[cells[0]] = cells[1:]
    assert rows, f"no markdown table with {header_contains!r} in its header"
    return rows


# --------------------------------------------------------------------------- #
# 1. Personalisation invariants in the README
# --------------------------------------------------------------------------- #
PRIMARY_PREFIX = "24.0"


def test_readme_zero_shot_control_is_reported_as_exactly_zero(personalization):
    """`none` scored twice must be bit-identical; the README says 0.0000 and that
    claim is the live check that the harness adds no drift."""
    none = personalization["significance"][PRIMARY_PREFIX]["none"]["rmse"]
    assert none["mean_delta"] == 0.0
    assert "0.0000" in _read(README)


# --------------------------------------------------------------------------- #
# 2. The soft-label band (F10)
# --------------------------------------------------------------------------- #
def test_the_readme_band_is_what_the_estimator_actually_produces():
    """The README's three-tier table quotes a measured MARD and band for
    OhioT1DM. Recompute both from the raw records rather than trusting the
    prose, since this is the number the whole F10 story rests on.
    """
    root = os.path.join(REPO, "OhioT1DM")
    if not os.path.isdir(root):
        pytest.skip("OhioT1DM not present")

    from causalcgm_mt.config import DataConfig
    from causalcgm_mt.data import SoftLabeler, estimate_reference_mard
    from causalcgm_mt.ohio_loader import collect_reference_pairs, discover_records

    pairs = collect_reference_pairs(discover_records(root))
    est = estimate_reference_mard(pairs.cgm, pairs.reference, pairs.patient)
    lab = SoftLabeler.fit(np.array([100.0]), DataConfig(band_mode="mard"), noise=est)

    doc = _read(README)
    mard_pct, band = est.mard * 100, lab.band

    tiers = _markdown_table(doc, "on OhioT1DM")
    cell = tiers["`paired_reference`"][-1]
    assert f"{mard_pct:.1f}% MARD" in cell, (
        f"tier table quotes a stale cohort MARD in {cell!r}; "
        f"measured {mard_pct:.1f}%")
    assert f"{band:.2f} mg/dL" in cell, (
        f"tier table quotes a stale band in {cell!r}; measured {band:.2f} mg/dL")

    for text, what in ((f"{mard_pct:.1f}% MARD", "cohort MARD"),
                       (f"{band:.2f}", "band")):
        assert doc.count(text) >= 2, (
            f"the {what} is quoted in the tier table and again in Status; "
            f"only one copy now reads {text!r}, so the other is stale")

    assert lab.band_source == "mard", (
        "the README presents this band as MEASURED; the estimator now reports "
        f"{lab.band_source!r}")
