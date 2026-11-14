"""The shipped MetaboNet summary carries no subject identifiers, and redaction
changes no number."""
from __future__ import annotations

import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "runs", "metabonet_results", "benchmarks",
                       "metabonet_summary.json")
SUBJECT_KEY = re.compile(r"^[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+$")


def _load_redactor():
    spec = importlib.util.spec_from_file_location(
        "redact_metabonet_summary",
        os.path.join(ROOT, "scripts", "redact_metabonet_summary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def _numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


@pytest.mark.skipif(not os.path.exists(SUMMARY), reason="summary not present")
def test_shipped_summary_has_no_subject_identifiers():
    with open(SUMMARY) as f:
        payload = json.load(f)
    assert payload.get("subject_ids_redacted") is True
    leaked = [s for s in _strings(payload) if SUBJECT_KEY.match(s)]
    assert not leaked, f"subject identifiers in the shipped summary: {leaked[:5]}"
    for row in payload["per_fold"]:
        assert "test_subjects" not in row and "scaler_fitted_on" not in row
        assert sum(row["test_subjects_by_source"].values()) == row["n_test_subjects"]


def test_redaction_keeps_every_number_and_counts_by_source():
    R = _load_redactor()
    raw = {
        "provenance": {"n_subjects": 3, "sources": ["A", "B"]},
        "per_fold": [{
            "test_subjects": ["A:1", "A:2", "B:x"],
            "scaler_fitted_on": ["B:y"],
            "n_test_subjects": 3,
            "personalization": {"per_subject": [
                {"subject": "A:1", "rmse_before": 20.5, "rmse_after": 21.0}]},
        }],
    }
    red = R.redact(raw)
    fold = red["per_fold"][0]
    assert fold["test_subjects_by_source"] == {"A": 2, "B": 1}
    assert fold["scaler_fitted_on_by_source"] == {"B": 1}
    assert fold["personalization"]["per_subject"][0]["subject"] == "A"
    assert not [s for s in _strings(red) if SUBJECT_KEY.match(s)]
    # Every numeric leaf of the original survives at the same path.
    red_numbers = dict(_numbers(red))
    for path, value in _numbers(raw):
        assert red_numbers.get(path) == value, path


def test_redaction_refuses_a_malformed_subject_key():
    R = _load_redactor()
    with pytest.raises(ValueError):
        R.redact({"test_subjects": ["no-source-prefix"]})
