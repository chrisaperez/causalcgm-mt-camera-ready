#!/usr/bin/env python3
"""Redact per-subject identifiers from a MetaboNet benchmark summary for release.

`run_metabonet_benchmarks.py` records, per fold, which subjects were held out
(`test_subjects`) and which the feature scaler was fitted on
(`scaler_fitted_on`), and the LoRA arm records one row per adapted subject.
Every identifier is "<source>:<id>". No renderer reads those identifiers: the
tables, macros and Figure 7 use counts and the numeric per-subject fields only.
The released copy therefore replaces

  * each ID list with a count per source (`<key>_by_source`), and
  * each per-subject `subject` with its source name alone,

and leaves every number untouched. `scripts/rebuild_paper_tables.py` renders
byte-identical macros and tables from the redacted copy (tests check this).

Usage:  redact_metabonet_summary.py IN.json OUT.json   (IN may equal OUT)
"""
from __future__ import annotations

import collections
import json
import sys

ID_LISTS = ("test_subjects", "scaler_fitted_on")


def _source(subject_id: str) -> str:
    if ":" not in subject_id:
        raise ValueError(f"not a '<source>:<id>' subject key: {subject_id!r}")
    return subject_id.split(":", 1)[0]


def _by_source(ids: list[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(_source(s) for s in ids).items()))


def redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ID_LISTS and isinstance(v, list):
                out[f"{k}_by_source"] = _by_source(v)
            elif k == "subject" and isinstance(v, str):
                out[k] = _source(v)
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        payload = json.load(f)
    if payload.get("subject_ids_redacted"):
        print(f"{argv[1]} is already redacted; nothing to do")
        out = payload
    else:
        out = redact(payload)
        out["subject_ids_redacted"] = True
    with open(argv[2], "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
