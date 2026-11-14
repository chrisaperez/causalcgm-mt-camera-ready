"""Shared CLI conventions for the drivers in this directory.

EXIT CODES
----------
The convention `reproduce_all_paper_results.sh` and `run_metabonet_benchmarks.py`
already follow, written down so the rest of the drivers can follow it too:

    0   the run finished
    1   the run started and then failed
    2   a PRECONDITION failed -- nothing was run

The difference between 1 and 2 is not cosmetic. Exit 1 may have left partial
artefacts worth resuming from; exit 2 cannot have, because nothing ran. It also
matters to the person at the keyboard: a missing licensed dataset is not a bug
report, it is a one-line instruction, and a traceback buries that instruction
under twenty frames of `pandas.io.common`.

WHY THIS IS A MODULE AND NOT FIVE COPIES OF A try/except
--------------------------------------------------------
Seven drivers take a path to data this repository cannot ship. Each one needs
the same three things -- the loader's message, the acquisition pointer, and the
exit code -- and the failure mode of duplicating them is that they drift, so
some drivers exit 1 and some exit 2 for the same missing file, and a caller
cannot branch on either.
"""

from __future__ import annotations

import contextlib
import sys

EXIT_PRECONDITION = 2

# Every path through this repository that needs licensed data has one that does
# not, and a person who just cloned the repo usually wants that one.
NO_DATA_HINT = (
    "To check the pipeline itself without any licensed data:\n"
    "    .venv/bin/python scripts/run_experiment.py --synthetic --epochs 3 --folds 2"
)

OHIO_HINT = (
    "OhioT1DM is licensed and cannot be downloaded automatically. Request it at\n"
    "    https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html\n"
    "and place the release directories at <root>/2018 and <root>/2020.\n\n"
    + NO_DATA_HINT
)

ARTIFACT_HINT = (
    "That artefact is built by prepare_ohio.py, not shipped:\n"
    "    .venv/bin/python scripts/prepare_ohio.py --root OhioT1DM --out runs/ohio\n\n"
    + NO_DATA_HINT
)

# No acquisition URL here on purpose: this repository does not record one for
# HUPA-UCM anywhere else, and a pointer invented in an error message is worse
# than none. The OhioT1DM URL above is the one `reproduce_all_paper_results.sh`
# already gives.
HUPA_HINT = (
    "--dir takes the DIRECTORY of per-subject HUPA-UCM CSVs, not a single file.\n\n"
    "For the synchronisation path with no download:\n"
    "    .venv/bin/python scripts/run_hupa.py --synthetic --epochs 2 --folds 2"
)

CSV_HINT = (
    "--csv takes a long-format CSV with columns: patient_id, timestamp, glucose.\n\n"
    + NO_DATA_HINT
)


@contextlib.contextmanager
def missing_input(hint: str = ""):
    """Report a missing input file as a message and `EXIT_PRECONDITION`.

    Wrap ONLY the load, never a whole `main()`. A `FileNotFoundError` raised
    from inside training is a bug, and swallowing its traceback to report
    "precondition failed" would misreport what happened -- the narrow scope is
    the point of the context manager.
    """
    try:
        yield
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        if hint:
            print(f"\n{hint}", file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
