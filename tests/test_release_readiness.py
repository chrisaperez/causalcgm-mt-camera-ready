"""What a stranger with a fresh clone hits first.

Three properties, each of which was broken at some point in this repository's
history and none of which any other test covered:

  1. every third-party module the code imports is DECLARED, and every
     declaration is BOUNDED. `matplotlib` and `statsmodels` were reachable only
     as transitive dependencies of `methcomp` -- the one package
     `causalcgm_mt.clinical` treats as optional -- so declining the optional
     Parkes backend removed the `figures` stage and the ARIMA row of Table 1;

  2. the synthetic cohort gives every patient both classes. `run_fold` carves
     its inner validation slice out of the TRAINING patients, and at
     `n_folds=2` that slice is a single patient; drawing a hypo-free one makes
     `val/pr_auc` nan and `_require_finite_monitor` aborts the run. Four of the
     twelve patients in the shipped spec were hypo-free, so the documented
     quickstart failed on a clean checkout;

  3. a driver pointed at data this repository cannot ship exits 2 with the
     acquisition pointer, not a traceback whose last line is the only part that
     matters.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from causalcgm_mt.config import Config                      # noqa: E402
from causalcgm_mt.data import build_windows                 # noqa: E402
from causalcgm_mt.simulate import simulate_cohort           # noqa: E402

REQUIREMENTS = os.path.join(REPO, "requirements.txt")
LOCKFILE = os.path.join(REPO, "requirements-lock.txt")

# Import name -> distribution name, for the cases where they differ.
DIST_OF_MODULE = {"sklearn": "scikit-learn"}


def _requirements(path: str) -> dict[str, str]:
    """{distribution: specifier} from a requirements file."""
    from packaging.requirements import Requirement

    out: dict[str, str] = {}
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                r = Requirement(line)
                out[r.name.lower()] = str(r.specifier)
    return out


def _source_files() -> list[str]:
    files = [os.path.join(REPO, f) for f in ("app.py", "conftest.py")]
    for sub in ("causalcgm_mt", "scripts", "tests"):
        for root, _dirs, names in os.walk(os.path.join(REPO, sub)):
            if "__pycache__" in root:
                continue
            files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    return sorted(files)


def _local_module_names() -> set[str]:
    """Modules importable by bare name because a script or test sits next to them."""
    names = {"causalcgm_mt", "conftest", "scripts", "tests", "app"}
    for sub in ("scripts", "tests"):
        for n in os.listdir(os.path.join(REPO, sub)):
            if n.endswith(".py"):
                names.add(n[:-3])
    return names


def _imported_top_level_modules() -> dict[str, list[str]]:
    """{top-level module: files that import it}, third-party only."""
    local, stdlib = _local_module_names(), set(sys.stdlib_module_names)
    found: dict[str, list[str]] = {}
    for path in _source_files():
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        for m in mods:
            if m and m not in local and m not in stdlib and not m.startswith("_"):
                found.setdefault(m, []).append(os.path.relpath(path, REPO))
    return found


# --------------------------------------------------------------------------- #
# 1. Dependencies: declared, and bounded
# --------------------------------------------------------------------------- #
def test_every_imported_third_party_module_is_declared():
    declared = _requirements(REQUIREMENTS)
    # setuptools ships with any venv and pulling it in explicitly would pin the
    # build backend; packaging arrives with pip. Neither is imported by the
    # library, only by tooling that already has them.
    exempt = {"setuptools", "packaging", "pkg_resources"}
    missing = {
        mod: files
        for mod, files in _imported_top_level_modules().items()
        if mod not in exempt
        and DIST_OF_MODULE.get(mod, mod).lower() not in declared
    }
    assert not missing, (
        "imported but not in requirements.txt -- these currently install only "
        f"as somebody else's transitive dependency: {missing}")


def test_no_requirement_is_left_unbounded():
    """A bare `>=` floor resolves to whatever exists on the day of the install."""
    unbounded = [
        f"{name} ({spec or 'no specifier at all'})"
        for name, spec in _requirements(REQUIREMENTS).items()
        if "==" not in spec and "<" not in spec
    ]
    assert not unbounded, (
        "unbounded requirement(s); add a next-major ceiling so a future release "
        f"cannot silently change a published number: {unbounded}")


def test_lockfile_covers_the_direct_requirements_and_agrees_with_them():
    from packaging.specifiers import SpecifierSet

    req, lock = _requirements(REQUIREMENTS), _requirements(LOCKFILE)
    assert set(lock) == set(req), (
        "requirements-lock.txt must fix exactly the direct requirements: "
        f"missing={sorted(set(req) - set(lock))} extra={sorted(set(lock) - set(req))}")
    for name, spec in lock.items():
        assert spec.startswith("=="), f"{name} is not pinned in the lock file: {spec}"
        pinned = spec[2:]
        assert SpecifierSet(req[name]).contains(pinned, prereleases=True), (
            f"lock file pins {name}=={pinned}, which requirements.txt "
            f"({req[name]}) excludes")


# --------------------------------------------------------------------------- #
# 2. The synthetic cohort is usable by the pipeline that consumes it
# --------------------------------------------------------------------------- #
# Every (n_patients, days, seed) this repository actually generates: the
# quickstart, the dashboard checkpoint, compare_models, the Mondrian check, and
# the three test fixtures.
COHORT_SPECS = [
    dict(n_patients=12, days=6.0, seed=0),    # run_experiment.py --synthetic
    dict(n_patients=14, days=8.0, seed=0),    # scripts/train_demo.py COHORT
    dict(n_patients=12, days=7.0, seed=0),    # scripts/compare_models.py
    dict(n_patients=18, days=12.0, seed=7),   # verify_mondrian_coverage.py
    dict(n_patients=6, days=3.0, seed=7),     # tests/test_causality.py
    dict(n_patients=8, days=4.0, seed=11),    # tests/test_conformal_and_baseline.py
    dict(n_patients=6, days=4.0, seed=5),     # tests/test_training_stability.py
]


@pytest.mark.parametrize("spec", COHORT_SPECS, ids=lambda s: f"p{s['n_patients']}_s{s['seed']}")
def test_no_synthetic_patient_is_hypo_free(spec):
    ws = build_windows(simulate_cohort(**spec), Config().data)
    pat = np.asarray(ws.patient)
    empty = sorted({p for p in set(pat) if not ws.y_cls_hard[pat == p].any()})
    assert not empty, (
        f"{empty} have no positive window, so any fold whose inner validation "
        f"slice is one of them makes val/pr_auc nan and aborts the run")


def test_the_quickstart_cohort_is_not_pathologically_imbalanced():
    """The guarantee must not have turned the smoke cohort into a hypo cohort."""
    ws = build_windows(simulate_cohort(n_patients=12, days=6.0), Config().data)
    assert 0.005 < float(ws.y_cls_hard.mean()) < 0.25


def test_the_readme_quickstart_output_is_the_one_the_command_produces():
    """The README shows an abridged run of the smoke command so a reader can
    tell a healthy install from a broken one. Only the COHORT-level figures are
    quoted -- window count, patient count, prevalence -- because those are pure
    functions of the seeded generator and the windowing config. The fold-level
    metrics are elided in the README on purpose: they depend on training, and a
    number nothing checks is a number that goes stale."""
    ws = build_windows(simulate_cohort(n_patients=12, days=6.0), Config().data)
    doc = open(os.path.join(REPO, "README.md")).read()
    quoted = (f"{len(ws)} windows | {len(set(ws.patient))} patients "
              f"| hypo prevalence {float(ws.y_cls_hard.mean()):.2%}")
    assert quoted in doc, (
        f"the README's sample output has drifted from the command; the "
        f"quickstart now prints {quoted!r}")


# --------------------------------------------------------------------------- #
# 3. Missing input -> exit 2 and an instruction, not a traceback
# --------------------------------------------------------------------------- #
MISSING_INPUT_INVOCATIONS = [
    ("run_experiment.py", ["--csv", "/nonexistent/cohort.csv"]),
    ("run_hupa.py", ["--dir", "/nonexistent/hupa"]),
    ("prepare_ohio.py", ["--root", "/nonexistent/OhioT1DM", "--no-save"]),
    ("train_ohio.py", ["--data", "/nonexistent/ohio_windows.pt"]),
    ("run_paper_benchmarks.py", ["--root", "/nonexistent/OhioT1DM",
                                 "--cache", "/nonexistent/ws.pkl"]),
    ("run_paper_ablations.py", ["--root", "/nonexistent/OhioT1DM",
                                "--cache", "/nonexistent/ws.pkl", "--studies", "a"]),
    ("run_ohio_personalization_sweep.py", ["--root", "/nonexistent/OhioT1DM",
                                           "--cache", "/nonexistent/ws.pkl"]),
    ("run_metabonet_benchmarks.py", ["--parquet-path", "/nonexistent/mn.parquet",
                                     "--out", "/nonexistent/out"]),
]


@pytest.mark.parametrize("script,args", MISSING_INPUT_INVOCATIONS,
                         ids=[s for s, _ in MISSING_INPUT_INVOCATIONS])
def test_missing_input_exits_two_without_a_traceback(script, args, tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE")
    if script == "run_metabonet_benchmarks.py":
        args = ["--parquet-path", "/nonexistent/mn.parquet",
                "--out", str(tmp_path / "mn")]
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", script), *args],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=600)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, (
        f"{script} exited {proc.returncode}; 2 is this repository's "
        f"'precondition failed, nothing ran' code.\n{out[-2000:]}")
    assert "Traceback (most recent call last)" not in out, (
        f"{script} reported a missing input as a traceback:\n{out[-2000:]}")
    # Each driver points at ITS OWN no-data path where it has one (run_hupa
    # exercises synchronisation, which the generic synthetic cohort does not),
    # so the requirement is a --synthetic pointer, not one specific command.
    assert "--synthetic" in out, (
        f"{script} did not point at any path that needs no data:\n{out[-2000:]}")
