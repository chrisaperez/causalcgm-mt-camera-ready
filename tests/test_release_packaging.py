"""Gates on the reproduction runner and the release packager.

The packager decides what leaves this repository, and one of the things it must
never let out is OhioT1DM. That dataset is released under a data use agreement
which does not permit redistribution, and the per-window prediction dumps under
`runs/paper_results/*/predictions/` are OhioT1DM: CGM values, subject
identifiers and timestamps, per row.

The first version of `package_release.py` had a tier table that looked correct
and still swept thirty of those dumps into the PUBLIC tier, because one entry
meant to collect `.tex` tables was a directory glob that collected everything
beneath it. These tests exist because that class of mistake is silent -- the
archive builds, the manifest looks complete, and the leak is only visible if
someone lists the contents and knows what `.npz` means here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import package_release as P  # noqa: E402

RUNNER = os.path.join(REPO, "scripts", "reproduce_all_paper_results.sh")


def _have_results() -> bool:
    return os.path.exists(os.path.join(
        REPO, "runs", "paper_results", "benchmarks", "results_summary.json"))


# --------------------------------------------------------------------------- #
# 1. The licensed-data boundary
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_the_public_tier_carries_no_per_window_patient_data():
    """The headline guarantee. `.npz` here is OhioT1DM, not a numeric detail."""
    files = [f for f, _ in P.collect("public")]
    leaked = [f for f in files if f.endswith(".npz") or "predictions" in f]
    assert not leaked, (
        f"public tier would redistribute OhioT1DM-derived per-window data: {leaked}")


def test_the_publishability_gate_refuses_a_restricted_path():
    """The gate is independent of the tier table, and must fail on its own.

    Tested by injecting a restricted path into an otherwise fine file list: if
    this passes silently, the boundary is only as good as every glob in
    CONTENTS, which is exactly the assumption that failed once already.
    """
    ok = [("manuscript/numbers.tex", "macros")]
    P.assert_publishable(ok, "public")          # must not raise

    for bad in ("runs/paper_results/ablations/predictions/x.npz",
                "runs/paper_results/benchmarks/predictions/y.npz"):
        with pytest.raises(SystemExit, match="REFUSING"):
            P.assert_publishable(ok + [(bad, "oops")], "public")


def test_the_gate_does_not_block_the_full_tier():
    """`full` exists precisely to carry those files, for a DUA holder."""
    P.assert_publishable(
        [("runs/paper_results/ablations/predictions/x.npz", "restricted")], "full")


def test_the_full_tier_requires_an_explicit_dua_acknowledgement():
    """`--tier full` alone must not produce an archive."""
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "package_release.py"),
         "--tier", "full", "--list"],
        capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r.returncode == 2
    assert "data use agreement" in (r.stderr + r.stdout)


@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_the_full_tier_actually_adds_the_restricted_files():
    """Otherwise the tier split would be decoration."""
    pub = {f for f, _ in P.collect("public")}
    full = {f for f, _ in P.collect("full")}
    if pub == full:
        pytest.skip("the restricted full-tier artefacts (per-window .npz) are not "
                    "tracked, so a fresh checkout has none to add")
    assert pub < full
    assert any(f.endswith(".npz") for f in full - pub)


# --------------------------------------------------------------------------- #
# 2. Provenance
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_smoke_artefacts_are_refused_not_warned_about(tmp_path):
    """A two-epoch budget must not reach a citable archive."""
    src = os.path.join(REPO, "runs/paper_results/ablations/ablations_summary.json")
    bak = str(tmp_path / "bak.json")
    shutil.copy(src, bak)
    try:
        with open(src) as f:
            d = json.load(f)
        d["smoke"] = True
        with open(src, "w") as f:
            json.dump(d, f, indent=2)
        with pytest.raises(SystemExit, match="smoke"):
            P.check_provenance(allow_smoke=False)
        P.check_provenance(allow_smoke=True)     # the override still works
    finally:
        shutil.copy(bak, src)
    with open(src) as f:
        assert json.load(f).get("smoke") is False, "fixture not restored"


@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_regenerable_caches_are_never_packaged():
    """Hundreds of megabytes, and a stale one reproduces the wrong windows."""
    for tier in ("public", "full"):
        files = [f for f, _ in P.collect(tier)]
        for junk in ("ohio_ws.pkl", "ohio_windows.pt", "naive_ws.pkl",
                     "_snapshots", "_superseded"):
            assert not any(junk in f for f in files), f"{junk} in {tier} tier"


@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_the_manifest_hashes_match_the_files_it_lists():
    """A manifest whose digests are wrong is worse than none."""
    files = P.collect("public")[:5]
    manifest = P.build_manifest(files, "public", ["(test)"])
    for rel, _ in files:
        digest = hashlib.sha256(open(os.path.join(REPO, rel), "rb").read()).hexdigest()
        assert f"{digest}" in manifest, f"digest for {rel} absent or wrong"
    assert "window pipeline version" in manifest


@pytest.mark.skipif(not _have_results(), reason="no results tree to package")
def test_every_explicitly_listed_file_keeps_its_own_description():
    """A file listed by name must not inherit a directory sweep's label.

    `base_model.pt` lives under the personalisation directory and was being
    described as "personalisation tables" -- harmless in isolation, and the kind
    of thing that makes a manifest untrustworthy in aggregate.
    """
    by_file = dict(P.collect("public"))
    ck = "runs/paper_results/personalization/base_model.pt"
    if ck in by_file:
        assert "checkpoint" in by_file[ck].lower()


# --------------------------------------------------------------------------- #
# 3. The reproduction runner
# --------------------------------------------------------------------------- #
def test_the_runner_is_syntactically_valid():
    r = subprocess.run(["bash", "-n", RUNNER], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_runner_invokes_only_scripts_that_exist():
    """A typo in a driver name would surface hours into a run, not at minute one."""
    body = open(RUNNER).read()
    import re
    referenced = set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", body))
    assert referenced, "the runner should call the drivers by path"
    missing = sorted(s for s in referenced
                     if not os.path.exists(os.path.join(REPO, "scripts", s)))
    assert not missing, f"runner calls scripts that do not exist: {missing}"


def test_the_runner_passes_study_letters_the_driver_accepts():
    """`--studies a b c d e f` must survive the driver's argparse `choices`.

    An invalid letter aborts the ablation stage instantly, after the benchmark
    stage has already spent an hour.
    """
    import re
    body = open(RUNNER).read()
    m = re.search(r'STUDIES="\$\{STUDIES:-([a-z ]+)\}"', body)
    assert m, "could not find the runner's default study list"
    letters = m.group(1).split()

    drv = open(os.path.join(REPO, "scripts", "run_paper_ablations.py")).read()
    c = re.search(r'choices=\[([^\]]*)\]', drv[drv.index('"--studies"'):])
    allowed = set(re.findall(r'"([a-z])"', c.group(1)))
    assert set(letters) <= allowed, (
        f"runner passes {sorted(set(letters) - allowed)}, which the driver rejects")


def test_the_runner_refuses_to_start_without_the_dataset(tmp_path):
    """Exit 2, before any driver runs, with instructions rather than a traceback."""
    # PY is set so the dataset check, not the interpreter check that precedes it
    # on a checkout without a .venv, is what refuses.
    env = dict(os.environ, ROOT=str(tmp_path / "nope"), PY=sys.executable)
    r = subprocess.run([RUNNER], capture_output=True, text=True, cwd=REPO,
                       env=env, timeout=120)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "OhioT1DM" in (r.stdout + r.stderr)


def test_the_runner_rejects_an_unknown_stage():
    for flag in ("--from", "--only"):
        r = subprocess.run([RUNNER, flag, "nosuchstage"], capture_output=True,
                           text=True, cwd=REPO, timeout=120)
        assert r.returncode == 2, f"{flag} accepted a bad stage"


# --------------------------------------------------------------------------- #
# 4. The container definition
# --------------------------------------------------------------------------- #
def test_the_dockerfile_copies_only_paths_that_exist():
    """A missing COPY source fails the build after the slow dependency layer."""
    import re
    df = os.path.join(REPO, "Dockerfile")
    if not os.path.exists(df):
        pytest.skip("no Dockerfile")
    missing = []
    for line in open(df):
        if not line.startswith("COPY "):
            continue
        parts = line.split()[1:]
        for src in parts[:-1]:                    # last token is the destination
            if src.startswith("--"):
                continue
            if not os.path.exists(os.path.join(REPO, src.rstrip("/"))):
                missing.append(src)
    assert not missing, f"Dockerfile COPYs paths that do not exist: {missing}"


def test_the_build_context_excludes_the_licensed_dataset():
    """OhioT1DM must not reach the daemon, let alone an image layer."""
    ign = os.path.join(REPO, ".dockerignore")
    if not os.path.exists(ign):
        pytest.skip("no .dockerignore")
    body = open(ign).read()
    for must in ("OhioT1DM/", "runs/", ".venv/"):
        assert must in body, f".dockerignore does not exclude {must}"
