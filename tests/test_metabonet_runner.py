"""Audit of the MetaboNet benchmark driver.

The failure modes a results driver has to be protected against are not "does it
crash" but:

  * does it write anywhere near the paper's artefacts?
  * does the contamination guard actually run, on the rows that matter?
  * does an arm's spec change what gets trained, or is one arm a copy of another
    wearing a different name?
  * does the serialised table say which protocol produced it?

Runs on a synthetic MetaboNet-shaped parquet, so it needs no licensed data.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_metabonet_benchmarks as R  # noqa: E402
from causalcgm_mt import benchmark as B  # noqa: E402
from causalcgm_mt.config import DataConfig  # noqa: E402
from causalcgm_mt.metabonet_loader import MetaboNetWindowSet  # noqa: E402
from tests.test_metabonet_loader import canonical, make_subject  # noqa: E402


@pytest.fixture(scope="module")
def parquet(tmp_path_factory) -> str:
    """Six subjects over three sources, all with hypo events and full aux."""
    corpus = pd.concat(
        [make_subject(src, pid, n=1100, seed=i, start="2021-03-01")
         for i, (src, pid) in enumerate([("StudyA", "1"), ("StudyA", "2"),
                                         ("StudyB", "1"), ("StudyB", "5"),
                                         ("StudyC", "3"), ("StudyC", "9")])],
        ignore_index=True)
    p = tmp_path_factory.mktemp("mnrun") / "synthetic.parquet"
    corpus.to_parquet(p, row_group_size=800)
    return str(p)


@pytest.fixture(scope="module")
def smoke_run(parquet, tmp_path_factory):
    """One real end-to-end invocation, reused by the assertions below."""
    out = tmp_path_factory.mktemp("mnout")
    rc = R.main(["--parquet-path", parquet, "--out", str(out), "--smoke",
                 "--folds", "2", "--epochs", "1", "--max-subjects", "6",
                 "--prefix-hours", "12", "--adapt-steps", "5",
                 # Big enough to hold a 12 h prefix, the embargo and 50
                 # evaluation windows for the lora arm; see the guard in main().
                 "--max-windows-per-subject", "400"])
    assert rc == 0
    with open(os.path.join(out, "benchmarks", "metabonet_summary.json")) as f:
        return json.load(f), str(out)


# --------------------------------------------------------------------------- #
# Output isolation
# --------------------------------------------------------------------------- #
def test_default_output_root_is_not_the_paper_tree():
    assert R.DEFAULT_OUT == "runs/metabonet_results"
    assert "paper_results" not in R.DEFAULT_OUT


def test_refuses_to_write_into_the_paper_tree(parquet):
    """And refuses BEFORE creating anything.

    The check used to run after `configure_logging` had opened
    `<out>/logs/run.log` and after `makedirs(<out>/benchmarks)`, so every run of
    this very test seeded two directories inside the tree the driver is not
    allowed to touch. The refusal held for results and leaked directories, which
    is a strange shape for an invariant to have: `runs/paper_results/x/` was
    sitting in the repository, created by the test that proves it cannot be.
    """
    out = "runs/paper_results/x"
    existed = os.path.exists(out)
    rc = R.main(["--parquet-path", parquet, "--out", out, "--smoke"])
    assert rc == 2, "the driver wrote into the paper artefact tree"
    if not existed:
        assert not os.path.exists(out), (
            f"refused, but still created {out}; the guard must run before any "
            f"directory is made")


def test_the_cap_is_documented_as_blocks_everywhere_it_is_described():
    """A docstring that describes the rejected design is worse than none.

    The development notes stated the invariant as "never change the thinning to
    every-k-th", yet the config field and the CLI help both described exactly
    that -- so a reader following the documentation would have reintroduced the
    bug the code was changed to fix, and would have had the docs on their side.
    """
    import inspect

    from causalcgm_mt import metabonet_loader as M

    src = inspect.getsource(M.MetaboNetConfig)
    field = src[src.index("# ---- window budget"):src.index("memory_budget_gb")]
    assert "contiguous blocks" in field.lower()
    assert "every k-th, k chosen" not in field, (
        "max_windows_per_subject still documents every-k-th thinning")

    help_text = R.parse_args(["--parquet-path", "x.parquet"])
    parser_src = inspect.getsource(R.parse_args)
    cap_help = parser_src[parser_src.index("--max-windows-per-subject"):
                          parser_src.index("--memory-budget-gb")]
    assert "contiguous blocks" in cap_help
    assert "k-th in time" not in cap_help, "the CLI help still promises every k-th"
    assert help_text.max_windows_per_subject == 400


def test_the_paper_tree_guard_does_not_depend_on_the_working_directory(tmp_path,
                                                                       monkeypatch):
    """`os.path.abspath("runs/paper_results")` resolved against the CWD.

    Run this driver from anywhere but the repository root -- a cron job, a
    Docker WORKDIR, `cd runs && python ../scripts/...` -- and the guard compared
    `--out` against a directory that does not exist, so every path passed,
    including the real paper tree. The invariant the development notes call load-bearing
    held only for people who happened to be standing in the right place.
    """
    inside = [R.PAPER_TREE,
              R.PAPER_TREE / "x",
              R.PAPER_TREE / "benchmarks" / "deep" / "er"]
    outside = [R.REPO_ROOT / "runs" / "metabonet_results",
               R.REPO_ROOT / "runs" / "paper_results_metabonet",
               tmp_path / "out"]

    for cwd in (R.REPO_ROOT, tmp_path):
        monkeypatch.chdir(cwd)
        for path in inside:
            assert R.is_inside_paper_tree(str(path)), f"{path} from {cwd}"
        for path in outside:
            assert not R.is_inside_paper_tree(str(path)), f"{path} from {cwd}"

    # A relative `--out` means the paper tree wherever it is typed: refused
    # from the repo root (where it IS the tree) and from elsewhere (where the
    # operator plainly meant it).
    for cwd in (R.REPO_ROOT, tmp_path):
        monkeypatch.chdir(cwd)
        assert R.is_inside_paper_tree("runs/paper_results")
        assert R.is_inside_paper_tree("runs/paper_results/x")
        assert R.is_inside_paper_tree("./runs/./paper_results/../paper_results/y")


def test_the_guard_has_a_separator_boundary():
    """`startswith` on a path string has none, so a perfectly legitimate
    `runs/paper_results_metabonet` was refused -- while a symlink pointing into
    the real tree walked straight past. Resolved, component-wise comparison
    fixes both directions at once."""
    assert not R.is_inside_paper_tree(str(R.REPO_ROOT / "runs" / "paper_results_x"))
    assert not R.is_inside_paper_tree(str(R.REPO_ROOT / "runs" / "paper_resultsX"))
    assert R.is_inside_paper_tree(str(R.PAPER_TREE / "paper_results_x"))


def test_the_guard_follows_a_symlink_into_the_paper_tree(tmp_path):
    """The path the driver would actually write through is the resolved one."""
    link = tmp_path / "innocent"
    try:
        link.symlink_to(R.PAPER_TREE, target_is_directory=True)
    except (OSError, NotImplementedError):                # pragma: no cover
        pytest.skip("symlinks unavailable here")
    assert R.is_inside_paper_tree(str(link))
    assert R.is_inside_paper_tree(str(link / "benchmarks"))


def test_package_release_cannot_pick_these_artefacts_up():
    r"""Isolation is structural: `package_release` ships an explicit allow-list,
    never a sweep over `runs/`.

    The invariant is about DIRECTORY entries, which is where the hazard lives --
    one entry meant to collect tables also swept `ablations/predictions/*.npz`
    into the public tier. A directory entry may therefore only name the paper
    tree or the demo artefacts. A single explicitly-named FILE cannot sweep
    anything, so the MetaboNet appendix table -- which `main.tex` \inputs, and
    which no other tier ships -- is allowed through by name and by name only.
    """
    import scripts.package_release as P

    entries = [(rel, suffixes) for rel, _t, _d, suffixes in P.CONTENTS]
    assert entries, "package_release has no contents table any more"

    for rel, suffixes in entries:
        if suffixes:                      # a directory sweep, filtered by suffix
            assert rel.startswith(("runs/paper_results", "runs/demo",
                                   "manuscript")), (
                f"directory entry {rel!r} sweeps a tree outside the paper "
                f"artefacts; this is how predictions reached the public tier")

    # Exactly one MetaboNet artefact may ship, by name, with no suffix sweep.
    metabonet = [(rel, sfx) for rel, sfx in entries
                 if rel.startswith("runs/metabonet")]
    assert metabonet == [
        ("runs/metabonet_results/benchmarks/table_metabonet_comparison.tex", ())], (
        f"unexpected MetaboNet entries in CONTENTS: {metabonet}")


def test_run_writes_only_under_its_own_root(smoke_run):
    payload, out = smoke_run
    written = {os.path.relpath(os.path.join(r, f), out)
               for r, _d, fs in os.walk(out) for f in fs}
    assert written, "nothing was written"
    assert any(w.endswith("metabonet_summary.json") for w in written)
    # A SMOKE run writes the sidecar, never the manuscript's own \input target.
    # See `comparison_table_path`: a two-epoch table sitting under
    # publication-budget \NumMetabo* macros is how the appendix ends up
    # describing two different experiments.
    assert any(w.endswith(R.MANUSCRIPT_TABLE_NAME + R.SMOKE_TABLE_SUFFIX)
               for w in written)
    assert not any(w.endswith(R.MANUSCRIPT_TABLE_NAME) for w in written), (
        "a smoke run overwrote the comparison table the manuscript \\inputs")
    assert not os.path.exists("runs/paper_results/benchmarks/metabonet_summary.json")


# --------------------------------------------------------------------------- #
# Contamination guard
# --------------------------------------------------------------------------- #
def test_guard_runs_on_selection_windows_and_every_test_fold(monkeypatch, parquet,
                                                             tmp_path):
    """Three calls, not one. Each catches something the others cannot."""
    calls: list[list[str]] = []
    real = R.assert_no_benchmark_contamination

    def spy(subjects, allow=()):
        keys = (subjects.keys() if hasattr(subjects, "keys")
                else [str(s) for s in subjects])
        calls.append(list(keys))
        return real(subjects, allow=allow)

    monkeypatch.setattr(R, "assert_no_benchmark_contamination", spy)
    rc = R.main(["--parquet-path", parquet, "--out", str(tmp_path), "--smoke",
                 "--folds", "2", "--epochs", "1", "--arms", "cgm_only"])
    assert rc == 0
    assert len(calls) >= 4, (
        f"guard ran {len(calls)} times: expected selection + window set + one "
        f"per test fold")
    # The per-fold calls are the ones a reader of the table relies on.
    fold_calls = calls[2:]
    assert all(c for c in fold_calls), "a fold's guard saw an empty subject list"
    assert all(":" in k for c in calls for k in c), (
        "the guard was handed bare ids, which are ambiguous across source_file")


def test_a_contaminated_corpus_is_refused(tmp_path):
    corpus = pd.concat([
        make_subject("StudyA", "1", n=700, seed=1),
        make_subject("StudyA", "2", n=700, seed=3),
        make_subject("StudyB", "4", n=700, seed=4),
        make_subject("StudyB", "6", n=700, seed=5),
        make_subject("OhioT1DM", "559", n=700, seed=2),
    ], ignore_index=True)
    p = tmp_path / "dirty.parquet"
    corpus.to_parquet(p, row_group_size=400)

    # Excluded by default, so the run is clean and Ohio simply is not there.
    rc = R.main(["--parquet-path", str(p), "--out", str(tmp_path / "o1"),
                 "--smoke", "--folds", "2", "--epochs", "1", "--arms", "cgm_only"])
    assert rc == 0
    with open(tmp_path / "o1" / "benchmarks" / "metabonet_summary.json") as f:
        payload = json.load(f)
    assert "OhioT1DM" not in payload["provenance"]["sources"]
    assert payload["provenance"]["allowed_benchmark_sources"] == []

    # Waiving the exclusion is possible, is explicit, and is recorded.
    rc = R.main(["--parquet-path", str(p), "--out", str(tmp_path / "o2"),
                 "--smoke", "--folds", "2", "--epochs", "1", "--arms", "cgm_only",
                 "--allow-benchmark-source", "OhioT1DM"])
    assert rc == 0
    with open(tmp_path / "o2" / "benchmarks" / "metabonet_summary.json") as f:
        payload = json.load(f)
    assert "OhioT1DM" in payload["provenance"]["sources"]
    assert payload["provenance"]["allowed_benchmark_sources"] == ["OhioT1DM"], (
        "a contaminated run must be distinguishable from a clean one in its "
        "own artefact")


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def test_arms_are_armspecs_with_distinct_configurations():
    specs = {s.name: s for s in R.build_arms(30, 5, 1337, 4, 24.0, 200)}
    assert set(specs) == {"cgm_only", "fusion", "lora"}
    assert all(s.family in B.FAMILIES for s in specs.values())
    assert specs["cgm_only"].unimodal and not specs["fusion"].unimodal
    assert specs["lora"].base_arm == "cgm_only"
    # A spec that does not change what gets trained turns an ablation into
    # three copies of one run.
    assert specs["cgm_only"].as_dict() != specs["fusion"].as_dict()
    assert specs["lora"].as_dict() != specs["cgm_only"].as_dict()


def test_arm_config_is_the_shared_mapping():
    """The MetaboNet runner and the paper drivers must read a spec identically."""
    spec = R.build_arms(30, 7, 99, 4, 24.0, 200)[0]
    cfg = B.arm_config(spec, DataConfig(horizon_min=30))
    assert cfg.train.max_epochs == 7 and cfg.train.seed == 99
    assert cfg.data.horizon_min == 30
    assert cfg.model.n_horizons == 1


def test_multi_horizon_specs_are_refused_with_the_reason():
    spec = R.ArmSpec(name="x", family="tcn", horizons=(30, 60))
    with pytest.raises(SystemExit, match="single-horizon"):
        R.check_single_horizon([spec])


def test_unimodal_view_removes_the_aux_branch_entirely(parquet):
    """Zeroing the aux tensor would still build and run the fusion model: a
    different model reported under the baseline's name."""
    from causalcgm_mt.metabonet_loader import (build_metabonet_windows,
                                               load_metabonet_frame)
    cfg = DataConfig(history_min=120, horizon_min=30)
    df = load_metabonet_frame(parquet)
    ws = build_metabonet_windows(df, cfg, horizons_min=(30,))
    view = R.unimodal_view(ws)

    assert isinstance(ws, MetaboNetWindowSet) and hasattr(ws, "x_aux")
    assert not hasattr(view, "x_aux"), (
        "run_fold detects multimodality with hasattr(ws, 'x_aux')")
    # Same rows, in the same order, so the arms stay comparable.
    assert len(view) == len(ws)
    assert np.array_equal(view.x, ws.x)
    assert np.array_equal(view.y_reg, ws.y_reg)
    assert list(view.patient) == list(ws.patient)


def test_lora_requires_its_base_arm(parquet, tmp_path):
    rc = R.main(["--parquet-path", parquet, "--out", str(tmp_path), "--smoke",
                 "--arms", "lora"])
    assert rc == 2, "lora ran without the base model it adapts"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def test_summary_records_protocol_smoke_flag_and_provenance(smoke_run):
    payload, _out = smoke_run
    assert payload["smoke"] is True, (
        "a two-epoch run must be labelled, or it can be mistaken for a result")
    assert payload["protocol"] == "patient_wise_cv"
    assert payload["provenance"]["shipped_is_test_used"] is False
    assert payload["provenance"]["excluded_sources"] == ["OhioT1DM", "HUPA-UCM"]
    assert len(payload["provenance"]["aux_channels"]) == 24
    assert payload["per_fold"] and payload["summary"]


def test_every_arm_is_scored_and_the_rows_say_on_what(smoke_run):
    payload, _out = smoke_run
    by_arm = {r["arm"]: r for r in payload["summary"]}
    assert set(by_arm) == {"cgm_only", "fusion", "lora"}

    for name in ("cgm_only", "fusion"):
        block = by_arm[name]["per_horizon"]["30"]
        assert np.isfinite(block["regression"]["rmse"])
        assert np.isfinite(block["classification"]["pr_auc"])
        # The alarm block is the clinically load-bearing one; a smoke run that
        # silently drops it is not exercising the path it claims to.
        assert np.isfinite(block["alarm"]["fapd_at_5pct_fpr"])
        assert np.isfinite(block["regression"]["time_lag_min"])

    for r in payload["per_fold"]:
        assert r["scored_on"] in ("fold_test_rows",
                                  "per_subject_post_embargo_eval_regions")


def test_the_lora_arm_is_not_a_copy_of_its_base(smoke_run):
    """The defect this test exists for: scoring the adapted arm with the BASE
    model's predictions produces a row identical to `cgm_only` in every digit,
    and nothing in the output says so."""
    payload, _out = smoke_run
    by_arm = {r["arm"]: r for r in payload["summary"]}
    lora = by_arm["lora"]

    assert lora["per_horizon"] == {}, (
        "the LoRA arm was scored on the fold's full test rows, which include "
        "the prefix it adapted on")
    assert lora["scored_on"] == "per_subject_post_embargo_eval_regions"
    assert lora["paired_against"] == "cgm_only"

    pers = lora["personalization"]
    assert pers["n_subjects"] > 0, "no subject was adapted; the arm is vacuous"
    assert pers["n_trainable"] > 0
    for s in pers["per_subject"]:
        # A paired before/after that is bit-identical means the adaptation ran
        # but changed nothing -- which for LoRA (B initialised to zero) is what
        # a no-op looks like.
        assert s["rmse_before"] != s["rmse_after"], (
            f"{s['subject']}: adaptation left RMSE bit-identical")


def test_folds_hold_out_whole_subjects(smoke_run):
    payload, _out = smoke_run
    for r in payload["per_fold"]:
        assert r["test_subjects"], "a fold had no test subject"
        assert all(":" in s for s in r["test_subjects"])
        if r["scored_on"] == "fold_test_rows":
            fitted = set(r["scaler_fitted_on"])
            assert not (fitted & set(r["test_subjects"])), (
                "the scaler was fitted on a test subject")


def test_rendered_table_carries_the_protocol_in_its_caption(smoke_run):
    _payload, out = smoke_run
    # The smoke sidecar, not the manuscript's \input target -- a smoke run is
    # not allowed to touch the latter.
    with open(R.comparison_table_path(os.path.join(out, "benchmarks"),
                                      smoke=True)) as f:
        tex = f.read()
    assert "SMOKE RUN" in tex, "a smoke table must say so on its face"
    assert "patient-wise cross validation" in tex
    assert "is\\_test" in tex and "not used here" in tex
    assert "OhioT1DM and HUPA-UCM are excluded" in tex
    # Arm names are LaTeX-escaped in the rendered table.
    assert "cgm\\_only" in tex and "fusion" in tex
    # The LoRA arm must not appear as a peer row of the population arms.
    assert "\nlora " not in tex


def test_prediction_dumps_match_the_scored_rows(smoke_run):
    payload, out = smoke_run
    pred_dir = os.path.join(out, "benchmarks", "predictions")
    dumps = sorted(os.listdir(pred_dir))
    assert dumps, "no per-window dumps were written"
    assert not any(d.startswith("lora") for d in dumps), (
        "the LoRA arm has no fold-test-row predictions to dump")

    by_key = {(r["arm"], r["fold"]): r for r in payload["per_fold"]}
    for name in dumps:
        arm, _, fold = name[:-4].rpartition("_fold")
        d = np.load(os.path.join(pred_dir, name), allow_pickle=False)
        row = by_key[(arm, int(fold))]
        assert d["bg_hat"].shape[0] == row["n_test"]
        assert d["y_reg"].shape[0] == row["n_test"]
        assert set(np.unique(d["patient"])) == set(row["test_subjects"])


def test_a_fold_too_thin_to_train_is_refused_with_the_arithmetic(tmp_path):
    """`run_fold` holds an inner validation subject out of the TRAINING side, so
    a fold whose training side has one subject leaves none to fit on -- and
    fails deep in `CausalScaler.fit` with "No training windows to fit the
    scaler on", a message that says nothing about the fold arithmetic.

    Two subjects over two folds is the smallest case that reaches it: each fold
    tests one subject and trains on the other, whose single record is then
    entirely consumed by the inner validation split.
    """
    corpus = pd.concat([make_subject("StudyA", "1", n=700, seed=1),
                        make_subject("StudyA", "2", n=700, seed=2)],
                       ignore_index=True)
    p = tmp_path / "thin.parquet"
    corpus.to_parquet(p, row_group_size=400)

    rc = R.main(["--parquet-path", str(p), "--out", str(tmp_path / "out"),
                 "--smoke", "--folds", "2", "--epochs", "1", "--arms", "cgm_only"])
    assert rc == 2, "a fold with one training subject was allowed to run"


def test_a_window_cap_too_small_for_the_lora_arm_is_refused(parquet, tmp_path):
    """The cap and the personalisation protocol interact, and the failure mode
    is silence: every subject is skipped and the arm reports nothing.

    `adapt_patient` needs a wall-clock prefix of at least AdaptConfig.batch_size
    windows plus the embargo plus 50 evaluation windows, all from one subject's
    kept windows. A cap below that sum cannot satisfy it for any subject.
    """
    rc = R.main(["--parquet-path", parquet, "--out", str(tmp_path), "--smoke",
                 "--folds", "2", "--epochs", "1", "--prefix-hours", "24",
                 "--max-windows-per-subject", "100"])
    assert rc == 2, "a cap too small for the lora arm was allowed to run"


def test_block_thinning_preserves_the_anchor_cadence(parquet):
    """Every-k-th thinning would bound memory and silently empty the alarm
    metrics: `clinical.segment_ids` starts a new segment whenever consecutive
    anchors are more than one period apart, so thinned anchors make every
    window its own segment and lead time comes back nan."""
    from causalcgm_mt.metabonet_loader import (MetaboNetConfig,
                                               build_metabonet_windows,
                                               load_metabonet_frame)
    cfg = DataConfig(history_min=120, horizon_min=30)
    df = load_metabonet_frame(parquet)
    capped = build_metabonet_windows(
        df, cfg, MetaboNetConfig(max_windows_per_subject=300,
                                 thin_block_windows=150))

    period = np.timedelta64(cfg.sample_period_min, "m")
    for pid in capped.subjects():
        t = np.sort(capped.t_time[capped.patient == pid])
        assert len(t) <= 300
        gaps = np.diff(t)
        contiguous = int((gaps == period).sum())
        assert contiguous >= 0.8 * len(gaps), (
            f"{pid}: only {contiguous}/{len(gaps)} consecutive anchors are one "
            f"period apart -- the cadence was destroyed, and every alarm metric "
            f"computed on these rows would be empty")


# --------------------------------------------------------------------------- #
# 9. Resumability
# --------------------------------------------------------------------------- #
def test_a_second_run_reuses_completed_fold_checkpoints(parquet, tmp_path):
    """A run that dies on the last fold must not lose the earlier ones.

    Each (arm, fold) result is written the moment it completes, and a rerun of
    the same experiment adopts them instead of refitting.
    """
    out = str(tmp_path / "out")
    common = ["--parquet-path", parquet, "--out", out, "--smoke", "--folds", "2",
              "--epochs", "1", "--arms", "cgm_only", "--max-subjects", "6",
              "--window-cache", str(tmp_path / "wc")]

    assert R.main(common) == 0
    fold_dir = os.path.join(out, "benchmarks", "_folds")
    checkpoints = sorted(os.listdir(fold_dir))
    assert checkpoints == ["cgm_only_fold0.json", "cgm_only_fold1.json"]

    with open(os.path.join(fold_dir, checkpoints[0])) as f:
        first = json.load(f)
    assert first["run_fingerprint"], "a checkpoint with no fingerprint is unsafe"

    # Second run: same experiment, so the checkpoints are adopted verbatim.
    assert R.main(common) == 0
    with open(os.path.join(out, "benchmarks", "metabonet_summary.json")) as f:
        payload = json.load(f)
    reused = [r for r in payload["per_fold"] if r["arm"] == "cgm_only"]
    assert len(reused) == 2
    assert reused[0]["per_horizon"]["30"]["regression"]["rmse"] == \
        first["per_horizon"]["30"]["regression"]["rmse"], (
        "the reused row differs from the checkpoint it came from")


def test_checkpoints_from_a_different_experiment_are_ignored(parquet, tmp_path):
    """Resuming across a changed horizon, seed or fold count would assemble one
    table out of two experiments, and the table would not say so."""
    out = str(tmp_path / "out")
    base = ["--parquet-path", parquet, "--out", out, "--smoke", "--folds", "2",
            "--epochs", "1", "--arms", "cgm_only", "--max-subjects", "6",
            "--window-cache", str(tmp_path / "wc")]
    assert R.main(base) == 0

    fold_dir = os.path.join(out, "benchmarks", "_folds")
    with open(os.path.join(fold_dir, "cgm_only_fold0.json")) as f:
        original = json.load(f)

    # Same output directory, different seed: a different experiment.
    assert R.main(base + ["--seed", "4242"]) == 0
    with open(os.path.join(fold_dir, "cgm_only_fold0.json")) as f:
        rerun = json.load(f)
    assert rerun["run_fingerprint"] != original["run_fingerprint"], (
        "the seed change did not reach the fingerprint, so a stale checkpoint "
        "would have been adopted")


def test_a_changed_window_sample_invalidates_the_fold_checkpoints(monkeypatch):
    """The window version has to reach the checkpoint fingerprint too.

    `metabonet_loader._fingerprint` rebuilds the window STORE when the sample
    changes, but a fold checkpoint is a result, not a window: without the
    version in `run_fingerprint`, a rerun after the thinning fix would adopt
    rows computed from the old sample -- each subject's opening ~33 h -- and
    render them into the same table as freshly computed ones. The cap alone
    cannot stand in for it: it says how many windows, never which.
    """
    args = R.parse_args(["--parquet-path", "x.parquet", "--arms", "cgm_only"])
    keys = ["StudyA:1", "StudyA:2"]

    before = R.run_fingerprint(args, keys)
    monkeypatch.setattr(R, "METABONET_WINDOW_VERSION",
                        R.METABONET_WINDOW_VERSION + 1)
    assert R.run_fingerprint(args, keys) != before, (
        "a change to which windows are sampled leaves the fingerprint alone, "
        "so stale fold results would be adopted after the thinning fix")


def test_epoch_history_is_recorded_for_a_run_diagnosed_after_the_fact(smoke_run):
    payload, _out = smoke_run
    trained = [r for r in payload["per_fold"] if r["scored_on"] == "fold_test_rows"]
    assert trained
    for r in trained:
        assert r["epochs_run"] >= 1
        for h in r["history"]:
            assert {"epoch", "train_loss", "val_rmse", "val_pr_auc"} <= set(h)
            assert np.isfinite(h["train_loss"]), (
                "a non-finite training loss is the first sign of a diverged "
                "run, and the log is where it has to be visible")


def test_the_rendered_table_is_stamped_with_the_run_that_produced_it(smoke_run):
    """The table half of the appendix must say which run wrote it.

    `rebuild_paper_tables.check_metabonet_coupling` matches this stamp against
    the `\\NumMetabo*` macros rendered beside it. Without it the two halves of
    the appendix are only assumed to belong together -- which is how a smoke
    table came to sit under publication-budget prose (audit finding C1).
    """
    payload, out = smoke_run
    path = R.comparison_table_path(os.path.join(out, "benchmarks"), smoke=True)
    prov = B.read_latex_provenance(path)

    assert prov is not None, "the comparison table carries no provenance stamp"
    assert prov["fingerprint"] == payload["run_fingerprint"]
    assert prov["smoke"] is True, "a smoke run must stamp itself as one"
    # A LaTeX comment: it identifies the run without printing in the PDF.
    with open(path) as f:
        assert f.readline().startswith("%")
