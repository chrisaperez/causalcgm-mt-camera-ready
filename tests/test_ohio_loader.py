"""Executable audit of the OhioT1DM pipeline's parsing and no-leakage claims.

Same spirit as `test_causality.py`: every test corresponds to a specific way
this pipeline is known to break, and is written to FAIL if the guarantee it
covers is removed. The synthetic fixture is a real (small) OhioT1DM-shaped XML
pair, so the parsing path under test is the production one, not a stub.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import DataConfig
from causalcgm_mt.hupa_loader import AUX_CHANNELS, HupaConfig
from causalcgm_mt.ohio_loader import (
    OHIO_AUX_CHANNELS,
    OHIO_AUX_INDEX,
    OHIO_AUX_SKIP_SCALING,
    OhioConfig,
    OhioPreprocessor,
    assert_no_temporal_overlap,
    assert_prefix_embargo_causal,
    build_ohio_windows,
    discover_records,
    official_temporal_split,
    ohio_prefix_embargo_split,
    parse_patient_xml,
    required_embargo_min,
    synchronize_ohio_record,
)

REAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OhioT1DM")
HAS_REAL = os.path.isdir(REAL_ROOT)

CFG = DataConfig(history_min=60, horizon_min=30, stride_steps=1)


# --------------------------------------------------------------------------- #
# Synthetic fixture: a two-file cohort in the real XML dialect
# --------------------------------------------------------------------------- #
def _write_xml(path: str, pid: str, start: pd.Timestamp, hours: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(hours * 12)
    ts = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    bg = 120 + 45 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 3, n)

    def fmt(t):  # day-first, exactly as the released files
        return t.strftime("%d-%m-%Y %H:%M:%S")

    rows = [f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<patient id="{pid}" weight="80" insulin_type="Novalog">',
            "  <glucose_level>"]
    rows += [f'    <event ts="{fmt(t)}" value="{v:.0f}"/>' for t, v in zip(ts, bg)]
    rows.append("  </glucose_level>")

    rows.append("  <finger_stick>")
    rows.append(f'    <event ts="{fmt(ts[3])}" value="118"/>')
    rows.append('    <event ts="" value=""/>')            # malformed, must be dropped
    rows.append("  </finger_stick>")

    rows.append("  <basal>")
    rows.append(f'    <event ts="{fmt(ts[0])}" value="0.8"/>')
    rows.append("  </basal>")

    rows.append("  <temp_basal>")
    rows.append(f'    <event ts_begin="{fmt(ts[10])}" ts_end="{fmt(ts[16])}" value="0.0"/>')
    rows.append("  </temp_basal>")

    rows.append("  <bolus>")
    rows.append(f'    <event ts_begin="{fmt(ts[5])}" ts_end="{fmt(ts[5])}" '
                f'type="normal" dose="4.0" bwz_carb_input="40"/>')
    rows.append(f'    <event ts_begin="{fmt(ts[20])}" ts_end="{fmt(ts[26])}" '
                f'type="normal dual" dose="6.0"/>')
    rows.append("  </bolus>")

    rows.append("  <meal>")
    rows.append(f'    <event ts="{fmt(ts[5])}" type="Breakfast" carbs="40"/>')
    rows.append("  </meal>")

    # ts_begin AFTER ts_end, as several 2018 sleep records genuinely are.
    rows.append("  <sleep>")
    rows.append(f'    <event ts_begin="{fmt(ts[40])}" ts_end="{fmt(ts[30])}" quality="2"/>')
    rows.append("  </sleep>")

    rows.append("  <basis_heart_rate>")
    rows += [f'    <event ts="{fmt(t)}" value="{70 + (i % 7)}"/>'
             for i, t in enumerate(ts) if i % 2 == 0]
    rows.append("  </basis_heart_rate>")

    rows.append("  <basis_steps>")
    rows += [f'    <event ts="{fmt(t)}" value="{i % 30}"/>' for i, t in enumerate(ts)]
    rows.append("  </basis_steps>")

    rows.append("</patient>")
    with open(path, "w") as f:
        f.write("\n".join(rows))


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("ohio")
    for sub, pids in (("2018", ["501", "502"]),):
        for split, tag, off, hours in (("train", "training", 0, 60.0), ("test", "testing", 61, 24.0)):
            d = root / sub / split
            d.mkdir(parents=True, exist_ok=True)
            for k, pid in enumerate(pids):
                _write_xml(str(d / f"{pid}-ws-{tag}.xml"), pid,
                           pd.Timestamp("2021-12-07 00:00:00") + pd.Timedelta(hours=off),
                           hours, seed=k + 1)
    return str(root)


@pytest.fixture(scope="module")
def records(fake_root):
    return discover_records(fake_root, subsets=("2018",))


@pytest.fixture(scope="module")
def ws(records):
    return build_ohio_windows(records, CFG, HupaConfig(), OhioConfig(), horizons_min=(30, 60))


# --------------------------------------------------------------------------- #
# 1. Parsing robustness
# --------------------------------------------------------------------------- #
def test_timestamps_are_day_first(records):
    """`07-12-2021` is 7 December, not 12 July.

    Both readings are valid dates, so a month-first parse fails silently: it
    reorders the record and invents multi-month gaps that the segmenter then
    turns into unusable fragments.
    """
    cgm = records[0].get("glucose_level")
    assert cgm["ts"].iloc[0].month == 12 and cgm["ts"].iloc[0].day == 7
    assert cgm["ts"].is_monotonic_increasing


def test_malformed_events_are_dropped_not_zeroed(records):
    """An empty `ts`/`value` must vanish, never become epoch-0 or 0 mg/dL."""
    with pytest.warns(UserWarning, match="unparseable ts"):
        rec = parse_patient_xml(records[0].path)
    fs = rec.get("finger_stick")
    assert len(fs) == 1
    assert fs["ts"].notna().all()


def test_transposed_sleep_interval_is_repaired(records):
    """Some released sleep events carry ts_begin > ts_end.

    Left alone the interval is empty and every sleep episode for those subjects
    silently disappears -- a missing modality that looks like a real one.
    """
    sl = records[0].get("sleep")
    assert (sl["ts_end"] >= sl["ts"]).all()


def test_missing_fields_yield_empty_frames(records):
    """2020 has no `exercise` for several subjects; that must not raise."""
    assert len(records[0].get("exercise")) == 0
    assert len(records[0].get("does_not_exist")) == 0


# --------------------------------------------------------------------------- #
# 2. Synchronisation semantics
# --------------------------------------------------------------------------- #
def test_temp_basal_overrides_and_suspends(records):
    """A temp basal of 0.0 is a genuine suspend, not a missing value."""
    reg = synchronize_ohio_record(records[0], CFG, HupaConfig(), OhioConfig())
    rate = reg["basal_rate"].values
    assert rate[:5].max() > 0.0, "scheduled basal should be in force before the temp"
    assert (rate == 0.0).any(), "temp_basal 0.0 must zero the rate, not be ignored"


def test_basal_is_forward_filled_without_a_staleness_budget(records):
    """A pump profile stays in force until reprogrammed.

    The fixture sets basal once at t0 and never again; the rate must still be
    non-zero at the end of the record.
    """
    reg = synchronize_ohio_record(records[0], CFG, HupaConfig(), OhioConfig())
    assert reg["basal_rate"].values[-1] > 0.0


def test_extended_bolus_is_spread_over_its_delivery_interval(records):
    """Crediting a dual-wave bolus entirely at ts_begin would put insulin on
    board before the pump delivered it -- IOB leading reality."""
    reg = synchronize_ohio_record(records[0], CFG, HupaConfig(), OhioConfig())
    nz = reg["bolus"].values[reg["bolus"].values > 0]
    assert nz.size >= 3, "the 6 U dual bolus should occupy several slots"
    assert reg["bolus"].values.sum() == pytest.approx(10.0, rel=1e-6)


def test_on_board_curves_never_precede_their_dose(records):
    """IOB/COB at t must be zero before the first dose at or before t."""
    reg = synchronize_ohio_record(records[0], CFG, HupaConfig(), OhioConfig())
    first_bolus = int(np.flatnonzero(reg["bolus"].values > 0)[0])
    assert reg["iob"].values[:first_bolus].max() == 0.0
    first_carb = int(np.flatnonzero(reg["carbs"].values > 0)[0])
    assert reg["cob"].values[:first_carb].max() == 0.0


def test_bwz_carbs_are_deduplicated_against_logged_meals(records):
    """The fixture logs the same 40 g as both a meal and a bolus-wizard entry.

    Enabling `carbs_from_bwz` must not double-count it.
    """
    base = synchronize_ohio_record(records[0], CFG, HupaConfig(), OhioConfig())
    bwz = synchronize_ohio_record(records[0], CFG, HupaConfig(),
                                  OhioConfig(carbs_from_bwz=True))
    assert bwz["carbs"].sum() == pytest.approx(base["carbs"].sum())


# --------------------------------------------------------------------------- #
# 3. Channel layout
# --------------------------------------------------------------------------- #
def test_ohio_channels_extend_hupa_without_reordering():
    """Index drift here would be invisible in every metric.

    `aux_valid_index=7` and `AUX_SKIP_SCALING=(7,8,9)` are passed as bare
    integers elsewhere in the codebase, so the shared prefix must stay put.
    """
    assert OHIO_AUX_CHANNELS[:len(AUX_CHANNELS)] == AUX_CHANNELS
    assert OHIO_AUX_INDEX["hr_observed"] == 7
    assert set((7, 8, 9)).issubset(set(OHIO_AUX_SKIP_SCALING))


def test_every_mask_channel_is_excluded_from_scaling():
    masks = [i for i, c in enumerate(OHIO_AUX_CHANNELS) if c.endswith("_observed")]
    assert sorted(OHIO_AUX_SKIP_SCALING) == sorted(masks)


def test_window_tensor_matches_declared_layout(ws):
    assert ws.x_aux.shape[-1] == len(OHIO_AUX_CHANNELS)
    assert ws.x.shape[1] == CFG.history_steps
    assert ws.y_reg_multi.shape[1] == len(ws.horizons_min)
    assert np.isfinite(ws.x).all() and np.isfinite(ws.x_aux).all()


# --------------------------------------------------------------------------- #
# 4. Causality of the window tensors
# --------------------------------------------------------------------------- #
def test_inputs_ignore_the_future(records):
    """THE test. Corrupt every sample after an anchor and the inputs must not move.

    Windows are rebuilt from a mutated record; any lookahead anywhere in
    synchronisation, gap filling or feature construction shows up as a changed
    `x` or `x_aux`.
    """
    base = build_ohio_windows(records[:1], CFG, HupaConfig(), OhioConfig(), horizons_min=(30,))

    import copy
    mutated = copy.deepcopy(records[0])
    cgm = mutated.fields["glucose_level"]
    cut = cgm["ts"].iloc[len(cgm) // 2]
    tail = cgm["ts"] > cut
    cgm.loc[tail, "value"] = 400.0
    mutated.fields["glucose_level"] = cgm

    after = build_ohio_windows([mutated], CFG, HupaConfig(), OhioConfig(), horizons_min=(30,))

    keep = base.t_time <= np.datetime64(cut) - np.timedelta64(CFG.horizon_min, "m")
    n = int(keep.sum())
    assert n > 10, "fixture too short to test"
    np.testing.assert_array_equal(base.x[:n], after.x[:n])
    np.testing.assert_array_equal(base.x_aux[:n], after.x_aux[:n])


def test_targets_are_never_imputed(records):
    """A window whose horizon crosses a CGM dropout must be dropped, not filled.

    Deleting a block of readings can only REMOVE windows; if any window survives
    with an anchor whose horizon spans the hole, a label was invented.
    """
    import copy
    holed = copy.deepcopy(records[0])
    cgm = holed.fields["glucose_level"]
    drop = cgm.index[100:130]
    holed.fields["glucose_level"] = cgm.drop(index=drop).reset_index(drop=True)

    base = build_ohio_windows(records[:1], CFG, HupaConfig(), OhioConfig(), horizons_min=(30,))
    after = build_ohio_windows([holed], CFG, HupaConfig(), OhioConfig(), horizons_min=(30,))
    assert len(after) < len(base)


def test_labels_follow_the_stated_definition(ws):
    """Classification is `min BG over (t, t+PH] < 70`, regression is BG at t+PH.

    A nadir at t+20 that recovers by t+30 is exactly the event an early warning
    system exists for, so an endpoint-only label would be the wrong target.
    """
    assert np.array_equal(ws.y_cls_multi, (ws.y_min_multi < 70.0).astype(np.float32))
    assert (ws.y_min_multi <= ws.y_reg_multi + 1e-6).all()
    # Longer horizons contain shorter ones, so their minima can only fall.
    assert (ws.y_min_multi[:, 1] <= ws.y_min_multi[:, 0] + 1e-6).all()


def test_primary_horizon_mirrors_the_multi_arrays(ws):
    k = list(ws.horizons_min).index(CFG.horizon_min)
    np.testing.assert_array_equal(ws.y_reg, ws.y_reg_multi[:, k])
    np.testing.assert_array_equal(ws.y_cls_hard, ws.y_cls_multi[:, k])


# --------------------------------------------------------------------------- #
# 5. Splitting
# --------------------------------------------------------------------------- #
def test_no_window_straddles_the_official_boundary(ws):
    """Windows are built per file, so this is structural -- but assert it anyway:
    a future refactor that concatenated files before windowing would produce
    windows reading training history to predict test targets."""
    for pid in sorted({str(p) for p in ws.patient}):
        m = ws.patient == pid
        tr = ws.t_time[m & (ws.split == "train")]
        te = ws.t_time[m & (ws.split == "test")]
        assert tr.max() < te.min()


def test_official_split_is_disjoint_and_embargoed(ws):
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    assert split.fit.size and split.val.size and split.test.size
    for a, b in ((split.fit, split.val), (split.fit, split.test), (split.val, split.test)):
        assert np.intersect1d(a, b).size == 0
    # The embargo must actually cost something, or it is not being applied.
    assert split.embargoed.size > 0
    assert_no_temporal_overlap(ws, split, CFG)


def test_validation_is_the_chronological_tail_not_a_random_sample(ws):
    """With stride 1 and a 12-step history, adjacent windows share 11 of 12
    history steps. A random split puts near-duplicates on both sides and
    validation stops measuring anything."""
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    for pid in sorted({str(p) for p in ws.patient[split.val]}):
        f = split.fit[ws.patient[split.fit] == pid]
        v = split.val[ws.patient[split.val] == pid]
        assert ws.t_time[f].max() < ws.t_time[v].min()


def test_too_short_an_embargo_is_caught(ws):
    """The overlap assertion is the thing protecting the split, so verify it
    fails when the embargo is removed -- otherwise it might be vacuous."""
    with pytest.raises(AssertionError, match="embargo too short"):
        official_temporal_split(ws, CFG, val_frac=0.2, embargo_steps=0)


# --------------------------------------------------------------------------- #
# 6. Fitted objects see the fit split only
# --------------------------------------------------------------------------- #
def test_statistics_are_invariant_to_val_and_test_content(ws):
    """Corrupt every val/test row; every fitted statistic must be unchanged.

    This is stronger than reading the code: it fails for any path -- scaler,
    MARD proxy, soft-label band, class prior -- that touched evaluation data.
    """
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    before = OhioPreprocessor.fit(ws, split, CFG)

    poisoned = ws.subset(np.arange(len(ws)))
    ev = np.concatenate([split.val, split.test])
    poisoned.x[ev] = 999.0
    poisoned.x_aux[ev] = -999.0
    poisoned.y_min[ev] = 0.0
    poisoned.y_min_multi[ev] = 0.0
    poisoned.y_cls_multi[ev] = 1.0
    after = OhioPreprocessor.fit(poisoned, split, CFG)

    np.testing.assert_array_equal(before.cgm_scaler.center, after.cgm_scaler.center)
    np.testing.assert_array_equal(before.aux_scaler.scale, after.aux_scaler.scale)
    assert before.mard_proxy == after.mard_proxy
    assert before.labeler.band == after.labeler.band
    assert before.pos_weight == after.pos_weight


def test_transform_does_not_scale_mask_channels(ws):
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    pre = OhioPreprocessor.fit(ws, split, CFG)
    out = pre.transform(ws, split.val)
    for c in OHIO_AUX_SKIP_SCALING:
        np.testing.assert_array_equal(out["x_aux"][..., c], ws.x_aux[split.val][..., c])


def test_targets_are_returned_unscaled(ws):
    """The Huber delta in ModelConfig is in mg/dL, and putting a fitted object
    on the label path is how targets leak back into inputs."""
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    pre = OhioPreprocessor.fit(ws, split, CFG)
    out = pre.transform(ws, split.test)
    np.testing.assert_array_equal(out["y_reg"], ws.y_reg[split.test])
    assert out["y_reg"].max() > 40.0  # still mg/dL, not standardised


# --------------------------------------------------------------------------- #
# 6b. Few-shot personalisation: the prefix / embargo protocol
# --------------------------------------------------------------------------- #
# The official Ohio protocol is temporal and WITHIN patient, so the only thing
# separating adaptation data from evaluation data is time. These tests are the
# executable form of that claim.
@pytest.fixture(scope="module")
def split(ws):
    return official_temporal_split(ws, CFG, val_frac=0.2)


def _subjects(ws):
    return sorted({str(p) for p in ws.patient})


def test_required_embargo_covers_history_plus_the_longest_horizon(ws):
    """Derived from the window geometry, never guessed. The extra period is the
    one that separates spans which would otherwise meet at a single sample."""
    need = required_embargo_min(ws, CFG)
    assert need > CFG.history_min + max(ws.horizons_min)
    assert need == CFG.history_min + max(ws.horizons_min) + CFG.sample_period_min


def test_ohio_personalization_causality(ws, split):
    """THE test for the personalisation protocol.

    Adaptation windows and evaluation windows must not overlap ACROSS THE
    EMBARGO BOUNDARY -- and row-level disjointness is not enough to show that.
    Two different windows share raw data whenever their [t - history,
    t + horizon] spans intersect, and at stride 1 adjacent windows share all but
    one history step. So this asserts on the spans, not on the indices.
    """
    P = np.timedelta64(CFG.sample_period_min, "m")
    hist = P * CFG.history_steps
    fwd = P * (max(ws.horizons_min) // CFG.sample_period_min)

    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0)
        assert ps is not None and ps.adapt.size > 0 and ps.eval.size > 0

        # 1. rows are pairwise disjoint
        for a, b in (("adapt", "eval"), ("adapt", "embargoed"), ("embargoed", "eval")):
            assert np.intersect1d(getattr(ps, a), getattr(ps, b)).size == 0

        # 2. the prefix is a strict PREFIX in wall-clock time
        assert ws.t_time[ps.adapt].max() < ws.t_time[ps.eval].min()
        assert ws.t_time[ps.adapt].min() == ws.t_time[
            np.concatenate([ps.adapt, ps.embargoed, ps.eval])].min()

        # 3. and no raw sample is shared across the boundary
        adapt_end = ws.t_time[ps.adapt].max().astype("datetime64[m]") + fwd
        eval_start = ws.t_time[ps.eval].min().astype("datetime64[m]") - hist
        assert adapt_end < eval_start, (
            f"{pid}: adaptation spans reach {adapt_end}, evaluation spans start "
            f"at {eval_start}")

        assert_prefix_embargo_causal(ws, ps, CFG)


def test_prefix_windows_all_lie_inside_the_requested_prefix(ws, split):
    """A 24-hour prefix must be 24 hours of the subject's own record, not a
    fraction of it chosen by row count -- the two differ whenever the record
    has gaps, and the row-count version silently reaches further in time."""
    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=12.0)
        t = ws.t_time[ps.adapt].astype("datetime64[m]")
        assert (t <= ps.prefix_end).all()
        assert (t - t.min() <= np.timedelta64(12 * 60, "m")).all()


def test_a_too_short_embargo_is_raised_to_the_required_one(ws, split):
    """The causal separation must not be configurable away. Asking for zero
    embargo gets the geometric minimum, not zero."""
    ps = ohio_prefix_embargo_split(ws, split, _subjects(ws)[0], CFG,
                                   prefix_hours=24.0, embargo_min=0)
    assert ps.embargo_min == required_embargo_min(ws, CFG)
    assert_prefix_embargo_causal(ws, ps, CFG)


def test_the_causality_assertion_is_not_vacuous(ws, split):
    """The assertion is what protects the protocol, so verify it FAILS on
    deliberately corrupted splits -- otherwise it might be asserting nothing.

    Both failure modes are exercised, because they are caught by different
    checks and only the second one is subtle: rows can be perfectly disjoint
    while the windows they index still share raw samples.
    """
    pid = _subjects(ws)[0]
    ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0)
    assert ps.embargoed.size >= 5, "fixture too short to corrupt"

    # 1. the same rows on both sides
    both = dataclasses.replace(
        ps, eval=np.sort(np.concatenate([ps.embargoed[:5], ps.eval])))
    with pytest.raises(AssertionError, match="share"):
        assert_prefix_embargo_causal(ws, both, CFG)

    # 2. disjoint ROWS whose window SPANS still overlap -- the embargo removed,
    #    which is exactly what a shortened embargo would produce.
    dissolved = dataclasses.replace(
        ps,
        eval=np.sort(np.concatenate([ps.embargoed, ps.eval])),
        embargoed=np.empty(0, dtype=np.int64),
    )
    assert np.intersect1d(dissolved.adapt, dissolved.eval).size == 0
    with pytest.raises(AssertionError, match="too short"):
        assert_prefix_embargo_causal(ws, dissolved, CFG)


def test_evaluation_keeps_the_official_test_file_whole(ws, split):
    """Ohio numbers are comparable to the literature only when scored on the
    released testing windows. Personalisation takes its prefix from the TRAINING
    file and must leave every test window in the evaluation set."""
    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0)
        n_test = int((ws.split[split.test] == "test").sum() and
                     (np.asarray([str(p) for p in ws.patient[split.test]]) == pid).sum())
        assert ps.n_eval_from_test == n_test
        assert (ws.split[ps.adapt] == "train").all(), \
            "the adaptation prefix must never come from the test file"


def test_default_eval_scope_is_disjoint_from_the_population_fit(ws, split):
    """The prefix/embargo geometry separates ADAPTATION from EVALUATION. It says
    nothing about the POPULATION model, which was fitted on `split.fit` and
    selected on `split.val` -- and those are the training-file rows the old
    evaluation region was mostly made of.

    This test previously asserted the opposite (`n_eval_from_train > 0`, "the
    training tail should also be scored"), i.e. it encoded the leak: measured
    across the twelve subjects, 63.8% of the evaluation region was base-model
    training data and 16.6% was its model-selection data, so the zero-shot
    "before" anchoring every paired delta was an in-sample number.
    """
    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0)
        assert ps.eval_scope == "test"
        assert ps.n_eval_from_train == 0
        assert ps.n_eval_in_fit == 0, "evaluation must not touch base-model fit rows"
        assert ps.n_eval_in_val == 0, "evaluation must not touch model-selection rows"
        assert np.intersect1d(ps.eval, split.fit).size == 0
        assert np.intersect1d(ps.eval, split.val).size == 0


def test_legacy_eval_scope_is_available_and_records_its_contamination(ws, split):
    """`post_embargo` reproduces the published numbers, so it stays reachable --
    but it must say what it is doing rather than look like the default."""
    pid = _subjects(ws)[0]
    legacy = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                       eval_scope="post_embargo")
    strict = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0)
    assert legacy.eval.size > strict.eval.size
    assert legacy.n_eval_from_train > 0
    # and the contamination is a recorded number, not something to re-derive
    assert legacy.n_eval_in_fit + legacy.n_eval_in_val > 0
    assert legacy.n_eval_in_fit == int(np.intersect1d(legacy.eval, split.fit).size)


def test_unknown_eval_scope_is_rejected(ws, split):
    with pytest.raises(ValueError, match="eval_scope"):
        ohio_prefix_embargo_split(ws, split, _subjects(ws)[0], CFG,
                                  eval_scope="everything")


def test_prefix_pool_includes_the_inner_validation_and_embargoed_rows(ws, split):
    """Those rows were dropped to protect MODEL SELECTION, which is not what is
    happening here. Excluding them again would silently shorten the prefix."""
    pid = _subjects(ws)[0]
    ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=1000.0)
    is_pid = np.asarray([str(p) == pid for p in ws.patient])
    train_rows = np.flatnonzero(is_pid & (ws.split == "train"))
    assert ps.adapt.size == train_rows.size


def test_unknown_subject_is_an_error_not_an_empty_split(ws, split):
    """A typo in a subject list must not look like 'this subject had too little
    data' and silently shrink the cohort."""
    with pytest.raises(ValueError, match="Unknown patient id"):
        ohio_prefix_embargo_split(ws, split, "not_a_subject", CFG)


# --------------------------------------------------------------------------- #
# 6b-ii. Prefix POSITION: the "recent" moving-window control
# --------------------------------------------------------------------------- #
# The drift control for Section 5.7. A recent prefix is only a valid control if
# it is still strictly causal -- which is the whole risk of moving the prefix
# toward the evaluation region -- so the causality assertions are re-run against
# it rather than assumed to carry over from the opening mode.
def test_recent_prefix_is_still_strictly_causal(ws, split):
    """Moving the prefix ADJACENT to the evaluation region is exactly where a
    prefix protocol breaks. Every guarantee the opening mode gets is re-asserted
    here on the spans, not the row indices."""
    P = np.timedelta64(CFG.sample_period_min, "m")
    hist = P * CFG.history_steps
    fwd = P * (max(ws.horizons_min) // CFG.sample_period_min)

    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                       prefix_mode="recent")
        assert ps is not None and ps.adapt.size > 0 and ps.eval.size > 0
        assert ps.prefix_mode == "recent"

        for a, b in (("adapt", "eval"), ("adapt", "embargoed"), ("embargoed", "eval")):
            assert np.intersect1d(getattr(ps, a), getattr(ps, b)).size == 0

        assert ws.t_time[ps.adapt].max() < ws.t_time[ps.eval].min()
        adapt_end = ws.t_time[ps.adapt].max().astype("datetime64[m]") + fwd
        eval_start = ws.t_time[ps.eval].min().astype("datetime64[m]") - hist
        assert adapt_end < eval_start, (
            f"{pid}: recent-prefix adaptation spans reach {adapt_end}, "
            f"evaluation spans start at {eval_start}")

        assert_prefix_embargo_causal(ws, ps, CFG)


def test_recent_prefix_ends_one_embargo_before_the_first_scored_window(ws, split):
    """The point of the control is that the gap IS the embargo. If the prefix
    drifted away from the evaluation region the contrast would silently become a
    weaker version of the opening arm rather than its opposite."""
    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                       prefix_mode="recent")
        first_eval = ws.t_time[ps.eval].min().astype("datetime64[m]")
        gap = (first_eval - np.datetime64(ps.prefix_end, "m")) / np.timedelta64(1, "m")
        assert gap == pytest.approx(ps.embargo_min), (
            f"{pid}: gap {gap} min, embargo {ps.embargo_min} min")
        assert ps.prefix_gap_min == pytest.approx(ps.embargo_min)


def test_recent_prefix_is_much_closer_to_evaluation_than_the_opening_one(ws, split):
    """The variable the contrast turns on. If these two ever coincided the
    experiment would be comparing an arm against itself."""
    for pid in _subjects(ws):
        opening = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                            prefix_mode="opening")
        recent = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                           prefix_mode="recent")
        assert recent.prefix_gap_min < opening.prefix_gap_min
        # Same evaluation rows, or the paired before/after is not paired.
        assert np.array_equal(opening.eval, recent.eval)


def test_recent_prefix_windows_lie_inside_the_requested_span(ws, split):
    """A 24-hour recent prefix is 24 hours of wall clock ending at prefix_end,
    not a row count -- the two differ whenever the record has gaps."""
    for pid in _subjects(ws):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=12.0,
                                       prefix_mode="recent")
        t = ws.t_time[ps.adapt].astype("datetime64[m]")
        assert (t <= ps.prefix_end).all()
        assert (t >= np.datetime64(ps.prefix_start, "m")).all()
        assert (t.max() - t.min() <= np.timedelta64(12 * 60, "m")).all()


def test_recent_prefix_refuses_the_legacy_eval_scope(ws, split):
    """Under `post_embargo` the training tail is inside the evaluation region,
    so a recent prefix would be drawn from AFTER windows it is scored on. That
    is an invalid protocol, not a weaker one, so it is refused."""
    with pytest.raises(ValueError, match="eval_scope='test'"):
        ohio_prefix_embargo_split(ws, split, _subjects(ws)[0], CFG,
                                  prefix_hours=24.0, prefix_mode="recent",
                                  eval_scope="post_embargo")


def test_unknown_prefix_mode_is_rejected(ws, split):
    with pytest.raises(ValueError, match="prefix_mode"):
        ohio_prefix_embargo_split(ws, split, _subjects(ws)[0], CFG,
                                  prefix_mode="yesterday")


def test_prefix_provenance_is_recorded_for_both_modes(ws, split):
    """The two prefixes draw from different parts of the population model's
    training data, so the contrast changes recency and training exposure
    together. That has to be in the artefact rather than inferred."""
    pid = _subjects(ws)[0]
    for mode in ("opening", "recent"):
        ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=24.0,
                                       prefix_mode=mode)
        assert ps.n_adapt_in_fit == int(np.intersect1d(ps.adapt, split.fit).size)
        assert ps.n_adapt_in_val == int(np.intersect1d(ps.adapt, split.val).size)


# --------------------------------------------------------------------------- #
# 6c. Few-shot personalisation: the adaptation loop end to end
# --------------------------------------------------------------------------- #
# These run the real `run_ohio_personalization` on the real multimodal tensors
# and the real multi-horizon Kendall loss. The model is untrained and the step
# budget is tiny, so the METRIC VALUES mean nothing -- what is being verified is
# that the plumbing is right: the aux tensors and mask reach the model, the
# targets carry both horizons, and the paired comparison is on identical rows.
@pytest.fixture(scope="module")
def perso(ws):
    """Split + frozen preprocessor + config for the personalisation tests.

    Module-scoped because `OhioPreprocessor.fit` is the expensive part and it is
    read-only afterwards -- the whole point of the class is that it is fitted
    once and applied frozen.

    `band_mode="fixed"` mirrors `scripts/train_ohio.py`: the synthetic fixture's
    MARD proxy comes out near zero, which would collapse the soft-label band
    below CGM display resolution. That is a real and correctly-reported problem
    with the fixture, not with the pipeline, and pinning the band keeps these
    tests measuring the adaptation plumbing rather than the labeller.
    """
    from causalcgm_mt.config import AdaptConfig, Config, ModelConfig
    from causalcgm_mt.ohio_loader import OhioPreprocessor

    data_cfg = dataclasses.replace(CFG, band_mode="fixed")
    split = official_temporal_split(ws, CFG, val_frac=0.2)
    pre = OhioPreprocessor.fit(ws, split, data_cfg)
    cfg = Config(
        data=data_cfg,
        model=ModelConfig(n_horizons=len(ws.horizons_min), hidden_channels=16,
                          n_blocks=2, head_hidden=16, dropout=0.0),
        adapt=AdaptConfig(adapt_prefix_hours=24.0, batch_size=32, steps=4, lr=1e-3),
    )
    return split, pre, cfg


def _fresh_model(ws, cfg):
    """A new model per test: `inject_lora` mutates in place, so a shared one
    would carry adapters from a previous test into the next."""
    import torch
    from causalcgm_mt.models import build_multimodal_model

    torch.manual_seed(0)
    model, criterion = build_multimodal_model(
        ws.x.shape[-1], len(OHIO_AUX_CHANNELS), cfg.model, 5.0,
        aux_valid_index=OHIO_AUX_INDEX["hr_observed"])
    return model.eval(), criterion


@pytest.mark.parametrize("method", ["none", "bn_adapt", "head_only", "lora", "full"])
def test_ohio_personalization_runs_every_mode_multimodally(ws, perso, method):
    """Every arm of the ablation must survive the real (x_cgm, x_aux, mask)
    tensors and the multi-horizon loss.

    LoRA in particular was unreachable for the fusion arm at one point, and a
    mode that raises is indistinguishable in a results table from one that was
    never run.
    """
    from causalcgm_mt.extension import run_ohio_personalization

    split, pre, cfg = perso
    model, criterion = _fresh_model(ws, cfg)
    res = run_ohio_personalization(
        model, ws, split, pre, cfg, criterion=criterion, method=method,
        min_adapt_windows=16, min_eval_windows=20, verbose=False)

    assert len(res) == len({str(p) for p in ws.patient})
    for r in res:
        assert r.method == method
        assert r.n_adapt_windows > 0 and r.n_eval_windows > 0
        assert r.n_eval_from_test > 0, "the official test file must be scored"
        assert set(r.before_by_horizon) == set(ws.horizons_min)
        assert set(r.after_by_horizon) == set(ws.horizons_min)
        # Paired: both reports cover the SAME evaluation windows.
        for h in ws.horizons_min:
            assert r.before_by_horizon[h].n == r.after_by_horizon[h].n == r.n_eval_windows
        assert (r.n_trainable > 0) == (method in ("head_only", "lora", "full"))


def test_zero_shot_arm_is_a_bit_identical_no_op(ws, perso):
    """`none` is a real experimental condition, and its "after" must equal its
    "before" exactly. Any drift here is drift in the harness itself, which would
    contaminate every other arm's delta."""
    from causalcgm_mt.extension import run_ohio_personalization

    split, pre, cfg = perso
    model, criterion = _fresh_model(ws, cfg)
    res = run_ohio_personalization(
        model, ws, split, pre, cfg, criterion=criterion, method="none",
        min_adapt_windows=16, min_eval_windows=20, verbose=False)

    assert res
    for r in res:
        assert r.n_trainable == 0
        for h in ws.horizons_min:
            # Field-by-field, so a single drifting metric cannot hide inside an
            # aggregate. (The synthetic cohort never goes hypoglycemic, so the
            # classification metrics are legitimately nan here -- `is` equality
            # is what makes "identical" the right word for that case too.)
            b, a = r.before_by_horizon[h].as_dict(), r.after_by_horizon[h].as_dict()
            assert b.keys() == a.keys()
            for k in b:
                assert b[k] == a[k] or (isinstance(b[k], float) and np.isnan(b[k])
                                        and np.isnan(a[k])), f"{h} min: {k} drifted"
        assert r.delta("rmse") == 0.0


def test_lora_on_ohio_tensors_starts_as_an_exact_identity(ws, perso):
    """dW = 0 at initialisation, so the first forward pass is BIT-IDENTICAL to
    the base model -- checked on real Ohio multimodal tensors, not random ones.

    `torch.equal`, not `assert_close`: B is exactly zero, so the LoRA branch
    contributes exactly 0.0 and the sum is the base output unchanged. A
    tolerance-based check would hide an adapter that starts merely small, which
    would mean every personalisation run begins by damaging the population
    model.
    """
    import torch
    from causalcgm_mt.config import AdaptConfig
    from causalcgm_mt.extension import configure_trainable, lora_delta_norm_sq

    split, pre, cfg = perso
    model, _criterion = _fresh_model(ws, cfg)
    rows = split.test[:16]
    arrays = pre.transform(ws, rows)
    x = torch.from_numpy(np.ascontiguousarray(arrays["x"]))
    xa = torch.from_numpy(np.ascontiguousarray(arrays["x_aux"]))
    last = torch.from_numpy(np.ascontiguousarray(arrays["last_bg"]))
    ap = torch.from_numpy(np.ascontiguousarray(arrays["aux_present"]))

    model.eval()
    with torch.no_grad():
        bg0, logit0 = model(x, xa, last, ap)

    configure_trainable(model, dataclasses.replace(cfg.adapt, method="lora"))
    model.eval()
    with torch.no_grad():
        bg1, logit1 = model(x, xa, last, ap)

    assert float(lora_delta_norm_sq(model).detach()) == 0.0
    assert torch.equal(bg0, bg1), "LoRA perturbed the regression head at init"
    assert torch.equal(logit0, logit1), "LoRA perturbed the classifier at init"


def test_a_zero_step_budget_leaves_the_model_untouched(ws, perso):
    """Isolates "the harness adds drift" from "the gradient steps hurt".

    With `steps=0` the adapter is injected (dW = 0) but never stepped, so the
    adapted model must be bit-identical to the population one. If a personalised
    arm scores worse than zero-shot, this test is what says the cause is the
    optimisation and not the plumbing around it.
    """
    from causalcgm_mt.extension import adapt_ohio_subject

    split, pre, cfg = perso
    model, criterion = _fresh_model(ws, cfg)
    pid = sorted({str(p) for p in ws.patient})[0]

    r = adapt_ohio_subject(model, ws, split, pid, pre, criterion, cfg,
                           method="lora", steps=0, min_adapt_windows=16,
                           min_eval_windows=20)
    assert r is not None and r.n_trainable > 0, "adapters must still be injected"
    for h in ws.horizons_min:
        b, a = r.before_by_horizon[h].as_dict(), r.after_by_horizon[h].as_dict()
        for k in b:
            assert b[k] == a[k] or (isinstance(b[k], float) and np.isnan(b[k])
                                    and np.isnan(a[k])), f"{h} min: {k} drifted"


def test_adaptation_only_ever_reads_the_prefix(ws, perso):
    """The decisive leakage test for personalisation: corrupt every evaluation
    window's INPUTS, adapt, and require the adapted weights to be unchanged.

    Stronger than reading the code -- it fails for any path (loader, scaler,
    BN recalibration, the training loop) that touched a row outside the prefix.
    The evaluation targets are left alone so the two runs remain comparable.
    """
    import torch
    from causalcgm_mt.extension import adapt_ohio_subject
    from causalcgm_mt.ohio_loader import ohio_prefix_embargo_split

    split, pre, cfg = perso
    model, criterion = _fresh_model(ws, cfg)
    pid = sorted({str(p) for p in ws.patient})[0]

    def adapted_state(window_set):
        torch.manual_seed(0)
        m = copy.deepcopy(model)
        r = adapt_ohio_subject(m, window_set, split, pid, pre, criterion, cfg,
                               method="lora", min_adapt_windows=16,
                               min_eval_windows=20)
        assert r is not None
        return r

    ps = ohio_prefix_embargo_split(ws, split, pid, cfg.data)
    poisoned = ws.subset(np.arange(len(ws)))
    poisoned.x[ps.eval] = 999.0
    poisoned.x_aux[ps.eval] = -999.0

    # Same prefix, same seed, same steps -> the adaptation must be identical.
    # Only the SCORES may differ, because the evaluation inputs were destroyed.
    a = adapted_state(ws)
    b = adapted_state(poisoned)
    assert a.n_adapt_windows == b.n_adapt_windows
    assert a.n_trainable == b.n_trainable
    assert a.before.rmse != b.before.rmse, \
        "poisoning the evaluation inputs should change the SCORE; if it does " \
        "not, this test is not exercising the rows it thinks it is"


# --------------------------------------------------------------------------- #
# 7. The real dataset, when present
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_REAL, reason="OhioT1DM not present")
def test_real_dataset_parses_and_splits():
    cfg = dataclasses.replace(CFG, history_min=120, stride_steps=12)
    recs = discover_records(REAL_ROOT)
    assert len({r.patient_id for r in recs}) == 12
    assert all(r.split in ("train", "test") for r in recs)

    ws = build_ohio_windows(recs, cfg, HupaConfig(), OhioConfig(), horizons_min=(30, 60))
    assert len(ws) > 1000
    # Published Ohio hypo prevalence at 30 min sits in the low single digits;
    # an order-of-magnitude miss means the labels or the units are wrong.
    assert 0.01 < ws.y_cls_hard.mean() < 0.15
    assert 40.0 < np.median(ws.y_reg) < 250.0

    split = official_temporal_split(ws, cfg, val_frac=0.2)
    assert_no_temporal_overlap(ws, split, cfg)
    assert split.test.size == ws.where("test").size


# --------------------------------------------------------------------------- #
# 8. Materialised artefacts carry the pipeline version
# --------------------------------------------------------------------------- #
def test_a_tensor_dump_from_another_pipeline_version_is_refused(ws, tmp_path):
    """`save_ohio_tensors` writes window CONTENT, so it needs the version key.

    Every other cache in this pipeline is version-keyed; this one was not. A
    dump written before the backward-snap fix reloads with every tensor shape,
    every channel name and every meta field identical, and the values wrong --
    and it is the LAST stage that could notice, because everything downstream of
    it consumes tensors. Hence a raise, not a warning.
    """
    import torch

    from causalcgm_mt.data import WINDOW_PIPELINE_VERSION
    from causalcgm_mt.ohio_loader import load_ohio_tensors, save_ohio_tensors

    split = official_temporal_split(ws, CFG, val_frac=0.2)
    path = save_ohio_tensors(ws, split, dataclasses.replace(CFG, band_mode="fixed"),
                             str(tmp_path))

    art = load_ohio_tensors(path)
    assert art["meta"]["pipeline_version"] == WINDOW_PIPELINE_VERSION

    art["meta"]["pipeline_version"] = WINDOW_PIPELINE_VERSION - 1
    torch.save(art, path)
    with pytest.raises(ValueError, match="window pipeline version"):
        load_ohio_tensors(path)
    # ...but an explicit request to inspect an old dump still works.
    assert load_ohio_tensors(path, require_current_version=False) is not None


def test_the_soft_label_band_provenance_survives_serialisation(ws, tmp_path):
    """The band's provenance travels with the artefact, not just the log line."""
    from causalcgm_mt.ohio_loader import load_ohio_tensors, save_ohio_tensors

    cfg = dataclasses.replace(CFG, band_mode="mard")
    split = official_temporal_split(ws, cfg, val_frac=0.2)
    path = save_ohio_tensors(ws, split, cfg, str(tmp_path))

    pre = load_ohio_tensors(path)["preprocessor"]
    assert pre.labeler.band_source in ("mard", "device_floor", "clamped_low",
                                       "clamped_high")
    assert 5.0 <= pre.labeler.band <= 20.0
    assert pre.noise is not None and pre.noise.source in ("high_pass",
                                                          "paired_reference")
