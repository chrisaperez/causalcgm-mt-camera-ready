"""Executable audit of the MetaboNet loader.

Same standard as `tests/test_causality.py`: these are not smoke tests. Each one
corresponds to a specific way this loader could leak or could silently mix
subjects, and is written to FAIL if the corresponding guarantee is removed.

Everything runs on a small synthetic MetaboNet-shaped corpus written to a real
parquet, so the ingestion path (row-group streaming, schema validation, subject
scanning) is exercised without the 1.3 GB licensed file. The handful of tests
that need the real file are skipped when it is absent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config, DataConfig, ModelConfig, TrainConfig
from causalcgm_mt.data import CausalScaler
from causalcgm_mt.hupa_loader import AUX_CHANNELS, AUX_SKIP_SCALING
from causalcgm_mt import metabonet_loader as mn

REAL_PARQUET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metabonet_public.parquet")
HAS_REAL = os.path.exists(REAL_PARQUET)


# --------------------------------------------------------------------------- #
# Synthetic corpus, shaped exactly like the released file
# --------------------------------------------------------------------------- #
def make_subject(
    source: str,
    pid: str,
    n: int = 900,
    start: str = "2021-01-01",
    test_frac: float = 0.0,
    seed: int = 0,
    with_aux: bool = True,
    hypo: bool = True,
    freq: str = "5min",
) -> pd.DataFrame:
    """One subject in MetaboNet's RELEASED column names and semantics."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq=freq)
    g = 130 + 45 * np.sin(np.arange(n) / 19.0) + rng.normal(0, 3, n)
    if hypo:                       # guarantee both classes exist
        g[np.arange(n) % 173 < 4] = 58.0

    n_test = int(round(n * test_frac))
    is_test = np.zeros(n, bool)
    if n_test:
        is_test[n - n_test:] = True

    d = pd.DataFrame({
        "date": t,
        "CGM": g,
        "id": pid,
        "source_file": source,
        "is_test": is_test,
        "subject_split_across_traintest": bool(n_test),
        "insulin": np.nan,          # never ingested: it is basal + bolus
        "basal": np.full(n, 0.0375),          # UNITS PER BIN, not U/hr
        "bolus": np.where(rng.random(n) < 0.02, rng.uniform(1, 6, n), 0.0),
        "carbs": np.where(rng.random(n) < 0.02, rng.uniform(10, 60, n), 0.0),
    })
    aux = {
        "heartrate": 70 + rng.normal(0, 5, n),
        "steps": rng.poisson(20, n).astype(float),
        "galvanic_skin_response": rng.random(n),
        "skin_temp": 90 + rng.normal(0, 1, n),
        "air_temp": 70 + rng.normal(0, 2, n),
        "calories_burned": rng.random(n) * 3,
        "workout_intensity": np.full(n, np.nan),
        "workout_duration": np.full(n, np.nan),
    }
    for k, v in aux.items():
        d[k] = v if with_aux else np.nan
    return d


def canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw released names -> the canonical frame `load_metabonet_frame` returns."""
    schema = mn.MetaboNetSchema()
    d = raw.rename(columns=schema.rename_map(raw.columns)).copy()
    d["patient"] = [mn.subject_key(s, p)
                    for s, p in zip(d["source"].astype(str), d["patient"].astype(str))]
    d["time"] = pd.to_datetime(d["time"])
    d["is_test"] = d["is_test"].astype(bool)
    return d.sort_values(["patient", "time"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def cfg() -> DataConfig:
    return DataConfig(history_min=120, horizon_min=30)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    """Three sources, a colliding id, a temporal subject and a pure-test one."""
    return pd.concat([
        make_subject("StudyA", "1", seed=1),
        make_subject("StudyA", "2", seed=2, test_frac=0.3),
        make_subject("StudyB", "1", seed=3, test_frac=0.3),   # id collides
        make_subject("StudyB", "7", seed=4, test_frac=1.0),   # test_only
    ], ignore_index=True)


@pytest.fixture(scope="module")
def ws(corpus, cfg):
    return mn.build_metabonet_windows(canonical(corpus), cfg, horizons_min=(30, 60))


@pytest.fixture(scope="module")
def parquet_path(corpus, tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("metabonet") / "synthetic.parquet"
    corpus.to_parquet(p, row_group_size=500)     # several row groups on purpose
    return str(p)


# --------------------------------------------------------------------------- #
# 1. The decisive causality test
# --------------------------------------------------------------------------- #
def test_future_perturbation_changes_targets_but_not_inputs(corpus, cfg):
    """Corrupt the tail of every subject's record; windows anchored before the
    corruption must have BYTE-IDENTICAL inputs.

    This is the mirror of `test_causality.py`'s decisive test, and it is the one
    that breaks if any stage of this loader reads forward: a two-sided
    interpolation, a centred rolling statistic, a backward fill, a
    `resample()` left the pandas default, or a target array that leaks into a
    feature. Both CGM and an auxiliary stream are corrupted, so the aux path is
    covered too -- x_aux is checked byte-for-byte alongside x.
    """
    clean = canonical(corpus)
    cut = clean["time"].min() + pd.Timedelta(hours=48)

    dirty = clean.copy()
    tail = dirty["time"] >= cut
    dirty.loc[tail, "glucose"] = dirty.loc[tail, "glucose"] + 120.0
    dirty.loc[tail, "heart_rate"] = dirty.loc[tail, "heart_rate"] + 60.0
    dirty.loc[tail, "steps"] = dirty.loc[tail, "steps"] + 500.0

    a = mn.build_metabonet_windows(clean, cfg)
    b = mn.build_metabonet_windows(dirty, cfg)

    horizon = pd.Timedelta(minutes=cfg.horizon_min)
    key_a = {(str(p), pd.Timestamp(t)): i
             for i, (p, t) in enumerate(zip(a.patient, a.t_time))}
    key_b = {(str(p), pd.Timestamp(t)): i
             for i, (p, t) in enumerate(zip(b.patient, b.t_time))}
    shared = sorted(set(key_a) & set(key_b))
    assert shared, "no windows survived both builds"

    n_before = n_after = 0
    for k in shared:
        i, j = key_a[k], key_b[k]
        anchor = k[1]
        if anchor + horizon < cut:
            # Entirely before the corruption: inputs AND targets untouched.
            assert np.array_equal(a.x[i], b.x[j]), (
                f"window at {anchor} reads the future: CGM features changed when "
                f"data at/after {cut} was corrupted")
            assert np.array_equal(a.x_aux[i], b.x_aux[j]), (
                f"window at {anchor} reads the future through an AUX channel")
            assert a.y_reg[i] == b.y_reg[j]
            n_before += 1
        elif anchor >= cut:
            # Entirely inside it: targets must have moved, or the perturbation
            # never landed and this test proves nothing.
            n_after += 1

    assert n_before > 50, f"only {n_before} pre-corruption windows compared"
    assert n_after > 50, f"only {n_after} post-corruption windows compared"

    post = [(key_a[k], key_b[k]) for k in shared if k[1] >= cut]
    changed = sum(1 for i, j in post if a.y_reg[i] != b.y_reg[j])
    assert changed == len(post), (
        "targets inside the corrupted region did not all change -- the "
        "perturbation did not reach the targets, so the input check above is "
        "vacuous")


def test_history_and_target_index_ranges_are_disjoint(corpus, cfg):
    """The window geometry itself: inputs [t-H+1, t], targets [t+1, t+PH].

    Checked against the raw series rather than by re-deriving the slices: the
    last history step must equal the anchor's own reading, and the target must
    equal the reading one horizon later, with no shared index.
    """
    clean = canonical(corpus)
    ws = mn.build_metabonet_windows(clean, cfg)
    H = cfg.history_steps

    for i in np.linspace(0, len(ws) - 1, 40).astype(int):
        pid, anchor = str(ws.patient[i]), pd.Timestamp(ws.t_time[i])
        series = clean[clean["patient"] == pid].set_index("time")["glucose"]

        # x[..., 0] is the raw value channel; its last step is the anchor.
        assert ws.x[i, -1, 0] == pytest.approx(series.loc[anchor], abs=1e-4)
        assert ws.last_bg[i] == pytest.approx(series.loc[anchor], abs=1e-4)

        tgt_time = anchor + pd.Timedelta(minutes=cfg.horizon_min)
        assert ws.y_reg[i] == pytest.approx(series.loc[tgt_time], abs=1e-4)

        first_hist = anchor - pd.Timedelta(minutes=cfg.history_min - cfg.sample_period_min)
        assert ws.x[i, 0, 0] == pytest.approx(series.loc[first_hist], abs=1e-4)
        assert first_hist < anchor < tgt_time


def test_targets_are_never_imputed(cfg):
    """A window with any unobserved target step is DROPPED, not filled.

    Imputing a label is leakage by definition, and it is the one gap the history
    fill must never be extended to cover.
    """
    raw = make_subject("StudyA", "1", n=600, seed=11)
    d = canonical(raw)
    # Punch a hole that can only ever be a target: short enough that a history
    # window would happily interpolate it.
    hole = d.index[300:302]
    d.loc[hole, "glucose"] = np.nan
    hole_times = set(pd.Timestamp(t) for t in d.loc[hole, "time"])

    ws = mn.build_metabonet_windows(d, cfg)
    horizon = pd.Timedelta(minutes=cfg.horizon_min)
    period = pd.Timedelta(minutes=cfg.sample_period_min)

    for i in range(len(ws)):
        anchor = pd.Timestamp(ws.t_time[i])
        span = pd.date_range(anchor + period, anchor + horizon, freq=period)
        assert not (set(span) & hole_times), (
            f"window at {anchor} has a target step at a missing reading -- the "
            f"label was imputed")


def test_history_gap_fill_is_bounded_and_backward(cfg):
    """A long history gap drops the window; a short one is interpolated between
    two endpoints that are BOTH inside the history slice."""
    raw = make_subject("StudyA", "1", n=600, seed=12)
    d = canonical(raw)
    long_gap = d.index[200:210]           # 50 min > max_interpolation_gap_min
    d.loc[long_gap, "glucose"] = np.nan
    gap_times = set(pd.Timestamp(t) for t in d.loc[long_gap, "time"])

    ws = mn.build_metabonet_windows(d, cfg)
    period = pd.Timedelta(minutes=cfg.sample_period_min)
    hist = pd.Timedelta(minutes=cfg.history_min - cfg.sample_period_min)

    for i in range(len(ws)):
        anchor = pd.Timestamp(ws.t_time[i])
        span = pd.date_range(anchor - hist, anchor, freq=period)
        assert not (set(span) & gap_times), (
            f"window at {anchor} spans a {len(long_gap)*5}-minute gap that "
            f"exceeds max_interpolation_gap_min")


def test_a_sample_after_the_slot_never_lands_in_it(cfg):
    """Backward binning: slot t summarises (t-P, t], never [t, t+P).

    The pandas default (`closed="left", label="left"`) attributes up to a full
    period of FUTURE data to the present slot. Off-grid timestamps are what make
    the difference observable, so the fixture is deliberately off-grid.
    """
    raw = make_subject("StudyA", "1", n=400, seed=13)
    raw["date"] = raw["date"] + pd.Timedelta(seconds=150)   # half a period late
    d = canonical(raw)

    reg = mn.synchronize_subject(d, cfg)
    grid = pd.DatetimeIndex(reg["time"])
    src = d.set_index("time")["heart_rate"]

    for slot in range(50, 60):
        t = grid[slot]
        if reg["hr_observed"].iloc[slot] < 0.5:
            continue
        window = src[(src.index > t - pd.Timedelta(minutes=cfg.sample_period_min))
                     & (src.index <= t)]
        assert len(window), f"slot {t} is marked observed with no sample in (t-P, t]"
        assert reg["heart_rate"].iloc[slot] == pytest.approx(window.mean(), rel=1e-9), (
            f"slot {t} does not equal the mean of its BACKWARD bin -- it is "
            f"reading forward")


# --------------------------------------------------------------------------- #
# 2. Subject identity: the composite key
# --------------------------------------------------------------------------- #
def test_colliding_ids_from_different_sources_stay_different_subjects(ws):
    """`id` is not a subject key: 241 of MetaboNet's 1,291 ids appear under more
    than one source_file, and those are different people."""
    subjects = ws.subjects()
    assert "StudyA:1" in subjects and "StudyB:1" in subjects
    assert len({s for s in subjects}) == 4

    a = ws.patient == "StudyA:1"
    b = ws.patient == "StudyB:1"
    assert a.sum() and b.sum()
    assert not (a & b).any()
    # And the raw id alone is never used as a patient label anywhere.
    assert not any(str(p) == "1" for p in ws.patient)


def test_patient_wise_split_rejects_a_bare_id(ws, cfg):
    with pytest.raises(ValueError, match="composite"):
        mn.patient_wise_metabonet_split(ws, cfg, held_out=["1"])


def test_folds_group_by_subject_key_and_optionally_by_source(ws):
    folds = mn.metabonet_folds(ws, n_folds=2, seed=0)
    for train, test in folds:
        assert not (set(map(str, train)) & set(map(str, test)))
        assert all(":" in str(k) for k in list(train) + list(test))

    by_source = mn.metabonet_folds(ws, n_folds=2, seed=0, group_by_source=True)
    for train, test in by_source:
        src_train = {mn.split_subject_key(k)[0] for k in train}
        src_test = {mn.split_subject_key(k)[0] for k in test}
        assert not (src_train & src_test), (
            "a source_file appears on both sides: subjects within one source "
            "share a device and protocol, so this leaks study identity")


# --------------------------------------------------------------------------- #
# 3. Splitting
# --------------------------------------------------------------------------- #
def test_no_window_straddles_the_official_boundary(corpus, cfg, ws):
    """Segments break at the train/test boundary, so a window is entirely on
    one side of it by construction."""
    clean = canonical(corpus)
    period = pd.Timedelta(minutes=cfg.sample_period_min)
    hist = pd.Timedelta(minutes=cfg.history_min - cfg.sample_period_min)
    horizon = pd.Timedelta(minutes=max(ws.horizons_min))

    starts = {}
    for pid, g in clean.groupby("patient"):
        te = g.loc[g["is_test"], "time"]
        if len(te):
            starts[str(pid)] = te.min()

    for i in range(len(ws)):
        pid = str(ws.patient[i])
        if pid not in starts:
            continue
        anchor = pd.Timestamp(ws.t_time[i])
        span_lo, span_hi = anchor - hist, anchor + horizon
        boundary = starts[pid]
        if ws.split[i] == "train":
            assert span_hi < boundary, (
                f"train window at {anchor} reaches {span_hi}, past the test "
                f"boundary at {boundary}")
        else:
            assert span_lo >= boundary, (
                f"test window at {anchor} reads back to {span_lo}, before the "
                f"test boundary at {boundary}")


def test_official_split_is_disjoint_and_embargoed(ws, cfg):
    split = mn.official_metabonet_split(ws, cfg)
    mn.assert_disjoint(split)
    mn.assert_no_temporal_overlap(ws, split, cfg)      # raises on failure

    assert split.fit.size and split.val.size and split.test.size
    assert split.embargoed.size > 0, (
        "nothing was embargoed; with a temporal subject in the corpus the "
        "fit/val cut alone should have dropped windows")
    allocated = np.concatenate([split.fit, split.val, split.test, split.embargoed])
    assert allocated.size == np.unique(allocated).size


def test_embargo_removal_makes_the_overlap_assertion_fire(ws, cfg):
    """The embargo is load-bearing, not decorative: with it set to zero the
    overlap check must FAIL. A guarantee whose removal changes nothing is not a
    guarantee."""
    with pytest.raises(AssertionError, match="embargo too short"):
        mn.official_metabonet_split(ws, cfg, embargo_steps=0)


def test_val_is_the_chronological_tail_not_a_random_sample(ws, cfg):
    """A random inner split would place near-duplicate windows on both sides:
    at stride 1 adjacent windows share 23 of 24 history steps."""
    split = mn.official_metabonet_split(ws, cfg)
    for pid in {str(p) for p in ws.patient[split.val]}:
        f = split.fit[ws.patient[split.fit] == pid]
        v = split.val[ws.patient[split.val] == pid]
        if f.size == 0 or v.size == 0:
            continue
        assert ws.t_time[f].max() < ws.t_time[v].min(), (
            f"subject {pid}: a fit window is anchored after a val window")


def test_patient_wise_split_moves_whole_subjects(ws, cfg):
    split = mn.patient_wise_metabonet_split(ws, cfg, held_out=["StudyA:1"])
    assert set(map(str, ws.patient[split.test])) == {"StudyA:1"}
    assert "StudyA:1" not in set(map(str, ws.patient[split.fit]))
    assert "StudyA:1" not in set(map(str, ws.patient[split.val]))
    assert split.protocol == "patient_wise"
    mn.assert_disjoint(split)


def test_interleaved_subjects_are_quarantined(cfg):
    """The three CTR3 subjects whose train and test rows overlap in time are the
    reason nothing here trusts `subject_split_across_traintest`."""
    raw = make_subject("CTR3", "10", n=700, seed=21)
    # Interleave: alternate blocks of test rows through the whole record, and
    # leave the shipped flag saying False, exactly as the real file does.
    raw["is_test"] = (np.arange(len(raw)) // 50) % 2 == 1
    raw["subject_split_across_traintest"] = False
    d = canonical(pd.concat([raw, make_subject("StudyA", "1", seed=22)],
                            ignore_index=True))

    dropped = mn.build_metabonet_windows(d, cfg)
    assert "CTR3:10" not in dropped.subjects(), (
        "an interleaved subject survived: its own official split overlaps in "
        "time, so no temporal holdout of it is valid")
    assert "StudyA:1" in dropped.subjects()

    kept = mn.build_metabonet_windows(
        d, cfg, mn.MetaboNetConfig(on_interleaved="train"))
    assert "CTR3:10" in kept.subjects()
    assert set(kept.split[kept.patient == "CTR3:10"]) == {"train"}, (
        "on_interleaved='train' must force the whole subject onto the train "
        "side, not preserve the broken split")

    with pytest.raises(ValueError, match="interleaved"):
        mn.build_metabonet_windows(
            d, cfg, mn.MetaboNetConfig(on_interleaved="raise"))


# --------------------------------------------------------------------------- #
# 4. Benchmark contamination
# --------------------------------------------------------------------------- #
def test_benchmark_sources_are_excluded_by_default():
    assert mn.MetaboNetConfig().exclude_sources == ("OhioT1DM", "HUPA-UCM")


def test_contamination_guard_refuses_ohio_and_hupa():
    with pytest.raises(AssertionError, match="OhioT1DM"):
        mn.assert_no_benchmark_contamination(["OhioT1DM:559", "StudyA:1"])
    with pytest.raises(AssertionError, match="HUPA-UCM"):
        mn.assert_no_benchmark_contamination(["HUPA-UCM:3"])
    # Clean sets pass, and the waiver is per source and explicit.
    mn.assert_no_benchmark_contamination(["StudyA:1", "StudyB:2"])
    mn.assert_no_benchmark_contamination(["OhioT1DM:559"], allow=("OhioT1DM",))
    with pytest.raises(AssertionError, match="HUPA-UCM"):
        mn.assert_no_benchmark_contamination(
            ["OhioT1DM:559", "HUPA-UCM:3"], allow=("OhioT1DM",))


def test_scan_excludes_benchmark_sources_from_a_real_parquet(tmp_path):
    corpus = pd.concat([
        make_subject("StudyA", "1", n=300, seed=31),
        make_subject("OhioT1DM", "559", n=300, seed=32),
        make_subject("HUPA-UCM", "3", n=300, seed=33),
    ], ignore_index=True)
    p = tmp_path / "with_benchmarks.parquet"
    corpus.to_parquet(p, row_group_size=200)

    idx = mn.scan_subjects(str(p))
    assert idx.keys() == ["StudyA:1"]
    assert "OhioT1DM" not in idx.sources()

    # Opting in is possible, and then the guard is what stops a silent run.
    permissive = mn.MetaboNetConfig(exclude_sources=())
    idx_all = mn.scan_subjects(str(p), mcfg=permissive)
    assert set(idx_all.sources()) == {"StudyA", "OhioT1DM", "HUPA-UCM"}
    with pytest.raises(AssertionError, match="benchmark cohorts"):
        mn.assert_no_benchmark_contamination(idx_all)


# --------------------------------------------------------------------------- #
# 5. Ingestion path
# --------------------------------------------------------------------------- #
def test_scan_subjects_classifies_from_rows_not_from_the_flag(parquet_path):
    idx = mn.scan_subjects(parquet_path)
    cls = dict(zip(idx.frame.key, idx.frame.split_class))
    assert cls["StudyA:1"] == "train_only"
    assert cls["StudyA:2"] == "temporal"
    assert cls["StudyB:1"] == "temporal"
    assert cls["StudyB:7"] == "test_only"
    assert set(idx.keys(classes=("temporal",))) == {"StudyA:2", "StudyB:1"}


def test_scan_reports_a_flag_that_contradicts_the_rows(tmp_path):
    raw = make_subject("CTR3", "10", n=300, seed=41, test_frac=0.3)
    raw["subject_split_across_traintest"] = False        # the released lie
    p = tmp_path / "flag.parquet"
    raw.to_parquet(p, row_group_size=100)

    idx = mn.scan_subjects(str(p))
    assert idx.frame.loc[0, "split_class"] == "temporal"
    assert not idx.frame.loc[0, "flag_says_split"]
    assert list(idx.disagreements.key) == ["CTR3:10"], (
        "a flag contradicting its own rows must surface, not be silently used")


def test_frame_loading_streams_and_keys_by_subject(parquet_path):
    df = mn.load_metabonet_frame(parquet_path, keys=["StudyA:1", "StudyB:1"])
    assert set(df["patient"]) == {"StudyA:1", "StudyB:1"}
    assert {"time", "glucose", "heart_rate", "basal", "bolus", "carbs"} <= set(df.columns)
    assert df["time"].is_monotonic_increasing is False       # per patient, not global
    for _, g in df.groupby("patient"):
        assert g["time"].is_monotonic_increasing


def test_implausible_cgm_is_masked_not_dropped(cfg, tmp_path):
    """A 1.8 mg/dL reading (the real file's T1D-UOM slice has them) is not a
    measurement. Dropping the ROW would close the gap in time and let it be
    interpolated into a plausible value; blanking the reading keeps the gap
    visible to the window filter."""
    raw = make_subject("StudyA", "1", n=300, seed=51)
    raw.loc[100:104, "CGM"] = 1.8
    p = tmp_path / "implausible.parquet"
    raw.to_parquet(p, row_group_size=150)

    df = mn.load_metabonet_frame(str(p), keys=["StudyA:1"])
    assert len(df) == 300, "rows were dropped rather than masked"
    assert df["glucose"].isna().sum() == 5
    assert not (df["glucose"] < 20).any()


def test_missing_required_column_raises_with_what_it_found(tmp_path):
    raw = make_subject("StudyA", "1", n=100, seed=61).drop(columns=["CGM"])
    p = tmp_path / "broken.parquet"
    raw.to_parquet(p)
    with pytest.raises(ValueError, match="missing required MetaboNet column"):
        mn.scan_subjects(str(p))


def test_end_to_end_entry_point(parquet_path, cfg):
    ws, split = mn.load_metabonet(parquet_path, cfg, verbose=False)
    assert len(ws) > 0
    assert split.protocol == "official"
    mn.assert_disjoint(split)
    mn.assert_no_temporal_overlap(ws, split, cfg)
    assert "MetaboNet window set" in mn.describe(ws)


# --------------------------------------------------------------------------- #
# 6. Channel layout, units and graceful degradation
# --------------------------------------------------------------------------- #
def test_first_eleven_aux_channels_are_byte_identical_to_hupa():
    """`train.run_fold` and the model read these indices by position. Appending
    at the end is what keeps an Ohio, HUPA or MetaboNet tensor mutually
    intelligible."""
    assert mn.METABONET_AUX_CHANNELS[:11] == AUX_CHANNELS
    assert mn.METABONET_AUX_INDEX["hr_observed"] == 7
    assert AUX_SKIP_SCALING == (7, 8, 9)
    assert mn.METABONET_AUX_SKIP_SCALING[:3] == AUX_SKIP_SCALING
    assert len(mn.METABONET_AUX_SKIP_SCALING) > 3, "extra masks must be skipped too"
    for i in mn.METABONET_AUX_SKIP_SCALING:
        assert mn.METABONET_AUX_CHANNELS[i].endswith("_observed")


def test_basal_is_units_per_bin_not_units_per_hour(cfg):
    """Verified against the OhioT1DM XML this repo ships: MetaboNet reports
    subject 596's basal as 0.0375 where the source says 0.45 U/hr.

    Reading it as a rate would understate insulin on board by 12x."""
    raw = make_subject("StudyA", "1", n=400, seed=71)
    raw["bolus"] = 0.0                       # isolate the basal contribution
    reg = mn.synchronize_subject(canonical(raw), cfg)

    assert reg["basal_dose"].iloc[50] == pytest.approx(0.0375, rel=1e-6)
    assert reg["basal_rate"].iloc[50] == pytest.approx(0.45, rel=1e-6)
    # IOB from basal alone accumulates and is causal (monotone from zero).
    assert reg["basal_iob"].iloc[0] < reg["basal_iob"].iloc[100]
    assert reg["iob"].iloc[100] == 0.0, "bolus IOB must be zero when no bolus was given"


def test_insulin_column_is_never_ingested(cfg):
    """`insulin == basal + bolus` in the released file. Ingesting it alongside
    its own components would double count."""
    raw = make_subject("StudyA", "1", n=300, seed=72)
    raw["insulin"] = 999.0
    reg = mn.synchronize_subject(canonical(raw), cfg)
    assert "insulin" not in reg.columns
    assert (reg[["iob", "basal_iob", "bolus"]].to_numpy() < 999.0).all()


def test_absent_sleep_is_masked_not_invented(cfg, ws):
    """MetaboNet has no sleep column. The channel stays in the layout at index
    2, permanently zero with its mask clear -- an honestly absent modality
    rather than a renumbered tensor."""
    sleep = mn.METABONET_AUX_INDEX["sleep"]
    mask = mn.METABONET_AUX_INDEX["sleep_observed"]
    assert np.all(ws.x_aux[:, :, sleep] == 0.0)
    assert np.all(ws.x_aux[:, :, mask] == 0.0)


def test_a_subject_with_no_aux_degrades_to_the_unimodal_baseline(cfg):
    """Requirement 3: missing auxiliary streams must not break ingestion, and
    must reach the model as cleared masks so the gated residual fusion reduces
    exactly to the CGM-only path."""
    d = canonical(pd.concat([
        make_subject("StudyA", "1", seed=81, with_aux=True),
        make_subject("StudyB", "9", seed=82, with_aux=False),
    ], ignore_index=True))

    ws = mn.build_metabonet_windows(d, cfg)
    assert "StudyB:9" in ws.subjects(), "a CGM-only subject was silently dropped"

    bare = ws.patient == "StudyB:9"
    rich = ws.patient == "StudyA:1"
    assert ws.aux_frac[bare].max() == 0.0
    assert ws.aux_frac[rich].min() > 0.0

    # Every mask channel clear, and every value channel it gates at zero: an
    # absent modality, not a zero-valued present one.
    for name in mn.METABONET_AUX_CHANNELS:
        if name.endswith("_observed"):
            assert np.all(ws.x_aux[bare][:, :, mn.METABONET_AUX_INDEX[name]] == 0.0)
    for name in ("heart_rate", "steps", "gsr", "skin_temp", "air_temp", "calories"):
        assert np.all(ws.x_aux[bare][:, :, mn.METABONET_AUX_INDEX[name]] == 0.0), (
            f"{name} carries a value while its mask is clear -- the signature "
            f"of a fill that invented data")

    # `aux_present` is what train.run_fold passes to the fusion gate.
    aux_present = (ws.aux_frac > 0).astype(np.float32)
    assert aux_present[bare].sum() == 0.0
    assert aux_present[rich].all()


def test_gated_fusion_with_absent_aux_equals_the_cgm_only_path(cfg):
    """The bound the gated residual design promises: an absent modality reduces
    the multimodal model to its CGM branch exactly, so MetaboNet's sparser
    subjects cannot do worse than the unimodal baseline."""
    import torch
    from causalcgm_mt.models import build_multimodal_model

    d = canonical(make_subject("StudyB", "9", n=400, seed=83, with_aux=False))
    ws = mn.build_metabonet_windows(d, cfg)

    mask_idx = tuple(i for i, c in enumerate(ws.aux_channels) if c.endswith("_observed"))
    model, _ = build_multimodal_model(
        ws.x.shape[-1], ws.x_aux.shape[-1], ModelConfig(), 1.0,
        aux_valid_index=mask_idx, modality_dropout=0.0)
    model.eval()

    x = torch.from_numpy(ws.x[:16]).float()              # (B, T, C_cgm)
    aux = torch.from_numpy(ws.x_aux[:16]).float()        # (B, T, C_aux)
    last_bg = torch.from_numpy(ws.last_bg[:16]).float()
    present = torch.zeros(16)

    with torch.no_grad():
        a = model(x, aux, last_bg, present)
        # Any aux content at all, still absent: the gate must ignore it.
        b = model(x, aux + 7.5, last_bg, present)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-6), (
            "aux content changed the output while the modality was marked "
            "absent -- the fusion gate is not closed")


# --------------------------------------------------------------------------- #
# 7. Harness integration
# --------------------------------------------------------------------------- #
def test_window_set_works_with_the_causal_scaler(ws):
    """The scaler must fit on FIT subjects only and record who it fitted on."""
    fit_pat = ["StudyA:1"]
    scaler = CausalScaler.fit(ws, fit_pat)
    assert scaler.fitted_on == ("StudyA:1",)

    aux_scaler = CausalScaler.fit_array(
        ws.x_aux, ws.patient, fit_pat, mn.METABONET_AUX_SKIP_SCALING)
    for i in mn.METABONET_AUX_SKIP_SCALING:
        assert aux_scaler.center[i] == 0.0 and aux_scaler.scale[i] == 1.0, (
            "a 0/1 mask was rescaled and no longer means 'observed'")

    out = aux_scaler.transform(ws.x_aux[:8])
    assert out.shape == ws.x_aux[:8].shape
    assert np.isfinite(out).all()


def test_run_fold_trains_on_a_metabonet_window_set(ws):
    """End of requirement 4: the window set is a drop-in for the training loop,
    with no MetaboNet-specific branch inside it."""
    from causalcgm_mt.train import run_fold

    cfg = Config(
        data=DataConfig(history_min=120, horizon_min=30),
        model=ModelConfig(hidden_channels=8, n_blocks=2, head_hidden=8),
        train=TrainConfig(max_epochs=1, batch_size=64, num_workers=0,
                          amp=False, early_stop_patience=1),
    )
    res = run_fold(ws, cfg, np.array(["StudyA:1", "StudyA:2", "StudyB:1"], dtype=object),
                   np.array(["StudyB:7"], dtype=object), fold=0)

    assert res.scaler.fitted_on and "StudyB:7" not in res.scaler.fitted_on, (
        "the scaler saw a test subject")
    assert np.isfinite(res.report.rmse)
    assert res.test_patients == ["StudyB:7"]


def test_run_fold_derives_the_mask_indices_from_the_window_set(ws):
    """`run_fold` used to hardcode HUPA's `(7, 8, 9)`. On a 24-channel MetaboNet
    tensor that left the appended mask channels rescaled by an IQR and invisible
    to the fusion gate."""
    from causalcgm_mt.hupa_loader import AUX_CHANNELS as HUPA
    names = tuple(getattr(ws, "aux_channels", HUPA))
    derived = tuple(i for i, c in enumerate(names) if c.endswith("_observed"))
    assert derived == mn.METABONET_AUX_SKIP_SCALING
    assert len(derived) == 8, "MetaboNet has eight mask channels, not three"


# --------------------------------------------------------------------------- #
# 8. The real file, when it is present
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_REAL, reason="metabonet_public.parquet not present")
def test_real_schema_matches_the_declared_one():
    import pyarrow.parquet as pq
    names = pq.ParquetFile(REAL_PARQUET).schema_arrow.names
    found = mn.MetaboNetSchema().validate(names, REAL_PARQUET)
    for c in ("heartrate", "steps", "basal", "bolus", "carbs",
              "galvanic_skin_response", "skin_temp", "air_temp",
              "calories_burned", "workout_intensity", "workout_duration"):
        assert c in found, f"{c} is no longer in the released file"
    assert "sleep" not in names, (
        "the release now has a sleep column; wire it into synchronize_subject "
        "instead of leaving channel 2 permanently masked")


@pytest.mark.skipif(not HAS_REAL, reason="metabonet_public.parquet not present")
def test_real_file_contains_the_benchmark_cohorts_and_they_are_excluded():
    """The finding this loader is built around, asserted rather than described:
    MetaboNet ships OhioT1DM and HUPA-UCM, and the default config drops them."""
    permissive = mn.MetaboNetConfig(exclude_sources=())
    idx_all = mn.scan_subjects(REAL_PARQUET, mcfg=permissive)
    sources = set(idx_all.sources())
    assert {"OhioT1DM", "HUPA-UCM"} <= sources

    ohio = idx_all.frame[idx_all.frame.source == "OhioT1DM"]
    assert len(ohio) == 12, "the Ohio slice is the full 12-subject cohort"
    assert (ohio.n_test == 0).all(), (
        "every OhioT1DM row is labelled is_test=False, INCLUDING the official "
        "test spans -- the contamination this loader guards against")

    idx = mn.scan_subjects(REAL_PARQUET)
    assert not ({"OhioT1DM", "HUPA-UCM"} & set(idx.sources()))
    mn.assert_no_benchmark_contamination(idx)


@pytest.mark.skipif(not HAS_REAL, reason="metabonet_public.parquet not present")
def test_real_ids_collide_across_sources():
    idx = mn.scan_subjects(REAL_PARQUET, mcfg=mn.MetaboNetConfig(exclude_sources=()))
    per_id = idx.frame.groupby("id")["source"].nunique()
    assert (per_id > 1).sum() > 100, (
        "raw ids no longer collide across source_file; the composite key is "
        "still correct but this test's premise has changed")


@pytest.mark.skipif(not HAS_REAL, reason="metabonet_public.parquet not present")
def test_real_interleaved_subjects_are_found_and_the_flag_cannot_find_them():
    """The shipped flag fails in BOTH directions, which is why `split_class` is
    derived from the rows.

    It misses straddling subjects (CTR3 9, 10, 13 are flagged False while their
    rows appear on both sides), and it marks interleaved subjects (CTR3 5, 12)
    no differently from the 164 subjects whose temporal holdout is valid. A
    loader that trusted it would evaluate two subjects on rows drawn from the
    same hours it trained them on.
    """
    idx = mn.scan_subjects(REAL_PARQUET, mcfg=mn.MetaboNetConfig(exclude_sources=()))
    f = idx.frame
    inter = f[f.split_class == "interleaved"]
    assert len(inter) >= 1

    assert (~inter.flag_says_split).any(), (
        "no interleaved subject is missed by the flag any more")
    assert inter.flag_says_split.any(), (
        "the flag no longer marks any interleaved subject as a straddler; if "
        "the release now distinguishes them, `split_class` can say so too")

    # And the flag never separates a valid temporal holdout from a broken one.
    flagged = f[f.flag_says_split]
    assert set(flagged.split_class) == {"temporal", "interleaved"}, (
        "the flag is a straddle hint, not a validity statement")


# --------------------------------------------------------------------------- #
# 9. Streaming, memory bounds and resumability
# --------------------------------------------------------------------------- #
def test_streaming_and_eager_builders_agree(parquet_path, cfg):
    """The two builders share `_subject_windows`, and this is what says so.

    If they ever diverge, one of the causality guarantees asserted above holds
    for a path nothing runs in production.
    """
    from causalcgm_mt.metabonet_loader import build_metabonet_windows_streaming

    keys = mn.scan_subjects(parquet_path).keys()
    eager = mn.build_metabonet_windows(
        mn.load_metabonet_frame(parquet_path, keys), cfg, horizons_min=(30,))
    streamed = build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,))

    assert len(streamed) == len(eager)
    order_e = np.lexsort((eager.t_time, eager.patient.astype(str)))
    order_s = np.lexsort((streamed.t_time, streamed.patient.astype(str)))
    assert np.array_equal(np.asarray(eager.x)[order_e],
                          np.asarray(streamed.x)[order_s])
    assert np.array_equal(np.asarray(eager.x_aux)[order_e],
                          np.asarray(streamed.x_aux)[order_s])
    assert np.array_equal(eager.y_reg[order_e], streamed.y_reg[order_s])
    assert list(eager.patient[order_e]) == list(streamed.patient[order_s])


def test_streaming_memmaps_the_large_arrays(parquet_path, cfg, tmp_path):
    """`x` and `x_aux` are ~99% of the bytes; they live on disk so a fold pages
    in only the rows it touches."""
    from causalcgm_mt.metabonet_loader import build_metabonet_windows_streaming

    keys = mn.scan_subjects(parquet_path).keys()
    ws = build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,),
        cache_dir=str(tmp_path / "store"))

    assert isinstance(ws.x, np.memmap) and isinstance(ws.x_aux, np.memmap)
    assert os.path.exists(tmp_path / "store" / "arrays" / "x.npy")
    assert os.path.exists(tmp_path / "store" / "arrays" / "x_aux.npy")
    # The small arrays stay resident: an object array of a repeated subject key
    # costs one pointer a row, so keeping them is cheaper than paging them.
    assert not isinstance(ws.y_reg, np.memmap)
    assert np.isfinite(np.asarray(ws.x[:64])).all()


def test_an_interrupted_build_resumes_from_its_shards(parquet_path, cfg, tmp_path):
    from causalcgm_mt.metabonet_loader import WindowStore, _fingerprint

    store_dir = str(tmp_path / "store")
    keys = mn.scan_subjects(parquet_path).keys()
    full = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir)
    n_full = len(full)
    del full

    # Simulate an interruption: drop the last subject's shard from the manifest.
    store = WindowStore(store_dir, _fingerprint(cfg, mn.MetaboNetConfig(), (30,)))
    assert store.load_manifest()
    assert len(store.entries) == len(keys)
    dropped = store.entries.pop()
    store.save_manifest()

    resumed = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir,
        resume=True)
    assert len(resumed) == n_full, (
        f"resuming did not rebuild {dropped['key']}")


def test_a_cache_built_under_different_geometry_is_rebuilt(parquet_path, tmp_path):
    """A stale store is the OhioT1DM `WINDOW_PIPELINE_VERSION` problem again:
    reusing windows built under a different history or horizon reproduces
    numbers from a pipeline that no longer exists."""
    from causalcgm_mt.metabonet_loader import WindowStore, _fingerprint

    store_dir = str(tmp_path / "store")
    keys = mn.scan_subjects(parquet_path).keys()
    cfg_a = DataConfig(history_min=120, horizon_min=30)
    cfg_b = DataConfig(history_min=60, horizon_min=30)      # different geometry
    assert _fingerprint(cfg_a, mn.MetaboNetConfig(), (30,)) != \
        _fingerprint(cfg_b, mn.MetaboNetConfig(), (30,))

    a = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg_a, horizons_min=(30,), cache_dir=store_dir)
    assert a.x.shape[1] == cfg_a.history_steps

    b = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg_b, horizons_min=(30,), cache_dir=store_dir,
        resume=True)
    assert b.x.shape[1] == cfg_b.history_steps, (
        "the stale cache was adopted; windows would be the wrong shape")


def test_the_eager_loader_refuses_a_corpus_sized_request(parquet_path):
    """`load_metabonet_frame` on the whole corpus is ~24 GB and the process is
    killed rather than raising, so the size is checked from the index first."""
    keys = mn.scan_subjects(parquet_path).keys()
    tiny = mn.MetaboNetConfig(max_eager_rows=10)
    with pytest.raises(MemoryError, match="eager limit"):
        mn.load_metabonet_frame(parquet_path, keys, mcfg=tiny)
    # And the message names the way out.
    try:
        mn.load_metabonet_frame(parquet_path, keys, mcfg=tiny)
    except MemoryError as exc:
        assert "build_metabonet_windows_streaming" in str(exc)


def test_the_memory_budget_stops_a_build_it_cannot_finish(parquet_path, cfg,
                                                          tmp_path):
    """The anti-OOM guarantee: a run projected past its budget stops early,
    with the knobs that would make it fit, instead of being killed."""
    mcfg = mn.MetaboNetConfig(memory_budget_gb=1e-6)     # nothing fits
    keys = mn.scan_subjects(parquet_path).keys()
    with pytest.raises(MemoryError, match="over the .* GB budget"):
        mn.build_metabonet_windows_streaming(
            parquet_path, keys, cfg, mcfg, horizons_min=(30,),
            cache_dir=str(tmp_path / "store"))


def test_window_cap_thins_in_blocks_not_every_kth(parquet_path, cfg):
    from causalcgm_mt.metabonet_loader import _thin_blocks

    # The unit itself: contiguous runs, spread across the record.
    idx = _thin_blocks(n=1000, cap=300, block=100)
    assert len(idx) == 300
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    assert len(runs) == 3 and all(len(r) == 100 for r in runs)
    assert idx[0] == 0 and idx[-1] < 1000
    # No cap, or a cap above the count, keeps everything in order.
    assert np.array_equal(_thin_blocks(50, 0, 10), np.arange(50))
    assert np.array_equal(_thin_blocks(50, 100, 10), np.arange(50))

    capped = mn.build_metabonet_windows(
        mn.load_metabonet_frame(parquet_path), cfg,
        mn.MetaboNetConfig(max_windows_per_subject=200, thin_block_windows=100),
        horizons_min=(30,))
    for pid in capped.subjects():
        assert (capped.patient == pid).sum() <= 200


def test_memory_helpers_report_something_usable():
    assert np.isfinite(mn.rss_gb()) and mn.rss_gb() > 0
    note = mn.memory_note()
    assert "RAM" in note and "GB" in note


def test_memory_reporting_degrades_without_psutil(monkeypatch):
    """`psutil` is a convenience, not a dependency of correctness.

    Without it the fallback reports the process PEAK RSS, which is a different
    quantity -- it never goes down -- so the label changes with it rather than
    presenting a peak as if it were the live figure.
    """
    monkeypatch.setattr(mn, "_PROC", None)
    monkeypatch.setattr(mn, "_psutil", None)
    assert np.isfinite(mn.rss_gb()) and mn.rss_gb() > 0
    assert "peak RAM" in mn.memory_note()
    assert not np.isfinite(mn.available_gb())


def test_run_fold_trains_on_a_memmap_backed_window_set(parquet_path, tmp_path):
    """The integration the whole refactor exists for: a window set whose big
    arrays live on disk must train exactly like one in RAM.

    Fancy-indexing a memmap returns a real array, so the scaler, the dataset and
    `torch.from_numpy` all see ordinary memory -- but only the rows the fold
    actually touches are ever paged in, which is what keeps a corpus-scale run
    inside its budget.
    """
    from causalcgm_mt.train import run_fold

    keys = mn.scan_subjects(parquet_path).keys()
    ws = mn.build_metabonet_windows_streaming(
        parquet_path, keys, DataConfig(history_min=120, horizon_min=30),
        horizons_min=(30,), cache_dir=str(tmp_path / "store"))
    assert isinstance(ws.x, np.memmap)

    cfg = Config(
        data=DataConfig(history_min=120, horizon_min=30),
        model=ModelConfig(hidden_channels=8, n_blocks=2, head_hidden=8),
        train=TrainConfig(max_epochs=1, batch_size=64, num_workers=0,
                          amp=False, early_stop_patience=1),
    )
    subjects = ws.subjects()
    res = run_fold(ws, cfg, np.array(subjects[:-1], dtype=object),
                   np.array(subjects[-1:], dtype=object), fold=0)

    assert np.isfinite(res.report.rmse)
    assert res.history and res.history[0]["train_loss"] > 0, (
        "the epoch history is what a long run is diagnosed from afterwards")
    assert set(res.history[0]) >= {"epoch", "train_loss", "val_rmse", "val_pr_auc"}
    assert subjects[-1] not in res.scaler.fitted_on


def test_a_change_to_the_windowing_logic_invalidates_the_cache(parquet_path, cfg,
                                                               tmp_path,
                                                               monkeypatch):
    """`data.WINDOW_PIPELINE_VERSION` covers what a window CONTAINS. Changing
    how windows are SELECTED -- the thinning, the per-subject loop -- is a
    different axis, and a store built the old way must not be mixed with shards
    built the new way."""
    from causalcgm_mt.metabonet_loader import _fingerprint

    before = _fingerprint(cfg, mn.MetaboNetConfig(), (30,))
    monkeypatch.setattr(mn, "METABONET_WINDOW_VERSION",
                        mn.METABONET_WINDOW_VERSION + 1)
    after = _fingerprint(cfg, mn.MetaboNetConfig(), (30,))
    assert before != after, (
        "the window-store fingerprint ignores this module's own version, so a "
        "change to _thin_blocks would silently reuse shards built the old way")


def test_a_fully_cached_store_finalises_without_reading_the_parquet(parquet_path,
                                                                    cfg, tmp_path):
    """A complete resume must not fall through to the reader.

    With every subject already sharded there is nothing to ingest, and asking
    the reader for an empty subject set is (correctly) an error -- so a fully
    cached store would fail where a partially cached one succeeds.
    """
    store_dir = str(tmp_path / "store")
    keys = mn.scan_subjects(parquet_path).keys()
    first = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir)
    n = len(first)
    del first

    missing = str(tmp_path / "not-a-file.parquet")
    again = mn.build_metabonet_windows_streaming(
        missing, keys, cfg, horizons_min=(30,), cache_dir=store_dir, resume=True)
    assert len(again) == n, (
        "a fully cached store should finalise from its shards alone")


def test_two_builders_cannot_share_one_window_cache(parquet_path, cfg, tmp_path):
    """Concurrent builds on one cache interleave shards and overwrite each
    other's manifest, and the result reads as a valid store: the manifest of
    whichever wrote last, over a shard set neither produced."""
    from causalcgm_mt.metabonet_loader import WindowStore, _fingerprint

    store_dir = str(tmp_path / "store")
    os.makedirs(store_dir, exist_ok=True)
    fp = _fingerprint(cfg, mn.MetaboNetConfig(), (30,))

    import json as _json
    # A live process that is not this one. pid 1 always exists on a unix host,
    # and acquiring in-process would be indistinguishable from re-entering.
    with open(os.path.join(store_dir, "LOCK"), "w") as f:
        _json.dump({"pid": 1, "started": "2020-01-01T00:00:00"}, f)
    keys = mn.scan_subjects(parquet_path).keys()
    with pytest.raises(RuntimeError, match="being built by process 1"):
        mn.build_metabonet_windows_streaming(
            parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir)
    os.remove(os.path.join(store_dir, "LOCK"))
    WindowStore(store_dir, fp)          # constructible after the lock is gone

    # Released: the build proceeds.
    ws = mn.build_metabonet_windows_streaming(
        parquet_path, mn.scan_subjects(parquet_path).keys(), cfg,
        horizons_min=(30,), cache_dir=store_dir)
    assert len(ws) > 0


def test_a_stale_lock_is_reclaimed_rather_than_stranding_the_cache(parquet_path,
                                                                   cfg, tmp_path):
    """A lock held by a dead process must not make a cache unresumable."""
    from causalcgm_mt.metabonet_loader import WindowStore, _fingerprint
    import json as _json

    store_dir = str(tmp_path / "store")
    os.makedirs(store_dir, exist_ok=True)
    with open(os.path.join(store_dir, "LOCK"), "w") as f:
        # A pid that cannot be running: pid 0 is never a normal process here.
        _json.dump({"pid": 999999999, "started": "2020-01-01T00:00:00"}, f)

    ws = mn.build_metabonet_windows_streaming(
        parquet_path, mn.scan_subjects(parquet_path).keys(), cfg,
        horizons_min=(30,), cache_dir=store_dir)
    assert len(ws) > 0
    assert not os.path.exists(os.path.join(store_dir, "LOCK")), (
        "the lock must be released when the build finishes")


def test_subjects_that_produce_no_windows_are_not_re_read_on_resume(cfg, tmp_path):
    """A barren subject is a decision, not an absence of one.

    Every input to it is already covered by the store's fingerprint, so
    re-deriving it on each resume buys nothing -- and on the released corpus 35
    of 1,873 subjects contribute nothing, which is a fifth of a minute of
    re-reading on every restart.
    """
    from causalcgm_mt.metabonet_loader import WindowStore, _fingerprint

    # One ordinary subject and one too short to yield any window: a window
    # needs history + horizon = 30 steps, so 20 rows cannot produce one.
    corpus = pd.concat([make_subject("StudyA", "1", n=700, seed=1),
                        make_subject("StudyA", "2", n=20, seed=2)],
                       ignore_index=True)
    path = tmp_path / "barren.parquet"
    corpus.to_parquet(path, row_group_size=400)
    store_dir = str(tmp_path / "store")

    keys = mn.scan_subjects(str(path)).keys()
    assert set(keys) == {"StudyA:1", "StudyA:2"}
    ws = mn.build_metabonet_windows_streaming(
        str(path), keys, cfg, horizons_min=(30,), cache_dir=store_dir)
    assert ws.subjects() == ["StudyA:1"]

    store = WindowStore(store_dir, _fingerprint(cfg, mn.MetaboNetConfig(), (30,)))
    assert store.load_manifest()
    assert store.barren == ["StudyA:2"]
    assert store.done_keys == {"StudyA:1", "StudyA:2"}, (
        "the barren subject is not marked done, so a resume would re-read it")

    # A resume has nothing left to do -- including nothing to re-derive.
    again = mn.build_metabonet_windows_streaming(
        str(tmp_path / "gone.parquet"), keys, cfg, horizons_min=(30,),
        cache_dir=store_dir, resume=True)
    assert len(again) == len(ws)


# --------------------------------------------------------------------------- #
# 10. The window budget: how it is spent, and what it costs to spend it
# --------------------------------------------------------------------------- #
def test_the_cap_never_collapses_to_the_records_opening():
    """The shipped default must not keep ONE block anchored at index 0.

    `round(400 / 288) == 1`, so the previously documented invocation
    (`--max-windows-per-subject 400`, `thin_block_windows=288`) took a single
    block from the start of the record: the opening ~33 h of a multi-week
    stream, 0.4% of the corpus's longest subject, while this function's own
    docstring promised a sample spread across the subject's weeks.

    Nothing showed it. The anchor cadence INSIDE the block is correct, so
    `clinical.segment_ids`, the alarm episodes and lead time all came back
    healthy -- computed from the wrong days. That is what makes it worth a test
    rather than a comment: there is no metric that goes wrong when it recurs.
    """
    from causalcgm_mt.metabonet_loader import _thin_blocks

    n = 100_000
    idx = _thin_blocks(n, cap=400, block=288)          # the shipped default
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)

    assert len(runs) >= 2, (
        "the cap degenerated to a single contiguous block, which is 'the first "
        "N windows' under another name")
    assert idx[-1] > 0.5 * n, "the sample does not reach the second half"
    assert idx[0] == 0 and idx[-1] == n - 1, "it must straddle the whole record"
    assert len(idx) == 400, "and still spend the budget it was given"

    # The degeneracy was not one unlucky pair: it covered every cap below
    # 1.5x the block length, which is where a per-subject budget naturally sits.
    for cap in (289, 300, 350, 400, 430):
        idx = _thin_blocks(n, cap, 288)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        assert len(runs) >= 2, f"cap={cap} still collapses to one block"
        assert idx[-1] == n - 1, f"cap={cap} does not reach the record's end"
        assert len(idx) >= cap - 1, f"cap={cap} leaves budget unspent"

    # Blocks stay CONTIGUOUS -- the property the whole function exists for,
    # since `clinical.segment_ids` breaks a segment on any gap over one period.
    idx = _thin_blocks(n, cap=1000, block=288)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    assert all(len(r) > 1 for r in runs), "thinned to isolated anchors"
    assert len({len(r) for r in runs}) == 1, "blocks are not equal length"

    # A budget too small for two blocks is the one case that keeps one.
    assert np.array_equal(_thin_blocks(n, cap=1, block=288), np.array([0]))


def test_the_window_version_covers_the_thinning_change():
    """A cache built under the biased sample must not be adopted.

    `_fingerprint` reads `METABONET_WINDOW_VERSION`, so the version is the only
    thing standing between a corrected run and shards holding each subject's
    opening 33 h. Bumping it is part of the fix, not bookkeeping.
    """
    assert mn.METABONET_WINDOW_VERSION >= 3, (
        "the block-thinning fix must invalidate stores built before it")


def test_the_cap_is_paid_before_the_windows_are_materialised(cfg):
    """Peak memory is O(cap), not O(windows in the record).

    The module's contract is that peak RSS during a build is one subject. It
    was not: every stride-1 window went into Python lists before the cap was
    consulted, so the corpus's longest subject built ~126k (24, 24) float64 aux
    slices -- ~580 MB in the list, plus a full copy in `np.stack` before `[keep]`
    sliced it to 400 -- roughly 1.2 GB peak for a megabyte of output. The
    `--memory-budget-gb` guard prices the FINISHED store and cannot see any of
    it, so this is the only place that bound can be enforced.

    Measured against the floor the old shape could not go below: the aux slices
    alone, for every valid window. (tracemalloc sees numpy's allocations here;
    where it does not, this test weakens rather than turning flaky.)
    """
    import gc
    import tracemalloc

    long_record = canonical(make_subject("Long", "L1", n=8000, seed=11))
    key = str(long_record["patient"].iloc[0])
    mcfg = mn.MetaboNetConfig(max_windows_per_subject=200, thin_block_windows=100)

    gc.collect()
    tracemalloc.start()
    part = mn._subject_windows(key, long_record, cfg, mcfg, (30,))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert part is not None
    assert part["n_before_cap"] > 20 * part["n"], (
        "this record is too short to distinguish O(cap) from O(windows)")

    aux_slices = (part["n_before_cap"] * cfg.history_steps
                  * len(mn.METABONET_AUX_CHANNELS) * 8)
    assert peak < 0.5 * aux_slices, (
        f"peak {peak/2**20:.1f} MB against a {aux_slices/2**20:.1f} MB floor "
        f"for holding every window: the cap is being applied after the "
        f"tensors were cut, not before")
    assert part["n"] == 200, "and the budget is still spent in full"


def test_capping_selects_windows_it_does_not_change_them(cfg):
    """The two-pass build must be a pure subset of the one-pass result.

    Deciding validity in one pass and cutting tensors in another is a memory
    change, not a semantic one -- so every capped window has to be byte-identical
    to the window the uncapped pass emits at the same anchor. This is what says
    the refactor did not quietly move a slice by one step.
    """
    record = canonical(make_subject("Sub", "s1", n=1500, seed=3))
    key = str(record["patient"].iloc[0])

    full = mn._subject_windows(key, record, cfg, mn.MetaboNetConfig(), (30, 60))
    capped = mn._subject_windows(
        key, record, cfg,
        mn.MetaboNetConfig(max_windows_per_subject=200, thin_block_windows=100),
        (30, 60))

    assert capped["n"] == 200 and full["n"] == capped["n_before_cap"]
    # Locate each kept window in the uncapped set by its anchor, then compare
    # every array the window is made of.
    where = {int(t): i for i, t in enumerate(full["t_index"])}
    rows = [where[int(t)] for t in capped["t_index"]]
    assert len(set(rows)) == len(rows), "the cap emitted a window twice"

    for field in ("x", "x_aux", "y_reg_multi", "y_min_multi", "y_cls_multi",
                  "t_index", "t_time", "last_bg", "aux_frac", "split"):
        assert np.array_equal(np.asarray(full[field])[rows],
                              np.asarray(capped[field])), field


def test_the_finalised_store_is_handed_back_read_only(parquet_path, cfg, tmp_path):
    """`ws.x` must not be a writable handle onto the resumable cache.

    `finalize` built the arrays "w+" and returned them still open that way, so
    any in-place write downstream -- an `out[...] =` that lost its `.copy()`, a
    stray `np.clip(..., out=ws.x)` -- would rewrite the CACHE, not a copy of it.
    Every later resume would then read corrupted windows under a matching
    fingerprint, with nothing to warn about: the store is the artefact the whole
    resumability design exists to protect.
    """
    store_dir = str(tmp_path / "store")
    keys = mn.scan_subjects(parquet_path).keys()
    ws = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir)

    assert isinstance(ws.x, np.memmap) and isinstance(ws.x_aux, np.memmap)
    for name, arr in (("x", ws.x), ("x_aux", ws.x_aux)):
        assert not arr.flags.writeable, f"ws.{name} is a writable memmap"
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0.0

    before = np.asarray(ws.x[:32]).copy()

    # The read paths every arm actually uses still work, because they all copy.
    scaler = CausalScaler.fit_array(ws.x, ws.patient, ws.subjects()[:1],
                                    skip_channels=(4,))
    scaled = scaler.transform(ws.x[:32])
    assert scaled.shape == (32,) + ws.x.shape[1:]
    assert np.array_equal(np.asarray(ws.x[:32]), before), (
        "transform wrote through to the store")
    assert ws.subset(np.arange(16)).x.shape[0] == 16

    # And the bytes on disk survived the attempted writes above, so a resume
    # sees the same windows.
    resumed = mn.build_metabonet_windows_streaming(
        parquet_path, keys, cfg, horizons_min=(30,), cache_dir=store_dir,
        resume=True)
    assert np.array_equal(np.asarray(resumed.x[:32]), before)
