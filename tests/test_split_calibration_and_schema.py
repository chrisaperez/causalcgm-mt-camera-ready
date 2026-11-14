"""Regression tests for three defects that were invisible in every artefact.

Each of the three had the same shape: the pipeline produced a well-formed
result, every field in the results JSON was a correct description of *something*,
and the thing being described was not the thing the paper claimed. So each test
here asserts the MECHANISM rather than a number -- a number would have passed
before the fix as readily as after it.

  1. CONFORMAL CALIBRATION vs MODEL SELECTION (`split.calib` vs `split.val`).
     `_conformal_block` calibrated on the same rows early stopping selected the
     checkpoint on, so qhat was a quantile of residuals from the rows the model
     had been chosen to fit best -- biased low, intervals too narrow, coverage
     below nominal. `calibrated_on`, `n_calib` and `n_effective` were all still
     accurate descriptions of the wrong rows.

  2. THE MULTI-TASK CRITERION IS FITTED STATE. `run_ohio_personalization_sweep.py`
     saved the base model without its criterion, so a RESUMED sweep adapted under
     a uniform Kendall task balance and a pos_weight of 1.0 while a cold run
     adapted under the trained ones. The same command produced two different
     tables depending on whether a cache file happened to exist.

  3. ONE CHECKPOINT SCHEMA. Three drivers wrote three shapes and only
     `train_demo.py`'s could be exported, so the shipped ONNX graph -- and every
     latency number taken from it -- was necessarily the synthetic demo model.

Runs on a small synthetic OhioT1DM-shaped cohort; no licensed data needed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B
from causalcgm_mt.config import Config, DataConfig
from causalcgm_mt.export import (REQUIRED_CHECKPOINT_KEYS,
                                 REQUIRED_MULTIMODAL_KEYS,
                                 assert_checkpoint_schema)
from causalcgm_mt.hupa_loader import HupaConfig
from causalcgm_mt.models import MultiTaskLoss, criterion_from_state
from causalcgm_mt.ohio_loader import (OhioConfig, build_ohio_windows,
                                      discover_records,
                                      official_temporal_split)

CFG = DataConfig(history_min=60, horizon_min=30, stride_steps=1, band_mode="fixed")
HORIZONS = (30, 60)


# --------------------------------------------------------------------------- #
# Fixtures: a two-cohort cohort in the real XML dialect
# --------------------------------------------------------------------------- #
def _write_xml(path, pid, start, hours, seed):
    rng = np.random.default_rng(seed)
    n = int(hours * 12)
    ts = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    bg = np.clip(125 + 62 * np.sin(np.arange(n) / 8.5) + rng.normal(0, 4, n), 40, 380)
    fmt = lambda t: t.strftime("%d-%m-%Y %H:%M:%S")  # noqa: E731

    rows = ['<?xml version="1.0" encoding="UTF-8"?>',
            f'<patient id="{pid}" weight="80" insulin_type="Novalog">',
            "  <glucose_level>"]
    rows += [f'    <event ts="{fmt(t)}" value="{v:.0f}"/>' for t, v in zip(ts, bg)]
    rows += ["  </glucose_level>", "  <basal>",
             f'    <event ts="{fmt(ts[0])}" value="0.8"/>', "  </basal>",
             "  <basis_heart_rate>"]
    rows += [f'    <event ts="{fmt(t)}" value="{70 + (i % 9)}"/>'
             for i, t in enumerate(ts) if i % 2 == 0]
    rows += ["  </basis_heart_rate>", "  <basis_steps>"]
    rows += [f'    <event ts="{fmt(t)}" value="{i % 30}"/>' for i, t in enumerate(ts)]
    rows += ["  </basis_steps>", "</patient>"]
    with open(path, "w") as f:
        f.write("\n".join(rows))


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("ohio_split")
    d = root / "2018"
    for split, tag, off, hours in (("train", "training", 0, 96.0),
                                   ("test", "testing", 97, 36.0)):
        (d / split).mkdir(parents=True, exist_ok=True)
        for k, pid in enumerate(["501", "502"]):
            _write_xml(str(d / split / f"{pid}-ws-{tag}.xml"), pid,
                       pd.Timestamp("2021-12-07 00:00:00") + pd.Timedelta(hours=off),
                       hours, seed=k + 1)
    return str(root)


@pytest.fixture(scope="module")
def ws(fake_root):
    return build_ohio_windows(discover_records(fake_root), CFG, HupaConfig(),
                              OhioConfig(), horizons_min=HORIZONS)


@pytest.fixture(scope="module")
def split(ws):
    return official_temporal_split(ws, CFG, val_frac=0.2)


@pytest.fixture(scope="module")
def data(fake_root) -> B.CohortData:
    return B.load_cohort(fake_root, history_min=CFG.history_min,
                         horizons=HORIZONS, val_frac=0.2, cache=None)


def _subjects(ws):
    return sorted({str(p) for p in ws.patient})


# --------------------------------------------------------------------------- #
# 1. Conformal calibration is disjoint from model selection
# --------------------------------------------------------------------------- #
def test_split_has_four_disjoint_regions(ws, split):
    """`calib` is a real region, not an alias of `val`. If it were empty or
    equal to `val`, every test below would pass vacuously."""
    assert split.calib.size > 0
    assert split.val.size > 0
    for a, b in (("fit", "val"), ("fit", "calib"), ("fit", "test"),
                 ("val", "calib"), ("val", "test"), ("calib", "test")):
        shared = np.intersect1d(getattr(split, a), getattr(split, b))
        assert shared.size == 0, f"{a}/{b} share {shared.size} rows"

    # And every window is accounted for exactly once.
    regions = np.concatenate([split.fit, split.val, split.calib,
                              split.test, split.embargoed])
    assert regions.size == len(ws)
    assert np.unique(regions).size == len(ws)


def test_calibration_region_is_chronologically_after_selection(ws, split):
    """Order matters: `calib` sits between `val` and `test`, so calibration
    happens on the rows temporally nearest the ones being scored -- where
    exchangeability with test is strongest -- and model selection stays on rows
    further from the boundary."""
    for pid in _subjects(ws):
        per = {}
        for name in ("fit", "val", "calib", "test"):
            idx = getattr(split, name)
            idx = idx[ws.patient[idx] == pid]
            if idx.size:
                per[name] = (ws.t_time[idx].min(), ws.t_time[idx].max())
        assert {"fit", "val", "calib", "test"} <= set(per), f"{pid}: {sorted(per)}"
        assert per["fit"][1] < per["val"][0]
        assert per["val"][1] < per["calib"][0]
        assert per["calib"][1] < per["test"][0]


def test_every_internal_boundary_is_embargoed(ws, split):
    """Row disjointness is not enough: at stride 1 adjacent windows share 23 of
    24 history steps, so two regions can be row-disjoint and still share raw
    samples. The gap between consecutive regions must exceed history + horizon.

    The calib|test boundary is the one this fix added, and it is the one the
    coverage claim rests on: OhioT1DM's testing file is the immediately
    following stretch of one continuous record, so without an embargo the last
    calibration windows and the first scored windows genuinely overlap."""
    P = np.timedelta64(CFG.sample_period_min, "m")
    span = P * int(CFG.history_steps) + P * int(max(HORIZONS) // CFG.sample_period_min)

    for pid in _subjects(ws):
        def bounds(name):
            idx = getattr(split, name)
            idx = idx[ws.patient[idx] == pid]
            return (ws.t_time[idx].min(), ws.t_time[idx].max()) if idx.size else None

        for earlier, later in (("fit", "val"), ("val", "calib"), ("calib", "test")):
            a, b = bounds(earlier), bounds(later)
            assert a is not None and b is not None
            gap = b[0] - a[1]
            assert gap > span, (
                f"{pid}: {earlier}->{later} gap {gap} does not clear "
                f"history+horizon {span}")


def test_conformal_calibrates_on_calib_and_ignores_val_targets(data):
    """THE regression test for defect 1.

    Poison every VALIDATION target and require qhat to be unchanged. Before the
    fix, `_conformal_block` read `split.val`, so this moved qhat; now qhat is a
    quantile of `split.calib` residuals and cannot see a val label at all.

    The paired assertion is what makes it a leakage test rather than a
    tautology: poisoning `split.calib` MUST move qhat, or the poisoning never
    reached the calibration path and the first half proves nothing."""
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    calib = B.persistence_predictions(data.ws, data.split.calib, HORIZONS)
    test = B.persistence_predictions(data.ws, data.split.test, HORIZONS)
    base = B.score_arm(spec, data, data.split, calib, test)

    def rescore(poison_rows):
        ws2 = data.ws.subset(np.arange(len(data.ws)))
        ws2.y_reg_multi[poison_rows] += 400.0
        d2 = B.CohortData(ws2, data.split, data.pre, data.cfg, data.horizons)
        return B.score_arm(spec, d2, data.split, calib, test)

    val_poisoned = rescore(data.split.val)
    cal_poisoned = rescore(data.split.calib)

    for h in HORIZONS:
        q0 = base["per_horizon"][str(h)]["conformal"]["split"]["qhat"]
        qv = val_poisoned["per_horizon"][str(h)]["conformal"]["split"]["qhat"]
        qc = cal_poisoned["per_horizon"][str(h)]["conformal"]["split"]["qhat"]
        assert qv == q0, "a validation label reached qhat -- calibration is on val"
        assert qc != q0, "poisoning calib did not move qhat; the probe is inert"


def test_conformal_block_refuses_a_split_with_no_calibration_region(data):
    """Falling back to `split.val` would silently reinstate the leak, and the
    resulting table would look exactly like a valid one. So an empty `calib`
    is an error, named as such."""
    empty = dataclasses.replace(data.split, calib=np.empty(0, dtype=np.int64))
    calib = B.persistence_predictions(data.ws, data.split.calib, HORIZONS)
    test = B.persistence_predictions(data.ws, data.split.test, HORIZONS)
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    with pytest.raises(ValueError, match="split.calib is empty"):
        B.score_arm(spec, data, empty, calib, test)


def test_neural_arm_selects_on_val_and_calibrates_on_calib(data):
    """End to end through `run_arm`: the arm that early-stops is the arm most
    exposed to this defect, so the artefact has to record that the two jobs used
    different rows -- and the counts must not be equal by accident."""
    spec = B.ArmSpec("tcn", "tcn", horizons=HORIZONS, epochs=1, patience=1,
                     batch_size=64, fit_subsample=8)
    res, trained = B.run_arm(spec, data, verbose=False)
    c = res["per_horizon"]["30"]["conformal"]

    assert c["calibrated_on_split"] == "calib"
    assert c["n_calib_rows_available"] == data.split.calib.size
    assert c["n_selection_rows"] == data.split.val.size
    assert res["n_calib"] == data.split.calib.size
    assert res["n_val"] == data.split.val.size
    # The model really was selected on val (an epoch was chosen), and the rows
    # that chose it are not the rows that calibrated it.
    assert trained.best_epoch >= 0
    assert np.intersect1d(data.split.val, data.split.calib).size == 0


def test_lgbm_early_stops_on_val_and_calibrates_on_calib(data):
    """LightGBM's early stopping reads `split.val` through `LGBMBaseline.fit`,
    so its calibration predictions must come from `split.calib` -- and must be
    the right LENGTH, since a mismatched array would be indexed by the mask and
    silently misalign residuals with labels."""
    spec = B.ArmSpec("lgbm", "lgbm", horizons=HORIZONS)
    calib_pred, test_pred = B.lgbm_predictions(data, spec, data.split)
    assert calib_pred.bg.shape == (data.split.calib.size, len(HORIZONS))
    assert test_pred.bg.shape == (data.split.test.size, len(HORIZONS))


def test_restrict_split_carries_the_calibration_region(data):
    """A cohort-restricted arm must keep a calibration region, or its coverage
    numbers would be built from an empty set. `restrict_split` cuts every region
    with the same mask."""
    spec = B.ArmSpec("x", "tcn_mm", horizons=HORIZONS, cohorts=("2018",))
    s = B.restrict_split(data, spec)
    assert s.calib.size > 0
    assert np.isin(s.calib, data.split.calib).all()
    assert np.intersect1d(s.calib, s.val).size == 0


def test_calib_frac_zero_is_refused_at_scoring_not_silently_tolerated(ws):
    """`calib_frac=0` builds a split with no calibration region. That is a
    legitimate thing to construct (for a run that computes no intervals) and an
    illegitimate thing to score conformally, so the refusal belongs at the
    scoring boundary rather than at construction."""
    s = official_temporal_split(ws, CFG, val_frac=0.2, calib_frac=0.0)
    assert s.calib.size == 0
    assert s.val.size > 0
    with pytest.raises(ValueError):
        official_temporal_split(ws, CFG, val_frac=0.2, calib_frac=1.0)


def test_adaptation_prefix_pool_still_covers_the_whole_training_file(ws, split):
    """`calib` is carved out of the training file, so the personalisation prefix
    pool has to include it. Omitting it would truncate the record the prefix is
    drawn from -- and would silently shorten `prefix_mode="recent"`, whose whole
    definition is the LAST hours that clear the embargo."""
    from causalcgm_mt.ohio_loader import ohio_prefix_embargo_split

    pid = _subjects(ws)[0]
    ps = ohio_prefix_embargo_split(ws, split, pid, CFG, prefix_hours=1000.0)
    is_pid = np.asarray([str(p) == pid for p in ws.patient])
    train_rows = np.flatnonzero(is_pid & (ws.split == "train"))
    assert ps.adapt.size == train_rows.size
    assert ps.n_adapt_in_calib > 0, "no calibration rows reached the prefix pool"


# --------------------------------------------------------------------------- #
# 2. The multi-task criterion is fitted state
# --------------------------------------------------------------------------- #
def _fitted_criterion(seed: int = 0) -> MultiTaskLoss:
    """A criterion whose Kendall weights and class prior have MOVED, so a reset
    to the initialisation is detectable."""
    cfg = Config().model
    crit = MultiTaskLoss(dataclasses.replace(cfg, n_horizons=1), pos_weight=18.5)
    with torch.no_grad():
        crit.log_var_reg.fill_(-0.75)
        crit.log_var_cls.fill_(0.4)
    return crit


def test_criterion_from_state_restores_task_weights_and_class_prior():
    crit = _fitted_criterion()
    state = {k: v.detach().cpu() for k, v in crit.state_dict().items()}

    restored = criterion_from_state(state, Config().model)
    assert torch.allclose(restored.log_var_reg, torch.tensor([-0.75]))
    assert torch.allclose(restored.log_var_cls, torch.tensor([0.4]))
    assert float(restored.pos_weight) == pytest.approx(18.5)


def test_criterion_from_state_refuses_empty_state():
    """A fresh `MultiTaskLoss` is not a neutral default: it is a uniform task
    balance and an unweighted BCE, i.e. a different objective. Silently
    substituting it is what made a resumed sweep a different experiment."""
    with pytest.raises(RuntimeError, match="no multi-task criterion state"):
        criterion_from_state({}, Config().model, context="a fold")
    with pytest.raises(RuntimeError, match="a fold"):
        criterion_from_state({}, Config().model, context="a fold")


def test_a_fresh_criterion_really_is_a_different_objective():
    """Guards the premise of the fix. If a reset criterion computed the same
    loss as a fitted one, none of this would matter."""
    fitted = _fitted_criterion()
    fresh = MultiTaskLoss(Config().model)

    torch.manual_seed(0)
    bg, logit = torch.randn(32) * 20 + 120, torch.randn(32)
    y_reg, y_cls = torch.randn(32) * 20 + 120, (torch.rand(32) > 0.9).float()

    l_fit, _ = fitted(bg, logit, y_reg, y_cls)
    l_new, _ = fresh(bg, logit, y_reg, y_cls)
    assert not torch.isclose(l_fit, l_new), (
        "a reset criterion computes the same loss as a fitted one; the "
        "persistence fix would be pointless")


def test_sweep_checkpoint_round_trips_the_criterion(tmp_path):
    """The exact schema `run_ohio_personalization_sweep.base_checkpoint` writes
    and reads. Written as a round-trip rather than by importing the driver
    (which needs a cohort), because the property under test is the KEY, and a
    driver that stops writing it is what the regression is."""
    crit = _fitted_criterion()
    path = tmp_path / "base_model.pt"
    torch.save({
        "state_dict": {},
        "criterion_state_dict": {k: v.detach().cpu()
                                 for k, v in crit.state_dict().items()},
        "spec": {}, "pipeline_version": B.WINDOW_PIPELINE_VERSION,
        "in_channels": 5, "aux_channels": 24, "pos_weight": 18.5,
    }, path)

    ck = torch.load(path, weights_only=False, map_location="cpu")
    assert ck.get("criterion_state_dict"), "the sweep checkpoint lost its criterion"
    restored = criterion_from_state(ck["criterion_state_dict"], Config().model)
    assert torch.allclose(restored.log_var_reg, crit.log_var_reg)
    assert torch.allclose(restored.log_var_cls, crit.log_var_cls)


def test_sweep_source_persists_and_refuses_a_criterion_free_checkpoint():
    """The driver must both SAVE the key and REFUSE to resume without it.
    Asserted against the source because reaching the code path needs a trained
    cohort, and the failure being guarded is a line going missing."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts",
        "run_ohio_personalization_sweep.py")).read()
    assert '"criterion_state_dict":' in src, "the sweep no longer saves it"
    assert 'criterion_from_state(' in src, "the sweep no longer restores it"
    assert 'not ck.get("criterion_state_dict")' in src, (
        "the sweep no longer refuses to resume from a checkpoint without one, "
        "so a resumed run would adapt under a uniform task balance")


def test_cross_validate_fold_checkpoint_carries_the_criterion():
    """`FoldResult` has carried `criterion_state_dict` and `aux_scaler` all
    along; `cross_validate` dropped both when it wrote the file, so a multimodal
    fold could not be reloaded at all."""
    import inspect

    from causalcgm_mt.train import cross_validate

    src = inspect.getsource(cross_validate)
    for key in ("criterion_state_dict", "aux_scaler", "config"):
        assert f'"{key}"' in src, f"cross_validate no longer persists {key}"


# --------------------------------------------------------------------------- #
# 3. One checkpoint schema, and it is exportable
# --------------------------------------------------------------------------- #
def _minimal_ckpt(**over) -> dict:
    ck = {"state_dict": {}, "in_channels": 5,
          "config": dataclasses.asdict(Config()), "scaler": object()}
    ck.update(over)
    return ck


def test_schema_check_names_every_missing_key():
    """The failure used to be `KeyError: 'scaler'` from four frames down, which
    says nothing about which driver wrote a short checkpoint or why the key
    matters."""
    with pytest.raises(ValueError, match=r"missing \['scaler'\]"):
        assert_checkpoint_schema({k: v for k, v in _minimal_ckpt().items()
                                  if k != "scaler"}, "x.pt")
    with pytest.raises(ValueError, match="config"):
        assert_checkpoint_schema({k: v for k, v in _minimal_ckpt().items()
                                  if k != "config"}, "x.pt")
    # A well-formed one passes.
    assert_checkpoint_schema(_minimal_ckpt(), "x.pt") is None


def test_multimodal_checkpoints_additionally_require_the_aux_scaler():
    """Exporting a fusion model without its auxiliary scaler used to fall back
    to the CGM-only wrapper -- a graph of a different architecture than the
    model being reported."""
    ck = _minimal_ckpt(multimodal=True, aux_channels=24)
    with pytest.raises(ValueError, match="aux_scaler"):
        assert_checkpoint_schema(ck, "mm.pt")
    ck["aux_scaler"] = object()
    assert_checkpoint_schema(ck, "mm.pt") is None
    # ...and a unimodal checkpoint is not asked for one.
    assert set(REQUIRED_MULTIMODAL_KEYS) - set(REQUIRED_CHECKPOINT_KEYS)


@pytest.mark.parametrize("driver", ["train_ohio", "cross_validate", "train_demo"])
def test_every_driver_writes_the_full_schema(driver):
    """The defect was three drivers writing three shapes. Asserted at the source
    because two of the three need licensed data or a full training run to reach,
    and what regressed was a key going missing from a dict literal."""
    import inspect

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if driver == "cross_validate":
        from causalcgm_mt.train import cross_validate
        src = inspect.getsource(cross_validate)
    else:
        src = open(os.path.join(repo, "scripts", f"{driver}.py")).read()

    for key in REQUIRED_CHECKPOINT_KEYS:
        assert f'"{key}"' in src, (
            f"{driver} does not write '{key}', so its checkpoints cannot be "
            f"exported -- see export.REQUIRED_CHECKPOINT_KEYS")


def test_train_ohio_saves_the_fitted_scalers_not_just_weights():
    """The specific omission behind audit finding F6: `train_ohio.py` wrote
    weights + config and no scaler, so the only exportable checkpoint in the
    repository was the synthetic demo's."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo, "scripts", "train_ohio.py")).read()
    assert '"scaler": pre.cgm_scaler' in src
    assert '"aux_scaler": pre.aux_scaler if multimodal else None' in src
    # The preprocessor has to reach the save site to be saved at all.
    assert "art[\"preprocessor\"]" in src


def test_load_and_export_dispatches_on_the_multimodal_flag():
    """A fusion checkpoint must build the fusion wrapper. Reading the source
    rather than exporting one, because a real export needs onnxruntime and a
    trained fusion model; `tests/test_export.py` covers the numeric path."""
    import inspect

    from causalcgm_mt.export import load_and_export

    src = inspect.getsource(load_and_export)
    assert 'ck.get("multimodal")' in src
    assert "build_multimodal_model" in src
    assert 'aux_scaler=ck.get("aux_scaler")' in src


def test_demo_checkpoint_round_trips_through_the_schema_check(tmp_path):
    """An end-to-end shape check on the one checkpoint the repo can build
    without licensed data: train a single fold on synthetic windows, write it
    the way `train_demo.py` does, and require the schema gate to accept it."""
    from causalcgm_mt.data import build_windows
    from causalcgm_mt.simulate import simulate_cohort
    from causalcgm_mt.train import run_fold

    cfg = Config(out_dir=str(tmp_path))
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, max_epochs=1, n_folds=2,
                                       batch_size=64))
    df = simulate_cohort(n_patients=6, days=4.0, seed=0)
    w = build_windows(df, cfg.data)
    pats = sorted({str(p) for p in w.patient})
    r = run_fold(w, cfg, np.array(pats[:4], dtype=object),
                 np.array(pats[4:], dtype=object), fold=0)

    ck = {
        "state_dict": r.state_dict,
        "criterion_state_dict": r.criterion_state_dict,
        "scaler": r.scaler,
        "aux_scaler": r.aux_scaler,
        "labeler": r.labeler,
        "multimodal": False,
        "aux_channels": 0,
        "in_channels": int(w.x.shape[-1]),
        "config": dataclasses.asdict(cfg),
    }
    assert_checkpoint_schema(ck, "demo.pt")
    # The criterion survives the round trip through the file, not just the dict.
    path = tmp_path / "demo.pt"
    torch.save(ck, path)
    back = torch.load(path, weights_only=False, map_location="cpu")
    assert_checkpoint_schema(back, str(path))
    assert criterion_from_state(back["criterion_state_dict"], cfg.model) is not None

    # And `config` survives JSON, which is the path it takes to
    # `Config.from_dict` inside `load_and_export`. Asserted field by field on
    # what the export depends on rather than by whole-object equality: `asdict`
    # renders the two SEQUENCE fields (`data.exog_cols`,
    # `adapt.lora_attention_targets`) as lists and `Config.from_dict` does not
    # convert them back, so `from_dict(asdict(cfg)) != cfg` for reasons that
    # predate this schema and do not reach the graph. `history_steps` is what
    # sizes the ONNX input, and `exog_cols`' truthiness is what `build_windows`
    # branches on; both round-trip exactly.
    rt = Config.from_dict(json.loads(json.dumps(back["config"])))
    assert rt.data.history_steps == cfg.data.history_steps
    assert rt.data.horizon_min == cfg.data.horizon_min
    assert rt.model == cfg.model
    assert bool(rt.data.exog_cols) == bool(cfg.data.exog_cols)
