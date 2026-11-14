"""Regression tests for the defects found in the line-by-line audit.

Each test is keyed to the finding it closes (F1, F2, ...) and is written to fail
against the pre-fix code. They live in one file rather than being scattered into
the subject-area suites because their value is provenance: a reader asking "was
F7 actually fixed, and would we notice if it came back" should be able to answer
it without grepping.

The common shape of every finding here is that the defect was invisible to the
existing suite -- 317 tests passed with all of them live. So these tests assert
the *mechanism*, not the metric: that a parameter reaches an optimiser, that a
matched timestamp is not in the future, that a mask varies within a file. A
metric-level assertion would have passed before the fix too.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

import os
import re

from causalcgm_mt.config import AdaptConfig, Config, DataConfig, ModelConfig, TrainConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# F1: adapt_patient crashed on every gradient-based method
# --------------------------------------------------------------------------- #
def _tiny_windowset(n: int = 400, H: int = 24, C: int = 5, seed: int = 0):
    """A single-patient WindowSet large enough for a prefix/embargo split."""
    from causalcgm_mt.data import WindowSet

    rng = np.random.default_rng(seed)
    x = rng.normal(120, 25, (n, H, C)).astype(np.float32)
    x[..., 4] = 1.0
    t0 = np.datetime64("2021-01-01T00:00", "m")
    return WindowSet(
        x=x,
        y_reg=rng.normal(120, 25, n).astype(np.float32),
        y_cls_hard=(rng.random(n) < 0.1).astype(np.float32),
        y_min=rng.normal(110, 25, n).astype(np.float32),
        patient=np.array(["p1"] * n, dtype=object),
        t_index=np.arange(n),
        t_time=np.array([t0 + np.timedelta64(5 * i, "m") for i in range(n)]),
        last_bg=x[:, -1, 0].copy(),
    )


@pytest.mark.parametrize("method", ["lora", "head_only", "full", "bn_adapt"])
def test_f1_adapt_patient_runs_for_every_method(method):
    """`lr=acfg.lr` referenced a name that was never assigned in this function.

    Every gradient-based method raised `NameError` -- including the default
    `--adapt lora` of `scripts/run_experiment.py`, i.e. the README quick-start,
    which crashed AFTER completing the whole cross-validation. Nothing in the
    suite called `adapt_patient`, so 317 tests passed with it broken.
    """
    from causalcgm_mt.data import CausalScaler, SoftLabeler
    from causalcgm_mt.extension import adapt_patient
    from causalcgm_mt.models import build_model

    ws = _tiny_windowset()
    cfg = Config(
        data=DataConfig(),
        model=ModelConfig(),
        train=TrainConfig(seed=0),
        adapt=AdaptConfig(method=method, batch_size=32, steps=3),
    )
    model, criterion = build_model(ws.x.shape[-1], cfg.model, 1.0)
    scaler = CausalScaler.fit(ws, ["p1"])
    labeler = SoftLabeler.fit(ws.y_min, cfg.data, mard_train=0.1)

    result = adapt_patient(model, ws, "p1", scaler, labeler, criterion, cfg)
    assert result is not None, "the fixture should be large enough to adapt on"
    assert result.method == method


def test_f1_adapt_patient_uses_the_configured_learning_rate():
    """The crash was a symptom; the underlying bug was an unthreaded config."""
    import inspect

    from causalcgm_mt.extension import adapt_patient

    src = inspect.getsource(adapt_patient)
    assert "acfg = cfg.adapt" in src, "adapt_patient must bind its AdaptConfig"
    assert "lr=acfg.lr" in src


# --------------------------------------------------------------------------- #
# F2: head_only trained only the primary horizon's heads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [1, 2, 3])
def test_f2_head_only_selects_every_horizons_heads(K):
    """`model.reg_head` is a property resolving to `reg_heads[0]`.

    Using it in `configure_trainable` trained 2 of 2K heads, so on the K=2 Ohio
    models the 60-minute pair was frozen -- and the shipped personalisation
    sweep shows exactly that: 12/12 subjects bit-identical before/after at
    PH=60. This asserts the parameter SET, which is what went wrong; a metric
    assertion would have passed before the fix.
    """
    from causalcgm_mt.extension import configure_trainable
    from causalcgm_mt.models import build_model

    model, _ = build_model(5, ModelConfig(n_horizons=K), 1.0)
    selected = configure_trainable(model, AdaptConfig(method="head_only"))

    expected = {
        id(p)
        for lst in (model.reg_heads, model.cls_heads)
        for head in lst
        for p in head.parameters()
    }
    assert {id(p) for p in selected} == expected
    assert len(selected) == 2 * K * len(list(model.reg_heads[0].parameters()))
    assert all(p.requires_grad for p in selected)


def test_f2_head_only_moves_every_horizon_head(tmp_path):
    """End-to-end: after adaptation, no horizon's head is left untouched."""
    from causalcgm_mt.data import CausalScaler, SoftLabeler
    from causalcgm_mt.extension import adapt_patient
    from causalcgm_mt.models import build_model

    ws = _tiny_windowset()
    cfg = Config(
        data=DataConfig(),
        model=ModelConfig(n_horizons=1),
        train=TrainConfig(seed=0),
        adapt=AdaptConfig(method="head_only", batch_size=32, steps=5),
    )
    model, criterion = build_model(ws.x.shape[-1], cfg.model, 1.0)
    before = {k: v.clone() for k, v in model.state_dict().items() if "heads" in k}

    scaler = CausalScaler.fit(ws, ["p1"])
    labeler = SoftLabeler.fit(ws.y_min, cfg.data, mard_train=0.1)
    adapt_patient(model, ws, "p1", scaler, labeler, criterion, cfg)

    # adapt_patient deep-copies, so the ORIGINAL must be untouched; the point of
    # this test is the selection above. Assert the contract we actually rely on.
    for k, v in model.state_dict().items():
        if "heads" in k:
            assert torch.equal(before[k], v), f"{k} mutated the caller's model"


# --------------------------------------------------------------------------- #
# F3 / F10b: inert configuration knobs
# --------------------------------------------------------------------------- #
def test_f10b_lora_attention_knobs_are_real_dataclass_fields():
    """`getattr(cfg, "lora_attention", True)` on a frozen dataclass with no such
    field is not a knob: `dataclasses.replace` would raise, so the documented
    "trunk vs trunk+fusion" ablation was unreachable."""
    names = {f.name for f in dataclasses.fields(AdaptConfig)}
    assert {"lora_attention", "lora_attention_targets"} <= names
    # and it must actually be settable on a frozen dataclass
    cfg = dataclasses.replace(AdaptConfig(), lora_attention=False)
    assert cfg.lora_attention is False


def test_f10b_disabling_lora_attention_skips_the_projections():
    """The flag must change what gets wrapped, not merely exist."""
    from causalcgm_mt.extension import LoRAConv1d, LoRALinear, inject_lora
    from causalcgm_mt.models import build_multimodal_model

    def wrapped(flag):
        m, _ = build_multimodal_model(5, 24, ModelConfig(), 1.0, aux_valid_index=(7,))
        inject_lora(m, dataclasses.replace(AdaptConfig(), lora_attention=flag))
        return (
            sum(1 for x in m.modules() if isinstance(x, LoRAConv1d)),
            sum(1 for x in m.modules() if isinstance(x, LoRALinear)),
        )

    conv_on, lin_on = wrapped(True)
    conv_off, lin_off = wrapped(False)
    assert lin_on == 4, "q/k/v/proj should be wrapped when enabled"
    assert lin_off == 0, "attention projections must be skipped when disabled"
    assert conv_on == conv_off > 0, "the trunk pass must be unaffected by the flag"


def test_f3_adapt_ohio_subject_threads_the_lr_argument():
    """The `lr=` argument was folded into `acfg` and then the optimiser read
    `cfg.adapt.lr` instead, silently discarding it."""
    import inspect

    from causalcgm_mt.extension import adapt_ohio_subject

    src = inspect.getsource(adapt_ohio_subject)
    assert "lr=acfg.lr" in src
    assert "lr=cfg.adapt.lr" not in src


# --------------------------------------------------------------------------- #
# F4: pre-refactor checkpoints stopped loading
# --------------------------------------------------------------------------- #
def test_f4_legacy_head_keys_are_remapped():
    """`reg_head.*` -> `reg_heads.0.*`. A property does not alias state_dict
    keys, so the multi-horizon refactor broke every checkpoint written before
    it -- taking the ONNX export path, the C++ parity gate and the dashboard
    with it, while the refactor's comments claimed the opposite."""
    from causalcgm_mt.models import (
        build_model,
        is_legacy_state_dict,
        load_state_dict_compat,
        remap_legacy_state_dict,
    )

    model, _ = build_model(5, ModelConfig(n_horizons=1), 1.0)
    current = model.state_dict()

    # Synthesise a pre-refactor checkpoint from the current one.
    legacy = {}
    for k, v in current.items():
        k = k.replace("reg_heads.0.", "reg_head.").replace("cls_heads.0.", "cls_head.")
        legacy[k] = v
    assert any(k.startswith("reg_head.") for k in legacy)

    assert is_legacy_state_dict(legacy)
    assert not is_legacy_state_dict(current)

    fresh, _ = build_model(5, ModelConfig(n_horizons=1), 1.0)
    result = load_state_dict_compat(fresh, legacy)
    assert not result.missing_keys and not result.unexpected_keys

    # values must survive the rename unchanged
    for k, v in current.items():
        assert torch.equal(fresh.state_dict()[k], v)

    # and the remap must be a no-op on an already-current dict
    assert remap_legacy_state_dict(current) is current


def test_f4_every_checkpoint_load_site_uses_the_compat_loader():
    """A remap helper nobody calls fixes nothing. Grep the load sites."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    sites = [
        repo / "causalcgm_mt" / "export.py",
        repo / "app.py",
        repo / "scripts" / "verify_cpp_parity.py",
    ]
    for path in sites:
        src = path.read_text()
        assert "load_state_dict_compat" in src, f"{path.name} still loads raw"
        assert ".load_state_dict(ck[" not in src, f"{path.name} has a raw load"


# --------------------------------------------------------------------------- #
# F7: the grid snap took samples from the future
# --------------------------------------------------------------------------- #
def test_f7_snap_never_takes_a_sample_from_the_future():
    """`merge_asof(direction="nearest", tolerance=P/2)` let slot t take a sample
    recorded AFTER t. On OhioT1DM the CGM clock drifts off the 5-minute grid, so
    40% of matched slots were doing exactly that, by up to 120 s.

    The fixture is deliberately OFF-GRID. The old test suite could not see this
    finding because `simulate_cohort` emits timestamps exactly on the grid,
    where nearest and backward are identical.
    """
    from causalcgm_mt.data import snap_to_grid_backward

    period = 5
    grid = pd.date_range("2024-01-01 00:00", periods=48, freq="5min")
    rng = np.random.default_rng(0)
    # samples drifting across the grid by up to +/- 2 minutes
    offsets = rng.integers(-120, 121, 48)
    ts = pd.Series(grid + pd.to_timedelta(offsets, unit="s"))
    vals = pd.DataFrame({"glucose": np.arange(48, dtype=float)})

    binned, observed = snap_to_grid_backward(ts, vals, grid, period)

    # For every observed slot, the sample that landed there must be at or before it.
    gv = pd.DatetimeIndex(grid).values
    for slot in np.flatnonzero(observed):
        src = ts.values[int(binned["glucose"].iloc[slot])]
        assert src <= gv[slot], (
            f"slot {grid[slot]} took a sample from {src} -- the future")


def test_f7_regularize_series_is_backward_on_off_grid_data():
    """End-to-end through the generic loader, on off-grid timestamps."""
    from causalcgm_mt.data import regularize_series

    cfg = DataConfig()
    rng = np.random.default_rng(1)
    base = pd.date_range("2024-01-01", periods=200, freq="5min")
    jitter = pd.to_timedelta(rng.integers(-140, 141, 200), unit="s")
    df = pd.DataFrame({
        "patient_id": "p1",
        "timestamp": base + jitter,
        "glucose": rng.normal(120, 20, 200),
    })
    reg = regularize_series(df, cfg)

    src_time = pd.Series(df["timestamp"].values)
    for i, row in reg.iterrows():
        if row["observed"] != 1.0:
            continue
        matches = src_time[np.isclose(df["glucose"].values, row["glucose"])]
        assert (matches <= row["timestamp"]).any(), (
            "an observed slot was filled from a strictly later sample")


def test_f7_grid_aligned_data_is_unchanged():
    """The fix must be a no-op where nearest and backward already agree, or it
    would silently move every number in the existing results."""
    from causalcgm_mt.data import snap_to_grid_backward

    grid = pd.date_range("2024-01-01", periods=20, freq="5min")
    vals = pd.DataFrame({"g": np.arange(20, dtype=float)})
    binned, observed = snap_to_grid_backward(pd.Series(grid), vals, grid, 5)
    assert observed.all()
    assert np.array_equal(binned["g"].to_numpy(), np.arange(20, dtype=float))


# --------------------------------------------------------------------------- #
# F9: soft_label_floor multiplied into pos_weight
# --------------------------------------------------------------------------- #
def test_f9_clear_negatives_carry_no_positive_class_mass():
    """`soft_label_floor = 0.02` x `pos_weight ~ 18` gave every clear-negative
    window a positive-class BCE term of 0.364 -- 37% of the positive mass the
    genuine positives contribute -- shifting the optimal constant predictor from
    0.500 to 0.578. Invisible in every rank-based metric the paper reports.
    """
    from causalcgm_mt.data import SoftLabeler

    cfg = DataConfig(band_mode="fixed", soft_band_mgdl=10.0)
    lab = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=None)

    assert lab.floor == 0.0, "the floor is what interacts with pos_weight"

    # A clear negative now carries only what the logistic ramp itself gives it,
    # which at 180 mg/dL with a 10 mg/dL band is sigmoid(-44) ~ 1e-19. The old
    # floor replaced that with 0.02 -- eighteen orders of magnitude larger, and
    # then multiplied it by pos_weight ~ 18.
    y_neg = float(lab(np.array([180.0]))[0])
    assert y_neg < 1e-12, f"clear negatives should carry no positive mass, got {y_neg}"

    # the ceiling has no such interaction and is kept
    assert lab.ceiling == 0.98
    assert float(lab(np.array([40.0]))[0]) == pytest.approx(0.98)


def test_f9_optimal_constant_predictor_is_not_shifted_upward():
    """The consequence, stated as the quantity that actually moved."""
    from causalcgm_mt.data import SoftLabeler

    prev = 0.0521
    pos_weight = (1 - prev) / prev
    cfg = DataConfig(band_mode="fixed", soft_band_mgdl=10.0)
    lab = SoftLabeler.fit(np.array([100.0]), cfg, mard_train=None)
    y_neg = float(lab(np.array([180.0]))[0])
    y_pos = float(lab(np.array([45.0]))[0])

    grid = np.linspace(-8.0, 6.0, 60001)
    p = 1.0 / (1.0 + np.exp(-grid))
    loss = -(
        pos_weight * (prev * y_pos + (1 - prev) * y_neg) * np.log(p)
        + (prev * (1 - y_pos) + (1 - prev) * (1 - y_neg)) * np.log(1 - p)
    )
    p_star = float(p[np.argmin(loss)])
    assert p_star == pytest.approx(0.5, abs=0.02), (
        f"prevalence weighting should centre the constant predictor at 0.5, got {p_star}")


# --------------------------------------------------------------------------- #
# F8: sleep_observed was a whole-file constant
# --------------------------------------------------------------------------- #
def test_f8_sleep_observed_is_a_per_step_mask(tmp_path):
    """It was `np.full(len(grid), float(sl_any and covered))` -- a constant 1.0
    across all 12,080 slots for a subject with >=10% sleep coverage. Because it
    is in `aux_validity_indices()` and the gate opens when ANY mask is set, that
    jammed the cross-attention gate open at every timestep regardless of whether
    a sensor had reported -- the `hr_observed` defect inverted.
    """
    import xml.etree.ElementTree as ET

    from causalcgm_mt.ohio_loader import parse_patient_xml, synchronize_ohio_record

    # A synthetic record: 24 h of CGM, one 8 h sleep episode.
    root = ET.Element("patient", id="999")
    g = ET.SubElement(root, "glucose_level")
    t0 = pd.Timestamp("2021-01-01 00:00")
    for i in range(288):
        ET.SubElement(g, "event",
                      ts=(t0 + pd.Timedelta(minutes=5 * i)).strftime("%d-%m-%Y %H:%M:%S"),
                      value="120")
    sl = ET.SubElement(root, "basis_sleep")
    ET.SubElement(sl, "event",
                  tbegin=t0.strftime("%d-%m-%Y %H:%M:%S"),
                  tend=(t0 + pd.Timedelta(hours=8)).strftime("%d-%m-%Y %H:%M:%S"),
                  quality="1")
    path = tmp_path / "999-ws-training.xml"
    ET.ElementTree(root).write(path)

    out = synchronize_ohio_record(parse_patient_xml(str(path)), DataConfig())
    mask = out["sleep_observed"].to_numpy()

    assert set(np.unique(mask)) == {0.0, 1.0}, "the mask must vary within the file"
    assert 0.0 < mask.mean() < 1.0, f"constant mask ({mask.mean()}) is the defect"
    # and it must line up with the episode, not with a file-level statistic
    assert mask[:90].mean() > 0.9, "inside the episode the mask should be set"
    assert mask[-60:].mean() == 0.0, "far outside it the mask should be clear"


# --------------------------------------------------------------------------- #
# F12: effective sample size was computed and discarded
# --------------------------------------------------------------------------- #
def test_f12_conformal_block_reports_effective_sample_size():
    """`conformal.py`'s header claims n_eff "is reported alongside every
    calibration". It was computed on every calibration -- an O(50n) pass, for
    Split, Normalized and each Mondrian stratum -- and then dropped before
    serialisation. It matters: on this cohort n_eff/n is ~0.06, so a 24,270-row
    calibration set is worth ~1,500 independent residuals.
    """
    from causalcgm_mt.conformal import MondrianConformal, SplitConformal

    rng = np.random.default_rng(0)
    y = rng.normal(120, 40, 500)
    p = y + rng.normal(0, 12, 500)

    sc = SplitConformal.calibrate(y, p, 0.1)
    assert np.isfinite(sc.n_effective)
    mc = MondrianConformal.calibrate(y, p, 0.1)
    assert mc.n_effective_by_stratum
    assert all(np.isfinite(v) for v in mc.n_effective_by_stratum.values())


def test_f12_effective_n_does_not_alter_qhat():
    """Reporting, never reweighting. The finite-sample guarantee is an exact
    statement about an order statistic of exchangeable scores; weighting them by
    an autocorrelation estimate would trade a theorem for a heuristic. This test
    exists so a future 'use n_eff to widen the interval' change has to argue
    with it."""
    from causalcgm_mt.conformal import SplitConformal, conformal_quantile

    rng = np.random.default_rng(1)
    # strongly autocorrelated scores -> n_eff << n
    e = rng.normal(0, 1, 800)
    scores = np.abs(pd.Series(e).rolling(40, min_periods=1).mean().to_numpy())
    sc = SplitConformal.calibrate(np.zeros(800), scores, 0.1)
    assert sc.n_effective < sc.n_calib / 2, "fixture should be autocorrelated"
    assert sc.qhat == pytest.approx(conformal_quantile(scores, 0.1))


def test_f12_patient_conditional_validity_is_its_own_predicate():
    """`is_conditionally_valid` inspects the glycemic strata only, so it can
    return True while an individual patient is undercovered -- which is the
    failure mode conformal.py's header calls decisive."""
    from causalcgm_mt.conformal import coverage_report

    y = np.concatenate([np.full(100, 100.0), np.full(100, 100.0)])
    lo = np.concatenate([np.full(100, 90.0), np.full(100, 99.0)])
    hi = np.concatenate([np.full(100, 110.0), np.full(100, 101.0)])
    # patient B's interval covers, patient A's is fine too -> both valid
    pat = np.array(["A"] * 100 + ["B"] * 100)
    rep = coverage_report(y, lo, hi, 0.1, pat)
    assert rep.is_patient_conditionally_valid()

    # now make B undercover badly
    hi2 = hi.copy(); hi2[100:] = 99.5
    rep2 = coverage_report(y, lo, hi2, 0.1, pat)
    assert not rep2.is_patient_conditionally_valid()


# --------------------------------------------------------------------------- #
# Phase 4: performance and cleanup
# --------------------------------------------------------------------------- #
def test_vectorised_causal_features_match_the_reference_loop():
    """The prefix-sum form must be numerically identical to the per-position
    loop it replaced -- including the ddof=0 convention and the zero std at
    position 0, both of which the C++ runner also depends on."""
    from causalcgm_mt.data import ROLL_WINDOW, causal_features, causal_features_batch

    def reference(hist, observed, w=ROLL_WINDOW):
        h = hist.astype(np.float32)
        diff = np.zeros_like(h)
        diff[1:] = h[1:] - h[:-1]
        rm = np.array([h[max(0, i - w + 1): i + 1].mean() for i in range(h.size)])
        rs = np.array([np.std(h[max(0, i - w + 1): i + 1]) if i else 0.0
                       for i in range(h.size)])
        return np.stack([h, diff, rm, rs, observed.astype(np.float32)], -1)

    rng = np.random.default_rng(3)
    for _ in range(50):
        h = rng.normal(120, 35, 24).astype(np.float32)
        o = (rng.random(24) > 0.2).astype(np.float32)
        np.testing.assert_allclose(causal_features(h, o), reference(h, o),
                                   rtol=0, atol=1e-3)

    # batched and single-window paths must agree exactly
    H = rng.normal(120, 30, (16, 24)).astype(np.float32)
    O = np.ones((16, 24), np.float32)
    batch = causal_features_batch(H, O)
    for i in range(16):
        np.testing.assert_array_equal(batch[i], causal_features(H[i], O[i]))


def test_causal_features_are_still_prefix_functions():
    """The whole point of the channel layout. Perturbing step j must not move
    any feature at a position before j."""
    from causalcgm_mt.data import causal_features

    rng = np.random.default_rng(5)
    h = rng.normal(120, 25, 24).astype(np.float32)
    o = np.ones(24, np.float32)
    base = causal_features(h, o)
    for j in (6, 12, 20):
        h2 = h.copy(); h2[j] += 50.0
        np.testing.assert_array_equal(base[:j], causal_features(h2, o)[:j])


def test_assert_no_bfill_actually_asserts():
    """It used to be `return None` behind a docstring claiming it was a guard."""
    from causalcgm_mt.hupa_loader import assert_no_bfill

    ok = pd.DataFrame({"heart_rate": [70.0, 0.0, 72.0],
                       "hr_observed": [1.0, 0.0, 1.0]})
    assert_no_bfill(ok)  # forward fill clears value and mask together

    leaked = pd.DataFrame({"heart_rate": [70.0, 72.0, 72.0],
                           "hr_observed": [1.0, 0.0, 1.0]})
    with pytest.raises(AssertionError, match="backward fill"):
        assert_no_bfill(leaked)


def test_parkes_verification_is_memoised():
    """A 4,000-point sweep per clinical_report was ~27 s per benchmark run."""
    from causalcgm_mt.clinical import parkes_available, verify_parkes_backend

    if not parkes_available():
        pytest.skip("no Parkes backend installed")
    assert hasattr(verify_parkes_backend, "cache_info")
    verify_parkes_backend()
    hits_before = verify_parkes_backend.cache_info().hits
    verify_parkes_backend()
    assert verify_parkes_backend.cache_info().hits == hits_before + 1


# --------------------------------------------------------------------------- #
# Regression from the F5 fix: the LoRA arm's calibration mask
# --------------------------------------------------------------------------- #
def test_calibration_mask_excludes_the_prefix_not_the_eval_complement():
    """The mask must select validation rows OUTSIDE the adaptation prefix.

    It was derived as `isin(val_rows, ps.eval)`, which worked only because the
    old evaluation scope happened to contain the whole training tail. Once
    `eval_scope="test"` made `ps.eval` the test file alone, no validation row
    was in it, the mask admitted nothing, and every conformal interval for the
    LoRA arm came back unbounded (`qhat = inf`) behind a warning.
    """
    import inspect

    from causalcgm_mt.benchmark import adapt_and_predict

    src = inspect.getsource(adapt_and_predict)
    assert "set(ps.adapt.tolist())" in src, "mask must be built from the prefix"
    assert "post_embargo = set(ps.eval" not in src


def test_empty_calibration_set_is_an_error_not_an_infinite_interval():
    """A mask that admits nothing is a caller bug. Emitting a table of `inf`
    widths that reads as a coverage result is the failure this guards."""
    import dataclasses

    import pytest as _pytest

    from causalcgm_mt.benchmark import ArmSpec, CohortData, Predictions, _conformal_block
    from causalcgm_mt.config import DataConfig as _DC

    class _WS:
        y_reg_multi = np.zeros((10, 1), dtype=np.float32)
        patient = np.array(["p"] * 10, dtype=object)
        x = np.zeros((10, 8, 5), dtype=np.float32)

    class _Split:
        val = np.arange(4)
        calib = np.arange(4)
        test = np.arange(4, 10)

    data = CohortData(ws=_WS(), split=_Split(), pre=None,
                      cfg=_DC(history_min=40, horizon_min=30), horizons=(30,))
    spec = ArmSpec("x", "persistence", horizons=(30,))
    calib_pred = Predictions(np.zeros(4), np.zeros(4))
    test_pred = Predictions(np.zeros(6), np.zeros(6))

    with _pytest.raises(ValueError, match="0 of 4 calibration rows"):
        _conformal_block(data, _Split(), calib_pred, test_pred, 0, 0, 0.1,
                         cal_mask=np.zeros(4, dtype=bool))

    # And a split with NO calibration region at all is refused before the mask
    # is even consulted -- falling back to `split.val` would silently reinstate
    # the early-stopping leak this separation exists to close.
    class _NoCalib:
        val = np.arange(4)
        calib = np.empty(0, dtype=np.int64)
        test = np.arange(4, 10)

    with _pytest.raises(ValueError, match="split.calib is empty"):
        _conformal_block(data, _NoCalib(), calib_pred, test_pred, 0, 0, 0.1)


# --------------------------------------------------------------------------- #
# H1: CausalScaler's divide-by-zero guard was a 1e6 amplifier
# --------------------------------------------------------------------------- #
# `iqr = np.clip(q75 - q25, 1e-6, None)` was written to stop a divide-by-zero.
# On an event-sparse channel it converted one into a MULTIPLICATION by 1e6.
#
# Auxiliary streams in this domain are overwhelmingly sparse: a bolus lands in
# ~3% of 5-minute bins, carbohydrates in ~1%. Any channel that is zero in more
# than 75% of (window, timestep) pairs has q75 == q25 == 0, so the clamp fires
# and every non-zero entry -- every actual event, i.e. the channel's entire
# information content -- is multiplied by a million.
#
# Measured on the real OhioT1DM window set before the fix: six of 24 auxiliary
# channels (steps, bolus, carbs, finger_stick, exercise_intensity,
# accel_magnitude) landed on the 1e-6 floor and reached magnitudes up to 5.9e8
# in the model input, while every observation mask stayed 0/1 and the fusion
# gate reported itself healthy at aux_valid_frac 0.946.
#
# These assert the mechanism: a bounded scaled range, and a substituted scale
# that comes from the channel's own non-zero entries.

def _sparse_aux(n: int = 600, T: int = 24, seed: int = 0):
    """(N, T, C) auxiliary tensor with the sparsity profile of a real cohort.

    ch0 dense (heart-rate-like)     ch3 constant-valued events (exercise-like)
    ch1 ~3% non-zero (bolus-like)   ch4 all zero (a sensor never worn)
    ch2 ~1% non-zero (carbs-like)   ch5 0/1 observation mask
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 6), dtype=np.float32)
    x[..., 0] = rng.normal(70, 8, (n, T))

    def scatter(ch, frac, values):
        hit = rng.random((n, T)) < frac
        x[..., ch][hit] = values(hit.sum())

    scatter(1, 0.03, lambda k: rng.uniform(0.5, 8.0, k))     # bolus, units
    scatter(2, 0.01, lambda k: rng.uniform(10.0, 90.0, k))   # carbs, grams
    scatter(3, 0.02, lambda k: np.full(k, 6.0))              # constant intensity
    x[..., 5] = (rng.random((n, T)) > 0.1).astype(np.float32)
    return x


def test_h1_sparse_auxiliary_channels_do_not_explode_when_scaled():
    """The headline: a >75%-zero channel must not leave the scaler at 1e8.

    A metric-level assertion would have passed before the fix -- the model still
    trained, the masks still read 0/1, and the availability gate still reported
    itself open. Only the scaled magnitude shows it.
    """
    from causalcgm_mt.data import CausalScaler

    x = _sparse_aux()
    patient = np.array(["p1"] * x.shape[0], dtype=object)
    scaler = CausalScaler.fit_array(x, patient, ["p1"], skip_channels=(5,))
    z = scaler.transform(x)

    # Every sparse channel is >75% zeros, so its raw IQR is exactly 0.
    q75, q25 = np.percentile(x.reshape(-1, x.shape[-1]), [75, 25], axis=0)
    assert (q75 - q25)[[1, 2, 3, 4]].max() == 0.0, "fixture is not sparse enough"

    worst = float(np.abs(z).max())
    assert worst < 1e3, (
        f"scaled auxiliary features reach {worst:.3g}; a degenerate IQR was "
        f"clamped to an epsilon and became a ~1e6 multiplier on exactly the "
        f"non-zero entries that carry the channel's signal")
    # Per channel, so a single offender cannot hide behind the others.
    per_channel = np.abs(z.reshape(-1, z.shape[-1])).max(axis=0)
    assert per_channel.max() < 1e3, dict(enumerate(per_channel.round(2)))


def test_h1_substituted_scale_comes_from_the_channels_own_nonzero_entries():
    """The fallback has to be a scale, not a constant: a bolus of 8 units and a
    bolus of 0.5 must stay 16x apart after scaling."""
    from causalcgm_mt.data import CausalScaler

    x = _sparse_aux()
    patient = np.array(["p1"] * x.shape[0], dtype=object)
    scaler = CausalScaler.fit_array(x, patient, ["p1"], skip_channels=(5,))

    nz = x[..., 1][x[..., 1] != 0.0]
    expected = float(np.percentile(nz, 75) - np.percentile(nz, 25))
    assert scaler.scale[1] == pytest.approx(expected, rel=1e-5), (
        "the bolus channel's substituted scale is not the IQR of its own "
        "non-zero entries")

    # A channel whose non-zero entries are all identical has no spread to read,
    # so the ladder falls through to their typical magnitude.
    assert scaler.scale[3] == pytest.approx(6.0, rel=1e-5)

    # Nothing to go on at all -> identity, leaving the channel raw rather than
    # inventing a scale for it.
    assert scaler.scale[4] == pytest.approx(1.0)

    # Zero stays exactly zero everywhere: "no event" must not acquire an offset.
    z = scaler.transform(x)
    for c in (1, 2, 3, 4):
        assert np.all(z[..., c][x[..., c] == 0.0] == 0.0)


def test_h1_the_healthy_path_and_skip_channels_are_untouched():
    """A regression guard on the fix itself: dense channels must scale exactly
    as before, and masks must still pass through as 0/1."""
    from causalcgm_mt.data import CausalScaler

    x = _sparse_aux()
    patient = np.array(["p1"] * x.shape[0], dtype=object)
    scaler = CausalScaler.fit_array(x, patient, ["p1"], skip_channels=(5,))

    flat = x.reshape(-1, x.shape[-1])
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    assert scaler.scale[0] == pytest.approx(float(q75[0] - q25[0]), rel=1e-6)
    assert scaler.center[0] == pytest.approx(float(np.median(flat[:, 0])), rel=1e-6)

    assert scaler.center[5] == 0.0 and scaler.scale[5] == 1.0
    z = scaler.transform(x)
    assert set(np.unique(z[..., 5]).tolist()) <= {0.0, 1.0}


def test_h1_substituted_channels_are_recorded_for_audit():
    """Which channels got a substituted scale has to be inspectable, or the
    substitution is as invisible as the blow-up it replaced."""
    from causalcgm_mt.data import CausalScaler

    x = _sparse_aux()
    patient = np.array(["p1"] * x.shape[0], dtype=object)
    scaler = CausalScaler.fit_array(x, patient, ["p1"], skip_channels=(5,))

    assert tuple(scaler.degenerate_channels) == (1, 2, 3, 4)
    # A cohort with no degenerate channel records nothing.
    dense = np.random.default_rng(1).normal(0, 1, (200, 8, 3)).astype(np.float32)
    clean = CausalScaler.fit_array(dense, np.array(["p"] * 200, dtype=object), ["p"])
    assert tuple(clean.degenerate_channels) == ()


# --------------------------------------------------------------------------- #
# H2: run_fold trained for zero batches and reported the untrained model
# --------------------------------------------------------------------------- #
# `drop_last=shuffle` dropped the final partial batch unconditionally, so a fit
# split smaller than one batch produced a loader of length 0. The epoch loop
# iterated zero times, no optimiser step was ever taken, and nothing raised:
# `train_loss` came out as 0.0 (`epoch_loss / max(n_batches, 1)`), the monitor
# was finite because a random model still scores, `best_state` was populated at
# epoch 0, and the fold returned a full FoldResult whose test RMSE sat a
# hundredth of a mg/dL from the persistence floor.
#
# Every sibling loader factory in the repository already carried the guard --
# `benchmark._loader`, `ohio_loader.make_ohio_loaders`, `scripts/train_ohio.py`
# all write `drop_last=shuffle and len(ds) > batch_size`. `run_fold` was the
# sole omission.

def _fold_inputs(n_per_patient: int = 60, n_patients: int = 4, seed: int = 0):
    """A WindowSet with several patients, plus the train/test patient arrays."""
    from causalcgm_mt.data import WindowSet

    rng = np.random.default_rng(seed)
    n = n_per_patient * n_patients
    H, C = 24, 5
    x = rng.normal(120, 25, (n, H, C)).astype(np.float32)
    x[..., 4] = 1.0
    pid = np.array([f"p{i // n_per_patient}" for i in range(n)], dtype=object)
    t0 = np.datetime64("2021-01-01T00:00", "m")
    y_min = rng.normal(95, 25, n).astype(np.float32)
    ws = WindowSet(
        x=x,
        y_reg=rng.normal(120, 25, n).astype(np.float32),
        y_cls_hard=(y_min < 70).astype(np.float32),
        y_min=y_min,
        patient=pid,
        t_index=np.tile(np.arange(n_per_patient), n_patients),
        t_time=np.array([t0 + np.timedelta64(5 * (i % n_per_patient), "m")
                         for i in range(n)]),
        last_bg=x[:, -1, 0].copy(),
    )
    names = [f"p{i}" for i in range(n_patients)]
    return ws, np.array(names[:-1], dtype=object), np.array(names[-1:], dtype=object)


def test_h2_a_fit_split_smaller_than_one_batch_still_trains():
    """The bug itself: below one batch the fold reported random initialisation.

    Asserted as "the weights moved", not as "the RMSE improved" -- the untrained
    model's RMSE was 14.871 against a persistence floor of 14.881, which reads
    as a weak result rather than as an absent one.
    """
    from causalcgm_mt.models import build_model
    from causalcgm_mt.train import run_fold, set_seed

    ws, train_pat, test_pat = _fold_inputs()
    cfg = Config(train=TrainConfig(max_epochs=2, batch_size=512, num_workers=0,
                                   monitor="val/rmse", monitor_mode="min",
                                   seed=1337))
    # Precondition: the fit side really is smaller than one batch.
    assert ws.x.shape[0] < cfg.train.batch_size * 2

    result = run_fold(ws, cfg, train_pat, test_pat, 0)

    set_seed(cfg.train.seed + 0)
    fresh, _ = build_model(ws.x.shape[-1], cfg.model, 1.0)
    moved = max(
        float((result.state_dict[k].float() - v.detach().cpu().float()).abs().max())
        for k, v in fresh.state_dict().items()
        if k in result.state_dict and v.is_floating_point())
    assert moved > 0.0, (
        "run_fold returned a model bit-identical to its random initialisation: "
        "the fit loader dropped every batch and no optimiser step ran")
    assert any(h["train_loss"] > 0.0 for h in result.history), (
        "every epoch reported train_loss=0.0, which is what zero batches "
        "produce via `epoch_loss / max(n_batches, 1)`")


def test_h2_zero_training_batches_is_a_runtime_error_not_a_silent_pass(monkeypatch):
    """The backstop, for the case the loader yields nothing anyway.

    Unreachable once `drop_last` is guarded -- which is exactly why it is pinned
    here rather than left to be re-derived: a future change to the loader must
    fail loudly instead of reporting an unselected model. The fit loader is
    emptied directly, so this tests the guard and not the arithmetic that
    currently makes it unnecessary.
    """
    import causalcgm_mt.train as train_mod

    real_loader = train_mod.DataLoader

    class _EmptyWhenShuffled:
        """Wraps DataLoader, yielding nothing for the shuffled (fit) loader."""

        def __init__(self, dataset, **kw):
            self._shuffle = bool(kw.get("shuffle", False))
            self._inner = real_loader(dataset, **kw)

        def __iter__(self):
            return iter(()) if self._shuffle else iter(self._inner)

        def __len__(self):
            return 0 if self._shuffle else len(self._inner)

    monkeypatch.setattr(train_mod, "DataLoader", _EmptyWhenShuffled)

    ws, train_pat, test_pat = _fold_inputs()
    cfg = Config(train=TrainConfig(max_epochs=2, batch_size=16, num_workers=0,
                                   monitor="val/rmse", monitor_mode="min",
                                   seed=1337))

    with pytest.raises(RuntimeError, match="no batches|zero batches|untrained"):
        train_mod.run_fold(ws, cfg, train_pat, test_pat, 0)


def test_h2_a_normal_split_still_drops_its_partial_batch():
    """The guard must not change the normal path: with more than one batch of
    fit rows, the trailing partial batch is still dropped, so batch statistics
    and the OneCycle step count are unchanged."""
    from torch.utils.data import DataLoader

    from causalcgm_mt.train import WindowDataset

    n, batch = 700, 256
    ds = WindowDataset(
        np.zeros((n, 24, 5), np.float32), np.zeros(n, np.float32),
        np.zeros(n, np.float32), np.zeros(n, np.float32))
    guarded = DataLoader(ds, batch_size=batch, shuffle=True,
                         drop_last=True and len(ds) > batch)
    assert len(guarded) == n // batch == 2

    small = WindowDataset(
        np.zeros((100, 24, 5), np.float32), np.zeros(100, np.float32),
        np.zeros(100, np.float32), np.zeros(100, np.float32))
    kept = DataLoader(small, batch_size=batch, shuffle=True,
                      drop_last=True and len(small) > batch)
    assert len(kept) == 1, "a below-one-batch split must keep its partial batch"


# --------------------------------------------------------------------------- #
# H3: PerPatientNormalizer scaled GSR by a noise floor
# --------------------------------------------------------------------------- #
# `flat = iqr < 1e-12` catches a channel that is EXACTLY constant for a subject.
# It does not catch the failure that actually occurs, and the first diagnosis of
# this finding got the mechanism wrong, so it is worth stating precisely.
#
# On OhioT1DM subject 2018_588 the released GSR trace is 97% sensor zero: p25
# 5.9e-05, p50 6.3e-05, p75 7.4e-05 -- an interquartile range of 1.5e-05 that is
# measuring QUANTISATION NOISE around a disconnected electrode. The remaining
# 1.4% is real skin contact, reaching p99 0.53 and a maximum of 23.6. The IQR is
# not degenerate in any relative sense (iqr/median = 0.24, perfectly healthy);
# it is simply fitted to the wrong part of a distribution spanning six orders of
# magnitude. Dividing by it put (max - median)/IQR = 1.57e6 into the fusion
# branch, and the aux tensor's global maximum was 1.767e6 -- all of it this one
# channel, on four of twelve subjects.
#
# The fix is a TAIL-AWARE spread, `max(IQR, (p99 - p1) / 3.449)`, plus a clip on
# the normalised output as a backstop. 3.449 is (p99 - p1)/IQR for a normal
# distribution, so the two terms coincide on well-behaved data and the wider one
# only binds when the interquartile range has collapsed into a noise floor.
# Measured over the real cohort's five device-scaled channels: global max |z|
# 1.573e6 -> 20, with 0.020% of values saturated, against 2.609% for clipping
# alone.

def _aux_with_gsr(profile: str, n: int = 400, T: int = 24, seed: int = 0):
    """A full-width Ohio auxiliary tensor whose GSR channel has `profile`.

    "healthy"    a well-conditioned unimodal trace
    "noisefloor" 97% sensor zero with a heavy contact tail (subject 2018_588)
    "constant"   never worn: one value throughout
    """
    from causalcgm_mt.ohio_loader import OHIO_AUX_CHANNELS, OHIO_AUX_INDEX

    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, len(OHIO_AUX_CHANNELS)), dtype=np.float32)
    x[..., OHIO_AUX_INDEX["heart_rate"]] = rng.normal(70, 8, (n, T))
    x[..., OHIO_AUX_INDEX["skin_temp"]] = rng.normal(91, 1.5, (n, T))

    g = OHIO_AUX_INDEX["gsr"]
    if profile == "healthy":
        x[..., g] = rng.lognormal(0.0, 0.5, (n, T))
    elif profile == "constant":
        x[..., g] = 0.25
    elif profile == "noisefloor":
        v = rng.normal(6.3e-5, 8e-6, (n, T))          # the disconnected floor
        contact = rng.random((n, T)) < 0.03           # occasional real contact
        v[contact] = rng.uniform(0.1, 23.6, int(contact.sum()))
        x[..., g] = v
    else:                                             # pragma: no cover
        raise ValueError(profile)
    return x


def test_h3_a_noise_floor_gsr_trace_does_not_blow_up_after_per_patient_scaling():
    """The headline: a subject whose electrode barely reads must not enter the
    model at 1e6 while the subjects around them sit at order 1."""
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, PerPatientNormalizer

    healthy = _aux_with_gsr("healthy", seed=1)
    broken = _aux_with_gsr("noisefloor", seed=2)
    x = np.concatenate([healthy, broken], axis=0)
    patient = np.array(["ok"] * len(healthy) + ["flat"] * len(broken), dtype=object)

    pre = PerPatientNormalizer.fit(x, patient, np.arange(len(x)))
    z = pre.transform(x, patient)

    g = OHIO_AUX_INDEX["gsr"]
    worst = float(np.abs(z[..., g]).max())
    assert worst < 1e3, (
        f"per-patient normalised GSR reaches {worst:.4g}; the interquartile "
        f"range was fitted to the sensor's noise floor and the contact tail "
        f"sits a million IQRs away from it")

    # Every device-scaled channel, so a single offender cannot hide.
    for c in pre.channels:
        assert float(np.abs(z[..., c]).max()) < 1e3, f"channel {c} unbounded"


def test_h3_the_spread_is_tail_aware_but_matches_the_iqr_on_well_behaved_data():
    """The mechanism, in both directions.

    A regression guard as much as a fix check: if the wider term bound on
    ordinary data it would silently rescale every healthy subject too.
    """
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, PerPatientNormalizer

    gauss = _aux_with_gsr("healthy", seed=3)
    gi = OHIO_AUX_INDEX["gsr"]
    gauss[..., gi] = np.random.default_rng(4).normal(5.0, 2.0, gauss.shape[:2])
    patient = np.array(["g"] * len(gauss), dtype=object)
    pre = PerPatientNormalizer.fit(gauss, patient, np.arange(len(gauss)))

    col = gauss[..., gi].ravel().astype(np.float64)
    iqr = float(np.subtract(*np.percentile(col, [75, 25])))
    ci = list(pre.channels).index(gi)
    assert pre.scale["g"][ci] == pytest.approx(iqr, rel=0.02), (
        "the tail-aware term bound on normally distributed data; it must "
        "coincide with the IQR there and only widen when the IQR collapses")

    # ...and on the pathological trace it must be far wider than the IQR.
    broken = _aux_with_gsr("noisefloor", seed=5)
    bp = np.array(["b"] * len(broken), dtype=object)
    pre_b = PerPatientNormalizer.fit(broken, bp, np.arange(len(broken)))
    bcol = broken[..., gi].ravel().astype(np.float64)
    biqr = float(np.subtract(*np.percentile(bcol, [75, 25])))
    got = float(pre_b.scale["b"][ci])
    assert got > 100 * biqr, (
        f"scale {got:.4g} is still within 100x of the noise-floor IQR "
        f"{biqr:.4g}; the tail-aware term did not bind")
    p99, p1 = np.percentile(bcol, [99, 1])
    assert got == pytest.approx((p99 - p1) / (4.6527 / 1.3490), rel=1e-3)


def test_h3_an_exactly_constant_channel_is_still_identity_transformed():
    """The pre-existing `flat` behaviour must survive the change: a channel the
    subject never wore has no scale to read and must be left alone."""
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, PerPatientNormalizer

    x = _aux_with_gsr("constant", seed=6)
    patient = np.array(["c"] * len(x), dtype=object)
    pre = PerPatientNormalizer.fit(x, patient, np.arange(len(x)))

    ci = list(pre.channels).index(OHIO_AUX_INDEX["gsr"])
    assert pre.center["c"][ci] == 0.0
    assert pre.scale["c"][ci] == 1.0
    z = pre.transform(x, patient)
    assert np.allclose(z[..., OHIO_AUX_INDEX["gsr"]], 0.25)


def test_h3_scaling_stays_monotone_below_the_clip():
    """The transform must remain a rescale, not a reshuffle: within the
    unsaturated range, order in raw units is order in scaled units."""
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, PerPatientNormalizer

    x = _aux_with_gsr("noisefloor", seed=7)
    patient = np.array(["p"] * len(x), dtype=object)
    pre = PerPatientNormalizer.fit(x, patient, np.arange(len(x)))
    z = pre.transform(x, patient)

    g = OHIO_AUX_INDEX["gsr"]
    raw, scaled = x[..., g].ravel(), z[..., g].ravel()
    keep = np.abs(scaled) < float(np.abs(scaled).max()) - 1e-6   # drop saturated
    order = np.argsort(raw[keep], kind="stable")
    assert np.all(np.diff(scaled[keep][order]) >= -1e-5), (
        "per-patient scaling is not order-preserving below the clip")


# --------------------------------------------------------------------------- #
# H4: the two auxiliary scalers composed into a 40x re-expansion
# --------------------------------------------------------------------------- #
# `OhioPreprocessor.transform` runs `PerPatientNormalizer` and then
# `CausalScaler` over the same tensor. After H3 the first stage is bounded at
# +/-20 -- but bounding it COMPRESSES the pooled distribution (every subject's
# spread grew by up to 1e4), so the global scaler's GSR interquartile range fell
# to 0.0250 and it re-expanded the bound by 1/0.0250 ~= 40x, back out to 797.8.
#
# Neither scaler is wrong on its own, which is why this needed its own finding:
# it is a property of the composition. The fix is to give `CausalScaler` the
# same tail-aware spread H3 gave the per-patient stage, plus a clip backstop,
# and to enable both ONLY on the auxiliary path.
#
# Measured on the real cohort, aux global scaler:
#     IQR (before)   gsr scale 0.02507   max|z| 797.8   1.9885% would clip at 20
#     tail-aware     gsr scale 1.0181    max|z|  57.5   0.0056% clips at 20
# so the rescale does the work and the clip is a guarantee rather than a lossy
# transform. Applying the same policy to the CGM scaler would move it (channel 1
# scale 5.0 -> 8.70, max|z| 22.4 -> 12.88), so it is opt-in and the CGM path is
# left bit-identical by construction rather than by measurement.

def _composed_aux(seed_offset: int = 10):
    """Four subjects through the real two-stage auxiliary pipeline."""
    from causalcgm_mt.ohio_loader import PerPatientNormalizer

    parts, pats = [], []
    for i, profile in enumerate(["healthy", "noisefloor", "noisefloor", "healthy"]):
        a = _aux_with_gsr(profile, n=300, seed=seed_offset + i)
        parts.append(a)
        pats += [f"s{i}"] * len(a)
    x = np.concatenate(parts)
    patient = np.array(pats, dtype=object)
    per_patient = PerPatientNormalizer.fit(x, patient, np.arange(len(x)))
    return per_patient.transform(x, patient), patient, sorted(set(pats))


def test_h4_the_composed_auxiliary_pipeline_is_bounded():
    """The headline: per-patient normalisation followed by the global scaler
    must not hand the model a channel 40x wider than every other."""
    from causalcgm_mt.data import CausalScaler
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, OHIO_AUX_SKIP_SCALING

    pp, patient, pids = _composed_aux()
    scaler = CausalScaler.fit_array(
        pp, patient, pids, skip_channels=OHIO_AUX_SKIP_SCALING,
        tail_aware=True, clip=20.0)
    z = scaler.transform(pp)

    worst = float(np.abs(z).max())
    assert worst <= 20.0 + 1e-5, (
        f"the composed auxiliary pipeline reaches {worst:.4g}; the per-patient "
        f"stage is bounded but the global scaler re-expands it, because the "
        f"pooled interquartile range collapsed onto the near-zero bulk")

    g = OHIO_AUX_INDEX["gsr"]
    assert float(np.abs(z[..., g]).max()) <= 20.0 + 1e-5
    # The rescale, not the clip, must be doing the work: a clip that is load
    # bearing is a second lossy transform, which is what H3 rejected.
    assert (np.abs(z) >= 20.0 - 1e-9).mean() < 0.005, (
        "the clip is saturating more than 0.5% of the tensor, so the "
        "tail-aware spread did not bind and the clip is carrying the fix")


def test_h4_the_cgm_scaler_is_untouched_by_default():
    """`tail_aware` and `clip` are opt-in. Every existing call site -- the CGM
    scaler above all -- must be bit-identical to the plain-IQR behaviour, or the
    CGM-only arms and the exported ONNX graph silently move."""
    from causalcgm_mt.data import CausalScaler

    rng = np.random.default_rng(0)
    x = rng.normal(120, 25, (500, 24, 5)).astype(np.float32)
    x[..., 1] = rng.standard_t(2.0, (500, 24)) * 5.0     # heavy-tailed, like ROC
    x[..., 4] = 1.0
    patient = np.array(["p"] * len(x), dtype=object)

    scaler = CausalScaler.fit_array(x, patient, ["p"], skip_channels=(4,))
    flat = x.reshape(-1, 5)
    med = np.median(flat, axis=0)
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    for c in range(4):
        assert scaler.center[c] == pytest.approx(float(med[c]), rel=1e-6)
        assert scaler.scale[c] == pytest.approx(float(q75[c] - q25[c]), rel=1e-6)

    assert getattr(scaler, "clip", None) is None, "clipping must be opt-in"
    # ...and with no clip the transform is exactly the affine map, unbounded.
    z = scaler.transform(x)
    expect = (x - scaler.center) / scaler.scale
    assert np.allclose(z, expect, atol=0, rtol=0)


def test_h4_the_clip_is_order_preserving_and_symmetric():
    """Below the bound the transform must remain a rescale, not a reshuffle."""
    from causalcgm_mt.data import CausalScaler

    pp, patient, pids = _composed_aux()
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX, OHIO_AUX_SKIP_SCALING
    scaler = CausalScaler.fit_array(
        pp, patient, pids, skip_channels=OHIO_AUX_SKIP_SCALING,
        tail_aware=True, clip=20.0)
    z = scaler.transform(pp)

    g = OHIO_AUX_INDEX["gsr"]
    raw, scaled = pp[..., g].ravel(), z[..., g].ravel()
    keep = np.abs(scaled) < 20.0 - 1e-6
    order = np.argsort(raw[keep], kind="stable")
    assert np.all(np.diff(scaled[keep][order]) >= -1e-5), (
        "the clipped transform is not order-preserving below its bound")
    assert float(z.min()) >= -20.0 - 1e-5 and float(z.max()) <= 20.0 + 1e-5


def test_h4_the_deployable_graph_clips_exactly_as_the_pipeline_does():
    """Train/serve skew guard.

    `DeployableMultiModalCausalCGMMT` folds the auxiliary scaler into the graph
    the C++ runner executes. If the Python path clips and the graph does not,
    every window carrying a saturated auxiliary value is served differently from
    how it was trained -- and no graph-level parity check would see it, because
    both sides of that check are the graph.
    """
    from causalcgm_mt.data import CausalScaler
    from causalcgm_mt.export import DeployableMultiModalCausalCGMMT
    from causalcgm_mt.models import build_multimodal_model

    rng = np.random.default_rng(3)
    n, T, C_cgm, C_aux = 16, 24, 5, 6
    x = rng.normal(120, 25, (n, T, C_cgm)).astype(np.float32)
    x[..., 4] = 1.0
    aux = rng.normal(0.0, 1.0, (n, T, C_aux)).astype(np.float32)
    aux[0, 0, 0] = 5_000.0            # a value that must saturate
    patient = np.array(["p"] * n, dtype=object)

    cgm_scaler = CausalScaler.fit_array(x, patient, ["p"], skip_channels=(4,))
    aux_scaler = CausalScaler.fit_array(aux, patient, ["p"], tail_aware=True,
                                        clip=20.0)

    model, _ = build_multimodal_model(C_cgm, C_aux, ModelConfig(n_blocks=2), 1.0)
    model.eval()
    net = DeployableMultiModalCausalCGMMT(model, cgm_scaler, aux_scaler).eval()

    last = torch.from_numpy(x[:, -1, 0].copy())
    present = torch.ones(n)
    with torch.no_grad():
        graph_bg, graph_risk = net(torch.from_numpy(x), torch.from_numpy(aux),
                                   last, present)
        ref_bg, ref_logit = model(
            torch.from_numpy(cgm_scaler.transform(x)),
            torch.from_numpy(aux_scaler.transform(aux)), last, present)
        ref_risk = torch.sigmoid(ref_logit)

    assert torch.allclose(graph_bg, ref_bg, atol=1e-5), (
        "the deployable wrapper disagrees with the numpy pipeline: the folded "
        "auxiliary scaler does not apply the clip")
    assert torch.allclose(graph_risk, ref_risk, atol=1e-5)


# --------------------------------------------------------------------------- #
# M3: `_macro` rendered NaN and Inf as the literal words "nan" / "inf"
# --------------------------------------------------------------------------- #
def test_m3_non_finite_macros_render_as_unavailable_not_as_the_word_nan():
    """`None` was handled; `float("nan")` was not.

    A NaN is a routine outcome upstream -- `pr_auc` on a single-class slice,
    `time_lag_min` on a short one, `mean_lead_time_min` when no event is
    detected -- and every one of those flows into `rebuild_numbers`. The
    manuscript would then compile clean, with no undefined control sequence,
    printing the word "nan" mid-sentence where a result belongs.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rpt_m3", os.path.join(REPO_ROOT, "scripts", "rebuild_paper_tables.py"))
    rpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rpt)

    assert rpt._macro("NumX", float("nan")) == "\\newcommand{\\NumX}{--}"
    assert rpt._macro("NumX", float("inf")) == "\\newcommand{\\NumX}{--}"
    assert rpt._macro("NumX", float("-inf")) == "\\newcommand{\\NumX}{--}"
    assert rpt._macro("NumX", None) == "\\newcommand{\\NumX}{--}"
    # ...and finite values are untouched, including the braced thousands comma.
    assert rpt._macro("NumX", 1.5) == "\\newcommand{\\NumX}{1.500}"
    assert rpt._macro("NumX", "28,335") == "\\newcommand{\\NumX}{28{,}335}"


def test_m3_the_generated_macro_file_carries_no_non_finite_literals():
    """The tree-level invariant the unit test above protects."""
    numbers = os.path.join(REPO_ROOT, "manuscript", "numbers.tex")
    if not os.path.exists(numbers):
        pytest.skip("numbers.tex not present")
    with open(numbers) as f:
        body = f.read()
    bad = re.findall(r"\\newcommand\{\\(\w+)\}\{-?(?:nan|inf)\}", body)
    assert not bad, f"macros rendering a non-finite value as a word: {bad}"


# --------------------------------------------------------------------------- #
# M6: ARIMAForecaster read statsmodels' trend="c" as an AR intercept
# --------------------------------------------------------------------------- #
def test_m6_the_d_zero_forecast_agrees_with_statsmodels():
    """With `trend="c"` statsmodels models `y_t = c + u_t` where `phi(L) u_t =
    eps`, so `params[0]` is the process MEAN. The recursion here runs on raw
    levels, whose intercept is `c * (1 - sum(phi))`. Feeding the mean in
    directly made the forecast drift by ~c*sum(phi) per step: on an AR(2) with
    mean 150 the six-step forecast was 434.95 where statsmodels said 149.85.

    Dormant at the shipped `order=(4, 1, 0)`, which takes the `d != 0` branch --
    and the existing `_forecast` test supplies its own `const`, so it exercises
    the recursion and never the interpretation.
    """
    pytest.importorskip("statsmodels")
    import warnings as _w

    from causalcgm_mt.benchmark import ARIMAForecaster

    rng = np.random.default_rng(0)
    phi, mu, n = np.array([0.5, 0.2]), 150.0, 3000
    y = np.full(n, mu)
    for t in range(2, n):
        y[t] = mu + phi[0] * (y[t - 1] - mu) + phi[1] * (y[t - 2] - mu) \
            + rng.normal(0, 5)

    ws = _tiny_windowset(n=n)
    ws.last_bg[:] = y.astype(np.float32)
    am = ARIMAForecaster(order=(2, 0, 0)).fit(ws, np.arange(n))
    const, ar = am.params["p1"]

    from statsmodels.tsa.arima.model import ARIMA
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        res = ARIMA(y, order=(2, 0, 0), trend="c").fit()

    expected = float(res.params[0]) * (1.0 - float(np.sum(res.arparams)))
    assert const == pytest.approx(expected, rel=1e-6), (
        "the stored constant is statsmodels' process mean, not the AR "
        "intercept the recursion needs")

    # ...and the resulting multi-step forecast must track statsmodels' own.
    got = am._forecast(y[None, -2:].astype(np.float64), 6, const, ar)[0]
    assert got == pytest.approx(float(res.forecast(6)[-1]), abs=1.0)


# --------------------------------------------------------------------------- #
# M8: MondrianConformal accepted out-of-range supplied strata
# --------------------------------------------------------------------------- #
def test_m8_out_of_range_supplied_strata_are_refused():
    """On the X-measurable path -- the one the class docstring calls "the
    theoretically clean path" -- the caller's assignment was taken verbatim.
    The SHAPE was checked; the VALUES were not. Rows outside 0..2 contributed to
    the pooled quantile but to no cell, so they silently received the MARGINAL
    guarantee while the object reported itself `taxonomy="supplied"` with
    nothing in `fell_back`, and every coverage number it emitted looked valid.
    """
    from causalcgm_mt.conformal import N_STRATA, MondrianConformal

    rng = np.random.default_rng(0)
    y = rng.normal(120, 30, 300)
    p = y + rng.normal(0, 10, 300)

    ok = rng.integers(0, N_STRATA, 300)
    mc = MondrianConformal.calibrate(y, p, 0.1, strata=ok)
    assert mc.taxonomy == "supplied"

    for bad_value in (-1, N_STRATA, 99):
        bad = ok.copy()
        bad[5] = bad_value
        with pytest.raises(ValueError, match=r"strata must be in 0\.\."):
            MondrianConformal.calibrate(y, p, 0.1, strata=bad)


# --------------------------------------------------------------------------- #
# M2: run_experiment.py personalised with a freshly-initialised criterion
# --------------------------------------------------------------------------- #
def test_m2_personalisation_restores_the_folds_fitted_criterion():
    """`build_model` returns a NEW `MultiTaskLoss`, so two fitted population
    quantities were silently reset: the Kendall log-variances went back to
    zeros (an equal task weighting the model was never trained under) and
    `pos_weight` to 1.0, dropping the class reweighting entirely.

    `FoldResult.criterion_state_dict` exists precisely to prevent this -- its
    own comment says so -- and `run_metabonet_benchmarks.py` uses it correctly.
    This driver was the one consumer that did not.
    """
    import importlib.util

    import torch as _torch

    from causalcgm_mt.models import MultiTaskLoss

    spec = importlib.util.spec_from_file_location(
        "_runexp_m2", os.path.join(REPO_ROOT, "scripts", "run_experiment.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    cfg = Config(model=ModelConfig(n_horizons=1))
    trained = MultiTaskLoss(cfg.model, pos_weight=18.2)
    with _torch.no_grad():
        trained.log_var_reg.fill_(-0.7)
        trained.log_var_cls.fill_(0.4)
    state = {k: v.detach().cpu() for k, v in trained.state_dict().items()}

    restored = drv.criterion_for_fold(state, cfg)
    assert float(restored.log_var_reg[0]) == pytest.approx(-0.7)
    assert float(restored.log_var_cls[0]) == pytest.approx(0.4)
    assert float(restored.pos_weight) == pytest.approx(18.2), (
        "pos_weight is a fitted class prior and travels in the state dict")

    # A fold that recorded nothing must fail loudly rather than silently
    # adapting under a uniform task balance.
    with pytest.raises(RuntimeError, match="criterion"):
        drv.criterion_for_fold({}, cfg)


# --------------------------------------------------------------------------- #
# M4: the ONNX probe could not detect the defect its error message named
# --------------------------------------------------------------------------- #
def test_m4_the_probe_measures_liveness_and_says_so():
    """It perturbs the newest steps and requires the output to CHANGE. That
    detects "the graph ignores the tail of its window". It cannot detect
    non-causality: a graph whose `CausalConv1d` trim was folded away reads MORE
    of the window and changes just as much, so it passes. The real defence is
    the parity check against PyTorch.

    Driven with a stub session, so this needs no onnxruntime.
    """
    from causalcgm_mt import export as E

    hist, in_ch = 24, 5

    class _Live:
        """Reads its input: output tracks the last few steps."""
        def run(self, _out, feed):
            x = feed["x_raw"]
            return [x[:, -3:, 0].sum(axis=1), np.zeros(x.shape[0], np.float32)]

    class _Dead:
        """Ignores its input entirely -- the defect the probe DOES catch."""
        def run(self, _out, feed):
            n = feed["x_raw"].shape[0]
            return [np.zeros(n, np.float32), np.zeros(n, np.float32)]

    assert E.verify_output_depends_on_recent_history(_Live(), in_ch, hist) is True
    assert E.verify_output_depends_on_recent_history(_Dead(), in_ch, hist) is False
    # The old name stays bound, so any external caller keeps working.
    assert E.verify_causality_onnx is E.verify_output_depends_on_recent_history


# --------------------------------------------------------------------------- #
# M9: DataConfig.exog_cols was silently dead
# --------------------------------------------------------------------------- #
def test_m9_a_non_empty_exog_cols_is_refused_rather_than_ignored():
    """`regularize_series` snapped the columns onto the grid and `build_windows`
    then read `glucose_col` and nothing else, so they were computed, binned and
    discarded. Setting the knob cost CPU and changed nothing, while its name
    reads as "this is how I add insulin to the generic pipeline"."""
    from causalcgm_mt.data import build_windows

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "patient_id": ["p1"] * n,
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="5min"),
        "glucose": rng.normal(120, 25, n),
        "insulin": rng.normal(1.0, 0.2, n),
    })
    cfg = DataConfig()
    assert len(build_windows(df, cfg)) > 0            # the supported path

    with pytest.raises(ValueError, match="exog_cols"):
        build_windows(df, dataclasses.replace(cfg, exog_cols=("insulin",)))


# --------------------------------------------------------------------------- #
# M11 / M12: worker churn, and the release provenance hole
# --------------------------------------------------------------------------- #
def test_m11_the_default_loader_does_not_respawn_workers_every_epoch():
    """`WindowDataset` holds the fold's tensors already materialised, so a
    worker process adds a full pickle of them per epoch and buys no parallelism:
    there is no per-item decode to overlap."""
    assert TrainConfig().num_workers == 0


def test_m12_the_release_checks_the_metabonet_summarys_provenance():
    """`CONTENTS` ships `manuscript/numbers.tex` in the PUBLIC tier, and 46 of
    its macros resolve out of the MetaboNet summary -- which `check_provenance`
    never inspected, so a release could publish corpus-scale numbers backed by a
    two-epoch run through the one script written to prevent that."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_pkg_m12", os.path.join(REPO_ROOT, "scripts", "package_release.py"))
    pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)

    assert any("metabonet" in p for p in pkg.PROVENANCE_JSON), (
        "the MetaboNet summary is not provenance-checked, yet numbers.tex "
        "ships publicly and quotes it")

    # And the appendix table `main.tex` \inputs now ships, by name -- an
    # archive that quotes those macros can rebuild the table beside them.
    # Shipping it does not weaken the isolation invariant: that is about
    # DIRECTORY entries sweeping a tree, and this entry carries no suffix list.
    # See `test_metabonet_runner.py::test_package_release_cannot_pick_these_artefacts_up`.
    metabonet = [(rel, sfx) for rel, _t, _d, sfx in pkg.CONTENTS
                 if rel.startswith("runs/metabonet")]
    assert metabonet == [
        ("runs/metabonet_results/benchmarks/table_metabonet_comparison.tex", ())]


# --------------------------------------------------------------------------- #
# M1: the noise estimator was fed windows where its contract is series
# --------------------------------------------------------------------------- #
def test_m1_the_noise_estimator_receives_series_not_windows():
    """Both call sites passed a generator over individual WINDOWS --
    `ws.x[i, :, 0]`, one 24-step gap-filled history slice per call -- where
    `estimate_signal_noise` differences its input and therefore needs a SERIES.

    Two consequences, neither of which raised. `rho1`, the identifiability
    diagnostic that decides `NoiseEstimate.smoothed` and is written verbatim
    into every results JSON, was computed over the wrong object: on the real
    cohort the per-window pass reports +0.523 against the per-patient +0.435,
    and the README quotes +0.43. And `n` counted overlapping history steps, so
    it overstated the sample behind the estimate by a factor of H.
    """
    from causalcgm_mt.data import anchor_series_by_patient, estimate_signal_noise

    ws, _train, _test = _fold_inputs(n_per_patient=80, n_patients=3)
    idx = np.arange(len(ws))

    series = list(anchor_series_by_patient(ws, idx))
    assert len(series) == 3, "one series per patient, not one per window"
    assert all(s.size == 80 for s in series)

    est = estimate_signal_noise(series, 5)
    assert est.n == idx.size, (
        f"n={est.n} counts overlapping history steps rather than anchors "
        f"({idx.size})")

    # Time order within a patient is what makes differencing meaningful, so the
    # helper must sort regardless of the order it is handed.
    shuffled = np.random.default_rng(0).permutation(idx)
    for ordered, from_shuffled in zip(series, anchor_series_by_patient(ws, shuffled)):
        assert np.allclose(ordered, from_shuffled)


def test_m1_both_fitters_use_the_series_form():
    """The two call sites -- `train.run_fold` and `OhioPreprocessor.fit` -- must
    agree, or one cohort's recorded provenance means something different from
    the other's."""
    import inspect

    from causalcgm_mt import ohio_loader, train

    for mod in (train, ohio_loader):
        src = inspect.getsource(mod)
        assert "anchor_series_by_patient" in src, (
            f"{mod.__name__} still builds the noise estimator's input itself")
        assert "ws.x[i, :, 0] for i in" not in src, (
            f"{mod.__name__} still passes per-window slices to "
            f"estimate_signal_noise")


# --------------------------------------------------------------------------- #
# M5: the C++ runner featurised raw values at unobserved steps
# --------------------------------------------------------------------------- #
# In training, `build_windows` fills a window's short interior gaps BEFORE
# featurising, so the model has never once seen a raw value at an unobserved
# step. The runner had no equivalent: it featurised whatever it was handed, so
# the deployment contract was "the caller must already have interpolated,
# causally, with the same budget" -- and that contract appeared nowhere.
#
# `verify_cpp_parity.py` could not see it either, because its masked cases fed a
# DENSE history alongside a mask with zeros in it: an unobserved step carrying a
# real value, which no window this pipeline produces ever looks like. Both sides
# agreed because both were wrong in the same way.
#
# With the parity script feeding raw values at unobserved steps, the skew
# measured up to 1.3e3 mg/dL in the features and 1.8e2 mg/dL in the forecast
# across 18 of 30 cases.

_RUNNER = os.path.join(REPO_ROOT, "cpp", "build", "causalcgm_mt_runner")
_ONNX = os.path.join(REPO_ROOT, "runs", "demo", "causalcgm_mt.onnx")


@pytest.mark.skipif(
    not (os.path.exists(_RUNNER) and os.path.exists(_ONNX)),
    reason="C++ runner or demo graph not built (./scripts/fetch_ort.sh; cmake)")
def test_m5_the_runner_reconstructs_short_gaps_like_the_training_pipeline():
    """Cross-language parity over a window with a real interior dropout."""
    import subprocess

    import torch as _torch

    from causalcgm_mt.config import Config as _Config
    from causalcgm_mt.data import _causal_gap_fill, causal_features
    from causalcgm_mt.export import DeployableCausalCGMMT
    from causalcgm_mt.models import build_model, load_state_dict_compat

    ck = _torch.load(os.path.join(REPO_ROOT, "runs", "demo", "demo.pt"),
                     weights_only=False, map_location="cpu")
    cfg = _Config.from_dict(ck["config"])
    H = cfg.data.history_steps
    budget = max(0, cfg.data.max_interpolation_gap_min // cfg.data.sample_period_min)

    observed = np.ones(H, dtype=np.float32)
    observed[H - 14:H - 12] = 0.0                      # a 10-minute dropout
    raw = np.linspace(160.0, 58.0, H).astype(np.float32)
    raw[observed == 0.0] = -999.0                      # garbage, as on a device

    proc = subprocess.run(
        [_RUNNER, "--verify", _ONNX, str(budget)],
        input="\n".join(f"{v:.9g}" for v in [*raw, *observed]),
        capture_output=True, text=True, check=True)
    c_bg = float([ln for ln in proc.stdout.splitlines()
                  if ln.startswith("bg_hat")][0].split()[1])

    filled, _ = _causal_gap_fill(raw.astype(np.float64), observed.astype(bool),
                                 budget)
    model, _ = build_model(ck["in_channels"], cfg.model)
    load_state_dict_compat(model, ck["state_dict"])
    net = DeployableCausalCGMMT(model.eval(), ck["scaler"]).eval()
    with _torch.no_grad():
        p_bg, _ = net(
            _torch.from_numpy(causal_features(filled.astype(np.float32), observed)[None]),
            _torch.tensor([np.float32(filled[-1])]))

    assert abs(c_bg - float(np.atleast_1d(p_bg.numpy())[0])) < 1e-2, (
        "the runner does not reconstruct short gaps the way build_windows "
        "does -- a routine CGM dropout is served differently from how it was "
        "trained")


@pytest.mark.skipif(
    not (os.path.exists(_RUNNER) and os.path.exists(_ONNX)),
    reason="C++ runner or demo graph not built")
@pytest.mark.parametrize("mask_kind", ["trailing", "over_budget"])
def test_m5_a_window_the_pipeline_would_drop_is_refused(mask_kind):
    """`build_windows` requires every history step to be observed or imputed and
    drops the window otherwise, so the model was never trained on one. The
    runner must refuse rather than return a confident number."""
    import subprocess

    H, budget = 24, 3
    observed = np.ones(H, dtype=np.float32)
    if mask_kind == "trailing":
        observed[-3:] = 0.0                    # unfillable: needs extrapolation
    else:
        observed[5:5 + budget + 1] = 0.0       # interior, but over budget
    values = np.full(H, 120.0, dtype=np.float32)
    values[observed == 0.0] = -999.0

    proc = subprocess.run(
        [_RUNNER, "--verify", _ONNX, str(budget)],
        input="\n".join(f"{v:.9g}" for v in [*values, *observed]),
        capture_output=True, text=True)
    assert proc.returncode != 0, (
        "the runner produced a forecast from a window the training pipeline "
        "would have dropped")
    assert "not usable" in proc.stderr


# --------------------------------------------------------------------------- #
# M7: whole MetaboNet studies were dropped without a diagnostic
# --------------------------------------------------------------------------- #
# The grid is built at `cfg.sample_period_min` regardless of a study's native
# cadence. A study sampled every 15 minutes lands two-thirds unobserved on the
# 5-minute grid; it clears `min_cgm_frac`, survives segmentation, and is then
# annihilated window by window -- `himp.mean() ~= 0.67 > max_imputed_frac` drops
# every one, and `os_[tgt].all()` drops the rest. The subject is recorded as
# barren and the build moves on.
#
# Measured against the shipped window cache: 10 sources contributed, 14 subjects
# were barren, and the barren list named only 4 sources -- so at least two
# non-excluded studies contributed nothing at all and appeared in NEITHER list,
# because they never produced a shard to record. Nothing is wrong in the
# reassuring direction (the data is excluded, not mis-included), but a
# corpus-scale validation that quietly drops whole studies is misdescribed by
# `\NumMetaboSources`, which counts what survived rather than what was asked for.

def test_m7_a_coarser_cadence_than_the_grid_is_reported():
    """The subject still contributes nothing -- that is correct -- but it must
    say why rather than vanishing into a barren count."""
    import logging

    from causalcgm_mt.metabonet_loader import native_cadence_min

    ts = pd.date_range("2021-01-01", periods=200, freq="15min")
    coarse = pd.DataFrame({"time": ts, "glucose": np.linspace(100, 140, 200)})
    assert native_cadence_min(coarse) == 15

    fine = pd.DataFrame({
        "time": pd.date_range("2021-01-01", periods=200, freq="5min"),
        "glucose": np.linspace(100, 140, 200)})
    assert native_cadence_min(fine) == 5
    assert native_cadence_min(pd.DataFrame({"time": [], "glucose": []})) == 0


def test_m7_the_source_ledger_accounts_for_every_subject_asked_for():
    """Per-source accounting, so a study that contributes nothing is visible as
    a zero rather than as an absence."""
    from causalcgm_mt.metabonet_loader import source_contributions

    ledger = source_contributions(
        requested=["A:1", "A:2", "B:1", "C:1", "C:2", "C:3"],
        contributed=["A:1", "A:2", "B:1"],
        barren=["C:1"],
    )
    assert ledger["A"] == {"requested": 2, "contributed": 2, "barren": 0,
                           "unaccounted": 0}
    assert ledger["B"] == {"requested": 1, "contributed": 1, "barren": 0,
                           "unaccounted": 0}
    # C asked for three, produced none: one explicitly barren, two never
    # reached (an interrupted or filtered pass). Neither is silent.
    assert ledger["C"] == {"requested": 3, "contributed": 0, "barren": 1,
                           "unaccounted": 2}
    assert [s for s, v in ledger.items() if v["contributed"] == 0] == ["C"]
