"""Audit of the Phase-2 experiment harness: `benchmark.py`, `leakage_control.py`
and the three paper drivers.

The harness is what produces every number in the paper, so the failure modes it
has to be protected against are not "does it crash" but:

  * does an arm's spec actually change what gets trained (a flag that silently
    does nothing turns an ablation into three copies of the same run)?
  * does every arm get scored on the same rows, through the same code?
  * is the leakage CONTROL arm actually leaky -- if it is not, Ablation D
    measures nothing and reports a reassuring zero?
  * do the tables serialise the numbers that were computed, rather than a
    plausible-looking transposition of them?

Everything here runs on a small synthetic OhioT1DM-shaped cohort, so it is fast
and needs no licensed data.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt import benchmark as B
from causalcgm_mt.config import DataConfig
from causalcgm_mt.hupa_loader import HupaConfig
from causalcgm_mt.leakage_control import (
    LEAK_LADDER, build_naive_windows, centered_features,
    interpolate_whole_series, naive_random_split,
)
from causalcgm_mt.ohio_loader import (
    OhioConfig, OhioPreprocessor, build_ohio_windows, discover_records,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = DataConfig(history_min=60, horizon_min=30, stride_steps=1, band_mode="fixed")
HORIZONS = (30, 60)


# --------------------------------------------------------------------------- #
# A small two-cohort fixture in the real XML dialect
# --------------------------------------------------------------------------- #
def _write_xml(path, pid, start, hours, seed, cohort):
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
             f'    <event ts="{fmt(ts[0])}" value="0.8"/>', "  </basal>"]
    if cohort == "2018":
        rows.append("  <basis_heart_rate>")
        rows += [f'    <event ts="{fmt(t)}" value="{70 + (i % 9)}"/>'
                 for i, t in enumerate(ts) if i % 2 == 0]
        rows += ["  </basis_heart_rate>", "  <basis_steps>"]
        rows += [f'    <event ts="{fmt(t)}" value="{i % 30}"/>' for i, t in enumerate(ts)]
        rows.append("  </basis_steps>")
        rows.append("  <basis_gsr>")
        rows += [f'    <event ts="{fmt(t)}" value="{1e-4 * (1 + i % 5):.6f}"/>'
                 for i, t in enumerate(ts)]
        rows.append("  </basis_gsr>")
    else:
        rows.append("  <acceleration>")
        rows += [f'    <event ts="{fmt(t)}" value="{20 + (i % 40)}"/>'
                 for i, t in enumerate(ts)]
        rows += ["  </acceleration>", "  <basis_gsr>"]
        rows += [f'    <event ts="{fmt(t)}" value="{1e-2 * (1 + i % 5):.6f}"/>'
                 for i, t in enumerate(ts)]
        rows.append("  </basis_gsr>")
    rows.append("</patient>")
    with open(path, "w") as f:
        f.write("\n".join(rows))


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("ohio_bench")
    for cohort, pids in (("2018", ["501", "502"]), ("2020", ["540", "544"])):
        for split, tag, off, hours in (("train", "training", 0, 96.0),
                                       ("test", "testing", 97, 36.0)):
            d = root / cohort / split
            d.mkdir(parents=True, exist_ok=True)
            for k, pid in enumerate(pids):
                _write_xml(str(d / f"{pid}-ws-{tag}.xml"), pid,
                           pd.Timestamp("2021-12-07 00:00:00") + pd.Timedelta(hours=off),
                           hours, seed=k + 1, cohort=cohort)
    return str(root)


@pytest.fixture(scope="module")
def data(fake_root) -> B.CohortData:
    return B.load_cohort(fake_root, history_min=CFG.history_min,
                         horizons=HORIZONS, val_frac=0.2, cache=None)


# --------------------------------------------------------------------------- #
# 1. ArmSpec is the experiment record, so it has to be strict
# --------------------------------------------------------------------------- #
def test_unknown_family_is_rejected_at_construction():
    """A typo must fail immediately, not two hours into a sweep when the
    dispatch falls through."""
    with pytest.raises(ValueError, match="unknown family"):
        B.ArmSpec("typo", "transformerr")


@pytest.mark.parametrize("bad", [{"fit_subsample": 0}, {"horizons": ()}])
def test_incoherent_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        B.ArmSpec("x", "tcn", **bad)


def test_spec_round_trips_through_json():
    """The spec is written into the results file verbatim, so a number in the
    paper can be traced to the hyperparameters that produced it. That only works
    if it survives serialisation."""
    spec = B.ArmSpec("A", "tcn_mm", horizons=(30, 60), modalities=("gsr",),
                     cohorts=("2018",), hypo_asymmetry=5.0)
    d = json.loads(json.dumps(spec.as_dict()))
    assert d["horizons"] == [30, 60]
    assert d["modalities"] == ["gsr"] and d["cohorts"] == ["2018"]
    assert d["hypo_asymmetry"] == 5.0
    assert B.ArmSpec(**{**d, "horizons": tuple(d["horizons"]),
                        "modalities": tuple(d["modalities"]),
                        "cohorts": tuple(d["cohorts"])}) == spec


def test_every_family_is_reachable_from_the_dispatch():
    """`FAMILIES` is what the drivers validate `--arms` against; a family listed
    there but unhandled in `run_arm` would fail only at runtime."""
    import inspect
    src = inspect.getsource(B.run_arm)
    for fam in B.FAMILIES:
        assert f'"{fam}"' in src or fam in ("lstm", "tcn", "tcn_mm"), \
            f"{fam} is advertised but never dispatched"
    # lstm/tcn/tcn_mm share the `is_neural` branch.
    assert 'spec.is_neural' in src


# --------------------------------------------------------------------------- #
# 2. Predictions: shape discipline
# --------------------------------------------------------------------------- #
def test_one_horizon_is_a_column_not_a_row():
    """`np.atleast_2d` on an (N,) array yields (1, N) -- the transpose. Every
    metric downstream would then be computed over a single row without raising,
    which is the kind of bug that produces a plausible table of nonsense."""
    p = B.Predictions(np.arange(7.0), np.zeros(7))
    assert p.bg.shape == (7, 1) and p.n_horizons == 1
    bg, risk = p.col(0)
    assert bg.shape == (7,) and risk.shape == (7,)


def test_mismatched_prediction_shapes_raise():
    with pytest.raises(ValueError, match="bg .* vs risk"):
        B.Predictions(np.zeros((5, 2)), np.zeros((5, 3)))


def test_risk_from_forecast_is_monotone_in_the_forecast():
    """The score is a strictly decreasing map of the forecast, so ROC-AUC and
    PR-AUC for a regression-only baseline are exactly those of its forecast --
    no information is invented by the wrapper."""
    bg = np.array([40.0, 60.0, 70.0, 120.0, 300.0])
    r = B.risk_from_forecast(bg)
    assert np.all(np.diff(r) < 0)
    assert 0.0 < r.min() and r.max() < 1.0
    assert B.risk_from_forecast(np.array([70.0]))[0] == pytest.approx(0.5)


def test_risk_from_forecast_survives_extreme_inputs():
    r = B.risk_from_forecast(np.array([-1e6, 1e6]))
    assert np.isfinite(r).all() and r[0] > 0.99 and r[1] < 0.01


# --------------------------------------------------------------------------- #
# 3. Classical baselines
# --------------------------------------------------------------------------- #
def test_persistence_is_exactly_the_last_observed_value(data):
    idx = data.split.test[:50]
    p = B.persistence_predictions(data.ws, idx, HORIZONS)
    for k in range(len(HORIZONS)):
        np.testing.assert_allclose(p.bg[:, k], data.ws.last_bg[idx], rtol=0, atol=0)


def test_ridge_ar_reads_only_the_causal_history(data):
    """The design matrix is channel 0 of the already-causal feature tensor, so
    corrupting every target must leave it bit-identical."""
    X = B._lag_matrix(data.ws, data.split.fit[:200])
    poisoned = data.ws.subset(np.arange(len(data.ws)))
    poisoned.y_reg_multi = poisoned.y_reg_multi + 500.0
    poisoned.y_reg = poisoned.y_reg + 500.0
    np.testing.assert_array_equal(X, B._lag_matrix(poisoned, data.split.fit[:200]))


def test_ridge_ar_beats_persistence(data):
    ar = B.RidgeAR().fit(data.ws, data.split.fit, HORIZONS, [0, 1])
    pred = ar.predict(data.ws, data.split.test)
    per = B.persistence_predictions(data.ws, data.split.test, HORIZONS)
    y = data.ws.y_reg_multi[data.split.test, 0]
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))  # noqa: E731
    assert rmse(pred.bg[:, 0]) < rmse(per.bg[:, 0]), \
        "a linear AR that cannot beat persistence is misconfigured"


# --------------------------------------------------------------------------- #
# Nadir target (Ablation F): the control for the risk-mapping handicap
# --------------------------------------------------------------------------- #
def test_nadir_target_fits_the_window_minimum_not_the_endpoint(data):
    """The whole study rests on the two arms differing in their TARGET alone.
    If `target='nadir'` silently fell back to the level target the table would
    show two identical rows and read as 'the mapping costs nothing'."""
    fit = data.split.fit
    level = B.RidgeAR(target="level").fit(data.ws, fit, HORIZONS, [0, 1])
    nadir = B.RidgeAR(target="nadir").fit(data.ws, fit, HORIZONS, [0, 1])
    assert not np.allclose(level.models[0].coef_, nadir.models[0].coef_), \
        "nadir and level fits produced the same coefficients"

    # And the nadir arm really is fitting the minimum: it must predict at or
    # below the level arm on average, because min g over (t, t+PH] <= g(t+PH).
    idx = data.split.test
    assert nadir.predict(data.ws, idx).bg[:, 0].mean() < \
        level.predict(data.ws, idx).bg[:, 0].mean()


def test_nadir_target_lowers_error_against_the_window_minimum(data):
    """Each target should win on its own quantity. Anything else means the
    target flag is wired to the wrong array."""
    fit, idx = data.split.fit, data.split.test
    rmse = lambda p, y: float(np.sqrt(np.mean((y - p) ** 2)))  # noqa: E731
    y_min = data.ws.y_min_multi[idx, 0]
    y_lvl = data.ws.y_reg_multi[idx, 0]
    lv = B.RidgeAR(target="level").fit(data.ws, fit, HORIZONS, [0, 1]).predict(data.ws, idx)
    nd = B.RidgeAR(target="nadir").fit(data.ws, fit, HORIZONS, [0, 1]).predict(data.ws, idx)
    assert rmse(nd.bg[:, 0], y_min) < rmse(lv.bg[:, 0], y_min)
    assert rmse(lv.bg[:, 0], y_lvl) < rmse(nd.bg[:, 0], y_lvl)


def test_nadir_target_is_rejected_for_arms_with_their_own_classifier():
    """LightGBM and the neural arms fit a classifier on the nadir label
    directly, so a nadir REGRESSION target there would change the regression
    head's meaning while leaving the risk score untouched -- a different
    experiment wearing this one's name."""
    for family in ("lgbm", "tcn_mm", "lstm"):
        with pytest.raises(ValueError, match="target='nadir'"):
            B.ArmSpec("x", family, target="nadir")


def test_nadir_target_round_trips_through_the_serialised_spec():
    """`target` has to reach the results JSON, or a reader cannot tell the two
    Ablation F rows apart from the artefact alone."""
    spec = B.ArmSpec("Ridge-AR, nadir target", "ridge_ar", target="nadir")
    assert spec.as_dict()["target"] == "nadir"
    assert B.ArmSpec("x", "ridge_ar").as_dict()["target"] == "level"


def test_arima_forecast_matches_an_explicit_recursion():
    """The vectorised h-step forecast must equal the AR recursion done by hand.
    A transposed lag order would still produce plausible numbers, so this checks
    the recursion itself rather than the output's shape."""
    am = B.ARIMAForecaster(order=(2, 0, 0))
    ar = np.array([0.6, -0.2])
    const = 5.0
    hist = np.array([[100.0, 110.0, 120.0, 130.0]])
    got = am._forecast(hist, 3, const, ar)

    w = [120.0, 130.0]                      # last p values, oldest first
    for _ in range(3):
        nxt = const + ar[0] * w[-1] + ar[1] * w[-2]
        w = [w[-1], nxt]
    assert got[0] == pytest.approx(w[-1])


def test_arima_differenced_forecast_returns_to_a_level():
    """With d=1 the recursion runs on differences and is cumulated back, so a
    perfectly linear history must extrapolate the same slope."""
    am = B.ARIMAForecaster(order=(1, 1, 0))
    hist = np.array([[100.0, 105.0, 110.0, 115.0]])
    got = am._forecast(hist, 2, 0.0, np.array([1.0]))   # differences persist
    assert got[0] == pytest.approx(125.0)


# --------------------------------------------------------------------------- #
# 4. Split and subsample discipline
# --------------------------------------------------------------------------- #
def test_fit_subsample_touches_only_the_rows_it_is_given(data):
    idx = data.split.fit
    sub = B.subsample_rows(data.ws, idx, 3)
    assert sub.size < idx.size
    assert set(sub.tolist()) <= set(idx.tolist()), "subsampling invented rows"
    assert np.array_equal(sub, np.sort(sub))
    assert np.array_equal(B.subsample_rows(data.ws, idx, 1), idx)


def test_subsample_keeps_every_patient(data):
    """Dropping a whole subject would change what the arm is trained on in a way
    the budget knob is not supposed to."""
    sub = B.subsample_rows(data.ws, data.split.fit, 4)
    assert ({str(p) for p in data.ws.patient[sub]}
            == {str(p) for p in data.ws.patient[data.split.fit]})


def test_subsampling_stringifies_the_patient_column_once(data, monkeypatch):
    """The inner `np.asarray([str(p) for p in ws.patient[idx]])` was rebuilt on
    every loop iteration -- a Python-level `str()` over every fit row, once per
    patient. ~1.2M calls per arm on the Ohio cohort, quadratic in subject count
    on a MetaboNet-scale index.

    Counted rather than timed: a wall-clock threshold on a shared machine is a
    flaky test, and the defect is exactly "how many times", not "how slow".
    """
    idx = data.split.fit
    n_patients = len({str(p) for p in data.ws.patient[idx]})
    assert n_patients >= 3, "too few patients for the quadratic term to show"

    calls = {"n": 0}
    real_str = builtins.str

    class CountingStr(str):
        pass

    def counting_str(x=""):
        calls["n"] += 1
        return real_str(x)

    monkeypatch.setattr(builtins, "str", counting_str)
    try:
        sub = B.subsample_rows(data.ws, idx, 3)
    finally:
        monkeypatch.undo()

    # One pass over the rows, plus a handful for the patient keys themselves.
    assert calls["n"] <= idx.size + 4 * n_patients + 32, (
        f"{calls['n']} str() calls over {idx.size} rows and {n_patients} "
        f"patients: the patient column is being rebuilt inside the loop")
    assert sub.size > 0


def test_subsampling_result_is_unchanged_by_the_hoist(data):
    """The optimisation must be exactly that. Compared against the original
    formulation, row for row."""
    for k in (2, 3, 5):
        idx = data.split.fit
        got = B.subsample_rows(data.ws, idx, k)

        out = []
        for pid in sorted({str(p) for p in data.ws.patient[idx]}):
            rows = idx[np.asarray([str(p) for p in data.ws.patient[idx]]) == pid]
            rows = rows[np.argsort(data.ws.t_time[rows], kind="stable")]
            out.append(rows[::k])
        want = np.sort(np.concatenate(out)).astype(np.int64)

        assert np.array_equal(got, want), f"k={k}"
        assert got.dtype == want.dtype


def test_cohort_restriction_keeps_indices_valid_and_disjoint(data):
    """Cutting the SPLIT rather than the window set is what lets restricted and
    unrestricted arms share one scoring path."""
    spec = B.ArmSpec("2018 only", "tcn_mm", cohorts=("2018",), horizons=HORIZONS)
    s = B.restrict_split(data, spec)
    for name in ("fit", "val", "test"):
        idx = getattr(s, name)
        assert idx.size > 0
        assert (np.asarray([str(t) for t in data.ws.subset_tag[idx]]) == "2018").all()
        assert set(idx.tolist()) <= set(getattr(data.split, name).tolist())
    assert np.intersect1d(s.fit, s.test).size == 0


def test_unrestricted_arm_gets_the_official_split_object(data):
    assert B.restrict_split(data, B.ArmSpec("all", "tcn_mm")) is data.split


# --------------------------------------------------------------------------- #
# 5. Scoring: one path, and no test label on the calibration path
# --------------------------------------------------------------------------- #
def test_score_arm_reports_every_metric_family(data):
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    res, trained = B.run_arm(spec, data, verbose=False)
    assert trained is None
    for h in HORIZONS:
        block = res["per_horizon"][str(h)]
        assert set(block) >= {"regression", "classification", "alarm",
                              "clinical", "conformal"}
        assert np.isfinite(block["regression"]["rmse"])
        assert 0.0 <= block["clinical"]["zone_ab_pct"] <= 100.0
        assert set(block["clinical"]["zone_pct"]) == set("ABCDE")
        assert block["conformal"]["alpha"] == 0.1
        assert block["n"] == data.split.test.size


def test_conformal_is_calibrated_on_calibration_not_on_test(data):
    """The decisive leakage test for the harness: corrupt every TEST target and
    require qhat to be unchanged. It is a quantile of CALIBRATION residuals, and
    there is no API by which a test label could reach it."""
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    calib = B.persistence_predictions(data.ws, data.split.calib, HORIZONS)
    test = B.persistence_predictions(data.ws, data.split.test, HORIZONS)
    before = B.score_arm(spec, data, data.split, calib, test)

    poisoned = data.ws.subset(np.arange(len(data.ws)))
    poisoned.y_reg_multi[data.split.test] += 400.0
    after = B.score_arm(spec, B.CohortData(poisoned, data.split, data.pre,
                                           data.cfg, data.horizons),
                        data.split, calib, test)
    for h in HORIZONS:
        a = before["per_horizon"][str(h)]["conformal"]
        b = after["per_horizon"][str(h)]["conformal"]
        assert a["split"]["qhat"] == b["split"]["qhat"]
        assert a["mondrian"]["qhat_by_stratum"] == b["mondrian"]["qhat_by_stratum"]
        # ...and the COVERAGE must move, or the poisoning did not reach the
        # metric and this test proves nothing.
        assert a["split"]["marginal_coverage"] != b["split"]["marginal_coverage"]


def test_calibration_mask_shrinks_the_calibration_set_not_the_test_set(data):
    """The mask exists for the personalised arm, whose adaptation prefix may
    overlap calibration rows. Excluding them must not touch what is scored."""
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    calib = B.persistence_predictions(data.ws, data.split.calib, HORIZONS)
    test = B.persistence_predictions(data.ws, data.split.test, HORIZONS)
    mask = np.zeros(data.split.calib.size, dtype=bool)
    mask[::2] = True

    full = B.score_arm(spec, data, data.split, calib, test)
    part = B.score_arm(spec, data, data.split, calib, test, cal_mask=mask)
    c_full = full["per_horizon"]["30"]["conformal"]
    c_part = part["per_horizon"]["30"]["conformal"]
    assert c_part["n_calib_rows_used"] == int(mask.sum()) < c_full["n_calib_rows_used"]
    assert part["per_horizon"]["30"]["n"] == full["per_horizon"]["30"]["n"]


def test_scoring_uses_the_right_horizon_column(data):
    """Column k of the prediction must be scored against horizon k's target. A
    swap here would be invisible -- both columns are plausible glucose values."""
    spec = B.ArmSpec("oracle", "persistence", horizons=HORIZONS)
    idx = data.split.test
    oracle = B.Predictions(np.stack([data.ws.y_reg_multi[idx, 0],
                                     data.ws.y_reg_multi[idx, 1]], axis=1),
                           np.zeros((idx.size, 2)))
    reps = B.score_horizons(data, idx, oracle, spec, [0, 1])
    assert reps[30].rmse == pytest.approx(0.0, abs=1e-4)
    assert reps[60].rmse == pytest.approx(0.0, abs=1e-4)

    swapped = B.Predictions(oracle.bg[:, ::-1], np.zeros((idx.size, 2)))
    bad = B.score_horizons(data, idx, swapped, spec, [0, 1])
    assert bad[30].rmse > 1.0 and bad[60].rmse > 1.0


# --------------------------------------------------------------------------- #
# 6. Neural arms: the flags must actually do something
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny(data):
    """A 1-epoch spec: enough to check that a flag changes the model, not enough
    to mean anything about performance."""
    return dict(horizons=HORIZONS, epochs=1, patience=1, fit_subsample=6,
                batch_size=64)


def test_unimodal_arm_builds_a_unimodal_model(data, tiny):
    t = B.train_neural_arm(data, B.ArmSpec("cgm", "tcn", unimodal=True, **tiny),
                           verbose=False)
    assert not t.multimodal
    assert hasattr(t.model, "backbone") and not hasattr(t.model, "cross_attn")


def test_fusion_arm_builds_the_cross_attention_model(data, tiny):
    t = B.train_neural_arm(data, B.ArmSpec("mm", "tcn_mm", **tiny), verbose=False)
    assert t.multimodal
    assert hasattr(t.model, "cross_attn") and hasattr(t.model, "aux_backbone")


def test_backbone_flag_selects_the_lstm(data, tiny):
    from causalcgm_mt.models import DilatedLSTMBackbone, TCNBackbone
    t = B.train_neural_arm(data, B.ArmSpec("lstm", "lstm", unimodal=True,
                                           backbone="dilated_lstm", **tiny),
                           verbose=False)
    assert isinstance(t.model.backbone, DilatedLSTMBackbone)
    t2 = B.train_neural_arm(data, B.ArmSpec("tcn", "tcn", unimodal=True, **tiny),
                            verbose=False)
    assert isinstance(t2.model.backbone, TCNBackbone)


def test_single_task_lambda_actually_silences_the_other_head(data, tiny):
    """Ablation A's single-task arms are built by zeroing a loss weight. If the
    gradient still reached the silenced head, the ablation would be comparing
    four copies of the same joint model."""
    import torch

    spec = B.ArmSpec("reg only", "tcn_mm", loss_weighting="fixed",
                     lambda_reg=1.0, lambda_cls=0.0, monitor="val/rmse",
                     monitor_mode="min", **tiny)
    cfg = B._arm_config(spec, data)
    from causalcgm_mt.models import build_multimodal_model
    model, crit = build_multimodal_model(data.ws.x.shape[-1],
                                         data.ws.x_aux.shape[-1], cfg.model, 5.0)
    x = torch.randn(4, CFG.history_steps, data.ws.x.shape[-1])
    xa = torch.randn(4, CFG.history_steps, data.ws.x_aux.shape[-1])
    bg, logit = model(x, xa, torch.full((4,), 120.0), torch.ones(4))
    loss, _ = crit(bg, logit, torch.full((4, 2), 110.0), torch.zeros(4, 2))
    loss.backward()

    cls_grad = max(float(p.grad.abs().max()) for p in model.cls_heads.parameters()
                   if p.grad is not None)
    reg_grad = max(float(p.grad.abs().max()) for p in model.reg_heads.parameters()
                   if p.grad is not None)
    assert cls_grad == 0.0, "lambda_cls=0 must silence the classification head"
    assert reg_grad > 0.0, "the regression head must still be trained"


def test_hypo_asymmetry_weights_only_over_predicted_lows():
    """The kappa knob's contract, checked on the weight function directly.

    kappa = 1 must be an EXACT no-op (all-ones), and above 1 the multiplier must
    apply only where the reference is low AND the prediction sits above it --
    the direction that produces Clarke zone D.
    """
    import torch

    from causalcgm_mt.models import hypo_cost_weights

    y = torch.tensor([55.0, 55.0, 150.0, 150.0])
    p = torch.tensor([95.0, 40.0, 190.0, 110.0])    # over, under, over, under
    off = hypo_cost_weights(p, y, kappa=1.0, threshold=70.0, tau=10.0)
    assert torch.equal(off, torch.ones_like(off)), "kappa=1 must be exactly off"

    on = hypo_cost_weights(p, y, kappa=5.0, threshold=70.0, tau=10.0)
    assert on[0] > 4.0, "an over-predicted low must be up-weighted"
    assert on[1] == pytest.approx(1.0), "an under-predicted low is the safe direction"
    assert on[2] == pytest.approx(1.0, abs=0.01), "the ramp must be ~off at 150"
    assert on[3] == pytest.approx(1.0)


def test_hypo_asymmetry_flag_changes_the_loss(data, tiny):
    """And the flag has to reach the loss.

    The batch is deliberately HETEROGENEOUS. `asymmetric_huber` normalises by
    the weight sum -- so that raising kappa does not rescale the loss and
    silently disturb the learned task weights -- which means a batch where every
    sample carries the same weight is kappa-invariant by construction. The
    effect appears only when some samples are up-weighted relative to others,
    which is also the only situation in which the mechanism does anything.
    """
    import torch

    from causalcgm_mt.models import build_multimodal_model

    args = (data.ws.x.shape[-1], data.ws.x_aux.shape[-1])
    #     over-predicted low  |  correctly predicted euglycemic
    bg = torch.tensor([[95.0, 95.0], [150.0, 150.0]])
    y = torch.tensor([[55.0, 55.0], [150.0, 150.0]])
    losses = []
    for kappa in (1.0, 5.0):
        spec = B.ArmSpec(f"k{kappa}", "tcn_mm", hypo_asymmetry=kappa, **tiny)
        torch.manual_seed(0)
        _model, crit = build_multimodal_model(*args, B._arm_config(spec, data).model, 5.0)
        loss, _ = crit(bg, torch.zeros(2, 2), y, torch.ones(2, 2))
        losses.append(float(loss.detach()))
    assert losses[1] > losses[0], "kappa must up-weight over-prediction of a low"


# --------------------------------------------------------------------------- #
# 7. The leakage control arm must actually leak
# --------------------------------------------------------------------------- #
def test_full_series_interpolation_uses_the_future():
    """The whole point of the control arm. A gap filled from both sides carries
    information from after the anchor; the strict per-window fill cannot."""
    v = np.array([100.0, 0.0, 0.0, 160.0, 170.0])
    obs = np.array([True, False, False, True, True])
    filled, imputed = interpolate_whole_series(v, obs)
    assert filled[1] == pytest.approx(120.0) and filled[2] == pytest.approx(140.0)
    assert imputed.tolist() == [False, True, True, False, False]

    # Change only the FUTURE endpoint; the filled past must move, which is the
    # leak stated as an executable fact.
    v2 = v.copy(); v2[3] = 400.0
    assert interpolate_whole_series(v2, obs)[0][1] != pytest.approx(120.0)


def test_leading_gap_is_back_filled_from_the_future():
    v = np.array([0.0, 0.0, 130.0])
    obs = np.array([False, False, True])
    assert interpolate_whole_series(v, obs)[0].tolist() == [130.0, 130.0, 130.0]


def test_centered_features_read_forward_and_causal_ones_do_not():
    """Side-by-side with the strict extractor, on the same history."""
    from causalcgm_mt.data import causal_features

    h = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 300.0])
    obs = np.ones(6)
    cen = centered_features(h, obs)
    cau = causal_features(h, obs)
    assert cen.shape == cau.shape

    # Perturb only the LAST step; a causal feature at position 2 cannot move.
    h2 = h.copy(); h2[-1] = 40.0
    assert np.allclose(cau[2], causal_features(h2, obs)[2])
    # The centred rolling mean at position 2 uses a window reaching to 4, and
    # position 3 reaches the perturbed step.
    assert not np.allclose(cen[3], centered_features(h2, obs)[3])
    # The centred first difference at i reads i+1 by construction.
    assert cen[1, 1] == pytest.approx((h[2] - h[0]) / 2.0)
    assert cau[1, 1] == pytest.approx(h[1] - h[0])


def test_naive_random_split_has_disjoint_rows_and_is_still_worthless(data):
    """Row disjointness is exactly why this flaw survives review: the obvious
    assertion passes. What fails is temporal separation."""
    s = naive_random_split(data.ws, seed=0)
    assert np.intersect1d(s.fit, s.test).size == 0
    assert np.intersect1d(s.fit, s.val).size == 0
    assert s.embargoed.size == 0

    t_fit, t_test = data.ws.t_time[s.fit], data.ws.t_time[s.test]
    assert t_fit.max() > t_test.min(), "a random split must interleave in time"
    # Interleaving is not a technicality: nearly every test window has a
    # training window within one sampling period of it.
    for pid in sorted({str(p) for p in data.ws.patient})[:1]:
        m_f = s.fit[np.asarray([str(p) for p in data.ws.patient[s.fit]]) == pid]
        m_t = s.test[np.asarray([str(p) for p in data.ws.patient[s.test]]) == pid]
        tf = np.sort(data.ws.t_time[m_f].astype("datetime64[m]").astype(np.int64))
        tt = data.ws.t_time[m_t].astype("datetime64[m]").astype(np.int64)
        nearest = np.abs(tf[np.clip(np.searchsorted(tf, tt), 0, tf.size - 1)] - tt)
        assert np.median(nearest) <= 10.0, "test windows should sit inside training time"


def test_naive_windows_straddle_the_official_file_boundary(fake_root):
    """The strict builder windows per FILE, so a window cannot read training
    history to predict a test target. The control arm concatenates first, which
    is the leak this ladder rung exists to price."""
    recs = discover_records(fake_root, subsets=("2018", "2020"))
    strict = build_ohio_windows(recs, CFG, HupaConfig(), OhioConfig(),
                                horizons_min=HORIZONS)
    naive = build_naive_windows(recs, CFG, HupaConfig(), OhioConfig(),
                                horizons_min=HORIZONS)
    assert len(naive) > len(strict), \
        "dropping the observed-target and segment rules must ADD windows"

    # In the strict set every subject's train windows precede every test window.
    for pid in sorted({str(p) for p in strict.patient}):
        m = strict.patient == pid
        assert strict.t_time[m & (strict.split == "train")].max() < \
            strict.t_time[m & (strict.split == "test")].min()

    # In the naive set, windows exist whose history reaches back across it.
    P = np.timedelta64(CFG.sample_period_min, "m")
    hist = P * CFG.history_steps
    straddles = 0
    for pid in sorted({str(p) for p in naive.patient}):
        m = naive.patient == pid
        first_test = naive.t_time[m & (naive.split == "test")].min()
        starts = naive.t_time[m & (naive.split == "test")] - hist
        straddles += int((starts < first_test).sum())
    assert straddles > 0, "the control arm is not actually straddling the boundary"


def test_leak_ladder_is_a_ladder():
    """Three rungs that differ in exactly one thing at a time, so the ablation
    can attribute the inflation rather than just report it."""
    keys = [a.key for a in LEAK_LADDER]
    assert keys == ["strict", "random_split", "naive"]
    assert LEAK_LADDER[0].windows == "strict" and LEAK_LADDER[0].split == "official"
    assert LEAK_LADDER[1].windows == "strict" and LEAK_LADDER[1].split == "random"
    assert LEAK_LADDER[2].windows == "naive" and LEAK_LADDER[2].split == "random"


def test_naive_windows_keep_the_strict_channel_layout(fake_root):
    """The comparison must measure leakage, not architecture: if the tensors had
    different widths the two arms would not even build the same model."""
    recs = discover_records(fake_root, subsets=("2018",))
    strict = build_ohio_windows(recs, CFG, HupaConfig(), OhioConfig(),
                                horizons_min=HORIZONS)
    naive = build_naive_windows(recs, CFG, HupaConfig(), OhioConfig(),
                                horizons_min=HORIZONS)
    assert naive.x.shape[1:] == strict.x.shape[1:]
    assert naive.x_aux.shape[1:] == strict.x_aux.shape[1:]
    assert naive.horizons_min == strict.horizons_min
    assert np.isfinite(naive.x).all() and np.isfinite(naive.y_reg_multi).all()


def test_a_preprocessor_fits_on_the_naive_split_without_complaint(data, fake_root):
    """Ablation D refits the preprocessor on each rung's own (leaky) fit rows,
    because borrowing the strict arm's would repair half the flaw."""
    s = naive_random_split(data.ws, seed=1)
    pre = OhioPreprocessor.fit(data.ws, s, CFG)
    assert pre.fitted_on_rows == s.fit.size
    assert set(pre.fitted_on_patients) == {str(p) for p in data.ws.patient}


# --------------------------------------------------------------------------- #
# 8. Table serialisation
# --------------------------------------------------------------------------- #
def _rows():
    return [
        {"arm": "A", "per_horizon": {"30": {"regression": {"rmse": 20.0},
                                            "classification": {"pr_auc": 0.5}}}},
        {"arm": "B_x", "per_horizon": {"30": {"regression": {"rmse": 18.0},
                                              "classification": {"pr_auc": 0.7}}}},
        {"arm": "C", "per_horizon": {"30": {"regression": {},
                                            "classification": {"pr_auc": None}}}},
    ]


COLS = (B.Column("RMSE", "regression.rmse", "{:.2f}", "min"),
        B.Column("PR-AUC", "classification.pr_auc", "{:.3f}", "max"))


def test_latex_bolds_the_winner_in_each_direction():
    tex = B.latex_table(_rows(), COLS, 30, "cap", "tab:t")
    assert "\\textbf{18.00}" in tex and "\\textbf{0.700}" in tex
    assert "\\textbf{20.00}" not in tex


def test_missing_values_render_as_a_dash_not_as_zero():
    """A metric that is undefined for an arm (time lag on a shuffled test set,
    Parkes without a backend) must not silently become 0.00 and win its column."""
    tex = B.latex_table(_rows(), COLS, 30, "cap", "tab:t")
    assert " -- " in tex or "& -- " in tex
    assert "0.00" not in tex.replace("18.00", "").replace("20.00", "")


def test_latex_escapes_characters_that_would_break_the_document():
    tex = B.latex_table(_rows(), COLS, 30, "cap", "tab:t")
    assert "B\\_x" in tex, "an underscore in an arm name must be escaped"


def test_text_and_latex_tables_report_the_same_numbers():
    """Two renderers, one set of numbers -- a divergence would mean the console
    output and the paper disagree."""
    txt = B.text_table(_rows(), COLS, 30)
    tex = B.latex_table(_rows(), COLS, 30, "cap", "tab:t")
    for v in ("20.00", "18.00", "0.500", "0.700"):
        assert v in txt and v in tex


def test_json_writer_survives_nan_and_numpy(tmp_path):
    """A three-hour run must not die at the serialisation step, and a nan must
    become null rather than the literal `NaN`, which is not valid JSON."""
    p = str(tmp_path / "r.json")
    B.write_json(p, {"a": np.float32(1.5), "b": float("nan"), "c": np.arange(3),
                     "d": {"e": float("inf")}, "f": np.bool_(True)})
    with open(p) as f:
        raw = f.read()
    assert "NaN" not in raw and "Infinity" not in raw
    loaded = json.loads(raw)
    assert loaded["a"] == 1.5 and loaded["b"] is None
    assert loaded["c"] == [0, 1, 2] and loaded["d"]["e"] is None
    assert loaded["f"] is True


def test_dig_returns_the_default_for_a_missing_path():
    assert B.dig({"a": {"b": 1}}, "a.b") == 1
    assert B.dig({"a": {"b": 1}}, "a.c") is None
    assert B.dig({"a": 1}, "a.b.c") is None


# --------------------------------------------------------------------------- #
# 9. Paired statistics for the personalisation sweep
# --------------------------------------------------------------------------- #
def _sweep_mod():
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_ohio_personalization_sweep.py")
    spec = importlib.util.spec_from_file_location("sweep_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_stats_detects_a_real_improvement():
    m = _sweep_mod()
    rng = np.random.default_rng(0)
    before = rng.normal(20.0, 1.0, 12)
    after = before - 1.0 + rng.normal(0, 0.1, 12)
    st = m.paired_stats(before, after, better="min")
    assert st["n"] == 12 and st["n_improved"] == 12
    assert st["mean_delta"] < 0 and st["p_t"] < 0.001
    assert st["ci95_high"] < 0, "the CI should exclude zero for an obvious effect"
    assert st["wilcoxon_p"] < 0.01


def test_paired_stats_reports_no_effect_when_there_is_none():
    m = _sweep_mod()
    rng = np.random.default_rng(1)
    before = rng.normal(20.0, 1.0, 12)
    st = m.paired_stats(before, before + rng.normal(0, 1.0, 12), better="min")
    assert st["p_t"] > 0.05 and st["ci95_low"] < 0 < st["ci95_high"]


def test_paired_stats_handles_the_zero_shot_control():
    """A mode that changes nothing has no variance and no test to run; saying so
    beats emitting a nan with no reason."""
    m = _sweep_mod()
    v = [20.0, 21.0, 19.0]
    st = m.paired_stats(v, list(v), better="min")
    assert st["mean_delta"] == 0.0 and np.isnan(st["p_t"])
    assert "zero-shot" in st["note"]


def test_paired_stats_drops_non_finite_pairs_instead_of_propagating_nan():
    m = _sweep_mod()
    before = [20.0, 21.0, np.nan, 19.0, 22.0]
    after = [19.0, 20.4, 18.0, 18.1, 21.5]
    st = m.paired_stats(before, after, better="min")
    assert st["n"] == 4 and st["n_dropped"] == 1
    assert np.isfinite(st["mean_delta"]) and np.isfinite(st["p_t"])


def test_paired_stats_refuses_a_degenerate_constant_delta():
    """Every subject moving by exactly the same amount has no between-subject
    variance, so a paired t-test is undefined -- scipy would emit a
    catastrophic-cancellation warning and a p-value of 0 that means nothing."""
    m = _sweep_mod()
    st = m.paired_stats([20.0, 21.0, 19.0], [19.0, 20.0, 18.0], better="min")
    assert st["mean_delta"] == pytest.approx(-1.0)
    assert np.isnan(st["p_t"]) and "degenerate" in st["note"]


def test_paired_stats_sign_convention_follows_the_metric():
    """`n_improved` normalises the direction so a reader does not have to
    remember which way each metric runs."""
    m = _sweep_mod()
    before = [0.50, 0.52, 0.48, 0.55, 0.51, 0.49]
    after = [0.60, 0.58, 0.61, 0.62, 0.57, 0.63]
    assert m.paired_stats(before, after, "max")["n_improved"] == 6
    assert m.paired_stats(before, after, "min")["n_improved"] == 0


def test_holm_bonferroni_is_monotone_and_conservative():
    m = _sweep_mod()
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    adj = m.holm_bonferroni(raw)
    assert all(adj[k] >= raw[k] for k in raw), "adjustment must never lower a p"
    ordered = [adj[k] for k in sorted(raw, key=raw.get)]
    assert ordered == sorted(ordered), "Holm's step-down must be monotone"
    assert adj["a"] == pytest.approx(0.004)
    assert all(v <= 1.0 for v in adj.values())


def test_holm_ignores_undefined_pvalues():
    m = _sweep_mod()
    adj = m.holm_bonferroni({"a": 0.01, "none": float("nan")})
    assert adj["a"] == pytest.approx(0.01)      # family of size 1
    assert np.isnan(adj["none"])


# --------------------------------------------------------------------------- #
# 10. The drivers are importable and their flags parse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["run_paper_benchmarks.py",
                                    "run_paper_ablations.py",
                                    "run_ohio_personalization_sweep.py"])
def test_driver_exposes_a_working_cli(script):
    """`--help` exercises the whole argparse setup, which is where a typo in a
    default or a bad `choices=` list would otherwise sit until the run starts."""
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", script),
                          "--help"], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "--out" in out.stdout and "--smoke" in out.stdout


def test_benchmark_arms_cover_every_model_the_paper_claims():
    m = _sweep_mod  # noqa: F841  (keep the loader symbol used above)
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_paper_benchmarks.py")
    spec = importlib.util.spec_from_file_location("bench_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import argparse
    ns = argparse.Namespace(horizons=[30, 60], epochs=1, batch_size=64, lr=1e-3,
                            patience=1, seed=1, fit_subsample=1, lora_rank=4,
                            prefix_hours=24.0, adapt_steps=5, arima_order="(4,1,0)")
    specs = mod.build_specs(ns)
    families = {s.family for s in specs}
    assert families == {"persistence", "ridge_ar", "arima", "lgbm", "lstm",
                        "tcn", "tcn_mm", "lora"}
    assert len({s.name for s in specs}) == len(specs), "duplicate arm names"
    lora = [s for s in specs if s.family == "lora"][0]
    assert lora.base_arm in {s.name for s in specs}, \
        "the LoRA arm must name a base arm that is actually trained"


def test_ablation_studies_enumerate_distinct_arms():
    import argparse
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_paper_ablations.py")
    spec = importlib.util.spec_from_file_location("abl_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ns = argparse.Namespace(horizons=[30, 60], epochs=1, batch_size=64, lr=1e-3,
                            patience=1, seed=1, fit_subsample=1,
                            kappas=[1.0, 2.0, 3.0, 5.0, 8.0])
    a = mod.specs_a(ns)
    assert len(a) == 4 and len({s.name for s in a}) == 4
    # The four arms must differ in the loss configuration, or the ablation is
    # four runs of the same model.
    assert len({(s.loss_weighting, s.lambda_reg, s.lambda_cls) for s in a}) == 4
    assert any(s.lambda_cls == 0.0 for s in a) and any(s.lambda_reg == 0.0 for s in a)
    assert any(s.loss_weighting == "uncertainty" for s in a)
    # A regression-only arm selected on PR-AUC would early-stop on noise.
    reg_only = [s for s in a if s.lambda_cls == 0.0][0]
    assert reg_only.monitor.endswith("rmse") and reg_only.monitor_mode == "min"

    b = mod.specs_b(ns)
    assert [s.hypo_asymmetry for s in b] == [1.0, 2.0, 3.0, 5.0, 8.0]
    assert len({s.name for s in b}) == 5

    c = dict(mod.specs_c(ns))
    assert set(c) == {"2018", "2020"}
    for cohort, arms in c.items():
        assert all(s.cohorts == (cohort,) for s in arms)
        assert sum(1 for s in arms if s.unimodal) == 1, "exactly one CGM-only control"
        assert any(s.modalities is None for s in arms), "one full-multimodal arm"
    assert any(s.modalities == ("heart_rate",) for s in c["2018"])
    assert any(s.modalities == ("accel",) for s in c["2020"])
    # 2018 wore a Basis Peak, 2020 an Empatica: neither arm may appear in the
    # other cohort, or it would train on windows that are entirely mask.
    assert not any(s.modalities == ("accel",) for s in c["2018"])
    assert not any(s.modalities == ("heart_rate",) for s in c["2020"])


def test_leakage_inflation_is_reported_relative_to_the_strict_arm():
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_paper_ablations.py")
    spec = importlib.util.spec_from_file_location("abl_mod2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"arm": "strict", "per_horizon": {"30": {"regression": {"rmse": 20.0},
                                                 "classification": {"pr_auc": 0.40}}}},
        {"arm": "leaky", "per_horizon": {"30": {"regression": {"rmse": 16.0},
                                                "classification": {"pr_auc": 0.60}}}},
    ]
    infl = mod.leakage_inflation(rows, 30)
    assert set(infl) == {"leaky"}
    v = infl["leaky"]
    assert v["rmse_absolute_improvement"] == pytest.approx(4.0)
    assert v["rmse_relative_improvement_pct"] == pytest.approx(20.0)
    assert v["pr_auc_absolute_gain"] == pytest.approx(0.20)
    assert v["pr_auc_relative_gain_pct"] == pytest.approx(50.0)


def _tabular_blocks(tex: str) -> list[tuple[str, str, list[str]]]:
    r"""(column spec, header row, data rows) for each tabular environment.

    Rows are the lines ending in `\`, so `\midrule` and `ottomrule` are
    excluded and the header is separated from the body -- parsing by splitting
    on `\` instead leaves a fragment of `ottomrule` behind as a phantom row.
    """
    out = []
    for block in tex.split("\\begin{tabular}")[1:]:
        spec = block[block.index("{") + 1:block.index("}")]
        body = block.split("\\toprule")[1].split("\\bottomrule")[0]
        rows = [ln.strip()[:-2].strip() for ln in body.splitlines()
                if ln.strip().endswith("\\\\")]
        out.append((spec, rows[0], rows[1:]))
    return out


def _n_separators(row: str) -> int:
    """Ampersands that are column separators, i.e. not the escaped `\\&`."""
    import re
    return len(re.findall(r"(?<!\\)&", row))


def test_emitted_latex_is_structurally_valid():
    """Every row must have exactly one fewer separator than the column spec has
    columns, and braces must balance.

    An arm name containing `&` or `_` is the realistic way this breaks: the
    table still looks fine in the console and fails to compile in the paper.
    """
    rows = [
        {"arm": "strict", "per_horizon": {"30": {"r": {"v": 1.0}}}},
        {"arm": "+ interpolation & centred features_v2",
         "per_horizon": {"30": {"r": {"v": 2.0}}}},
        {"arm": "100% naive", "per_horizon": {"30": {"r": {}}}},
    ]
    cols = (B.Column("A", "r.v", "{:.2f}", "min"), B.Column("B", "r.v", "{:.2f}"))
    tex = B.latex_table(rows, cols, 30, "cap with 50% & a _underscore", "tab:x")

    assert tex.count("{") == tex.count("}"), "unbalanced braces"
    blocks = _tabular_blocks(tex)
    assert len(blocks) == 1
    spec, header, body = blocks[0]
    assert spec == "l" + "r" * len(cols)
    assert len(body) == len(rows)
    for row in [header, *body]:
        assert _n_separators(row) == len(cols), \
            f"row has {_n_separators(row)} separators, expected {len(cols)}: {row}"
    assert "\\&" in tex and "\\_" in tex, "special characters must be escaped"


def test_latex_row_count_matches_the_arms_given():
    rows = [{"arm": f"a{i}", "per_horizon": {"30": {"r": {"v": float(i)}}}}
            for i in range(5)]
    tex = B.latex_table(rows, (B.Column("V", "r.v"),), 30, "c", "tab:y")
    _spec, header, body = _tabular_blocks(tex)[0]
    assert header.startswith("Model")
    assert len(body) == 5
    assert [r.split("&")[0].strip() for r in body] == [f"a{i}" for i in range(5)]


# --------------------------------------------------------------------------- #
# 11. Auxiliary validity: the bug that made three ablation arms identical
# --------------------------------------------------------------------------- #
# `MultiModalCausalCGMMT` reads per-step auxiliary validity from a set of
# observation-mask channels. It used to read ONE channel, defaulted to
# `hr_observed`, and that was wrong in two compounding ways:
#
#   * the 2020 OhioT1DM subjects wore an Empatica, which reports no heart rate,
#     so `hr_observed` is identically zero across every one of their windows --
#     the cross-attention was fully masked and the "multimodal" model was the
#     CGM-only model for half the cohort, while still reporting itself as
#     multimodal;
#   * every modality-ablation arm that excluded the heart-rate group zeroed that
#     channel too, so CGM+GSR, CGM+steps and CGM+accel all silently became the
#     same CGM-only model and produced bit-identical metrics.
#
# These tests fail if either behaviour comes back.
def test_2020_cohort_has_no_heart_rate_but_does_have_other_sensors(data):
    """The premise of the bug, asserted on the fixture so the tests below are
    not merely testing a hypothetical."""
    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX

    sub = np.asarray([str(t) for t in data.ws.subset_tag])
    hr = data.ws.x_aux[sub == "2020"][..., OHIO_AUX_INDEX["hr_observed"]]
    assert hr.size > 0 and (hr > 0.5).mean() == 0.0, \
        "2020 subjects should carry no heart rate at all"
    gsr = data.ws.x_aux[sub == "2020"][..., OHIO_AUX_INDEX["gsr_observed"]]
    assert (gsr > 0.5).mean() > 0.5, "...but they do carry other wearables"


def test_aux_validity_indices_follow_the_arm_not_one_sensor():
    from causalcgm_mt.ohio_loader import (
        MODALITY_GROUPS, OHIO_AUX_INDEX, aux_validity_indices,
    )

    assert aux_validity_indices(["gsr"]) == (OHIO_AUX_INDEX["gsr_observed"],)
    assert aux_validity_indices(["heart_rate"]) == (OHIO_AUX_INDEX["hr_observed"],)
    assert aux_validity_indices(["accel"]) == (OHIO_AUX_INDEX["accel_observed"],)
    # Different arms must get different validity channels, or they collapse.
    assert aux_validity_indices(["gsr"]) != aux_validity_indices(["steps"])
    # Every group's masks, and only masks.
    every = aux_validity_indices()
    assert set(every) == {OHIO_AUX_INDEX[c] for g in MODALITY_GROUPS
                          for c in MODALITY_GROUPS[g] if c.endswith("_observed")}
    assert len(every) == len(set(every)) and list(every) == sorted(every)


def test_a_group_with_no_observation_mask_yields_every_step_valid():
    """Insulin-on-board has no mask because a zero is a real zero. An arm made
    only of such streams must be ALWAYS valid, never never-valid."""
    from causalcgm_mt.ohio_loader import aux_validity_indices
    assert aux_validity_indices(["treatment"]) == ()
    assert aux_validity_indices(["exercise"]) == ()


def test_unknown_modality_group_is_rejected():
    from causalcgm_mt.ohio_loader import aux_validity_indices
    with pytest.raises(ValueError, match="unknown modality group"):
        aux_validity_indices(["heartrate"])


@pytest.mark.parametrize("given,expected", [
    (None, ()), (7, (7,)), ([7, 16], (7, 16)), ((16,), (16,)), (np.int64(9), (9,)),
])
def test_model_normalises_the_validity_channel_spec(given, expected):
    """A bare int must keep working -- HUPA-UCM and older checkpoints pass one."""
    import torch

    from causalcgm_mt.models import build_multimodal_model
    from causalcgm_mt.config import ModelConfig
    m, _ = build_multimodal_model(5, 24, ModelConfig(), 1.0, aux_valid_index=given)
    assert m.aux_valid_index == expected
    x, xa = torch.randn(2, 12, 5), torch.zeros(2, 12, 24)
    with torch.no_grad():
        bg, logit = m.eval()(x, xa, torch.full((2,), 120.0), torch.ones(2))
    assert bg.shape == (2,) and logit.shape == (2,)


def test_a_step_is_valid_when_any_configured_mask_is_set():
    """`any`, not `all`: a subject wearing one working sensor out of four is
    still contributing auxiliary information at that step."""
    import torch

    from causalcgm_mt.models import build_multimodal_model
    from causalcgm_mt.config import ModelConfig

    m, _ = build_multimodal_model(5, 24, ModelConfig(dropout=0.0), 1.0,
                                  aux_valid_index=[7, 17], modality_dropout=0.0)
    m.eval()
    x = torch.randn(1, 12, 5)
    last, ap = torch.full((1,), 120.0), torch.ones(1)

    none_set = torch.zeros(1, 12, 24)
    hr_only = none_set.clone(); hr_only[:, :, 7] = 1.0
    gsr_only = none_set.clone(); gsr_only[:, :, 17] = 1.0
    with torch.no_grad():
        out_none = m(x, none_set, last, ap)[0]
        out_hr = m(x, hr_only, last, ap)[0]
        out_gsr = m(x, gsr_only, last, ap)[0]
    # Either mask alone must change the output away from the fully-masked case.
    assert not torch.allclose(out_none, out_hr, atol=1e-7)
    assert not torch.allclose(out_none, out_gsr, atol=1e-7)


def test_the_fusion_branch_is_live_for_a_heart_rate_free_arm(data, tiny):
    """The regression test for the bug itself.

    A GSR-only arm on the 2020 cohort has no heart rate anywhere. Under the old
    single-channel gate its `aux_valid` was zero at every timestep, the
    cross-attention was fully masked, and the model was CGM-only. Here it must
    see live auxiliary steps.
    """
    import torch

    from causalcgm_mt.ohio_loader import aux_validity_indices

    spec = B.ArmSpec("gsr 2020", "tcn_mm", modalities=("gsr",), cohorts=("2020",),
                     **tiny)
    split = B.restrict_split(data, spec)
    cols = [list(data.horizons).index(h) for h in spec.horizons]
    arrays = B._arm_arrays(data, spec, split.test[:64], cols)
    xa = torch.from_numpy(np.ascontiguousarray(arrays["x_aux"]))

    idx = list(aux_validity_indices(spec.modalities))
    assert idx, "a GSR arm must have a validity channel"
    valid = (xa[..., idx] > 0.5).any(dim=-1).float()
    assert float(valid.mean()) > 0.5, \
        "the GSR arm's auxiliary stream is masked out at nearly every step"

    from causalcgm_mt.ohio_loader import OHIO_AUX_INDEX
    old_gate = (xa[..., OHIO_AUX_INDEX["hr_observed"]] > 0.5).float()
    assert float(old_gate.mean()) == 0.0, \
        "this fixture no longer reproduces the condition that caused the bug"


def test_two_modality_arms_receive_different_auxiliary_tensors(data, tiny):
    """The observable symptom: arms that differ in their sensors must differ in
    what reaches the model. Bit-identical inputs are how three ablation rows
    ended up with bit-identical metrics."""
    cols = [0, 1]
    rows = data.split.test[:128]
    made = {}
    for mods in (("gsr",), ("steps",), ("heart_rate",), None):
        spec = B.ArmSpec(str(mods), "tcn_mm", modalities=mods, **tiny)
        made[mods] = B._arm_arrays(data, spec, rows, cols)["x_aux"]
    keys = list(made)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(made[keys[i]], made[keys[j]]), \
                f"{keys[i]} and {keys[j]} produced identical auxiliary tensors"


def test_training_records_and_warns_about_a_dead_fusion_gate(data, tiny):
    """A gate that never opens is invisible in every metric, so it is recorded
    in the results file and warned about at training time.

    Constructed by handing an arm a modality whose sensor the cohort does not
    carry -- heart rate on the 2020 Empatica subjects -- which is exactly the
    real configuration that produced three bit-identical ablation rows.
    """
    spec = B.ArmSpec("hr on 2020", "tcn_mm", modalities=("heart_rate",),
                     cohorts=("2020",), **tiny)
    with pytest.warns(UserWarning, match="gated off on every fit window"):
        t = B.train_neural_arm(data, spec, B.restrict_split(data, spec),
                               verbose=False)
    assert t.aux_valid_frac == pytest.approx(0.0)

    ok = B.ArmSpec("gsr on 2020", "tcn_mm", modalities=("gsr",),
                   cohorts=("2020",), **tiny)
    t2 = B.train_neural_arm(data, ok, B.restrict_split(data, ok), verbose=False)
    assert t2.aux_valid_frac > 0.5, "a GSR arm on 2020 must have a live gate"
    assert t2.aux_valid_index and t2.aux_valid_index != t.aux_valid_index


# --------------------------------------------------------------------------- #
# 12. The control arm must fail the causality test the strict pipeline passes
# --------------------------------------------------------------------------- #
# `test_ohio_loader.test_inputs_ignore_the_future` corrupts every sample after
# an anchor and asserts the strict pipeline's inputs do not move. Ablation D is
# only meaningful if the control arm FAILS that same test -- otherwise it is
# measuring nothing and will report a reassuring zero.
#
# This is not hypothetical. The first version of `centered_features` was called
# on each window's own history slice, where centring has no future to reach
# into, so the "leaky" arm leaked nothing past the anchor and the ablation
# measured almost no inflation.
def _mutate_future(records, cut_frac=0.5):
    import copy as _copy
    out = []
    cut = None
    for rec in records:
        r = _copy.deepcopy(rec)
        cgm = r.fields["glucose_level"]
        if cut is None:
            cut = cgm["ts"].iloc[int(len(cgm) * cut_frac)]
        tail = cgm["ts"] > cut
        cgm.loc[tail, "value"] = 400.0
        r.fields["glucose_level"] = cgm
        out.append(r)
    return out, cut


def test_naive_features_do_look_past_the_anchor(fake_root):
    """The decisive test for the control arm: corrupt the future and the naive
    inputs MUST move, at anchors whose own history is untouched."""
    recs = discover_records(fake_root, subsets=("2018",))[:1]
    mutated, cut = _mutate_future(recs)

    base = build_naive_windows(recs, CFG, HupaConfig(), OhioConfig(),
                               horizons_min=HORIZONS)
    after = build_naive_windows(mutated, CFG, HupaConfig(), OhioConfig(),
                                horizons_min=HORIZONS)
    assert len(base) == len(after), "the mutation must not change the geometry"

    # Every window anchored at or before the cut has an ENTIRELY unmutated
    # history, because its history is [t-H+1, t] and t <= cut. If any of their
    # feature tensors move, the only possible source is a lookahead.
    m = base.t_time <= np.datetime64(cut)
    assert m.sum() > 50, "fixture too short to test"
    assert not np.array_equal(base.x[m], after.x[m]), (
        "the control arm's features do not read the future -- Ablation D would "
        "measure nothing and report a reassuring zero")

    # Localise it: only the anchors whose centred window reaches across the cut
    # may move, and they must be a small minority. A change everywhere would
    # mean the mutation reached the history too and the test proves nothing.
    moved = ~np.all(base.x[m] == after.x[m], axis=(1, 2))
    assert 0 < moved.sum() < 0.2 * m.sum(), (
        f"{moved.sum()} of {m.sum()} unmutated-history windows moved; expected "
        f"only those within the centring half-window of the cut")


def test_strict_features_ignore_the_future_on_the_same_fixture(fake_root):
    """The other half of the contrast, on the identical records, so the two
    arms differ in the pipeline and not in the data."""
    recs = discover_records(fake_root, subsets=("2018",))[:1]
    mutated, cut = _mutate_future(recs)

    base = build_ohio_windows(recs, CFG, HupaConfig(), OhioConfig(),
                              horizons_min=HORIZONS)
    after = build_ohio_windows(mutated, CFG, HupaConfig(), OhioConfig(),
                               horizons_min=HORIZONS)
    m = base.t_time <= np.datetime64(cut)
    assert m.sum() > 50
    np.testing.assert_array_equal(base.x[m], after.x[m])
    np.testing.assert_array_equal(base.x_aux[m], after.x_aux[m])


def test_centered_features_leak_only_when_computed_before_windowing():
    """The order of operations IS the flaw, stated as an executable fact.

    Per-window centring cannot see past the anchor; whole-series centring
    followed by slicing can. Both use the same function.
    """
    from causalcgm_mt.data import causal_features

    full = np.arange(24, dtype=np.float64) * 10.0
    obs = np.ones(24)
    H, anchor = 8, 15
    hs = slice(anchor - H + 1, anchor + 1)

    fut = full.copy()
    fut[anchor + 1:] = -999.0

    per_window = centered_features(full[hs], obs[hs])
    per_window_mut = centered_features(fut[hs], obs[hs])
    assert np.allclose(per_window, per_window_mut), \
        "a window slice has no future inside it to leak"

    sliced = centered_features(full, obs)[hs]
    sliced_mut = centered_features(fut, obs)[hs]
    assert not np.allclose(sliced, sliced_mut), \
        "whole-series centring followed by slicing must leak past the anchor"

    # And the strict extractor never moves, whichever way it is called.
    assert np.allclose(causal_features(full, obs)[hs],
                       causal_features(fut, obs)[hs])


def test_ablation_study_keys_cover_every_letter():
    """A partial re-run merges into the existing summary, so the merge has to
    know exactly which JSON keys it is replacing. A letter missing from this map
    would silently preserve a stale study alongside its own replacement."""
    import argparse
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_paper_ablations.py")
    spec = importlib.util.spec_from_file_location("abl_keys", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    all_keys = mod._study_keys(argparse.Namespace(studies=["a", "b", "c", "d"]))
    assert all_keys == {"a_multitask", "b_asymmetry", "c_modality", "d_leakage"}
    assert mod._study_keys(argparse.Namespace(studies=["d"])) == {"d_leakage"}
    # Every key the driver writes must be one the merge knows about.
    import inspect
    src = inspect.getsource(mod.main)
    for k in all_keys:
        assert f'payload["studies"]["{k}"]' in src, \
            f"{k} is written but not declared in _study_keys"


# --------------------------------------------------------------------------- #
# 13. Prediction dumps and the third conformal variant
# --------------------------------------------------------------------------- #
def test_saved_predictions_round_trip_and_match_the_scored_rows(data, tmp_path):
    """A figure must be a pure function of one artefact file.

    If the dump did not carry its own references, a plotting script would have
    to re-derive the split -- and would silently draw a different one the moment
    any windowing parameter changed.
    """
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    res, _ = B.run_arm(spec, data, verbose=False,
                       predictions_dir=str(tmp_path / "preds"))
    path = res["predictions_file"]
    assert os.path.exists(path)

    z = np.load(path, allow_pickle=False)
    n = data.split.test.size
    assert z["bg_hat"].shape == (n, len(HORIZONS))
    assert z["risk"].shape == z["bg_hat"].shape
    assert list(z["horizons"]) == list(HORIZONS)
    assert str(z["arm"]) == "persistence"
    # References must be the rows that were actually scored.
    np.testing.assert_allclose(z["y_reg"][:, 0], data.ws.y_reg_multi[data.split.test, 0])
    np.testing.assert_allclose(z["last_bg"], data.ws.last_bg[data.split.test])
    # Persistence predicts last_bg, so the dump is checkable against the method.
    np.testing.assert_allclose(z["bg_hat"][:, 0], z["last_bg"], rtol=0, atol=0)
    # And the RMSE recomputed from the dump must equal the reported one.
    rmse = float(np.sqrt(np.mean((z["bg_hat"][:, 0] - z["y_reg"][:, 0]) ** 2)))
    assert rmse == pytest.approx(
        res["per_horizon"]["30"]["regression"]["rmse"], rel=1e-6)


def test_predictions_are_not_written_unless_asked(data, tmp_path):
    res, _ = B.run_arm(B.ArmSpec("p", "persistence", horizons=HORIZONS), data,
                       verbose=False, predictions_dir=None)
    assert "predictions_file" not in res


def test_arm_names_with_latex_become_safe_filenames(data, tmp_path):
    r"""Ablation arms are named things like `$\kappa$ = 3`; those characters
    cannot go into a path unescaped."""
    spec = B.ArmSpec(r"$\kappa$ = 3 / fusion", "persistence", horizons=HORIZONS)
    res, _ = B.run_arm(spec, data, verbose=False,
                       predictions_dir=str(tmp_path / "p"))
    name = os.path.basename(res["predictions_file"])
    assert name.endswith(".npz") and "/" not in name and "\\" not in name
    assert os.path.exists(res["predictions_file"])


def test_conformal_block_reports_all_three_constructions(data):
    """Figure 4 contrasts constant-width, normalised and Mondrian coverage, so
    all three have to be computed on the same rows from the same predictions --
    otherwise the figure compares methods and splits at once."""
    spec = B.ArmSpec("persistence", "persistence", horizons=HORIZONS)
    res, _ = B.run_arm(spec, data, verbose=False)
    c = res["per_horizon"]["30"]["conformal"]
    assert set(c) >= {"alpha", "split", "normalized", "mondrian"}
    for name in ("split", "normalized", "mondrian"):
        blk = c[name]
        assert 0.0 <= blk["marginal_coverage"] <= 1.0
        assert blk["mean_width"] > 0.0
        assert set(blk["by_stratum"]) <= set(
            __import__("causalcgm_mt.conformal", fromlist=["STRATUM_NAMES"]).STRATUM_NAMES)
        assert blk["n_calib"] == data.split.calib.size
    # And the artefact must say WHICH region calibrated it, so a reader can
    # confirm calibration and model selection were disjoint without re-running.
    assert c["calibrated_on_split"] == "calib"
    assert c["n_selection_rows"] == data.split.val.size
    # Constant-width is by construction one width everywhere; the other two
    # are not, which is the whole point of using them.
    assert c["split"]["mean_width"] == pytest.approx(c["split"]["median_width"])
    assert c["normalized"]["mean_width"] != pytest.approx(
        c["normalized"]["median_width"])


def test_normalized_conformal_difficulty_uses_no_label(data):
    """The normalised variant's sigma must be a function of the input window
    alone; a label on the calibration path would be leakage dressed as
    adaptivity."""
    from causalcgm_mt.conformal import rate_of_change_difficulty

    sigma = rate_of_change_difficulty(data.ws, data.split.test)
    poisoned = data.ws.subset(np.arange(len(data.ws)))
    poisoned.y_reg_multi = poisoned.y_reg_multi + 500.0
    poisoned.y_reg = poisoned.y_reg + 500.0
    np.testing.assert_array_equal(
        sigma, rate_of_change_difficulty(poisoned, data.split.test))


# --------------------------------------------------------------------------- #
# 14. LaTeX escaping must not mangle deliberate math
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    (r"$\kappa$ = 3", r"$\kappa$ = 3"),                 # math passes through
    (r"Joint, fixed $\lambda=1$", r"Joint, fixed $\lambda=1$"),
    ("CausalCGM-MT + LoRA r=4", "CausalCGM-MT + LoRA r=4"),  # nothing to do
    ("B_x", r"B\_x"),                                    # unsafe, escaped
    ("100% naive", r"100\% naive"),
    ("a & b", r"a \& b"),
    ("cost $5 only", r"cost \$5 only"),                  # unpaired $, escaped
    (r"$\kappa$ & b_1", r"$\kappa$ \& b\_1"),            # mixed
])
def test_latex_escaping_preserves_math_and_escapes_the_rest(raw, expected):
    r"""Arm names are authored WITH math in them, on purpose.

    A blanket escaper turns `$\kappa$ = 3` into the literal characters
    `\$\textbackslash{}kappa\$ = 3`, which still compiles and still looks like a
    table -- the damage only shows in the rendered PDF, which is the worst place
    to find it. It must still escape the genuinely unsafe characters, because an
    arm named with a bare `_` or `&` breaks the document outright.
    """
    assert B._escape_latex(raw) == expected


def test_kappa_rows_render_as_symbols_in_a_real_table():
    """End-to-end through `latex_table`, since that is what the drivers call."""
    rows = [{"arm": rf"$\kappa$ = {k}",
             "per_horizon": {"30": {"r": {"v": float(k)}}}} for k in (1, 3, 8)]
    tex = B.latex_table(rows, (B.Column("V", "r.v", "{:.2f}", "min"),), 30,
                        "cap", "tab:k", row_label="Arm")
    assert r"$\kappa$ = 1 &" in tex
    assert "textbackslash" not in tex, "math was escaped into literal text"
    _spec, _header, body = _tabular_blocks(tex)[0]
    assert len(body) == 3
    for row in body:
        assert _n_separators(row) == 1
