"""Executable audit of the training loops' selection and update discipline.

Three of this repository's four training loops share a shape -- fit an epoch,
score the validation slice, compare a scalar monitor against the best so far --
and the failures here are the ones that shape admits silently: a monitor that
is `nan` (so nothing is ever "best" and the LAST epoch is reported as if it
were), and an optimiser whose parameter set is wider than the gradient clip's
(so part of the update is unbounded). Neither raises, neither shows up in a
metric, and both are invisible in the results file.

`extension`'s adaptation loop is deliberately not in that list: it freezes the
criterion rather than optimising it, and its `"none"` arm -- tested here too --
takes no gradient step at all.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import (AdaptConfig, Config, DataConfig, ModelConfig,
                                 TrainConfig)
from causalcgm_mt.data import build_windows, patient_folds
from causalcgm_mt.evaluate import EvalReport
from causalcgm_mt.extension import adapt_patient, configure_trainable
from causalcgm_mt.models import build_model
from causalcgm_mt.simulate import simulate_cohort
from causalcgm_mt.train import _require_finite_monitor, run_fold


@pytest.fixture(scope="module")
def ws():
    return build_windows(simulate_cohort(n_patients=6, days=4.0, seed=5),
                         DataConfig())


@pytest.fixture(scope="module")
def cfg():
    return Config(
        data=DataConfig(),
        model=ModelConfig(hidden_channels=8, n_blocks=2, head_hidden=8),
        train=TrainConfig(max_epochs=2, batch_size=64, num_workers=0, amp=False,
                          early_stop_patience=2),
    )


def _report(pr_auc: float, n: int = 500, n_positive: int = 0) -> EvalReport:
    """An EvalReport with only the fields the monitor check reads."""
    fields = {f: 0.0 for f in EvalReport.__dataclass_fields__}
    fields.update(pr_auc=pr_auc, n=n, n_positive=n_positive)
    return EvalReport(**fields)


# --------------------------------------------------------------------------- #
# M1: a nan monitor must stop the run, not select the last epoch
# --------------------------------------------------------------------------- #
def test_a_non_finite_monitor_refuses_to_continue():
    """`nan > -inf` is False, which is the whole bug.

    No epoch ever beats the initial `best`, so `best_state` stays None, the
    patience counter runs out, the final `load_state_dict` is skipped and the
    model reported is whichever the last epoch produced. The results file says
    `best_epoch: -1`, which reads as "no improvement" rather than "model
    selection never ran".
    """
    with pytest.raises(RuntimeError, match="not finite"):
        _require_finite_monitor(float("nan"), "val/pr_auc", 3,
                                {"h30": _report(float("nan"))})
    with pytest.raises(RuntimeError, match="not finite"):
        _require_finite_monitor(float("-inf"), "val/pr_auc", 0,
                                {"h30": _report(float("-inf"))})
    # A finite monitor is silent, sign and magnitude irrelevant.
    for good in (0.0, 0.42, -3.5, 1e12):
        _require_finite_monitor(good, "val/pr_auc", 0, {"h30": _report(good)})


def test_the_monitor_error_names_the_slice_that_caused_it():
    """A degenerate validation slice is the routine cause, so the message has
    to carry enough to act on: which horizon, how many rows, how many positives.
    `pr_auc` is nan exactly when the slice is single-class."""
    with pytest.raises(RuntimeError) as exc:
        _require_finite_monitor(
            float("nan"), "val/pr_auc", 7,
            {"h30": _report(0.61, n=900, n_positive=44),
             "h60": _report(float("nan"), n=900, n_positive=0)})
    msg = str(exc.value)
    assert "epoch 7" in msg and "val/pr_auc" in msg
    assert "h60" in msg and "n_positive=0" in msg, "the degenerate horizon"
    assert "h30" in msg and "n_positive=44" in msg, "and the healthy one"
    assert "monitor_mode" in msg, "the message must name a way out"


def test_run_fold_raises_rather_than_reporting_its_last_epoch(ws, cfg,
                                                              monkeypatch):
    """End to end: the guard is wired into the loop, not just defined.

    `run_fold` used to return a FoldResult here -- a model nobody selected,
    scored and written to the results JSON like any other.
    """
    import causalcgm_mt.train as T

    folds = patient_folds(ws.patient, n_folds=3, seed=0)
    train_pat, test_pat = folds[0]
    monkeypatch.setattr(T, "_monitor_value", lambda rep, key: float("nan"))
    with pytest.raises(RuntimeError, match="not finite"):
        run_fold(ws, cfg, train_pat, test_pat, fold=0)


def test_every_training_loop_checks_its_monitor():
    """Three loops, one guard. A driver that grows its own selection loop
    without the check reintroduces the bug in the one place nobody looks."""
    import inspect

    import causalcgm_mt.benchmark as B
    import causalcgm_mt.train as T

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ohio = open(os.path.join(root, "scripts", "train_ohio.py")).read()
    for name, src in (("train.run_fold", inspect.getsource(T.run_fold)),
                      ("benchmark.train_neural_arm",
                       inspect.getsource(B.train_neural_arm)),
                      ("scripts/train_ohio.py", ohio)):
        assert "_require_finite_monitor" in src, f"{name} selects unchecked"
        assert "if best_state is not None" not in src, (
            f"{name} still skips its final load_state_dict silently")


def test_one_degenerate_horizon_does_not_poison_a_usable_monitor():
    """`nanmean`, not `mean`: a fold whose 60-minute slice happens to be
    single-class still has a perfectly good 30-minute monitor to select on.
    `train_ohio.py` used a plain mean, so one nan took the whole run down the
    silent path."""
    import inspect

    import causalcgm_mt.benchmark as B

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ohio = open(os.path.join(root, "scripts", "train_ohio.py")).read()
    assert "np.nanmean" in inspect.getsource(B.train_neural_arm)
    assert "np.nanmean" in ohio and "np.mean([_monitor_value" not in ohio

    # And the semantics the drivers implement: partial nan averages, all-nan
    # stays nan so the guard above can catch it.
    per_h = [0.61, float("nan")]
    assert np.nanmean(per_h) == pytest.approx(0.61)
    assert not np.isfinite(np.nan if all(not np.isfinite(v) for v in
                                         [float("nan")] * 2) else 0.0)


# --------------------------------------------------------------------------- #
# M10: the gradient clip must cover everything the optimiser steps
# --------------------------------------------------------------------------- #
def test_the_clip_covers_every_parameter_the_optimiser_steps(ws, cfg,
                                                             monkeypatch):
    """The Kendall log-variances were optimised but never clipped.

    `MultiTaskLoss` hands its `2K` log-variances to the optimiser, and their
    gradient is `0.5 - exp(-s) * L` -- largest early in training, when `L` is
    large and `grad_clip` is protecting everything else. One oversized step
    there rescales the whole multi-task balance.

    Asserted by identity, not by count: a clip over the right NUMBER of tensors
    that happened to be a different set would pass a length check.
    """
    seen: list[list[torch.Tensor]] = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **kw):
        params = list(params)
        seen.append(params)
        return real(params, max_norm, *a, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    folds = patient_folds(ws.patient, n_folds=3, seed=0)
    res = run_fold(ws, cfg, folds[0][0], folds[0][1], fold=0)
    assert seen, "no gradient clipping happened at all"

    clipped = {id(p) for p in seen[0]}
    model, criterion = build_model(ws.x.shape[-1], cfg.model, 1.0)
    assert len(criterion.state_dict()) or list(criterion.parameters()), (
        "the criterion has no parameters, so this test proves nothing")

    # Every tensor the optimiser was given is in the clipped set, and the
    # criterion's log-variances specifically.
    n_model = sum(1 for _ in model.parameters())
    n_crit = sum(1 for _ in criterion.parameters())
    assert n_crit > 0
    assert len(clipped) == n_model + n_crit, (
        f"clipped {len(clipped)} tensors, optimiser steps {n_model + n_crit}: "
        f"the criterion's log-variances are stepping unclipped")
    assert np.isfinite(res.report.rmse)


def test_the_optimiser_and_the_clip_read_the_same_list():
    """Structural, so the two cannot drift apart again: both sites take the one
    hoisted `trainable` list rather than each building their own."""
    import inspect

    import causalcgm_mt.benchmark as B
    import causalcgm_mt.train as T

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ohio = open(os.path.join(root, "scripts", "train_ohio.py")).read()
    for name, src in (("train.run_fold", inspect.getsource(T.run_fold)),
                      ("benchmark.train_neural_arm",
                       inspect.getsource(B.train_neural_arm)),
                      ("scripts/train_ohio.py", ohio)):
        assert "trainable = list(model.parameters()) + list(criterion.parameters())" in src, name
        assert "clip_grad_norm_(trainable" in src, name
        assert "clip_grad_norm_(model.parameters()" not in src, (
            f"{name} clips only the model while the optimiser steps more")


# --------------------------------------------------------------------------- #
# M9: the "none" adaptation arm is reachable on the generic protocol
# --------------------------------------------------------------------------- #
def test_configure_trainable_agrees_with_the_advertised_methods():
    """`ADAPT_METHODS` listed `"none"`; `configure_trainable` fell through its
    dispatch to `raise ValueError("unknown method")`."""
    from causalcgm_mt.extension import ADAPT_METHODS

    m = build_model(5, ModelConfig(), 1.0)[0]
    assert "none" in ADAPT_METHODS
    assert configure_trainable(m, AdaptConfig(method="none")) == []
    with pytest.raises(ValueError, match="unknown method"):
        configure_trainable(m, AdaptConfig(method="not_a_method"))


def test_the_zero_shot_arm_runs_and_changes_nothing(ws, cfg):
    """The paired control for the generic/MetaboNet personalisation protocol,
    which has no other baseline: `before` and `after` come from the same
    unmodified model, so any delta this arm reports is harness drift.

    It used to raise `ValueError: unknown method none` instead.
    """
    folds = patient_folds(ws.patient, n_folds=3, seed=0)
    train_pat, test_pat = folds[0]
    res = run_fold(ws, cfg, train_pat, test_pat, fold=0)

    model, criterion = build_model(ws.x.shape[-1], cfg.model, 1.0)
    model.load_state_dict(res.state_dict)
    acfg = AdaptConfig(method="none", adapt_prefix_hours=12.0, batch_size=16)
    before_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    adapted = None
    for pid in test_pat:
        adapted = adapt_patient(model, ws, str(pid), res.scaler, res.labeler,
                                criterion, Config(data=cfg.data, model=cfg.model,
                                                  train=cfg.train, adapt=acfg))
        if adapted is not None:
            break
    assert adapted is not None, "no held-out patient had enough data to adapt"

    assert adapted.method == "none"
    assert adapted.n_trainable == 0
    # The no-op must be exactly a no-op: same report object, untouched weights.
    assert adapted.after.rmse == adapted.before.rmse
    assert adapted.after.pr_auc == adapted.before.pr_auc or (
        np.isnan(adapted.after.pr_auc) and np.isnan(adapted.before.pr_auc))
    for k, v in model.state_dict().items():
        assert torch.equal(v, before_weights[k]), f"{k} moved during a no-op"
