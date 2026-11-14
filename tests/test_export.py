"""Executable audit of the ONNX export path.

`export_onnx` already gates itself on parity and causality, and those gates are
exercised by `scripts/verify_cpp_parity.py`. What was never covered is the
plumbing AROUND them: which model gets exported, and where the file lands. Both
of those failed silently, which is why they are asserted here rather than left
to the parity gate -- a graph at the wrong path still passes every numerical
check ever run against it.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.config import Config, DataConfig, ModelConfig
from causalcgm_mt.data import CausalScaler
from causalcgm_mt.export import export_onnx, load_and_export
from causalcgm_mt.models import build_model

ort = pytest.importorskip("onnxruntime", reason="ONNX export needs onnxruntime")
pytest.importorskip("onnx", reason="ONNX export reads the emitted opset back")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """A checkpoint in the shape every training driver writes."""
    cfg = Config(
        data=DataConfig(history_min=120, horizon_min=30),
        model=ModelConfig(hidden_channels=8, n_blocks=2, head_hidden=8),
    )
    in_ch = 5
    model, _ = build_model(in_ch, cfg.model)
    model.eval()

    rng = np.random.default_rng(0)
    x = rng.normal(140, 40, (32, cfg.data.history_steps, in_ch)).astype(np.float32)
    x[..., 4] = 1.0                                    # the observation mask
    scaler = CausalScaler.fit_array(
        x, np.full(32, "p1", dtype=object), ["p1"], skip_channels=(4,))

    path = str(tmp_path_factory.mktemp("ckpt") / "demo.pt")
    torch.save({"state_dict": model.state_dict(),
                "scaler": scaler,
                "in_channels": in_ch,
                "config": dataclasses.asdict(cfg)}, path)
    return path


def test_load_and_export_writes_to_the_path_it_was_given(checkpoint, tmp_path):
    """The output path used to be swallowed by `aux_scaler`.

    `export_onnx(model, scaler, cfg, out_path)` bound the path to the fourth
    POSITIONAL parameter, which is the auxiliary scaler, so `out_path` kept its
    default and every export landed on `runs/demo/causalcgm_mt.onnx`. Nothing
    raised: the unimodal branch never reads `aux_scaler`. Nothing caught it
    either, because `scripts/verify_cpp_parity.py` hardcodes that same default,
    so the parity gate passed on a file the caller had not asked for -- and two
    checkpoints exported to two paths produced one file, silently overwritten.
    """
    asked_for = str(tmp_path / "nested" / "my_model.onnx")
    default = os.path.join("runs", "demo", "causalcgm_mt.onnx")
    default_before = (os.path.getmtime(default) if os.path.exists(default)
                      else None)

    rep = load_and_export(checkpoint, asked_for)

    assert os.path.exists(asked_for), (
        f"the export went somewhere else: report says {rep.path}")
    assert rep.path == asked_for
    assert os.path.getsize(asked_for) > 0
    # ... and specifically NOT to the default the bug always wrote to.
    default_after = (os.path.getmtime(default) if os.path.exists(default)
                     else None)
    assert default_after == default_before, (
        "the export also touched runs/demo/causalcgm_mt.onnx, which is the "
        "path the aux_scaler mis-binding always wrote to")

    # The emitted graph is loadable and is the model that was asked for.
    sess = ort.InferenceSession(asked_for, providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["x_raw", "last_bg"]


def test_two_exports_to_two_paths_are_two_files(checkpoint, tmp_path):
    """The user-visible symptom: `scripts/export_onnx.py` advertises a
    positional output argument, so exporting two checkpoints must not collapse
    onto one file whose name came from the first call and weights from the
    second."""
    a, b = str(tmp_path / "a.onnx"), str(tmp_path / "b.onnx")
    load_and_export(checkpoint, a)
    load_and_export(checkpoint, b)
    assert os.path.exists(a) and os.path.exists(b)


def test_out_path_is_keyword_only(checkpoint, tmp_path):
    """The structural guard, so the mis-binding cannot come back.

    There is no positional slot for `out_path` any more, so a caller that means
    the output path has to name it. A fourth positional argument still lands on
    `aux_scaler` -- that is what the slot IS -- but it can no longer be a path
    that the exporter was supposed to honour, and a fifth is a TypeError rather
    than a silent rebinding of the tolerances.

    Deliberately signature-only below the TypeError: calling `export_onnx` with
    a mis-bound path would write a graph to the default `runs/demo/` location,
    which is the artefact `scripts/verify_cpp_parity.py` gates on.
    """
    params = inspect.signature(export_onnx).parameters
    assert params["out_path"].kind is inspect.Parameter.KEYWORD_ONLY
    # Every knob after it too -- they sit behind the same `*`.
    for name in ("opset", "atol_bg_mgdl", "atol_risk"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    positional = [n for n, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == ["model", "scaler", "cfg", "aux_scaler"]

    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    cfg = Config.from_dict(ck["config"])
    model, _ = build_model(ck["in_channels"], cfg.model)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with pytest.raises(TypeError, match="positional argument"):
        # Five positionals: the old signature accepted this and shifted every
        # tolerance by one. It raises before anything is written.
        export_onnx(model, ck["scaler"], cfg, None,
                    str(tmp_path / "positional.onnx"))

    # And `load_and_export` -- the only caller -- passes the path by keyword.
    src = inspect.getsource(load_and_export)
    assert "out_path=out_path" in src
