"""ONNX export with numerical-parity verification.

Why parity is checked as a hard gate, not a nicety
--------------------------------------------------
The risky part of this export is `CausalConv1d`. In PyTorch it pads
`(k-1)*dilation` on BOTH sides (nn.Conv1d has no asymmetric-padding option) and
then slices the tail off in `forward`. That slice is what makes the layer
causal. If the exporter folds the padding and drops or mis-sizes the slice, the
result is a model that reads the future -- numerically different, still fast,
and completely silent about it. A latency benchmark would never catch it.

So `export_onnx` refuses to write a model that does not match PyTorch to
tolerance -- and that parity check, against a model whose causality is asserted
structurally, is what actually rules the defect out.
`verify_output_depends_on_recent_history` adds a narrower liveness probe on top:
it catches a graph wired to nothing, and deliberately does not claim more.

The exported graph folds the population scaler into the model as a constant
affine op, so the C++ runner takes raw features and cannot desynchronise from
the training-time transform -- a scaler mismatch between training and
deployment is otherwise a very easy production bug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import CausalScaler
from .models import (CausalCGMMT, MultiModalCausalCGMMT, build_model,
                     load_state_dict_compat)


class DeployableCausalCGMMT(nn.Module):
    """Inference wrapper: raw features in, (bg_hat, risk_probability) out.

    Folding the scaler and the final sigmoid into the graph means the C++ caller
    has exactly one contract -- raw window in, mg/dL and probability out -- with
    no preprocessing constants to keep in sync on the other side of the FFI.
    """

    def __init__(self, model: CausalCGMMT, scaler: CausalScaler):
        super().__init__()
        self.model = model
        self.register_buffer("center", torch.from_numpy(np.asarray(scaler.center, np.float32)))
        self.register_buffer("scale", torch.from_numpy(np.asarray(scaler.scale, np.float32)))
        # `getattr`: a scaler pickled before `CausalScaler.clip` existed has no
        # such attribute. None for every CGM scaler, which is deliberate -- see
        # `data.SCALED_FEATURE_CLIP`.
        self.clip = getattr(scaler, "clip", None)

    def forward(self, x_raw: torch.Tensor, last_bg: torch.Tensor):
        x = (x_raw - self.center) / self.scale
        if self.clip is not None:
            x = torch.clamp(x, -self.clip, self.clip)
        bg_hat, logit = self.model(x, last_bg)
        return bg_hat, torch.sigmoid(logit)


class DeployableMultiModalCausalCGMMT(nn.Module):
    """Inference wrapper for the FUSION model: raw features in, (bg, risk) out.

    Four inputs, because the fusion model genuinely has four:

        x_raw        (B, T, C_cgm)   raw CGM features
        x_aux_raw    (B, T, C_aux)   auxiliary features
        last_bg      (B,)            mg/dL, the residual base point
        aux_present  (B,)            1 where the auxiliary stream is usable

    Both population scalers are folded in as constant affine ops and the output
    sigmoid with them, so the caller's contract is the same shape as the
    unimodal one: no preprocessing constants live on the other side of the FFI.

    ONE THING THE GRAPH CANNOT FOLD
    -------------------------------
    `OhioPreprocessor` also applies a PER-PATIENT normaliser to the
    device-scaled auxiliary channels (GSR, skin/air temperature, acceleration,
    heart rate), because their raw scale is a property of the electrode contact
    rather than of the physiology. Those statistics are per subject, so they are
    not a graph constant. `x_aux_raw` is therefore expected to be
    per-patient-normalised already; `PerPatientNormalizer.transform` is the one
    definition of that step, and a deployment computes it from the subject's own
    history exactly as the personalisation prefix does.
    """

    def __init__(self, model: MultiModalCausalCGMMT, cgm_scaler: CausalScaler,
                 aux_scaler: CausalScaler):
        super().__init__()
        self.model = model
        for name, sc in (("cgm", cgm_scaler), ("aux", aux_scaler)):
            self.register_buffer(f"{name}_center",
                                 torch.from_numpy(np.asarray(sc.center, np.float32)))
            self.register_buffer(f"{name}_scale",
                                 torch.from_numpy(np.asarray(sc.scale, np.float32)))
        # THE CLIP HAS TO BE FOLDED IN TOO, or the graph and the training
        # pipeline disagree on exactly the windows that carry a saturated
        # auxiliary value -- and no graph-level parity check would see it,
        # because both sides of that check are the graph. The auxiliary scaler
        # is clipped (`data.SCALED_FEATURE_CLIP`); the CGM one is not.
        self.cgm_clip = getattr(cgm_scaler, "clip", None)
        self.aux_clip = getattr(aux_scaler, "clip", None)

    def forward(self, x_raw: torch.Tensor, x_aux_raw: torch.Tensor,
                last_bg: torch.Tensor, aux_present: torch.Tensor):
        x = (x_raw - self.cgm_center) / self.cgm_scale
        if self.cgm_clip is not None:
            x = torch.clamp(x, -self.cgm_clip, self.cgm_clip)
        x_aux = (x_aux_raw - self.aux_center) / self.aux_scale
        if self.aux_clip is not None:
            x_aux = torch.clamp(x_aux, -self.aux_clip, self.aux_clip)
        bg_hat, logit = self.model(x, x_aux, last_bg, aux_present)
        return bg_hat, torch.sigmoid(logit)


def strip_onnx_metadata(path: str) -> int:
    """Remove every `metadata_props` entry from a graph on disk; return the count.

    WHY THIS IS NOT COSMETIC
    ------------------------
    torch's exporter annotates each emitted node with `pkg.torch.onnx.stack_trace`,
    which is a VERBATIM Python traceback line -- absolute source path included:

        File "/Users/<name>/<project>/causalcgm_mt/export.py", line 53, in forward

    So an exported graph carries the author's home directory, username and the
    local project directory name on every one of its ~300 nodes. For a published
    artefact that is a privacy leak which survives every audit of the source
    tree, because it lives in a binary nobody greps.

    These props are debug provenance only -- no shape, no runtime and no numeric
    information is stored there -- so clearing them cannot change what the graph
    computes. `export_onnx` strips BEFORE its parity check rather than after, so
    that claim is checked against PyTorch on the file that actually ships instead
    of merely asserted here.

    `load_external_data=False` keeps initializers as their external references:
    loading the weights in and writing them back would inline the tensor data
    into the `.onnx` and orphan the sibling `.onnx.data`.
    """
    import onnx

    model = onnx.load(path, load_external_data=False)
    n = 0

    def _clear(obj) -> None:
        nonlocal n
        n += len(obj.metadata_props)
        del obj.metadata_props[:]

    def _walk(graph) -> None:
        _clear(graph)
        for node in graph.node:
            _clear(node)
            # Recurse into control-flow bodies rather than walking `graph.node`
            # alone: a stack trace inside an If/Loop subgraph is no less a leak.
            for attr in node.attribute:
                if attr.HasField("g"):
                    _walk(attr.g)
                for sub in attr.graphs:
                    _walk(sub)

    _clear(model)
    _walk(model.graph)
    onnx.save(model, path)
    return n


@dataclass
class ExportReport:
    path: str
    max_abs_diff_bg: float
    max_abs_diff_risk: float
    causal_ok: bool
    n_params: int
    input_shape: tuple[int, int, int]
    opset: int
    # Recorded because the audit's F6 was exactly this going unnoticed: the
    # shipped graph was a unimodal, single-horizon, synthetic-trained model
    # while the paper reported its latency as the fusion model's.
    multimodal: bool = False
    n_horizons: int = 1
    aux_channels: int = 0
    input_names: tuple[str, ...] = ()

    def __str__(self) -> str:
        kind = "multimodal fusion" if self.multimodal else "CGM-only"
        return (
            f"ONNX export -> {self.path}\n"
            f"  model  : {kind}, K={self.n_horizons} horizon(s), "
            f"aux_channels={self.aux_channels}\n"
            f"  inputs : {', '.join(self.input_names)}\n"
            f"  parity : bg {self.max_abs_diff_bg:.3e} mg/dL, "
            f"risk {self.max_abs_diff_risk:.3e}\n"
            f"  causal : {'PASS' if self.causal_ok else 'FAIL'}\n"
            f"  opset  : {self.opset}\n"
            f"  params : {self.n_params:,}"
        )


def verify_output_depends_on_recent_history(sess, in_ch: int, hist: int,
                                           n_trials: int = 8,
                                           aux_channels: int = 0) -> bool:
    """LIVENESS probe: the graph must actually read the newest steps it is fed.

    THIS IS NOT A CAUSALITY TEST, despite what its old name and its caller's
    error message claimed. It corrupts the last few history steps and requires
    the output to change. A graph whose `CausalConv1d` trim was folded away
    reads MORE of the window and therefore changes just as much -- so it passes
    this probe. Both a causal and an acausal graph pass; only a graph that
    ignores its tail entirely fails.

    Causality is inherited from the PARITY CHECK in `export_onnx`, which
    compares the graph against the PyTorch model whose causality is asserted
    structurally in `tests/test_causality.py`. That check is the guarantee; this
    one catches the narrower failure of a graph wired to nothing.

    The complementary direction -- that outputs do not depend on data AFTER the
    window -- is structural, since post-anchor samples are never placed in the
    input tensor (see `data.build_windows`).
    """
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        x = rng.normal(120, 25, (1, hist, in_ch)).astype(np.float32)
        x[..., 4] = 1.0                              # observation mask channel
        last = np.array([x[0, -1, 0]], dtype=np.float32)

        feed = {"x_raw": x, "last_bg": last}
        if aux_channels:
            aux = rng.normal(0.0, 1.0, (1, hist, aux_channels)).astype(np.float32)
            feed |= {"x_aux": aux, "aux_present": np.ones(1, dtype=np.float32)}

        y0 = sess.run(None, feed)

        # Corrupting the newest steps must move the prediction: the anchor is
        # the single most informative input, so invariance here means the graph
        # is not reading the window it claims to.
        x2 = x.copy()
        x2[:, -3:, 0] += 60.0
        y2 = sess.run(None, {**feed, "x_raw": x2})

        if np.allclose(y0[0], y2[0], atol=1e-5) and np.allclose(y0[1], y2[1], atol=1e-5):
            return False
    return True


def export_onnx(
    model: CausalCGMMT | MultiModalCausalCGMMT,
    scaler: CausalScaler,
    cfg: Config,
    aux_scaler: CausalScaler | None = None,
    # Everything past here is keyword-only. `out_path` used to sit FIFTH
    # positionally, one slot behind `aux_scaler`, and `load_and_export` passed
    # it fourth: the path bound to the scaler, `out_path` silently kept its
    # default, and every export landed on `runs/demo/causalcgm_mt.onnx` whatever
    # the caller asked for. Nothing raised -- the unimodal branch never reads
    # `aux_scaler` -- and nothing caught it either, because the C++ parity gate
    # hardcodes that same default and so passed on a file nobody asked for.
    # There is no positional slot for the output path now: a caller that means
    # the path has to name it, and the tolerances below can no longer be shifted
    # along by one either.
    *,
    out_path: str = "runs/demo/causalcgm_mt.onnx",
    # torch's exporter treats this as a request, not a guarantee: if the
    # downgrade pass fails it falls back to its native opset and swallows the
    # error. We therefore read the opset back off the emitted graph below and
    # report what was ACTUALLY produced rather than what was asked for.
    opset: int = 18,
    # Tolerances are per-output and expressed in each output's own units,
    # because the two live on very different scales. bg_hat is mg/dL with
    # magnitude ~40-400, so a single shared absolute tolerance would either be
    # unmeetable in fp32 for bg or meaninglessly loose for risk.
    #
    # 1e-3 mg/dL is three orders of magnitude below CGM display resolution
    # (1 mg/dL) -- tight enough that any real graph defect trips it, loose
    # enough to absorb fp32 reassociation from constant folding.
    atol_bg_mgdl: float = 1e-3,
    # risk is a probability in [0, 1]; 1e-5 is genuinely strict here.
    atol_risk: float = 1e-5,
) -> ExportReport:
    """Export, then refuse to ship anything that fails parity or causality.

    Handles both architectures. The fusion model takes two extra inputs
    (`x_aux`, `aux_present`) and needs its auxiliary scaler passed as
    `aux_scaler`; exporting it without one is an error rather than a silent
    fall back to the CGM-only wrapper, because that silent fall back is exactly
    how the shipped graph came to be a different model from the reported one.
    """
    import onnxruntime as ort

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    multimodal = isinstance(model, MultiModalCausalCGMMT) or hasattr(model, "cgm_backbone")
    hist, in_ch = cfg.data.history_steps, int(scaler.center.shape[0])

    if multimodal:
        if aux_scaler is None:
            raise ValueError(
                "exporting a MultiModalCausalCGMMT needs its auxiliary scaler "
                "(pass aux_scaler=...). Silently exporting the CGM-only wrapper "
                "would emit a graph of a different architecture than the model.")
        aux_ch = int(aux_scaler.center.shape[0])
        net = DeployableMultiModalCausalCGMMT(model, scaler, aux_scaler).eval()
        dummy = (torch.randn(1, hist, in_ch), torch.randn(1, hist, aux_ch),
                 torch.full((1,), 120.0), torch.ones(1))
        input_names = ["x_raw", "x_aux", "last_bg", "aux_present"]
    else:
        aux_ch = 0
        net = DeployableCausalCGMMT(model, scaler).eval()
        dummy = (torch.randn(1, hist, in_ch), torch.full((1,), 120.0))
        input_names = ["x_raw", "last_bg"]

    # Batch is dynamic so the runner can amortise over multiple patients; the
    # time axis is FIXED, because a variable-length window would change the
    # receptive field and silently alter what the model conditions on.
    dynamic_axes = {n: {0: "batch"} for n in input_names + ["bg_hat", "risk"]}

    torch.onnx.export(
        net, dummy, out_path,
        input_names=input_names,
        output_names=["bg_hat", "risk"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # Strip the exporter's debug provenance BEFORE anything else touches the
    # file: those props embed absolute source paths (and therefore the author's
    # username and local directory names) on every node. Doing it here rather
    # than in the calling script means no export path can forget to -- and it
    # puts the strip ahead of the parity gate, so the graph that gets compared
    # against PyTorch is byte-for-byte the graph that ships.
    strip_onnx_metadata(out_path)

    # --- parity against PyTorch on a realistic batch -------------------------
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    x = rng.normal(140, 40, (64, hist, in_ch)).astype(np.float32)
    x[..., 4] = (rng.random((64, hist)) > 0.05).astype(np.float32)
    last = x[:, -1, 0].copy()
    feed = {"x_raw": x, "last_bg": last}
    torch_args = [torch.from_numpy(x), torch.from_numpy(last)]

    if multimodal:
        x_aux = rng.normal(0.0, 1.0, (64, hist, aux_ch)).astype(np.float32)
        # Exercise the availability gate on both settings, and the fully-masked
        # rows with it: the dead-attention path is the part most likely to
        # export differently from PyTorch.
        present = (rng.random(64) > 0.3).astype(np.float32)
        feed = {"x_raw": x, "x_aux": x_aux, "last_bg": last, "aux_present": present}
        torch_args = [torch.from_numpy(x), torch.from_numpy(x_aux),
                      torch.from_numpy(last), torch.from_numpy(present)]

    with torch.no_grad():
        t_bg, t_risk = net(*torch_args)
    o_bg, o_risk = sess.run(None, feed)

    d_bg = float(np.max(np.abs(t_bg.numpy() - o_bg)))
    d_risk = float(np.max(np.abs(t_risk.numpy() - o_risk)))
    causal_ok = verify_output_depends_on_recent_history(
        sess, in_ch, hist, aux_channels=aux_ch if multimodal else 0)

    import onnx as _onnx
    actual_opset = max(
        (i.version for i in _onnx.load(out_path).opset_import if i.domain in ("", "ai.onnx")),
        default=-1,
    )
    if actual_opset != opset:
        import warnings
        warnings.warn(
            f"Requested opset {opset} but the emitted graph is opset {actual_opset}. "
            "torch's downgrade pass failed silently. Check that your ONNX Runtime "
            "build supports the emitted version before deploying.",
            stacklevel=2,
        )

    rep = ExportReport(
        path=out_path,
        max_abs_diff_bg=d_bg,
        max_abs_diff_risk=d_risk,
        causal_ok=causal_ok,
        n_params=sum(p.numel() for p in net.parameters()),
        input_shape=(1, hist, in_ch),
        opset=int(actual_opset),
        multimodal=bool(multimodal),
        n_horizons=int(getattr(model, "n_horizons", 1)),
        aux_channels=int(aux_ch),
        input_names=tuple(input_names),
    )

    if d_bg > atol_bg_mgdl or d_risk > atol_risk:
        raise RuntimeError(
            f"ONNX parity failure: bg {d_bg:.2e} mg/dL (limit {atol_bg_mgdl:.0e}), "
            f"risk {d_risk:.2e} (limit {atol_risk:.0e}). "
            "Refusing to emit a checkpoint that disagrees with PyTorch.\n" + str(rep)
        )
    if not causal_ok:
        raise RuntimeError(
            "Exported graph failed the LIVENESS probe -- its output does not "
            "depend on the newest history steps at all, so it is wired to "
            "something other than the window it is being fed. (Causality is "
            "covered by the parity check above, not by this probe -- see "
            "`verify_output_depends_on_recent_history`.)\n" + str(rep)
        )
    return rep


# --------------------------------------------------------------------------- #
# The checkpoint schema
# --------------------------------------------------------------------------- #
# ONE schema, named here, because there were three and only one of them could be
# exported. `scripts/train_demo.py` wrote config+scaler and could; `cross_validate`
# omitted `config`; `scripts/train_ohio.py` -- the driver that produces the
# model the paper reports -- omitted `scaler`. So `load_and_export` raised a bare
# `KeyError` on both real drivers, and the only graph anyone could actually
# build was the synthetic demo's. That is audit finding F6's root cause: the
# shipped ONNX graph was a unimodal, single-horizon, synthetic-trained model
# while the paper quoted its latency as the fusion model's, and no amount of
# care at export time could have fixed it, because the artefact carrying the
# real model had nowhere to put its scaler.
#
# A checkpoint must carry the fitted PREPROCESSING, not just the weights: a
# graph folds the scaler in as a constant (see `DeployableCausalCGMMT`), so a
# checkpoint without one cannot produce a deployable graph at all.
REQUIRED_CHECKPOINT_KEYS: tuple[str, ...] = (
    "state_dict", "in_channels", "config", "scaler",
)
# Additionally required when `multimodal` is set. The fusion graph takes two
# extra inputs and folds a SECOND scaler; exporting it without one used to fall
# back to the CGM-only wrapper, which is how a graph of a different architecture
# than the reported model came to be shipped.
REQUIRED_MULTIMODAL_KEYS: tuple[str, ...] = ("aux_scaler", "aux_channels")


def assert_checkpoint_schema(ck: dict, path: str = "<checkpoint>") -> None:
    """Refuse a checkpoint that cannot be exported, and say what is missing.

    Called before anything is unpacked, so the failure names the absent artefact
    instead of surfacing as `KeyError: 'scaler'` from four frames down.
    """
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ck]
    if ck.get("multimodal"):
        missing += [k for k in REQUIRED_MULTIMODAL_KEYS if k not in ck]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}, so it cannot be exported.\n"
            f"  A checkpoint must carry the fitted preprocessing, not just the "
            f"weights: the exported graph folds the scaler in as a constant, so "
            f"a checkpoint without one has no deployable form.\n"
            f"  Required: {list(REQUIRED_CHECKPOINT_KEYS)}"
            f"{' + ' + str(list(REQUIRED_MULTIMODAL_KEYS)) if ck.get('multimodal') else ''}\n"
            f"  Present : {sorted(ck)}\n"
            f"  See `causalcgm_mt.export.REQUIRED_CHECKPOINT_KEYS`; "
            f"scripts/train_ohio.py and causalcgm_mt.train.cross_validate both "
            f"write it.")


def load_and_export(ckpt_path: str, out_path: str) -> ExportReport:
    """Export a checkpoint written to `REQUIRED_CHECKPOINT_KEYS`.

    Dispatches on the checkpoint's own `multimodal` flag, so the OhioT1DM fusion
    model exports through `DeployableMultiModalCausalCGMMT` with its auxiliary
    scaler rather than being silently narrowed to the CGM-only wrapper.
    """
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    assert_checkpoint_schema(ck, ckpt_path)

    cfg = Config.from_dict(ck["config"])
    if ck.get("multimodal"):
        from .models import build_multimodal_model
        model, _ = build_multimodal_model(
            ck["in_channels"], ck["aux_channels"], cfg.model,
            aux_valid_index=ck.get("aux_valid_index"))
    else:
        model, _ = build_model(ck["in_channels"], cfg.model)
    # `load_state_dict_compat` so a checkpoint written before the multi-horizon
    # refactor still exports; the remap is a no-op for current ones.
    load_state_dict_compat(model, ck["state_dict"])
    model.eval()
    return export_onnx(model, ck["scaler"], cfg,
                       aux_scaler=ck.get("aux_scaler"), out_path=out_path)


# Backwards-compatible alias. The old name asserted more than the function
# proves; it is kept bound so an external caller does not break on the rename.
verify_causality_onnx = verify_output_depends_on_recent_history
