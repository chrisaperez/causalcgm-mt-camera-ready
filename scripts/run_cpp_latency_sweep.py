"""Sweep the C++ ONNX Runtime engine across batch sizes and thread counts.

    python scripts/run_cpp_latency_sweep.py --model runs/demo/causalcgm_mt.onnx

Writes `runs/paper_results/edge/latency_sweep.json`, which `make_paper_figures.py`
turns into Figure 6. Nothing here re-implements the measurement: it shells out to
`cpp/build/causalcgm_mt_runner` and parses that binary's own report, so the figure
and the README quote the same numbers from the same source.

WHAT THE RUNNER REPORTS, AND WHAT IT DOES NOT
---------------------------------------------
Percentiles (min, p50, p90, p99, p99.9, max) over `iterations` timed Run calls,
after a warmup. It does NOT emit the raw sample vector, so Figure 6 is a
percentile chart rather than a kernel-density boxplot -- drawn from the numbers
that were actually measured instead of a distribution inferred from five of
them. For a latency budget the tail is what matters anyway: a p50 of 2 ms with a
p99.9 of 80 ms misses the deadline on one alarm in a thousand, and a mean would
hide that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNNER = os.path.join("cpp", "build", "causalcgm_mt_runner")
_PCTL = re.compile(r"^\s*(min|mean|p50|p90|p99|p99\.9|max)\s+([0-9.]+)\s*$", re.M)
_PERWINDOW = re.compile(r"Per-window p99\.9\s*:\s*([0-9.]+)")
_FEATURE = re.compile(r"Feature construction\s*:\s*([0-9.]+)")
_BUDGET = re.compile(r"Budget\s+([0-9.]+)\s+ms\s*:\s*(PASS|FAIL)")


def cpu_brand() -> str:
    """The CPU's marketing name, or "" when the platform will not say.

    `platform.processor()` returns "arm" on Apple silicon and "" on many Linux
    builds, which is not enough context for a latency claim: "0.1 ms on arm" is
    unfalsifiable, while "0.1 ms on an Apple M4 Pro" tells a reader what to scale
    from. Recorded in the artefact rather than written into the prose so the
    manuscript's hardware sentence is generated from the run that produced the
    numbers beside it.
    """
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def run_one(model: str, iterations: int, warmup: int, batch: int, threads: int) -> dict:
    """One (batch, threads) configuration. Positional args, in the runner's order."""
    out = subprocess.run(
        [RUNNER, model, str(iterations), str(warmup), str(batch), str(threads)],
        capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise RuntimeError(f"runner failed (batch={batch}, threads={threads}):\n"
                           f"{out.stdout}\n{out.stderr}")
    pct = {k: float(v) for k, v in _PCTL.findall(out.stdout)}
    missing = {"min", "p50", "p99", "max"} - set(pct)
    if missing:
        raise RuntimeError(f"runner output missing {sorted(missing)}:\n{out.stdout}")
    budget = _BUDGET.search(out.stdout)
    per_window = _PERWINDOW.search(out.stdout)
    feature = _FEATURE.search(out.stdout)
    return {
        "batch": batch, "threads": threads,
        "iterations": iterations, "warmup": warmup,
        "latency_ms": pct,
        # The per-window figure is the one to quote for a wearable: a batch of
        # 128 that takes 12 ms is 0.094 ms per prediction, not 12.
        "per_window_p999_ms": float(per_window.group(1)) if per_window else None,
        "feature_construction_us": float(feature.group(1)) if feature else None,
        "budget_ms": float(budget.group(1)) if budget else None,
        "budget_met": (budget.group(2) == "PASS") if budget else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/demo/causalcgm_mt.onnx")
    ap.add_argument("--out", default="runs/paper_results/edge")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--deadline-ms", type=float, default=50.0)
    args = ap.parse_args()

    if not os.path.exists(RUNNER):
        raise SystemExit(f"{RUNNER} not found. Build it first:\n"
                         f"    cmake -S cpp -B cpp/build && cmake --build cpp/build")
    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found. Export it first:\n"
                         f"    python scripts/export_onnx.py")

    rows = []
    for threads in args.threads:
        for batch in args.batches:
            r = run_one(args.model, args.iterations, args.warmup, batch, threads)
            p = r["latency_ms"]
            rows.append(r)
            print(f"  batch {batch:>4} threads {threads}: p50 {p['p50']:8.3f} ms | "
                  f"p99 {p['p99']:8.3f} | max {p['max']:8.3f} | "
                  f"per-window p99.9 {r['per_window_p999_ms']} ms")

    import platform
    payload = {
        "experiment": "cpp_edge_latency",
        "model": args.model,
        "runner": RUNNER,
        "deadline_ms": args.deadline_ms,
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "cpu": cpu_brand(),
                 "physical_cores": os.cpu_count()},
        "note": ("percentiles as reported by the runner over `iterations` timed "
                 "Run calls after warmup; the raw sample vector is not emitted, "
                 "so downstream figures are percentile charts, not KDE boxplots"),
        "runs": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "latency_sweep.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
