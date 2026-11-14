"""Structured multimodal feature ablation on OhioT1DM.

    python scripts/run_ablation.py --data runs/ohio/ohio_windows.pt --epochs 30
    python scripts/run_ablation.py --data runs/ohio/ohio_windows.pt --cohort 2018
    python scripts/run_ablation.py --data runs/ohio/ohio_windows.pt \
        --arms cgm_only cgm_hr cgm_gsr --horizons 30 60

WHAT THIS CONTROLS FOR, AND WHY IT HAS TO
-----------------------------------------
The 2018 subjects wore a Basis Peak and the 2020 subjects an Empatica Embrace,
and the two devices expose DISJOINT sensor sets:

    heart rate, steps, air temperature   ->  2018 only (0 events in every 2020 file)
    acceleration                         ->  2020 only
    GSR, skin temperature                ->  both, but in different units

So "CGM + heart rate" can only ever be measured on six subjects, and
"CGM + accelerometry" on a DIFFERENT six. Reporting those two side by side as
evidence about sensors is really a comparison of two disjoint patient groups --
with n=6 each, between-patient variance in glycemic control swamps any sensor
effect. This runner therefore:

  1. restricts every arm to the cohorts that actually carry its sensors,
  2. re-runs the CGM-only control INSIDE each cohort restriction, so every
     comparison has a control drawn from the same subjects, and
  3. groups the report by cohort, and refuses to print a single cross-cohort
     ranking.

Everything else is held identical across arms: same windows, same official
split, same embargo, same preprocessing, same model shape (channels outside the
arm are zeroed, not removed, so parameter count cannot differ), same seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalcgm_mt.ohio_loader import DEFAULT_ABLATION, resolve_arm

HERE = os.path.dirname(os.path.abspath(__file__))


def run_arm(arm: str, horizon: int, args, cohorts=None) -> dict | None:
    """Train one arm in a subprocess and read back its report.

    A subprocess rather than an in-process loop, deliberately: each arm gets a
    fresh interpreter, so no CUDA/MPS allocator state, RNG state or module-level
    cache can carry from one arm into the next and become a confound.
    """
    scope = "".join(cohorts) if cohorts else "all"
    out = os.path.join(args.out, f"{arm}_ph{horizon}_{scope}")
    cmd = [sys.executable, os.path.join(HERE, "train_ohio.py"),
           "--data", args.data, "--horizon", str(horizon),
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
           "--seed", str(args.seed), "--out", out]
    if arm == "cgm_only":
        cmd.append("--unimodal")
    else:
        cmd += ["--arm", arm]
    if cohorts and set(cohorts) != {"2018", "2020"}:
        cmd += ["--cohort", *cohorts]

    print(f"\n{'=' * 72}\n### ARM {arm} @ PH={horizon} min\n{'=' * 72}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  arm {arm} FAILED (exit {r.returncode}); continuing")
        return None

    tag = f"ph{horizon}_{arm if arm != 'cgm_only' else 'uni'}"
    if cohorts and set(cohorts) != {"2018", "2020"}:
        tag += "_" + "".join(sorted(cohorts))
    path = os.path.join(out, f"report_{tag}.json")
    if not os.path.exists(path):
        print(f"  no report at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="default: every arm in DEFAULT_ABLATION")
    ap.add_argument("--horizons", nargs="*", type=int, default=[30])
    ap.add_argument("--cohort", default=None, choices=["2018", "2020"],
                    help="only run arms this cohort can support")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="runs/ohio_ablation")
    args = ap.parse_args()

    arms = args.arms or [a for a, _ in DEFAULT_ABLATION]
    plan = []
    for arm in arms:
        groups = dict(DEFAULT_ABLATION).get(arm, ())
        _, cohorts = resolve_arm(groups) if groups else (None, ["2018", "2020"])
        if not cohorts:
            print(f"skipping {arm}: no cohort carries all of {groups}")
            continue
        if args.cohort and args.cohort not in cohorts:
            continue
        plan.append((arm, tuple(cohorts)))

    print("Ablation plan:")
    for arm, cohorts in plan:
        print(f"  {arm:16s} cohorts={list(cohorts)}")

    # Every cohort scope that appears in the plan needs its OWN CGM-only
    # control, trained and scored on exactly the subjects its arms use.
    # Comparing a 6-subject 2018 arm against a 12-subject control would attribute
    # a patient-mix difference to the sensor.
    scopes = {c for _, c in plan}
    for scope in sorted(scopes):
        if ("cgm_only", scope) not in [(a, c) for a, c in plan]:
            plan.append(("cgm_only", scope))

    results = {}
    for horizon in args.horizons:
        for arm, cohorts in plan:
            rep = run_arm(arm, horizon, args, cohorts=list(cohorts))
            if rep:
                results[(horizon, arm, cohorts)] = (rep, cohorts)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.json"), "w") as f:
        json.dump({f"ph{h}/{''.join(c)}/{a}": {"cohorts": list(c), **r}
                   for (h, a, c), (r, _) in results.items()}, f, indent=2)

    # Report grouped by cohort scope. Arms measured on different subject sets
    # are never placed in the same ranking.
    for horizon in args.horizons:
        by_scope: dict[tuple, list] = {}
        for (h, arm, scope), (rep, cohorts) in results.items():
            if h == horizon:
                by_scope.setdefault(scope, []).append((arm, rep))
        for cohorts, rows in sorted(by_scope.items()):
            print(f"\n=== PH={horizon} min | cohort scope {list(cohorts)} "
                  f"({'all 12' if len(cohorts) == 2 else '6'} subjects) ===")
            print(f"{'arm':18s} {'RMSE':>7s} {'PR-AUC':>7s} {'Sens@5%':>8s} "
                  f"{'A+B%':>7s} {'D+E n':>6s}")
            for arm, rep in sorted(rows, key=lambda r: r[1]["statistical"]["rmse"]):
                st, cl = rep["statistical"], rep["clinical"]
                print(f"{arm:18s} {st['rmse']:7.3f} {st['pr_auc']:7.4f} "
                      f"{st['sens_at_5pct_fpr']:8.4f} {cl['zone_ab_pct']:7.2f} "
                      f"{cl['zone_de_n']:6d}")
            print("  (compare only within this block: each block is one "
                  "subject set, with its own cgm_only control)")


if __name__ == "__main__":
    main()
