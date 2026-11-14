"""Synthetic CGM generator, for pipeline smoke-testing only.

This exists so the repository is runnable end to end without OhioT1DM/DIAtrend
access. It is a crude oscillator with meal spikes, insulin-driven dips and
device dropout -- NOT a physiological model. Never report numbers from it as
results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig

# Read from the config rather than retyped, so a change to the label threshold
# cannot leave this generator quietly emitting a cohort with no positives.
HYPO_THRESHOLD_MGDL: float = DataConfig().hypo_threshold_mgdl


def simulate_hupa_cohort(
    n_patients: int = 10,
    days: float = 6.0,
    seed: int = 0,
    cgm_period_min: int = 5,
    hr_period_min: int = 1,
    hr_dropout: float = 0.12,
    sleep_available: bool = True,
) -> pd.DataFrame:
    """HUPA-UCM-SHAPED synthetic data: multiple streams at different cadences.

    The point of this generator is to exercise the *synchronisation* code, so it
    deliberately reproduces the awkward parts of the real thing:

      * heart rate at 1-minute cadence (finer than the CGM grid) with dropout
        blocks, so the binning and staleness logic actually get used;
      * steps as a per-minute count that must be SUMMED, not averaged;
      * sleep as sparse episodic rows, so forward fill with a budget matters;
      * boluses and carbs as rare point events at arbitrary offsets, which must
        land in the correct backward-looking bin.

    The physiology is crude -- meals raise glucose, insulin and steps lower it,
    with insulin acting at a lag. It is not a metabolic model, and no clinical
    conclusion should be drawn from it. What it does support is a genuine test
    that the aux streams carry signal: glucose here really is a function of the
    auxiliary variables at physiological lags, so a fusion model that ignores
    them should be measurably worse than one that does not.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for p in range(n_patients):
        start = pd.Timestamp("2024-03-01")
        minutes = int(days * 24 * 60)
        t_min = pd.date_range(start, periods=minutes, freq="1min")

        base = rng.normal(140, 18)
        sens = rng.uniform(0.7, 1.5)          # insulin sensitivity
        carb_ratio = rng.uniform(2.5, 4.5)    # mg/dL per gram
        rest_hr = rng.normal(64, 6)

        # --- events ---------------------------------------------------------
        n_meals = int(days * rng.integers(3, 5))
        meal_t = np.sort(rng.choice(minutes - 300, n_meals, replace=False)) + 60
        meal_g = rng.uniform(25, 80, n_meals)
        # Bolus accompanies the meal, with realistic timing error and occasional
        # over-correction (the mechanism that actually produces hypoglycemia).
        bolus_t = meal_t + rng.integers(-12, 20, n_meals)
        bolus_u = meal_g / rng.uniform(8, 11, n_meals) * rng.uniform(0.85, 1.35, n_meals)

        n_walks = int(days * rng.integers(1, 4))
        walk_t = np.sort(rng.choice(minutes - 200, n_walks, replace=False)) + 60
        walk_dur = rng.integers(20, 70, n_walks)

        # --- glucose --------------------------------------------------------
        g = np.full(minutes, base, dtype=float)
        g += 18 * np.sin(2 * np.pi * np.arange(minutes) / 1440 + rng.uniform(0, 6.3))

        k = np.arange(300)
        for mt, mg in zip(meal_t, meal_g):
            end = min(minutes, mt + 300)
            rise = carb_ratio * mg / 40 * (1 - np.exp(-k / 25)) * np.exp(-k / 110)
            g[mt:end] += rise[: end - mt] * 40 / max(rise.max(), 1e-6) * (mg / 40)

        for bt, bu in zip(bolus_t, bolus_u):
            bt = max(bt, 0)
            end = min(minutes, bt + 300)
            act = (k / 75) * np.exp(-k / 75)
            act = act / max(act.max(), 1e-6)
            g[bt:end] -= sens * bu * 18 * act[: end - bt]

        steps = np.zeros(minutes)
        for wt, wd in zip(walk_t, walk_dur):
            end = min(minutes, wt + wd)
            steps[wt:end] = rng.uniform(80, 130, end - wt)
            # Exercise drops glucose during and after the bout.
            tail = min(minutes, end + 120)
            g[wt:tail] -= np.linspace(0, rng.uniform(15, 35), tail - wt)
        steps += rng.poisson(2.0, minutes) * (rng.random(minutes) > 0.7)

        g = np.clip(g + rng.normal(0, 2.5, minutes), 38, 400)

        # --- heart rate: rest + exercise + glucose-driven counterregulation --
        hr = rest_hr + 0.10 * steps + rng.normal(0, 3, minutes)
        hr += np.clip(70 - g, 0, None) * 0.45          # tachycardia when low
        sleep = np.zeros(minutes)
        hours = np.array([ts.hour for ts in t_min])
        night = (hours >= 23) | (hours < 7)
        sleep[night] = 1.0
        hr[night] -= 8

        # --- emit each stream at its own cadence ----------------------------
        cgm_idx = np.arange(0, minutes, cgm_period_min)
        rows.append(pd.DataFrame({
            "patient_id": f"H{p:03d}", "time": t_min[cgm_idx],
            "glucose": g[cgm_idx],
        }))

        hr_idx = np.arange(0, minutes, hr_period_min)
        # Dropout in contiguous blocks, like a watch coming off the wrist.
        keep = np.ones(hr_idx.size, bool)
        for _ in range(int(days * 3)):
            s = rng.integers(0, max(hr_idx.size - 120, 1))
            keep[s : s + rng.integers(20, 120)] = False
        keep &= rng.random(hr_idx.size) > hr_dropout * 0.2
        rows.append(pd.DataFrame({
            "patient_id": f"H{p:03d}", "time": t_min[hr_idx[keep]],
            "heart_rate": hr[hr_idx[keep]], "steps": steps[hr_idx[keep]],
        }))

        if sleep_available:
            # Sparse episodic sleep rows, every 30 min while a state holds.
            sl_idx = np.arange(0, minutes, 30)
            rows.append(pd.DataFrame({
                "patient_id": f"H{p:03d}", "time": t_min[sl_idx],
                "sleep": sleep[sl_idx],
            }))

        rows.append(pd.DataFrame({
            "patient_id": f"H{p:03d}", "time": t_min[np.clip(meal_t, 0, minutes - 1)],
            "carb_input": meal_g,
        }))
        rows.append(pd.DataFrame({
            "patient_id": f"H{p:03d}",
            "time": t_min[np.clip(bolus_t, 0, minutes - 1)],
            "bolus_volume_delivered": bolus_u,
        }))

    out = pd.concat(rows, ignore_index=True).sort_values(["patient_id", "time"])
    for c in ("glucose", "heart_rate", "steps", "sleep", "carb_input",
              "bolus_volume_delivered"):
        if c not in out.columns:
            out[c] = np.nan
    return out.reset_index(drop=True)


def simulate_cohort(
    n_patients: int = 12,
    days: float = 6.0,
    period_min: int = 5,
    seed: int = 0,
    dropout_rate: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 60 / period_min)
    rows = []

    for p in range(n_patients):
        # Per-patient physiology, so patient-wise splitting is a real test.
        base = rng.normal(140, 20)
        amp = rng.uniform(20, 45)
        sens = rng.uniform(0.6, 1.6)
        noise = rng.uniform(2.0, 5.0)

        t = np.arange(steps)
        circadian = amp * np.sin(2 * np.pi * t / (24 * 60 / period_min) + rng.uniform(0, 6.3))
        g = base + circadian + rng.normal(0, noise, steps)

        # Meals: sharp rise then decay. Insulin: delayed dip, occasionally
        # overshooting into hypoglycemia.
        for _ in range(int(days * rng.integers(3, 5))):
            s = rng.integers(0, steps - 60)
            k = np.arange(60)
            g[s : s + 60] += rng.uniform(40, 90) * np.exp(-k / 18) * (1 - np.exp(-k / 3))
            d = s + rng.integers(6, 18)
            if d + 60 <= steps:
                g[d : d + 60] -= sens * rng.uniform(45, 95) * np.exp(-np.arange(60) / 22)

        # Every patient must actually reach hypoglycemia.
        #
        # The loop above subtracts `sens * U(45, 95)` mg/dL from whatever level
        # the patient happens to sit at, so a low-sensitivity draw on a high
        # baseline never crosses 70. At the shipped cohort spec four of twelve
        # patients came out hypo-free, and that is not a cosmetic imbalance:
        # `run_fold` carves its inner validation slice out of the TRAINING
        # patients, and at `n_folds=2` that slice is a single patient. Drawing a
        # hypo-free one makes `val/pr_auc` nan, and `_require_finite_monitor`
        # then aborts the run -- correctly, but it meant the documented
        # quickstart (`run_experiment.py --synthetic --epochs 3 --folds 2`)
        # failed on a clean checkout every time.
        #
        # The guarantee is a MECHANISM, not a clamp on the output: an insulin
        # over-correction with the same PK shape used in
        # `simulate_hupa_cohort`, scaled against the patient's own level at the
        # nadir so the trough lands in the 45-65 mg/dL band whatever their
        # baseline is. Two properties are deliberate:
        #   * only hypo-free patients are touched, so every other trace stays
        #     byte-identical to what earlier runs produced;
        #   * the draws come from a per-patient generator, so injecting into one
        #     patient does not shift the main stream for every patient after it.
        if g.min() >= HYPO_THRESHOLD_MGDL:
            span = int(min(48, steps))            # up to 4 h of insulin action
            k = np.arange(span)
            act = (k / 9.0) * np.exp(-k / 9.0)    # nadir at ~45 min, then recovery
            act = act / max(act.max(), 1e-9)
            hypo_rng = np.random.default_rng([seed, p])
            for _ in range(int(hypo_rng.integers(2, 4))):
                s = int(hypo_rng.integers(0, max(steps - span, 1)))
                end = min(steps, s + span)
                seg = act[: end - s]
                nadir = s + int(np.argmax(seg))
                depth = float(g[nadir]) - float(hypo_rng.uniform(45.0, 65.0))
                if depth > 0:
                    g[s:end] -= depth * seg

        g = np.clip(g, 40, 400)
        ts = pd.date_range("2024-01-01", periods=steps, freq=f"{period_min}min")

        keep = rng.random(steps) > dropout_rate
        rows.append(pd.DataFrame({
            "patient_id": f"P{p:03d}",
            "timestamp": ts[keep],
            "glucose": g[keep],
        }))

    return pd.concat(rows, ignore_index=True)
