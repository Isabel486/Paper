from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

import random_walk_submission_analysis as base
from extra_analyses import absolute_distance_sd_exact, expected_absolute_distance_exact

OUTPUT_DIR = base.OUTPUT_DIR
ANALYSIS_ID_APPROXIMATION = 5
STEP_COUNTS = [10, 20, 50, 100]
TRIALS_PER_N_SQUARED = [1, 3, 10, 30, 100, 300]
REPETITIONS = 200
Z_CRITICAL = 1.96


def asymptotic(metric: str, steps: int) -> float:
    if metric == "average_absolute_distance":
        return math.sqrt(2 * steps / math.pi)
    return math.sqrt(2 / (math.pi * steps))


def exact_and_sd(metric: str, steps: int) -> tuple[float, float]:
    if metric == "average_absolute_distance":
        return expected_absolute_distance_exact(steps, 0.5), absolute_distance_sd_exact(steps, 0.5)
    p0 = base.endpoint_origin_probability_exact(steps, 0.5)
    return p0, math.sqrt(p0 * (1 - p0))


def false_alarm_probability(delta: float) -> float:
    return float(stats.norm.cdf(-Z_CRITICAL - delta) + 1 - stats.norm.cdf(Z_CRITICAL - delta))


def critical_trials(metric: str, steps: int) -> dict:
    exact, sd = exact_and_sd(metric, steps)
    approx = asymptotic(metric, steps)
    relative_error = (approx - exact) / exact
    cv = sd / exact
    n_star = (Z_CRITICAL * cv / relative_error) ** 2
    return {
        "metric": metric,
        "steps": steps,
        "exact": exact,
        "asymptotic": approx,
        "relative_error": relative_error,
        "relative_error_times_4n": relative_error * 4 * steps,
        "cv": cv,
        "critical_trials": n_star,
        "critical_trials_over_n_squared": n_star / steps**2,
    }


def run_false_alarm_simulation() -> pd.DataFrame:
    metric = "average_absolute_distance"
    rows = []
    for condition_index, (steps, ratio) in enumerate(
        (s, r) for s in STEP_COUNTS for r in TRIALS_PER_N_SQUARED
    ):
        trials = ratio * steps**2
        exact, sd = exact_and_sd(metric, steps)
        approx = asymptotic(metric, steps)
        se = sd / math.sqrt(trials)
        z_vs_approx, z_vs_exact = [], []
        for repetition in range(REPETITIONS):
            seed = base.condition_seed(ANALYSIS_ID_APPROXIMATION, condition_index, repetition)
            positions = base.simulate_final_positions(steps, trials, 0.5, seed)
            estimate = float(np.mean(np.abs(positions)))
            z_vs_approx.append((estimate - approx) / se)
            z_vs_exact.append((estimate - exact) / se)
        z_vs_approx, z_vs_exact = np.array(z_vs_approx), np.array(z_vs_exact)
        delta = (exact - approx) / se
        observed = float(np.mean(np.abs(z_vs_approx) > Z_CRITICAL))
        rows.append(
            {
                "steps": steps,
                "trials": trials,
                "trials_over_n_squared": ratio,
                "repetitions": REPETITIONS,
                "false_alarm_rate_vs_approximation": observed,
                "false_alarm_rate_se": math.sqrt(observed * (1 - observed) / REPETITIONS),
                "predicted_false_alarm_rate": false_alarm_probability(delta),
                "rejection_rate_vs_exact": float(np.mean(np.abs(z_vs_exact) > Z_CRITICAL)),
                "mean_z_vs_approximation": float(np.mean(z_vs_approx)),
                "predicted_mean_z": delta,
            }
        )
        print(f"n={steps:4d} N={trials:>9,d}  observed {observed:.3f}  predicted {rows[-1]['predicted_false_alarm_rate']:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    critical = pd.DataFrame(
        [critical_trials(m, s) for m in ["average_absolute_distance", "endpoint_origin_probability"]
         for s in [10, 20, 50, 100, 200, 500, 1000, 5000]]
    )
    critical.to_csv(OUTPUT_DIR / "approximation_critical_trials.csv", index=False)
    simulated = run_false_alarm_simulation()
    simulated.to_csv(OUTPUT_DIR / "approximation_false_alarms.csv", index=False)

    se_rate = np.sqrt(simulated.predicted_false_alarm_rate * (1 - simulated.predicted_false_alarm_rate)
                      / simulated.repetitions).replace(0, np.nan)
    deviation_in_se = ((simulated.false_alarm_rate_vs_approximation - simulated.predicted_false_alarm_rate).abs()
                       / se_rate)
    pd.DataFrame([{
        "conditions": len(simulated),
        "repetitions_per_condition": REPETITIONS,
        "mean_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.mean(),
        "min_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.min(),
        "max_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.max(),
        "max_abs_observed_minus_predicted": (simulated.false_alarm_rate_vs_approximation
                                             - simulated.predicted_false_alarm_rate).abs().max(),
        "max_deviation_in_se": deviation_in_se.max(),
    }]).to_csv(OUTPUT_DIR / "approximation_summary.csv", index=False)

    cv_limit = math.sqrt(math.pi / 2 - 1)
    limit = (Z_CRITICAL * 4 * cv_limit) ** 2
    from scipy.optimize import brentq
    delta_10 = brentq(lambda d: false_alarm_probability(d) - 0.10, 0.0, Z_CRITICAL)
    limit_10 = (delta_10 * 4 * cv_limit) ** 2
    exact_10 = {}
    for steps in [10, 20, 50, 100]:
        exact, sd = exact_and_sd("average_absolute_distance", steps)
        r = (asymptotic("average_absolute_distance", steps) - exact) / exact
        exact_10[steps] = (delta_10 * (sd / exact) / r) ** 2
    pd.DataFrame([{
        "limit_critical_trials_over_n_squared": limit,
        "cv_limit": cv_limit,
        "delta_for_10pct_false_alarms": delta_10,
        "limit_10pct_trials_over_n_squared": limit_10,
        "steps_needed_for_10pct_at_10000_trials": math.sqrt(10_000 / limit_10),
        **{f"trials_for_10pct_n{k}": v for k, v in exact_10.items()},
        **{f"predicted_false_alarm_n{k}_N10000": false_alarm_probability(
            -(asymptotic("average_absolute_distance", k) - exact_and_sd("average_absolute_distance", k)[0])
            / (exact_and_sd("average_absolute_distance", k)[1] / 100.0)) for k in [10, 20, 50, 100]},
    }]).to_csv(OUTPUT_DIR / "approximation_rule_constant.csv", index=False)

    manifest_path = OUTPUT_DIR / "run_manifest.txt"
    if "5. Approximation" not in manifest_path.read_text(encoding="utf-8"):
        with open(manifest_path, "a", encoding="utf-8") as manifest:
            manifest.write(
                "\n5. Approximation false-alarm study (approximation_study.py)\n"
                f"    Step counts: {STEP_COUNTS}; trials N = ratio * n^2 for ratio in {TRIALS_PER_N_SQUARED}\n"
                f"    Repetitions per condition: {REPETITIONS}; p_right: 0.50; seed family analysis_id = 5.\n"
            )
    print(critical.round(4).to_string(index=False))
    print(f"large-n limit of N*/n^2: {limit:.2f}")


if __name__ == "__main__":
    main()
