from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

import random_walk_submission_analysis as base

OUTPUT_DIR = base.OUTPUT_DIR
ANALYSIS_ID_PATHS = 4
PATH_STEPS = 1000
PATH_COUNT = 20
PATH_PROBABILITIES = [0.50, 0.55]
HISTOGRAM_STEP_COUNTS = [10, 100, 1000]


def endpoint_pmf(steps: int, p_right: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(steps + 1)
    return 2 * k - steps, stats.binom.pmf(k, steps, p_right)


def expected_absolute_distance_exact(steps: int, p_right: float) -> float:
    positions, probabilities = endpoint_pmf(steps, p_right)
    return float(np.sum(np.abs(positions) * probabilities))


def absolute_distance_sd_exact(steps: int, p_right: float) -> float:
    mean = steps * (2 * p_right - 1)
    second_moment = 4 * steps * p_right * (1 - p_right) + mean**2
    first = expected_absolute_distance_exact(steps, p_right)
    return math.sqrt(second_moment - first**2)


def variance_standard_error(x: np.ndarray) -> float:
    n = len(x)
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    m4 = np.mean(centered**4)
    return math.sqrt(max(m4 - m2**2 * (n - 3) / (n - 1), 0.0) / n)


def comparison_rows(final_positions: np.ndarray, steps: int, p_right: float) -> dict:
    n_trials = len(final_positions)
    absolute = np.abs(final_positions).astype(float)
    theory_mean = steps * (2 * p_right - 1)
    theory_var = 4 * steps * p_right * (1 - p_right)
    theory_abs = expected_absolute_distance_exact(steps, p_right)
    theory_p0 = base.endpoint_origin_probability_exact(steps, p_right)

    estimates = base.calculate_endpoint_estimators(final_positions)
    se = {
        "mean_position": math.sqrt(theory_var / n_trials),
        "variance": variance_standard_error(final_positions.astype(float)),
        "average_absolute_distance": absolute_distance_sd_exact(steps, p_right) / math.sqrt(n_trials),
        "endpoint_origin_probability": math.sqrt(theory_p0 * (1 - theory_p0) / n_trials),
    }
    theory = {
        "mean_position": theory_mean,
        "variance": theory_var,
        "average_absolute_distance": theory_abs,
        "endpoint_origin_probability": theory_p0,
    }
    row = {"steps": steps, "trials": n_trials, "p_right": p_right}
    for metric, estimate in estimates.items():
        row[f"{metric}"] = estimate
        row[f"{metric}_theory"] = theory[metric]
        row[f"{metric}_se"] = se[metric]
        row[f"{metric}_ci_lower"] = estimate - 1.96 * se[metric]
        row[f"{metric}_ci_upper"] = estimate + 1.96 * se[metric]
        row[f"{metric}_z"] = (estimate - theory[metric]) / se[metric] if se[metric] > 0 else np.nan
    if p_right == 0.5:
        asymptotic = {
            "average_absolute_distance": math.sqrt(2 * steps / math.pi),
            "endpoint_origin_probability": math.sqrt(2 / (math.pi * steps)) if steps % 2 == 0 else 0.0,
        }
        for metric, value in asymptotic.items():
            row[f"{metric}_asymptotic"] = value
            row[f"{metric}_asymptotic_relative_error_percent"] = 100 * (value - theory[metric]) / theory[metric]
            row[f"{metric}_z_vs_asymptotic"] = (estimates[metric] - value) / se[metric]
    return row


def run_symmetric_with_uncertainty() -> tuple[pd.DataFrame, dict]:
    rows, samples = [], {}
    for condition_index, steps in enumerate(base.SYMMETRIC_STEP_COUNTS):
        seed = base.condition_seed(base.ANALYSIS_ID_SYMMETRIC, condition_index)
        positions = base.simulate_final_positions(steps, base.SYMMETRIC_TRIALS, 0.5, seed)
        rows.append({"seed": seed, **comparison_rows(positions, steps, 0.5)})
        samples[f"symmetric_n{steps}"] = positions
    return pd.DataFrame(rows), samples


def run_biased_with_uncertainty() -> tuple[pd.DataFrame, dict]:
    rows, samples = [], {}
    for condition_index, p_right in enumerate(base.BIASED_PROBABILITIES):
        seed = base.condition_seed(base.ANALYSIS_ID_BIASED, condition_index)
        positions = base.simulate_final_positions(
            base.BIASED_STEP_COUNT, base.BIASED_TRIALS, p_right, seed
        )
        row = comparison_rows(positions, base.BIASED_STEP_COUNT, p_right)
        row["drift_to_spread_ratio"] = abs(row["mean_position_theory"]) / math.sqrt(
            row["variance_theory"]
        )
        rows.append({"seed": seed, **row})
        samples[f"biased_p{p_right:.2f}"] = positions
    return pd.DataFrame(rows), samples


def convergence_rates_with_ci(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in summary.groupby("metric", sort=False):
        ordered = group.sort_values("trials")
        x = np.log(ordered["trials"].to_numpy(dtype=float))
        y = np.log(ordered["rmse"].to_numpy(dtype=float))
        fit = stats.linregress(x, y)
        t_critical = stats.t.ppf(0.975, len(x) - 2)
        at_10k = ordered.loc[ordered["trials"] == 10_000].iloc[0]
        theory_value = float(at_10k["theoretical_value"])
        trials_for_1pct = (
            10_000 * (float(at_10k["rmse"]) / (0.01 * abs(theory_value))) ** 2
            if theory_value != 0
            else np.nan
        )
        rows.append(
            {
                "metric": metric,
                "slope": fit.slope,
                "slope_se": fit.stderr,
                "slope_ci_lower": fit.slope - t_critical * fit.stderr,
                "slope_ci_upper": fit.slope + t_critical * fit.stderr,
                "intercept": fit.intercept,
                "r_squared": fit.rvalue**2,
                "contains_minus_half": fit.slope - t_critical * fit.stderr <= -0.5
                <= fit.slope + t_critical * fit.stderr,
                "rmse_at_10000": float(at_10k["rmse"]),
                "relative_rmse_percent_at_10000": float(at_10k["relative_rmse_percent"]),
                "trials_for_1pct_relative_rmse": trials_for_1pct,
            }
        )
    return pd.DataFrame(rows)


def simulate_paths() -> dict:
    paths = {}
    for condition_index, p_right in enumerate(PATH_PROBABILITIES):
        rng = np.random.default_rng(base.condition_seed(ANALYSIS_ID_PATHS, condition_index))
        steps = np.where(rng.random((PATH_COUNT, PATH_STEPS)) < p_right, 1, -1)
        positions = np.concatenate(
            [np.zeros((PATH_COUNT, 1), dtype=np.int64), np.cumsum(steps, axis=1)], axis=1
        )
        paths[f"paths_p{p_right:.2f}"] = positions
    return paths


def drift_thresholds() -> pd.DataFrame:
    rows = []
    for p in [0.51, 0.55, 0.60, 0.70]:
        bias = 2 * p - 1
        rows.append(
            {
                "p_right": p,
                "steps_for_drift_1sd": 4 * p * (1 - p) / bias**2,
                "steps_for_drift_2sd": 4 * 4 * p * (1 - p) / bias**2,
                "ever_return_probability": 1 - abs(bias),
            }
        )
    return pd.DataFrame(rows)


def trials_for_relative_precision(steps: int = base.CONVERGENCE_STEP_COUNT, target: float = 0.01) -> pd.DataFrame:
    p0 = base.endpoint_origin_probability_exact(steps, 0.5)
    abs_mean = expected_absolute_distance_exact(steps, 0.5)
    per_trial = {
        "variance": (math.sqrt(2 * steps * (steps - 1)), float(steps)),
        "average_absolute_distance": (math.sqrt(steps - abs_mean**2), abs_mean),
        "endpoint_origin_probability": (math.sqrt(p0 * (1 - p0)), p0),
    }
    rows = [
        {
            "metric": metric,
            "steps": steps,
            "per_trial_sd": sd,
            "theoretical_value": value,
            "target_relative_se": target,
            "trials_needed": (sd / (target * value)) ** 2,
        }
        for metric, (sd, value) in per_trial.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    symmetric, symmetric_samples = run_symmetric_with_uncertainty()
    biased, biased_samples = run_biased_with_uncertainty()

    original_symmetric = pd.read_csv(OUTPUT_DIR / "symmetric_step_analysis.csv")
    original_biased = pd.read_csv(OUTPUT_DIR / "biased_probability_analysis.csv")
    for metric in ["mean_position", "variance", "average_absolute_distance", "endpoint_origin_probability"]:
        assert np.allclose(symmetric[metric], original_symmetric[metric]), metric
        assert np.allclose(biased[metric], original_biased[metric]), metric

    summary = pd.read_csv(OUTPUT_DIR / "convergence_summary.csv")
    rates = convergence_rates_with_ci(summary)

    symmetric.to_csv(OUTPUT_DIR / "symmetric_with_uncertainty.csv", index=False)
    biased.to_csv(OUTPUT_DIR / "biased_with_uncertainty.csv", index=False)
    rates.to_csv(OUTPUT_DIR / "convergence_rates_with_ci.csv", index=False)
    drift_thresholds().to_csv(OUTPUT_DIR / "drift_thresholds.csv", index=False)
    trials_for_relative_precision().to_csv(OUTPUT_DIR / "trials_for_1pct_precision.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "endpoint_samples.npz", **symmetric_samples, **biased_samples)
    np.savez_compressed(OUTPUT_DIR / "sample_paths.npz", **simulate_paths())

    manifest_path = OUTPUT_DIR / "run_manifest.txt"
    if "4. Sample paths" not in manifest_path.read_text(encoding="utf-8"):
        with open(manifest_path, "a", encoding="utf-8") as manifest:
            manifest.write(
                "\n4. Sample paths (extra_analyses.py)\n"
                f"    p_right values: {PATH_PROBABILITIES}\n"
                f"    Steps per path: {PATH_STEPS}; paths per value: {PATH_COUNT}\n"
                "    Step-by-step +/-1 increments; seed family analysis_id = 4.\n"
            )
    print("Extra analyses complete.")


if __name__ == "__main__":
    main()
