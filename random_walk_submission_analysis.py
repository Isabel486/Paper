from __future__ import annotations

import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

BASE_SEED = 20260729
SYMMETRIC_STEP_COUNTS = [10, 50, 100, 500, 1000, 5000]
SYMMETRIC_TRIALS = 10_000

BIASED_STEP_COUNT = 1000
BIASED_TRIALS = 10_000
BIASED_PROBABILITIES = [0.50, 0.55, 0.60, 0.70]

CONVERGENCE_STEP_COUNT = 1000
CONVERGENCE_TRIAL_COUNTS = [100, 500, 1000, 5000, 10_000, 50_000]
CONVERGENCE_REPETITIONS = 30
CONVERGENCE_P_RIGHT = 0.50

ANALYSIS_ID_SYMMETRIC = 1
ANALYSIS_ID_BIASED = 2
ANALYSIS_ID_CONVERGENCE = 3


def condition_seed(
    analysis_id: int,
    condition_index: int,
    repetition_index: int = 0,
) -> int:
    seed = (
        BASE_SEED
        + analysis_id * 10_000_000
        + condition_index * 10_000
        + repetition_index
    )
    if seed >= 2**63:
        raise ValueError("Derived seed exceeds the supported integer range.")
    return seed


def simulate_final_positions(
    steps: int,
    trials: int,
    p_right: float,
    seed: int,
) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if trials <= 1:
        raise ValueError("trials must exceed 1 because sample variance uses N-1.")
    if not 0.0 <= p_right <= 1.0:
        raise ValueError("p_right must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    right_steps = rng.binomial(n=steps, p=p_right, size=trials)
    return (2 * right_steps - steps).astype(np.int64, copy=False)


def endpoint_origin_probability_exact(steps: int, p_right: float = 0.5) -> float:
    if steps % 2 == 1:
        return 0.0

    k = steps // 2
    if p_right == 0.0 or p_right == 1.0:
        return float(steps == 0)

    log_probability = (
        math.lgamma(steps + 1)
        - math.lgamma(k + 1)
        - math.lgamma(steps - k + 1)
        + k * math.log(p_right)
        + (steps - k) * math.log1p(-p_right)
    )
    return math.exp(log_probability)


def symmetric_expected_absolute_distance_exact(steps: int) -> float:
    if steps % 2 == 0:
        return steps * endpoint_origin_probability_exact(steps, 0.5)
    return steps * endpoint_origin_probability_exact(steps - 1, 0.5)


def theoretical_endpoint_metrics(steps: int, p_right: float) -> Dict[str, float]:
    mean = steps * (2.0 * p_right - 1.0)
    variance = 4.0 * steps * p_right * (1.0 - p_right)

    values = {
        "theoretical_mean_position": mean,
        "theoretical_variance": variance,
        "theoretical_endpoint_origin_probability":
            endpoint_origin_probability_exact(steps, p_right),
    }

    if p_right == 0.5:
        values["theoretical_average_absolute_distance"] = (
            symmetric_expected_absolute_distance_exact(steps)
        )
    else:
        values["theoretical_average_absolute_distance"] = np.nan

    return values


def calculate_endpoint_estimators(final_positions: np.ndarray) -> Dict[str, float]:
    return {
        "mean_position": float(np.mean(final_positions)),
        "variance": float(np.var(final_positions, ddof=1)),
        "average_absolute_distance": float(np.mean(np.abs(final_positions))),
        "endpoint_origin_probability": float(np.mean(final_positions == 0)),
    }


def run_symmetric_analysis() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for condition_index, steps in enumerate(SYMMETRIC_STEP_COUNTS):
        seed = condition_seed(
            ANALYSIS_ID_SYMMETRIC,
            condition_index,
            repetition_index=0,
        )
        final_positions = simulate_final_positions(
            steps=steps,
            trials=SYMMETRIC_TRIALS,
            p_right=0.5,
            seed=seed,
        )
        estimates = calculate_endpoint_estimators(final_positions)
        theory = theoretical_endpoint_metrics(steps, 0.5)

        rows.append(
            {
                "analysis": "symmetric_step_comparison",
                "condition_index": condition_index,
                "seed": seed,
                "steps": steps,
                "trials": SYMMETRIC_TRIALS,
                "p_right": 0.5,
                **estimates,
                **theory,
                "mean_absolute_error": abs(
                    estimates["mean_position"]
                    - theory["theoretical_mean_position"]
                ),
                "variance_absolute_error": abs(
                    estimates["variance"]
                    - theory["theoretical_variance"]
                ),
                "average_absolute_distance_error": abs(
                    estimates["average_absolute_distance"]
                    - theory["theoretical_average_absolute_distance"]
                ),
                "endpoint_origin_probability_error": abs(
                    estimates["endpoint_origin_probability"]
                    - theory["theoretical_endpoint_origin_probability"]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_biased_analysis() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for condition_index, p_right in enumerate(BIASED_PROBABILITIES):
        seed = condition_seed(
            ANALYSIS_ID_BIASED,
            condition_index,
            repetition_index=0,
        )
        final_positions = simulate_final_positions(
            steps=BIASED_STEP_COUNT,
            trials=BIASED_TRIALS,
            p_right=p_right,
            seed=seed,
        )
        estimates = calculate_endpoint_estimators(final_positions)
        theory = theoretical_endpoint_metrics(BIASED_STEP_COUNT, p_right)

        rows.append(
            {
                "analysis": "biased_probability_comparison",
                "condition_index": condition_index,
                "seed": seed,
                "steps": BIASED_STEP_COUNT,
                "trials": BIASED_TRIALS,
                "p_right": p_right,
                **estimates,
                **theory,
                "mean_absolute_error": abs(
                    estimates["mean_position"]
                    - theory["theoretical_mean_position"]
                ),
                "variance_absolute_error": abs(
                    estimates["variance"]
                    - theory["theoretical_variance"]
                ),
                "endpoint_origin_probability_error": abs(
                    estimates["endpoint_origin_probability"]
                    - theory["theoretical_endpoint_origin_probability"]
                ),
            }
        )

    return pd.DataFrame(rows)


METRIC_TO_THEORY_COLUMN = {
    "mean_position": "theoretical_mean_position",
    "variance": "theoretical_variance",
    "average_absolute_distance":
        "theoretical_average_absolute_distance",
    "endpoint_origin_probability":
        "theoretical_endpoint_origin_probability",
}


def run_convergence_raw_analysis() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    theory = theoretical_endpoint_metrics(
        CONVERGENCE_STEP_COUNT,
        CONVERGENCE_P_RIGHT,
    )

    for condition_index, trials in enumerate(CONVERGENCE_TRIAL_COUNTS):
        for repetition_index in range(CONVERGENCE_REPETITIONS):
            seed = condition_seed(
                ANALYSIS_ID_CONVERGENCE,
                condition_index,
                repetition_index,
            )
            final_positions = simulate_final_positions(
                steps=CONVERGENCE_STEP_COUNT,
                trials=trials,
                p_right=CONVERGENCE_P_RIGHT,
                seed=seed,
            )
            estimates = calculate_endpoint_estimators(final_positions)

            row = {
                "analysis": "finite_sample_convergence",
                "condition_index": condition_index,
                "repetition_index": repetition_index,
                "seed": seed,
                "steps": CONVERGENCE_STEP_COUNT,
                "trials": trials,
                "p_right": CONVERGENCE_P_RIGHT,
                **estimates,
                **theory,
            }

            for metric, theory_column in METRIC_TO_THEORY_COLUMN.items():
                error = estimates[metric] - theory[theory_column]
                row[f"{metric}_error"] = error
                row[f"{metric}_absolute_error"] = abs(error)
                row[f"{metric}_squared_error"] = error**2

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_convergence(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for trials, group in raw.groupby("trials", sort=True):
        for metric, theory_column in METRIC_TO_THEORY_COLUMN.items():
            estimates = group[metric].to_numpy(dtype=float)
            theory_value = float(group[theory_column].iloc[0])
            errors = estimates - theory_value

            mean_estimate = float(np.mean(estimates))
            bias = float(np.mean(errors))
            between_run_sd = float(np.std(estimates, ddof=1))
            repeated_run_se = between_run_sd / math.sqrt(len(estimates))
            margin = 1.96 * repeated_run_se
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors**2)))

            relative_rmse_percent = (
                100.0 * rmse / abs(theory_value)
                if theory_value != 0.0
                else np.nan
            )

            rows.append(
                {
                    "steps": int(group["steps"].iloc[0]),
                    "trials": int(trials),
                    "repetitions": int(len(estimates)),
                    "metric": metric,
                    "theoretical_value": theory_value,
                    "mean_estimate": mean_estimate,
                    "bias": bias,
                    "between_run_sd": between_run_sd,
                    "repeated_run_se": repeated_run_se,
                    "ci_lower": mean_estimate - margin,
                    "ci_upper": mean_estimate + margin,
                    "mean_absolute_error": mae,
                    "rmse": rmse,
                    "relative_rmse_percent": relative_rmse_percent,
                }
            )

    return pd.DataFrame(rows)


def estimate_convergence_rates(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for metric, group in summary.groupby("metric", sort=False):
        ordered = group.sort_values("trials")
        x = np.log(ordered["trials"].to_numpy(dtype=float))
        y = np.log(ordered["rmse"].to_numpy(dtype=float))

        slope, intercept = np.polyfit(x, y, deg=1)
        predicted = intercept + slope * x
        residual_sum_squares = float(np.sum((y - predicted) ** 2))
        total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )

        rows.append(
            {
                "metric": metric,
                "log_log_slope": float(slope),
                "log_log_intercept": float(intercept),
                "r_squared": r_squared,
                "expected_monte_carlo_slope": -0.5,
                "difference_from_expected_slope": float(slope + 0.5),
            }
        )

    return pd.DataFrame(rows)


def save_original_figures(
    symmetric: pd.DataFrame,
    biased: pd.DataFrame,
) -> None:
    figure_specs = [
        (
            symmetric,
            "steps",
            "mean_position",
            "Number of Steps",
            "Mean Final Position",
            "Steps vs Mean Final Position",
            "figure1_mean_position.png",
        ),
        (
            symmetric,
            "steps",
            "variance",
            "Number of Steps",
            "Variance of Final Position",
            "Steps vs Variance",
            "figure2_variance.png",
        ),
        (
            symmetric,
            "steps",
            "average_absolute_distance",
            "Number of Steps",
            "Average Absolute Distance from Origin",
            "Steps vs Average Absolute Distance",
            "figure3_average_distance.png",
        ),
        (
            symmetric,
            "steps",
            "endpoint_origin_probability",
            "Number of Steps",
            "Endpoint Origin Probability",
            "Steps vs Endpoint Origin Probability",
            "figure4_endpoint_origin_probability.png",
        ),
        (
            biased,
            "p_right",
            "mean_position",
            "Probability of Moving Right",
            "Mean Final Position",
            "Effect of Directional Bias on Mean Final Position",
            "figure5_biased_mean_position.png",
        ),
    ]

    for (
        frame,
        x_column,
        y_column,
        x_label,
        y_label,
        title,
        filename,
    ) in figure_specs:
        plt.figure()
        plt.plot(frame[x_column], frame[y_column], marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()


def save_convergence_figures(
    summary: pd.DataFrame,
    rates: pd.DataFrame,
) -> None:
    y_labels = {
        "mean_position": "RMSE of Mean Position",
        "variance": "RMSE of Variance",
        "average_absolute_distance":
            "RMSE of Average Absolute Distance",
        "endpoint_origin_probability":
            "RMSE of Endpoint Origin Probability",
    }

    filenames = {
        "mean_position": "figure6_convergence_mean_rmse.png",
        "variance": "figure7_convergence_variance_rmse.png",
        "average_absolute_distance":
            "figure8_convergence_absolute_distance_rmse.png",
        "endpoint_origin_probability":
            "figure9_convergence_endpoint_probability_rmse.png",
    }

    for metric in METRIC_TO_THEORY_COLUMN:
        selected = summary.loc[summary["metric"] == metric].sort_values("trials")
        rate_row = rates.loc[rates["metric"] == metric].iloc[0]
        slope = float(rate_row["log_log_slope"])
        intercept = float(rate_row["log_log_intercept"])

        trials = selected["trials"].to_numpy(dtype=float)
        rmse = selected["rmse"].to_numpy(dtype=float)
        fitted_rmse = np.exp(intercept) * trials**slope

        plt.figure()
        plt.loglog(trials, rmse, marker="o", label="Observed RMSE")
        plt.loglog(
            trials,
            fitted_rmse,
            linestyle="--",
            label=f"Log-log fit (slope = {slope:.3f})",
        )
        plt.xlabel("Number of Monte Carlo Trials")
        plt.ylabel(y_labels[metric])
        plt.title(
            "Finite-Sample Convergence of "
            + metric.replace("_", " ").title()
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / filenames[metric],
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def save_environment_files() -> None:
    environment_lines = [
        f"Python version: {sys.version.replace(chr(10), ' ')}",
        f"Platform: {platform.platform()}",
        f"NumPy version: {np.__version__}",
        f"pandas version: {pd.__version__}",
        f"Matplotlib version: {matplotlib.__version__}",
        "Random-number generator: numpy.random.default_rng",
        "Generator distribution call: numpy.random.Generator.binomial",
        "Variance convention: unbiased sample variance, ddof=1 (N-1 denominator)",
        f"Base seed: {BASE_SEED}",
        (
            "Seed formula: BASE_SEED + analysis_id*10,000,000 "
            "+ condition_index*10,000 + repetition_index"
        ),
    ]
    (OUTPUT_DIR / "environment_info.txt").write_text(
        "\n".join(environment_lines) + "\n",
        encoding="utf-8",
    )

    requirements = "\n".join(
        [
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"matplotlib=={matplotlib.__version__}",
        ]
    ) + "\n"
    (OUTPUT_DIR / "requirements.txt").write_text(
        requirements,
        encoding="utf-8",
    )


def save_run_manifest() -> None:
    manifest_lines = [
        "RUN MANIFEST",
        "",
        "1. Symmetric step-count comparison",
        f"   Step counts, in order: {SYMMETRIC_STEP_COUNTS}",
        f"   Trials per condition: {SYMMETRIC_TRIALS}",
        "   p_right: 0.50",
        "",
        "2. Biased probability comparison",
        f"   p_right values, in order: {BIASED_PROBABILITIES}",
        f"   Steps per condition: {BIASED_STEP_COUNT}",
        f"   Trials per condition: {BIASED_TRIALS}",
        "",
        "3. Finite-sample convergence analysis",
        f"   Trial counts, in order: {CONVERGENCE_TRIAL_COUNTS}",
        f"   Steps per walk: {CONVERGENCE_STEP_COUNT}",
        f"   Repetitions per trial count: {CONVERGENCE_REPETITIONS}",
        f"   p_right: {CONVERGENCE_P_RIGHT:.2f}",
        "",
        (
            "Every condition and repetition uses its own deterministic seed. "
            "Seeds are stored in the raw CSV files."
        ),
    ]
    (OUTPUT_DIR / "run_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    symmetric = run_symmetric_analysis()
    biased = run_biased_analysis()
    convergence_raw = run_convergence_raw_analysis()
    convergence_summary = summarize_convergence(convergence_raw)
    convergence_rates = estimate_convergence_rates(convergence_summary)

    symmetric.to_csv(
        OUTPUT_DIR / "symmetric_step_analysis.csv",
        index=False,
    )
    biased.to_csv(
        OUTPUT_DIR / "biased_probability_analysis.csv",
        index=False,
    )
    convergence_raw.to_csv(
        OUTPUT_DIR / "convergence_raw_results.csv",
        index=False,
    )
    convergence_summary.to_csv(
        OUTPUT_DIR / "convergence_summary.csv",
        index=False,
    )
    convergence_rates.to_csv(
        OUTPUT_DIR / "convergence_rate_estimates.csv",
        index=False,
    )

    save_original_figures(symmetric, biased)
    save_convergence_figures(convergence_summary, convergence_rates)
    save_environment_files()
    save_run_manifest()

    print("Analysis completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("\nEstimated convergence rates:")
    print(
        convergence_rates[
            [
                "metric",
                "log_log_slope",
                "r_squared",
                "expected_monte_carlo_slope",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

