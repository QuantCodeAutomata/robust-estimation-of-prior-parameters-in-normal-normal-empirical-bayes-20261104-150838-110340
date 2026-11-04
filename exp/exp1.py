"""Experiment 1: Illustrative Simulations -- Sensitivity to Extreme Observations.

Reproduces the contamination simulation of Burson and Kessler (2026),
"Robust Estimation of Prior Parameters in Normal-Normal Empirical Bayes",
Section 6: for each number of contaminated observations k in {0, ..., 5},
simulate N = 1000 tests from the normal-normal model, replace k of their
Delta_hat_i with the extreme value 50, and estimate (mu, sigma^2) with MLE,
MAP, and BIMLE. The whole exercise is repeated 100 times per k, and the table
reports the average estimate across replications for each method.

Run as: python -m exp.exp1
"""
from __future__ import annotations

import logging
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import generate_contamination_sample
from src import config
from src.eb_model import fit_bimle, fit_map, fit_mle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_one_replication(k: int, rng: np.random.Generator) -> dict[str, float]:
    """One Monte Carlo replication for a given k (paper Section 6, steps 1-5).

    Args:
        k: number of contaminated observations.
        rng: numpy random Generator shared across the replication.

    Returns:
        Dict with mu/sigma2 estimates for mle, map, and bimle.
    """
    sample = generate_contamination_sample(
        n_tests=config.EXP1_N_TESTS,
        true_mu=config.EXP1_TRUE_MU,
        true_sigma2=config.EXP1_TRUE_SIGMA2,
        tau_sq=config.EXP1_TAU_SQ,
        k_contaminated=k,
        extreme_value=config.EXP1_EXTREME_VALUE,
        rng=rng,
    )
    delta_hat = sample["delta_hat"].to_numpy()
    tau_sq = sample["tau_sq"].to_numpy()

    mle = fit_mle(delta_hat, tau_sq)
    map_est = fit_map(
        delta_hat,
        tau_sq,
        mu_prior_mean=config.EXP1_MAP_MU_PRIOR_MEAN,
        mu_prior_sd=config.EXP1_MAP_MU_PRIOR_SD,
        logsigma2_prior_mean=config.EXP1_MAP_LOGSIGMA2_PRIOR_MEAN,
        logsigma2_prior_sd=config.EXP1_MAP_LOGSIGMA2_PRIOR_SD,
    )
    bimle = fit_bimle(delta_hat, tau_sq, b=config.EXP1_BIMLE_B)

    return {
        "mle_mu": mle.mu, "mle_sigma2": mle.sigma2,
        "map_mu": map_est.mu, "map_sigma2": map_est.sigma2,
        "bimle_mu": bimle.mu, "bimle_sigma2": bimle.sigma2,
    }


def run_simulation(
    k_values: list[int] = config.EXP1_K_VALUES,
    n_replications: int = config.EXP1_N_REPLICATIONS,
    seed: int = config.EXP1_SEED,
) -> pd.DataFrame:
    """Run the full contamination simulation of paper Section 6.

    Args:
        k_values: numbers of contaminated observations to evaluate.
        n_replications: Monte Carlo replications per k.
        seed: base random seed (one independent stream is derived per k).

    Returns:
        DataFrame with one row per k, averaging mu and sigma^2 estimates
        across replications for MLE, MAP, and BIMLE.
    """
    rows = []
    for k in k_values:
        rng = np.random.default_rng(seed + k)
        replications = [run_one_replication(k, rng) for _ in range(n_replications)]
        rep_df = pd.DataFrame(replications)
        row = {"k": k}
        for col in rep_df.columns:
            row[col] = rep_df[col].mean()
        rows.append(row)
        logger.info(
            "k=%d: MLE sigma2=%.3f, MAP sigma2=%.3f, BIMLE sigma2=%.3f",
            k, row["mle_sigma2"], row["map_sigma2"], row["bimle_sigma2"],
        )
    return pd.DataFrame(rows)


def make_plots(results: pd.DataFrame, output_dir: str) -> None:
    """Reproduce paper Figure 1: sensitivity of mu_hat and sigma2_hat to k."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for method, label in [("mle", "MLE"), ("map", "MAP"), ("bimle", "BIMLE")]:
        axes[0].plot(results["k"], results[f"{method}_mu"], marker="o", label=label)
        axes[1].plot(results["k"], results[f"{method}_sigma2"], marker="o", label=label)

    axes[0].set_title("Estimated prior mean $\\hat{\\mu}$")
    axes[0].set_xlabel("Number of contaminated observations (k)")
    axes[0].set_ylabel("$\\hat{\\mu}$")
    axes[0].axhline(config.EXP1_TRUE_MU, color="gray", linestyle="--", linewidth=0.8, label="true $\\mu$")
    axes[0].legend()

    axes[1].set_title("Estimated prior variance $\\hat{\\sigma}^2$")
    axes[1].set_xlabel("Number of contaminated observations (k)")
    axes[1].set_ylabel("$\\hat{\\sigma}^2$")
    axes[1].axhline(config.EXP1_TRUE_SIGMA2, color="gray", linestyle="--", linewidth=0.8, label="true $\\sigma^2$")
    axes[1].legend()

    fig.suptitle("Sensitivity to extreme observations: MLE vs. MAP vs. BIMLE (paper Figure 1)")
    fig.tight_layout()
    path = os.path.join(output_dir, "exp1_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot to %s", path)


def main() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    start = time.time()
    logger.info(
        "Running Experiment 1: N=%d tests, k in %s, %d replications per k, BIMLE b=%.1f",
        config.EXP1_N_TESTS, config.EXP1_K_VALUES, config.EXP1_N_REPLICATIONS, config.EXP1_BIMLE_B,
    )
    results = run_simulation()
    elapsed = time.time() - start
    logger.info("Simulation finished in %.1f seconds", elapsed)

    csv_path = os.path.join(config.RESULTS_DIR, "exp1_results.csv")
    results.to_csv(csv_path, index=False)
    logger.info("Saved results table to %s", csv_path)

    make_plots(results, config.RESULTS_DIR)

    comparison_rows = []
    for method in ["mle", "map", "bimle"]:
        for i, k in enumerate(config.EXP1_K_VALUES):
            comparison_rows.append({
                "method": method.upper(),
                "k": k,
                "sigma2_hat": results.loc[results["k"] == k, f"{method}_sigma2"].iloc[0],
                "sigma2_expected_paper": config.EXP1_EXPECTED_SIGMA2[method.upper()][i],
                "mu_hat": results.loc[results["k"] == k, f"{method}_mu"].iloc[0],
                "mu_expected_paper": config.EXP1_EXPECTED_MU[method.upper()][i],
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(config.RESULTS_DIR, "exp1_comparison_to_paper.csv")
    comparison.to_csv(comparison_path, index=False)
    logger.info("Saved paper comparison table to %s", comparison_path)


if __name__ == "__main__":
    main()
