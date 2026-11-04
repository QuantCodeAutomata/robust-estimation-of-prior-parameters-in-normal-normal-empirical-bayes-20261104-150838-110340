"""Experiment 2: Empirical Application -- Amazon A/B Tests.

Reproduces the estimation pipeline of Burson and Kessler (2026), Section 7:
for each of four illustrative metrics and six half-year periods, estimate
the prior mean and variance (mu, sigma^2) with MLE, MAP, and BIMLE, attach
95% nonparametric bootstrap confidence intervals, and summarize stability
across periods with the coefficient of variation (CV). For BIMLE, also
report the share of downweighted observations (DW) and their average weight
(w | DW), and run a sensitivity analysis over the tuning constant
b in {10, 15, 20} (paper Tables 3-4).

The underlying Amazon A/B-testing dataset is proprietary and not released
with the paper; see data/generate_synthetic.py and
"Deviations from the paper" in results/RESULTS.md. This module accepts any
DataFrame with columns ["metric", "half_year", "delta_hat", "tau_sq"],
so it runs unchanged on the real dataset if one becomes available.

Run as: python -m exp.exp2
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import generate_synthetic_ab_tests
from src import config
from src.bootstrap import bootstrap_ci
from src.eb_model import fit_bimle, fit_map, fit_mle
from src.stability_metrics import coefficient_of_variation, downweight_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def estimate_period(
    delta_hat: np.ndarray, tau_sq: np.ndarray, bimle_b: float
) -> dict[str, object]:
    """Fit MLE, MAP, and BIMLE on one (metric, half_year) cell.

    Args:
        delta_hat: estimated treatment impacts for the tests in this cell.
        tau_sq: known sampling variances for the same tests.
        bimle_b: BIMLE tuning constant.

    Returns:
        Dict of point estimates, bootstrap confidence intervals, and BIMLE
        downweighting diagnostics for this cell.
    """
    mle = fit_mle(delta_hat, tau_sq)
    map_est = fit_map(
        delta_hat,
        tau_sq,
        mu_prior_mean=config.EXP2_MAP_MU_PRIOR_MEAN,
        mu_prior_sd=np.sqrt(config.EXP2_MAP_MU_PRIOR_VAR),
        logsigma2_prior_mean=config.EXP2_MAP_LOGSIGMA2_PRIOR_MEAN,
        logsigma2_prior_sd=config.EXP2_MAP_LOGSIGMA2_PRIOR_SD,
    )
    bimle = fit_bimle(delta_hat, tau_sq, b=bimle_b)
    dw_pct, avg_w_dw = downweight_stats(bimle.weights)

    def _mle_fn(d, t):
        return fit_mle(d, t).mu, fit_mle(d, t).sigma2

    def _map_fn(d, t):
        r = fit_map(
            d, t,
            mu_prior_mean=config.EXP2_MAP_MU_PRIOR_MEAN,
            mu_prior_sd=np.sqrt(config.EXP2_MAP_MU_PRIOR_VAR),
            logsigma2_prior_mean=config.EXP2_MAP_LOGSIGMA2_PRIOR_MEAN,
            logsigma2_prior_sd=config.EXP2_MAP_LOGSIGMA2_PRIOR_SD,
        )
        return r.mu, r.sigma2

    def _bimle_fn(d, t):
        r = fit_bimle(d, t, b=bimle_b)
        return r.mu, r.sigma2

    ci_mle = bootstrap_ci(_mle_fn, delta_hat, tau_sq, config.EXP2_N_BOOTSTRAP, config.EXP2_CI_LEVEL, config.EXP2_BOOTSTRAP_SEED)
    ci_map = bootstrap_ci(_map_fn, delta_hat, tau_sq, config.EXP2_N_BOOTSTRAP, config.EXP2_CI_LEVEL, config.EXP2_BOOTSTRAP_SEED)
    ci_bimle = bootstrap_ci(_bimle_fn, delta_hat, tau_sq, config.EXP2_N_BOOTSTRAP, config.EXP2_CI_LEVEL, config.EXP2_BOOTSTRAP_SEED)

    return {
        "n_tests": delta_hat.size,
        "mle_mu": mle.mu, "mle_sigma2": mle.sigma2,
        "mle_mu_ci_lo": ci_mle["mu"][0], "mle_mu_ci_hi": ci_mle["mu"][1],
        "mle_sigma2_ci_lo": ci_mle["sigma2"][0], "mle_sigma2_ci_hi": ci_mle["sigma2"][1],
        "map_mu": map_est.mu, "map_sigma2": map_est.sigma2,
        "map_mu_ci_lo": ci_map["mu"][0], "map_mu_ci_hi": ci_map["mu"][1],
        "map_sigma2_ci_lo": ci_map["sigma2"][0], "map_sigma2_ci_hi": ci_map["sigma2"][1],
        "bimle_mu": bimle.mu, "bimle_sigma2": bimle.sigma2,
        "bimle_mu_ci_lo": ci_bimle["mu"][0], "bimle_mu_ci_hi": ci_bimle["mu"][1],
        "bimle_sigma2_ci_lo": ci_bimle["sigma2"][0], "bimle_sigma2_ci_hi": ci_bimle["sigma2"][1],
        "bimle_dw_pct": dw_pct, "bimle_avg_weight_downweighted": avg_w_dw,
    }


def run_panel_estimation(data: pd.DataFrame, bimle_b: float = config.EXP2_BIMLE_B_MAIN) -> pd.DataFrame:
    """Estimate (mu, sigma^2) for every (metric, half_year) cell.

    Args:
        data: DataFrame with columns ["metric", "half_year", "delta_hat", "tau_sq"].
        bimle_b: BIMLE tuning constant used for the main table (paper Table 1).

    Returns:
        DataFrame with one row per (metric, half_year), point estimates,
        bootstrap confidence intervals, and BIMLE downweighting diagnostics.
    """
    required_cols = {"metric", "half_year", "delta_hat", "tau_sq"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"data must contain columns {required_cols}, got {list(data.columns)}")

    rows = []
    for (metric, half_year), group in data.groupby(["metric", "half_year"], sort=False):
        delta_hat = group["delta_hat"].to_numpy()
        tau_sq = group["tau_sq"].to_numpy()
        stats = estimate_period(delta_hat, tau_sq, bimle_b)
        stats.update({"metric": metric, "half_year": half_year})
        rows.append(stats)
        logger.info(
            "%s %s (n=%d): MLE sigma2=%.6g, MAP sigma2=%.6g, BIMLE sigma2=%.6g, DW=%.3f%%",
            metric, half_year, stats["n_tests"], stats["mle_sigma2"], stats["map_sigma2"], stats["bimle_sigma2"], stats["bimle_dw_pct"],
        )
    return pd.DataFrame(rows)


def summarize_stability(panel: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation and downweighting summary per metric/method.

    Args:
        panel: output of ``run_panel_estimation``, with one row per
            (metric, half_year).

    Returns:
        DataFrame with one row per metric, reporting the CV of sigma_hat =
        sqrt(sigma2_hat) across half-year periods for MLE, MAP, and BIMLE,
        and BIMLE's average DW / w|DW across periods (paper Table 1).
    """
    rows = []
    for metric, group in panel.groupby("metric", sort=False):
        row = {"metric": metric}
        for method in ["mle", "map", "bimle"]:
            sigma_hat = np.sqrt(group[f"{method}_sigma2"].to_numpy())
            row[f"{method}_sigma_cv"] = coefficient_of_variation(sigma_hat)
            row[f"{method}_mu_cv"] = coefficient_of_variation(group[f"{method}_mu"].to_numpy())
        row["bimle_dw_pct_avg"] = group["bimle_dw_pct"].mean()
        row["bimle_avg_weight_downweighted_avg"] = group["bimle_avg_weight_downweighted"].mean(skipna=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _bimle_point_estimates(data: pd.DataFrame, bimle_b: float) -> pd.DataFrame:
    """BIMLE point estimates and downweighting diagnostics per cell, no bootstrap.

    Used by ``sensitivity_analysis``, which only needs point estimates across
    the b grid (paper Tables 3-4 report Mean/CV/DW, not confidence intervals).
    """
    rows = []
    for (metric, half_year), group in data.groupby(["metric", "half_year"], sort=False):
        delta_hat = group["delta_hat"].to_numpy()
        tau_sq = group["tau_sq"].to_numpy()
        bimle = fit_bimle(delta_hat, tau_sq, b=bimle_b)
        dw_pct, avg_w_dw = downweight_stats(bimle.weights)
        rows.append({
            "metric": metric, "half_year": half_year,
            "bimle_mu": bimle.mu, "bimle_sigma2": bimle.sigma2,
            "bimle_dw_pct": dw_pct, "bimle_avg_weight_downweighted": avg_w_dw,
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(data: pd.DataFrame, b_values: list[float] = config.EXP2_BIMLE_B_SENSITIVITY) -> pd.DataFrame:
    """BIMLE sensitivity to the tuning constant b (paper Tables 3-4).

    Args:
        data: DataFrame with columns ["metric", "half_year", "delta_hat", "tau_sq"].
        b_values: grid of tuning constants to evaluate.

    Returns:
        DataFrame with one row per (metric, b), reporting the mean, CV, and
        DW of sigma_hat and mu_hat across half-year periods.
    """
    rows = []
    for b in b_values:
        panel_b = _bimle_point_estimates(data, bimle_b=b)
        for metric, group in panel_b.groupby("metric", sort=False):
            sigma_hat = np.sqrt(group["bimle_sigma2"].to_numpy())
            mu_hat = group["bimle_mu"].to_numpy()
            rows.append({
                "metric": metric,
                "b": b,
                "sigma_hat_mean": float(np.mean(sigma_hat)),
                "sigma_hat_cv": coefficient_of_variation(sigma_hat),
                "mu_hat_mean": float(np.mean(mu_hat)),
                "mu_hat_cv": coefficient_of_variation(mu_hat),
                "dw_pct_mean": float(group["bimle_dw_pct"].mean()),
            })
    return pd.DataFrame(rows)


def make_plots(panel: pd.DataFrame, output_dir: str) -> None:
    """Plot sigma_hat across half-year periods per metric and method (paper Section 7)."""
    metrics = panel["metric"].unique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = panel[panel["metric"] == metric].sort_values("half_year")
        for method, label in [("mle", "MLE"), ("map", "MAP"), ("bimle", "BIMLE")]:
            ax.plot(sub["half_year"], np.sqrt(sub[f"{method}_sigma2"]), marker="o", label=label)
        ax.set_title(metric)
        ax.set_xlabel("Half-year period")
        ax.set_ylabel("$\\hat{\\sigma}$")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("Stability of the estimated prior standard deviation across half-years (paper Section 7)")
    fig.tight_layout()
    path = os.path.join(output_dir, "exp2_stability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot to %s", path)


def main(full: bool = False) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logger.info(
        "Amazon A/B-testing data is proprietary and not released with the paper; "
        "running the pipeline on synthetic data generated in data/generate_synthetic.py "
        "with the paper's described panel structure (%d metrics x %d half-years).",
        len(config.EXP2_METRICS), len(config.EXP2_HALF_YEARS),
    )
    data = generate_synthetic_ab_tests(seed=config.SYN_SEED)
    logger.info("Synthetic panel: %d rows, metrics=%s, half-years=%s", len(data), data["metric"].unique().tolist(), data["half_year"].unique().tolist())

    panel = run_panel_estimation(data, bimle_b=config.EXP2_BIMLE_B_MAIN)
    panel_path = os.path.join(config.RESULTS_DIR, "exp2_panel_estimates.csv")
    panel.to_csv(panel_path, index=False)
    logger.info("Saved per-period estimates to %s", panel_path)

    stability = summarize_stability(panel)
    stability_path = os.path.join(config.RESULTS_DIR, "exp2_stability_summary.csv")
    stability.to_csv(stability_path, index=False)
    logger.info("Saved stability summary to %s", stability_path)

    b_values = config.EXP2_BIMLE_B_SENSITIVITY if full else config.EXP2_BIMLE_B_SENSITIVITY
    sensitivity = sensitivity_analysis(data, b_values=b_values)
    sensitivity_path = os.path.join(config.RESULTS_DIR, "exp2_bimle_sensitivity.csv")
    sensitivity.to_csv(sensitivity_path, index=False)
    logger.info("Saved BIMLE sensitivity table to %s", sensitivity_path)

    make_plots(panel, config.RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2: Empirical Application to A/B tests")
    parser.add_argument("--full", action="store_true", help="Run the full paper configuration (sensitivity grid over b).")
    args = parser.parse_args()
    main(full=args.full)
