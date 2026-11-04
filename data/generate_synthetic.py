"""Synthetic data generation for both experiments.

Experiment 1 (contamination simulation, paper Section 6) draws its data
directly from the model described in the paper, so ``generate_contamination_sample``
is simply the sampling procedure of Section 6, not a substitute for
unavailable data.

Experiment 2 (empirical application, paper Section 7) analyzes a proprietary
Amazon A/B-testing dataset that the paper does not release and that cannot be
obtained from any market-data API: it is not price or OHLCV data, but
internal experimentation-platform telemetry (estimated treatment impacts and
their sampling variances). ``generate_synthetic_ab_tests`` builds a
dataset with the structure the paper describes (four metrics, six half-year
periods) and injects a small number of extreme observations in a few cells,
so that the full MLE/MAP/BIMLE pipeline, bootstrap confidence intervals, and
stability diagnostics can be exercised end-to-end. It is not a reproduction
of the paper's reported figures; see "Deviations from the paper" in
results/RESULTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def generate_contamination_sample(
    n_tests: int,
    true_mu: float,
    true_sigma2: float,
    tau_sq: float,
    k_contaminated: int,
    extreme_value: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One contamination-experiment sample, paper Section 6.

    Draws Delta_hat_i ~ N(true_mu, true_sigma2 + tau_sq) for n_tests tests
    with known sampling variance tau_sq, then replaces k_contaminated randomly
    selected observations with extreme_value (paper: "replace k ... with
    Delta_hat_i = 50"). Random selection of the contaminated tests is used
    because the paper leaves the selection rule for which tests are
    contaminated unspecified while describing an i.i.d. contamination
    experiment.

    Args:
        n_tests: number of simulated tests N.
        true_mu: true prior mean used to draw Delta_hat_i.
        true_sigma2: true prior variance used to draw Delta_hat_i.
        tau_sq: common known sampling variance tau_i^2 for every test.
        k_contaminated: number of tests whose Delta_hat_i is replaced with extreme_value.
        extreme_value: the contaminated Delta_hat_i value.
        rng: numpy random Generator for reproducibility.

    Returns:
        DataFrame with columns ["delta_hat", "tau_sq"] of length n_tests.
    """
    if n_tests <= 0:
        raise ValueError("n_tests must be positive")
    if not 0 <= k_contaminated <= n_tests:
        raise ValueError("k_contaminated must be between 0 and n_tests")
    total_sd = np.sqrt(true_sigma2 + tau_sq)
    delta_hat = true_mu + total_sd * rng.standard_normal(n_tests)
    if k_contaminated > 0:
        contaminated_idx = rng.choice(n_tests, size=k_contaminated, replace=False)
        delta_hat[contaminated_idx] = extreme_value
    tau_sq_arr = np.full(n_tests, tau_sq)
    return pd.DataFrame({"delta_hat": delta_hat, "tau_sq": tau_sq_arr})


def generate_synthetic_ab_tests(seed: int = config.SYN_SEED) -> pd.DataFrame:
    """Synthetic analogue of the Amazon A/B-testing panel described in Section 7.

    Builds one row per synthetic test, with columns ["metric", "half_year",
    "delta_hat", "tau_sq"], matching the generic panel structure of Section 7
    (per-test estimated treatment impact and sampling variance, grouped by
    metric and half-year period). Each (metric, half_year) cell draws
    SYN_TESTS_PER_PERIOD tests from a normal-normal
    model with metric-specific (mu, sigma) and heterogeneous sampling
    variances, then injects extreme observations in the cells listed in
    SYN_CONTAMINATED_PERIODS, mimicking the paper's qualitative description of
    a small number of anomalous tests in some but not all periods.

    Args:
        seed: random seed controlling every draw.

    Returns:
        DataFrame with columns ["metric", "half_year", "delta_hat", "tau_sq"].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for metric in config.EXP2_METRICS:
        true_mu = config.SYN_TRUE_MU[metric]
        true_sigma = config.SYN_TRUE_SIGMA[metric]
        for half_year in config.EXP2_HALF_YEARS:
            n = config.SYN_TESTS_PER_PERIOD
            tau = rng.uniform(*config.SYN_TAU_SD_RANGE, size=n)
            tau_sq = tau**2
            delta_hat = true_mu + np.sqrt(true_sigma**2 + tau_sq) * rng.standard_normal(n)
            k = config.SYN_CONTAMINATED_PERIODS.get((metric, half_year), 0)
            if k > 0:
                contaminated_idx = rng.choice(n, size=k, replace=False)
                sign = rng.choice([-1.0, 1.0], size=k)
                delta_hat[contaminated_idx] = sign * config.SYN_EXTREME_MULTIPLIER * tau[contaminated_idx]
            rows.append(pd.DataFrame({"metric": metric, "half_year": half_year, "delta_hat": delta_hat, "tau_sq": tau_sq}))
    return pd.concat(rows, ignore_index=True)
