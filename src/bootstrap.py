"""Nonparametric bootstrap confidence intervals for the prior-parameter
estimators, paper Table 1 notes: "Brackets show 95% confidence intervals
from a nonparametric bootstrap (200 resamples of the tests within each
half-year window)."

``scipy.stats.bootstrap`` assumes a scalar statistic computed independently
per resample from a single sample array; the paper's estimators take two
paired arrays (delta_hat, tau_sq) and return a 2-vector (mu, sigma^2), and
BIMLE additionally needs the b parameter threaded through. Implementing the
resampling loop directly with ``numpy.random.Generator.integers`` is simpler
and more transparent than adapting scipy's generic interface to this
paired, vector-valued statistic, so this module is a thin, custom wrapper
around the estimators in ``eb_model.py``, not a reimplementation of a
statistical primitive.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    estimator_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    delta_hat: np.ndarray,
    tau_sq: np.ndarray,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Nonparametric bootstrap percentile confidence intervals for (mu, sigma^2).

    Resamples the N paired observations (delta_hat_i, tau_i^2) with
    replacement n_boot times, applies ``estimator_fn`` to each resample, and
    reports the percentile interval implied by ``ci_level`` for mu and
    sigma^2 separately.

    Args:
        estimator_fn: a function mapping (delta_hat, tau_sq) to (mu_hat, sigma2_hat),
            e.g. a closure around fit_mle/fit_map/fit_bimle.
        delta_hat: array of estimated treatment impacts for one metric/period.
        tau_sq: array of known sampling variances for the same tests.
        n_boot: number of bootstrap resamples.
        ci_level: confidence level in (0, 1), e.g. 0.95.
        seed: random seed for the resampling generator.

    Returns:
        A dict with keys "mu" and "sigma2", each mapping to a (lower, upper)
        percentile interval.
    """
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    tau_sq = np.asarray(tau_sq, dtype=np.float64)
    if delta_hat.shape != tau_sq.shape or delta_hat.size == 0:
        raise ValueError("delta_hat and tau_sq must be non-empty arrays of equal length")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be strictly between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    rng = np.random.default_rng(seed)
    n = delta_hat.size
    mu_draws = np.empty(n_boot)
    sigma2_draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mu_hat, sigma2_hat = estimator_fn(delta_hat[idx], tau_sq[idx])
        mu_draws[b] = mu_hat
        sigma2_draws[b] = sigma2_hat

    alpha = 1.0 - ci_level
    lower_pct, upper_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mu": (float(np.percentile(mu_draws, lower_pct)), float(np.percentile(mu_draws, upper_pct))),
        "sigma2": (float(np.percentile(sigma2_draws, lower_pct)), float(np.percentile(sigma2_draws, upper_pct))),
    }
