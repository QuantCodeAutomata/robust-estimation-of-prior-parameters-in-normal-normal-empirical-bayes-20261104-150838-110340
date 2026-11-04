"""Unit tests for src/bootstrap.py."""
import numpy as np
import pytest

from src.bootstrap import bootstrap_ci
from src.eb_model import fit_mle


def _mle_estimator(delta_hat, tau_sq):
    result = fit_mle(delta_hat, tau_sq)
    return result.mu, result.sigma2


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(42)
    delta_hat = rng.standard_normal(500) * np.sqrt(2.0)
    tau_sq = np.ones(500)
    point = _mle_estimator(delta_hat, tau_sq)
    ci = bootstrap_ci(_mle_estimator, delta_hat, tau_sq, n_boot=200, ci_level=0.95, seed=1)
    lower_mu, upper_mu = ci["mu"]
    lower_sigma2, upper_sigma2 = ci["sigma2"]
    assert lower_mu <= point[0] <= upper_mu
    assert lower_sigma2 <= point[1] <= upper_sigma2


def test_bootstrap_ci_widens_with_more_variable_data():
    rng = np.random.default_rng(7)
    small_n = rng.standard_normal(50) * np.sqrt(2.0)
    large_n = rng.standard_normal(2000) * np.sqrt(2.0)
    tau_small = np.ones(50)
    tau_large = np.ones(2000)
    ci_small = bootstrap_ci(_mle_estimator, small_n, tau_small, n_boot=200, ci_level=0.95, seed=2)
    ci_large = bootstrap_ci(_mle_estimator, large_n, tau_large, n_boot=200, ci_level=0.95, seed=2)
    width_small = ci_small["sigma2"][1] - ci_small["sigma2"][0]
    width_large = ci_large["sigma2"][1] - ci_large["sigma2"][0]
    assert width_small > width_large


def test_bootstrap_ci_rejects_invalid_ci_level():
    with pytest.raises(ValueError):
        bootstrap_ci(_mle_estimator, np.ones(5), np.ones(5), n_boot=10, ci_level=1.5, seed=0)


def test_bootstrap_ci_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        bootstrap_ci(_mle_estimator, np.ones(5), np.ones(4), n_boot=10, ci_level=0.95, seed=0)
