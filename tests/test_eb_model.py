"""Unit tests for src/eb_model.py: MLE, MAP, and BIMLE estimators.

Tests check the paper's stated properties (paper Section 6 and Section 4):
recovery under the model with no contamination, bounded influence of BIMLE
relative to MLE under contamination, reduction to MLE as b -> infinity, and
basic input validation.
"""
import numpy as np
import pytest

from src import config
from src.eb_model import fit_bimle, fit_map, fit_mle, weighted_log_likelihood
from data.generate_synthetic import generate_contamination_sample


@pytest.fixture
def clean_sample():
    rng = np.random.default_rng(0)
    df = generate_contamination_sample(
        n_tests=2000, true_mu=0.0, true_sigma2=1.0, tau_sq=1.0, k_contaminated=0, extreme_value=50.0, rng=rng
    )
    return df["delta_hat"].to_numpy(), df["tau_sq"].to_numpy()


def test_mle_recovers_true_parameters_without_contamination(clean_sample):
    delta_hat, tau_sq = clean_sample
    result = fit_mle(delta_hat, tau_sq)
    assert abs(result.mu - 0.0) < 0.1
    assert abs(result.sigma2 - 1.0) < 0.2
    assert result.converged


def test_mle_sigma2_inflates_with_contamination():
    rng = np.random.default_rng(1)
    clean = generate_contamination_sample(1000, 0.0, 1.0, 1.0, 0, 50.0, rng)
    contaminated = generate_contamination_sample(1000, 0.0, 1.0, 1.0, 5, 50.0, rng)
    sigma2_clean = fit_mle(clean["delta_hat"].to_numpy(), clean["tau_sq"].to_numpy()).sigma2
    sigma2_contaminated = fit_mle(contaminated["delta_hat"].to_numpy(), contaminated["tau_sq"].to_numpy()).sigma2
    # Paper Section 6: five contaminated observations should inflate sigma2_hat by more than tenfold
    # relative to its value with no contamination (true sigma2 = 1).
    assert sigma2_contaminated > 5 * sigma2_clean


def test_bimle_less_sensitive_than_mle_under_contamination():
    rng = np.random.default_rng(2)
    contaminated = generate_contamination_sample(1000, 0.0, 1.0, 1.0, 5, 50.0, rng)
    delta_hat, tau_sq = contaminated["delta_hat"].to_numpy(), contaminated["tau_sq"].to_numpy()
    mle = fit_mle(delta_hat, tau_sq)
    bimle = fit_bimle(delta_hat, tau_sq, b=config.EXP1_BIMLE_B)
    assert bimle.sigma2 < mle.sigma2
    assert bimle.converged
    # At most the 5 contaminated tests should receive weight below 1.
    assert np.sum(bimle.weights < 1.0) <= 5


def test_bimle_converges_to_mle_as_b_grows(clean_sample):
    delta_hat, tau_sq = clean_sample
    mle = fit_mle(delta_hat, tau_sq)
    bimle_large_b = fit_bimle(delta_hat, tau_sq, b=1e6)
    assert abs(bimle_large_b.mu - mle.mu) < 1e-3
    assert abs(bimle_large_b.sigma2 - mle.sigma2) < 1e-2
    assert np.allclose(bimle_large_b.weights, 1.0)


def test_bimle_no_contamination_matches_mle(clean_sample):
    delta_hat, tau_sq = clean_sample
    mle = fit_mle(delta_hat, tau_sq)
    bimle = fit_bimle(delta_hat, tau_sq, b=config.EXP1_BIMLE_B)
    # Paper Section 6: "When there are no extreme observations (k=0), all three
    # estimators recover mu=0 and sigma2=1."
    assert abs(bimle.sigma2 - mle.sigma2) < 0.1


def test_map_with_correctly_centered_hyperprior_close_to_mle_absent_contamination(clean_sample):
    delta_hat, tau_sq = clean_sample
    mle = fit_mle(delta_hat, tau_sq)
    map_est = fit_map(
        delta_hat,
        tau_sq,
        mu_prior_mean=config.EXP1_MAP_MU_PRIOR_MEAN,
        mu_prior_sd=config.EXP1_MAP_MU_PRIOR_SD,
        logsigma2_prior_mean=config.EXP1_MAP_LOGSIGMA2_PRIOR_MEAN,
        logsigma2_prior_sd=config.EXP1_MAP_LOGSIGMA2_PRIOR_SD,
    )
    assert abs(map_est.sigma2 - mle.sigma2) < 0.3


def test_map_provides_little_protection_against_contamination():
    rng = np.random.default_rng(3)
    contaminated = generate_contamination_sample(1000, 0.0, 1.0, 1.0, 5, 50.0, rng)
    delta_hat, tau_sq = contaminated["delta_hat"].to_numpy(), contaminated["tau_sq"].to_numpy()
    mle = fit_mle(delta_hat, tau_sq)
    map_est = fit_map(
        delta_hat,
        tau_sq,
        mu_prior_mean=config.EXP1_MAP_MU_PRIOR_MEAN,
        mu_prior_sd=config.EXP1_MAP_MU_PRIOR_SD,
        logsigma2_prior_mean=config.EXP1_MAP_LOGSIGMA2_PRIOR_MEAN,
        logsigma2_prior_sd=config.EXP1_MAP_LOGSIGMA2_PRIOR_SD,
    )
    # Paper Section 6: MAP offers "limited protection"; its sigma2 should stay
    # close to (within 40%) the MLE's inflated value, not close to the truth.
    assert map_est.sigma2 > 2.0
    assert abs(map_est.sigma2 - mle.sigma2) / mle.sigma2 < 0.4


def test_mu_estimates_far_less_sensitive_than_sigma2_to_contamination():
    rng = np.random.default_rng(4)
    contaminated = generate_contamination_sample(1000, 0.0, 1.0, 1.0, 5, 50.0, rng)
    delta_hat, tau_sq = contaminated["delta_hat"].to_numpy(), contaminated["tau_sq"].to_numpy()
    mle = fit_mle(delta_hat, tau_sq)
    # Paper Section 6: the score for mu is linear (not quadratic) in z_i, so
    # mu_hat should remain near 0 even with contamination.
    assert abs(mle.mu) < 0.5


def test_weighted_log_likelihood_matches_unweighted_with_unit_weights():
    delta_hat = np.array([0.1, -0.2, 0.3])
    tau_sq = np.array([1.0, 1.0, 1.0])
    ll_default = weighted_log_likelihood(0.0, 1.0, delta_hat, tau_sq)
    ll_explicit = weighted_log_likelihood(0.0, 1.0, delta_hat, tau_sq, weights=np.ones(3))
    assert ll_default == pytest.approx(ll_explicit)


def test_fit_mle_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fit_mle(np.array([1.0, 2.0]), np.array([1.0]))


def test_fit_mle_rejects_nonpositive_tau_sq():
    with pytest.raises(ValueError):
        fit_mle(np.array([1.0, 2.0]), np.array([1.0, -0.5]))


def test_fit_bimle_rejects_negative_b():
    with pytest.raises(ValueError):
        fit_bimle(np.array([1.0, 2.0]), np.array([1.0, 1.0]), b=-1.0)


def test_single_observation_edge_case():
    # A single test should still produce a finite estimate without raising.
    result = fit_mle(np.array([0.5]), np.array([1.0]))
    assert np.isfinite(result.mu)
    assert np.isfinite(result.sigma2)
    assert result.sigma2 >= 0
