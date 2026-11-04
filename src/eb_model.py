"""Normal-normal empirical Bayes prior-parameter estimators.

Implements the three estimators compared in Burson and Kessler (2026),
"Robust Estimation of Prior Parameters in Normal-Normal Empirical Bayes":

* MLE  -- maximum likelihood (paper Section 2, equations 4-8).
* MAP  -- maximum a posteriori with hyperpriors on (mu, log sigma^2)
          (paper Section 5, equation 12).
* BIMLE -- bounded influence maximum likelihood via iteratively reweighted
          maximum likelihood (paper Section 4, equations 9-11).

statsmodels provides generic M-estimation for linear regression
(``statsmodels.robust.robust_linear_model.RLM``, Huber/Tukey weight
functions on regression residuals), but it has no notion of a normal-normal
empirical Bayes prior with test-specific sampling variances tau_i^2, nor the
paper's model-implied weighting w_i = min(1, b^2 / z_i^2) on the *prior*
parameters. No library equivalent covers this estimator, so BIMLE, MAP, and
the pooled MLE below are a custom implementation of paper equations 4-13.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.stats import norm

from src import config


@dataclass
class EstimationResult:
    """Container for a fitted (mu, sigma^2) pair and diagnostics.

    Attributes:
        mu: estimated prior mean.
        sigma2: estimated prior variance (>= 0).
        weights: per-observation weights used in the final iteration
            (all ones for MLE and MAP; BIMLE weights for BIMLE).
        n_iter: number of iterations used by the estimator (1 for MLE/MAP).
        converged: whether the optimizer/IRWLS loop reported convergence.
    """

    mu: float
    sigma2: float
    weights: np.ndarray
    n_iter: int
    converged: bool


def _validate_inputs(delta_hat: np.ndarray, tau_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce the estimated impacts and sampling variances.

    Args:
        delta_hat: array of estimated treatment impacts Delta_hat_i.
        tau_sq: array of known sampling variances tau_i^2 (must be > 0).

    Returns:
        The two arrays as float64 numpy arrays of equal length.
    """
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    tau_sq = np.asarray(tau_sq, dtype=np.float64)
    if delta_hat.shape != tau_sq.shape:
        raise ValueError("delta_hat and tau_sq must have the same shape")
    if delta_hat.size == 0:
        raise ValueError("delta_hat and tau_sq must contain at least one observation")
    if np.any(tau_sq <= 0):
        raise ValueError("tau_sq must be strictly positive for every test")
    if not np.all(np.isfinite(delta_hat)) or not np.all(np.isfinite(tau_sq)):
        raise ValueError("delta_hat and tau_sq must be finite")
    return delta_hat, tau_sq


def weighted_log_likelihood(
    mu: float, sigma2: float, delta_hat: np.ndarray, tau_sq: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted normal-normal log likelihood, paper equations (4)-(6).

    L(mu, sigma^2) = -1/2 * sum_i w_i * [log(sigma^2 + tau_i^2)
                                          + (Delta_hat_i - mu)^2 / (sigma^2 + tau_i^2)]

    With w_i = 1 for all i this is the ordinary marginal log likelihood used
    by the MLE; with BIMLE weights it is the weighted objective maximized in
    step 4 of the iteratively reweighted algorithm (Section 4).

    Args:
        mu: candidate prior mean.
        sigma2: candidate prior variance (must be >= 0).
        delta_hat: estimated treatment impacts.
        tau_sq: known sampling variances.
        weights: per-observation weights; defaults to all ones.

    Returns:
        The (weighted) log-likelihood value.
    """
    if weights is None:
        weights = np.ones_like(delta_hat)
    total_var = sigma2 + tau_sq
    return -0.5 * np.sum(weights * (np.log(total_var) + (delta_hat - mu) ** 2 / total_var))


def _neg_ll_reparam(theta: np.ndarray, delta_hat: np.ndarray, tau_sq: np.ndarray, weights: np.ndarray) -> float:
    """Negative weighted log likelihood in (mu, log sigma^2), for ``minimize``.

    Optimizing over log(sigma^2) rather than sigma^2 enforces sigma^2 >= 0
    without a constrained optimizer, as suggested in the implementation
    notes.
    """
    mu, log_sigma2 = theta
    sigma2 = np.exp(log_sigma2)
    return -weighted_log_likelihood(mu, sigma2, delta_hat, tau_sq, weights)


def fit_mle(delta_hat: np.ndarray, tau_sq: np.ndarray) -> EstimationResult:
    """Maximum likelihood estimate of (mu, sigma^2), paper equations (4)-(8).

    Args:
        delta_hat: array of estimated treatment impacts Delta_hat_i.
        tau_sq: array of known sampling variances tau_i^2.

    Returns:
        An EstimationResult with unit weights for every observation.
    """
    delta_hat, tau_sq = _validate_inputs(delta_hat, tau_sq)
    weights = np.ones_like(delta_hat)
    theta0 = np.array([np.mean(delta_hat), np.log(max(np.var(delta_hat) - np.mean(tau_sq), 1e-6))])
    result = optimize.minimize(
        _neg_ll_reparam,
        theta0,
        args=(delta_hat, tau_sq, weights),
        method=config.OPTIMIZER_METHOD,
        tol=config.OPTIMIZER_TOL,
    )
    mu_hat, log_sigma2_hat = result.x
    return EstimationResult(mu=float(mu_hat), sigma2=float(np.exp(log_sigma2_hat)), weights=weights, n_iter=1, converged=bool(result.success))


def _map_objective(
    theta: np.ndarray,
    delta_hat: np.ndarray,
    tau_sq: np.ndarray,
    mu_prior_mean: float,
    mu_prior_sd: float,
    logsigma2_prior_mean: float,
    logsigma2_prior_sd: float,
) -> float:
    """Negative penalized log likelihood maximized by MAP, paper equation (12).

    L(mu, sigma^2) + log pi(mu) + log pi(sigma^2), with pi(mu) Gaussian and
    log(sigma^2) ~ N(logsigma2_prior_mean, logsigma2_prior_sd^2). The change
    of variables contributes no extra Jacobian term because the hyperprior on
    sigma^2 is specified directly through its log (as in the paper's Table 1
    notes: "log(sigma^2) ~ N(-16, 2^2)"), so pi(sigma^2) is evaluated via the
    log-normal density of sigma^2 implied by that specification.
    """
    mu, log_sigma2 = theta
    sigma2 = np.exp(log_sigma2)
    weights = np.ones_like(delta_hat)
    log_lik = weighted_log_likelihood(mu, sigma2, delta_hat, tau_sq, weights)
    log_prior_mu = norm.logpdf(mu, loc=mu_prior_mean, scale=mu_prior_sd)
    # log(sigma^2) ~ N(logsigma2_prior_mean, logsigma2_prior_sd^2); evaluate at log(sigma^2) (paper Table 1 notes).
    log_prior_sigma2 = norm.logpdf(log_sigma2, loc=logsigma2_prior_mean, scale=logsigma2_prior_sd)
    return -(log_lik + log_prior_mu + log_prior_sigma2)


def fit_map(
    delta_hat: np.ndarray,
    tau_sq: np.ndarray,
    mu_prior_mean: float,
    mu_prior_sd: float,
    logsigma2_prior_mean: float,
    logsigma2_prior_sd: float,
) -> EstimationResult:
    """Maximum a posteriori estimate of (mu, sigma^2), paper equation (12).

    Args:
        delta_hat: array of estimated treatment impacts Delta_hat_i.
        tau_sq: array of known sampling variances tau_i^2.
        mu_prior_mean: hyperprior mean for mu.
        mu_prior_sd: hyperprior standard deviation for mu.
        logsigma2_prior_mean: hyperprior mean for log(sigma^2).
        logsigma2_prior_sd: hyperprior standard deviation for log(sigma^2).

    Returns:
        An EstimationResult with unit weights for every observation.
    """
    delta_hat, tau_sq = _validate_inputs(delta_hat, tau_sq)
    if mu_prior_sd <= 0 or logsigma2_prior_sd <= 0:
        raise ValueError("hyperprior standard deviations must be strictly positive")
    weights = np.ones_like(delta_hat)
    theta0 = np.array([mu_prior_mean, logsigma2_prior_mean])
    result = optimize.minimize(
        _map_objective,
        theta0,
        args=(delta_hat, tau_sq, mu_prior_mean, mu_prior_sd, logsigma2_prior_mean, logsigma2_prior_sd),
        method=config.OPTIMIZER_METHOD,
        tol=config.OPTIMIZER_TOL,
    )
    mu_hat, log_sigma2_hat = result.x
    return EstimationResult(mu=float(mu_hat), sigma2=float(np.exp(log_sigma2_hat)), weights=weights, n_iter=1, converged=bool(result.success))


def fit_bimle(
    delta_hat: np.ndarray,
    tau_sq: np.ndarray,
    b: float,
    max_iter: int = config.BIMLE_MAX_ITER,
    tol: float = config.BIMLE_CONV_TOL,
) -> EstimationResult:
    """Bounded influence maximum likelihood estimate, paper equations (9)-(11).

    Iteratively reweighted maximum likelihood (paper Section 4, steps 1-5):
      1. Initialize (mu, sigma^2) via MLE on the full sample.
      2. Compute standardized residuals z_i = (Delta_hat_i - mu) / sqrt(sigma^2 + tau_i^2).
      3. Update weights w_i = min(1, b^2 / z_i^2).
      4. Re-estimate (mu, sigma^2) by maximizing sum_i w_i * L_i(mu, sigma^2).
      5. Repeat 2-4 until convergence (paper reports typically < 10 iterations).

    Args:
        delta_hat: array of estimated treatment impacts Delta_hat_i.
        tau_sq: array of known sampling variances tau_i^2.
        b: BIMLE tuning constant (b >= 0).
        max_iter: maximum number of reweighting iterations.
        tol: convergence tolerance on the change in (mu, sigma^2) across iterations.

    Returns:
        An EstimationResult with the final weights, iteration count, and a
        convergence flag.
    """
    delta_hat, tau_sq = _validate_inputs(delta_hat, tau_sq)
    if b < 0:
        raise ValueError("BIMLE tuning constant b must be non-negative")

    mle = fit_mle(delta_hat, tau_sq)
    mu, sigma2 = mle.mu, mle.sigma2
    weights = np.ones_like(delta_hat)
    converged = False

    for iteration in range(1, max_iter + 1):
        # Step 2: standardized residuals under the current parameter estimates.
        z = (delta_hat - mu) / np.sqrt(sigma2 + tau_sq)
        z_sq = np.maximum(z**2, config.Z_EPS)
        # Step 3: bounded-influence weights (paper equation 9).
        weights = np.minimum(1.0, (b**2) / z_sq)

        # Step 4: re-estimate (mu, sigma^2) holding the weights fixed.
        theta0 = np.array([mu, np.log(max(sigma2, 1e-6))])
        result = optimize.minimize(
            _neg_ll_reparam,
            theta0,
            args=(delta_hat, tau_sq, weights),
            method=config.OPTIMIZER_METHOD,
            tol=config.OPTIMIZER_TOL,
        )
        new_mu, new_log_sigma2 = result.x
        new_sigma2 = np.exp(new_log_sigma2)

        if abs(new_mu - mu) < tol and abs(new_sigma2 - sigma2) < tol:
            mu, sigma2 = float(new_mu), float(new_sigma2)
            converged = True
            break
        mu, sigma2 = float(new_mu), float(new_sigma2)

    return EstimationResult(mu=mu, sigma2=sigma2, weights=weights, n_iter=iteration, converged=converged)
