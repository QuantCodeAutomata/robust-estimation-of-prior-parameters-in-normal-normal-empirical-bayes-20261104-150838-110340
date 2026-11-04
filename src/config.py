"""Experiment parameters for the replication of Burson and Kessler (2026),
"Robust Estimation of Prior Parameters in Normal-Normal Empirical Bayes".

Every parameter used by an experiment script lives here as a module-level
UPPER_CASE variable, grouped by experiment, each with a comment stating its
meaning, unit, and source (paper section/equation, or "not specified in the
paper"). No classes or dictionaries are used so every value stays directly
greppable and overridable.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared optimizer settings (paper sections 3-4; implementation notes)
# ---------------------------------------------------------------------------
OPTIMIZER_METHOD = "BFGS"  # scipy.optimize.minimize method, not specified in the paper
OPTIMIZER_TOL = 1e-10  # convergence tolerance for scipy.optimize.minimize, not specified in the paper
BIMLE_MAX_ITER = 50  # cap on IRWLS iterations; paper (Section 4, step 5) reports convergence "typically < 10 iterations"
BIMLE_CONV_TOL = 1e-8  # parameter-change tolerance to declare IRWLS convergence, not specified in the paper
Z_EPS = 1e-12  # numerical floor on z_i^2 to avoid division by zero in BIMLE weights (implementation note)

# ---------------------------------------------------------------------------
# Experiment 1: Illustrative Simulations (paper Section 6, Figure 1)
# ---------------------------------------------------------------------------
EXP1_N_TESTS = 1000  # number of simulated A/B tests N, paper Section 6
EXP1_TRUE_MU = 0.0  # true prior mean mu, paper Section 6
EXP1_TRUE_SIGMA2 = 1.0  # true prior variance sigma^2, paper Section 6
EXP1_TAU_SQ = 1.0  # known sampling variance tau_i^2 for every test i, paper Section 6
EXP1_K_VALUES = [0, 1, 2, 3, 4, 5]  # number of contaminated observations k, paper Section 6
EXP1_EXTREME_VALUE = 50.0  # contaminated Delta_hat_i value, paper Section 6
EXP1_N_REPLICATIONS = 100  # Monte Carlo replications per k, paper Section 6 ("We repeat this exercise 100 times")
EXP1_BIMLE_B = 10.0  # BIMLE tuning constant used in this simulation, paper Section 6 footnote 3
EXP1_MAP_MU_PRIOR_MEAN = 0.0  # hyperprior mean for mu: pi(mu) = N(0, 1), paper Section 6
EXP1_MAP_MU_PRIOR_SD = 1.0  # hyperprior sd for mu: pi(mu) = N(0, 1), paper Section 6
EXP1_MAP_LOGSIGMA2_PRIOR_MEAN = 0.0  # hyperprior mean for log(sigma^2) ~ N(0, 0.25^2), paper Section 6
EXP1_MAP_LOGSIGMA2_PRIOR_SD = 0.25  # hyperprior sd for log(sigma^2) ~ N(0, 0.25^2), paper Section 6
EXP1_SEED = 20261104  # random seed for reproducibility, not specified in the paper

# Expected outcomes read from Figure 1 of the paper (page 6-7), for comparison only.
EXP1_EXPECTED_MU = {  # Figure 1, left chart (number of contaminated obs -> mu_hat)
    "MLE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "MAP": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "BIMLE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
EXP1_EXPECTED_SIGMA2 = {  # Figure 1, right chart (number of contaminated obs -> sigma2_hat)
    "MLE": [0.8, 3.2, 5.8, 8.6, 11.0, 13.5],
    "MAP": [0.8, 3.0, 5.5, 7.8, 10.0, 12.2],
    "BIMLE": [0.8, 1.0, 1.5, 1.8, 2.5, 3.0],
}

# ---------------------------------------------------------------------------
# Experiment 2: Empirical Application to Amazon A/B Tests (paper Section 7)
# ---------------------------------------------------------------------------
# The Amazon A/B-testing dataset underlying Tables 1-4 is proprietary and is
# not released with the paper (Section 7). It cannot be retrieved from the
# QuantX Dataloader or Massive REST APIs, which serve market/OHLCV data, not
# internal experimentation-platform records. The pipeline below is exercised
# on synthetic data generated in data/generate_synthetic.py to match the
# paper's described structure (four metrics, six half-year periods,
# occasional contaminated periods), so that every estimator, confidence
# interval, and stability statistic in the paper's methodology is computed by
# the same code that would run on the real data.
EXP2_METRICS = ["Metric A", "Metric B", "Metric C", "Metric D"]  # four illustrative metrics, paper Section 7
EXP2_HALF_YEARS = ["2023-H1", "2023-H2", "2024-H1", "2024-H2", "2025-H1", "2025-H2"]  # paper Section 7
EXP2_BIMLE_B_MAIN = 15.0  # recommended BIMLE tuning constant, paper Section 4 ("We recommend a default of b = 15")
EXP2_BIMLE_B_SENSITIVITY = [10.0, 15.0, 20.0]  # sensitivity grid, paper Appendix A / Tables 3-4
EXP2_MAP_MU_PRIOR_MEAN = 0.0  # hyperprior mean, mu ~ N(0, 1e-6), paper Table 1 notes
EXP2_MAP_MU_PRIOR_VAR = 1e-6  # hyperprior variance, mu ~ N(0, 1e-6), paper Table 1 notes
EXP2_MAP_LOGSIGMA2_PRIOR_MEAN = -16.0  # hyperprior mean, log(sigma^2) ~ N(-16, 2^2), paper Table 1 notes
EXP2_MAP_LOGSIGMA2_PRIOR_SD = 2.0  # hyperprior sd, log(sigma^2) ~ N(-16, 2^2), paper Table 1 notes
EXP2_N_BOOTSTRAP = 200  # nonparametric bootstrap resamples per half-year window, paper Table 1 notes
EXP2_CI_LEVEL = 0.95  # confidence level for bootstrap intervals, paper Table 1 notes ("95% confidence intervals")
EXP2_BOOTSTRAP_SEED = 20261104  # random seed for the bootstrap, not specified in the paper

# Synthetic-data generation parameters (data/generate_synthetic.py). These
# values are not from the paper: they only let the shared estimation pipeline
# be exercised end-to-end in the documented absence of the proprietary
# Amazon dataset (see "Deviations from the paper" in results/RESULTS.md).
SYN_TESTS_PER_PERIOD = 3000  # synthetic A/B tests per metric x half-year cell, not specified in the paper.
# Chosen so that a handful of contaminated tests (below) make up well under
# 1/BIMLE_b^2 of the cell, the share above which the MLE-initialized IRWLS
# scale already absorbs the contamination before the first reweighting step
# (masking); this keeps the qualitative contrast of paper Section 7 (MLE/MAP
# swings vs. BIMLE stability) visible instead of an estimator artifact.
SYN_TRUE_MU = {"Metric A": 0.0002, "Metric B": -0.0001, "Metric C": 0.0, "Metric D": 0.0005}  # not specified in the paper
SYN_TRUE_SIGMA = {"Metric A": 0.00016, "Metric B": 0.0006, "Metric C": 0.003, "Metric D": 0.0012}  # not specified in the paper
SYN_TAU_SD_RANGE = (0.0008, 0.004)  # range of sampling standard deviations tau_i across tests, not specified in the paper
SYN_CONTAMINATED_PERIODS = {  # (metric, half_year) -> number of injected extreme observations, not specified in the paper
    ("Metric A", "2024-H1"): 2,
    ("Metric A", "2025-H1"): 1,
    ("Metric C", "2024-H2"): 3,
}
SYN_EXTREME_MULTIPLIER = 60.0  # contaminated Delta_hat = multiplier * tau_i, not specified in the paper
SYN_SEED = 20261104  # random seed for synthetic data generation, not specified in the paper

RESULTS_DIR = "results"  # output directory for all experiment artifacts
RNG = np.random.default_rng  # convenience alias used by experiment scripts
