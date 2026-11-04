# Robust Estimation of Prior Parameters in Normal-Normal Empirical Bayes

Replication of selected experiments from Burson, J. and Kessler, D. (2026),
"Robust Estimation of Prior Parameters in Normal-Normal Empirical Bayes".

This repository implements and compares three estimators of the prior
parameters (mu, sigma^2) of a normal-normal empirical Bayes model for A/B
test impacts: the Maximum Likelihood Estimator (MLE), the Maximum A
Posteriori estimator (MAP) with informative hyperpriors, and the
Bounded-Influence Maximum Likelihood Estimator (BIMLE), a robust estimator
obtained by iteratively reweighted maximum likelihood with a tuning
constant `b`.

## Reproduced experiments

* **Experiment 1 -- Illustrative Simulations: Sensitivity to Extreme
  Observations** (paper Section 6). A Monte Carlo study on simulated
  normal-normal data: `N = 1000` tests are drawn with true prior mean
  `mu = 0`, true prior variance `sigma^2 = 1`, and known sampling variance
  `tau_i^2 = 1`; for each `k` in `{0, 1, 2, 3, 4, 5}`, `k` randomly chosen
  observations are replaced with the extreme value `50`, and MLE, MAP, and
  BIMLE are re-estimated. The exercise is repeated 100 times per `k` and the
  table reports average estimates (paper Figure 1).

* **Experiment 2 -- Empirical Application: Amazon A/B Tests** (paper
  Section 7). The same three estimators are applied per metric and
  half-year period to A/B-test panels, with nonparametric bootstrap
  confidence intervals, a coefficient-of-variation (CV) stability summary,
  BIMLE downweighting diagnostics (DW, w|DW), and a sensitivity analysis of
  BIMLE over the tuning constant `b in {10, 15, 20}` (paper Tables 1, 3, 4).
  The Amazon dataset underlying the paper's tables is proprietary and is not
  released with the paper; see "Deviations from the paper" in
  `results/RESULTS.md`.

## Repository layout

```
src/     estimators (eb_model.py), bootstrap CIs (bootstrap.py),
         stability diagnostics (stability_metrics.py), config.py
exp/     exp1.py (Section 6 simulation), exp2.py (Section 7 application)
data/    generate_synthetic.py (Section 6 sampling design and a
         structurally analogous synthetic panel for Section 7)
tests/   unit tests for every src/ and data/ module
results/ tables, plots, and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No market-data API keys are required: both experiments use simulated data
and a proprietary internal A/B-testing dataset, not price or OHLCV data.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1          # Experiment 1: contamination simulation (paper Section 6)
python -m exp.exp2          # Experiment 2: empirical application (paper Section 7)
python -m exp.exp2 --full   # same pipeline, full sensitivity grid over b
```

Or run everything with:

```bash
./reproduce.sh
```

Each run overwrites the tables and plots in `results/`. After a rerun, check
the numbers and verdicts in `results/RESULTS.md` against the new files.

## Deviations and limitations

See `results/RESULTS.md` for the full list of deviations from the paper and
the paper's results that could not be reproduced here.
