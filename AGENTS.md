# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of selected experiments from Burson, J. and Kessler, D.
(2026), "Robust Estimation of Prior Parameters in Normal-Normal Empirical
Bayes": three estimators of the prior parameters `(mu, sigma^2)` of a
normal-normal empirical Bayes model -- MLE, MAP, and the Bounded-Influence
MLE (BIMLE) -- and two experiments from the paper:

* Experiment 1 (paper Section 6): a Monte Carlo contamination study on
  simulated data.
* Experiment 2 (paper Section 7): the same estimators applied per metric and
  half-year period, with bootstrap confidence intervals and a stability/
  downweighting summary, run on a synthetic analogue of the paper's
  proprietary Amazon A/B-testing panel (see `results/RESULTS.md`).

## Layout

```
src/     eb_model.py (MLE/MAP/BIMLE), bootstrap.py (bootstrap CIs),
         stability_metrics.py (CV, downweighting diagnostics), config.py
exp/     exp1.py, exp2.py -- run the experiments, read src/, write results/
data/    generate_synthetic.py -- Section 6 sampling design and the
         synthetic analogue of the Section 7 panel; no market data is used
         by either experiment, so there is no data_loader.py
tests/   unit tests for every src/ and data/ module
results/ tables, plots, and RESULTS.md (overwritten by each experiment run)
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -m "not market_data"   # equivalent to `python -m pytest` here: no test is marked market_data

python -m exp.exp1          # Experiment 1 (paper Section 6)
python -m exp.exp2          # Experiment 2 (paper Section 7)
python -m exp.exp2 --full   # same pipeline, full b-sensitivity grid (paper Tables 3-4)

./reproduce.sh              # tests, then both experiments
```

## Conventions

* All experiment parameters live in `src/config.py` as UPPER_CASE
  module-level variables, grouped by experiment, each with a comment stating
  its meaning, unit, and source (a paper section/equation, or "not specified
  in the paper"). No classes or parameter dictionaries.
* Data preparation lives only in `data/`: `generate_synthetic.py` draws the
  Section 6 contamination samples and the Section 7 synthetic analogue.
  Neither experiment calls a market-data API; no API keys are used anywhere
  in this repository.
* `exp/exp1.py` and `exp/exp2.py` call the estimators in `src/eb_model.py`
  (`fit_mle`, `fit_map`, `fit_bimle`), the bootstrap routine in
  `src/bootstrap.py` (`bootstrap_ci`), and the diagnostics in
  `src/stability_metrics.py` (`coefficient_of_variation`,
  `downweight_stats`) with the signatures as written; they do not define a
  private estimator or performance shortcut.
* Tests live only in `tests/`, run with `python -m pytest` from the
  repository root. No test in this repository depends on a live API call or
  network access.
* After a rerun of `python -m exp.exp1` / `python -m exp.exp2`, the files in
  `results/` are overwritten; the numbers and verdicts in
  `results/RESULTS.md` must then be checked against the new files and
  updated by hand.
