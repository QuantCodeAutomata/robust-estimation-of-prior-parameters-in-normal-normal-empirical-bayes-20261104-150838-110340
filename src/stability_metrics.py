"""Stability and downweighting diagnostics used in the empirical application
(paper Section 7 and Table 1 notes).

- Coefficient of variation (CV) across half-year periods, used to compare
  estimator stability (paper Section 7: "CV is the coefficient of variation
  across half-year periods").
- Downweighting diagnostics for BIMLE: the share of observations receiving
  weight below 1 (DW) and the average weight among those downweighted
  observations (w | DW), paper Table 1 notes.
"""
from __future__ import annotations

import numpy as np


def coefficient_of_variation(values: np.ndarray) -> float:
    """CV = standard deviation / mean of a set of period estimates.

    Args:
        values: array of estimates across half-year periods (e.g. sigma_hat
            for one metric and one estimator across six periods).

    Returns:
        The coefficient of variation. Uses the sample standard deviation
        (ddof=1) as is standard when summarizing variability across a small
        number of discrete periods.

    Raises:
        ValueError: if fewer than two values are given, or the mean is zero.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("coefficient_of_variation requires at least two period estimates")
    mean = np.mean(values)
    if mean == 0:
        raise ValueError("coefficient_of_variation is undefined when the mean is zero")
    return float(np.std(values, ddof=1) / mean)


def downweight_stats(weights: np.ndarray) -> tuple[float, float]:
    """Share of downweighted observations (DW) and their average weight.

    Args:
        weights: BIMLE weights w_i in [0, 1] for one metric/period fit.

    Returns:
        A tuple (dw_pct, avg_weight_downweighted):
            dw_pct: percent of observations with weight strictly below 1.
            avg_weight_downweighted: average weight among those observations
                (NaN if none are downweighted).
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(weights < 0) or np.any(weights > 1 + 1e-9):
        raise ValueError("weights must lie in [0, 1]")
    downweighted = weights < 1.0 - 1e-12
    dw_pct = 100.0 * np.mean(downweighted)
    avg_weight = float(np.mean(weights[downweighted])) if np.any(downweighted) else float("nan")
    return float(dw_pct), avg_weight
