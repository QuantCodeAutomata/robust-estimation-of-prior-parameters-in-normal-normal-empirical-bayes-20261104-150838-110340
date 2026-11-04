"""Unit tests for src/stability_metrics.py."""
import numpy as np
import pytest

from src.stability_metrics import coefficient_of_variation, downweight_stats


def test_cv_zero_for_constant_series():
    assert coefficient_of_variation(np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)


def test_cv_matches_manual_computation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.std(values, ddof=1) / np.mean(values)
    assert coefficient_of_variation(values) == pytest.approx(expected)


def test_cv_requires_at_least_two_values():
    with pytest.raises(ValueError):
        coefficient_of_variation(np.array([1.0]))


def test_downweight_stats_all_full_weight():
    dw_pct, avg_w = downweight_stats(np.ones(10))
    assert dw_pct == 0.0
    assert np.isnan(avg_w)


def test_downweight_stats_partial_downweighting():
    weights = np.array([1.0, 1.0, 0.5, 0.2, 1.0])
    dw_pct, avg_w = downweight_stats(weights)
    assert dw_pct == pytest.approx(40.0)
    assert avg_w == pytest.approx(0.35)


def test_downweight_stats_rejects_out_of_range_weights():
    with pytest.raises(ValueError):
        downweight_stats(np.array([0.5, 1.2]))
