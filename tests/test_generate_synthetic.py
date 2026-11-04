"""Unit tests for data/generate_synthetic.py."""
import numpy as np

from data.generate_synthetic import generate_contamination_sample, generate_synthetic_ab_tests
from src import config


def test_contamination_sample_shape_and_columns():
    rng = np.random.default_rng(0)
    df = generate_contamination_sample(100, 0.0, 1.0, 1.0, 3, 50.0, rng)
    assert list(df.columns) == ["delta_hat", "tau_sq"]
    assert len(df) == 100
    assert np.sum(df["delta_hat"] == 50.0) == 3
    assert np.all(df["tau_sq"] == 1.0)


def test_contamination_sample_zero_k_has_no_extreme_values():
    rng = np.random.default_rng(1)
    df = generate_contamination_sample(200, 0.0, 1.0, 1.0, 0, 50.0, rng)
    assert np.sum(df["delta_hat"] == 50.0) == 0


def test_contamination_sample_rejects_invalid_k():
    rng = np.random.default_rng(2)
    try:
        generate_contamination_sample(10, 0.0, 1.0, 1.0, 20, 50.0, rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_synthetic_ab_tests_has_expected_schema_and_cells():
    df = generate_synthetic_ab_tests(seed=config.SYN_SEED)
    assert set(df.columns) == {"metric", "half_year", "delta_hat", "tau_sq"}
    assert set(df["metric"].unique()) == set(config.EXP2_METRICS)
    assert set(df["half_year"].unique()) == set(config.EXP2_HALF_YEARS)
    n_cells = len(config.EXP2_METRICS) * len(config.EXP2_HALF_YEARS)
    assert len(df) == n_cells * config.SYN_TESTS_PER_PERIOD
    assert np.all(df["tau_sq"] > 0)
