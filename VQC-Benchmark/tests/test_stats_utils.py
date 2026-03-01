"""Tests for statistical utilities."""
PHASE_NAME = "Stats Utilities"

import numpy as np
from src.stats_utils import paired_statistical_test, compute_stats


def test_paired_test_identical_scores():
    scores = [0.9, 0.85, 0.88, 0.92, 0.87]
    result = paired_statistical_test(scores, scores)
    assert not result["significant"]


def test_paired_test_different_scores():
    a = [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.96, 0.93, 0.95, 0.94]
    b = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.49]
    result = paired_statistical_test(a, b)
    assert result["significant"]


def test_paired_test_returns_required_keys():
    a = [0.9, 0.85, 0.88]
    b = [0.8, 0.75, 0.78]
    result = paired_statistical_test(a, b)
    for key in ["test", "statistic", "p_value", "significant"]:
        assert key in result


def test_compute_stats():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, std = compute_stats(values)
    assert abs(mean - 3.0) < 1e-10
    assert abs(std - np.std(values)) < 1e-10
