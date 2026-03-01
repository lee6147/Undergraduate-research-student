"""Statistical testing utilities for VQC benchmark."""
import numpy as np
from scipy import stats


def paired_statistical_test(scores_a, scores_b, test_name="auto"):
    """Run paired statistical test between two sets of scores.

    For n >= 20: paired t-test
    For n < 20: Wilcoxon signed-rank test (non-parametric)

    Returns dict with test_name, statistic, p_value, significant (at alpha=0.05).
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    if test_name == "auto":
        test_name = "t-test" if n >= 20 else "wilcoxon"

    if test_name == "t-test":
        stat, p_val = stats.ttest_rel(scores_a, scores_b)
    elif test_name == "wilcoxon":
        diff = scores_a - scores_b
        if np.all(diff == 0):
            return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0, "significant": False}
        stat, p_val = stats.wilcoxon(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test_name}")

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def compute_stats(values):
    """Compute mean and std for a list of values."""
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))
