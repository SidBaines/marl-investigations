"""Hand-computable statistics pin denominators, task pairing and voting semantics."""

from __future__ import annotations

import random

import pytest

from marli.eval.stats import (
    avg_at_k,
    bootstrap_mean,
    maj_at_k,
    mcnemar_exact,
    paired_comparison,
    percentile,
    wilson_ci,
)


def test_wilson_known_values_and_extremes() -> None:
    assert wilson_ci(50, 100) == pytest.approx((0.40383153, 0.59616847))
    assert wilson_ci(0, 10) == pytest.approx((0, 0.2775328))
    assert wilson_ci(10, 10) == pytest.approx((0.7224672, 1))
    assert wilson_ci(0, 0) == (0, 1)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_bootstrap_deterministic_local_rng_and_constant() -> None:
    state = random.getstate()
    first = bootstrap_mean([0, 1, 2, 3, 4], seed=12)
    assert first == bootstrap_mean([0, 1, 2, 3, 4], seed=12)
    assert first.mean == 2 and first.low < 2 < first.high
    assert random.getstate() == state
    constant = bootstrap_mean([0.4] * 10)
    assert constant.mean == constant.low == constant.high == 0.4


def test_mcnemar_exact_small_table_and_symmetry() -> None:
    assert mcnemar_exact(1, 5) == 14 / 64
    assert mcnemar_exact(5, 1) == 14 / 64
    assert mcnemar_exact(0, 4) == 0.125
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(50, 50) == 1
    assert 0 <= mcnemar_exact(10000, 0) <= 1


def test_paired_per_task_not_unpaired_episodes() -> None:
    cell = {"a": [1] * 10, "b": [0], "extra": [1]}
    baseline = {"b": [1] * 8, "a": [0], "other": [0]}
    result = paired_comparison(cell, baseline, seed=7)
    assert result.n_tasks == 2 and result.difference == 0
    assert result.low == -1 and result.high == 1
    assert result.wins == result.losses == 1 and result.permutation_p == 1
    assert result.mcnemar_p is None
    assert result == paired_comparison(dict(reversed(list(cell.items()))), baseline, seed=7)
    all_better = paired_comparison({str(i): [1] for i in range(4)}, {str(i): [0] for i in range(4)})
    assert all_better.difference == all_better.low == all_better.high == 1
    assert all_better.mcnemar_p == 0.125
    ties = paired_comparison({"a": [0, 1]}, {"a": [1]})
    assert ties.losses == 1 and ties.difference == -0.5
    with pytest.raises(ValueError, match="shared task"):
        paired_comparison({"a": [1]}, {"b": [1]})


def test_average_and_majority_are_distinct_from_pass_at_k() -> None:
    assert avg_at_k([1, 0, 0]) == 1 / 3
    assert avg_at_k([1, 0, 1], 2) == 0.5
    assert maj_at_k(["right", "right", "wrong"], [1, 1, 0]) == 1
    # Plurality of equivalent answers can win without >50% of episodes correct.
    assert maj_at_k(["right", "wrong1", "right", "wrong2", "wrong3"], [1, 0, 1, 0, 0]) == 1
    assert maj_at_k(["wrong", "right"], [0, 1]) == 0
    assert maj_at_k([None, None, "right"], [0, 0, 1]) == 1
    assert maj_at_k([None, None], [0, 0]) == 0
    assert maj_at_k(["wrong", "right", "right"], [0, 1, 1], 1) == 0
    with pytest.raises(ValueError):
        avg_at_k([0, 1], 3)


def test_percentiles_interpolate_and_validate() -> None:
    assert percentile([40, 10, 30, 20], 50) == 25
    assert percentile([0, 10], 90) == 9
    assert percentile([5], 90) == 5
    assert percentile([3, 1, 2], 0) == 1
    assert percentile([3, 1, 2], 100) == 3
    with pytest.raises(ValueError):
        percentile([], 50)
