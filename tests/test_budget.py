"""Budget checks must not lose incurred spend or discount the wrong tokens."""

from __future__ import annotations

import pytest

from marli.budget import SpendGuard, tinker_cost
from marli.errors import BudgetExceededError, ConfigError
from marli.model import load_model


def test_spend_guard_check_charge_remaining_and_summary() -> None:
    guard = SpendGuard(2.0)
    assert guard.spent == 0.0
    assert guard.remaining() == 2.0
    assert guard.summary() == {"max_usd": 2.0, "spent_usd": 0.0, "by_item": {}}
    guard.check(2.0, "estimate")
    assert guard.spent == 0.0
    assert guard.summary()["by_item"] == {}
    guard.charge(0.5, "sample")
    guard.charge(0.25, "sample")
    guard.charge(0.25, "train")
    assert guard.spent == 1.0
    assert guard.remaining() == 1.0
    assert guard.summary() == {
        "max_usd": 2.0,
        "spent_usd": 1.0,
        "by_item": {"sample": 0.75, "train": 0.25},
    }
    guard.check(1.0, "exact limit")
    guard.charge(1.0, "exact limit")
    assert guard.spent == 2.0
    assert guard.remaining() == 0.0


def test_check_exceeding_limit_does_not_record_spend() -> None:
    guard = SpendGuard(1.0)
    guard.charge(0.75, "earlier")
    before = guard.summary()
    with pytest.raises(BudgetExceededError) as caught:
        guard.check(0.5, "next sample")
    assert caught.value.exit_code == 4
    message = str(caught.value)
    assert "limit $1" in message
    assert "spent so far $0.75" in message
    assert "next sample" in message
    assert guard.summary() == before
    assert guard.remaining() == 0.25


def test_charge_exceeding_limit_keeps_incurred_spend() -> None:
    guard = SpendGuard(1.0)
    guard.charge(0.75, "sample")
    with pytest.raises(BudgetExceededError) as caught:
        guard.charge(0.5, "sample")
    assert caught.value.exit_code == 4
    message = str(caught.value)
    assert "limit $1" in message
    assert "spent so far $1.25" in message
    assert "sample" in message
    assert guard.spent == 1.25
    assert guard.remaining() == -0.25
    assert guard.summary() == {"max_usd": 1.0, "spent_usd": 1.25, "by_item": {"sample": 1.25}}
    with pytest.raises(BudgetExceededError):
        guard.check(0, "already over budget")


def test_unlimited_budget_still_tracks_spend() -> None:
    guard = SpendGuard(None)
    guard.check(1_000_000.0, "estimate")
    assert guard.spent == 0.0
    guard.charge(1_000_000.0, "sample")
    guard.charge(2.0, "sample")
    assert guard.remaining() is None
    assert guard.spent == 1_000_002.0
    assert guard.summary() == {
        "max_usd": None,
        "spent_usd": 1_000_002.0,
        "by_item": {"sample": 1_000_002.0},
    }


def test_zero_budget_allows_only_zero_spend() -> None:
    guard = SpendGuard(0)
    guard.check(0, "free")
    guard.charge(0, "free")
    assert guard.remaining() == 0
    with pytest.raises(BudgetExceededError):
        guard.check(0.000001, "paid")


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_budget_and_charges_are_rejected_without_mutation(amount: float) -> None:
    with pytest.raises(ValueError, match="max_usd"):
        SpendGuard(amount)
    for max_usd in (None, 1.0):
        guard = SpendGuard(max_usd)
        guard.charge(0.25, "existing")
        before = guard.summary()
        for operation in (guard.check, guard.charge):
            with pytest.raises(ValueError, match="usd"):
                operation(amount, "invalid")
            assert guard.summary() == before


def test_summary_is_an_independent_snapshot() -> None:
    guard = SpendGuard(1.0)
    guard.charge(0.25, "sample")
    snapshot = guard.summary()
    items = snapshot["by_item"]
    assert isinstance(items, dict)
    items["sample"] = 100
    snapshot["spent_usd"] = 100
    assert guard.spent == 0.25
    assert guard.summary()["by_item"] == {"sample": 0.25}


@pytest.mark.parametrize(
    "counts,expected",
    [
        ({}, 0.0),
        ({"prefill": 1_000_000}, 0.20),
        ({"cached_prefill": 1_000_000}, 0.04),
        ({"sample": 1_000_000}, 0.60),
        ({"train": 1_000_000}, 0.44),
        ({"prefill": 1000, "cached_prefill": 2000, "sample": 3000, "train": 4000}, 0.00384),
    ],
)
def test_tinker_cost_uses_usd_per_million_and_cached_discount(
    counts: dict[str, int], expected: float
) -> None:
    assert tinker_cost(load_model("qwen3_8b"), **counts) == pytest.approx(expected)


def test_tinker_cost_uses_the_selected_models_prices() -> None:
    assert tinker_cost(load_model("qwen3_5_4b"), prefill=1_000_000) == pytest.approx(0.33)


@pytest.mark.parametrize("field", ["prefill", "cached_prefill", "sample", "train"])
@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_tinker_cost_rejects_invalid_token_counts(field: str, tokens: object) -> None:
    with pytest.raises(ValueError, match=field):
        tinker_cost(load_model("qwen3_8b"), **{field: tokens})


def test_tinker_cost_rejects_non_tinker_models_even_for_zero_tokens() -> None:
    with pytest.raises(ConfigError, match="qwen3_4b_instruct_2507.*no Tinker prices"):
        tinker_cost(load_model("qwen3_4b_instruct_2507"))
