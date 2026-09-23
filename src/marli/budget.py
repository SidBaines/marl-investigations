"""Guard cumulative run spend while retaining charges that exceed the limit."""

from __future__ import annotations

from math import isfinite

from marli.errors import BudgetExceededError, ConfigError
from marli.model import ModelSpec


class SpendGuard:
    """Cumulative spend limit for one run (CLAUDE.md: spend guards; exit code 4)."""

    def __init__(self, max_usd: float | None) -> None:
        if max_usd is not None:
            _validate_amount(max_usd, "max_usd")
        self._max_usd = max_usd
        self._spent = 0.0
        self._by_item: dict[str, float] = {}

    @property
    def spent(self) -> float:
        return self._spent

    def remaining(self) -> float | None:
        """Return the unspent budget, negative after an overcharge, or None if unlimited."""
        return None if self._max_usd is None else self._max_usd - self._spent

    def check(self, usd: float, what: str) -> None:
        """Check an estimated charge without recording or reserving spend."""
        _validate_amount(usd, "usd")
        if self._max_usd is not None and self._spent + usd > self._max_usd:
            raise BudgetExceededError(
                f"budget limit ${self._max_usd:.4f} exceeded for {what!r}: "
                f"spent so far ${self._spent:.4f}, additional spend ${usd:.4f}"
            )

    def charge(self, usd: float, what: str) -> None:
        """Record incurred spend before checking the limit, even when it is exceeded."""
        _validate_amount(usd, "usd")
        self._spent += usd
        self._by_item[what] = self._by_item.get(what, 0.0) + usd
        if self._max_usd is not None and self._spent > self._max_usd:
            raise BudgetExceededError(
                f"budget limit ${self._max_usd:.4f} exceeded by {what!r} (charged ${usd:.4f}): "
                f"spent so far ${self._spent:.4f}"
            )

    def summary(self) -> dict[str, float | None | dict[str, float]]:
        """Return a snapshot of the limit, total, and accumulated spend per item."""
        return {
            "max_usd": self._max_usd,
            "spent_usd": self._spent,
            "by_item": dict(self._by_item),
        }


def _validate_amount(usd: float, name: str) -> None:
    if not isfinite(usd) or usd < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def tinker_cost(
    model: ModelSpec,
    *,
    prefill: int = 0,
    cached_prefill: int = 0,
    sample: int = 0,
    train: int = 0,
) -> float:
    """Price uncached and cached prefill separately; cached tokens receive 80% off."""
    prices = model.tinker_prices
    if prices is None:
        raise ConfigError(f"model {model.name!r} has no Tinker prices")
    for name, tokens in (
        ("prefill", prefill),
        ("cached_prefill", cached_prefill),
        ("sample", sample),
        ("train", train),
    ):
        if type(tokens) is not int or tokens < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return (
        (prefill + 0.2 * cached_prefill) * prices.prefill
        + sample * prices.sample
        + train * prices.train
    ) / 1_000_000
