"""Keep model capabilities and dated prices reproducible across backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path

from marli.errors import ConfigError
from marli.registry import Registry


@dataclass(frozen=True)
class TinkerPrices:
    """Published USD per million tokens, with the date the prices were read."""

    prefill: float
    sample: float
    train: float
    as_of: str

    def __post_init__(self) -> None:
        for field in ("prefill", "sample", "train"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{field} price must be finite and non-negative")
        if (
            not isinstance(self.as_of, str)
            or date.fromisoformat(self.as_of).isoformat() != self.as_of
        ):
            raise ValueError("as_of must be an ISO date (YYYY-MM-DD)")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    family: str
    renderer: str
    architecture: str
    max_ctx: int
    default_max_tokens: int
    thinking: bool
    tool_format: str
    tinker_id: str | None = None
    tinker_max_ctx: int | None = None
    tinker_prices: TinkerPrices | None = None
    local: str = "unverified"
    notes: str = ""

    def __post_init__(self) -> None:
        for field, choices in (
            ("family", ("qwen3", "qwen3_5", "gpt_oss")),
            ("renderer", ("qwen3", "qwen3_5", "gpt_oss")),
            ("tool_format", ("qwen3_json", "qwen3_5_xml", "harmony")),
            ("local", ("yes", "no", "unverified")),
        ):
            if getattr(self, field) not in choices:
                raise ValueError(f"{field} must be one of {choices}")
        for field in ("max_ctx", "default_max_tokens", "tinker_max_ctx"):
            value = getattr(self, field)
            if field == "tinker_max_ctx" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.thinking, bool):
            raise ValueError("thinking must be a boolean")

        tinker_fields = (self.tinker_id, self.tinker_max_ctx, self.tinker_prices)
        if any(value is not None for value in tinker_fields) and any(
            value is None for value in tinker_fields
        ):
            raise ValueError(
                "tinker_id, tinker_max_ctx, and tinker_prices must be all set or all None"
            )
        if isinstance(self.tinker_prices, Mapping):
            try:
                prices = TinkerPrices(**self.tinker_prices)
            except TypeError as exc:
                raise ValueError(f"invalid tinker_prices: {exc}") from exc
            object.__setattr__(self, "tinker_prices", prices)
        elif self.tinker_prices is not None and not isinstance(self.tinker_prices, TinkerPrices):
            raise ValueError("tinker_prices must be a mapping or TinkerPrices")


MODELS: Registry[ModelSpec] = Registry("models", Path(__file__).parent / "models", ModelSpec)


def load_model(name: str) -> ModelSpec:
    """Load a validated model by its registry name."""
    return MODELS.load(name)


def list_models() -> list[str]:
    """Return available model names in sorted order."""
    return MODELS.names()


def for_hf_id(hf_id: str) -> ModelSpec:
    """Resolve one HF identifier, rejecting missing or ambiguous registry entries."""
    matches = [model for model in MODELS.load_all().values() if model.hf_id == hf_id]
    if not matches:
        raise ConfigError(f"no model registered for HF id {hf_id!r}")
    if len(matches) > 1:
        raise ConfigError(
            f"duplicate HF id {hf_id!r} in model entries: {[model.name for model in matches]}"
        )
    return matches[0]


def tinker_models() -> list[ModelSpec]:
    """Return Tinker-supported models in registry name order."""
    return [model for model in MODELS.load_all().values() if model.tinker_id is not None]
