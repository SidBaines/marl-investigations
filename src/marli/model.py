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
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} price must be finite and non-negative, got {value!r}")
        date_error = (
            "as_of must be a quoted ISO date string 'YYYY-MM-DD', "
            f"got {self.as_of!r}"
        )
        if not isinstance(self.as_of, str):
            raise ValueError(date_error)
        try:
            canonical_date = date.fromisoformat(self.as_of).isoformat()
        except ValueError as exc:
            raise ValueError(date_error) from exc
        if canonical_date != self.as_of:
            raise ValueError(date_error)



def _renderer_names() -> tuple[str, ...]:
    from marli.render.registry import renderer_names

    return renderer_names()


@dataclass(frozen=True)
class ModelSpec:
    """Reproducible model identity, rendering, backend capabilities and defaults.

    ``max_ctx`` caps prompt plus completion and must fit the model and selected
    backend's limits. Tinker limits are checked here; local limits also depend
    on the server configuration. Tinker fields are all set or all None.
    """

    name: str  # registry key, identical to the YAML filename stem
    hf_id: str  # Hugging Face model identifier for weights and tokenizer
    family: str  # model family: qwen3 | qwen3_5 | gpt_oss
    renderer: str  # default token renderer name (marli.render.registry.renderer_names())
    architecture: str  # Hugging Face architecture class name
    max_ctx: int  # default prompt+completion cap; <= model and backend limits
    default_max_tokens: int  # default completion token cap per call
    thinking: bool  # whether the model uses a reasoning channel
    tool_format: str  # native tool syntax: qwen3_json | qwen3_5_xml | harmony
    tinker_id: str | None = None  # Tinker base model identifier; None = not on Tinker
    tinker_max_ctx: int | None = None  # Tinker's prompt+completion token limit
    tinker_prices: TinkerPrices | None = None  # dated USD per million Tinker tokens
    # Local learner support: yes = supported, no = unsupported, unverified = untested.
    local: str = "unverified"
    notes: str = ""  # capability caveats and provenance for this catalog entry

    def __post_init__(self) -> None:
        for field, choices in (
            ("family", ("qwen3", "qwen3_5", "gpt_oss")),
            ("renderer", _renderer_names()),
            ("tool_format", ("qwen3_json", "qwen3_5_xml", "harmony")),
            ("local", ("yes", "no", "unverified")),
        ):
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(f"{field} must be one of {choices}, got {value!r}")
        for field in ("max_ctx", "default_max_tokens", "tinker_max_ctx"):
            value = getattr(self, field)
            if field == "tinker_max_ctx" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")
        if not isinstance(self.thinking, bool):
            raise ValueError(f"thinking must be a boolean, got {self.thinking!r}")

        tinker_fields = (self.tinker_id, self.tinker_max_ctx, self.tinker_prices)
        if any(value is not None for value in tinker_fields) and any(
            value is None for value in tinker_fields
        ):
            raise ValueError(
                "tinker_id, tinker_max_ctx, and tinker_prices must be all set or all None, "
                f"got {tinker_fields!r}"
            )
        if self.tinker_max_ctx is not None and self.max_ctx > self.tinker_max_ctx:
            raise ValueError(
                f"max_ctx must be <= tinker_max_ctx ({self.tinker_max_ctx!r}), "
                f"got {self.max_ctx!r}"
            )
        if isinstance(self.tinker_prices, Mapping):
            try:
                prices = TinkerPrices(**self.tinker_prices)
            except TypeError as exc:
                raise ValueError(
                    f"invalid tinker_prices: {exc}; got {self.tinker_prices!r}"
                ) from exc
            object.__setattr__(self, "tinker_prices", prices)
        elif self.tinker_prices is not None and not isinstance(self.tinker_prices, TinkerPrices):
            raise ValueError(
                f"tinker_prices must be a mapping or TinkerPrices, got {self.tinker_prices!r}"
            )


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
