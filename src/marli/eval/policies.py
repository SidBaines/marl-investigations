"""Evaluation seats share policies and tokenizers, but never renderer state."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

from marli.budget import SpendGuard
from marli.errors import ConfigError
from marli.model import MODELS, ModelSpec, load_model
from marli.policy.base import Policy, TokenPolicy
from marli.policy.refs import PolicyRef, parse_ref
from marli.policy.resolve import resolve_policy
from marli.render.base import DeltaRenderer
from marli.render.registry import get_renderer, renderer_names


@dataclass
class SamplingOverrides:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ConfigError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ConfigError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ConfigError("top_k must be -1 or a positive integer")


@dataclass
class PolicySpec:
    ref: str
    renderer: str | None = None
    model: str | None = None
    trainable: bool = False
    sampling: SamplingOverrides = field(default_factory=SamplingOverrides)

    def __post_init__(self) -> None:
        if self.trainable:
            raise ConfigError("evaluation policies must have trainable=False")
        ref = parse_ref(self.ref)
        if ref.base_url and (urlsplit(ref.base_url).query or urlsplit(ref.base_url).fragment):
            raise ConfigError("policy URLs must not contain query strings or fragments")
        if self.renderer is not None and self.renderer not in renderer_names():
            raise ConfigError(f"unknown renderer {self.renderer!r}")
        self.sampling.__post_init__()


def _served_base(server_json: str) -> str | None:
    """The HF id a marli ``serve vllm`` manifest serves, or None if unreadable."""
    try:
        data = json.loads(Path(server_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    hf_id = data.get("hf_id") if isinstance(data, dict) else None
    return hf_id if isinstance(hf_id, str) and hf_id else None


def resolve_spec(spec: PolicySpec) -> tuple[PolicyRef, ModelSpec | None, str | None]:
    """Resolve catalog defaults without constructing a backend or tokenizer."""
    spec.__post_init__()
    ref = parse_ref(spec.ref, resolve_paths=True)
    if ref.kind == "ckpt":
        # A trained sampler: resolve to the concrete ref it points at (model and
        # renderer defaults follow the base model), e.g. tinker:<base>#sampler=…
        from marli.train.checkpoint import resolve_checkpoint_ref

        ref = parse_ref(resolve_checkpoint_ref(ref, ref.learner), resolve_paths=True)
    model = load_model(spec.model) if spec.model else None
    if model is None and ref.kind in {"tinker", "vllm", "api"}:
        entries = list(MODELS.load_all().values())
        target = ref.target
        if ref.kind == "vllm" and ref.server_json and all(e.hf_id != target for e in entries):
            # A LoRA adapter on a marli-served vLLM inherits the server's base model.
            target = _served_base(ref.server_json) or target
        matches = [entry for entry in entries if entry.hf_id == target]
        if len(matches) > 1:
            raise ConfigError(f"ambiguous model registry HF id {target!r}")
        model = next(iter(matches), None)
    renderer = spec.renderer or (model.renderer if model and ref.kind != "api" else None)
    if ref.kind in {"tinker", "vllm"} and renderer is None:
        raise ConfigError(f"policy {spec.ref!r} needs a renderer or registered model")
    if ref.kind == "tinker" and (model is None or model.tinker_prices is None):
        raise ConfigError("Tinker evaluation needs a registered model with prices")
    if ref.kind == "api" and (spec.sampling.top_p != 1 or spec.sampling.top_k != -1):
        raise ConfigError("chat policies support only temperature sampling overrides")
    return ref, model, renderer


async def build_policies(
    specs: dict[str, PolicySpec], *, spend: SpendGuard
) -> tuple[dict[str, Policy], dict[str, Callable[[], DeltaRenderer]]]:
    resolved = {name: resolve_spec(spec) for name, spec in specs.items()}
    for name, (_, model, renderer) in resolved.items():
        if model and renderer and renderer != model.renderer:
            warnings.warn(
                f"policy {name!r}: renderer {renderer!r} disagrees with model "
                f"{model.name!r} default {model.renderer!r}",
                stacklevel=2,
            )
    policies: dict[str, Policy] = {}
    renderers: dict[str, Callable[[], DeltaRenderer]] = {}
    for name, (ref, model, renderer) in resolved.items():
        policy = await resolve_policy(
            ref, policy_id=name, trainable=False, renderer_name=renderer, model=model, spend=spend
        )
        if isinstance(policy, TokenPolicy):
            renderer = renderer or policy.renderer_name
            renderers[name] = partial(
                get_renderer, renderer, hf_id=model.hf_id if model else ref.target
            )
        elif specs[name].sampling.top_p != 1 or specs[name].sampling.top_k != -1:
            raise ConfigError("chat policies support only temperature sampling overrides")
        policies[name] = policy
    return policies, renderers
