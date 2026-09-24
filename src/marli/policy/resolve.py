"""Resolve config identities only when a backend is actually needed."""

from __future__ import annotations

from marli.budget import SpendGuard
from marli.errors import ConfigError
from marli.model import ModelSpec
from marli.policy import refs
from marli.policy.base import Policy
from marli.policy.refs import PolicyRef


async def resolve_policy(
    ref: PolicyRef | str,
    *,
    policy_id: str,
    trainable: bool,
    renderer_name: str | None,
    model: ModelSpec | None = None,
    spend: SpendGuard | None = None,
    api_prices: dict[str, dict[str, float]] | None = None,
) -> Policy:
    if isinstance(ref, str):
        ref = refs.parse_ref(ref)
    if ref.kind == "ckpt":
        from marli.train.checkpoint import resolve_checkpoint_ref

        ref = refs.parse_ref(resolve_checkpoint_ref(ref, ref.learner))
    if ref.kind == "scripted":
        policy = refs.resolve_scripted(ref)
        if policy.trainable != trainable:
            raise ConfigError("scripted policy trainable disagrees with the requested trainable")
        if renderer_name is not None and getattr(policy, "renderer_name", None) != renderer_name:
            raise ConfigError(
                "scripted policy renderer_name disagrees with requested renderer_name"
            )
        return policy
    if ref.kind == "api":
        if trainable:
            raise ConfigError("API policies cannot be trainable")
        from marli.llm.client import ChatClient
        from marli.policy.api import APIPolicy

        factories = {
            "openrouter": ChatClient.openrouter,
            "anthropic": ChatClient.anthropic,
            "openai": ChatClient.openai,
        }
        if ref.provider not in factories:
            raise ConfigError(f"Unknown API provider {ref.provider!r}")
        return APIPolicy(
            policy_id,
            factories[ref.provider](ref.target),
            provider=ref.provider,
            spend=spend,
            api_prices=api_prices,
        )
    if ref.kind not in ("tinker", "vllm"):
        raise ConfigError(f"Unknown policy kind {ref.kind!r}")
    if not renderer_name:
        raise ConfigError(f"{ref.kind} token policies require renderer_name")
    if ref.kind == "tinker":
        if spend is not None and model is None:
            raise ConfigError("Tinker refs with spend require a model for pricing")
        from marli.policy.tinker import TinkerPolicy, make_service_client, sampling_client_for

        service = make_service_client(ref.base_url)
        client = await sampling_client_for(
            service,
            model_path=ref.sampler,
            base_model=None if ref.sampler is not None else ref.target,
        )
        return TinkerPolicy(
            policy_id,
            client,
            renderer_name=renderer_name,
            trainable=trainable,
            model=model,
            spend=spend,
        )

    from marli.policy.vllm import VLLMPolicy, read_server_json

    base_url = ref.base_url
    if ref.server_json is not None:
        base_url, models = read_server_json(ref.server_json)
        if ref.target not in models:
            raise ConfigError(f"vLLM model {ref.target!r} is not served by {ref.server_json}")
    if not base_url:
        raise ConfigError("vLLM refs require base_url or server_json")
    return VLLMPolicy(
        policy_id,
        base_url,
        ref.target,
        renderer_name=renderer_name,
        trainable=trainable,
    )
