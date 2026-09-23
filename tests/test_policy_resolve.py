"""Resolution validates seating before constructing only the requested backend."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from marli.budget import SpendGuard
from marli.errors import ConfigError
from marli.llm.client import ChatClient, Endpoint
from marli.model import load_model
from marli.policy import tinker, vllm
from marli.policy.api import APIPolicy
from marli.policy.refs import PolicyRef, parse_ref
from marli.policy.resolve import resolve_policy
from marli.policy.scripted import ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter"])
@pytest.mark.parametrize("as_string", [True, False])
async def test_api_dispatch(
    monkeypatch: pytest.MonkeyPatch, provider: str, as_string: bool
) -> None:
    client = ChatClient(Endpoint("https://api.invalid", "org/model"))
    factory = Mock(return_value=client)
    monkeypatch.setattr(ChatClient, provider, factory)
    ref = f"api:{provider}/org/model"
    prices = {f"{provider}/org/model": {"output": 3}}
    spend = SpendGuard(1)
    try:
        policy = await resolve_policy(
            ref if as_string else parse_ref(ref),
            policy_id="teacher",
            trainable=False,
            renderer_name=None,
            spend=spend,
            api_prices=prices,
        )
        assert isinstance(policy, APIPolicy)
        assert policy.policy_id == "teacher" and not policy.trainable
        assert policy.client is client and policy.provider == provider
        assert policy.spend is spend and policy.api_prices == prices
        factory.assert_called_once_with("org/model")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "ref,base_url,sampler",
    [
        ("tinker:org/base", None, None),
        ("tinker@http://local:8000/api|org/base", "http://local:8000/api", None),
        ("tinker:org/base#sampler=tinker://saved/weights", None, "tinker://saved/weights"),
    ],
)
async def test_tinker_dispatch(
    monkeypatch: pytest.MonkeyPatch, ref: str, base_url: str | None, sampler: str | None
) -> None:
    service = Mock()
    make = Mock(return_value=service)
    client = SimpleNamespace(sample_async=AsyncMock())
    sampling = AsyncMock(return_value=client)
    monkeypatch.setattr(tinker, "make_service_client", make)
    monkeypatch.setattr(tinker, "sampling_client_for", sampling)
    model, spend = load_model("qwen3_8b"), SpendGuard(1)
    policy = await resolve_policy(
        ref, policy_id="learner", trainable=True, renderer_name="fake", model=model, spend=spend
    )
    assert isinstance(policy, tinker.TinkerPolicy)
    assert policy.policy_id == "learner" and policy.trainable
    assert policy.renderer_name == "fake" and policy.sampling_client is client
    assert policy.model is model and policy.spend is spend
    make.assert_called_once_with(base_url)
    sampling.assert_awaited_once_with(
        service, model_path=sampler, base_model="org/base" if sampler is None else None
    )


@pytest.mark.parametrize("manifest", [False, True])
async def test_vllm_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, manifest: bool
) -> None:
    factory = Mock()
    monkeypatch.setattr(vllm, "VLLMPolicy", factory)
    ref = "vllm:http://local:8000#learner@1"
    if manifest:
        path = tmp_path / "server.json"
        path.write_text(json.dumps({"base_url": "http://local:8000", "models": ["learner@1"]}))
        ref = f"vllm:@{path}#learner@1"
    policy = await resolve_policy(ref, policy_id="learner", trainable=True, renderer_name="fake")
    assert policy is factory.return_value
    factory.assert_called_once_with(
        "learner", "http://local:8000", "learner@1", renderer_name="fake", trainable=True
    )


async def test_scripted_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = FakeRenderer()

    def turn(ctx: Any) -> Turn:
        return Turn(content="ok")

    expected = ScriptedPolicy("scripted", renderer, from_callable(turn, renderer))
    module = ModuleType("policy_test_factory")
    module.factory = Mock(return_value=expected)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = await resolve_policy(
        "scripted:policy_test_factory:factory",
        policy_id="ignored",
        trainable=False,
        renderer_name=None,
    )
    assert result is expected
    module.factory.assert_called_once_with()


@pytest.mark.parametrize(
    "ref,trainable,renderer,error",
    [
        ("api:openai/model", True, None, "cannot be trainable"),
        ("ckpt:checkpoint", False, None, r"ckpt: refs are resolved by marli.train \(M3\)"),
        ("tinker:base", False, None, "require renderer_name"),
        ("vllm:http://local#model", True, "", "require renderer_name"),
        (PolicyRef("unknown", "x"), False, None, "Unknown policy kind"),
        (PolicyRef("api", "x", provider="unknown"), False, None, "Unknown API provider"),
        (PolicyRef("vllm", "x"), False, "fake", "require base_url"),
        ("invalid", False, None, "Invalid policy ref"),
    ],
)
async def test_invalid_resolution_is_early(
    monkeypatch: pytest.MonkeyPatch,
    ref: PolicyRef | str,
    trainable: bool,
    renderer: str | None,
    error: str,
) -> None:
    make = Mock(side_effect=AssertionError("backend must not be constructed"))
    monkeypatch.setattr(tinker, "make_service_client", make)
    monkeypatch.setattr(vllm, "VLLMPolicy", make)
    monkeypatch.setattr(ChatClient, "openai", make)
    with pytest.raises(ConfigError, match=error):
        await resolve_policy(ref, policy_id="p", trainable=trainable, renderer_name=renderer)
    make.assert_not_called()


async def test_unserved_model_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "server.json"
    path.write_text(json.dumps({"base_url": "http://local", "models": ["base"]}))
    factory = Mock()
    monkeypatch.setattr(vllm, "VLLMPolicy", factory)
    with pytest.raises(ConfigError, match="not served"):
        await resolve_policy(
            f"vllm:@{path}#missing", policy_id="p", trainable=False, renderer_name="fake"
        )
    factory.assert_not_called()


def test_policy_imports_remain_light(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] not in {"tinker", "torch", "transformers", "vllm", "datasets"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for name in ("api", "tinker", "vllm", "resolve"):
        importlib.reload(importlib.import_module(f"marli.policy.{name}"))
