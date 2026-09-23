"""A fake SDK pins the installed Tinker signatures without loading heavy modules."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from marli.budget import SpendGuard, tinker_cost
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.interact.types import Termination
from marli.model import load_model
from marli.policy.base import CallMeta, SamplingSpec, TokenPolicy
from marli.policy.tinker import TinkerPolicy, make_service_client, sampling_client_for
from marli.render.base import Msg
from marli.render.fake import FakeRenderer


class APIConnectionError(Exception):
    pass


class APIStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RetryableException(Exception):
    pass


@dataclass(frozen=True)
class ModelInput:
    tokens: tuple[int, ...]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> ModelInput:
        assert isinstance(tokens, list)
        return cls(tuple(tokens))


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: list[int]
    seed: int


@dataclass(frozen=True)
class RetryConfig:
    enable_retry_logic: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        asyncio.TimeoutError,
        APIConnectionError,
        httpx.TimeoutException,
        RetryableException,
    )


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    sdk = ModuleType("tinker")
    sdk.ModelInput = ModelInput
    sdk.SamplingParams = SamplingParams
    sdk.APIStatusError = APIStatusError
    sdk.ServiceClient = Mock()
    retry = ModuleType("tinker.lib.retry_handler")
    retry.RetryConfig = RetryConfig

    def is_retryable_status_code(status_code: int) -> bool:
        return status_code in (408, 409, 429) or 500 <= status_code < 600

    retry.is_retryable_status_code = is_retryable_status_code
    base_url = ModuleType("tinker.lib.base_url")
    base_url.DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"
    monkeypatch.setitem(sys.modules, "tinker", sdk)
    monkeypatch.setitem(sys.modules, "tinker.lib.retry_handler", retry)
    monkeypatch.setitem(sys.modules, "tinker.lib.base_url", base_url)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    return sdk


def response(stop_reason: str = "stop") -> SimpleNamespace:
    ids = FakeRenderer().encode_completion("ok", stop=stop_reason == "stop")
    return SimpleNamespace(
        sequences=[
            SimpleNamespace(
                tokens=ids,
                logprobs=[-0.5] * len(ids),
                stop_reason=stop_reason,
            )
        ]
    )


@pytest.mark.parametrize(
    "reason,termination",
    [
        ("stop", Termination.STOP),
        ("length", Termination.LENGTH),
        ("other", Termination.ERROR),
    ],
)
async def test_exact_sampling_request_and_response(
    sdk: ModuleType, reason: str, termination: Termination
) -> None:
    renderer = FakeRenderer()
    prompt = renderer.initial("system", [], [Msg("user", "query")])
    result = response(reason)

    async def sample_async(
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        topk_sample_logprobs: int = 0,
        target_prompt_logprobs: Any = None,
    ) -> SimpleNamespace:
        assert prompt == ModelInput(tuple(original_prompt))
        assert num_samples == 1
        assert sampling_params == SamplingParams(20, 0.7, 0.8, 5, list(renderer.stop_token_ids), 42)
        return result

    original_prompt = list(prompt)
    client = SimpleNamespace(sample_async=sample_async)
    policy = TinkerPolicy(
        "p", client, renderer_name=renderer.name, trainable=False, policy_version=7
    )
    assert isinstance(policy, TokenPolicy)
    sample = await policy.sample(
        prompt,
        SamplingSpec(20, 0.7, 0.8, 5, renderer.stop_token_ids),
        seed=42,
        meta=CallMeta(agent_id="peer"),
    )
    assert sample.completion_ids == tuple(result.sequences[0].tokens)
    assert sample.logprobs == tuple(result.sequences[0].logprobs)
    assert (sample.termination, sample.policy_version) == (termination, 7)
    assert sample.usage.prompt_tokens == len(prompt)
    assert sample.usage.completion_tokens == len(sample.completion_ids)
    assert sample.usage.tokenizer == "fake"
    assert prompt == original_prompt


async def test_spend_preflight_and_actual_charge(sdk: ModuleType) -> None:
    renderer = FakeRenderer()
    prompt = renderer.initial(None, [], [Msg("user", "hi")])
    client = SimpleNamespace(sample_async=AsyncMock(return_value=response()))
    model = load_model("qwen3_8b")
    guard = Mock(wraps=SpendGuard(1))
    policy = TinkerPolicy(
        "p", client, renderer_name="fake", trainable=True, model=model, spend=guard
    )
    sample = await policy.sample(prompt, SamplingSpec(20), seed=1)
    expected = tinker_cost(model, prefill=len(prompt), sample=len(sample.completion_ids))
    guard.check.assert_called_once_with(tinker_cost(model, prefill=len(prompt), sample=20), "p")
    guard.charge.assert_called_once_with(expected, "p")
    assert sample.usage.cost_usd == expected and guard._mock_wraps.spent == expected
    policy.spend = SpendGuard(0)
    with pytest.raises(BudgetExceededError):
        await policy.sample(prompt, SamplingSpec(20), seed=1)
    assert client.sample_async.await_count == 1


@pytest.mark.parametrize("logprobs", [None, [], [-1.0]])
async def test_misaligned_logprobs_fail(sdk: ModuleType, logprobs: list[float] | None) -> None:
    result = response()
    result.sequences[0].logprobs = logprobs
    client = SimpleNamespace(sample_async=AsyncMock(return_value=result))
    policy = TinkerPolicy("p", client, renderer_name="fake", trainable=True)
    with pytest.raises(BackendError, match="length mismatch"):
        await policy.sample(FakeRenderer().encode_text("prompt"), SamplingSpec(20), seed=1)
    assert client.sample_async.await_count == 1


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        ConnectionError(),
        APIConnectionError(),
        httpx.ReadTimeout("timeout"),
        RetryableException(),
        APIStatusError(408),
        APIStatusError(409),
        APIStatusError(429),
        APIStatusError(503),
    ],
)
async def test_transient_retries_preserve_request(
    sdk: ModuleType, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    client = SimpleNamespace(sample_async=AsyncMock(side_effect=[error, error, response()]))
    policy = TinkerPolicy("p", client, renderer_name="fake", trainable=True)
    sample = await policy.sample(FakeRenderer().encode_text("prompt"), SamplingSpec(20), seed=1)
    assert sample.termination == Termination.STOP
    calls = client.sample_async.call_args_list
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert [call.args for call in sleep.call_args_list] == [(1.0,), (2.0,)]


@pytest.mark.parametrize("error,attempts", [(APIStatusError(400), 1), (TimeoutError("gone"), 3)])
async def test_failures_are_bounded(
    sdk: ModuleType, monkeypatch: pytest.MonkeyPatch, error: Exception, attempts: int
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    client = SimpleNamespace(sample_async=AsyncMock(side_effect=error))
    with pytest.raises(BackendError) as caught:
        await TinkerPolicy("p", client, renderer_name="fake", trainable=True).sample(
            FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1
        )
    assert caught.value.__cause__ is error
    assert client.sample_async.await_count == attempts and sleep.await_count == attempts - 1


async def test_cancellation_is_not_retried(sdk: ModuleType) -> None:
    client = SimpleNamespace(sample_async=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await TinkerPolicy("p", client, renderer_name="fake", trainable=True).sample(
            FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1
        )
    assert client.sample_async.await_count == 1


@pytest.mark.parametrize(
    "spec",
    [SamplingSpec(10, temperature=0.5), SamplingSpec(10, top_p=0.9), SamplingSpec(10, top_k=5)],
)
async def test_trainable_guard_before_sdk(sdk: ModuleType, spec: SamplingSpec) -> None:
    client = SimpleNamespace(sample_async=AsyncMock())
    with pytest.raises(ConfigError, match="raw|off-policy"):
        await TinkerPolicy("p", client, renderer_name="fake", trainable=True).sample(
            FakeRenderer().encode_text("p"), spec, seed=1
        )
    client.sample_async.assert_not_called()


@pytest.mark.parametrize(
    "explicit,env,allowed",
    [
        (None, None, True),
        ("http://local:8000", None, True),
        ("http://local:8000", "http://local:8000", True),
        (None, "http://local:8000", False),
        ("http://local:8000", "https://cloud.invalid", False),
        (None, "", False),
    ],
)
def test_explicit_endpoint_guard(
    sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    env: str | None,
    allowed: bool,
) -> None:
    if env is not None:
        monkeypatch.setenv("TINKER_BASE_URL", env)
    if not allowed:
        with pytest.raises(ConfigError, match="TINKER_BASE_URL"):
            make_service_client(explicit)
        sdk.ServiceClient.assert_not_called()
        return
    assert make_service_client(explicit) is sdk.ServiceClient.return_value
    sdk.ServiceClient.assert_called_once_with(
        base_url=explicit or "https://tinker.thinkingmachines.dev/services/tinker-prod",
        max_retries=0,
    )


@pytest.mark.parametrize("kwargs", [{"base_model": "base"}, {"model_path": "tinker://saved"}])
async def test_sampling_client_constructor(sdk: ModuleType, kwargs: dict[str, str]) -> None:
    service = SimpleNamespace(create_sampling_client_async=AsyncMock())
    result = await sampling_client_for(service, **kwargs)
    assert result is service.create_sampling_client_async.return_value
    service.create_sampling_client_async.assert_awaited_once_with(
        base_model=kwargs.get("base_model"),
        model_path=kwargs.get("model_path"),
        retry_config=RetryConfig(enable_retry_logic=False),
    )
