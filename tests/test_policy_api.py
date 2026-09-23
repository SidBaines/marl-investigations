"""API seats preserve tool history and independent samples over a mocked wire."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from marli.budget import SpendGuard
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.interact.types import Termination, Usage
from marli.llm.client import OPENROUTER_BASE_URL, ChatClient, Endpoint, UnsupportedRequestError
from marli.policy.api import APIPolicy
from marli.policy.base import CallMeta, ChatPolicy
from marli.render.base import Msg, ToolCall, ToolSpec


@pytest.fixture
async def wire() -> AsyncIterator[tuple[ChatClient, list[dict[str, Any]], list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = ChatClient(Endpoint("https://api.invalid/v1", "model"), cache="memory")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        yield client, requests, responses
    finally:
        await client.aclose()


def reply(reason: str | None = "stop", **message: Any) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "answer", **message}, "finish_reason": reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


async def test_message_and_tool_conversion_both_ways(wire: tuple) -> None:
    client, requests, responses = wire
    raw_call = {
        "id": "native-id",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key": "4"}'},
    }
    responses.append(
        reply("tool_calls", content=None, reasoning_content="thinking", tool_calls=[raw_call])
    )
    messages = [
        Msg("system", "system"),
        Msg("user", "query"),
        Msg(
            "assistant",
            "checking",
            tool_calls=(ToolCall("lookup", {"key": "a"}, "kept"), ToolCall("lookup", {})),
        ),
        Msg("tool", "first", name="lookup", tool_call_id="kept"),
        Msg("tool", "second", name="lookup"),
    ]
    tool = ToolSpec("lookup", "Look up a key", {"type": "object"})
    policy = APIPolicy("teacher", client)
    result = await policy.chat(
        messages, [tool], max_tokens=40, temperature=0.7, seed=21, cache_salt="call/0"
    )
    assert isinstance(policy, ChatPolicy)
    assert policy.trainable is False
    sent = requests[0]
    assert sent.keys() == {"messages", "tools", "max_tokens", "temperature", "seed", "model"}
    assert (sent["model"], sent["seed"], sent["max_tokens"], sent["temperature"]) == (
        "model",
        21,
        40,
        0.7,
    )
    assert sent["messages"][:2] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "query"},
    ]
    calls = sent["messages"][2]["tool_calls"]
    assert calls[0] == {
        "id": "kept",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key": "a"}'},
    }
    assert calls[1]["id"] and calls[1]["id"] != "kept"
    assert sent["messages"][3:] == [
        {"role": "tool", "tool_call_id": "kept", "content": "first"},
        {"role": "tool", "tool_call_id": calls[1]["id"], "content": "second"},
    ]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a key",
                "parameters": tool.parameters,
            },
        }
    ]
    assert (result.content, result.thinking, result.termination) == (
        "",
        "thinking",
        Termination.STOP,
    )
    (parsed,) = result.tool_calls
    assert (parsed.name, parsed.arguments, parsed.id, parsed.ok) == (
        "lookup",
        {"key": "4"},
        "native-id",
        True,
    )
    assert json.loads(parsed.raw) == raw_call
    assert result.raw["choices"][0]["message"]["tool_calls"] == [raw_call]
    responses.append(reply())
    await policy.chat(
        messages, [tool], max_tokens=40, temperature=0.7, seed=21, cache_salt="call/1"
    )
    assert requests[0] == requests[1]
    assert messages[2].tool_calls[1].id is None


@pytest.mark.parametrize("arguments", ["{bad", "[]", "null", "4", '"text"'])
async def test_bad_tool_json_is_retained(wire: tuple, arguments: str) -> None:
    client, _, responses = wire
    call = {"id": "broken", "function": {"name": "lookup", "arguments": arguments}}
    responses.append(reply("tool_calls", tool_calls=[call]))
    result = await APIPolicy("p", client).chat(
        [], [], max_tokens=4, temperature=1, seed=1, cache_salt="x"
    )
    (parsed,) = result.tool_calls
    assert not parsed.ok and parsed.arguments is None
    assert parsed.id == "broken" and json.loads(parsed.raw) == call


@pytest.mark.parametrize(
    "reason,termination",
    [
        ("stop", Termination.STOP),
        ("tool_calls", Termination.STOP),
        ("length", Termination.LENGTH),
        ("content_filter", Termination.ERROR),
        ("unknown", Termination.ERROR),
        (None, Termination.ERROR),
    ],
)
async def test_termination_mapping(
    wire: tuple, reason: str | None, termination: Termination
) -> None:
    client, _, responses = wire
    responses.append(reply(reason))
    result = await APIPolicy("p", client).chat(
        [], [], max_tokens=4, temperature=1, seed=1, cache_salt="x"
    )
    assert result.termination == termination
    assert result.raw["choices"][0]["finish_reason"] == reason


@pytest.mark.parametrize(
    "reason,termination,normalized",
    [
        ("end_turn", Termination.STOP, "stop"),
        ("stop_sequence", Termination.STOP, "stop"),
        ("max_tokens", Termination.LENGTH, "length"),
        ("tool_use", Termination.STOP, "tool_calls"),
        ("refusal", Termination.ERROR, "content_filter"),
        ("unknown", Termination.ERROR, "unknown"),
    ],
)
async def test_anthropic_termination_through_transport(
    wire: tuple, reason: str, termination: Termination, normalized: str
) -> None:
    client, requests, responses = wire
    client.endpoint = Endpoint("https://anthropic.invalid", "model", provider="anthropic")
    content = (
        [{"type": "tool_use", "id": "call", "name": "lookup", "input": {"key": "a"}}]
        if reason == "tool_use"
        else [{"type": "text", "text": "answer"}]
    )
    responses.append(
        {
            "content": content,
            "stop_reason": reason,
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
    )
    result = await APIPolicy("p", client).chat(
        [Msg("user", "query")], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
    )
    assert len(requests) == 1
    assert result.termination == termination
    assert result.raw["choices"][0]["finish_reason"] == normalized
    if reason == "tool_use":
        assert result.tool_calls[0].id == "call"
        assert result.tool_calls[0].arguments == {"key": "a"}


async def test_usage_price_estimate_and_actual_charge(wire: tuple) -> None:
    client, _, responses = wire
    responses.append(reply())
    responses[0]["usage"].update(cost=0.03, prompt_tokens_details={"cached_tokens": 5})
    guard = Mock(wraps=SpendGuard(1))
    policy = APIPolicy(
        "p",
        client,
        provider="openrouter",
        spend=guard,
        api_prices={"openrouter/model": {"input": 1000, "output": 2000}},
    )
    result = await policy.chat([], [], max_tokens=40, temperature=1, seed=1, cache_salt="x")
    assert result.usage == Usage(12, 4, 5, 0.03, "api:openrouter/model")
    guard.check.assert_called_once_with(0.08, "p")
    guard.charge.assert_called_once_with(0.03, "p")
    assert guard._mock_wraps.spent == 0.03


async def test_preflight_budget_prevents_request(wire: tuple) -> None:
    client, requests, _ = wire
    policy = APIPolicy(
        "p", client, spend=SpendGuard(0), api_prices={"openai/model": {"input": 1, "output": 1}}
    )
    with pytest.raises(BudgetExceededError):
        await policy.chat([], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    assert not requests


async def test_no_price_still_charges_actual_cost(wire: tuple) -> None:
    client, requests, responses = wire
    responses.append(reply())
    responses[0]["usage"]["cost"] = 0.2
    guard = SpendGuard(0.1)
    with pytest.raises(BudgetExceededError):
        await APIPolicy("p", client, provider="openrouter", spend=guard).chat(
            [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
        )
    assert len(requests) == 1 and guard.spent == 0.2


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter"])
@pytest.mark.parametrize("reported_cost", [None, 0.0, 0.03])
async def test_cost_accounting_for_every_provider(
    wire: tuple, provider: str, reported_cost: float | None
) -> None:
    client, _, responses = wire
    if provider == "anthropic":
        client.endpoint = Endpoint("https://anthropic.invalid", "model", provider="anthropic")
        data = {
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }
    else:
        data = reply()
    if reported_cost is not None:
        data["usage"]["cost"] = reported_cost
    responses.append(data)
    guard = SpendGuard(1)
    result = await APIPolicy(
        "p",
        client,
        provider=provider,
        spend=guard,
        api_prices={f"{provider}/model": {"input": 2, "output": 3}},
    ).chat([], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    expected = (12 * 2 + 4 * 3) / 1e6 if reported_cost is None else reported_cost
    assert result.usage.cost_usd == pytest.approx(expected)
    assert guard.spent == pytest.approx(expected)


async def test_price_cost_is_recorded_without_spend_guard(wire: tuple) -> None:
    client, _, responses = wire
    responses.append(reply())
    result = await APIPolicy(
        "p", client, api_prices={"openai/model": {"input": 2, "output": 3}}
    ).chat([], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    assert result.usage.cost_usd == pytest.approx(36 / 1e6)


async def test_price_based_actual_charge_can_exceed_budget(wire: tuple) -> None:
    client, requests, responses = wire
    responses.append(reply())
    guard = SpendGuard(30 / 1e6)
    policy = APIPolicy(
        "p", client, spend=guard, api_prices={"openai/model": {"input": 2, "output": 3}}
    )
    with pytest.raises(BudgetExceededError):
        await policy.chat([], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    assert len(requests) == 1
    assert guard.spent == pytest.approx(36 / 1e6)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_finite_budget_requires_pricing_at_construction(wire: tuple, provider: str) -> None:
    client, requests, _ = wire
    with pytest.raises(ConfigError, match=f"price entry for {provider}/model"):
        APIPolicy("p", client, provider=provider, spend=SpendGuard(1))
    assert not requests
    APIPolicy("p", client, provider=provider, spend=SpendGuard(None))
    APIPolicy("p", client, provider=provider)


async def test_openrouter_inferred_provider_can_use_reported_cost(wire: tuple) -> None:
    client, _, responses = wire
    client.endpoint = Endpoint(OPENROUTER_BASE_URL + "/", "model")
    data = reply()
    data["usage"]["cost"] = 0.01
    responses.append(data)
    guard = SpendGuard(1)
    result = await APIPolicy("p", client, spend=guard).chat(
        [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
    )
    assert result.usage.cost_usd == guard.spent == 0.01


async def test_openrouter_missing_cost_with_finite_budget_fails(wire: tuple) -> None:
    client, _, responses = wire
    responses.append(reply())
    with pytest.raises(BackendError, match="no cost or configured price"):
        await APIPolicy("p", client, provider="openrouter", spend=SpendGuard(1)).chat(
            [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
        )


async def test_salt_pass_through_and_independent_samples(wire: tuple) -> None:
    client, requests, responses = wire
    responses.extend([reply(content="first"), reply(content="second")])
    client.chat = AsyncMock(wraps=client.chat)
    policy = APIPolicy("p", client)
    outputs = []
    for salt in ("a", "b", "a"):
        result = await policy.chat(
            [],
            [],
            max_tokens=10,
            temperature=1,
            seed=1,
            cache_salt=salt,
            meta=CallMeta(agent_id=salt),
        )
        outputs.append(result.content)
    assert outputs == ["first", "second", "first"] and len(requests) == 2
    assert [call.kwargs for call in client.chat.call_args_list] == [
        {"cache_salt": "a"},
        {"cache_salt": "b"},
        {"cache_salt": "a"},
    ]
    assert requests[0] == requests[1]


async def test_anthropic_seed_omission_and_usage(wire: tuple) -> None:
    client, requests, responses = wire
    client.endpoint = Endpoint("https://anthropic.invalid", "model", provider="anthropic")
    responses.append(
        {
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "cache_read_input_tokens": 3, "output_tokens": 2},
        }
    )
    guard = Mock(wraps=SpendGuard(1))
    result = await APIPolicy(
        "p", client, spend=guard, api_prices={"anthropic/model": {"input": 2, "output": 3}}
    ).chat([Msg("user", "query")], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    assert "seed" not in requests[0]
    assert result.usage == Usage(11, 2, 3, 28 / 1e6, "api:anthropic/model")
    guard.check.assert_called_once_with(30 / 1e6, "p")
    guard.charge.assert_called_once_with(28 / 1e6, "p")


@pytest.mark.parametrize(
    "error",
    [RuntimeError("failed"), UnsupportedRequestError("unsupported"), ValueError("invalid")],
)
async def test_client_failures_are_backend_errors(wire: tuple, error: Exception) -> None:
    client, _, _ = wire
    client.chat = AsyncMock(side_effect=error)
    with pytest.raises(BackendError) as caught:
        await APIPolicy("p", client).chat(
            [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
        )
    assert caught.value.__cause__ is error
    client.chat.assert_awaited_once()


@pytest.mark.parametrize("error", [BudgetExceededError("over budget"), ConfigError("bad config")])
async def test_client_budget_and_config_errors_pass_through(wire: tuple, error: Exception) -> None:
    client, _, _ = wire
    client.chat = AsyncMock(side_effect=error)
    with pytest.raises(type(error)) as caught:
        await APIPolicy("p", client).chat(
            [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
        )
    assert caught.value is error


@pytest.mark.parametrize(
    "message,thinking",
    [
        ({"reasoning": "OpenRouter reasoning"}, "OpenRouter reasoning"),
        ({"reasoning_content": "preferred", "reasoning": "fallback"}, "preferred"),
        ({"reasoning_content": None, "reasoning": "fallback"}, None),
    ],
)
async def test_reasoning_fallback(wire: tuple, message: dict, thinking: str | None) -> None:
    client, _, _ = wire
    client.chat = AsyncMock(return_value=reply(**message))
    result = await APIPolicy("p", client, provider="openrouter").chat(
        [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
    )
    assert result.thinking == thinking


@pytest.mark.parametrize("new_call", [False, True])
async def test_tool_result_never_binds_to_earlier_assistant_turn(
    wire: tuple, new_call: bool
) -> None:
    client, requests, responses = wire
    messages = [
        Msg("assistant", "old", tool_calls=(ToolCall("lookup", {}, "old-id"),)),
        Msg(
            "assistant",
            "new",
            tool_calls=(ToolCall("lookup", {}, "new-id"),) if new_call else (),
        ),
        Msg("tool", "result", name="lookup"),
    ]
    policy = APIPolicy("p", client)
    if new_call:
        responses.append(reply())
        await policy.chat(messages, [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
        assert requests[0]["messages"][-1]["tool_call_id"] == "new-id"
    else:
        with pytest.raises(ConfigError, match="tool_call_id"):
            await policy.chat(messages, [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
        assert not requests


async def test_invalid_inputs_and_response(wire: tuple) -> None:
    client, requests, responses = wire
    policy = APIPolicy("p", client)
    with pytest.raises(ConfigError, match="cache_salt"):
        await policy.chat([], [], max_tokens=4, temperature=1, seed=1, cache_salt="")
    with pytest.raises(ConfigError, match="tool_call_id"):
        await policy.chat(
            [Msg("tool", "orphan")], [], max_tokens=4, temperature=1, seed=1, cache_salt="x"
        )
    assert not requests
    responses.append({"choices": []})
    with pytest.raises(BackendError, match="invalid chat response"):
        await policy.chat([], [], max_tokens=4, temperature=1, seed=1, cache_salt="x")
