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
from marli.llm.client import ChatClient, Endpoint
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
        ("end_turn", Termination.ERROR),
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
        api_prices={"openrouter/model": {"output": 2000}},
    )
    result = await policy.chat([], [], max_tokens=40, temperature=1, seed=1, cache_salt="x")
    assert result.usage == Usage(12, 4, 5, 0.03, "api:openrouter/model")
    guard.check.assert_called_once_with(0.08, "p")
    guard.charge.assert_called_once_with(0.03, "p")
    assert guard._mock_wraps.spent == 0.03


async def test_preflight_budget_prevents_request(wire: tuple) -> None:
    client, requests, _ = wire
    policy = APIPolicy("p", client, spend=SpendGuard(0), api_prices={"openai/model": {"output": 1}})
    with pytest.raises(BudgetExceededError):
        await policy.chat([], [], max_tokens=10, temperature=1, seed=1, cache_salt="x")
    assert not requests


async def test_no_price_still_charges_actual_cost(wire: tuple) -> None:
    client, requests, responses = wire
    responses.append(reply())
    responses[0]["usage"]["cost"] = 0.2
    guard = SpendGuard(0.1)
    with pytest.raises(BudgetExceededError):
        await APIPolicy("p", client, spend=guard).chat(
            [], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
        )
    assert len(requests) == 1 and guard.spent == 0.2


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
    result = await APIPolicy("p", client, spend=guard).chat(
        [Msg("user", "query")], [], max_tokens=10, temperature=1, seed=1, cache_salt="x"
    )
    assert "seed" not in requests[0]
    assert result.usage == Usage(11, 2, 3, 0, "api:anthropic/model")
    guard.check.assert_not_called()
    guard.charge.assert_not_called()


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
