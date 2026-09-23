"""Provider and cache behavior is exercised through httpx without network access."""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from marli.llm.client import (
    ChatClient,
    Endpoint,
    UnsupportedRequestError,
    cached_client,
    from_anthropic,
    to_anthropic,
    usage_of,
)

PAYLOAD = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Find a value",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    },
}
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"key": "a"}'},
}


def completion(text: str | None = "answer", **message: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text, **message},
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


@dataclass
class MockServer:
    responses: list[httpx.Response | httpx.HTTPError] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.responses, "unexpected request"
        response = self.responses.pop(0)
        if isinstance(response, httpx.HTTPError):
            raise response
        return response

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)


@pytest.fixture
async def server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MockServer]:
    server = MockServer()
    original = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []

    def client(**kw: Any) -> httpx.AsyncClient:
        result = original(transport=httpx.MockTransport(server.handle), **kw)
        clients.append(result)
        return result

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(asyncio, "sleep", server.sleep)
    yield server
    for instance in clients:
        await instance.aclose()


@pytest.mark.parametrize(
    ("provider", "url", "env_name"),
    [
        ("openai", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
        ("anthropic", "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"),
    ],
)
async def test_provider_wire_contract(
    server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    url: str,
    env_name: str,
) -> None:
    monkeypatch.delenv(env_name, raising=False)
    client = getattr(ChatClient, provider)("model-id")
    monkeypatch.setenv(env_name, "test-key-one")
    raw = completion()
    if provider == "anthropic":
        raw = {"content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn"}
    server.responses = [httpx.Response(200, json=raw), httpx.Response(200, json=raw)]
    payload = {**PAYLOAD, "temperature": 1.0}
    first = await client.chat(payload)
    monkeypatch.setenv(env_name, "test-key-two")
    await client.chat(payload)
    assert first["choices"][0]["message"]["content"] == "answer"
    expected = {"model": "model-id", **payload}
    if provider == "anthropic":
        expected.pop("temperature")
    for request, secret in zip(server.requests, ["test-key-one", "test-key-two"], strict=True):
        assert str(request.url) == url
        assert json.loads(request.content) == expected
        assert request.headers["content-type"] == "application/json"
        assert request.extensions["timeout"]["read"] == 120
        if provider == "anthropic":
            assert request.headers["x-api-key"] == secret
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == f"Bearer {secret}"
    assert "test-key" not in repr(client)


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_openai_tools_and_extra_params_pass_through(
    server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "fake-key")
    extra = {
        "provider": {"order": ["vendor"], "allow_fallbacks": False},
        "reasoning": {"effort": "low"},
    }
    client = getattr(ChatClient, provider)("model", extra_params=extra)
    raw = completion(None, tool_calls=[CALL])
    server.responses = [httpx.Response(200, json=raw)]
    payload = {**PAYLOAD, "tools": [TOOL], "tool_choice": "required"}
    original = deepcopy(payload)
    assert await client.chat(payload) == raw
    assert json.loads(server.requests[0].content) == {"model": "model", **extra, **payload}
    assert payload == original


@pytest.mark.parametrize("status", [408, 409, 429, 500, 501, 503, 529, 599])
async def test_retryable_status(server: MockServer, status: int) -> None:
    server.responses = [httpx.Response(status), httpx.Response(200, json=completion())]
    client = ChatClient(Endpoint("https://local.invalid/v1/", "model"))
    assert await client.chat(PAYLOAD) == completion()
    assert len(server.requests) == 2
    assert server.sleeps == [1.0]
    assert "authorization" not in server.requests[0].headers


async def test_retry_sequence_and_backoff(server: MockServer) -> None:
    server.responses = [httpx.Response(code) for code in [429, 503, 529]]
    server.responses.append(httpx.Response(200, json=completion()))
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"))
    assert await client.chat(PAYLOAD) == completion()
    assert server.sleeps == [1, 2, 4]


async def test_retry_exhaustion_is_bounded(server: MockServer) -> None:
    server.responses = [httpx.Response(503) for _ in range(8)]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), max_retries=8)
    with pytest.raises(RuntimeError, match="failed after 8 retries.*HTTP 503"):
        await client.chat(PAYLOAD)
    assert len(server.requests) == 8
    assert server.sleeps == [1, 2, 4, 8, 16, 30, 30]


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("offline"), httpx.Response(200, text="<html>")]
)
async def test_transport_and_non_json_retry(
    server: MockServer,
    failure: httpx.Response | httpx.HTTPError,
) -> None:
    server.responses = [failure, httpx.Response(200, json=completion())]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"))
    assert await client.chat(PAYLOAD) == completion()
    assert server.sleeps == [1]


async def test_400_is_not_retried(server: MockServer) -> None:
    server.responses = [httpx.Response(400, text="unsupported parameter")]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"))
    with pytest.raises(UnsupportedRequestError, match="HTTP 400.*unsupported parameter"):
        await client.chat(PAYLOAD)
    assert len(server.requests) == 1
    assert not server.sleeps


async def test_max_tokens_switch_is_remembered_with_canonical_cache(server: MockServer) -> None:
    server.responses = [httpx.Response(400, text="use max_completion_tokens instead")]
    server.responses.extend(httpx.Response(200, json=completion()) for _ in range(2))
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), cache="memory")
    original = deepcopy(PAYLOAD)
    await client.chat(PAYLOAD, cache_salt="a")
    await client.chat(PAYLOAD, cache_salt="a")
    await client.chat(PAYLOAD, cache_salt="b")
    bodies = [json.loads(request.content) for request in server.requests]
    assert bodies[0] == {"model": "model", **PAYLOAD}
    assert (
        bodies[1:]
        == [{"model": "model", "messages": PAYLOAD["messages"], "max_completion_tokens": 32}] * 2
    )
    assert not server.sleeps
    assert original == PAYLOAD


def test_anthropic_translation_with_tools_and_grouped_results() -> None:
    second = {**CALL, "id": "call_2"}
    payload = {
        "model": "claude",
        "tools": [TOOL],
        "tool_choice": "required",
        "stop": "STOP",
        "max_tokens": 80,
        "temperature": 0.2,
        "top_p": 0.8,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "metadata": {"user_id": "test"},
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "look up both"},
            {"role": "assistant", "content": "checking", "tool_calls": [CALL, second]},
            {"role": "tool", "tool_call_id": "call_1", "content": "one"},
            {"role": "tool", "tool_call_id": "call_2", "content": "two"},
            {"role": "user", "content": "continue"},
        ],
    }
    original = deepcopy(payload)
    body = to_anthropic(payload)
    assert body == {
        "model": "claude",
        "system": "first\n\nsecond",
        "max_tokens": 80,
        "temperature": 0.2,
        "top_p": 0.8,
        "stop_sequences": ["STOP"],
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "metadata": {"user_id": "test"},
        "tool_choice": {"type": "any"},
        "tools": [
            {
                "name": "lookup",
                "description": "Find a value",
                "input_schema": TOOL["function"]["parameters"],
            }
        ],
        "messages": [
            {"role": "user", "content": "look up both"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"key": "a"}},
                    {"type": "tool_use", "id": "call_2", "name": "lookup", "input": {"key": "a"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "one"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "two"},
                ],
            },
            {"role": "user", "content": "continue"},
        ],
    }
    assert payload == original


@pytest.mark.parametrize("content", [None, ""])
def test_anthropic_tool_only_assistant(content: str | None) -> None:
    body = to_anthropic(
        {"messages": [{"role": "assistant", "content": content, "tool_calls": [CALL]}]}
    )
    assert body["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"key": "a"}},
            ],
        }
    ]
    assert body["max_tokens"] == 4096
    assert "temperature" not in body


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "lookup"}}, {"type": "tool", "name": "lookup"}),
    ],
)
def test_anthropic_tool_choice(choice: Any, expected: dict[str, str]) -> None:
    assert to_anthropic({**PAYLOAD, "tool_choice": choice})["tool_choice"] == expected


@pytest.mark.parametrize("extra", [{"seed": 1}, {"logprobs": True}, {"typo": True}])
def test_anthropic_unknown_keys_rejected(extra: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedRequestError, match="payload key"):
        to_anthropic({**PAYLOAD, **extra})


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        {**PAYLOAD, "tools": [{"type": "custom"}]},
        {**PAYLOAD, "tool_choice": "unrecognized"},
        {"messages": [{"role": "developer", "content": "unsupported"}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {**CALL, "function": {"name": "lookup", "arguments": "[]"}},
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {**CALL, "function": {"name": "lookup", "arguments": "not json"}},
                    ],
                }
            ]
        },
    ],
)
def test_anthropic_unsupported_message_or_tool(payload: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedRequestError):
        to_anthropic(payload)


async def test_anthropic_response_tools_thinking_and_usage(
    server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    raw = {
        "id": "msg_1",
        "model": "claude",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "first ", "signature": "opaque"},
            {"type": "thinking", "thinking": "second"},
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"key": "a"}},
            {"type": "tool_use", "id": "call_2", "name": "lookup", "input": {"key": "b"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }
    server.responses = [httpx.Response(200, json=raw)]
    client = ChatClient.anthropic("claude")
    result = await client.chat({**PAYLOAD, "tools": [TOOL]})
    assert result["id"] == "msg_1"
    assert result["model"] == "claude"
    assert result["usage"] == raw["usage"]
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "checking"
    assert choice["message"]["reasoning_content"] == "first second"
    calls = choice["message"]["tool_calls"]
    assert calls[0] == CALL
    assert calls[1]["id"] == "call_2"
    assert json.loads(calls[1]["function"]["arguments"]) == {"key": "b"}
    assert usage_of(result)["prompt_tokens"] == 10


def test_anthropic_embedded_error_is_not_normalized_to_empty() -> None:
    with pytest.raises(UnsupportedRequestError, match="anthropic error response"):
        from_anthropic({"type": "error", "error": {"message": "overloaded"}})


async def test_cache_off_identical_calls_are_independent(server: MockServer) -> None:
    server.responses = [httpx.Response(200, json=completion(str(i))) for i in range(2)]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"))
    results = await asyncio.gather(client.chat(PAYLOAD), client.chat(PAYLOAD))
    assert [result["choices"][0]["message"]["content"] for result in results] == ["0", "1"]
    assert len(server.requests) == 2
    assert client._cache == {}


async def test_memory_cache_requires_matching_payload_and_salt(server: MockServer) -> None:
    server.responses = [httpx.Response(200, json=completion(str(i))) for i in range(3)]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), cache="memory")
    first = await client.chat(PAYLOAD, cache_salt="a")
    assert await client.chat(dict(reversed(list(PAYLOAD.items()))), cache_salt="a") == first
    assert await client.chat(PAYLOAD, cache_salt="b") != first
    assert await client.chat({**PAYLOAD, "temperature": 0.5}, cache_salt="a") != first
    assert len(server.requests) == 3
    assert all("cache_salt" not in json.loads(request.content) for request in server.requests)


@pytest.mark.parametrize("cache", ["memory", "disk"])
async def test_cached_calls_require_explicit_salt(
    server: MockServer,
    tmp_path: Path,
    cache: str,
) -> None:
    client = ChatClient(
        Endpoint("https://local.invalid/v1", "model"),
        cache=cache,
        cache_path=tmp_path / "cache.jsonl",
    )
    with pytest.raises(
        ValueError, match="^cached ChatClient calls require an explicit cache_salt$"
    ):
        await client.chat(PAYLOAD)
    assert not server.requests


@pytest.mark.parametrize("cache", ["off", "memory"])
async def test_non_disk_modes_ignore_cache_path(
    server: MockServer,
    tmp_path: Path,
    cache: str,
) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_bytes(b"invalid cache\n")
    server.responses = [httpx.Response(200, json=completion())]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), cache=cache, cache_path=path)
    await client.chat(PAYLOAD, cache_salt="a")
    assert path.read_bytes() == b"invalid cache\n"


async def test_disk_cache_survives_clients_and_repairs_torn_tail(
    server: MockServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "never-persist-this-key")
    endpoint = Endpoint("https://local.invalid/v1", "model", api_key_env="TEST_API_KEY")
    server.responses = [httpx.Response(200, json=completion(str(i))) for i in range(2)]
    first = cached_client(endpoint, tmp_path, "run")
    response = await first.chat(PAYLOAD, cache_salt="a")
    await first.aclose()
    path = tmp_path / "cache_run.jsonl"
    good_bytes = path.read_bytes()
    with path.open("ab") as output:
        output.write(b'{"key": "partial", "response": "\xe2')
    with pytest.warns(UserWarning, match="truncated trailing cache fragment"):
        second = cached_client(endpoint, tmp_path, "run")
    assert path.read_bytes() == good_bytes
    assert await second.chat(PAYLOAD, cache_salt="a") == response
    await second.chat(PAYLOAD, cache_salt="b")
    await second.aclose()
    third = cached_client(endpoint, tmp_path, "run")
    await third.chat(PAYLOAD, cache_salt="a")
    await third.chat(PAYLOAD, cache_salt="b")
    assert len(server.requests) == 2
    assert len(path.read_text().splitlines()) == 2
    assert "never-persist-this-key" not in path.read_text()


async def test_complete_cache_tail_without_newline_can_be_appended(
    server: MockServer,
    tmp_path: Path,
) -> None:
    server.responses = [httpx.Response(200, json=completion()) for _ in range(2)]
    endpoint = Endpoint("https://local.invalid/v1", "model")
    first = cached_client(endpoint, tmp_path, "tail")
    await first.chat(PAYLOAD, cache_salt="a")
    path = tmp_path / "cache_tail.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    second = cached_client(endpoint, tmp_path, "tail")
    await second.chat(PAYLOAD, cache_salt="b")
    third = cached_client(endpoint, tmp_path, "tail")
    await third.chat(PAYLOAD, cache_salt="a")
    await third.chat(PAYLOAD, cache_salt="b")
    assert len(server.requests) == 2


@pytest.mark.parametrize("content", [b"not json\n", b"not json\n{}\n"])
def test_interior_cache_corruption_is_loud(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "cache_bad.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="malformed cache record 1"):
        cached_client(Endpoint("https://local.invalid/v1", "model"), tmp_path, "bad")
    assert path.read_bytes() == content


async def test_extra_params_participate_in_disk_key(server: MockServer, tmp_path: Path) -> None:
    server.responses = [httpx.Response(200, json=completion(str(i))) for i in range(2)]
    results = []
    for effort in ["low", "high", "low", "high"]:
        client = cached_client(
            Endpoint(
                "https://local.invalid/v1",
                "model",
                extra_params={
                    "reasoning": {"effort": effort},
                },
            ),
            tmp_path,
            "shared",
        )
        results.append(await client.chat(PAYLOAD, cache_salt="same"))
        await client.aclose()
    assert results[0] != results[1]
    assert results[:2] == results[2:]
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    "raw",
    [
        completion(""),
        completion(None),
        {"choices": []},
        completion(
            "",
            reasoning_content="only reasoning",
        ),
    ],
)
async def test_empty_completions_are_not_cached(
    server: MockServer,
    tmp_path: Path,
    raw: dict[str, Any],
) -> None:
    server.responses = [httpx.Response(200, json=raw) for _ in range(2)]
    client = cached_client(Endpoint("https://local.invalid/v1", "model"), tmp_path, "empty")
    assert await client.chat(PAYLOAD, cache_salt="same") == raw
    assert await client.chat(PAYLOAD, cache_salt="same") == raw
    assert len(server.requests) == 2
    assert not (tmp_path / "cache_empty.jsonl").exists()


@pytest.mark.parametrize(
    "raw",
    [
        {"error": {"message": "upstream failure"}},
        {"choices": [{"error": {"message": "upstream failure"}}]},
        {"choices": [{"finish_reason": "error", "message": {"content": "partial"}}]},
        {"type": "error", "error": {"type": "overloaded_error"}},
    ],
)
async def test_embedded_errors_retry_without_poisoning_cache(
    server: MockServer,
    tmp_path: Path,
    raw: dict[str, Any],
) -> None:
    server.responses = [httpx.Response(200, json=raw) for _ in range(2)]
    client = ChatClient(
        Endpoint("https://local.invalid/v1", "model"),
        cache="disk",
        cache_path=tmp_path / "cache.jsonl",
        max_retries=2,
    )
    with pytest.raises(RuntimeError, match="embedded error"):
        await client.chat(PAYLOAD, cache_salt="same")
    assert not (tmp_path / "cache.jsonl").exists()
    server.responses = [httpx.Response(200, json=completion())]
    assert await client.chat(PAYLOAD, cache_salt="same") == completion()
    assert await client.chat(PAYLOAD, cache_salt="same") == completion()
    assert len(server.requests) == 3


async def test_tool_only_responses_are_cacheable(server: MockServer) -> None:
    raw = completion(None, tool_calls=[CALL])
    server.responses = [httpx.Response(200, json=raw)]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), cache="memory")
    assert await client.chat(PAYLOAD, cache_salt="same") == raw
    assert await client.chat(PAYLOAD, cache_salt="same") == raw
    assert len(server.requests) == 1


async def test_anthropic_embedded_error_retries_and_tool_only_result_is_cached(
    server: MockServer,
) -> None:
    raw = {
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"key": "a"}},
        ],
        "stop_reason": "tool_use",
    }
    server.responses = [
        httpx.Response(200, json={"type": "error", "error": {"type": "overloaded_error"}}),
        httpx.Response(200, json=raw),
    ]
    client = ChatClient(
        Endpoint("https://local.invalid", "model", provider="anthropic"),
        cache="memory",
    )
    result = await client.chat(PAYLOAD, cache_salt="same")
    assert result["choices"][0]["message"]["tool_calls"] == [CALL]
    assert await client.chat(PAYLOAD, cache_salt="same") == result
    assert len(server.requests) == 2
    assert server.sleeps == [1]


async def test_anthropic_never_switches_to_openai_max_completion_tokens(server: MockServer) -> None:
    server.responses = [httpx.Response(400, text="max_completion_tokens is unsupported")]
    client = ChatClient(Endpoint("https://local.invalid", "model", provider="anthropic"))
    with pytest.raises(UnsupportedRequestError, match="HTTP 400"):
        await client.chat(PAYLOAD)
    assert len(server.requests) == 1


async def test_missing_api_key_names_env_without_request(
    server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)
    client = ChatClient(
        Endpoint("https://local.invalid/v1", "model", api_key_env="MISSING_TEST_API_KEY")
    )
    with pytest.raises(
        ValueError, match="missing API key environment variable 'MISSING_TEST_API_KEY'"
    ):
        await client.chat(PAYLOAD)
    assert not server.requests


@pytest.mark.parametrize("failure", ["400", "503", "embedded", "transport"])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_credentials_redacted_from_exceptions(
    server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    provider: str,
) -> None:
    secret = "fake-secret-must-not-appear"
    monkeypatch.setenv("TEST_API_KEY", secret)
    if failure == "transport":
        response = httpx.ConnectError(f"failed using {secret}")
    elif failure == "embedded":
        response = httpx.Response(200, json={"error": {"message": secret}})
    else:
        response = httpx.Response(int(failure), text=f"failed using {secret}")
    server.responses = [response]
    client = ChatClient(
        Endpoint(
            "https://local.invalid/v1", "model", api_key_env="TEST_API_KEY", provider=provider
        ),
        max_retries=1,
    )
    with pytest.raises(RuntimeError) as caught:
        await client.chat(PAYLOAD)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert "[redacted]" in str(caught.value)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"prompt_tokens": 10, "completion_tokens": 4},
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cached_prompt_tokens": None,
                "cost_usd": None,
            },
        ),
        (
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 3},
                "cost": 0.005,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cached_prompt_tokens": 3,
                "cost_usd": 0.005,
            },
        ),
        (
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0,
            },
            {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0, "cost_usd": 0},
        ),
        (
            {
                "input_tokens": 4,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
            },
            {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "cached_prompt_tokens": 3,
                "cost_usd": None,
            },
        ),
        (
            {},
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_prompt_tokens": None,
                "cost_usd": None,
            },
        ),
    ],
)
def test_usage_of(usage: dict[str, Any], expected: dict[str, Any]) -> None:
    response = {"usage": deepcopy(usage)}
    assert usage_of(response) == expected
    assert response == {"usage": usage}


@pytest.mark.parametrize(
    "kw", [{"cache": "invalid"}, {"cache": "disk"}, {"concurrency": 0}, {"max_retries": 0}]
)
def test_client_validation(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ChatClient(Endpoint("https://local.invalid/v1", "model"), **kw)


def test_endpoint_validation_and_frozen_record() -> None:
    with pytest.raises(ValueError, match="provider"):
        Endpoint("https://local.invalid", "model", provider="invalid")
    with pytest.raises(ValueError, match="reserved key 'model'"):
        Endpoint("https://local.invalid", "model", extra_params={"model": "other"})
    endpoint = Endpoint("https://local.invalid", "model")
    with pytest.raises(FrozenInstanceError):
        endpoint.model = "other"


async def test_payload_cannot_override_endpoint_model(server: MockServer) -> None:
    client = ChatClient(Endpoint("https://local.invalid", "model"))
    with pytest.raises(ValueError, match="payload must not include 'model'"):
        await client.chat({**PAYLOAD, "model": "other"})
    assert not server.requests


async def test_raw_completions_route_and_cache(server: MockServer) -> None:
    raw = {"choices": [{"text": "continued text"}]}
    server.responses = [httpx.Response(200, json=raw)]
    client = ChatClient(Endpoint("https://local.invalid/v1", "model"), cache="memory")
    assert await client.completions({"prompt": "prefix"}, cache_salt="a") == raw
    assert await client.completions({"prompt": "prefix"}, cache_salt="a") == raw
    assert len(server.requests) == 1
    assert str(server.requests[0].url) == "https://local.invalid/v1/completions"
    anthropic = ChatClient(Endpoint("https://local.invalid", "model", provider="anthropic"))
    with pytest.raises(UnsupportedRequestError, match="route"):
        await anthropic.completions({"prompt": "prefix"})
