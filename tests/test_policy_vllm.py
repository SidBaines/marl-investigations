"""The mocked completion endpoint must return ids and aligned raw logprobs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from marli.errors import BackendError, ConfigError
from marli.interact.types import Termination, Usage
from marli.policy.base import SamplingSpec, TokenPolicy
from marli.policy.vllm import VLLMPolicy, read_server_json
from marli.render.base import Msg
from marli.render.fake import FakeRenderer


@pytest.fixture
async def wire() -> AsyncIterator[tuple[httpx.AsyncClient, list[httpx.Request], list[Any]]]:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response | Exception] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        yield client, requests, responses


def response(reason: str = "stop") -> dict[str, Any]:
    ids = FakeRenderer().encode_completion("actual", stop=reason == "stop")
    return {
        "choices": [
            {
                "token_ids": ids,
                "logprobs": {"token_logprobs": [-0.5] * len(ids)},
                "text": "wrong text; must never be tokenized",
                "finish_reason": reason,
            }
        ],
        "usage": {"prompt_tokens": 999, "completion_tokens": 999},
    }


@pytest.mark.parametrize("base_url", ["http://local:8000", "http://local:8000/v1/"])
async def test_exact_token_request_and_response(wire: tuple, base_url: str) -> None:
    client, requests, responses = wire
    renderer = FakeRenderer()
    prompt = renderer.initial(None, [], [Msg("user", "hi")])
    data = response()
    responses.append(httpx.Response(200, json=data))
    policy = VLLMPolicy(
        "p",
        base_url,
        "learner@3",
        renderer_name="fake",
        trainable=False,
        policy_version=3,
        client=client,
        timeout_s=90,
    )
    assert isinstance(policy, TokenPolicy)
    sample = await policy.sample(
        prompt, SamplingSpec(30, 0.7, 0.8, 5, renderer.stop_token_ids), seed=42
    )
    assert str(requests[0].url) == "http://local:8000/v1/completions"
    assert json.loads(requests[0].content) == {
        "model": "learner@3",
        "prompt": prompt,
        "max_tokens": 30,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 5,
        "seed": 42,
        "logprobs": 1,
        "return_token_ids": True,
        "stop_token_ids": list(renderer.stop_token_ids),
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
    }
    assert requests[0].extensions["timeout"]["read"] == 90
    assert sample.completion_ids == tuple(data["choices"][0]["token_ids"])
    assert sample.logprobs == tuple(data["choices"][0]["logprobs"]["token_logprobs"])
    assert sample.policy_version == 3 and sample.termination == Termination.STOP
    assert renderer.parse(sample.completion_ids).content == "actual"
    assert sample.usage == Usage(len(prompt), len(sample.completion_ids), tokenizer="fake")
    await policy.aclose()
    assert not client.is_closed


@pytest.mark.parametrize(
    "reason,termination", [("length", Termination.LENGTH), ("unknown", Termination.ERROR)]
)
async def test_termination(wire: tuple, reason: str, termination: Termination) -> None:
    client, _, responses = wire
    responses.append(httpx.Response(200, json=response(reason)))
    sample = await VLLMPolicy(
        "p", "http://local", "m", renderer_name="fake", trainable=True, client=client
    ).sample(FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1)
    assert sample.termination == termination


@pytest.mark.parametrize("malformed", ["missing_ids", "missing_logprobs", "mismatch", "no_choices"])
async def test_malformed_response_fails_without_retry(wire: tuple, malformed: str) -> None:
    client, requests, responses = wire
    data = response()
    if malformed == "missing_ids":
        del data["choices"][0]["token_ids"]
    elif malformed == "missing_logprobs":
        data["choices"][0]["logprobs"] = None
    elif malformed == "mismatch":
        data["choices"][0]["logprobs"]["token_logprobs"].pop()
    else:
        data["choices"] = []
    responses.append(httpx.Response(200, json=data))
    with pytest.raises(BackendError):
        await VLLMPolicy(
            "p", "http://local", "m", renderer_name="fake", trainable=True, client=client
        ).sample(FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection lost"),
        httpx.ReadTimeout("timed out"),
        httpx.Response(503, text="unavailable"),
    ],
)
@pytest.mark.parametrize("recover", [True, False])
async def test_bounded_retries(
    wire: tuple, monkeypatch: pytest.MonkeyPatch, failure: Exception | httpx.Response, recover: bool
) -> None:
    client, requests, responses = wire
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    responses.extend(
        [failure, failure, httpx.Response(200, json=response()) if recover else failure]
    )
    policy = VLLMPolicy(
        "p", "http://local", "m", renderer_name="fake", trainable=True, client=client
    )
    if recover:
        assert (
            await policy.sample(FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1)
        ).termination == Termination.STOP
    else:
        with pytest.raises(BackendError, match="connection lost|timed out|unavailable"):
            await policy.sample(FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1)
    assert len(requests) == 3
    assert requests[0].content == requests[1].content == requests[2].content
    assert [call.args for call in sleep.call_args_list] == [(1.0,), (2.0,)]


@pytest.mark.parametrize("status", [400, 401, 404, 429])
async def test_4xx_immediate_server_message(wire: tuple, status: int) -> None:
    client, requests, responses = wire
    responses.append(httpx.Response(status, json={"error": {"message": "prompt too long"}}))
    with pytest.raises(BackendError, match=f"HTTP {status}.*prompt too long"):
        await VLLMPolicy(
            "p", "http://local", "m", renderer_name="fake", trainable=True, client=client
        ).sample(FakeRenderer().encode_text("p"), SamplingSpec(10), seed=1)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "spec",
    [SamplingSpec(10, temperature=0.5), SamplingSpec(10, top_p=0.9), SamplingSpec(10, top_k=5)],
)
async def test_trainable_guard(wire: tuple, spec: SamplingSpec) -> None:
    client, requests, _ = wire
    with pytest.raises(ConfigError, match="off-policy"):
        await VLLMPolicy(
            "p", "http://local", "m", renderer_name="fake", trainable=True, client=client
        ).sample(FakeRenderer().encode_text("p"), spec, seed=1)
    assert not requests


async def test_owned_client_is_closed() -> None:
    policy = VLLMPolicy("p", "http://local", "m", renderer_name="fake", trainable=False)
    await policy.aclose()
    assert policy.client.is_closed


def test_server_manifest(tmp_path: Path) -> None:
    path = tmp_path / "server.json"
    path.write_text(
        json.dumps({"base_url": "http://local:8000", "models": ["base", "learner@3"], "pid": 123})
    )
    assert read_server_json(path) == (
        "http://local:8000",
        {"base": "base", "learner@3": "learner@3"},
    )


@pytest.mark.parametrize(
    "data",
    [
        None,
        "{bad",
        "{}",
        '"not an object"',
        '{"base_url": "http://local", "models": [1]}',
        '{"base_url": "", "models": []}',
    ],
)
def test_invalid_server_manifest(tmp_path: Path, data: str | None) -> None:
    path = tmp_path / "server.json"
    if data is not None:
        path.write_text(data)
    with pytest.raises(ConfigError, match="server manifest"):
        read_server_json(path)
