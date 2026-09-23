"""Judge transport preserves the original retry and exhaustion contract."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from marli.llm.judge import DEFAULT_JUDGE_MODEL, anthropic_judge, judge_headers


def test_headers_read_environment_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "first-key")
    assert judge_headers() == {
        "x-api-key": "first-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    monkeypatch.setenv("ANTHROPIC_API_KEY", "second-key")
    assert judge_headers()["x-api-key"] == "second-key"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(KeyError, match="ANTHROPIC_API_KEY"):
        judge_headers()
    assert DEFAULT_JUDGE_MODEL == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize("temperature", [None, 0.0, 0.5])
async def test_success_payload_url_headers_and_semaphore(
    monkeypatch: pytest.MonkeyPatch,
    temperature: float | None,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "judge-test-key")
    sem = asyncio.Semaphore(1)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert sem.locked()
        requests.append(request)
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "judge-test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["content-type"] == "application/json"
        assert request.extensions["timeout"]["read"] == 60
        return httpx.Response(200, json={"content": [{"text": "YES"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await anthropic_judge(
            client,
            sem,
            judge_headers(),
            model=DEFAULT_JUDGE_MODEL,
            system="rubric",
            user="submission",
            max_tokens=16,
            temperature=temperature,
        )
    assert result == "YES"
    assert not sem.locked()
    expected = {
        "model": DEFAULT_JUDGE_MODEL,
        "system": "rubric",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "submission"}],
    }
    if temperature is not None:
        expected["temperature"] = temperature
    assert json.loads(requests[0].content) == expected


@pytest.mark.parametrize("failure", ["429", "503", "529", "400", "transport", "malformed"])
@pytest.mark.parametrize("exhaust", [False, True])
async def test_retries_and_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    exhaust: bool,
) -> None:
    calls = 0
    sleeps: list[float] = []
    sem = asyncio.Semaphore(1)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["max_tokens"] == 8
        if calls == 4 and not exhaust:
            return httpx.Response(200, json={"content": [{"text": "NO"}]})
        if failure == "transport":
            raise httpx.ReadTimeout("test timeout")
        if failure == "malformed":
            return httpx.Response(200, json={"content": []})
        return httpx.Response(int(failure))

    monkeypatch.setattr(asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await anthropic_judge(
            client,
            sem,
            {},
            model=DEFAULT_JUDGE_MODEL,
            system="rubric",
            user="submission",
        )
    assert result == (None if exhaust else "NO")
    assert calls == 4
    assert sleeps == [2, 4, 6]
    assert not sem.locked()
