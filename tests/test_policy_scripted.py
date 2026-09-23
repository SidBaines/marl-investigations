"""Scripted seats must reproduce token traces and expose every sampling input."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import pytest

import marli.policy.scripted as scripted_module
from marli.interact.types import Termination, Usage
from marli.policy.base import CallMeta, ChatPolicy, ChatReply, SamplingSpec, TokenPolicy
from marli.policy.scripted import (
    ScriptCtx,
    ScriptedChatPolicy,
    ScriptedPolicy,
    Turn,
    by_role,
    from_callable,
    turns_by_agent,
)
from marli.render.base import Msg, ParsedToolCall, ToolSpec
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer
from marli.seeds import derive_seed


class FakeClock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.elapsed += seconds


async def test_sample_context_usage_and_call_recording() -> None:
    renderer = FakeRenderer()
    seen: list[ScriptCtx] = []

    def script(ctx: ScriptCtx) -> Sequence[int]:
        seen.append(ctx)
        return renderer.encode_completion("answer")

    policy = ScriptedPolicy("script", renderer, script)
    assert isinstance(policy, TokenPolicy)
    assert policy.policy_id == "script"
    assert policy.renderer_name == renderer.name
    assert policy.trainable is False
    assert policy.calls == []
    prompt = renderer.initial(None, [], [Msg("user", "question π")])
    original_prompt = tuple(prompt)
    spec = SamplingSpec(100)
    first = await policy.sample(prompt, spec, seed=17)
    meta = CallMeta("episode", "peer1", "peer", 3, "compact")
    second = await policy.sample(prompt, spec, seed=18, meta=meta)
    prompt.clear()

    assert (
        policy.calls
        == seen
        == [
            ScriptCtx(original_prompt, renderer.decode(original_prompt), spec, 17, CallMeta()),
            ScriptCtx(original_prompt, renderer.decode(original_prompt), spec, 18, meta),
        ]
    )
    assert policy.calls[1].meta is meta
    assert (
        first.completion_ids == second.completion_ids == tuple(renderer.encode_completion("answer"))
    )
    assert first.usage == Usage(len(original_prompt), len(first.completion_ids), tokenizer="fake")
    assert first.policy_version is None
    with pytest.raises(FrozenInstanceError):
        policy.calls[0].seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("ids", "max_tokens", "stops", "expected_ids", "termination"),
    [
        ([65, S["eot"]], 3, (), (65, S["eot"]), Termination.STOP),
        ([65, S["eot"]], 2, (), (65, S["eot"]), Termination.STOP),
        ([65, 66], 3, (), (65, 66), Termination.LENGTH),
        ([65, 66], 1, (), (65,), Termination.LENGTH),
        ([65, S["eot"], 66], 2, (), (65, S["eot"]), Termination.LENGTH),
        ([65, S["eot"], 66], 3, (), (65, S["eot"], 66), Termination.LENGTH),
        ([65, 66], 2, (66, 67), (65, 66), Termination.STOP),
        ([65, S["eot"]], 2, (66,), (65, S["eot"]), Termination.LENGTH),
        ([], 1, (), (), Termination.LENGTH),
    ],
)
async def test_sample_termination(
    ids: list[int],
    max_tokens: int,
    stops: tuple[int, ...],
    expected_ids: tuple[int, ...],
    termination: Termination,
) -> None:
    policy = ScriptedPolicy("script", FakeRenderer(), lambda ctx: ids)
    sample = await policy.sample([], SamplingSpec(max_tokens, stop_token_ids=stops), seed=1)
    assert sample.completion_ids == expected_ids
    assert sample.termination == termination
    assert sample.logprobs is not None
    assert len(sample.logprobs) == len(expected_ids) == sample.usage.completion_tokens
    assert all(-3.0 <= value < 0 for value in sample.logprobs)


async def test_logprobs_are_seeded_aligned_and_independent_of_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derivations: list[tuple[object, ...]] = []

    def tracked_seed(*parts: object) -> int:
        derivations.append(parts)
        return derive_seed(*parts)

    monkeypatch.setattr(scripted_module, "derive_seed", tracked_seed)
    renderer = FakeRenderer()
    policy = ScriptedPolicy("script", renderer, lambda ctx: renderer.encode_completion("abcdef"))
    random_state = random.getstate()
    full = await policy.sample([], SamplingSpec(100), seed=10)
    assert derivations == [(10, i) for i in range(len(full.completion_ids))]
    other = await policy.sample([], SamplingSpec(100), seed=11)
    repeat = await policy.sample([], SamplingSpec(100), seed=10)
    short = await policy.sample([], SamplingSpec(3), seed=10)
    assert repeat == full
    assert other.logprobs != full.logprobs
    assert short.completion_ids == full.completion_ids[:3]
    assert full.logprobs is not None
    assert short.logprobs == full.logprobs[:3]
    assert len(set(full.logprobs)) > 1
    assert random.getstate() == random_state


async def test_trainable_guard_is_left_to_runtime_and_version_is_preserved() -> None:
    policy = ScriptedPolicy(
        "learner", FakeRenderer(), lambda ctx: [65], trainable=True, policy_version=12
    )
    sample = await policy.sample([], SamplingSpec(10, temperature=0.5, top_p=0.8, top_k=4), seed=1)
    assert policy.trainable is True
    assert sample.policy_version == 12


async def test_script_failure_still_records_the_call() -> None:
    def script(ctx: ScriptCtx) -> Sequence[int]:
        raise RuntimeError("script failed")

    policy = ScriptedPolicy("broken", FakeRenderer(), script)
    with pytest.raises(RuntimeError, match="script failed"):
        await policy.sample([], SamplingSpec(10), seed=1)
    assert len(policy.calls) == 1


@pytest.mark.parametrize("latency", [0.0, 2.5])
async def test_fixed_latency_uses_clock(latency: float) -> None:
    clock = FakeClock()
    policy = ScriptedPolicy(
        "script", FakeRenderer(), lambda ctx: [], clock=clock, latency_s=latency
    )
    await policy.sample([], SamplingSpec(1), seed=1)
    assert clock.sleeps == ([latency] if latency else [])
    assert clock.now() == latency


async def test_callable_latency_receives_recorded_context() -> None:
    clock = FakeClock()
    seen: list[ScriptCtx] = []

    def latency(ctx: ScriptCtx) -> float:
        seen.append(ctx)
        return float(ctx.meta.call_index)

    policy = ScriptedPolicy(
        "script", FakeRenderer(), lambda ctx: [], clock=clock, latency_s=latency
    )
    await policy.sample([], SamplingSpec(1), seed=1, meta=CallMeta(call_index=2))
    await policy.sample([], SamplingSpec(1), seed=2, meta=CallMeta(call_index=0))
    assert seen == policy.calls
    assert clock.sleeps == [2.0]


async def test_default_clock_awaits_asyncio_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    policy = ScriptedPolicy("script", FakeRenderer(), lambda ctx: [], latency_s=0.25)
    await policy.sample([], SamplingSpec(1), seed=1)
    sleep.assert_awaited_once_with(0.25)


def test_turn_callable_encodes_content_thinking_tools_and_raw_bodies() -> None:
    renderer = FakeRenderer()
    turn = Turn(
        "answer π",
        thinking="reasoning",
        tool_calls=(("submit", {"answer": "42"}),),
        raw_tool_bodies=("not json",),
        stop=False,
    )
    ctx = ScriptCtx((), "", SamplingSpec(100), 1, CallMeta())
    seen: list[ScriptCtx] = []

    def choose(context: ScriptCtx) -> Turn:
        seen.append(context)
        return turn

    parsed = renderer.parse(from_callable(choose, renderer)(ctx))
    assert seen == [ctx]
    assert parsed.content == turn.content
    assert parsed.thinking == turn.thinking
    assert parsed.termination == "length"
    assert parsed.tool_calls[0].ok
    assert parsed.tool_calls[0].name == "submit"
    assert parsed.tool_calls[0].arguments == {"answer": "42"}
    assert not parsed.tool_calls[1].ok
    assert parsed.tool_calls[1].raw == "not json"
    with pytest.raises(FrozenInstanceError):
        turn.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("by_agent", [True, False])
def test_turn_lookup_uses_metadata_and_has_no_shared_cursor(by_agent: bool) -> None:
    renderer = FakeRenderer()
    helper = turns_by_agent if by_agent else by_role
    script = helper(renderer, {"a": [Turn("first"), Turn("second")], "b": [Turn("other")]})
    for key, index, expected in [("a", 1, "second"), ("b", 0, "other"), ("a", 0, "first")]:
        meta = CallMeta(
            agent_id=key if by_agent else "unused",
            role="unused" if by_agent else key,
            call_index=index,
        )
        ctx = ScriptCtx((), "", SamplingSpec(100), 1, meta)
        assert tuple(script(ctx)) == tuple(renderer.encode_completion(expected))
        assert tuple(script(ctx)) == tuple(renderer.encode_completion(expected))


@pytest.mark.parametrize("by_agent", [True, False])
@pytest.mark.parametrize(("key", "index"), [("peer0", 1), ("missing", 0), ("peer0", -1)])
def test_turn_lookup_exhaustion_and_default(by_agent: bool, key: str, index: int) -> None:
    renderer = FakeRenderer()
    helper = turns_by_agent if by_agent else by_role
    turns = {"peer0": [Turn("first")]}
    ctx = ScriptCtx(
        (), "", SamplingSpec(100), 1, CallMeta(agent_id=key, role=key, call_index=index)
    )
    with pytest.raises(IndexError) as caught:
        helper(renderer, turns)(ctx)
    assert repr(key) in str(caught.value)
    assert f"index {index}" in str(caught.value)
    assert tuple(helper(renderer, turns, default=Turn())(ctx)) == tuple(
        renderer.encode_completion()
    )


async def test_chat_records_inputs_and_does_not_cache_identical_prompts() -> None:
    reply = ChatReply(
        "reply",
        "thinking",
        (ParsedToolCall("submit", {}, "{}", True),),
        Termination.STOP,
        Usage(3, 2, tokenizer="foreign"),
        {"response_id": "test"},
    )
    seen: list[tuple[Sequence[Msg], CallMeta]] = []

    def replies(messages: Sequence[Msg], meta: CallMeta) -> ChatReply:
        seen.append((messages, meta))
        return reply

    policy = ScriptedChatPolicy("chat", replies)
    assert isinstance(policy, ChatPolicy)
    assert policy.policy_id == "chat"
    assert policy.trainable is False
    messages = [Msg("user", "question")]
    tools = [ToolSpec("submit", "Submit", {})]
    meta = CallMeta(agent_id="peer1", role="peer", call_index=1)
    for index, call_meta in enumerate((None, meta)):
        result = await policy.chat(
            messages,
            tools,
            max_tokens=10,
            temperature=0.7,
            seed=42,
            cache_salt=f"salt-{index}",
            meta=call_meta,
        )
        assert result is reply
    messages.clear()
    tools.clear()
    assert len(policy.calls) == len(seen) == 2
    for index, call in enumerate(policy.calls):
        assert call.messages == (Msg("user", "question"),)
        assert call.tools == (ToolSpec("submit", "Submit", {}),)
        assert (call.max_tokens, call.temperature, call.seed) == (10, 0.7, 42)
        assert call.cache_salt == f"salt-{index}"
        assert call.meta == (CallMeta() if index == 0 else meta)
        assert seen[index] == (call.messages, call.meta)
    with pytest.raises(FrozenInstanceError):
        policy.calls[0].cache_salt = "changed"  # type: ignore[misc]


async def test_chat_rejects_empty_cache_salt_before_invoking_script() -> None:
    def replies(messages: Sequence[Msg], meta: CallMeta) -> ChatReply:
        pytest.fail("invalid calls must not invoke replies")

    policy = ScriptedChatPolicy("chat", replies)
    with pytest.raises(ValueError, match="non-empty cache_salt"):
        await policy.chat([], [], max_tokens=10, temperature=1.0, seed=0, cache_salt="")
    assert policy.calls == []
