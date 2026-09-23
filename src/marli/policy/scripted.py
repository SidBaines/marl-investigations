"""Scripted seats make multi-agent traces reproducible without a model backend.

The policy preserves completion ids and supplies aligned, seeded logprobs so
CPU tests can exercise the same recording and credit paths as real samplers.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marli.interact.scheduler import Clock
from marli.interact.types import Termination, Usage
from marli.policy.base import CallMeta, ChatReply, Sample, SamplingSpec
from marli.render.base import DeltaRenderer, Msg, ToolSpec
from marli.render.fake import FakeRenderer
from marli.seeds import derive_seed


@dataclass(frozen=True)
class ScriptCtx:
    prompt_ids: tuple[int, ...]
    prompt_text: str
    spec: SamplingSpec
    seed: int
    meta: CallMeta


ScriptFn = Callable[[ScriptCtx], Sequence[int]]


class ScriptedPolicy:
    def __init__(
        self,
        policy_id: str,
        renderer: DeltaRenderer,
        script: ScriptFn,
        *,
        trainable: bool = False,
        clock: Clock | None = None,
        latency_s: float | Callable[[ScriptCtx], float] = 0.0,
        policy_version: int | None = None,
    ) -> None:
        if not callable(latency_s) and latency_s < 0:
            raise ValueError(f"latency_s must be non-negative, got {latency_s!r}")
        self.policy_id = policy_id
        self.trainable = trainable
        self.renderer_name = renderer.name
        self.policy_version = policy_version
        self.calls: list[ScriptCtx] = []
        self._renderer = renderer
        self._script = script
        self._sleep = asyncio.sleep if clock is None else clock.sleep
        self._latency_s = latency_s

    async def sample(
        self,
        prompt_ids: Sequence[int],
        spec: SamplingSpec,
        *,
        seed: int,
        meta: CallMeta | None = None,
    ) -> Sample:
        prompt = tuple(prompt_ids)
        ctx = ScriptCtx(prompt, self._renderer.decode(prompt), spec, seed, meta or CallMeta())
        self.calls.append(ctx)
        scripted_ids = tuple(self._script(ctx))
        ids = scripted_ids[: spec.max_tokens]
        stops = spec.stop_token_ids or self._renderer.stop_token_ids
        termination = Termination.LENGTH
        if len(scripted_ids) <= spec.max_tokens and ids and ids[-1] in stops:
            termination = Termination.STOP
        logprobs = tuple(
            -3.0 + 3.0 * random.Random(derive_seed(seed, i)).random() for i in range(len(ids))
        )
        latency = self._latency_s(ctx) if callable(self._latency_s) else self._latency_s
        if latency < 0:
            raise ValueError(f"latency_s must be non-negative, got {latency!r}")
        if latency:
            await self._sleep(latency)
        return Sample(
            completion_ids=ids,
            logprobs=logprobs,
            termination=termination,
            policy_version=self.policy_version,
            usage=Usage(
                prompt_tokens=len(prompt), completion_tokens=len(ids), tokenizer=self.renderer_name
            ),
        )


@dataclass(frozen=True)
class Turn:
    """One assistant turn, including deliberately malformed tool bodies for tests."""

    content: str = ""
    thinking: str | None = None
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    raw_tool_bodies: tuple[str, ...] = ()
    stop: bool = True


def from_callable(fn: Callable[[ScriptCtx], Turn], renderer: FakeRenderer) -> ScriptFn:
    def script(ctx: ScriptCtx) -> Sequence[int]:
        turn = fn(ctx)
        return renderer.encode_completion(
            turn.content,
            thinking=turn.thinking,
            tool_calls=turn.tool_calls,
            raw_tool_bodies=turn.raw_tool_bodies,
            stop=turn.stop,
        )

    return script


def _lookup_turn(
    turns: Mapping[str, Sequence[Turn]], key: str, index: int, default: Turn | None, *, kind: str
) -> Turn:
    sequence = turns.get(key, ())
    if 0 <= index < len(sequence):
        return sequence[index]
    if default is not None:
        return default
    raise IndexError(
        f"no scripted turn for {kind} {key!r} at call index {index} ({len(sequence)} scripted)"
    )


def turns_by_agent(
    renderer: FakeRenderer,
    turns: Mapping[str, Sequence[Turn]],
    *,
    default: Turn | None = None,
) -> ScriptFn:
    """Select by agent id and call index; missing/exhausted agents use the default.

    Without a default, raise IndexError naming the agent, call index and scripted turn count.
    """

    def select(ctx: ScriptCtx) -> Turn:
        return _lookup_turn(turns, ctx.meta.agent_id, ctx.meta.call_index, default, kind="agent")

    return from_callable(select, renderer)


def by_role(
    renderer: FakeRenderer,
    turns: Mapping[str, Sequence[Turn]],
    *,
    default: Turn | None = None,
) -> ScriptFn:
    """Select by role and the caller's call index, without a shared role cursor.

    Missing/exhausted roles use the default. Without one, raise IndexError naming the
    role, call index and scripted turn count.
    """

    def select(ctx: ScriptCtx) -> Turn:
        return _lookup_turn(turns, ctx.meta.role, ctx.meta.call_index, default, kind="role")

    return from_callable(select, renderer)


@dataclass(frozen=True)
class ChatCall:
    """Chat inputs retained for assertions, including the required cache salt."""

    messages: tuple[Msg, ...]
    tools: tuple[ToolSpec, ...]
    max_tokens: int
    temperature: float
    seed: int
    cache_salt: str
    meta: CallMeta


class ScriptedChatPolicy:
    trainable: bool = False

    def __init__(
        self, policy_id: str, replies: Callable[[Sequence[Msg], CallMeta], ChatReply]
    ) -> None:
        self.policy_id = policy_id
        self.calls: list[ChatCall] = []
        self._replies = replies

    async def chat(
        self,
        messages: Sequence[Msg],
        tools: Sequence[ToolSpec],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        cache_salt: str,
        meta: CallMeta | None = None,
    ) -> ChatReply:
        if not cache_salt:
            raise ValueError("ScriptedChatPolicy requires a non-empty cache_salt for every call")
        call = ChatCall(
            tuple(messages),
            tuple(tools),
            max_tokens,
            temperature,
            seed,
            cache_salt,
            meta or CallMeta(),
        )
        self.calls.append(call)
        return self._replies(call.messages, call.meta)
