"""Runtime traces test observations, actions, and boundaries through real integration."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import replace

import pytest
from _marli_test_envs import ArithEnv, ScratchEnv

from marli.errors import BackendError, BudgetExceededError, ConfigError, MarliError
from marli.interact.agent import AgentRuntime
from marli.interact.limits import AgentLimits, CallLimits, EpisodeLimits, Limits, SessionLimits
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.scheduler import FakeClock
from marli.interact.system import AgentResult, ContextSpec, Protocol, RoleSpec, SystemIO
from marli.interact.tools import ToolCtx, ToolResult
from marli.interact.types import (
    Episode,
    EventKind,
    Outcome,
    Purpose,
    ReadVia,
    SegmentStart,
    Termination,
    Usage,
    record_to_dict,
)
from marli.interact.workspace import DeliverySpec
from marli.policy.base import CallMeta, ChatReply
from marli.policy.scripted import (
    ScriptCtx,
    ScriptedChatPolicy,
    ScriptedPolicy,
    Turn,
    turns_by_agent,
)
from marli.render.base import Msg, ParsedToolCall, ToolSpec
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer


class RuntimeProtocol(Protocol):
    name = "runtime_test"

    def __init__(
        self,
        *,
        tools: tuple[str, ...] = ("submit",),
        count: int = 1,
        context: ContextSpec | None = None,
    ) -> None:
        self.role = RoleSpec(
            "solver",
            tools,
            "Solve {agent_id} of {n_agents}.",
            count,
            context or ContextSpec(),
        )
        self.results: list[AgentResult] = []

    def roles(self) -> list[RoleSpec]:
        return [self.role]

    def adjust_limits(self, limits: Limits) -> Limits:
        # This test protocol exercises runtime session limits directly.
        return limits

    async def run(self, io: SystemIO) -> Outcome:
        handles = [
            await io.start_agent(
                "solver",
                agent_id=f"solver{i}",
                seat_key=("solver", i),
                first_message=io.task.prompt,
            )
            for i in range(self.role.count or 0)
        ]
        self.results = await io.wait(handles)
        answers = {result.agent_id: result.submission for result in self.results}
        return Outcome(self.results[-1].submission, answers, "test")


def spec_for(
    turns: dict[str, list[Turn]],
    protocol: RuntimeProtocol | None = None,
    *,
    renderer: FakeRenderer | None = None,
    limits: Limits | None = None,
) -> EpisodeSpec:
    env = ScratchEnv()
    renderer = renderer or FakeRenderer()
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, turns))
    policy.deterministic = True
    return EpisodeSpec(
        protocol or RuntimeProtocol(),
        env,
        env.task,
        {"solver": "script"},
        {"script": policy},
        {"script": type(renderer)},
        limits or Limits(),
        run_seed=17,
    )


def submit(answer: str = "5") -> Turn:
    return Turn(tool_calls=(("submit", {"answer": answer}),))


def assert_token_buffers(
    episode: Episode, buffers: dict[str, list[int]], policy: ScriptedPolicy
) -> None:
    prompts = {(ctx.meta.agent_id, ctx.meta.call_index): ctx.prompt_ids for ctx in policy.calls}
    for call in episode.calls:
        buf = buffers[call.segment_id]
        index = int(call.call_id.rsplit("/c", 1)[1])
        assert tuple(buf[: call.prompt_len]) == prompts[call.agent_id, index]
        assert tuple(buf[call.prompt_len : call.prompt_len + len(call.completion_ids)]) == (
            call.completion_ids
        )


def forced_answer(renderer: FakeRenderer) -> list[int]:
    return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]


def assert_tool_pairs(messages: Sequence[Msg]) -> None:
    pending: list[str] = []
    for msg in messages:
        if msg.role == "tool":
            assert pending and msg.tool_call_id == pending.pop(0)
        else:
            assert not pending
            if msg.role == "assistant":
                assert all(call.id for call in msg.tool_calls)
                pending = [call.id for call in msg.tool_calls]
    assert not pending


async def test_tool_loop_preserves_every_prompt_and_sampled_offset() -> None:
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"))
    spec = spec_for(
        {
            "solver0": [
                Turn(thinking="work", tool_calls=(("write_scratchpad", {"content": "2+3=5"}),)),
                submit(),
            ]
        },
        protocol,
    )
    episode, buffers = await run_episode(spec)
    policy = spec.policies["script"]
    renderer = FakeRenderer()
    assert episode.ok and len(episode.calls) == 2
    assert len(episode.segments) == 1
    first, second = episode.calls
    buf = buffers[first.segment_id]
    for call, actual in zip(episode.calls, policy.calls, strict=True):
        assert tuple(buf[: call.prompt_len]) == actual.prompt_ids
        assert (
            tuple(buf[call.prompt_len : call.prompt_len + len(call.completion_ids)])
            == call.completion_ids
        )
    expected_delta = renderer.continuation(
        "stop",
        [
            Msg(
                "tool",
                "ok: scratchpad v1 (5 chars) (visible to others next tick)",
                name="write_scratchpad",
            )
        ],
    )
    assert buf[first.prompt_len + len(first.completion_ids) : second.prompt_len] == expected_delta
    assert episode.workspace_log[0].content == "2+3=5"
    assert protocol.results[0].ended_by == "submit"
    for call in episode.calls:
        start = episode.events[call.seq]
        assert start.kind == EventKind.CALL_START and start.data["call_id"] == call.call_id
    assert [event.kind for event in episode.events].count(EventKind.TOOL_END) == 2


@pytest.mark.parametrize(
    ("mode", "expected", "calls"),
    [
        ("nudge", "5", 2),
        ("end_agent", None, 1),
        ("final_text_as_answer", "5", 1),
    ],
)
async def test_no_tool_modes(mode: str, expected: str | None, calls: int) -> None:
    cfg = Limits(on_no_tool_call=mode)
    spec = spec_for({"solver0": [Turn("5"), submit()]}, limits=cfg)
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == expected
    assert len(episode.calls) == calls
    if mode == "nudge":
        assert "Please use a tool" in spec.policies["script"].calls[1].prompt_text


async def test_nudges_are_bounded_and_tool_attempts_reset_them() -> None:
    turns = [Turn("thinking"), Turn(raw_tool_bodies=("broken",)), Turn("more"), submit()]
    spec = spec_for({"solver0": turns}, limits=Limits(max_nudges=1))
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5"
    assert episode.calls[1].tool_calls[0].result == "error: could not parse tool call"
    assert "error: could not parse tool call" in spec.policies["script"].calls[2].prompt_text
    spec = spec_for({"solver0": [Turn("a"), Turn("b")]}, limits=Limits(max_nudges=1))
    episode, _ = await run_episode(spec)
    assert len(episode.calls) == 2 and episode.limits_hit == {"solver0": ("max_nudges",)}


async def test_forced_final_prefix_is_observation_and_parsed_with_completion() -> None:
    class SuppressingRenderer(FakeRenderer):
        def suppress_thinking_prefix(self) -> list[int]:
            return [S["think"], S["/think"]]

    renderer = SuppressingRenderer()
    cfg = Limits(agent=AgentLimits(max_calls=1), call=CallLimits(min_call_tokens=1))
    spec = spec_for({"solver0": []}, renderer=renderer, limits=cfg)

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "act":
            return renderer.encode_completion("work")
        return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]

    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    final = episode.calls[-1]
    assert episode.outcome.final_answer == "5"
    assert final.purpose == Purpose.FINAL and final.forced
    assert final.tool_calls[0].name == "submit" and final.tool_calls[0].parsed_ok
    # FINAL calls use only the forced tool prefix (it skips reasoning natively);
    # suppress_thinking_prefix is for summary calls (COMPACT/CARRY).
    prefix = renderer.forced_tool_prefix("submit")
    assert renderer.suppress_thinking_prefix() != [] and final.prompt_len >= len(prefix)
    assert buffers[final.segment_id][final.prompt_len - len(prefix) : final.prompt_len] == prefix
    assert list(final.completion_ids) == script(policy.calls[-1])
    assert (
        "You have run out of budget. Submit your final answer now." in policy.calls[-1].prompt_text
    )
    assert episode.limits_hit == {"solver0": ("agent.max_calls",)}


async def test_on_exhaust_none_and_max_ticks() -> None:
    cfg = Limits(on_exhaust="none")
    cfg.episode.max_ticks = 1
    spec = spec_for({"solver0": [Turn("work")]}, limits=cfg)
    episode, _ = await run_episode(spec)
    assert len(episode.calls) == 1 and episode.outcome.final_answer is None
    assert episode.limits_hit == {"_episode": ("episode.max_ticks",)}
    assert spec.protocol.results[0].ended_by == "max_ticks"


async def test_compaction_drops_pending_observation_and_starts_summary_segment() -> None:
    protocol = RuntimeProtocol(
        tools=("submit", "write_scratchpad"),
        context=ContextSpec(kind="compaction", compact_threshold=1000, compact_reserve=512),
    )
    cfg = Limits()
    cfg.session.carry_max_tokens = 256  # the summary cap must fit in compact_reserve
    spec = spec_for(
        {
            "solver0": [
                Turn(tool_calls=(("write_scratchpad", {"content": "x" * 600}),)),
                Turn("We found five."),
                submit(),
            ]
        },
        protocol,
        limits=cfg,
    )
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.COMPACT, Purpose.ACT]
    assert [seg.start_reason for seg in episode.segments] == [
        SegmentStart.START,
        SegmentStart.COMPACTION,
    ]
    assert episode.segments[1].carry_from == episode.segments[0].segment_id
    compact_prompt = spec.policies["script"].calls[1].prompt_text
    assert "summary you write in this reply" in compact_prompt
    assert "ok: scratchpad" not in compact_prompt
    new_prompt = spec.policies["script"].calls[2].prompt_text
    assert "We found five." in new_prompt and "What is 2+3?" in new_prompt
    for call in episode.calls:
        buf = buffers[call.segment_id]
        assert (
            tuple(buf[call.prompt_len : call.prompt_len + len(call.completion_ids)])
            == call.completion_ids
        )


async def test_sessions_with_notes_include_same_turn_write() -> None:
    protocol = RuntimeProtocol(tools=("submit", "end_session"), context=ContextSpec(kind="notes"))
    cfg = Limits(session=SessionLimits(max_sessions=2))
    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(("write_notes", {"content": "answer is five"}), ("end_session", {}))
                ),
                submit(),
            ]
        },
        protocol,
        limits=cfg,
    )
    episode, _ = await run_episode(spec)
    assert [seg.start_reason for seg in episode.segments] == [
        SegmentStart.START,
        SegmentStart.SESSION,
    ]
    assert [call.session_idx for call in episode.calls] == [0, 1]
    assert "[Your notes]\nanswer is five" in spec.policies["script"].calls[1].prompt_text
    assert protocol.results[0].n_sessions == 2
    assert episode.metrics["n_sessions"] == 2


@pytest.mark.parametrize("exhausted", [False, True])
async def test_session_carry_on_request_or_budget(exhausted: bool) -> None:
    renderer = FakeRenderer()
    first = Turn("x" * 199) if exhausted else Turn(tool_calls=(("end_session", {}),))
    protocol = RuntimeProtocol(
        tools=("submit", "end_session"), context=ContextSpec(kind="compaction")
    )
    cfg = Limits(session=SessionLimits(2, 500, 300, 300), call=CallLimits(min_call_tokens=1))
    spec = spec_for({"solver0": [first, Turn("Carry five."), submit()]}, protocol, limits=cfg)
    episode, _ = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.CARRY, Purpose.ACT]
    assert episode.calls[1].forced == exhausted
    assert episode.calls[1].session_idx == 0 and episode.calls[2].session_idx == 1
    assert "Carry five." in renderer.decode(spec.policies["script"].calls[2].prompt_ids)
    assert episode.outcome.final_answer == "5"


async def test_single_session_end_session_ends_agent() -> None:
    protocol = RuntimeProtocol(tools=("submit", "end_session"))
    spec = spec_for({"solver0": [Turn(tool_calls=(("end_session", {}),))]}, protocol)
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer is None
    assert protocol.results[0].ended_by == "end_agent"


async def test_lockstep_delivers_only_other_agents_previous_tick_and_replays() -> None:
    async def once() -> tuple[Episode, dict[str, list[int]]]:
        protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"), count=2)
        spec = spec_for(
            {
                f"solver{i}": [
                    Turn(tool_calls=(("write_scratchpad", {"content": f"note {i}"}),)),
                    submit(),
                ]
                for i in range(2)
            },
            protocol,
        )
        episode, buffers = await run_episode(spec)
        for ctx in spec.policies["script"].calls:
            if ctx.meta.call_index == 1:
                peer = "solver1" if ctx.meta.agent_id == "solver0" else "solver0"
                assert f"[workspace] {peer} wrote scratchpad v1" in ctx.prompt_text
                assert f"[workspace] {ctx.meta.agent_id} wrote" not in ctx.prompt_text
        return episode, buffers

    first, second = await once(), await once()
    assert first == second
    episode, _ = first
    assert episode.replayable
    assert [write.writer for write in episode.workspace_log] == ["solver0", "solver1"]
    for call in episode.calls:
        assert call.tick in (0, 1)
        if call.tick == 0:
            assert not call.reads
        else:
            assert len(call.reads) == 1
            assert call.reads[0].via == ReadVia.NOTIFY and call.reads[0].version == 1
            assert call.reads[0].writer != call.agent_id


@pytest.mark.parametrize("delivery_role", ["tool", "append_tool", "user"])
async def test_delivery_roles_and_pull_reads_are_attached_to_next_prompt(
    delivery_role: str,
) -> None:
    class DeliveryRenderer(FakeRenderer):
        pass

    DeliveryRenderer.delivery_role = delivery_role
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad", "read_scratchpad"), count=2)
    turns = {
        f"solver{i}": [
            Turn(tool_calls=(("write_scratchpad", {"content": f"note {i}"}),)),
            Turn(tool_calls=(("read_scratchpad", {"agent_id": f"solver{1 - i}"}),)),
            submit(),
        ]
        for i in range(2)
    }
    spec = spec_for(turns, protocol, renderer=DeliveryRenderer())
    episode, _ = await run_episode(spec)
    for call in episode.calls:
        if call.tick == 1:
            assert [read.via for read in call.reads] == [ReadVia.NOTIFY]
        if call.tick == 2:
            assert [read.via for read in call.reads] == [ReadVia.PULL]
    ctx = next(ctx for ctx in spec.policies["script"].calls if ctx.meta.call_index == 1)
    marker = ctx.prompt_text.index("[workspace]")
    if delivery_role == "user":
        assert ctx.prompt_text[marker - len("⟨user⟩") : marker] == "⟨user⟩"
    elif delivery_role == "tool":
        assert ctx.prompt_text[marker - len("⟨res⟩") : marker] == "⟨res⟩"
    else:
        assert ctx.prompt_text[marker - 2 : marker] == "\n\n"


async def test_rerender_keeps_assistant_thinking_and_new_segment_per_call() -> None:
    class FullRenderer(FakeRenderer):
        supports_delta = False

    spec = spec_for(
        {"solver0": [Turn("work", thinking="secret reasoning"), submit()]}, renderer=FullRenderer()
    )
    episode, buffers = await run_episode(spec)
    assert len(episode.segments) == 2
    assert all(seg.start_reason == SegmentStart.RERENDER for seg in episode.segments)
    assert "⟨think⟩secret reasoning⟨/think⟩" in spec.policies["script"].calls[1].prompt_text
    for call, ctx in zip(episode.calls, spec.policies["script"].calls, strict=True):
        assert buffers[call.segment_id] == list(ctx.prompt_ids + call.completion_ids)


async def test_backend_failure_ends_only_failed_agent() -> None:
    protocol = RuntimeProtocol(count=2)
    spec = spec_for({"solver0": [], "solver1": []}, protocol)
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.agent_id == "solver0":
            raise BackendError("backend unavailable")
        return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert not episode.ok and "backend unavailable" in episode.errors[0]
    assert [result.ended_by for result in protocol.results] == ["error", "submit"]
    failed = next(call for call in episode.calls if call.agent_id == "solver0")
    assert failed.termination == Termination.ERROR
    assert episode.outcome.submissions["solver1"] == "5"


async def test_tools_after_control_are_recorded_errors_and_not_executed() -> None:
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"))
    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(
                        ("submit", {"answer": "5"}),
                        ("write_scratchpad", {"content": "forbidden"}),
                    )
                )
            ]
        },
        protocol,
    )
    episode, _ = await run_episode(spec)
    assert not episode.workspace_log
    assert episode.calls[0].tool_calls[1].error == "tool call after control tool"


async def test_env_tools_are_filtered_and_output_is_truncated() -> None:
    class EchoTool:
        shared = False
        control = False
        spec = ToolSpec("echo", "Echo", {"type": "object", "properties": {}})

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            return ToolResult("X" * 200)

    class ToolEnv(ArithEnv):
        def tools(self, role: str) -> list[EchoTool]:
            return [EchoTool()]

    protocol = RuntimeProtocol(tools=("submit", "echo"))
    spec = spec_for(
        {"solver0": [Turn(tool_calls=(("echo", {}),)), submit()]},
        protocol,
        limits=Limits(tool_output_chars=40),
    )
    spec.env = ToolEnv()
    episode, _ = await run_episode(spec)
    result = episode.calls[0].tool_calls[0].result
    assert len(result) <= 40 and "truncated" in result
    assert result in spec.policies["script"].calls[1].prompt_text


async def test_async_writes_are_recorded_and_not_replayable() -> None:
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"))
    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(
                        ("write_scratchpad", {"content": "first"}),
                        ("write_scratchpad", {"content": "second"}),
                    )
                ),
                submit(),
            ]
        },
        protocol,
    )
    spec.schedule = "async"
    episode, _ = await run_episode(spec)
    assert not episode.replayable and all(call.tick is None for call in episode.calls)
    assert [write.version for write in episode.workspace_log] == [1, 2]
    assert episode.workspace_log[-1].content == "first\nsecond"
    assert len([event for event in episode.events if event.kind == EventKind.COMMIT]) == 2


async def test_programming_errors_propagate_and_teardown_runs() -> None:
    spec = spec_for({"solver0": []})

    def broken(ctx: ScriptCtx) -> Sequence[int]:
        raise TypeError("programming mistake")

    spec.policies["script"] = ScriptedPolicy("script", FakeRenderer(), broken)
    with pytest.raises(TypeError, match="programming mistake"):
        await run_episode(spec)
    assert spec.env.teardown_count == 1


async def test_compaction_can_discard_delta_that_would_exceed_context_cap() -> None:
    class VerboseTool:
        shared = False
        control = False
        spec = ToolSpec("verbose", "Return a long observation.", {"properties": {}})

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            return ToolResult("y" * 1500)

    class VerboseEnv(ScratchEnv):
        def tools(self, role: str) -> list[VerboseTool]:
            return [VerboseTool()]

    protocol = RuntimeProtocol(
        tools=("submit", "verbose"),
        context=ContextSpec(kind="compaction", compact_threshold=1100, compact_reserve=300),
    )
    cfg = Limits()
    cfg.ctx.max_ctx = 1800
    cfg.session.carry_max_tokens = 256
    spec = spec_for(
        {
            "solver0": [
                Turn(tool_calls=(("verbose", {}),)),
                Turn("Summary five."),
                submit(),
            ]
        },
        protocol,
        limits=cfg,
    )
    spec.env = VerboseEnv()
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.COMPACT, Purpose.ACT]
    assert episode.outcome.final_answer == "5" and not episode.limits_hit
    assert all(len(buf) <= cfg.ctx.max_ctx for buf in buffers.values())
    assert "y" * 100 not in spec.policies["script"].calls[1].prompt_text


async def test_tick_zero_delivery_is_in_first_user_message() -> None:
    class PrimedProtocol(RuntimeProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            handles = [
                await io.start_agent(
                    "solver",
                    agent_id=f"solver{i}",
                    seat_key=("solver", i),
                    first_message=io.task.prompt,
                )
                for i in range(2)
            ]
            io.workspace.write(
                "solver1", "scratchpad", "initial note", mode="overwrite", tick=None, seq=0
            )
            for write in io.workspace.commit_staged(["solver1"]):
                io.recorder.add_write(write)
            results = await io.wait(handles)
            return Outcome(
                results[0].submission,
                {result.agent_id: result.submission for result in results},
                "primed",
            )

    spec = spec_for({"solver0": [submit()], "solver1": [submit()]}, PrimedProtocol(count=2))
    episode, _ = await run_episode(spec)
    ctx = next(ctx for ctx in spec.policies["script"].calls if ctx.meta.agent_id == "solver0")
    assert "⟨user⟩What is 2+3?\n\n[workspace] solver1 wrote scratchpad v1" in ctx.prompt_text
    call = next(call for call in episode.calls if call.agent_id == "solver0")
    assert call.tick == 0 and call.reads[0].writer == "solver1"


async def test_last_session_forces_final_and_tail_is_observation() -> None:
    protocol = RuntimeProtocol(
        tools=("submit", "end_session"), context=ContextSpec(kind="tail", tail_tokens=12)
    )
    cfg = Limits(session=SessionLimits(max_sessions=2))
    spec = spec_for({"solver0": []}, protocol, limits=cfg)
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "final":
            return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion("tail content", tool_calls=(("end_session", {}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.ACT, Purpose.FINAL]
    assert episode.outcome.final_answer == "5"
    second = episode.calls[1]
    prompt = buffers[second.segment_id][: second.prompt_len]
    assert "[Tail of your previous session]\ntail content⟨eot⟩⟨asst⟩" in renderer.decode(prompt)
    assert prompt[-1] == S["asst"]
    assert_token_buffers(episode, buffers, spec.policies["script"])
    assert episode.limits_hit["solver0"] == ("session.max_sessions",)


async def test_tools_receive_call_context_and_renderers_are_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marli.interact.tools import TOOLS
    from marli.registry import FnRegistry

    registry = FnRegistry("test-tools")
    seen: list[ToolCtx] = []
    for name in TOOLS.names():
        registry.register(name)(TOOLS.get(name))

    @registry.register("context_tool")
    class ContextTool:
        shared = False
        control = False
        spec = ToolSpec("context_tool", "Context", {"properties": {}})

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            seen.append(ctx)
            return ToolResult(ctx.agent_id)

    monkeypatch.setattr("marli.interact.agent.TOOLS", registry)
    monkeypatch.setattr("marli.interact.run.TOOLS", registry)
    made: list[FakeRenderer] = []

    def factory() -> FakeRenderer:
        renderer = FakeRenderer()
        made.append(renderer)
        return renderer

    protocol = RuntimeProtocol(tools=("submit", "context_tool"), count=2)
    probe = Turn(tool_calls=(("context_tool", {}),))
    spec = spec_for({"solver0": [probe, submit()], "solver1": [probe, submit()]}, protocol)
    spec.renderers["script"] = factory
    await run_episode(spec)
    assert sorted(ctx.agent_id for ctx in seen) == ["solver0", "solver1"]
    assert len(made) == 3 and len({id(renderer) for renderer in made}) == 3


@pytest.mark.parametrize("report", [True, False])
async def test_worker_handle_reservation_forced_report_and_fallback(report: bool) -> None:
    class WorkerProtocol(Protocol):
        name = "worker_test"
        results: list[AgentResult]

        def roles(self) -> list[RoleSpec]:
            return [
                RoleSpec("solver", ("submit",), "Solve."),
                RoleSpec("worker", ("return_report",), "Report.", count=None, limits_key="worker"),
            ]

        async def run(self, io: SystemIO) -> Outcome:
            parent = await io.start_agent(
                "solver", agent_id="solver0", seat_key=("solver", 0), first_message=io.task.prompt
            )
            assert parent.agent_id == "solver0"
            assert io.ledger.reserve_workers("solver0", 1) == 1
            worker = await io.start_agent(
                "worker",
                agent_id="worker0",
                seat_key=("worker", 0),
                first_message="Help solve.",
                parent="solver0",
            )
            self.results = await io.wait([parent, worker])
            return Outcome(
                self.results[0].submission, {"solver0": self.results[0].submission}, "worker"
            )

    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.agent_id == "solver0":
            return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))
        if ctx.meta.purpose == "act":
            return renderer.encode_completion("working")
        if report:
            return [*renderer.encode_text('{"report": "five"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion("no valid report")

    env = ScratchEnv()
    protocol = WorkerProtocol()
    spec = EpisodeSpec(
        protocol,
        env,
        env.task,
        {"solver": "script", "worker": "script"},
        {"script": ScriptedPolicy("script", renderer, script)},
        {"script": FakeRenderer},
        Limits(worker=AgentLimits(max_calls=1)),
    )
    episode, buffers = await run_episode(spec)
    worker_result = protocol.results[1]
    assert worker_result.report == ("five" if report else None)
    assert worker_result.ended_by == "budget"
    worker_segment = next(segment for segment in episode.segments if segment.agent_id == "worker0")
    assert worker_segment.start_reason == SegmentStart.SPAWN
    call = next(call for call in episode.calls if call.purpose == Purpose.REPORT)
    assert call.forced and call.role == "worker"
    assert (
        renderer.forced_tool_prefix("return_report")
        == buffers[call.segment_id][
            call.prompt_len - len(renderer.forced_tool_prefix("return_report")) : call.prompt_len
        ]
    )
    assert episode.limits_hit["worker0"] == ("worker.max_calls",)


async def test_no_tool_delivery_uses_user_message() -> None:
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"), count=2)
    spec = spec_for(
        {
            "solver0": [Turn("thinking"), submit()],
            "solver1": [Turn(tool_calls=(("write_scratchpad", {"content": "note"}),)), submit()],
        },
        protocol,
    )
    await run_episode(spec)
    ctx = next(
        ctx
        for ctx in spec.policies["script"].calls
        if ctx.meta.agent_id == "solver0" and ctx.meta.call_index == 1
    )
    assert "⟨user⟩[workspace] solver1" in ctx.prompt_text


async def test_role_permissions_restrict_reads() -> None:
    from marli.interact.workspace import Permissions

    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad", "read_scratchpad"), count=2)
    protocol.role = replace(protocol.role, permissions=Permissions(read_others=False))
    spec = spec_for(
        {
            "solver0": [
                Turn(tool_calls=(("write_scratchpad", {"content": "secret"}),)),
                Turn(tool_calls=(("write_scratchpad", {"content": "more"}),)),
                submit(),
            ],
            "solver1": [
                Turn(tool_calls=(("write_scratchpad", {"content": "mine"}),)),
                Turn(tool_calls=(("read_scratchpad", {"agent_id": "solver0"}),)),
                submit(),
            ],
        },
        protocol,
    )
    episode, _ = await run_episode(spec)
    assert not any(read.writer == "solver0" for call in episode.calls for read in call.reads)
    assert "secret" not in spec.policies["script"].calls[-1].prompt_text


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_publish_final_text_records_visible_content_once_per_act(schedule: str) -> None:
    protocol = RuntimeProtocol(count=2)
    protocol.role = replace(protocol.role, publish_final_text=True)
    spec = spec_for(
        {
            f"solver{i}": [
                Turn(f"visible {i}", thinking=f"private {i}"),
                Turn(f"revised {i}", tool_calls=(("submit", {"answer": "5"}),)),
            ]
            for i in range(2)
        },
        protocol,
    )
    spec.schedule = schedule
    episode, _ = await run_episode(spec)
    assert episode.ok
    assert len([event for event in episode.events if event.kind == EventKind.COMMIT]) == 4
    for i in range(2):
        writes = [write for write in episode.workspace_log if write.writer == f"solver{i}"]
        assert [write.content for write in writes] == [f"visible {i}", f"revised {i}"]
        assert [write.version for write in writes] == [1, 2]
        assert all("private" not in write.content for write in writes)
        if schedule == "lockstep":
            calls = [call for call in episode.calls if call.agent_id == f"solver{i}"]
            assert calls[0].reads == ()
            assert [(read.writer, read.version) for read in calls[1].reads] == [
                (f"solver{1 - i}", 1)
            ]


async def test_publish_final_text_skips_empty_content_and_forced_calls() -> None:
    protocol = RuntimeProtocol()
    protocol.role = replace(protocol.role, publish_final_text=True)
    limits = Limits(agent=AgentLimits(max_calls=1))
    spec = spec_for({}, protocol, limits=limits)
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> list[int]:
        if ctx.meta.purpose == "act":
            return renderer.encode_completion(thinking="unpublished reasoning")
        return (
            renderer.encode_text('{"answer": "5"}}')
            + [S["/call"]]
            + renderer.encode_completion("final-only content")
        )

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5"
    assert not episode.workspace_log
    assert not any(event.kind == EventKind.COMMIT for event in episode.events)


@pytest.mark.parametrize("on_exhaust", ["force_final", "none"])
async def test_stop_request_before_first_turn_respects_exhaustion(on_exhaust: str) -> None:
    class StoppedProtocol(RuntimeProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            handle = await io.start_agent(
                "solver", agent_id="solver0", seat_key=("solver", 0), first_message=io.task.prompt
            )
            io.stop(handle, "consensus")
            io.stop(handle, "later reason")
            result = await handle
            self.results = [result]
            io.stop(handle, "already done")
            return Outcome(result.submission, {result.agent_id: result.submission}, "test")

    protocol = StoppedProtocol()
    spec = spec_for({}, protocol, limits=Limits(on_exhaust=on_exhaust))
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> list[int]:
        assert ctx.meta.purpose == "final"
        return renderer.encode_text('{"answer": "5"}}') + [S["/call"], S["eot"]]

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert episode.ok and protocol.results[0].ended_by == "consensus"
    assert len(episode.calls) == (1 if on_exhaust == "force_final" else 0)
    assert episode.outcome.final_answer == ("5" if on_exhaust == "force_final" else None)
    if episode.calls:
        assert episode.calls[0].forced and episode.calls[0].purpose == Purpose.FINAL


async def test_stop_with_existing_submission_does_not_sample() -> None:
    class AnsweredProtocol(RuntimeProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            handle = await io.start_agent(
                "solver", agent_id="solver0", seat_key=("solver", 0), first_message=io.task.prompt
            )
            io.runtimes[handle.agent_id].submission = "5"
            io.stop(handle, "consensus")
            result = await handle
            self.results = [result]
            return Outcome(result.submission, {result.agent_id: result.submission}, "test")

    protocol = AnsweredProtocol()
    episode, _ = await run_episode(spec_for({}, protocol))
    assert not episode.calls
    assert episode.outcome.final_answer == "5" and protocol.results[0].ended_by == "consensus"


async def test_publish_final_text_with_no_tools_keeps_answer_and_scratchpad() -> None:
    protocol = RuntimeProtocol(tools=())
    protocol.role = replace(protocol.role, publish_final_text=True)
    spec = spec_for(
        {"solver0": [Turn("visible answer", thinking="private reasoning")]},
        protocol,
        limits=Limits(on_no_tool_call="final_text_as_answer"),
    )
    episode, _ = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "visible answer"
    assert len(episode.calls) == len(episode.workspace_log) == 1
    assert not episode.calls[0].tool_calls
    assert episode.workspace_log[0].content == "visible answer"


@pytest.mark.parametrize("reset", ["notes", "compaction", "both", "tail"])
async def test_harness_final_on_fresh_segment(reset: str) -> None:
    context = ContextSpec(
        kind=reset, compact_threshold=600 if reset == "compaction" else 0, tail_tokens=12
    )
    protocol = RuntimeProtocol(tools=("submit", "end_session", "write_scratchpad"), context=context)
    cfg = Limits(
        agent=AgentLimits(max_calls=2 if reset in {"compaction", "both"} else 1),
        session=SessionLimits(max_sessions=2),
    )
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "final":
            return forced_answer(renderer)
        if ctx.meta.purpose in {"compact", "carry"}:
            return renderer.encode_completion("summary: five")
        if reset == "compaction":
            return renderer.encode_completion(
                tool_calls=(("write_scratchpad", {"content": "x" * 500}),)
            )
        return renderer.encode_completion(tool_calls=(("end_session", {}),))

    policy = ScriptedPolicy("script", renderer, script)
    spec = spec_for({}, protocol, limits=cfg)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    final = episode.calls[-1]
    assert final.purpose == Purpose.FINAL and final.forced
    assert episode.outcome.final_answer == "5"
    assert protocol.results[0].ended_by == "budget"
    assert final.segment_id != episode.calls[-2].segment_id
    assert final.tick == 1
    prompt = policy.calls[-1].prompt_text
    assert "⟨eot⟩⟨user⟩You have run out of budget" in prompt
    assert "⟨asst⟩⟨user⟩" not in prompt
    if reset == "compaction":
        assert "[Summary of your earlier work]" in prompt
        assert "Previous session summary" not in prompt
    if reset == "both":
        assert "[Previous session summary]" in prompt
    assert_token_buffers(episode, buffers, policy)


@pytest.mark.parametrize("mode", ["notify", "push"])
@pytest.mark.parametrize("reset", ["compaction", "notes", "both"])
async def test_deliveries_survive_compaction_and_session_reset(mode: str, reset: str) -> None:
    renderer = FakeRenderer()
    protocol = RuntimeProtocol(
        tools=("submit", "write_scratchpad", "end_session"),
        count=2,
        context=ContextSpec(kind=reset, compact_threshold=1400 if reset == "compaction" else 0),
    )
    spec = spec_for({}, protocol, limits=Limits(session=SessionLimits(max_sessions=2)))
    spec.delivery = DeliverySpec(mode=mode)

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose in {"compact", "carry"}:
            return renderer.encode_completion("summary")
        if ctx.meta.call_index == 0:
            if ctx.meta.agent_id == "solver1":
                calls = (("write_scratchpad", {"content": "PEER-SECRET-VALUE"}),)
            elif reset == "compaction":
                calls = (("write_scratchpad", {"content": "z" * 900}),)
            else:
                calls = (("end_session", {}),)
            return renderer.encode_completion(tool_calls=calls)
        return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))

    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    calls = [call for call in episode.calls if call.agent_id == "solver0"]
    final = calls[-1]
    assert final.purpose == Purpose.ACT and final.tick == 1
    assert final.reads == tuple(read for call in calls for read in call.reads)
    (read,) = final.reads
    assert (read.writer, read.version, read.via) == ("solver1", 1, ReadVia(mode))
    prompt = renderer.decode(buffers[final.segment_id][: final.prompt_len])
    assert "PEER-SECRET-VALUE" in prompt
    assert "ok: scratchpad" not in prompt and "Please use a tool" not in prompt
    session = 1 if reset == "compaction" else 2
    label = f"[Session {session} of 2]"
    if session == 2:
        label += " — final session: you must submit before it ends"
    assert f"⟨user⟩{label}\n\nWhat is 2+3?" in prompt
    assert_token_buffers(episode, buffers, policy)


async def test_compaction_never_loops_on_a_fresh_segment() -> None:
    protocol = RuntimeProtocol(context=ContextSpec(kind="compaction", compact_threshold=1))
    spec = spec_for({"solver0": [Turn("work"), Turn("summary"), submit()]}, protocol)
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.COMPACT, Purpose.ACT]
    assert [call.tick for call in episode.calls] == [0, 1, 1]
    assert episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, spec.policies["script"])


@pytest.mark.parametrize("on_exhaust", ["none", "force_final"])
async def test_fresh_prompt_over_max_ctx_ends_without_compaction(on_exhaust: str) -> None:
    protocol = RuntimeProtocol(context=ContextSpec(kind="compaction", compact_threshold=1))
    cfg = Limits(on_exhaust=on_exhaust)
    cfg.ctx.max_ctx = 1100
    protocol.role = replace(protocol.role, system_prompt="x" * 2000)
    spec = spec_for({}, protocol, limits=cfg)
    episode, buffers = await run_episode(spec)
    assert not episode.calls and not episode.segments
    assert protocol.results[0].ended_by == "ctx"
    assert episode.limits_hit["solver0"] == ("ctx.max_ctx",)
    assert_token_buffers(episode, buffers, spec.policies["script"])


@pytest.mark.parametrize(
    ("schedule", "exhaustion"),
    [
        ("lockstep", "ctx"),
        ("lockstep", "episode"),
        ("lockstep", "max_ticks"),
        ("async", "ctx"),
        ("async", "episode"),
    ],
)
async def test_exhaustion_reserves_forced_final(exhaustion: str, schedule: str) -> None:
    cfg = Limits(
        agent=AgentLimits(max_gen_tokens=100000, final_reserve=64),
        call=CallLimits(max_tokens=100000),
        episode=EpisodeLimits(max_gen_tokens=300 if exhaustion == "episode" else 131072),
    )
    cfg.ctx.max_ctx = 2000 if exhaustion == "ctx" else 32768
    cfg.episode.max_ticks = 1 if exhaustion == "max_ticks" else 64
    spec = spec_for({}, limits=cfg)
    spec.schedule = schedule
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "final":
            return forced_answer(renderer)
        return renderer.encode_completion("hmm" if exhaustion == "max_ticks" else "x" * 5000)

    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.FINAL]
    assert episode.outcome.final_answer == "5"
    expected = "budget" if exhaustion == "episode" else exhaustion
    assert spec.protocol.results[0].ended_by == expected
    assert episode.calls[-1].forced
    assert episode.calls[-1].prompt_len + len(episode.calls[-1].completion_ids) <= cfg.ctx.max_ctx
    assert sum(len(call.completion_ids) for call in episode.calls) <= cfg.episode.max_gen_tokens
    assert_token_buffers(episode, buffers, policy)


@pytest.mark.parametrize("stop", [True, False])
async def test_malformed_turn_stop_token_controls_continuation(stop: bool) -> None:
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.call_index == 0:
            return [
                S["call"],
                *renderer.encode_text('{"name": "submit"'),
                *([S["eot"]] if stop else []),
            ]
        return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))

    spec = spec_for({})
    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    first, second = episode.calls
    assert first.termination == (Termination.MALFORMED if stop else Termination.LENGTH)
    buf = buffers[first.segment_id]
    delta = buf[first.prompt_len + len(first.completion_ids) : second.prompt_len]
    assert delta[0] == (S["tool"] if stop else S["eot"])
    assert [S["eot"], S["eot"]] not in [buf[i : i + 2] for i in range(len(buf) - 1)]
    assert_token_buffers(episode, buffers, policy)


@pytest.mark.parametrize("error", [BudgetExceededError, ConfigError, MarliError, RuntimeError])
async def test_non_backend_errors_propagate(error: type[Exception]) -> None:
    spec = spec_for({})

    def script(ctx: ScriptCtx) -> Sequence[int]:
        raise error("must propagate")

    spec.policies["script"] = ScriptedPolicy("script", FakeRenderer(), script)
    with pytest.raises(error, match="must propagate"):
        await run_episode(spec)
    assert spec.env.teardown_count == 1


async def test_failed_control_tool_allows_later_calls() -> None:
    protocol = RuntimeProtocol(tools=("submit", "write_scratchpad"))
    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(
                        ("submit", {"wrong": "5"}),
                        ("write_scratchpad", {"content": "saved"}),
                    )
                ),
                submit(),
            ]
        },
        protocol,
    )
    episode, buffers = await run_episode(spec)
    first, second = episode.calls[0].tool_calls
    assert first.error is not None and second.error is None
    assert episode.workspace_log[0].content == "saved"
    assert episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, spec.policies["script"])


async def test_chat_message_log_tool_calls_pair_one_to_one() -> None:
    def reply(messages: Sequence[Msg], meta: CallMeta) -> ChatReply:
        calls = (
            (
                ParsedToolCall("write_scratchpad", {"content": "a"}, "raw", True),
                ParsedToolCall(None, None, "garbage", False),
                ParsedToolCall("unknown", {}, "raw", True, "native-id"),
            )
            if meta.call_index == 0
            else (ParsedToolCall("submit", {"answer": "5"}, "raw", True),)
        )
        return ChatReply("", None, calls, Termination.STOP, Usage(5, 3, tokenizer="api:test"))

    policy = ScriptedChatPolicy("chat", reply)
    spec = spec_for({}, RuntimeProtocol(tools=("submit", "write_scratchpad")))
    spec.policies, spec.seating, spec.renderers = {"chat": policy}, {"solver": "chat"}, {}
    episode, buffers = await run_episode(spec)
    messages = policy.calls[1].messages
    assert_tool_pairs(messages)
    (assistant,) = [msg for msg in messages if msg.role == "assistant"]
    tools = [msg for msg in messages if msg.role == "tool"]
    assert [call.id for call in assistant.tool_calls] == [msg.tool_call_id for msg in tools]
    assert len(tools) == 3 and len({msg.tool_call_id for msg in tools}) == 3
    assert all(msg.tool_call_id for msg in tools)
    assert assistant.tool_calls[-1].id == "native-id"
    assert tools[1].content == "error: could not parse tool call"
    assert tools[2].content == "error: unknown tool unknown"
    assert episode.outcome.final_answer == "5" and not buffers
    assert policy.calls[0].cache_salt != policy.calls[1].cache_salt


@pytest.mark.parametrize("rerender", [False, True])
async def test_summary_suppress_prefixes_and_same_ticket_act(rerender: bool) -> None:
    class SuppressingRenderer(FakeRenderer):
        supports_delta = not rerender

        def initial(
            self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
        ) -> list[int]:
            assert_tool_pairs(msgs)
            return super().initial(system, tools, msgs)

        def suppress_thinking_prefix(self) -> list[int]:
            return [S["think"], S["/think"]]

    renderer = SuppressingRenderer()
    protocol = RuntimeProtocol(
        tools=("submit", "write_scratchpad", "end_session"),
        context=ContextSpec(kind="both", compact_threshold=1500),
    )
    spec = spec_for(
        {}, protocol, renderer=renderer, limits=Limits(session=SessionLimits(max_sessions=2))
    )

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose in {"compact", "carry"}:
            return renderer.encode_completion(f"{ctx.meta.purpose} summary")
        if ctx.meta.call_index == 0:
            calls = (("write_scratchpad", {"content": "x" * 700}),)
        elif ctx.meta.call_index == 2:
            calls = (("end_session", {}),)
        else:
            calls = (("submit", {"answer": "5"}),)
        return renderer.encode_completion(tool_calls=calls)

    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [
        Purpose.ACT,
        Purpose.COMPACT,
        Purpose.ACT,
        Purpose.CARRY,
        Purpose.ACT,
    ]
    assert [call.tick for call in episode.calls] == [0, 1, 1, 2, 2]
    for call in (episode.calls[1], episode.calls[3]):
        assert (
            buffers[call.segment_id][call.prompt_len - 2 : call.prompt_len]
            == renderer.suppress_thinking_prefix()
        )
    assert episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, policy)


async def test_rerender_message_log_tool_calls_pair_one_to_one() -> None:
    class PairedRenderer(FakeRenderer):
        supports_delta = False

        def initial(
            self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
        ) -> list[int]:
            assert_tool_pairs(msgs)
            return super().initial(system, tools, msgs)

    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(("write_scratchpad", {"content": "saved"}), ("unknown", {})),
                    raw_tool_bodies=("garbage",),
                ),
                submit(),
            ]
        },
        RuntimeProtocol(tools=("submit", "write_scratchpad")),
        renderer=PairedRenderer(),
    )
    episode, buffers = await run_episode(spec)
    assert len(episode.calls[0].tool_calls) == 3 and episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, spec.policies["script"])


async def test_lockstep_byte_determinism_under_seeded_random_latency() -> None:
    async def once(latency: bool) -> tuple[dict[str, object], dict[str, list[int]]]:
        renderer = FakeRenderer()
        protocol = RuntimeProtocol(tools=("submit", "write_scratchpad", "read_scratchpad"), count=3)
        spec = spec_for({}, protocol)

        def script(ctx: ScriptCtx) -> Sequence[int]:
            index = ctx.meta.call_index
            if index == 3:
                calls = (("submit", {"answer": "5"}),)
            else:
                calls = (("write_scratchpad", {"content": f"{ctx.meta.agent_id} t{index}"}),)
                if index == 1:
                    peer = (int(ctx.meta.agent_id[-1]) + 1) % 3
                    calls += (("read_scratchpad", {"agent_id": f"solver{peer}"}),)
            return renderer.encode_completion(tool_calls=calls)

        policy = ScriptedPolicy(
            "script",
            renderer,
            script,
            latency_s=lambda ctx: random.Random(ctx.seed).random() / 100 if latency else 0,
        )
        policy.deterministic = True
        spec.policies["script"] = policy
        episode, buffers = await run_episode(spec)
        assert_token_buffers(episode, buffers, policy)
        assert [(c.tick, c.agent_id) for c in episode.calls] == [
            (tick, f"solver{i}") for tick in range(4) for i in range(3)
        ]
        for call in episode.calls:
            assert episode.events[call.seq].data["call_id"] == call.call_id
        for write in episode.workspace_log:
            event = episode.events[write.seq]
            assert event.kind == EventKind.TOOL_START and event.agent_id == write.writer
        data = record_to_dict(episode)
        for call in data["calls"]:
            del call["timing"]  # timing is intentionally outside replay equivalence
        return data, buffers

    first, second = await once(False), await once(True)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize("purpose", [Purpose.COMPACT, Purpose.CARRY])
async def test_harness_summary_on_fresh_segment(
    purpose: Purpose, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FreshRenderer(FakeRenderer):
        def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
            pytest.fail("a fresh segment must use initial(), including for reserve calculation")

        def suppress_thinking_prefix(self) -> list[int]:
            return [S["think"], S["/think"]]

    async def summary_only(runtime: AgentRuntime) -> None:
        ticket = await runtime.scheduler.turn(runtime.info.agent_id)
        runtime._carry = "[Tail of your previous session]\ntail content"
        runtime._initial("delivery")
        assert runtime.segment_id == ""  # recorder segments are lazy
        await runtime._harness_call(ticket, purpose, "Summarize.", [], [], forced=False)

    monkeypatch.setattr(AgentRuntime, "_loop", summary_only)
    spec = spec_for({"solver0": [Turn("summary")]}, renderer=FreshRenderer())
    episode, buffers = await run_episode(spec)
    (call,) = episode.calls
    assert call.purpose == purpose
    policy = spec.policies["script"]
    prompt = policy.calls[0].prompt_text
    assert (
        "⟨user⟩What is 2+3?\n\n[Tail of your previous session]\ntail content"
        "\n\ndelivery⟨eot⟩⟨user⟩Summarize.⟨eot⟩⟨asst⟩" in prompt
    )
    assert prompt.endswith("⟨asst⟩⟨think⟩⟨/think⟩")
    assert_token_buffers(episode, buffers, policy)


async def test_worker_spawned_without_act_budget_reports_on_fresh_segment() -> None:
    class LateWorkerProtocol(Protocol):
        name = "late_worker"
        results: list[AgentResult]

        def roles(self) -> list[RoleSpec]:
            return [
                RoleSpec("solver", ("submit",), "Solve with {n_agents}."),
                RoleSpec(
                    "worker",
                    ("return_report",),
                    "Workers: {n_agents}.",
                    count=None,
                    limits_key="worker",
                ),
            ]

        async def run(self, io: SystemIO) -> Outcome:
            parent = await io.start_agent(
                "solver", agent_id="solver0", seat_key=("solver", 0), first_message=io.task.prompt
            )
            first = await parent
            assert io.ledger.reserve_workers(parent.agent_id, 1) == 1
            worker = await io.start_agent(
                "worker",
                agent_id="worker0",
                seat_key=("worker", 0),
                first_message="Help.",
                parent=parent.agent_id,
            )
            self.results = [first, await worker]
            return Outcome(first.submission, {first.agent_id: first.submission}, "late_worker")

    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.role == "worker":
            assert ctx.meta.purpose == "report" and ctx.meta.call_index == 0
            assert "Workers: 0." in ctx.prompt_text
            return [*renderer.encode_text('{"report": "five"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))

    spec = spec_for({})
    protocol = LateWorkerProtocol()
    spec.protocol = protocol
    spec.seating["worker"] = "script"
    spec.limits.episode.max_ticks = 1
    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    worker = protocol.results[-1]
    assert worker.report == "five" and worker.ended_by == "max_ticks"
    assert episode.segments[-1].start_reason == SegmentStart.SPAWN
    assert "Help.⟨eot⟩⟨user⟩You have run out of budget" in policy.calls[-1].prompt_text
    assert_token_buffers(episode, buffers, policy)


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_wallclock_cancellation_preserves_completed_writes(schedule: str) -> None:
    clock = FakeClock()
    renderer = FakeRenderer()
    protocol = RuntimeProtocol(count=2, tools=("submit", "write_scratchpad"))
    spec = spec_for({}, protocol)
    spec.schedule, spec.clock = schedule, clock
    spec.limits.episode.max_wall_s = 5

    def script(ctx: ScriptCtx) -> Sequence[int]:
        return renderer.encode_completion(tool_calls=(("write_scratchpad", {"content": "saved"}),))

    policy = ScriptedPolicy(
        "script",
        renderer,
        script,
        clock=clock,
        latency_s=lambda ctx: 100 if ctx.meta.call_index else 0,
    )
    spec.policies["script"] = policy
    before = set(asyncio.all_tasks())
    task = asyncio.create_task(run_episode(spec))
    for _ in range(100):
        await asyncio.sleep(0)
        if len(policy.calls) == 4:
            break
    assert len(policy.calls) == 4
    clock.advance(5)
    episode, buffers = await asyncio.wait_for(task, timeout=2)
    assert not episode.ok and episode.errors == ("episode wall-clock limit",)
    assert sorted(write.writer for write in episode.workspace_log) == ["solver0", "solver1"]
    assert all(write.content == "saved" and write.version == 1 for write in episode.workspace_log)
    assert len([event for event in episode.events if event.kind == EventKind.COMMIT]) == 2
    assert spec.env.teardown_count == 1 and not clock._sleepers
    assert not [other for other in asyncio.all_tasks() - before if not other.done()]
    assert_token_buffers(episode, buffers, policy)


async def test_last_tick_write_and_submit_commits_both_agents() -> None:
    protocol = RuntimeProtocol(count=2, tools=("submit", "write_scratchpad"))
    turn = Turn(
        tool_calls=(("write_scratchpad", {"content": "final notes"}), ("submit", {"answer": "5"}))
    )
    spec = spec_for({"solver0": [turn], "solver1": [turn]}, protocol)
    episode, buffers = await run_episode(spec)
    assert [write.writer for write in episode.workspace_log] == ["solver0", "solver1"]
    assert all(
        write.content == "final notes" and write.tick == 0 for write in episode.workspace_log
    )
    assert episode.metrics["scratchpad_writes"] == 2
    assert episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, spec.policies["script"])


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_episode_share_after_peer_submits(schedule: str) -> None:
    renderer = FakeRenderer()
    cfg = Limits(
        episode=EpisodeLimits(max_gen_tokens=3000),
        agent=AgentLimits(max_gen_tokens=100000, final_reserve=10),
        call=CallLimits(max_tokens=100000, min_call_tokens=1),
    )
    cfg.ctx.max_ctx = 100000
    spec = spec_for({}, RuntimeProtocol(count=3), limits=cfg)
    spec.schedule = schedule

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.agent_id == "solver0" or ctx.meta.call_index >= 2:
            return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))
        return renderer.encode_completion("w" * 99)

    policy = ScriptedPolicy("script", renderer, script)
    spec.policies["script"] = policy
    episode, buffers = await run_episode(spec)
    if schedule == "lockstep":
        spent = sum(len(call.completion_ids) for call in episode.calls if call.tick == 0)
        expected = (3000 - spent) // 2 - 10
    else:
        expected = 1000 - 100 - 10
    for ctx in policy.calls:
        if ctx.meta.call_index == 0:
            assert ctx.spec.max_tokens == 990
        elif ctx.meta.call_index == 1:
            assert ctx.spec.max_tokens == expected
    assert episode.outcome.final_answer == "5"
    assert_token_buffers(episode, buffers, policy)


@pytest.mark.parametrize("stop", ["ticks", "budget"])
async def test_provisional_text_continues_then_survives_exhaustion_without_forced_final(
    stop: str,
) -> None:
    cfg = Limits(on_no_tool_call="final_text_continue")
    if stop == "ticks":
        cfg.episode.max_ticks = 2
    else:
        cfg.agent.max_calls = 2
    protocol = RuntimeProtocol(tools=())
    spec = spec_for({"solver0": [Turn("draft"), Turn("revised")]}, protocol, limits=cfg)
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "revised"
    assert len(episode.calls) == 2 and not any(call.forced for call in episode.calls)
    assert protocol.results[0].ended_by == ("max_ticks" if stop == "ticks" else "budget")


async def test_no_tool_continuation_forces_visible_final_when_act_budget_cannot_fit() -> None:
    class TextRenderer(FakeRenderer):
        def forced_tool_prefix(self, tool_name: str) -> list[int]:
            pytest.fail(f"No tools were advertised: {tool_name}")

    spec = spec_for(
        {"solver0": [Turn("5")]},
        RuntimeProtocol(tools=()),
        renderer=TextRenderer(),
        limits=Limits(
            on_no_tool_call="final_text_continue",
            agent=AgentLimits(max_gen_tokens=80, final_reserve=64),
            call=CallLimits(min_call_tokens=32),
        ),
    )
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5"
    (call,) = episode.calls
    assert call.purpose == Purpose.FINAL and call.forced and not call.tool_calls
    assert spec.protocol.results[0].ended_by == "budget"
    assert episode.metrics["total_gen"] <= spec.limits.agent.max_gen_tokens
    assert_token_buffers(episode, buffers, spec.policies["script"])


async def test_async_blocking_tool_releases_then_reacquires_shared_phase() -> None:
    from marli.interact.tools import Tool

    seen: list[str] = []

    class SharedAction:
        shared = True
        control = False
        spec = ToolSpec(
            "shared_action",
            "Shared action",
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            assert ctx.scheduler._tool_lock.locked()
            seen.append("shared")
            return ToolResult("ok")

    class BlockingAction:
        shared = False
        control = False
        blocking = True
        spec = ToolSpec(
            "blocking_action",
            "Wait for shared work",
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            async with ctx.scheduler.tool_phase(ctx.agent_id):
                seen.append("blocking")
            return ToolResult("ok")

    class SharedEnv(ScratchEnv):
        def tools(self, role: str) -> list[Tool]:
            return [SharedAction(), BlockingAction()]

    spec = spec_for(
        {
            "solver0": [
                Turn(
                    tool_calls=(
                        ("shared_action", {}),
                        ("blocking_action", {}),
                        ("shared_action", {}),
                        ("submit", {"answer": "5"}),
                    )
                )
            ]
        },
        RuntimeProtocol(tools=("shared_action", "blocking_action", "submit")),
    )
    spec.env = SharedEnv()
    spec.schedule = "async"
    episode, _ = await asyncio.wait_for(run_episode(spec), 2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert seen == ["shared", "blocking", "shared"]
