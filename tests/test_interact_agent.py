"""Runtime traces test observations, actions, and boundaries through real integration."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from _marli_test_envs import ArithEnv, ScratchEnv

from marli.interact.limits import AgentLimits, CallLimits, Limits, SessionLimits
from marli.interact.run import EpisodeSpec, run_episode
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
)
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, turns_by_agent
from marli.render.base import Msg, ToolSpec
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
    prefix = renderer.suppress_thinking_prefix() + renderer.forced_tool_prefix("submit")
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
    spec = spec_for(
        {
            "solver0": [
                Turn(tool_calls=(("write_scratchpad", {"content": "x" * 600}),)),
                Turn("We found five."),
                submit(),
            ]
        },
        protocol,
    )
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.COMPACT, Purpose.ACT]
    assert [seg.start_reason for seg in episode.segments] == [
        SegmentStart.START,
        SegmentStart.COMPACTION,
    ]
    assert episode.segments[1].carry_from == episode.segments[0].segment_id
    compact_prompt = spec.policies["script"].calls[1].prompt_text
    assert "Write a self-contained summary" in compact_prompt
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
            raise RuntimeError("backend unavailable")
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
        return renderer.encode_completion(tool_calls=(("end_session", {}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, buffers = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, Purpose.ACT, Purpose.FINAL]
    assert episode.outcome.final_answer == "5"
    first, second = episode.calls[:2]
    assert (
        buffers[second.segment_id][second.prompt_len - 12 : second.prompt_len]
        == buffers[first.segment_id][-12:]
    )
    assert episode.limits_hit["solver0"] == ("session.max_sessions",)


async def test_factory_receives_tool_context_and_renderers_are_fresh(
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

        def __init__(self, ctx: ToolCtx) -> None:
            seen.append(ctx)

        async def __call__(self, ctx: ToolCtx) -> ToolResult:
            return ToolResult(ctx.agent_id)

    monkeypatch.setattr("marli.interact.agent.TOOLS", registry)
    monkeypatch.setattr("marli.interact.run.TOOLS", registry)
    made: list[FakeRenderer] = []

    def factory() -> FakeRenderer:
        renderer = FakeRenderer()
        made.append(renderer)
        return renderer

    protocol = RuntimeProtocol(tools=("submit", "context_tool"), count=2)
    spec = spec_for({"solver0": [submit()], "solver1": [submit()]}, protocol)
    spec.renderers["script"] = factory
    await run_episode(spec)
    assert [ctx.agent_id for ctx in seen] == ["solver0", "solver1"]
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
    assert worker_result.report == ("five" if report else "[worker worker0: no report]")
    assert worker_result.ended_by == ("report" if report else "budget")
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
