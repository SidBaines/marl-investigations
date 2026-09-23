"""Delegation must preserve scheduler barriers, fresh contexts and compute accounting."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from _marli_test_envs import ArithEnv

from marli.errors import ConfigError
from marli.interact.configs import resolve_protocol
from marli.interact.limits import AgentLimits, Ledger, Limits, SpawnLimits
from marli.interact.protocols.coordinator import CoordinatorConfig, CoordinatorProtocol
from marli.interact.records import compute_metrics
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.scheduler import FakeClock, Scheduler
from marli.interact.system import AgentResult, get_protocol
from marli.interact.tools import TOOLS, ToolCtx, run_tool
from marli.interact.types import EventKind, Purpose, SegmentStart, record_to_dict
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import FakeRenderer, S


def tool_turn(name: str, **arguments: Any) -> Turn:
    return Turn(tool_calls=((name, arguments),))


def coordinator_spec(
    turns: dict[str, list[Turn]] | None = None,
    *,
    config: CoordinatorConfig | None = None,
    limits: Limits | None = None,
    schedule: str = "lockstep",
) -> EpisodeSpec:
    renderer = FakeRenderer()
    env = ArithEnv()
    if turns is None:
        turns = {
            "coord0": [
                replace(
                    tool_turn(
                        "spawn_workers",
                        tasks=[
                            {"task": "Compute the sum.", "context": "Check each addend."},
                            {"task": "Check the result independently."},
                        ],
                    ),
                    content="Private coordinator deliberation.",
                ),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [
                Turn(content="I will verify each addend before reporting. " * 3),
                tool_turn("return_report", report="The sum is five."),
            ],
            "coord0/w1": [tool_turn("return_report", report="Confirmed: 5.")],
        }
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, turns), clock=FakeClock())
    return EpisodeSpec(
        CoordinatorProtocol(config),
        env,
        env.task,
        {"coordinator": "script", "worker": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        limits or Limits(),
        schedule=schedule,
        clock=FakeClock(),
        config_hash="coordinator-test",
        backend="scripted",
        run_seed=17,
    )


async def test_parallel_workers_join_before_next_tick_with_exact_prompts_and_metrics() -> None:
    spec = coordinator_spec()
    episode, buffers = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.replayable and not episode.errors
    assert episode.outcome.final_answer == "5"
    assert episode.outcome.submissions == {"coord0": "5"}
    assert episode.outcome.aggregation == "coordinator"
    assert episode.grades == {"coord0": {"correct": 1.0}, "_system": {"correct": 1.0}}
    assert spec.env.setup_count == spec.env.teardown_count == 1

    calls = sorted(episode.calls, key=lambda call: call.seq)
    assert {
        agent.agent_id: [call.tick for call in calls if call.agent_id == agent.agent_id]
        for agent in episode.agents
    } == {"coord0": [0, 3], "coord0/w0": [1, 2], "coord0/w1": [1]}
    assert [(agent.agent_id, agent.parent, agent.seat_key) for agent in episode.agents] == [
        ("coord0", None, ("coord", 0)),
        ("coord0/w0", "coord0", ("coord", 0, "w", 0)),
        ("coord0/w1", "coord0", ("coord", 0, "w", 1)),
    ]
    spawns = [event for event in episode.events if event.kind == EventKind.SPAWN]
    assert [(event.agent_id, event.tick, event.data["child"]) for event in spawns] == [
        ("coord0", 0, "coord0/w0"),
        ("coord0", 0, "coord0/w1"),
    ]
    assert {segment.agent_id: segment.start_reason for segment in episode.segments} == {
        "coord0": SegmentStart.START,
        "coord0/w0": SegmentStart.SPAWN,
        "coord0/w1": SegmentStart.SPAWN,
    }
    spawn = calls[0].tool_calls[0]
    assert not spawn.shared and spawn.error is None
    assert spawn.result == "[worker coord0/w0] The sum is five.\n\n[worker coord0/w1] Confirmed: 5."

    prompts = spec.policies["script"].calls
    assert spec.env.task.prompt in prompts[0].prompt_text
    assert "fresh context" in prompts[0].prompt_text
    assert "self-contained subtask" in prompts[0].prompt_text
    assert spawn.result in prompts[-1].prompt_text
    worker_prompts = {
        prompt.meta.agent_id: prompt.prompt_text
        for prompt in prompts
        if prompt.meta.role == "worker" and prompt.meta.call_index == 0
    }
    for agent_id, prompt in worker_prompts.items():
        assert "You are helping solve: What is 2+3?" in prompt
        assert "End by calling return_report with your findings" in prompt
        assert "Private coordinator deliberation" not in prompt
        assert "spawn_workers" not in prompt
        assert f"You are worker {agent_id}" in prompt
    assert "Compute the sum." in worker_prompts["coord0/w0"]
    assert "Context: Check each addend." in worker_prompts["coord0/w0"]
    assert "Check the result independently." in worker_prompts["coord0/w1"]
    assert "Check each addend." not in worker_prompts["coord0/w1"]
    # lockstep records calls in (tick, seat) order; policies see them in arrival order
    by_agent: dict[str, list] = {}
    for prompt in prompts:
        by_agent.setdefault(prompt.meta.agent_id, []).append(prompt)
    paired = []
    for call in calls:
        paired.append((call, by_agent[call.agent_id].pop(0)))
    for call, prompt in paired:
        assert buffers[call.segment_id][: call.prompt_len] == list(prompt.prompt_ids)
        assert buffers[call.segment_id][
            call.prompt_len : call.prompt_len + len(call.completion_ids)
        ] == list(call.completion_ids)

    metrics = compute_metrics(episode, buffers)
    coordinator_tokens = sum(
        len(call.completion_ids) for call in calls if call.agent_id == "coord0"
    )
    worker_tokens = [
        sum(len(call.completion_ids) for call in calls if call.agent_id == agent)
        for agent in ("coord0/w0", "coord0/w1")
    ]
    assert metrics["cp_tokens"] == coordinator_tokens + max(worker_tokens)
    assert metrics["cp_calls"] == 4
    assert metrics["total_gen"] == coordinator_tokens + sum(worker_tokens)
    assert metrics["calls"] == 5 and metrics["n_workers"] == 2


@pytest.mark.parametrize("count,max_per_call,max_total", [(0, 2, 4), (3, 2, 4), (2, 3, 1)])
async def test_spawn_limits_reject_entire_batch(
    count: int, max_per_call: int, max_total: int
) -> None:
    limits = Limits(spawn=SpawnLimits(max_per_call=max_per_call, max_total=max_total))
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "Check"}] * count),
                tool_turn("submit", answer="5"),
            ]
        },
        limits=limits,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert len(episode.agents) == 1
    result = episode.calls[0].tool_calls[0]
    assert result.error is not None
    assert f"1 <= len(tasks) <= {max_per_call}" in result.result
    assert f"running total <= {max_total}" in result.result


@pytest.mark.parametrize("episode_tokens,affordable", [(512, 0), (2048, 1)])
async def test_unaffordable_batch_starts_no_workers(episode_tokens: int, affordable: int) -> None:
    limits = Limits(
        agent=AgentLimits(final_reserve=64),
        worker=AgentLimits(max_gen_tokens=1024, final_reserve=128),
    )
    limits.episode.max_gen_tokens = episode_tokens
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "Check"}] * 2),
                tool_turn("submit", answer="5"),
            ]
        },
        limits=limits,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert len(episode.agents) == 1
    result = episode.calls[0].tool_calls[0]
    assert result.error == f"only {affordable} workers are affordable; requested 2"


@pytest.mark.parametrize(
    "tasks",
    [[{}], ["task"], [{"task": 5}], [{"task": "x", "context": 5}], [{"task": "x", "extra": "y"}]],
)
async def test_invalid_subtask_is_a_tool_error_before_reservation(tasks: list[Any]) -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    system = SimpleNamespace(start_agent=AsyncMock())
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, None, system, ledger)
    result = await run_tool(TOOLS.get("spawn_workers")(), ctx, {"tasks": tasks})
    assert result.error is not None and "tasks[0]" in result.error
    ledger.reserve_workers.assert_not_called()
    system.start_agent.assert_not_awaited()


async def test_exhausted_worker_returns_no_report_and_coordinator_still_submits() -> None:
    limits = Limits(on_exhaust="none")
    limits.worker.max_calls = 1
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "Check"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [Turn(content="Not finished yet.")],
        },
        limits=limits,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert "worker.max_calls" in episode.limits_hit["coord0/w0"]
    assert "[worker coord0/w0: no report (budget)]" in spec.policies["script"].calls[-1].prompt_text


async def test_forced_worker_report_is_retained() -> None:
    limits = Limits()
    limits.worker.max_calls = 1
    turns = {
        "coord0": [
            tool_turn("spawn_workers", tasks=[{"task": "Check"}]),
            tool_turn("submit", answer="5"),
        ],
        "coord0/w0": [Turn(content="Still checking.")],
    }
    spec = coordinator_spec(turns, limits=limits)
    renderer = FakeRenderer()
    ordinary_turns = turns_by_agent(renderer, turns)

    def script(ctx: ScriptCtx) -> list[int]:
        if ctx.meta.purpose == "report":
            return [*renderer.encode_text('{"report": "5"}}'), S["/call"], S["eot"]]
        return list(ordinary_turns(ctx))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    (forced,) = [call for call in episode.calls if call.forced]
    assert forced.purpose == Purpose.REPORT
    assert "[worker coord0/w0] 5" in spec.policies["script"].calls[-1].prompt_text


@pytest.mark.parametrize("enabled", [False, True])
async def test_worker_scratchpads_are_optional_and_visible_only_to_coordinator(
    enabled: bool,
) -> None:
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "Check"}] * 2),
                Turn(
                    tool_calls=(
                        ("list_scratchpads", {}),
                        ("read_scratchpad", {"agent_id": "coord0/w0"}),
                    )
                ),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [
                tool_turn("write_scratchpad", content="First worker private calculation."),
                tool_turn("return_report", report="5"),
            ],
            "coord0/w1": [
                tool_turn("write_scratchpad", content="Second worker private calculation."),
                tool_turn("return_report", report="5"),
            ],
        },
        config=CoordinatorConfig(worker_scratchpads=enabled),
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert len(episode.workspace_log) == (2 if enabled else 0)
    coordinator_calls = sorted(
        (call for call in episode.calls if call.agent_id == "coord0"), key=lambda call: call.seq
    )
    listed, read = coordinator_calls[1].tool_calls
    if enabled:
        assert read.result == "First worker private calculation." and read.error is None
        assert "coord0/w0 v1" in listed.result and "coord0/w1 v1" in listed.result
        assert episode.metrics["cross_reads"] >= 2
        assert "First worker private calculation." in spec.policies["script"].calls[-1].prompt_text
    else:
        assert listed.error == "unknown tool list_scratchpads"
        assert read.error == "unknown tool read_scratchpad"
    for prompt in spec.policies["script"].calls:
        if prompt.meta.agent_id == "coord0/w0":
            assert "Second worker private calculation." not in prompt.prompt_text
        elif prompt.meta.agent_id == "coord0/w1":
            assert "First worker private calculation." not in prompt.prompt_text


async def test_two_spawn_rounds_have_stable_ids_release_budget_and_enforce_running_total() -> None:
    limits = Limits(spawn=SpawnLimits(max_per_call=2, max_total=2))
    limits.episode.max_gen_tokens = 2000
    limits.worker = AgentLimits(max_gen_tokens=1200, final_reserve=128)
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "First check"}]),
                tool_turn("spawn_workers", tasks=[{"task": "Second check"}]),
                tool_turn("spawn_workers", tasks=[{"task": "Over total"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [tool_turn("return_report", report="First report")],
            "coord0/w1": [tool_turn("return_report", report="Second report")],
        },
        limits=limits,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert [agent.agent_id for agent in episode.agents] == ["coord0", "coord0/w0", "coord0/w1"]
    assert [agent.seat_key for agent in episode.agents[1:]] == [
        ("coord", 0, "w", 0),
        ("coord", 0, "w", 1),
    ]
    calls = sorted(episode.calls, key=lambda call: call.seq)
    assert [(call.agent_id, call.tick) for call in calls] == [
        ("coord0", 0),
        ("coord0/w0", 1),
        ("coord0", 2),
        ("coord0/w1", 3),
        ("coord0", 4),
        ("coord0", 5),
    ]
    assert calls[-2].tool_calls[0].error is not None
    assert "already spawned 2" in calls[-2].tool_calls[0].result
    assert "First report" in spec.policies["script"].calls[2].prompt_text
    assert "Second report" in spec.policies["script"].calls[4].prompt_text
    assert "First report" not in spec.policies["script"].calls[3].prompt_text


async def test_lockstep_runs_are_byte_deterministic_and_reset_spawn_numbering() -> None:
    first = coordinator_spec()
    second = coordinator_spec()
    second.protocol = first.protocol
    episode1, buffers1 = await asyncio.wait_for(run_episode(first), timeout=2)
    episode2, buffers2 = await asyncio.wait_for(run_episode(second), timeout=2)
    assert episode1.ok and episode2.ok
    assert buffers1 == buffers2
    assert json.dumps(record_to_dict(episode1), sort_keys=True) == json.dumps(
        record_to_dict(episode2), sort_keys=True
    )


async def test_async_schedule_joins_reports_and_records_parent_dependencies() -> None:
    spec = coordinator_spec(schedule="async", config=CoordinatorConfig(worker_scratchpads=True))
    episode, buffers = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and not episode.replayable
    assert all(call.tick is None for call in episode.calls)
    assert episode.outcome.final_answer == "5"
    assert "[worker coord0/w0] The sum is five." in spec.policies["script"].calls[-1].prompt_text
    assert "[worker coord0/w1] Confirmed: 5." in spec.policies["script"].calls[-1].prompt_text
    metrics = compute_metrics(episode, buffers)
    chains = {
        agent.agent_id: sum(
            len(call.completion_ids) for call in episode.calls if call.agent_id == agent.agent_id
        )
        for agent in episode.agents
    }
    assert metrics["cp_tokens"] == chains["coord0"] + max(chains["coord0/w0"], chains["coord0/w1"])
    assert metrics["cp_calls"] == 4


@pytest.mark.parametrize("failure", [RuntimeError("wait failed"), asyncio.CancelledError()])
async def test_wait_always_unblocks_coordinator(failure: BaseException) -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    ledger.reserve_workers.return_value = 1
    scheduler = Mock(spec=Scheduler)
    system = SimpleNamespace(
        task=ArithEnv().task,
        start_agent=AsyncMock(return_value="worker-handle"),
        wait=AsyncMock(side_effect=failure),
    )
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, scheduler, system, ledger)
    with pytest.raises(type(failure)):
        await run_tool(TOOLS.get("spawn_workers")(), ctx, {"tasks": [{"task": "Check"}]})
    scheduler.block.assert_called_once_with("coord0")
    scheduler.unblock.assert_called_once_with("coord0")
    system.wait.assert_awaited_once_with(["worker-handle"])


@pytest.mark.parametrize(
    "report,ended_by",
    [(None, "error"), (None, "budget"), ("", "report"), ("forced findings", "budget")],
)
async def test_report_content_does_not_depend_on_termination_reason(
    report: str | None, ended_by: str
) -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    ledger.reserve_workers.return_value = 1
    system = SimpleNamespace(
        task=ArithEnv().task,
        start_agent=AsyncMock(return_value="worker-handle"),
        wait=AsyncMock(return_value=[AgentResult("coord0/w0", None, report, ended_by)]),
    )
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, Mock(), system, ledger)
    tool = TOOLS.get("spawn_workers")()
    result = await run_tool(tool, ctx, {"tasks": [{"task": "Check"}]})
    assert result.control == {} and result.error is None
    assert not tool.shared and tool.control and tool.blocking
    expected = (
        f"[worker coord0/w0: no report ({ended_by})]"
        if report is None
        else f"[worker coord0/w0] {report}"
    )
    assert result.content == expected


@pytest.mark.parametrize("scratchpads", [False, True])
def test_config_roles_and_registry(scratchpads: bool) -> None:
    config = CoordinatorConfig(
        env_tools=("python",),
        worker_scratchpads=scratchpads,
        coordinator_system_prompt="Coordinator {agent_id}",
        worker_system_prompt="Worker {agent_id}",
    )
    protocol = get_protocol("coordinator", config)
    assert protocol.config is config
    coordinator, worker = protocol.roles()
    assert coordinator.role == "coordinator" and coordinator.count == 1
    assert coordinator.limits_key == "agent" and coordinator.system_prompt.startswith(
        "Coordinator {agent_id}"
    )
    assert worker.role == "worker" and worker.count is None and worker.limits_key == "worker"
    assert worker.system_prompt.startswith("Worker {agent_id}")
    assert coordinator.tools == ("spawn_workers", "submit", "python") + (
        ("read_scratchpad", "list_scratchpads") if scratchpads else ()
    )
    assert worker.tools == ("return_report", "python") + (
        ("write_scratchpad",) if scratchpads else ()
    )
    assert coordinator.permissions.read_others == scratchpads
    assert coordinator.permissions.readable_roles == ("worker",)
    assert not worker.permissions.read_others
    assert worker.permissions.write_scratchpad == scratchpads


@pytest.mark.parametrize("field", ["worker_tools", "env_tools"])
def test_config_forbids_worker_spawning(field: str) -> None:
    with pytest.raises(ConfigError, match="spawn.max_depth=1"):
        CoordinatorConfig(**{field: ("spawn_workers",)})


@pytest.mark.parametrize(
    "name,scratchpads", [("coordinator_default", False), ("coordinator_scratch", True)]
)
def test_yaml_configs_compose(name: str, scratchpads: bool) -> None:
    protocol_name, config, _ = resolve_protocol(name)
    assert protocol_name == "coordinator" and config.worker_scratchpads == scratchpads
    coordinator, worker = CoordinatorProtocol(config).roles()
    assert ("write_scratchpad" in worker.tools) == scratchpads
    assert ("read_scratchpad" in coordinator.tools) == scratchpads
    with pytest.raises(ConfigError):
        resolve_protocol(name, {"unknown_option": True})


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
@pytest.mark.parametrize("shared_tool", ["write_scratchpad", "python"])
async def test_shared_tool_before_spawn_never_holds_workers_lock(
    schedule: str,
    shared_tool: str,
) -> None:
    from marli.interact.tools import Tool, ToolResult
    from marli.render.base import ToolSpec

    effects: list[str] = []

    class SharedPython:
        shared = True
        control = False
        spec = ToolSpec(
            "python",
            "Shared environment action",
            {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        )

        async def __call__(self, ctx: ToolCtx, *, content: str) -> ToolResult:
            effects.append(ctx.agent_id)
            await asyncio.sleep(0)
            return ToolResult(content)

    class SharedEnv(ArithEnv):
        def tools(self, role: str) -> list[Tool]:
            return [SharedPython()]

    config = CoordinatorConfig(
        coordinator_tools=("spawn_workers", "submit", "write_scratchpad"),
        env_tools=("python",),
        worker_scratchpads=True,
    )
    spec = coordinator_spec(
        {
            "coord0": [
                Turn(
                    tool_calls=(
                        (shared_tool, {"content": "coordinator"}),
                        ("spawn_workers", {"tasks": [{"task": "A"}, {"task": "B"}]}),
                    )
                ),
                tool_turn("submit", answer="5"),
            ],
            **{
                f"coord0/w{i}": [
                    tool_turn(shared_tool, content=f"worker {i}"),
                    tool_turn("return_report", report=f"report {i}"),
                ]
                for i in range(2)
            },
        },
        config=config,
        schedule=schedule,
    )
    spec.env = SharedEnv()
    episode, _ = await asyncio.wait_for(run_episode(spec), 2)
    assert episode.ok and episode.outcome.final_answer == "5"
    result = next(c for c in episode.calls if c.agent_id == "coord0").tool_calls[1]
    assert result.error is None and "report 0" in result.result and "report 1" in result.result
    if shared_tool == "python":
        assert effects == ["coord0", "coord0/w0", "coord0/w1"]
    else:
        # The denied coordinator write still takes the shared-tool phase.
        assert len(episode.workspace_log) == 2


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_rejected_spawn_allows_later_shared_tools_in_the_same_turn(schedule: str) -> None:
    spec = coordinator_spec(
        {
            "coord0": [
                Turn(
                    tool_calls=(
                        ("write_scratchpad", {"content": "before"}),
                        ("spawn_workers", {"tasks": [{"task": " "}]}),
                        ("write_scratchpad", {"content": "after"}),
                        ("spawn_workers", {"tasks": [{"task": "check"}]}),
                    )
                ),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [
                tool_turn("write_scratchpad", content="work"),
                tool_turn("return_report", report="findings"),
            ],
        },
        config=CoordinatorConfig(
            coordinator_tools=("spawn_workers", "submit", "write_scratchpad"),
            worker_scratchpads=True,
        ),
        schedule=schedule,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), 2)
    assert episode.ok and episode.outcome.final_answer == "5"
    calls = min(episode.calls, key=lambda call: call.seq).tool_calls
    assert "non-blank" in calls[1].error
    assert calls[2].error == "agent 'coord0' may not write scratchpad"
    assert calls[3].result == "[worker coord0/w0] findings"
    assert len(episode.agents) == 2


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_spawn_ends_turn_and_next_batch_has_a_critical_path_edge(schedule: str) -> None:
    spawn = ("spawn_workers", {"tasks": [{"task": "check"}]})
    spec = coordinator_spec(
        {
            "coord0": [
                Turn(tool_calls=(spawn, spawn, ("submit", {"answer": "wrong"}))),
                tool_turn("spawn_workers", tasks=[{"task": "second batch"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [tool_turn("return_report", report="first")],
            "coord0/w1": [tool_turn("return_report", report="second")],
        },
        schedule=schedule,
    )
    episode, _ = await asyncio.wait_for(run_episode(spec), 2)
    first = min(episode.calls, key=lambda c: c.seq)
    assert [tool.error for tool in first.tool_calls] == [
        None,
        "tool call after control tool",
        "tool call after control tool",
    ]
    assert episode.outcome.final_answer == "5" and len(episode.agents) == 3
    assert episode.metrics["cp_calls"] == 5
    assert episode.metrics["cp_tokens"] == episode.metrics["total_gen"]


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_rejected_batch_leaves_account_unchanged_and_smaller_spawn_succeeds(
    schedule: str,
) -> None:
    from dataclasses import asdict

    limits = Limits(worker=AgentLimits(max_gen_tokens=1024, final_reserve=128))
    limits.episode.max_gen_tokens = 2600
    ledger = Ledger(limits, schedule=schedule)
    ledger.register("coord0", kind="agent")
    before = asdict(ledger._agents["coord0"])
    system = SimpleNamespace(
        task=ArithEnv().task,
        start_agent=AsyncMock(return_value="handle"),
        wait=AsyncMock(return_value=[AgentResult("coord0/w0", None, "ok", "report")]),
    )
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, Mock(), system, ledger)
    tool = TOOLS.get("spawn_workers")()
    rejected = await run_tool(tool, ctx, {"tasks": [{"task": "A"}] * 3})
    assert rejected.error == "only 2 workers are affordable; requested 3"
    assert asdict(ledger._agents["coord0"]) == before
    system.start_agent.assert_not_awaited()
    accepted = await run_tool(tool, ctx, {"tasks": [{"task": "B"}]})
    assert accepted.error is None
    assert system.start_agent.await_args.kwargs["agent_id"] == "coord0/w0"

    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A"}] * 3),
                tool_turn("spawn_workers", tasks=[{"task": "B"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [tool_turn("return_report", report="ok")],
        },
        limits=limits,
        schedule=schedule,
    )
    episode, _ = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5" and len(episode.agents) == 2
    prompts = [ctx for ctx in spec.policies["script"].calls if ctx.meta.agent_id == "coord0"]
    assert prompts[1].spec.max_tokens == (
        limits.episode.max_gen_tokens
        - len(episode.calls[0].completion_ids)
        - limits.agent.final_reserve
    )


@pytest.mark.parametrize("task", ["", " ", "\n\t"])
async def test_blank_task_rejected_without_reservation(task: str) -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, None, None, ledger)
    result = await run_tool(TOOLS.get("spawn_workers")(), ctx, {"tasks": [{"task": task}]})
    assert "non-blank" in result.error
    ledger.reserve_workers.assert_not_called()


async def test_null_context_is_absent_and_budget_fields_reach_prompts() -> None:
    limits = Limits(spawn=SpawnLimits(max_per_call=2, max_total=3))
    limits.worker.max_gen_tokens = 7000
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A", "context": None}]),
                tool_turn("list_scratchpads"),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [
                tool_turn("write_scratchpad", content="work"),
                tool_turn("return_report", report="ok"),
            ],
        },
        config=resolve_protocol("coordinator_scratch")[1],
        limits=limits,
    )
    episode, _ = await run_episode(spec)
    prompts = spec.policies["script"].calls
    assert "2 workers per call" in prompts[0].prompt_text
    assert "3 workers total" in prompts[0].prompt_text
    assert "7000 generated tokens" in prompts[0].prompt_text
    worker = next(ctx.prompt_text for ctx in prompts if ctx.meta.role == "worker")
    assert "Context: None" not in worker
    assert (
        "visible to the coordinator" in worker
        and "visible to the coordinator" in prompts[0].prompt_text
    )
    listing = next(
        t.result for c in episode.calls for t in c.tool_calls if t.name == "list_scratchpads"
    )
    assert "coord0 v0" not in listing and "coord0/w0 v1" in listing


@pytest.mark.parametrize("field", ["coordinator_system_prompt", "worker_system_prompt"])
@pytest.mark.parametrize("template", ["{unknown}", "{}", "{agent_id.missing}", "{"])
def test_bad_coordinator_template_fails_at_config_time(field: str, template: str) -> None:
    with pytest.raises(ConfigError, match=field):
        CoordinatorConfig(**{field: template})


def test_worker_tools_require_return_report() -> None:
    with pytest.raises(ConfigError, match="return_report"):
        CoordinatorConfig(worker_tools=("python",))


async def test_worker_nudges_and_final_text_are_reports() -> None:
    turns = {
        "coord0": [
            tool_turn("spawn_workers", tasks=[{"task": "A"}]),
            tool_turn("submit", answer="5"),
        ],
        "coord0/w0": [Turn("findings"), tool_turn("return_report", report="findings")],
    }
    spec = coordinator_spec(turns)
    await run_episode(spec)
    worker_prompts = [
        ctx.prompt_text for ctx in spec.policies["script"].calls if ctx.meta.role == "worker"
    ]
    assert "use a tool or call return_report with your findings" in worker_prompts[-1]
    assert "submit" not in worker_prompts[-1]
    spec = coordinator_spec(turns, limits=Limits(on_no_tool_call="final_text_as_answer"))
    episode, _ = await run_episode(spec)
    assert episode.calls[0].tool_calls[0].result == "[worker coord0/w0] findings"
    assert "coord0/w0" not in episode.outcome.submissions and "coord0/w0" not in episode.grades
    assert (
        next(
            e.data["ended_by"]
            for e in episode.events
            if e.kind == EventKind.DONE and e.agent_id == "coord0/w0"
        )
        == "report"
    )


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_wall_timeout_records_reason_and_cleans_blocked_workers(schedule: str) -> None:
    clock = FakeClock()
    renderer = FakeRenderer()
    entered = asyncio.Event()

    def latency(ctx: ScriptCtx) -> float:
        if ctx.meta.role == "worker":
            entered.set()
            return 100
        return 0

    spec = coordinator_spec(
        {
            "coord0": [tool_turn("spawn_workers", tasks=[{"task": "A"}])],
            "coord0/w0": [tool_turn("return_report", report="late")],
        },
        schedule=schedule,
    )
    spec.clock = clock
    spec.limits.episode.max_wall_s = 10
    spec.policies["script"] = ScriptedPolicy(
        "script",
        renderer,
        spec.policies["script"]._script,
        clock=clock,
        latency_s=latency,
    )
    before = asyncio.all_tasks()
    task = asyncio.create_task(run_episode(spec))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        clock.advance(10)
        episode, _ = await asyncio.wait_for(task, 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not episode.ok and episode.errors == ("episode wall-clock limit",)
    assert {e.data["ended_by"] for e in episode.events if e.kind == EventKind.DONE} == {
        "max_wall_s"
    }
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_worker_critical_path_accounts_for_lockstep_barrier(schedule: str) -> None:
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A"}, {"task": "B"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [Turn("short"), tool_turn("return_report", report="r0")],
            "coord0/w1": [replace(tool_turn("return_report", report="r1"), content="long" * 100)],
        },
        schedule=schedule,
    )
    episode, _ = await run_episode(spec)
    chains = {
        a.agent_id: [len(c.completion_ids) for c in episode.calls if c.agent_id == a.agent_id]
        for a in episode.agents
    }
    a, b = chains["coord0/w0"], chains["coord0/w1"]
    expected = max(a[0], b[0]) + a[1] if schedule == "lockstep" else max(sum(a), sum(b))
    assert episode.metrics["cp_tokens"] == sum(chains["coord0"]) + expected


@pytest.mark.parametrize("tasks", [None, 5, {}, "do it", '[{"task": "A"}]', [None]])
async def test_invalid_tasks_schema_never_reserves_workers(tasks: Any) -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, None, None, ledger)
    result = await run_tool(TOOLS.get("spawn_workers")(), ctx, {"tasks": tasks})
    assert result.error
    ledger.reserve_workers.assert_not_called()


async def test_extra_spawn_argument_is_a_tool_error() -> None:
    ledger = Mock(spec=Ledger, limits=Limits())
    ctx = ToolCtx("coord0", "coordinator", 0, 0, None, None, None, None, ledger)
    result = await run_tool(
        TOOLS.get("spawn_workers")(),
        ctx,
        {
            "tasks": [{"task": "A"}],
            "wait": True,
        },
    )
    assert result.error == "unknown arguments: ['wait']"
    ledger.reserve_workers.assert_not_called()


async def test_worker_permissions_block_nested_spawn_and_other_workers_pads() -> None:
    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A"}, {"task": "B"}]),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [
                Turn(
                    tool_calls=(
                        ("spawn_workers", {"tasks": [{"task": "nested"}]}),
                        ("read_scratchpad", {"agent_id": "coord0/w1"}),
                        ("list_scratchpads", {}),
                    )
                ),
                tool_turn("return_report", report="first"),
            ],
            "coord0/w1": [
                tool_turn("write_scratchpad", content="worker-one-private"),
                tool_turn("return_report", report="second"),
            ],
        },
        config=CoordinatorConfig(
            worker_scratchpads=True,
            worker_tools=("return_report", "read_scratchpad", "list_scratchpads"),
        ),
    )
    episode, _ = await run_episode(spec)
    first = next(c for c in episode.calls if c.agent_id == "coord0/w0")
    assert first.tool_calls[0].error == "unknown tool spawn_workers"
    assert "may not read" in first.tool_calls[1].error
    assert first.tool_calls[2].result == ""
    assert len(episode.agents) == 3
    for ctx in spec.policies["script"].calls:
        if ctx.meta.agent_id == "coord0/w0":
            assert "worker-one-private" not in ctx.prompt_text


async def test_worker_backend_failure_and_malformed_forced_report_still_join() -> None:
    from marli.errors import BackendError

    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A"}, {"task": "B"}]),
                tool_turn("submit", answer="5"),
            ],
        }
    )
    renderer = FakeRenderer()
    ordinary = spec.policies["script"]._script

    def script(ctx: ScriptCtx) -> list[int]:
        if ctx.meta.agent_id == "coord0/w0":
            raise BackendError("backend exploded")
        if ctx.meta.agent_id == "coord0/w1":
            if ctx.meta.purpose == "report":
                return renderer.encode_text("garbage") + [S["eot"]]
            return renderer.encode_completion("still working")
        return list(ordinary(ctx))

    spec.limits.worker.max_calls = 1
    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert not episode.ok and episode.outcome.final_answer == "5"
    assert episode.calls[0].tool_calls[0].result == (
        "[worker coord0/w0: no report (error)]\n\n[worker coord0/w1: no report (budget)]"
    )
    assert set(episode.grades) == {"coord0", "_system"}


async def test_coordinator_ignores_caller_session_count_and_keeps_worker_report() -> None:
    from marli.interact.limits import SessionLimits

    spec = coordinator_spec(
        {
            "coord0": [
                tool_turn("spawn_workers", tasks=[{"task": "A"}]),
                Turn("y" * 400),
                tool_turn("submit", answer="5"),
            ],
            "coord0/w0": [tool_turn("return_report", report="REPORT-FROM-W0")],
        },
        limits=Limits(
            session=SessionLimits(
                max_sessions=2, max_gen_tokens=300, carry_reserve=50, carry_max_tokens=50
            )
        ),
    )
    with pytest.warns(UserWarning, match="sessions are owned"):
        episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5"
    assert len(episode.segments) == 2 and all(s.session_idx == 0 for s in episode.segments)
    assert "REPORT-FROM-W0" in spec.policies["script"].calls[-1].prompt_text
