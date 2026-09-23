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
    assert "[worker coord0/w0: no report]" in spec.policies["script"].calls[-1].prompt_text


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
    assert not tool.shared and not tool.control
    expected_report = "[worker coord0/w0: no report]" if report is None else report
    assert result.content == f"[worker coord0/w0] {expected_report}"


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
    assert (
        coordinator.limits_key == "agent" and coordinator.system_prompt == "Coordinator {agent_id}"
    )
    assert worker.role == "worker" and worker.count is None and worker.limits_key == "worker"
    assert worker.system_prompt == "Worker {agent_id}"
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
