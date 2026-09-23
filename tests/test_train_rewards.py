"""Aux rewards count recorded actions without conflating tools, calls, or agents."""

from __future__ import annotations

from dataclasses import replace

import pytest

from marli.interact.types import (
    AgentInfo,
    Call,
    Episode,
    Outcome,
    Purpose,
    ReadVia,
    Termination,
    ToolCallRecord,
    WorkspaceRead,
)
from marli.render.fake import FakeRenderer
from marli.train.rewards import AUX_REWARDS


def call(
    agent_id: str = "a",
    *,
    tool_calls: tuple[tuple[str, dict[str, str]], ...] = (),
    malformed: tuple[str, ...] = (),
    purpose: Purpose = Purpose.ACT,
    reads: tuple[WorkspaceRead, ...] = (),
) -> Call:
    renderer = FakeRenderer()
    ids = tuple(renderer.encode_completion(tool_calls=tool_calls, raw_tool_bodies=malformed))
    turn = renderer.parse(ids)
    tools = tuple(
        ToolCallRecord(i, tool.name, tool.arguments, tool.raw, tool.ok, "")
        for i, tool in enumerate(turn.tool_calls)
    )
    return Call(
        call_id=f"ep/{agent_id}/c0",
        episode_id="ep",
        agent_id=agent_id,
        role="peer",
        policy_id="fake",
        policy_version=0,
        segment_id=f"{agent_id}/g0",
        prompt_len=1,
        completion_ids=ids,
        logprobs=(-0.2,) * len(ids),
        text=turn.content,
        termination=Termination(turn.termination),
        purpose=purpose,
        forced=False,
        tool_calls=tools,
        reads=reads,
        tick=0,
        seq=0,
        session_idx=0,
        seed=0,
    )


def episode(calls: tuple[Call, ...] = ()) -> Episode:
    return Episode(
        episode_id="ep",
        group_id="g",
        episode_idx=0,
        task_id="task",
        protocol="test",
        config_hash="hash",
        backend="fake",
        agents=(AgentInfo("a", "peer", "fake", None, (0,)),),
        segments=(),
        calls=calls,
        events=(),
        workspace_log=(),
        outcome=Outcome("system answer", {"a": "answer"}, "test"),
        grades={"_system": {"correct": 1}},
        limits_hit={},
        metrics={},
        replayable=True,
        ok=True,
    )


def test_builtin_registry_names() -> None:
    assert AUX_REWARDS.names() == [
        "cross_reads",
        "cross_reads_all",
        "format",
        "submitted",
        "workers_spawned",
    ]


def test_format_is_fraction_of_act_calls_not_fraction_of_tools() -> None:
    calls = (
        call(tool_calls=(("submit", {"answer": "1"}), ("write_notes", {"text": "notes"}))),
        call(tool_calls=(("read_notes", {}),), malformed=("{bad",)),
        call(),
        call("b", malformed=("{bad",)),
        call(malformed=("{bad",), purpose=Purpose.COMPACT),
        call(malformed=("{bad",), purpose=Purpose.CARRY),
        call(malformed=("{bad",), purpose=Purpose.FINAL),
        call(malformed=("{bad",), purpose=Purpose.REPORT),
    )
    assert AUX_REWARDS.get("format")(episode(calls), "a") == pytest.approx(2 / 3)
    assert AUX_REWARDS.get("format")(episode(calls), "b") == 0.0


def test_format_counts_parsing_even_if_tool_execution_failed() -> None:
    valid = call(tool_calls=(("submit", {"answer": "1"}),))
    failed = replace(valid, tool_calls=(replace(valid.tool_calls[0], error="execution failed"),))
    assert AUX_REWARDS.get("format")(episode((failed,)), "a") == 1.0


@pytest.mark.parametrize("calls", [(), (call("b"),), (call(purpose=Purpose.COMPACT),)])
def test_format_defaults_to_one_with_no_own_act_calls(calls: tuple[Call, ...]) -> None:
    assert AUX_REWARDS.get("format")(episode(calls), "a") == 1.0


@pytest.mark.parametrize(
    "submissions,expected",
    [
        ({"a": "answer"}, 1.0),
        ({"a": ""}, 1.0),
        ({"a": None}, 0.0),
        ({"b": "answer"}, 0.0),
        ({}, 0.0),
    ],
)
def test_submitted_uses_own_submission_not_team_answer_or_grade(
    submissions: dict[str, str | None],
    expected: float,
) -> None:
    ep = episode()
    ep = replace(
        ep, outcome=replace(ep.outcome, submissions=submissions), grades={"a": {"correct": 1}}
    )
    assert AUX_REWARDS.get("submitted")(ep, "a") == expected
    assert (
        AUX_REWARDS.get("submitted")(
            replace(ep, outcome=replace(ep.outcome, final_answer=None)), "a"
        )
        == expected
    )


@pytest.mark.parametrize(
    "name,counts", [("cross_reads", (2.0, 0.0)), ("cross_reads_all", (4.0, 1.0))]
)
def test_cross_reads_respects_delivery_mode(name: str, counts: tuple[float, float]) -> None:
    reads = tuple(
        WorkspaceRead(writer, "scratchpad", 1, via) for writer in ("a", "b") for via in ReadVia
    )
    calls = (
        call(reads=reads),
        call(reads=(WorkspaceRead("b", "notes", 1, ReadVia.PULL),), purpose=Purpose.COMPACT),
        call("b", reads=(WorkspaceRead("c", "scratchpad", 0, ReadVia.NOTIFY),)),
    )
    assert AUX_REWARDS.get(name)(episode(calls), "a") == counts[0]
    assert AUX_REWARDS.get(name)(episode(calls), "b") == counts[1]
    assert AUX_REWARDS.get(name)(episode(calls), "missing") == 0.0
    assert AUX_REWARDS.get(name)(episode(), "a") == 0.0


def test_workers_spawned_counts_direct_children_including_without_calls() -> None:
    ep = episode()
    ep = replace(
        ep,
        agents=ep.agents
        + (
            AgentInfo("w0", "worker", "api:fake", "a", (1,)),
            AgentInfo("w1", "worker", "fake", "a", (2,)),
            AgentInfo("grandchild", "worker", "fake", "w0", (3,)),
            AgentInfo("peer", "peer", "fake", None, (4,)),
        ),
    )
    assert AUX_REWARDS.get("workers_spawned")(ep, "a") == 2.0
    assert AUX_REWARDS.get("workers_spawned")(ep, "w0") == 1.0
    assert AUX_REWARDS.get("workers_spawned")(ep, "w1") == 0.0
    assert AUX_REWARDS.get("workers_spawned")(ep, "missing") == 0.0
