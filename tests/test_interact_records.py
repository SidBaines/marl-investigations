"""Small traces pin compute accounting and crash-safe episode persistence."""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from marli.errors import MarliError
from marli.interact.records import Recorder, compute_metrics, read_episodes, write_episode
from marli.interact.types import (
    AgentInfo,
    Call,
    Episode,
    Event,
    EventKind,
    Outcome,
    Purpose,
    ReadVia,
    SegmentInfo,
    SegmentStart,
    Termination,
    ToolCallRecord,
    Usage,
    WorkspaceRead,
    Write,
    record_from_dict,
    record_to_dict,
)
from marli.render.base import Msg
from marli.render.fake import FakeRenderer

R = FakeRenderer()
PROMPT = R.initial(None, [], [Msg("user", "q")])  # four tokens


class FakeClock:
    def now(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        pass


def call(agent: str, seq: int, n_gen: int, **changes: Any) -> Call:
    ids = tuple(R.encode_completion("x" * (n_gen - 1)))
    return replace(
        Call(
            call_id=f"episode/{agent}/c{seq}",
            episode_id="episode",
            agent_id=agent,
            role="peer",
            policy_id="learner",
            policy_version=0,
            segment_id=f"{agent}/g0",
            prompt_len=len(PROMPT),
            completion_ids=ids,
            logprobs=(-0.5,) * len(ids),
            text="x",
            termination=Termination.STOP,
            purpose=Purpose.ACT,
            forced=False,
            tool_calls=(),
            reads=(),
            tick=None,
            seq=seq,
            session_idx=0,
            seed=seq,
        ),
        **changes,
    )


def episode(calls: tuple[Call, ...] = (), **changes: Any) -> Episode:
    agents = {
        c.agent_id: AgentInfo(c.agent_id, c.role, c.policy_id, None, (i, c.agent_id))
        for i, c in enumerate(calls)
    }
    return replace(
        Episode(
            episode_id="episode",
            group_id="group",
            episode_idx=0,
            task_id="task",
            protocol="test",
            config_hash="hash",
            backend="scripted",
            agents=tuple(agents.values()),
            segments=(),
            calls=calls,
            events=(),
            workspace_log=(),
            outcome=Outcome("answer", {}, "single"),
            grades={"_system": {"correct": 1.0}},
            limits_hit={},
            metrics={},
            replayable=True,
            ok=True,
        ),
        **changes,
    )


def independent_buffers(calls: tuple[Call, ...]) -> dict[str, list[int]]:
    return {c.segment_id: PROMPT + list(c.completion_ids) for c in calls if c.segment_id}


def segment(key: str = "peer0/g0", **changes: Any) -> SegmentInfo:
    return replace(
        SegmentInfo(
            key,
            key.split("/")[0],
            0,
            SegmentStart.START,
            None,
            R.name,
            R.tokenizer_sha,
        ),
        **changes,
    )


def test_recorder_buffers_sequence_and_build_snapshot() -> None:
    clock = FakeClock()
    recorder = Recorder("episode", clock=clock)
    assert recorder.clock is clock
    agent = AgentInfo("peer0", "peer", "learner", None, (0, "peer0"))
    recorder.add_agent(agent)
    initial = list(PROMPT)
    recorder.new_segment(segment(), initial)
    initial.clear()
    assert recorder.buffer("peer0/g0") == PROMPT
    seq = recorder.add_event(EventKind.CALL_START, "peer0", 0, call_id="c0")
    c = call("peer0", seq, 3, tick=0)
    recorder.extend(c.segment_id, list(c.completion_ids))
    recorder.add_call(c)
    w = Write("peer0", "scratchpad", 1, "note", 0, 1)
    recorder.add_write(w)
    assert recorder.add_event(EventKind.CALL_END, "peer0", 0, call_id=c.call_id) == 1
    assert recorder.add_event(EventKind.DONE, "peer0", None) == 2
    recorder.limit_hit("peer0", "max_calls")
    recorder.limit_hit("_episode", "max_ticks")
    recorder.limit_hit("peer0", "max_calls")
    recorder.limit_hit("peer0", "max_gen_tokens")
    assert recorder.calls_of("peer0") == [c]
    assert recorder.calls_of("missing") == []
    recorder.calls_of("peer0").clear()

    fields = record_to_dict(episode())
    for key in (
        "episode_id",
        "agents",
        "segments",
        "calls",
        "events",
        "workspace_log",
        "limits_hit",
    ):
        del fields[key]
    fields["outcome"] = Outcome("answer", {"peer0": "answer"}, "single")
    fields["errors"] = ()
    built, buffers = recorder.build(**fields)
    assert built.episode_id == "episode"
    assert built.agents == (agent,)
    assert built.calls == (c,)
    assert built.workspace_log == (w,)
    assert [e.seq for e in built.events] == [0, 1, 2]
    assert built.events[0].data == {"call_id": "c0"}
    assert list(built.limits_hit) == ["peer0", "_episode"]
    assert built.limits_hit["peer0"] == ("max_calls", "max_gen_tokens")
    assert built.segments == (replace(segment(), n_tokens=7),)
    assert buffers == {"peer0/g0": PROMPT + list(c.completion_ids)}
    recorder.extend("peer0/g0", R.encode_text("later"))
    recorder.limit_hit("peer0", "another")
    assert len(buffers["peer0/g0"]) == built.segments[0].n_tokens == 7
    assert built.limits_hit["peer0"] == ("max_calls", "max_gen_tokens")
    buffers["peer0/g0"].clear()
    assert len(recorder.buffer("peer0/g0")) == 12


def test_recorder_preserves_segment_boundaries() -> None:
    recorder = Recorder("episode", clock=FakeClock())
    recorder.new_segment(segment(), PROMPT)
    second = segment("peer0/g1", start_reason=SegmentStart.COMPACTION, carry_from="peer0/g0")
    recorder.new_segment(second, R.initial(None, [], []))
    recorder.extend("peer0/g1", R.encode_completion("summary"))
    assert recorder.buffer("peer0/g0") == PROMPT
    with pytest.raises(ValueError, match="peer0/g0"):
        recorder.new_segment(segment(), [])
    assert recorder.buffer("peer0/g0") == PROMPT


def test_recorder_buffer_returns_the_live_list() -> None:
    recorder = Recorder("episode", clock=FakeClock())
    recorder.new_segment(segment(), PROMPT)
    live = recorder.buffer("peer0/g0")
    live.extend(R.encode_text("more"))
    assert recorder.buffer("peer0/g0") is live
    assert live == PROMPT + R.encode_text("more")


def test_recorder_build_rejects_a_different_episode_id() -> None:
    recorder = Recorder("expected", clock=FakeClock())
    with pytest.raises(ValueError, match="different.*expected"):
        recorder.build(episode_id="different")


@pytest.mark.parametrize("seat_key", [[1, "peer"], ((1, 2),), ([1],), ({"a": 1},), (object(),)])
def test_recorder_rejects_non_flat_seat_keys(seat_key: Any) -> None:
    recorder = Recorder("episode", clock=FakeClock())
    with pytest.raises(ValueError, match="peer0.*seat_key.*flat tuple"):
        recorder.add_agent(AgentInfo("peer0", "peer", "learner", None, seat_key))


@pytest.mark.parametrize("seat_key", [(), (0, "peer", 2.5, True, None)])
def test_recorder_events_and_seat_keys_round_trip_in_json(seat_key: tuple[Any, ...]) -> None:
    recorder = Recorder("episode", clock=FakeClock())
    recorder.add_agent(AgentInfo("peer0", "peer", "learner", None, seat_key))
    versions = {1: ("peer0", [2, 3])}
    recorder.add_event(EventKind.PUSH, "peer0", 0, versions=versions)
    versions[1][1].append(4)
    supplied = vars(episode()).copy()
    for key in ("agents", "segments", "calls", "events", "workspace_log", "limits_hit"):
        del supplied[key]
    built, _ = recorder.build(**supplied)
    assert built.events[0].data == {"versions": {"1": ["peer0", [2, 3]]}}
    assert record_from_dict(Episode, json.loads(json.dumps(record_to_dict(built)))) == built


def test_single_agent_three_calls_count_only_new_prompt_tokens() -> None:
    delta = R.continuation("stop", [Msg("tool", "ok")])  # seven tokens
    first = call("peer0", 0, 3)
    second = call("peer0", 1, 5, prompt_len=14, purpose=Purpose.COMPACT)
    third = call("peer0", 2, 2, prompt_len=26, purpose=Purpose.FINAL, forced=True)
    buffer = PROMPT + list(first.completion_ids) + delta + list(second.completion_ids)
    buffer += delta + list(third.completion_ids)
    metrics = compute_metrics(episode((third, first, second)), {first.segment_id: buffer})
    assert len(delta) == 7
    assert metrics["total_gen"] == metrics["cp_tokens"] == 10
    assert metrics["calls"] == metrics["cp_calls"] == 3
    assert metrics["total_prompt"] == 4 + 14 + 26
    assert metrics["total_uncached"] == 4 + 7 + 7
    assert metrics["peak_ctx"] == metrics["peak_active_ctx"] == 28
    assert metrics["gen/peer"] == 10
    assert metrics["gen_purpose/compact"] == 5
    assert metrics["calls/final"] == metrics["forced_calls"] == 1
    assert "calls/report" not in metrics
    assert "gen/worker" not in metrics
    assert all(type(value) is float for value in metrics.values())


@pytest.mark.parametrize("third_tick", [False, True])
def test_lockstep_barriers_follow_existing_ticks_and_changing_peers(third_tick: bool) -> None:
    calls = (
        call("p0", 0, 8, tick=2),
        call("p1", 1, 2, tick=2),
        call("p0", 2, 3, tick=5, segment_id="p0/g1"),
        call("p1", 3, 9, tick=5, segment_id="p1/g1"),
    )
    if third_tick:
        calls += (call("p1", 4, 4, tick=8, segment_id="p1/g2"),)
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["total_gen"] == 22 + (4 if third_tick else 0)
    assert metrics["cp_tokens"] == 8 + 9 + (4 if third_tick else 0)
    assert metrics["cp_calls"] == 2 + third_tick
    assert metrics["peak_ctx"] == 13
    assert metrics["peak_active_ctx"] == (4 + 3) + (4 + 9)


def test_coordinator_spawn_and_join_parallel_worker_chains() -> None:
    calls = (
        call("coord", 0, 2, role="coordinator"),
        call("w0", 1, 4, role="worker"),
        call("w1", 2, 7, role="worker"),
        call("w0", 3, 6, role="worker", segment_id="w0/g1"),
        call("w1", 4, 2, role="worker", segment_id="w1/g1", purpose=Purpose.REPORT),
        call("coord", 5, 3, role="coordinator", segment_id="coord/g1"),
    )
    ep = episode(calls)
    ep = replace(
        ep, agents=tuple(replace(a, parent="coord") if a.role == "worker" else a for a in ep.agents)
    )
    metrics = compute_metrics(ep, independent_buffers(calls))
    assert metrics["total_gen"] == 24
    assert metrics["cp_tokens"] == 2 + max(4 + 6, 7 + 2) + 3
    assert metrics["cp_calls"] == 4
    assert metrics["n_agents"] == 3
    assert metrics["n_workers"] == 2
    assert metrics["gen/worker"] == 19
    assert metrics["gen/coordinator"] == 5


def test_spawn_uses_nearest_parent_calls_at_each_boundary() -> None:
    calls = (
        call("coord", 0, 2, role="coordinator"),
        call("coord", 1, 3, role="coordinator", segment_id="coord/g1"),
        call("worker", 2, 7, role="worker"),
        call("coord", 3, 5, role="coordinator", segment_id="coord/g2"),
        call("coord", 4, 4, role="coordinator", segment_id="coord/g3"),
    )
    ep = episode(calls)
    ep = replace(
        ep, agents=tuple(replace(a, parent="coord") if a.role == "worker" else a for a in ep.agents)
    )
    metrics = compute_metrics(ep, independent_buffers(calls))
    assert metrics["cp_tokens"] == 2 + 3 + 7 + 5 + 4
    assert metrics["cp_calls"] == 5


def test_finalizer_depends_on_last_call_of_each_peer() -> None:
    calls = (
        call("p0", 0, 8),
        call("p1", 1, 2, reads=(WorkspaceRead("p0", "scratchpad", 1, ReadVia.PULL),)),
        call("p1", 2, 3, segment_id="p1/g1"),
        call("judge", 3, 4, role="finalizer", purpose=Purpose.FINAL),
    )
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["total_gen"] == 17
    assert metrics["cp_tokens"] == max(8, 2 + 3) + 4
    # Longest paths for tokens and call counts can be different.
    assert metrics["cp_calls"] == 3
    assert metrics["gen/finalizer"] == 4


@pytest.mark.parametrize("role", ["worker", "coordinator", "custom"])
def test_finalizer_depends_on_every_earlier_non_finalizer(role: str) -> None:
    calls = (call("agent", 0, 8, role=role), call("judge", 1, 4, role="finalizer"))
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["cp_tokens"] == 12
    assert metrics["cp_calls"] == 2


def test_finalizer_skips_agents_whose_last_call_is_later_and_other_finalizers() -> None:
    calls = (
        call("peer", 0, 2),
        call("first_judge", 1, 20, role="finalizer"),
        call("second_judge", 2, 30, role="finalizer"),
        call("peer", 3, 2, segment_id="peer/g1"),
    )
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["cp_tokens"] == 30
    assert metrics["cp_calls"] == 2


def test_backward_tick_edges_name_both_calls() -> None:
    calls = (call("later_tick", 0, 2, tick=1), call("earlier_tick", 1, 3, tick=0))
    with pytest.raises(MarliError) as error:
        compute_metrics(episode(calls), independent_buffers(calls))
    assert all(c.call_id in str(error.value) for c in calls)


def test_api_call_keeps_foreign_token_usage_separate() -> None:
    token_call = call("peer", 0, 3, usage=Usage(999, 999, 999))
    api = call(
        "api",
        1,
        1,
        segment_id="",
        prompt_len=0,
        completion_ids=(),
        logprobs=None,
        role="finalizer",
        purpose=Purpose.FINAL,
        policy_version=None,
        usage=Usage(prompt_tokens=100, completion_tokens=70, tokenizer="api:model"),
    )
    metrics = compute_metrics(episode((token_call, api)), independent_buffers((token_call,)))
    assert metrics["total_gen"] == metrics["cp_tokens"] == 3
    assert metrics["calls"] == metrics["cp_calls"] == 2
    assert metrics["total_prompt"] == metrics["total_uncached"] == 4
    assert "gen/finalizer" not in metrics
    assert "gen_purpose/final" not in metrics
    assert metrics["api_calls"] == 1
    assert metrics["api_gen_tokens"] == 70
    assert metrics["api_prompt_tokens"] == 100
    assert metrics["peak_ctx"] == metrics["peak_active_ctx"] == 7


@pytest.mark.parametrize("termination", [Termination.BUDGET, Termination.CTX])
@pytest.mark.parametrize("api", [False, True])
def test_refused_calls_contribute_no_compute(termination: Termination, api: bool) -> None:
    refused = call(
        "refused",
        0,
        100,
        termination=termination,
        purpose=Purpose.CARRY,
        segment_id="" if api else "refused/g0",
        prompt_len=1_000,
        usage=Usage(prompt_tokens=1_000, completion_tokens=100),
    )
    sampled = call("judge", 1, 3, role="finalizer")
    # Refused prompts need no token buffer and must not populate the ideal cache.
    metrics = compute_metrics(episode((refused, sampled)), independent_buffers((sampled,)))
    assert metrics["refused_calls"] == 1
    assert metrics["calls"] == metrics["cp_calls"] == 1
    assert metrics["calls/act"] == 1
    assert "calls/carry" not in metrics
    assert "gen/peer" not in metrics
    assert "gen_purpose/carry" not in metrics
    assert metrics["total_gen"] == metrics["cp_tokens"] == 3
    assert metrics["total_prompt"] == metrics["total_uncached"] == 4
    assert metrics["peak_ctx"] == metrics["peak_active_ctx"] == (7 if api else 1_000)
    assert metrics["api_calls"] == metrics["api_gen_tokens"] == metrics["api_prompt_tokens"] == 0


def test_api_calls_preserve_dependencies_between_token_calls() -> None:
    calls = (
        call("coord", 0, 3, role="coordinator"),
        call(
            "api",
            1,
            1,
            role="worker",
            segment_id="",
            completion_ids=(),
            logprobs=None,
            usage=Usage(500, 100),
        ),
        call("coord", 2, 4, role="coordinator", segment_id="coord/g1"),
    )
    ep = episode(calls)
    ep = replace(
        ep,
        agents=tuple(replace(a, parent="coord") if a.agent_id == "api" else a for a in ep.agents),
    )
    metrics = compute_metrics(ep, independent_buffers(calls))
    assert metrics["cp_tokens"] == metrics["total_gen"] == 7
    assert metrics["cp_calls"] == metrics["calls"] == 3
    assert metrics["api_gen_tokens"] == 100


def test_cache_reuses_all_earlier_full_sequences_but_not_future_tokens() -> None:
    first = call("p0", 0, 3)
    delta = R.continuation("stop", [Msg("tool", "ok")])
    second = call("p0", 1, 4, prompt_len=14)
    full = PROMPT + list(first.completion_ids) + delta + list(second.completion_ids)
    copied = call("p1", 2, 2, prompt_len=18)
    calls = (copied, second, first)  # storage order must not affect cache eligibility
    buffers = {"p0/g0": full, "p1/g0": full + list(copied.completion_ids)}
    metrics = compute_metrics(episode(calls), buffers)
    assert metrics["total_uncached"] == 4 + 7 + 0
    for changes in ({"policy_id": "other"}, {"policy_version": 1}, {"policy_version": None}):
        changed = (replace(copied, **changes), second, first)
        assert compute_metrics(episode(changed), buffers)["total_uncached"] == 4 + 7 + 18


def test_cache_uses_partial_common_prefixes() -> None:
    first = call("p0", 0, 3)
    second = call("p1", 1, 2)
    buffers = independent_buffers((first, second))
    buffers[second.segment_id] = R.initial(None, [], [Msg("user", "z")]) + list(
        second.completion_ids
    )
    assert compute_metrics(episode((first, second)), buffers)["total_uncached"] == 4 + 3


@pytest.mark.parametrize("shared", [0, 1, 2, 3, 7, 15, 31, 63, 64])
@pytest.mark.parametrize("mixed_sequences", [False, True])
def test_cache_prefix_length_is_exact(shared: int, mixed_sequences: bool) -> None:
    calls = (call("a", 0, 3, prompt_len=64), call("b", 1, 2, prompt_len=64))
    first = R.encode_text("a" * 64) + list(calls[0].completion_ids)
    second = R.encode_text("a" * shared + "b" * (64 - shared)) + list(calls[1].completion_ids)
    buffers: dict[str, Sequence[int]] = {
        "a/g0": first,
        "b/g0": tuple(second) if mixed_sequences else second,
    }
    metrics = compute_metrics(episode(calls), buffers)
    assert metrics["total_uncached"] == 128 - shared


def test_uncached_long_append_only_episode_is_fast() -> None:
    buffer = R.initial(None, [], [Msg("user", "q" * (32_000 - 3))])
    delta = R.continuation("stop", [Msg("tool", "ok")])
    calls = []
    for seq in range(150):
        c = call("peer", seq, 1, prompt_len=len(buffer))
        calls.append(c)
        buffer.extend(c.completion_ids)
        if seq < 149:
            buffer.extend(delta)
    ep = episode(tuple(calls))
    started = perf_counter()
    metrics = compute_metrics(ep, {"peer/g0": buffer})
    elapsed = perf_counter() - started
    assert metrics["total_uncached"] == 32_000 + 149 * len(delta)
    assert elapsed < 0.5


@pytest.mark.parametrize("lockstep", [True, False])
def test_blocked_coordinator_counts_toward_peak_context(lockstep: bool) -> None:
    calls = (
        call("coord", 0, 6, role="coordinator", tick=0 if lockstep else None),
        call("worker", 1, 20, role="worker", tick=1 if lockstep else None),
        call(
            "coord", 2, 1, role="coordinator", tick=2 if lockstep else None, segment_id="coord/g1"
        ),
    )
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["peak_active_ctx"] == (4 + 6) + (4 + 20)


def test_lockstep_and_async_peak_match_with_call_events() -> None:
    calls = (
        call("coord", 0, 6, role="coordinator", tick=0),
        call("worker", 2, 20, role="worker", tick=1),
        call("coord", 5, 1, role="coordinator", tick=2, segment_id="coord/g1"),
    )
    events = tuple(
        event
        for c in calls
        for event in (
            Event(c.seq, EventKind.CALL_START, c.agent_id, c.tick, {"call_id": c.call_id}),
            Event(c.seq + 1, EventKind.CALL_END, c.agent_id, c.tick, {"call_id": c.call_id}),
        )
    ) + (Event(4, EventKind.DONE, "worker", 1), Event(7, EventKind.DONE, "coord", 2))
    ep = episode(calls, events=events)
    asynchronous = replace(
        ep,
        calls=tuple(replace(c, tick=None) for c in calls),
        events=tuple(replace(e, tick=None) for e in events),
    )
    buffers = independent_buffers(calls)
    assert compute_metrics(ep, buffers)["peak_active_ctx"] == 34
    assert compute_metrics(asynchronous, buffers)["peak_active_ctx"] == 34


def test_async_without_done_releases_context_at_last_call_end() -> None:
    calls = (call("a", 0, 8), call("b", 3, 10))
    events = (
        Event(0, EventKind.CALL_START, "a", None, {"call_id": calls[0].call_id}),
        Event(2, EventKind.CALL_END, "a", None, {"call_id": calls[0].call_id}),
        Event(3, EventKind.CALL_START, "b", None, {"call_id": calls[1].call_id}),
        Event(4, EventKind.CALL_END, "b", None, {"call_id": calls[1].call_id}),
    )
    metrics = compute_metrics(episode(calls, events=events), independent_buffers(calls))
    assert metrics["peak_active_ctx"] == 14


@pytest.mark.parametrize("lockstep", [True, False])
@pytest.mark.parametrize("record_done", [True, False])
def test_active_context_ends_at_done_or_last_call(lockstep: bool, record_done: bool) -> None:
    calls = tuple(
        call(agent, seq, n_gen, tick=tick if lockstep else None)
        for agent, seq, n_gen, tick in (("a", 0, 8, 0), ("b", 2, 10, 1), ("c", 4, 20, 3))
    )
    events = (Event(3, EventKind.DONE, "a", 2 if lockstep else None),) if record_done else ()
    metrics = compute_metrics(episode(calls, events=events), independent_buffers(calls))
    assert metrics["peak_active_ctx"] == (26 if record_done else 24)


def test_lockstep_uses_max_within_tick_and_replaces_carried_context() -> None:
    calls = (
        call("a", 0, 8, tick=0),
        call("a", 1, 1, tick=0, segment_id="a/g1"),
        call("b", 2, 2, tick=0),
        call("a", 3, 1, tick=1, segment_id="a/g2"),
        call("c", 4, 5, tick=1),
        call("b", 5, 1, tick=2, segment_id="b/g1"),
    )
    metrics = compute_metrics(episode(calls), independent_buffers(calls))
    assert metrics["peak_active_ctx"] == 5 + 6 + 9


def test_async_peak_context_tracks_events_and_releases_done_agents() -> None:
    calls = (call("p0", 0, 8), call("p1", 1, 2), call("p2", 6, 10))
    events = (
        Event(0, EventKind.CALL_START, "p0", None, {"call_id": calls[0].call_id}),
        Event(1, EventKind.CALL_START, "p1", None, {"call_id": calls[1].call_id}),
        Event(2, EventKind.CALL_END, "p0", None, {"call_id": calls[0].call_id}),
        Event(3, EventKind.DONE, "p0", None),
        Event(4, EventKind.CALL_END, "p1", None, {"call_id": calls[1].call_id}),
        Event(5, EventKind.DONE, "p1", None),
        Event(6, EventKind.CALL_START, "p2", None, {"call_id": calls[2].call_id}),
        Event(7, EventKind.CALL_END, "p2", None, {"call_id": calls[2].call_id}),
    )
    metrics = compute_metrics(episode(calls, events=events[::-1]), independent_buffers(calls))
    assert metrics["peak_ctx"] == 14
    assert metrics["peak_active_ctx"] == (4 + 8) + 4


def test_protocol_diagnostics_and_sessions() -> None:
    tools = (
        ToolCallRecord(0, "write", {}, "", True, "ok", shared=True),
        ToolCallRecord(1, "read", {}, "", True, "failed", error="permission"),
        ToolCallRecord(2, None, None, "bad", False, "bad", error="unparsed"),
    )
    reads = (
        WorkspaceRead("p0", "scratchpad", 1, ReadVia.PULL),
        WorkspaceRead("p1", "scratchpad", 1, ReadVia.PUSH),
        WorkspaceRead("p1", "scratchpad", 1, ReadVia.NOTIFY),
    )
    calls = (
        call("p0", 0, 3, tool_calls=tools, reads=reads, termination=Termination.MALFORMED),
        call("p0", 1, 2, session_idx=1, segment_id="p0/g1", forced=True, purpose=Purpose.CARRY),
    )
    ep = episode(
        calls,
        workspace_log=(
            Write("p0", "scratchpad", 1, "note", None, 2),
            Write("p0", "notes", 1, "carry", None, 3),
            Write("p0", "scratchpad", 2, "update", None, 4),
        ),
        segments=(
            segment("p0/g0"),
            segment("p0/g1", session_idx=1),
            segment("p0/g2", session_idx=1),
        ),
    )
    metrics = compute_metrics(ep, independent_buffers(calls))
    assert {
        key: metrics[key]
        for key in (
            "n_sessions",
            "tool_calls",
            "tool_errors",
            "malformed_calls",
            "forced_calls",
            "cross_reads",
            "notify_reads",
            "pull_reads",
            "push_reads",
            "scratchpad_writes",
        )
    } == {
        "n_sessions": 2,
        "tool_calls": 3,
        "tool_errors": 2,
        "malformed_calls": 1,
        "forced_calls": 1,
        "cross_reads": 2,
        "notify_reads": 1,
        "pull_reads": 1,
        "push_reads": 1,
        "scratchpad_writes": 2,
    }


def test_empty_episode_has_all_fixed_metrics_zero() -> None:
    metrics = compute_metrics(episode(), {})
    assert set(metrics) == {
        "total_gen",
        "total_prompt",
        "total_uncached",
        "calls",
        "refused_calls",
        "api_calls",
        "api_gen_tokens",
        "api_prompt_tokens",
        "cp_tokens",
        "cp_calls",
        "peak_ctx",
        "peak_active_ctx",
        "n_agents",
        "n_workers",
        "n_sessions",
        "tool_calls",
        "tool_errors",
        "malformed_calls",
        "forced_calls",
        "cross_reads",
        "notify_reads",
        "pull_reads",
        "push_reads",
        "scratchpad_writes",
    }
    assert set(metrics.values()) == {0.0}


def row_writer(directory: Path) -> Callable[[str, Mapping[str, Any]], None]:
    def append_row(name: str, row: Mapping[str, Any]) -> None:
        with (directory / name).open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    return append_row


def test_persistence_round_trip_with_little_endian_tokens(tmp_path: Path) -> None:
    c = call("p0", 0, 3)
    buffers = independent_buffers((c,))
    buffers["empty/g0"] = []
    ep = episode((c,), segments=(segment("p0/g0", n_tokens=7),))
    rows: list[tuple[str, Mapping[str, Any]]] = []

    def append_row(name: str, row: Mapping[str, Any]) -> None:
        rows.append((name, row))
        row_writer(tmp_path)(name, row)

    write_episode(append_row, ep, buffers, record_tokens=True)
    assert [name for name, _ in rows] == ["tokens.jsonl", "episodes.jsonl"]
    raw = base64.b64decode(rows[0][1]["segments"]["p0/g0"])
    assert raw == struct.pack("<7i", *buffers["p0/g0"])
    assert list(read_episodes(tmp_path, with_tokens=True)) == [(ep, buffers)]
    assert list(read_episodes(tmp_path, with_tokens=False)) == [(ep, {})]


def test_tokens_join_by_episode_id_and_ignore_orphans(tmp_path: Path) -> None:
    first = episode(episode_id="first")
    second = episode(episode_id="second")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for ep, text in ((first, "a"), (second, "b"), (episode(episode_id="orphan"), "c")):
        write_episode(
            lambda name, row: rows.append((name, row)),
            ep,
            {"g0": R.encode_text(text)},
            record_tokens=True,
        )
    writer = row_writer(tmp_path)
    for name, row in rows[::-1]:
        if name == "tokens.jsonl":
            writer(name, row)
    for ep in (first, second):
        writer("episodes.jsonl", record_to_dict(ep))
    assert list(read_episodes(tmp_path, with_tokens=True)) == [
        (first, {"g0": R.encode_text("a")}),
        (second, {"g0": R.encode_text("b")}),
    ]


def test_read_episodes_sees_rows_appended_after_iteration_starts(tmp_path: Path) -> None:
    first = episode(episode_id="first")
    second = episode(episode_id="second")
    writer = row_writer(tmp_path)
    write_episode(writer, first, {"g0": R.encode_text("a")}, record_tokens=True)
    rows = read_episodes(tmp_path, with_tokens=True)
    assert next(rows) == (first, {"g0": R.encode_text("a")})
    write_episode(writer, second, {"g0": R.encode_text("b")}, record_tokens=True)
    assert next(rows) == (second, {"g0": R.encode_text("b")})
    assert list(rows) == []


def test_tokens_are_read_only_as_far_as_the_current_episode(tmp_path: Path) -> None:
    first = episode(episode_id="first")
    writer = row_writer(tmp_path)
    write_episode(writer, first, {}, record_tokens=True)
    with (tmp_path / "tokens.jsonl").open("a") as stream:
        stream.write("not JSON\n")
    write_episode(writer, episode(episode_id="second"), {}, record_tokens=True)
    rows = read_episodes(tmp_path, with_tokens=True)
    assert next(rows) == (first, {})
    with pytest.raises(MarliError, match="tokens.jsonl"):
        next(rows)


@pytest.mark.parametrize("actual", [None, 2, 4])
def test_read_validates_segment_presence_and_length(tmp_path: Path, actual: int | None) -> None:
    ep = episode(episode_id="broken-episode", segments=(segment("peer/g3", n_tokens=3),))
    buffers = {} if actual is None else {"peer/g3": R.encode_text("x" * actual)}
    write_episode(row_writer(tmp_path), ep, buffers, record_tokens=True)
    with pytest.raises(MarliError, match="broken-episode.*peer/g3"):
        list(read_episodes(tmp_path, with_tokens=True))
    assert list(read_episodes(tmp_path, with_tokens=False)) == [(ep, {})]


def test_read_accepts_empty_recorded_segment(tmp_path: Path) -> None:
    ep = episode(segments=(segment("peer/g0", n_tokens=0),))
    write_episode(row_writer(tmp_path), ep, {"peer/g0": []}, record_tokens=True)
    assert list(read_episodes(tmp_path, with_tokens=True)) == [(ep, {"peer/g0": []})]


@pytest.mark.parametrize("tail", [b'{"episode_id":', b'{"episode_id":"\xe2'])
def test_torn_final_lines_in_both_files(tmp_path: Path, tail: bytes) -> None:
    ep = episode()
    write_episode(row_writer(tmp_path), ep, {}, record_tokens=True)
    for name in ("episodes.jsonl", "tokens.jsonl"):
        with (tmp_path / name).open("ab") as stream:
            stream.write(tail)
    assert list(read_episodes(tmp_path, with_tokens=True)) == [(ep, {})]


@pytest.mark.parametrize("sidecar", [None, b"", b'{"episode_id":'])
def test_missing_tokens_names_the_episode(tmp_path: Path, sidecar: bytes | None) -> None:
    ep = episode(episode_id="missing-id")
    write_episode(row_writer(tmp_path), ep, {}, record_tokens=False)
    if sidecar is not None:
        (tmp_path / "tokens.jsonl").write_bytes(sidecar)
    with pytest.raises(MarliError, match="missing-id"):
        list(read_episodes(tmp_path, with_tokens=True))


def test_record_tokens_false_never_writes_or_reads_tokens(tmp_path: Path) -> None:
    ep = episode()
    write_episode(row_writer(tmp_path), ep, {}, record_tokens=False)
    assert not (tmp_path / "tokens.jsonl").exists()
    (tmp_path / "tokens.jsonl").write_text("not JSON\n")
    assert list(read_episodes(tmp_path, with_tokens=False)) == [(ep, {})]


@pytest.mark.parametrize("name", ["episodes.jsonl", "tokens.jsonl"])
def test_complete_corrupt_line_is_not_silently_skipped(tmp_path: Path, name: str) -> None:
    ep = episode()
    write_episode(row_writer(tmp_path), ep, {}, record_tokens=True)
    with (tmp_path / name).open("a") as stream:
        stream.write("not JSON\n")
    write_episode(row_writer(tmp_path), ep, {}, record_tokens=True)
    with pytest.raises(MarliError, match=name):
        list(read_episodes(tmp_path, with_tokens=True))


def test_sidecar_failure_does_not_write_episode() -> None:
    names: list[str] = []

    def fail(name: str, row: Mapping[str, Any]) -> None:
        names.append(name)
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_episode(fail, episode(), {}, record_tokens=True)
    assert names == ["tokens.jsonl"]
