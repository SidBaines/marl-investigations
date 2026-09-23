"""Small traces pin compute accounting and crash-safe episode persistence."""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
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


def test_api_call_counts_usage_without_a_buffer_or_prompt_tokens() -> None:
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
        usage=Usage(prompt_tokens=100, completion_tokens=7, tokenizer="api:model"),
    )
    metrics = compute_metrics(episode((token_call, api)), independent_buffers((token_call,)))
    assert metrics["total_gen"] == metrics["cp_tokens"] == 10
    assert metrics["calls"] == metrics["cp_calls"] == 2
    assert metrics["total_prompt"] == metrics["total_uncached"] == 4
    assert metrics["gen/finalizer"] == 7
    assert metrics["peak_ctx"] == metrics["peak_active_ctx"] == 7


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
