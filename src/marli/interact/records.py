"""Recording an episode, persisting it, and computing compute metrics.

``Recorder`` is shared by every agent of one episode; the rollout layer turns
it into an :class:`~marli.interact.types.Episode`. Token buffers are stored in
a ``tokens.jsonl`` sidecar next to ``episodes.jsonl`` — one row per episode,
``{"episode_id": ..., "segments": {segment_id: base64(int32 little-endian)}}``
(stdlib only; ~5.3 chars/token) — appended and fsynced **before** the
episode's own row, so a row on disk always has its tokens (crash-safety; see
rundir.py). Eval runs may skip tokens (``record_tokens=False``).

Compute metrics (``compute_metrics(episode, buffers) -> dict[str, float]``,
all computed offline from records so every protocol is measured identically):

- ``gen_tokens`` (per call) = ``len(completion_ids)`` (includes stop token and
  thinking; excludes prefill / forced-close / injected tokens, which are never
  in completion_ids). API calls use ``usage.completion_tokens`` instead.
  ``total_gen`` = sum over all calls of all agents
  (workers, finalizers, compaction/carry/final calls included); also broken
  down as ``gen/<role>`` and ``gen_purpose/<purpose>``.
- ``prompt_tokens`` (per call) = ``prompt_len``; ``total_prompt`` = sum.
  API calls have no token buffer and contribute zero prompt/uncached tokens.
- ``uncached_ideal`` (per call) = ``prompt_len`` minus the longest common
  prefix between this call's prompt and the full token sequence
  (prompt + completion) of any earlier call **of the same policy_id and
  policy_version** in this episode. ``total_uncached`` = sum.
  (Server-reported cache hits are logged separately as a diagnostic only.)
- ``calls`` = number of calls; ``calls/<purpose>``.
- Critical path: build a DAG over calls with edges (i) consecutive calls of
  the same agent, (ii) the spawning call -> each worker's first call,
  (iii) each worker's last call -> the coordinator's next call,
  (iv) under lockstep, every call at one tick -> every call at the next
  recorded tick (i.e. tick barrier edges for the agents active at each tick),
  (v) every peer's last call -> the finalizer's first call.
  Async reads are not edges. ``cp_tokens`` = longest path weighted by
  gen_tokens; ``cp_calls`` = longest path with unit weights (≈ Kimi
  CriticalSteps).
- ``peak_ctx`` = max over calls of prompt_len + gen_tokens;
  ``peak_active_ctx`` = max over ticks (lockstep) or event times (async) of the
  sum of active agents' current buffer lengths (a KV-cache proxy).
  Within a tick use each agent's largest context. Async lengths are observed
  at CALL_START/CALL_END and released at DONE; tokens within a call have no
  individual timestamps. The peak is at least the largest individual buffer.
  API calls have no buffer to contribute to ``peak_active_ctx``.
- Protocol diagnostics: ``n_agents``, ``n_workers``, ``n_sessions``,
  ``tool_calls``, ``tool_errors``, ``malformed_calls``, ``forced_calls``,
  ``cross_reads`` (WorkspaceReads whose writer != reader), ``notify_reads`` /
  ``pull_reads`` / ``push_reads``, ``scratchpad_writes``.
  Sessions are distinct (agent_id, session_idx) pairs in calls or segments.

Interface (M1-10)::

    class Recorder:
        def __init__(self, episode_id: str, *, clock) -> None
        def new_segment(self, info: SegmentInfo, initial_ids: list[int]) -> None
        def extend(self, segment_id: str, ids: list[int]) -> None
        def buffer(self, segment_id: str) -> list[int]
        def add_call(self, call: Call) -> None
        def add_agent(self, info: AgentInfo) -> None
        def add_event(self, kind: EventKind, agent_id: str, tick: int | None, **data) -> int
        def add_write(self, w: Write) -> None
        def limit_hit(self, key: str, limit: str) -> None
        def calls_of(self, agent_id: str) -> list[Call]
        def build(self, **episode_fields) -> tuple[Episode, dict[str, list[int]]]
    def compute_metrics(episode: Episode, buffers: Mapping[str, Sequence[int]]) -> dict[str, float]
    def write_episode(append_row: Callable[[str, Mapping[str, Any]], None],
                      episode: Episode, buffers: Mapping[str, Sequence[int]], *,
                      record_tokens: bool) -> None
    def read_episodes(dir: Path, *, with_tokens: bool
                      ) -> Iterator[tuple[Episode, dict[str, list[int]]]]
"""

from __future__ import annotations

import base64
import json
import sys
from array import array
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from marli.errors import MarliError
from marli.interact.types import (
    AgentInfo,
    Call,
    Episode,
    Event,
    EventKind,
    ReadVia,
    SegmentInfo,
    Termination,
    Write,
    record_from_dict,
    record_to_dict,
)

if TYPE_CHECKING:
    from marli.interact.scheduler import Clock


class Recorder:
    """Collect one episode's records without deciding its outcome or grades."""

    def __init__(self, episode_id: str, *, clock: Clock) -> None:
        self.episode_id = episode_id
        self.clock = clock
        self._segments: dict[str, SegmentInfo] = {}
        self._buffers: dict[str, list[int]] = {}
        self._calls: list[Call] = []
        self._agents: list[AgentInfo] = []
        self._events: list[Event] = []
        self._writes: list[Write] = []
        self._limits: dict[str, list[str]] = {}

    def new_segment(self, info: SegmentInfo, initial_ids: list[int]) -> None:
        if info.segment_id in self._segments:
            raise ValueError(f"Segment already recorded: {info.segment_id}")
        self._segments[info.segment_id] = info
        self._buffers[info.segment_id] = list(initial_ids)

    def extend(self, segment_id: str, ids: list[int]) -> None:
        self._buffers[segment_id].extend(ids)

    def buffer(self, segment_id: str) -> list[int]:
        return self._buffers[segment_id]

    def add_call(self, call: Call) -> None:
        self._calls.append(call)

    def add_agent(self, info: AgentInfo) -> None:
        self._agents.append(info)

    def add_event(self, kind: EventKind, agent_id: str, tick: int | None, **data: Any) -> int:
        seq = len(self._events)
        self._events.append(Event(seq, kind, agent_id, tick, data))
        return seq

    def add_write(self, w: Write) -> None:
        self._writes.append(w)

    def limit_hit(self, key: str, limit: str) -> None:
        limits = self._limits.setdefault(key, [])
        if limit not in limits:
            limits.append(limit)

    def calls_of(self, agent_id: str) -> list[Call]:
        return [call for call in self._calls if call.agent_id == agent_id]

    def build(self, **episode_fields: Any) -> tuple[Episode, dict[str, list[int]]]:
        """Snapshot records and buffers; the caller supplies rollout decisions."""
        episode_fields.setdefault("episode_id", self.episode_id)
        buffers = {key: list(ids) for key, ids in self._buffers.items()}
        episode = Episode(
            **episode_fields,
            agents=tuple(self._agents),
            segments=tuple(
                replace(info, n_tokens=len(buffers[key])) for key, info in self._segments.items()
            ),
            calls=tuple(self._calls),
            events=tuple(self._events),
            workspace_log=tuple(self._writes),
            limits_hit={key: tuple(limits) for key, limits in self._limits.items()},
        )
        return episode, buffers


def _gen_tokens(call: Call) -> int:
    return len(call.completion_ids) if call.segment_id else call.usage.completion_tokens


def _critical_path(episode: Episode, calls: Sequence[Call]) -> tuple[int, int]:
    predecessors: list[set[int]] = [set() for _ in calls]
    by_agent: dict[str, list[int]] = defaultdict(list)
    by_tick: dict[int, list[int]] = defaultdict(list)
    for index, call in enumerate(calls):
        by_agent[call.agent_id].append(index)
        if call.tick is not None:
            by_tick[call.tick].append(index)

    for chain in by_agent.values():
        for previous, current in pairwise(chain):
            predecessors[current].add(previous)

    for agent in episode.agents:
        chain = by_agent[agent.agent_id]
        if not chain:
            continue
        if agent.parent is not None:
            parent_chain = by_agent[agent.parent]
            before = [index for index in parent_chain if index < chain[0]]
            after = [index for index in parent_chain if index > chain[-1]]
            if before:
                predecessors[chain[0]].add(before[-1])
            if after:
                predecessors[after[0]].add(chain[-1])
        if agent.role == "finalizer":
            predecessors[chain[0]].update(
                by_agent[peer.agent_id][-1]
                for peer in episode.agents
                if peer.role == "peer" and by_agent[peer.agent_id]
            )

    ticks = sorted(by_tick)
    for previous, current in pairwise(ticks):
        for index in by_tick[current]:
            predecessors[index].update(by_tick[previous])

    token_paths: list[int] = []
    call_paths: list[int] = []
    for index, call in enumerate(calls):
        parents = predecessors[index]
        token_paths.append(_gen_tokens(call) + max((token_paths[p] for p in parents), default=0))
        call_paths.append(1 + max((call_paths[p] for p in parents), default=0))
    return max(token_paths, default=0), max(call_paths, default=0)


def _peak_active_ctx(calls: Sequence[Call], events: Sequence[Event]) -> int:
    by_tick: dict[int, dict[str, int]] = defaultdict(dict)
    by_id = {call.call_id: call for call in calls}
    peak = 0
    for call in calls:
        if not call.segment_id:
            continue
        length = call.prompt_len + len(call.completion_ids)
        peak = max(peak, length)
        if call.tick is not None:
            tick = by_tick[call.tick]
            tick[call.agent_id] = max(tick.get(call.agent_id, 0), length)
    peak = max(peak, max((sum(tick.values()) for tick in by_tick.values()), default=0))

    active: dict[str, int] = {}
    for event in sorted(events, key=lambda event: event.seq):
        if event.tick is not None:
            continue
        if event.kind == EventKind.DONE:
            active.pop(event.agent_id, None)
        elif event.kind in (EventKind.CALL_START, EventKind.CALL_END):
            call = by_id[event.data["call_id"]]
            if call.segment_id:
                active[call.agent_id] = call.prompt_len
                if event.kind == EventKind.CALL_END:
                    active[call.agent_id] += len(call.completion_ids)
        peak = max(peak, sum(active.values()))
    return peak


def compute_metrics(episode: Episode, buffers: Mapping[str, Sequence[int]]) -> dict[str, float]:
    """Measure saved token sequences and call dependencies, independent of backend."""
    calls = sorted(episode.calls, key=lambda call: call.seq)
    metrics: dict[str, float] = {
        "total_gen": 0.0,
        "total_prompt": 0.0,
        "total_uncached": 0.0,
        "calls": float(len(calls)),
        "peak_ctx": 0.0,
        "n_agents": float(len(episode.agents)),
        "n_workers": float(sum(agent.role == "worker" for agent in episode.agents)),
        "n_sessions": float(
            len(
                {(call.agent_id, call.session_idx) for call in calls}
                | {(segment.agent_id, segment.session_idx) for segment in episode.segments}
            )
        ),
        "tool_calls": 0.0,
        "tool_errors": 0.0,
        "malformed_calls": 0.0,
        "forced_calls": 0.0,
        "cross_reads": 0.0,
        "notify_reads": 0.0,
        "pull_reads": 0.0,
        "push_reads": 0.0,
        "scratchpad_writes": float(sum(w.key == "scratchpad" for w in episode.workspace_log)),
    }
    history: dict[tuple[str, int | None], list[Call]] = defaultdict(list)
    for call in calls:
        gen = _gen_tokens(call)
        metrics["total_gen"] += gen
        for key, value in (
            (f"gen/{call.role}", gen),
            (f"gen_purpose/{call.purpose.value}", gen),
            (f"calls/{call.purpose.value}", 1),
        ):
            metrics[key] = metrics.get(key, 0.0) + value
        prompt_len = call.prompt_len if call.segment_id else 0
        metrics["total_prompt"] += prompt_len
        metrics["peak_ctx"] = max(metrics["peak_ctx"], float(prompt_len + gen))
        if call.segment_id:
            prompt = buffers[call.segment_id][:prompt_len]
            earlier = history[call.policy_id, call.policy_version]
            longest = 0
            for previous in earlier:
                previous_buffer = buffers[previous.segment_id]
                end = min(prompt_len, previous.prompt_len + len(previous.completion_ids))
                common = 0
                while common < end and prompt[common] == previous_buffer[common]:
                    common += 1
                longest = max(longest, common)
            metrics["total_uncached"] += prompt_len - longest
            earlier.append(call)
        metrics["tool_calls"] += len(call.tool_calls)
        metrics["tool_errors"] += sum(tool.error is not None for tool in call.tool_calls)
        metrics["malformed_calls"] += call.termination == Termination.MALFORMED
        metrics["forced_calls"] += call.forced
        for read in call.reads:
            metrics["cross_reads"] += read.writer != call.agent_id
            if read.via == ReadVia.NOTIFY:
                metrics["notify_reads"] += 1
            elif read.via == ReadVia.PULL:
                metrics["pull_reads"] += 1
            elif read.via == ReadVia.PUSH:
                metrics["push_reads"] += 1
    cp_tokens, cp_calls = _critical_path(episode, calls)
    metrics["cp_tokens"] = float(cp_tokens)
    metrics["cp_calls"] = float(cp_calls)
    metrics["peak_active_ctx"] = float(_peak_active_ctx(calls, episode.events))
    return metrics


def write_episode(
    append_row: Callable[[str, Mapping[str, Any]], None],
    episode: Episode,
    buffers: Mapping[str, Sequence[int]],
    *,
    record_tokens: bool,
) -> None:
    """Append tokens before the episode; the supplied writer owns fsync."""
    if record_tokens:
        segments: dict[str, str] = {}
        for segment_id, ids in buffers.items():
            packed = array("i", ids)
            if sys.byteorder == "big":
                packed.byteswap()
            segments[segment_id] = base64.b64encode(packed.tobytes()).decode("ascii")
        append_row("tokens.jsonl", {"episode_id": episode.episode_id, "segments": segments})
    append_row("episodes.jsonl", record_to_dict(episode))


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if not line.endswith(b"\n"):
                    return  # An interrupted append may end mid-JSON or mid-UTF-8.
                raise MarliError(f"Invalid JSON in {path.name} at line {line_number}") from exc


def read_episodes(
    dir: Path, *, with_tokens: bool
) -> Iterator[tuple[Episode, dict[str, list[int]]]]:
    """Stream episodes, joining the optional sidecar by episode identity."""
    dir = Path(dir)
    tokens: dict[str, dict[str, str]] = {}
    if with_tokens and (dir / "tokens.jsonl").exists():
        tokens = {row["episode_id"]: row["segments"] for row in _read_rows(dir / "tokens.jsonl")}
    for row in _read_rows(dir / "episodes.jsonl"):
        episode = record_from_dict(Episode, row)
        buffers: dict[str, list[int]] = {}
        if with_tokens:
            if episode.episode_id not in tokens:
                raise MarliError(f"Missing tokens for episode {episode.episode_id}")
            for segment_id, encoded in tokens[episode.episode_id].items():
                packed = array("i")
                packed.frombytes(base64.b64decode(encoded, validate=True))
                if sys.byteorder == "big":
                    packed.byteswap()
                buffers[segment_id] = packed.tolist()
        yield episode, buffers
