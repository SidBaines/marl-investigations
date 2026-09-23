"""Recording an episode, persisting it, and computing compute metrics.

``Recorder`` is shared by every agent of one episode; the rollout layer turns
it into an :class:`~marli.interact.types.Episode`. Token buffers are stored in
an ``.npz`` sidecar (one int32 array per segment, key = segment_id) that is
written and fsynced **before** the episode's JSONL row, so a row on disk
always has its tokens (crash-safety; see rundir.py).

Compute metrics (``compute_metrics(episode, buffers) -> dict[str, float]``,
all computed offline from records so every protocol is measured identically):

- ``gen_tokens`` (per call) = ``len(completion_ids)`` (includes stop token and
  thinking; excludes prefill / forced-close / injected tokens, which are never
  in completion_ids). ``total_gen`` = sum over all calls of all agents
  (workers, finalizers, compaction/carry/final calls included); also broken
  down as ``gen/<role>`` and ``gen_purpose/<purpose>``.
- ``prompt_tokens`` (per call) = ``prompt_len``; ``total_prompt`` = sum.
- ``uncached_ideal`` (per call) = ``prompt_len`` minus the longest common
  prefix between this call's prompt and the full token sequence
  (prompt + completion) of any earlier call **of the same policy_id and
  policy_version** in this episode — in practice compare against (a) the
  previous call in the same segment and (b) the first call of every other
  segment. ``total_uncached`` = sum. (Server-reported cache hits are logged
  separately as a diagnostic only.)
- ``calls`` = number of calls; ``calls/<purpose>``.
- Critical path: build a DAG over calls with edges (i) consecutive calls of
  the same agent, (ii) the spawning call -> each worker's first call,
  (iii) each worker's last call -> the coordinator's next call,
  (iv) under lockstep, every call at tick t -> every call at tick t+1 of the
  *same agent set that was active* (i.e. tick barrier edges),
  (v) every peer's last call -> the finalizer's first call.
  Async reads are not edges. ``cp_tokens`` = longest path weighted by
  gen_tokens; ``cp_calls`` = longest path with unit weights (≈ Kimi
  CriticalSteps).
- ``peak_ctx`` = max over calls of prompt_len + gen_tokens;
  ``peak_active_ctx`` = max over ticks (lockstep) or event times (async) of the
  sum of active agents' current buffer lengths (a KV-cache proxy).
- Protocol diagnostics: ``n_agents``, ``n_workers``, ``n_sessions``,
  ``tool_calls``, ``tool_errors``, ``malformed_calls``, ``forced_calls``,
  ``cross_reads`` (WorkspaceReads whose writer != reader), ``notify_reads`` /
  ``pull_reads`` / ``push_reads``, ``scratchpad_writes``.

Interface (M1-10)::

    class Recorder:
        def __init__(self, episode_id: str, *, clock) -> None
        def new_segment(self, info: SegmentInfo, initial_ids: list[int]) -> None
        def extend(self, segment_id: str, ids: list[int]) -> None          # observation or
        completion ids
        def buffer(self, segment_id: str) -> list[int]
        def add_call(self, call: Call) -> None
        def add_agent(self, info: AgentInfo) -> None
        def add_event(self, kind: EventKind, agent_id: str, tick: int | None, **data) -> int  #
        returns seq
        def add_write(self, w: Write) -> None
        def limit_hit(self, key: str, limit: str) -> None
        def calls_of(self, agent_id: str) -> list[Call]
    def compute_metrics(episode: Episode, buffers: Mapping[str, Sequence[int]]) -> dict[str, float]
    def write_episode(path_jsonl: Path, sidecar_dir: Path, episode: Episode, buffers) -> None
    def read_episodes(path_jsonl: Path, sidecar_dir: Path, *, with_tokens: bool) ->
    Iterator[tuple[Episode, dict]]
"""

from __future__ import annotations
