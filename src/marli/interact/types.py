"""Records of one multi-agent episode — the contract between rollout, eval and RL.

Every LLM call an agent makes becomes one :class:`Call`. Token-level policies
append into a per-agent, append-only token buffer (a *segment*); a context
reset (compaction, new session, worker spawn, renderer fallback) starts a new
segment. Training builds datums from exactly these buffers — the ids the
policy actually saw and sampled — never from re-rendered text (see CLAUDE.md
"Token-level RL invariants"). Eval computes compute metrics (total and
critical-path tokens, calls, peak context) offline from the same records.

Layout of a segment buffer for one agent::

    [ initial prompt ........ | completion_1 | delta_1 ........ | completion_2 | ... ]
      ^ Call_1.prompt_len = len(initial)       ^ Call_2.prompt_len = len(initial)+len(c1)+len(d1)

``Call.prompt_len`` indexes into the buffer: the call's prompt is
``buffer[:prompt_len]`` and its sampled completion immediately follows it.
Every non-completion position (initial prompt, deltas carrying tool results /
pushed messages / forced closes / prefilled openers) is an *observation* and
never carries loss.

All records are frozen dataclasses of JSON-able values (enums serialize by
value, tuples as lists); ``record_to_dict``/``record_from_dict`` round-trip
them. ``Timing`` fields compare as equal regardless of value, so seeded
lockstep episodes with scripted policies are byte-identical modulo timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Termination(StrEnum):
    """Why a call's completion ended."""

    STOP = "stop"  # sampled a stop token (end of turn / tool-call close)
    LENGTH = "length"  # hit the per-call max_tokens
    MALFORMED = "malformed"  # renderer could not parse the completion (e.g. unclosed tool call)
    # the harness refused to sample: an agent/session/episode limit was exhausted
    BUDGET = "budget"
    CTX = "ctx"  # the harness refused to sample: the prompt leaves no room in max_ctx
    ERROR = "error"  # the policy backend raised after retries


class Purpose(StrEnum):
    """What the harness asked for with this call."""

    ACT = "act"  # a normal agent turn
    COMPACT = "compact"  # in-session compaction summary (context manager)
    CARRY = "carry"  # end-of-session carry (notes / summary for the next session)
    FINAL = "final"  # forced final answer on exhaustion (on_exhaust=force_final)
    REPORT = "report"  # forced worker report when a worker exhausts its budget


class SegmentStart(StrEnum):
    """Why a new token buffer (segment) was started for an agent."""

    START = "start"  # the agent's first context
    COMPACTION = "compaction"  # in-session compaction replaced the context
    SESSION = "session"  # a new session of a multi-session agent
    SPAWN = "spawn"  # a worker spawned by a coordinator (first segment of a worker)
    RERENDER = "rerender"  # renderer without delta support: one segment per call


class ReadVia(StrEnum):
    """How workspace content reached an agent."""

    PULL = "pull"  # the agent called a read tool
    PUSH = "push"  # the harness appended new writes to the agent's context
    NOTIFY = "notify"  # the harness pushed an index entry (writer, version, size, first line)


@dataclass(frozen=True)
class Usage:
    """Token and money accounting for one call (or a sum of calls)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Server-reported prefix-cache hits: a diagnostic only. The comparable
    # "uncached" metric is computed offline from buffers (eval/compute.py).
    cached_prompt_tokens: int | None = None
    cost_usd: float = 0.0
    # Tokenizer family the counts are in ("qwen3_5", "o200k_harmony", "api:<model>", ...).
    # Token counts are never compared across tokenizers.
    tokenizer: str | None = None


@dataclass(frozen=True)
class Timing:
    """Wall-clock facts about a call. Excluded from equality (see module docstring)."""

    started_at: float = field(default=0.0, compare=False)
    latency_s: float = field(default=0.0, compare=False)


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call parsed from a completion, and what the agent got back."""

    index: int  # position within the turn (tools run in this order)
    name: str | None  # None when the call could not be parsed
    arguments: dict[str, Any] | None  # parsed JSON arguments; None if unparseable
    raw: str  # the call's raw text as sampled (for unparsed calls / audit)
    parsed_ok: bool
    result: str  # tool result text exactly as delivered (after truncation)
    error: str | None = None  # tool-level error message, if the call failed
    shared: bool = False  # touched shared state (ordering-relevant under lockstep)


@dataclass(frozen=True)
class WorkspaceRead:
    """A piece of shared state that became visible to an agent during a call."""

    writer: str  # agent_id that owns the scratchpad / notes
    key: str  # "scratchpad" | "notes" | an artifact name
    version: int  # version that was read (0 = empty/never written)
    via: ReadVia


@dataclass(frozen=True)
class Call:
    """One LLM call. The atomic unit of training data and compute accounting."""

    call_id: str  # f"{episode_id}/{agent_id}/c{n}" (deterministic)
    episode_id: str
    agent_id: str
    role: str
    policy_id: str  # learner name or frozen policy ref (resolved)
    policy_version: int | None  # sampler/weights version for learners; None for frozen/API
    segment_id: str  # token buffer this call lives in ("" for API/chat policies)
    prompt_len: int  # prompt = segment buffer[:prompt_len]; 0 for API/chat policies
    # exactly the sampled ids (incl. the stop token); () for API policies
    completion_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None  # sampler logprobs aligned with completion_ids; None for API
    text: str  # decoded completion for humans/eval — never used for training
    termination: Termination
    purpose: Purpose
    forced: bool  # the harness prefilled/forced this call (force_final, forced carry)
    tool_calls: tuple[ToolCallRecord, ...]
    reads: tuple[WorkspaceRead, ...]  # workspace state visible in THIS call's prompt delta
    tick: int | None  # global lockstep tick; None under async scheduling
    seq: int  # global per-episode event order (monotone)
    session_idx: int  # 0 for single-session agents
    seed: int  # per-call sampling seed (derived; see marli.seeds)
    usage: Usage = Usage()
    timing: Timing = Timing()


@dataclass(frozen=True)
class SegmentInfo:
    """Metadata of one token buffer. Buffers live in the tokens.jsonl sidecar (records.py)."""

    segment_id: str  # f"{agent_id}/g{n}" (deterministic)
    agent_id: str
    session_idx: int
    start_reason: SegmentStart
    carry_from: str | None  # segment_id whose carry (summary/notes/tail) seeded this one
    renderer: str | None  # renderer name (None for API policies)
    tokenizer_sha: str | None  # identity of the tokenizer that produced the ids
    n_tokens: int = 0


@dataclass(frozen=True)
class AgentInfo:
    """One agent instance in an episode (workers are created dynamically)."""

    agent_id: str  # deterministic: "peer0", "coord0", "coord0/w3", "finalizer0", ...
    role: str
    policy_id: str
    parent: str | None  # spawning agent for workers; None otherwise
    seat_key: tuple[Any, ...]  # total order used for deterministic lockstep commits


@dataclass(frozen=True)
class Write:
    """One versioned write to the shared workspace."""

    writer: str
    key: str  # "scratchpad" | "notes" | artifact name
    version: int  # 1-based, per (writer, key)
    content: str
    tick: int | None
    seq: int


class EventKind(StrEnum):
    CALL_START = "call_start"
    CALL_END = "call_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    COMMIT = "commit"
    PUSH = "push"
    SPAWN = "spawn"
    DONE = "done"


@dataclass(frozen=True)
class Event:
    """Global event log entry (always recorded; essential for async episodes)."""

    seq: int
    kind: EventKind
    agent_id: str
    tick: int | None
    # e.g. {"call_id":...} or {"versions": {...}}
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """How the system answered."""

    final_answer: str | None
    submissions: dict[str, str | None]  # agent_id -> that agent's own submission (None = none)
    aggregation: str  # "single" | "vote" | "finalizer" | "coordinator" | ...
    votes: dict[str, int] | None = None  # canonical answer -> count, for vote aggregation


@dataclass(frozen=True)
class Episode:
    """Everything that happened in one episode of one protocol on one task."""

    episode_id: str  # f"{group_id}/e{idx}"
    group_id: str  # f"{task_id}/{config_hash[:8]}" — the G episodes compared by GRPO
    episode_idx: int
    task_id: str
    protocol: str  # protocol config name
    config_hash: str
    backend: str  # resolved sampling backend description (never secrets)
    agents: tuple[AgentInfo, ...]
    segments: tuple[SegmentInfo, ...]
    calls: tuple[Call, ...]
    events: tuple[Event, ...]
    workspace_log: tuple[Write, ...]
    outcome: Outcome
    # Raw grades from env.grade: agent_id -> component -> value. The system-level
    # grade (of outcome.final_answer) is stored under the reserved agent id "_system".
    # Rewards are computed from grades by the training loop (train/credit.py).
    grades: dict[str, dict[str, float]]
    limits_hit: dict[str, tuple[str, ...]]  # agent_id (or "_episode") -> limit names that fired
    metrics: dict[str, float]  # compute + protocol diagnostics (eval/compute.py)
    replayable: bool  # True for lockstep with deterministic policies
    ok: bool
    errors: tuple[str, ...] = ()


SYSTEM_GRADE_KEY = "_system"
EPISODE_LIMIT_KEY = "_episode"


def record_to_dict(obj: Any) -> Any:
    """JSON-able form of any record above (recursive; enums by value; tuples as lists)."""
    raise NotImplementedError  # M1-10


def record_from_dict(cls: type, data: Any) -> Any:
    """Inverse of :func:`record_to_dict` for ``cls`` (rebuilds nested records, enums, tuples)."""
    raise NotImplementedError  # M1-10
