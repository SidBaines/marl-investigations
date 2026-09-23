"""Token/turn limits for agent systems, and what happens when they run out.

Distinct from ``marli.budget`` (the ``max_usd`` spend guard): these are
*protocol* limits that define the compute budget of an episode and are part of
the config hash. Every limit that fires is recorded in ``Episode.limits_hit``.

Per-call allocation (``Ledger.allocate``)::

    max_tokens = min(call.max_tokens,
                     agent_remaining - agent.final_reserve,
                     session_remaining - session.carry_reserve   (multi-session agents),
                     episode_share,                              (lockstep: episode_remaining //
                     n_active at tick start;
                                                                  async: fixed per-agent share
                                                                  assigned at registration)
                     ctx.max_ctx - prompt_len)

If the result is < ``min_call_tokens`` the call is not made and the limit that
bound it is reported as *exhausted*. Then ``on_exhaust`` decides:

- ``force_final`` (default): one last call with ``final_reserve`` tokens,
  purpose FINAL, ``forced=True``. The runtime appends an instruction message
  plus the renderer's ``forced_tool_prefix("submit")`` (both observation tokens)
  so the model only completes the answer arguments. Workers use
  ``return_report`` instead (purpose REPORT); a worker that still fails
  returns ``[worker <id>: no report]`` to its coordinator.
- ``none``: the agent simply stops with no submission.

Spawning k workers reserves ``k * worker.max_gen_tokens`` from the spawning
agent's episode budget up front; if that does not fit, ``spawn_workers``
returns an error result stating how many workers are affordable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from marli.config import doc_field
from marli.errors import ConfigError
from marli.interact.types import Purpose

# Reserves may be 0 (no forced/carry call budget); the context manager and
# on_exhaust validation decide when a reserve must be positive.
_NON_NEGATIVE = frozenset({"agent.final_reserve", "worker.final_reserve", "session.carry_reserve"})


@dataclass
class CallLimits:
    max_tokens: int = doc_field(4096, help="cap on tokens sampled by a single call")
    min_call_tokens: int = doc_field(
        16, help="below this allocation a call is not made (limit exhausted)"
    )


@dataclass
class AgentLimits:
    max_gen_tokens: int = doc_field(32768, help="total generated tokens for one agent instance")
    max_calls: int = doc_field(64, help="max LLM calls for one agent instance")
    final_reserve: int = doc_field(512, help="tokens held back for the forced final/report call")


@dataclass
class SessionLimits:
    max_sessions: int = doc_field(1, help="sessions per multi-session agent (1 = single-session)")
    max_gen_tokens: int = doc_field(
        16384, help="generated tokens per session (incl. compaction calls)"
    )
    carry_reserve: int = doc_field(
        2048, help="tokens held back for the end-of-session carry call (must cover the summary)"
    )
    carry_max_tokens: int = doc_field(1024, help="cap on the carried summary/notes length")


@dataclass
class EpisodeLimits:
    max_gen_tokens: int = doc_field(
        131072, help="generated tokens for the whole episode (all agents)"
    )
    max_ticks: int = doc_field(64, help="lockstep ticks before the episode is forced to finish")
    max_wall_s: float = doc_field(
        1800.0, help="wall-clock cap (straggler guard; logged, not used for matching)"
    )


@dataclass
class SpawnLimits:
    max_per_call: int = doc_field(4, help="workers one spawn_workers call may start")
    max_total: int = doc_field(8, help="workers per coordinator per episode")
    max_depth: int = doc_field(1, help="spawn depth (v1: workers cannot spawn)")


@dataclass
class ContextLimits:
    max_ctx: int = doc_field(
        32768, help="max prompt+completion tokens in one segment; <= model and backend max_seq_len"
    )


@dataclass
class Limits:
    call: CallLimits = field(default_factory=CallLimits)
    agent: AgentLimits = field(default_factory=AgentLimits)
    worker: AgentLimits = field(
        default_factory=lambda: AgentLimits(max_gen_tokens=8192, max_calls=24)
    )
    session: SessionLimits = field(default_factory=SessionLimits)
    episode: EpisodeLimits = field(default_factory=EpisodeLimits)
    spawn: SpawnLimits = field(default_factory=SpawnLimits)
    ctx: ContextLimits = field(default_factory=ContextLimits)
    on_exhaust: str = doc_field("force_final", help="force_final | none")
    on_no_tool_call: str = doc_field(
        "nudge",
        help="what to do when a turn makes no tool call: nudge | end_agent | final_text_as_answer",
    )
    max_nudges: int = doc_field(2, help="consecutive nudges before the agent is ended")
    tool_output_chars: int = doc_field(
        8000, help="tool results are truncated (head+tail) to this many chars"
    )

    def __post_init__(self) -> None:
        if self.on_exhaust not in {"force_final", "none"}:
            raise ConfigError("on_exhaust must be force_final or none")
        if self.on_no_tool_call not in {"nudge", "end_agent", "final_text_as_answer"}:
            raise ConfigError("on_no_tool_call must be nudge, end_agent or final_text_as_answer")
        for block in ("call", "agent", "worker", "session", "episode", "spawn", "ctx"):
            for item in fields(getattr(self, block)):
                value = getattr(getattr(self, block), item.name)
                name = f"{block}.{item.name}"
                if name == "episode.max_wall_s":
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ConfigError(f"{name} must be finite and positive")
                elif name in _NON_NEGATIVE:
                    if type(value) is not int or value < 0:
                        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")
                elif type(value) is not int or value <= 0:
                    raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        for name in ("max_nudges", "tool_output_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        for name in ("agent", "worker"):
            block = getattr(self, name)
            if self.on_exhaust == "force_final" and block.final_reserve <= 0:
                raise ConfigError(f"{name}.final_reserve must be > 0 when on_exhaust=force_final")
            if block.final_reserve >= block.max_gen_tokens:
                raise ConfigError(f"{name}.final_reserve must be < {name}.max_gen_tokens")
        if self.session.carry_reserve >= self.session.max_gen_tokens:
            raise ConfigError("session.carry_reserve must be < session.max_gen_tokens")
        if self.spawn.max_depth != 1:
            raise ConfigError("spawn.max_depth must equal 1")


@dataclass(frozen=True)
class Allocation:
    """Result of asking the ledger for a call's budget."""

    max_tokens: int  # > 0 when the call may proceed
    # name of the binding limit when the call may not proceed, e.g. "agent.max_gen_tokens"
    exhausted: str | None = None


class Ledger:
    """Tracks remaining budgets for one episode (all agents, sessions, spawns).

    Methods (M1-8):
      register(agent_id, *, kind: "agent" | "worker", parent: str | None) -> None
      start_session(agent_id) -> None
      allocate(agent_id, *, prompt_len: int, purpose: Purpose, tick: int | None, n_active: int) ->
      Allocation
      charge(agent_id, *, gen_tokens: int, purpose: Purpose) -> None   # after each call
      reserve_workers(parent_id, k: int) -> int                        # returns how many fit (0..k)
      release_worker(worker_id) -> None                                # return unused reservation
      to the parent
      limits_hit() -> dict[str, tuple[str, ...]]
    """

    def __init__(
        self, limits: Limits, *, schedule: str = "lockstep", expected_agents: int = 1
    ) -> None:
        if schedule not in {"lockstep", "async"}:
            raise ConfigError("schedule must be lockstep or async")
        if type(expected_agents) is not int or expected_agents <= 0:
            raise ConfigError("expected_agents must be a positive integer")
        self.limits = limits
        self.schedule = schedule
        self.expected_agents = expected_agents
        self._agents: dict[str, _Account] = {}
        self._spent = 0
        self._hits: dict[str, list[str]] = {}
        self._tick: int | None = None
        self._tick_share = 0
        self._tick_tokens: dict[str, int] = {}
        self._tick_reserved: dict[str, int] = {}

    @property
    def n_active(self) -> int:
        return sum(not state.done for state in self._agents.values())

    def register(self, agent_id: str, *, kind: str, parent: str | None = None) -> None:
        if agent_id in self._agents:
            raise ConfigError(f"agent_id already registered: {agent_id}")
        if kind not in {"agent", "worker"}:
            raise ConfigError("kind must be agent or worker")
        share = self.limits.episode.max_gen_tokens // self.expected_agents
        if parent is not None:
            owner = self._agents[parent]
            if kind != "worker" or owner.pending_workers < 1:
                raise ConfigError("worker must have a parent reservation")
            owner.pending_workers -= 1
            share = self.limits.worker.max_gen_tokens
        self._agents[agent_id] = _Account(kind, parent, share)

    def start_session(self, agent_id: str) -> None:
        state = self._agents[agent_id]
        state.session_tokens = 0

    def done(self, agent_id: str) -> None:
        self._agents[agent_id].done = True

    def limit_hit(self, agent_id: str, limit: str) -> None:
        key = "_episode" if limit.startswith("episode.") else agent_id
        hits = self._hits.setdefault(key, [])
        if limit not in hits:
            hits.append(limit)

    def allocate(
        self,
        agent_id: str,
        *,
        prompt_len: int,
        purpose: Purpose,
        tick: int | None,
        n_active: int,
    ) -> Allocation:
        state = self._agents[agent_id]
        block = getattr(self.limits, state.kind)
        final = purpose in (Purpose.FINAL, Purpose.REPORT)
        carry = purpose == Purpose.CARRY
        candidates: list[tuple[str, int]] = []
        if not final and not carry:
            if state.calls >= block.max_calls:
                candidates.append((f"{state.kind}.max_calls", 0))
            if tick is not None and tick >= self.limits.episode.max_ticks:
                candidates.append(("episode.max_ticks", 0))
        candidates.extend(
            [
                ("call.max_tokens", self.limits.call.max_tokens),
                (
                    f"{state.kind}.max_gen_tokens",
                    block.max_gen_tokens - state.tokens - (0 if final else block.final_reserve),
                ),
            ]
        )
        if final:
            candidates.append((f"{state.kind}.final_reserve", block.final_reserve))
        if self.limits.session.max_sessions > 1:
            candidates.append(
                (
                    "session.max_gen_tokens",
                    self.limits.session.max_gen_tokens
                    - state.session_tokens
                    - (0 if final or carry else self.limits.session.carry_reserve),
                )
            )
        if carry:
            candidates.append(("session.carry_reserve", self.limits.session.carry_reserve))
            candidates.append(("session.carry_max_tokens", self.limits.session.carry_max_tokens))
        remaining = self.limits.episode.max_gen_tokens - self._spent
        reserved = sum(account.reserved for account in self._agents.values())
        if self.schedule == "lockstep":
            # All allocations in a tick use the budget before any of its calls finish.
            if tick is None or self._tick != tick:
                self._tick = tick
                self._tick_share = max(0, remaining - reserved) // max(n_active, 1)
                self._tick_tokens = {key: account.tokens for key, account in self._agents.items()}
                self._tick_reserved = {
                    key: account.reserved for key, account in self._agents.items()
                }
            share = self._tick_share
            if state.parent is not None:
                share = state.share - state.tokens
            else:
                share = min(share, max(0, remaining - reserved))
        else:
            share = state.share - state.tokens - state.reserved
        candidates.extend(
            [
                ("episode.max_gen_tokens", min(remaining, share)),
                ("ctx.max_ctx", self.limits.ctx.max_ctx - prompt_len),
            ]
        )
        name, available = min(candidates, key=lambda item: item[1])
        if available < self.limits.call.min_call_tokens:
            self.limit_hit(agent_id, name)
            return Allocation(0, name)
        return Allocation(available)

    def charge(self, agent_id: str, *, gen_tokens: int, purpose: Purpose) -> None:
        state = self._agents[agent_id]
        state.tokens += gen_tokens
        state.session_tokens += gen_tokens
        state.calls += 1
        self._spent += gen_tokens
        if state.parent is not None:
            owner = self._agents[state.parent]
            owner.reserved -= gen_tokens
            owner.share -= gen_tokens

    def reserve_workers(self, parent_id: str, k: int) -> int:
        state = self._agents[parent_id]
        if state.kind == "worker":
            self.limit_hit(parent_id, "spawn.max_depth")
            return 0
        if k <= 0:
            return 0
        remaining = self.limits.episode.max_gen_tokens - self._spent
        remaining -= sum(account.reserved for account in self._agents.values())
        if self.schedule == "async":
            remaining = min(remaining, state.share - state.tokens - state.reserved)
        elif self._tick is not None:
            # Other seats may still spend their allocations from this tick. A spawn
            # can reserve only its parent's unspent share, never those in-flight tokens.
            spent_this_tick = state.tokens - self._tick_tokens[parent_id]
            reserved_this_tick = max(0, state.reserved - self._tick_reserved[parent_id])
            remaining = min(remaining, self._tick_share - spent_this_tick - reserved_this_tick)
        else:
            remaining //= max(self.n_active, 1)
        name, available = min(
            ("spawn.max_per_call", self.limits.spawn.max_per_call),
            ("spawn.max_total", self.limits.spawn.max_total - state.workers),
            ("episode.max_gen_tokens", max(0, remaining) // self.limits.worker.max_gen_tokens),
            key=lambda item: item[1],
        )
        count = min(k, available)
        if count < k:
            self.limit_hit(parent_id, name)
        state.reserved += count * self.limits.worker.max_gen_tokens
        state.pending_workers += count
        state.workers += count
        return count

    def release_worker(self, worker_id: str) -> None:
        state = self._agents[worker_id]
        if state.parent is not None and not state.released:
            self._agents[state.parent].reserved -= state.share - state.tokens
            state.released = True

    def limits_hit(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(value) for key, value in self._hits.items()}


@dataclass
class _Account:
    kind: str
    parent: str | None
    share: int
    tokens: int = 0
    session_tokens: int = 0
    calls: int = 0
    reserved: int = 0
    pending_workers: int = 0
    workers: int = 0
    done: bool = False
    released: bool = False
