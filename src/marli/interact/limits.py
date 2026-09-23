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

from dataclasses import dataclass, field

from marli.config import doc_field


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
    carry_reserve: int = doc_field(1024, help="tokens held back for the end-of-session carry call")
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
        # Validation (M1-8): enum-like strings, positive values, reserves smaller than
        # their budgets, spawn.max_depth == 1 in v1, max_ticks >= 1; raise
        # marli.errors.ConfigError naming the offending field.
        ...


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
