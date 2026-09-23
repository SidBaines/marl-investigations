"""Keep session ablations in configuration, with token carry owned by the runtime.

``sessions`` must match ``EpisodeSpec.limits.session.max_sessions``; this
protocol validates that setting before starting the solver and never changes
the episode's scientific limits. The current SystemIO contract has no limits
accessor, so this check uses the concrete EpisodeSystem's public ``spec``.
Session counts come from the runtime's call/segment records in compute_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from marli.errors import ConfigError
from marli.interact.system import (
    PROTOCOLS,
    ContextSpec,
    EpisodeSystem,
    Protocol,
    RoleSpec,
    SystemIO,
)
from marli.interact.types import Outcome


@dataclass(frozen=True)
class MultiSessionConfig:
    """Session settings; zero compact_threshold enables only session-end summaries.

    Prompt fields are sessions, carry, carry_instructions, tail_tokens and
    notes_cap_chars, plus the runtime's agent_id, role and n_agents fields.
    The token budget itself is configured by Limits.session.max_gen_tokens.
    With sessions=1 the runtime uses single-session semantics: end_session
    stops the agent without a forced final call.
    """

    sessions: int = 3
    carry: str = "compaction"
    tail_tokens: int = 2048
    notes_cap_chars: int = 4000
    compact_threshold: int = 0
    tools: tuple[str, ...] = ("submit", "end_session")
    env_tools: tuple[str, ...] = ()
    system_prompt: str = (
        "Solve the task carefully. You have up to {sessions} sessions, each bounded by "
        "its session token budget. Your context is cleared between sessions. "
        "{carry_instructions} A session ends when its token budget runs out or you call "
        "end_session. Use submit for your final answer; submitting ends the whole episode."
    )

    def __post_init__(self) -> None:
        if self.carry not in ("compaction", "notes", "both", "tail"):
            raise ConfigError("carry must be compaction, notes, both or tail")
        for name in ("sessions", "tail_tokens", "notes_cap_chars"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.compact_threshold) is not int or self.compact_threshold < 0:
            raise ConfigError("compact_threshold must be a non-negative integer")
        if self.compact_threshold and self.carry not in ("compaction", "both"):
            raise ConfigError("compact_threshold requires compaction or both carry")


@PROTOCOLS.register("multi_session")
class MultiSessionProtocol(Protocol):
    name = "multi_session"

    def __init__(self, config: MultiSessionConfig | None = None) -> None:
        self.config = config if config is not None else MultiSessionConfig()

    def roles(self) -> list[RoleSpec]:
        cfg = self.config
        carry_instructions = {
            "compaction": (
                "At each session boundary you will be asked to write a summary for your next "
                "session; that summary accompanies the original task in the fresh context."
            ),
            "notes": (
                f"Use write_notes and read_notes to maintain private notes capped at "
                f"{cfg.notes_cap_chars} characters. Your next session sees those notes "
                "alongside the original task."
            ),
            "both": (
                f"Use write_notes and read_notes to maintain private notes capped at "
                f"{cfg.notes_cap_chars} characters. At each session boundary you will also "
                "write a summary; your next session sees the original task, summary and notes."
            ),
            "tail": (
                f"Only the last {cfg.tail_tokens} tokens of the previous context are carried "
                "as a prefill alongside the original task in your next session."
            ),
        }[cfg.carry]
        prompt = cfg.system_prompt.format(
            sessions=cfg.sessions,
            carry=cfg.carry,
            carry_instructions=carry_instructions,
            tail_tokens=cfg.tail_tokens,
            notes_cap_chars=cfg.notes_cap_chars,
            agent_id="{agent_id}",
            role="{role}",
            n_agents="{n_agents}",
        )
        notes_tools = ("read_notes", "write_notes") if cfg.carry in ("notes", "both") else ()
        return [
            RoleSpec(
                "solver",
                tuple(dict.fromkeys((*cfg.tools, *notes_tools, *cfg.env_tools))),
                prompt,
                context=ContextSpec(
                    kind=cfg.carry,
                    compact_threshold=cfg.compact_threshold,
                    tail_tokens=cfg.tail_tokens if cfg.carry == "tail" else 0,
                    notes_cap_chars=cfg.notes_cap_chars,
                ),
            )
        ]

    async def run(self, io: SystemIO) -> Outcome:
        limits = cast(EpisodeSystem, io).spec.limits
        if limits.session.max_sessions != self.config.sessions:
            raise ConfigError(
                f"multi_session.sessions={self.config.sessions} must match "
                f"limits.session.max_sessions={limits.session.max_sessions}"
            )
        handle = await io.start_agent(
            "solver",
            agent_id="solver0",
            seat_key=("solver", 0),
            first_message=io.env.task_message("solver"),
        )
        result = await handle
        return Outcome(result.submission, {"solver0": result.submission}, "multi_session")
