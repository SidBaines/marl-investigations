"""Divide one agent's compute across explicit sessions and controlled carry modes.

The protocol owns session counts and budgets, leaving the agent's final reserve
outside that division. The runtime manages session identity and context resets;
protocol prompts explain exactly which state survives them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from string import Formatter

from marli.errors import ConfigError
from marli.interact.limits import Limits
from marli.interact.system import (
    PROTOCOLS,
    ContextSpec,
    Protocol,
    RoleSpec,
    SystemIO,
)
from marli.interact.types import Outcome


@dataclass(frozen=True)
class MultiSessionConfig:
    """Session settings; zero compact_threshold enables only session-end summaries.

    Prompt fields are sessions, carry, carry_instructions, tail_tokens and
    notes_cap_chars, plus the runtime's agent_id, role, n_agents and
    session_tokens fields. An omitted session_tokens divides the agent's
    available budget equally; explicit budgets must fit that same total.
    """

    sessions: int = 3
    session_tokens: int | None = None
    carry: str = "compaction"
    tail_tokens: int = 2048
    notes_cap_chars: int = 4000
    compact_threshold: int = 0
    tools: tuple[str, ...] = ("submit", "end_session")
    env_tools: tuple[str, ...] = ()
    system_prompt: str = (
        "Solve the task carefully. You have up to {sessions} sessions, each bounded by "
        "a session token budget of {session_tokens} generated tokens. "
        "Your context is cleared between sessions. {carry_instructions} "
        "Use submit for your final answer; submitting ends the whole episode."
    )

    def __post_init__(self) -> None:
        if self.carry not in ("compaction", "notes", "both", "tail"):
            raise ConfigError("carry must be compaction, notes, both or tail")
        for name in ("sessions", "tail_tokens", "notes_cap_chars"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if self.sessions == 1:
            raise ConfigError("multi_session requires at least two sessions; use single")
        if self.session_tokens is not None and (
            type(self.session_tokens) is not int or self.session_tokens <= 0
        ):
            raise ConfigError("session_tokens must be a positive integer or None")
        if type(self.compact_threshold) is not int or self.compact_threshold < 0:
            raise ConfigError("compact_threshold must be a non-negative integer")
        if self.compact_threshold and self.carry not in ("compaction", "both"):
            raise ConfigError("compact_threshold requires compaction or both carry")
        try:
            self.system_prompt.format(
                sessions=self.sessions,
                session_tokens=1024,
                carry=self.carry,
                carry_instructions="carry",
                tail_tokens=self.tail_tokens,
                notes_cap_chars=self.notes_cap_chars,
                agent_id="solver0",
                role="solver",
                n_agents=1,
                max_workers_per_call=4,
                max_workers_total=8,
                worker_tokens=8192,
            )
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ConfigError(f"invalid system_prompt template: {exc}") from exc


@PROTOCOLS.register("multi_session")
class MultiSessionProtocol(Protocol):
    name = "multi_session"
    config_type = MultiSessionConfig

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
                "alongside the original task. Save progress to notes regularly: sessions "
                "end without warning when the budget runs out."
            ),
            "both": (
                f"Use write_notes and read_notes to maintain private notes capped at "
                f"{cfg.notes_cap_chars} characters. At each session boundary you will also "
                "write a summary; your next session sees the original task, summary and notes."
            ),
            "tail": (
                f"Only the last {cfg.tail_tokens} tokens of your own replies (including "
                "thinking, excluding tool calls) are carried as text alongside the original "
                "task in your next session. Sessions end when their token budget runs out."
            ),
        }[cfg.carry]
        if cfg.carry != "tail":
            carry_instructions += (
                " A session ends when its token budget runs out or you call end_session."
            )
        fields = {
            "sessions": cfg.sessions,
            "carry": cfg.carry,
            "carry_instructions": carry_instructions,
            "tail_tokens": cfg.tail_tokens,
            "notes_cap_chars": cfg.notes_cap_chars,
        }
        # Keep runtime fields and literal braces intact for the final format pass.
        parts = []
        for literal, name, spec, conversion in Formatter().parse(cfg.system_prompt):
            parts.append(literal.replace("{", "{{").replace("}", "}}"))
            if name is not None:
                field = "{" + name + ("!" + conversion if conversion else "")
                field += (":" + spec if spec else "") + "}"
                if name in fields:
                    field = field.format(**fields).replace("{", "{{").replace("}", "}}")
                parts.append(field)
        prompt = "".join(parts)
        notes_tools = ("read_notes", "write_notes") if cfg.carry in ("notes", "both") else ()
        tools = (
            ("submit",)
            if cfg.carry == "tail" and cfg.tools == ("submit", "end_session")
            else cfg.tools
        )
        return [
            RoleSpec(
                "solver",
                tuple(dict.fromkeys((*tools, *notes_tools, *cfg.env_tools))),
                prompt,
                context=ContextSpec(
                    kind=cfg.carry,
                    compact_threshold=cfg.compact_threshold,
                    tail_tokens=cfg.tail_tokens if cfg.carry == "tail" else 0,
                    notes_cap_chars=cfg.notes_cap_chars,
                ),
            )
        ]

    def adjust_limits(self, limits: Limits) -> Limits:
        """Compute-match all sessions to the agent budget, preserving its final reserve."""
        cfg = self.config
        available = limits.agent.max_gen_tokens - limits.agent.final_reserve
        tokens = available // cfg.sessions if cfg.session_tokens is None else cfg.session_tokens
        if cfg.sessions * tokens > available:
            raise ConfigError("sessions * session_tokens exceeds agent budget minus final_reserve")
        return replace(
            limits,
            session=replace(
                limits.session,
                max_sessions=cfg.sessions,
                max_gen_tokens=tokens,
                carry_reserve=0 if cfg.carry in ("notes", "tail") else limits.session.carry_reserve,
            ),
        )

    async def run(self, io: SystemIO) -> Outcome:
        handle = await io.start_agent(
            "solver",
            agent_id="solver0",
            seat_key=("solver", 0),
            first_message=io.env.task_message("solver"),
        )
        result = await handle
        return Outcome(result.submission, {"solver0": result.submission}, "multi_session")
