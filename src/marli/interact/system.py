"""Protocols (agent-system orchestration) and the IO surface they run against.

A **protocol** is orchestration code: which roles exist, which tools and
limits each role gets, how agents are started (all at once, spawned by a
coordinator, one session after another), and how the system's answer is
formed. Protocols never sample or render tokens themselves — they create
agents through :class:`SystemIO` and await them. Registered by name in
``PROTOCOLS`` and configured by YAML under ``interact/configs/``.

A protocol's ``run`` returns an :class:`~marli.interact.types.Outcome`; the
rollout layer (``eval/rollout.py``) assembles the full ``Episode`` (calls,
segments, events, workspace log, grades, compute metrics) from the SystemIO
recorder, so every protocol gets identical record-keeping for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from marli.interact.types import Outcome
from marli.registry import FnRegistry


@dataclass(frozen=True)
class ContextSpec:
    """How an agent manages its context (interact/context.py)."""

    kind: str = "none"  # none | compaction | notes | tail | both (compaction + notes)
    # compaction: trigger when the next prompt would exceed this many tokens
    compact_threshold: int = 0
    compact_reserve: int = 1024  # tokens reserved for the compaction call itself
    notes_cap_chars: int = 4000  # notes: max size of the notes file carried across sessions
    tail_tokens: int = 0  # tail (Delethink-style): last m ids carried as a prefill


@dataclass(frozen=True)
class RoleSpec:
    role: str
    # tool names from the tool registry (interact/tools.py) + env tools
    tools: tuple[str, ...]
    system_prompt: str  # template; formatted with {agent_id}, {role}, {n_agents}, ...
    count: int | None = 1  # None = dynamic (workers)
    context: ContextSpec = ContextSpec()
    limits_key: str = "agent"  # which Limits block applies: "agent" | "worker"


@dataclass(frozen=True)
class AgentResult:
    agent_id: str
    submission: str | None  # the agent's own submitted answer (None if none)
    report: str | None  # workers: returned report
    ended_by: str  # submit | end_agent | budget | error | report | max_ticks
    n_sessions: int = 1


class SystemIO(ABC):
    """What a protocol can do. Implemented once (interact/system.py, M1-9)."""

    task: Any  # marli.envs.base.Task
    rng: Any  # random.Random seeded per episode
    config: Any  # the protocol's own config dataclass

    @abstractmethod
    async def start_agent(
        self,
        role: str,
        *,
        agent_id: str,
        seat_key: tuple[Any, ...],
        first_message: str,
        parent: str | None = None,
    ) -> Any: ...  # -> AgentHandle (awaitable -> AgentResult)

    @abstractmethod
    async def wait(self, handles: list[Any]) -> list[AgentResult]: ...

    @abstractmethod
    async def finalize(self, role: str, *, agent_id: str, first_message: str) -> AgentResult: ...

    # a fresh-context agent that runs after the others (e.g. swarm finalizer: 'separate')

    @abstractmethod
    # env verifier-equivalence key (votes)
    def canonicalize(self, answer: str | None) -> str | None: ...


class Protocol(ABC):
    name: str

    @abstractmethod
    def roles(self) -> list[RoleSpec]: ...

    @abstractmethod
    async def run(self, io: SystemIO) -> Outcome: ...


PROTOCOLS: FnRegistry = FnRegistry("protocols")  # name -> factory(config) -> Protocol
