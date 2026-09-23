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

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from marli.errors import ConfigError
from marli.interact.types import AgentInfo, EventKind, Outcome
from marli.policy.base import TokenPolicy
from marli.registry import FnRegistry
from marli.seeds import derive_seed

if TYPE_CHECKING:
    from marli.envs.base import Env
    from marli.interact.agent import AgentRuntime
    from marli.interact.limits import Ledger
    from marli.interact.records import Recorder
    from marli.interact.run import EpisodeSpec
    from marli.interact.scheduler import Scheduler
    from marli.interact.workspace import Workspace


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
    """What a protocol can do, independently of policy and renderer details."""

    task: Any  # marli.envs.base.Task
    env: Env
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


def get_protocol(name: str, config: Any) -> Protocol:
    """Resolve built-in or externally registered orchestration by name."""
    from marli.interact.protocols import single  # noqa: F401

    return PROTOCOLS.get(name)(config)


@dataclass(frozen=True)
class AgentHandle:
    """An agent's stable identity and awaitable completion."""

    agent_id: str
    task: asyncio.Task[AgentResult]

    def __await__(self) -> Generator[Any, None, AgentResult]:
        return self.task.__await__()


class EpisodeSystem(SystemIO):
    """Own agent lifetimes and share one episode's injected services."""

    def __init__(
        self,
        spec: EpisodeSpec,
        *,
        workspace: Workspace,
        scheduler: Scheduler,
        ledger: Ledger,
        recorder: Recorder,
    ) -> None:
        self.spec = spec
        self.protocol = spec.protocol
        self.env = spec.env
        self.task = spec.task
        self.rng = random.Random(derive_seed(spec.run_seed, spec.task.task_id, spec.episode_idx))
        self.config = getattr(spec.protocol, "config", None)
        self.workspace = workspace
        self.scheduler = scheduler
        self.ledger = ledger
        self.recorder = recorder
        self.handles: list[AgentHandle] = []
        self.runtimes: dict[str, AgentRuntime] = {}
        self._roles = {role.role: role for role in spec.protocol.roles()}

    async def start_agent(
        self,
        role: str,
        *,
        agent_id: str,
        seat_key: tuple[Any, ...],
        first_message: str,
        parent: str | None = None,
    ) -> AgentHandle:
        from marli.interact.agent import AgentRuntime, SamplingOverrides
        from marli.interact.context import make_context_manager

        if role not in self._roles:
            raise ConfigError(f"unknown role {role!r}")
        if agent_id in self.runtimes:
            raise ConfigError(f"agent_id already registered: {agent_id}")
        role_spec = self._roles[role]
        policy_id = self.spec.seating[role]
        policy = self.spec.policies[policy_id]
        renderer = self.spec.renderers[policy_id]() if isinstance(policy, TokenPolicy) else None
        info = AgentInfo(agent_id, role, policy_id, parent, seat_key)
        context = make_context_manager(role_spec.context, self.spec.limits)
        runtime = AgentRuntime(
            info,
            policy=policy,
            renderer=renderer,
            role=role_spec,
            tools=self.env.tools(role),
            first_message=first_message,
            scheduler=self.scheduler,
            workspace=self.workspace,
            ledger=self.ledger,
            limits=self.spec.limits,
            context=context,
            recorder=self.recorder,
            sandbox=self.env.sandbox,
            system=self,
            run_seed=self.spec.run_seed,
            task_id=self.task.task_id,
            episode_idx=self.spec.episode_idx,
            sampling=self.spec.sampling.get(role, SamplingOverrides()),
        )
        self.ledger.register(agent_id, kind=role_spec.limits_key, parent=parent)
        self.scheduler.register(agent_id, seat_key)
        self.workspace.add_agent(agent_id, role)
        self.recorder.add_agent(info)
        if parent is not None:
            self.recorder.add_event(EventKind.SPAWN, parent, self.scheduler.tick, child=agent_id)
        self.runtimes[agent_id] = runtime
        handle = AgentHandle(agent_id, asyncio.create_task(runtime.run(), name=agent_id))
        self.handles.append(handle)
        return handle

    async def wait(self, handles: list[AgentHandle]) -> list[AgentResult]:
        return list(await asyncio.gather(*(handle.task for handle in handles)))

    async def finalize(self, role: str, *, agent_id: str, first_message: str) -> AgentResult:
        handle = await self.start_agent(
            role,
            agent_id=agent_id,
            seat_key=(role, 0),
            first_message=first_message,
        )
        return await handle

    def canonicalize(self, answer: str | None) -> str | None:
        return self.env.canonical(answer)
