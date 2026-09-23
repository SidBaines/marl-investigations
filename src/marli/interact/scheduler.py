"""Schedulers: a *gate* on agents' LLM calls, shared-state tools and commits.

Agents are long-lived coroutines (``AgentRuntime.run``); protocols start them
and the scheduler decides when each may take its next turn. It does not drive
agents — it gates them, which is what makes a blocking ``spawn_workers`` tool
(a coordinator waiting on workers that run their own multi-turn loops) fit
cleanly into lockstep scheduling.

Agent turn protocol (every agent, both schedulers)::

    t = await sched.turn(agent_id)          # -> Ticket(tick, view): may wait for the tick to open
    ... sample one completion ...
    async with sched.tool_phase(agent_id):   # ordered/serialized section for shared-state tools
        ... run tools in call order ...
    # loop; or sched.done(agent_id) when finished
    # a tool that waits on other agents calls sched.block(agent_id) ... sched.unblock(agent_id)

Lockstep (default; deterministic given deterministic policies):
  * Tick ``t`` opens when every registered, not-done, not-blocked agent is
    waiting in ``turn()``. All of them are released together with the same
    ``WorkspaceView`` snapshot (state committed through tick ``t-1``).
  * Generation runs concurrently. ``tool_phase`` admits agents one at a time in
    ``seat_key`` order, and only once every agent released at ``t`` has either
    reached its tool phase or finished (so shared tools never race).
  * Workspace writes made during tick ``t`` are *staged*; they commit in
    ``seat_key`` order when tick ``t`` closes (every agent released at ``t`` is
    back in ``turn()``, blocked, or done). An agent sees its own staged writes
    immediately; others see them from tick ``t+1``.
  * ``register`` during tick ``t`` (a spawn) makes the new agent eligible from
    tick ``t+1``. A blocked agent does not hold ticks open; when unblocked it
    joins the next tick.
  * ``Call.tick`` is the single global tick counter. ``max_ticks`` is enforced
    by the protocol via ``tick`` (the scheduler reports it).
  * Caveat (documented, not a bug): the sandbox filesystem is not snapshotted;
    shared-sandbox effects within a tick are visible in seat order.

Async (``schedule: async``):
  * ``turn()`` returns immediately (``tick=None``) with a view of the *live*
    committed state; writes commit immediately; ``tool_phase`` is a per-episode
    mutex for shared tools only (pure tools don't take it).
  * Every call start/end, tool start/end, commit and push is appended to the
    event log with a monotone ``seq`` and the version vector visible at call
    start; episodes are marked ``replayable=False``.
  * An optional ``wait_for_update(timeout)`` tool lets an agent sleep until any
    other agent commits.

Construction: ``LockstepScheduler(workspace, recorder, *, clock=None)`` and
``AsyncScheduler(workspace, recorder, *, clock=None)``. The recorder
(``interact/records.py``) is the single authority for event ``seq`` numbers:
the scheduler records TICK-level facts through it (commits: one
``EventKind.COMMIT`` event per committed write, plus ``recorder.add_write``),
while call/tool events are recorded by the agent runtime. At tick close the
lockstep scheduler calls ``workspace.commit_staged(seat_order)`` with the
agents that were released at that tick, in ``seat_key`` order.

``register`` returns immediately; a registered agent must eventually call
``turn()`` or ``done()`` — the scheduler never times agents out (the protocol
enforces ``max_wall_s``). ``done`` is idempotent. ``block``/``unblock`` nest
(a counter), so a tool may block while waiting on several workers.

Both schedulers take an injectable clock for tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from marli.interact.types import EventKind, Write


@dataclass(frozen=True)
class Ticket:
    """The tick and workspace snapshot assigned to one generation."""

    tick: int | None  # None under async
    view: Any  # WorkspaceView: read-only snapshot for this turn (see workspace.py)


class Scheduler(Protocol):
    """Gates used by agent runtimes and episode orchestration."""

    kind: str  # "lockstep" | "async"

    def register(self, agent_id: str, seat_key: tuple[Any, ...]) -> None: ...
    async def turn(self, agent_id: str) -> Ticket: ...
    def tool_phase(self, agent_id: str) -> AbstractAsyncContextManager[None]: ...
    def block(self, agent_id: str) -> None: ...
    def unblock(self, agent_id: str) -> None: ...
    def done(self, agent_id: str) -> None: ...
    def notify_commit(self, writer: str) -> None: ...
    async def wait_idle(self) -> None: ...
    @property
    def tick(self) -> int | None: ...
    # async only; False on timeout
    async def wait_for_update(self, agent_id: str, timeout: float) -> bool: ...


class Clock(Protocol):
    """Monotonic time and an awaitable delay, injectable per episode."""

    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Production clock using monotonic time and asyncio delays."""

    def now(self) -> float:
        """Return elapsed monotonic seconds."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for the requested delay."""
        await asyncio.sleep(seconds)


class FakeClock:
    """Manual monotonic clock; advancing time wakes due sleepers."""

    def __init__(self) -> None:
        self._now = 0.0
        self._sleepers: dict[asyncio.Future[None], float] = {}

    def now(self) -> float:
        """Return seconds advanced since construction."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Wait until advance reaches this sleep's deadline."""
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._sleepers[future] = self._now + seconds
        try:
            await future
        finally:
            self._sleepers.pop(future, None)

    def advance(self, seconds: float) -> None:
        """Advance without running coroutines; time cannot move backwards."""
        if seconds < 0:
            raise ValueError("cannot advance a clock backwards")
        self._now += seconds
        for future, deadline in sorted(self._sleepers.items(), key=lambda item: item[1]):
            if deadline <= self._now and not future.done():
                future.set_result(None)


class _Workspace(Protocol):
    """Only the workspace operations owned by a scheduler."""

    def view(self, reader: str) -> Any: ...
    def commit_staged(self, seat_order: list[str]) -> list[Write]: ...


class _Recorder(Protocol):
    """The recorder remains the authority for write and event sequencing."""

    def add_event(self, kind: EventKind, agent_id: str, tick: int | None, **data: Any) -> int: ...
    def add_write(self, write: Write) -> None: ...


@dataclass
class _Agent:
    """Mutable lifecycle state, independent of any particular tick."""

    seat_key: tuple[Any, ...]
    phase: str = "registered"
    blocks: int = 0
    ticket: Ticket | None = None


@dataclass
class _CohortMember:
    """Tick-local state survives a blocked tool's context-manager exit."""

    phase: str = "generating"


class _Lifecycle:
    """Shared registration, nested blocking, and episode completion accounting."""

    def __init__(
        self, workspace: _Workspace, recorder: _Recorder, *, clock: Clock | None = None
    ) -> None:
        self._workspace = workspace
        self._recorder = recorder
        self._clock = clock if clock is not None else SystemClock()
        self._agents: dict[str, _Agent] = {}
        self._changed = asyncio.Event()
        self._failure: str | None = None

    def register(self, agent_id: str, seat_key: tuple[Any, ...]) -> None:
        """Register a new id and a unique seat, including dynamically spawned agents."""
        if agent_id in self._agents:
            raise ValueError(f"agent already registered: {agent_id}")
        if any(agent.seat_key == seat_key for agent in self._agents.values()):
            raise ValueError(f"seat key already registered: {seat_key}")
        self._agents[agent_id] = _Agent(seat_key)
        self._state_changed()

    def block(self, agent_id: str) -> None:
        """Exclude this agent from barriers until all matching unblocks arrive."""
        self._agents[agent_id].blocks += 1
        self._state_changed()

    def unblock(self, agent_id: str) -> None:
        """Release one nesting level; an unblocked agent must enter its next turn."""
        agent = self._agents[agent_id]
        if agent.blocks == 0:
            raise ValueError(f"agent is not blocked: {agent_id}")
        agent.blocks -= 1
        self._state_changed()

    def done(self, agent_id: str) -> None:
        """Finish an agent permanently; repeated calls have no effect."""
        agent = self._agents[agent_id]
        if agent.phase != "done":
            agent.phase = "done"
            self._state_changed()

    async def wait_idle(self) -> None:
        """Wait for every registered agent to finish, or report lockstep deadlock."""
        while True:
            self._raise_failure()
            if all(agent.phase == "done" for agent in self._agents.values()):
                return
            await self._changed.wait()

    def _state_changed(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    def _live_agent(self, agent_id: str) -> _Agent:
        self._raise_failure()
        agent = self._agents[agent_id]
        if agent.phase == "done":
            raise RuntimeError(f"agent is done: {agent_id}")
        return agent


class LockstepScheduler(_Lifecycle):
    """Release tick cohorts together and serialize their tools in seat order."""

    kind = "lockstep"

    def __init__(
        self, workspace: _Workspace, recorder: _Recorder, *, clock: Clock | None = None
    ) -> None:
        super().__init__(workspace, recorder, clock=clock)
        self._tick: int | None = None
        self._cohort: dict[str, _CohortMember] = {}
        self._deadlock_check: asyncio.TimerHandle | None = None

    @property
    def tick(self) -> int | None:
        """Current global tick, or None before the first cohort opens."""
        return self._tick

    async def turn(self, agent_id: str) -> Ticket:
        """Finish the preceding turn and wait for the next cohort's snapshot."""
        agent = self._live_agent(agent_id)
        if agent.phase == "waiting":
            raise RuntimeError(f"agent is already waiting for a turn: {agent_id}")
        agent.phase = "waiting"
        self._state_changed()
        try:
            while agent.ticket is None:
                self._live_agent(agent_id)
                await self._changed.wait()
            self._live_agent(agent_id)
            ticket, agent.ticket = agent.ticket, None
            return ticket
        finally:
            if agent.phase == "waiting":
                agent.phase = "registered"
                self._state_changed()

    @asynccontextmanager
    async def tool_phase(self, agent_id: str) -> AsyncIterator[None]:
        """Wait for all generations, then admit this seat's shared tools."""
        self._live_agent(agent_id)
        member = self._cohort.get(agent_id)
        if member is None or member.phase != "generating":
            raise RuntimeError(f"agent has no generation awaiting tools: {agent_id}")
        member.phase = "queued"
        self._state_changed()
        try:
            while member.phase != "active":
                self._live_agent(agent_id)
                await self._changed.wait()
            self._live_agent(agent_id)
            yield
        finally:
            # A spawn may already have retired this member and opened later ticks.
            if member.phase != "finished":
                member.phase = "tools_done"
            self._state_changed()

    def notify_commit(self, writer: str) -> None:
        """No-op: lockstep commits are handled at tick close."""

    async def wait_for_update(self, agent_id: str, timeout: float) -> bool:
        """The update-waiting tool is available only under async scheduling."""
        raise NotImplementedError("wait_for_update requires async scheduling")

    def _state_changed(self) -> None:
        if self._failure is None:
            self._advance()
        if self._deadlock_check is not None:
            self._deadlock_check.cancel()
            self._deadlock_check = None
        if self._failure is None and self._all_blocked():
            # Task completion and gather callbacks must let joining parents unblock
            # before an otherwise quiescent episode is declared deadlocked.
            self._deadlock_check = asyncio.get_running_loop().call_later(
                0, self._check_deadlock, False
            )
        super()._state_changed()

    def _advance(self) -> None:
        for agent_id, member in self._cohort.items():
            agent = self._agents[agent_id]
            if agent.blocks or agent.phase in {"waiting", "done"}:
                member.phase = "finished"

        if self._cohort and all(m.phase == "finished" for m in self._cohort.values()):
            writes = self._workspace.commit_staged(list(self._cohort))
            for write in writes:
                self._recorder.add_write(write)
                self._recorder.add_event(
                    EventKind.COMMIT, write.writer, self._tick, key=write.key, version=write.version
                )
            self._cohort = {}

        if not self._cohort:
            eligible = [
                agent_id
                for agent_id, agent in self._agents.items()
                if agent.phase != "done" and not agent.blocks
            ]
            if eligible and all(self._agents[a].phase == "waiting" for a in eligible):
                eligible.sort(key=lambda agent_id: self._agents[agent_id].seat_key)
                self._tick = 0 if self._tick is None else self._tick + 1
                # Capture every view before releasing even the coroutine opening the tick.
                tickets = {a: Ticket(self._tick, self._workspace.view(a)) for a in eligible}
                self._cohort = {a: _CohortMember() for a in eligible}
                for agent_id, ticket in tickets.items():
                    self._agents[agent_id].ticket = ticket
                    self._agents[agent_id].phase = "running"

        if any(m.phase in {"generating", "active"} for m in self._cohort.values()):
            return
        for member in self._cohort.values():
            if member.phase == "queued":
                member.phase = "active"
                return

    def _all_blocked(self) -> bool:
        remaining = [agent for agent in self._agents.values() if agent.phase != "done"]
        return bool(remaining) and all(agent.blocks for agent in remaining)

    def _check_deadlock(self, settled: bool) -> None:
        self._deadlock_check = None
        if not self._all_blocked():
            return
        if not settled:
            self._deadlock_check = asyncio.get_running_loop().call_later(
                0, self._check_deadlock, True
            )
            return
        blocked = sorted(a for a, state in self._agents.items() if state.phase != "done")
        self._failure = f"lockstep deadlock: all remaining agents are blocked: {', '.join(blocked)}"
        super()._state_changed()


class AsyncScheduler(_Lifecycle):
    """Free-running turns with FIFO shared tools and commit notifications."""

    kind = "async"

    def __init__(
        self, workspace: _Workspace, recorder: _Recorder, *, clock: Clock | None = None
    ) -> None:
        super().__init__(workspace, recorder, clock=clock)
        self._tool_lock = asyncio.Lock()
        self._updates: dict[asyncio.Future[bool], tuple[str, float]] = {}

    @property
    def tick(self) -> None:
        """Async turns do not have global ticks."""
        return None

    async def turn(self, agent_id: str) -> Ticket:
        """Return immediately with this agent's current live workspace view."""
        agent = self._live_agent(agent_id)
        agent.phase = "running"
        return Ticket(None, self._workspace.view(agent_id))

    @asynccontextmanager
    async def tool_phase(self, agent_id: str) -> AsyncIterator[None]:
        """Serialize shared tools with the episode's FIFO asyncio lock."""
        self._live_agent(agent_id)
        async with self._tool_lock:
            self._live_agent(agent_id)
            yield

    def notify_commit(self, writer: str) -> None:
        """Wake current update waiters belonging to other agents."""
        for future, (reader, deadline) in self._updates.items():
            if reader != writer and not future.done():
                future.set_result(self._clock.now() < deadline)

    async def wait_for_update(self, agent_id: str, timeout: float) -> bool:
        """Wait for another writer's next commit, returning False at the deadline."""
        self._live_agent(agent_id)
        if timeout <= 0:
            return False
        deadline = self._clock.now() + timeout
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._updates[future] = (agent_id, deadline)

        async def expire() -> None:
            await self._clock.sleep(max(0.0, deadline - self._clock.now()))
            if not future.done():
                future.set_result(False)

        timer = asyncio.create_task(expire())
        try:
            return await future
        finally:
            self._updates.pop(future, None)
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
