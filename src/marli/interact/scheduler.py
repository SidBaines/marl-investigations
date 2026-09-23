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

Both schedulers take an injectable clock for tests.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Ticket:
    tick: int | None  # None under async
    view: Any  # WorkspaceView: read-only snapshot for this turn (see workspace.py)
    seq: int  # event seq assigned to this turn's start


class Scheduler(Protocol):
    kind: str  # "lockstep" | "async"

    def register(self, agent_id: str, seat_key: tuple[Any, ...]) -> None: ...
    async def turn(self, agent_id: str) -> Ticket: ...
    def tool_phase(self, agent_id: str) -> AbstractAsyncContextManager[None]: ...
    def block(self, agent_id: str) -> None: ...
    def unblock(self, agent_id: str) -> None: ...
    def done(self, agent_id: str) -> None: ...
    @property
    def tick(self) -> int | None: ...
    def next_seq(self) -> int: ...
    # async only; False on timeout
    async def wait_for_update(self, agent_id: str, timeout: float) -> bool: ...


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...
