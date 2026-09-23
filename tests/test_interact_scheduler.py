"""Scheduler gates preserve tick isolation without depending on workspace or recorder code."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from marli.interact.scheduler import AsyncScheduler, FakeClock, LockstepScheduler, SystemClock
from marli.interact.types import EventKind, Write


@dataclass(frozen=True)
class FakeView:
    versions: dict[tuple[str, str], int]
    contents: dict[tuple[str, str, int], str]
    own_staged: dict[str, str] = field(default_factory=dict)


class FakeWorkspace:
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log
        self.committed: list[Write] = []
        self.staged: list[Write] = []
        self.commits: list[list[str]] = []

    def view(self, reader: str) -> FakeView:
        self.log.append(("view", reader))
        return FakeView(
            {(w.writer, w.key): w.version for w in self.committed},
            {(w.writer, w.key, w.version): w.content for w in self.committed},
            {w.key: w.content for w in self.staged if w.writer == reader},
        )

    def stage(self, writer: str, tick: int | None) -> Write:
        version = 1 + sum(w.writer == writer for w in self.committed + self.staged)
        write = Write(writer, "scratchpad", version, f"{writer}/{version}", tick, version)
        self.log.append(("stage", writer, tick))
        self.staged.append(write)
        return write

    def commit_staged(self, seat_order: list[str]) -> list[Write]:
        self.commits.append(seat_order.copy())
        self.log.append(("commit", tuple(seat_order)))
        writes = [w for agent in seat_order for w in self.staged if w.writer == agent]
        self.staged = [w for w in self.staged if w.writer not in seat_order]
        self.committed.extend(writes)
        return writes


class FakeRecorder:
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log
        self.writes: list[Write] = []
        self.events: list[tuple[Any, ...]] = []

    def add_event(self, kind: EventKind, agent_id: str, tick: int | None, **data: Any) -> int:
        seq = len(self.events)
        event = (seq, kind, agent_id, tick, data)
        self.events.append(event)
        self.log.append(("event", *event))
        return seq

    def add_write(self, write: Write) -> None:
        self.writes.append(write)
        self.log.append(("write", write))


async def settle() -> None:
    """Run ready coroutines and zero-delay callbacks without using wall-clock sleeps."""
    for _ in range(10):
        await asyncio.sleep(0)


async def scripted_trace() -> tuple[list[tuple[Any, ...]], FakeRecorder]:
    log: list[tuple[Any, ...]] = []
    workspace, recorder = FakeWorkspace(log), FakeRecorder(log)
    sched = LockstepScheduler(workspace, recorder)
    # Registration order differs from seat order.
    sched.register("high", (1,))
    sched.register("low", (0,))
    assert sched.tick is None

    async def agent(name: str) -> None:
        for tick in range(3):
            ticket = await sched.turn(name)
            log.append(("run", name, ticket.tick))
            assert ticket.tick == sched.tick == tick
            expected = {} if tick == 0 else {(a, "scratchpad"): tick for a in ("low", "high")}
            assert ticket.view.versions == expected
            # Different completion orders also exercise pure-tool turns, which skip tool_phase.
            delay = 3 if (name == "low") == (tick % 2 == 0) else 1
            for _ in range(delay):
                await asyncio.sleep(0)
            workspace.stage(name, ticket.tick)
            assert len(workspace.committed) == tick * 2
            assert ticket.view.versions == expected
        sched.done(name)

    async with asyncio.timeout(2):
        await asyncio.gather(agent("high"), agent("low"), sched.wait_idle())
    assert workspace.commits == [["low", "high"]] * 3
    assert [w.writer for w in recorder.writes] == ["low", "high"] * 3
    assert [w.tick for w in recorder.writes] == [0, 0, 1, 1, 2, 2]
    for tick in range(3):
        views = [i for i, item in enumerate(log) if item[0] == "view"][tick * 2 : tick * 2 + 2]
        runs = [i for i, item in enumerate(log) if item[0] == "run" and item[2] == tick]
        stages = [i for i, item in enumerate(log) if item[0] == "stage" and item[2] == tick]
        commit = [i for i, item in enumerate(log) if item[0] == "commit"][tick]
        assert max(views) < min(runs)
        assert max(stages) < commit
        assert recorder.events[tick * 2 : tick * 2 + 2] == [
            (
                tick * 2 + seat,
                EventKind.COMMIT,
                name,
                tick,
                {"key": "scratchpad", "version": tick + 1},
            )
            for seat, name in enumerate(("low", "high"))
        ]
    assert [item[1] for item in log if item[0] == "stage"] == [
        "high", "low", "low", "high", "high", "low"
    ]
    return log, recorder


async def test_three_ticks_snapshot_isolation_and_commit_order() -> None:
    await scripted_trace()


async def test_scripted_lockstep_event_log_is_deterministic() -> None:
    first_log, first_recorder = await scripted_trace()
    second_log, second_recorder = await scripted_trace()
    assert first_log == second_log
    assert first_recorder.events == second_recorder.events


async def test_tools_wait_for_slow_generator_then_run_in_seat_order() -> None:
    log: list[tuple[Any, ...]] = []
    workspace, recorder = FakeWorkspace(log), FakeRecorder(log)
    sched = LockstepScheduler(workspace, recorder)
    sched.register("low", (0,))
    sched.register("high", (1,))
    generated_high, finish_low_generation = asyncio.Event(), asyncio.Event()
    low_entered, release_low = asyncio.Event(), asyncio.Event()
    entered: list[str] = []

    async def agent(name: str) -> None:
        ticket = await sched.turn(name)
        if name == "low":
            await finish_low_generation.wait()
        else:
            generated_high.set()
        async with sched.tool_phase(name):
            entered.append(name)
            workspace.stage(name, ticket.tick)
            if name == "low":
                low_entered.set()
                await release_low.wait()

    async with asyncio.timeout(2):
        tasks = [asyncio.create_task(agent(name)) for name in ("low", "high")]
        await generated_high.wait()
        await settle()
        assert entered == []
        finish_low_generation.set()
        await low_entered.wait()
        await settle()
        assert entered == ["low"]
        assert workspace.commits == []
        release_low.set()
        await asyncio.gather(*tasks)
        assert entered == ["low", "high"]
        assert workspace.commits == []  # Exiting tools alone does not close a tick.
        sched.done("high")
        assert workspace.commits == []
        sched.done("low")
        await sched.wait_idle()
    assert workspace.commits == [["low", "high"]]


@pytest.mark.parametrize("finish", ["turn", "done", "block"])
async def test_skipped_tool_phase_does_not_stall_other_seats(finish: str) -> None:
    log: list[tuple[Any, ...]] = []
    sched = LockstepScheduler(FakeWorkspace(log), FakeRecorder(log))
    sched.register("low", (0,))
    sched.register("high", (1,))
    await asyncio.gather(sched.turn("low"), sched.turn("high"))
    entered = asyncio.Event()

    async def high() -> None:
        async with sched.tool_phase("high"):
            entered.set()

    async with asyncio.timeout(2):
        task = asyncio.create_task(high())
        await settle()
        assert not entered.is_set()
        next_turn = None
        if finish == "turn":
            next_turn = asyncio.create_task(sched.turn("low"))
        elif finish == "done":
            sched.done("low")
        else:
            sched.block("low")
        await task
        assert entered.is_set()
        if finish == "block":
            sched.unblock("low")
        sched.done("high")
        if next_turn is not None:
            assert (await next_turn).tick == 1
        sched.done("low")
        await sched.wait_idle()


async def test_registration_during_a_tick_joins_only_the_next_cohort() -> None:
    log: list[tuple[Any, ...]] = []
    workspace = FakeWorkspace(log)
    sched = LockstepScheduler(workspace, FakeRecorder(log))
    sched.register("parent", (0,))
    assert (await sched.turn("parent")).tick == 0
    async with sched.tool_phase("parent"):
        workspace.stage("parent", 0)
        sched.register("worker", (0, 0))
        worker_turn = asyncio.create_task(sched.turn("worker"))
        await settle()
        assert not worker_turn.done()
        assert sched.tick == 0
        assert workspace.commits == []
    async with asyncio.timeout(2):
        tickets = await asyncio.gather(sched.turn("parent"), worker_turn)
    assert [ticket.tick for ticket in tickets] == [1, 1]
    assert workspace.commits == [["parent"]]
    assert tickets[1].view.versions == {("parent", "scratchpad"): 1}
    sched.done("parent")
    sched.done("worker")
    await sched.wait_idle()
    assert workspace.commits == [["parent"], ["parent", "worker"]]


async def test_nested_blocking_spawn_workers_and_parent_resume() -> None:
    log: list[tuple[Any, ...]] = []
    workspace = FakeWorkspace(log)
    sched = LockstepScheduler(workspace, FakeRecorder(log))
    sched.register("parent", (0,))
    ticks: dict[str, list[int | None]] = {a: [] for a in ("parent", "w0", "w1")}

    async def worker(name: str, count: int) -> None:
        for _ in range(count):
            ticket = await sched.turn(name)
            ticks[name].append(ticket.tick)
            async with sched.tool_phase(name):
                workspace.stage(name, ticket.tick)
        sched.done(name)

    async def parent() -> None:
        ticks["parent"].append((await sched.turn("parent")).tick)
        async with sched.tool_phase("parent"):
            workspace.stage("parent", sched.tick)
            sched.register("w0", (0, 0))
            sched.register("w1", (0, 1))
            sched.block("parent")
            sched.block("parent")
            first = asyncio.create_task(worker("w0", 1))
            last = asyncio.create_task(worker("w1", 3))
            await first
            sched.unblock("parent")
            await asyncio.gather(last)
            sched.unblock("parent")
        ticks["parent"].append((await sched.turn("parent")).tick)
        sched.done("parent")

    async with asyncio.timeout(2):
        await asyncio.gather(parent(), sched.wait_idle())
    assert ticks == {"parent": [0, 4], "w0": [1], "w1": [1, 2, 3]}
    assert workspace.commits == [["parent"], ["w0", "w1"], ["w1"], ["w1"], ["parent"]]


async def test_unblock_mid_tick_rejoins_next_tick_without_reopening_old_barriers() -> None:
    log: list[tuple[Any, ...]] = []
    workspace = FakeWorkspace(log)
    sched = LockstepScheduler(workspace, FakeRecorder(log))
    sched.register("low", (0,))
    sched.register("high", (1,))
    await asyncio.gather(sched.turn("low"), sched.turn("high"))
    sched.block("low")
    sched.unblock("low")
    resumed = asyncio.create_task(sched.turn("low"))
    await settle()
    assert not resumed.done()
    assert sched.tick == 0
    async with sched.tool_phase("high"):
        workspace.stage("high", 0)
    async with asyncio.timeout(2):
        tickets = await asyncio.gather(resumed, sched.turn("high"))
    assert [ticket.tick for ticket in tickets] == [1, 1]
    assert workspace.commits == [["low", "high"]]
    sched.done("low")
    sched.done("high")


async def test_blocked_tool_context_exit_cannot_release_a_later_ticks_tools() -> None:
    log: list[tuple[Any, ...]] = []
    sched = LockstepScheduler(FakeWorkspace(log), FakeRecorder(log))
    sched.register("parent", (0,))
    await sched.turn("parent")
    worker_entered, release_worker = asyncio.Event(), asyncio.Event()
    entered: list[str] = []

    async def worker(name: str) -> None:
        await sched.turn(name)
        async with sched.tool_phase(name):
            entered.append(name)
            if name == "w0":
                worker_entered.set()
                await release_worker.wait()
        sched.done(name)

    async with asyncio.timeout(2):
        async with sched.tool_phase("parent"):
            sched.register("w0", (0, 0))
            sched.register("w1", (0, 1))
            sched.block("parent")
            tasks = [asyncio.create_task(worker(a)) for a in ("w0", "w1")]
            await worker_entered.wait()
            sched.unblock("parent")
        await settle()
        assert entered == ["w0"]
        release_worker.set()
        await asyncio.gather(*tasks)
        assert (await sched.turn("parent")).tick == 2
        sched.done("parent")
        await sched.wait_idle()
    assert entered == ["w0", "w1"]


@pytest.mark.parametrize("scheduler_type", [LockstepScheduler, AsyncScheduler])
async def test_lifecycle_validation_done_idempotence_and_wait_idle(
    scheduler_type: type[LockstepScheduler] | type[AsyncScheduler],
) -> None:
    log: list[tuple[Any, ...]] = []
    workspace = FakeWorkspace(log)
    sched = scheduler_type(workspace, FakeRecorder(log))
    await sched.wait_idle()
    sched.register("a", (0,))
    sched.register("b", (1,))
    with pytest.raises(ValueError, match="already registered"):
        sched.register("a", (2,))
    with pytest.raises(ValueError, match="seat key"):
        sched.register("c", (1,))
    with pytest.raises(ValueError, match="not blocked"):
        sched.unblock("a")
    idle = asyncio.create_task(sched.wait_idle())
    await settle()
    assert not idle.done()
    sched.block("a")
    sched.block("a")
    sched.unblock("a")
    sched.unblock("a")
    sched.done("a")
    sched.done("a")
    await settle()
    assert not idle.done()
    sched.done("b")
    await asyncio.wait_for(idle, 2)
    await sched.wait_idle()
    with pytest.raises(RuntimeError, match="is done"):
        await sched.turn("a")
    assert workspace.commits == []


async def test_deadlock_reaches_all_pending_turn_tool_and_idle_waiters() -> None:
    log: list[tuple[Any, ...]] = []
    sched = LockstepScheduler(FakeWorkspace(log), FakeRecorder(log))
    sched.register("low", (0,))
    sched.register("high", (1,))
    await asyncio.gather(sched.turn("low"), sched.turn("high"))

    async def tool_waiter() -> None:
        async with sched.tool_phase("high"):
            pytest.fail("slow generator has not finished")

    tasks = [asyncio.create_task(tool_waiter()), asyncio.create_task(sched.wait_idle())]
    await settle()
    sched.register("new", (2,))
    tasks.append(asyncio.create_task(sched.turn("new")))
    await settle()
    sched.block("new")
    sched.block("high")
    sched.block("low")
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all(str(result).startswith("lockstep deadlock:") for result in results)
    with pytest.raises(RuntimeError, match="lockstep deadlock:"):
        await sched.wait_idle()


async def test_registered_agent_is_not_mistaken_for_a_deadlock() -> None:
    log: list[tuple[Any, ...]] = []
    sched = LockstepScheduler(FakeWorkspace(log), FakeRecorder(log))
    sched.register("late", (0,))
    sched.register("waiting", (1,))
    turn = asyncio.create_task(sched.turn("waiting"))
    idle = asyncio.create_task(sched.wait_idle())
    await settle()
    assert not turn.done() and not idle.done()
    sched.done("late")
    assert (await asyncio.wait_for(turn, 2)).tick == 0
    sched.done("waiting")
    await asyncio.wait_for(idle, 2)


async def test_done_closes_last_tick_exactly_once() -> None:
    log: list[tuple[Any, ...]] = []
    workspace, recorder = FakeWorkspace(log), FakeRecorder(log)
    sched = LockstepScheduler(workspace, recorder)
    sched.register("a", (0,))
    await sched.turn("a")
    workspace.stage("a", 0)
    sched.done("a")
    completed_log = log.copy()
    sched.done("a")
    await sched.wait_idle()
    assert log == completed_log
    assert workspace.commits == [["a"]]
    assert len(recorder.writes) == len(recorder.events) == 1


async def test_lockstep_update_notification_is_noop_and_wait_tool_is_async_only() -> None:
    log: list[tuple[Any, ...]] = []
    sched = LockstepScheduler(FakeWorkspace(log), FakeRecorder(log))
    sched.register("a", (0,))
    sched.notify_commit("a")
    assert log == []
    assert sched.tick is None
    with pytest.raises(NotImplementedError, match="async"):
        await sched.wait_for_update("a", 1)


async def test_async_turns_are_immediate_and_views_are_live_snapshots() -> None:
    log: list[tuple[Any, ...]] = []
    workspace, recorder = FakeWorkspace(log), FakeRecorder(log)
    sched = AsyncScheduler(workspace, recorder)
    sched.register("a", (0,))
    sched.register("b", (1,))
    other_ran = False

    async def other() -> None:
        nonlocal other_ran
        other_ran = True

    other_task = asyncio.create_task(other())
    first = await sched.turn("a")
    assert not other_ran  # No suspension, even though b has not called turn.
    write = Write("b", "scratchpad", 1, "live", None, 0)
    workspace.committed.append(write)
    sched.notify_commit("b")
    second = await sched.turn("a")
    assert not other_ran
    assert first.tick is second.tick is sched.tick is None
    assert first.view.versions == {}
    assert second.view.versions == {("b", "scratchpad"): 1}
    assert recorder.events == recorder.writes == workspace.commits == []
    await other_task
    sched.done("a")
    sched.done("b")
    await sched.wait_idle()


async def test_async_tools_are_serialized_fifo_and_release_on_exception() -> None:
    log: list[tuple[Any, ...]] = []
    sched = AsyncScheduler(FakeWorkspace(log), FakeRecorder(log))
    for seat, name in enumerate(("low", "middle", "high")):
        sched.register(name, (seat,))
    entered: list[str] = []
    active = 0

    async def tools(name: str) -> None:
        nonlocal active
        async with sched.tool_phase(name):
            assert active == 0
            active += 1
            entered.append(name)
            await asyncio.sleep(0)
            active -= 1

    async with asyncio.timeout(2):
        with pytest.raises(ValueError, match="tool failed"):
            async with sched.tool_phase("low"):
                # Queue in reverse seat order: async admission is FIFO, not by seat.
                first = asyncio.create_task(tools("high"))
                await settle()
                second = asyncio.create_task(tools("middle"))
                await settle()
                assert entered == []
                raise ValueError("tool failed")
        await asyncio.gather(first, second)
    assert entered == ["high", "middle"]


async def test_async_wait_for_other_commit_ignores_self_and_old_notifications() -> None:
    log: list[tuple[Any, ...]] = []
    clock = FakeClock()
    sched = AsyncScheduler(FakeWorkspace(log), FakeRecorder(log), clock=clock)
    sched.register("a", (0,))
    sched.register("b", (1,))
    sched.notify_commit("b")
    a_waiters = [asyncio.create_task(sched.wait_for_update("a", 5)) for _ in range(2)]
    b_waiter = asyncio.create_task(sched.wait_for_update("b", 5))
    await settle()
    assert not any(task.done() for task in [*a_waiters, b_waiter])
    sched.notify_commit("a")
    assert await asyncio.wait_for(b_waiter, 2) is True
    assert not any(task.done() for task in a_waiters)
    clock.advance(4)
    await settle()
    assert not any(task.done() for task in a_waiters)
    sched.notify_commit("b")
    assert await asyncio.wait_for(asyncio.gather(*a_waiters), 2) == [True, True]
    clock.advance(10)  # Cancelled timeout sleepers cannot affect a later wait.
    assert log == []


async def test_async_update_timeout_and_cancellation_use_injected_clock() -> None:
    log: list[tuple[Any, ...]] = []
    clock = FakeClock()
    sched = AsyncScheduler(FakeWorkspace(log), FakeRecorder(log), clock=clock)
    sched.register("a", (0,))
    waiting = asyncio.create_task(sched.wait_for_update("a", 5))
    await settle()
    clock.advance(4.5)
    await settle()
    assert not waiting.done()
    sched.notify_commit("a")
    clock.advance(0.5)
    assert await asyncio.wait_for(waiting, 2) is False
    assert await sched.wait_for_update("a", 0) is False
    assert await sched.wait_for_update("a", -1) is False
    cancelled = asyncio.create_task(sched.wait_for_update("a", 10))
    await settle()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    clock.advance(10)
    sched.notify_commit("b")
    await settle()
    assert log == []


async def test_update_timeout_deadline_starts_when_wait_is_called() -> None:
    log: list[tuple[Any, ...]] = []
    clock = FakeClock()
    sched = AsyncScheduler(FakeWorkspace(log), FakeRecorder(log), clock=clock)
    sched.register("a", (0,))
    waiting = asyncio.create_task(sched.wait_for_update("a", 5))
    await asyncio.sleep(0)  # The timer task has been created but has not run yet.
    clock.advance(5)
    assert await asyncio.wait_for(waiting, 2) is False


async def test_commit_after_deadline_cannot_win_before_timer_task_resumes() -> None:
    log: list[tuple[Any, ...]] = []
    clock = FakeClock()
    sched = AsyncScheduler(FakeWorkspace(log), FakeRecorder(log), clock=clock)
    sched.register("a", (0,))
    sched.register("b", (1,))
    waiting = asyncio.create_task(sched.wait_for_update("a", 5))
    await settle()
    clock.advance(6)
    sched.notify_commit("b")
    assert await asyncio.wait_for(waiting, 2) is False


async def test_fake_clock_deadlines_and_cancelled_sleepers() -> None:
    clock = FakeClock()
    assert clock.now() == 0
    short = asyncio.create_task(clock.sleep(2))
    long = asyncio.create_task(clock.sleep(3))
    cancelled = asyncio.create_task(clock.sleep(1))
    await settle()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    clock.advance(1.5)
    await settle()
    assert not short.done() and not long.done()
    clock.advance(0.5)
    await settle()
    assert short.done() and not long.done()
    clock.advance(2)
    await asyncio.wait_for(asyncio.gather(short, long), 2)
    assert clock.now() == 4
    await clock.sleep(0)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)


async def test_system_clock_uses_monotonic_time() -> None:
    clock = SystemClock()
    before = clock.now()
    await clock.sleep(0)
    assert clock.now() >= before
