"""Budget tests pin binding limits, concurrent shares, and reserved work."""

from __future__ import annotations

from dataclasses import replace

import pytest

from marli.errors import ConfigError
from marli.interact.limits import (
    AgentLimits,
    Allocation,
    CallLimits,
    ContextLimits,
    EpisodeLimits,
    Ledger,
    Limits,
    SessionLimits,
)
from marli.interact.types import Purpose


def limits() -> Limits:
    return Limits(
        call=CallLimits(1000, 1),
        agent=AgentLimits(100, 4, 10),
        worker=AgentLimits(40, 3, 5),
        session=SessionLimits(2, 60, 10, 10),
        episode=EpisodeLimits(1000, 4, 20),
        ctx=ContextLimits(2000),
    )


@pytest.mark.parametrize(
    ("block", "field", "value"),
    [
        (None, "on_exhaust", "wrong"),
        (None, "on_no_tool_call", "wrong"),
        (None, "max_nudges", 0),
        (None, "tool_output_chars", 0),
        ("call", "max_tokens", -1),
        ("call", "min_call_tokens", True),
        ("agent", "max_calls", 0),
        ("agent", "max_gen_tokens", 0),
        ("agent", "final_reserve", 100),
        ("worker", "final_reserve", 40),
        ("session", "max_sessions", 0),
        ("session", "carry_reserve", 60),
        ("session", "carry_max_tokens", 0),
        ("episode", "max_ticks", 0),
        ("episode", "max_wall_s", float("inf")),
        ("episode", "max_wall_s", float("nan")),
        ("spawn", "max_depth", 2),
        ("spawn", "max_total", 1.5),
        ("ctx", "max_ctx", 0),
    ],
)
def test_validation_names_field(block: str | None, field: str, value: object) -> None:
    cfg = limits()
    target = cfg if block is None else getattr(cfg, block)
    setattr(target, field, value)
    with pytest.raises(ConfigError, match=field):
        cfg.__post_init__()


def alloc(
    ledger: Ledger,
    agent: str = "a",
    *,
    purpose: Purpose = Purpose.ACT,
    prompt: int = 0,
    tick: int | None = 0,
    active: int = 1,
) -> Allocation:
    return ledger.allocate(agent, prompt_len=prompt, purpose=purpose, tick=tick, n_active=active)


def test_agent_session_context_and_call_caps() -> None:
    cfg = limits()
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    assert alloc(ledger).max_tokens == 50  # session less its carry reserve
    assert alloc(ledger, prompt=1990).max_tokens == 10
    ledger.charge("a", gen_tokens=50, purpose=Purpose.ACT)
    assert alloc(ledger).exhausted == "session.max_gen_tokens"
    assert alloc(ledger, purpose=Purpose.CARRY).max_tokens == 10
    ledger.charge("a", gen_tokens=10, purpose=Purpose.CARRY)
    ledger.start_session("a")
    assert alloc(ledger).max_tokens == 30  # total agent budget still applies
    ledger.charge("a", gen_tokens=30, purpose=Purpose.ACT)
    assert alloc(ledger).exhausted == "agent.max_gen_tokens"
    assert alloc(ledger, purpose=Purpose.FINAL).max_tokens == 10
    assert alloc(ledger, purpose=Purpose.FINAL, prompt=1999).max_tokens == 1
    assert alloc(ledger, purpose=Purpose.FINAL, prompt=2000).exhausted == "ctx.max_ctx"
    cfg.call.max_tokens = 3
    assert alloc(ledger, purpose=Purpose.FINAL).max_tokens == 3
    assert ledger.limits_hit() == {
        "a": ("session.max_gen_tokens", "agent.max_gen_tokens", "ctx.max_ctx"),
    }


def test_call_and_tick_exhaustion_allow_final_reserve() -> None:
    cfg = limits()
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    for _ in range(4):
        ledger.charge("a", gen_tokens=1, purpose=Purpose.ACT)
    assert alloc(ledger).exhausted == "agent.max_calls"
    assert alloc(ledger, purpose=Purpose.FINAL).max_tokens == 10
    other = Ledger(cfg)
    other.register("b", kind="worker")
    assert alloc(other, "b", tick=4).exhausted == "episode.max_ticks"
    assert alloc(other, "b", tick=4, purpose=Purpose.REPORT).max_tokens == 5
    assert other.limits_hit() == {"_episode": ("episode.max_ticks",)}


def test_lockstep_share_is_fixed_at_tick_start() -> None:
    cfg = limits()
    cfg.episode.max_gen_tokens = 61
    ledger = Ledger(cfg)
    for agent in ("a", "b"):
        ledger.register(agent, kind="agent")
    assert alloc(ledger, active=2).max_tokens == 20
    ledger.charge("a", gen_tokens=20, purpose=Purpose.ACT)
    assert alloc(ledger, "b", active=2).max_tokens == 20
    ledger.charge("b", gen_tokens=20, purpose=Purpose.ACT)
    assert alloc(ledger, "b", tick=1, active=2).exhausted == "episode.max_gen_tokens"
    assert alloc(ledger, "a", tick=1, active=2).exhausted == "episode.max_gen_tokens"
    assert alloc(ledger, "a", tick=1, active=2, purpose=Purpose.FINAL).max_tokens == 10


def test_async_shares_are_fixed_at_registration() -> None:
    cfg = limits()
    cfg.episode.max_gen_tokens = 61
    ledger = Ledger(cfg, schedule="async", expected_agents=2)
    ledger.register("a", kind="agent")
    assert alloc(ledger, tick=None).max_tokens == 20
    ledger.charge("a", gen_tokens=20, purpose=Purpose.ACT)
    ledger.register("b", kind="agent")
    assert alloc(ledger, "b", tick=None).max_tokens == 20
    assert alloc(ledger, tick=None).exhausted == "episode.max_gen_tokens"
    assert ledger.limits_hit() == {"_episode": ("episode.max_gen_tokens",)}


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_worker_reservations_and_unused_release(schedule: str) -> None:
    cfg = limits()
    cfg.episode.max_gen_tokens = 100
    cfg.session = replace(cfg.session, max_sessions=1)
    ledger = Ledger(cfg, schedule=schedule)
    ledger.register("a", kind="agent")
    assert ledger.reserve_workers("a", 3) == 2
    assert alloc(ledger).max_tokens == 10
    for worker in ("w0", "w1"):
        ledger.register(worker, kind="worker", parent="a")
        assert alloc(ledger, worker).max_tokens == 35
    assert ledger.reserve_workers("w0", 1) == 0
    ledger.charge("w0", gen_tokens=12, purpose=Purpose.ACT)
    ledger.release_worker("w0")
    ledger.release_worker("w0")  # no double refund
    ledger.done("w0")
    assert alloc(ledger, tick=1).max_tokens == 38
    ledger.release_worker("w1")
    assert alloc(ledger, tick=2).max_tokens == 78
    assert ledger.n_active == 2


def test_minimum_allocation_records_the_binding_limit() -> None:
    cfg = limits()
    cfg.call.min_call_tokens = 16
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    assert alloc(ledger, prompt=1990) == Allocation(0, "ctx.max_ctx")
    assert alloc(ledger, prompt=1990) == Allocation(0, "ctx.max_ctx")
    assert ledger.limits_hit()["a"] == ("ctx.max_ctx",)


def test_registration_requires_worker_reservation() -> None:
    ledger = Ledger(limits())
    ledger.register("a", kind="agent")
    with pytest.raises(ConfigError, match="reservation"):
        ledger.register("w", kind="worker", parent="a")


def test_spawn_limits_are_recorded() -> None:
    cfg = limits()
    cfg.spawn.max_per_call = 2
    cfg.spawn.max_total = 3
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    assert ledger.reserve_workers("a", 3) == 2
    assert ledger.reserve_workers("a", 2) == 1
    assert ledger.limits_hit() == {"a": ("spawn.max_per_call", "spawn.max_total")}


def test_lockstep_spawn_cannot_reserve_another_seats_inflight_allocation() -> None:
    cfg = limits()
    cfg.episode.max_gen_tokens = 100
    cfg.session.max_sessions = 1
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    ledger.register("b", kind="agent")
    assert alloc(ledger, "a", active=2).max_tokens == 40
    assert alloc(ledger, "b", active=2).max_tokens == 40
    ledger.charge("a", gen_tokens=10, purpose=Purpose.ACT)
    assert ledger.reserve_workers("a", 2) == 1
    assert ledger.reserve_workers("a", 1) == 0
    ledger.register("w", kind="worker", parent="a")
    ledger.charge("b", gen_tokens=40, purpose=Purpose.ACT)
    assert alloc(ledger, "w", tick=1, active=1).max_tokens == 35
    ledger.charge("w", gen_tokens=35, purpose=Purpose.ACT)
    assert alloc(ledger, "w", tick=2, active=1, purpose=Purpose.REPORT).max_tokens == 5


@pytest.mark.parametrize("kind", ["agent", "worker"])
def test_final_reserve_must_cover_min_call_tokens(kind: str) -> None:
    with pytest.raises(ConfigError, match=rf"{kind}.final_reserve.*min_call_tokens"):
        Limits(**{kind: AgentLimits(final_reserve=8)}, call=CallLimits(min_call_tokens=16))
    Limits(**{kind: AgentLimits(final_reserve=16)}, call=CallLimits(min_call_tokens=16))
    Limits(**{kind: AgentLimits(final_reserve=0)}, on_exhaust="none")


@pytest.mark.parametrize(
    "purpose", [Purpose.ACT, Purpose.COMPACT, Purpose.CARRY, Purpose.FINAL, Purpose.REPORT]
)
def test_ctx_reserve_applies_only_to_non_final_allocations(purpose: Purpose) -> None:
    ledger = Ledger(limits())
    ledger.register("a", kind="agent")
    allocation = ledger.allocate(
        "a", prompt_len=1990, purpose=purpose, tick=0, n_active=1, ctx_reserve=8
    )
    assert allocation.max_tokens == (10 if purpose in {Purpose.FINAL, Purpose.REPORT} else 2)


def test_reserve_workers_for_parent_registered_after_tick_snapshot() -> None:
    cfg = limits()
    cfg.episode.max_gen_tokens = 100
    ledger = Ledger(cfg)
    ledger.register("a", kind="agent")
    ledger.register("b", kind="agent")
    alloc(ledger, active=2)
    ledger.register("later", kind="agent")
    assert ledger.reserve_workers("later", 1) == 1
    assert ledger.reserve_workers("later", 1) == 0
    ledger.register("w", kind="worker", parent="later")
    assert alloc(ledger, "w").max_tokens == 35


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_episode_reserve_and_same_ticket_calls_cannot_spend_another_seats_share(
    schedule: str,
) -> None:
    cfg = limits()
    cfg.session.max_sessions = 1
    cfg.episode.max_gen_tokens = 100
    ledger = Ledger(cfg, schedule=schedule, expected_agents=2)
    for agent in ("a", "b"):
        ledger.register(agent, kind="agent")
    assert alloc(ledger, active=2).max_tokens == 40
    ledger.charge("a", gen_tokens=15, purpose=Purpose.COMPACT)
    assert alloc(ledger, active=2).max_tokens == 25
    ledger.charge("a", gen_tokens=25, purpose=Purpose.ACT)
    assert alloc(ledger, active=2).exhausted == "episode.max_gen_tokens"
    assert alloc(ledger, purpose=Purpose.FINAL, active=2).max_tokens == 10
    assert alloc(ledger, "b", active=2).max_tokens == 40
