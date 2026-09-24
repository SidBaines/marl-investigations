"""Hand-computed ablations pin the population and denominator at each stage."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import sqrt
from random import Random

import pytest

from marli.errors import ConfigError
from marli.interact.system import ContextSpec, RoleSpec
from marli.interact.types import (
    AgentInfo,
    Call,
    Episode,
    Outcome,
    Purpose,
    SegmentInfo,
    SegmentStart,
    Termination,
    Usage,
)
from marli.render.fake import FakeRenderer
from marli.train.credit import (
    CreditContext,
    _aux_weight,
    _segment_weights,
    assign_credit,
    validate_credit,
)
from marli.train.types import AuxReward, CreditConfig, CreditStats, SegmentCredit


@dataclass(frozen=True)
class Seat:
    agent_id: str = "a"
    role: str = "peer"
    tokens: tuple[int, ...] = (4,)
    sessions: tuple[int, ...] = ()
    own: float | None = None
    submission: str | None = "answer"
    parent: str | None = None


def episode(
    idx: int,
    team: float,
    seats: Sequence[Seat] = (Seat(),),
    *,
    group: str = "g",
    final_answer: str | None = "answer",
    ok: bool = True,
) -> Episode:
    renderer = FakeRenderer()
    episode_id = f"{group}/e{idx}"
    agents: list[AgentInfo] = []
    segments: list[SegmentInfo] = []
    calls: list[Call] = []
    grades = {"_system": {"correct": team}}
    for position, seat in enumerate(seats):
        agents.append(
            AgentInfo(seat.agent_id, seat.role, "recorded-policy", seat.parent, (position,))
        )
        if seat.own is not None:
            grades[seat.agent_id] = {"correct": seat.own}
        for index, n_tokens in enumerate(seat.tokens):
            session_idx = seat.sessions[index] if seat.sessions else 0
            segment_id = f"{seat.agent_id}/g{index}"
            ids = tuple(renderer.encode_completion("x" * (n_tokens - 1))) if n_tokens else ()
            assert len(ids) == n_tokens
            reason = SegmentStart.START
            if index:
                reason = SegmentStart.SESSION if session_idx else SegmentStart.COMPACTION
            segments.append(
                SegmentInfo(
                    segment_id,
                    seat.agent_id,
                    session_idx,
                    reason,
                    None,
                    renderer.name,
                    renderer.tokenizer_sha,
                    n_tokens + 1,
                )
            )
            calls.append(
                Call(
                    call_id=f"{episode_id}/{seat.agent_id}/c{index}",
                    episode_id=episode_id,
                    agent_id=seat.agent_id,
                    role=seat.role,
                    policy_id="recorded-policy",
                    policy_version=7,
                    segment_id=segment_id,
                    prompt_len=1,
                    completion_ids=ids,
                    logprobs=(-0.2,) * n_tokens,
                    text="not used for credit",
                    termination=Termination.STOP,
                    purpose=Purpose.ACT,
                    forced=False,
                    tool_calls=(),
                    reads=(),
                    tick=index,
                    seq=len(calls),
                    session_idx=session_idx,
                    seed=index,
                    usage=Usage(completion_tokens=9999),
                )
            )
    return Episode(
        episode_id=episode_id,
        group_id=group,
        episode_idx=idx,
        task_id="task",
        protocol="test",
        config_hash="hash",
        backend="fake",
        agents=tuple(agents),
        segments=tuple(segments),
        calls=tuple(calls),
        events=(),
        workspace_log=(),
        outcome=Outcome(final_answer, {seat.agent_id: seat.submission for seat in seats}, "test"),
        grades=grades,
        limits_hit={},
        metrics={},
        replayable=True,
        ok=ok,
    )


def context(*roles: RoleSpec, seating: dict[str, str] | None = None) -> CreditContext:
    roles = roles or (RoleSpec("peer", ("submit",), "", count=3),)
    return CreditContext(
        {role.role: role for role in roles},
        seating if seating is not None else {role.role: "learner:shared" for role in roles},
        protocol_hash="protocol-hash",
    )


def advantages(credits: Sequence[SegmentCredit]) -> list[float]:
    return [credit.advantage for credit in credits]


def coordinator() -> tuple[list[Episode], CreditContext]:
    ctx = context(
        RoleSpec("coordinator", ("submit", "spawn_workers"), ""),
        RoleSpec("worker", ("return_report",), "", count=None),
        seating={"coordinator": "learner:coord", "worker": "learner:work"},
    )
    episodes = [
        episode(
            idx,
            team,
            (
                Seat("c", "coordinator", coordinator_tokens),
                Seat("w0", "worker", (worker_tokens[0],), parent="c"),
                Seat("w1", "worker", (worker_tokens[1],), parent="c"),
            ),
        )
        for idx, (team, coordinator_tokens, worker_tokens) in enumerate(
            (
                (1, (300,), (200, 150)),
                (0, (250,), (200, 200)),
                (1, (100, 200), (150, 200)),
            ),
            start=1,
        )
    ]
    return episodes, ctx


@pytest.mark.parametrize("loss_agg", ["token_sum", "token_mean_per_learner"])
def test_g1_coordinator_role_baseline_and_exact_tokens(loss_agg: str) -> None:
    episodes, ctx = coordinator()
    cfg = CreditConfig(loss_agg=loss_agg)
    assert validate_credit(cfg, ctx, group_size=3) == []
    credits, stats, state = assign_credit(episodes, cfg, ctx)
    assert len(credits) == 10
    for credit in credits:
        expected = -1.0 if credit.episode_id == "g/e2" else 0.5
        if loss_agg == "token_mean_per_learner":
            expected /= 850 if credit.learner == "coord" else 1100
        assert credit.advantage == pytest.approx(expected, abs=1e-12)
        assert credit.reward == (0 if credit.episode_id == "g/e2" else 1)
        assert credit.weight == 1.0
    assert stats.n_episodes == 3
    assert stats.n_groups == 1
    assert stats.action_tokens == {"coord": 850, "work": 1100}
    assert stats.role_token_share == {"coord": {"coordinator": 1.0}, "work": {"worker": 1.0}}
    assert stats.reward_mean == {"coordinator": 2 / 3, "worker": 2 / 3}
    assert stats.advantage_abs_mean == pytest.approx(
        {
            "coordinator": 0.625 / (850 if loss_agg == "token_mean_per_learner" else 1),
            "worker": (2 / 3) / (1100 if loss_agg == "token_mean_per_learner" else 1),
        }
    )
    assert state == {}


@pytest.mark.parametrize("unit,coord_tokens,n_credits", [("segment", 750, 9), ("session", 850, 10)])
def test_g1_last_segment_vs_last_session(unit: str, coord_tokens: int, n_credits: int) -> None:
    episodes, ctx = coordinator()
    cfg = CreditConfig(segment_credit="last", segment_unit=unit)
    credits, stats, _ = assign_credit(episodes, cfg, ctx)
    assert len(credits) == n_credits
    third = [credit for credit in credits if credit.episode_id == "g/e3" and credit.agent_id == "c"]
    assert [credit.segment_id for credit in third] == (
        ["c/g1"] if unit == "segment" else ["c/g0", "c/g1"]
    )
    assert advantages(third) == [0.5] * len(third)
    assert stats.action_tokens == {"coord": coord_tokens, "work": 1100}
    scaled, _, _ = assign_credit(episodes, replace(cfg, loss_agg="token_mean_per_learner"), ctx)
    for credit in scaled:
        denominator = coord_tokens if credit.learner == "coord" else 1100
        assert credit.advantage == pytest.approx(
            (-1 if credit.episode_id == "g/e2" else 0.5) / denominator
        )


def test_g1_coordinator_only_and_idle_learner_opt_in() -> None:
    episodes, ctx = coordinator()
    cfg = CreditConfig(recipients=("coordinator",))
    with pytest.raises(ConfigError, match="idle.*work"):
        validate_credit(cfg, ctx, group_size=3)
    assert validate_credit(cfg, ctx, group_size=3, allow_idle=True) == []
    credits, stats, _ = assign_credit(episodes, cfg, ctx)
    assert len(credits) == 4
    assert {credit.role for credit in credits} == {"coordinator"}
    assert advantages(credits) == [0.5, -1, 0.5, 0.5]
    assert stats.action_tokens == {"coord": 850}
    assert stats.reward_mean["worker"] == 2 / 3


@pytest.mark.parametrize(
    "target,expected",
    [
        ("individual", [2 / 3, -1 / 3, 2 / 3, -2 / 3, -2 / 3, 1 / 3]),
        ("team", [1, 1, 1, -1, -1, -1]),
        ("mix:0.5", [5 / 6, 1 / 3, 5 / 6, -5 / 6, -5 / 6, -1 / 3]),
    ],
)
def test_g2_swarm(target: str, expected: list[float]) -> None:
    episodes = [
        episode(idx, team, [Seat(f"p{a}", own=own) for a, own in enumerate(owns)])
        for idx, (team, owns) in enumerate([(1, [1, 0, 1]), (0, [0, 0, 1])])
    ]
    cfg = CreditConfig(default_target=target)
    assert validate_credit(cfg, context(), group_size=2) == []
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "mode,normalize,weights",
    [
        ("all", "none", [1, 1, 1]),
        ("last", "none", [0, 0, 1]),
        ("geometric", "none", [0.25, 0.5, 1]),
        ("geometric", "sum_to_one", [1 / 7, 2 / 7, 4 / 7]),
        ("all", "sum_to_one", [1 / 3, 1 / 3, 1 / 3]),
        ("last", "sum_to_one", [0, 0, 1]),
    ],
)
def test_g3_multi_session(mode: str, normalize: str, weights: list[float]) -> None:
    seats = [Seat(tokens=(3, 4, 5), sessions=(0, 1, 2))]
    episodes = [episode(0, 1, seats), episode(1, 0, seats)]
    ctx = context(RoleSpec("peer", ("submit", "end_session"), "", context=ContextSpec("notes")))
    cfg = CreditConfig(segment_credit=mode, segment_normalize=normalize)
    assert validate_credit(cfg, ctx, group_size=2) == []
    assert _segment_weights(episodes[0].segments, cfg) == pytest.approx(weights, abs=1e-12)
    credits, _, _ = assign_credit(episodes, cfg, ctx)
    nonzero = [weight for weight in weights if weight > 0]
    assert advantages(credits) == pytest.approx(
        nonzero + [-weight for weight in nonzero], abs=1e-12
    )
    assert [credit.weight for credit in credits] == pytest.approx(nonzero * 2, abs=1e-12)
    if mode == "last":
        assert [credit.segment_id for credit in credits] == ["a/g2", "a/g2"]


def test_session_weights_normalize_over_sessions_not_compaction_segments() -> None:
    ep = episode(0, 1, [Seat(tokens=(2, 3, 4, 5), sessions=(0, 0, 1, 2))])
    cfg = CreditConfig(segment_credit="geometric", segment_normalize="sum_to_one")
    assert _segment_weights(ep.segments, cfg) == pytest.approx([1 / 7, 1 / 7, 2 / 7, 4 / 7])
    assert _segment_weights([], cfg) == []


def test_episode_baseline_is_leave_one_out_over_four_episodes() -> None:
    episodes = [episode(i, reward) for i, reward in enumerate([1, 0, 1, 1])]
    cfg = CreditConfig(baseline="episode")
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx([1 / 3, -1, 1 / 3, 1 / 3], abs=1e-12)


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("episode", [-0.5, 1, 1, 1, -0.5]),
        ("instance", [-0.75, 1, 1, 1, -0.75]),
    ],
)
def test_role_counts_change_instance_weighting(unit: str, expected: list[float]) -> None:
    episodes = [
        episode(0, 0),
        episode(1, 1, [Seat("a"), Seat("b"), Seat("c")]),
        episode(2, 0),
    ]
    credits, _, _ = assign_credit(episodes, CreditConfig(baseline_unit=unit), context())
    assert advantages(credits) == expected


def test_no_baseline_counts_role_episode_once_and_zeros_all_its_instances() -> None:
    episodes = [
        episode(0, 1, [Seat(), Seat("w0", "worker"), Seat("w1", "worker")]),
        episode(1, 0),
    ]
    ctx = context(RoleSpec("peer", ("submit",), ""), RoleSpec("worker", (), "", count=None))
    credits, stats, _ = assign_credit(episodes, CreditConfig(), ctx)
    assert advantages(credits) == [1, 0, 0, -1]
    assert stats.no_baseline == 1
    assert stats.zero_variance_groups == 0


def test_rae_first_sight_then_pre_update_use_across_groups_and_calls() -> None:
    cfg = CreditConfig(baseline="rae", rae_gamma=0.75)
    ctx = context()
    episodes = [episode(0, 1), episode(1, 0), episode(0, 3, group="h"), episode(1, 0, group="h")]
    original = {"unrelated/peer": 42.0}
    credits, _, state = assign_credit(episodes, cfg, ctx, rae_state=original)
    assert advantages(credits) == [0, -1, 2, -1]
    assert state == {"unrelated/peer": 42.0, "protocol-hash/peer": 1.0}
    assert original == {"unrelated/peer": 42.0}
    next_credits, _, next_state = assign_credit(
        [episode(0, 2), episode(1, 4)], cfg, ctx, rae_state=state
    )
    assert advantages(next_credits) == [1, 3]
    assert next_state["protocol-hash/peer"] == 1.5
    assert state["protocol-hash/peer"] == 1.0


def test_rae_uses_all_role_instances_and_updates_even_zero_variance_groups() -> None:
    cfg = CreditConfig(baseline="rae")
    episodes = [episode(0, 1, [Seat("a"), Seat("b")]), episode(1, 0)]
    credits, _, state = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx([1 / 3, 1 / 3, -2 / 3])
    assert state["protocol-hash/peer"] == 2 / 3
    credits, stats, state = assign_credit([episode(0, 1), episode(1, 1)], cfg, context())
    assert credits == []
    assert stats.zero_variance_groups == 1
    assert state == {"protocol-hash/peer": 1}


@pytest.mark.parametrize("baseline", ["episode", "role", "none"])
def test_mean_std_population_rewards_include_focal_episode(baseline: str) -> None:
    cfg = CreditConfig(baseline=baseline, norm="mean_std", std_eps=0.1)
    credits, _, _ = assign_credit([episode(0, 1), episode(1, 0)], cfg, context())
    assert advantages(credits) == pytest.approx([1 / 0.6, (0 if baseline == "none" else -1) / 0.6])


@pytest.mark.parametrize(
    "unit,std,raw",
    [
        ("episode", sqrt(2 / 9), [-0.5, 1, 1, 1, -0.5]),
        ("instance", sqrt(0.24), [-0.75, 1, 1, 1, -0.75]),
    ],
)
def test_mean_std_respects_role_baseline_population(
    unit: str, std: float, raw: list[float]
) -> None:
    episodes = [episode(0, 0), episode(1, 1, [Seat("a"), Seat("b"), Seat("c")]), episode(2, 0)]
    cfg = CreditConfig(baseline_unit=unit, norm="mean_std", std_eps=0.1)
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx([value / (std + 0.1) for value in raw])


def test_per_learner_normalizes_once_per_emitting_agent_across_roles_and_groups() -> None:
    ctx = context(
        RoleSpec("peer", ("submit",), ""),
        RoleSpec("worker", ("submit",), ""),
        RoleSpec("other", ("submit",), ""),
        seating={"peer": "learner:shared", "worker": "learner:shared", "other": "learner:other"},
    )
    episodes = [
        episode(
            0,
            0,
            [Seat(own=1, tokens=(2, 8)), Seat("w", "worker", own=3), Seat("o", "other", own=2)],
        ),
        episode(1, 0, [Seat(own=0), Seat("w", "worker", own=0), Seat("o", "other", own=0)]),
        episode(0, 0, [Seat(own=2)], group="h"),
        episode(1, 0, [Seat(own=0)], group="h"),
    ]
    cfg = CreditConfig(default_target="individual", norm="per_learner", std_eps=0.1)
    assert validate_credit(cfg, ctx, group_size=2) == [
        "learner 'other' serves a single role: per_learner scales by RMS; "
        "use norm=mean_std for reward-standard-deviation scaling"
    ]
    credits, _, _ = assign_credit(episodes, cfg, ctx)
    shared_std = sqrt(14 / 3)
    assert advantages(credits) == pytest.approx(
        [
            1 / (shared_std + 0.1),
            1 / (shared_std + 0.1),
            3 / (shared_std + 0.1),
            2 / 2.1,
            -1 / (shared_std + 0.1),
            -3 / (shared_std + 0.1),
            -2 / 2.1,
            2 / (shared_std + 0.1),
            -2 / (shared_std + 0.1),
        ]
    )


def test_per_learner_population_excludes_masked_nonrecipient_frozen_and_empty_agents() -> None:
    ctx = context(
        RoleSpec("peer", ("submit",), ""),
        RoleSpec("ignored", ("submit",), ""),
        RoleSpec("frozen", ("submit",), ""),
        seating={"peer": "learner:shared", "ignored": "learner:shared", "frozen": "api:fake"},
    )
    episodes = [
        episode(
            0,
            0,
            [
                Seat("kept", own=0),
                Seat("masked", own=100, submission=None),
                Seat("empty", own=1000, tokens=(0,)),
                Seat("ignored", "ignored", own=10000),
                Seat("frozen", "frozen", own=100000),
            ],
        ),
        episode(1, 0, [Seat("kept", own=2)]),
    ]
    cfg = CreditConfig(
        baseline="none",
        default_target="individual",
        norm="per_learner",
        std_eps=0.1,
        recipients=("peer",),
        overlong="mask_no_answer",
        overlong_scope="agent",
    )
    credits, _, _ = assign_credit(episodes, cfg, ctx)
    assert [credit.agent_id for credit in credits] == ["kept", "kept"]
    assert advantages(credits) == pytest.approx([0, 2 / (sqrt(2) + 0.1)])


def test_zero_variance_drop_is_after_baselines_before_normalization() -> None:
    episodes = [episode(0, 1), episode(1, 1)]
    cfg = CreditConfig()
    credits, stats, _ = assign_credit(episodes, cfg, context())
    assert credits == []
    assert asdict(stats) == {
        **asdict(
            CreditStats(n_episodes=2, n_groups=1, zero_variance_groups=1, reward_mean={"peer": 1})
        ),
        "norm_skipped": 0,
    }
    credits, stats, _ = assign_credit(episodes, replace(cfg, drop_zero_variance=False), context())
    assert advantages(credits) == [0, 0]
    assert stats.action_tokens == {"shared": 8}
    assert stats.zero_variance_groups == 0
    credits, _, _ = assign_credit(episodes, replace(cfg, baseline="none"), context())
    assert advantages(credits) == [1, 1]


def test_failed_episodes_drop_before_small_groups_and_do_not_affect_rewards_or_rae() -> None:
    bad = replace(episode(2, 100, ok=False), grades={})
    episodes = [episode(0, 1), episode(1, 0), bad, episode(0, 100, group="small")]
    cfg = CreditConfig(baseline="rae")
    credits, stats, state = assign_credit(episodes, cfg, context())
    assert advantages(credits) == [0.5, -0.5]
    assert stats.n_episodes == 4
    assert stats.dropped_not_ok == 1
    assert stats.dropped_small_groups == 1
    assert stats.n_groups == 1
    assert stats.reward_mean == {"peer": 0.5}
    assert state == {"protocol-hash/peer": 0.5}


def test_min_group_is_configurable_and_small_groups_count_episodes() -> None:
    cfg = CreditConfig(baseline="none", min_group=3)
    credits, stats, _ = assign_credit([episode(0, 1), episode(1, 0)], cfg, context())
    assert credits == []
    assert stats.dropped_small_groups == 2
    assert stats.n_groups == 0
    credits, _, _ = assign_credit([episode(0, 2)], replace(cfg, min_group=1), context())
    assert advantages(credits) == [2]


def test_empty_input_preserves_rae_state_without_aliasing() -> None:
    state = {"protocol-hash/peer": 0.5}
    credits, stats, new_state = assign_credit([], CreditConfig(), context(), rae_state=state)
    assert credits == []
    assert asdict(stats) == {**asdict(CreditStats()), "norm_skipped": 0}
    assert new_state == state
    assert new_state is not state


@pytest.mark.parametrize(
    "scope,expected_ids,masked",
    [
        ("episode", [("g/e1", "a"), ("g/e1", "b"), ("g/e1", "w")], 3),
        ("agent", [("g/e0", "a"), ("g/e1", "a"), ("g/e1", "b"), ("g/e1", "w")], 2),
    ],
)
def test_mask_no_answer_scopes_and_nonsubmitting_role_follows_episode(
    scope: str,
    expected_ids: list[tuple[str, str]],
    masked: int,
) -> None:
    ctx = context(RoleSpec("peer", ("submit",), ""), RoleSpec("worker", (), ""))
    episodes = [
        episode(
            0,
            1,
            [Seat("a"), Seat("b", submission=None), Seat("w", "worker", submission=None)],
            final_answer=None,
        ),
        episode(1, 0, [Seat("a"), Seat("b"), Seat("w", "worker", submission=None)]),
    ]
    cfg = CreditConfig(overlong="mask_no_answer", overlong_scope=scope)
    credits, stats, _ = assign_credit(episodes, cfg, ctx)
    assert [(credit.episode_id, credit.agent_id) for credit in credits] == expected_ids
    assert advantages(credits) == [1 if ep == "g/e0" else -1 for ep, _ in expected_ids]
    assert stats.masked_overlong == masked
    assert stats.reward_mean == {"peer": 0.5, "worker": 0.5}
    assert stats.no_baseline == 0


@pytest.mark.parametrize(
    "scope,kept,masked", [("episode", ["a", "b", "w"], 3), ("agent", ["a", "w", "a", "b", "w"], 1)]
)
@pytest.mark.parametrize("purpose", [Purpose.FINAL, Purpose.REPORT])
def test_mask_forced_scopes_include_final_and_report(
    scope: str, kept: list[str], masked: int, purpose: Purpose
) -> None:
    ctx = context(RoleSpec("peer", ("submit",), ""), RoleSpec("worker", (), ""))
    seats = [Seat("a"), Seat("b"), Seat("w", "worker")]
    first = episode(0, 1, seats)
    first = replace(
        first,
        calls=tuple(
            replace(call, forced=True, purpose=purpose) if call.agent_id == "b" else call
            for call in first.calls
        ),
    )
    episodes = [first, episode(1, 0, seats)]
    cfg = CreditConfig(overlong="mask_forced", overlong_scope=scope)
    credits, stats, _ = assign_credit(episodes, cfg, ctx)
    assert [credit.agent_id for credit in credits] == kept
    assert advantages(credits) == ([1, 1, -1, -1, -1] if scope == "agent" else [-1, -1, -1])
    assert stats.masked_overlong == masked
    assert stats.no_baseline == 0
    all_credits, unmasked, _ = assign_credit(episodes, replace(cfg, overlong="none"), ctx)
    assert len(all_credits) == 6
    assert unmasked.masked_overlong == 0


def test_masked_agent_reward_stays_in_role_baseline_and_mean_std() -> None:
    episodes = [
        episode(0, 0, [Seat("a", own=1), Seat("b", own=3, submission=None)]),
        episode(1, 0, [Seat("a", own=0), Seat("b", own=0)]),
    ]
    cfg = CreditConfig(
        default_target="individual",
        overlong="mask_no_answer",
        overlong_scope="agent",
        norm="mean_std",
        std_eps=0.1,
    )
    credits, stats, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx([a / (sqrt(1.5) + 0.1) for a in (1, -2, -2)])
    assert stats.reward_mean == {"peer": 1}
    assert stats.masked_overlong == 1


def test_masked_agents_still_update_rae() -> None:
    episodes = [episode(0, 3, final_answer=None), episode(1, 1)]
    cfg = CreditConfig(baseline="rae", overlong="mask_no_answer", rae_gamma=0.5)
    credits, stats, state = assign_credit(
        episodes, cfg, context(), rae_state={"protocol-hash/peer": 0}
    )
    assert advantages(credits) == [1]
    assert state == {"protocol-hash/peer": 1}
    assert stats.masked_overlong == 1


def test_nonrecipient_advantages_prevent_zero_variance_drop() -> None:
    ctx = context(
        RoleSpec("peer", ("submit",), ""),
        RoleSpec("observer", ("submit",), ""),
        seating={"peer": "learner:shared", "observer": "api:fake"},
    )
    episodes = [
        episode(0, 0, [Seat(own=0), Seat("o", "observer", own=1)]),
        episode(1, 0, [Seat(own=0), Seat("o", "observer", own=0)]),
    ]
    credits, stats, _ = assign_credit(episodes, CreditConfig(default_target="individual"), ctx)
    assert advantages(credits) == [0, 0]
    assert stats.zero_variance_groups == 0
    assert stats.reward_mean == {"observer": 0.5, "peer": 0}
    assert stats.action_tokens == {"shared": 8}


@pytest.mark.parametrize(
    "mode,tokens,coord_denominators",
    [
        ("all", 850, [900, 750, 900, 900]),
        ("last", 750, [900, 750, 600]),
    ],
)
def test_agent_mean_counts_agents_not_segments_and_only_emitted_tokens(
    mode: str,
    tokens: int,
    coord_denominators: list[int],
) -> None:
    episodes, ctx = coordinator()
    cfg = CreditConfig(loss_agg="agent_mean", segment_credit=mode, segment_unit="segment")
    credits, stats, _ = assign_credit(episodes, cfg, ctx)
    coords = [credit for credit in credits if credit.role == "coordinator"]
    workers = [credit for credit in credits if credit.role == "worker"]
    raw = [0.5, -1, 0.5, 0.5] if mode == "all" else [0.5, -1, 0.5]
    assert advantages(coords) == pytest.approx(
        [
            advantage / denominator
            for advantage, denominator in zip(raw, coord_denominators, strict=True)
        ]
    )
    assert advantages(workers) == pytest.approx(
        [0.5 / 1200, 0.5 / 900, -1 / 1200, -1 / 1200, 0.5 / 900, 0.5 / 1200]
    )
    assert stats.action_tokens == {"coord": tokens, "work": 1100}


def test_shared_learner_token_shares_and_loss_denominator() -> None:
    episodes, ctx = coordinator()
    ctx = replace(ctx, seating={role: "learner:shared" for role in ctx.roles})
    credits, stats, _ = assign_credit(
        episodes, CreditConfig(loss_agg="token_mean_per_learner"), ctx
    )
    assert stats.action_tokens == {"shared": 1950}
    assert stats.role_token_share["shared"] == pytest.approx(
        {"coordinator": 850 / 1950, "worker": 1100 / 1950}
    )
    assert advantages(credits) == pytest.approx(
        [value / 1950 for value in [0.5, 0.5, 0.5, -1, -1, -1, 0.5, 0.5, 0.5, 0.5]]
    )


def test_segments_require_own_nonempty_token_calls_and_count_every_sampled_purpose() -> None:
    ep = episode(0, 1, [Seat(tokens=(2, 0, 3, 0, 0))])
    calls = [replace(ep.calls[0], forced=True, purpose=Purpose.COMPACT), ep.calls[1], ep.calls[2]]
    calls.append(replace(ep.calls[2], call_id="carry", purpose=Purpose.CARRY))
    calls.append(replace(ep.calls[0], call_id="api", segment_id=""))
    calls.append(replace(ep.calls[2], call_id="wrong-owner", agent_id="other", segment_id="a/g3"))
    calls.append(replace(ep.calls[2], call_id="unknown-segment", segment_id="missing"))
    ep = replace(
        ep, calls=tuple(calls), segments=ep.segments + (replace(ep.segments[0], segment_id=""),)
    )
    cfg = CreditConfig(baseline="none", min_group=1, segment_credit="last", segment_unit="segment")
    credits, stats, _ = assign_credit([ep], cfg, context())
    assert [credit.segment_id for credit in credits] == ["a/g2"]
    assert advantages(credits) == [1]
    assert stats.action_tokens == {"shared": 6}
    credits, stats, _ = assign_credit([ep], replace(cfg, segment_credit="all"), context())
    assert [credit.segment_id for credit in credits] == ["a/g0", "a/g2"]
    assert stats.action_tokens == {"shared": 8}


@pytest.mark.parametrize("seat", ["api:fake", "tinker:frozen", "vllm:fake"])
def test_frozen_and_api_seats_never_emit_even_with_recorded_ids(seat: str) -> None:
    ctx = context(seating={"peer": seat})
    cfg = CreditConfig()
    assert validate_credit(cfg, ctx, group_size=2) == []
    credits, stats, _ = assign_credit([episode(0, 1), episode(1, 0)], cfg, ctx)
    assert credits == []
    assert stats.action_tokens == {}
    assert stats.reward_mean == {"peer": 0.5}


@pytest.mark.parametrize("step,weight", [(0, 1), (2, 1), (4, 2), (6, 3), (8, 3)])
def test_aux_linear_interpolation_clamps_endpoints(step: int, weight: float) -> None:
    aux = AuxReward(
        "submitted", weight=1, schedule="linear", end_weight=3, start_step=2, end_step=6
    )
    assert _aux_weight(aux, step) == weight
    cfg = CreditConfig(baseline="none", min_group=1, aux_rewards=[aux])
    credits, _, _ = assign_credit([episode(0, 0)], cfg, replace(context(), step=step))
    assert [credit.reward for credit in credits] == [weight]


@pytest.mark.parametrize("weight,expected", [(3, 4), (-3, -4), (0, 0)])
def test_aux_cap_applies_after_weighting_and_role_filter(weight: float, expected: float) -> None:
    episodes, ctx = coordinator()
    aux = AuxReward("workers_spawned", weight=weight, cap=4, roles=("coordinator",))
    assert _aux_weight(aux, 100) == weight
    cfg = CreditConfig(baseline="none", drop_zero_variance=False, aux_rewards=[aux])
    credits, _, _ = assign_credit(episodes, cfg, ctx)
    for credit in credits:
        team = 0 if credit.episode_id == "g/e2" else 1
        assert credit.reward == team + (expected if credit.role == "coordinator" else 0)
        assert credit.advantage == credit.reward


def test_aux_contributions_add_and_zero_cap_disables_contribution() -> None:
    cfg = CreditConfig(
        baseline="none",
        min_group=1,
        aux_rewards=[
            AuxReward("submitted", weight=2),
            AuxReward("format", weight=3),
            AuxReward("format", weight=9, cap=0),
        ],
    )
    credits, _, _ = assign_credit([episode(0, 0.25)], cfg, context())
    assert advantages(credits) == [5.25]


def test_aux_rewards_are_included_before_episode_baseline() -> None:
    episodes = [episode(0, 1), episode(1, 0)]
    cfg = CreditConfig(baseline="episode", aux_rewards=[AuxReward("submitted", weight=0.5)])
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert [credit.reward for credit in credits] == [1.5, 0.5]
    assert advantages(credits) == [1, -1]


def test_role_targets_override_default_and_missing_own_grade_is_zero() -> None:
    ctx = context(RoleSpec("peer", ("submit",), ""), RoleSpec("worker", (), ""))
    ep = episode(0, 0.5, [Seat("graded", own=0.25), Seat("ungraded"), Seat("w", "worker")])
    cfg = CreditConfig(
        baseline="none", min_group=1, default_target="individual", reward_target={"worker": "team"}
    )
    assert validate_credit(cfg, ctx, group_size=1) == []
    credits, _, _ = assign_credit([ep], cfg, ctx)
    assert [credit.reward for credit in credits] == [0.25, 0, 0.5]
    credits, _, _ = assign_credit([ep], replace(cfg, default_target="mix:0.2"), ctx)
    assert [credit.reward for credit in credits] == pytest.approx([0.3, 0.1, 0.5])


def test_reward_key_uses_selected_grade_component() -> None:
    ep = episode(0, 1, [Seat(own=1)])
    ep = replace(
        ep, grades={"_system": {"correct": 1, "partial": 0.8}, "a": {"correct": 1, "partial": 0.2}}
    )
    cfg = CreditConfig(baseline="none", min_group=1, reward_key="partial", default_target="mix:0.5")
    credits, _, _ = assign_credit([ep], cfg, context())
    assert advantages(credits) == [0.5]


@pytest.mark.parametrize("target", ["individual", "mix:0.5", "mix:1"])
def test_validate_rejects_episode_baseline_with_nonteam_target(target: str) -> None:
    cfg = CreditConfig(baseline="episode", reward_target={"peer": target})
    with pytest.raises(ConfigError, match="baseline=episode requires team.*peer"):
        validate_credit(cfg, context(), group_size=2)


@pytest.mark.parametrize("target", ["individual", "mix:0.5", "mix:1"])
def test_validate_rejects_own_target_without_submit_tool(target: str) -> None:
    cfg = CreditConfig(default_target=target)
    ctx = context(RoleSpec("worker", ("return_report",), "", count=None))
    with pytest.raises(ConfigError, match="worker.*no submission"):
        validate_credit(cfg, ctx, group_size=2)


def test_validate_rejects_oracle_any() -> None:
    with pytest.raises(ConfigError, match="oracle_any.*eval-only"):
        validate_credit(CreditConfig(reward_key="oracle_any"), context(), group_size=2)


def test_validate_rejects_unknown_recipient() -> None:
    with pytest.raises(ConfigError, match="recipient.*missing.*not a protocol role"):
        validate_credit(CreditConfig(recipients=("missing",)), context(), group_size=2)


@pytest.mark.parametrize("seat", ["api:fake", "tinker:frozen"])
def test_validate_rejects_frozen_or_api_recipient(seat: str) -> None:
    with pytest.raises(ConfigError, match="recipient.*peer.*frozen/API"):
        validate_credit(
            CreditConfig(recipients=("peer",)), context(seating={"peer": seat}), group_size=2
        )


def test_validate_rejects_missing_seating_even_for_nonrecipient_role() -> None:
    ctx = context(
        RoleSpec("peer", ("submit",), ""),
        RoleSpec("worker", (), ""),
        seating={"peer": "learner:shared"},
    )
    with pytest.raises(ConfigError, match="worker.*missing from seating"):
        validate_credit(CreditConfig(recipients=("peer",)), ctx, group_size=2)


@pytest.mark.parametrize("baseline", ["episode", "role"])
@pytest.mark.parametrize("size", [0, 1])
def test_validate_rejects_small_group_with_group_baseline(baseline: str, size: int) -> None:
    with pytest.raises(ConfigError, match=f"baseline={baseline} requires group_size >= 2"):
        validate_credit(CreditConfig(baseline=baseline), context(), group_size=size)


@pytest.mark.parametrize("baseline", ["none", "rae"])
def test_validate_allows_singleton_without_group_baseline(baseline: str) -> None:
    assert (
        validate_credit(CreditConfig(baseline=baseline, min_group=1), context(), group_size=1) == []
    )


def test_validate_rejects_unknown_aux_reward_or_role() -> None:
    with pytest.raises(ConfigError, match="unknown aux_rewards.*missing"):
        validate_credit(CreditConfig(aux_rewards=[AuxReward("missing")]), context(), group_size=2)
    with pytest.raises(ConfigError, match="aux reward.*submitted.*missing.*not a protocol role"):
        validate_credit(
            CreditConfig(aux_rewards=[AuxReward("submitted", roles=("missing",))]),
            context(),
            group_size=2,
        )


def test_validate_warns_when_segment_weights_cannot_have_an_effect() -> None:
    ctx = context(RoleSpec("peer", ("submit",), ""))
    for mode in ("last", "geometric"):
        warnings = validate_credit(CreditConfig(segment_credit=mode), ctx, group_size=2)
        assert len(warnings) == 1
        assert "U == 1" in warnings[0]
    assert validate_credit(CreditConfig(), ctx, group_size=2) == []


@pytest.mark.parametrize(
    "role",
    [
        RoleSpec("peer", ("submit",), "", context=ContextSpec("compaction")),
        RoleSpec("peer", ("submit", "end_session"), ""),
    ],
)
def test_validate_does_not_warn_when_multiple_units_are_possible(role: RoleSpec) -> None:
    assert validate_credit(CreditConfig(segment_credit="last"), context(role), group_size=2) == []


def test_validate_per_learner_warning_counts_roles_not_instances() -> None:
    warnings = validate_credit(CreditConfig(norm="per_learner"), context(), group_size=2)
    assert len(warnings) == 1
    assert "single role: per_learner scales by RMS" in warnings[0]
    ctx = context(RoleSpec("peer", ("submit",), ""), RoleSpec("worker", (), "", count=None))
    assert validate_credit(CreditConfig(norm="per_learner"), ctx, group_size=2) == []


def test_validate_idle_extra_learner_declared_in_seating() -> None:
    ctx = context(seating={"peer": "learner:shared", "unused": "learner:idle"})
    with pytest.raises(ConfigError, match="idle learners.*idle"):
        validate_credit(CreditConfig(), ctx, group_size=2)
    assert validate_credit(CreditConfig(), ctx, group_size=2, allow_idle=True) == []


def test_contract_rejects_mean_std_with_rae() -> None:
    with pytest.raises(ConfigError, match="mean_std.*incompatible.*rae"):
        CreditConfig(baseline="rae", norm="mean_std")


def test_output_order_is_group_episode_seat_segment_and_inputs_are_unchanged() -> None:
    seats = [Seat("z", tokens=(2, 3)), Seat("a", tokens=(4,))]
    eps = [
        episode(i, reward, seats, group=group)
        for group in ("b", "a")
        for i, reward in ((10, 0), (2, 1))
    ]
    eps = [replace(ep, agents=tuple(reversed(ep.agents))) for ep in eps]
    cfg = CreditConfig(
        segment_credit="geometric",
        segment_unit="segment",
        norm="per_learner",
        loss_agg="agent_mean",
    )
    ctx = context()
    state = {"unrelated": 4.0}
    before = deepcopy((eps, cfg, ctx, state))
    result = assign_credit(eps, cfg, ctx, rae_state=state)
    assert result == assign_credit(tuple(reversed(eps)), cfg, ctx, rae_state=state)
    assert (eps, cfg, ctx, state) == before
    assert [(credit.episode_id, credit.agent_id, credit.segment_id) for credit in result[0]] == [
        (f"{group}/e{idx}", agent, segment)
        for group in ("a", "b")
        for idx in (2, 10)
        for agent, segment in (("z", "z/g0"), ("z", "z/g1"), ("a", "a/g0"))
    ]


@pytest.mark.parametrize("unit", ["episode", "instance"])
@pytest.mark.parametrize("target,scale", [("individual", 1.0), ("mix:0.5", 0.5)])
def test_mean_std_includes_spread_with_tied_episode_means(
    unit: str, target: str, scale: float
) -> None:
    episodes = [
        episode(i, 1, [Seat(f"p{j}", own=r) for j, r in enumerate(rewards)])
        for i, rewards in enumerate(([1, 0, 0], [0, 1, 0]))
    ]
    cfg = CreditConfig(default_target=target, norm="mean_std", baseline_unit=unit)
    credits, _, _ = assign_credit(episodes, cfg, context())
    denominator = scale * sqrt(2 / 9) + cfg.std_eps
    assert advantages(credits) == pytest.approx(
        [scale * r / denominator for r in (2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3, -1 / 3)]
    )


@pytest.mark.parametrize("unit", ["episode", "instance"])
def test_mean_std_swarm_individual_golden(unit: str) -> None:
    episodes = [
        episode(i, 1 - i, [Seat(f"p{j}", own=r) for j, r in enumerate(rewards)])
        for i, rewards in enumerate(([1, 0, 1], [0, 0, 1]))
    ]
    cfg = CreditConfig(default_target="individual", norm="mean_std", baseline_unit=unit)
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx(
        [r / (0.5 + cfg.std_eps) for r in (2 / 3, -1 / 3, 2 / 3, -2 / 3, -2 / 3, 1 / 3)]
    )


def test_mean_std_episode_baseline_preserves_aux_spread() -> None:
    episodes = [
        episode(0, 1, [Seat("p0"), Seat("p1", submission=None)]),
        episode(1, 1, [Seat("p0", submission=None), Seat("p1")]),
    ]
    cfg = CreditConfig(
        baseline="episode", norm="mean_std", aux_rewards=[AuxReward("submitted", weight=0.1)]
    )
    credits, _, _ = assign_credit(episodes, cfg, context())
    assert advantages(credits) == pytest.approx(
        [a / (0.05 + cfg.std_eps) for a in (0.05, -0.05, -0.05, 0.05)]
    )


@pytest.mark.parametrize("unit", ["episode", "instance"])
def test_mean_std_weights_instance_spread_by_episode(unit: str) -> None:
    episodes = [
        episode(i, 0, [Seat(f"p{j}", own=r) for j, r in enumerate(rewards)])
        for i, rewards in enumerate(([1, 0], [0, 1, 0, 1], [0.5]))
    ]
    # Episode means are all 0.5, so within-episode spread alone sets the scale.
    cfg = CreditConfig(default_target="individual", norm="mean_std", baseline_unit=unit)
    credits, _, _ = assign_credit(episodes, cfg, context())
    expected_std = sqrt(1 / 6) if unit == "episode" else sqrt(3 / 14)
    assert advantages(credits) == pytest.approx(
        [r / (expected_std + cfg.std_eps) for r in (0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0)]
    )


def test_per_learner_single_emitter_uses_rms() -> None:
    cfg = CreditConfig(norm="per_learner", overlong="mask_no_answer")
    credits, stats, _ = assign_credit(
        [episode(0, 1), episode(1, 0, final_answer=None)], cfg, context()
    )
    assert advantages(credits) == pytest.approx([1 / (1 + cfg.std_eps)])
    assert stats.norm_skipped == 0


def test_per_learner_identical_rae_advantages_use_rms_and_keep_ema() -> None:
    episodes = [episode(i, 1, [Seat("p0"), Seat("p1")], group=f"g{i // 2}") for i in range(4)]
    cfg = CreditConfig(baseline="rae", norm="per_learner", rae_gamma=0.9)
    state = {"protocol-hash/peer": 0.4}
    credits, stats, updated = assign_credit(episodes, cfg, context(), rae_state=state)
    assert advantages(credits) == pytest.approx([0.6 / (0.6 + cfg.std_eps)] * 8)
    assert stats.norm_skipped == 0
    assert updated == pytest.approx({"protocol-hash/peer": 0.46})
    assert state == {"protocol-hash/peer": 0.4}


def test_per_learner_zero_rms_skips_once_per_learner_and_serializes() -> None:
    cfg = CreditConfig(norm="per_learner", drop_zero_variance=False)
    credits, stats, _ = assign_credit([episode(0, 1), episode(1, 1)], cfg, context())
    assert advantages(credits) == [0, 0]
    assert asdict(stats)["norm_skipped"] == 1


@pytest.mark.parametrize("norm", ["mean_std", "per_learner"])
def test_randomized_normalization_does_not_blow_up(norm: str) -> None:
    rng = Random(20260923)
    for _ in range(100):
        baseline = rng.choice(["role", "episode"])
        target = "team" if baseline == "episode" else rng.choice(["team", "individual", "mix:0.3"])
        episodes = [
            episode(
                i,
                rng.choice([0.0, 0.5, 1.0]),
                [
                    Seat(
                        f"p{j}",
                        own=rng.choice([0.0, 0.5, 1.0]),
                        submission=rng.choice([None, "answer"]),
                    )
                    for j in range(rng.randint(1, 4))
                ],
            )
            for i in range(rng.randint(2, 4))
        ]
        # Fix the reward scale: a reward-relative bound is not scale invariant.
        first = episodes[0]
        episodes[0] = replace(
            first,
            grades={
                **first.grades,
                "_system": {"correct": 1.0},
                first.agents[0].agent_id: {"correct": 1.0},
            },
        )
        cfg = CreditConfig(
            baseline=baseline,
            norm=norm,
            default_target=target,
            baseline_unit=rng.choice(["episode", "instance"]),
            aux_rewards=[AuxReward("submitted", weight=0.1)],
        )
        credits, _, _ = assign_credit(episodes, cfg, context())
        bound = 10 * max((abs(c.reward) for c in credits), default=0) + 1
        assert all(abs(c.advantage) <= bound for c in credits)


@pytest.mark.parametrize("scope", ["episode", "agent"])
async def test_budget_ended_multi_session_carry_is_not_masked(scope: str) -> None:
    from test_protocols_multi_session import session_spec

    from marli.interact.run import run_episode

    spec = session_spec("compaction", budget_end=True)
    ep, _ = await run_episode(spec)
    assert ep.ok and ep.outcome.final_answer == "5"
    assert any(c.forced and c.purpose == Purpose.CARRY for c in ep.calls)
    assert not any(c.forced and c.purpose in (Purpose.FINAL, Purpose.REPORT) for c in ep.calls)
    ctx = context(*spec.protocol.roles())
    cfg = CreditConfig(baseline="none", min_group=1, overlong="mask_forced", overlong_scope=scope)
    credits, stats, _ = assign_credit([ep], cfg, ctx)
    assert {c.segment_id for c in credits} == {s.segment_id for s in ep.segments}
    assert advantages(credits) == [1, 1]
    assert stats.masked_overlong == 0


def test_validate_rejects_unknown_target_role_and_impossible_min_group() -> None:
    with pytest.raises(ConfigError, match="reward_target.*typo.*not a protocol role"):
        validate_credit(CreditConfig(reward_target={"typo": "individual"}), context(), group_size=2)
    with pytest.raises(ConfigError, match="min_group.*group_size"):
        validate_credit(CreditConfig(min_group=4), context(), group_size=2)


@pytest.mark.parametrize("source", ["_system", "a"])
def test_missing_reward_key_is_config_error(source: str) -> None:
    ep = episode(0, 1, [Seat(own=1)])
    ep = replace(ep, grades={**ep.grades, source: {"partial": 0.2}})
    with pytest.raises(ConfigError, match="g/e0.*missing reward_key.*correct"):
        assign_credit([ep], CreditConfig(baseline="none", min_group=1), context())


@pytest.mark.parametrize("count", [1, 3, None])
@pytest.mark.parametrize("kind", ["none", "notes", "tail"])
def test_unit_warning_is_per_recipient_agent(count: int | None, kind: str) -> None:
    ctx = context(
        RoleSpec("peer", ("submit",), "", count=count, context=ContextSpec(kind)),
        RoleSpec("frozen", ("end_session",), "", context=ContextSpec("compaction")),
        seating={"peer": "learner:shared", "frozen": "api:fake"},
    )
    warnings = validate_credit(CreditConfig(segment_credit="last"), ctx, group_size=2)
    assert len(warnings) == 1 and "U == 1" in warnings[0]


def test_multi_session_tail_carry_does_not_warn_segment_credit_has_no_effect() -> None:
    from marli.interact.protocols.multi_session import MultiSessionConfig, MultiSessionProtocol

    [role] = MultiSessionProtocol(MultiSessionConfig(carry="tail")).roles()
    assert "end_session" not in role.tools
    ctx = CreditContext({role.role: role}, {role.role: "learner:shared"})
    assert validate_credit(CreditConfig(segment_credit="last"), ctx, group_size=2) == []
