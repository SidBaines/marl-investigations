"""Keep credit ablations independent of token rendering and backend loss code.

The order is part of the scientific contract: drop failed/small groups, compute
all agents' rewards, mark overlong agents, subtract baselines, drop all-zero
groups, normalize, weight segments, select recipients, and scale the summed loss.
Masking and recipient selection never remove observations from a baseline.

Group reward standard deviations include the focal episode. Episode baselines
use episode means of agent rewards (team plus any aux); role baselines use
episode role means or individual role instances according to baseline_unit.
Without a baseline, mean_std uses all
agent rewards in the group. Per-learner normalization counts each emitting
agent once, before segment weights or loss scaling, including zero advantages.

Stats count input episodes, groups surviving min_group (including zero-variance
drops), and masked agents. Reward means include all agents surviving min_group;
advantage means summarize final emitted segment credits. Token shares count
only sampled completion ids in emitted segments.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean, pstdev

from marli.errors import ConfigError
from marli.interact.system import RoleSpec
from marli.interact.types import SYSTEM_GRADE_KEY, AgentInfo, Episode, SegmentInfo
from marli.train.rewards import AUX_REWARDS
from marli.train.types import AuxReward, CreditConfig, CreditStats, SegmentCredit, seat_learner


@dataclass(frozen=True)
class CreditContext:
    roles: dict[str, RoleSpec]
    seating: dict[str, str]
    step: int = 0
    protocol_hash: str = ""


def validate_credit(
    cfg: CreditConfig, ctx: CreditContext, *, group_size: int, allow_idle: bool = False
) -> list[str]:
    """Validate protocol-dependent combinations before sampling spends compute."""
    if cfg.reward_key == "oracle_any":
        raise ConfigError("credit.reward_key=oracle_any is eval-only")
    for role, spec in ctx.roles.items():
        if role not in ctx.seating:
            raise ConfigError(f"role {role!r} is missing from seating")
        target, _ = cfg.target_for(role)
        if cfg.baseline == "episode" and target != "team":
            raise ConfigError(
                f"baseline=episode requires team rewards; role {role!r} uses {target}"
            )
        if target != "team" and "submit" not in spec.tools:
            raise ConfigError(f"role {role!r} targets {target} but has no submission (submit tool)")
    for role in cfg.recipients:
        if role not in ctx.roles:
            raise ConfigError(f"recipient {role!r} is not a protocol role")
        if seat_learner(ctx.seating[role]) is None:
            raise ConfigError(f"recipient {role!r} is seated on a frozen/API ref")
    recipients = cfg.recipients or tuple(ctx.roles)
    learner_roles: dict[str, list[str]] = defaultdict(list)
    for role in ctx.roles:
        learner = seat_learner(ctx.seating[role])
        if learner is not None:
            learner_roles[learner].append(role)
    declared = {
        learner for seat in ctx.seating.values() if (learner := seat_learner(seat)) is not None
    }
    active = {
        learner for role in recipients if (learner := seat_learner(ctx.seating[role])) is not None
    }
    if not allow_idle and (idle := declared - active):
        raise ConfigError(f"idle learners have no recipient role: {sorted(idle)}; set allow_idle")
    if group_size < 2 and cfg.baseline in {"episode", "role"}:
        raise ConfigError(f"baseline={cfg.baseline} requires group_size >= 2")
    for aux in cfg.aux_rewards:
        AUX_REWARDS.get(aux.name)
        for role in aux.roles:
            if role not in ctx.roles:
                raise ConfigError(f"aux reward {aux.name!r} role {role!r} is not a protocol role")

    warnings: list[str] = []
    if (
        cfg.segment_credit != "all"
        and ctx.roles
        and all(
            role.count == 1 and role.context.kind == "none" and "end_session" not in role.tools
            for role in ctx.roles.values()
        )
    ):
        warnings.append(
            "segment_credit != all but roles have count=1, context=none and no session tool; "
            "for a single-session protocol U == 1, so segment credit has no effect"
        )
    if cfg.norm == "per_learner":
        for learner, roles in sorted(learner_roles.items()):
            if len(roles) == 1:
                warnings.append(
                    f"learner {learner!r} serves a single role: per_learner == mean_std-like; "
                    "did you mean norm=mean_std?"
                )
    return warnings


@dataclass(frozen=True)
class _AgentCredit:
    episode: Episode
    agent: AgentInfo
    reward: float
    advantage: float = 0.0
    reward_std: float = 0.0
    masked: bool = False


@dataclass(frozen=True)
class _WeightedSegment:
    row: _AgentCredit
    segment: SegmentInfo
    weight: float
    action_tokens: int


@dataclass(frozen=True)
class _Emission:
    credit: SegmentCredit
    action_tokens: int


def _drop_episodes(
    episodes: Sequence[Episode], cfg: CreditConfig, stats: CreditStats
) -> list[Episode]:
    groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        if not episode.ok:
            stats.dropped_not_ok += 1
        else:
            groups[episode.group_id].append(episode)
    kept: list[Episode] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        if len(group) < cfg.min_group:
            stats.dropped_small_groups += len(group)
        else:
            stats.n_groups += 1
            kept.extend(sorted(group, key=lambda episode: episode.episode_idx))
    return kept


def _aux_weight(aux: AuxReward, step: int) -> float:
    if aux.schedule == "constant":
        return aux.weight
    fraction = min(1.0, max(0.0, (step - aux.start_step) / (aux.end_step - aux.start_step)))
    return aux.weight + fraction * (aux.end_weight - aux.weight)


def _compute_rewards(
    episodes: Sequence[Episode], cfg: CreditConfig, ctx: CreditContext
) -> list[_AgentCredit]:
    auxiliaries = [
        (aux, _aux_weight(aux, ctx.step), AUX_REWARDS.get(aux.name)) for aux in cfg.aux_rewards
    ]
    rows: list[_AgentCredit] = []
    for episode in episodes:
        team = episode.grades[SYSTEM_GRADE_KEY][cfg.reward_key]
        for agent in sorted(episode.agents, key=lambda agent: agent.seat_key):
            target, alpha = cfg.target_for(agent.role)
            own_grade = episode.grades.get(agent.agent_id)
            own = own_grade[cfg.reward_key] if own_grade is not None else 0.0
            reward = team if target == "team" else own
            if target == "mix":
                reward = alpha * team + (1.0 - alpha) * own
            for aux, weight, function in auxiliaries:
                if not aux.roles or agent.role in aux.roles:
                    contribution = weight * function(episode, agent.agent_id)
                    if aux.cap is not None:
                        contribution = max(-aux.cap, min(aux.cap, contribution))
                    reward += contribution
            rows.append(_AgentCredit(episode, agent, reward))
    return rows


def _mask_overlong(
    rows: Sequence[_AgentCredit], cfg: CreditConfig, ctx: CreditContext, stats: CreditStats
) -> list[_AgentCredit]:
    result: list[_AgentCredit] = []
    for row in rows:
        episode, agent = row.episode, row.agent
        masked = False
        if cfg.overlong == "mask_no_answer":
            if cfg.overlong_scope == "agent" and "submit" in ctx.roles[agent.role].tools:
                masked = episode.outcome.submissions.get(agent.agent_id) is None
            else:
                masked = episode.outcome.final_answer is None
        elif cfg.overlong == "mask_forced":
            masked = any(
                call.forced and (cfg.overlong_scope == "episode" or call.agent_id == agent.agent_id)
                for call in episode.calls
            )
        stats.masked_overlong += int(masked)
        result.append(replace(row, masked=masked))
    return result


def _group_baseline(
    rows: Sequence[_AgentCredit], cfg: CreditConfig, stats: CreditStats
) -> list[_AgentCredit]:
    by_episode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode.episode_id].append(row.reward)
    means = {episode_id: fmean(rewards) for episode_id, rewards in by_episode.items()}
    population = (
        list(means.values())
        if cfg.baseline_unit == "episode" or cfg.baseline == "episode"
        else [reward for rewards in by_episode.values() for reward in rewards]
    )
    reward_std = pstdev(population)
    baselines: dict[str, float] = {}
    for episode_id in by_episode:
        others = (
            [mean for other, mean in means.items() if other != episode_id]
            if cfg.baseline_unit == "episode" or cfg.baseline == "episode"
            else [
                r for other, rewards in by_episode.items() if other != episode_id for r in rewards
            ]
        )
        if others:
            baselines[episode_id] = fmean(others)
    if not baselines:
        stats.no_baseline += len({(row.agent.role, row.episode.episode_id) for row in rows})
    return [
        replace(
            row,
            advantage=row.reward - baselines[row.episode.episode_id] if baselines else 0.0,
            reward_std=reward_std,
        )
        for row in rows
    ]


def _apply_baselines(
    rows: Sequence[_AgentCredit],
    cfg: CreditConfig,
    ctx: CreditContext,
    stats: CreditStats,
    rae_state: dict[str, float],
) -> tuple[list[_AgentCredit], dict[str, float]]:
    new_state = dict(rae_state)
    groups: dict[tuple[str, str], list[_AgentCredit]] = defaultdict(list)
    for row in rows:
        group_id = "" if cfg.baseline == "rae" else row.episode.group_id
        role = row.agent.role if cfg.baseline in {"role", "rae"} else ""
        groups[group_id, role].append(row)
    updated: dict[tuple[str, str], _AgentCredit] = {}
    for (_, role), group in groups.items():
        if cfg.baseline == "rae":
            key = f"{ctx.protocol_hash}/{role}"
            batch_mean = fmean(row.reward for row in group)
            baseline = rae_state.get(key, batch_mean)
            new_state[key] = (
                cfg.rae_gamma * baseline + (1.0 - cfg.rae_gamma) * batch_mean
                if key in rae_state
                else batch_mean
            )
            result = [replace(row, advantage=row.reward - baseline) for row in group]
        elif cfg.baseline == "none":
            reward_std = pstdev(row.reward for row in group)
            result = [replace(row, advantage=row.reward, reward_std=reward_std) for row in group]
        else:
            result = _group_baseline(group, cfg, stats)
        updated.update(((row.episode.episode_id, row.agent.agent_id), row) for row in result)
    return [updated[row.episode.episode_id, row.agent.agent_id] for row in rows], new_state


def _drop_zero_variance(
    rows: Sequence[_AgentCredit], cfg: CreditConfig, stats: CreditStats
) -> list[_AgentCredit]:
    if not cfg.drop_zero_variance:
        return list(rows)
    nonzero = {row.episode.group_id for row in rows if row.advantage != 0.0}
    dropped = {row.episode.group_id for row in rows} - nonzero
    stats.zero_variance_groups += len(dropped)
    return [row for row in rows if row.episode.group_id not in dropped]


def _agent_segments(episode: Episode, agent_id: str) -> list[tuple[SegmentInfo, int]]:
    tokens: dict[str, int] = defaultdict(int)
    for call in episode.calls:
        if call.agent_id == agent_id and call.segment_id:
            tokens[call.segment_id] += len(call.completion_ids)
    return [
        (segment, tokens[segment.segment_id])
        for segment in episode.segments
        if segment.agent_id == agent_id and tokens.get(segment.segment_id, 0) > 0
    ]


def _recipient_learner(row: _AgentCredit, cfg: CreditConfig, ctx: CreditContext) -> str | None:
    if row.masked or (cfg.recipients and row.agent.role not in cfg.recipients):
        return None
    return seat_learner(ctx.seating[row.agent.role])


def _normalize(
    rows: Sequence[_AgentCredit], cfg: CreditConfig, ctx: CreditContext
) -> list[_AgentCredit]:
    if cfg.norm == "mean":
        return list(rows)
    deviations: dict[str, float] = {}
    if cfg.norm == "per_learner":
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            learner = _recipient_learner(row, cfg, ctx)
            if learner is not None and _agent_segments(row.episode, row.agent.agent_id):
                values[learner].append(row.advantage)
        deviations = {learner: pstdev(advantages) for learner, advantages in values.items()}
    result: list[_AgentCredit] = []
    for row in rows:
        if cfg.norm == "mean_std":
            result.append(replace(row, advantage=row.advantage / (row.reward_std + cfg.std_eps)))
        else:
            learner = _recipient_learner(row, cfg, ctx)
            result.append(
                replace(row, advantage=row.advantage / (deviations[learner] + cfg.std_eps))
                if learner in deviations
                else row
            )
    return result


def _segment_weights(segments: Sequence[SegmentInfo], cfg: CreditConfig) -> list[float]:
    units = (
        [segment.session_idx for segment in segments]
        if cfg.segment_unit == "session"
        else list(range(len(segments)))
    )
    order = list(dict.fromkeys(units))
    weights = {
        unit: (
            1.0
            if cfg.segment_credit == "all"
            else float(index == len(order) - 1)
            if cfg.segment_credit == "last"
            else cfg.segment_gamma ** (len(order) - index - 1)
        )
        for index, unit in enumerate(order)
    }
    total = sum(weights.values()) if cfg.segment_normalize == "sum_to_one" else 1.0
    return [weights[unit] / total for unit in units]


def _weight_segments(rows: Sequence[_AgentCredit], cfg: CreditConfig) -> list[_WeightedSegment]:
    result: list[_WeightedSegment] = []
    for row in rows:
        segments = _agent_segments(row.episode, row.agent.agent_id)
        weights = _segment_weights([segment for segment, _ in segments], cfg)
        result.extend(
            _WeightedSegment(row, segment, weight, tokens)
            for (segment, tokens), weight in zip(segments, weights, strict=True)
            if weight > 0.0
        )
    return result


def _select_recipients(
    segments: Sequence[_WeightedSegment], cfg: CreditConfig, ctx: CreditContext
) -> list[_Emission]:
    result: list[_Emission] = []
    for unit in segments:
        row = unit.row
        learner = _recipient_learner(row, cfg, ctx)
        if learner is not None:
            result.append(
                _Emission(
                    SegmentCredit(
                        episode_id=row.episode.episode_id,
                        agent_id=row.agent.agent_id,
                        role=row.agent.role,
                        segment_id=unit.segment.segment_id,
                        learner=learner,
                        advantage=row.advantage * unit.weight,
                        reward=row.reward,
                        weight=unit.weight,
                    ),
                    unit.action_tokens,
                )
            )
    return result


def _aggregate_loss(emissions: Sequence[_Emission], cfg: CreditConfig) -> list[_Emission]:
    learner_tokens: dict[str, int] = defaultdict(int)
    agent_tokens: dict[tuple[str, str, str], int] = defaultdict(int)
    for emission in emissions:
        credit = emission.credit
        learner_tokens[credit.learner] += emission.action_tokens
        agent_tokens[credit.learner, credit.episode_id, credit.agent_id] += emission.action_tokens
    agent_counts: dict[str, int] = defaultdict(int)
    for learner, _, _ in agent_tokens:
        agent_counts[learner] += 1
    result: list[_Emission] = []
    for emission in emissions:
        credit = emission.credit
        denominator = 1
        if cfg.loss_agg == "token_mean_per_learner":
            denominator = learner_tokens[credit.learner]
        elif cfg.loss_agg == "agent_mean":
            denominator = (
                agent_tokens[credit.learner, credit.episode_id, credit.agent_id]
                * agent_counts[credit.learner]
            )
        result.append(
            replace(emission, credit=replace(credit, advantage=credit.advantage / denominator))
        )
    return result


def _summarize(
    rewards: Sequence[_AgentCredit], emissions: Sequence[_Emission], stats: CreditStats
) -> None:
    role_rewards: dict[str, list[float]] = defaultdict(list)
    role_advantages: dict[str, list[float]] = defaultdict(list)
    role_tokens: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rewards:
        role_rewards[row.agent.role].append(row.reward)
    for emission in emissions:
        credit = emission.credit
        role_advantages[credit.role].append(abs(credit.advantage))
        role_tokens[credit.learner][credit.role] += emission.action_tokens
    stats.reward_mean = {role: fmean(values) for role, values in sorted(role_rewards.items())}
    stats.advantage_abs_mean = {
        role: fmean(values) for role, values in sorted(role_advantages.items())
    }
    stats.action_tokens = {
        learner: sum(counts.values()) for learner, counts in sorted(role_tokens.items())
    }
    stats.role_token_share = {
        learner: {
            role: count / stats.action_tokens[learner] for role, count in sorted(counts.items())
        }
        for learner, counts in sorted(role_tokens.items())
    }


def assign_credit(
    episodes: Sequence[Episode],
    cfg: CreditConfig,
    ctx: CreditContext,
    *,
    rae_state: dict[str, float] | None = None,
) -> tuple[list[SegmentCredit], CreditStats, dict[str, float]]:
    """Assign credit from recorded episodes after preflight with validate_credit.

    RAE reads a snapshot of the input state and updates once per role over the
    retained batch, including masked agents and groups later dropped as all-zero.
    The caller validates the configured group size before rollout; actual groups
    may shrink after failed episodes are discarded. Idle learners are permitted
    here so a caller's explicit allow_idle validation remains effective.
    """
    stats = CreditStats(n_episodes=len(episodes))
    kept = _drop_episodes(episodes, cfg, stats)
    rewards = _compute_rewards(kept, cfg, ctx)
    rows = _mask_overlong(rewards, cfg, ctx, stats)
    rows, new_state = _apply_baselines(
        rows, cfg, ctx, stats, rae_state if rae_state is not None else {}
    )
    rows = _drop_zero_variance(rows, cfg, stats)
    rows = _normalize(rows, cfg, ctx)
    segments = _weight_segments(rows, cfg)
    emissions = _select_recipients(segments, cfg, ctx)
    emissions = _aggregate_loss(emissions, cfg)
    _summarize(rewards, emissions, stats)
    return [emission.credit for emission in emissions], stats, new_state
