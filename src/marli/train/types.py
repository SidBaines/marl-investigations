"""Training contracts: learners, seating, credit configuration, datums.

The training loop consumes recorded :class:`~marli.interact.types.Episode`\\ s
(plus their token sidecars) and never re-renders anything: a datum is built
per **(agent, segment)** from exactly the ids the policy saw.

Pipeline (``train/credit.py`` implements it; its docstring and the golden tests
in ``tests/test_train_credit.py`` are the spec)::

    episodes (G per task group)
      1. drop ok=False episodes; drop groups with < credit.min_group episodes
      2. rewards r[e, a] from grades via reward_target (team / individual / mix)
         + aux rewards (annealed, capped)
      3. overlong filter (mask_no_answer / mask_forced; scope episode | agent)
      4. baseline per credit.baseline (episode LOO | role LOO | rae EMA | none)
      5. zero-variance filter over the same groups
      6. norm (mean | mean_std | per_learner)
      7. segment_credit weights over units (sessions or segments) of each agent
      8. recipients: only these roles emit datums
      9. loss_agg scaling (the RL loss is SUM-reduced; this sets the effective mean)
     10. -> per-(episode, agent, segment) scalar advantage -> datums (train/datums.py)

A datum's arrays are aligned to ``targets = tokens[1:]``: position ``p``
predicts ``tokens[p + 1]``. Only the agent's own **sampled** completion
positions have ``mask = 1`` (and a nonzero advantage / sampler logprob);
prompts, tool results, pushed deliveries, forced prefixes, forced closes and
carries are observations (mask 0, logprob 0, advantage 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marli.config import doc_field
from marli.errors import ConfigError

# ---------------------------------------------------------------- learners / seating


@dataclass
class LearnerSpec:
    """One trainable LoRA. Several roles may share a learner (shared LoRA)."""

    base_model: str = doc_field("", help="models registry name (marli list models)")
    backend: str = doc_field("tinker", help="tinker | local | fake")
    rank: int = 32
    learning_rate: float = 1e-5
    # optimizer (AdamW) — defaults follow tinker-cookbook RL recipes
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    loss: str = doc_field("importance_sampling", help="importance_sampling | ppo")
    init_from: str | None = doc_field(None, help="checkpoint manifest (state) to resume/warm-start")

    def __post_init__(self) -> None:
        if self.backend not in {"tinker", "local", "fake"}:
            raise ConfigError(
                f"learner.backend must be tinker, local or fake, got {self.backend!r}"
            )
        if self.rank <= 0 or self.learning_rate <= 0:
            raise ConfigError("learner.rank and learner.learning_rate must be positive")
        if self.loss not in {"importance_sampling", "ppo"}:
            raise ConfigError(f"learner.loss must be importance_sampling or ppo, got {self.loss!r}")


# seating: role -> "learner:<name>" | a frozen PolicyRef string ("tinker:…", "api:…", …).
# Every role of the protocol must be seated (workers / finalizers included).
LEARNER_PREFIX = "learner:"


def seat_learner(seat: str) -> str | None:
    """The learner name for a trainable seat, or None for a frozen / API seat."""
    return seat[len(LEARNER_PREFIX) :] if seat.startswith(LEARNER_PREFIX) else None


# ---------------------------------------------------------------- credit configuration


@dataclass
class AuxReward:
    """An auxiliary reward from the ``marli.train.rewards.AUX_REWARDS`` registry."""

    name: str = ""
    weight: float = 0.0
    # constant | linear (weight -> end_weight over [start_step, end_step]) — annealing
    schedule: str = "constant"
    end_weight: float = 0.0
    start_step: int = 0
    end_step: int = 0
    cap: float | None = None  # |aux contribution| <= cap per agent (after weighting)
    roles: tuple[str, ...] = ()  # empty = all roles

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("aux reward needs a name")
        if self.schedule not in {"constant", "linear"}:
            raise ConfigError(f"aux schedule must be constant or linear, got {self.schedule!r}")
        if self.schedule == "linear" and self.end_step <= self.start_step:
            raise ConfigError("aux linear schedule needs end_step > start_step")
        if self.cap is not None and self.cap < 0:
            raise ConfigError("aux cap must be >= 0")


REWARD_TARGETS = ("team", "individual")  # plus "mix:<alpha>" with 0 <= alpha <= 1


def parse_reward_target(value: str) -> tuple[str, float]:
    """``team`` → (team, 1.0); ``individual`` → (individual, 0.0); ``mix:0.3`` → (mix, 0.3).

    The float is alpha in ``r = alpha * team + (1 - alpha) * own``.
    """
    if value == "team":
        return "team", 1.0
    if value == "individual":
        return "individual", 0.0
    if value.startswith("mix:"):
        try:
            alpha = float(value[4:])
        except ValueError:
            alpha = float("nan")
        if 0.0 <= alpha <= 1.0:
            return "mix", alpha
    raise ConfigError(f"reward_target must be team, individual or mix:<0..1>, got {value!r}")


@dataclass
class CreditConfig:
    """Which agents/sessions receive credit, and how it is normalized.

    Field-level validation happens here; combination validation (which needs the
    protocol roles, seating and group size) is ``train.credit.validate_credit``.
    """

    reward_key: str = doc_field("correct", help="grade component used as the reward")
    # role -> team | individual | mix:<alpha>. Roles not listed use `default_target`.
    reward_target: dict[str, str] = field(default_factory=dict)
    default_target: str = "team"
    aux_rewards: list[AuxReward] = field(default_factory=list)
    # overlong / exhaustion filter: none | mask_no_answer | mask_forced
    overlong: str = "none"
    overlong_scope: str = "episode"  # episode | agent
    baseline: str = "role"  # episode | role | rae | none
    baseline_unit: str = "episode"  # role baseline weighting: episode | instance
    rae_gamma: float = 0.95
    drop_zero_variance: bool = True
    min_group: int = 2
    norm: str = "mean"  # mean | mean_std | per_learner
    std_eps: float = 1e-6
    segment_credit: str = "all"  # all | last | geometric
    segment_gamma: float = 0.5
    segment_unit: str = "session"  # session | segment
    segment_normalize: str = "none"  # none | sum_to_one
    # roles that emit datums; empty = every role seated on a learner
    recipients: tuple[str, ...] = ()
    loss_agg: str = "token_sum"  # token_sum | token_mean_per_learner | agent_mean

    def __post_init__(self) -> None:
        choices = {
            "overlong": ("none", "mask_no_answer", "mask_forced"),
            "overlong_scope": ("episode", "agent"),
            "baseline": ("episode", "role", "rae", "none"),
            "baseline_unit": ("episode", "instance"),
            "norm": ("mean", "mean_std", "per_learner"),
            "segment_credit": ("all", "last", "geometric"),
            "segment_unit": ("session", "segment"),
            "segment_normalize": ("none", "sum_to_one"),
            "loss_agg": ("token_sum", "token_mean_per_learner", "agent_mean"),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ConfigError(f"credit.{name} must be one of {allowed}, got {value!r}")
        for target in (self.default_target, *self.reward_target.values()):
            parse_reward_target(target)
        if not 0.0 < self.rae_gamma < 1.0:
            raise ConfigError("credit.rae_gamma must be in (0, 1)")
        if not 0.0 < self.segment_gamma <= 1.0:
            raise ConfigError("credit.segment_gamma must be in (0, 1]")
        if self.min_group < 1:
            raise ConfigError("credit.min_group must be >= 1")
        if self.std_eps <= 0:
            raise ConfigError("credit.std_eps must be > 0")
        if self.norm == "mean_std" and self.baseline == "rae":
            raise ConfigError("credit.norm=mean_std is incompatible with baseline=rae")

    def target_for(self, role: str) -> tuple[str, float]:
        return parse_reward_target(self.reward_target.get(role, self.default_target))


# ---------------------------------------------------------------- datums


@dataclass(frozen=True)
class TrainDatum:
    """One (agent, segment) training sequence for one learner.

    ``tokens`` is the segment buffer up to the end of its last sampled
    completion. ``logprobs`` / ``mask`` / ``advantages`` have length
    ``len(tokens) - 1`` and are aligned to ``tokens[1:]``. Tinker's datum is
    ``model_input = tokens[:-1]``, ``loss_fn_inputs = {target_tokens: tokens[1:],
    logprobs, advantages}``; ``mask`` is a sidecar (advantage 0 ⇒ no gradient).
    """

    learner: str
    episode_id: str
    agent_id: str
    role: str
    segment_id: str
    session_idx: int
    policy_version: int | None
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    mask: tuple[float, ...]
    advantages: tuple[float, ...]
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        n = len(self.tokens) - 1
        if n < 1:
            raise ValueError("a datum needs at least two tokens")
        if not (len(self.logprobs) == len(self.mask) == len(self.advantages) == n):
            raise ValueError("logprobs/mask/advantages must align with tokens[1:]")

    @property
    def n_action_tokens(self) -> int:
        return int(sum(self.mask))


@dataclass(frozen=True)
class SegmentCredit:
    """The scalar advantage assigned to one (episode, agent, segment) unit."""

    episode_id: str
    agent_id: str
    role: str
    segment_id: str
    learner: str
    advantage: float  # after baseline, norm, segment weight and loss_agg scaling
    reward: float  # r[e, a] before the baseline (for logging)
    weight: float  # segment_credit weight w_u


@dataclass
class CreditStats:
    """Counters and summaries for logging (progress.json / tracking)."""

    n_episodes: int = 0
    dropped_not_ok: int = 0
    dropped_small_groups: int = 0
    zero_variance_groups: int = 0
    n_groups: int = 0
    no_baseline: int = 0  # (role, episode) units with no other-episode baseline → A = 0
    masked_overlong: int = 0
    reward_mean: dict[str, float] = field(default_factory=dict)  # per role
    advantage_abs_mean: dict[str, float] = field(default_factory=dict)  # per role
    action_tokens: dict[str, int] = field(default_factory=dict)  # per learner
    # learner -> role -> share of that learner's action tokens (role-capture watch)
    role_token_share: dict[str, dict[str, float]] = field(default_factory=dict)
