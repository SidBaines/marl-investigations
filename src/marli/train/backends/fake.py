"""Exercise learner routing, sampler versions and resume without paid compute.

Weights are a step counter, not a language model. Scripted snapshot refs identify
the factory only; they cannot reload learned weights or versions across runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from marli.errors import ConfigError
from marli.handles import atomic_write_text
from marli.model import ModelSpec
from marli.policy.refs import PolicyRef, resolve_scripted
from marli.policy.scripted import ScriptedPolicy
from marli.train.backends.base import SamplerSnapshot, StepResult
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec, TrainDatum


@dataclass(frozen=True)
class FakeStepCall:
    learner: str
    step: int
    datums: tuple[TrainDatum, ...]
    learning_rate: float


class FakeLearner:
    def __init__(
        self,
        name: str,
        spec: LearnerSpec,
        *,
        model: ModelSpec,
        policy_factory: str | None,
        state_dir: Path | None,
    ) -> None:
        self.name = name
        self.spec = spec
        self.model = model
        self.version = 0
        self.weights = 0
        self.steps: list[list[TrainDatum]] = []
        self.calls: list[FakeStepCall] = []
        self.policy_factory = policy_factory
        self.state_dir = state_dir
        self._sampler_names: set[str] = set()

    async def train_step(
        self, datums: Sequence[TrainDatum], *, learning_rate: float | None = None
    ) -> StepResult:
        if not datums:
            raise ConfigError("train_step requires non-empty datums")
        for datum in datums:
            if datum.learner != self.name:
                raise ConfigError(f"datum for {datum.learner!r} routed to learner {self.name!r}")
            if len(datum.tokens) > self.model.max_ctx:
                raise ConfigError(f"datum exceeds model max ctx {self.model.max_ctx}")
        versions = {datum.policy_version for datum in datums}
        if len(versions) != 1 or versions != {self.version}:
            raise ConfigError(f"datums must use current policy version {self.version}")
        lr = self.spec.learning_rate if learning_rate is None else learning_rate
        self.steps.append(list(datums))
        self.calls.append(FakeStepCall(self.name, len(self.steps), tuple(datums), lr))
        self.weights += 1
        return StepResult(
            learner=self.name,
            n_datums=len(datums),
            n_tokens=sum(len(d.tokens) - 1 for d in datums),
            n_action_tokens=sum(d.n_action_tokens for d in datums),
            loss=-sum(m * a for d in datums for m, a in zip(d.mask, d.advantages, strict=True)),
            kl_sample_train=0.0,
        )

    def policy(self, *, policy_id: str | None = None) -> ScriptedPolicy:
        if self.policy_factory is None:
            raise ConfigError("fake policy() requires policy_factory=module:attr")
        policy = resolve_scripted(PolicyRef("scripted", self.policy_factory))
        if not isinstance(policy, ScriptedPolicy):
            raise ConfigError("fake policy_factory must return a ScriptedPolicy")
        policy.trainable = True
        policy.policy_version = self.version
        policy.policy_id = self.name if policy_id is None else policy_id
        return policy

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        if name in self._sampler_names:
            raise ValueError(f"sampler name {name!r} already used by learner {self.name!r}")
        if self.policy_factory is None:
            raise ConfigError("fake sync_sampler requires policy_factory=module:attr")
        ref = str(PolicyRef("scripted", self.policy_factory))
        self._sampler_names.add(name)
        self.version += 1
        return SamplerSnapshot(self.name, self.version, f"fake://{self.name}/{name}", ref)

    async def save_state(self, name: str) -> str:
        if self.state_dir is None:
            raise ConfigError("fake save_state requires an explicit state_dir")
        path = (self.state_dir / self.name / f"{name}.json").resolve()
        atomic_write_text(path, json.dumps({"version": self.version, "weights": self.weights}))
        return str(path)

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version = state["version"]
        self.weights = state["weights"]

    async def close(self) -> None:
        pass


class FakeBackend:
    name = "fake"

    def __init__(
        self, policy_factory: str | None = None, *, state_dir: str | Path | None = None
    ) -> None:
        self.policy_factory = policy_factory
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.learners: dict[str, FakeLearner] = {}

    async def create_learner(
        self, name: str, spec: LearnerSpec, *, model: ModelSpec, seed: int
    ) -> FakeLearner:
        if name in self.learners:
            raise ConfigError(f"learner {name!r} already exists")
        learner = FakeLearner(
            name, spec, model=model, policy_factory=self.policy_factory, state_dir=self.state_dir
        )
        if spec.init_from is not None:
            checkpoint = Checkpoint.load(spec.init_from)
            await learner.load_state(checkpoint.require_state(name))
            learner.version = checkpoint.learners[name]["version"]
        self.learners[name] = learner
        return learner

    async def close(self) -> None:
        for learner in self.learners.values():
            await learner.close()
