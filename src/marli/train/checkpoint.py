"""Keep resumable optimizer state separate from published sampling policies.

History holds the learner records for every saved step. Samplers may be concrete
PolicyRefs, or Tinker paths accompanied by a base model (registry name or Tinker
id). Fake samplers must retain their scripted factory ref: a fake:// path alone
cannot reconstruct an in-memory policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypedDict

from marli.errors import ConfigError
from marli.handles import Handle, register_handle
from marli.model import load_model
from marli.policy.refs import PolicyRef, parse_ref


class LearnerCheckpoint(TypedDict):
    state: str | None
    sampler: str | None
    version: int
    base_model: str
    backend: str
    rank: int


class SavedStep(TypedDict):
    step: int
    learners: dict[str, LearnerCheckpoint]


@register_handle
@dataclass(frozen=True, kw_only=True)
class Checkpoint(Handle):
    KIND: ClassVar[str] = "checkpoint"
    MANIFEST: ClassVar[str] = "checkpoint.json"

    step: int
    learners: dict[str, LearnerCheckpoint]
    run_config_hash: str
    rae_state: dict[str, Any] = field(default_factory=dict)
    data_cursor: dict[str, Any] = field(default_factory=dict)
    history: list[SavedStep] = field(default_factory=list)

    def require_state(self, learner: str) -> str:
        record = self._record(learner)
        state = record.get("state")
        if not state:
            raise ConfigError(f"checkpoint learner {learner!r} has no resumable state")
        return state if "://" in state else str((self.root / state).resolve())

    def _records(self, step: int | None) -> dict[str, LearnerCheckpoint]:
        if step is None or step == self.step:
            return self.learners
        for saved in self.history:
            if saved["step"] == step:
                return saved["learners"]
        raise ConfigError(f"checkpoint has no saved step {step}")

    def _record(self, learner: str, step: int | None = None) -> LearnerCheckpoint:
        records = self._records(step)
        if learner not in records:
            selected_step = self.step if step is None else step
            raise ConfigError(f"checkpoint has no learner {learner!r} at step {selected_step}")
        return records[learner]

    def policy_ref(self, learner: str, step: int | None = None) -> str:
        record = self._record(learner, step)
        sampler = record.get("sampler")
        if not sampler:
            raise ConfigError(f"checkpoint learner {learner!r} has no sampler at step {step}")
        if sampler.startswith("tinker://"):
            if record["backend"] != "tinker":
                raise ConfigError("Tinker sampler path requires backend=tinker")
            base = record["base_model"]
            if "/" not in base:
                model = load_model(base)
                if model.tinker_id is None:
                    raise ConfigError(f"model {base!r} has no tinker_id")
                base = model.tinker_id
            return str(PolicyRef("tinker", base, sampler=sampler))
        ref = parse_ref(sampler)
        if ref.kind not in {"tinker", "scripted", "vllm"}:
            raise ConfigError(f"checkpoint sampler must be a concrete token policy: {sampler!r}")
        return str(ref)


def resolve_checkpoint_ref(ref: PolicyRef, learner: str | None) -> str:
    """Resolve a checkpoint without changing the public PolicyRef grammar."""
    if ref.kind != "ckpt":
        raise ConfigError(f"expected ckpt policy ref, got {ref.kind!r}")
    checkpoint = Checkpoint.load(Path(ref.target))
    step = ref.step if isinstance(ref.step, int) else None
    if learner is None:
        names = checkpoint._records(step)
        if len(names) != 1:
            raise ConfigError("checkpoint must have exactly one learner or specify learner")
        learner = next(iter(names))
    return checkpoint.policy_ref(learner, step)
