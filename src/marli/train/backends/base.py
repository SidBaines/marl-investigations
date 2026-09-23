"""Training backends: one algorithm loop, several ways to run a LoRA learner.

A :class:`TrainBackend` creates :class:`Learner`\\ s — one per trainable LoRA
(``train rl`` config ``learners:``). Several roles may share one learner (shared
LoRA) or each role gets its own (per-role LoRAs); the loop routes datums by
seating and never mixes learners in one batch.

Backends:

- ``tinker`` (``train/backends/tinker.py``): one ``TrainingClient`` per learner.
- ``local`` (M4, ``train/backends/local/``): PEFT multi-adapter learner + vLLM
  multi-LoRA sampler with versioned adapter names.
- ``fake`` (``train/backends/fake.py``): in-memory, deterministic, CPU — for the
  loop's tests and the CLI end-to-end chain.

**Concurrency contract.** ``train_step`` *submits* the forward/backward and the
optimizer step before awaiting either result (Tinker pipelines them), so the
loop can ``asyncio.gather`` every learner's ``train_step`` and all learners'
work is in flight together. A learner with no informative datums this step is
simply not stepped.

**Versions.** ``version`` is the sampler version the learner's ``policy()``
currently serves (0 = base weights / ``init_from``). ``sync_sampler`` publishes
the current weights for sampling under a new, strictly increasing version and a
unique name (never overwrite a name that a sampler may still hold — the local
backend relies on this for vLLM's LoRA cache). Every rollout Call records
``policy_version``; datums refuse mixed versions (on-policy sync training).

**Checkpoints.** ``save_state`` returns a resumable *state* path (weights +
optimizer); ``sync_sampler`` returns a *sampler* path for eval/serving. They are
never interchangeable (``Checkpoint.require_state()``). Tinker checkpoints are
saved TTL-free and named deterministically from the run.

Loss: ``LearnerSpec.loss`` (``importance_sampling`` default | ``ppo``), always
**sum-reduced** over action tokens (Tinker semantics); credit/normalization is
already folded into the advantages (train/credit.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from marli.model import ModelSpec
    from marli.policy.base import TokenPolicy
    from marli.train.types import LearnerSpec, TrainDatum


@dataclass(frozen=True)
class StepResult:
    """One learner's train step: forward/backward + optimizer."""

    learner: str
    n_datums: int
    n_tokens: int  # model-input tokens across datums
    n_action_tokens: int
    loss: float  # sum-reduced loss reported by the backend
    grad_norm: float | None = None
    # mean over action tokens of (sampler logprob - training logprob): the
    # off-policy gap. Large values mean sampler/trainer mismatch (renderer, dtype, …).
    kl_sample_train: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerSnapshot:
    """A published sampler version."""

    learner: str
    version: int
    path: str  # backend sampler path (tinker://… | local adapter dir)
    policy_ref: str  # PolicyRef string that samples this snapshot (never secrets)


class Learner(Protocol):
    name: str
    spec: LearnerSpec
    model: ModelSpec
    version: int

    async def train_step(
        self, datums: Sequence[TrainDatum], *, learning_rate: float | None = None
    ) -> StepResult:
        """Submit forward/backward then optim before awaiting; raise on empty ``datums``."""
        ...

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        """Publish current weights for sampling as ``version + 1`` and switch ``policy()``."""
        ...

    def policy(self, *, policy_id: str | None = None) -> TokenPolicy:
        """A trainable TokenPolicy (T=1, raw logprobs) serving the current version."""
        ...

    async def save_state(self, name: str) -> str:
        """Save weights + optimizer state (TTL-free); return the state path."""
        ...

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None: ...

    async def close(self) -> None: ...


class TrainBackend(Protocol):
    name: str

    async def create_learner(
        self, name: str, spec: LearnerSpec, *, model: ModelSpec, seed: int
    ) -> Learner:
        """Create (or, with ``spec.init_from``, restore) one LoRA learner."""
        ...

    async def close(self) -> None: ...
