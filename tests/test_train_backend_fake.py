"""The CPU backend preserves the routing and version boundaries used by the loop."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from marli.errors import ConfigError
from marli.model import load_model
from marli.policy.base import SamplingSpec, TokenPolicy
from marli.policy.scripted import ScriptedPolicy, Turn, by_role
from marli.render.fake import FakeRenderer
from marli.train.backends.fake import FakeBackend
from marli.train.backends.registry import make_backend
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec, TrainDatum

FACTORY = "test_train_backend_fake:policy_factory"


def policy_factory() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy("original", renderer, by_role(renderer, {}, default=Turn("answer")))


def datum(learner: str = "a", *, version: int = 0) -> TrainDatum:
    tokens = tuple(FakeRenderer().encode_text("abcde"))
    return TrainDatum(
        learner=learner,
        episode_id="e",
        agent_id="peer",
        role="peer",
        segment_id="s",
        session_idx=0,
        policy_version=version,
        tokens=tokens,
        logprobs=(0.0, -0.5, 0.0, -1.0),
        mask=(0.0, 1.0, 0.0, 1.0),
        advantages=(0.0, 2.0, 0.0, -0.5),
    )


async def test_routing_records_and_deterministic_loss(tmp_path: Path) -> None:
    backend = FakeBackend(FACTORY, state_dir=tmp_path)
    model = load_model("qwen3_8b")
    spec = LearnerSpec(backend="fake", learning_rate=0.003)
    a = await backend.create_learner("a", spec, model=model, seed=1)
    b = await backend.create_learner("b", spec, model=model, seed=2)
    batch = [datum(), replace(datum(), episode_id="e2")]
    ra, rb = await asyncio.gather(
        a.train_step(batch, learning_rate=0.01), b.train_step([datum("b")])
    )
    batch.clear()
    assert a.steps == [[datum(), replace(datum(), episode_id="e2")]]
    assert b.steps == [[datum("b")]]
    assert (a.calls[0].learner, a.calls[0].step, a.calls[0].learning_rate) == ("a", 1, 0.01)
    assert a.calls[0].datums == tuple(a.steps[0])
    assert b.calls[0].learning_rate == spec.learning_rate
    assert (ra.learner, ra.n_datums, ra.n_tokens, ra.n_action_tokens, ra.loss) == (
        "a",
        2,
        8,
        4,
        -3.0,
    )
    assert (rb.loss, rb.kl_sample_train) == (-1.5, 0.0)
    assert a.weights == b.weights == 1
    assert a.version == b.version == 0
    await a.train_step([datum()])
    assert a.weights == 2 and b.weights == 1
    assert a.calls[1].step == 2
    with pytest.raises(ConfigError, match="already exists"):
        await backend.create_learner("a", spec, model=model, seed=3)
    await backend.close()


@pytest.mark.parametrize("invalid", ["empty", "routing", "context", "versions", "stale"])
async def test_invalid_batch_does_not_change_weights(tmp_path: Path, invalid: str) -> None:
    model = load_model("qwen3_8b")
    if invalid == "context":
        model = replace(model, max_ctx=4)
    learner = await FakeBackend(state_dir=tmp_path).create_learner(
        "a", LearnerSpec(backend="fake"), model=model, seed=0
    )
    batch = {
        "empty": [],
        "routing": [datum("b")],
        "context": [datum()],
        "versions": [datum(), datum(version=1)],
        "stale": [datum(version=1)],
    }[invalid]
    with pytest.raises(ConfigError):
        await learner.train_step(batch)
    assert learner.steps == learner.calls == []
    assert learner.weights == 0


async def test_trainable_policy_versions_and_unique_names(tmp_path: Path) -> None:
    backend = FakeBackend(FACTORY, state_dir=tmp_path)
    a = await backend.create_learner("a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0)
    initial = a.policy()
    assert isinstance(initial, TokenPolicy)
    assert initial.trainable and initial.policy_version == 0 and initial.policy_id == "a"
    renderer = FakeRenderer()
    sample = await initial.sample(renderer.encode_text("prompt"), SamplingSpec(100), seed=42)
    assert sample.completion_ids == tuple(renderer.encode_completion("answer"))
    assert len(sample.logprobs) == len(sample.completion_ids)
    assert sample.policy_version == 0
    snap = await a.sync_sampler("run-a-v1")
    assert (snap.learner, snap.version, snap.path, snap.policy_ref) == (
        "a",
        1,
        "fake://a/run-a-v1",
        f"scripted:{FACTORY}",
    )
    policy = a.policy(policy_id="role")
    assert policy.policy_version == 1 and policy.policy_id == "role"
    assert initial.policy_version == 0
    with pytest.raises(ValueError, match="already used"):
        await a.sync_sampler("run-a-v1")
    assert a.version == 1
    assert (await a.sync_sampler("run-a-v2")).version == 2
    b = await backend.create_learner("b", LearnerSpec(), model=a.model, seed=1)
    assert (await b.sync_sampler("run-a-v1")).version == 1


async def test_policy_requires_scripted_factory(tmp_path: Path) -> None:
    learner = await FakeBackend(state_dir=tmp_path).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    with pytest.raises(ConfigError, match="policy_factory"):
        learner.policy()
    learner.policy_factory = "missing_module:factory"
    with pytest.raises(ConfigError, match="Cannot resolve"):
        learner.policy()


@pytest.mark.parametrize("with_optimizer", [True, False])
async def test_save_load_and_checkpoint_resume(tmp_path: Path, with_optimizer: bool) -> None:
    model = load_model("qwen3_8b")
    backend = FakeBackend(FACTORY, state_dir=tmp_path / "states")
    learner = await backend.create_learner("a", LearnerSpec(), model=model, seed=0)
    await learner.train_step([datum()])
    snap = await learner.sync_sampler("first")
    path = await learner.save_state("step-1")
    assert Path(path).is_relative_to(tmp_path / "states")
    await learner.train_step([datum(version=1)])
    await learner.sync_sampler("second")
    await learner.load_state(path, with_optimizer=with_optimizer)
    assert (learner.weights, learner.version) == (1, 1)
    checkpoint = Checkpoint(
        root=tmp_path / "checkpoint",
        step=1,
        run_config_hash="config",
        learners={
            "a": {
                "state": path,
                "sampler": snap.policy_ref,
                "version": 1,
                "base_model": model.name,
                "backend": "fake",
                "rank": 32,
            }
        },
    )
    checkpoint.save()
    fresh = await FakeBackend(FACTORY, state_dir=tmp_path / "fresh").create_learner(
        "a", LearnerSpec(init_from=str(checkpoint.manifest_path)), model=model, seed=0
    )
    assert (fresh.weights, fresh.version, fresh.policy().policy_version) == (1, 1, 1)
    await fresh.train_step([datum(version=1)])
    assert fresh.weights == 2


async def test_save_requires_explicit_state_directory() -> None:
    learner = await FakeBackend().create_learner(
        "a", LearnerSpec(backend="fake"), model=load_model("qwen3_8b"), seed=0
    )
    await learner.train_step([datum()])
    with pytest.raises(ConfigError, match="explicit state_dir"):
        await learner.save_state("step-1")


async def test_resume_version_comes_from_checkpoint_record(tmp_path: Path) -> None:
    model = load_model("qwen3_8b")
    learner = await FakeBackend(FACTORY, state_dir=tmp_path / "states").create_learner(
        "a", LearnerSpec(backend="fake"), model=model, seed=0
    )
    await learner.train_step([datum()])
    state = await learner.save_state("before-publishing")
    await learner.sync_sampler("published")
    checkpoint = Checkpoint(
        root=tmp_path / "ckpt",
        step=1,
        run_config_hash="h",
        learners={
            "a": {
                "state": state,
                "sampler": None,
                "version": 1,
                "base_model": model.name,
                "backend": "fake",
                "rank": 32,
            }
        },
    )
    fresh = await FakeBackend(FACTORY).create_learner(
        "a", LearnerSpec(backend="fake", init_from=str(checkpoint.save())), model=model, seed=0
    )
    assert fresh.weights == 1
    assert fresh.version == fresh.policy().policy_version == 1
    await fresh.train_step([datum(version=1)])
    assert fresh.weights == 2


def test_registry_selection_and_lazy_imports() -> None:
    assert isinstance(make_backend("fake", spend=None, policy_factory=FACTORY), FakeBackend)
    with pytest.raises(ConfigError, match="not implemented until M4"):
        make_backend("local", spend=None)
    with pytest.raises(ConfigError, match="unknown training backend"):
        make_backend("unknown", spend=None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import marli.train.backends.registry; "
            "assert not {'torch', 'tinker', 'tinker_cookbook'} & sys.modules.keys()",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr
