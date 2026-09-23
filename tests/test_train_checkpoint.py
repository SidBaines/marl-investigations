"""Checkpoint selection never substitutes a sampler for resumable training state."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from marli.errors import ConfigError
from marli.handles import load_any
from marli.model import load_model
from marli.policy.refs import parse_ref
from marli.train.checkpoint import Checkpoint, LearnerCheckpoint, resolve_checkpoint_ref


def record(step: int, **updates: object) -> LearnerCheckpoint:
    return {
        "state": f"tinker://run/weights/step-{step}",
        "sampler": f"tinker://run/sampler/step-{step}",
        "version": step,
        "base_model": "qwen3_8b",
        "backend": "tinker",
        "rank": 32,
        **updates,
    }


@pytest.fixture
def checkpoint(tmp_path: Path) -> Checkpoint:
    return Checkpoint(
        root=tmp_path,
        step=4,
        learners={"a": record(4)},
        run_config_hash="hash",
        rae_state={"protocol/peer": {"mean": 0.4}},
        data_cursor={"epoch": 1, "offset": 12},
        history=[{"step": s, "learners": {"a": record(s)}} for s in (0, 2, 4)],
    )


def test_round_trip_history_and_frozen_handle(checkpoint: Checkpoint) -> None:
    path = checkpoint.save()
    assert path.name == "checkpoint.json"
    assert json.loads(path.read_text())["kind"] == "checkpoint"
    assert Checkpoint.load(path) == checkpoint
    assert Checkpoint.load(path.parent) == checkpoint
    assert load_any(path) == checkpoint
    with pytest.raises(FrozenInstanceError):
        checkpoint.step = 8
    restored = Checkpoint.load(path)
    assert restored.history == checkpoint.history
    assert restored.rae_state == checkpoint.rae_state
    assert restored.data_cursor == checkpoint.data_cursor


def test_state_is_never_sampler(checkpoint: Checkpoint) -> None:
    assert checkpoint.require_state("a") == "tinker://run/weights/step-4"
    sampler_only = replace(checkpoint, learners={"a": record(4, state=None)})
    with pytest.raises(ConfigError, match="no resumable state"):
        sampler_only.require_state("a")
    with pytest.raises(ConfigError, match="no learner"):
        checkpoint.require_state("missing")
    local = replace(checkpoint, learners={"a": record(4, state="state.json")})
    assert local.require_state("a") == str(checkpoint.root / "state.json")


@pytest.mark.parametrize("step", [None, 0, 2, 4])
def test_sampler_at_saved_step(checkpoint: Checkpoint, step: int | None) -> None:
    expected_step = checkpoint.step if step is None else step
    expected = f"tinker:{load_model('qwen3_8b').tinker_id}#sampler=tinker://run/sampler/step-"
    assert checkpoint.policy_ref("a", step) == f"{expected}{expected_step}"
    with pytest.raises(ConfigError, match="no learner"):
        checkpoint.policy_ref("missing", step)


def test_concrete_sampler_refs_and_raw_model_id(checkpoint: Checkpoint) -> None:
    for sampler in ("scripted:example:factory", "tinker:Qwen/Qwen3-8B#sampler=tinker://s"):
        saved = replace(checkpoint, learners={"a": record(4, sampler=sampler)})
        assert saved.policy_ref("a") == sampler
    saved = replace(checkpoint, learners={"a": record(4, base_model="Qwen/Qwen3-8B")})
    assert saved.policy_ref("a") == "tinker:Qwen/Qwen3-8B#sampler=tinker://run/sampler/step-4"


def test_missing_step_sampler_and_recursive_refs(checkpoint: Checkpoint) -> None:
    with pytest.raises(ConfigError, match="no saved step"):
        checkpoint.policy_ref("a", 3)
    missing = replace(checkpoint, learners={"a": record(4, sampler=None)})
    with pytest.raises(ConfigError, match="no sampler"):
        missing.policy_ref("a")
    recursive = replace(checkpoint, learners={"a": record(4, sampler="ckpt:somewhere")})
    with pytest.raises(ConfigError, match="concrete"):
        recursive.policy_ref("a")


def test_resolve_checkpoint_and_learner_selection(checkpoint: Checkpoint) -> None:
    checkpoint.save()
    ref = parse_ref(f"ckpt:{checkpoint.root}#step=2")
    assert resolve_checkpoint_ref(ref, None) == checkpoint.policy_ref("a", 2)
    assert resolve_checkpoint_ref(ref, "a") == checkpoint.policy_ref("a", 2)
    final = parse_ref(f"ckpt:{checkpoint.manifest_path}#step=final")
    assert resolve_checkpoint_ref(final, None) == checkpoint.policy_ref("a")
    with pytest.raises(ConfigError, match="expected ckpt"):
        resolve_checkpoint_ref(parse_ref("scripted:example:factory"), None)
    with pytest.raises(ConfigError, match="no learner"):
        resolve_checkpoint_ref(ref, "missing")
    multi = replace(checkpoint, learners={"a": record(4), "b": record(4)})
    multi.save()
    with pytest.raises(ConfigError, match="specify learner"):
        resolve_checkpoint_ref(final, None)
    assert resolve_checkpoint_ref(final, "b") == multi.policy_ref("b")
    # Historical steps choose among their own learners, not the latest roster.
    assert resolve_checkpoint_ref(ref, None) == checkpoint.policy_ref("a", 2)
    replace(checkpoint, learners={}).save()
    with pytest.raises(ConfigError, match="exactly one learner"):
        resolve_checkpoint_ref(final, None)
