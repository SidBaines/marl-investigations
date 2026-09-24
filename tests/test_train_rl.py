"""Pin CLI identity and the boundary between checkpoint state and eval samplers."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_train_loop import FakeSetup, config, make_taskset
from test_train_loop import fake_setup as fake_setup

from marli import verbs
from marli.cli.main import main
from marli.config import compose, config_hash, save
from marli.errors import ConfigError, DirtyTreeError
from marli.eval.policies import PolicySpec, SamplingOverrides
from marli.eval.rollout import RolloutConfig
from marli.interact.records import read_episodes
from marli.model import load_model
from marli.policy.base import CallMeta, SamplingSpec
from marli.policy.refs import parse_ref
from marli.policy.resolve import resolve_policy
from marli.render.fake import FakeRenderer
from marli.train import loop
from marli.train.checkpoint import Checkpoint, resolve_checkpoint_ref
from marli.train.rl import TrainRLConfig
from marli.train.types import CreditConfig, LearnerSpec
from marli.verbs import run_verb


def test_cli_yaml_one_line_complete_noop(
    tmp_path: Path,
    fake_setup: FakeSetup,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"))
    path = save(cfg, tmp_path / "train.yaml")
    composed = compose(TrainRLConfig, path)
    assert config_hash(composed) == config_hash(cfg)
    assert composed.learners == cfg.learners
    out = tmp_path / "train"
    args = ["train", "rl", str(path), "--out", str(out)]
    assert main(args) == 0
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1
    first = json.loads(output.out)
    assert first["ok"] and first["kind"] == "checkpoint" and first["status"] == "fresh"
    assert first["manifest"] == str(out / "checkpoint.json")
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert len(fake_setup.backends) == 1


def test_cli_budget_exit_four_then_resume(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), max_usd=1, steps=2)
    path = save(cfg, tmp_path / "train.yaml")
    out = tmp_path / "train"
    original = loop.run_episode

    async def billed(spec: Any) -> Any:
        if "/s1/" in spec.group_id:
            fake_setup.guards[-1].charge(2, "sample")
        return await original(spec)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_episode", billed)
        assert main(["train", "rl", str(path), "--out", str(out)]) == 4
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1 and json.loads(output.out)["exit_code"] == 4
    assert not (out / "checkpoint.json").exists()
    assert Checkpoint.load(out / "checkpoints/step_00000").step == 0
    assert main(["train", "rl", str(path), "max_usd=10", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "resume"
    assert Checkpoint.load(out).step == 1


async def test_credit_warnings_returned_by_verb(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pytest replaces the process-wide hook used by run_verb's collector.
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    cfg = config(make_taskset(tmp_path / "tasks"), credit=CreditConfig(segment_credit="last"))
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    assert any("segment credit has no effect" in warning for warning in result.warnings)


async def test_dirty_tree_refused_before_backend(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marli.runlog import GitInfo

    monkeypatch.delenv("MARLI_ALLOW_DIRTY")
    monkeypatch.setattr(loop.runlog, "git_info", lambda repo_dir=None: GitInfo("commit", True))
    with pytest.raises(DirtyTreeError):
        await run_verb("train rl", config(make_taskset(tmp_path / "tasks")), out=tmp_path / "train")
    assert not fake_setup.backends


async def test_tinker_context_checked_before_backend(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = replace(load_model("qwen3_8b"), max_ctx=1024, tinker_max_ctx=1024)
    monkeypatch.setattr(loop, "load_model", lambda name: model)
    cfg = config(make_taskset(tmp_path / "tasks"), max_usd=1)
    for learner in cfg.learners.values():
        learner.backend = "tinker"
    with pytest.raises(ConfigError, match="ctx.max_ctx"):
        await run_verb("train rl", cfg, out=tmp_path / "train")
    assert not fake_setup.backends


def test_config_runtime_fields_do_not_change_identity() -> None:
    cfg = TrainRLConfig()
    assert config_hash(cfg) == config_hash(
        replace(cfg, max_usd=9, concurrency=3, base_url="http://localhost:1234")
    )
    assert config_hash(cfg) != config_hash(replace(cfg, group_size=2))
    assert config_hash(cfg) != config_hash(replace(cfg, seed=1))
    assert TrainRLConfig().learners is not cfg.learners
    with pytest.raises(ConfigError):
        compose(TrainRLConfig, overrides=["unknown=1"])
    for changes in (
        {"steps": 0},
        {"group_size": 0},
        {"checkpoint_every": 0},
        {"concurrency": 0},
        {"schedule": "other"},
    ):
        with pytest.raises(ConfigError):
            replace(cfg, **changes)
    with pytest.raises(ConfigError):
        SamplingOverrides(temperature=-1)


async def test_ckpt_resolves_sampler_and_eval_accepts_concrete_ref(
    tmp_path: Path,
    fake_setup: FakeSetup,
) -> None:
    taskset = make_taskset(tmp_path / "tasks")
    cfg = config(taskset)
    cfg.learners.pop("b")
    cfg.seating["b"] = "learner:a"
    out = tmp_path / "train"
    await run_verb("train rl", cfg, out=out)
    for suffix in ("", "#step=0", "#step=final"):
        policy = await resolve_policy(
            f"ckpt:{out}{suffix}", policy_id="eval", trainable=False, renderer_name="fake"
        )
        assert not policy.trainable
        sample = await policy.sample(
            FakeRenderer().encode_text("prompt"),
            SamplingSpec(1000),
            seed=1,
            meta=CallMeta(episode_id="test/e0"),
        )
        assert FakeRenderer().parse(sample.completion_ids).tool_calls[0].arguments == {
            "answer": "5"
        }
    concrete = resolve_checkpoint_ref(parse_ref(f"ckpt:{out}"), None)
    assert concrete == "scripted:test_train_loop:mixed_policy"
    result = await run_verb(
        "eval rollout",
        RolloutConfig(
            tasks=str(taskset.manifest_path),
            env="rl_arith",
            max_tasks=1,
            policies={"trained": PolicySpec(concrete)},
            seating={"solver": "trained"},
        ),
        out=tmp_path / "eval",
    )
    [(episode, _)] = read_episodes(result.handle.root, with_tokens=False)
    assert episode.grades["_system"]["correct"] == 1


async def test_ckpt_multiple_learners_error_names_them(
    tmp_path: Path,
    fake_setup: FakeSetup,
) -> None:
    out = tmp_path / "train"
    await run_verb("train rl", config(make_taskset(tmp_path / "tasks")), out=out)
    with pytest.raises(ConfigError, match="learners: .*'a'.*'b'"):
        await resolve_policy(f"ckpt:{out}", policy_id="eval", trainable=False, renderer_name="fake")


async def test_eval_rollout_with_checkpoint_seat(tmp_path: Path, fake_setup: FakeSetup) -> None:
    taskset = make_taskset(tmp_path / "tasks")
    cfg = config(taskset)
    cfg.learners.pop("b")
    cfg.seating["b"] = "learner:a"
    out = tmp_path / "train"
    await run_verb("train rl", cfg, out=out)
    result = await run_verb(
        "eval rollout",
        RolloutConfig(
            tasks=str(taskset.manifest_path),
            env="rl_arith",
            max_tasks=1,
            policies={"trained": PolicySpec(f"ckpt:{out}")},
            seating={"solver": "trained"},
        ),
        out=tmp_path / "eval",
    )
    assert result.handle.n == 1


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_failed_fraction_validation(fraction: float) -> None:
    with pytest.raises(ConfigError, match="max_failed_frac"):
        TrainRLConfig(max_failed_frac=fraction)


@pytest.mark.parametrize("backend", ["tinker", "local"])
def test_paid_learners_require_budget(backend: str) -> None:
    with pytest.raises(ConfigError, match="paid learners require max_usd"):
        TrainRLConfig(learners={"a": LearnerSpec(backend=backend)})


async def test_rl_rejects_cross_entropy_before_backend(
    tmp_path: Path, fake_setup: FakeSetup
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"))
    cfg.learners["a"].loss = "cross_entropy"
    with pytest.raises(ConfigError, match="use train sft"):
        await run_verb("train rl", cfg, out=tmp_path / "train")
    assert not fake_setup.backends


@pytest.mark.parametrize("missing_manifest", [False, True])
async def test_init_from_validation_precedes_all_backends(
    tmp_path: Path, fake_setup: FakeSetup, missing_manifest: bool
) -> None:
    path = tmp_path / "warm"
    if not missing_manifest:
        Checkpoint(root=path, step=0, learners={}, run_config_hash="warm").save()
    cfg = config(make_taskset(tmp_path / "tasks"))
    cfg.learners["b"].init_from = str(path)
    message = "no checkpoint" if missing_manifest else "no learner 'b'"
    with pytest.raises(ConfigError, match=message):
        await run_verb("train rl", cfg, out=tmp_path / "train")
    assert not fake_setup.backends


@pytest.mark.parametrize("suffix", ["#learner=b", "#step=0&learner=b"])
async def test_frozen_training_seat_selects_checkpoint_learner(
    tmp_path: Path, fake_setup: FakeSetup, suffix: str
) -> None:
    taskset = make_taskset(tmp_path / "tasks")
    warm = tmp_path / "warm"
    await run_verb("train rl", config(taskset, steps=1), out=warm)
    cfg = config(taskset, steps=1)
    cfg.learners.pop("b")
    cfg.seating["b"] = f"ckpt:{warm}{suffix}"
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    assert set(result.handle.learners) == {"a"}
    assert all(
        c.policy_version is None
        for e, _ in read_episodes(result.handle.root / "rollouts/step_00000", with_tokens=False)
        for c in e.calls
        if c.role == "b"
    )


async def test_training_provenance_uses_marli_checkout_outside_cwd(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marli
    from marli.runlog import GitInfo

    expected = Path(marli.__file__).resolve().parent
    checked: list[Path | None] = []

    def git_info(repo_dir: Path | None = None) -> GitInfo:
        checked.append(repo_dir)
        return GitInfo("checkout", repo_dir != expected)

    cfg = config(make_taskset(tmp_path / "tasks"), steps=1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARLI_ALLOW_DIRTY")
    monkeypatch.setattr(loop.runlog, "git_info", git_info)
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    assert expected in checked
    assert result.handle.meta["provenance"]["git_commit"] == "checkout"
    assert result.handle.meta["provenance"]["git_dirty"] is False


def test_all_failed_cli_returns_backend_error(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), steps=1)
    original = loop.run_episode

    async def failed(spec: Any) -> Any:
        episode, buffers = await original(spec)
        return replace(episode, ok=False, grades={}), buffers

    monkeypatch.setattr(loop, "run_episode", failed)
    path = save(cfg, tmp_path / "config.yaml")
    with pytest.warns(UserWarning, match="2/2 episodes failed"):
        assert main(["train", "rl", str(path), "--out", str(tmp_path / "train")]) == 5
    output = json.loads(capsys.readouterr().out)
    assert not output["ok"] and output["exit_code"] == 5
