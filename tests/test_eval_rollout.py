"""Durable rows, bounded work and per-seat controls are rollout's public contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _marli_test_envs import ArithEnv

from marli.budget import SpendGuard
from marli.cli.main import main
from marli.config import save
from marli.envs.base import Task
from marli.envs.registry import ENVS
from marli.errors import ConfigError
from marli.eval import policies as policies_module
from marli.eval import rollout as rollout_module
from marli.eval.policies import PolicySpec, SamplingOverrides, build_policies
from marli.eval.rollout import EpisodeSet, RolloutConfig
from marli.interact.records import read_episodes
from marli.interact.run import EpisodeSpec
from marli.interact.types import Episode
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.tasks.taskset import TaskSet, write_tasks
from marli.verbs import run_verb


@ENVS.register("eval_arith")
def arith_factory(config: dict[str, Any], task: Task) -> ArithEnv:
    env = ArithEnv()
    env.task = task
    return env


def correct_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy(
        "test",
        renderer,
        from_callable(lambda ctx: Turn(tool_calls=(("submit", {"answer": "05"}),)), renderer),
    )


def wrong_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy(
        "wrong",
        renderer,
        from_callable(lambda ctx: Turn(tool_calls=(("submit", {"answer": "4"}),)), renderer),
    )


def mixed_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        index = int(ctx.meta.episode_id.rsplit("/e", 1)[1])
        return Turn(tool_calls=(("submit", {"answer": ("05", "5", "4")[index % 3]}),))

    return ScriptedPolicy("mixed", renderer, from_callable(turn, renderer))


def make_taskset(root: Path, n: int = 4) -> TaskSet:
    tasks = [Task(f"t{index}", "What is 2 + 3?", answer=5) for index in range(n)]
    write_tasks(root, tasks)
    handle = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="arithmetic",
        split="test",
        kind="math",
        n=n,
        answer_format="integer",
        commit_text=True,
    )
    handle.save()
    return handle


def rollout_config(taskset: TaskSet, **overrides: Any) -> RolloutConfig:
    cfg = RolloutConfig(
        tasks=str(taskset.manifest_path),
        env="eval_arith",
        policies={"test": PolicySpec("scripted:test_eval_rollout:correct_policy", renderer="fake")},
        seating={"solver": "test"},
    )
    return replace(cfg, **overrides)


def json_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_rollout_resume_incremental_crash_and_token_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taskset = make_taskset(tmp_path / "tasks")
    cfg = rollout_config(taskset, concurrency=1, record_tokens=True, episodes_per_task=2)
    original = RunDir.append_row
    out = tmp_path / "rollout"

    def crash(run: RunDir, rel: str, row: Mapping[str, Any]) -> None:
        if rel == "episodes.jsonl" and len(json_rows(out / rel)) == 3:
            raise RuntimeError("simulated crash after token append")
        original(run, rel, row)

    with monkeypatch.context() as patch:
        patch.setattr(RunDir, "append_row", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            await run_verb("eval rollout", cfg, out=out)
    completed = json_rows(out / "episodes.jsonl")
    assert len(completed) == 3 and len(json_rows(out / "tokens.jsonl")) == 4
    assert not (out / "episodes.json").exists()
    assert json.loads((out / "progress.json").read_text())["done"] == 3
    with (out / "episodes.jsonl").open("ab") as stream:
        stream.write(b'{"episode_id": "torn')
    result = await run_verb("eval rollout", replace(cfg, concurrency=2), out=out)
    assert result.status is RunStatus.RESUME
    rows = json_rows(out / "episodes.jsonl")
    assert rows[:3] == completed
    assert len(rows) == len({row["episode_id"] for row in rows}) == 8
    tokens = json_rows(out / "tokens.jsonl")
    assert len(tokens) == len({row["episode_id"] for row in tokens}) == 8
    episodes = list(read_episodes(out, with_tokens=True))
    assert all(episode.ok and buffers for episode, buffers in episodes)
    assert result.handle.summary() == {"n": 8, "n_failed": 0, "cost_usd": 0}
    handle = EpisodeSet.load(out)
    assert handle.inputs[0].resolve() == taskset.manifest_path
    assert handle.protocol == "single" and "system_prompt" in handle.protocol_config
    assert handle.usage["completion_tokens"] > 0
    before = (out / "episodes.jsonl").read_bytes()
    assert (await run_verb("eval rollout", cfg, out=out)).status is RunStatus.COMPLETE
    assert (out / "episodes.jsonl").read_bytes() == before


async def test_concurrency_and_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 6), concurrency=2)
    original = rollout_module.run_episode
    active = peak = 0
    second_done = asyncio.Event()
    out = tmp_path / "rollout"

    async def run(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if spec.task.task_id == "t0":
                await second_done.wait()
                while not json_rows(out / "episodes.jsonl"):
                    await asyncio.sleep(0)
                assert json_rows(out / "episodes.jsonl")[0]["task_id"] == "t1"
            result = await original(spec)
            if spec.task.task_id == "t1":
                second_done.set()
            return result
        finally:
            active -= 1

    monkeypatch.setattr(rollout_module, "run_episode", run)
    await asyncio.wait_for(run_verb("eval rollout", cfg, out=out), timeout=5)
    assert peak == 2 and active == 0
    assert len(json_rows(out / "episodes.jsonl")) == 6


async def test_sampling_overrides_and_fresh_renderer_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1))
    cfg.policies["test"].sampling = SamplingOverrides(0.7, 0.8, 5)
    policies, renderers = await build_policies(cfg.policies, spend=SpendGuard(None))
    assert renderers["test"]() is not renderers["test"]()
    assert renderers["test"]().name == "fake"
    assert not policies["test"].trainable

    async def built(*args: Any, **kwargs: Any) -> Any:
        return policies, renderers

    monkeypatch.setattr(rollout_module, "build_policies", built)
    result = await run_verb("eval rollout", cfg, out=tmp_path / "out")
    assert result.handle.n == 1
    assert result.handle.meta["policy_specs"]["test"]["sampling"]["temperature"] == 0.7
    (call,) = policies["test"].calls
    assert (call.spec.temperature, call.spec.top_p, call.spec.top_k) == (0.7, 0.8, 5)


def test_budget_exit_four_preserves_inflight_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 6), concurrency=2, max_usd=1)
    original = rollout_module.run_episode
    original_build = rollout_module.build_policies
    guard: SpendGuard | None = None
    launched: list[str] = []

    async def build(specs: dict[str, PolicySpec], *, spend: SpendGuard) -> Any:
        nonlocal guard
        guard = spend
        return await original_build(specs, spend=spend)

    async def spend_then_run(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        launched.append(spec.task.task_id)
        if spec.task.task_id == "t0":
            await asyncio.sleep(0)
            assert guard is not None
            guard.charge(2, "simulated backend bill")
        return await original(spec)

    monkeypatch.setattr(rollout_module, "build_policies", build)
    monkeypatch.setattr(rollout_module, "run_episode", spend_then_run)
    config_path = save(cfg, tmp_path / "cfg.yaml")
    out = tmp_path / "rollout"
    assert main(["eval", "rollout", str(config_path), "--out", str(out)]) == 4
    output = capfd.readouterr()
    assert len(output.out.splitlines()) == 1
    assert json.loads(output.out)["exit_code"] == 4
    assert launched == ["t0", "t1"]
    assert [row["task_id"] for row in json_rows(out / "episodes.jsonl")] == ["t1"]
    assert json.loads((out / "progress.json").read_text())["spend"]["spent_usd"] == 2
    assert not (out / "episodes.json").exists()
    monkeypatch.setattr(rollout_module, "run_episode", original)
    assert main(["eval", "rollout", str(config_path), "max_usd=3", "--out", str(out)]) == 0
    capfd.readouterr()
    assert EpisodeSet.load(out).cost_usd == 2
    assert len(json_rows(out / "episodes.jsonl")) == 6


async def test_build_model_defaults_and_spend_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    from marli.model import load_model

    model = load_model("qwen3_8b")
    seen: dict[str, Any] = {}

    async def resolve(ref: Any, **kwargs: Any) -> ScriptedPolicy:
        seen.update(kwargs)
        policy = correct_policy()
        policy.renderer_name = kwargs["renderer_name"]
        return policy

    monkeypatch.setattr(policies_module, "resolve_policy", resolve)
    spend = SpendGuard(1)
    _, renderers = await build_policies({"model": PolicySpec(f"tinker:{model.hf_id}")}, spend=spend)
    assert seen["spend"] is spend and seen["model"] == model
    assert seen["renderer_name"] == model.renderer
    assert renderers["model"].keywords["hf_id"] == model.hf_id


@pytest.mark.parametrize("overrides", [{"trainable": True}, {"renderer": "missing"}])
def test_invalid_policies(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        PolicySpec("scripted:test_eval_rollout:correct_policy", **overrides)


async def test_unseated_roles_rejected_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks"), seating={})

    async def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("should validate before building paid clients")

    monkeypatch.setattr(rollout_module, "build_policies", forbidden)
    with pytest.raises(ConfigError, match="unseated"):
        await run_verb("eval rollout", cfg, out=tmp_path / "out")


async def test_empty_rollout_and_failed_episode_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taskset = make_taskset(tmp_path / "tasks", 1)
    cfg = rollout_config(taskset)
    original = rollout_module.run_episode

    async def failed(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        episode, tokens = await original(spec)
        return replace(episode, ok=False, errors=("test failure",)), tokens

    monkeypatch.setattr(rollout_module, "run_episode", failed)
    result = await run_verb("eval rollout", cfg, out=tmp_path / "failed")
    assert result.handle.summary()["n_failed"] == 1
    empty = await run_verb("eval rollout", replace(cfg, max_tasks=0), out=tmp_path / "empty")
    assert empty.handle.n == 0 and empty.handle.file("episodes").read_text() == ""


async def test_pause_after_k_tasks_then_resume_keeps_sampled_episodes(tmp_path: Path) -> None:
    taskset = make_taskset(tmp_path / "tasks", 4)
    cfg = rollout_config(taskset, episodes_per_task=2, concurrency=3)
    out = tmp_path / "rollout"
    paused = await run_verb("eval rollout", replace(cfg, stop_after_tasks=1), out=out)
    assert paused.status is RunStatus.FRESH
    assert paused.handle.n == 2 and paused.handle.meta["paused"]
    assert (paused.handle.meta["n_tasks_sampled"], paused.handle.meta["n_tasks"]) == (1, 4)
    assert any("paused after 1 of 4 tasks" in warning for warning in paused.warnings)
    first = (out / "episodes.jsonl").read_bytes()
    assert {row["task_id"] for row in json_rows(out / "episodes.jsonl")} == {"t0"}
    # The same or a smaller pause point is already satisfied; a larger one extends the run.
    for stop in (1, 0):
        again = await run_verb("eval rollout", replace(cfg, stop_after_tasks=stop), out=out)
        assert again.status is RunStatus.COMPLETE and again.handle.n == 2
    more = await run_verb("eval rollout", replace(cfg, stop_after_tasks=2), out=out)
    assert more.status is not RunStatus.COMPLETE and more.handle.n == 4
    assert more.config_hash == paused.config_hash
    rest = await run_verb("eval rollout", cfg, out=out)
    assert rest.status is not RunStatus.COMPLETE
    rows = json_rows(out / "episodes.jsonl")
    assert (out / "episodes.jsonl").read_bytes().startswith(first)
    assert len(rows) == len({row["episode_id"] for row in rows}) == 8
    assert not rest.handle.meta["paused"] and rest.handle.meta["n_tasks_sampled"] == 4
    assert (await run_verb("eval rollout", cfg, out=out)).status is RunStatus.COMPLETE


def test_invalid_stop_after_tasks(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="stop_after_tasks"):
        rollout_config(make_taskset(tmp_path / "tasks"), stop_after_tasks=-1).__post_init__()
