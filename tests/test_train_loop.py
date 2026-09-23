"""Exercise the synchronous learner barrier through real token-level episodes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from _marli_test_envs import ArithEnv

from marli.budget import SpendGuard
from marli.envs.base import Task
from marli.envs.registry import ENVS
from marli.errors import BudgetExceededError, ConfigError
from marli.eval.policies import SamplingOverrides
from marli.interact.records import read_episodes
from marli.interact.run import EpisodeSpec
from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.types import Episode, Outcome
from marli.model import load_model
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunStatus
from marli.tasks.taskset import TaskSet, write_tasks
from marli.train import loop
from marli.train.backends.base import StepResult
from marli.train.backends.fake import FakeBackend, FakeLearner
from marli.train.checkpoint import Checkpoint
from marli.train.rl import TrainRLConfig
from marli.train.types import CreditConfig, LearnerSpec, TrainDatum
from marli.verbs import run_verb


@ENVS.register("rl_arith")
def arith_factory(config: dict[str, Any], task: Task) -> ArithEnv:
    env = ArithEnv()
    env.task = task
    return env


@PROTOCOLS.register("rl_pair")
class PairProtocol(Protocol):
    name = "rl_pair"

    def __init__(self, config: None = None) -> None:
        pass

    def roles(self) -> list[RoleSpec]:
        return [RoleSpec(role, ("submit",), "Solve as {agent_id}.") for role in ("a", "b")]

    async def run(self, io: SystemIO) -> Outcome:
        handles = [
            await io.start_agent(
                role, agent_id=f"{role}0", seat_key=(role, 0), first_message=io.task.prompt
            )
            for role in ("a", "b")
        ]
        results = await io.wait(handles)
        return Outcome(results[0].submission, {r.agent_id: r.submission for r in results}, "pair")


def mixed_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        idx = int(ctx.meta.episode_id.rsplit("/e", 1)[1])
        answer = "5" if idx % 2 == 0 else "4"
        return Turn(tool_calls=(("submit", {"answer": answer}),))

    return ScriptedPolicy("mixed", renderer, from_callable(turn, renderer))


def session_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        if ctx.meta.call_index == 0:
            return Turn(tool_calls=(("end_session", {}),))
        idx = int(ctx.meta.episode_id.rsplit("/e", 1)[1])
        return Turn(tool_calls=(("submit", {"answer": "5" if idx % 2 == 0 else "4"}),))

    return ScriptedPolicy("sessions", renderer, from_callable(turn, renderer))


def rae_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        group, episode = ctx.meta.episode_id.rsplit("/e", 1)
        step = int(group.rsplit("/s", 1)[1].split("/b", 1)[0])
        correct = (int(episode) % 2 == 0, True, False, int(episode) % 2 == 0)[step]
        return Turn(tool_calls=(("submit", {"answer": "5" if correct else "4"}),))

    return ScriptedPolicy("rae", renderer, from_callable(turn, renderer))


def make_taskset(root: Path, n: int = 5) -> TaskSet:
    write_tasks(root, [Task(f"t{i}", "What is 2+3?", answer=5) for i in range(n)])
    taskset = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="train",
        kind="math",
        n=n,
        answer_format="integer",
        commit_text=True,
    )
    taskset.save()
    return taskset


def config(taskset: TaskSet, **overrides: Any) -> TrainRLConfig:
    return replace(
        TrainRLConfig(
            tasks=str(taskset.manifest_path),
            env="rl_arith",
            protocol="rl_pair",
            learners={
                name: LearnerSpec(base_model="qwen3_8b", backend="fake") for name in ("a", "b")
            },
            seating={"a": "learner:a", "b": "learner:b"},
            batch_tasks=1,
            group_size=2,
            steps=2,
            checkpoint_every=1,
        ),
        **overrides,
    )


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@dataclass
class FakeSetup:
    backends: list[FakeBackend]
    guards: list[SpendGuard]
    factory: str = "test_train_loop:mixed_policy"


@pytest.fixture
def fake_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSetup:
    setup = FakeSetup([], [])
    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "1")
    monkeypatch.setattr(loop, "load_model", lambda name: replace(load_model(name), renderer="fake"))

    def backend(name: str, *, spend: SpendGuard, **kwargs: Any) -> FakeBackend:
        instance = FakeBackend(setup.factory, state_dir=tmp_path / f"states{len(setup.backends)}")
        setup.backends.append(instance)
        setup.guards.append(spend)
        return instance

    monkeypatch.setattr(loop, "make_backend", backend)
    return setup


@pytest.mark.parametrize("shared", [False, True])
async def test_routing_versions_metrics_and_checkpoint(
    tmp_path: Path,
    fake_setup: FakeSetup,
    shared: bool,
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"))
    if shared:
        cfg.learners.pop("b")
        cfg.seating["b"] = "learner:a"
    out = tmp_path / "train"
    result = await run_verb("train rl", cfg, out=out)
    checkpoint = Checkpoint.load(out)
    assert result.status is RunStatus.FRESH
    assert checkpoint.step == 1
    assert [saved["step"] for saved in checkpoint.history] == [0, 1]
    for name, learner in fake_setup.backends[0].learners.items():
        assert learner.version == learner.weights == 2
        for version, batch in enumerate(learner.steps):
            assert {d.learner for d in batch} == {name}
            assert {d.role for d in batch} == ({"a", "b"} if shared else {name})
            assert {d.policy_version for d in batch} == {version}
            assert all(d.n_action_tokens > 0 for d in batch)
        assert Path(checkpoint.require_state(name)).is_file()
        assert checkpoint.policy_ref(name) == "scripted:test_train_loop:mixed_policy"
    metrics = rows(out / "metrics.jsonl")
    assert [row["accuracy"] for row in metrics] == [0.5, 0.5]
    assert metrics[-1]["credit"]["n_groups"] == 1
    assert metrics[-1]["calls"] == 2
    assert metrics[-1]["n_sessions"] == 2
    for record in metrics[-1]["learners"].values():
        assert record["n_datums"] == (4 if shared else 2)
        assert record["n_action_tokens"] > 0
        assert "grad_norm" in record and record["kl_sample_train"] == 0
    for step in range(2):
        episodes = list(read_episodes(out / f"rollouts/step_{step:05d}", with_tokens=True))
        assert len(episodes) == 2
        for episode, buffers in episodes:
            assert episode.group_id.startswith(f"{episode.task_id}/s{step}/b")
            assert {call.policy_version for call in episode.calls} == {step}
            assert buffers and episode.ok
    assert checkpoint.meta["provenance"]["git_commit"]
    assert checkpoint.inputs[0].resolve() == Path(cfg.tasks)
    assert json.loads((out / "progress.json").read_text())["step"] == 1
    before = (out / "metrics.jsonl").read_bytes()
    assert (await run_verb("train rl", cfg, out=out)).status is RunStatus.COMPLETE
    assert (out / "metrics.jsonl").read_bytes() == before
    assert len(fake_setup.backends) == 1


@pytest.mark.parametrize("frozen", [False, True])
async def test_recipient_subset_idle_and_frozen(
    tmp_path: Path,
    fake_setup: FakeSetup,
    frozen: bool,
) -> None:
    cfg = config(
        make_taskset(tmp_path / "tasks"), allow_idle=True, credit=CreditConfig(recipients=("a",))
    )
    if frozen:
        cfg.learners.pop("b")
        cfg.seating["b"] = "scripted:test_train_loop:mixed_policy"
        cfg.frozen_sampling = {"b": SamplingOverrides(0.5, 0.8, 3)}
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    learners = fake_setup.backends[0].learners
    assert all(d.role == "a" for step in learners["a"].steps for d in step)
    if not frozen:
        assert learners["b"].steps == [] and learners["b"].version == 0
        assert rows(result.handle.root / "metrics.jsonl")[-1]["idle_learners"] == ["b"]
        assert result.handle.learners["b"]["sampler"] is None
    else:
        saved = list(read_episodes(result.handle.root / "rollouts/step_00001", with_tokens=True))
        assert {c.policy_version for e, _ in saved for c in e.calls if c.role == "b"} == {None}


@pytest.mark.parametrize("aggregation", ["token_sum", "token_mean_per_learner", "agent_mean"])
async def test_last_session_and_loss_aggregation_reach_backend(
    tmp_path: Path,
    fake_setup: FakeSetup,
    aggregation: str,
) -> None:
    fake_setup.factory = "test_train_loop:session_policy"
    cfg = config(
        make_taskset(tmp_path / "tasks"),
        protocol="multi_session",
        protocol_config={"sessions": 2, "carry": "notes"},
        learners={"a": LearnerSpec(base_model="qwen3_8b", backend="fake")},
        seating={"solver": "learner:a"},
        steps=1,
        credit=CreditConfig(segment_credit="last", loss_agg=aggregation),
    )
    out = tmp_path / "train"
    await run_verb("train rl", cfg, out=out)
    [batch] = fake_setup.backends[0].learners["a"].steps
    assert len(batch) == 2 and {d.session_idx for d in batch} == {1}
    assert (
        sum(
            len(e.segments) for e, _ in read_episodes(out / "rollouts/step_00000", with_tokens=True)
        )
        == 4
    )
    for datum in batch:
        sign = 1 if datum.episode_id.endswith("e0") else -1
        denom = 1 if aggregation == "token_sum" else sum(d.n_action_tokens for d in batch)
        if aggregation == "agent_mean":
            denom = datum.n_action_tokens * len(batch)
        assert {a for a, mask in zip(datum.advantages, datum.mask, strict=True) if mask} == {
            sign / denom
        }


def test_data_cursor_determinism_and_epoch_wrap(tmp_path: Path) -> None:
    tasks = [Task(f"t{i}", "synthetic") for i in range(5)]
    whole, cursor = loop.take_tasks(tasks, loop.DataCursor(), 14, seed=23)
    chunks: list[Task] = []
    resumed = loop.DataCursor()
    for n in (3, 4, 1, 6):
        batch, resumed = loop.take_tasks(tasks, resumed, n, seed=23)
        chunks.extend(batch)
    assert whole == chunks and cursor == resumed == loop.DataCursor(2, 4)
    assert {t.task_id for t in whole[:5]} == {t.task_id for t in tasks}
    assert {t.task_id for t in whole[5:10]} == {t.task_id for t in tasks}
    assert whole != loop.take_tasks(tasks, loop.DataCursor(), 14, seed=24)[0]


async def test_resume_restores_cursor_rae_optimizer_and_versions(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_setup.factory = "test_train_loop:rae_policy"
    cfg = config(make_taskset(tmp_path / "tasks", 3), steps=4, credit=CreditConfig(baseline="rae"))
    original = Checkpoint.save
    out = tmp_path / "interrupted"

    def crash(checkpoint: Checkpoint) -> Path:
        path = original(checkpoint)
        if checkpoint.step == 1:
            raise RuntimeError("crash after step 1 checkpoint")
        return path

    with monkeypatch.context() as patch:
        patch.setattr(Checkpoint, "save", crash)
        with pytest.raises(RuntimeError, match="crash after step 1"):
            await run_verb("train rl", cfg, out=out)
    assert not (out / "checkpoint.json").exists()
    assert [len(learner.steps) for learner in fake_setup.backends[0].learners.values()] == [2, 2]
    first_rows = (out / "rollouts/step_00000/episodes.jsonl").read_bytes()
    partial = out / "rollouts/step_00002"
    partial.mkdir()
    (partial / "episodes.jsonl").write_text("torn partial step")
    (partial / "tokens.jsonl").write_text("orphan tokens")
    loaded: list[tuple[str, bool]] = []
    original_load = FakeLearner.load_state

    async def load(learner: FakeLearner, path: str, *, with_optimizer: bool = True) -> None:
        loaded.append((path, with_optimizer))
        await original_load(learner, path, with_optimizer=with_optimizer)

    monkeypatch.setattr(FakeLearner, "load_state", load)
    result = await run_verb("train rl", cfg, out=out)
    assert result.status is RunStatus.RESUME
    resumed = fake_setup.backends[1].learners
    assert all(len(learner.steps) == 2 and learner.weights == 4 for learner in resumed.values())
    assert all({d.policy_version for d in learner.steps[0]} == {2} for learner in resumed.values())
    assert loaded and all(with_optimizer for _, with_optimizer in loaded)
    assert (out / "rollouts/step_00000/episodes.jsonl").read_bytes() == first_rows
    assert [saved["step"] for saved in result.handle.history] == [0, 1, 2, 3]
    expected_rae = 0.5
    for reward in (1.0, 0.0, 0.5):
        expected_rae = cfg.credit.rae_gamma * expected_rae + (1 - cfg.credit.rae_gamma) * reward
    assert list(result.handle.rae_state.values()) == pytest.approx([expected_rae, expected_rae])
    assert result.handle.data_cursor == {"epoch": 1, "index": 1}
    clean = await run_verb("train rl", cfg, out=tmp_path / "uninterrupted")
    assert result.handle.rae_state == clean.handle.rae_state
    assert result.handle.data_cursor == clean.handle.data_cursor
    for name, learner in resumed.items():
        assert learner.steps == fake_setup.backends[2].learners[name].steps[2:]
    assert [row["step"] for row in rows(out / "metrics.jsonl")] == [0, 1, 2, 3]
    for step in range(4):
        directory = f"rollouts/step_{step:05d}"
        assert list(read_episodes(out / directory, with_tokens=True)) == list(
            read_episodes(clean.handle.root / directory, with_tokens=True)
        )


async def test_concurrent_training_and_bounded_completion_order(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), steps=1, concurrency=2, group_size=4)
    out = tmp_path / "train"
    original_rollout = loop.run_episode
    second_done = asyncio.Event()
    active = peak = 0
    seen_seeds: list[int] = []

    async def rollout(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen_seeds.append(spec.run_seed)
        assert spec.renderers["a"]() is not spec.renderers["a"]()
        assert not spec.sampling
        try:
            if spec.episode_idx == 0:
                await second_done.wait()
            result = await original_rollout(spec)
            if spec.episode_idx == 1:
                second_done.set()
            return result
        finally:
            active -= 1

    entered: set[str] = set()
    both = asyncio.Event()
    original_train = FakeLearner.train_step

    async def train(
        learner: FakeLearner, datums: Sequence[TrainDatum], **kwargs: Any
    ) -> StepResult:
        entered.add(learner.name)
        if len(entered) == 2:
            both.set()
        await both.wait()
        return await original_train(learner, datums, **kwargs)

    monkeypatch.setattr(loop, "run_episode", rollout)
    monkeypatch.setattr(FakeLearner, "train_step", train)
    await asyncio.wait_for(run_verb("train rl", cfg, out=out), 10)
    assert peak == 2 and active == 0 and entered == {"a", "b"}
    assert len(seen_seeds) == 4
    assert rows(out / "rollouts/step_00000/episodes.jsonl")[0]["episode_idx"] == 1


async def test_budget_checkpoints_last_complete_unsaved_step(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), steps=3, checkpoint_every=3, max_usd=1)
    original = loop.run_episode
    out = tmp_path / "train"

    async def bill(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        if "/s1/" in spec.group_id:
            fake_setup.guards[-1].charge(2, "interrupted sampling")
        return await original(spec)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_episode", bill)
        with pytest.raises(BudgetExceededError):
            await run_verb("train rl", cfg, out=out)
    saved = Checkpoint.load(out / "checkpoints/step_00000")
    assert saved.step == 0 and saved.learners["a"]["version"] == 1
    assert not (out / "checkpoint.json").exists()
    spent = json.loads((out / "progress.json").read_text())["spend"]["spent_usd"]
    assert spent >= 2
    result = await run_verb("train rl", replace(cfg, max_usd=10), out=out)
    assert result.status is RunStatus.RESUME and result.handle.step == 2
    assert result.handle.meta["spend"]["spent_usd"] == spent
    assert [row["step"] for row in rows(out / "metrics.jsonl")] == [0, 1, 2]


async def test_stale_call_fails_before_training(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = loop.run_episode

    async def stale(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        episode, buffers = await original(spec)
        return replace(
            episode, calls=tuple(replace(c, policy_version=99) for c in episode.calls)
        ), buffers

    monkeypatch.setattr(loop, "run_episode", stale)
    with pytest.raises(AssertionError, match="expected learner"):
        await run_verb("train rl", config(make_taskset(tmp_path / "tasks")), out=tmp_path / "train")
    assert all(not learner.steps for learner in fake_setup.backends[0].learners.values())


@pytest.mark.parametrize(
    "bad",
    [
        {"seating": {"a": "learner:a"}},
        {"seating": {"a": "learner:unknown", "b": "learner:b"}},
        {"frozen_sampling": {"a": SamplingOverrides(0.5)}},
        {"credit": CreditConfig(recipients=("a",))},
        {"seating": {"a": "learner:a", "b": "invalid:frozen"}},
        {
            "learners": {
                "a": LearnerSpec(base_model="missing", backend="fake"),
                "b": LearnerSpec(base_model="qwen3_8b", backend="fake"),
            }
        },
    ],
)
async def test_validation_precedes_backend_creation(
    tmp_path: Path,
    fake_setup: FakeSetup,
    bad: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        await run_verb(
            "train rl", config(make_taskset(tmp_path / "tasks"), **bad), out=tmp_path / "train"
        )
    assert fake_setup.backends == []


async def test_no_datums_does_not_step_or_sync(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A step whose credit emits nothing (e.g. every group zero-variance) must not
    # step or sync any learner, even though the rollouts contain actions.
    original = loop.assign_credit

    def no_credit(*args: Any, **kwargs: Any) -> Any:
        _, stats, rae = original(*args, **kwargs)
        return [], stats, rae

    monkeypatch.setattr(loop, "assign_credit", no_credit)
    cfg = config(make_taskset(tmp_path / "tasks"), group_size=2, steps=1)
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    assert all(
        learner.steps == [] and learner.version == 0
        for learner in fake_setup.backends[0].learners.values()
    )
    metric = rows(result.handle.root / "metrics.jsonl")[0]
    assert metric["idle_learners"] == ["a", "b"]


async def test_min_group_above_group_size_is_rejected_before_spend(
    tmp_path: Path, fake_setup: FakeSetup
) -> None:
    cfg = config(
        make_taskset(tmp_path / "tasks"), group_size=2, steps=1, credit=CreditConfig(min_group=3)
    )
    with pytest.raises(ConfigError, match="min_group"):
        await run_verb("train rl", cfg, out=tmp_path / "train")
    assert not fake_setup.backends


async def test_aggregate_training_budget_prevents_partial_update(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), steps=1, max_usd=1)
    for spec in cfg.learners.values():
        spec.backend = "tinker"
    monkeypatch.setattr(loop, "tinker_cost", lambda model, *, train: 0.75)
    out = tmp_path / "train"
    with pytest.raises(BudgetExceededError, match="all learners"):
        await run_verb("train rl", cfg, out=out)
    assert all(
        learner.weights == learner.version == 0
        for learner in fake_setup.backends[0].learners.values()
    )
    checkpoint = Checkpoint.load(out / "checkpoints/step_-0001")
    assert checkpoint.step == -1 and checkpoint.data_cursor == {"epoch": 0, "index": 0}
    result = await run_verb("train rl", replace(cfg, max_usd=2), out=out)
    assert result.status is RunStatus.RESUME and result.handle.step == 0


async def test_repeated_batch_tasks_get_distinct_groups(
    tmp_path: Path,
    fake_setup: FakeSetup,
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks", 1), batch_tasks=2, steps=1)
    await run_verb("train rl", cfg, out=tmp_path / "train")
    episodes = list(read_episodes(tmp_path / "train/rollouts/step_00000", with_tokens=False))
    groups = {episode.group_id for episode, _ in episodes}
    assert len(groups) == 2 and all(group.endswith(("/b0", "/b1")) for group in groups)
