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
from test_train_backend_tinker import sdk as sdk

from marli.budget import SpendGuard
from marli.envs.base import Task
from marli.envs.registry import ENVS
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.eval.policies import SamplingOverrides
from marli.interact.records import read_episodes
from marli.interact.run import EpisodeSpec
from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.types import Episode, Outcome
from marli.model import load_model
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunStatus
from marli.seeds import derive_seed
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
    assert metrics[-1]["agent_sessions"] == metrics[-1]["n_agents"] == 2
    assert "n_sessions" not in metrics[-1]
    assert metrics[-1]["cp_tokens"] > 0 and metrics[-1]["peak_ctx"] > 0
    assert metrics[-1]["sample_seconds"] > 0 and metrics[-1]["train_seconds"] >= 0
    assert metrics[-1]["learner_versions"] == {name: 2 for name in cfg.learners}
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
    assert len(loaded) == 2 and all(with_optimizer for _, with_optimizer in loaded)
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
        {"env": "missing_environment"},
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = loop.run_episode
    seeds: dict[str, int] = {}

    async def record_seed(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        seeds[spec.group_id] = spec.run_seed
        return await original(spec)

    monkeypatch.setattr(loop, "run_episode", record_seed)
    cfg = config(make_taskset(tmp_path / "tasks", 1), batch_tasks=2, steps=1)
    await run_verb("train rl", cfg, out=tmp_path / "train")
    episodes = list(read_episodes(tmp_path / "train/rollouts/step_00000", with_tokens=False))
    groups = {episode.group_id for episode, _ in episodes}
    assert len(groups) == 2 and all(group.endswith(("/b0", "/b1")) for group in groups)
    assert seeds == {f"t0/s0/b{slot}": derive_seed(cfg.seed, 0, slot) for slot in range(2)}
    assert len(set(seeds.values())) == 2


@pytest.mark.parametrize(
    ("failed", "max_failed_frac", "fails"),
    [(4, 1.0, True), (3, 0.5, True), (2, 0.5, False), (1, 0.0, True)],
)
async def test_failed_episodes_warn_and_enforce_step_threshold(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
    failed: int,
    max_failed_frac: float,
    fails: bool,
) -> None:
    cfg = config(
        make_taskset(tmp_path / "tasks"),
        steps=1,
        group_size=4,
        max_failed_frac=max_failed_frac,
    )
    out = tmp_path / "train"
    original = loop.run_episode

    async def fail(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        episode, buffers = await original(spec)
        if spec.episode_idx < failed:
            episode = replace(episode, ok=False, grades={})
        return episode, buffers

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_episode", fail)
        with pytest.warns(UserWarning, match=f"{failed}/4 episodes failed"):
            if fails:
                with pytest.raises(BackendError) as error:
                    await run_verb("train rl", cfg, out=out)
                assert error.value.exit_code == 5
            else:
                await run_verb("train rl", cfg, out=out)
    learners = fake_setup.backends[0].learners.values()
    if fails:
        assert all(not learner.steps and learner.version == 0 for learner in learners)
        assert rows(out / "metrics.jsonl") == []
        assert not list(out.glob("**/checkpoint.json"))
        assert json.loads((out / "progress.json").read_text())["step"] == -1
        resumed = await run_verb("train rl", cfg, out=out)
        assert resumed.status is RunStatus.RESUME
        assert resumed.handle.data_cursor == {"epoch": 0, "index": 1}
        assert [row["step"] for row in rows(out / "metrics.jsonl")] == [0]
    else:
        assert all(len(learner.steps) == 1 for learner in learners)
        assert rows(out / "metrics.jsonl")[0]["accuracy"] == 0.25


@pytest.mark.parametrize("shared", [False, True])
async def test_zero_advantage_datums_are_not_trained(
    tmp_path: Path,
    fake_setup: FakeSetup,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
) -> None:
    cfg = config(
        make_taskset(tmp_path / "tasks"),
        steps=1,
        max_usd=1,
        credit=CreditConfig(default_target="individual", drop_zero_variance=False),
    )
    for spec in cfg.learners.values():
        spec.backend = "tinker"
    reservations: list[int] = []

    def cost(model: Any, *, train: int) -> float:
        reservations.append(train)
        return 0.75

    monkeypatch.setattr(loop, "tinker_cost", cost)
    if shared:
        cfg.learners.pop("b")
        cfg.seating["b"] = "learner:a"
    original = loop.run_episode

    async def constant_worker(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        episode, buffers = await original(spec)
        return replace(episode, grades={**episode.grades, "b0": {"correct": 0.0}}), buffers

    monkeypatch.setattr(loop, "run_episode", constant_worker)
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    learners = fake_setup.backends[0].learners
    [batch] = learners["a"].steps
    assert len(batch) == 2 and {d.role for d in batch} == {"a"}
    assert reservations == [sum(len(d.tokens) - 1 for d in batch)]
    if not shared:
        assert learners["b"].steps == [] and learners["b"].version == 0
    metric = rows(result.handle.root / "metrics.jsonl")[0]
    assert metric["datums_zero_adv"] == ({"a": 2} if shared else {"a": 0, "b": 2})
    assert metric["learners"]["a"]["n_tokens"] == sum(len(d.tokens) - 1 for d in batch)
    assert metric["datums"]["a/n"] == 2


async def test_zero_advantage_checks_only_masked_positions(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = loop.build_datums

    def observation_only(*args: Any, **kwargs: Any) -> list[TrainDatum]:
        return [
            replace(d, advantages=tuple(0.0 if mask else 7.0 for mask in d.mask))
            for d in original(*args, **kwargs)
        ]

    monkeypatch.setattr(loop, "build_datums", observation_only)
    result = await run_verb(
        "train rl", config(make_taskset(tmp_path / "tasks"), steps=1), out=tmp_path / "train"
    )
    assert all(
        not learner.steps and learner.version == 0
        for learner in fake_setup.backends[0].learners.values()
    )
    metric = rows(result.handle.root / "metrics.jsonl")[0]
    assert metric["datums_zero_adv"] == {"a": 2, "b": 2}
    assert metric["learners"] == metric["datums"] == {}


async def test_custom_reward_metrics_are_captured_before_training(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(
        make_taskset(tmp_path / "tasks"), steps=1, credit=CreditConfig(reward_key="quality")
    )
    original_episode, original_train = loop.run_episode, FakeLearner.train_step
    sampled: list[Episode] = []

    async def quality(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        episode, buffers = await original_episode(spec)
        episode = replace(
            episode,
            grades={name: {"quality": g["correct"]} for name, g in episode.grades.items()},
        )
        sampled.append(episode)
        return episode, buffers

    async def train(
        learner: FakeLearner, datums: Sequence[TrainDatum], **kwargs: Any
    ) -> StepResult:
        for episode in sampled:
            episode.grades.clear()
            episode.metrics.clear()
        return await original_train(learner, datums, **kwargs)

    monkeypatch.setattr(loop, "run_episode", quality)
    monkeypatch.setattr(FakeLearner, "train_step", train)
    result = await run_verb("train rl", cfg, out=tmp_path / "train")
    metric = rows(result.handle.root / "metrics.jsonl")[0]
    assert metric["accuracy"] == 0.5
    assert metric["calls"] == metric["agent_sessions"] == metric["n_agents"] == 2


async def test_tinker_resume_restores_once_without_exporting_sampler(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch, sdk: Any
) -> None:
    from marli.train.backends.registry import make_backend

    cfg = config(make_taskset(tmp_path / "tasks"), max_usd=1)
    for spec in cfg.learners.values():
        spec.backend = "tinker"
    out = tmp_path / "train"
    original_save = Checkpoint.save

    def crash(checkpoint: Checkpoint) -> Path:
        original_save(checkpoint)
        raise RuntimeError("interrupt after checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(Checkpoint, "save", crash)
        with pytest.raises(RuntimeError, match="interrupt"):
            await run_verb("train rl", cfg, out=out)
    saved = Checkpoint.load(out / "checkpoints/step_00000")
    replace(
        saved,
        learners={
            name: {
                **record,
                "state": f"tinker://old/weights/{name}",
                "sampler": f"tinker://old/sampler_weights/{name}",
            }
            for name, record in saved.learners.items()
        },
    ).save()

    async def stop_before_sampling(*args: Any, **kwargs: Any) -> Any:
        assert {name: learner.version for name, learner in kwargs["learners"].items()} == {
            "a": 1,
            "b": 1,
        }
        raise RuntimeError("restored")

    monkeypatch.setattr(loop, "make_backend", make_backend)
    monkeypatch.setattr(loop, "_rollouts", stop_before_sampling)
    with pytest.raises(RuntimeError, match="restored"):
        await run_verb("train rl", cfg, out=out)
    service = sdk.ServiceClient.return_value
    assert [
        call.kwargs["path"]
        for call in service.create_training_client_from_state_with_optimizer_async.await_args_list
    ] == ["tinker://old/weights/a", "tinker://old/weights/b"]
    service.create_training_client_from_state_async.assert_not_awaited()
    service.create_lora_training_client_async.assert_not_awaited()
    assert [call.kwargs for call in service.create_sampling_client_async.await_args_list] == [
        {"model_path": "tinker://old/sampler_weights/a"},
        {"model_path": "tinker://old/sampler_weights/b"},
    ]
    assert len(sdk.clients) == 2 and all(not client.saves for client in sdk.clients)
    service.close.assert_awaited_once()


async def test_final_checkpoint_resume_needs_no_backend(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), steps=1)
    out = tmp_path / "train"
    original = Checkpoint.save

    def crash(checkpoint: Checkpoint) -> Path:
        original(checkpoint)
        raise RuntimeError("before finalize")

    with monkeypatch.context() as patch:
        patch.setattr(Checkpoint, "save", crash)
        with pytest.raises(RuntimeError, match="before finalize"):
            await run_verb("train rl", cfg, out=out)
    before = (out / "metrics.jsonl").read_bytes()
    result = await run_verb("train rl", cfg, out=out)
    assert result.status is RunStatus.RESUME and result.handle.step == 0
    assert len(fake_setup.backends) == 1
    assert (out / "metrics.jsonl").read_bytes() == before
    assert (out / "checkpoint.json").is_file()


async def test_resume_preserves_itemized_spend_even_over_budget(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"), max_usd=10)
    out = tmp_path / "train"
    original = loop.run_episode

    async def billed(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        guard = fake_setup.guards[-1]
        guard.charge(0.25, "sampling")
        guard.charge(0.5, "other")
        if "/s1/" in spec.group_id:
            raise BackendError("interrupted sampling")
        return await original(spec)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_episode", billed)
        with pytest.raises(BackendError, match="interrupted sampling"):
            await run_verb("train rl", cfg, out=out)
    previous = json.loads((out / "progress.json").read_text())["spend"]
    for _ in range(2):
        with pytest.raises(BudgetExceededError):
            await run_verb("train rl", replace(cfg, max_usd=0.1), out=out)
        ledger = json.loads((out / "progress.json").read_text())["spend"]
        assert ledger["by_item"] == previous["by_item"]
        assert ledger["spent_usd"] == previous["spent_usd"]
    assert len(fake_setup.backends) == 1

    async def bill_again(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        fake_setup.guards[-1].charge(0.25, "sampling")
        return await original(spec)

    monkeypatch.setattr(loop, "run_episode", bill_again)
    result = await run_verb("train rl", cfg, out=out)
    assert result.handle.meta["spend"]["by_item"] == {
        **previous["by_item"],
        "sampling": previous["by_item"]["sampling"] + 0.5,
    }
    assert result.handle.meta["spend"]["spent_usd"] == previous["spent_usd"] + 0.5


async def test_checkpoint_provenance_retains_each_attempt_and_warns_on_commit_change(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"))
    out = tmp_path / "train"
    original = Checkpoint.save
    first = {"git_commit": "first", "git_dirty": False, "host": "one"}
    second = {"git_commit": "second", "git_dirty": False, "host": "two"}

    def crash(checkpoint: Checkpoint) -> Path:
        original(checkpoint)
        raise RuntimeError("interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(loop.runlog, "provenance", lambda repo_dir=None: first)
        patch.setattr(Checkpoint, "save", crash)
        with pytest.raises(RuntimeError, match="interrupt"):
            await run_verb("train rl", cfg, out=out)
    monkeypatch.setattr(loop.runlog, "provenance", lambda repo_dir=None: second)
    with pytest.warns(UserWarning, match="commit changed across resume: first -> second"):
        result = await run_verb("train rl", cfg, out=out)
    assert result.handle.meta["history_state"]["0"]["provenance"] == first
    assert result.handle.meta["history_state"]["1"]["provenance"] == second
    assert result.handle.meta["provenance"] == second


async def test_wrong_restored_version_fails_before_sampling(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(make_taskset(tmp_path / "tasks"))
    out = tmp_path / "train"
    original_save, original_create = Checkpoint.save, FakeBackend.create_learner

    def crash(checkpoint: Checkpoint) -> Path:
        original_save(checkpoint)
        raise RuntimeError("interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(Checkpoint, "save", crash)
        with pytest.raises(RuntimeError, match="interrupt"):
            await run_verb("train rl", cfg, out=out)

    async def wrong_version(*args: Any, **kwargs: Any) -> FakeLearner:
        learner = await original_create(*args, **kwargs)
        learner.version = 999
        return learner

    monkeypatch.setattr(FakeBackend, "create_learner", wrong_version)
    with pytest.raises(AssertionError, match="restored version 999, expected 1"):
        await run_verb("train rl", cfg, out=out)
    assert not (out / "rollouts/step_00001").exists()


@ENVS.register("rl_relay_slots")
def relay_env_factory(config: dict[str, Any], task: Task) -> Any:
    from test_interact_relay import SlottedEnv

    return SlottedEnv()


def relay_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        slot = int(ctx.meta.agent_id.removeprefix("contrib"))
        index = ctx.meta.call_index
        episode_idx = int(ctx.meta.episode_id.rsplit("/e", 1)[1])
        if slot == 0:
            if index == 0:
                return Turn(tool_calls=(("ci_review", {}),))
            if index == 1:
                return Turn(tool_calls=(("bash", {"content": "shared rule"}),))
        else:
            if index == 0:
                assert "shared rule" not in ctx.prompt_text
                return Turn(tool_calls=(("bash", {}),))
            if index == 1:
                assert "shared rule" in ctx.prompt_text
                return Turn(tool_calls=(("ci_submit", {"score": 3 if episode_idx == 0 else 1}),))
        assert index == 2
        return Turn(tool_calls=(("end_session", {}),))

    return ScriptedPolicy("relay", renderer, from_callable(turn, renderer))


@pytest.mark.parametrize("target", ["individual", "team"])
async def test_relay_two_training_steps_rewards_and_grade_metrics(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from marli.train.types import CreditStats, SegmentCredit

    fake_setup.factory = "test_train_loop:relay_policy"
    cfg = config(
        make_taskset(tmp_path / "tasks"),
        protocol="relay_n4",
        env="rl_relay_slots",
        learners={"shared": LearnerSpec(base_model="qwen3_8b", backend="fake")},
        seating={"contributor": "learner:shared"},
        credit=CreditConfig(reward_key="score", reward_target={"contributor": target}),
    )
    captured: list[list[SegmentCredit]] = []
    original = loop.assign_credit

    def credit(
        episodes: Sequence[Episode], cfg: CreditConfig, ctx: loop.CreditContext,
        *, rae_state: dict[str, float],
    ) -> tuple[list[SegmentCredit], CreditStats, dict[str, float]]:
        result = original(episodes, cfg, ctx, rae_state=rae_state)
        captured.append(result[0])
        return result

    monkeypatch.setattr(loop, "assign_credit", credit)
    out = tmp_path / "train"
    result = await run_verb("train rl", cfg, out=out)
    assert result.handle.step == 1
    learner = fake_setup.backends[0].learners["shared"]
    assert learner.version == learner.weights == len(learner.steps) == 2
    assert len(captured) == 2
    for step, (credits, datums) in enumerate(zip(captured, learner.steps, strict=True)):
        assert len(credits) == len(datums) == 8
        by_agent = {(credit.episode_id, credit.agent_id): credit for credit in credits}
        for credit in credits:
            first_episode = credit.episode_id.endswith("/e0")
            own = 0 if credit.agent_id == "contrib0" else (3 if first_episode else 1)
            team = 2.25 if first_episode else 0.75
            assert credit.reward == (team if target == "team" else own)
            assert credit.advantage == credit.reward - (0.75 if first_episode else 2.25)
        for datum in datums:
            assert datum.policy_version == step and datum.session_idx == 0
            expected = by_agent[datum.episode_id, datum.agent_id].advantage
            assert all(
                advantage == (expected if mask else 0)
                for advantage, mask in zip(datum.advantages, datum.mask, strict=True)
            )
        saved = list(read_episodes(out / f"rollouts/step_{step:05d}", with_tokens=True))
        assert len(saved) == 2 and all(episode.ok for episode, _ in saved)
        assert all(len(episode.segments) == 4 for episode, _ in saved)
    metrics = rows(out / "metrics.jsonl")
    assert len(metrics) == 2
    for metric in metrics:
        assert metric["grades"] == {
            "contrib0": {"score": 0, "probed": 1, "notes_had_rule": 0, "n": 2},
            **{
                f"contrib{k}": {"score": 2, "probed": 0, "notes_had_rule": 1, "n": 2}
                for k in range(1, 4)
            },
            "_system": {"score": 1.5, "probed": 0.25, "notes_had_rule": 0.75, "n": 2},
        }
        assert metric["accuracy"] == 1.5 and metric["n_agents"] == 4
        assert metric["reward_mean"] == {"contributor": 1.5}
        assert metric["calls"] == 12 and metric["datums"]["shared/n"] == 8


def test_grade_metrics_use_successful_episodes_and_observed_components() -> None:
    from test_train_credit import Seat, episode

    first = episode(0, 1, [Seat("a", own=1), Seat("b", own=0)])
    second = episode(1, 0, [Seat("a", own=0), Seat("ungraded")])
    failed = episode(2, 99, [Seat("a", own=99), Seat("failed_only", own=99)], ok=False)
    first.grades["a"].update({"probed": 1, "diagnostic": "text"})
    second.grades["a"]["probed"] = 0
    first.grades["_system"]["sparse"] = 6
    assert loop._grade_metrics([first, second, failed]) == {
        "a": {"correct": 0.5, "probed": 0.5, "n": 2},
        "b": {"correct": 0, "n": 1},
        "ungraded": {"n": 1},
        "_system": {"correct": 0.5, "sparse": 6, "n": 2},
    }
    assert loop._grade_metrics([failed]) == {}


from test_envs_sandbox import sandbox_host as sandbox_host  # noqa: E402


def code_rules_policy() -> ScriptedPolicy:
    import re
    import shlex

    from _marli_code_fixtures import SUM_SOLUTION

    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        slot = int(ctx.meta.agent_id.removeprefix("contrib"))
        index = ctx.meta.call_index
        first_episode = ctx.meta.episode_id.endswith("/e0")
        if slot == 0 and first_episode:
            if index == 0:
                return Turn(tool_calls=(("ci_review", {}),))
            if index == 1:
                header = re.search(r"# [\w-]+: [A-Z]{3}-\d{4}", ctx.prompt_text).group()
                command = f"printf '%s' {shlex.quote(header)} > NOTES.md"
                return Turn(tool_calls=(("bash", {"command": command}),))
        else:
            if index == 0:
                return Turn(tool_calls=(("bash", {"command": "cat NOTES.md"}),))
            if index == 1:
                match = re.search(r"# [\w-]+: [A-Z]{3}-\d{4}", ctx.prompt_text)
                source = (match.group() + "\n" if match else "") + SUM_SOLUTION
                command = f"printf '%s' {shlex.quote(source)} > tasks/task_{slot + 1}/solution.py"
                return Turn(tool_calls=(("bash", {"command": command}),))
            if index == 2:
                return Turn(tool_calls=(("ci_submit", {}),))
        return Turn(tool_calls=(("end_session", {}),))

    return ScriptedPolicy("code_rules", renderer, from_callable(turn, renderer))


@pytest.mark.parametrize(
    "target,fail_grading", [("team", False), ("individual", False), ("team", True)]
)
@pytest.mark.usefixtures("sandbox_host")
def test_cli_train_code_rules_relay_and_failed_episode_guard(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], target: str, fail_grading: bool,
) -> None:
    from test_envs_code_rules import repo_task

    from marli.cli.main import main
    from marli.envs.code_fn import CodeFnEnv

    task_root = tmp_path / "tasks"
    write_tasks(task_root, [repo_task(family="header")])
    taskset = TaskSet(
        root=task_root, tasks="tasks.jsonl", source="synthetic", split="train",
        kind="code", n=1, answer_format="code_repo", commit_text=True,
    )
    taskset.save()
    fake_setup.factory = "test_train_loop:code_rules_policy"
    if fail_grading:
        original_grade = CodeFnEnv.grade

        async def fail(self: CodeFnEnv, source: str | None) -> dict[str, float]:
            if source is not None and source.startswith("# "):
                raise RuntimeError("grader uid pool exhausted")
            return await original_grade(self, source)

        monkeypatch.setattr(CodeFnEnv, "grade", fail)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(json.dumps({
        "tasks": str(taskset.manifest_path), "env": "code_rules", "protocol": "relay_n4",
        "learners": {"shared": {"base_model": "qwen3_8b", "backend": "fake"}},
        "seating": {"contributor": "learner:shared"},
        "credit": {"reward_key": "score", "reward_target": {"contributor": target}},
        "batch_tasks": 1, "group_size": 2, "steps": 1, "checkpoint_every": 1,
        "concurrency": 1, "max_usd": 1, "max_failed_frac": 0,
    }))
    out = tmp_path / "train"
    exit_code = main(["train", "rl", str(config_path), "--out", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    saved = list(read_episodes(out / "rollouts/step_00000", with_tokens=True))
    assert len(saved) == 2 and all(len(ep.agents) == 4 for ep, _ in saved)
    learner = fake_setup.backends[0].learners["shared"]
    if fail_grading:
        assert exit_code == 5 and not result["ok"]
        assert "max_failed_frac=0" in result["message"]
        assert not learner.steps
        failed = [ep for ep, _ in saved if not ep.ok]
        assert len(failed) == 1 and "grader uid pool exhausted" in str(failed[0].errors)
    else:
        assert exit_code == 0 and result["ok"] and result["kind"] == "checkpoint"
        assert Checkpoint.load(result["manifest"]).step == 0
        assert len(learner.steps) == learner.weights == 1
        assert all(ep.ok and len(ep.segments) == 4 for ep, _ in saved)
        first, second = sorted((ep for ep, _ in saved), key=lambda ep: ep.episode_idx)
        assert first.grades["_system"]["score"] == 2.25
        assert second.grades["_system"]["score"] == 1
        assert first.grades["_system"]["rule_known_at_start"] == 0.75
        assert first.grades["_system"]["rule_known_at_ci"] == 0.75
        assert first.grades["_system"]["ran_ci"] == 1
        metrics = rows(out / "metrics.jsonl")
        assert len(metrics) == 1 and metrics[0]["grades"]["_system"]["score"] == 1.625


@pytest.mark.usefixtures("sandbox_host")
def test_cli_train_relay_frozen_opener_trains_only_later_contributors(
    tmp_path: Path, fake_setup: FakeSetup, capsys: pytest.CaptureFixture[str]
) -> None:
    """A frozen opener plays slot 0; only contributors 1-3 emit datums (individual reward)."""
    from test_envs_code_rules import repo_task

    from marli.cli.main import main

    task_root = tmp_path / "tasks"
    write_tasks(task_root, [repo_task(family="header")])
    taskset = TaskSet(
        root=task_root, tasks="tasks.jsonl", source="synthetic", split="train",
        kind="code", n=1, answer_format="code_repo", commit_text=True,
    )
    taskset.save()
    fake_setup.factory = "test_train_loop:code_rules_policy"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(json.dumps({
        "tasks": str(taskset.manifest_path), "env": "code_rules", "protocol": "relay_n4",
        "protocol_config": {"opener_role": "opener"},
        "learners": {"shared": {"base_model": "qwen3_8b", "backend": "fake"}},
        "seating": {
            "opener": "scripted:test_train_loop:code_rules_policy",
            "contributor": "learner:shared",
        },
        "credit": {"reward_key": "score", "reward_target": {"contributor": "individual"}},
        "batch_tasks": 1, "group_size": 2, "steps": 1, "checkpoint_every": 1,
        "concurrency": 1, "max_usd": 1, "max_failed_frac": 0,
    }))
    out = tmp_path / "train"
    assert main(["train", "rl", str(config_path), "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["ok"]
    saved = list(read_episodes(out / "rollouts/step_00000", with_tokens=True))
    assert len(saved) == 2
    for episode, _ in saved:
        roles = {agent.agent_id: agent.role for agent in episode.agents}
        assert roles == {"contrib0": "opener", "contrib1": "contributor",
                         "contrib2": "contributor", "contrib3": "contributor"}
        opener_calls = [c for c in episode.calls if c.agent_id == "contrib0"]
        assert opener_calls and all(c.policy_version is None for c in opener_calls)
        assert set(episode.grades) == {"_system", "contrib0", "contrib1", "contrib2", "contrib3"}
    learner = fake_setup.backends[0].learners["shared"]
    datums = [d for step in learner.steps for d in step]
    assert datums and {d.role for d in datums} == {"contributor"}
    assert "contrib0" not in {d.agent_id for d in datums}
