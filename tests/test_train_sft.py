"""Exercise SFT batching, crash recovery, and checkpoint consumers without remote compute."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from test_data_sft import build_set, save_teacher, teacher_episode
from test_train_backend_tinker import (
    TensorData,
    TrainingClient,
    sdk,  # noqa: F401 -- shared fake SDK fixture
)

from marli.budget import SpendGuard
from marli.cli.main import main
from marli.data import sft as data_sft
from marli.data.sft import SFTSet, read_datums
from marli.errors import BudgetExceededError, ConfigError, DirtyTreeError
from marli.eval.policies import PolicySpec, build_policies, resolve_spec
from marli.model import ModelSpec, load_model
from marli.policy.base import CallMeta, SamplingSpec
from marli.render.fake import FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.train import loop, sft
from marli.train.backends.base import StepResult
from marli.train.backends.fake import FakeBackend, FakeLearner
from marli.train.backends.registry import make_backend
from marli.train.backends.tinker import TinkerBackend
from marli.train.checkpoint import Checkpoint
from marli.train.loop import DataCursor
from marli.train.rl import TrainRLConfig
from marli.train.sft import TrainSFTConfig, batches
from marli.train.types import CreditConfig, LearnerSpec, TrainDatum
from marli.verbs import run_verb


@dataclass(frozen=True)
class Setup:
    backends: list[FakeBackend]
    load_model: Callable[[str], ModelSpec]


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> Setup:
    def model(name: str) -> ModelSpec:
        return replace(load_model(name), renderer="fake")

    instances: list[FakeBackend] = []

    def backend(name: str, *, spend: SpendGuard, base_url: str | None, **kw: Any) -> FakeBackend:
        assert name == "fake"
        instance = FakeBackend(**kw)
        instances.append(instance)
        return instance

    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "1")
    monkeypatch.setattr(sft, "load_model", model)
    monkeypatch.setattr(data_sft, "load_model", model)
    monkeypatch.setattr(sft, "make_backend", backend)
    return Setup(instances, model)


async def training_data(root: Path, n: int = 7) -> SFTSet:
    source = save_teacher(root / "teacher", [await teacher_episode(i) for i in range(n)])
    return await build_set(root / "sft", source)


def config(data: SFTSet, **kw: Any) -> TrainSFTConfig:
    length = max(len(d.tokens) - 1 for d in read_datums(data))
    return replace(
        TrainSFTConfig(
            data=str(data.root),
            learner=LearnerSpec(base_model="qwen3_8b", backend="fake", loss="cross_entropy"),
            epochs=2,
            batch_tokens=2 * length,
            checkpoint_every=2,
            seed=13,
        ),
        **kw,
    )


def identities(steps: Sequence[Sequence[TrainDatum]]) -> list[list[str]]:
    return [[d.episode_id for d in batch] for batch in steps]


def small_datum(index: int, n_tokens: int) -> TrainDatum:
    tokens = tuple(FakeRenderer().encode_text("x" * (n_tokens + 1)))
    mask = (0.0,) + (1.0,) * (n_tokens - 1)
    zeros = (0.0,) * n_tokens
    return TrainDatum("student", f"e{index}", "a", "solver", "g0", 0, 0, tokens, zeros, mask, zeros)


def test_greedy_packing_seeded_epochs_and_cursor() -> None:
    datums = [small_datum(i, n) for i, n in enumerate((4, 6, 5, 3, 7, 4))]
    args = {"epochs": 3, "batch_tokens": 11, "seed": 5}
    result = list(batches(datums, DataCursor(), **args))
    assert any(len(batch) > 1 for batch, _ in result)
    for i, (batch, cursor) in enumerate(result):
        size = sum(len(d.tokens) - 1 for d in batch)
        assert size <= 11
        if cursor.index and i + 1 < len(result):
            assert size + len(result[i + 1][0][0].tokens) - 1 > 11
    permutations: list[list[str]] = [[]]
    for batch, cursor in result:
        permutations[-1].extend(d.episode_id for d in batch)
        if cursor.index == 0:
            permutations.append([])
    assert all(sorted(p) == [f"e{i}" for i in range(6)] for p in permutations[:-1])
    assert permutations[0] != permutations[1]
    assert list(batches(datums, DataCursor(), **args)) == result
    assert list(batches(datums, DataCursor(), **{**args, "seed": 6})) != result
    assert list(batches(datums, result[2][1], **args)) == result[3:]
    with pytest.raises(ConfigError, match="batch_tokens"):
        list(batches(datums, DataCursor(), epochs=1, batch_tokens=2, seed=0))


@pytest.mark.parametrize("loss_agg", ["token_sum", "mean_per_token"])
async def test_training_weights_metrics_and_checkpoints(
    tmp_path: Path,
    setup: Setup,
    loss_agg: str,
) -> None:
    data = await training_data(tmp_path)
    cfg = config(data, loss_agg=loss_agg)
    result = await run_verb("train sft", cfg, out=tmp_path / "train")
    checkpoint = Checkpoint.load(result.handle.root)
    learner = setup.backends[0].learners["student"]
    originals = {d.episode_id: d for d in read_datums(data)}
    expected = list(
        batches(
            list(originals.values()),
            DataCursor(),
            epochs=2,
            batch_tokens=cfg.batch_tokens,
            seed=cfg.seed,
        )
    )
    assert identities(learner.steps) == identities([b for b, _ in expected])
    metrics = [
        json.loads(row) for row in (checkpoint.root / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == len(learner.steps) == 8
    for batch, row in zip(learner.steps, metrics, strict=True):
        n_action = sum(originals[d.episode_id].n_action_tokens for d in batch)
        scale = 1 / n_action if loss_agg == "mean_per_token" else 1
        for d in batch:
            original = originals[d.episode_id]
            assert d.tokens == original.tokens
            assert d.mask == tuple(m * scale for m in original.mask)
            assert set(d.logprobs) == set(d.advantages) == {0.0}
        assert row["loss"] == pytest.approx(-n_action * scale)
        assert row["n_tokens"] == sum(len(d.tokens) - 1 for d in batch)
        assert row["n_action_tokens"] == n_action
        assert row["tokens/sec"] > 0
    assert checkpoint.step == 7 and checkpoint.data_cursor == {"epoch": 2, "index": 0}
    assert [s["step"] for s in checkpoint.history] == [1, 3, 5, 7]
    assert checkpoint.meta["provenance"]["git_commit"]
    assert checkpoint.inputs[0].sha256 == data.sha256()
    assert Path(checkpoint.require_state("student")).is_file()
    assert learner.weights == 8 and learner.version == 4
    for saved in checkpoint.history:
        assert checkpoint.policy_ref("student", saved["step"]).startswith("scripted:")
    before = (checkpoint.root / "metrics.jsonl").read_bytes()
    assert (await run_verb("train sft", cfg, out=checkpoint.root)).status is RunStatus.COMPLETE
    assert (checkpoint.root / "metrics.jsonl").read_bytes() == before
    assert len(setup.backends) == 1


async def test_resume_repeats_only_unsaved_batches(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path)
    cfg = config(data)
    baseline = await run_verb("train sft", cfg, out=tmp_path / "baseline")
    expected = setup.backends[0].learners["student"].steps
    original = FakeLearner.train_step

    async def crash(learner: FakeLearner, datums: Sequence[TrainDatum], **kw: Any) -> StepResult:
        result = await original(learner, datums, **kw)
        if len(learner.steps) == 4:
            raise RuntimeError("lost optimizer response")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(FakeLearner, "train_step", crash)
        with pytest.raises(RuntimeError, match="lost optimizer response"):
            await run_verb("train sft", cfg, out=tmp_path / "resume")
    assert not (tmp_path / "resume/checkpoint.json").exists()
    with (tmp_path / "resume/metrics.jsonl").open("ab") as stream:
        stream.write(b'{"step":')
    recovered = await run_verb("train sft", cfg, out=tmp_path / "resume")
    assert recovered.status is RunStatus.RESUME
    learner = setup.backends[-1].learners["student"]
    assert identities(learner.steps) == identities(expected[2:])
    assert learner.weights == len(expected)
    assert recovered.handle.data_cursor == baseline.handle.data_cursor
    assert [r["step"] for r in recovered.handle.history] == [1, 3, 5, 7]
    metrics = [
        json.loads(line) for line in (tmp_path / "resume/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in metrics] == list(range(8))


async def test_resume_after_final_checkpoint_before_manifest(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path, n=1)
    cfg = config(data, epochs=1)
    with monkeypatch.context() as patch:

        def crash(*args: Any) -> None:
            raise RuntimeError("manifest crash")

        patch.setattr(RunDir, "finalize", crash)
        with pytest.raises(RuntimeError, match="manifest crash"):
            await run_verb("train sft", cfg, out=tmp_path / "train")
    recovered = await run_verb("train sft", cfg, out=tmp_path / "train")
    assert recovered.handle.step == 0
    assert setup.backends[-1].learners["student"].steps == []


async def test_eval_checkpoint_resolution_and_rl_warm_start(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path, n=1)
    trained = await run_verb("train sft", config(data, epochs=1), out=tmp_path / "train")
    checkpoint = trained.handle
    spec = PolicySpec(f"ckpt:{checkpoint.root}")
    ref, _, _ = resolve_spec(spec)
    assert str(ref) == checkpoint.policy_ref("student")
    policies, renderers = await build_policies({"solver": spec}, spend=SpendGuard(0))
    assert renderers["solver"]().tokenizer_sha == FakeRenderer.tokenizer_sha
    sampled = await policies["solver"].sample(
        FakeRenderer().encode_text("question"),
        SamplingSpec(max_tokens=256),
        seed=0,
        meta=CallMeta(),
    )
    assert FakeRenderer().parse(sampled.completion_ids).tool_calls[0].arguments == {"answer": "5"}
    rl = TrainRLConfig(
        learners={
            "student": LearnerSpec(
                base_model="qwen3_8b", backend="fake", init_from=str(checkpoint.root)
            )
        },
        seating={"solver": "learner:student"},
        group_size=1,
        batch_tasks=1,
        steps=1,
        credit=CreditConfig(baseline="none", min_group=1, drop_zero_variance=False),
    )
    monkeypatch.setattr(loop, "load_model", setup.load_model)
    loop._preflight(rl)
    backend = FakeBackend("test_data_sft:teacher_policy", state_dir=tmp_path / "rl-states")
    learner = await backend.create_learner(
        "student", rl.learners["student"], model=setup.load_model("qwen3_8b"), seed=0
    )
    assert learner.weights == 1
    assert learner.version == checkpoint.learners["student"]["version"]
    assert learner.spec.loss == "importance_sampling"


@pytest.mark.parametrize("mean", [False, True])
async def test_tinker_cross_entropy_conversion_and_metrics(sdk: Any, mean: bool) -> None:  # noqa: F811
    backend = TinkerBackend(SpendGuard(1))
    learner = await backend.create_learner(
        "student",
        LearnerSpec(base_model="qwen3_8b", loss="cross_entropy"),
        model=load_model("qwen3_8b"),
        seed=0,
    )
    datum = small_datum(0, 4)
    if mean:
        datum = replace(datum, mask=tuple(m / 3 for m in datum.mask))
    result = await learner.train_step([datum])
    client = sdk.clients[0]
    assert client.loss_fn == "cross_entropy"
    [converted] = client.data
    assert converted.model_input.tokens == datum.tokens[:-1]
    assert converted.loss_fn_inputs == {
        "target_tokens": list(datum.tokens[1:]),
        "weights": list(datum.mask),
    }
    assert all(type(t) is int for t in converted.loss_fn_inputs["target_tokens"])
    assert all(type(w) is float for w in converted.loss_fn_inputs["weights"])
    assert result.loss == 1.25 and result.kl_sample_train is None
    assert result.n_action_tokens == 3
    assert not any("kl" in key for key in result.metrics)
    await learner.sync_sampler("sft-sampler")
    await learner.save_state("sft-state")
    assert [ttl for _, _, ttl in client.saves] == [None, None]
    await backend.close()


async def test_tinker_sft_loop_checkpoint_and_resume(
    tmp_path: Path,
    setup: Setup,
    sdk: Any,  # noqa: F811 -- shared fake SDK fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path, n=3)
    original = TrainingClient.forward_backward_async
    failed = False

    async def forward(client: TrainingClient, batch: list[Any], *, loss_fn: str) -> Any:
        nonlocal failed
        if len(sdk.clients) == 1 and len(client.saves) == 2 and not failed:
            failed = True
            raise RuntimeError("interrupt after checkpoint")
        future = await original(client, batch, loss_fn=loss_fn)
        client.forward_result.loss_fn_outputs = [
            {"logprobs": TensorData([-0.5] * len(d.loss_fn_inputs["weights"]))} for d in batch
        ]
        return future

    monkeypatch.setattr(TrainingClient, "forward_backward_async", forward)
    monkeypatch.setattr(sft, "make_backend", make_backend)
    cfg = config(
        data,
        learner=LearnerSpec(base_model="qwen3_8b", loss="cross_entropy"),
        epochs=1,
        checkpoint_every=1,
        max_usd=1.0,
        base_url="https://tinker.example.test",
    )
    with pytest.raises(Exception, match="interrupt after checkpoint"):
        await run_verb("train sft", cfg, out=tmp_path / "train")
    resumed = await run_verb("train sft", cfg, out=tmp_path / "train")
    checkpoint = resumed.handle
    assert checkpoint.data_cursor == {"epoch": 1, "index": 0}
    assert [saved["step"] for saved in checkpoint.history] == [0, 1]
    ref, model, _ = resolve_spec(PolicySpec(f"ckpt:{checkpoint.root}"))
    assert ref.kind == "tinker" and ref.sampler and ref.base_url == cfg.base_url
    assert model.name == "qwen3_8b"
    assert checkpoint.require_state("student").startswith("tinker://state/")
    service = sdk.ServiceClient.return_value
    service.create_training_client_from_state_with_optimizer_async.assert_awaited_once()
    assert all(ttl is None for client in sdk.clients for _, _, ttl in client.saves)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learner": LearnerSpec(loss="ppo")},
        {"epochs": 0},
        {"batch_tokens": 0},
        {"checkpoint_every": 0},
        {"loss_agg": "mean"},
        {"run_name": "../escape"},
        {"base_url": "https://user:secret@example.test"},
    ],
)
def test_config_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        TrainSFTConfig(**kwargs)


async def test_preflight_and_dirty_tree_refusal(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path, n=1)
    with pytest.raises(ConfigError, match="batch_tokens"):
        await run_verb("train sft", config(data, batch_tokens=2), out=tmp_path / "too-long")
    with monkeypatch.context() as patch:
        patch.setattr(
            sft, "get_renderer", lambda *a, **kw: type("Renderer", (), {"tokenizer_sha": "bad"})()
        )
        with pytest.raises(ConfigError, match="tokenizer_sha"):
            await run_verb("train sft", config(data), out=tmp_path / "bad-sha")
    monkeypatch.delenv("MARLI_ALLOW_DIRTY")

    def dirty() -> None:
        raise DirtyTreeError("git tree is dirty")

    monkeypatch.setattr(sft.runlog, "require_clean_tree", dirty)
    with pytest.raises(DirtyTreeError, match="dirty"):
        await run_verb("train sft", config(data), out=tmp_path / "dirty")
    assert setup.backends == []


async def test_failed_step_spend_survives_resume(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = await training_data(tmp_path, n=1)
    cfg = config(data, epochs=1, max_usd=0.5)
    original = sft.make_backend

    def charged(name: str, *, spend: SpendGuard, **kw: Any) -> FakeBackend:
        backend = original(name, spend=spend, **kw)

        async def fail(
            learner: FakeLearner, datums: Sequence[TrainDatum], **kwargs: Any
        ) -> StepResult:
            spend.charge(0.3, "failed step")
            raise RuntimeError("remote failure")

        monkeypatch.setattr(FakeLearner, "train_step", fail)
        return backend

    monkeypatch.setattr(sft, "make_backend", charged)
    with pytest.raises(RuntimeError, match="remote failure"):
        await run_verb("train sft", cfg, out=tmp_path / "train")
    with pytest.raises(BudgetExceededError):
        await run_verb("train sft", cfg, out=tmp_path / "train")
    progress = json.loads((tmp_path / "train/progress.json").read_text())
    assert progress["spend"]["spent_usd"] == pytest.approx(0.6)


def test_train_cli_one_json_line_and_idempotent(
    tmp_path: Path,
    setup: Setup,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = asyncio.run(training_data(tmp_path, n=1))
    args = [
        "train",
        "sft",
        f"data={data.root}",
        "learner.base_model=qwen3_8b",
        "learner.backend=fake",
        "--out",
        str(tmp_path / "train"),
    ]
    for status in ("fresh", "complete"):
        assert main(args) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["ok"] and row["kind"] == "checkpoint" and row["status"] == status
