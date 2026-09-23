"""Fake SDK futures expose request ordering, accounting and learner isolation."""

from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from marli.budget import SpendGuard, tinker_cost
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.model import load_model
from marli.policy.tinker import TinkerPolicy
from marli.render.fake import FakeRenderer
from marli.train.backends.registry import make_backend
from marli.train.backends.tinker import TinkerBackend, TinkerLearner
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec, TrainDatum


class BadRequestError(Exception):
    pass


class UnprocessableEntityError(Exception):
    pass


class APIError(Exception):
    pass


@dataclass(frozen=True)
class ModelInput:
    tokens: tuple[int, ...]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> ModelInput:
        assert isinstance(tokens, list)
        return cls(tuple(tokens))


@dataclass(frozen=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict[str, list[int] | list[float]]


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: float


@dataclass(frozen=True)
class TensorData:
    values: list[float]

    def tolist(self) -> list[float]:
        return self.values.copy()


class Future:
    def __init__(self, sdk: Any, label: str, result: Any) -> None:
        self.sdk = sdk
        self.label = label
        self.result = result

    async def result_async(self) -> Any:
        self.sdk.events.append(f"{self.label}:result")
        await asyncio.sleep(0)
        return self.result


class TrainingClient:
    def __init__(self, sdk: Any, name: str) -> None:
        self.sdk = sdk
        self.name = name
        self.data: list[Datum] = []
        self.adam: AdamParams | None = None
        self.forward_result = SimpleNamespace(
            metrics={"loss:sum": 1.25, "extra:mean": 4.0}, loss_fn_outputs=[]
        )
        self.optim_result = SimpleNamespace(metrics={"grad_norm": 3.0, "optim_extra": 2.0})
        self.saves: list[tuple[str, str, int | None]] = []
        self.loads: list[tuple[str, bool]] = []

    async def forward_backward_async(self, data: list[Datum], *, loss_fn: str) -> Future:
        if self.sdk.spend is not None:
            assert self.sdk.spend.spent > 0, "must charge before submission"
        self.sdk.events.append(f"{self.name}:forward:submit")
        self.data = data
        self.loss_fn = loss_fn
        self.forward_result.loss_fn_outputs = [
            {"logprobs": TensorData([-100.0, -0.75, -200.0, -1.75])} for _ in data
        ]
        return Future(self.sdk, f"{self.name}:forward", self.forward_result)

    async def optim_step_async(self, params: AdamParams) -> Future:
        self.sdk.events.append(f"{self.name}:optim:submit")
        self.adam = params
        return Future(self.sdk, f"{self.name}:optim", self.optim_result)

    async def save_weights_for_sampler_async(self, name: str, *, ttl_seconds: int | None) -> Future:
        self.saves.append(("sampler", name, ttl_seconds))
        return Future(self.sdk, "sampler", SimpleNamespace(path=f"tinker://sampler/{name}"))

    async def save_state_async(self, name: str, *, ttl_seconds: int | None) -> Future:
        self.saves.append(("state", name, ttl_seconds))
        return Future(self.sdk, "state", SimpleNamespace(path=f"tinker://state/{name}"))

    async def load_state_async(self, path: str) -> Future:
        self.loads.append((path, False))
        return Future(self.sdk, "load", None)

    async def load_state_with_optimizer_async(self, path: str) -> Future:
        self.loads.append((path, True))
        return Future(self.sdk, "load_optimizer", None)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    sdk = ModuleType("tinker")
    sdk.ModelInput = ModelInput
    sdk.Datum = Datum
    sdk.AdamParams = AdamParams
    sdk.BadRequestError = BadRequestError
    sdk.UnprocessableEntityError = UnprocessableEntityError
    sdk.events = []
    sdk.clients = []
    sdk.spend = None

    async def training_client(**kwargs: Any) -> TrainingClient:
        client = TrainingClient(sdk, f"learner{len(sdk.clients)}")
        sdk.clients.append(client)
        return client

    weights_info = SimpleNamespace(base_model="Qwen/Qwen3-8B", is_lora=True, lora_rank=32)
    rest = SimpleNamespace(get_weights_info_by_tinker_path=AsyncMock(return_value=weights_info))
    service = SimpleNamespace(
        create_lora_training_client_async=AsyncMock(side_effect=training_client),
        create_training_client_from_state_async=AsyncMock(side_effect=training_client),
        create_training_client_from_state_with_optimizer_async=AsyncMock(
            side_effect=training_client
        ),
        create_rest_client=Mock(return_value=rest),
        create_sampling_client_async=AsyncMock(side_effect=lambda **kw: object()),
        close=AsyncMock(),
    )
    sdk.ServiceClient = Mock(return_value=service)
    base_url = ModuleType("tinker.lib.base_url")
    base_url.DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"
    monkeypatch.setitem(sys.modules, "tinker", sdk)
    monkeypatch.setitem(sys.modules, "tinker.lib.base_url", base_url)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    return sdk


def datum(learner: str = "a", *, version: int = 0) -> TrainDatum:
    return TrainDatum(
        learner=learner,
        episode_id="e",
        agent_id="peer",
        role="peer",
        segment_id="s",
        session_idx=0,
        policy_version=version,
        tokens=tuple(FakeRenderer().encode_text("abcde")),
        logprobs=(0.0, -0.5, 0.0, -1.0),
        mask=(0.0, 1.0, 0.0, 1.0),
        advantages=(0.0, 2.0, 0.0, -0.5),
    )


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo"])
async def test_conversion_pipeline_metrics_and_optimizer(sdk: Any, loss: str) -> None:
    guard = SpendGuard(1.0)
    sdk.spend = guard
    backend = make_backend("tinker", spend=guard)
    spec = LearnerSpec(
        rank=8,
        loss=loss,
        learning_rate=0.1,
        beta1=0.8,
        beta2=0.9,
        eps=1e-7,
        weight_decay=0.2,
        grad_clip=0.4,
    )
    model = load_model("qwen3_8b")
    learner = await backend.create_learner("a", spec, model=model, seed=42)
    service = sdk.ServiceClient.return_value
    service.create_lora_training_client_async.assert_awaited_once_with(
        base_model=model.tinker_id, rank=8, seed=42
    )
    service.create_sampling_client_async.assert_awaited_once_with(base_model=model.tinker_id)
    result = await learner.train_step([datum()], learning_rate=0.03)
    client = sdk.clients[0]
    assert client.data == [
        Datum(
            ModelInput(datum().tokens[:-1]),
            {
                "target_tokens": list(datum().tokens[1:]),
                "logprobs": list(datum().logprobs),
                "advantages": list(datum().advantages),
            },
        )
    ]
    assert "mask" not in client.data[0].loss_fn_inputs
    assert client.loss_fn == loss
    assert client.adam == AdamParams(0.03, 0.8, 0.9, 1e-7, 0.2, 0.4)
    assert sdk.events == [
        "learner0:forward:submit",
        "learner0:optim:submit",
        "learner0:forward:result",
        "learner0:optim:result",
    ]
    assert (result.n_datums, result.n_tokens, result.n_action_tokens) == (1, 4, 2)
    assert result.loss == 1.25 and result.grad_norm == 3.0
    assert result.kl_sample_train == pytest.approx(0.5)
    assert result.metrics == {
        "loss:sum": 1.25,
        "extra:mean": 4.0,
        "grad_norm": 3.0,
        "optim_extra": 2.0,
    }
    assert guard.spent == tinker_cost(model, train=4)
    assert learner.version == 0
    await learner.train_step([datum()])
    assert client.adam.learning_rate == spec.learning_rate


async def test_two_learners_submit_before_any_result(sdk: Any) -> None:
    backend = TinkerBackend(SpendGuard(1))
    a, b = await asyncio.gather(
        *(
            backend.create_learner(name, LearnerSpec(), model=load_model("qwen3_8b"), seed=i)
            for i, name in enumerate(("a", "b"))
        )
    )
    ra, rb = await asyncio.gather(a.train_step([datum()]), b.train_step([datum("b")]))
    assert a.training_client is not b.training_client
    assert sdk.ServiceClient.call_count == 1
    assert sdk.events[:4] == [
        "learner0:forward:submit",
        "learner0:optim:submit",
        "learner1:forward:submit",
        "learner1:optim:submit",
    ]
    assert all(event.endswith(":result") for event in sdk.events[4:])
    assert (ra.learner, rb.learner) == ("a", "b")
    await a.close()
    sdk.ServiceClient.return_value.close.assert_not_called()
    await backend.close()
    sdk.ServiceClient.return_value.close.assert_awaited_once_with("success")


async def test_budget_failure_never_submits_or_records_spend(sdk: Any) -> None:
    guard = SpendGuard(0)
    learner = await TinkerBackend(guard).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    with pytest.raises(BudgetExceededError):
        await learner.train_step([datum()])
    assert sdk.events == [] and guard.spent == 0


async def test_kl_is_token_weighted_and_optimizer_metrics_optional(sdk: Any) -> None:
    learner = await TinkerBackend(None).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    learner.training_client.optim_result.metrics = None
    second = replace(
        datum(),
        mask=(0.0, 1.0, 0.0, 0.0),
        logprobs=(0.0, -0.5, 0.0, 0.0),
        advantages=(0.0, 1.0, 0.0, 0.0),
    )
    result = await learner.train_step([datum(), second])
    assert result.kl_sample_train == pytest.approx((0.25 + 0.75 + 0.25) / 3)
    assert result.n_action_tokens == 3 and result.n_tokens == 8
    assert result.grad_norm is None
    assert result.metrics == {"loss:sum": 1.25, "extra:mean": 4.0}


@pytest.mark.parametrize("invalid", ["empty", "routing", "context", "versions"])
async def test_invalid_datums_rejected_before_spend(sdk: Any, invalid: str) -> None:
    model = load_model("qwen3_8b")
    if invalid == "context":
        model = replace(model, max_ctx=4)
    guard = SpendGuard(1)
    learner = await TinkerBackend(guard).create_learner("a", LearnerSpec(), model=model, seed=0)
    batch = {
        "empty": [],
        "routing": [datum("b")],
        "context": [datum()],
        "versions": [datum(), datum(version=1)],
    }[invalid]
    with pytest.raises(ConfigError):
        await learner.train_step(batch)
    assert sdk.events == [] and guard.spent == 0


async def test_model_preflight(sdk: Any) -> None:
    backend = TinkerBackend(SpendGuard(1))
    model = replace(load_model("qwen3_8b"), tinker_id=None, tinker_max_ctx=None, tinker_prices=None)
    with pytest.raises(ConfigError, match="tinker_id"):
        await backend.create_learner("a", LearnerSpec(), model=model, seed=0)
    assert sdk.clients == []
    # ModelSpec normally rejects partial Tinker metadata; pin the backend guard too.
    model = load_model("qwen3_8b")
    object.__setattr__(model, "tinker_prices", None)
    with pytest.raises(ConfigError, match="prices"):
        await backend.create_learner("a", LearnerSpec(), model=model, seed=0)
    assert sdk.clients == []


@pytest.mark.parametrize("guard", [None, SpendGuard(None)])
async def test_unlimited_guard_allows_unpriced_model(sdk: Any, guard: SpendGuard | None) -> None:
    model = load_model("qwen3_8b")
    object.__setattr__(model, "tinker_prices", None)
    learner = await TinkerBackend(guard).create_learner("a", LearnerSpec(), model=model, seed=0)
    await learner.train_step([datum()])
    assert len(sdk.events) == 4


@pytest.mark.parametrize("stage", ["submit", "result"])
@pytest.mark.parametrize(
    "error,expected",
    [
        (ValueError("runtime failure"), BackendError),
        (BadRequestError("bad request"), ConfigError),
        (UnprocessableEntityError("invalid"), ConfigError),
        (APIError("remote failure"), BackendError),
        (ConnectionError("offline"), BackendError),
        (TimeoutError("timeout"), BackendError),
    ],
)
async def test_step_errors_chained_without_retry(
    sdk: Any, stage: str, error: Exception, expected: type[Exception]
) -> None:
    guard = SpendGuard(1)
    learner = await TinkerBackend(guard).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    future = SimpleNamespace(result_async=AsyncMock(side_effect=error))
    submit = AsyncMock(side_effect=error) if stage == "submit" else AsyncMock(return_value=future)
    learner.training_client.forward_backward_async = submit
    with pytest.raises(expected) as caught:
        await learner.train_step([datum()])
    assert caught.value.__cause__ is error
    submit.assert_awaited_once()
    if stage == "result":
        future.result_async.assert_awaited_once()
    assert guard.spent == tinker_cost(learner.model, train=4)


async def test_cancellation_passes_through(sdk: Any) -> None:
    learner = await TinkerBackend(None).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    learner.training_client.forward_backward_async = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await learner.train_step([datum()])
    learner.training_client.forward_backward_async.assert_awaited_once()


async def test_snapshots_state_and_policy_identity(sdk: Any) -> None:
    guard = SpendGuard(1)
    model = load_model("qwen3_8b")
    backend = TinkerBackend(guard)
    learner = await backend.create_learner("a", LearnerSpec(), model=model, seed=0)
    initial = learner.policy()
    assert isinstance(initial, TinkerPolicy) and initial.trainable
    assert initial.policy_version == 0 and initial.policy_id == "a"
    assert initial.model is model and initial.spend is guard
    assert initial.renderer_name == model.renderer
    snap = await learner.sync_sampler("run-a-v1")
    assert (snap.learner, snap.version, snap.path) == ("a", 1, "tinker://sampler/run-a-v1")
    assert snap.policy_ref == f"tinker:{model.tinker_id}#sampler={snap.path}"
    sdk.ServiceClient.return_value.create_sampling_client_async.assert_awaited_with(
        model_path=snap.path
    )
    policy = learner.policy(policy_id="peer")
    assert policy.policy_id == "peer" and policy.policy_version == 1
    assert initial.policy_version == 0 and initial.sampling_client is not policy.sampling_client
    with pytest.raises(ValueError, match="already used"):
        await learner.sync_sampler("run-a-v1")
    assert learner.version == 1
    assert (await learner.sync_sampler("run-a-v2")).version == 2
    state = await learner.save_state("run-a-step2")
    assert state == "tinker://state/run-a-step2"
    assert learner.training_client.saves == [
        ("sampler", "run-a-v1", None),
        ("sampler", "run-a-v2", None),
        ("state", "run-a-step2", None),
    ]
    before_restore = learner.policy()
    await learner.load_state(state)
    service = sdk.ServiceClient.return_value
    service.create_training_client_from_state_with_optimizer_async.assert_awaited_once_with(
        path=state
    )
    assert learner.training_client.saves == [("sampler", "a-init", None)]
    assert learner.policy().sampling_client is not before_restore.sampling_client
    await learner.load_state(state, with_optimizer=False)
    service.create_training_client_from_state_async.assert_awaited_once_with(path=state)
    assert learner.training_client.saves == [("sampler", "a-restore-1", None)]
    assert learner.version == 2


@pytest.mark.parametrize("source_kind", ["raw", "manifest", "manifest_sampler"])
@pytest.mark.parametrize("init_mode", ["resume", "weights"])
async def test_init_from_restores_weights_optimizer_sampler_and_version(
    sdk: Any, tmp_path: Path, source_kind: str, init_mode: str
) -> None:
    state = "tinker://run/weights/step-5"
    model = load_model("qwen3_8b")
    source = state
    sampler = "tinker://run/sampler_weights/step-5" if source_kind == "manifest_sampler" else None
    if source_kind != "raw":
        source = str(
            Checkpoint(
                root=tmp_path,
                step=5,
                run_config_hash="hash",
                learners={
                    "a": {
                        "state": state,
                        "sampler": sampler,
                        "version": 5,
                        "base_model": model.name,
                        "backend": "tinker",
                        "rank": 32,
                    }
                },
            ).save()
        )
    learner = await TinkerBackend(SpendGuard(1), init_mode=init_mode).create_learner(
        "a", LearnerSpec(init_from=source), model=model, seed=0
    )
    service = sdk.ServiceClient.return_value
    resume = service.create_training_client_from_state_with_optimizer_async
    warm = service.create_training_client_from_state_async
    (resume if init_mode == "resume" else warm).assert_awaited_once_with(path=state)
    (warm if init_mode == "resume" else resume).assert_not_called()
    service.create_lora_training_client_async.assert_not_called()
    assert learner.training_client.loads == []
    version = 0 if source_kind == "raw" else 5
    assert learner.policy().policy_version == learner.version == version
    expected_sampler = sampler or "tinker://sampler/a-init"
    service.create_sampling_client_async.assert_awaited_once_with(model_path=expected_sampler)
    assert learner.training_client.saves == ([] if sampler else [("sampler", "a-init", None)])
    await learner.train_step([datum(version=version)])
    assert (await learner.sync_sampler("next")).version == version + 1


async def test_reuses_base_url_guard(sdk: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_BASE_URL", "http://local:8000")
    with pytest.raises(ConfigError, match="TINKER_BASE_URL"):
        TinkerBackend(None)
    sdk.ServiceClient.assert_not_called()
    backend = TinkerBackend(None, base_url="http://local:8000")
    sdk.ServiceClient.assert_called_once_with(base_url="http://local:8000")
    learner = await backend.create_learner("a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0)
    snap = await learner.sync_sampler("custom")
    assert snap.policy_ref.startswith("tinker@http://local:8000|Qwen/")


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "fake"),
        ("base_model", "qwen3_4b"),
        ("rank", 8),
    ],
)
async def test_checkpoint_metadata_mismatch_fails_before_sdk_restore(
    sdk: Any, tmp_path: Path, field: str, value: str | int
) -> None:
    record = {
        "state": "tinker://run/weights/step-5",
        "sampler": None,
        "version": 5,
        "base_model": "qwen3_8b",
        "backend": "tinker",
        "rank": 32,
        field: value,
    }
    ck = Checkpoint(root=tmp_path, step=5, run_config_hash="hash", learners={"a": record})
    with pytest.raises(ConfigError, match=field):
        await TinkerBackend(None).create_learner(
            "a", LearnerSpec(init_from=str(ck.save())), model=load_model("qwen3_8b"), seed=0
        )
    service = sdk.ServiceClient.return_value
    assert sdk.clients == []
    service.create_rest_client.assert_not_called()
    service.create_sampling_client_async.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_model", "Qwen/Qwen3-4B"),
        ("lora_rank", 8),
        ("is_lora", False),
    ],
)
async def test_raw_state_metadata_is_checked_before_restore(
    sdk: Any, field: str, value: str | int | bool
) -> None:
    service = sdk.ServiceClient.return_value
    info = service.create_rest_client.return_value.get_weights_info_by_tinker_path.return_value
    setattr(info, field, value)
    with pytest.raises(ConfigError, match="base_model|rank"):
        await TinkerBackend(None).create_learner(
            "a",
            LearnerSpec(init_from="tinker://run/weights/step-5"),
            model=load_model("qwen3_8b"),
            seed=0,
        )
    service.create_training_client_from_state_with_optimizer_async.assert_not_called()
    service.create_training_client_from_state_async.assert_not_called()
    service.create_sampling_client_async.assert_not_called()


async def test_restore_respects_explicit_spec_rank(sdk: Any) -> None:
    service = sdk.ServiceClient.return_value
    info = service.create_rest_client.return_value.get_weights_info_by_tinker_path.return_value
    info.lora_rank = 8
    learner = await TinkerBackend(None).create_learner(
        "a",
        LearnerSpec(rank=8, init_from="tinker://run/weights/step-5"),
        model=load_model("qwen3_8b"),
        seed=0,
    )
    assert learner.spec.rank == 8
    assert learner.training_client.saves == [("sampler", "a-init", None)]


def test_missing_tinker_extra_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    with pytest.raises(ConfigError, match=r"\[tinker\] extra"):
        TinkerBackend(None)


def test_unknown_init_mode_is_config_error(sdk: Any) -> None:
    with pytest.raises(ConfigError, match="init_mode"):
        TinkerBackend(None, init_mode="unknown")
    sdk.ServiceClient.assert_not_called()


async def test_local_optimizer_validation_is_config_error(sdk: Any) -> None:
    backend = TinkerBackend(None)
    learner = await backend.create_learner("a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0)
    sdk.AdamParams = Mock(side_effect=ValueError("invalid optimizer"))
    with pytest.raises(ConfigError, match="invalid optimizer"):
        await learner.train_step([datum()])
    assert sdk.events == []


@pytest.mark.parametrize(
    "error,status",
    [
        (ValueError("remote failure"), "errored"),
        (asyncio.CancelledError(), "interrupted"),
    ],
)
async def test_close_reports_training_failure(sdk: Any, error: BaseException, status: str) -> None:
    backend = TinkerBackend(None)
    learner = await backend.create_learner("a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0)
    learner.training_client.optim_step_async = AsyncMock(
        return_value=SimpleNamespace(result_async=AsyncMock(side_effect=error))
    )
    with pytest.raises(BackendError if isinstance(error, ValueError) else asyncio.CancelledError):
        await learner.train_step([datum()])
    await backend.close()
    sdk.ServiceClient.return_value.close.assert_awaited_once_with(status)


async def test_close_reports_external_failure_in_finally(sdk: Any) -> None:
    backend = TinkerBackend(None)
    with pytest.raises(RuntimeError, match="rollout"):
        try:
            raise RuntimeError("rollout failed")
        finally:
            await backend.close()
    sdk.ServiceClient.return_value.close.assert_awaited_once_with("errored")


async def test_grad_norm_from_forward_metrics(sdk: Any) -> None:
    learner = await TinkerBackend(None).create_learner(
        "a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0
    )
    learner.training_client.optim_result.metrics = None
    learner.training_client.forward_result.metrics["grad_norm"] = 4.5
    result = await learner.train_step([datum()])
    assert result.grad_norm == 4.5


async def test_invalid_training_response_marks_close_errored(sdk: Any) -> None:
    backend = TinkerBackend(None)
    learner = await backend.create_learner("a", LearnerSpec(), model=load_model("qwen3_8b"), seed=0)
    learner.training_client.forward_result.metrics = {}
    with pytest.raises(BackendError, match="missing loss:sum"):
        await learner.train_step([datum()])
    await backend.close()
    sdk.ServiceClient.return_value.close.assert_awaited_once_with("errored")


@pytest.mark.tinker
async def test_train_step_with_installed_sdk_types_and_fake_transport() -> None:
    tinker = pytest.importorskip("tinker")
    sdk = SimpleNamespace(events=[])
    data = datum()
    forward_result = tinker.types.ForwardBackwardOutput(
        loss_fn_output_type="ArrayRecord",
        loss_fn_outputs=[
            {
                "logprobs": tinker.TensorData(
                    data=[-100.0, -0.75, -200.0, -1.75], dtype="float32", shape=[4]
                )
            }
        ],
        metrics={"loss:sum": 3.5},
    )
    optim_result = tinker.types.OptimStepResponse(metrics=None)
    client = SimpleNamespace(
        forward_backward_async=AsyncMock(return_value=Future(sdk, "forward", forward_result)),
        optim_step_async=AsyncMock(return_value=Future(sdk, "optim", optim_result)),
    )
    guard = SpendGuard(1)
    learner = TinkerLearner(
        "a",
        LearnerSpec(),
        model=load_model("qwen3_8b"),
        service=None,
        training_client=client,
        sampling_client=object(),
        spend=guard,
        base_url=None,
    )
    result = await learner.train_step([data])
    sent = client.forward_backward_async.call_args.args[0][0]
    assert sent.model_input.to_ints() == list(data.tokens[:-1])
    assert sent.loss_fn_inputs["target_tokens"].tolist() == list(data.tokens[1:])
    assert sent.loss_fn_inputs["logprobs"].tolist() == list(data.logprobs)
    assert sent.loss_fn_inputs["advantages"].tolist() == list(data.advantages)
    adam = client.optim_step_async.call_args.args[0]
    assert isinstance(adam, tinker.AdamParams)
    assert adam.grad_clip_norm == 0.0
    assert result.loss == 3.5 and result.grad_norm is None
    assert result.kl_sample_train == 0.5
    assert guard.spent == tinker_cost(learner.model, train=4)


@pytest.mark.tinker
def test_installed_sdk_contract_drift() -> None:
    tinker = pytest.importorskip("tinker")
    service_calls = {
        "__init__": {"base_url": "http://local:8000"},
        "create_lora_training_client_async": {"base_model": "base", "rank": 8, "seed": 0},
        "create_training_client_from_state_async": {"path": "tinker://state"},
        "create_training_client_from_state_with_optimizer_async": {"path": "tinker://state"},
        "create_rest_client": {},
        "create_sampling_client_async": {"base_model": "base", "model_path": "tinker://sampler"},
        "close": {"status": "success"},
    }
    for method, kwargs in service_calls.items():
        inspect.signature(getattr(tinker.ServiceClient, method)).bind(None, **kwargs)
    training_calls = {
        "forward_backward_async": {"data": [], "loss_fn": "importance_sampling"},
        "optim_step_async": {"optim_params": None},
        "save_weights_for_sampler_async": {"name": "sampler", "ttl_seconds": None},
        "save_state_async": {"name": "state", "ttl_seconds": None},
        "load_state_async": {"path": "tinker://state"},
        "load_state_with_optimizer_async": {"path": "tinker://state"},
    }
    for method, kwargs in training_calls.items():
        inspect.signature(getattr(tinker.TrainingClient, method)).bind(None, **kwargs)
    from tinker.lib.public_interfaces.rest_client import RestClient

    inspect.signature(RestClient.get_weights_info_by_tinker_path).bind(None, "tinker://state")
    inspect.signature(tinker.APIFuture.result_async).bind(None)
    inspect.signature(tinker.ModelInput.from_ints).bind([1, 2])
    inspect.signature(tinker.TensorData.tolist).bind(None)
    for error in (tinker.BadRequestError, tinker.UnprocessableEntityError):
        assert issubclass(error, Exception)
    assert {"learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"} <= (
        tinker.AdamParams.model_fields.keys()
    )
    for cls, fields in (
        (tinker.Datum, {"model_input", "loss_fn_inputs"}),
        (tinker.types.ForwardBackwardOutput, {"metrics", "loss_fn_outputs"}),
        (tinker.types.OptimStepResponse, {"metrics"}),
        (tinker.types.SaveWeightsResponse, {"path"}),
        (tinker.types.SaveWeightsForSamplerResponse, {"path"}),
    ):
        assert fields <= (getattr(cls, "model_fields", None) or inspect.get_annotations(cls)).keys()
    converted = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(datum().tokens[:-1])),
        loss_fn_inputs={
            "target_tokens": list(datum().tokens[1:]),
            "logprobs": list(datum().logprobs),
            "advantages": list(datum().advantages),
        },
    )
    assert converted.loss_fn_inputs["target_tokens"].tolist() == list(datum().tokens[1:])
    assert converted.loss_fn_inputs["logprobs"].tolist() == list(datum().logprobs)
    assert converted.loss_fn_inputs["advantages"].tolist() == list(datum().advantages)
