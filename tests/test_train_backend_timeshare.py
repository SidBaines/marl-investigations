"""A time-shared server sleeps for training and is awake for every adapter load."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from test_train_backend_local import FakeLearner, FakePool, prompt_response

from marli.errors import ConfigError
from marli.model import load_model
from marli.serve.vllm import Server
from marli.train.backends.base import StepResult
from marli.train.backends.local.backend import LocalBackend
from marli.train.types import LearnerSpec


class TrainableFakeLearner(FakeLearner):
    events: list[str]
    fail = False

    async def train_step(self, datums: Any, *, learning_rate: float | None = None) -> StepResult:
        self.events.append(f"train:{self.name}")
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("simulated OOM")
        return StepResult(self.name, 1, 1, 1, 0.0, 0.0, 0.0, {})


class Vllm:
    """Mock vLLM: sleep state, adapter loads and probe scoring, in call order."""

    def __init__(self, events: list[str], *, sleep_mode: bool = True) -> None:
        self.events = events
        self.asleep = False
        self.sleep_mode = sleep_mode
        self.base = "qwen3_5_4b"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in {"/sleep", "/wake_up", "/is_sleeping"} and not self.sleep_mode:
            return httpx.Response(404, json={"detail": "Not Found"})
        if path == "/sleep":
            assert request.method == "POST"
            assert dict(request.url.params) == {"level": "1", "mode": "wait"}
            self.asleep = True
            self.events.append("sleep")
            return httpx.Response(200)
        if path == "/wake_up":
            self.asleep = False
            self.events.append("wake")
            return httpx.Response(200)
        if path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": self.asleep})
        assert not self.asleep, f"{path} sent to a sleeping server"
        body = json.loads(request.content)
        if body.get("echo"):
            self.events.append("probe")
            return prompt_response(body["prompt"], -1.0 if body["model"] == self.base else -0.9)
        self.events.append(path.removeprefix("/v1/"))
        return httpx.Response(200, json={})


@pytest.fixture
def fake_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("marli.train.backends.local.learner")
    module.LocalLearner = TrainableFakeLearner
    module.LocalLearnerPool = FakePool
    monkeypatch.setitem(sys.modules, module.__name__, module)
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, "torch", torch)


def server(tmp_path: Path, *, sleep_mode: bool = True) -> Server:
    record = Server(
        root=tmp_path,
        base_url="http://vllm.test",
        models=["qwen3_5_4b"],
        pid=12345,
        log="server.log",
        max_loras=4,
        hf_id="Qwen/Qwen3.5-4B",
        tensor_parallel_size=2,
        enable_sleep_mode=sleep_mode,
    )
    record.save()
    return record


@pytest.fixture
async def shared(
    tmp_path: Path, fake_stack: None
) -> AsyncIterator[tuple[LocalBackend, Vllm, list[str]]]:
    events: list[str] = []
    backend = LocalBackend(
        str(server(tmp_path).manifest_path),
        adapters_dir=str(tmp_path / "adapters"),
        devices=["cpu"],
        sleep_sampler=True,
    )
    await backend.client.aclose()
    vllm = Vllm(events)
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(vllm))
    try:
        yield backend, vllm, events
    finally:
        await backend.close()


async def new_learner(backend: LocalBackend, events: list[str], name: str = "x") -> Any:
    model = replace(load_model("qwen3_5_4b"), renderer="fake")
    learner = await backend.create_learner(
        name, LearnerSpec(base_model=model.name, backend="local", rank=16), model=model, seed=1
    )
    learner.events = events
    return learner


async def test_sleep_train_wake_then_load_and_probe(shared: tuple) -> None:
    backend, vllm, events = shared
    learner = await new_learner(backend, events)
    await learner.train_step([])
    assert vllm.asleep and events == ["sleep", "train:x"]
    await learner.sync_sampler("x-s0")
    assert not vllm.asleep
    assert events == ["sleep", "train:x", "wake", "load_lora_adapter", "probe", "probe"]
    metrics = backend.pop_step_metrics()
    assert metrics.keys() == {"sleep_s", "wake_s"} and min(metrics.values()) >= 0
    assert backend.pop_step_metrics() == {}
    events.clear()
    await learner.train_step([])  # the next step sleeps again
    await learner.sync_sampler("x-s1")
    assert events[:3] == ["sleep", "train:x", "wake"] and events[3] == "load_lora_adapter"


async def test_concurrent_learners_share_one_sleep_and_one_wake(shared: tuple) -> None:
    backend, vllm, events = shared
    first = await new_learner(backend, events, "a")
    second = await new_learner(backend, events, "b")
    await asyncio.gather(first.train_step([]), second.train_step([]))
    assert events.count("sleep") == 1 and events.index("sleep") == 0
    await asyncio.gather(first.sync_sampler("a-s0"), second.sync_sampler("b-s0"))
    assert events.count("wake") == 1 and not vllm.asleep
    assert events.index("wake") < events.index("load_lora_adapter")


async def test_failed_train_step_wakes_the_server(shared: tuple) -> None:
    backend, vllm, events = shared
    learner = await new_learner(backend, events)
    learner.fail = True
    with pytest.raises(RuntimeError, match="simulated OOM"):
        await learner.train_step([])
    assert events == ["sleep", "train:x", "wake"] and not vllm.asleep


async def test_close_wakes_a_sleeping_server(
    tmp_path: Path, fake_stack: None
) -> None:
    events: list[str] = []
    backend = LocalBackend(
        str(server(tmp_path).manifest_path),
        adapters_dir=str(tmp_path / "adapters"),
        sleep_sampler=True,
    )
    await backend.client.aclose()
    vllm = Vllm(events)
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(vllm))
    learner = await new_learner(backend, events)
    await learner.train_step([])
    assert vllm.asleep
    await backend.close()
    assert events[-1] == "wake" and not vllm.asleep


async def test_no_sleep_unless_requested(tmp_path: Path, fake_stack: None) -> None:
    events: list[str] = []
    backend = LocalBackend(
        str(server(tmp_path).manifest_path), adapters_dir=str(tmp_path / "adapters")
    )
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(Vllm(events)))
    try:
        learner = await new_learner(backend, events)
        await learner.train_step([])
        await learner.sync_sampler("x-s0")
        assert "sleep" not in events and "wake" not in events
        assert backend.pop_step_metrics() == {}
    finally:
        await backend.close()


def test_sleep_sampler_needs_a_sleep_mode_server(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="enable_sleep_mode"):
        LocalBackend(
            str(server(tmp_path, sleep_mode=False).manifest_path),
            adapters_dir=str(tmp_path / "adapters"),
            sleep_sampler=True,
        )


async def test_missing_sleep_endpoints_explain_the_fix(tmp_path: Path, fake_stack: None) -> None:
    events: list[str] = []
    backend = LocalBackend(
        str(server(tmp_path).manifest_path),
        adapters_dir=str(tmp_path / "adapters"),
        sleep_sampler=True,
    )
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(Vllm(events, sleep_mode=False))
    )
    try:
        learner = await new_learner(backend, events)
        with pytest.raises(ConfigError, match="VLLM_SERVER_DEV_MODE"):
            await learner.train_step([])
        assert backend._training == 0 and not backend._asleep
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "devices,match",
    [(["cuda:0", "cuda:0"], "duplicate"), ([], "non-empty"), (["cpu", "cuda:1"], "all cpu")],
)
def test_invalid_devices_fail_before_training(
    tmp_path: Path, devices: list[str], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        LocalBackend(
            str(server(tmp_path).manifest_path),
            adapters_dir=str(tmp_path / "adapters"),
            devices=devices,
        )


def test_single_device_list_selects_that_device(tmp_path: Path) -> None:
    backend = LocalBackend(
        str(server(tmp_path).manifest_path),
        adapters_dir=str(tmp_path / "adapters"),
        devices=["cuda:1"],
    )
    assert backend.device == "cuda:1" and backend.devices == ("cuda:1",)
