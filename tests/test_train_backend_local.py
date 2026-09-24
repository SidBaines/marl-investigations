"""Fake training weights and HTTP responses pin immutable adapter publication."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from marli.errors import BackendError, ConfigError
from marli.model import ModelSpec, load_model
from marli.policy.base import SamplingSpec
from marli.policy.resolve import resolve_policy
from marli.policy.vllm import read_server_json
from marli.render.base import Msg
from marli.render.fake import FakeRenderer
from marli.serve.vllm import Server
from marli.train.backends.local.backend import LocalBackend
from marli.train.backends.registry import make_backend
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec


class FakePool:
    def __init__(self, model: ModelSpec, *, device: str) -> None:
        self.model = model
        self.device = device


def prompt_response(tokens: list[int], score: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "prompt_logprobs": [None]
                    + [{str(token): {"logprob": score}} for token in tokens[1:]]
                }
            ]
        },
    )


class FakeLearner:
    def __init__(self, name: str, spec: LearnerSpec, pool: FakePool) -> None:
        self.name, self.spec, self.pool = name, spec, pool
        self.model = pool.model
        self.version = 0
        self.saved: list[Path] = []
        self.loaded: list[tuple[str, bool]] = []
        self.closed = False

    async def save_adapter(self, directory: str | Path) -> Path:
        directory = Path(directory)
        self.saved.append(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "adapter_config.json").write_text("{}")
        return directory

    async def probe_logprobs(
        self, tokens: list[int]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        self.probe_tokens = tuple(tokens)
        return (-0.9,) * (len(tokens) - 1), (-1.0,) * (len(tokens) - 1)

    async def save_state(self, name: str) -> str:
        return f"states/{name}"

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None:
        self.loaded.append((path, with_optimizer))

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
async def local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[LocalBackend, list[httpx.Request], list[Any]]]:
    module = ModuleType("marli.train.backends.local.learner")
    module.LocalLearner = FakeLearner
    module.LocalLearnerPool = FakePool
    monkeypatch.setitem(sys.modules, module.__name__, module)
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    server = Server(
        root=tmp_path,
        base_url="http://vllm.test",
        models=["qwen3_5_4b"],
        pid=12345,
        log="server.log",
        max_loras=3,
        hf_id="Qwen/Qwen3.5-4B",
    )
    server.save()
    backend = LocalBackend(
        str(server.manifest_path), adapters_dir=str(tmp_path / "adapters"), device="cpu"
    )
    await backend.client.aclose()
    calls: list[httpx.Request] = []
    replies: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content)
        if body.get("echo"):
            score = -1.0 if body["model"] == backend.base_name else -0.9
            return prompt_response(body["prompt"], score)
        response = replies.pop(0) if replies else httpx.Response(200, json={})
        if isinstance(response, Exception):
            raise response
        return response

    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        yield backend, calls, replies
    finally:
        replies.clear()
        await backend.close()


async def new_learner(backend: LocalBackend, name: str = "x") -> FakeLearner:
    model = replace(load_model("qwen3_5_4b"), renderer="fake")
    return await backend.create_learner(
        name,
        LearnerSpec(base_model=model.name, backend="local", rank=16),
        model=model,
        seed=41,
    )


async def test_publication_versions_policy_and_eviction(local: tuple) -> None:
    backend, calls, replies = local
    learner = await new_learner(backend)
    assert isinstance(learner, FakeLearner)
    initial = learner.policy()
    assert initial.model == "qwen3_5_4b" and initial.policy_version == 0
    assert initial.trainable and initial.renderer_name == "fake"
    for step in range(1, 5):
        name = f"run-x-s{step}"
        snapshot = await learner.sync_sampler(name)
        assert snapshot.version == step == learner.version
        assert snapshot.learner == "x"
        assert snapshot.path == str(backend.adapters_dir / name)
        assert snapshot.policy_ref == f"vllm:@{backend.server_json}#{name}"
        policy = learner.policy(policy_id="peer")
        assert policy.model == name and policy.policy_version == step
        assert policy.trainable and policy.policy_id == "peer"
        server = Server.load(backend.server_json)
        assert len(server.adapters) <= server.max_loras - 1
        assert read_server_json(backend.server_json)[1][name] == name
    assert [
        (request.url.path, json.loads(request.content)["lora_name"])
        for request in calls
        if not json.loads(request.content).get("echo")
    ] == [
        ("/v1/load_lora_adapter", "run-x-s1"),
        ("/v1/load_lora_adapter", "run-x-s2"),
        ("/v1/load_lora_adapter", "run-x-s3"),
        ("/v1/unload_lora_adapter", "run-x-s1"),
        ("/v1/load_lora_adapter", "run-x-s4"),
        ("/v1/unload_lora_adapter", "run-x-s2"),
    ]
    assert json.loads(calls[0].content) == {
        "lora_name": "run-x-s1",
        "lora_path": str(backend.adapters_dir / "run-x-s1"),
    }
    assert learner.saved == [backend.adapters_dir / f"run-x-s{n}" for n in range(1, 5)]
    with pytest.raises(ConfigError, match="never be reused"):
        await learner.sync_sampler("run-x-s1")
    assert learner.version == 4

    # Policy requests keep the exact renderer ids, raw logprobs and sampler version.
    renderer = FakeRenderer()
    prompt = renderer.initial(None, [], [Msg("user", "hi")])
    completion = renderer.encode_completion("answer")
    replies.append(
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": completion,
                        "logprobs": {"token_logprobs": [-0.1] * len(completion)},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )
    sample = await policy.sample(
        prompt,
        SamplingSpec(100, 1, 1, -1, renderer.stop_token_ids),
        seed=12,
    )
    body = json.loads(calls[-1].content)
    assert body["prompt"] == prompt and body["model"] == "run-x-s4"
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1, 1, -1)
    assert sample.completion_ids == tuple(completion) and sample.policy_version == 4
    resolved = await resolve_policy(
        snapshot.policy_ref, policy_id="eval", trainable=False, renderer_name="fake"
    )
    try:
        assert resolved.model == "run-x-s4"
    finally:
        await resolved.aclose()


async def test_pool_shared_and_other_learner_current_preserved(local: tuple) -> None:
    backend, calls, _ = local
    x, y = await new_learner(backend, "x"), await new_learner(backend, "y")
    assert x.pool is y.pool and x.pool.device == "cpu"
    await asyncio.gather(x.sync_sampler("run-x-s1"), y.sync_sampler("run-y-s1"))
    await x.sync_sampler("run-x-s2")
    assert y.policy().model == "run-y-s1"
    server = Server.load(backend.server_json)
    assert {a["name"] for a in server.adapters} == {"run-x-s2", "run-y-s1"}
    assert json.loads(calls[-1].content)["lora_name"] == "run-x-s1"
    await x.close()
    assert Server.load(backend.server_json).models == ["qwen3_5_4b", "run-y-s1"]
    await backend.close()
    assert Server.load(backend.server_json).models == ["qwen3_5_4b"]
    assert x.closed and y.closed
    assert all(path.exists() for path in x.saved + y.saved)


@pytest.mark.parametrize(
    "response,error,match",
    [
        (
            httpx.Response(400, text="adapter rank invalid"),
            ConfigError,
            "400: adapter rank invalid",
        ),
        (httpx.Response(500, text="engine crashed"), BackendError, "500: engine crashed"),
        (httpx.ConnectError("connection refused"), BackendError, "connection refused"),
    ],
)
async def test_load_errors_do_not_advance_or_reuse_name(
    local: tuple, response: Any, error: type[Exception], match: str
) -> None:
    backend, calls, replies = local
    learner = await new_learner(backend)
    replies.append(response)
    with pytest.raises(error, match=match):
        await learner.sync_sampler("run-x-s1")
    assert learner.version == 0 and learner.policy().model == "qwen3_5_4b"
    assert len(calls) == 1
    with pytest.raises(ConfigError, match="already used"):
        await learner.sync_sampler("run-x-s1")
    assert len(calls) == 1


async def test_failed_unload_keeps_new_policy_and_manifest_truthful(local: tuple) -> None:
    backend, calls, replies = local
    learner = await new_learner(backend)
    await learner.sync_sampler("run-x-s1")
    await learner.sync_sampler("run-x-s2")
    replies.extend([httpx.Response(200), httpx.Response(500, text="cannot unload")])
    with pytest.raises(BackendError, match="cannot unload"):
        await learner.sync_sampler("run-x-s3")
    assert learner.version == 3 and learner.policy().model == "run-x-s3"
    assert len(Server.load(backend.server_json).adapters) == 3
    await backend.close()
    assert Server.load(backend.server_json).adapters == []


async def test_state_methods_delegate_and_invalid_names_do_not_export(local: tuple) -> None:
    backend, calls, _ = local
    learner = await new_learner(backend)
    # Bare names resolve under the backend's state_dir (never the CWD).
    assert backend.state_dir == backend.adapters_dir.parent / "states"
    assert await learner.save_state("step") == f"states/{backend.state_dir / 'step'}"
    assert await learner.save_state("/abs/step") == "states//abs/step"
    await learner.load_state("state.json", with_optimizer=False)
    assert learner.loaded == [("state.json", False)]
    for name in ("../escape", "", ".", "..", "/absolute", "has#suffix"):
        with pytest.raises(ConfigError, match="sampler name"):
            await learner.sync_sampler(name)
    assert not learner.saved and not calls


async def test_validate_before_loading_heavy_stack(local: tuple) -> None:
    backend, _, _ = local
    model = load_model("qwen3_5_4b")
    with pytest.raises(ConfigError, match="rank"):
        await backend.create_learner(
            "x", LearnerSpec(base_model=model.name, backend="local", rank=64), model=model, seed=0
        )
    assert backend.pool is None
    await new_learner(backend)
    with pytest.raises(ConfigError, match="already exists"):
        await new_learner(backend)
    other = load_model("qwen3_5_9b")
    with pytest.raises(ConfigError, match="same base"):
        await backend.create_learner(
            "y", LearnerSpec(base_model=other.name, backend="local"), model=other, seed=0
        )
    await new_learner(backend, "y")
    with pytest.raises(ConfigError, match="reserve"):
        await new_learner(backend, "z")


async def test_registry_and_names_survive_backend_close(local: tuple) -> None:
    backend, _, _ = local
    with pytest.raises(ConfigError, match="requires server_json and adapters_dir"):
        make_backend("local", spend=None)
    learner = await new_learner(backend)
    await learner.sync_sampler("run-x-s1")
    await backend.close()
    reopened = make_backend(
        "local",
        spend=None,
        server_json=str(backend.server_json),
        adapters_dir=str(backend.adapters_dir / "another-run"),
        device="cpu",
    )
    assert isinstance(reopened, LocalBackend)
    try:
        fresh = await new_learner(reopened)
        with pytest.raises(ConfigError, match="already used"):
            await fresh.sync_sampler("run-x-s1")
    finally:
        await reopened.close()


async def test_concurrent_publications_increase_versions(local: tuple) -> None:
    backend, _, _ = local
    learner = await new_learner(backend)
    snapshots = await asyncio.gather(
        learner.sync_sampler("run-x-s1"), learner.sync_sampler("run-x-s2")
    )
    assert [snapshot.version for snapshot in snapshots] == [1, 2]
    assert learner.policy().model == "run-x-s2" and learner.version == 2


async def test_init_from_publishes_restored_weights_at_the_recorded_version(local: tuple) -> None:
    backend, calls, _ = local
    checkpoint = Checkpoint(
        root=backend.adapters_dir.parent / "checkpoint",
        step=2,
        run_config_hash="hash",
        learners={
            "x": {
                "state": "state",
                "sampler": None,
                "version": 2,
                "base_model": "qwen3_5_4b",
                "backend": "local",
                "rank": 16,
            }
        },
    )
    checkpoint.save()
    spec = LearnerSpec(
        base_model="qwen3_5_4b", backend="local", rank=16, init_from=str(checkpoint.manifest_path)
    )
    learner = await backend.create_learner(
        "x", spec, model=replace(load_model(spec.base_model), renderer="fake"), seed=1
    )
    assert learner.loaded == [(str(checkpoint.root / "state"), True)]
    # A resume continues the checkpoint's version sequence (train rl asserts it).
    assert learner.version == 2 and learner.policy().policy_version == 2
    assert learner.policy().model.endswith("-x-init")
    assert calls[0].url.path == "/v1/load_lora_adapter"
    assert (await learner.sync_sampler("run-x-s3")).version == 3


@pytest.mark.parametrize(
    "record,match",
    [
        ({"backend": "tinker"}, "backend=local"),
        ({"rank": 8}, "rank"),
        ({"base_model": "qwen3_5_9b"}, "base_model"),
    ],
)
async def test_init_from_rejects_mismatched_checkpoints(
    local: tuple, record: dict[str, Any], match: str
) -> None:
    backend, _, _ = local
    checkpoint = Checkpoint(
        root=backend.adapters_dir.parent / "checkpoint",
        step=2,
        run_config_hash="hash",
        learners={
            "x": {
                "state": "state",
                "sampler": None,
                "version": 2,
                "base_model": "qwen3_5_4b",
                "backend": "local",
                "rank": 16,
                **record,
            }
        },
    )
    checkpoint.save()
    spec = LearnerSpec(
        base_model="qwen3_5_4b", backend="local", rank=16, init_from=str(checkpoint.manifest_path)
    )
    with pytest.raises(ConfigError, match=match):
        await backend.create_learner(
            "x", spec, model=replace(load_model(spec.base_model), renderer="fake"), seed=1
        )


async def test_close_preserves_adapters_owned_by_another_run(local: tuple) -> None:
    backend, calls, _ = local
    server = Server.load(backend.server_json)
    replace(
        server,
        models=[*server.models, "foreign"],
        adapters=[
            {
                "name": "foreign",
                "learner": "x",
                "owner": "other-run",
                "path": "/foreign",
                "owner_pid": os.getpid(),  # a live process on this host
                "owner_host": socket.gethostname(),
            }
        ],
    ).save()
    learner = await new_learner(backend)
    await learner.sync_sampler("run-x-s1")
    await learner.sync_sampler("run-x-s2")
    await backend.close()
    assert Server.load(backend.server_json).models == ["qwen3_5_4b", "foreign"]
    assert all(json.loads(call.content).get("lora_name") != "foreign" for call in calls)


@pytest.mark.parametrize("legacy", [False, True])
async def test_adapters_of_a_dead_run_are_evicted_before_publishing(
    local: tuple, legacy: bool
) -> None:
    # A killed run never calls close(): its adapters would hold every slot and
    # the resumed run could not publish (max_loras=3 leaves 2 snapshot slots).
    backend, calls, _ = local
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    owner = {} if legacy else {"owner_pid": dead.pid, "owner_host": socket.gethostname()}
    stale = [
        {"name": f"old-{i}", "learner": "x", "owner": "killed-run", "path": "/old", **owner}
        for i in range(2)
    ]
    server = Server.load(backend.server_json)
    replace(server, models=[*server.models, "old-0", "old-1"], adapters=stale).save()
    learner = await new_learner(backend)
    await learner.sync_sampler("run-x-s5")
    server = Server.load(backend.server_json)
    assert [a["name"] for a in server.adapters] == ["run-x-s5"]
    assert server.models == ["qwen3_5_4b", "run-x-s5"]
    unloaded = [
        json.loads(call.content)["lora_name"]
        for call in calls
        if call.url.path.endswith("unload_lora_adapter")
    ]
    assert unloaded == ["old-0", "old-1"]


@pytest.mark.parametrize(
    "score,error", [(-1.0, "ignored adapter"), (-0.9, None), (-0.7, "effect drift")]
)
async def test_hot_load_checks_effect_and_parity(
    local: tuple, score: float, error: str | None
) -> None:
    backend, _, _ = local
    learner = await new_learner(backend)
    assert learner.policy().policy_version == 0
    assert not hasattr(learner, "probe_tokens")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("completions"):
            assert body["echo"] is True and body["prompt_logprobs"] == 0
            assert body["max_tokens"] == 1 and body["temperature"] == 1
            return prompt_response(
                body["prompt"], -1 if body["model"] == backend.base_name else score
            )
        return httpx.Response(200, json={})

    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if error:
        with pytest.raises(BackendError, match=error):
            await learner.sync_sampler("checked")
        assert learner.version == 0 and learner.policy().model == backend.base_name
        assert not Server.load(backend.server_json).adapters
        assert requests[-1][0] == "/v1/unload_lora_adapter"
    else:
        await learner.sync_sampler("checked")
        assert learner.version == 1 and learner.policy().model == "checked"
    probes = [body for path, body in requests if path.endswith("completions")]
    assert len(probes) == 2
    assert probes[0]["prompt"] == probes[1]["prompt"] == list(learner.probe_tokens)


@pytest.mark.parametrize("inplace", [False, True])
async def test_adapter_names_option_keeps_immutable_exports(local: tuple, inplace: bool) -> None:
    backend, calls, _ = local
    assert backend.adapter_names == "versioned"
    backend.adapter_names = "inplace" if inplace else "versioned"
    learner = await new_learner(backend)
    first = await learner.sync_sampler("first")
    second = await learner.sync_sampler("second")
    loads = [
        json.loads(call.content) for call in calls if call.url.path.endswith("load_lora_adapter")
    ]
    assert loads[0] == {"lora_name": "first", "lora_path": first.path}
    assert loads[1] == {
        "lora_name": "first" if inplace else "second",
        "lora_path": second.path,
        **({"load_inplace": True} if inplace else {}),
    }
    assert first.path != second.path and all(path.exists() for path in learner.saved)
    assert second.version == 2
    assert second.policy_ref.endswith("#first" if inplace else "#second")
    adapters = Server.load(backend.server_json).adapters
    assert len(adapters) == (1 if inplace else 2) and adapters[-1]["path"] == second.path
    assert sum(call.url.path.endswith("completions") for call in calls) == 4


async def test_failed_inplace_load_blocks_sampling_until_verified(local: tuple) -> None:
    backend, _, replies = local
    backend.adapter_names = "inplace"
    learner = await new_learner(backend)
    await learner.sync_sampler("first")
    replies.append(httpx.ConnectError("uncertain load"))
    with pytest.raises(BackendError, match="uncertain load"):
        await learner.sync_sampler("second")
    assert learner.version == 1
    with pytest.raises(BackendError, match="sync_sampler must succeed"):
        learner.policy()
    await learner.sync_sampler("third")
    assert learner.version == 2 and learner.policy().model == "first"


@pytest.mark.parametrize(
    "options",
    [
        {"adapter_names": "wrong"},
        {"adapter_check_tol": -1},
        {"adapter_check_tol": float("nan")},
    ],
)
def test_invalid_adapter_options_fail_before_server_access(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="adapter_"):
        LocalBackend("missing.json", adapters_dir="unused", **options)


async def test_prompt_logprobs_rejects_malformed_scores(local: tuple) -> None:
    backend, _, _ = local
    await backend.client.aclose()
    for entries in ([None], [None, {"2": {"logprob": "nan"}}], [None, {"3": {"logprob": -1}}]):
        backend.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request, entries=entries: httpx.Response(
                    200, json={"choices": [{"prompt_logprobs": entries}]}
                )
            )
        )
        with pytest.raises(BackendError, match="invalid prompt_logprobs"):
            await backend._prompt_logprobs("adapter", [1, 2])
        await backend.client.aclose()


async def test_zero_initialized_adapter_allows_backend_roundoff(
    local: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _, _ = local
    learner = await new_learner(backend)

    async def zero_probe(tokens: list[int]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (-0.99,) * (len(tokens) - 1), (-0.99,) * (len(tokens) - 1)

    monkeypatch.setattr(learner, "probe_logprobs", zero_probe)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("echo"):
            return prompt_response(body["prompt"], -1.0)
        return httpx.Response(200, json={})

    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await learner.sync_sampler("zero-initialized")
    assert learner.version == 1


@pytest.mark.parametrize("inplace", [False, True])
async def test_bad_effect_check_preserves_publication_version(
    local: tuple,
    monkeypatch: pytest.MonkeyPatch,
    inplace: bool,
) -> None:
    backend, _, _ = local
    backend.adapter_names = "inplace" if inplace else "versioned"
    learner = await new_learner(backend)
    await learner.sync_sampler("first")

    async def changed_probe(tokens: list[int]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (-0.7,) * (len(tokens) - 1), (-1.0,) * (len(tokens) - 1)

    monkeypatch.setattr(learner, "probe_logprobs", changed_probe)
    with pytest.raises(BackendError, match="effect drift"):
        await learner.sync_sampler("rejected")
    assert learner.version == 1
    if inplace:
        with pytest.raises(BackendError, match="sync_sampler must succeed"):
            learner.policy()
    else:
        assert learner.policy().model == "first"
        assert [a["name"] for a in Server.load(backend.server_json).adapters] == ["first"]


async def test_effect_check_tolerance_is_configurable(local: tuple) -> None:
    backend, _, _ = local
    backend.adapter_check_tol = 0.2
    learner = await new_learner(backend)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("echo"):
            return prompt_response(
                body["prompt"], -1 if body["model"] == backend.base_name else -0.8
            )
        return httpx.Response(200, json={})

    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await learner.sync_sampler("tolerated")
    assert learner.version == 1
