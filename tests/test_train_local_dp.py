"""Data-parallel local learners (gloo on CPU) must match one device and fail loudly."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from marli.errors import BackendError, ConfigError
from marli.train.backends.local.parallel import shard_by_tokens, validate_devices


def test_shards_are_deterministic_balanced_and_rank0_first() -> None:
    lengths = [5, 40, 7, 40, 12, 3, 9]
    shards = shard_by_tokens(lengths, 2)
    assert shards == shard_by_tokens(lengths, 2)
    assert sorted(i for shard in shards for i in shard) == list(range(len(lengths)))
    loads = [sum(lengths[i] for i in shard) for shard in shards]
    assert abs(loads[0] - loads[1]) <= max(lengths)
    assert 1 in shards[0]  # the first of the longest items goes to rank 0
    assert shard_by_tokens([10], 2) == [[0], []]
    assert shard_by_tokens([], 2) == [[], []]
    with pytest.raises(ValueError):
        shard_by_tokens([1], 0)


@pytest.mark.parametrize(
    "devices,match",
    [
        ([], "non-empty"),
        (["cuda:0", "cuda:0"], "duplicate"),
        (["cuda", "cuda:1"], "explicit indices"),
        (["cpu", "cuda:0"], "all cpu or all cuda"),
        (["tpu:0"], "unsupported"),
    ],
)
def test_invalid_devices(devices: list[str], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        validate_devices(devices)
    assert validate_devices(["cpu", "cpu"]) == ("cpu", "cpu")
    assert validate_devices(["cuda:0", "cuda:1"]) == ("cuda:0", "cuda:1")


torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from _marli_tiny_models import tiny_qwen3  # noqa: E402
from test_train_local_learner import adapter_weights, datum  # noqa: E402

from marli.render.fake import FakeRenderer  # noqa: E402
from marli.train.backends.local import LocalLearner, LocalLearnerPool  # noqa: E402
from marli.train.backends.local.parallel import PoolRecipe, parameter_digest  # noqa: E402
from marli.train.types import LearnerSpec  # noqa: E402

FACTORY = "_marli_tiny_models:tiny_qwen3"


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Spawned ranks inherit the environment: keep the contended test box usable.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def recipe(factory: str = FACTORY) -> PoolRecipe:
    return PoolRecipe(model=None, factory=factory, tokenizer_sha=FakeRenderer.tokenizer_sha)


def dp_pool(tmp_path: Path, factory: str = FACTORY, timeout_s: float = 60) -> LocalLearnerPool:
    return LocalLearnerPool.data_parallel(
        recipe(factory), ["cpu", "cpu"], timeout_s=timeout_s, log_dir=tmp_path / "dp"
    )


def learner(pool: LocalLearnerPool, name: str = "alice") -> LocalLearner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        return LocalLearner(
            name, LearnerSpec(backend="local", rank=4, learning_rate=1e-3), pool
        )


def single() -> LocalLearner:
    return learner(
        LocalLearnerPool.from_model(tiny_qwen3(), tokenizer_sha=FakeRenderer.tokenizer_sha)
    )


TEXTS = [("hi", "ok"), ("a much longer prompt about sums", "five"), ("x", "a long answer here")]


def assert_same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)


def worker_digest(pool: LocalLearnerPool, name: str) -> float:
    assert pool.replicas is not None
    return pool.replicas.call("digest", name)[0]


@pytest.fixture(scope="class")
def pool(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalLearnerPool]:
    """One pool per class: a spawned rank costs a torch import, so tests share it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OMP_NUM_THREADS", "1")
        shared = dp_pool(tmp_path_factory.mktemp("healthy"))
    try:
        yield shared
    finally:
        shared.close()
        assert shared.replicas is not None
        assert not any(w.process.is_alive() for w in shared.replicas.workers)


class TestHealthyPool:
    """One adapter per test on the shared pool."""

    async def test_matches_one_device_over_two_steps(self, pool: LocalLearnerPool) -> None:
        reference = single()
        datums = [datum(reference, text, answer) for text, answer in TEXTS]
        parallel = learner(pool, "alice")  # the reference learner is also "alice"
        assert parameter_digest(parallel.parameters) == worker_digest(pool, "alice")
        assert_same(adapter_weights(parallel), adapter_weights(reference))
        for _ in range(2):
            expected = await reference.train_step(datums)
            result = await parallel.train_step(datums)
            assert (result.n_datums, result.n_tokens, result.n_action_tokens) == (
                expected.n_datums,
                expected.n_tokens,
                expected.n_action_tokens,
            )
            for field in ("loss", "grad_norm", "kl_sample_train"):
                assert getattr(result, field) == pytest.approx(getattr(expected, field), rel=1e-5)
            assert result.metrics.keys() == expected.metrics.keys()
            for key, value in expected.metrics.items():
                assert result.metrics[key] == pytest.approx(value, rel=1e-5, abs=1e-7)
            # The reduced gradient is shared, so the ranks agree bitwise.
            assert parameter_digest(parallel.parameters) == worker_digest(pool, "alice")
        assert_same(adapter_weights(parallel), adapter_weights(reference))
        assert parallel.step == reference.step == 2

    async def test_one_datum_leaves_the_worker_shard_empty(self, pool: LocalLearnerPool) -> None:
        reference = single()
        datums = [datum(reference, "a single datum", "only")]
        parallel = learner(pool, "bob")
        expected = await reference.train_step(datums)
        result = await parallel.train_step([replace(d, learner="bob") for d in datums])
        assert result.loss == pytest.approx(expected.loss, rel=1e-5)
        assert parameter_digest(parallel.parameters) == worker_digest(pool, "bob")
        assert_same(adapter_weights(parallel), adapter_weights(reference))

    async def test_state_round_trip_restores_every_rank(
        self, pool: LocalLearnerPool, tmp_path: Path
    ) -> None:
        reference = single()
        datums = [
            replace(datum(reference, text, answer), learner="carol") for text, answer in TEXTS
        ]
        parallel = learner(pool, "carol")
        await parallel.train_step(datums)
        saved = await parallel.save_state(str(tmp_path / "states"))
        after_one = parameter_digest(parallel.parameters)
        await parallel.train_step(datums)
        after_two = parameter_digest(parallel.parameters)
        assert after_two != after_one
        await parallel.load_state(saved)
        assert parallel.step == 1
        assert parameter_digest(parallel.parameters) == worker_digest(pool, "carol") == after_one
        await parallel.train_step(datums)
        assert parameter_digest(parallel.parameters) == worker_digest(pool, "carol") == after_two


async def test_dead_worker_fails_fast_and_is_reaped(tmp_path: Path) -> None:
    reference = single()
    datums = [datum(reference, text, answer) for text, answer in TEXTS]
    pool = dp_pool(tmp_path, timeout_s=20)
    try:
        parallel = learner(pool)
        assert pool.replicas is not None
        worker = pool.replicas.workers[0].process
        pool.replicas.send("exit", (3,))
        worker.join(20)
        start = time.monotonic()
        with pytest.raises(BackendError, match="rank 1"):
            await parallel.train_step(datums)
        assert time.monotonic() - start < 15
        with pytest.raises(BackendError, match="unusable"):
            await parallel.train_step(datums)
    finally:
        pool.close()
    assert not worker.is_alive() and worker.exitcode == 3


async def test_unresponsive_worker_times_out_with_its_log(tmp_path: Path) -> None:
    pool = dp_pool(tmp_path, timeout_s=20)
    try:
        assert pool.replicas is not None
        start = time.monotonic()
        with pytest.raises(BackendError, match="did not reply within 0.5s") as info:
            pool.replicas.call("sleep", 5, timeout_s=0.5)
        assert time.monotonic() - start < 5
        assert str(pool.replicas.workers[0].log) in str(info.value)
    finally:
        pool.close()
    assert not pool.replicas.workers[0].process.is_alive()


def test_worker_load_failure_surfaces_with_log_tail(tmp_path: Path) -> None:
    start = time.monotonic()
    with pytest.raises(BackendError, match="simulated worker load failure"):
        dp_pool(tmp_path, factory="_marli_tiny_models:tiny_qwen3_fails_in_worker", timeout_s=20)
    assert time.monotonic() - start < 60
    assert not [p for p in _children() if p.name.startswith("marli-dp-")]


def test_ranks_with_different_base_weights_refuse_to_train(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="different base weights"):
        dp_pool(tmp_path, factory="_marli_tiny_models:tiny_qwen3_other_seed", timeout_s=20)
    assert not [p for p in _children() if p.name.startswith("marli-dp-")]


def _children() -> list:
    import multiprocessing

    return [p for p in multiprocessing.active_children() if p.pid != os.getpid()]
