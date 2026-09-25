"""Replicate one local learner pool over several devices with summed LoRA gradients.

Rank 0 is the caller's in-process pool; ranks 1.. are spawned worker processes
holding identical pools, driven over pipes. The RL loss is sum-reduced, so the
sum of per-shard gradients equals the full-batch gradient: every rank clips the
all-reduced gradient and takes the same optimizer step, which keeps parameters
identical (up to nothing: the reduced gradient is bitwise shared). Only rank 0
exports adapters, writes checkpoints and scores probes; every rank loads state.

A worker that dies or stops answering surfaces as a ``BackendError`` carrying
its log tail. Rank 0's compute runs in a daemon thread watched from the event
loop, so a peer lost inside a collective cannot hang the caller.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import multiprocessing
import os
import socket
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from marli.errors import BackendError, ConfigError
from marli.model import ModelSpec

if TYPE_CHECKING:
    import torch

    from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool

_T = TypeVar("_T")
_BUCKET_BYTES = 256 << 20
_LOG_TAIL_BYTES = 4000


def shard_by_tokens(lengths: Sequence[int], world: int) -> list[list[int]]:
    """Greedy longest-first: each item goes to the least-loaded rank (ties: lowest rank).

    Returns datum indices per rank, each ascending. Deterministic; rank 0 receives
    the longest item, so it is never empty when there is at least one item.
    """
    if world < 1:
        raise ValueError("world must be positive")
    loads, shards = [0] * world, [[] for _ in range(world)]
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(world), key=lambda r: (loads[r], r))
        shards[rank].append(index)
        loads[rank] += lengths[index]
    return [sorted(shard) for shard in shards]


def validate_devices(devices: Sequence[str]) -> tuple[str, ...]:
    """``cpu`` may repeat (gloo tests); CUDA devices must be explicit and distinct."""
    devices = tuple(devices)
    if not devices or any(not isinstance(d, str) or not d for d in devices):
        raise ConfigError("learner devices must be a non-empty list of device strings")
    for device in devices:
        if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
            raise ConfigError(f"unsupported learner device {device!r}; use cpu or cuda:N")
    if len(devices) > 1:
        if len({d == "cpu" for d in devices}) > 1:
            raise ConfigError("data-parallel learner devices must be all cpu or all cuda")
        if "cuda" in devices:
            raise ConfigError("data-parallel learner devices need explicit indices (cuda:N)")
        cuda = [d for d in devices if d != "cpu"]
        if len(set(cuda)) != len(cuda):
            raise ConfigError(f"duplicate learner devices: {list(devices)}")
    return devices


@dataclass(frozen=True)
class PoolRecipe:
    """How every rank builds an identical frozen base (picklable for spawned workers).

    ``factory`` (``module:function`` returning a ``PreTrainedModel``) replaces the
    registry download, e.g. for tiny test models; ``model`` may then be None.
    """

    model: ModelSpec | None
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    attn_implementation: str | None = None
    factory: str | None = None
    tokenizer_sha: str | None = None

    def build(self, device: str) -> LocalLearnerPool:
        from marli.train.backends.local.learner import LocalLearnerPool

        if self.factory:
            module, _, name = self.factory.partition(":")
            model = getattr(importlib.import_module(module), name)()
            return LocalLearnerPool.from_model(
                model,
                tokenizer_sha=self.tokenizer_sha or "",
                device=device,
                gradient_checkpointing=self.gradient_checkpointing,
                model_spec=self.model,
            )
        if self.model is None:
            raise ConfigError("a pool recipe needs a model spec or a factory")
        return LocalLearnerPool(
            self.model,
            device=device,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
            attn_implementation=self.attn_implementation,
        )


# ---------------------------------------------------------------- collectives (every rank)


def _dist() -> Any:
    import torch.distributed as dist

    return dist


def parameter_digest(parameters: Iterable[torch.Tensor]) -> float:
    """A float64 fingerprint of parameter values (identical models give identical digests)."""
    import torch

    total = 0.0
    for index, parameter in enumerate(parameters):
        # Chunked float64 sums bound the temporary copy (an embedding can be >1e9 values);
        # position weighting catches permuted but otherwise equal tensors.
        flat = parameter.detach().reshape(-1)
        value = sum(
            float(chunk.to(torch.float64).sum()) for chunk in flat.split(1 << 24)
        )
        total += (index + 1) * value
    return total


def _all_equal(value: float, device: torch.device) -> bool:
    import torch

    dist = _dist()
    mine = torch.tensor([value], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, mine)
    # Different checkpoints differ by far more than float64 reduction noise.
    return all(math.isclose(float(t.item()), value, rel_tol=1e-9, abs_tol=1e-6) for t in gathered)


def all_ok(ok: bool, device: torch.device) -> bool:
    """Agree on whether every rank's local pass succeeded (keeps collectives aligned)."""
    import torch

    flag = torch.tensor([0.0 if ok else 1.0], device=device)
    _dist().all_reduce(flag)
    return float(flag.item()) == 0.0


def broadcast_parameters(parameters: Sequence[torch.Tensor]) -> None:
    for parameter in parameters:
        _dist().broadcast(parameter.data, src=0)


def all_reduce_gradients(parameters: Sequence[torch.Tensor]) -> None:
    """Sum gradients across ranks in dtype-homogeneous buckets; missing grads count as 0."""
    import torch

    dist = _dist()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    bucket: list[torch.Tensor] = []
    size = 0

    def flush() -> None:
        nonlocal bucket, size
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat)
        offset = 0
        for p in bucket:
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n
        bucket, size = [], 0

    for parameter in parameters:
        if bucket and (
            parameter.grad.dtype != bucket[0].grad.dtype
            or parameter.grad.device != bucket[0].grad.device
            or size + parameter.grad.numel() * parameter.grad.element_size() > _BUCKET_BYTES
        ):
            flush()
        bucket.append(parameter)
        size += parameter.grad.numel() * parameter.grad.element_size()
    flush()


def _init_group(rank: int, world: int, port: int, device: str, timeout_s: float) -> None:
    import torch

    dist = _dist()
    if dist.is_initialized():
        raise ConfigError("only one data-parallel learner pool may exist per process")
    kwargs: dict[str, Any] = {}
    backend = "gloo"
    if device.startswith("cuda"):
        backend = "nccl"
        torch.cuda.set_device(torch.device(device))
        kwargs["device_id"] = torch.device(device)
    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout_s),
        **kwargs,
    )


def _check_identical_base(pool: LocalLearnerPool) -> None:
    if not _all_equal(parameter_digest(pool.base_model.parameters()), pool.device):
        raise BackendError("data-parallel ranks loaded different base weights")


def _empty_cache(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- worker process (ranks 1..)


def _worker_main(
    conn: Any,
    rank: int,
    world: int,
    port: int,
    device: str,
    recipe: PoolRecipe,
    timeout_s: float,
    log_path: str,
    parent_pid: int,
) -> None:
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    learners: dict[str, LocalLearner] = {}
    try:
        import torch  # noqa: F401  (import cost is paid before the rendezvous)

        conn.send(("started", None))
        _init_group(rank, world, port, device, timeout_s)
        pool = recipe.build(device)
        conn.send(("built", None))
        _check_identical_base(pool)
        conn.send(("ready", None))
    except BaseException:
        traceback.print_exc()
        try:
            conn.send(("error", traceback.format_exc()))
        finally:
            os._exit(1)
    handlers: dict[str, Callable[..., Any]] = {
        "create": lambda *a: _worker_create(pool, learners, *a),
        "train": lambda *a: _worker_train(pool, learners, *a),
        "load_state": lambda name, path, with_optimizer: learners[name]._load_state_here(
            Path(path), with_optimizer
        ),
        "digest": lambda name: parameter_digest(learners[name].parameters),
        "sleep": lambda seconds: time.sleep(seconds),
        "exit": lambda code: os._exit(code),
    }
    while True:
        if not conn.poll(1.0):
            if os.getppid() != parent_pid:  # the trainer is gone: never linger as an orphan
                break
            continue
        try:
            op, args = conn.recv()
        except (EOFError, OSError):
            break
        if op == "close":
            conn.send(("ok", None))
            break
        try:
            conn.send(("ok", handlers[op](*args)))
        except BaseException:
            traceback.print_exc()
            conn.send(("error", traceback.format_exc()))
    try:
        _dist().destroy_process_group()
    finally:
        os._exit(0)


def _worker_create(
    pool: LocalLearnerPool,
    learners: dict[str, LocalLearner],
    name: str,
    spec: Any,
    kwargs: dict[str, Any],
) -> None:
    from marli.train.backends.local.learner import LocalLearner

    learner = LocalLearner(name, spec, pool, **kwargs)
    broadcast_parameters(learner.parameters)
    learners[name] = learner


def _worker_train(
    pool: LocalLearnerPool,
    learners: dict[str, LocalLearner],
    name: str,
    datums: Sequence[Any],
    learning_rate: float | None,
) -> dict[str, Any] | None:
    learner = learners[name]
    learner._begin_step(learning_rate)
    partial: dict[str, Any] | None = None
    error: str | None = None
    try:
        try:
            partial = learner._local_pass(datums)
        except Exception:
            error = traceback.format_exc()
        if not all_ok(error is None, pool.device):
            if error is not None:
                raise BackendError(f"local pass failed:\n{error}")
            return None  # another rank failed; it reports the cause
        all_reduce_gradients(learner.parameters)
        partial["grad_norm"] = learner._apply_gradients()
        return partial
    finally:
        learner.optimizer.zero_grad(set_to_none=True)
        _empty_cache(pool.device)


# ---------------------------------------------------------------- rank-0 handle


@dataclass
class _Worker:
    rank: int
    process: Any
    conn: Any
    log: Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Replicas:
    """Rank 0's view of the worker ranks: start, command, watch and stop them."""

    def __init__(
        self,
        recipe: PoolRecipe,
        devices: Sequence[str],
        *,
        timeout_s: float = 1800.0,
        start_timeout_s: float = 1800.0,
        log_dir: str | Path | None = None,
    ) -> None:
        self.devices = validate_devices(devices)
        if len(self.devices) < 2:
            raise ConfigError("data parallelism needs at least two devices")
        self.world = len(self.devices)
        self.timeout_s = timeout_s
        self._broken: BackendError | None = None
        self._closed = False
        self._group = False
        self.log_dir = Path(log_dir or tempfile.mkdtemp(prefix="marli-dp-"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        port = _free_port()
        context = multiprocessing.get_context("spawn")
        self.workers: list[_Worker] = []
        try:
            for rank in range(1, self.world):
                parent, child = context.Pipe()
                log = self.log_dir / f"rank{rank}.log"
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child,
                        rank,
                        self.world,
                        port,
                        self.devices[rank],
                        recipe,
                        timeout_s,
                        str(log),
                        os.getpid(),
                    ),
                    name=f"marli-dp-rank{rank}",
                    daemon=True,
                )
                process.start()
                child.close()
                self.workers.append(_Worker(rank, process, parent, log))
            for worker in self.workers:
                self._expect(worker, "started", start_timeout_s)
            _init_group(0, self.world, port, self.devices[0], timeout_s)
            self._group = True
        except BaseException:
            self.close()
            raise

    def after_build(self, pool: LocalLearnerPool, start_timeout_s: float = 1800.0) -> None:
        """Rank 0 built its pool: agree the bases are identical, then wait for readiness."""
        try:
            for worker in self.workers:
                self._expect(worker, "built", start_timeout_s)
            _check_identical_base(pool)
            for worker in self.workers:
                self._expect(worker, "ready", start_timeout_s)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ messaging

    def _tail(self, worker: _Worker) -> str:
        try:
            data = worker.log.read_bytes()[-_LOG_TAIL_BYTES:]
        except OSError:
            return "(no log)"
        return data.decode("utf-8", errors="replace") or "(empty log)"

    def _failure(self, worker: _Worker, what: str) -> BackendError:
        error = BackendError(
            f"data-parallel learner rank {worker.rank} ({self.devices[worker.rank]}) {what}; "
            f"log {worker.log} tail:\n{self._tail(worker)}"
        )
        self._broken = error
        return error

    def _recv(self, worker: _Worker, timeout_s: float) -> tuple[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if worker.conn.poll(0.2):
                try:
                    return worker.conn.recv()
                except (EOFError, OSError) as exc:
                    raise self._failure(worker, f"closed its pipe ({exc})") from exc
            if not worker.process.is_alive():
                if worker.conn.poll(0):
                    continue
                raise self._failure(worker, f"exited with code {worker.process.exitcode}")
            if time.monotonic() > deadline:
                raise self._failure(worker, f"did not reply within {timeout_s:g}s")

    def _expect(self, worker: _Worker, kind: str, timeout_s: float) -> Any:
        status, payload = self._recv(worker, timeout_s)
        if status == "error":
            raise self._failure(worker, f"failed:\n{payload}")
        if status != kind:
            raise self._failure(worker, f"sent {status!r}, expected {kind!r}")
        return payload

    @property
    def broken(self) -> bool:
        return self._broken is not None

    def raise_if_broken(self) -> None:
        if self._broken is not None:
            raise BackendError(f"data-parallel learner pool is unusable: {self._broken}")
        if self._closed:
            raise BackendError("data-parallel learner pool is closed")

    def send(self, op: str, per_rank: Sequence[tuple[Any, ...]] | tuple[Any, ...]) -> None:
        """Send ``op`` to every worker; ``per_rank`` is one args tuple or one per worker."""
        self.raise_if_broken()
        many = isinstance(per_rank, list)
        for index, worker in enumerate(self.workers):
            args = per_rank[index] if many else per_rank
            try:
                worker.conn.send((op, args))
            except (OSError, BrokenPipeError) as exc:
                raise self._failure(worker, f"refused a command ({exc})") from exc

    def collect(self, timeout_s: float | None = None) -> list[Any]:
        """One reply per worker, in rank order; worker errors become BackendError."""
        replies, errors = [], []
        for worker in self.workers:
            status, payload = self._recv(worker, self.timeout_s if timeout_s is None else timeout_s)
            if status == "error":
                errors.append(f"rank {worker.rank}: {payload}")
            replies.append(payload)
        if errors:
            raise BackendError("data-parallel learner failed:\n" + "\n".join(errors))
        return replies

    def call(self, op: str, *args: Any, timeout_s: float | None = None) -> list[Any]:
        self.send(op, args)
        return self.collect(timeout_s)

    def dead_worker(self) -> _Worker | None:
        return next((w for w in self.workers if not w.process.is_alive()), None)

    async def run_watched(self, operation: Callable[[], _T]) -> _T:
        """Run rank 0's part in a daemon thread; a lost worker fails fast instead of hanging."""
        self.raise_if_broken()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_T] = loop.create_future()

        def settle(result: Any, error: BaseException | None) -> None:
            if future.done():
                return
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

        def target() -> None:
            try:
                if self.devices[0].startswith("cuda"):
                    import torch

                    # The current CUDA device is per thread; NCCL expects rank 0's.
                    torch.cuda.set_device(torch.device(self.devices[0]))
                result = operation()
            except BaseException as exc:  # delivered to the awaiting task
                loop.call_soon_threadsafe(settle, None, exc)
            else:
                loop.call_soon_threadsafe(settle, result, None)

        thread = threading.Thread(target=target, name="marli-dp-rank0", daemon=True)
        self._thread = thread
        thread.start()
        cancellation: asyncio.CancelledError | None = None
        while not future.done():
            try:
                await asyncio.wait({future}, timeout=0.5)
            except asyncio.CancelledError as exc:
                # Like the single-device pool: torch cannot be interrupted, so the
                # pool stays locked until rank 0's thread finishes.
                cancellation = exc
            dead = self.dead_worker()
            if dead is not None and not future.done():
                raise self._failure(dead, f"exited with code {dead.process.exitcode}")
        try:
            result = future.result()
        except BaseException as exc:
            dead = self.dead_worker()
            if dead is not None and not isinstance(exc, BackendError):
                raise self._failure(dead, f"exited with code {dead.process.exitcode}") from exc
            raise
        if cancellation is not None:
            raise cancellation
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self.workers:
            if worker.process.is_alive():
                try:
                    worker.conn.send(("close", ()))
                    if worker.conn.poll(30):
                        worker.conn.recv()
                except (OSError, EOFError):
                    pass
        for worker in self.workers:
            worker.process.join(10)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(5)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(5)
            worker.conn.close()
        thread = getattr(self, "_thread", None)
        if self._group and (thread is None or not thread.is_alive()):
            # Never tear the group down under a rank-0 thread still inside a collective.
            self._group = False
            with suppress(Exception):  # a broken group must not mask the caller's error
                _dist().destroy_process_group()
