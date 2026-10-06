"""Verify exported adapters before exposing their serving policies.

One backend owns one shared training base. Publication is serialized across its
learners: load the new snapshot, switch the current policy, then evict an old
snapshot. Versioned names protect the prefix cache by default; inplace names
are an opt-in trade-off. Exported directories survive unloading and shutdown.

Time sharing (``sleep_sampler``): the base may sit on the same GPUs as the
server (optionally data-parallel over several, ``devices``). vLLM sleeps before
the first train step after sampling (weights to CPU, KV cache freed) and wakes
before the next adapter is loaded and probe-checked. The server is never left
asleep: close and failed steps wake it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx

from marli.budget import SpendGuard
from marli.errors import BackendError, ConfigError
from marli.model import ModelSpec
from marli.policy.vllm import VLLMPolicy, http_client
from marli.render.registry import get_renderer
from marli.serve.vllm import Server
from marli.train.backends.base import SamplerSnapshot, StepResult
from marli.train.backends.local.parallel import validate_devices
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool
    from marli.train.types import TrainDatum


class _SamplerMixin:
    """Add serving to the lazy learner subclass without wrapping its optimizer."""

    name: str
    spec: LearnerSpec
    model: ModelSpec
    version: int
    _backend: LocalBackend
    _current_adapter: str | None = None
    _sampler_error: bool = False

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        return await self._backend._publish(self, name)

    def policy(self, *, policy_id: str | None = None) -> VLLMPolicy:
        if self._sampler_error:
            raise BackendError(
                "inplace adapter update failed; sync_sampler must succeed before sampling"
            )
        return VLLMPolicy(
            self.name if policy_id is None else policy_id,
            self._backend.base_url,
            self._current_adapter or self._backend.base_name,
            renderer_name=self.model.renderer,
            trainable=True,
            policy_version=self.version,
            client=self._backend.client,
        )

    async def close(self) -> None:
        await self._backend._close_learner(self.name)
        await super().close()


# A fixed probe (math prose + code) scored in both engines after every hot-load.
_PROBE_TEXT = (
    "The sum of two and three is five. To check it in Python: print(2 + 3)\n"
    "Let x be a positive integer with x^2 - 5x + 6 = 0. Factoring gives (x - 2)(x - 3) = 0, "
    "so x = 2 or x = 3, and the sum of all solutions is 5. Therefore the answer is \\boxed{5}.\n"
)


def _owner_alive(adapter: dict[str, object]) -> bool:
    """Whether the process that published ``adapter`` may still be using it.

    Records from before owner pids were stored, and records from a dead process
    on this host (a killed or crashed run), are stale. Other hosts are assumed live.
    """
    pid, host = adapter.get("owner_pid"), adapter.get("owner_host")
    if not isinstance(pid, int):
        return False
    if host != socket.gethostname():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalBackend:
    name = "local"

    def __init__(
        self,
        server_json: str,
        *,
        adapters_dir: str,
        device: str = "cuda",
        devices: Sequence[str] | None = None,
        sleep_sampler: bool = False,
        sleep_timeout_s: float = 600.0,
        spend: SpendGuard | None = None,
        adapter_check_tol: float = 0.05,
        adapter_names: str = "versioned",
        state_dir: str | None = None,
    ) -> None:
        if not isfinite(adapter_check_tol) or adapter_check_tol <= 0:
            raise ConfigError("adapter_check_tol must be finite and positive")
        if adapter_names not in {"versioned", "inplace"}:
            raise ConfigError("adapter_names must be 'versioned' or 'inplace'")
        if type(sleep_sampler) is not bool:
            raise ConfigError("sleep_sampler must be a boolean")
        if not isfinite(sleep_timeout_s) or sleep_timeout_s <= 0:
            raise ConfigError("sleep_timeout_s must be finite and positive")
        # One device, or several for a data-parallel base (the first is rank 0).
        self.devices = validate_devices(devices) if devices is not None else (device,)
        device = self.devices[0]
        self.adapter_check_tol = adapter_check_tol
        self.adapter_names = adapter_names
        self.server_json = Path(server_json).resolve()
        server = Server.load(self.server_json)
        if not server.models or not server.enable_lora or server.max_loras < 2:
            raise ConfigError("local training requires a LoRA server with max_loras >= 2")
        if sleep_sampler and not server.enable_sleep_mode:
            raise ConfigError(
                "sleep_sampler requires a server started with enable_sleep_mode=true "
                f"({self.server_json} does not record it)"
            )
        self.sleep_sampler = sleep_sampler
        self.sleep_timeout_s = sleep_timeout_s
        self._asleep = False
        self._training = 0
        self._sleep_state = asyncio.Condition()
        self._step_metrics: dict[str, float] = {}
        self.base_url = server.base_url.rstrip("/")
        self.base_name = server.models[0]
        self.adapters_dir = Path(adapters_dir).resolve()
        # Learner state (adapter + optimizer) for checkpoints. Bare names from the
        # loop resolve here, never against the CWD (which is the git checkout).
        self.state_dir = (
            Path(state_dir).resolve() if state_dir else self.adapters_dir.parent / "states"
        )
        self.device = device
        self.spend = spend
        self.pool: LocalLearnerPool | None = None
        self.learners: dict[str, LocalLearner] = {}
        self.client = http_client(600, trust_env=False)
        self._lock = asyncio.Lock()
        self._owned: dict[str, str] = {}
        self._owner = uuid4().hex

    async def create_learner(
        self, name: str, spec: LearnerSpec, *, model: ModelSpec, seed: int
    ) -> LocalLearner:
        if name in self.learners:
            raise ConfigError(f"learner {name!r} already exists")
        server = Server.load(self.server_json)
        if spec.backend != "local" or spec.base_model != model.name:
            raise ConfigError("local learner spec must identify its supplied base model")
        if model.name != self.base_name or (server.hf_id and server.hf_id != model.hf_id):
            raise ConfigError("local learners and server must use the same base model")
        if self.pool is not None and self.pool.model.hf_id != model.hf_id:
            raise ConfigError("local learners must share one base model per run")
        if spec.rank > server.max_lora_rank:
            raise ConfigError("learner rank exceeds server max_lora_rank")
        if server.lora_target_modules:
            # vLLM silently drops adapter weights for modules it did not wrap.
            targets = model.lora.get("target_modules", [])
            missing = sorted(set(targets) - set(server.lora_target_modules))
            if not targets or missing:
                raise ConfigError(
                    f"server LoRA is restricted to {server.lora_target_modules}; the learner "
                    f"would also train {missing or 'every nn.Linear (no registry targets)'}"
                )
        if len(self.learners) + 2 > server.max_loras:
            raise ConfigError("max_loras must reserve one slot beyond the learner count")
        # Like Tinker: a checkpoint manifest restores its recorded sampler version
        # (a resume continues the version sequence the datums were sampled under).
        state, version = None, 0
        if spec.init_from is not None:
            checkpoint = Checkpoint.load(spec.init_from)
            state = checkpoint.require_state(name)
            record = checkpoint.learners[name]
            if record["backend"] != "local":
                raise ConfigError("local init_from requires a backend=local checkpoint")
            if record["base_model"] != model.name:
                raise ConfigError("local checkpoint base_model does not match the model")
            if record["rank"] != spec.rank:
                raise ConfigError("local checkpoint rank does not match learner.rank")
            version = record["version"]

        # M4-1 owns torch/PEFT construction, training and checkpoint I/O. Its
        # constructors are synchronous; no heavy import occurs before validation.
        import torch

        from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool

        class ServingLearner(_SamplerMixin, LocalLearner):
            async def train_step(
                self, datums: Sequence[TrainDatum], *, learning_rate: float | None = None
            ) -> StepResult:
                await self._backend._enter_training()
                failed = True
                try:
                    result = await LocalLearner.train_step(
                        self, datums, learning_rate=learning_rate
                    )
                    failed = False
                    return result
                finally:
                    await self._backend._leave_training(failed)

            async def save_state(self, name: str) -> str:
                path = Path(name)
                if not path.is_absolute():
                    path = self._backend.state_dir / path
                return await LocalLearner.save_state(self, str(path))

        torch.manual_seed(seed)
        if self.pool is None:
            if len(self.devices) > 1:
                from marli.train.backends.local.parallel import PoolRecipe

                self.pool = LocalLearnerPool.data_parallel(
                    PoolRecipe(model),
                    self.devices,
                    log_dir=self.state_dir.parent / "learner-ranks",
                )
            else:
                self.pool = LocalLearnerPool(model, device=self.device)
        learner = ServingLearner(name, spec, self.pool)
        learner._backend = self
        self.learners[name] = learner
        if state is not None:
            await learner.load_state(state)
            await self._publish(learner, f"{self._owner}-{name}-init", version=version)
        return learner

    # ------------------------------------------------------------ time sharing

    def _metric(self, key: str, seconds: float) -> None:
        self._step_metrics[key] = self._step_metrics.get(key, 0.0) + seconds

    def pop_step_metrics(self) -> dict[str, float]:
        """Seconds spent putting the server to sleep and waking it since the last call."""
        metrics, self._step_metrics = self._step_metrics, {}
        return metrics

    async def _control(
        self, method: str, path: str, params: dict[str, str] | None = None
    ) -> httpx.Response:
        # vLLM's sleep endpoints live at the server root, not under /v1.
        root = self.base_url.removesuffix("/v1")
        try:
            response = await self.client.request(method, root + path, params=params)
        except httpx.TransportError as exc:
            raise BackendError(f"vLLM {path}: {exc}") from exc
        if response.status_code == 404:
            raise ConfigError(
                f"vLLM {path} is not served: start the server with enable_sleep_mode=true "
                "(it also sets VLLM_SERVER_DEV_MODE=1)"
            )
        if 400 <= response.status_code < 500:
            raise ConfigError(f"vLLM {path}: HTTP {response.status_code}: {response.text[:500]}")
        if not response.is_success:
            raise BackendError(f"vLLM {path}: HTTP {response.status_code}: {response.text[:500]}")
        return response

    async def _await_sleeping(self, target: bool) -> None:
        deadline = perf_counter() + self.sleep_timeout_s
        while True:
            response = await self._control("GET", "/is_sleeping")
            try:
                state = response.json()["is_sleeping"]
            except (ValueError, KeyError, TypeError) as exc:
                raise BackendError(
                    f"vLLM /is_sleeping: invalid reply {response.text[:200]}"
                ) from exc
            if state is target:
                return
            if perf_counter() > deadline:
                raise BackendError(
                    f"vLLM did not {'sleep' if target else 'wake'} within {self.sleep_timeout_s:g}s"
                )
            await asyncio.sleep(0.2)

    async def _wake(self) -> None:
        start = perf_counter()
        await self._control("POST", "/wake_up")
        await self._await_sleeping(False)
        self._asleep = False
        self._metric("wake_s", perf_counter() - start)

    async def _enter_training(self) -> None:
        """Count a train step in; the first one after sampling puts the server to sleep."""
        async with self._sleep_state:
            self._training += 1
            if not self.sleep_sampler or self._asleep:
                return
            try:
                start = perf_counter()
                # mode=wait: nothing should be in flight in the sync loop; never abort it.
                await self._control("POST", "/sleep", {"level": "1", "mode": "wait"})
                await self._await_sleeping(True)
            except BaseException:
                self._training -= 1
                self._sleep_state.notify_all()
                raise
            self._asleep = True
            self._metric("sleep_s", perf_counter() - start)

    async def _leave_training(self, failed: bool) -> None:
        async with self._sleep_state:
            self._training -= 1
            self._sleep_state.notify_all()
            if failed and self._training == 0 and self._asleep:
                # Never leave the server asleep after a failed step.
                try:
                    await self._wake()
                except (BackendError, ConfigError) as exc:
                    log.warning("could not wake vLLM after a failed train step: %s", exc)

    async def _ensure_awake(self) -> None:
        """Wake the server once every train step has finished (before loading adapters)."""
        async with self._sleep_state:
            await self._sleep_state.wait_for(lambda: self._training == 0)
            if self._asleep:
                await self._wake()

    async def _request(self, endpoint: str, body: dict[str, object]) -> httpx.Response:
        root = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        try:
            response = await self.client.post(root + endpoint, json=body)
        except httpx.TransportError as exc:
            raise BackendError(f"vLLM {endpoint}: {exc}") from exc
        if 400 <= response.status_code < 500:
            raise ConfigError(
                f"vLLM {endpoint}: HTTP {response.status_code}: {response.text[:500]}"
            )
        if not response.is_success:
            raise BackendError(
                f"vLLM {endpoint}: HTTP {response.status_code}: {response.text[:500]}"
            )
        return response

    async def _prompt_logprobs(self, model: str, tokens: list[int]) -> tuple[float, ...]:
        # vLLM 0.30 returns a separate prompt_logprobs field (first position is
        # null); logprobs.token_logprobs describes generated tokens, not this probe.
        response = await self._request(
            "/completions",
            {
                "model": model,
                "prompt": tokens,
                "max_tokens": 1,
                "echo": True,
                "prompt_logprobs": 0,
                "temperature": 1.0,
            },
        )
        try:
            entries = response.json()["choices"][0]["prompt_logprobs"]
            if len(entries) != len(tokens):
                raise ValueError("prompt length mismatch")
            scores = tuple(
                float(entry[str(token)]["logprob"])
                for token, entry in zip(tokens[1:], entries[1:], strict=True)
            )
            if not all(isfinite(score) for score in scores):
                raise ValueError("nonfinite prompt logprobs")
            return scores
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"vLLM invalid prompt_logprobs for {model!r}: {exc}") from exc

    async def _check_adapter(self, learner: LocalLearner, name: str) -> None:
        renderer = get_renderer(learner.model.renderer, hf_id=learner.model.hf_id)
        tokens = renderer.encode_text(_PROBE_TEXT)
        adapted, base = await learner.probe_logprobs(tokens)
        served = await self._prompt_logprobs(name, tokens)
        served_base = await self._prompt_logprobs(self.base_name, tokens)
        if (
            len(adapted) != len(served)
            or len(base) != len(served)
            or not all(isfinite(score) for score in (*adapted, *base))
        ):
            raise BackendError("learner returned invalid probe logprobs")
        if all(abs(a - b) <= 1e-6 for a, b in zip(served, served_base, strict=True)) and any(
            abs(a - b) > 1e-6 for a, b in zip(adapted, base, strict=True)
        ):
            raise BackendError(
                f"vLLM ignored adapter {name!r}: probe equals base but learner differs"
            )
        # Compare the adapter's *effect* (adapted - base) in both engines: vLLM and
        # HF kernels disagree by ~0.03 nats on short context-free prompts even
        # for the base model, and that shared mismatch cancels here.
        n = len(served)
        drift = sum(
            abs((s - sb) - (a - b))
            for s, sb, a, b in zip(served, served_base, adapted, base, strict=True)
        ) / n
        base_drift = sum(abs(sb - b) for sb, b in zip(served_base, base, strict=True)) / n
        log.info(
            "adapter %s probe: effect drift %.4f, base drift %.4f nats (n=%d)",
            name, drift, base_drift, n,
        )
        if drift > self.adapter_check_tol:
            raise BackendError(
                f"vLLM adapter {name!r} probe effect drift {drift:.6f} nats exceeds "
                f"adapter_check_tol={self.adapter_check_tol} (base drift {base_drift:.6f})"
            )

    async def _publish(
        self, learner: _SamplerMixin, name: str, *, version: int | None = None
    ) -> SamplerSnapshot:
        if name in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ConfigError("sampler name must contain only letters, digits, _.-")
        async with self._lock:
            await self._ensure_awake()
            version = learner.version + 1 if version is None else version
            server = Server.load(self.server_json)
            used = server.meta.get("used_adapter_names", [])
            directory = self.adapters_dir / name
            if name in used or name in server.models or directory.exists():
                raise ConfigError(f"sampler name {name!r} already used; names must never be reused")
            inplace = self.adapter_names == "inplace" and learner._current_adapter is not None
            served_name = learner._current_adapter if inplace else name
            assert served_name is not None
            server = await self._evict_stale(server)
            candidates = [
                adapter
                for adapter in server.adapters
                if self._owned.get(adapter["name"]) == learner.name
            ]
            excess = len(server.adapters) + (not inplace) - (server.max_loras - 1)
            if excess > len(candidates):
                raise ConfigError("max_loras has no free snapshot slot for this learner")
            # Persist the reservation before exporting or sending a load request:
            # an uncertain HTTP result can still have installed these weights.
            server = replace(server, meta={**server.meta, "used_adapter_names": [*used, name]})
            server.save()
            path = Path(await learner.save_adapter(directory)).resolve()
            self._owned[served_name] = learner.name
            body: dict[str, object] = {"lora_name": served_name, "lora_path": str(path)}
            if inplace:
                body["load_inplace"] = True
                # Once replacement starts, even a timeout can mean weights changed.
                learner._sampler_error = True
            try:
                await self._request("/load_lora_adapter", body)
            except ConfigError:
                if not inplace:
                    del self._owned[served_name]
                raise
            adapter = {
                "name": served_name,
                "path": str(path),
                "learner": learner.name,
                "owner": self._owner,
                "owner_pid": os.getpid(),
                "owner_host": socket.gethostname(),
            }
            server = replace(
                server,
                models=server.models if inplace else [*server.models, served_name],
                adapters=[a for a in server.adapters if a["name"] != served_name] + [adapter],
            )
            server.save()
            try:
                await self._check_adapter(learner, served_name)
            except (BackendError, ConfigError) as exc:
                if not inplace:
                    try:
                        await self._unload(served_name)
                    except (BackendError, ConfigError) as cleanup:
                        exc.add_note(f"failed to unload rejected adapter: {cleanup}")
                raise
            learner._current_adapter = served_name
            learner._sampler_error = False
            learner.version = version
            # The previous current snapshot becomes eligible only after loading
            # and publishing its replacement. Never evict another learner's head.
            for old in candidates[: max(0, excess)] if not inplace else []:
                await self._unload(old["name"])
            return SamplerSnapshot(
                learner.name, version, str(path), f"vllm:@{self.server_json}#{served_name}"
            )

    async def _evict_stale(self, server: Server) -> Server:
        """Unload adapters left by a dead run (e.g. killed before ``close()``).

        Their slots would otherwise count against ``max_loras`` forever, so a
        resumed run could not publish. Adapters of live owners are untouched.
        """
        stale = [
            adapter
            for adapter in server.adapters
            if adapter.get("owner") != self._owner and not _owner_alive(adapter)
        ]
        for adapter in stale:
            log.warning("unloading stale adapter %s (owner process gone)", adapter["name"])
            with suppress(ConfigError):  # vLLM no longer has it (e.g. it restarted)
                await self._request("/unload_lora_adapter", {"lora_name": adapter["name"]})
        if not stale:
            return server
        names = {adapter["name"] for adapter in stale}
        server = replace(
            server,
            models=[model for model in server.models if model not in names],
            adapters=[adapter for adapter in server.adapters if adapter["name"] not in names],
        )
        server.save()
        return server

    async def _unload(self, name: str) -> None:
        await self._request("/unload_lora_adapter", {"lora_name": name})
        server = Server.load(self.server_json)
        replace(
            server,
            models=[model for model in server.models if model != name],
            adapters=[adapter for adapter in server.adapters if adapter["name"] != name],
        ).save()
        self._owned.pop(name, None)

    async def _close_learner(self, learner: str) -> None:
        async with self._lock:
            errors = []
            for name, owner in list(self._owned.items()):
                if owner == learner:
                    try:
                        await self._unload(name)
                    except (BackendError, ConfigError) as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]

    async def close(self) -> None:
        errors: list[Exception] = []
        try:
            if self._asleep and self._training == 0:
                try:
                    await self._wake()
                except (BackendError, ConfigError) as exc:
                    errors.append(exc)
            elif self._asleep:
                log.warning("closing with a train step in flight; vLLM stays asleep")
            for learner in self.learners.values():
                try:
                    await learner.close()
                except (BackendError, ConfigError) as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            self.learners.clear()
        finally:
            try:
                if self.pool is not None and hasattr(self.pool, "close"):
                    self.pool.close()  # stops data-parallel workers
                self.pool = None
            finally:
                await self.client.aclose()
