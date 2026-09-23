"""Publish immutable adapters so vLLM never reuses a cache key for new weights.

One backend owns one shared training base. Publication is serialized across its
learners: load the new snapshot, switch the current policy, then evict an old
snapshot. Exported directories and used names survive unloading and shutdown.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx

from marli.budget import SpendGuard
from marli.errors import BackendError, ConfigError
from marli.model import ModelSpec
from marli.policy.vllm import VLLMPolicy
from marli.serve.vllm import Server
from marli.train.backends.base import SamplerSnapshot
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec

if TYPE_CHECKING:
    from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool


class _SamplerMixin:
    """Add serving to the lazy learner subclass without wrapping its optimizer."""

    name: str
    spec: LearnerSpec
    model: ModelSpec
    version: int
    _backend: LocalBackend
    _current_adapter: str | None = None

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        return await self._backend._publish(self, name)

    def policy(self, *, policy_id: str | None = None) -> VLLMPolicy:
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


class LocalBackend:
    name = "local"

    def __init__(
        self,
        server_json: str,
        *,
        adapters_dir: str,
        device: str = "cuda",
        spend: SpendGuard | None = None,
    ) -> None:
        self.server_json = Path(server_json).resolve()
        server = Server.load(self.server_json)
        if not server.models or not server.enable_lora or server.max_loras < 2:
            raise ConfigError("local training requires a LoRA server with max_loras >= 2")
        self.base_url = server.base_url.rstrip("/")
        self.base_name = server.models[0]
        self.adapters_dir = Path(adapters_dir).resolve()
        self.device = device
        self.spend = spend
        self.pool: LocalLearnerPool | None = None
        self.learners: dict[str, LocalLearner] = {}
        self.client = httpx.AsyncClient(timeout=600, trust_env=False)
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
        if len(self.learners) + 2 > server.max_loras:
            raise ConfigError("max_loras must reserve one slot beyond the learner count")
        state = (
            Checkpoint.load(spec.init_from).require_state(name)
            if spec.init_from is not None
            else None
        )

        # M4-1 owns torch/PEFT construction, training and checkpoint I/O. Its
        # constructors are synchronous; no heavy import occurs before validation.
        import torch

        from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool

        class ServingLearner(_SamplerMixin, LocalLearner):
            pass

        torch.manual_seed(seed)
        if self.pool is None:
            self.pool = LocalLearnerPool(model, device=self.device)
        learner = ServingLearner(name, spec, self.pool)
        learner._backend = self
        self.learners[name] = learner
        if state is not None:
            await learner.load_state(state)
            await self._publish(learner, f"{self._owner}-{name}-init", version=0)
        return learner

    async def _request(self, endpoint: str, body: dict[str, str]) -> None:
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

    async def _publish(
        self, learner: _SamplerMixin, name: str, *, version: int | None = None
    ) -> SamplerSnapshot:
        if name in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ConfigError("sampler name must contain only letters, digits, _.-")
        async with self._lock:
            version = learner.version + 1 if version is None else version
            server = Server.load(self.server_json)
            used = server.meta.get("used_adapter_names", [])
            directory = self.adapters_dir / name
            if name in used or name in server.models or directory.exists():
                raise ConfigError(f"sampler name {name!r} already used; names must never be reused")
            candidates = [
                adapter
                for adapter in server.adapters
                if self._owned.get(adapter["name"]) == learner.name
            ]
            excess = len(server.adapters) + 1 - (server.max_loras - 1)
            if excess > len(candidates):
                raise ConfigError("max_loras has no free snapshot slot for this learner")
            # Persist the reservation before exporting or sending a load request:
            # an uncertain HTTP result can still have installed these weights.
            server = replace(server, meta={**server.meta, "used_adapter_names": [*used, name]})
            server.save()
            path = Path(await learner.save_adapter(directory)).resolve()
            self._owned[name] = learner.name
            try:
                await self._request(
                    "/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}
                )
            except ConfigError:
                del self._owned[name]
                raise
            adapter = {
                "name": name,
                "path": str(path),
                "learner": learner.name,
                "owner": self._owner,
            }
            server = replace(
                server, models=[*server.models, name], adapters=[*server.adapters, adapter]
            )
            server.save()
            learner._current_adapter = name
            learner.version = version
            # The previous current snapshot becomes eligible only after loading
            # and publishing its replacement. Never evict another learner's head.
            for old in candidates[: max(0, excess)]:
                await self._unload(old["name"])
            return SamplerSnapshot(
                learner.name, version, str(path), f"vllm:@{self.server_json}#{name}"
            )

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
        errors = []
        try:
            for learner in self.learners.values():
                try:
                    await learner.close()
                except (BackendError, ConfigError) as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            self.learners.clear()
            self.pool = None
        finally:
            await self.client.aclose()
