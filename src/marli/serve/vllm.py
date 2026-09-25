"""Publish a ready token-native server before any learner can attach to it.

The manifest is also the ownership record for detached servers. Status and stop
produce separate observation handles, preserving that record and its log.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from marli.config import doc_field, input_field, runtime_field
from marli.errors import ConfigError
from marli.handles import Handle, InputRef, register_handle
from marli.model import ModelSpec, load_model
from marli.rundir import RunDir
from marli.serve._supervise import Supervisor, process_alive, stop_process_group


@dataclass
class VLLMServeConfig:
    model: str = doc_field("", help="models registry name")
    python: str = doc_field("", help="python of the separate vLLM venv, e.g. /opt/vllm/bin/python")
    port: int = 8000
    host: str = "127.0.0.1"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.85
    enable_lora: bool = True
    max_loras: int = 4
    max_lora_rank: int = 32
    tensor_parallel_size: int = 1
    enable_sleep_mode: bool = doc_field(
        False,
        help="vLLM sleep/wake endpoints so a co-located learner can use the GPUs between "
        "sampling phases (also sets VLLM_SERVER_DEV_MODE=1)",
    )
    cuda_visible_devices: str | None = None
    extra_args: list[str] = doc_field(
        default_factory=list, help="escape hatch: argv entries appended verbatim to vLLM"
    )
    learner_ranks: list[int] = doc_field(
        default_factory=list, help="optional planned learner ranks; reserve one snapshot slot"
    )
    ready_timeout_s: float = runtime_field(900.0)
    detach: bool = runtime_field(False, help="return after ready; serve stop owns later cleanup")

    def __post_init__(self) -> None:
        if not self.model or not self.python or not self.host:
            raise ConfigError("serve vllm requires model, python and host")
        for name in ("port", "max_loras", "max_lora_rank", "tensor_parallel_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.enable_sleep_mode) is not bool:
            raise ConfigError("enable_sleep_mode must be a boolean")
        if self.port > 65535:
            raise ConfigError("port must be <= 65535")
        if self.max_model_len is not None and (
            type(self.max_model_len) is not int or self.max_model_len <= 0
        ):
            raise ConfigError("max_model_len must be a positive integer")
        if not 0 < self.gpu_memory_utilization < 1 or not 0 < self.ready_timeout_s < float("inf"):
            raise ConfigError("GPU utilization must be in (0, 1); ready timeout must be finite > 0")
        if any(type(rank) is not int or rank <= 0 for rank in self.learner_ranks):
            raise ConfigError("learner_ranks must contain positive integers")
        if self.learner_ranks and (
            not self.enable_lora or max(self.learner_ranks) > self.max_lora_rank
        ):
            raise ConfigError("enable_lora and max_lora_rank must support every learner rank")
        if self.enable_lora and self.max_loras < len(self.learner_ranks) + 1:
            raise ConfigError("max_loras must reserve space for learners plus a snapshot")


# vLLM 0.30.x api_server / EngineArgs spellings. vLLM is pod-only; keep the
# version-sensitive flags in one table and pin the complete argv in CPU tests.
VLLM_FLAGS = {
    "model": "--model",
    "served_model_name": "--served-model-name",
    "generation_config": "--generation-config",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "enable_lora": "--enable-lora",
    "max_loras": "--max-loras",
    "max_lora_rank": "--max-lora-rank",
    "tensor_parallel_size": "--tensor-parallel-size",
    "enable_sleep_mode": "--enable-sleep-mode",
    "seed": "--seed",
    "port": "--port",
    "host": "--host",
}


# Flags that take no value.
_SWITCHES = frozenset({"enable_lora", "enable_sleep_mode"})


def launch_args(cfg: VLLMServeConfig) -> tuple[ModelSpec, list[str], dict[str, str]]:
    """Validate registry-dependent settings before starting any GPU process."""
    cfg.__post_init__()
    model = load_model(cfg.model)
    length = cfg.max_model_len if cfg.max_model_len is not None else model.max_ctx
    if length > model.max_ctx:
        raise ConfigError(f"max_model_len {length} exceeds model.max_ctx {model.max_ctx}")
    values: dict[str, str | int | float | bool] = {
        "model": model.hf_id,
        "served_model_name": model.name,
        "generation_config": "vllm",
        "max_model_len": length,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enable_lora": cfg.enable_lora,
        "max_loras": cfg.max_loras,
        "max_lora_rank": cfg.max_lora_rank,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "enable_sleep_mode": cfg.enable_sleep_mode,
        "seed": 0,
        "port": cfg.port,
        "host": cfg.host,
    }
    argv = [cfg.python, "-m", "vllm.entrypoints.openai.api_server"]
    for key, flag in VLLM_FLAGS.items():
        if not cfg.enable_lora and key in {"enable_lora", "max_loras", "max_lora_rank"}:
            continue
        if key == "enable_sleep_mode" and not cfg.enable_sleep_mode:
            continue
        argv.append(flag)
        if key not in _SWITCHES:
            argv.append(str(values[key]))
    argv.extend(cfg.extra_args)
    env = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    if cfg.enable_sleep_mode:
        # vLLM 0.30 registers /sleep, /wake_up and /is_sleeping only in dev mode.
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return model, argv, env


@register_handle
@dataclass(frozen=True, kw_only=True)
class Server(Handle):
    KIND: ClassVar[str] = "server"
    MANIFEST: ClassVar[str] = "server.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("log",)

    base_url: str
    models: list[str]
    pid: int
    log: str
    adapters: list[dict[str, str]] = field(default_factory=list)
    hf_id: str = ""
    enable_lora: bool = True
    max_loras: int = 4
    max_lora_rank: int = 32
    tensor_parallel_size: int = 1
    enable_sleep_mode: bool = False

    def summary(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "models": self.models, "pid": self.pid}


def venv_path(python: str, path: str | None = None) -> str:
    """PATH for the server with the vLLM venv's ``bin`` first, as activation would.

    vLLM JIT-builds kernels at startup (flashinfer's sampler runs ``ninja``) and
    finds the build tools on PATH; they live beside the venv's interpreter.
    """
    path = os.environ.get("PATH", "") if path is None else path
    bin_dir = os.path.dirname(os.path.abspath(python))
    return os.pathsep.join([bin_dir, *(p for p in path.split(os.pathsep) if p and p != bin_dir)])


async def vllm(cfg: VLLMServeConfig, run: RunDir) -> Server:
    model, argv, env = launch_args(cfg)
    env = {**env, "PATH": venv_path(cfg.python)}
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(cfg.host, cfg.host)
    base_url = f"http://{'[' + host + ']' if ':' in host else host}:{cfg.port}"
    supervisor = Supervisor(
        argv,
        env=env,
        log=run.path("server.log"),
        health_url=base_url + "/v1/models",
        ready_timeout_s=cfg.ready_timeout_s,
        detach=cfg.detach,
        exit_on_signal=False,
    )
    await supervisor.start()
    try:
        server = Server(
            root=run.out,
            config_hash=run.config_hash,
            base_url=base_url,
            models=[model.name],
            pid=supervisor.pid,
            log="server.log",
            hf_id=model.hf_id,
            enable_lora=cfg.enable_lora,
            max_loras=cfg.max_loras,
            max_lora_rank=cfg.max_lora_rank,
            tensor_parallel_size=cfg.tensor_parallel_size,
            enable_sleep_mode=cfg.enable_sleep_mode,
        )
        # Foreground consumers need the ready manifest while this verb is alive.
        # Detached ownership is released only after that record is durable.
        server.save()
        if cfg.detach:
            supervisor.release()
        else:
            await supervisor.wait()
        return Server.load(server.manifest_path)
    finally:
        supervisor.stop()


@dataclass
class ServerControlConfig:
    server_json: str = input_field("", help="server.json written by serve vllm")
    timeout_s: float = runtime_field(5.0, help="HTTP timeout and stop grace period")

    def __post_init__(self) -> None:
        if not self.server_json or not 0 < self.timeout_s < float("inf"):
            raise ConfigError("server_json and a positive finite timeout_s are required")


@register_handle
@dataclass(frozen=True, kw_only=True)
class ServerStatus(Handle):
    KIND: ClassVar[str] = "server_status"
    MANIFEST: ClassVar[str] = "server-status.json"

    pid: int
    running: bool
    healthy: bool

    def summary(self) -> dict[str, Any]:
        return {"pid": self.pid, "running": self.running, "healthy": self.healthy}


async def status(cfg: ServerControlConfig, run: RunDir) -> ServerStatus:
    server = Server.load(cfg.server_json)
    running = process_alive(server.pid)
    healthy = False
    if running:
        async with httpx.AsyncClient(trust_env=False, timeout=cfg.timeout_s) as client:
            try:
                response = await client.get(server.base_url.rstrip("/") + "/v1/models")
                healthy = response.is_success
            except httpx.TransportError:
                pass
    return ServerStatus(
        root=run.out,
        inputs=(InputRef.of(server),),
        pid=server.pid,
        running=running,
        healthy=healthy,
    )


async def stop(cfg: ServerControlConfig, run: RunDir) -> ServerStatus:
    server = Server.load(cfg.server_json)
    await asyncio.to_thread(stop_process_group, server.pid, grace_s=cfg.timeout_s)
    return ServerStatus(
        root=run.out,
        inputs=(InputRef.of(server),),
        pid=server.pid,
        running=process_alive(server.pid),
        healthy=False,
    )
