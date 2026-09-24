"""Episode sandboxes separate model-written programs from the training process."""

from __future__ import annotations

from typing import Any

from marli.envs.sandbox.base import ExecResult, Sandbox
from marli.envs.sandbox.docker import DockerPool, DockerSandbox, DockerSandboxConfig
from marli.envs.sandbox.subprocess import SubprocessSandbox
from marli.errors import ConfigError

__all__ = [
    "DockerPool",
    "DockerSandbox",
    "DockerSandboxConfig",
    "ExecResult",
    "Sandbox",
    "SubprocessSandbox",
    "make_sandbox",
]


async def make_sandbox(kind: str, **cfg: Any) -> Sandbox:
    """Create and start a sandbox; pooled leases must be returned via pool.release."""
    pool = cfg.pop("pool", None)
    if kind == "docker" and pool is not None:
        if not isinstance(pool, DockerPool) or cfg:
            raise ConfigError("pool must be a DockerPool and supplies its own config")
        return await pool.acquire()
    if pool is not None:
        raise ConfigError("pool is only supported for Docker sandboxes")
    try:
        if kind == "docker":
            sandbox: Sandbox = DockerSandbox(DockerSandboxConfig(**cfg))
        elif kind == "subprocess":
            sandbox = SubprocessSandbox(**cfg)
        else:
            raise ConfigError(f"unknown sandbox kind: {kind}")
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
    await sandbox.start()
    return sandbox
