"""``marli`` sandbox for Inspect: one hardened subprocess sandbox per sample, no Docker.

RunPod pods have no Docker, and Inspect's ``local`` sandbox runs model commands as the
eval user with the eval's environment, network and whole filesystem. This provider wraps
marli's :class:`~marli.envs.sandbox.subprocess.SubprocessSandbox` (the sandbox our coding
envs train in on pods):

- a fresh work directory per sample, owned by a dedicated unprivileged uid;
- Landlock confines file access to that directory (system files read-only);
- seccomp denies network sockets, ptrace and mount; rlimits bound memory, processes,
  file size and CPU; a scrubbed environment (no API keys or tokens);
- every process of that uid is killed and the directory removed at sample cleanup.

Inspect's bash tool asks for a login shell; here it gets a plain one (``--noprofile``),
because the sandbox cannot read the host's login profiles.

It needs root (for the uid pool) and Landlock ABI >= 4, which pods and the dev box have;
elsewhere use ``sandbox="local"`` with trusted policies only. Importing this module
registers the provider with Inspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, overload

from inspect_ai.util import (
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxUserUnsupportedError,
    sandboxenv,
)

from marli.envs.sandbox.subprocess import SubprocessSandbox

DEFAULT_TIMEOUT_S = 300


@sandboxenv(name="marli")
class MarliSandboxEnvironment(SandboxEnvironment):
    def __init__(self, sandbox: SubprocessSandbox) -> None:
        super().__init__()
        self.sandbox = sandbox

    @classmethod
    def default_concurrency(cls) -> int | None:
        return 64

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        sandbox = SubprocessSandbox()
        await sandbox.start()
        return {"default": cls(sandbox)}

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            await environment.as_type(cls).sandbox.close()

    def _relative(self, file: str) -> str:
        path = Path(file)
        if path.is_absolute():
            path = path.relative_to(self.sandbox.workdir)  # ValueError outside the workdir
        return path.as_posix()

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if user is not None:
            raise SandboxUserUnsupportedError("the marli sandbox runs as its own uid only")
        argv = list(cmd)
        if argv[:2] == ["bash", "--login"]:
            # Inspect's bash tool asks for a login shell; Landlock keeps /etc/profile and the
            # host's dotfiles unreadable, so every output would start with a permission error.
            argv = ["bash", "--noprofile", "--norc", *argv[2:]]
        if env:  # applied by absolute path, never through the sandbox PATH
            argv = ["/usr/bin/env", *(f"{key}={value}" for key, value in env.items()), *argv]
        if cwd is not None:
            argv = [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'cd "$1" && shift && exec "$@"',
                "_",
                cwd,
                *argv,
            ]
        stdin = input.decode() if isinstance(input, bytes) else input
        result = await self.sandbox.exec(
            argv, timeout_s=float(timeout or DEFAULT_TIMEOUT_S), stdin=stdin
        )
        if result.timed_out:
            raise TimeoutError(f"command timed out after {timeout or DEFAULT_TIMEOUT_S} s")
        code = result.exit_code if result.exit_code is not None else 1
        return ExecResult(code == 0, code, result.stdout, result.stderr)

    async def write_file(self, file: str, contents: str | bytes) -> None:
        text = contents.decode() if isinstance(contents, bytes) else contents
        await self.sandbox.write_file(self._relative(file), text)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> Any:
        try:
            content = await self.sandbox.read_file(self._relative(file))
        except FileNotFoundError:
            raise
        except OSError as exc:  # e.g. a directory
            raise FileNotFoundError(f"{file}: {exc}") from exc
        return content if text else content.encode()
