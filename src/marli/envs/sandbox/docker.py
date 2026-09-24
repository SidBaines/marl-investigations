"""Keep episode files and processes inside a disposable Docker container.

The workspace is a writable tmpfs (charged to the container's memory limit),
retained between commands and removed with the container. Images must provide
bash, GNU timeout, realpath, cat, find, and sleep. Docker's init reaps orphaned children;
GNU timeout kills the command's process group, and cleanup after each call
kills all remaining processes except init and the initial sleep keeper.
File operations hold the command lock after this cleanup, so no model process
can replace a checked path with a symlink during exec-based file I/O. Exec
writes as the container user into tmpfs even with a read-only root filesystem.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import math
import os
import re
import signal
import subprocess
import time
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from marli.envs.sandbox.base import ExecResult
from marli.envs.sandbox.subprocess import _OutputBuffer
from marli.errors import BackendError, ConfigError

# Register before `run`: even a lost CLI response must leave a cleanup target.
_CONTAINERS: set[tuple[int, str, str]] = set()
_CONTROL_TIMEOUT = 30.0

# This executes only inside the private PID namespace. Re-scan until every
# non-keeper process has stopped; a single kill pass can race a final fork.
# Read the whole stat file: process names may contain newlines and parentheses.
_STOP_COMMANDS = """# marli-stop
keeper=$1
while :; do
    live=no
    for path in /proc/[0-9]*/stat; do
        pid=${path#/proc/}; pid=${pid%/stat}
        case "$pid" in 1|"$keeper"|"$$") continue ;; esac
        state=$(<"$path") 2>/dev/null || continue
        state=${state##*) }; state=${state%% *}
        case "$state" in Z|X) continue ;; esac
        kill -KILL "$pid" 2>/dev/null && live=yes
    done
    [ "$live" = yes ] || break
done
if [ -n "$2" ] && [ -f "$2" ]; then cat -- "$2"; rm -f -- "$2"; fi
"""


def _sweep_containers() -> None:
    for owner, docker, label in tuple(_CONTAINERS):
        if owner != os.getpid():
            continue
        try:
            found = subprocess.run(
                [docker, "ps", "-aq", "--filter", f"label=marli.sandbox={label}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            ids = found.stdout.split()
            if ids:
                subprocess.run(
                    [docker, "rm", "-f", *ids],
                    capture_output=True,
                    timeout=10,
                    check=True,
                )
            _CONTAINERS.discard((owner, docker, label))
        except (OSError, subprocess.SubprocessError) as exc:
            warnings.warn(
                f"Docker sandbox exit cleanup failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


atexit.register(_sweep_containers)


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Resource and isolation settings, with no host filesystem mounts.

    extra_run_args accepts --platform, --hostname and --label (separate or
    equals-form values). Other flags are rejected: an escape hatch must not
    override isolation, the entrypoint, resource limits, or the workspace.
    """

    image: str = "python:3.12-slim"
    cpus: float = 1.0
    memory_mb: int = 2048
    pids: int = 256
    workdir: str = "/workspace"
    network: str = "none"
    allow_network: bool = False
    user: str = "65534:65534"
    read_only_root: bool = True
    max_output_bytes: int = 1 << 20
    docker: str = "docker"
    extra_run_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ConfigError("cpus must be positive and finite")
        for name in ("memory_mb", "pids", "max_output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        for name in ("image", "docker", "user", "network", "workdir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\0" in value or value.startswith("-"):
                raise ConfigError(f"invalid {name}")
        for name in ("allow_network", "read_only_root"):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"{name} must be a boolean")
        user = self.user.partition(":")[0]
        if user == "root" or (user.isdecimal() and int(user) == 0):
            raise ConfigError("sandbox user must not be root")
        if self.network == "host" or self.network.startswith(("container:", "ns:")):
            raise ConfigError("host and shared-namespace networking are forbidden")
        if self.network != "none" and not self.allow_network:
            raise ConfigError("network requires allow_network=True")
        path = PurePosixPath(self.workdir)
        if (
            not path.is_absolute()
            or self.workdir != str(path)
            or ".." in path.parts
            or self.workdir in ("/", "/tmp")
            or any(char in self.workdir for char in ":,\n")
        ):
            raise ConfigError(
                "workdir must be a normalized absolute directory other than / or /tmp"
            )
        args = iter(self.extra_run_args)
        for arg in args:
            if not isinstance(arg, str):
                raise ConfigError("extra_run_args must contain strings")
            key, separator, value = arg.partition("=")
            if key not in ("--platform", "--hostname", "--label"):
                raise ConfigError(f"unsafe or unsupported extra_run_args flag: {key}")
            if not separator:
                value = next(args, "")
            if not isinstance(value, str) or not value or value.startswith("-") or "\0" in value:
                raise ConfigError(f"missing or invalid value for {key}")
            if key == "--label" and value.partition("=")[0] == "marli.sandbox":
                raise ConfigError("marli.sandbox label is reserved for cleanup")


async def _capture(
    argv: Sequence[str],
    *,
    timeout_s: float,
    cap: int,
    stdin: str | None = None,
    overflow_limit: bool = False,
    strict_stdout: bool = False,
) -> ExecResult:
    """Bound memory while draining both pipes, including when the client hangs."""
    started = time.monotonic()
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            start_new_session=True,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    )
    cancelled = False
    try:
        process = await asyncio.shield(launch)
    except asyncio.CancelledError:
        process = await launch
        cancelled = True
    stdout, stderr = _OutputBuffer(cap), _OutputBuffer(cap)
    overflow = asyncio.Event()

    async def drain(stream: asyncio.StreamReader | None, buffer: _OutputBuffer) -> None:
        assert stream is not None
        while data := await stream.read(65536):
            buffer.append(data)
            if overflow_limit and buffer.total > 4 * cap:
                overflow.set()

    async def feed() -> None:
        if process.stdin is not None:
            try:
                assert stdin is not None
                data = stdin.encode()
                for offset in range(0, len(data), 65536):
                    process.stdin.write(data[offset : offset + 65536])
                    await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

    async def collect() -> None:
        await asyncio.gather(drain(process.stdout, stdout), drain(process.stderr, stderr), feed())
        await process.wait()

    output = asyncio.create_task(collect())
    excess = asyncio.create_task(overflow.wait())
    timed_out = False
    try:
        if cancelled:
            raise asyncio.CancelledError
        done, _ = await asyncio.wait(
            (output, excess),
            timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        timed_out = not done
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        await output
        excess.cancel()
        await asyncio.gather(excess, return_exceptions=True)
    if overflow.is_set():
        diagnostic = _OutputBuffer(cap)
        diagnostic.append(f"sandbox max_output_bytes exceeded ({cap} per stream)\n".encode())
        diagnostic.append(stderr.head + stderr.tail)
        stderr_text = diagnostic.text()
    else:
        stderr_text = stderr.text()
    return ExecResult(
        None
        if timed_out
        else (process.returncode or 1)
        if overflow.is_set()
        else process.returncode,
        (stdout.head + stdout.tail).decode(
            errors="strict" if strict_stdout and stdout.total <= cap else "replace"
        ),
        stderr_text,
        timed_out,
        time.monotonic() - started,
        stdout.total > cap or stderr.total > cap,
    )


class DockerSandbox:
    def __init__(self, config: DockerSandboxConfig | None = None) -> None:
        self.config = config or DockerSandboxConfig()
        self.config.__post_init__()
        self.workdir = Path(self.config.workdir)
        self.container_id: str | None = None
        self._label = uuid.uuid4().hex
        self._keeper: str | None = None
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._closed = False

    def _require_started(self) -> str:
        if self.container_id is None or self._closed:
            raise RuntimeError("sandbox is not running")
        return self.container_id

    async def _cli(self, *args: str) -> str:
        result = await _capture(
            [self.config.docker, *args],
            timeout_s=_CONTROL_TIMEOUT,
            cap=65536,
        )
        if result.timed_out or result.exit_code != 0:
            raise BackendError(f"docker {args[0]} failed: {result.stderr or result.stdout}")
        return result.stdout.strip()

    async def start(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("sandbox is closed")
            if self.container_id is not None:
                return
            cfg = self.config
            cfg.__post_init__()
            argv = [
                "run",
                "-d",
                "--rm",
                "--init",
                "--entrypoint",
                "",
                "--network",
                cfg.network,
                "--cpus",
                str(cfg.cpus),
                "--memory",
                f"{cfg.memory_mb}m",
                "--memory-swap",
                f"{cfg.memory_mb}m",
                "--pids-limit",
                str(cfg.pids),
                "--user",
                cfg.user,
            ]
            if cfg.read_only_root:
                argv.append("--read-only")
            argv += [
                "--tmpfs",
                "/tmp:rw,size=256m",
                "--tmpfs",
                f"{cfg.workdir}:rw,exec,mode=1777,size={cfg.memory_mb}m",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--workdir",
                cfg.workdir,
                "--label",
                f"marli.sandbox={self._label}",
                *cfg.extra_run_args,
                cfg.image,
                "sleep",
                "infinity",
            ]
            _CONTAINERS.add((os.getpid(), cfg.docker, self._label))
            try:
                launch = asyncio.create_task(self._cli(*argv))
                try:
                    self.container_id = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    self.container_id = await launch
                    raise
                if not re.fullmatch(r"[a-f0-9]{12,64}", self.container_id):
                    self.container_id = None
                    raise BackendError("docker run returned an invalid container id")
                # Fail at startup if a custom image cannot implement the contract.
                self._keeper = await self._cli(
                    "exec",
                    self.container_id,
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    "command -v timeout >/dev/null && command -v find >/dev/null && "
                    "command -v realpath >/dev/null && command -v cat >/dev/null && "
                    'read -r keeper _ < /proc/1/task/1/children; printf "%s" "$keeper"',
                    "marli-start",
                )
                if not self._keeper.isdecimal() or int(self._keeper) <= 1:
                    raise BackendError("cannot identify Docker init's keeper process")
            except BaseException:
                await self._remove()
                raise

    async def _stop_commands(self, completion: str | None = None) -> str:
        async with self._cleanup_lock:
            if self.container_id is not None and self._keeper is not None:
                return await self._cli(
                    "exec",
                    self.container_id,
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    _STOP_COMMANDS,
                    "marli-stop",
                    self._keeper,
                    completion or "",
                )
            return ""

    async def exec(
        self,
        cmd: Sequence[str] | str,
        *,
        timeout_s: float,
        stdin: str | None = None,
    ) -> ExecResult:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        argv = ["bash", "--noprofile", "--norc", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        if not argv or not all(isinstance(arg, str) and "\0" not in arg for arg in argv):
            raise ValueError("command must contain string arguments")
        async with self._lock:
            container = self._require_started()
            started = time.monotonic()
            completion = f"/tmp/marli-{uuid.uuid4().hex}"
            try:
                result = await _capture(
                    [
                        self.config.docker,
                        "exec",
                        *(["-i"] if stdin is not None else []),
                        container,
                        "timeout",
                        "--signal=KILL",
                        f"{timeout_s}s",
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        'marker=$1; shift; "$@"; status=$?; '
                        'printf "%s" "$status" > "$marker"; exit "$status"',
                        "marli-command",
                        completion,
                        *argv,
                    ],
                    # Allow the in-container watchdog to fire even with CLI startup latency.
                    timeout_s=timeout_s + 1,
                    cap=self.config.max_output_bytes,
                    stdin=stdin,
                    overflow_limit=True,
                )
            finally:
                try:
                    completed = await self._stop_commands(completion)
                except BaseException:
                    self._closed = True
                    await self._remove()
                    raise
            # A completion record distinguishes a command exiting 137 (including
            # an OOM victim) from GNU timeout killing its entire process group.
            timed_out = result.timed_out or (result.exit_code == 137 and not completed)
            return ExecResult(
                None if timed_out else result.exit_code,
                result.stdout,
                result.stderr,
                timed_out,
                time.monotonic() - started,
                result.truncated,
            )

    async def _file_path(self, rel: str, *, create: bool = False) -> str:
        container = self._require_started()
        path = PurePosixPath(rel)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\0" in rel:
            raise ValueError("expected a relative file path within the sandbox")
        # No model processes survive exec; hold _lock across check + file I/O.
        script = """
root=$1; create=$2; shift 2
[ ! -L "$root" ] || exit 42
resolved_root=$(realpath -e -- "$root") || exit 42
[ "$resolved_root" = "$root" ] || exit 42
cd -- "$root" || exit 42
while [ "$#" -gt 1 ]; do
    [ ! -L "$1" ] || exit 42
    if [ "$create" = yes ]; then mkdir -p -- "$1" || exit 42; fi
    cd -- "$1" || exit 42
    shift
done
[ ! -L "$1" ] || exit 42
if [ -e "$1" ]; then [ -f "$1" ] || exit 42; else [ "$create" = yes ] || exit 44; fi
resolved=$(realpath -m -- "$1" && printf /) || exit 42
resolved=${resolved%$'\n/'}
case "$resolved" in "$root"/*) printf '%s' "$resolved" ;; *) exit 42 ;; esac
"""
        result = await _capture(
            [
                self.config.docker,
                "exec",
                container,
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                script,
                "marli-file",
                self.config.workdir,
                "yes" if create else "no",
                *path.parts,
            ],
            timeout_s=_CONTROL_TIMEOUT,
            cap=65536,
        )
        if result.exit_code == 42:
            raise ValueError("sandbox file paths must not traverse symlinks or non-files")
        if result.exit_code == 44:
            raise FileNotFoundError(rel)
        if result.timed_out or result.exit_code != 0:
            raise BackendError(f"Docker file path check failed: {result.stderr}")
        resolved = result.stdout
        if result.truncated or not resolved.startswith(self.config.workdir + "/"):
            raise ValueError("sandbox file path escapes the workspace")
        return resolved

    async def read_file(self, rel: str) -> str:
        async with self._lock:
            path = await self._file_path(rel)
            result = await _capture(
                [self.config.docker, "exec", self._require_started(), "cat", "--", path],
                timeout_s=_CONTROL_TIMEOUT,
                cap=self.config.max_output_bytes,
                overflow_limit=True,
                strict_stdout=True,
            )
            if result.truncated:
                raise ValueError("sandbox file exceeds max_output_bytes")
            if result.timed_out or result.exit_code != 0:
                raise BackendError(f"Docker file read failed: {result.stderr}")
            return result.stdout

    async def write_file(self, rel: str, content: str) -> None:
        async with self._lock:
            path = await self._file_path(rel, create=True)
            result = await _capture(
                [
                    self.config.docker,
                    "exec",
                    "-i",
                    self._require_started(),
                    "sh",
                    "-c",
                    'umask 077; cat > "$1"',
                    "sh",
                    path,
                ],
                timeout_s=_CONTROL_TIMEOUT,
                cap=65536,
                stdin=content,
            )
            if result.timed_out or result.exit_code != 0:
                raise BackendError(f"Docker file write failed: {result.stderr}")

    async def reset(self) -> None:
        self._require_started()
        await self._stop_commands()
        async with self._lock:
            container = self._require_started()
            await self._cli(
                "exec", container, "find", self.config.workdir, "-mindepth", "1", "-delete"
            )

    async def _remove(self) -> None:
        ids = (
            await self._cli(
                "ps",
                "-aq",
                "--filter",
                f"label=marli.sandbox={self._label}",
            )
        ).split()
        if ids:
            await self._cli("rm", "-f", *ids)
        self.container_id = None
        _CONTAINERS.discard((os.getpid(), self.config.docker, self._label))

    async def close(self) -> None:
        self._closed = True
        try:
            if self.container_id is not None:
                # Forced removal remains authoritative if the container has
                # already exited or can no longer admit a cleanup process.
                with contextlib.suppress(BackendError):
                    await self._stop_commands()
        finally:
            async with self._lock:
                if self.container_id is not None:
                    await self._remove()


class DockerPool:
    """A bounded pool; call start before acquire and close to retire all leases.

    A failed reset retires that container. The next acquire replaces it, keeping
    the configured capacity without retrying a broken container indefinitely.
    """

    def __init__(self, config: DockerSandboxConfig, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ConfigError("pool size must be a positive integer")
        self.config, self.size = config, size
        self._available: list[DockerSandbox] = []
        self._leased: set[DockerSandbox] = set()
        self._condition = asyncio.Condition()
        self._started = False
        self._closed = False

    async def start(self) -> None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("pool is closed")
            if self._started:
                return
            try:
                for _ in range(self.size):
                    sandbox = DockerSandbox(self.config)
                    await sandbox.start()
                    self._available.append(sandbox)
            except BaseException:
                for sandbox in self._available:
                    await sandbox.close()
                self._available.clear()
                raise
            self._started = True

    async def acquire(self) -> DockerSandbox:
        await self.start()
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or bool(self._available) or len(self._leased) < self.size,
            )
            if self._closed:
                raise RuntimeError("pool is closed")
            sandbox = self._available.pop() if self._available else DockerSandbox(self.config)
            try:
                await sandbox.start()
                await sandbox.reset()
            except BaseException:
                await sandbox.close()
                self._condition.notify_all()
                raise
            self._leased.add(sandbox)
            return sandbox

    async def release(self, sandbox: DockerSandbox) -> None:
        async with self._condition:
            if sandbox not in self._leased:
                raise ValueError("sandbox is not leased from this pool")
            try:
                await sandbox.reset()
            except BaseException:
                await sandbox.close()
                raise
            else:
                self._available.append(sandbox)
            finally:
                self._leased.remove(sandbox)
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            try:
                results = await asyncio.gather(
                    *(sandbox.close() for sandbox in [*self._available, *self._leased]),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            finally:
                self._available.clear()
                self._leased.clear()
                self._condition.notify_all()
