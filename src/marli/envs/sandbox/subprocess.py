"""Restrict model-written code on hosts without containers or user namespaces.

Landlock, dedicated identities, process-group cleanup and rlimits protect the
host/training process against filesystem writes outside the workdir, TCP,
fork bombs, memory blowups and runaway CPU. Limits apply per process, not as
an aggregate cgroup budget. ABI 4 does NOT isolate UDP or Unix sockets, nor
provide a /proc namespace hiding other processes. Hidden tests must never be
on disk outside the grader's own fresh directory.

The default requires Landlock ABI >= 4. Explicitly disabling that requirement
warns and applies whatever filesystem protection the kernel supports; TCP
is unconfined below ABI 4. Non-root execution warns because it cannot allocate
an isolated identity, enforce a dedicated NPROC budget, or sweep detached
processes by uid. It is suitable only for trusted code in that mode.

The parent becomes a Linux child subreaper so killed grandchildren can be
reaped even on pods whose PID 1 does not reap. Only sandbox-owned children
are waited on. The launcher uses isolated Python startup so files placed in
the workdir cannot execute as root through module or sitecustomize shadowing.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import errno
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from marli.envs.sandbox import landlock
from marli.envs.sandbox.base import ExecResult
from marli.errors import ConfigError

_UID_LOCK = threading.Lock()
_UIDS_IN_USE: set[int] = set()


@dataclass(frozen=True)
class ResourceLimits:
    address_space_bytes: int = 4 * 1024**3
    cpu_grace_s: int = 5
    file_size_bytes: int = 256 * 1024**2
    open_files: int = 256
    processes: int = 256

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")


def _uid_pids(uid: int) -> set[int]:
    pids = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # Non-dumpable processes have root-owned /proc entries even after
            # setuid; the status record retains their actual identity.
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("Uid:") and int(line.split()[1]) == uid:
                    pids.add(int(entry.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
    return pids


def _allocate_uid(uid_range: tuple[int, int]) -> int:
    with _UID_LOCK:
        for uid in range(uid_range[0], uid_range[1] + 1):
            if uid not in _UIDS_IN_USE and not _uid_pids(uid):
                _UIDS_IN_USE.add(uid)
                return uid
    raise ConfigError(f"sandbox uid pool exhausted ({uid_range[0]}–{uid_range[1]})")


class SubprocessSandbox:
    def __init__(
        self,
        *,
        require_landlock: bool = True,
        limits: ResourceLimits | None = None,
        uid_range: tuple[int, int] = (61000, 61999),
    ) -> None:
        if not (0 < uid_range[0] <= uid_range[1] < 2**32 - 1):
            raise ConfigError("sandbox uid range must contain positive unprivileged identities")
        self.require_landlock = require_landlock
        self.limits = limits or ResourceLimits()
        self.uid_range = uid_range
        self.workdir: Path
        self.uid: int | None = None
        self._abi = 0
        self._started = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._processes: set[asyncio.subprocess.Process] = set()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        if self._started:
            return
        self._abi = landlock.abi_version()
        if self._abi < 4:
            message = f"SubprocessSandbox requires Landlock ABI >= 4; host supports {self._abi}"
            if self.require_landlock:
                raise ConfigError(message)
            warnings.warn(
                f"{message}; degraded mode: TCP is unrestricted and filesystem protection "
                "may be incomplete or absent",
                RuntimeWarning,
                stacklevel=2,
            )
        if os.geteuid() != 0:
            warnings.warn(
                "sandbox runs without a dedicated uid: NPROC and detached-process cleanup "
                "are unavailable; use only trusted code",
                RuntimeWarning,
                stacklevel=2,
            )
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise ConfigError("cannot enable sandbox child reaping")
        if os.geteuid() == 0:
            self.uid = _allocate_uid(self.uid_range)
        try:
            root = os.environ.get("MARLI_SANDBOX_ROOT")
            self.workdir = Path(tempfile.mkdtemp(prefix="marli-", dir=root)).resolve()
            if self.uid is not None:
                os.chown(self.workdir, self.uid, self.uid)
            self._make_tmp()
        except BaseException:
            if hasattr(self, "workdir"):
                shutil.rmtree(self.workdir)
            self._release_uid()
            raise
        self._started = True

    def _make_tmp(self) -> None:
        path = self.workdir / "tmp"
        path.mkdir(mode=0o700)
        if self.uid is not None:
            os.chown(path, self.uid, self.uid)

    def _release_uid(self) -> None:
        if self.uid is not None:
            with _UID_LOCK:
                _UIDS_IN_USE.remove(self.uid)
            self.uid = None

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("sandbox is not running")

    def _signal_processes(self) -> set[int]:
        for process in self._processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        pids = _uid_pids(self.uid) if self.uid is not None else set()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        return pids

    async def _stop_processes(self) -> None:
        deadline = time.monotonic() + 1.5
        while True:
            pids = self._signal_processes()
            direct = {process.pid for process in self._processes}
            for pid in pids - direct:
                with contextlib.suppress(ChildProcessError, ProcessLookupError):
                    os.waitpid(pid, os.WNOHANG)
            if not pids or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.01)
        if self.uid is not None and _uid_pids(self.uid):
            raise RuntimeError(f"could not clean up sandbox processes for uid {self.uid}")

    async def exec(
        self, cmd: Sequence[str] | str, *, timeout_s: float, stdin: str | None = None
    ) -> ExecResult:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        argv = ["bash", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        if not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("command must contain string arguments")
        async with self._lock:
            self._require_started()
            config = {
                "workdir": str(self.workdir),
                "uid": self.uid,
                "landlock_abi": self._abi,
                "limits": {
                    "as": self.limits.address_space_bytes,
                    "cpu": math.ceil(timeout_s + self.limits.cpu_grace_s),
                    "fsize": self.limits.file_size_bytes,
                    "nofile": self.limits.open_files,
                    "nproc": self.limits.processes,
                },
            }
            started = time.monotonic()
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-B",
                    "-m",
                    "marli.envs.sandbox._launch",
                    json.dumps(config),
                    "--",
                    *argv,
                    cwd=self.workdir,
                    env={
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "HOME": str(self.workdir),
                        "LANG": "C.UTF-8",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    start_new_session=True,
                    stdin=asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
            try:
                process = await asyncio.shield(launch)
            except asyncio.CancelledError:
                # Retain the pid even if cancellation arrives during creation,
                # so descendants cannot outlive an interrupted episode.
                process = await launch
                self._processes.add(process)
                try:
                    await self._stop_processes()
                    await process.communicate()
                finally:
                    self._processes.discard(process)
                raise
            self._processes.add(process)
            output = asyncio.create_task(
                process.communicate(None if stdin is None else stdin.encode())
            )
            timed_out = False
            try:
                if self._closed:
                    self._signal_processes()
                try:
                    stdout, stderr = await asyncio.wait_for(asyncio.shield(output), timeout_s)
                except TimeoutError:
                    timed_out = True
                    await self._stop_processes()
                    stdout, stderr = await output
            finally:
                try:
                    await self._stop_processes()
                    await output
                finally:
                    self._processes.discard(process)
            return ExecResult(
                None if timed_out else process.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
                timed_out,
                time.monotonic() - started,
            )

    @contextlib.contextmanager
    def _file_parent(self, rel: str, *, create: bool = False) -> Iterator[tuple[int, str]]:
        self._require_started()
        path = Path(rel)
        parts: list[str] = []
        if path.is_absolute():
            raise ValueError("sandbox paths must be relative")
        for part in path.parts:
            if part == "..":
                if not parts:
                    raise ValueError("path escapes sandbox workdir")
                parts.pop()
            else:
                parts.append(part)
        if not parts:
            raise ValueError("expected a file path within the sandbox")
        # Open each component relative to an already opened directory, refusing
        # symlinks so model code cannot race a privileged read/write into the host.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(self.workdir, flags)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        if self.uid is not None:
                            os.chown(part, self.uid, self.uid, dir_fd=fd, follow_symlinks=False)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError("sandbox file paths must not traverse symlinks") from exc
            raise
        finally:
            os.close(fd)

    async def read_file(self, rel: str) -> str:
        with self._file_parent(rel) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "r", encoding="utf-8") as file:
                return file.read()

    async def write_file(self, rel: str, content: str) -> None:
        with self._file_parent(rel, create=True) as (parent, name):
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
            fd = os.open(name, flags, mode=0o600, dir_fd=parent)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                if self.uid is not None:
                    os.fchown(file.fileno(), self.uid, self.uid)
                file.write(content)

    async def reset(self) -> None:
        self._require_started()
        await self._stop_processes()
        async with self._lock:
            self._require_started()
            for path in self.workdir.iterdir():
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            self._make_tmp()

    async def close(self) -> None:
        self._closed = True
        if not self._started:
            return
        await self._stop_processes()
        async with self._lock:
            if not self._started:
                return
            shutil.rmtree(self.workdir)
            self._release_uid()
            self._started = False
