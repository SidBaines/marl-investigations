"""Keep model-written programs within an episode's files and resource budget.

Landlock restricts filesystem access; seccomp denies sockets and dangerous
syscalls. A locked, dedicated uid permits cleanup even after double forks.
The trainer never becomes a subreaper. PID 1 remains responsible for reaping
orphan zombies, which cannot execute or hold memory. /dev/shm is denied, so
multiprocessing's named semaphores are unavailable. Anonymous Unix stream pairs
work; datagram pairs are denied because they can retarget named host peers.

RSS and disk limits are sampled every 250 ms, not enforced as cgroup quotas:
short-lived overshoot is possible. Per-process rlimits provide a second bound.
Explicitly disabling a required kernel/identity feature warns about degradation.
Non-root degraded use needs a writable, searchable MARLI_SANDBOX_ROOT; the
default root-owned 0711 directory is reserved for privileged sandbox creation.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from marli.envs.sandbox import landlock
from marli.envs.sandbox.base import ExecResult
from marli.errors import ConfigError

_DEFAULT_ROOT = Path("/tmp/marli-sandboxes")
_LOCK_ROOT = Path("/run/marli-sandbox")
_FALLBACK_LOCK_ROOT = Path("/tmp/marli-sandbox-locks")
_MARKER = ".marli-sandbox"
_LAUNCHER = Path(__file__).with_name("_launch.py").resolve()


@dataclass(frozen=True)
class ResourceLimits:
    address_space_bytes: int = 4 * 1024**3
    cpu_grace_s: int = 5
    file_size_bytes: int = 256 * 1024**2
    open_files: int = 256
    processes: int = 64

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")


def _uid_usage(uid: int) -> dict[int, int]:
    usage = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = dict(
                line.split(":", 1) for line in (entry / "status").read_text().splitlines()
            )
            if int(fields["Uid"].split()[0]) == uid and fields["State"].strip()[0] != "Z":
                usage[int(entry.name)] = int(fields.get("VmRSS", "0").split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            pass
    return usage


def _uid_pids(uid: int) -> set[int]:
    """Return live processes, including non-dumpable and detached descendants."""
    return set(_uid_usage(uid))


def _lock_directory() -> Path:
    try:
        _LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
        root = _LOCK_ROOT
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EROFS):
            raise
        _FALLBACK_LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
        root = _FALLBACK_LOCK_ROOT
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o700:
        raise ConfigError(f"sandbox lock directory must be root-owned mode 0700: {root}")
    return root


def _try_uid_lock(root: Path, uid: int) -> int | None:
    fd = os.open(root / f"{uid}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _allocate_uid(uid_range: tuple[int, int]) -> tuple[int, int]:
    root = _lock_directory()
    for uid in range(uid_range[0], uid_range[1] + 1):
        fd = _try_uid_lock(root, uid)
        if fd is not None:
            try:
                busy = bool(_uid_pids(uid))
            except BaseException:
                os.close(fd)
                raise
            if not busy:
                return uid, fd
            os.close(fd)
    raise ConfigError(f"sandbox uid pool exhausted ({uid_range[0]}–{uid_range[1]})")


def _sweep_stale(root: Path) -> None:
    """Delete only marked, idle directories while holding their uid's lock."""
    lock_root = _lock_directory()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            fd = os.open(path / _MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            info = os.fstat(fd)
            if info.st_uid != 0 or not stat.S_ISREG(info.st_mode) or info.st_size > 128:
                continue
            marker = json.loads(os.read(fd, 128))
            uid = marker["uid"]
            if marker != {"uid": uid, "version": 1} or type(uid) is not int or uid <= 0:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        finally:
            os.close(fd)
        lock_fd = _try_uid_lock(lock_root, uid)
        if lock_fd is None:
            continue
        try:
            if not _uid_pids(uid):
                # A concurrent close (this or another trainer process) may have just removed it.
                shutil.rmtree(path, onexc=_ignore_missing)
        finally:
            os.close(lock_fd)


def _ignore_missing(function: object, path: str, exc: BaseException) -> None:
    if not isinstance(exc, FileNotFoundError):
        raise exc


def _disk_usage(workdir: Path, limit: int) -> int:
    """Bound both bytes and scan work; never follow model-controlled symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(workdir, flags)
    stack = [(fd, os.scandir(fd))]
    total, entries = 0, 0
    try:
        while stack:
            parent, iterator = stack[-1]
            entry = next(iterator, None)
            if entry is None:
                iterator.close()
                os.close(parent)
                stack.pop()
                continue
            entries += 1
            if entries > 100_000 or len(stack) > 128:
                raise OSError("sandbox disk scan exceeded its entry/depth budget")
            try:
                info = entry.stat(follow_symlinks=False)
                total += max(info.st_size, info.st_blocks * 512)
                if total > limit:
                    return total
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, flags, dir_fd=parent)
                    stack.append((child, os.scandir(child)))
            except FileNotFoundError:
                pass
    finally:
        for parent, iterator in stack:
            iterator.close()
            os.close(parent)
    return total


class _OutputBuffer:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.total = 0
        self.head = bytearray()
        self.tail = bytearray()

    def append(self, data: bytes) -> None:
        self.total += len(data)
        head_size = (self.cap + 1) // 2
        take = min(len(data), head_size - len(self.head))
        self.head.extend(data[:take])
        tail_size = self.cap - head_size
        if tail_size:
            self.tail.extend(data[take:])
            del self.tail[:-tail_size]

    def text(self) -> str:
        return (self.head + self.tail).decode(errors="replace")


async def _helper(*args: str) -> tuple[int, str]:
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-I", "-S", "-B", str(_LAUNCHER), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
        timeout=5,
        check=False,
    )
    return result.returncode, (result.stderr if result.returncode else result.stdout).decode()


class SubprocessSandbox:
    def __init__(
        self,
        *,
        require_landlock: bool = True,
        require_seccomp: bool = True,
        require_dedicated_uid: bool = True,
        limits: ResourceLimits | None = None,
        uid_range: tuple[int, int] = (61000, 61999),
        max_output_bytes: int = 1024**2,
        max_memory_bytes: int = 4 * 1024**3,
        max_disk_bytes: int = 2 * 1024**3,
    ) -> None:
        if not (0 < uid_range[0] <= uid_range[1] < 2**32 - 1):
            raise ConfigError("sandbox uid range must contain positive unprivileged identities")
        for name, value in (
            ("max_output_bytes", max_output_bytes),
            ("max_memory_bytes", max_memory_bytes),
            ("max_disk_bytes", max_disk_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        self.require_landlock = require_landlock
        self.require_seccomp = require_seccomp
        self.require_dedicated_uid = require_dedicated_uid
        self.limits = limits or ResourceLimits()
        self.uid_range = uid_range
        self.max_output_bytes = max_output_bytes
        self.max_memory_bytes = max_memory_bytes
        self.max_disk_bytes = max_disk_bytes
        self.workdir: Path
        self.uid: int | None = None
        self._uid_lock_fd: int | None = None
        self._abi = 0
        self._seccomp = False
        self._started = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._processes: set[asyncio.subprocess.Process] = set()

    async def start(self) -> None:
        async with self._lock:
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
                    f"{message}; degraded mode: filesystem protection may be incomplete or absent",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if os.geteuid() != 0:
                message = "sandbox requires a dedicated uid (root required)"
                if self.require_dedicated_uid:
                    raise ConfigError(message)
                warnings.warn(
                    f"{message}; degraded mode: aggregate RSS, NPROC and detached-process "
                    "cleanup are unavailable; use only trusted code",
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                code, detail = await _helper("--probe-seccomp")
            except (OSError, subprocess.TimeoutExpired) as exc:
                code, detail = 1, str(exc)
            self._seccomp = code == 0
            if not self._seccomp:
                message = f"cannot install sandbox seccomp filter: {detail.strip()}"
                if self.require_seccomp:
                    raise ConfigError(message)
                warnings.warn(f"{message}; degraded mode", RuntimeWarning, stacklevel=2)
            # Finish setup even if cancelled, so its uid lock and directory can be released.
            prepare = asyncio.create_task(asyncio.to_thread(self._prepare_workdir))
            try:
                await asyncio.shield(prepare)
            except BaseException:
                await prepare
                if hasattr(self, "workdir"):
                    await asyncio.to_thread(shutil.rmtree, self.workdir)
                self._release_uid()
                raise
            self._started = True

    def _prepare_workdir(self) -> None:
        configured = os.environ.get("MARLI_SANDBOX_ROOT")
        root = Path(configured) if configured else _DEFAULT_ROOT
        if not configured:
            root.mkdir(mode=0o711, exist_ok=True)
            info = root.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) != 0o711
            ):
                raise ConfigError(f"default sandbox root must be root-owned mode 0711: {root}")
        root = root.resolve()
        for ancestor in (root, *root.parents):
            if not ancestor.stat().st_mode & stat.S_IXOTH:
                raise ConfigError(f"sandbox ancestor is not searchable by others (o+x): {ancestor}")
        if os.geteuid() == 0:
            _sweep_stale(root)
            self.uid, self._uid_lock_fd = _allocate_uid(self.uid_range)
        try:
            self.workdir = Path(tempfile.mkdtemp(prefix="marli-", dir=root)).resolve()
            if self.uid is not None:
                marker = self.workdir / _MARKER
                marker.write_text(json.dumps({"uid": self.uid, "version": 1}))
                marker.chmod(0o600)
                os.chown(self.workdir, self.uid, self.uid)
            self._make_tmp()
        except BaseException:
            if hasattr(self, "workdir"):
                shutil.rmtree(self.workdir)
                del self.workdir
            self._release_uid()
            raise

    def _make_tmp(self) -> None:
        path = self.workdir / "tmp"
        path.mkdir(mode=0o700)
        if self.uid is not None:
            os.chown(path, self.uid, self.uid)

    def _release_uid(self) -> None:
        if self._uid_lock_fd is not None:
            os.close(self._uid_lock_fd)
            self._uid_lock_fd = None
        self.uid = None

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("sandbox is not running")

    async def _stop_processes(self) -> None:
        async with self._cleanup_lock:
            for process in self._processes:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            if self.uid is None:
                return
            deadline = time.monotonic() + 1.5
            while await asyncio.to_thread(_uid_pids, self.uid):
                code, detail = await _helper("--kill-uid", str(self.uid))
                if code:
                    raise RuntimeError(f"sandbox uid cleanup failed: {detail.strip()}")
                if time.monotonic() >= deadline:
                    if await asyncio.to_thread(_uid_pids, self.uid):
                        raise RuntimeError(
                            f"could not clean up sandbox processes for uid {self.uid}"
                        )
                    break
                await asyncio.sleep(0.01)

    def _resource_error(self) -> str | None:
        if self.uid is not None and sum(_uid_usage(self.uid).values()) > self.max_memory_bytes:
            return f"sandbox max_memory_bytes exceeded ({self.max_memory_bytes})"
        try:
            if _disk_usage(self.workdir, self.max_disk_bytes) > self.max_disk_bytes:
                return f"sandbox max_disk_bytes exceeded ({self.max_disk_bytes})"
        except OSError as exc:
            return f"sandbox disk monitoring failed: {exc}"
        return None

    async def _watch_resources(self, stopped: asyncio.Event) -> str | None:
        """Poll resource use in short off-loop checks, never parking a worker thread.

        A watchdog thread held for a command's whole lifetime starved the shared
        default executor (32 threads) at ~32 concurrent commands: each command's
        cleanup awaited ``to_thread`` while every thread waited for that cleanup.
        """
        while not stopped.is_set():
            error = await asyncio.to_thread(self._resource_error)
            if error:
                return error
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stopped.wait(), 0.25)
        return None

    async def _collect(
        self,
        process: asyncio.subprocess.Process,
        stdin: bytes | None,
        stdout: _OutputBuffer,
        stderr: _OutputBuffer,
        overflow: asyncio.Event,
    ) -> None:
        async def drain(stream: asyncio.StreamReader | None, output: _OutputBuffer) -> None:
            assert stream is not None
            while data := await stream.read(64 * 1024):
                output.append(data)
                if output.total > 4 * output.cap:
                    overflow.set()

        async def feed() -> None:
            if process.stdin is not None:
                try:
                    assert stdin is not None
                    for offset in range(0, len(stdin), 64 * 1024):
                        process.stdin.write(stdin[offset : offset + 64 * 1024])
                        await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()

        await asyncio.gather(drain(process.stdout, stdout), drain(process.stderr, stderr), feed())
        await process.wait()

    async def exec(
        self, cmd: Sequence[str] | str, *, timeout_s: float, stdin: str | None = None
    ) -> ExecResult:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        argv = ["bash", "--noprofile", "--norc", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        if not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("command must contain string arguments")
        input_bytes = None if stdin is None else stdin.encode()
        async with self._lock:
            self._require_started()
            config = {
                "workdir": str(self.workdir),
                "uid": self.uid,
                "landlock_abi": self._abi,
                "seccomp": self._seccomp,
                "limits": {
                    "as": self.limits.address_space_bytes,
                    "cpu": math.ceil(timeout_s + self.limits.cpu_grace_s),
                    "fsize": self.limits.file_size_bytes,
                    "nofile": self.limits.open_files,
                    "nproc": self.limits.processes,
                },
            }
            started = time.monotonic()
            error = await asyncio.to_thread(self._resource_error)
            if error:
                diagnostic = _OutputBuffer(self.max_output_bytes)
                diagnostic.append(error.encode())
                return ExecResult(
                    1,
                    "",
                    diagnostic.text(),
                    False,
                    time.monotonic() - started,
                    diagnostic.total > diagnostic.cap,
                )
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(_LAUNCHER),
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
            cancelled = False
            try:
                process = await asyncio.shield(launch)
            except asyncio.CancelledError:
                process = await launch
                cancelled = True
            self._processes.add(process)
            stdout, stderr = (
                _OutputBuffer(self.max_output_bytes),
                _OutputBuffer(self.max_output_bytes),
            )
            overflow = asyncio.Event()
            output = asyncio.create_task(
                self._collect(process, input_bytes, stdout, stderr, overflow)
            )
            excess_output = asyncio.create_task(overflow.wait())
            stopped = asyncio.Event()
            watchdog = asyncio.create_task(self._watch_resources(stopped))
            timed_out = False
            try:
                if cancelled:
                    raise asyncio.CancelledError
                if self._closed:
                    await self._stop_processes()
                done, _ = await asyncio.wait(
                    (output, watchdog, excess_output),
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    timed_out = True
                elif watchdog in done:
                    error = watchdog.result()
                elif excess_output in done:
                    error = (
                        f"sandbox max_output_bytes exceeded ({self.max_output_bytes} per stream)"
                    )
            finally:
                try:
                    await self._stop_processes()
                    await output
                finally:
                    stopped.set()
                    watch_error = await watchdog
                    excess_output.cancel()
                    await asyncio.gather(excess_output, return_exceptions=True)
                    self._processes.discard(process)
            error = error or watch_error or await asyncio.to_thread(self._resource_error)
            truncated = stdout.total > stdout.cap or stderr.total > stderr.cap
            stderr_text = stderr.text()
            if error:
                diagnostic = _OutputBuffer(self.max_output_bytes)
                diagnostic.append((error + "\n").encode())
                diagnostic.append(stderr.head + stderr.tail)
                truncated |= diagnostic.total > diagnostic.cap
                stderr_text = diagnostic.text()
            return ExecResult(
                None if timed_out else (process.returncode or 1) if error else process.returncode,
                stdout.text(),
                stderr_text,
                timed_out,
                time.monotonic() - started,
                truncated,
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
                if path.name == _MARKER:
                    continue
                if path.is_dir() and not path.is_symlink():
                    await asyncio.to_thread(shutil.rmtree, path)
                else:
                    path.unlink()
            self._make_tmp()

    async def close(self) -> None:
        self._closed = True
        if self._started:
            await self._stop_processes()
        async with self._lock:
            if not self._started:
                return
            await asyncio.to_thread(shutil.rmtree, self.workdir)
            self._release_uid()
            self._started = False
