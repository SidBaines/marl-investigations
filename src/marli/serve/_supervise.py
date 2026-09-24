"""Bound a server's lifetime to its owner, including failed startup and signals.

Detached servers deliberately transfer ownership to their manifest. Their file
descriptors point directly at the log so exiting the launcher cannot break them.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import httpx

from marli.errors import BackendError, ConfigError


def process_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        # A reparented child may remain a zombie when the container's PID 1 does
        # not reap it; it no longer serves requests and cannot receive signals.
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
    except (ProcessLookupError, FileNotFoundError):
        return False


def stop_process_group(
    pid: int, *, grace_s: float = 5.0, child: subprocess.Popen[str] | None = None
) -> None:
    """Terminate an owned session, including workers left behind by its leader."""
    if pid <= 1 or pid == os.getpgrp():
        raise ConfigError(f"refusing to signal unsafe process group {pid}")
    try:
        if process_alive(pid) and os.getpgid(pid) != pid:
            raise ConfigError(f"server pid {pid} is not a process group leader")
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if child is not None:
            child.poll()
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
    if child is not None:
        child.wait(timeout=1)
    else:
        deadline = time.monotonic() + 1.0
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        if process_alive(pid):
            raise BackendError(f"server {pid} is still alive after SIGKILL")


class Supervisor:
    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        log: Path,
        health_url: str,
        ready_timeout_s: float = 900.0,
        grace_s: float = 5.0,
        detach: bool = False,
        exit_on_signal: bool = True,
    ) -> None:
        if isinstance(argv, str) or not argv or any(not isinstance(x, str) for x in argv):
            raise ConfigError("server argv must be a non-empty sequence of strings")
        if ready_timeout_s <= 0 or grace_s < 0:
            raise ConfigError("server ready timeout must be positive and grace non-negative")
        self.argv = list(argv)
        self.env = dict(env or {})
        self.log = Path(log)
        self.health_url = health_url
        self.ready_timeout_s = ready_timeout_s
        self.grace_s = grace_s
        self.detach = detach
        self.exit_on_signal = exit_on_signal
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self._released = False
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._handlers: dict[int, Any] = {}

    @property
    def pid(self) -> int:
        if self.process is None:
            raise BackendError("server has not been started")
        return self.process.pid

    def _signal(self, signum: int, frame: FrameType | None) -> None:
        self.stop_requested = True
        self.stop()
        if self.exit_on_signal:
            raise SystemExit(128 + signum)

    def _register_cleanup(self) -> None:
        atexit.register(self.stop)
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                self._handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal)

    def _unregister_cleanup(self) -> None:
        atexit.unregister(self.stop)
        for signum, previous in self._handlers.items():
            if signal.getsignal(signum) == self._signal:
                signal.signal(signum, previous)
        self._handlers.clear()

    def _tee(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.process.stdout, self.log.open("a", encoding="utf-8") as stream:
            for line in self.process.stdout:
                stream.write(line)
                stream.flush()
                # Logging must continue even if the caller's terminal is gone.
                with contextlib.suppress(OSError, ValueError):
                    sys.stderr.write(line)
                    sys.stderr.flush()

    def failure(self, message: str) -> BackendError:
        try:
            with self.log.open(encoding="utf-8", errors="replace") as stream:
                tail = "".join(deque(stream, maxlen=50))
        except FileNotFoundError:
            tail = ""
        return BackendError(f"{message}\nLast 50 log lines ({self.log}):\n{tail}")

    async def start(self) -> None:
        if self.process is not None or self._stopped:
            raise ConfigError("supervisor can only start one child")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._register_cleanup()
        try:
            with self.log.open("a", encoding="utf-8") as stream:
                self.process = subprocess.Popen(
                    self.argv,
                    env={**os.environ, **self.env},
                    stdin=subprocess.DEVNULL,
                    stdout=stream if self.detach else subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
            if not self.detach:
                self._thread = threading.Thread(target=self._tee, daemon=True)
                self._thread.start()
            deadline = time.monotonic() + self.ready_timeout_s
            async with httpx.AsyncClient(trust_env=False) as client:
                while not self.stop_requested:
                    if self.process.poll() is not None:
                        raise self.failure(f"server exited with code {self.process.returncode}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise self.failure("server readiness deadline exceeded")
                    try:
                        async with asyncio.timeout(remaining):
                            response = await client.get(
                                self.health_url, timeout=min(2.0, remaining)
                            )
                        if response.is_success and self.process.poll() is None:
                            return
                    except (httpx.TransportError, TimeoutError):
                        pass
                    await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                raise self.failure("server startup interrupted")
        except BaseException as exc:
            self.stop()
            if isinstance(exc, (BackendError, OSError)):
                # Draining the pipe during stop captures even the last crash line.
                raise self.failure(str(exc).split("\nLast 50 log lines", 1)[0]) from exc
            raise

    async def wait(self) -> None:
        """Stay with a foreground server, reporting an unsolicited exit loudly."""
        assert self.process is not None
        try:
            while not self.stop_requested:
                if self.process.poll() is not None:
                    self.stop()
                    raise self.failure(f"server exited with code {self.process.returncode}")
                await asyncio.sleep(0.1)
        finally:
            self.stop()

    def release(self) -> None:
        """Transfer a ready detached child to the caller's durable manifest."""
        if not self.detach or self.process is None or self.process.poll() is not None:
            raise BackendError("only a running detached server can be released")
        self._released = True
        self._unregister_cleanup()

    def stop(self) -> None:
        if self._released or self._stopped:
            return
        self._stopped = True
        self._unregister_cleanup()
        if self.process is not None:
            stop_process_group(self.pid, grace_s=self.grace_s, child=self.process)
        if self._thread is not None:
            self._thread.join(timeout=1)
