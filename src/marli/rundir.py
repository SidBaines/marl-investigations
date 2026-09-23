"""Locked run directories make verbs idempotent and recoverable after a crash.

The run record fixes configuration identity before work begins. Durable JSONL
rows preserve progress, and the handle manifest is written last to mark completion.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

from marli import runlog
from marli.errors import ConfigError, HashMismatchError, MarliError
from marli.handles import Handle, atomic_write_text


class RunStatus(str, Enum):  # noqa: UP042 -- preserve the specified str/Enum API.
    FRESH = "fresh"
    RESUME = "resume"
    COMPLETE = "complete"


class RunDirLockedError(MarliError):
    """Another owner holds this run directory's exclusive lock."""

    exit_code: int = 1


class RunDir:
    """Own one verb's --out directory while holding its non-blocking POSIX lock."""

    status: RunStatus

    def __init__(
        self,
        out: str | Path,
        *,
        kind: str,
        manifest_name: str,
        config_hash: str,
        force: bool = False,
    ) -> None:
        self.out = Path(out).resolve()
        self.kind = kind
        self.manifest_name = manifest_name
        self.config_hash = config_hash
        self.force = force
        self._lock_fd: int | None = None

    def open(self) -> RunStatus:
        """Lock the directory, then determine whether to start, resume, or reuse it."""
        if self._lock_fd is not None:
            return self.status
        self.out.mkdir(parents=True, exist_ok=True)
        control = self.out / ".marli"
        control.mkdir(exist_ok=True)
        self._lock_fd = os.open(control / "lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunDirLockedError(f"run directory is locked: {self.out}") from exc
            if self.force:
                # Never unlink the locked inode: a new lock file would permit a second owner.
                for child in self.out.iterdir():
                    if child == control:
                        for entry in control.iterdir():
                            if entry.name != "lock":
                                self._remove(entry)
                    else:
                        self._remove(child)
            manifest = self.path(self.manifest_name)
            record = control / "run.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                self._check_hash(data.get("config_hash"))
                self.status = RunStatus.COMPLETE
            elif record.exists():
                data = json.loads(record.read_text(encoding="utf-8"))
                self._check_hash(data.get("config_hash"))
                if data["kind"] != self.kind:
                    raise HashMismatchError(
                        f"{self.out}: run kind {data['kind']!r} differs from {self.kind!r}; "
                        "use --force or a different --out"
                    )
                self.status = RunStatus.RESUME
            elif any(child != control for child in self.out.iterdir()):
                raise ConfigError(
                    f"{self.out}: not a marli run dir; refusing to write into it; "
                    "use --force or an empty dir"
                )
            else:
                data = {
                    "kind": self.kind,
                    "config_hash": self.config_hash,
                    "manifest": self.manifest_name,
                    "provenance": runlog.provenance(),
                    "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                atomic_write_text(record, json.dumps(data, sort_keys=True, indent=2) + "\n")
                self.status = RunStatus.FRESH
            return self.status
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _check_hash(self, recorded: str | None) -> None:
        if recorded != self.config_hash:
            raise HashMismatchError(
                f"{self.out}: config hash {str(recorded)[:12]} differs from "
                f"{self.config_hash[:12]}; use --force or a different --out"
            )

    def _require_open(self) -> None:
        if self._lock_fd is None:
            raise MarliError(f"run directory is not open: {self.out}; call open() first")

    def close(self) -> None:
        """Release ownership; closing an already closed run is harmless."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def __enter__(self) -> RunDir:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def path(self, rel: str) -> Path:
        return self.out / rel

    def append_row(self, rel: str, row: Mapping[str, Any]) -> None:
        """Append and sync one complete JSONL row."""
        self._require_open()
        text = json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        path = self.path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def read_rows(self, rel: str) -> list[dict[str, Any]]:
        """Recover complete rows, truncating only an incomplete or invalid final line."""
        self._require_open()
        path = self.path(rel)
        if not path.exists():
            return []
        rows = []
        with path.open("r+b") as stream:
            size = os.fstat(stream.fileno()).st_size
            line_number = 0
            while line := stream.readline():
                line_number += 1
                end = stream.tell()
                try:
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if end == size:
                        break
                    raise MarliError(f"{path}: invalid JSON on line {line_number}") from exc
                rows.append(row)
            else:
                return rows
            # Binary offsets also handle a crash partway through a UTF-8 character.
            stream.truncate(end - len(line))
            stream.flush()
            os.fsync(stream.fileno())
        return rows

    def done_keys(self, rel: str, key: str) -> set[Any]:
        self._require_open()
        return {row[key] for row in self.read_rows(rel)}

    def write_progress(self, data: Mapping[str, Any]) -> None:
        """Atomically publish the latest progress with a UTC update timestamp."""
        self._require_open()
        record = {**data, "updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
        atomic_write_text(
            self.path("progress.json"),
            json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        )

    def finalize(self, handle: Handle) -> Handle:
        """Save the completion manifest last, carrying this run's configuration hash."""
        self._require_open()
        if handle.root.resolve() != self.out:
            raise ConfigError(f"handle root {handle.root} does not match run directory {self.out}")
        saved = replace(handle, config_hash=self.config_hash)
        saved.save()
        self.status = RunStatus.COMPLETE
        return saved
