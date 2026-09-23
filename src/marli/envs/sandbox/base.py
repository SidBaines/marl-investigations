"""The sandbox boundary exposes files and commands, never agent conversations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


class Sandbox(Protocol):
    workdir: Path

    async def start(self) -> None: ...

    async def exec(
        self, cmd: Sequence[str] | str, *, timeout_s: float, stdin: str | None = None
    ) -> ExecResult:
        """Run argv, or a string through bash -lc, from the workdir."""
        ...

    async def read_file(self, rel: str) -> str:
        """Read UTF-8 within workdir; absolute paths and escapes raise ValueError."""
        ...

    async def write_file(self, rel: str, content: str) -> None: ...

    async def reset(self) -> None:
        """Stop commands and wipe the workdir contents."""
        ...

    async def close(self) -> None:
        """Kill everything and remove the workdir; repeated calls are harmless."""
        ...
