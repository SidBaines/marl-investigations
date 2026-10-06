"""Keep task bytes portable and separate a handle's kind from its task domain."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from marli.envs.base import Task
from marli.errors import ConfigError
from marli.handles import Handle, register_handle


@register_handle
@dataclass(frozen=True)
class TaskSet(Handle):
    """A saved collection; ``kind`` is its domain (``math`` or ``code``).

    Handle reserves the top-level manifest ``kind`` for dispatch (``taskset``).
    Its extensible metadata therefore stores the domain as ``meta.task_kind``;
    construction and loading restore the public ``kind`` field from that value.
    """

    KIND: ClassVar[str] = "taskset"
    MANIFEST: ClassVar[str] = "taskset.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("tasks",)

    tasks: str
    source: str
    split: str
    kind: str
    n: int
    answer_format: str
    commit_text: bool

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 0:
            raise ValueError("TaskSet n must be a non-negative integer")
        if type(self.commit_text) is not bool:
            raise ValueError("commit_text must be a boolean")
        kind = self.meta.get("task_kind") if self.kind == self.KIND else self.kind
        if kind not in ("math", "code"):
            raise ValueError("TaskSet kind must be 'math' or 'code'")
        # code_repo: a bundle of code problems per task (code_rules; `marli data repos`).
        formats = ("tests", "code_repo") if kind == "code" else ("integer", "latex")
        if self.answer_format not in formats:
            raise ValueError(f"{kind} answer_format must be one of {formats}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "meta", {**self.meta, "task_kind": kind})

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, "source": self.source}


def write_tasks(dir: str | Path, tasks: Iterable[Task]) -> Path:
    """Durably replace task rows before the caller publishes a completion manifest."""
    path = Path(dir) / "tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for task in tasks:
                stream.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_tasks(taskset: TaskSet, *, stream: bool = False) -> list[Task] | Iterator[Task]:
    """Read rows and enforce the manifest count; atomic writes need no tail recovery.

    ``stream=True`` bounds memory to one row and checks the count on exhaustion.
    The default retains the list interface used by rollout and data filter.
    """

    def rows() -> Iterator[Task]:
        count = 0
        with taskset.file("tasks").open("rb") as file:
            for line in file:
                count += 1
                yield Task(**json.loads(line))
        if count != taskset.n:
            raise ConfigError(f"TaskSet row count mismatch: expected {taskset.n}, read {count}")

    return rows() if stream else list(rows())
