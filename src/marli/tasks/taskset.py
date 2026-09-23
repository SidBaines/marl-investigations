"""Keep task bytes portable and separate a handle's kind from its task domain."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from marli.envs.base import Task
from marli.handles import Handle, atomic_write_text, register_handle


@register_handle
@dataclass(frozen=True)
class TaskSet(Handle):
    """A saved collection; ``kind`` is its domain (currently ``math``).

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
        kind = self.meta.get("task_kind") if self.kind == self.KIND else self.kind
        if kind != "math":
            raise ValueError("TaskSet kind must be 'math'")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "meta", {**self.meta, "task_kind": kind})

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, "source": self.source}


def write_tasks(dir: str | Path, tasks: Iterable[Task]) -> Path:
    """Durably replace task rows before the caller publishes a completion manifest."""
    path = Path(dir) / "tasks.jsonl"
    atomic_write_text(
        path,
        "".join(json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in tasks),
    )
    return path


def read_tasks(taskset: TaskSet) -> list[Task]:
    """Read complete rows, tolerating a torn final JSON/UTF-8 line without rewriting it."""
    tasks = []
    with taskset.file("tasks").open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        for line in stream:
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if stream.tell() == size:
                    break
                raise
            tasks.append(Task(**row))
    return tasks
