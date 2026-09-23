"""Keep task bytes portable and separate a handle's kind from its task domain."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from marli.envs.base import Task
from marli.errors import ConfigError
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
        if type(self.n) is not int or self.n < 0:
            raise ValueError("TaskSet n must be a non-negative integer")
        if self.answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        if type(self.commit_text) is not bool:
            raise ValueError("commit_text must be a boolean")
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
    """Read every row and enforce the manifest count; atomic writes need no tail recovery."""
    tasks = []
    with taskset.file("tasks").open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            tasks.append(Task(**row))
    if len(tasks) != taskset.n:
        raise ConfigError(f"TaskSet row count mismatch: expected {taskset.n}, read {len(tasks)}")
    return tasks
