"""Record dataset schemas and text restrictions without importing dataset backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marli.registry import Registry


@dataclass(frozen=True)
class TaskSourceSpec:
    name: str
    hf_id: str
    split: str
    kind: str
    fields: dict[str, str]
    answer_format: str
    license: str
    commit_text: bool = False
    config: str | None = None
    answer_path: str | None = None
    prompt_path: str | None = None
    prompt_transform: str | None = None
    dedupe: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind != "math":
            raise ValueError("kind must be 'math'")
        if self.answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        allowed = {"id", "prompt", "answer", "difficulty", "topic"}
        if not isinstance(self.fields, dict) or self.fields.keys() - allowed:
            raise ValueError(f"fields must map names from {sorted(allowed)} to columns")
        if any(not isinstance(value, str) or not value for value in self.fields.values()):
            raise ValueError("field paths must be nonempty strings")
        for name in ("prompt", "answer"):
            if name not in self.fields and getattr(self, f"{name}_path") is None:
                raise ValueError(f"fields must include {name}, or supply {name}_path")
        if not isinstance(self.filters, dict) or self.filters.keys() - {"answer_nonempty"}:
            raise ValueError("unknown filters; supported: answer_nonempty")
        if any(type(value) is not bool for value in self.filters.values()):
            raise ValueError("answer_nonempty must be a boolean")
        if type(self.commit_text) is not bool or type(self.dedupe) is not bool:
            raise ValueError("commit_text and dedupe must be booleans")
        if self.prompt_transform is not None:
            from marli.tasks.loaders import PROMPT_TRANSFORMS

            if (
                not isinstance(self.prompt_transform, str)
                or self.prompt_transform not in PROMPT_TRANSFORMS
            ):
                raise ValueError(f"unknown prompt_transform: {self.prompt_transform!r}")


SOURCES: Registry[TaskSourceSpec] = Registry("tasks", Path(__file__).parent, TaskSourceSpec)
