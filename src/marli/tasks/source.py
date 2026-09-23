"""Record dataset schemas and text restrictions without importing dataset backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
    test_format: str | None = None
    subsets: list[str] = field(default_factory=list)
    data_files: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("math", "code"):
            raise ValueError("kind must be 'math' or 'code'")
        if self.kind == "math":
            if self.answer_format not in ("integer", "latex"):
                raise ValueError("answer_format must be 'integer' or 'latex'")
            if self.test_format is not None or self.subsets or self.data_files is not None:
                raise ValueError("test_format, subsets and data_files are code-only fields")
        else:
            if self.answer_format != "tests":
                raise ValueError("code answer_format must be 'tests'")
            if self.test_format not in ("deepcoder", "lcb"):
                raise ValueError("code test_format must be 'deepcoder' or 'lcb'")
            if self.config is not None and self.subsets:
                raise ValueError("supply config or subsets, not both")
            if not isinstance(self.subsets, list) or any(
                not isinstance(subset, str) or not subset for subset in self.subsets
            ):
                raise ValueError("subsets must be a list of nonempty strings")
            if len(set(self.subsets)) != len(self.subsets):
                raise ValueError("subsets must be unique")
            if self.data_files is not None and (
                not isinstance(self.data_files, str) or not self.data_files
            ):
                raise ValueError("data_files must be a nonempty filename")
            if self.test_format == "lcb" and self.data_files is None:
                raise ValueError("lcb sources require data_files")
        allowed = {"id", "prompt", "answer", "difficulty", "topic"}
        if not isinstance(self.fields, dict) or self.fields.keys() - allowed:
            raise ValueError(f"fields must map names from {sorted(allowed)} to columns")
        if any(not isinstance(value, str) or not value for value in self.fields.values()):
            raise ValueError("field paths must be nonempty strings")
        if self.kind == "code" and (self.answer_path is not None or "answer" in self.fields):
            raise ValueError("code sources use test_format, not an answer path")
        for name in ("prompt", "answer") if self.kind == "math" else ("prompt",):
            if name not in self.fields and getattr(self, f"{name}_path") is None:
                raise ValueError(f"fields must include {name}, or supply {name}_path")
        allowed_filters = (
            {"answer_nonempty"} if self.kind == "math" else {"contest_date_min", "contest_date_max"}
        )
        if not isinstance(self.filters, dict) or self.filters.keys() - allowed_filters:
            raise ValueError(f"unknown filters; supported: {sorted(allowed_filters)}")
        if self.kind == "math":
            if any(type(value) is not bool for value in self.filters.values()):
                raise ValueError("answer_nonempty must be a boolean")
        else:
            for value in self.filters.values():
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError("contest date filters must be YYYY-MM-DD strings")
            if self.filters.get("contest_date_min", "") > self.filters.get(
                "contest_date_max", "9999-12-31"
            ):
                raise ValueError("contest_date_min must not exceed contest_date_max")
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
