"""Normalize source rows once, preserving original row identity through sampling."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from typing import Any

from marli.envs.base import Task
from marli.tasks.source import TaskSourceSpec

type DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


def strip_dapo_wrapper(prompt: str) -> str:
    """Remove DAPO's instruction paragraph and reminder, preserving the problem bytes.

    Only the known leading instruction and a final reminder line are removed;
    blank lines separating those wrappers from the problem belong to the wrapper.
    Unwrapped prompts and internal paragraphs are otherwise left verbatim.
    """
    if prompt.startswith("Solve the following math problem step by step."):
        boundary = re.search(r"\r?\n[ \t]*\r?\n", prompt)
        if boundary is not None:
            prompt = prompt[boundary.end() :]
    return re.sub(r"(?:\r?\n[ \t]*)+Remember to put your answer[^\r\n]*(?:\r?\n)?\Z", "", prompt)


def _at_path(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
    return value


def _difficulty(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and "/" in value:
        return float(Fraction(value))
    return float(value)


def load_tasks(
    source: TaskSourceSpec,
    *,
    split: str | None = None,
    max_n: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
    loader: DatasetLoader | None = None,
) -> list[Task]:
    """Load, filter and deduplicate before seeded shuffling and truncation.

    Explicit dataset ids are kept verbatim; otherwise ids use the zero-based
    source row index. The injected loader has datasets.load_dataset's call shape.
    """
    if max_n is not None and (type(max_n) is not int or max_n < 0):
        raise ValueError("max_n must be a non-negative integer or None")
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    selected_split = source.split if split is None else split
    rows = loader(source.hf_id, source.config, split=selected_split)
    tasks = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        prompt = _at_path(row, source.prompt_path or source.fields["prompt"])
        answer = _at_path(row, source.answer_path or source.fields["answer"])
        if source.name == "dapo_math_17k":
            prompt = strip_dapo_wrapper(prompt)
        if source.dedupe:
            if prompt in seen:
                continue
            seen.add(prompt)
        if source.filters.get("answer_nonempty") and (answer is None or not str(answer).strip()):
            continue
        task_id = _at_path(row, source.fields["id"]) if "id" in source.fields else row_index
        difficulty = (
            _difficulty(_at_path(row, source.fields["difficulty"]))
            if "difficulty" in source.fields
            else None
        )
        topic = _at_path(row, source.fields["topic"]) if "topic" in source.fields else None
        tasks.append(
            Task(
                task_id=f"{source.name}/{task_id}",
                prompt=prompt,
                answer=None if answer is None else str(answer),
                meta={
                    "source": source.name,
                    "split": selected_split,
                    "answer_format": source.answer_format,
                    "difficulty": difficulty,
                    "topic": topic,
                    "row_index": row_index,
                },
            )
        )
    if shuffle:
        random.Random(seed).shuffle(tasks)
    return tasks if max_n is None else tasks[:max_n]
