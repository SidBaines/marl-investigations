"""Normalize source rows once, preserving original row identity through sampling."""

from __future__ import annotations

import json
import random
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
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


PROMPT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "dapo_strip_wrapper": strip_dapo_wrapper,
}


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


def _select_code_tasks(
    tasks: Iterable[Task], *, max_n: int | None, seed: int, shuffle: bool
) -> Iterator[Task]:
    """Shuffle offsets, not bundles; unshuffled builds need no scratch disk.

    Exhaust the input even after max_n so manifest counts cover the full pool.
    For exact seeded shuffle semantics, spool normalized rows to a temporary
    file in the working directory, retaining only byte offsets in memory. Read
    back one selected row at a time and close/remove the spool on exit. This
    trades bounded scratch disk for memory even when max_n is unlimited.
    """
    if not shuffle:
        for index, task in enumerate(tasks):
            if max_n is None or index < max_n:
                yield task
        return
    with tempfile.TemporaryFile(dir=Path.cwd()) as spool:
        offsets: list[int] = []
        for task in tasks:
            offsets.append(spool.tell())
            spool.write((json.dumps(asdict(task), ensure_ascii=False) + "\n").encode("utf-8"))
        random.Random(seed).shuffle(offsets)
        for offset in offsets[:max_n]:
            spool.seek(offset)
            yield Task(**json.loads(spool.readline()))


def load_tasks(
    source: TaskSourceSpec,
    *,
    split: str | None = None,
    max_n: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
    loader: DatasetLoader | None = None,
    meta: dict[str, Any] | None = None,
) -> Iterable[Task]:
    """Load, filter and deduplicate before seeded shuffling and truncation.

    Explicit dataset ids are kept verbatim; otherwise ids use the zero-based
    source row index. The injected loader has datasets.load_dataset's call shape.
    Deduplication keeps the first row only when all answers for that transformed
    prompt agree (as strings); any conflict drops the entire prompt group.

    Pass a dict as ``meta`` and then to ``TaskSet(meta=meta)`` to record counts
    before filtering, shuffle, or truncation: ``n_raw`` counts source rows,
    ``n_duplicates`` counts rows beyond each prompt's first, and
    ``n_conflicting_dropped`` counts unique prompt groups excluded for conflicts.

    Code sources return an iterator; consume it completely to finalize ``meta``.
    Their hidden tests are capped using the source settings and this seed.
    """
    if max_n is not None and (type(max_n) is not int or max_n < 0):
        raise ValueError("max_n must be a non-negative integer or None")
    if source.kind == "code":
        from marli.tasks.code import load_code_tasks

        return _select_code_tasks(
            load_code_tasks(source, split=split, loader=loader, meta=meta, seed=seed),
            max_n=max_n,
            seed=seed,
            shuffle=shuffle,
        )
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    selected_split = source.split if split is None else split
    rows = loader(source.hf_id, source.config, split=selected_split)
    tasks: list[Task] = []
    seen: dict[str, str | None] = {}
    conflicts: set[str] = set()
    n_raw = n_duplicates = 0
    transform = PROMPT_TRANSFORMS[source.prompt_transform] if source.prompt_transform else None
    for row_index, row in enumerate(rows):
        n_raw += 1
        prompt = _at_path(row, source.prompt_path or source.fields["prompt"])
        answer = _at_path(row, source.answer_path or source.fields["answer"])
        answer = None if answer is None else str(answer)
        if transform is not None:
            prompt = transform(prompt)
        if source.dedupe:
            if prompt in seen:
                n_duplicates += 1
                if answer != seen[prompt]:
                    conflicts.add(prompt)
                continue
            seen[prompt] = answer
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
                answer=answer,
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
    if conflicts:
        tasks = [task for task in tasks if task.prompt not in conflicts]
    if meta is not None:
        meta.update(n_raw=n_raw, n_duplicates=n_duplicates, n_conflicting_dropped=len(conflicts))
    if shuffle:
        random.Random(seed).shuffle(tasks)
    return tasks if max_n is None else tasks[:max_n]
