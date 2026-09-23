"""Decontamination removes overlapping prompts without changing source task bytes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marli.config import doc_field, input_field
from marli.errors import ConfigError
from marli.handles import InputRef
from marli.rundir import RunDir
from marli.tasks.loaders import load_tasks
from marli.tasks.source import SOURCES
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks


@dataclass
class BuildConfig:
    source: str = doc_field("aime_2025", help="task source registry name (marli list tasks)")
    split: str | None = None
    max_n: int | None = None
    seed: int = 0
    shuffle: bool = False
    exclude: str | None = input_field(
        None,
        help="TaskSet whose prompts are removed (decontamination by exact/normalised match)",
    )
    ngram_exclude: int = doc_field(
        0, help="if >0, also drop tasks sharing any n-gram of this length (words) with `exclude`"
    )

    def __post_init__(self) -> None:
        if self.max_n is not None and (type(self.max_n) is not int or self.max_n < 0):
            raise ConfigError("max_n must be a non-negative integer or None")
        if type(self.ngram_exclude) is not int or self.ngram_exclude < 0:
            raise ConfigError("ngram_exclude must be a non-negative integer")


def _normalize(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _ngrams(prompt: str, n: int) -> set[tuple[str, ...]]:
    words = prompt.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


async def build(cfg: BuildConfig, run: RunDir) -> TaskSet:
    """Load the requested sample, then exclude normalized exact and word n-gram matches.

    Counts partition the loaded sample; exact matches take precedence over n-grams.
    The runner publishes the manifest only after task rows are durably replaced.
    """
    source = SOURCES.load(cfg.source)
    excluded = TaskSet.load(cfg.exclude) if cfg.exclude is not None else None
    prompts = {_normalize(task.prompt) for task in read_tasks(excluded)} if excluded else set()
    ngrams: set[tuple[str, ...]] = set()
    if cfg.ngram_exclude:
        for prompt in prompts:
            ngrams.update(_ngrams(prompt, cfg.ngram_exclude))
    tasks = load_tasks(source, split=cfg.split, max_n=cfg.max_n, seed=cfg.seed, shuffle=cfg.shuffle)
    counts = {"loaded": len(tasks), "kept": 0, "dropped_exact": 0, "dropped_ngram": 0}
    kept = []
    for task in tasks:
        prompt = _normalize(task.prompt)
        if prompt in prompts:
            counts["dropped_exact"] += 1
        elif cfg.ngram_exclude and not ngrams.isdisjoint(_ngrams(prompt, cfg.ngram_exclude)):
            counts["dropped_ngram"] += 1
        else:
            kept.append(task)
    counts["kept"] = len(kept)
    inputs = (InputRef.of(excluded),) if excluded is not None else ()
    meta: dict[str, Any] = {"counts": counts}
    if excluded is not None:
        meta.update(decontaminated_against=inputs[0].path, ngram_exclude=cfg.ngram_exclude)
    path = write_tasks(run.out, kept)
    return TaskSet(
        root=run.out,
        tasks=path.name,
        source=source.name,
        split=source.split if cfg.split is None else cfg.split,
        n=len(kept),
        kind=source.kind,
        answer_format=source.answer_format,
        commit_text=source.commit_text,
        inputs=inputs,
        meta=meta,
    )
