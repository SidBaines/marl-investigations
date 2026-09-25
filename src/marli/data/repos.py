"""Group code problems without replacement while keeping episode rules unsampled.

Repo rows preserve the original problem bundles and provenance. Only whether a
repo has a rule is fixed here; the environment samples its secret each episode.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import batched

from marli.config import input_field
from marli.envs.base import Task
from marli.errors import ConfigError
from marli.handles import InputRef
from marli.rundir import RunDir
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks


@dataclass
class DataReposConfig:
    tasks: str | None = input_field(None, help="code TaskSet (answer_format tests)")
    n_per_repo: int = 4
    max_repos: int | None = None
    shuffle: bool = True
    seed: int = 0
    rule_prob: float = 1.0
    rule_families: tuple[str, ...] = ("header", "constant", "docstring")

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ConfigError("tasks is required")
        if type(self.n_per_repo) is not int or self.n_per_repo <= 0:
            raise ConfigError("n_per_repo must be a positive integer")
        if self.max_repos is not None and (type(self.max_repos) is not int or self.max_repos < 0):
            raise ConfigError("max_repos must be a non-negative integer or None")
        if type(self.shuffle) is not bool:
            raise ConfigError("shuffle must be a boolean")
        if type(self.seed) is not int:
            raise ConfigError("seed must be an integer")
        if (
            type(self.rule_prob) not in (int, float)
            or not math.isfinite(self.rule_prob)
            or not 0 <= self.rule_prob <= 1
        ):
            raise ConfigError("rule_prob must be between 0 and 1")
        if (
            not isinstance(self.rule_families, (tuple, list))
            or not self.rule_families
            or any(
                family not in ("header", "constant", "docstring") for family in self.rule_families
            )
        ):
            raise ConfigError("rule_families must contain header, constant or docstring")


async def repos(cfg: DataReposConfig, run: RunDir) -> TaskSet:
    assert cfg.tasks is not None
    source = TaskSet.load(cfg.tasks)
    if source.kind != "code" or source.answer_format != "tests":
        raise ConfigError("data repos requires a code TaskSet with answer_format tests")
    rng = random.Random(cfg.seed)
    tasks = read_tasks(source, stream=True)
    if cfg.shuffle:
        tasks = list(tasks)
        rng.shuffle(tasks)
    counts = {"input": 0, "kept": 0, "dropped_remainder": 0, "dropped_max_repos": 0}
    n_repos = 0
    n_with_rule = 0

    def grouped() -> Iterator[Task]:
        nonlocal n_repos, n_with_rule
        for problems in batched(tasks, cfg.n_per_repo):
            counts["input"] += len(problems)
            if len(problems) < cfg.n_per_repo:
                counts["dropped_remainder"] += len(problems)
                continue
            if cfg.max_repos is not None and n_repos >= cfg.max_repos:
                counts["dropped_max_repos"] += len(problems)
                continue
            has_rule = rng.random() < cfg.rule_prob
            task = Task(
                task_id=f"repo-{n_repos:05d}",
                prompt=f"A shared repository with {cfg.n_per_repo} contributor tasks.",
                answer={
                    "kind": "code_repo",
                    "problems": [asdict(problem) for problem in problems],
                    "has_rule": has_rule,
                    "rule_families": list(cfg.rule_families),
                },
            )
            n_repos += 1
            n_with_rule += has_rule
            counts["kept"] += len(problems)
            yield task

    path = write_tasks(run.out, grouped())
    reference = InputRef.of(source)
    return TaskSet(
        root=run.out,
        tasks=path.name,
        source=source.source,
        split=source.split,
        kind="code",
        n=n_repos,
        answer_format="code_repo",
        commit_text=source.commit_text,
        inputs=(reference,),
        meta={
            "n_repos": n_repos,
            "n_per_repo": cfg.n_per_repo,
            "counts": {**counts, "with_rule": n_with_rule, "without_rule": n_repos - n_with_rule},
            "source": asdict(reference),
            "rule_prob": cfg.rule_prob,
            "families": list(cfg.rule_families),
        },
    )
