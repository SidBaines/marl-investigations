"""Difficulty selection uses saved system grades so failed rollouts cannot bias pass rates."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import fmean

from marli.config import input_field
from marli.errors import ConfigError
from marli.handles import InputRef
from marli.interact import records
from marli.interact.types import SYSTEM_GRADE_KEY
from marli.rundir import RunDir
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks
from marli.verbs import input_manifest


@dataclass
class FilterConfig:
    tasks: str | None = input_field(None, help="TaskSet to filter")
    episodes: str | None = input_field(
        None, help="episodes dir of rollouts of that TaskSet (eval rollout output)"
    )
    lo: float = 0.0
    hi: float = 1.0
    inclusive: bool = False
    min_episodes: int = 2
    metric: str = "correct"

    def __post_init__(self) -> None:
        if not self.tasks or not self.episodes:
            raise ConfigError("tasks and episodes are required")
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ConfigError("lo and hi must be finite bounds with lo <= hi")
        if type(self.min_episodes) is not int or self.min_episodes < 1:
            raise ConfigError("min_episodes must be a positive integer")
        if not self.metric:
            raise ConfigError("metric must be nonempty")


async def filter(cfg: FilterConfig, run: RunDir) -> TaskSet:
    """Keep tasks by mean system grade over ok episodes, preserving their order.

    Drop reasons are exclusive: insufficient episodes, then the lower bound,
    then the upper bound. Strict bounds count equality as a rejection.
    """
    assert cfg.tasks is not None and cfg.episodes is not None
    taskset = TaskSet.load(cfg.tasks)
    tasks = read_tasks(taskset)
    task_ids = {task.task_id for task in tasks}
    manifest = input_manifest(cfg.episodes, "episodes")
    grades: dict[str, list[float]] = defaultdict(list)
    ignored_non_ok = 0
    for episode, _ in records.read_episodes(manifest.parent, with_tokens=False):
        if not episode.ok:
            ignored_non_ok += 1
            continue
        if episode.task_id not in task_ids:
            raise ConfigError(f"episode {episode.episode_id!r}: unknown task {episode.task_id!r}")
        try:
            value = episode.grades[SYSTEM_GRADE_KEY][cfg.metric]
        except KeyError as exc:
            raise ConfigError(
                f"episode {episode.episode_id!r}: missing system metric {cfg.metric!r}"
            ) from exc
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(
                f"episode {episode.episode_id!r}: metric {cfg.metric!r} must be finite and numeric"
            )
        grades[episode.task_id].append(value)
    counts = {
        "input": len(tasks),
        "kept": 0,
        "dropped_min_episodes": 0,
        "dropped_lo": 0,
        "dropped_hi": 0,
    }
    kept = []
    for task in tasks:
        values = grades[task.task_id]
        if len(values) < cfg.min_episodes:
            counts["dropped_min_episodes"] += 1
            continue
        pass_rate = fmean(values)
        if pass_rate < cfg.lo or (not cfg.inclusive and pass_rate == cfg.lo):
            counts["dropped_lo"] += 1
        elif pass_rate > cfg.hi or (not cfg.inclusive and pass_rate == cfg.hi):
            counts["dropped_hi"] += 1
        else:
            kept.append(replace(task, meta={**task.meta, "pass_rate": pass_rate}))
    counts["kept"] = len(kept)
    path = write_tasks(run.out, kept)
    return replace(
        taskset,
        root=run.out,
        tasks=path.name,
        config_hash=None,
        n=len(kept),
        inputs=(InputRef.of(taskset), InputRef.from_manifest(manifest)),
        meta={
            **taskset.meta,
            "filter": {
                "lo": cfg.lo,
                "hi": cfg.hi,
                "inclusive": cfg.inclusive,
                "min_episodes": cfg.min_episodes,
                "metric": cfg.metric,
                "counts": counts,
                "ignored_non_ok_episodes": ignored_non_ok,
            },
        },
    )
