"""Saved episodes can be rescored without changing their samples or compute records."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar

from marli.config import input_field
from marli.envs.registry import make_env
from marli.errors import ConfigError
from marli.eval.rollout import EpisodeSet
from marli.handles import Handle, InputRef, register_handle
from marli.interact.records import read_episodes
from marli.interact.types import Episode
from marli.rundir import RunDir
from marli.tasks.taskset import TaskSet, read_tasks


@dataclass
class ScoreConfig:
    episodes: str | None = input_field(None, help="EpisodeSet manifest or dir")
    regrade: bool = False


@register_handle
@dataclass(frozen=True)
class Scores(Handle):
    KIND: ClassVar[str] = "scores"
    MANIFEST: ClassVar[str] = "scores.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("rows",)

    rows: str
    n: int

    def summary(self) -> dict[str, Any]:
        return {"n": self.n}


async def score(cfg: ScoreConfig, run: RunDir) -> Scores:
    if cfg.episodes is None:
        raise ConfigError("score requires episodes")
    source = EpisodeSet.load(cfg.episodes)
    taskset = TaskSet.load(source.taskset)
    tasks = {task.task_id: task for task in read_tasks(taskset)}
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for episode, _ in read_episodes(source.root, with_tokens=False):
        by_task[episode.task_id].append(episode)
    done = run.done_keys("scores.jsonl", "episode_id")
    run.path("scores.jsonl").touch(exist_ok=True)
    policy = json.dumps(
        {role: source.policies[name] for role, name in source.meta.get("seating", {}).items()},
        sort_keys=True,
    )
    harness = json.dumps(
        {"taskset": taskset.sha256(), "env": source.env, "env_config": source.env_config},
        sort_keys=True,
    )
    for task_id, episodes in sorted(by_task.items()):
        episodes.sort(key=lambda episode: episode.episode_idx)
        if all(episode.episode_id in done for episode in episodes):
            continue
        env = make_env(source.env, source.env_config, tasks[task_id])
        representatives: list[str] = []
        needs_env = cfg.regrade or len(episodes) > 1
        try:
            if needs_env:
                await env.setup()
            for episode in episodes:
                answer = episode.outcome.final_answer
                answer_group = None
                if answer is not None and answer.strip():
                    for index, representative in enumerate(representatives):
                        if await env.same_answer(answer, representative):
                            answer_group = index
                            break
                    else:
                        answer_group = len(representatives)
                        representatives.append(answer)
                if episode.episode_id in done:
                    continue
                grades = episode.grades
                if cfg.regrade:
                    grades = {
                        agent: await env.grade(submission)
                        for agent, submission in episode.outcome.submissions.items()
                    }
                    grades["_system"] = await env.grade(answer)
                own = {
                    agent: grades.get(agent, {}).get("correct", 0.0)
                    for agent in sorted(
                        {a.agent_id for a in episode.agents}
                        | episode.outcome.submissions.keys()
                        | (grades.keys() - {"_system"})
                    )
                }
                run.append_row(
                    "scores.jsonl",
                    {
                        **episode.metrics,
                        "task_id": task_id,
                        "episode_id": episode.episode_id,
                        "episode_idx": episode.episode_idx,
                        "group_id": episode.group_id,
                        "protocol": episode.protocol,
                        "policy": policy,
                        "taskset": taskset.sha256(),
                        "harness": harness,
                        "correct": grades.get("_system", {}).get("correct", 0.0),
                        "answered": answer is not None and bool(answer.strip()),
                        "oracle_any": any(value == 1 for value in own.values()),
                        "own_correct": own,
                        "answer_group": answer_group,
                        "ok": episode.ok,
                    },
                )
                done.add(episode.episode_id)
        finally:
            if needs_env:
                await env.teardown()
    return Scores(root=run.out, inputs=(InputRef.of(source),), rows="scores.jsonl", n=len(done))
