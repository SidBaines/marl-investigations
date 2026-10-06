"""eval continue: a budget-cut episode continued under larger limits equals a fresh run there."""

from __future__ import annotations

import itertools
import json
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from _marli_code_fixtures import SUM_SOLUTION, code_task
from test_envs_sandbox import sandbox_host  # noqa: F401

from marli.cli.main import main
from marli.data.filter import FilterConfig
from marli.envs.base import Env, Task
from marli.envs.registry import ENVS
from marli.errors import ConfigError
from marli.eval.continuation import ContinueConfig
from marli.eval.policies import PolicySpec
from marli.eval.rollout import RolloutConfig
from marli.interact.limits import AgentLimits, CallLimits, EpisodeLimits, Limits
from marli.interact.records import read_episodes
from marli.interact.replay import normalize_result
from marli.interact.tools import Tool, ToolCtx, ToolResult, validate_tool_spec
from marli.interact.types import Episode
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.base import ToolSpec
from marli.render.fake import FakeRenderer
from marli.rundir import RunStatus
from marli.tasks.taskset import TaskSet, write_tasks
from marli.verbs import run_verb

# ---------------------------------------------------------------- a stateful single-agent env

_STAMPS = itertools.count()


class _Tool:
    shared = False
    control = False

    def __init__(self, env: FilesEnv, name: str, properties: dict[str, Any]) -> None:
        self.env = env
        self.spec = ToolSpec(
            name,
            f"{name} a file",
            {"type": "object", "properties": properties, "additionalProperties": False},
        )
        validate_tool_spec(self.spec)


class WriteTool(_Tool):
    def __init__(self, env: FilesEnv) -> None:
        super().__init__(env, "write", {"path": {"type": "string"}, "text": {"type": "string"}})

    async def __call__(self, ctx: ToolCtx, *, path: str, text: str) -> ToolResult:
        self.env.files[path] = text
        # A volatile field, as in real bash results: replays must not count it.
        return ToolResult(json.dumps({"ok": True, "duration_s": random.random()}))


class CatTool(_Tool):
    def __init__(self, env: FilesEnv) -> None:
        super().__init__(env, "cat", {"path": {"type": "string"}})

    async def __call__(self, ctx: ToolCtx, *, path: str) -> ToolResult:
        return ToolResult(self.env.files.get(path, "<missing>"))


class StampTool(_Tool):
    def __init__(self, env: FilesEnv) -> None:
        super().__init__(env, "stamp", {})

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        # Differs on every run: a real replay mismatch.
        return ToolResult(f"stamp {next(_STAMPS)}")


class FilesEnv(Env):
    name = "cont_files"
    supports_vote = False

    def __init__(self, task: Task) -> None:
        self.task = task
        self.files: dict[str, str] = {}

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def task_message(self, role: str) -> str:
        return self.task.prompt

    def tools(self, role: str) -> list[Tool]:
        return [WriteTool(self), CatTool(self), StampTool(self)]

    async def grade(self, submission: str | None) -> dict[str, float]:
        # Correct only if the file written before the cut exists in *this* run.
        return {"correct": float(submission is not None and submission == self.files.get("a.txt"))}

    def canonical(self, submission: str | None) -> str | None:
        return submission


@ENVS.register("cont_files")
def files_factory(config: dict[str, Any], task: Task) -> FilesEnv:
    return FilesEnv(task)


LONG = "think " * 60
TURNS = (
    Turn(thinking="plan", tool_calls=(("write", {"path": "a.txt", "text": "hello"}),)),
    Turn(tool_calls=(("stamp", {}), ("cat", {"path": "a.txt"}))),
    Turn(thinking=LONG, tool_calls=(("submit", {"answer": "hello"}),)),
)


def _turn(ctx: ScriptCtx) -> Turn:
    if "Just submit" in ctx.prompt_text:
        return Turn(tool_calls=(("submit", {"answer": "nothing"}),))
    return TURNS[min(ctx.meta.call_index, len(TURNS) - 1)]


def solver_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy("test", renderer, from_callable(_turn, renderer))


def _lengths() -> list[int]:
    renderer = FakeRenderer()
    return [
        len(renderer.encode_completion(t.content, thinking=t.thinking, tool_calls=t.tool_calls))
        for t in TURNS
    ]


def _limits(agent_tokens: int) -> Limits:
    return Limits(
        call=CallLimits(max_tokens=4096),
        agent=AgentLimits(max_gen_tokens=agent_tokens, final_reserve=0),
        episode=EpisodeLimits(max_gen_tokens=100_000),
        on_exhaust="none",
    )


def _taskset(root: Path) -> TaskSet:
    tasks = [
        Task("t0", "Write hello to a.txt, then submit its content.", answer="hello"),
        Task("t1", "Just submit.", answer="nothing"),
    ]
    write_tasks(root, tasks)
    handle = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="test",
        kind="math",
        n=len(tasks),
        answer_format="latex",
        commit_text=True,
    )
    handle.save()
    return handle


def _rollout(taskset: TaskSet, agent_tokens: int) -> RolloutConfig:
    return RolloutConfig(
        tasks=str(taskset.manifest_path),
        env="cont_files",
        protocol="single",
        protocol_config={"env_tools": ["write", "cat", "stamp"]},
        policies={"test": PolicySpec("scripted:test_eval_continue:solver_policy", renderer="fake")},
        seating={"solver": "test"},
        limits=_limits(agent_tokens),
    )


def _episodes(root: Path) -> dict[str, Episode]:
    return {episode.task_id: episode for episode, _ in read_episodes(root, with_tokens=False)}


def _provenance(root: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "continuations.jsonl").read_text().splitlines()]
    return {row["episode_id"]: row for row in rows}


@pytest.fixture
def source(tmp_path: Path) -> tuple[TaskSet, Path, int]:
    taskset = _taskset(tmp_path / "tasks")
    small = sum(_lengths()[:2]) + 40  # the third call is cut at 40 tokens
    out = tmp_path / "source"
    asyncio_run(run_verb("eval rollout", _rollout(taskset, small), out=out))
    return taskset, out, small


def asyncio_run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def test_single_agent_continuation_equals_a_fresh_run_at_the_larger_budget(
    tmp_path: Path, source: tuple[TaskSet, Path, int]
) -> None:
    taskset, src, small = source
    original = _episodes(src)["t0"]
    assert original.grades["_system"]["correct"] == 0  # cut before submitting
    assert [c.completion_ids for c in original.calls][2] and len(original.calls) == 3
    assert len(original.calls[2].completion_ids) == 40

    result = asyncio_run(
        run_verb(
            "eval continue",
            ContinueConfig(episodes=str(src), limits={"agent": {"max_gen_tokens": 5000}}),
            out=tmp_path / "continued",
        )
    )
    fresh_out = tmp_path / "fresh"
    asyncio_run(run_verb("eval rollout", _rollout(taskset, 5000), out=fresh_out))
    continued, fresh = _episodes(tmp_path / "continued")["t0"], _episodes(fresh_out)["t0"]

    assert continued.episode_id == original.episode_id and continued.ok
    # Same trajectory and outcome as running with the larger budget from the start.
    assert [c.completion_ids for c in continued.calls] == [c.completion_ids for c in fresh.calls]
    assert [
        [(t.name, t.arguments, normalize_result(t.result)) for t in c.tool_calls]
        for c in continued.calls
    ][:1] == [
        [(t.name, t.arguments, normalize_result(t.result)) for t in c.tool_calls]
        for c in fresh.calls
    ][:1]
    assert continued.grades == fresh.grades and continued.grades["_system"]["correct"] == 1
    # The recorded prefix is byte-identical (timing aside), including tool results.
    assert continued.calls[:2] == original.calls[:2]
    cut, recorded = continued.calls[2], original.calls[2]
    assert cut.completion_ids[:40] == recorded.completion_ids
    assert len(cut.completion_ids) > 40 and len(cut.logprobs) == len(cut.completion_ids)
    assert cut.logprobs[:40] == recorded.logprobs and cut.seed != recorded.seed

    rows = _provenance(tmp_path / "continued")
    row = rows[original.episode_id]
    assert row["action"] == "continued" and row["source_episode_id"] == original.episode_id
    assert row["cut"] == {
        "agent_id": "solver0",
        "call_index": 2,
        "extended": True,
        "reason": "budget (agent.max_gen_tokens)",
    }
    assert row["replayed_calls"] == 3 and row["tool_calls_replayed"] == 3
    assert row["tool_mismatches"] == 1  # the stamp; the write's duration is volatile
    assert rows[_episodes(src)["t1"].episode_id]["action"] == "copied"
    counts = result.handle.meta["continuation"]["counts"]
    assert counts["continued"] == counts["copied"] == counts["extended"] == 1
    assert result.handle.inputs[0].kind == "episodes" and result.handle.inputs[1].kind == "taskset"


def test_untruncated_copies_filter_resume_and_cli(
    tmp_path: Path, source: tuple[TaskSet, Path, int], capsys: pytest.CaptureFixture[str]
) -> None:
    taskset, src, _ = source
    out = tmp_path / "continued"
    cfg = ContinueConfig(episodes=str(src), limits={"agent": {"max_gen_tokens": 5000}})
    asyncio_run(run_verb("eval continue", cfg, out=out))
    copied = _episodes(out)["t1"]
    assert copied == _episodes(src)["t1"]  # verbatim
    before = (out / "episodes.jsonl").read_bytes()
    assert asyncio_run(run_verb("eval continue", cfg, out=out)).status is RunStatus.COMPLETE
    (out / "episodes.json").unlink()  # incomplete again: resume must skip every done row
    assert asyncio_run(run_verb("eval continue", cfg, out=out)).status is RunStatus.RESUME
    assert (out / "episodes.jsonl").read_bytes() == before
    filtered = asyncio_run(
        run_verb(
            "data filter",
            FilterConfig(
                tasks=str(taskset.root),
                episodes=str(out),
                metric="correct",
                lo=0.0,
                hi=1.0,
                inclusive=True,
                min_episodes=1,
            ),
            out=tmp_path / "filtered",
        )
    )
    assert filtered.handle.n >= 1, filtered.handle.meta
    capsys.readouterr()
    code = main(
        [
            "eval",
            "continue",
            f"episodes={src}",
            "limits.agent.max_gen_tokens=5000",
            "--out",
            str(tmp_path / "cli"),
        ]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and len(lines) == 1 and json.loads(lines[0])["ok"]


@pytest.mark.parametrize("mode", ["fail", "live"])
def test_divergence_fails_or_goes_live(
    tmp_path: Path, source: tuple[TaskSet, Path, int], mode: str
) -> None:
    _, src, _ = source
    tampered = tmp_path / "tampered"
    shutil.copytree(src, tampered)
    rows = [json.loads(line) for line in (tampered / "episodes.jsonl").read_text().splitlines()]
    for row in rows:
        if row["task_id"] == "t0":
            row["calls"][1]["prompt_len"] += 1
    (tampered / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / f"continued-{mode}"
    asyncio_run(
        run_verb(
            "eval continue",
            ContinueConfig(
                episodes=str(tampered),
                limits={"agent": {"max_gen_tokens": 5000}},
                on_divergence=mode,
            ),
            out=out,
        )
    )
    episode = _episodes(out)["t0"]
    row = _provenance(out)[episode.episode_id]
    assert "prompt length" in row["diverged"]
    if mode == "fail":
        assert not episode.ok and row["action"] == "diverged"
        assert any("prompt length" in error for error in episode.errors)
    else:
        assert episode.ok and row["action"] == "continued"
        assert episode.grades["_system"]["correct"] == 1


def test_limits_that_change_a_system_prompt_are_rejected(tmp_path: Path) -> None:
    taskset = _taskset(tmp_path / "tasks")
    cfg = _rollout(taskset, 10_000)
    cfg = replace(
        cfg,
        max_tasks=1,
        protocol_config={
            **cfg.protocol_config,
            "system_prompt": "Solve it. Workers get {worker_tokens} tokens.",
        },
    )
    src = tmp_path / "source"
    asyncio_run(run_verb("eval rollout", cfg, out=src))
    with pytest.raises(ConfigError, match="system prompt"):
        asyncio_run(
            run_verb(
                "eval continue",
                ContinueConfig(episodes=str(src), limits={"worker": {"max_gen_tokens": 999}}),
                out=tmp_path / "bad",
            )
        )


def test_a_larger_call_cap_that_changes_an_earlier_cut_diverges(tmp_path: Path) -> None:
    """Call 1 was cut by the call cap (not the budget): raising the cap breaks equivalence."""
    taskset = _taskset(tmp_path / "tasks")
    lengths = _lengths()
    cfg = _rollout(taskset, 10_000)
    cfg = replace(cfg, limits=replace(cfg.limits, call=CallLimits(max_tokens=lengths[2] - 20)))
    src = tmp_path / "source"
    asyncio_run(run_verb("eval rollout", replace(cfg, max_tasks=1), out=src))
    out = tmp_path / "continued"
    asyncio_run(
        run_verb(
            "eval continue",
            ContinueConfig(episodes=str(src), select="all", limits={"call": {"max_tokens": 4096}}),
            out=out,
        )
    )
    episode = _episodes(out)["t0"]
    assert (
        not episode.ok and "not be equivalent" in _provenance(out)[episode.episode_id]["diverged"]
    )


# ---------------------------------------------------------------- relay on code_rules


def _solution(slot: int) -> str:
    return f"cat > tasks/task_{slot}/solution.py <<'EOF'\n{SUM_SOLUTION}EOF"


RELAY_TURNS = {
    "contrib0": (
        Turn(tool_calls=(("bash", {"command": _solution(1)}),)),
        Turn(tool_calls=(("ci_submit", {}),)),
        Turn(tool_calls=(("end_session", {}),)),
    ),
    "contrib1": (
        Turn(tool_calls=(("bash", {"command": _solution(2)}),)),
        Turn(tool_calls=(("ci_review", {}),)),
        Turn(thinking="y " * 150, tool_calls=(("ci_submit", {}),)),
        Turn(tool_calls=(("end_session", {}),)),
    ),
    "contrib2": (
        Turn(tool_calls=(("bash", {"command": "cat NOTES.md"}),)),
        Turn(tool_calls=(("ci_submit", {}),)),
        Turn(tool_calls=(("end_session", {}),)),
    ),
}


def relay_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        turns = RELAY_TURNS[ctx.meta.agent_id]
        return turns[min(ctx.meta.call_index, len(turns) - 1)]

    return ScriptedPolicy("test", renderer, from_callable(turn, renderer))


def _relay_budget() -> int:
    renderer = FakeRenderer()
    lengths = [
        len(renderer.encode_completion(t.content, thinking=t.thinking, tool_calls=t.tool_calls))
        for t in RELAY_TURNS["contrib1"][:2]
    ]
    return sum(lengths) + 50


@pytest.mark.usefixtures("sandbox_host")
def test_relay_continuation_replays_earlier_agents_and_rebuilds_ci_state(tmp_path: Path) -> None:
    problems = [asdict(replace(code_task(), task_id=f"synthetic/{i}")) for i in range(3)]
    tasks = [
        Task(
            "repo-00000",
            "Synthetic shared repository.",
            {
                "kind": "code_repo",
                "problems": problems,
                "has_rule": True,
                "rule_families": ["header", "constant", "docstring"],
            },
        )
    ]
    root = tmp_path / "repos"
    write_tasks(root, tasks)
    taskset = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="test",
        kind="code",
        n=1,
        answer_format="code_repo",
        commit_text=True,
    )
    taskset.save()
    budget = _relay_budget()
    cfg = RolloutConfig(
        tasks=str(taskset.manifest_path),
        env="code_rules",
        protocol="relay",
        protocol_config={"n_agents": 3, "env_tools": ["bash", "ci_submit", "ci_review"]},
        policies={"test": PolicySpec("scripted:test_eval_continue:relay_policy", renderer="fake")},
        seating={"contributor": "test"},
        limits=Limits(
            call=CallLimits(max_tokens=4096),
            agent=AgentLimits(max_gen_tokens=budget, final_reserve=0),
            episode=EpisodeLimits(max_gen_tokens=3 * budget, max_ticks=200),
            on_exhaust="none",
        ),
    )
    src = tmp_path / "source"
    asyncio_run(run_verb("eval rollout", cfg, out=src))
    original = _episodes(src)["repo-00000"]
    assert original.ok
    out = tmp_path / "continued"
    asyncio_run(
        run_verb(
            "eval continue",
            ContinueConfig(
                episodes=str(src),
                limits={"agent": {"max_gen_tokens": 5000}, "episode": {"max_gen_tokens": 15000}},
            ),
            out=out,
        )
    )
    continued = _episodes(out)["repo-00000"]
    row = _provenance(out)[continued.episode_id]
    assert continued.ok and row["action"] == "continued"
    assert row["cut"]["agent_id"] == "contrib1" and row["cut"]["call_index"] == 2
    assert row["cut"]["extended"] and row["tool_mismatches"] == 0  # same rule, same CI reports

    def calls(episode: Episode, agent: str) -> list[Any]:
        return [c for c in episode.calls if c.agent_id == agent]

    assert calls(continued, "contrib0") == calls(original, "contrib0")  # replayed exactly
    assert calls(continued, "contrib1")[:2] == calls(original, "contrib1")[:2]
    after_cut = calls(continued, "contrib1")[2]
    assert after_cut.completion_ids[: len(calls(original, "contrib1")[2].completion_ids)] == (
        calls(original, "contrib1")[2].completion_ids
    )
    # ci_review before the cut was rebuilt: the continued ci_submit is refused.
    assert "already used your CI run" in after_cut.tool_calls[0].result
    later = calls(continued, "contrib2")
    assert later and all(
        c.seed != o.seed for c, o in zip(later, calls(original, "contrib2"), strict=False)
    )
