"""A tiny slotted workspace exercises relay without the parallel code_rules build."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

import pytest
from _marli_test_envs import ArithEnv

from marli.envs.base import Env, Task
from marli.errors import ConfigError
from marli.interact.limits import AgentLimits, CallLimits, EpisodeLimits, Limits, SessionLimits
from marli.interact.protocols.relay import RelayConfig, RelayProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.system import AgentHandle, EpisodeSystem
from marli.interact.tools import Tool, ToolCtx, ToolResult
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.base import ToolSpec
from marli.render.fake import FakeRenderer
from marli.train.datums import build_datums
from marli.train.types import SegmentCredit


class SlottedEnv(Env):
    name = "relay_slots"
    supports_vote = False

    def __init__(self, n_slots: int = 4) -> None:
        self.n_slots = n_slots
        self.task = Task("relay", "Shared repository")
        self.files: dict[str, str] = {}
        self.bindings: dict[str, int] = {}
        self.submissions: dict[str, str] = {}
        self.trace: list[tuple[str, str]] = []
        self.setup_count = 0
        self.teardown_count = 0

    @property
    def sandbox(self) -> dict[str, str]:
        return self.files

    async def setup(self) -> None:
        self.setup_count += 1

    async def teardown(self) -> None:
        self.teardown_count += 1

    def task_message(self, role: str) -> str:
        raise AssertionError("a slotted relay must use slot_message")

    def tools(self, role: str) -> list[Tool]:
        return [FileTool(self), CITool(self, "ci_submit"), CITool(self, "ci_review")]

    def bind_agent(self, agent_id: str, slot: int) -> None:
        assert agent_id not in self.bindings
        self.bindings[agent_id] = slot
        self.trace.append(("bind", agent_id))

    def slot_message(self, slot: int) -> str:
        assert self.bindings[f"contrib{slot}"] == slot
        return f"Slot {slot}: complete your contribution."

    def slot_submission(self, agent_id: str) -> str | None:
        return self.submissions.get(agent_id)

    def bundle(self, submissions: dict[str, str | None]) -> str:
        return json.dumps({"submissions": submissions})

    async def grade(self, submission: str | None) -> dict[str, float]:
        if submission is None:
            return {"score": 0.0, "probed": 0.0, "notes_had_rule": 0.0}
        payload = json.loads(submission)
        if "submissions" in payload:
            grades = [await self.grade(value) for value in payload["submissions"].values()]
            return {key: fmean(grade[key] for grade in grades) for key in grades[0]}
        return payload

    def canonical(self, submission: str | None) -> str | None:
        return None


@dataclass(frozen=True)
class FileTool:
    env: SlottedEnv
    shared = True
    blocking = False
    control = False
    spec = ToolSpec(
        "bash",
        "Read the shared file, or replace it with content.",
        {"type": "object", "properties": {"content": {"type": "string"}}},
    )

    async def __call__(self, ctx: ToolCtx, *, content: str | None = None) -> ToolResult:
        assert ctx.sandbox is self.env.files
        assert ctx.agent_id in self.env.bindings
        self.env.trace.append(("write" if content is not None else "read", ctx.agent_id))
        if content is not None:
            self.env.files["file"] = content
        return ToolResult(self.env.files.get("file", "empty"))


@dataclass(frozen=True)
class CITool:
    env: SlottedEnv
    name: str
    shared = True
    blocking = False
    control = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            self.name,
            "Record this contributor's CI result without ending the contributor.",
            {"type": "object", "properties": {"score": {"type": "number"}}},
        )

    async def __call__(self, ctx: ToolCtx, *, score: float = 0.0) -> ToolResult:
        assert ctx.agent_id in self.env.bindings
        assert ctx.agent_id not in self.env.submissions
        probed = self.name == "ci_review"
        self.env.submissions[ctx.agent_id] = json.dumps(
            {
                "score": 0.0 if probed else score,
                "probed": float(probed),
                "notes_had_rule": float(bool(self.env.files.get("file"))),
            }
        )
        self.env.trace.append((self.name, ctx.agent_id))
        return ToolResult("CI recorded")


def relay_spec(
    env: Env,
    turn: Callable[[ScriptCtx], Turn],
    *,
    n_agents: int = 2,
    schedule: str = "lockstep",
) -> EpisodeSpec:
    renderer = FakeRenderer()
    policy = ScriptedPolicy("relay", renderer, from_callable(turn, renderer), trainable=True)
    protocol = RelayProtocol(
        RelayConfig(
            n_agents=n_agents, env_tools=tuple(tool.spec.name for tool in env.tools("contributor"))
        )
    )
    return EpisodeSpec(
        protocol,
        env,
        Task("relay", "Shared task"),
        {"contributor": "relay"},
        {"relay": policy},
        {"relay": FakeRenderer},
        Limits(),
        schedule=schedule,
    )


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_sequential_shared_workspace_fresh_context_and_exact_datums(
    schedule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = SlottedEnv(2)
    original = EpisodeSystem.start_agent

    async def start(
        io: EpisodeSystem,
        role: str,
        *,
        agent_id: str,
        seat_key: tuple[Any, ...],
        first_message: str,
        parent: str | None = None,
    ) -> AgentHandle:
        assert all(handle.task.done() for handle in io.handles)
        assert env.bindings[agent_id] == seat_key[1]
        assert env.trace[-1] == ("bind", agent_id)
        env.trace.append(("start", agent_id))
        return await original(
            io,
            role,
            agent_id=agent_id,
            seat_key=seat_key,
            first_message=first_message,
            parent=parent,
        )

    monkeypatch.setattr(EpisodeSystem, "start_agent", start)

    def turn(ctx: ScriptCtx) -> Turn:
        slot = env.bindings[ctx.meta.agent_id]
        index = ctx.meta.call_index
        if index == 0:
            assert f"Slot {slot}:" in ctx.prompt_text
            assert "PRIVATE-THOUGHT" not in ctx.prompt_text
            assert "shared clue" not in ctx.prompt_text
            return Turn(
                thinking="PRIVATE-THOUGHT" if slot == 0 else None,
                tool_calls=(("bash", {"content": "shared clue"} if slot == 0 else {}),),
            )
        if index == 1:
            assert "shared clue" in ctx.prompt_text
            return Turn(tool_calls=(("ci_review" if slot == 0 else "ci_submit", {"score": 3}),))
        assert index == 2
        return Turn(tool_calls=(("end_session", {}),))

    spec = relay_spec(env, turn, schedule=schedule)
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.outcome.aggregation == "relay"
    assert env.setup_count == env.teardown_count == 1
    assert env.bindings == {"contrib0": 0, "contrib1": 1}
    assert env.files == {"file": "shared clue"}
    assert env.trace == [
        ("bind", "contrib0"),
        ("start", "contrib0"),
        ("write", "contrib0"),
        ("ci_review", "contrib0"),
        ("bind", "contrib1"),
        ("start", "contrib1"),
        ("read", "contrib1"),
        ("ci_submit", "contrib1"),
    ]
    assert [agent.seat_key for agent in episode.agents] == [("contributor", 0), ("contributor", 1)]
    assert [call.agent_id for call in episode.calls] == ["contrib0"] * 3 + ["contrib1"] * 3
    assert episode.outcome.submissions == env.submissions
    assert json.loads(episode.outcome.final_answer)["submissions"] == env.submissions
    assert episode.grades == {
        "contrib0": {"score": 0.0, "probed": 1.0, "notes_had_rule": 1.0},
        "contrib1": {"score": 3.0, "probed": 0.0, "notes_had_rule": 1.0},
        "_system": {"score": 1.5, "probed": 0.5, "notes_had_rule": 1.0},
    }
    assert len(episode.segments) == 2
    assert all(segment.session_idx == 0 for segment in episode.segments)
    credits = [
        SegmentCredit(
            episode.episode_id,
            segment.agent_id,
            "contributor",
            segment.segment_id,
            "shared",
            0.5,
            episode.grades[segment.agent_id]["score"],
            1.0,
        )
        for segment in episode.segments
    ]
    datums = build_datums(episode, buffers, credits)
    assert [datum.agent_id for datum in datums] == ["contrib0", "contrib1"]
    for datum in datums:
        assert datum.tokens == tuple(buffers[datum.segment_id])
        calls = [call for call in episode.calls if call.agent_id == datum.agent_id]
        assert datum.n_action_tokens == sum(len(call.completion_ids) for call in calls)
        sampled_calls = [
            sampled
            for sampled in spec.policies["relay"].calls
            if sampled.meta.agent_id == datum.agent_id
        ]
        for call, sampled in zip(calls, sampled_calls, strict=True):
            assert datum.tokens[: call.prompt_len] == sampled.prompt_ids
            start, end = call.prompt_len, call.prompt_len + len(call.completion_ids)
            assert datum.tokens[start:end] == call.completion_ids
            assert datum.logprobs[start - 1 : end - 1] == call.logprobs
            assert datum.advantages[start - 1 : end - 1] == (0.5,) * len(call.completion_ids)
        assert all(
            logprob == advantage == 0.0
            for mask, logprob, advantage in zip(
                datum.mask, datum.logprobs, datum.advantages, strict=True
            )
            if not mask
        )


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
@pytest.mark.parametrize("ending", ["budget", "end_session", "submit"])
async def test_missing_ci_is_none_and_zero_even_with_builtin_submit(
    schedule: str, ending: str
) -> None:
    env = SlottedEnv(2)

    def turn(ctx: ScriptCtx) -> Turn:
        if ending == "budget":
            return Turn(content="x" * 1000)
        return Turn(
            tool_calls=((ending, {"answer": "forged score"} if ending == "submit" else {}),)
        )

    spec = relay_spec(env, turn, schedule=schedule)
    spec.limits = Limits(
        call=CallLimits(max_tokens=128),
        agent=AgentLimits(max_gen_tokens=128, final_reserve=32),
        episode=EpisodeLimits(max_gen_tokens=256),
        session=SessionLimits(max_sessions=3),
    )
    episode, _ = await run_episode(spec)
    assert episode.ok
    assert episode.outcome.submissions == {"contrib0": None, "contrib1": None}
    assert set(episode.grades) == {"contrib0", "contrib1", "_system"}
    assert all(
        grade == {"score": 0, "probed": 0, "notes_had_rule": 0} for grade in episode.grades.values()
    )
    assert len(episode.calls) == 2 and not any(call.forced for call in episode.calls)
    if ending == "budget":
        assert all(len(call.completion_ids) == 128 for call in episode.calls)
        assert all("agent.max_gen_tokens" in episode.limits_hit[f"contrib{k}"] for k in range(2))


async def test_slot_count_mismatch_fails_before_binding_or_sampling() -> None:
    env = SlottedEnv(3)

    def turn(ctx: ScriptCtx) -> Turn:
        raise AssertionError("must not sample")

    with pytest.raises(ConfigError, match="n_slots.*n_agents"):
        await run_episode(relay_spec(env, turn))
    assert env.bindings == {} and env.teardown_count == 1


@pytest.mark.parametrize("answers", [("5", "4", None), (None, None, None), ("5", "4", "")])
@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_non_slotted_submissions_end_contributors_and_last_non_none_wins(
    answers: tuple[str | None, ...], schedule: str
) -> None:
    env = ArithEnv()

    def turn(ctx: ScriptCtx) -> Turn:
        assert env.task.prompt in ctx.prompt_text
        answer = answers[int(ctx.meta.agent_id[-1])]
        return Turn(
            tool_calls=(("end_session", {}) if answer is None else ("submit", {"answer": answer}),)
        )

    episode, _ = await run_episode(relay_spec(env, turn, n_agents=3, schedule=schedule))
    assert episode.ok and len(episode.calls) == 3
    assert episode.outcome.submissions == dict(
        zip((f"contrib{k}" for k in range(3)), answers, strict=True)
    )
    final = next((answer for answer in reversed(answers) if answer is not None), None)
    assert episode.outcome.final_answer == final
    assert episode.grades == {
        **{f"contrib{k}": {"correct": float(answer == "5")} for k, answer in enumerate(answers)},
        "_system": {"correct": float(final == "5")},
    }


def test_limits_preserve_full_agent_budget_and_require_enough_episode_tokens() -> None:
    protocol = RelayProtocol()
    limits = Limits(session=SessionLimits(max_sessions=3))
    adjusted = protocol.adjust_limits(limits)
    assert adjusted.agent == replace(limits.agent, final_reserve=0)
    assert adjusted.episode == limits.episode
    assert adjusted.session.max_sessions == 1 and adjusted.on_exhaust == "none"
    assert limits.agent.final_reserve == 512 and limits.session.max_sessions == 3
    with pytest.raises(ConfigError, match="episode.max_gen_tokens.*n_agents"):
        protocol.adjust_limits(
            replace(limits, episode=replace(limits.episode, max_gen_tokens=131071))
        )


@pytest.mark.parametrize("n_agents", [0, -1, True, 1.5])
def test_invalid_contributor_count(n_agents: int) -> None:
    with pytest.raises(ConfigError, match="n_agents"):
        RelayConfig(n_agents=n_agents)
