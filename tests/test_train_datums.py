"""Saved runtime traces must retain exact actions through training alignment."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _marli_test_envs import ScratchEnv

from marli.interact.limits import AgentLimits, CallLimits, Limits, SessionLimits
from marli.interact.records import read_episodes, write_episode
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.system import ContextSpec, Protocol, RoleSpec, SystemIO
from marli.interact.types import Call, Episode, Outcome, Purpose, ReadVia, SegmentStart
from marli.interact.workspace import DeliverySpec
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, turns_by_agent
from marli.render.base import Msg
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer
from marli.train.datums import build_datums, datum_stats
from marli.train.types import SegmentCredit, TrainDatum


class DatumProtocol(Protocol):
    name = "datum_test"

    def __init__(self, *, count: int = 1, context: ContextSpec | None = None) -> None:
        self.role = RoleSpec(
            "solver",
            ("submit", "write_scratchpad", "end_session"),
            "Solve {agent_id} of {n_agents}.",
            count,
            context or ContextSpec(),
        )

    def roles(self) -> list[RoleSpec]:
        return [self.role]

    async def run(self, io: SystemIO) -> Outcome:
        handles = [
            await io.start_agent(
                "solver",
                agent_id=f"solver{i}",
                seat_key=("solver", i),
                first_message=io.task.prompt,
            )
            for i in range(self.role.count or 0)
        ]
        results = await io.wait(handles)
        answers = {result.agent_id: result.submission for result in results}
        return Outcome(results[-1].submission, answers, "test")


def episode_spec(
    turns: dict[str, list[Turn]],
    *,
    context: ContextSpec | None = None,
    limits: Limits | None = None,
) -> EpisodeSpec:
    renderer = FakeRenderer()
    policy = ScriptedPolicy(
        "script",
        renderer,
        turns_by_agent(renderer, turns),
        trainable=True,
        policy_version=7,
    )
    env = ScratchEnv()
    return EpisodeSpec(
        DatumProtocol(count=len(turns), context=context),
        env,
        env.task,
        {"solver": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        limits or Limits(),
        run_seed=17,
    )


def round_trip(
    path: Path,
    episode: Episode,
    buffers: Mapping[str, Sequence[int]],
) -> tuple[Episode, dict[str, list[int]]]:
    def append_row(name: str, row: Mapping[str, Any]) -> None:
        with (path / name).open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    write_episode(append_row, episode, buffers, record_tokens=True)
    [(saved, saved_buffers)] = read_episodes(path, with_tokens=True)
    assert saved == episode
    assert saved_buffers == buffers
    return saved, saved_buffers


def credit_for(call: Call, *, learner: str = "shared", advantage: float = 0.75) -> SegmentCredit:
    return SegmentCredit(
        call.episode_id,
        call.agent_id,
        call.role,
        call.segment_id,
        learner,
        advantage=advantage,
        reward=1.0,
        weight=0.5,
    )


def assert_exact_datum(
    datum: TrainDatum,
    episode: Episode,
    buffer: Sequence[int],
    credit: SegmentCredit,
) -> None:
    calls = sorted(
        (
            call
            for call in episode.calls
            if call.agent_id == credit.agent_id
            and call.segment_id == credit.segment_id
            and call.completion_ids
        ),
        key=lambda call: call.seq,
    )
    expected: dict[int, tuple[int, float]] = {}
    for call in calls:
        assert call.logprobs is not None
        for offset, (token, logprob) in enumerate(
            zip(call.completion_ids, call.logprobs, strict=True)
        ):
            position = call.prompt_len + offset - 1
            assert position not in expected
            expected[position] = (token, logprob)
    end = max(call.prompt_len + len(call.completion_ids) for call in calls)
    assert datum.tokens == tuple(buffer[:end])
    assert len(datum.mask) == len(datum.logprobs) == len(datum.advantages) == end - 1
    assert {p for p, value in enumerate(datum.mask) if value} == set(expected)
    assert sum(datum.mask) == sum(len(call.completion_ids) for call in calls)
    for position, token in enumerate(datum.tokens[1:]):
        if position in expected:
            assert datum.mask[position] == 1.0
            assert (token, datum.logprobs[position]) == expected[position]
            assert datum.advantages[position] == credit.advantage
        else:
            assert datum.mask[position] == 0.0
            assert datum.logprobs[position] == datum.advantages[position] == 0.0
    assert datum.learner == credit.learner
    assert datum.episode_id == credit.episode_id
    assert datum.agent_id == credit.agent_id
    assert datum.role == credit.role
    assert datum.segment_id == credit.segment_id
    assert datum.session_idx == calls[0].session_idx
    assert datum.policy_version == 7
    assert datum.meta == {
        "call_ids": [call.call_id for call in calls],
        "purposes": [call.purpose.value for call in calls],
        "n_action": len(expected),
        "forced_calls": sum(call.forced for call in calls),
    }


@pytest.fixture
async def tool_episode(tmp_path: Path) -> tuple[Episode, dict[str, list[int]]]:
    spec = episode_spec(
        {
            "solver0": [
                Turn(tool_calls=(("write_scratchpad", {"content": "first"}),), stop=False),
                Turn(thinking="refine", tool_calls=(("write_scratchpad", {"content": "second"}),)),
                Turn(tool_calls=(("submit", {"answer": "5"}),)),
            ]
        }
    )
    episode, buffers = await run_episode(spec)
    assert episode.ok and len(episode.calls) == 3 and len(episode.segments) == 1
    segment = episode.segments[0]
    buffers[segment.segment_id].extend(
        FakeRenderer().continuation("stop", [Msg("tool", "trailing observation")])
    )
    episode = replace(
        episode,
        segments=(replace(segment, n_tokens=len(buffers[segment.segment_id])),),
    )
    return round_trip(tmp_path, episode, buffers)


@pytest.mark.parametrize("advantage", [0.75, -1.5, 0.0])
def test_tool_loop_exact_ids_and_observations(
    tool_episode: tuple[Episode, dict[str, list[int]]],
    advantage: float,
) -> None:
    episode, buffers = tool_episode
    first = episode.calls[0]
    credit = credit_for(first, advantage=advantage)
    [datum] = build_datums(episode, buffers, [credit])
    buffer = buffers[first.segment_id]
    assert_exact_datum(datum, episode, buffer, credit)
    assert len(datum.tokens) < len(buffer)
    forced_close = first.prompt_len + len(first.completion_ids)
    assert buffer[forced_close] == S["eot"]
    assert datum.mask[forced_close - 1] == 0.0
    assert datum.logprobs[forced_close - 1] == datum.advantages[forced_close - 1] == 0.0


@pytest.mark.parametrize("purpose", [Purpose.COMPACT, Purpose.CARRY])
async def test_summary_trained_in_ending_segment_and_copied_as_observation(
    tmp_path: Path,
    purpose: Purpose,
) -> None:
    if purpose == Purpose.COMPACT:
        context = ContextSpec(kind="compaction", compact_threshold=1000, compact_reserve=512)
        limits = Limits(session=SessionLimits(carry_max_tokens=256))
        first = Turn(tool_calls=(("write_scratchpad", {"content": "x" * 600}),))
        start_reason = SegmentStart.COMPACTION
    else:
        context = ContextSpec(kind="compaction")
        limits = Limits(session=SessionLimits(2, 500, 300, 300))
        first = Turn(tool_calls=(("end_session", {}),))
        start_reason = SegmentStart.SESSION
    spec = episode_spec(
        {"solver0": [first, Turn("Carry five."), Turn(tool_calls=(("submit", {"answer": "5"}),))]},
        context=context,
        limits=limits,
    )
    episode, buffers = await run_episode(spec)
    assert episode.ok
    assert [call.purpose for call in episode.calls] == [Purpose.ACT, purpose, Purpose.ACT]
    assert [segment.start_reason for segment in episode.segments] == [
        SegmentStart.START,
        start_reason,
    ]
    episode, buffers = round_trip(tmp_path, episode, buffers)
    first_call, summary, last_call = episode.calls
    assert summary.segment_id == first_call.segment_id != last_call.segment_id
    credits = [credit_for(first_call, advantage=-0.25), credit_for(last_call, advantage=1.5)]
    datums = build_datums(episode, buffers, credits)
    assert len(datums) == 2
    for datum, credit in zip(datums, credits, strict=True):
        assert_exact_datum(datum, episode, buffers[datum.segment_id], credit)
    assert summary.call_id in datums[0].meta["call_ids"]
    assert "Carry five." in FakeRenderer().decode(datums[1].tokens[: last_call.prompt_len])
    assert not any(datums[1].mask[: last_call.prompt_len - 1])
    assert [datum.session_idx for datum in datums] == (
        [0, 0] if purpose == Purpose.COMPACT else [0, 1]
    )


async def test_forced_final_prefix_masked_and_completion_trained(tmp_path: Path) -> None:
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "act":
            return renderer.encode_completion("work")
        return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]

    spec = episode_spec(
        {"solver0": []},
        limits=Limits(agent=AgentLimits(max_calls=1), call=CallLimits(min_call_tokens=1)),
    )
    spec.policies["script"] = ScriptedPolicy(
        "script",
        renderer,
        script,
        trainable=True,
        policy_version=7,
    )
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5"
    episode, buffers = round_trip(tmp_path, episode, buffers)
    final = episode.calls[-1]
    assert final.purpose == Purpose.FINAL and final.forced
    credit = credit_for(final)
    [datum] = build_datums(episode, buffers, [credit])
    assert_exact_datum(datum, episode, buffers[final.segment_id], credit)
    prefix = tuple(renderer.forced_tool_prefix("submit"))
    start = final.prompt_len - len(prefix)
    assert datum.tokens[start : final.prompt_len] == prefix
    assert not any(datum.mask[start - 1 : final.prompt_len - 1])
    assert datum.mask[final.prompt_len - 1] == 1.0
    assert datum.meta["forced_calls"] == 1


@pytest.mark.parametrize("delivery", ["notify", "push"])
async def test_lockstep_agents_train_only_their_own_completions(
    tmp_path: Path,
    delivery: str,
) -> None:
    spec = episode_spec(
        {
            f"solver{i}": [
                Turn(tool_calls=(("write_scratchpad", {"content": f"peer note {i}"}),)),
                Turn(tool_calls=(("submit", {"answer": str(5 + i)}),)),
            ]
            for i in range(2)
        }
    )
    spec.delivery = DeliverySpec(mode=delivery)
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable
    assert len(episode.calls) == 4 and len(episode.segments) == 2
    episode, buffers = round_trip(tmp_path, episode, buffers)
    credits = [
        credit_for(call, advantage=0.5 if call.agent_id == "solver0" else -0.5)
        for call in episode.calls
        if call.tick == 0
    ]
    datums = build_datums(episode, buffers, credits)
    assert len(datums) == 2
    for datum, credit in zip(datums, credits, strict=True):
        assert_exact_datum(datum, episode, buffers[datum.segment_id], credit)
        calls = [call for call in episode.calls if call.agent_id == datum.agent_id]
        assert [call.tick for call in calls] == [0, 1]
        [read] = calls[-1].reads
        assert read.writer != datum.agent_id and read.via == ReadVia(delivery)
        start = calls[0].prompt_len + len(calls[0].completion_ids)
        stop = calls[1].prompt_len
        assert "[workspace]" in FakeRenderer().decode(datum.tokens[start:stop])
        assert not any(datum.mask[start - 1 : stop - 1])


def test_filters_other_episodes_and_preserves_credit_order(
    tool_episode: tuple[Episode, dict[str, list[int]]],
) -> None:
    episode, buffers = tool_episode
    credit = credit_for(episode.calls[0])
    other = replace(credit, episode_id="other", agent_id="missing", segment_id="")
    second = replace(credit, learner="second", advantage=-2.0)
    datums = build_datums(episode, buffers, [other, second, credit, credit])
    assert [datum.learner for datum in datums] == ["second", "shared", "shared"]
    for datum, expected_credit in zip(datums, [second, credit, credit], strict=True):
        assert_exact_datum(datum, episode, buffers[credit.segment_id], expected_credit)
    assert build_datums(episode, {}, [other]) == []
    assert build_datums(episode, {}, []) == []


def test_call_storage_order_text_and_sequence_container_do_not_affect_datums(
    tool_episode: tuple[Episode, dict[str, list[int]]],
) -> None:
    episode, buffers = tool_episode
    credit = credit_for(episode.calls[0])
    expected = build_datums(episode, buffers, [credit])
    changed = replace(
        episode,
        calls=tuple(replace(call, text="not the sampled text") for call in reversed(episode.calls)),
    )
    [datum] = build_datums(changed, {key: tuple(ids) for key, ids in buffers.items()}, [credit])
    assert [datum] == expected
    assert datum.meta == expected[0].meta


@pytest.mark.parametrize("corruption", ["flip", "short", "missing"])
def test_rejects_corrupt_buffers(
    tool_episode: tuple[Episode, dict[str, list[int]]],
    corruption: str,
) -> None:
    episode, buffers = tool_episode
    call = episode.calls[0]
    if corruption == "flip":
        buffers[call.segment_id][call.prompt_len] ^= 1
    elif corruption == "short":
        buffers[call.segment_id] = buffers[call.segment_id][: call.prompt_len]
    else:
        del buffers[call.segment_id]
    with pytest.raises(ValueError, match=re.escape(call.call_id)):
        build_datums(episode, buffers, [credit_for(call)])


@pytest.mark.parametrize(
    "corruption",
    [
        "logprob_length",
        "nan",
        "positive_inf",
        "negative_inf",
        "no_logprobs",
        "overlap",
        "backwards",
        "version",
        "zero_prompt",
        "negative_prompt",
    ],
)
def test_rejects_corrupt_calls(
    tool_episode: tuple[Episode, dict[str, list[int]]],
    corruption: str,
) -> None:
    episode, buffers = tool_episode
    first, call, last = episode.calls
    assert call.logprobs is not None
    changes: dict[str, Any]
    if corruption == "logprob_length":
        changes = {"logprobs": call.logprobs[:-1]}
    elif corruption in {"nan", "positive_inf", "negative_inf"}:
        value = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[
            corruption
        ]
        changes = {"logprobs": (value, *call.logprobs[1:])}
    elif corruption == "no_logprobs":
        changes = {"logprobs": None}
    elif corruption in {"overlap", "backwards"}:
        start = first.prompt_len + len(first.completion_ids) - 1 if corruption == "overlap" else 1
        changes = {
            "prompt_len": start,
            "completion_ids": tuple(
                buffers[call.segment_id][start : start + len(call.completion_ids)]
            ),
        }
    elif corruption == "version":
        changes = {"policy_version": 8}
    else:
        changes = {"prompt_len": 0 if corruption == "zero_prompt" else -1}
    changed = replace(episode, calls=(first, replace(call, **changes), last))
    with pytest.raises(ValueError, match=re.escape(call.call_id)):
        build_datums(changed, buffers, [credit_for(first)])


def test_rejects_api_credit(
    tool_episode: tuple[Episode, dict[str, list[int]]],
) -> None:
    episode, _ = tool_episode
    api_call = replace(
        episode.calls[0],
        segment_id="",
        prompt_len=0,
        completion_ids=(),
        logprobs=None,
        policy_version=None,
    )
    episode = replace(episode, calls=(api_call,), segments=())
    with pytest.raises(ValueError, match=re.escape(api_call.call_id)):
        build_datums(episode, {}, [credit_for(api_call)])


@pytest.mark.parametrize("missing", ["agent", "segment", "completion"])
def test_credit_must_identify_calls_with_completions(
    tool_episode: tuple[Episode, dict[str, list[int]]],
    missing: str,
) -> None:
    episode, buffers = tool_episode
    credit = credit_for(episode.calls[0])
    if missing == "completion":
        episode = replace(
            episode,
            calls=tuple(replace(call, completion_ids=(), logprobs=()) for call in episode.calls),
        )
    else:
        credit = replace(credit, **{f"{missing}_id": "missing"})
    with pytest.raises(ValueError, match="no trainable calls"):
        build_datums(episode, buffers, [credit])


def test_empty_calls_are_not_actions_but_must_share_policy_version(
    tool_episode: tuple[Episode, dict[str, list[int]]],
) -> None:
    episode, buffers = tool_episode
    first, middle, last = episode.calls
    empty = replace(middle, completion_ids=(), logprobs=())
    episode = replace(episode, calls=(first, empty, last))
    credit = credit_for(first)
    [datum] = build_datums(episode, buffers, [credit])
    assert_exact_datum(datum, episode, buffers[first.segment_id], credit)
    episode = replace(episode, calls=(first, replace(empty, policy_version=8), last))
    with pytest.raises(ValueError, match=re.escape(empty.call_id)):
        build_datums(episode, buffers, [credit])


def test_max_len_uses_trimmed_sequence_and_selected_learner(
    tool_episode: tuple[Episode, dict[str, list[int]]],
) -> None:
    episode, buffers = tool_episode
    credit = credit_for(episode.calls[0])
    last = episode.calls[-1]
    length = last.prompt_len + len(last.completion_ids)
    assert length < len(buffers[credit.segment_id])
    [datum] = build_datums(episode, buffers, [credit], max_len={"shared": length, "other": 1})
    assert len(datum.tokens) == length
    assert build_datums(episode, buffers, [credit], max_len={"other": 1}) == [datum]
    with pytest.raises(ValueError, match=r"shared.*exceeds max_len"):
        build_datums(episode, buffers, [credit], max_len={"shared": length - 1})


def test_datum_stats_per_learner() -> None:
    renderer = FakeRenderer()
    tokens = tuple(renderer.encode_completion("abc"))
    first = TrainDatum(
        "alpha",
        "episode",
        "solver0",
        "solver",
        "solver0/g0",
        0,
        7,
        tokens,
        (0.0, -0.2, -0.3),
        (0.0, 1.0, 1.0),
        (0.0, 0.5, 0.5),
    )
    second = replace(
        first,
        tokens=tokens[:2],
        logprobs=(-0.1,),
        mask=(1.0,),
        advantages=(0.0,),
    )
    third = replace(first, learner="beta")
    assert datum_stats([first, second, third]) == {
        "alpha/n": 2.0,
        "alpha/tokens": 6.0,
        "alpha/action_tokens": 3.0,
        "alpha/max_len": 4.0,
        "beta/n": 1.0,
        "beta/tokens": 4.0,
        "beta/action_tokens": 2.0,
        "beta/max_len": 4.0,
    }
    assert all(isinstance(value, float) for value in datum_stats([first, second, third]).values())
    assert datum_stats([]) == {}
