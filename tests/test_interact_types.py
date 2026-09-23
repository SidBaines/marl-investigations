"""The record wire format preserves types and the exact ids needed by RL."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any

import pytest

from marli.interact import types
from marli.interact.types import (
    AgentInfo,
    Call,
    Episode,
    Event,
    EventKind,
    Outcome,
    Purpose,
    ReadVia,
    SegmentInfo,
    SegmentStart,
    Termination,
    Timing,
    ToolCallRecord,
    Usage,
    WorkspaceRead,
    Write,
    record_from_dict,
    record_to_dict,
)
from marli.render.fake import FakeRenderer

R = FakeRenderer()
IDS = tuple(R.encode_completion("answer", thinking="reason"))
USAGE = Usage(17, len(IDS), 4, 0.125, "fake")
TIMING = Timing(123.5, 2.75)
TOOL = ToolCallRecord(0, "write", {"text": "hi", "options": [1, None]}, "raw", True, "ok")
READ = WorkspaceRead("peer1", "scratchpad", 2, ReadVia.PUSH)
CALL = Call(
    call_id="episode/peer0/c0",
    episode_id="episode",
    agent_id="peer0",
    role="peer",
    policy_id="learner",
    policy_version=3,
    segment_id="peer0/g0",
    prompt_len=17,
    completion_ids=IDS,
    logprobs=(-0.5,) * len(IDS),
    text="answer",
    termination=Termination.STOP,
    purpose=Purpose.ACT,
    forced=False,
    tool_calls=(TOOL,),
    reads=(READ,),
    tick=0,
    seq=1,
    session_idx=0,
    seed=42,
    usage=USAGE,
    timing=TIMING,
)
SEGMENT = SegmentInfo("peer0/g0", "peer0", 0, SegmentStart.START, None, "fake", "sha", 31)
AGENT = AgentInfo("peer0", "peer", "learner", None, (0, "peer", 2))
WRITE = Write("peer0", "scratchpad", 1, "notes", 0, 2)
EVENT = Event(0, EventKind.CALL_START, "peer0", 0, {"call_id": CALL.call_id, "versions": {"p": 2}})
OUTCOME = Outcome("answer", {"peer0": "answer", "peer1": None}, "vote", {"answer": 1})
EPISODE = Episode(
    episode_id="episode",
    group_id="group",
    episode_idx=0,
    task_id="task",
    protocol="swarm",
    config_hash="hash",
    backend="scripted",
    agents=(AGENT,),
    segments=(SEGMENT,),
    calls=(CALL,),
    events=(EVENT,),
    workspace_log=(WRITE,),
    outcome=OUTCOME,
    grades={"_system": {"correct": 1.0}},
    limits_hit={"peer0": ("max_calls", "max_gen_tokens")},
    metrics={"total_gen": float(len(IDS))},
    replayable=True,
    ok=True,
    errors=("example error",),
)
RECORDS = [USAGE, TIMING, TOOL, READ, CALL, SEGMENT, AGENT, WRITE, EVENT, OUTCOME, EPISODE]


def assert_exact_types(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if is_dataclass(expected):
        for field in fields(expected):
            assert_exact_types(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_exact_types(a, e)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_exact_types(actual[key], value)
    else:
        assert actual == expected


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: type(record).__name__)
def test_json_round_trip_every_record(record: Any) -> None:
    encoded = record_to_dict(record)
    decoded = record_from_dict(type(record), json.loads(json.dumps(encoded)))
    assert decoded == record
    # StrEnum and Timing equality alone would hide incorrectly restored types/values.
    assert_exact_types(decoded, record)


def test_all_record_classes_are_covered() -> None:
    assert {type(record) for record in RECORDS} == {
        obj for obj in vars(types).values() if isinstance(obj, type) and is_dataclass(obj)
    }


@pytest.mark.parametrize("enum", [Termination, Purpose, SegmentStart, ReadVia, EventKind])
def test_all_enum_values(enum: type[Enum]) -> None:
    for member in enum:
        encoded = record_to_dict(member)
        assert type(encoded) is str
        assert encoded == member.value
        assert record_from_dict(enum, encoded) is member


def test_optional_values_and_empty_tuples() -> None:
    call = replace(
        CALL,
        policy_version=None,
        logprobs=None,
        tick=None,
        completion_ids=(),
        tool_calls=(ToolCallRecord(0, None, None, "bad", False, "", "unparsed", True),),
        reads=(),
        usage=Usage(),
    )
    episode = replace(
        EPISODE,
        calls=(call,),
        agents=(replace(AGENT, parent="coord0", seat_key=()),),
        segments=(replace(SEGMENT, carry_from="peer0/old", renderer=None, tokenizer_sha=None),),
        outcome=Outcome(None, {"peer0": None}, "single"),
        errors=(),
    )
    assert_exact_types(record_from_dict(Episode, record_to_dict(episode)), episode)


def test_nested_tuples_and_seat_keys_use_json_lists() -> None:
    encoded = record_to_dict(EPISODE)
    assert type(encoded["calls"]) is list
    assert type(encoded["calls"][0]["completion_ids"]) is list
    assert encoded["agents"][0]["seat_key"] == [0, "peer", 2]
    assert encoded["limits_hit"]["peer0"] == ["max_calls", "max_gen_tokens"]
    assert record_from_dict(tuple[tuple[int, ...], ...], [[1, 2], [3]]) == ((1, 2), (3,))
    assert record_from_dict(tuple[str, int], ["peer", 2]) == ("peer", 2)


def test_default_fields_can_be_omitted() -> None:
    assert record_from_dict(Usage, {}) == Usage()
    encoded = record_to_dict(CALL)
    del encoded["usage"]
    del encoded["timing"]
    restored = record_from_dict(Call, encoded)
    assert restored.usage == Usage()
    assert restored.timing.started_at == restored.timing.latency_s == 0.0
