"""Reject unsuitable teacher traces without changing their sampled token identities.

Each row is one agent segment, with int32 tokens and float32 target-aligned
masks packed little-endian as base64. RL-only arrays are reconstructed as zeros.
Recovery retains completed rows and recomputes rejection counts from the input.
"""

from __future__ import annotations

import base64
import sys
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar

from marli.config import input_field
from marli.errors import ConfigError
from marli.eval.rollout import EpisodeSet
from marli.handles import Handle, InputRef, register_handle
from marli.interact import records
from marli.interact.types import Call, Purpose, Termination
from marli.model import load_model
from marli.render.registry import get_renderer
from marli.rundir import RunDir
from marli.train.datums import build_datums
from marli.train.types import SegmentCredit, TrainDatum


@dataclass
class DataSFTConfig:
    episodes: str | None = input_field(
        None, help="teacher EpisodeSet (eval rollout with record_tokens=true)"
    )
    require_correct: bool = True
    require_ok: bool = True
    roles: tuple[str, ...] = ()
    require_conformant: bool = True
    student_model: str = ""
    max_len: int | None = None

    def __post_init__(self) -> None:
        if self.max_len is not None and (type(self.max_len) is not int or self.max_len < 2):
            raise ConfigError("max_len must be at least two tokens or None")


@register_handle
@dataclass(frozen=True, kw_only=True)
class SFTSet(Handle):
    KIND: ClassVar[str] = "sftset"
    MANIFEST: ClassVar[str] = "sft.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("datums",)

    datums: str
    n: int
    n_tokens: int
    n_action_tokens: int
    counts: dict[str, dict[str, int]]
    roles: tuple[str, ...]
    teacher_policies: dict[str, str]
    student_model: str
    tokenizer_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, "n_tokens": self.n_tokens, "n_action_tokens": self.n_action_tokens}


def _pack(values: Sequence[int] | Sequence[float], code: str) -> str:
    packed = array(code, values)
    if sys.byteorder == "big":
        packed.byteswap()
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _unpack(encoded: str, code: str) -> tuple[int, ...] | tuple[float, ...]:
    packed = array(code)
    packed.frombytes(base64.b64decode(encoded, validate=True))
    if sys.byteorder == "big":
        packed.byteswap()
    return tuple(packed)


def read_datums(data: SFTSet) -> Iterator[TrainDatum]:
    """Read exact SFT sequences; a torn final JSONL append is ignored."""
    for row in records._read_rows(data.file("datums")):
        tokens = _unpack(row.pop("tokens"), "i")
        mask = _unpack(row.pop("mask"), "f")
        zeros = (0.0,) * (len(tokens) - 1)
        yield TrainDatum(**row, tokens=tokens, mask=mask, logprobs=zeros, advantages=zeros)


def _conformance_reason(calls: Sequence[Call], *, nudges: bool) -> str | None:
    if any(
        call.termination == Termination.MALFORMED
        or any(not tool.parsed_ok for tool in call.tool_calls)
        for call in calls
    ):
        return "unparsed"
    if any(call.forced for call in calls):
        return "forced"
    if nudges and any(
        call.purpose == Purpose.ACT and call.completion_ids and not call.tool_calls
        for call in calls
    ):
        return "nudge"
    return None


async def data_sft(cfg: DataSFTConfig, run: RunDir) -> SFTSet:
    """Filter episodes, then agents, then segment lengths, with exclusive reasons.

    Counts are separated by unit: episode filters precede agent filters (role,
    conformance, no trainable calls); max_len is a datum filter. Token totals include
    the initial token. Every selected agent segment must use the student tokenizer.
    """
    cfg.__post_init__()
    if not cfg.episodes or not cfg.student_model:
        raise ConfigError("data sft requires episodes and student_model")
    source = EpisodeSet.load(cfg.episodes)
    if source.tokens is None:
        raise ConfigError("data sft requires teacher record_tokens=true")
    model = load_model(cfg.student_model)
    sha = get_renderer(model.renderer, hf_id=model.hf_id).tokenizer_sha
    counts = {
        "episodes": {"input": 0, "kept": 0, "dropped_not_ok": 0, "dropped_incorrect": 0},
        "agents": {
            "input": 0,
            "kept": 0,
            "dropped_role": 0,
            "dropped_unparsed": 0,
            "dropped_forced": 0,
            "dropped_nudge": 0,
            "dropped_no_trainable_calls": 0,
        },
        "datums": {"input": 0, "kept": 0, "dropped_max_len": 0},
    }
    done = {
        (row["episode_id"], row["agent_id"], row["segment_id"])
        for row in run.read_rows("datums.jsonl")
    }
    # Empty, fully rejected sets still own a readable data file.
    run.path("datums.jsonl").touch(exist_ok=True)
    roles: set[str] = set()
    n_tokens = n_action_tokens = 0
    limits = source.meta.get("limits", {})
    nudges = limits.get("on_no_tool_call", "nudge") == "nudge" and limits.get("max_nudges", 2) > 0
    for episode, buffers in records.read_episodes(source.root, with_tokens=True):
        ec = counts["episodes"]
        ec["input"] += 1
        if cfg.require_ok and not episode.ok:
            ec["dropped_not_ok"] += 1
            continue
        if cfg.require_correct and episode.grades.get("_system", {}).get("correct") != 1:
            ec["dropped_incorrect"] += 1
            continue
        ec["kept"] += 1
        for agent in episode.agents:
            ac = counts["agents"]
            ac["input"] += 1
            if cfg.roles and agent.role not in cfg.roles:
                ac["dropped_role"] += 1
                continue
            calls = [call for call in episode.calls if call.agent_id == agent.agent_id]
            reason = _conformance_reason(calls, nudges=nudges) if cfg.require_conformant else None
            if reason:
                ac[f"dropped_{reason}"] += 1
                continue
            trainable = {
                call.segment_id
                for call in calls
                if call.segment_id and call.completion_ids and call.logprobs is not None
            }
            if not trainable:
                ac["dropped_no_trainable_calls"] += 1
                continue
            segments = [s for s in episode.segments if s.agent_id == agent.agent_id]
            if trainable - {s.segment_id for s in segments}:
                raise ConfigError(f"agent {agent.agent_id!r}: missing SegmentInfo")
            for segment in segments:
                if segment.tokenizer_sha != sha:
                    raise ConfigError(
                        f"segment {segment.segment_id!r}: tokenizer_sha differs from student; "
                        "a same-family teacher with an identical tokenizer is required"
                    )
            ac["kept"] += 1
            credits = [
                SegmentCredit(
                    episode.episode_id,
                    agent.agent_id,
                    agent.role,
                    segment.segment_id,
                    "student",
                    1.0,
                    1.0,
                    1.0,
                )
                for segment in segments
                if segment.segment_id in trainable
            ]
            for datum in build_datums(episode, buffers, credits):
                counts["datums"]["input"] += 1
                if cfg.max_len is not None and len(datum.tokens) > cfg.max_len:
                    counts["datums"]["dropped_max_len"] += 1
                    continue
                zeros = (0.0,) * len(datum.mask)
                datum = replace(datum, logprobs=zeros, advantages=zeros)
                counts["datums"]["kept"] += 1
                n_tokens += len(datum.tokens)
                n_action_tokens += datum.n_action_tokens
                roles.add(datum.role)
                key = (datum.episode_id, datum.agent_id, datum.segment_id)
                if key not in done:
                    row = asdict(datum)
                    del row["logprobs"], row["advantages"]
                    row.update(tokens=_pack(datum.tokens, "i"), mask=_pack(datum.mask, "f"))
                    run.append_row("datums.jsonl", row)
                    done.add(key)
    return SFTSet(
        root=run.out,
        inputs=(InputRef.of(source),),
        datums="datums.jsonl",
        n=counts["datums"]["kept"],
        n_tokens=n_tokens,
        n_action_tokens=n_action_tokens,
        counts=counts,
        roles=tuple(sorted(roles)),
        teacher_policies=dict(source.policies),
        student_model=model.name,
        tokenizer_sha=sha,
    )
