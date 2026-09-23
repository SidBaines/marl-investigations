"""Preserve sampled token identities and agent boundaries when assigning loss.

Recorded offsets are the only source of action positions: observations remain
masked, and corrupt records fail rather than being re-rendered or truncated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from marli.interact.types import Episode
from marli.train.types import SegmentCredit, TrainDatum


def build_datums(
    episode: Episode,
    buffers: Mapping[str, Sequence[int]],
    credits: Sequence[SegmentCredit],
    *,
    max_len: dict[str, int] | None = None,
) -> list[TrainDatum]:
    """Build one exact, target-aligned datum per credit for this episode.

    Credits determine learner routing and advantages; this function validates
    recorded calls and never re-tokenizes, normalizes credit, or clips tokens.
    """
    datums: list[TrainDatum] = []
    for credit in credits:
        if credit.episode_id != episode.episode_id:
            continue
        segment_calls = sorted(
            (
                call
                for call in episode.calls
                if call.agent_id == credit.agent_id and call.segment_id == credit.segment_id
            ),
            key=lambda call: call.seq,
        )
        if segment_calls and not credit.segment_id:
            raise ValueError(f"Call {segment_calls[0].call_id}: API/chat calls cannot be trained")
        calls = [call for call in segment_calls if call.completion_ids]
        if not calls:
            raise ValueError(
                f"Credit for {episode.episode_id}/{credit.agent_id}, "
                f"segment {credit.segment_id!r}: no trainable calls"
            )
        version = calls[0].policy_version
        for call in segment_calls:
            if call.policy_version != version:
                raise ValueError(f"Call {call.call_id}: mixed policy versions in segment")
        if credit.segment_id not in buffers:
            raise ValueError(f"Call {calls[0].call_id}: missing buffer {credit.segment_id!r}")
        buffer = buffers[credit.segment_id]
        end = 0
        for call in calls:
            start = call.prompt_len
            n = len(call.completion_ids)
            if start <= 0:
                raise ValueError(f"Call {call.call_id}: completion needs a non-empty prompt")
            if start < end:
                raise ValueError(f"Call {call.call_id}: overlapping or out-of-order completions")
            if tuple(buffer[start : start + n]) != call.completion_ids:
                raise ValueError(f"Call {call.call_id}: completion ids do not match buffer")
            if call.logprobs is None:
                raise ValueError(f"Call {call.call_id}: missing logprobs; cannot train API/chat")
            if len(call.logprobs) != n:
                raise ValueError(f"Call {call.call_id}: logprob length does not match completion")
            if not all(isfinite(value) for value in call.logprobs):
                raise ValueError(f"Call {call.call_id}: logprobs must be finite")
            end = start + n
        limit = max_len.get(credit.learner) if max_len is not None else None
        if limit is not None and end > limit:
            raise ValueError(
                f"Learner {credit.learner!r}, segment {credit.segment_id!r}: "
                f"datum length {end} exceeds max_len {limit}"
            )
        tokens = tuple(buffer[:end])
        mask = [0.0] * (end - 1)
        logprobs = [0.0] * (end - 1)
        advantages = [0.0] * (end - 1)
        for call in calls:
            assert call.logprobs is not None
            start = call.prompt_len - 1
            stop = start + len(call.completion_ids)
            mask[start:stop] = [1.0] * len(call.completion_ids)
            logprobs[start:stop] = call.logprobs
            advantages[start:stop] = [credit.advantage] * len(call.completion_ids)
        datums.append(
            TrainDatum(
                learner=credit.learner,
                episode_id=episode.episode_id,
                agent_id=credit.agent_id,
                role=credit.role,
                segment_id=credit.segment_id,
                session_idx=calls[0].session_idx,
                policy_version=version,
                tokens=tokens,
                logprobs=tuple(logprobs),
                mask=tuple(mask),
                advantages=tuple(advantages),
                meta={
                    "call_ids": [call.call_id for call in calls],
                    "purposes": [call.purpose.value for call in calls],
                    "n_action": sum(len(call.completion_ids) for call in calls),
                    "forced_calls": sum(call.forced for call in calls),
                },
            )
        )
    return datums


def datum_stats(datums: Sequence[TrainDatum]) -> dict[str, float]:
    """Return ``<learner>/{n,tokens,action_tokens,max_len}`` for present learners.

    Token counts and maximum lengths include the initial token, matching the
    sequence lengths checked by ``build_datums``. An empty batch returns {}.
    """
    stats: dict[str, float] = {}
    for datum in datums:
        learner = datum.learner
        for name, value in (
            ("n", 1),
            ("tokens", len(datum.tokens)),
            ("action_tokens", datum.n_action_tokens),
        ):
            key = f"{learner}/{name}"
            stats[key] = stats.get(key, 0.0) + value
        key = f"{learner}/max_len"
        stats[key] = max(stats.get(key, 0.0), float(len(datum.tokens)))
    return stats
