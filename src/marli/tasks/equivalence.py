"""Votes count verifier-equivalent answers, keeping the first member as representative."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class AnswerVerifier(Protocol):
    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool: ...

    async def canonical(self, pred: str | None, answer_format: str) -> str | None: ...


@dataclass(frozen=True)
class Vote:
    representative: str
    count: int


async def group_answers(
    answers: Sequence[str | None], verifier: AnswerVerifier, answer_format: str
) -> list[int | None]:
    """Reuse canonical keys, otherwise compare representatives concurrently.

    The lowest matching cluster index wins, independent of completion order.
    Absent answers get no vote. Each distinct canonical key is compared once
    against the representatives present when it first appears.
    """
    representatives: list[str] = []
    canonical_groups: dict[str, int] = {}
    groups: list[int | None] = []
    for answer in answers:
        if answer is None or not answer.strip():
            groups.append(None)
            continue
        key = await verifier.canonical(answer, answer_format)
        if key is None:
            groups.append(None)
            continue
        if key in canonical_groups:
            groups.append(canonical_groups[key])
            continue
        matches = await asyncio.gather(
            *(
                verifier.verify(answer, representative, answer_format)
                for representative in representatives
            )
        )
        group = next((index for index, match in enumerate(matches) if match), len(representatives))
        if group == len(representatives):
            representatives.append(answer)
        canonical_groups[key] = group
        groups.append(group)
    return groups


async def majority(
    answers: Sequence[str | None],
    verifier: AnswerVerifier,
    answer_format: str,
    rng: random.Random,
) -> tuple[str | None, dict[int, Vote]]:
    """Return the winning answer and cluster-id votes with seeded tie-breaking."""
    groups = await group_answers(answers, verifier, answer_format)
    votes: dict[int, Vote] = {}
    for answer, group in zip(answers, groups, strict=True):
        if group is None:
            continue
        assert answer is not None
        previous = votes.get(group, Vote(answer, 0))
        votes[group] = Vote(previous.representative, previous.count + 1)
    if not votes:
        return None, {}
    largest = max(vote.count for vote in votes.values())
    tied = [group for group, vote in votes.items() if vote.count == largest]
    winner = tied[0] if len(tied) == 1 else rng.choice(tied)
    return votes[winner].representative, votes
