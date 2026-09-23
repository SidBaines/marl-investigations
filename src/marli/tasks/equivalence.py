"""Votes count verifier-equivalent answers, keeping the first member as representative."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol


class AnswerVerifier(Protocol):
    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool: ...

    async def canonical(self, pred: str | None, answer_format: str) -> str | None: ...


async def group_answers(
    answers: Sequence[str | None], verifier: AnswerVerifier, answer_format: str
) -> list[int | None]:
    """Greedily compare to each cluster's first member; absent answers get no vote."""
    representatives: list[str] = []
    groups: list[int | None] = []
    for answer in answers:
        if answer is None or not answer.strip():
            groups.append(None)
            continue
        if await verifier.canonical(answer, answer_format) is None:
            groups.append(None)
            continue
        for index, representative in enumerate(representatives):
            if await verifier.verify(answer, representative, answer_format):
                groups.append(index)
                break
        else:
            groups.append(len(representatives))
            representatives.append(answer)
    return groups


async def majority(
    answers: Sequence[str | None],
    verifier: AnswerVerifier,
    answer_format: str,
    rng: random.Random,
) -> tuple[str | None, dict[str, int]]:
    """Return the winning first-member answer and counts, with seeded tie-breaking."""
    groups = await group_answers(answers, verifier, answer_format)
    representatives: dict[int, str] = {}
    votes: dict[str, int] = {}
    for answer, group in zip(answers, groups, strict=True):
        if group is None:
            continue
        assert answer is not None
        representative = representatives.setdefault(group, answer)
        votes[representative] = votes.get(representative, 0) + 1
    if not votes:
        return None, {}
    largest = max(votes.values())
    tied = [answer for answer, count in votes.items() if count == largest]
    return (tied[0] if len(tied) == 1 else rng.choice(tied)), votes
