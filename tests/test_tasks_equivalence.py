"""Voting uses verifier equivalence, stable representatives and an explicit tie RNG."""

from __future__ import annotations

import asyncio
import random

import pytest

import marli.tasks.equivalence as equivalence_module
from marli.tasks.equivalence import Vote, group_answers, majority
from marli.tasks.verifiers import MathVerifier


async def test_integer_equivalence_and_original_representatives() -> None:
    answers = [None, "070", "70", r"\boxed{70}", "$70$", "71", "071", "", " "]
    async with MathVerifier() as verifier:
        assert await group_answers(answers, verifier, "integer") == [
            None,
            0,
            0,
            0,
            0,
            1,
            1,
            None,
            None,
        ]
        assert await majority(answers, verifier, "integer", random.Random(42)) == (
            "070",
            {0: Vote("070", 4), 1: Vote("71", 2)},
        )
        assert verifier._pool is None


async def test_symbolic_equivalence_groups_mixed_notation() -> None:
    pytest.importorskip("math_verify")
    pytest.importorskip("pebble")
    answers = [r"\frac{1}{2}", "0.5", "1/2", r"\boxed{0.5}", "2/3", "-1/21", r"-\frac{1}{21}"]
    async with MathVerifier(max_workers=1) as verifier:
        assert await group_answers(answers, verifier, "latex") == [0, 0, 0, 0, 1, 2, 2]
        assert await majority(answers, verifier, "latex", random.Random(42)) == (
            r"\frac{1}{2}",
            {0: Vote(r"\frac{1}{2}", 4), 1: Vote("2/3", 1), 2: Vote("-1/21", 2)},
        )


@pytest.mark.parametrize("answers", [[], [None, None], [None, "", " ", "$$", r"\boxed{}"]])
async def test_no_answers_means_no_vote(answers: list[str | None]) -> None:
    async with MathVerifier() as verifier:
        assert await group_answers(answers, verifier, "integer") == [None] * len(answers)
        assert await majority(answers, verifier, "integer", random.Random(1)) == (None, {})


async def test_ties_are_deterministic_and_exclude_smaller_clusters() -> None:
    answers = ["01", "2", "1", "02", "3"]
    async with MathVerifier() as verifier:
        winners = set()
        for seed in range(8):
            expected = random.Random(seed).choice(["01", "2"])
            result = await majority(answers, verifier, "integer", random.Random(seed))
            assert result == (expected, {0: Vote("01", 2), 1: Vote("2", 2), 2: Vote("3", 1)})
            winners.add(result[0])
        assert winners == {"01", "2"}


async def test_greedy_grouping_compares_only_first_cluster_members() -> None:
    class Verifier:
        def __init__(self) -> None:
            self.comparisons: list[tuple[str | None, str]] = []

        async def canonical(self, pred: str | None, answer_format: str) -> str | None:
            return pred

        async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
            self.comparisons.append((pred, gold))
            return (pred, gold) in {("b", "a"), ("c", "b")}

    verifier = Verifier()
    assert await group_answers(["a", "b", "c", "b"], verifier, "latex") == [0, 0, 1, 0]
    assert verifier.comparisons == [("b", "a"), ("c", "a")]


class HangingVerifier:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def canonical(self, pred: str | None, answer_format: str) -> str | None:
        return pred

    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)
        except TimeoutError:
            return False
        finally:
            self.active -= 1
        raise AssertionError("the fake worker must time out")


async def test_identical_pathological_keys_never_verify_and_votes_match_groups() -> None:
    tower = "2^{" * 100 + "2" + "}" * 100
    verifier = HangingVerifier()
    assert await group_answers([tower] * 12, verifier, "latex") == [0] * 12
    assert verifier.calls == 0
    answers = [tower, tower, "5"]
    assert await group_answers(answers, verifier, "latex") == [0, 0, 1]
    assert verifier.calls == 1
    assert await majority(answers, verifier, "latex", random.Random(0)) == (
        tower,
        {0: Vote(tower, 2), 1: Vote("5", 1)},
    )
    assert verifier.calls == 2


async def test_distinct_representatives_are_compared_concurrently_once_per_key() -> None:
    verifier = HangingVerifier()
    answers = ["a", "b", "c", "d"] * 3
    assert await group_answers(answers, verifier, "latex") == [0, 1, 2, 3] * 3
    assert verifier.calls == 6
    assert verifier.peak == 3


async def test_lowest_cluster_index_wins_regardless_of_completion_order() -> None:
    class Verifier(HangingVerifier):
        async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
            if pred == "c" and gold == "a":
                await asyncio.sleep(0.01)
            return pred == "c"

    assert await group_answers(["a", "b", "c"], Verifier(), "latex") == [0, 1, 0]


async def test_vote_identity_is_cluster_id_even_with_repeated_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def separate_groups(*args: object) -> list[int]:
        return [0, 1, 2]

    monkeypatch.setattr(equivalence_module, "group_answers", separate_groups)
    _, votes = await majority(["T", "T", "5"], HangingVerifier(), "latex", random.Random(0))
    assert votes == {0: Vote("T", 1), 1: Vote("T", 1), 2: Vote("5", 1)}
