"""The sacrifice-relay held-out independence check flags each kind of repeat; synthetic problems."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import pytest

from marli.data.overlap import case_key, case_keys, case_size

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-25_sacrifice-relay/exp2_eval/overlap.py"
)
WRAP = (
    "Solve the following coding problem using the programming language python:\n\n{}\n\n"
    "The input will be stdin and you should print your solution to stdout\n"
)
FN_WRAP = WRAP.format(
    "{}\n\nYour solution should implemented in the function. The inputs will "
    "be passed to it and it should return the correct solution."
)


def story(seed: int, n: int = 60) -> str:
    """A statement of n made-up words; different seeds share almost no 5-word run."""
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(5000)}" for _ in range(n)) + "."


@pytest.fixture
def overlap(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("sacrifice_relay_overlap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # A pool of a dozen problems: boilerplate is what most of them share.
    monkeypatch.setattr(module, "LINE_MIN_COUNT", 5)
    return module


def test_every_kind_of_repeat_is_flagged_and_clean_problems_pass(overlap, tmp_path: Path) -> None:
    alice, bob = story(1), story(2)
    prompts = {
        "u/alice": WRAP.format(alice),
        "u/bob": WRAP.format(bob),
        "u/tiny": FN_WRAP.format("Return the sum of the list."),
        # draws, in order
        # The same statement under another subset's layout, without the wrapper.
        "d/alice_raw": f"{alice}\n\n-----Input-----\n\n-----Output-----",
        "d/bob_tests": WRAP.format(story(3)),  # new text, bob's tests
        "d/short": WRAP.format("Print the answer."),
        "d/russian": WRAP.format("Даны два числа. Выведите их сумму, а затем их разность и "
                                 "произведение, каждое на отдельной строке, без пробелов."),
        "d/clean": WRAP.format(story(4)),
        "d/clean_again": story(4),  # repeats an earlier draw
        "d/tiny_wrapper": FN_WRAP.format("Return the product of the list."),  # wrapper only
    }
    for i in range(6):  # filler: the wrapper lines become boilerplate (>= 5 problems)
        prompts[f"pool/{i}"] = WRAP.format(story(10 + i))
    tests = {k: [{"input": f"{k} {i}\n", "output": f"{i}"}] for i, k in enumerate(prompts)}
    tests["u/bob"] = [{"input": "1000003 2718281 314159\n", "output": "577215664"},
                      {"input": "1\n", "output": "1"}]
    tests["d/bob_tests"] = list(tests["u/bob"])
    keys = {k: case_keys(v) for k, v in tests.items()}
    sizes = {case_key(t["input"], t["output"]): case_size(t["input"], t["output"])
             for v in tests.values() for t in v}
    used = ["u/alice", "u/bob", "u/tiny"]
    draw = [k for k in prompts if k.startswith("d/")]
    args = argparse.Namespace(upstream=False, models=[], out=tmp_path)
    result = overlap.assess(used, draw, args, data=(prompts, keys, sizes))
    flags = {row["id"]: set(row["flags"]) for row in result["rows"]}
    assert "text_used" in flags["d/alice_raw"]
    assert flags["d/bob_tests"] == {"tests_used"}
    assert "short" in flags["d/short"]
    assert "not_english" in flags["d/russian"]
    assert flags["d/clean"] == set()
    assert "text_earlier" in flags["d/clean_again"] and "text_used" not in flags["d/clean_again"]
    assert "raw_text_used" in flags["d/tiny_wrapper"]
    assert result["ran"] == {"short", "not_english", "text", "raw_text", "tests"}
    text = overlap.report(result, len(used), "check")
    assert "ids          not run" in text and "Kept: 1 of 7" in text
    assert not any(word in text for word in alice.split()[:10])  # numbers only, no text
