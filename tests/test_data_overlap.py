"""Overlap checks find reworded and re-wrapped copies of a problem; synthetic problems only."""

from __future__ import annotations

import pytest

from marli.data.overlap import (
    CaseMatch,
    Rules,
    TextMatch,
    case_key,
    case_keys,
    case_size,
    common_lines,
    document_frequency,
    histogram,
    rare,
    shared_cases,
    shingles,
    strip_lines,
    text_matches,
    words,
)

STORY = (
    "Alice has a row of n lamps numbered from one to n. Each second she toggles every lamp whose "
    "number is a multiple of the current second. After n seconds she counts the lamps that are on "
    "and reports that count to Bob, who must check it quickly."
)
WRAPPED = (
    "Solve the following coding problem using the programming language python:\n\n"
    f"{STORY}\n\nInput\n\nThe only line holds n.\n\nOutput\n\nPrint the count.\n\n"
    "The input will be stdin and you should print your solution to stdout\n"
)
RAW = (
    f"{STORY}\n\n-----Input-----\n\nThe only line holds n.\n\n-----Output-----\n\n"
    "Print the count.\n"
)
OTHER = (
    "A frog sits on stone zero of a pond with m stones in a circle and jumps k stones clockwise "
    "each minute, eating one fly per visit until it returns home to the starting stone."
)
WRAPPERS = [
    f"Solve the following coding problem using the programming language python:\n\nTask {i}: "
    f"compute value {i} quickly.\n\nInput\n\nOutput\n\n"
    "The input will be stdin and you should print your solution to stdout"
    for i in range(5)
]


def test_words_ignore_case_punctuation_and_markup() -> None:
    assert words("Hello, WORLD -- $x_1$!") == ["hello", "world", "x_1"]
    assert words("") == []


def test_common_lines_strip_wrappers_and_headers_but_keep_the_story() -> None:
    common = common_lines([WRAPPED, RAW, *WRAPPERS], min_count=3)
    assert "solve the following coding problem using the programming language python" in common
    assert {"input", "output", ""} <= common
    assert "the only line holds n" not in common  # two prompts only
    stripped = strip_lines(WRAPPED, common)
    assert "Solve the following" not in stripped and "stdin" not in stripped
    assert STORY in stripped and "\n\n" not in stripped
    # The wrapped and raw copies now differ only in the dashed headers, which are words-free.
    assert shingles(strip_lines(WRAPPED, common)) == shingles(strip_lines(RAW, common))
    with pytest.raises(ValueError):
        common_lines([RAW], 0)


def test_shingles_are_word_runs_and_short_texts_get_one() -> None:
    assert len(shingles("a b c d e f g")) == 3
    assert shingles("A, b. C d e") == shingles("a b c d e")
    assert len(shingles("only two")) == 1
    assert shingles("") == frozenset()
    assert shingles("a b c d e f", n=2) != shingles("a b c d e f", n=3)


def test_text_matches_find_rewordings_containment_and_respect_order() -> None:
    reworded = STORY.replace("Alice", "Carol").replace("quickly", "fast")
    longer = STORY + " " + OTHER + " Extra closing words that make the statement much longer."
    queries = {"q_reworded": shingles(reworded), "q_longer": shingles(longer)}
    pool = {"story": shingles(STORY), "other": shingles(OTHER)}
    found = text_matches(queries, pool)
    best = found["q_reworded"][0]
    assert best.other == "story" and 0.6 < best.jaccard < 1 and best.containment > 0.7
    contained = {m.other: m for m in found["q_longer"]}
    assert contained["story"].containment == 1.0 and contained["story"].jaccard < 0.6
    assert contained["other"].containment == 1.0
    assert all(m.other != "other" for m in found["q_reworded"])
    # A shingle in more pool problems than max_postings is ignored.
    stock = {f"p{i}": shingles(STORY) for i in range(3)}
    assert text_matches({"q": shingles(STORY)}, stock, max_postings=2) == {"q": []}
    # Within one set: compared only with earlier problems, never with itself.
    same = {"a": shingles(STORY), "b": shingles(reworded), "c": shingles(OTHER)}
    order = {"a": 0, "b": 1, "c": 2}
    within = text_matches(same, same, order=order)
    assert within["a"] == [] and [m.other for m in within["b"]] == ["a"] and within["c"] == []


def test_case_keys_match_layout_quotes_and_json_brackets_not_values() -> None:
    assert case_key("3\n1 2 3\n", "6") == case_key("3 1 2 3", "6\n")
    assert case_key("[1, 2, 3]", '"ab"') == case_key("1 2 3", "ab")
    assert case_key("Yes", "NO") == case_key("yes", "no")
    assert case_key("1 2", "3") != case_key("1 2", "4")
    assert case_key("1 2", "3") != case_key("12", "3")
    assert case_key("0.5", "x") != case_key("05", "x")
    assert len(case_keys([{"input": "1", "output": "1"}, {"input": "1\n", "output": "1"}])) == 1


def test_shared_cases_discount_common_pairs_and_report_containment() -> None:
    trivial = {"input": "1", "output": "1"}
    tests = {
        "used": [trivial, {"input": "7 3\n", "output": "21"}, {"input": "8 9", "output": "72"}],
        "copy": [{"input": "[7, 3]", "output": "21"}],  # one distinctive pair, contained
        "unrelated": [trivial, {"input": "5", "output": "120"}],
        "partial": [{"input": "8 9", "output": "72"}, {"input": "100 100", "output": "10000"}],
    }
    keys = {name: case_keys(cases) for name, cases in tests.items()}
    frequency: dict[int, int] = {}
    for items in keys.values():
        for item in items:
            frequency[item] = frequency.get(item, 0) + 1
    frequency[case_key("1", "1")] = 50  # common across the whole source pool
    queries = {name: keys[name] for name in ("copy", "unrelated", "partial")}
    found = shared_cases(queries, {"used": keys["used"]}, frequency, max_df=3)
    [copy] = found["copy"]
    assert (copy.other, copy.shared, copy.distinctive, copy.contained) == ("used", 1, 1, True)
    [unrelated] = found["unrelated"]
    assert unrelated.shared == 1 and unrelated.distinctive == 0 and not unrelated.contained
    [partial] = found["partial"]
    assert partial.distinctive == 1 and not partial.contained
    within = shared_cases(keys, keys, frequency, max_df=3, order={k: i for i, k in enumerate(keys)})
    assert within["used"] == [] and [m.other for m in within["copy"]] == ["used"]


def test_histogram_closes_the_last_bin() -> None:
    assert histogram([0.0, 0.1, 0.5, 1.0], [0, 0.5, 1]) == [("[0, 0.5)", 2), ("[0.5, 1]", 2)]


def test_case_size_counts_normalised_characters() -> None:
    assert case_size("[1, 2]", '"ab"') == 4
    assert case_size("1 1 1 9\n", "2") == 5
    assert case_size("", "") == 0


def test_frequency_and_rarity_count_sets_once_each() -> None:
    sets = [frozenset({1, 2}), frozenset({2, 3}), frozenset({2})]
    frequency = document_frequency(sets)
    assert frequency == {1: 1, 2: 3, 3: 1}
    assert rare(frozenset({1, 2, 4}), frequency, max_df=1) == frozenset({1, 4})


def test_long_rare_pairs_separate_copies_from_coincidences() -> None:
    copy = [{"input": "12 345 6789 1011 1213\n", "output": "424242\n"}]
    coincidence = [{"input": "1 1 1 9\n", "output": "2"}]
    used = {"used": case_keys(copy + coincidence + [{"input": "5", "output": "5"}])}
    queries = {"copy": case_keys(copy), "coincidence": case_keys(coincidence)}
    frequency = document_frequency([*used.values(), *queries.values()])
    sizes = {case_key(t["input"], t["output"]): case_size(t["input"], t["output"])
             for t in copy + coincidence}
    found = shared_cases(queries, used, frequency, max_df=3, sizes=sizes, min_chars=16)
    assert found["copy"][0].long == 1 and found["coincidence"][0].long == 0
    rules = Rules()
    # Contained, but a single pair: only its length decides.
    assert rules.cases(found["copy"]) and not rules.cases(found["coincidence"])


def test_rules_flag_text_and_tests_at_their_thresholds() -> None:
    rules = Rules(jaccard=0.05, containment=0.1, containment_min=15, contained_min=2,
                  long_min=1, distinctive_min=3)
    sizes = {"q": 100, "big": 100, "tiny": 5}
    assert rules.text([TextMatch("big", 0.05, 0.09, 9)], sizes, "q")
    assert not rules.text([TextMatch("big", 0.049, 0.099, 9)], sizes, "q")
    assert rules.text([TextMatch("big", 0.01, 0.1, 10)], sizes, "q")
    assert not rules.text([TextMatch("tiny", 0.01, 0.8, 4)], sizes, "q")  # too small to contain
    assert not rules.text([], sizes, "q")
    assert rules.cases([CaseMatch("u", 2, 0, 0, True)])
    assert not rules.cases([CaseMatch("u", 1, 1, 0, True)])
    assert rules.cases([CaseMatch("u", 1, 1, 1, False)])
    assert rules.cases([CaseMatch("u", 5, 3, 0, False)])
    assert not rules.cases([CaseMatch("u", 40, 2, 0, False)])  # many generic pairs only
