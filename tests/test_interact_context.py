"""Reset policies preserve only enabled carry state and report any lost text."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from marli.errors import ConfigError
from marli.interact.context import CarryText, ContextManager, make_context_manager
from marli.interact.limits import ContextLimits, Limits, SessionLimits
from marli.interact.system import ContextSpec
from marli.render.fake import FakeRenderer

KINDS = ("none", "compaction", "notes", "both", "tail")


@pytest.mark.parametrize(
    ("kind", "compacts", "has_notes", "expected_text"),
    [
        ("none", False, False, ""),
        ("compaction", True, False, "[Previous session summary]\nsummary"),
        ("notes", False, True, "[Your notes]\nnotes"),
        ("both", True, True, "[Previous session summary]\nsummary\n\n[Your notes]\nnotes"),
        ("tail", False, False, ""),
    ],
)
def test_kind_behavior(kind: str, compacts: bool, has_notes: bool, expected_text: str) -> None:
    spec = ContextSpec(kind=kind, compact_threshold=64, tail_tokens=8)
    manager = make_context_manager(spec, Limits())

    assert manager.spec == spec
    assert manager.should_compact(65) is compacts
    assert bool(manager.compaction_instruction()) is compacts
    assert (manager.session_carry_instruction() is not None) is compacts
    assert manager.has_notes_tools is has_notes
    assert manager.carry_text(summary="summary", notes="notes") == CarryText(expected_text, False)
    ids = FakeRenderer().encode_completion("progress so far")
    assert manager.tail_prefill(ids) == (ids[-8:] if kind == "tail" else [])


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize(("prompt_len", "expected"), [(63, False), (64, False), (65, True)])
def test_compaction_threshold_boundary(kind: str, prompt_len: int, expected: bool) -> None:
    manager = make_context_manager(ContextSpec(kind=kind, compact_threshold=64), Limits())
    assert manager.should_compact(prompt_len) is expected


@pytest.mark.parametrize("kind", ["compaction", "both"])
def test_disabled_compaction_still_has_session_carry(kind: str) -> None:
    manager = make_context_manager(ContextSpec(kind=kind, compact_threshold=0), Limits())
    assert manager.should_compact(100_000) is False
    assert manager.session_carry_instruction() is not None


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize("carry_max_tokens", [32, 257])
def test_summary_instructions(kind: str, carry_max_tokens: int) -> None:
    manager = make_context_manager(
        ContextSpec(kind=kind), Limits(session=SessionLimits(carry_max_tokens=carry_max_tokens))
    )
    compact = manager.compaction_instruction()
    carry = manager.session_carry_instruction()
    assert carry is not None
    body = (
        "Write the summary itself as your reply. Include what you have tried and what worked "
        "or failed (and why), intermediate results with exact values, your current best answer "
        "or the current state of your solution, and your next steps. Do not restate the task. "
        f"Keep it within about {carry_max_tokens} tokens "
        f"(roughly {carry_max_tokens * 3 // 4} words); anything longer is cut. "
        "Do not call any tools and do not continue working on the task in this reply."
    )
    assert compact == (
        "Your context is almost full. It will now be cleared and replaced by the summary you "
        "write in this reply; afterwards you will see only the original task and this summary. "
        + body
    )
    carry_opening = (
        "Your session is ending. Your next session will start with a fresh context and will see "
        "only the original task and the summary you write in this reply"
    )
    notes_reminder = " plus your saved notes; don't repeat them" if kind == "both" else ""
    assert carry == carry_opening + notes_reminder + ". " + body
    assert manager.compaction_instruction() == compact
    assert manager.session_carry_instruction() == carry


@pytest.mark.parametrize(
    ("summary", "notes", "expected"),
    [
        ("answer", "next steps", "[Previous session summary]\nanswer\n\n[Your notes]\nnext steps"),
        ("answer", None, "[Previous session summary]\nanswer"),
        ("answer", "", "[Previous session summary]\nanswer"),
        (None, "next steps", "[Your notes]\nnext steps"),
        ("", "next steps", "[Your notes]\nnext steps"),
        ("  \n", " \n ", ""),
        ("answer", " \n ", "[Previous session summary]\nanswer"),
        ("  \n", "notes", "[Your notes]\nnotes"),
        (" answer ", " notes ", "[Previous session summary]\n answer \n\n[Your notes]\n notes "),
    ],
)
def test_carry_format(summary: str | None, notes: str | None, expected: str) -> None:
    manager = make_context_manager(ContextSpec(kind="both"), Limits())
    assert manager.carry_text(summary=summary, notes=notes) == CarryText(expected, False)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("summary", [None, "", " \n\t" * 2000])
@pytest.mark.parametrize("notes", [None, "", " \n\t" * 2000])
def test_empty_carry(kind: str, summary: str | None, notes: str | None) -> None:
    manager = make_context_manager(ContextSpec(kind=kind, tail_tokens=8), Limits())
    assert manager.carry_text(summary=summary, notes=notes) == CarryText("", False)


@pytest.mark.parametrize(
    ("summary", "notes", "expected", "truncated"),
    [
        ("abc", "xy", "[Previous session summary]\nabc\n\n[Your notes]\nxy", False),
        ("abcd", "xyz", "[Previous session summary]\nabcd\n\n[Your notes]\nxyz", False),
        (
            "abcde",
            "xyz",
            "[Previous session summary]\n…[truncated]bcde\n\n[Your notes]\nxyz",
            True,
        ),
        (
            "abcd",
            "wxyz",
            "[Previous session summary]\nabcd\n\n[Your notes]\n…[truncated]xyz",
            True,
        ),
        (
            "abcde",
            "wxyz",
            "[Previous session summary]\n…[truncated]bcde\n\n[Your notes]\n…[truncated]xyz",
            True,
        ),
        ("古い記録新", None, "[Previous session summary]\n…[truncated]い記録新", True),
        (None, "古記録新", "[Your notes]\n…[truncated]記録新", True),
    ],
)
def test_carry_truncation(
    summary: str | None, notes: str | None, expected: str, truncated: bool
) -> None:
    manager = make_context_manager(
        ContextSpec(kind="both", notes_cap_chars=3),
        Limits(session=SessionLimits(carry_max_tokens=1)),
    )
    assert manager.carry_text(summary=summary, notes=notes) == CarryText(expected, truncated)
    assert manager.carry_text(summary="ok", notes="ok").truncated is False


@pytest.mark.parametrize(
    ("kind", "summary", "notes"),
    [
        ("none", "unused summary", "unused notes"),
        ("tail", "unused summary", "unused notes"),
        ("compaction", None, "unused notes"),
        ("notes", "unused summary", None),
    ],
)
def test_disabled_sources_do_not_flag_truncation(
    kind: str, summary: str | None, notes: str | None
) -> None:
    manager = make_context_manager(
        ContextSpec(kind=kind, notes_cap_chars=1, tail_tokens=8),
        Limits(session=SessionLimits(carry_max_tokens=1)),
    )
    assert manager.carry_text(summary=summary, notes=notes) == CarryText("", False)


def test_carry_record_is_frozen() -> None:
    carry = CarryText(text="summary", truncated=False)
    with pytest.raises(FrozenInstanceError):
        carry.text = "changed"
    with pytest.raises(FrozenInstanceError):
        carry.truncated = True


def test_context_spec_is_read_only() -> None:
    assert isinstance(ContextManager.spec, property)
    assert ContextManager.spec.fset is None
    manager = make_context_manager(ContextSpec(), Limits())
    with pytest.raises(FrozenInstanceError):
        manager.spec = ContextSpec(kind="notes")


@pytest.mark.parametrize("length", [0, 3, 5, 10])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_tail_is_an_exact_independent_copy(length: int, as_tuple: bool) -> None:
    manager = make_context_manager(ContextSpec(kind="tail", tail_tokens=5), Limits())
    renderer = FakeRenderer()
    ids = renderer.encode_completion("café", thinking="next steps")[-length:] if length else []
    original = ids.copy()
    carried = manager.tail_prefill(tuple(ids) if as_tuple else ids)
    assert isinstance(carried, list)
    assert carried == original[-5:]
    carried.append(999)
    assert ids == original
    assert manager.tail_prefill(ids) == original[-5:]


@pytest.mark.parametrize("kind", ["unknown", "", "Compaction", "session-carry"])
def test_unknown_kind_rejected(kind: str) -> None:
    with pytest.raises(ConfigError, match="kind") as caught:
        make_context_manager(ContextSpec(kind=kind), Limits())
    assert f"got {kind!r}" in str(caught.value)
    assert all(repr(valid) in str(caught.value) for valid in KINDS)


@pytest.mark.parametrize("kind", ["compaction", "both"])
def test_compaction_reserve_validation(kind: str) -> None:
    limits = Limits(ctx=ContextLimits(max_ctx=256), session=SessionLimits(carry_max_tokens=32))
    valid = make_context_manager(
        ContextSpec(kind=kind, compact_threshold=224, compact_reserve=32), limits
    )
    assert valid.should_compact(224) is False
    assert valid.should_compact(225) is True
    with pytest.raises(ConfigError, match="compact_threshold") as caught:
        make_context_manager(
            ContextSpec(kind=kind, compact_threshold=225, compact_reserve=32), limits
        )
    assert "got 225" in str(caught.value)
    assert "256 - 32" in str(caught.value)


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize("reserve", [-1, 0, 512])
def test_disabled_compaction_does_not_require_context_reserve(kind: str, reserve: int) -> None:
    manager = make_context_manager(
        ContextSpec(kind=kind, compact_threshold=0, compact_reserve=reserve),
        Limits(ctx=ContextLimits(max_ctx=256)),
    )
    assert manager.should_compact(257) is False


@pytest.mark.parametrize("tail_tokens", [-1, 0, 128, 129, 256])
@pytest.mark.parametrize("max_ctx", [256, 257])
def test_invalid_tail_length(tail_tokens: int, max_ctx: int) -> None:
    with pytest.raises(ConfigError, match="tail_tokens") as caught:
        make_context_manager(
            ContextSpec(kind="tail", tail_tokens=tail_tokens),
            Limits(ctx=ContextLimits(max_ctx=max_ctx)),
        )
    assert f"got {tail_tokens!r}" in str(caught.value)


@pytest.mark.parametrize("tail_tokens", [1, 127])
@pytest.mark.parametrize("max_ctx", [256, 257])
def test_valid_tail_length_boundaries(tail_tokens: int, max_ctx: int) -> None:
    manager = make_context_manager(
        ContextSpec(kind="tail", tail_tokens=tail_tokens),
        Limits(ctx=ContextLimits(max_ctx=max_ctx)),
    )
    ids = FakeRenderer().encode_text("a" * 200)
    assert manager.tail_prefill(ids) == ids[-tail_tokens:]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("notes_cap_chars", [0, -1])
def test_invalid_notes_cap(kind: str, notes_cap_chars: int) -> None:
    with pytest.raises(ConfigError, match="notes_cap_chars") as caught:
        make_context_manager(
            ContextSpec(kind=kind, notes_cap_chars=notes_cap_chars, tail_tokens=8), Limits()
        )
    assert f"got {notes_cap_chars!r}" in str(caught.value)


@pytest.mark.parametrize("kind", KINDS)
def test_negative_compaction_threshold_is_invalid(kind: str) -> None:
    with pytest.raises(ConfigError, match="compact_threshold.*got -1"):
        make_context_manager(ContextSpec(kind=kind, compact_threshold=-1, tail_tokens=8), Limits())


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize("reserve", [-1, 0, 31])
def test_enabled_compaction_reserve_must_cover_summary(kind: str, reserve: int) -> None:
    with pytest.raises(ConfigError, match="compact_reserve.*cover the summary") as caught:
        make_context_manager(
            ContextSpec(kind=kind, compact_threshold=64, compact_reserve=reserve),
            Limits(session=SessionLimits(carry_max_tokens=32)),
        )
    assert f"got {reserve!r}" in str(caught.value)


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize("carry_reserve", [0, -1, 31])
@pytest.mark.parametrize("threshold", [0, 64])
def test_summary_carry_requires_reserve(kind: str, carry_reserve: int, threshold: int) -> None:
    with pytest.raises(ConfigError, match=r"session\.carry_reserve.*cover the summary") as caught:
        make_context_manager(
            ContextSpec(kind=kind, compact_threshold=threshold),
            Limits(session=SessionLimits(carry_reserve=carry_reserve, carry_max_tokens=32)),
        )
    assert f"got {carry_reserve!r}" in str(caught.value)


@pytest.mark.parametrize("kind", ["compaction", "both"])
@pytest.mark.parametrize("reserve", [32, 33])
def test_summary_reserves_may_equal_or_exceed_summary_cap(kind: str, reserve: int) -> None:
    manager = make_context_manager(
        ContextSpec(kind=kind, compact_threshold=64, compact_reserve=reserve),
        Limits(session=SessionLimits(carry_reserve=reserve, carry_max_tokens=32)),
    )
    assert manager.should_compact(65)
    assert manager.session_carry_instruction() is not None


@pytest.mark.parametrize("kind", ["none", "notes", "tail"])
def test_kinds_without_summary_calls_do_not_require_reserves(kind: str) -> None:
    manager = make_context_manager(
        ContextSpec(kind=kind, compact_threshold=100_000, tail_tokens=8),
        Limits(session=SessionLimits(carry_reserve=0)),
    )
    assert manager.should_compact(100_001) is False
    assert manager.session_carry_instruction() is None


@pytest.mark.parametrize("kind", ["none", "compaction", "notes", "both"])
def test_non_tail_kinds_ignore_tail_tokens(kind: str) -> None:
    manager = make_context_manager(ContextSpec(kind=kind, tail_tokens=100_000), Limits())
    assert manager.tail_prefill(FakeRenderer().encode_completion("answer")) == []
