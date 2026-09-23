"""The fake renderer is test infrastructure for every CPU test of the harness,
so its own contract is pinned here: parse inverts encode_completion, and the
delta path (initial + completion + continuation) produces exactly the tokens a
full re-render of the same conversation would."""

from __future__ import annotations

from marli.render.base import DeltaRenderer, Msg, ToolCall, ToolSpec
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer

R = FakeRenderer()
TOOLS = [
    ToolSpec(
        "submit",
        "Submit the final answer.",
        {"type": "object", "properties": {"answer": {"type": "string"}}},
    )
]


def test_satisfies_protocol():
    assert isinstance(R, DeltaRenderer)
    assert R.supports_delta and R.stop_token_ids == (S["eot"],)


def test_parse_roundtrip_content_thinking_and_calls():
    ids = R.encode_completion("hello", thinking="hmm", tool_calls=[("submit", {"answer": "42"})])
    turn = R.parse(ids)
    assert turn.termination == "stop"
    assert turn.content == "hello" and turn.thinking == "hmm"
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.ok and call.name == "submit" and call.arguments == {"answer": "42"}


def test_parse_length_and_malformed():
    assert R.parse(R.encode_completion("cut off", stop=False)).termination == "length"
    bad = R.parse(R.encode_completion(raw_tool_bodies=["not json"]))
    assert (
        bad.termination == "stop"
        and not bad.tool_calls[0].ok
        and bad.tool_calls[0].raw == "not json"
    )
    unclosed = [*R.encode_completion("x", stop=False), S["call"], *R.encode_text('{"name": "sub')]
    assert R.parse(unclosed + [S["eot"]]).termination == "malformed"


def test_delta_equals_full_rerender():
    """buf = initial + completion + continuation(...) must equal initial(full history)."""
    system, user = "You are peer0.", "Solve 2+2."
    buf = R.initial(system, TOOLS, [Msg("user", user)])
    completion = R.encode_completion("thinking aloud", tool_calls=[("submit", {"answer": "4"})])
    buf += completion
    buf += R.continuation(
        "stop",
        [
            Msg("tool", "ok", name="submit"),
            Msg("tool", "second", name="x"),
            Msg("user", "[notify] peer1 wrote v1"),
        ],
    )
    full = R.initial(
        system,
        TOOLS,
        [
            Msg("user", user),
            Msg("assistant", "thinking aloud", tool_calls=(ToolCall("submit", {"answer": "4"}),)),
            Msg("tool", "ok", name="submit"),
            Msg("tool", "second", name="x"),
            Msg("user", "[notify] peer1 wrote v1"),
        ],
    )
    assert buf == full


def test_consecutive_tool_results_grouped_in_one_block():
    ids = R.continuation("stop", [Msg("tool", "a"), Msg("tool", "b")])
    assert ids.count(S["tool"]) == 1 and ids.count(S["res"]) == 2 and ids[-1] == S["asst"]


def test_length_termination_gets_forced_close():
    ids = R.continuation("length", [Msg("user", "continue")])
    assert ids[0] == S["eot"]
    assert R.continuation("stop", [Msg("user", "continue")])[0] == S["user"]


def test_forced_tool_prefix_completes_to_valid_call():
    prefix = R.forced_tool_prefix("submit")
    completion = [*R.encode_text('{"answer": "7"}}'), S["/call"], S["eot"]]
    turn = R.parse(prefix + completion)
    assert turn.tool_calls[0].ok and turn.tool_calls[0].arguments == {"answer": "7"}
