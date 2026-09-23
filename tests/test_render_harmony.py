"""Official Harmony conversations pin token parity and sampled-turn parsing.

These integration tests need only the optional, offline Harmony encoding;
harness tests elsewhere use FakeRenderer instead of a model tokenizer.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from marli.render.base import DeltaRenderer, Msg, ToolCall, ToolSpec
from marli.render.harmony import HARMONY_RENDERERS, HarmonyRenderer

if TYPE_CHECKING:
    from openai_harmony import HarmonyEncoding, Message

h = pytest.importorskip("openai_harmony")

TOOLS = [
    ToolSpec(
        "submit",
        "Submit the final answer.",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
]
SYSTEM = "You are peer0."
QUESTION = "Solve 2+2."


@pytest.fixture(scope="module")
def encoding() -> HarmonyEncoding:
    return h.load_harmony_encoding(h.HarmonyEncodingName.HARMONY_GPT_OSS)


@pytest.fixture(scope="module")
def renderer() -> HarmonyRenderer:
    return HarmonyRenderer()


def _reference_prompt(encoding: HarmonyEncoding, messages: Sequence[Message]) -> list[int]:
    system = h.SystemContent(
        reasoning_effort=h.ReasoningEffort.MEDIUM, conversation_start_date="2026-09-23"
    )
    developer = h.DeveloperContent(instructions=SYSTEM).with_function_tools(
        [h.ToolDescription.new(t.name, t.description, parameters=t.parameters) for t in TOOLS]
    )
    return encoding.render_conversation_for_completion(
        h.Conversation.from_messages(
            [
                h.Message.from_role_and_content(h.Role.SYSTEM, system),
                h.Message.from_role_and_content(h.Role.DEVELOPER, developer),
                *messages,
            ]
        ),
        h.Role.ASSISTANT,
        h.RenderConversationConfig(auto_drop_analysis=False),
    )


def _completion(encoding: HarmonyEncoding, messages: Sequence[Message]) -> list[int]:
    ids = encoding.render_conversation(
        h.Conversation.from_messages(messages), h.RenderConversationConfig(auto_drop_analysis=False)
    )
    header = encoding.encode("<|start|>assistant", allowed_special="all")
    assert ids[: len(header)] == header
    return ids[len(header) :]


@pytest.mark.parametrize("rounds,followup", [(1, False), (2, False), (1, True), (2, True)])
def test_delta_parity(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding, rounds: int, followup: bool
) -> None:
    history = [h.Message.from_role_and_content(h.Role.USER, QUESTION)]
    msgs = [Msg("user", QUESTION)]
    buffer = renderer.initial(SYSTEM, TOOLS, msgs)
    assert buffer == _reference_prompt(encoding, history)
    for i in range(rounds):
        thinking = f"Round {i}: 2+2=4."
        answer = str(4 + i)
        result = f"Round {i} received."
        assistant = [
            h.Message.from_role_and_content(h.Role.ASSISTANT, thinking).with_channel("analysis"),
            h.Message.from_role_and_content(h.Role.ASSISTANT, '{"answer": "' + answer + '"}')
            .with_channel("commentary")
            .with_recipient("functions.submit")
            .with_content_type("<|constrain|>json"),
        ]
        completion = _completion(encoding, assistant)
        buffer += completion
        new_msgs = [Msg("tool", result, name="submit")]
        history += [
            *assistant,
            h.Message.from_author_and_content(h.Author.new(h.Role.TOOL, "functions.submit"), result)
            .with_recipient("assistant")
            .with_channel("commentary"),
        ]
        if followup and i == rounds - 1:
            new_msgs.append(Msg("user", "Please check again."))
            history.append(h.Message.from_role_and_content(h.Role.USER, "Please check again."))
        buffer += renderer.continuation(renderer.parse(completion, TOOLS).termination, new_msgs)
        assert buffer == _reference_prompt(encoding, history)
        msgs += [
            Msg(
                "assistant",
                "",
                thinking=thinking,
                tool_calls=(ToolCall("submit", {"answer": answer}),),
            ),
            *new_msgs,
        ]
        assert renderer.initial(SYSTEM, TOOLS, msgs) == buffer


def test_parse_thinking_tools_and_visible_content(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding
) -> None:
    ids = encoding.encode(
        "<|channel|>analysis<|message|>First. <|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>Second.<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Checking. <|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.submit "
        '<|constrain|>json<|message|>{"answer": "4", "n": 4}<|call|>',
        allowed_special="all",
    )
    turn = renderer.parse(ids, TOOLS)
    assert turn.termination == "stop"
    assert turn.thinking == "First. Second."
    assert turn.content == "Checking. "
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.ok and call.name == "submit" and call.arguments == {"answer": "4", "n": 4}
    assert call.raw == '{"answer": "4", "n": 4}'
    assert call.id is None


def test_parse_final(renderer: HarmonyRenderer, encoding: HarmonyEncoding) -> None:
    ids = encoding.encode(
        "<|channel|>analysis<|message|>2+2=4.<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Answer: <|end|>"
        "<|start|>assistant<|channel|>final<|message|>4<|return|>",
        allowed_special="all",
    )
    turn = renderer.parse(ids)
    assert turn.termination == "stop"
    assert turn.thinking == "2+2=4."
    assert turn.content == "Answer: 4"
    assert turn.tool_calls == ()


@pytest.mark.parametrize("raw", ["not json", "[]", '"4"', "null", '{"answer":'])
def test_bad_tool_arguments_are_retained(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding, raw: str
) -> None:
    ids = encoding.encode(
        "<|channel|>commentary to=functions.submit <|constrain|>json<|message|>" + raw + "<|call|>",
        allowed_special="all",
    )
    turn = renderer.parse(ids)
    assert turn.termination == "stop"
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert not call.ok and call.name == "submit" and call.arguments is None and call.raw == raw


@pytest.mark.parametrize("ending", ["", "<|end|>"])
def test_length_truncation(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding, ending: str
) -> None:
    ids = encoding.encode(
        "<|channel|>analysis<|message|>Still thinking" + ending, allowed_special="all"
    )
    turn = renderer.parse(ids)
    assert turn.termination == "length"
    assert turn.thinking == "Still thinking"
    assert turn.content == ""


def test_length_truncated_tool(renderer: HarmonyRenderer, encoding: HarmonyEncoding) -> None:
    ids = encoding.encode(
        '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"answer": "',
        allowed_special="all",
    )
    turn = renderer.parse(ids)
    assert turn.termination == "length"
    assert not turn.tool_calls[0].ok
    assert turn.tool_calls[0].raw == '{"answer": "'


@pytest.mark.parametrize("ids", [[], [-1], [999999999], [200006, 200006]])
def test_garbage_never_raises(renderer: HarmonyRenderer, ids: list[int]) -> None:
    assert renderer.parse(ids).termination == "malformed"


@pytest.mark.parametrize("ending", ["", "<|return|>"])
def test_malformed_recovers_thinking_and_only_closes_without_stop(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding, ending: str
) -> None:
    ids = encoding.encode(
        "<|channel|>analysis<|message|>Keep this.<|end|>garbage" + ending, allowed_special="all"
    )
    turn = renderer.parse(ids)
    assert turn.termination == "malformed"
    assert turn.thinking == "Keep this."
    delta = renderer.continuation(turn.termination, [Msg("user", "Continue.")])
    assert renderer.decode(delta).startswith("<|start|>user" if ending else "<|end|><|start|>user")


@pytest.mark.parametrize("channel", ["analysis", "final"])
def test_malformed_recovers_partial_message(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding, channel: str
) -> None:
    ids = encoding.encode(f"<|channel|>{channel}<|message|>Keep this.", allowed_special="all")
    turn = renderer.parse([*ids, -1])
    assert turn.termination == "malformed"
    assert turn.thinking == ("Keep this." if channel == "analysis" else None)
    assert turn.content == ("Keep this." if channel == "final" else "")


def test_forced_tool_prefix(renderer: HarmonyRenderer, encoding: HarmonyEncoding) -> None:
    renderer.initial(SYSTEM, TOOLS, [])
    prefix = renderer.forced_tool_prefix("submit")
    assert renderer.decode(prefix) == (
        '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"answer": "'
    )
    turn = renderer.parse(prefix + encoding.encode('4"}<|call|>', allowed_special="all"), TOOLS)
    assert turn.termination == "stop" and len(turn.tool_calls) == 1
    assert turn.tool_calls[0].ok
    assert turn.tool_calls[0].name == "submit"
    assert turn.tool_calls[0].arguments == {"answer": "4"}


def test_forced_prefix_uses_first_required_parameter(renderer: HarmonyRenderer) -> None:
    tool = ToolSpec(
        "custom",
        "Custom.",
        {
            "type": "object",
            "properties": {"optional": {"type": "string"}, "chosen": {"type": "string"}},
            "required": ["chosen"],
        },
    )
    renderer.initial(None, [tool], [])
    assert renderer.decode(renderer.forced_tool_prefix("custom")).endswith('{"chosen": "')
    with pytest.raises(ValueError, match="not advertised"):
        renderer.forced_tool_prefix("submit")
    renderer.initial(None, [ToolSpec("custom", "Custom.", {"type": "object"})], [])
    with pytest.raises(ValueError, match="required parameter"):
        renderer.forced_tool_prefix("custom")


def test_length_continuation_closes_open_message(renderer: HarmonyRenderer) -> None:
    assert renderer.decode(renderer.continuation("length", [])).startswith(
        "<|end|><|start|>assistant"
    )
    assert renderer.decode(renderer.continuation("stop", [])) == "<|start|>assistant"


def test_plain_text_encoding_and_special_decoding(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding
) -> None:
    text = "Literal <|call|> and <|end|>, café 世界"
    ids = renderer.encode_text(text)
    assert renderer.decode(ids) == text
    assert not any(encoding.is_special_token(token) for token in ids)
    assert renderer.decode(renderer.stop_token_ids) == "<|return|><|call|>"


def test_profiles_and_fingerprint(encoding: HarmonyEncoding) -> None:
    assert set(HARMONY_RENDERERS) == {"gpt_oss_low", "gpt_oss_medium", "gpt_oss_high"}
    expected_sha = hashlib.sha256(f"{encoding.name}:201089".encode()).hexdigest()
    for effort in ("low", "medium", "high"):
        renderer = HARMONY_RENDERERS[f"gpt_oss_{effort}"]()
        assert isinstance(renderer, DeltaRenderer)
        assert renderer.name == f"gpt_oss_{effort}"
        assert renderer.supports_delta and renderer.delivery_role == "append_tool"
        assert renderer.stop_token_ids == (200002, 200012)
        assert renderer.tokenizer_sha == expected_sha
        prompt = renderer.initial(SYSTEM, TOOLS, [Msg("user", QUESTION)])
        assert f"Reasoning: {effort}" in renderer.decode(prompt)
        assert "Current date: 2026-09-23" in renderer.decode(prompt)
        assert prompt == renderer.initial(SYSTEM, TOOLS, [Msg("user", QUESTION)])
    changed = HARMONY_RENDERERS["gpt_oss_low"](conversation_start_date="2026-10-01")
    assert "Current date: 2026-10-01" in changed.decode(changed.initial(None, [], []))
    assert changed.tokenizer_sha == expected_sha


def test_bad_inputs_fail_loudly(renderer: HarmonyRenderer) -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        HarmonyRenderer("invalid")
    with pytest.raises(ValueError, match="tool name"):
        renderer.continuation("stop", [Msg("tool", "nameless")])
    with pytest.raises(ValueError, match="message role"):
        renderer.initial(None, [], [Msg("invalid", "content")])


def test_import_is_light() -> None:
    code = (
        "import sys; import marli.render.harmony; "
        "assert 'openai_harmony' not in sys.modules; "
        "assert not {'torch', 'transformers', 'tinker', 'vllm', 'datasets'} & sys.modules.keys()"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )


def test_initial_matches_cached_hf_template(renderer: HarmonyRenderer) -> None:
    hub = pytest.importorskip("huggingface_hub")
    cached = hub.try_to_load_from_cache("openai/gpt-oss-20b", "tokenizer_config.json")
    if not isinstance(cached, str):
        pytest.skip("gpt-oss tokenizer is not cached locally")
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b", local_files_only=True
        )
    except OSError:
        pytest.skip("gpt-oss tokenizer cache is incomplete")
    expected = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in TOOLS
        ],
        reasoning_effort="medium",
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    actual = renderer.initial(SYSTEM, TOOLS, [Msg("user", QUESTION)])
    if actual != expected:
        end = tokenizer.convert_tokens_to_ids("<|end|>")
        assert actual[actual.index(end) + 1 :] == expected[expected.index(end) + 1 :]
        pytest.skip("Only system header differs: Harmony pins the conversation date")
    assert actual == expected


def test_suppress_thinking_prefix_opens_final_channel(
    renderer: HarmonyRenderer, encoding: HarmonyEncoding
) -> None:
    prefix = renderer.suppress_thinking_prefix()
    completion = encoding.encode("Summary text.<|return|>", allowed_special="all")
    turn = renderer.parse(prefix + completion)
    assert turn.content == "Summary text." and turn.thinking is None
