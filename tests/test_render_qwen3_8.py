"""Pin Qwen3.8's append-only history against its cached, unmodified HF template."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import copy
from typing import Any

import pytest

from marli.render.base import Msg, ToolCall, ToolSpec
from marli.render.hf import HFTemplateRenderer, _tokenizer_sha
from marli.render.registry import get_renderer

pytestmark = pytest.mark.hf
transformers = pytest.importorskip("transformers")

PROFILES = {
    "qwen3_8": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    "qwen3_8_medium": {"enable_thinking": True, "reasoning_effort": "medium"},
    "qwen3_8_low": {"enable_thinking": True, "reasoning_effort": "low"},
    "qwen3_8_nothink": {"enable_thinking": False, "reasoning_effort": "xhigh"},
}
TOOLS = (
    ToolSpec(
        "lookup",
        "Look up a note.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["key", "count"],
        },
    ),
    ToolSpec("ping", "Check readiness.", {"type": "object", "properties": {}}),
)
SYSTEM = "You are a helpful peer."
USER = Msg("user", "Check the notes.")
START = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.content}]


@pytest.fixture(scope="module", autouse=True)
def offline() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        original = transformers.AutoTokenizer.from_pretrained

        def local_tokenizer(hf_id: str, **kwargs: Any) -> Any:
            return original(hf_id, **{**kwargs, "local_files_only": True})

        patch.setattr(transformers.AutoTokenizer, "from_pretrained", local_tokenizer)
        yield


@pytest.fixture(params=PROFILES)
def renderer(request: pytest.FixtureRequest) -> HFTemplateRenderer:
    try:
        result = get_renderer(request.param)
    except OSError as exc:
        pytest.skip(f"Cached Qwen3.8 tokenizer unavailable offline: {exc}")
    assert isinstance(result, HFTemplateRenderer)
    return result


def hf_ids(
    renderer: HFTemplateRenderer,
    messages: list[dict[str, Any]],
    *,
    tools: Sequence[ToolSpec] = TOOLS,
    generation: bool = True,
) -> list[int]:
    return renderer.tokenizer.apply_chat_template(
        messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=generation,
        **PROFILES[renderer.name],
    )


def assistant_turn(
    renderer: HFTemplateRenderer, *, tool_calls: bool = True
) -> tuple[Msg, dict[str, Any], list[int]]:
    thinking = "I will check the notes." if PROFILES[renderer.name]["enable_thinking"] else None
    calls = (
        (ToolCall("lookup", {"key": "4", "count": 2}), ToolCall("ping", {})) if tool_calls else ()
    )
    msg = Msg("assistant", "Ready.", thinking=thinking, tool_calls=calls)
    hf = {
        "role": "assistant",
        "content": msg.content,
        "reasoning_content": thinking or "",
        "tool_calls": [
            {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ],
    }
    text = (thinking + "\n</think>\n\n" if thinking else "") + msg.content
    if tool_calls:
        text += (
            "\n\n<tool_call>\n<function=lookup>\n<parameter=key>\n4\n</parameter>\n"
            "<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        )
    ids = renderer.tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
    return msg, hf, ids


def assert_reasoning_instruction(renderer: HFTemplateRenderer, ids: list[int]) -> None:
    text = renderer.decode(ids)
    effort = PROFILES[renderer.name]["reasoning_effort"]
    # The actual HF template deliberately has no instruction for medium.
    instructed = PROFILES[renderer.name]["enable_thinking"] and effort != "medium"
    assert text.count("Reasoning effort is set to ") == int(instructed)
    if instructed:
        assert text.startswith(f"<|im_start|>system\nReasoning effort is set to {effort}.")


@pytest.mark.parametrize("system", [SYSTEM, None, ""])
@pytest.mark.parametrize("tools", [TOOLS, ()])
def test_initial_parity(
    renderer: HFTemplateRenderer, system: str | None, tools: Sequence[ToolSpec]
) -> None:
    history = [] if system is None else [{"role": "system", "content": system}]
    history.append({"role": "user", "content": USER.content})
    ids = renderer.initial(system, tools, [USER])
    assert ids == hf_ids(renderer, history, tools=tools)
    if renderer.name == "qwen3_8" and not tools:
        assert ids == renderer.tokenizer.apply_chat_template(
            history,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    assert renderer.profile.chat_template_kwargs == PROFILES[renderer.name]
    assert renderer.tokenizer.name_or_path == "Qwen/Qwen3.8-27B"
    assert_reasoning_instruction(renderer, ids)
    suffix = (
        "<think>\n" if PROFILES[renderer.name]["enable_thinking"] else "<think>\n\n</think>\n\n"
    )
    assert renderer.decode(ids).endswith("<|im_start|>assistant\n" + suffix)


@pytest.mark.parametrize("user_after_turn", [False, True])
@pytest.mark.parametrize("stopped", [False, True])
def test_several_rounds_delta_parity(
    renderer: HFTemplateRenderer, user_after_turn: bool, stopped: bool
) -> None:
    history = list(START)
    messages = [USER]
    buf = renderer.initial(SYSTEM, TOOLS, messages)
    for round_index in range(3):
        msg, assistant, completion = assistant_turn(renderer)
        if not stopped:
            completion = completion[:-1]
        turn = renderer.parse(completion)
        assert turn.termination == ("stop" if stopped else "length")
        assert turn.content == msg.content and turn.thinking == msg.thinking
        assert [(c.name, c.arguments, c.ok) for c in turn.tool_calls] == [
            ("lookup", {"key": "4", "count": 2}, True),
            ("ping", {}, True),
        ]
        delivered = [
            Msg("tool", f"note {round_index}", name="lookup"),
            Msg("tool", "ready", name="ping"),
            Msg("tool", "[notify] peer1 wrote a note"),
        ]
        if user_after_turn:
            delivered.append(Msg("user", "Check again."))
        buf += completion + renderer.continuation(turn.termination, delivered)
        history.extend([assistant, *({"role": m.role, "content": m.content} for m in delivered)])
        messages.extend([msg, *delivered])
        assert buf == hf_ids(renderer, history)
        assert buf == renderer.initial(SYSTEM, TOOLS, messages)
        assert_reasoning_instruction(renderer, buf)
    assert renderer.supports_delta


def test_user_delivery_after_plain_turn_preserves_thinking(renderer: HFTemplateRenderer) -> None:
    buf = renderer.initial(SYSTEM, TOOLS, [USER])
    msg, assistant, completion = assistant_turn(renderer, tool_calls=False)
    turn = renderer.parse(completion)
    delivery = Msg("user", "[notify] peer1 wrote a note")
    buf += completion + renderer.continuation(turn.termination, [delivery])
    history = [*START, assistant, {"role": "user", "content": delivery.content}]
    assert buf == hf_ids(renderer, history)
    assert buf == renderer.initial(SYSTEM, TOOLS, [USER, msg, delivery])
    assert renderer.supports_delta
    assert_reasoning_instruction(renderer, buf)


def test_empty_string_arguments_roundtrip(renderer: HFTemplateRenderer) -> None:
    renderer.initial(SYSTEM, TOOLS, [USER])
    msg, assistant, _ = assistant_turn(renderer)
    assistant["tool_calls"] = [{"type": "function", "function": {"name": "ping", "arguments": ""}}]
    prompt = hf_ids(renderer, START)
    full = hf_ids(renderer, [*START, assistant], generation=False)
    assert full[: len(prompt)] == prompt
    end = full.index(renderer.stop_token_ids[0], len(prompt)) + 1
    completion = full[len(prompt) : end]
    assert renderer.decode(completion).endswith(
        "<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>"
    )
    turn = renderer.parse(completion)
    assert turn.termination == "stop" and turn.thinking == msg.thinking
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.ok and call.name == "ping" and call.arguments == {}
    parsed_msg = Msg(
        "assistant", turn.content, thinking=turn.thinking, tool_calls=(ToolCall("ping", {}),)
    )
    delivery = Msg("tool", "ready", name="ping")
    history = [*START, assistant, {"role": "tool", "content": delivery.content}]
    expected = hf_ids(renderer, history)
    assert prompt + completion + renderer.continuation("stop", [delivery]) == expected
    assert renderer.initial(SYSTEM, TOOLS, [USER, parsed_msg, delivery]) == expected


@pytest.mark.parametrize("stage", ["thinking", "content", "tool"])
def test_length_truncated_completion(renderer: HFTemplateRenderer, stage: str) -> None:
    thinking = PROFILES[renderer.name]["enable_thinking"]
    if stage == "thinking" and not thinking:
        pytest.skip("Non-thinking profile has no open reasoning block")
    prefix = "Reasoning.\n</think>\n\n" if thinking else ""
    text = {
        "thinking": "Still thinking <tool_call> is reasoning",
        "content": prefix + "Partial answer",
        "tool": prefix + "<tool_call>\n<function=lookup>\n<parameter=key>\n4",
    }[stage]
    ids = renderer.tokenizer.encode(text, add_special_tokens=False)
    turn = renderer.parse(ids, TOOLS)
    if stage == "thinking":
        assert turn.thinking == text and turn.content == "" and not turn.tool_calls
    else:
        assert turn.thinking == ("Reasoning." if thinking else None)
        assert turn.content == ("Partial answer" if stage == "content" else "")
    if stage == "tool":
        assert len(turn.tool_calls) == 1 and not turn.tool_calls[0].ok
        assert turn.tool_calls[0].raw == text.removeprefix(prefix)
    assert turn.termination == ("malformed" if stage == "tool" else "length")
    delta = renderer.continuation("length", [Msg("user", "Continue.")])
    assert delta[0] == renderer.tokenizer.convert_tokens_to_ids("<|im_end|>")


def test_tokenizer_fingerprint_includes_template(renderer: HFTemplateRenderer) -> None:
    assert renderer.tokenizer_sha == _tokenizer_sha(renderer.tokenizer)
    assert renderer.tokenizer_sha == get_renderer(renderer.name).tokenizer_sha
    changed = copy(renderer.tokenizer)
    changed.chat_template += "{# changed template #}"
    assert changed.get_vocab() == renderer.tokenizer.get_vocab()
    assert _tokenizer_sha(changed) != renderer.tokenizer_sha


def test_qwen38_fingerprint_differs_from_qwen36(renderer: HFTemplateRenderer) -> None:
    try:
        older = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.6-27B", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"Cached Qwen3.6 tokenizer unavailable offline: {exc}")
    assert older.get_vocab() == renderer.tokenizer.get_vocab()
    assert older.chat_template != renderer.tokenizer.chat_template
    assert _tokenizer_sha(older) != renderer.tokenizer_sha
