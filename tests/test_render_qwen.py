"""Use cached HF templates as independent oracles for rendering and parsing.

Some template history rewrites cannot be reproduced by an append-only buffer;
the parity cases exercise those boundaries as well as ordinary tool delivery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from marli.errors import ConfigError
from marli.render.base import DeltaRenderer, Msg, ToolCall, ToolSpec
from marli.render.fake import FakeRenderer
from marli.render.hf import HFTemplateRenderer
from marli.render.registry import get_renderer

transformers = pytest.importorskip("transformers")

NAMES = ("qwen3_5", "qwen3_5_nothink", "qwen3", "qwen3_nothink")
TOOLS = (
    ToolSpec(
        "submit",
        "Submit the answer.",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["answer", "count"],
        },
    ),
    ToolSpec(
        "lookup",
        "Look up a note.",
        {
            "type": "object",
            "properties": {"optional": {"type": "string"}, "key": {"type": "string"}},
            "required": ["key"],
        },
    ),
)
HF_TOOLS = [
    {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
    }
    for t in TOOLS
]
SYSTEM = "You are a helpful peer."
USER = Msg("user", "Compute two plus two.")
HF_START = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.content}]


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


@pytest.fixture(scope="module", params=NAMES)
def renderer(request: pytest.FixtureRequest) -> HFTemplateRenderer:
    try:
        result = get_renderer(request.param)
    except OSError as exc:
        pytest.skip(f"Cached tokenizer unavailable offline: {exc}")
    assert isinstance(result, HFTemplateRenderer)
    return result


def hf_text(
    renderer: HFTemplateRenderer, messages: list[dict[str, Any]], *, generation: bool
) -> str:
    return renderer.tokenizer.apply_chat_template(
        messages,
        tools=HF_TOOLS,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=not renderer.name.endswith("_nothink"),
    )


def assistant_turn(renderer: HFTemplateRenderer, calls: int = 1) -> tuple[Msg, dict[str, Any]]:
    thinking = "I will use a tool." if not renderer.name.endswith("_nothink") else ""
    tool_calls = (
        ToolCall("submit", {"answer": "4", "count": 2}, id="call-1"),
        ToolCall("lookup", {"key": "next"}, id="call-2"),
    )[:calls]
    msg = Msg("assistant", "Ready.", thinking=thinking, tool_calls=tool_calls)
    hf = {
        "role": "assistant",
        "content": msg.content,
        "reasoning_content": thinking,
        "tool_calls": [
            {"type": "function", "function": {"name": c.name, "arguments": c.arguments}, "id": c.id}
            for c in tool_calls
        ],
    }
    return msg, hf


def canonical_completion(
    renderer: HFTemplateRenderer, history: list[dict[str, Any]], assistant: dict[str, Any]
) -> list[int]:
    prompt = hf_text(renderer, history, generation=True)
    full = hf_text(renderer, [*history, assistant], generation=False)
    assert full.startswith(prompt)
    end = full.index("<|im_end|>", len(prompt)) + len("<|im_end|>")
    completion = full[len(prompt) : end]
    ids = renderer.tokenizer.encode(completion, add_special_tokens=False)
    assert ids[-1] == renderer.stop_token_ids[0]
    # The completion itself is sliced from canonical template output, never hand-built.
    assert renderer.encode_text(prompt) + ids == renderer.encode_text(full[:end])
    return ids


def new_messages(case: str) -> tuple[list[Msg], list[dict[str, Any]]]:
    msgs = [Msg("tool", "accepted", name="submit", tool_call_id="call-1")]
    if case == "two_results":
        msgs.append(Msg("tool", "found", name="lookup", tool_call_id="call-2"))
    msgs.append(Msg("tool", "[notify] peer1 wrote a note", name="delivery"))
    if case == "then_user":
        msgs.append(Msg("user", "Explain the result."))
    hf = [
        {"role": m.role, "content": m.content, "name": m.name, "tool_call_id": m.tool_call_id}
        for m in msgs
    ]
    return msgs, hf


@pytest.mark.parametrize("case", ["one_result", "two_results"])
def test_delta_parity(renderer: HFTemplateRenderer, case: str) -> None:
    _, assistant = assistant_turn(renderer, calls=2 if case == "two_results" else 1)
    msgs, hf_msgs = new_messages(case)
    initial = renderer.initial(SYSTEM, TOOLS, [USER])
    completion = canonical_completion(renderer, HF_START, assistant)
    delta = renderer.continuation("stop", msgs)
    canonical = hf_text(renderer, [*HF_START, assistant, *hf_msgs], generation=True)
    buf = initial + completion + delta
    expected = renderer.tokenizer.encode(canonical, add_special_tokens=False)
    if renderer.name == "qwen3_nothink":
        # The HF template removes this *already emitted* prefill before tool results.
        assert not renderer.supports_delta and buf != expected
        assert renderer.decode(buf).replace("<think>\n\n</think>\n\n", "", 1) == canonical
    else:
        assert buf == expected


def test_user_message_rewrites_history(renderer: HFTemplateRenderer) -> None:
    _, assistant = assistant_turn(renderer)
    msgs, hf_msgs = new_messages("then_user")
    buf = renderer.initial(SYSTEM, TOOLS, [USER])
    buf += canonical_completion(renderer, HF_START, assistant)
    buf += renderer.continuation("stop", msgs)
    full = hf_text(renderer, [*HF_START, assistant, *hf_msgs], generation=True)
    assert buf != renderer.tokenizer.encode(full, add_special_tokens=False)
    thinking = "<think>\n" + assistant["reasoning_content"] + "\n</think>\n\n"
    # Pin the exact rewrite; simply appending the new messages cannot reproduce it.
    assert renderer.decode(buf).replace(thinking, "", 1) == full
    # Known, documented deviation: the buffer keeps earlier reasoning after a user message.
    assert renderer.supports_delta == (renderer.name != "qwen3_nothink")


def test_second_round_delta_parity(renderer: HFTemplateRenderer) -> None:
    _, assistant = assistant_turn(renderer, calls=2)
    msgs, hf_msgs = new_messages("two_results")
    history = [*HF_START, assistant, *hf_msgs]
    buf = renderer.initial(SYSTEM, TOOLS, [USER])
    buf += canonical_completion(renderer, HF_START, assistant)
    buf += renderer.continuation("stop", msgs)
    _, second = assistant_turn(renderer)
    second["content"] = "Checked."
    buf += canonical_completion(renderer, history, second)
    buf += renderer.continuation("stop", msgs)
    full = hf_text(renderer, [*history, second, *hf_msgs], generation=True)
    expected = renderer.tokenizer.encode(full, add_special_tokens=False)
    if renderer.name == "qwen3_nothink":
        assert not renderer.supports_delta and buf != expected
        assert renderer.decode(buf).replace("<think>\n\n</think>\n\n", "", 2) == full
    else:
        assert buf == expected


def test_initial_converts_assistant_history(renderer: HFTemplateRenderer) -> None:
    msg, assistant = assistant_turn(renderer, calls=2)
    msgs, hf_msgs = new_messages("two_results")
    full = hf_text(renderer, [*HF_START, assistant, *hf_msgs], generation=True)
    assert renderer.initial(SYSTEM, TOOLS, [USER, msg, *msgs]) == renderer.encode_text(full)


@pytest.mark.parametrize("system", [None, ""])
def test_initial_without_tools(renderer: HFTemplateRenderer, system: str | None) -> None:
    messages = [{"role": "user", "content": USER.content}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    text = renderer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not renderer.name.endswith("_nothink"),
    )
    assert renderer.initial(system, (), [USER]) == renderer.encode_text(text)


def test_grouped_results_and_delivery(renderer: HFTemplateRenderer) -> None:
    msgs, _ = new_messages("two_results")
    text = renderer.decode(renderer.continuation("stop", msgs))
    assert text.startswith("\n<|im_start|>user\n<tool_response>\n")
    assert text.count("<|im_start|>user") == 1
    assert text.count("<tool_response>") == 3
    assert "[notify] peer1 wrote a note" in text


@pytest.mark.parametrize("termination", ["length", "malformed"])
def test_continuation_closes_unfinished_turn(
    renderer: HFTemplateRenderer, termination: str
) -> None:
    msgs = [Msg("tool", "continue")]
    close = renderer.encode_text("<|im_end|>")
    assert renderer.continuation(termination, msgs) == close + renderer.continuation("stop", msgs)


def test_parse_canonical_completion(renderer: HFTemplateRenderer) -> None:
    _, assistant = assistant_turn(renderer, calls=2)
    turn = renderer.parse(canonical_completion(renderer, HF_START, assistant), TOOLS)
    assert turn.termination == "stop"
    assert turn.content == "Ready."
    assert turn.thinking == (None if renderer.name.endswith("_nothink") else "I will use a tool.")
    assert [(c.name, c.arguments, c.ok) for c in turn.tool_calls] == [
        ("submit", {"answer": "4", "count": 2}, True),
        ("lookup", {"key": "next"}, True),
    ]
    assert type(turn.tool_calls[0].arguments["answer"]) is str
    assert type(turn.tool_calls[0].arguments["count"]) is int


def content_completion(
    renderer: HFTemplateRenderer, content: str, *, stop: bool = True
) -> list[int]:
    if renderer.name == "qwen3_5":
        content = "\n</think>\n\n" + content
    return renderer.encode_text(content + ("<|im_end|>" if stop else ""))


@pytest.mark.parametrize("stopped", [False, True])
def test_parse_unclosed_call(renderer: HFTemplateRenderer, stopped: bool) -> None:
    raw = "<tool_call>\nunfinished"
    turn = renderer.parse(content_completion(renderer, "before " + raw, stop=stopped), TOOLS)
    assert turn.content == "before"
    assert turn.termination == "malformed"
    assert len(turn.tool_calls) == 1
    assert not turn.tool_calls[0].ok and turn.tool_calls[0].raw == raw


@pytest.mark.parametrize("failure", ["bad_body", "unknown", "missing_close", "bad_arguments"])
def test_parse_invalid_calls(renderer: HFTemplateRenderer, failure: str) -> None:
    xml = renderer.name.startswith("qwen3_5")
    bodies = {
        "bad_body": "not a tool call",
        "unknown": (
            "<function=unknown></function>" if xml else '{"name":"unknown","arguments":{}}'
        ),
        "missing_close": (
            "<function=submit><parameter=answer>\n4\n</parameter>"
            if xml
            else '{"name":"submit","arguments":{'
        ),
        "bad_arguments": (
            "<function=submit><parameter=answer>4</function>"
            if xml
            else '{"name":"submit","arguments":"4"}'
        ),
    }
    raw = "<tool_call>" + bodies[failure] + "</tool_call>"
    turn = renderer.parse(content_completion(renderer, "before " + raw + " after"), TOOLS)
    assert turn.content == "before  after"
    assert turn.termination == "stop"
    assert len(turn.tool_calls) == 1
    assert not turn.tool_calls[0].ok and turn.tool_calls[0].raw == raw


def test_truncated_thinking(renderer: HFTemplateRenderer) -> None:
    if renderer.name.endswith("_nothink"):
        pytest.skip("No open thinking block in this profile")
    text = "working <tool_call>is still reasoning"
    if renderer.name == "qwen3":
        text = "<think>\n" + text
    turn = renderer.parse(renderer.encode_text(text), TOOLS)
    assert turn.thinking == "working <tool_call>is still reasoning"
    assert turn.content == "" and turn.tool_calls == () and turn.termination == "length"


def test_qwen3_optional_thinking(renderer: HFTemplateRenderer) -> None:
    if renderer.name.startswith("qwen3_5"):
        pytest.skip("Qwen3-specific hybrid thinking")
    turn = renderer.parse(renderer.encode_text("  <think> reason </think> answer <|im_end|>"))
    assert turn.thinking == "reason" and turn.content == "answer"
    plain = renderer.parse(renderer.encode_text("answer<|im_end|>"))
    assert plain.thinking is None and plain.content == "answer"


def test_schema_aware_xml_values(renderer: HFTemplateRenderer) -> None:
    if not renderer.name.startswith("qwen3_5"):
        pytest.skip("XML-specific parameter typing")
    values = {
        "answer": "\n 4 \n",
        "count": "4",
        "obj": '{"a":[1,true,null]}',
        "fallback": "not json",
    }
    body = (
        "<tool_call>\n<function=submit>\n"
        + "".join(f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in values.items())
        + "</function>\n</tool_call>"
    )
    turn = renderer.parse(content_completion(renderer, body), TOOLS)
    assert turn.tool_calls[0].ok
    assert turn.tool_calls[0].arguments == {
        "answer": "\n 4 \n",
        "count": 4,
        "obj": {"a": [1, True, None]},
        "fallback": "not json",
    }


def test_forced_submit_prefix(renderer: HFTemplateRenderer) -> None:
    renderer.initial(SYSTEM, TOOLS, [USER])
    prefix = renderer.forced_tool_prefix("submit")
    if renderer.name.startswith("qwen3_5"):
        suffix = "4\n</parameter>\n</function>\n</tool_call><|im_end|>"
        expected = "<tool_call>\n<function=submit>\n<parameter=answer>\n"
        if renderer.name == "qwen3_5":
            expected = "\n</think>\n\n" + expected
    else:
        suffix = '4"}}\n</tool_call><|im_end|>'
        expected = '<tool_call>\n{"name": "submit", "arguments": {"answer": "'
    assert renderer.decode(prefix) == expected
    turn = renderer.parse(prefix + renderer.encode_text(suffix), TOOLS)
    assert turn.content == "" and turn.termination == "stop"
    assert turn.tool_calls[0].ok
    assert turn.tool_calls[0].name == "submit"
    assert turn.tool_calls[0].arguments == {"answer": "4"}


def test_forced_prefix_uses_first_required_and_explicit_parameter(
    renderer: HFTemplateRenderer,
) -> None:
    renderer.initial(SYSTEM, TOOLS, [USER])
    prefix = renderer.decode(renderer.forced_tool_prefix("lookup"))
    explicit = renderer.decode(renderer.forced_tool_prefix("lookup", "optional"))
    if renderer.name.startswith("qwen3_5"):
        assert prefix.endswith("<parameter=key>\n")
        assert explicit.endswith("<parameter=optional>\n")
    else:
        assert prefix.endswith('{"key": "')
        assert explicit.endswith('{"optional": "')


def test_lineage_tool_state_is_not_shared(renderer: HFTemplateRenderer) -> None:
    other = get_renderer(renderer.name)
    assert other is not renderer and other.tokenizer is renderer.tokenizer
    renderer.initial(SYSTEM, TOOLS, [USER])
    other.initial(SYSTEM, (), [USER])
    assert renderer.forced_tool_prefix("submit")
    with pytest.raises(ConfigError, match="unknown tool"):
        other.forced_tool_prefix("submit")
    renderer.initial(SYSTEM, (ToolSpec("empty", "No params", {}),), [USER])
    with pytest.raises(ConfigError, match="required parameter"):
        renderer.forced_tool_prefix("empty")


def test_protocol_and_tokenizer_roundtrip(renderer: HFTemplateRenderer) -> None:
    assert isinstance(renderer, DeltaRenderer)
    assert renderer.delivery_role == "tool"
    text = "Unicode: λ → 4.\n<|im_end|>"
    ids = renderer.encode_text(text)
    assert renderer.decode(ids) == text
    assert renderer.encode_text(renderer.decode(ids)) == ids
    assert ids[-1] in renderer.stop_token_ids
    assert renderer.parse([]).termination == "length"
    text_turn = renderer.parse(content_completion(renderer, " short ", stop=False))
    assert text_turn.content == "short" and text_turn.termination == "length"


def subprocess_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_tokenizer_sha_stable(renderer: HFTemplateRenderer) -> None:
    assert len(renderer.tokenizer_sha) == 16
    int(renderer.tokenizer_sha, 16)
    assert renderer.tokenizer_sha == get_renderer(renderer.name).tokenizer_sha


def test_tokenizer_sha_across_processes() -> None:
    # Check a fresh interpreter with a different Python hash seed.
    hashes: dict[str, str] = {}
    for name in NAMES:
        try:
            hashes[name] = get_renderer(name).tokenizer_sha
        except OSError:
            continue
    if not hashes:
        pytest.skip("No cached Qwen tokenizers available offline")
    code = (
        "import json; "
        "from marli.render.registry import get_renderer; "
        "print(json.dumps({name: get_renderer(name).tokenizer_sha "
        f"for name in {list(hashes)!r}}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**subprocess_env(), "PYTHONHASHSEED": "17"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == hashes


def test_registry_unknown_and_fake() -> None:
    assert isinstance(get_renderer("fake"), FakeRenderer)
    with pytest.raises(ConfigError, match="Unknown renderer") as exc:
        get_renderer("not-a-renderer")
    assert all(name in str(exc.value) for name in (*NAMES, "fake"))


def test_registry_lazy_without_transformers() -> None:
    code = """
import importlib.abc
import sys

class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'torch', 'tinker_cookbook'}:
            raise ImportError(fullname)

sys.meta_path.insert(0, NoHeavyImports())
from marli.render.registry import get_renderer
assert get_renderer('fake').name == 'fake'
assert 'transformers' not in sys.modules
assert 'torch' not in sys.modules
assert 'tinker_cookbook' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], env=subprocess_env(), check=True)


def test_registry_caches_by_hf_id(
    renderer: HFTemplateRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marli.render import registry

    loads: list[str] = []

    def load(hf_id: str) -> Any:
        loads.append(hf_id)
        return renderer.tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(registry, "_tokenizer", cache(registry._tokenizer.__wrapped__))
    one = get_renderer("qwen3", hf_id="local/test-1")
    two = get_renderer("qwen3_nothink", hf_id="local/test-1")
    three = get_renderer("qwen3", hf_id="local/test-2")
    assert one is not two and two is not three
    assert loads == ["local/test-1", "local/test-2"]


def test_delta_support_per_profile(renderer: HFTemplateRenderer) -> None:
    assert renderer.supports_delta == (renderer.name != "qwen3_nothink")


def test_suppress_thinking_prefix(renderer: HFTemplateRenderer) -> None:
    text = renderer.decode(renderer.suppress_thinking_prefix())
    expected = {
        "qwen3_5": "\n</think>\n\n",
        "qwen3_5_nothink": "",
        "qwen3": "<think>\n\n</think>\n\n",
        "qwen3_nothink": "",
    }[renderer.name]
    assert text == expected
