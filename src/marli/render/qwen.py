"""Parse Qwen's native tool formats without coercing schema-declared strings.

Qwen3.5 prefills the thinking opener, whereas Qwen3 can sample it. The parser
therefore consumes only the thinking syntax that belongs to the completion.

For Qwen3.5, suppressing thinking or forcing a tool call closes an already
buffered ``<think>\n``. The resulting empty block is text-equal to the native
non-thinking form, but has two separately encoded newline ids instead of the
native merged double-newline id. Sampled/prefilled ids cannot be rewritten.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from marli.render.base import ParsedToolCall, ParsedTurn, ToolSpec
from marli.render.hf import TemplateProfile

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_FUNCTION = re.compile(r"\s*<function=([^<>\s]+)>(.*?)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"\s*<parameter=([^<>\s]+)>(.*?)</parameter>", re.DOTALL)


def qwen_profile(
    tokenizer: PreTrainedTokenizerBase, *, xml: bool, thinking: bool
) -> TemplateProfile:
    """Delta mode for every profile whose tool-loop parity holds.

    Within one user query (the agent loop: completions followed by tool results
    and workspace deliveries, which the runtime renders as tool messages), the
    append-only buffer is token-identical to the canonical HF template for
    ``qwen3_5``, ``qwen3_5_nothink`` and ``qwen3`` (tests pin this over several
    rounds). Known deviations: after a user message, the HF templates strip
    earlier reasoning while the buffer keeps it. This is common: every harness
    instruction and every delivery after a turn without tool calls is a user
    message. Qwen3 also drops empty sampled reasoning blocks on non-last turns,
    so its parity depends on non-empty reasoning. ``qwen3_nothink`` deletes the
    already-sampled empty think prefill before tool results, which no append-only
    buffer can reproduce, so that profile re-renders every call.
    """
    return TemplateProfile(
        chat_template_kwargs={"enable_thinking": thinking},
        stop_token_ids=tuple(
            tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")
        ),
        delivery_role="tool",
        close_turn="<|im_end|>",
        parser=partial(_parse, xml=xml, prefilled_thinking=xml and thinking),
        tool_prefix=partial(_tool_prefix, xml=xml, thinking=thinking),
        supports_delta=xml or thinking,
        suppress_thinking=_suppress_thinking(xml=xml, thinking=thinking),
    )


def _suppress_thinking(*, xml: bool, thinking: bool) -> str:
    """Observation text that skips reasoning in the template's own non-thinking form."""
    if not thinking:
        return ""  # the non-thinking generation prompt already carries the empty block
    if xml:
        return "\n</think>\n\n"  # Qwen3.5 prefills "<think>\n": close it
    return "<think>\n\n</think>\n\n"  # Qwen3 non-thinking form


def _tool_prefix(tool_name: str, first_param: str, *, xml: bool, thinking: bool) -> str:
    skip_thinking = _suppress_thinking(xml=xml, thinking=thinking)
    if xml:
        return f"{skip_thinking}<tool_call>\n<function={tool_name}>\n<parameter={first_param}>\n"
    return (
        skip_thinking
        + '<tool_call>\n{"name": '
        + json.dumps(tool_name)
        + ', "arguments": {'
        + json.dumps(first_param)
        + ': "'
    )


def _parse(
    text: str,
    tools: Sequence[ToolSpec],
    termination: str,
    *,
    xml: bool,
    prefilled_thinking: bool,
) -> ParsedTurn:
    thinking: str | None = None
    if prefilled_thinking or text.lstrip().startswith("<think>"):
        if not prefilled_thinking:
            text = text.lstrip().removeprefix("<think>")
        thinking_text, close, text = text.partition("</think>")
        thinking = thinking_text.strip()
        if not close:
            return ParsedTurn(content="", thinking=thinking, termination=termination)

    content_parts: list[str] = []
    calls: list[ParsedToolCall] = []
    specs = {tool.name: tool for tool in tools}
    while "<tool_call>" in text:
        before, _, after = text.partition("<tool_call>")
        content_parts.append(before)
        body, close, text = after.partition("</tool_call>")
        raw = "<tool_call>" + body + close
        if not close:
            calls.append(ParsedToolCall(name=None, arguments=None, raw=raw, ok=False))
            termination = "malformed"
            break
        calls.append(_parse_xml(body, raw, specs) if xml else _parse_json(body, raw, specs))
    content_parts.append(text)
    return ParsedTurn(
        content="".join(content_parts).strip(),
        thinking=thinking,
        tool_calls=tuple(calls),
        termination=termination,
    )


def _parse_json(body: str, raw: str, specs: dict[str, ToolSpec]) -> ParsedToolCall:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    name, arguments = obj["name"], obj.get("arguments")
    if not isinstance(arguments, dict):
        return ParsedToolCall(name=name, arguments=None, raw=raw, ok=False)
    ok = name in specs and all(k in arguments for k in specs[name].parameters.get("required", []))
    return ParsedToolCall(name=name, arguments=arguments, raw=raw, ok=ok)


def _parse_xml(body: str, raw: str, specs: dict[str, ToolSpec]) -> ParsedToolCall:
    function = _FUNCTION.fullmatch(body)
    if function is None:
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    name, parameters = function.groups()
    spec = specs.get(name)
    properties = spec.parameters.get("properties", {}) if spec else {}
    arguments: dict[str, Any] = {}
    position = 0
    while position < len(parameters) and parameters[position:].strip():
        parameter = _PARAMETER.match(parameters, position)
        if parameter is None or parameter[1] in arguments:
            return ParsedToolCall(name=name, arguments=None, raw=raw, ok=False)
        param_name, value = parameter.groups()
        value = value.removeprefix("\n").removesuffix("\n")
        arguments[param_name] = _xml_value(value, properties.get(param_name, {}))
        position = parameter.end()
    ok = spec is not None and all(k in arguments for k in spec.parameters.get("required", []))
    return ParsedToolCall(name=name, arguments=arguments, raw=raw, ok=ok)


def _xml_value(value: str, schema: dict[str, Any]) -> Any:
    """Undo the template's string conversion only when the schema asks for it."""
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    alternatives = schema.get("anyOf", [])
    if "string" in kinds or (
        alternatives and all(option.get("type") == "string" for option in alternatives)
    ):
        return value
    if ("null" in kinds or any(option.get("type") == "null" for option in alternatives)) and (
        value in ("None", "null")
    ):
        return None
    if "boolean" in kinds and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if "integer" in kinds or "number" in kinds:
        try:
            return json.loads(value)
        except ValueError:
            try:
                return int(value) if "integer" in kinds else float(value)
            except ValueError:
                return value
    if "object" in kinds or "array" in kinds:
        try:
            return json.loads(value)
        except ValueError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value
