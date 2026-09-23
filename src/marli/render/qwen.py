"""Parse Qwen's native tool formats without coercing schema-declared strings.

Qwen3.5 prefills the thinking opener, whereas Qwen3 can sample it. The parser
therefore consumes only the thinking syntax that belongs to the completion.
"""

from __future__ import annotations

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
    rounds). Two known deviations: (1) after a *user* message (a nudge), the HF
    templates strip earlier reasoning while the buffer keeps it — a bounded,
    documented shift that keeps "datums use exactly the ids the policy saw";
    (2) ``qwen3_nothink``'s template deletes the already-sampled empty think
    prefill before tool results, which no append-only buffer can reproduce, so
    that profile re-renders every call.
    """
    return TemplateProfile(
        chat_template_kwargs={"enable_thinking": thinking},
        stop_token_ids=(tokenizer.convert_tokens_to_ids("<|im_end|>"),),
        delivery_role="tool",
        close_turn="<|im_end|>",
        parser=partial(_parse, xml=xml, prefilled_thinking=xml and thinking),
        tool_prefix=partial(_tool_prefix, xml=xml, prefilled_thinking=xml and thinking),
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


def _tool_prefix(tool_name: str, first_param: str, *, xml: bool, prefilled_thinking: bool) -> str:
    if xml:
        close_thinking = "\n</think>\n\n" if prefilled_thinking else ""
        return f"{close_thinking}<tool_call>\n<function={tool_name}>\n<parameter={first_param}>\n"
    return (
        '<tool_call>\n{"name": '
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
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("name"), str)
        or obj["name"] not in specs
        or not isinstance(obj.get("arguments"), dict)
    ):
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    return ParsedToolCall(name=obj["name"], arguments=obj["arguments"], raw=raw, ok=True)


def _parse_xml(body: str, raw: str, specs: dict[str, ToolSpec]) -> ParsedToolCall:
    function = _FUNCTION.fullmatch(body)
    if function is None or function[1] not in specs:
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    name, parameters = function.groups()
    properties = specs[name].parameters.get("properties", {})
    arguments: dict[str, Any] = {}
    position = 0
    while position < len(parameters) and parameters[position:].strip():
        parameter = _PARAMETER.match(parameters, position)
        if parameter is None or parameter[1] in arguments:
            return ParsedToolCall(name=name, arguments=None, raw=raw, ok=False)
        param_name, value = parameter.groups()
        value = value.removeprefix("\n").removesuffix("\n")
        if properties.get(param_name, {}).get("type") == "string":
            arguments[param_name] = value
        else:
            try:
                arguments[param_name] = json.loads(value)
            except json.JSONDecodeError:
                arguments[param_name] = value
        position = parameter.end()
    return ParsedToolCall(name=name, arguments=arguments, raw=raw, ok=True)
