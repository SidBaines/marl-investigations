"""HF templates own the wire format; deltas never rewrite sampled tokens.

The sentinel assistant turn only locates a boundary. For supported profiles,
rendering subsequent messages is independent of that turn's content. Slicing
after its end-of-turn special token keeps BPE merges from crossing the slice
boundary. Generation prefills need not end at special tokens: appending a forced
prefix can prevent a BPE merge across that boundary (see Qwen's empty thinking
block). Templates that rewrite earlier history are still not append-only.

Untrusted content is masked before templating, then restored only inside plain
text gaps. Only template markup can produce special ids. Residual risk:
non-special added tokens such as ``<tool_call>``, ``</tool_response>`` and
``<think>`` are still matched with ``split_special_tokens=True``, exactly as
the HF template tokenizes them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from marli.errors import ConfigError
from marli.render.base import Msg, ParsedTurn, ToolSpec

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TemplateProfile:
    """Family-specific choices around the tokenizer's canonical chat template."""

    chat_template_kwargs: dict[str, Any]
    stop_token_ids: tuple[int, ...]
    delivery_role: str
    close_turn: str
    parser: Callable[[str, Sequence[ToolSpec], str], ParsedTurn]
    tool_prefix: Callable[[str, str], str]
    supports_delta: bool = False
    suppress_thinking: str = ""


class HFTemplateRenderer:
    """A renderer belongs to one agent lineage, even when tokenizers are shared.

    ``initial`` remembers that lineage's tool specs so forced calls can open the
    first required parameter without hard-coding a particular tool's schema.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        name: str,
        profile: TemplateProfile,
        *,
        tokenizer_sha: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.name = name
        self.profile = profile
        self.stop_token_ids = profile.stop_token_ids
        self.delivery_role = profile.delivery_role
        self.supports_delta = profile.supports_delta
        self._tools: tuple[ToolSpec, ...] = ()
        self._last_stop_id: int | None = None
        self._special_ids = {
            token.content: token_id
            for token_id, token in tokenizer.added_tokens_decoder.items()
            if token.special
        }
        self._special_pattern = re.compile(
            "|".join(re.escape(s) for s in sorted(self._special_ids, key=len, reverse=True))
        )
        self.tokenizer_sha = (
            tokenizer_sha if tokenizer_sha is not None else _tokenizer_sha(tokenizer)
        )

    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        self._tools = tuple(tools)
        self._last_stop_id = None
        mask = _ContentMask(self._special_pattern)
        messages = [] if system is None else [{"role": "system", "content": mask.text(system)}]
        messages.extend(_message_dict(msg, mask) for msg in msgs)
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self.profile.chat_template_kwargs,
        )
        return self._encode_markup(text, mask)

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        # A fresh sentinel cannot collide with a user-supplied message or template text.
        sentinel = f"MARLI_DELTA_{uuid4().hex}"
        mask = _ContentMask(self._special_pattern)
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "·"},
                {"role": "assistant", "content": sentinel},
                *(_message_dict(msg, mask) for msg in new_msgs),
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self.profile.chat_template_kwargs,
        )
        sentinel_end = text.index(sentinel) + len(sentinel)
        boundary = text.index(self.profile.close_turn, sentinel_end) + len(self.profile.close_turn)
        delta = text[boundary:]
        close_id = self._special_ids[self.profile.close_turn]
        # With no parsed stop, the caller's stop label supplies the framing state.
        needs_close = self._last_stop_id != close_id and (
            self._last_stop_id is not None or last_termination != "stop"
        )
        if needs_close:
            delta = self.profile.close_turn + delta
        return self._encode_markup(delta, mask)

    def suppress_thinking_prefix(self) -> list[int]:
        """Tokens that skip the reasoning channel for harness-requested calls."""
        text = self.profile.suppress_thinking
        return self._encode_markup(text) if text else []

    def forced_tool_prefix(self, tool_name: str, first_param: str | None = None) -> list[int]:
        if first_param is None:
            tool = next((tool for tool in self._tools if tool.name == tool_name), None)
            if tool is None:
                raise ConfigError(f"Cannot force unknown tool {tool_name!r}; call initial first")
            required = tool.parameters.get("required", [])
            if not required:
                raise ConfigError(f"Cannot force tool {tool_name!r} without a required parameter")
            first_param = required[0]
        return self._encode_markup(self.profile.tool_prefix(tool_name, first_param))

    def parse(self, completion_ids: Sequence[int], tools: Sequence[ToolSpec] = ()) -> ParsedTurn:
        ids = list(completion_ids)
        self._last_stop_id = ids[-1] if ids and ids[-1] in self.stop_token_ids else None
        termination = "length"
        if self._last_stop_id is not None:
            termination = "stop"
            ids.pop()
        return self.profile.parser(self.decode(ids), tools or self._tools, termination)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False, split_special_tokens=True)[
            "input_ids"
        ]

    def _encode_markup(self, text: str, mask: _ContentMask | None = None) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in self._special_pattern.finditer(text):
            gap = text[position : match.start()]
            ids.extend(self.encode_text(mask.restore(gap) if mask else gap))
            ids.append(self._special_ids[match[0]])
            position = match.end()
        gap = text[position:]
        ids.extend(self.encode_text(mask.restore(gap) if mask else gap))
        return ids


def _tokenizer_sha(tokenizer: PreTrainedTokenizerBase) -> str:
    serialized = json.dumps(
        {"vocab": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class _ContentMask:
    """One template call's placeholders, including JSON/Python-escaped forms."""

    def __init__(self, special_pattern: re.Pattern[str]) -> None:
        self._special_pattern = special_pattern
        self._nonce = uuid4().hex
        self._originals: list[str] = []
        # Nested XML arguments use Python repr; JSON may also escape the PUA.
        self._placeholder_pattern = re.compile(
            r"(?:\ue000|\\ue000)" + self._nonce + r"_(\d+)(?:\ue001|\\ue001)"
        )

    def text(self, text: str) -> str:
        if "\ue000" in text or "\ue001" in text:
            raise ValueError("Content contains reserved placeholder delimiters")

        def replace(match: re.Match[str]) -> str:
            self._originals.append(match[0])
            return f"\ue000{self._nonce}_{len(self._originals) - 1}\ue001"

        return self._special_pattern.sub(replace, text)

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.value(k): self.value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.value(v) for v in value)
        return value

    def restore(self, text: str) -> str:
        return self._placeholder_pattern.sub(lambda m: self._originals[int(m[1])], text)


def _message_dict(msg: Msg, mask: _ContentMask) -> dict[str, Any]:
    result: dict[str, Any] = {"role": msg.role, "content": mask.text(msg.content)}
    if msg.name is not None:
        result["name"] = msg.name
    if msg.tool_call_id is not None:
        result["tool_call_id"] = msg.tool_call_id
    if msg.role == "assistant":
        if msg.thinking is not None:
            result["reasoning_content"] = mask.text(msg.thinking)
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": mask.value(call.arguments)},
                    **({"id": call.id} if call.id is not None else {}),
                }
                for call in msg.tool_calls
            ]
    return result
