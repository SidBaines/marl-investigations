"""HF templates own the wire format; deltas never rewrite sampled tokens.

The sentinel assistant turn only locates a boundary. For supported profiles,
rendering subsequent messages is independent of that turn's content. Slicing
after its end-of-turn special token keeps BPE merges from crossing the slice
boundary. This does not make templates that rewrite earlier history append-only.
"""

from __future__ import annotations

import hashlib
import json
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
        self, tokenizer: PreTrainedTokenizerBase, name: str, profile: TemplateProfile
    ) -> None:
        self.tokenizer = tokenizer
        self.name = name
        self.profile = profile
        self.stop_token_ids = profile.stop_token_ids
        self.delivery_role = profile.delivery_role
        self.supports_delta = profile.supports_delta
        self._tools: tuple[ToolSpec, ...] = ()
        serialized = json.dumps(
            {"vocab": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.tokenizer_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        self._tools = tuple(tools)
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.extend(_message_dict(msg) for msg in msgs)
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
        return self.encode_text(text)

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        # A fresh sentinel cannot collide with a user-supplied message or template text.
        sentinel = f"MARLI_DELTA_{uuid4().hex}"
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "·"},
                {"role": "assistant", "content": sentinel},
                *(_message_dict(msg) for msg in new_msgs),
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self.profile.chat_template_kwargs,
        )
        sentinel_end = text.index(sentinel) + len(sentinel)
        boundary = text.index(self.profile.close_turn, sentinel_end) + len(self.profile.close_turn)
        delta = text[boundary:]
        if last_termination != "stop":
            delta = self.profile.close_turn + delta
        return self.encode_text(delta)

    def suppress_thinking_prefix(self) -> list[int]:
        """Tokens that skip the reasoning channel for harness-requested calls."""
        text = self.profile.suppress_thinking
        return self.encode_text(text) if text else []

    def forced_tool_prefix(self, tool_name: str, first_param: str | None = None) -> list[int]:
        if first_param is None:
            tool = next((tool for tool in self._tools if tool.name == tool_name), None)
            if tool is None:
                raise ConfigError(f"Cannot force unknown tool {tool_name!r}; call initial first")
            required = tool.parameters.get("required", [])
            if not required:
                raise ConfigError(f"Cannot force tool {tool_name!r} without a required parameter")
            first_param = required[0]
        return self.encode_text(self.profile.tool_prefix(tool_name, first_param))

    def parse(self, completion_ids: Sequence[int], tools: Sequence[ToolSpec] = ()) -> ParsedTurn:
        ids = list(completion_ids)
        termination = "length"
        if ids and ids[-1] in self.stop_token_ids:
            termination = "stop"
            ids.pop()
        return self.profile.parser(self.decode(ids), tools, termination)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def _message_dict(msg: Msg) -> dict[str, Any]:
    result: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.name is not None:
        result["name"] = msg.name
    if msg.tool_call_id is not None:
        result["tool_call_id"] = msg.tool_call_id
    if msg.role == "assistant":
        if msg.thinking is not None:
            result["reasoning_content"] = msg.thinking
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                    **({"id": call.id} if call.id is not None else {}),
                }
                for call in msg.tool_calls
            ]
    return result
