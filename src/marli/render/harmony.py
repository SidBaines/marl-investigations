"""Keep gpt-oss context append-only using the official Harmony wire format.

Harmony renders message lists without adding a system prefix, so deltas need
no sentinel slicing. Analysis dropping is disabled: sampled reasoning must
remain in the token buffer across tool rounds and user messages. Only action
stops end a completion; ``<|end|>`` can separate messages within one completion.

The conversation date defaults to 2026-09-23 for reproducible prompts. The
cached gpt-oss HF template instead reads the current date when rendering its
system header. Harmony, rather than that template's history rewriting, is the
reference for delta parity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from marli.render.base import DeltaRenderer, Msg, ParsedToolCall, ParsedTurn, ToolSpec

if TYPE_CHECKING:
    from openai_harmony import Message


class HarmonyRenderer:
    supports_delta = True
    delivery_role = "append_tool"

    def __init__(
        self,
        reasoning_effort: str = "medium",
        *,
        conversation_start_date: str = "2026-09-23",
    ) -> None:
        import openai_harmony as harmony

        if reasoning_effort not in ("low", "medium", "high"):
            raise ValueError(f"Unknown Harmony reasoning effort: {reasoning_effort!r}")
        self.name = f"gpt_oss_{reasoning_effort}"
        self._harmony = harmony
        self._encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
        self._system = harmony.SystemContent(
            reasoning_effort=harmony.ReasoningEffort[reasoning_effort.upper()],
            conversation_start_date=conversation_start_date,
        )
        self._render_config = harmony.RenderConversationConfig(auto_drop_analysis=False)
        self.stop_token_ids = tuple(sorted(self._encoding.stop_tokens_for_assistant_actions()))
        self._end_ids = self._encoding.encode("<|end|>", allowed_special="all")
        # Harmony 0.0.8 exposes no vocab size. Its reserved special tokens extend
        # through the final vocabulary id, including otherwise unused slots.
        vocab_size = 1 + max(
            self._encoding.encode(token, allowed_special="all")[0]
            for token in self._encoding.special_tokens_set
        )
        self.tokenizer_sha = hashlib.sha256(
            f"{self._encoding.name}:{vocab_size}".encode()
        ).hexdigest()
        self._tools: dict[str, ToolSpec] = {}
        # A malformed frame can still end on an action stop. Continuation must
        # not insert an extra close in that case.
        self._last_completion_stopped = False

    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        h = self._harmony
        self._tools = {tool.name: tool for tool in tools}
        self._last_completion_stopped = False
        developer = h.DeveloperContent(instructions=system)
        if tools:
            developer.with_function_tools(
                [
                    h.ToolDescription.new(t.name, t.description, parameters=t.parameters)
                    for t in tools
                ]
            )
        messages = [
            h.Message.from_role_and_content(h.Role.SYSTEM, self._system),
            h.Message.from_role_and_content(h.Role.DEVELOPER, developer),
            *self._messages(msgs),
        ]
        return self._encoding.render_conversation_for_completion(
            h.Conversation.from_messages(messages), h.Role.ASSISTANT, self._render_config
        )

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        h = self._harmony
        needs_close = last_termination == "length" or (
            last_termination == "malformed" and not self._last_completion_stopped
        )
        return (self._end_ids if needs_close else []) + (
            self._encoding.render_conversation_for_completion(
                h.Conversation.from_messages(self._messages(new_msgs)),
                h.Role.ASSISTANT,
                self._render_config,
            )
        )

    def _messages(self, msgs: Sequence[Msg]) -> list[Message]:
        h = self._harmony
        messages: list[Message] = []
        for msg in msgs:
            if msg.role == "tool":
                if not msg.name:
                    raise ValueError("Harmony tool results require a tool name")
                messages.append(
                    h.Message.from_author_and_content(
                        h.Author.new(h.Role.TOOL, f"functions.{msg.name}"), msg.content
                    )
                    .with_recipient("assistant")
                    .with_channel("commentary")
                )
            elif msg.role == "assistant":
                if msg.thinking is not None:
                    messages.append(
                        h.Message.from_role_and_content(
                            h.Role.ASSISTANT, msg.thinking
                        ).with_channel("analysis")
                    )
                if msg.content or not msg.tool_calls:
                    messages.append(
                        h.Message.from_role_and_content(h.Role.ASSISTANT, msg.content).with_channel(
                            "commentary" if msg.tool_calls else "final"
                        )
                    )
                for call in msg.tool_calls:
                    messages.append(
                        h.Message.from_role_and_content(
                            h.Role.ASSISTANT, json.dumps(call.arguments)
                        )
                        .with_channel("commentary")
                        .with_recipient(f"functions.{call.name}")
                        .with_content_type("<|constrain|>json")
                    )
            elif msg.role in ("user", "system"):
                messages.append(h.Message.from_role_and_content(h.Role(msg.role), msg.content))
            else:
                raise ValueError(f"Unknown Harmony message role: {msg.role!r}")
        return messages

    def suppress_thinking_prefix(self) -> list[int]:
        """Open the final channel directly, skipping analysis (harness summary calls)."""
        return self._encoding.encode("<|channel|>final<|message|>", allowed_special="all")

    def forced_tool_prefix(self, tool_name: str) -> list[int]:
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name!r} was not advertised in initial()")
        required = self._tools[tool_name].parameters.get("required", [])
        if not required:
            raise ValueError(f"Tool {tool_name!r} needs a required parameter for a forced prefix")
        return self._encoding.encode(
            f"<|channel|>commentary to=functions.{tool_name} <|constrain|>json<|message|>"
            + "{"
            + json.dumps(required[0])
            + ': "',
            allowed_special="all",
        )

    def parse(self, completion_ids: Sequence[int], tools: Sequence[ToolSpec] = ()) -> ParsedTurn:
        ids = list(completion_ids)
        self._last_completion_stopped = bool(ids and ids[-1] in self.stop_token_ids)
        termination = "stop" if self._last_completion_stopped else "length"
        try:
            messages = self._encoding.parse_messages_from_completion_tokens(
                ids, self._harmony.Role.ASSISTANT
            )
        except Exception:
            termination = "malformed"
            messages = self._recover_messages(ids)

        thinking: list[str] = []
        content: list[str] = []
        calls: list[ParsedToolCall] = []
        for message in messages:
            if message.author.role != self._harmony.Role.ASSISTANT:
                continue
            raw = "".join(part.text for part in message.content)
            if message.channel == "analysis":
                thinking.append(raw)
            elif message.channel == "final":
                content.append(raw)
            elif message.channel == "commentary":
                if message.recipient and message.recipient.startswith("functions."):
                    name = message.recipient.removeprefix("functions.")
                    try:
                        arguments = json.loads(raw)
                    except (ValueError, RecursionError):
                        arguments = None
                    ok = bool(name) and isinstance(arguments, dict)
                    calls.append(
                        ParsedToolCall(
                            name=name or None,
                            arguments=arguments if ok else None,
                            raw=raw,
                            ok=ok,
                        )
                    )
                elif message.recipient is None:
                    content.append(raw)
        return ParsedTurn(
            content="".join(content),
            thinking="".join(thinking) if thinking else None,
            tool_calls=tuple(calls),
            termination=termination,
        )

    def _recover_messages(self, ids: Sequence[int]) -> list[Message]:
        h = self._harmony
        parser = h.StreamableParser(self._encoding, h.Role.ASSISTANT)
        try:
            for token in ids:
                parser.process(token)
        except Exception:
            pass  # Keep messages preceding the first framing/token error.
        messages = parser.messages
        if parser.current_role == h.Role.ASSISTANT and parser.current_content:
            message = h.Message.from_role_and_content(h.Role.ASSISTANT, parser.current_content)
            message.channel = parser.current_channel
            message.recipient = parser.current_recipient
            messages.append(message)
        return messages

    def decode(self, ids: Sequence[int]) -> str:
        return self._encoding.decode(ids)

    def encode_text(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())


HARMONY_RENDERERS: dict[str, Callable[..., DeltaRenderer]] = {
    f"gpt_oss_{effort}": partial(HarmonyRenderer, reasoning_effort=effort)
    for effort in ("low", "medium", "high")
}
