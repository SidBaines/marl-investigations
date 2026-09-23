"""Token-level rendering contract (``DeltaRenderer``).

Agents keep an append-only token buffer per segment. A renderer turns chat
messages into token ids in two steps: ``initial`` renders the first prompt
(system + tools + messages, ending with the assistant generation header and
any generation prefill such as Qwen3.5's ``<think>\n``); ``continuation``
renders only what comes *after* a sampled completion (closing the previous turn
if it did not sample an end-of-turn token, the new tool results / delivered
messages, and the next generation header). The sampled ids are appended verbatim by
the agent runtime — the renderer never re-renders them. This keeps "datums use
exactly the ids the policy saw" true and makes one datum per segment possible.

**Canonical format = the model's own HF chat template**, not tinker-cookbook's
renderers: a 2026-09-23 prototype showed the cookbook's Qwen3.5 renderer
deviates from the HF template (unwrapped tool JSON in the system prompt,
history thinking stripped so re-renders don't prefix-extend what was sampled)
and its XML tool-call parser coerces string parameters (``"4"`` → ``4``).
Concrete renderers therefore build ``initial`` with
``tokenizer.apply_chat_template`` and ``continuation`` by rendering the new
messages after a *sentinel* assistant turn and slicing after the sentinel's
end-of-turn special token (turn boundaries are special tokens, so slices
tokenize independently); gpt-oss uses the official ``openai_harmony``
encoding. Parity tests compare ``initial + completion + continuation`` against
``apply_chat_template`` of the equivalent full conversation.

Renderers whose delta output has not passed parity tests set
``supports_delta = False``; the runtime then starts a new segment per call,
re-rendering the full message history with ``initial``
(``SegmentStart.RERENDER``). Chosen per renderer at config time, never per call.

``delivery_role`` says how the runtime injects workspace deliveries
(notify/push text): ``"tool"`` = as an extra tool-result message after the
turn's tool results (Qwen templates group consecutive tool responses, and tool
responses do not reset "last user query", so reasoning stays in-distribution);
``"append_tool"`` = appended to the content of the turn's last tool result
(formats where uncalled tool messages are invalid, e.g. Harmony); ``"user"`` =
as a user message. When the previous turn made no tool call, deliveries always
go in a user message.

``render/fake.py`` is a character-level renderer for CPU tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """A tool as advertised to the model (OpenAI function-schema shape)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class ToolCall:
    """A tool call inside an assistant message (for rendering history / API policies)."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class Msg:
    """A chat message as the renderer sees it.

    ``role`` is one of "system", "user", "assistant", "tool". Tool results use
    role "tool" with ``name`` (the tool) and ``tool_call_id`` where the format
    needs it. ``thinking`` is only used when rendering assistant history in
    re-render mode; the delta path never renders assistant messages.
    """

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None
    thinking: str | None = None


@dataclass(frozen=True)
class ParsedToolCall:
    name: str | None
    arguments: dict[str, Any] | None
    raw: str  # the call's raw text span
    ok: bool  # parsed into a name + JSON-object arguments
    id: str | None = None  # native call id if the format has one


@dataclass(frozen=True)
class ParsedTurn:
    """A sampled completion, parsed from token ids (never from server text)."""

    content: str  # visible text (thinking removed)
    thinking: str | None
    tool_calls: tuple[ParsedToolCall, ...] = ()
    # "stop": ended with the renderer's stop token; "length": no stop token
    # (truncated); "malformed": ended but a tool call could not be closed/parsed.
    termination: str = "stop"
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DeltaRenderer(Protocol):
    name: str
    supports_delta: bool
    tokenizer_sha: str
    stop_token_ids: tuple[int, ...]  # ids that end an assistant turn (included in completions)
    delivery_role: str  # "tool" | "append_tool" | "user" (see module docstring)

    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        """Full prompt for a fresh segment, ending with the assistant generation header."""
        ...

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        """Tokens to append after a sampled completion so the buffer again ends in a
        generation header. The runtime passes ``"stop"`` iff the completion ended
        with a stop id, even when parsing found a malformed call. Renderers close
        the turn iff the last completion did not end with their end-of-turn token
        (an alternate EOS can stop generation without closing the turn). Consecutive
        tool results are grouped the way the reference template groups them."""
        ...

    def forced_tool_prefix(self, tool_name: str) -> list[int]:
        """Tokens that open a call to ``tool_name`` right after the generation header
        (used by force_final: appended as observation, then the model completes the
        arguments)."""
        ...

    def suppress_thinking_prefix(self) -> list[int]:
        """Observation tokens appended after the generation header to skip the reasoning
        channel for COMPACT/CARRY calls (compaction/carry summaries),
        in the template's own non-thinking form (Qwen3.5: close the prefilled think block;
        gpt-oss: open the final channel). Forced FINAL/REPORT calls use only
        ``forced_tool_prefix``. ``[]`` for formats without a reasoning channel."""
        ...

    def parse(self, completion_ids: Sequence[int], tools: Sequence[ToolSpec] = ()) -> ParsedTurn:
        """Parse sampled ids (including the stop token, if any). ``tools`` lets formats
        with untyped parameter text (Qwen3.5 XML) type arguments by their JSON schema:
        string-capable parameters keep raw text; other declared types determine decoding."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def encode_text(self, text: str) -> list[int]:
        """Plain-text encoding (budget estimates, scripted policies). No special tokens."""
        ...
