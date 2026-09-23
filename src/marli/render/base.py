"""Token-level rendering contract (``DeltaRenderer``).

Agents keep an append-only token buffer per segment. A renderer turns chat
messages into token ids in two steps: ``initial`` renders the first prompt
(system + tools + messages, ending with the assistant generation header);
``continuation`` renders only what comes *after* a sampled completion (closing
the previous turn if it ended on ``length``, the new tool results / pushed
messages, and the next generation header). The sampled ids themselves are
appended verbatim by the agent runtime — the renderer never re-renders them.
This is what keeps "datums use exactly the ids the policy saw" true and makes
one datum per segment possible.

Renderers whose delta output has not passed parity tests against the model's
reference chat template set ``supports_delta = False``; the runtime then starts
a new segment per call, re-rendering the full message history with
``initial`` (``SegmentStart.RERENDER``). That is chosen per renderer at config
time, never per call.

Concrete renderers wrap tinker-cookbook renderers (``render/qwen3.py``,
``render/qwen3_5.py``, ``render/gpt_oss.py``); ``render/fake.py`` is a
character-level renderer for CPU tests.
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

    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        """Full prompt for a fresh segment, ending with the assistant generation header."""
        ...

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        """Tokens to append after a sampled completion so the buffer again ends in a
        generation header. ``last_termination`` is the previous completion's
        termination: after "length"/"malformed-without-stop" the renderer first emits
        the end-of-turn tokens the model did not sample. Consecutive tool results are
        grouped the way the model's reference template groups them."""
        ...

    def forced_tool_prefix(self, tool_name: str) -> list[int]:
        """Tokens that open a call to ``tool_name`` right after the generation header
        (used by force_final: appended as observation, then the model completes the
        arguments)."""
        ...

    def parse(self, completion_ids: Sequence[int]) -> ParsedTurn:
        """Parse sampled ids (including the stop token, if any)."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def encode_text(self, text: str) -> list[int]:
        """Plain-text encoding (budget estimates, scripted policies). No special tokens."""
        ...
