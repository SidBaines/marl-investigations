"""Context managers: compaction, notes, tail carry, and session carry.

Owned by the agent runtime; each decides *when* the context is reset and
*what* crosses the reset. Every reset starts a new segment
(``SegmentInfo.carry_from`` points at the segment it came from).

- ``none``: never reset (the agent may run out of context -> ``Termination.CTX``
  -> ``on_exhaust``).
- ``compaction(threshold)`` (SUPO-style): when the next prompt would exceed
  ``threshold`` tokens, the runtime drops the pending last action/observation
  delta, appends a compaction instruction (observation) and makes a
  ``purpose=COMPACT`` call *in the ending segment* (so the summary tokens are
  trained with the segment that produced them). The new segment is
  ``initial(system, tools, [user(first_message + summary)])``. Validation:
  ``threshold <= max_ctx - compact_reserve``.
- ``notes(cap)``: the agent has ``read_notes``/``write_notes`` tools; notes
  persist in the workspace. At a session boundary the new session's first
  user message includes the current notes (capped). No extra call.
- ``both``: notes tools + a compaction summary at session end.
- ``tail(m)`` (Delethink-style baseline): at a reset, the last ``m`` ids of the
  ending segment are spliced into the new segment after the generation header
  as prefill (observation tokens).
- Session carry (multi-session): at session end the runtime makes a
  ``purpose=CARRY`` call (for ``compaction``/``both``) asking for a summary for
  "your next session"; with budget exhaustion this call is forced and uses
  ``SessionLimits.carry_reserve``. The session budget counts compaction and
  carry calls.

Carry text is capped (``SessionLimits.carry_max_tokens`` /
``ContextSpec.notes_cap_chars``); overflow is truncated and flagged in
``limits_hit``.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from marli.errors import ConfigError
from marli.interact.limits import Limits
from marli.interact.system import ContextSpec


@dataclass(frozen=True)
class CarryText:
    """Carry payload and whether the runtime must record a truncation limit hit."""

    text: str
    truncated: bool


class ContextManager(Protocol):
    """Pure reset policy; the runtime owns calls, buffers, and segment lineage."""

    @property
    def spec(self) -> ContextSpec:
        """Read-only context policy configuration."""
        ...

    def should_compact(self, prompt_len: int) -> bool:
        """Whether the next prompt exceeds an enabled compaction threshold."""
        ...

    def compaction_instruction(self) -> str:
        """Request a replacement summary, or return empty text without compaction."""
        ...

    def session_carry_instruction(self) -> str | None:
        """Request the next session's summary, or None when no carry call is needed."""
        ...

    def carry_text(self, *, summary: str | None, notes: str | None) -> CarryText:
        """Format enabled carry sources after the task in the first user message.

        Empty or whitespace-only sources are omitted. Caps count retained source characters;
        labels and the leading ``…[truncated]`` marker are additional text.
        Truncation keeps the tail of each source.
        """
        ...

    def tail_prefill(self, ending_segment_ids: Sequence[int]) -> list[int]:
        """Copy the final tail_tokens ids, or return [] unless kind is tail."""
        ...

    @property
    def has_notes_tools(self) -> bool:
        """Whether this role receives read_notes and write_notes tools."""
        ...


@dataclass(frozen=True)
class _ContextManager:
    """Apply the configured carry policy without sampling or editing context."""

    spec: ContextSpec
    limits: Limits

    def should_compact(self, prompt_len: int) -> bool:
        return (
            self.spec.kind in ("compaction", "both")
            and self.spec.compact_threshold > 0
            and prompt_len > self.spec.compact_threshold
        )

    def compaction_instruction(self) -> str:
        if self.spec.kind not in ("compaction", "both"):
            return ""
        return (
            "Your context is almost full. It will now be cleared and replaced by the "
            "summary you write in this reply; afterwards you will see only the original "
            "task and this summary. "
        ) + self._summary_instruction()

    def session_carry_instruction(self) -> str | None:
        if self.spec.kind not in ("compaction", "both"):
            return None
        opening = (
            "Your session is ending. Your next session will start with a fresh context "
            "and will see only the original task and the summary you write in this reply"
        )
        if self.spec.kind == "both":
            opening += " plus your saved notes; don't repeat them"
        return opening + ". " + self._summary_instruction()

    def _summary_instruction(self) -> str:
        tokens = self.limits.session.carry_max_tokens
        return (
            "Write the summary itself as your reply. Include what you have tried and "
            "what worked or failed (and why), intermediate results with exact values, "
            "your current best answer or the current state of your solution, and your "
            "next steps. Do not restate the task. "
            f"Keep it within about {tokens} tokens (roughly {tokens * 3 // 4} words); "
            "anything longer is cut. Do not call any tools and do not continue working "
            "on the task in this reply."
        )

    def carry_text(self, *, summary: str | None, notes: str | None) -> CarryText:
        parts: list[str] = []
        truncated = False
        for label, content, cap in (
            (
                "Previous session summary",
                summary if self.spec.kind in ("compaction", "both") else None,
                self.limits.session.carry_max_tokens * 4,
            ),
            ("Your notes", notes if self.has_notes_tools else None, self.spec.notes_cap_chars),
        ):
            if not content or not content.strip():
                continue
            if len(content) > cap:
                content = "…[truncated]" + content[len(content) - cap :]
                truncated = True
            parts.append(f"[{label}]\n{content}")
        return CarryText(text="\n\n".join(parts), truncated=truncated)

    def tail_prefill(self, ending_segment_ids: Sequence[int]) -> list[int]:
        if self.spec.kind != "tail":
            return []
        return list(ending_segment_ids[-self.spec.tail_tokens :])

    @property
    def has_notes_tools(self) -> bool:
        return self.spec.kind in ("notes", "both")


def make_context_manager(spec: ContextSpec, limits: Limits) -> ContextManager:
    """Validate context settings against limits before the runtime spends compute."""
    kinds = ("none", "compaction", "notes", "both", "tail")
    if spec.kind not in kinds:
        raise ConfigError(f"context.kind must be one of {kinds}, got {spec.kind!r}")
    if spec.notes_cap_chars <= 0:
        raise ConfigError(f"context.notes_cap_chars must be > 0, got {spec.notes_cap_chars!r}")
    if spec.compact_threshold < 0:
        raise ConfigError(f"context.compact_threshold must be >= 0, got {spec.compact_threshold!r}")
    if spec.kind in ("compaction", "both"):
        if spec.compact_threshold > 0:
            if spec.compact_reserve <= 0:
                raise ConfigError(
                    "context.compact_reserve must be > 0 to cover the summary, "
                    f"got {spec.compact_reserve!r}"
                )
            if spec.compact_reserve < limits.session.carry_max_tokens:
                raise ConfigError(
                    "context.compact_reserve must cover the summary "
                    f"(>= session.carry_max_tokens={limits.session.carry_max_tokens!r}), "
                    f"got {spec.compact_reserve!r}"
                )
            if spec.compact_threshold > limits.ctx.max_ctx - spec.compact_reserve:
                raise ConfigError(
                    "context.compact_threshold must be <= ctx.max_ctx - context.compact_reserve "
                    f"({limits.ctx.max_ctx!r} - {spec.compact_reserve!r}), "
                    f"got {spec.compact_threshold!r}"
                )
        if limits.session.carry_reserve <= 0:
            raise ConfigError(
                "session.carry_reserve must be > 0 to cover the summary for compaction/both, "
                f"got {limits.session.carry_reserve!r}"
            )
        if limits.session.carry_reserve < limits.session.carry_max_tokens:
            raise ConfigError(
                "session.carry_reserve must cover the summary "
                f"(>= session.carry_max_tokens={limits.session.carry_max_tokens!r}), "
                f"got {limits.session.carry_reserve!r}"
            )
    if spec.kind == "tail":
        if spec.tail_tokens <= 0:
            raise ConfigError(f"context.tail_tokens must be > 0, got {spec.tail_tokens!r}")
        if spec.tail_tokens >= limits.ctx.max_ctx // 2:
            raise ConfigError(
                f"context.tail_tokens must be < ctx.max_ctx // 2 ({limits.ctx.max_ctx // 2}), "
                f"got {spec.tail_tokens!r}"
            )
    return _ContextManager(spec=spec, limits=limits)
