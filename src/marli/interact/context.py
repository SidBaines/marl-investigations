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

Interface (M1-7)::

    class ContextManager(Protocol):
        spec: ContextSpec
        def should_compact(self, prompt_len: int) -> bool
        def compaction_instruction(self) -> str
        def session_carry_instruction(self) -> str | None      # None: no carry call (notes-only /
        tail)
        def carry_text(self, *, summary: str | None, notes: str | None) -> str   # text for the next
        segment's first message
        def tail_prefill(self, ending_segment_ids: list[int]) -> list[int]       # [] unless kind ==
        tail
    def make_context_manager(spec: ContextSpec, limits: Limits) -> ContextManager  # validates spec
    vs limits
"""

from __future__ import annotations
