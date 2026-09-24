---
type: entity
title: "Protocol: multi-session"
description: One agent's budget divided across fresh-context sessions with a carry mode (notes, compaction, both, tail); each session is a new training segment.
resource: src/marli/interact/configs/multi_session_s3_notes.yaml
tags: [protocol, multi-session, carry, notes, compaction]
timestamp: 2026-09-24
---

# Protocol: `multi_session`

- **Configs:** all have `sessions: 3` and differ in carry.
  - `multi_session_s3_notes`: private notes-to-self, `notes_cap_chars` 4000.
  - `multi_session_s3_compaction`: session-end summary, `compact_threshold` 0.
  - `multi_session_s3_both`: summary plus notes.
  - `multi_session_s3_tail`: the last 2,048 tokens of parsed replies.

  Implementation: `src/marli/interact/protocols/multi_session.py`. Tools are
  `submit` (ends the episode) and `end_session`. With no explicit
  `session_tokens`, the budget is split equally across sessions, with the
  final reserve kept outside the split.
- **In the compute-matched pilot:** 3 sessions × ~10.7k. The pilot numbers
  came from the pre-fix harness and are superseded. The fixed-harness rerun
  on AIME25 gave:
  - notes: 0.500 [0.332, 0.668], lift −0.033 [−0.167, +0.067].
  - compaction: 0.400 [0.246, 0.577], lift −0.133 [−0.300, +0.033].
  - Mean `total_gen` was 15.3k / 13.5k, which is not compute-matched.

  HMMT25 is unmeasured. Regime: untrained [Qwen3.5-4B](qwen3-5-4b.md) on
  Tinker, n=30, G=1. Details are in
  [multi-session carry](../concepts/multi-session-carry.md). [pilot]
- **In the Tinker RL smoke:** `ms_all` (`segment_credit: all`) vs `ms_last`
  (`last`), both with `segment_unit: session`. The runs could not compare them
  ([zero-variance groups](../concepts/zero-variance-groups.md)). [pilot]
