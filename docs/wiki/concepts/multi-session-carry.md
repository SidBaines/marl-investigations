---
type: concept
title: "Multi-session carry: notes vs compaction"
description: One agent's budget split into 3 fresh-context sessions, carrying notes-to-self or a compaction summary; on the fixed harness notes ties single on AIME25 at ~60% of the tokens, but most episodes submit in session 1, so compute is not matched.
resource: src/marli/interact/protocols/multi_session.py
tags: [multi-session, carry, notes, compaction, compute, protocols]
timestamp: 2026-09-24
---

# Multi-session carry: notes vs compaction

**Setup.** A [multi-session](../entities/protocol-multi-session.md) agent works
in up to 3 sessions. The context is cleared between sessions, and a *carry*
decides what survives:

- `multi_session_s3_notes` carries private notes-to-self (≤ 4,000 chars).
- `multi_session_s3_compaction` carries a session-end summary.

The protocol splits the agent's budget equally across sessions, with the final
reserve kept outside the split. At a 32,768 episode cap with a 512 reserve,
that is about 10.7k per session. `submit` ends the whole episode, and
`end_session` hands over to the next session. Each session starts a new
training *segment*, which is what `segment_credit` acts on (see
[the RL smoke](../../sources/tinker-rl-smoke.md)).

**Regime:** untrained `tinker:Qwen/Qwen3.5-4B` ([entity](../entities/qwen3-5-4b.md))
on Tinker. The benchmark is [AIME25](../entities/aime-2025.md), n=30. G=1, one
seed, lockstep, math-verify, and a 32,768-token episode cap. Accuracy has Wilson
95% CIs. Lift is the paired difference vs `single` with a task-bootstrap 95% CI
and exact McNemar p. Sources:
[pilot README](../../sources/compute-matched-baselines-pilot.md) and
[report tables](../../sources/compute-matched-baselines-results.md).

## Pilot numbers (pre-fix harness): superseded

The pilot ran on a harness with the [nudge-strike bug](unanswered-episodes.md).
Agents whose sessions ended by budget were terminated with no answer. The
pilot numbers therefore measure the bug, not the carry:

- ~~AIME25: ms3_notes 0.467 [0.302, 0.639], lift −0.067 [−0.233, +0.100]
  (3/5, p=0.73); ms3_compaction 0.300 [0.167, 0.479], lift −0.233
  [−0.400, −0.067] (1/8, p=0.039)~~. Superseded by the fixed-harness rerun
  below. The pilot's answered rate was 0.900 (notes) and 0.533 (compaction).
- ~~HMMT25: ms3_notes 0.333 [0.192, 0.512] and ms3_compaction 0.333
  [0.192, 0.512], each lift −0.200 [−0.400, +0.000] (2/8, p=0.11)~~. No
  replacement yet. The HMMT25 rerun stopped at 7 / 6 of 30 episodes when its
  budget ran out and is not reported. HMMT25 multi-session is **unmeasured**.
  [open]

## Rerun on the fixed harness (76aa955), AIME25 [pilot]

`single` is reused from the pilot via `scores:`, so these are the same
baseline episodes.

| Cell | acc | lift vs single (W/L, p) | answered | total_gen mean / p50 / p90 | sessions used 1/2/3 | correct by sessions used |
|---|---|---|---|---|---|---|
| single | 0.533 [0.361, 0.698] | — | 0.967 | 25.5k / 31.2k / 31.8k | — | — |
| ms3_notes | 0.500 [0.332, 0.668] | −0.033 [−0.167, +0.067] (1/2, p=1.00) | 1.000 | 15.3k / 8.9k / 32.3k | 17 / 4 / 9 | 12/17, 2/4, 1/9 |
| ms3_compaction | 0.400 [0.246, 0.577] | −0.133 [−0.300, +0.033] (2/6, p=0.29) | 1.000 | 13.5k / 11.4k / 27.3k | 14 / 4 / 12 | 4/14, 4/4, 4/12 |

What this supports:

1. **Notes roughly ties `single` on ~60% of the tokens** (15.3k vs 25.5k mean
   `total_gen`). This is not a matched comparison, since the report flags both
   cells (see [compute-matching](compute-matching.md)). [pilot]
2. **Early submission.** More than half the episodes submit in session 1 (17/30
   notes, 14/30 compaction). The ~10.7k session budget ends a chain early, and
   the agent often answers instead of carrying over. So the protocol usually
   does not use the compute it was given. [pilot]
3. **Selection in later sessions.** The episodes that reach session 3 are
   mostly the hard problems (notes: 1/9 correct there). This is a selection
   effect. It is not evidence that later sessions hurt. [pilot]
4. **Compaction trails notes by 0.10.** 16/30 compaction episodes needed a
   forced final, against 8/30 for notes. No paired notes-vs-compaction test is
   in the committed tables, and the cause is open because the transcripts have
   not been read. [pilot]

## Tensions

- **Session 1 differs between carry modes.** Before the first reset, the two
  cells differ only in the carry instructions in the system prompt. Yet
  session-1 submitters are 12/17 correct under notes and 4/14 under
  compaction. Also, the 16 compaction forced finals outnumber the 12 episodes
  that reached session 3. That suggests some forced finals came before the last
  session, for example through the post-fix max_nudges → forced-final path.
  This is an inference from the counts and has not been checked in
  transcripts. It could be what drags compaction's session-1 accuracy down, or
  it could be noise at n=30, G=1. [open]
- **The rerun baseline predates the fix.** The rerun's `single` episodes ran on
  the harness before 76aa955, which changed nudge-strike accounting and forced
  finals for every agent. `single` left 1/30 AIME25 episodes unanswered, so the
  effect on the baseline is probably at most about one episode. Strictly,
  though, the rerun's paired lift crosses a harness change. [open]
- **Pilot and rerun differ** (compaction 0.300 → 0.400, notes 0.467 → 0.500).
  The direction fits the bug's no-answer terminations. Both are single n=30,
  G=1 draws, so part of the change is sampling noise.
- **The design question is untested.** The study hypothesised that notes and
  compaction "help most on problems whose reasoning exceeds one context". This
  protocol *divides* one agent's 32k budget into three ~10.7k sessions, so
  what it tests is "fresh contexts at equal budget". It does not test
  "reasoning beyond one context". [open]

## Open questions

- Finish the HMMT25 multi-session rerun (about $2 by the README's estimate). [open]
- Match realized compute: raise per-session budgets, or add a "use all
  sessions" instruction, then compare at equal mean `total_gen`. [open]
- Read the compaction transcripts to explain the forced finals. [open]
- Carry modes `both` and `tail` exist (`multi_session_s3_both`,
  `multi_session_s3_tail`) but have not been run. [open]
- Credit across sessions (all vs last) is untested: the RL smoke could not
  compare them (see [zero-variance groups](zero-variance-groups.md)). [open]

See also: [unanswered episodes](unanswered-episodes.md),
[the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md).
