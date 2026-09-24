---
type: concept
title: "Unanswered episodes: read the answered rate before accuracy"
description: A low answered rate means accuracy is measuring a harness or budget-design failure rather than reasoning; the known cases are the multi-session nudge-strike bug (fixed in 76aa955) and debate's per-round call cap.
resource: src/marli/interact/agent.py
tags: [evaluation, harness, measurement, bugs, multi-session, debate]
timestamp: 2026-09-24
---

# Unanswered episodes: read the answered rate before accuracy

**The rule.** Accuracy counts an unanswered episode as wrong. When a cell's
`Answered` rate is well below 1, its accuracy is mostly telling you how
episodes terminate, not how well the model reasons. Check `Answered` (and
`Failed`) in the eval report before you read `Accuracy`. [pilot]

**Regime:** untrained `tinker:Qwen/Qwen3.5-4B` ([entity](../entities/qwen3-5-4b.md))
on Tinker. Benchmarks are [AIME25](../entities/aime-2025.md) and
[HMMT25](../entities/hmmt-feb-2025.md), n=30 each. G=1, one seed, and a
32,768-token episode cap. Sources:
[pilot README](../../sources/compute-matched-baselines-pilot.md) and
[report tables](../../sources/compute-matched-baselines-results.md).

## Case 1: the nudge-strike bug (harness; fixed)

- **Mechanism, before the fix.** A turn cut off by its token allocation got a
  nudge message **and** a strike. Strikes were not reset when a fresh context
  started (new session or compaction), so they piled up across sessions.
  Reaching `max_nudges` (default 2 consecutive) ended the agent **with no
  answer**. Multi-session agents whose sessions ended by budget were therefore
  killed in their last session. (Commit 76aa955; the limits are in
  `src/marli/interact/limits.py`.)
- **Symptom in the pilot.** Answered rate:

| Cell | AIME25 answered | HMMT25 answered |
|---|---|---|
| ms3_compaction | 0.533 | 0.700 |
| ms3_notes | 0.900 | 1.000 |
| single (reference) | 0.967 | 1.000 |

- **Fix (76aa955).** A fresh context gets a fresh nudge budget. A
  length-truncated turn still gets the nudge message but no strike. Reaching
  `max_nudges` forces a FINAL/REPORT under `on_exhaust=force_final`. On the
  fixed harness, both multi-session cells answered 1.000 on AIME25.
- **Consequence.** The pilot's multi-session numbers are struck through and
  replaced by the rerun in [multi-session carry](multi-session-carry.md).

### Measurement lessons [pilot]

1. **A significant-looking result can be a harness artefact.** ms3_compaction
   on AIME25 (−0.233 [−0.400, −0.067], McNemar p=0.039) was the only pilot
   multi-session lift with p < 0.05. After the fix it shrank to −0.133
   [−0.300, +0.033] (p=0.29). The answered rate is what gave the bug away.
2. **Fix, rerun the affected cells, supersede.** Never patch numbers by hand.
   The rerun reused the baseline's saved Scores so the pairing was preserved.
   A related ops bug surfaced then: a grid that reused Scores charged their
   original cost against its own `max_usd` (fixed in 33d86f5).
3. **Only the multi-session cells were rerun.** The other pilot cells (single,
   sc4, swarm4, coordinator, debate3) ran on the pre-fix harness too. The fix
   changes strike accounting for every agent. Those cells answered 1.000,
   except single (0.967 on AIME25) and debate3 (below).

## Case 2: debate's per-round call cap (budget design)

debate3 caps each round's call at 5,461 tokens. That truncates a thinking
model's reasoning before a `\boxed{}` answer. The result is answered 0.233 on
AIME25 and 0.067 on HMMT25, with accuracy 0.133 [0.053, 0.297] and 0.033
[0.006, 0.167]. The pilot README treats this as a budget-design problem for
thinking models, not evidence against debate. [pilot] See
[budget splitting](budget-splitting-truncation.md) and the
[debate entity](../entities/protocol-debate.md).

## Tensions

- The fix commit says 13/30 ms3_compaction AIME25 episodes were ended
  unanswered by `max_nudges`. The report's answered rate of 0.533 implies
  14/30 unanswered. The extra episode was either unanswered by some other path
  or counted differently. It is unresolved but does not change the conclusion.

See also: [compute-matching](compute-matching.md),
[the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md).
