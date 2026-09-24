---
type: concept
title: Compute-matching
description: A shared episode token budget is a cap, not a match — compare protocols on realized total_gen and cp_tokens; the eval report flags cells more than 10% off the baseline's mean total_gen.
resource: src/marli/eval/report.py
tags: [evaluation, compute, methodology, protocols]
timestamp: 2026-09-24
---

# Compute-matching

**What it means here.** The repo's default comparison is compute-matched
(`CLAUDE.md`, *Evaluation conventions*). A grid gives every cell the same
`episode.max_gen_tokens`, and per-agent budgets split it by protocol shape
(see [budget splitting](budget-splitting-truncation.md)). Every episode records
total generated tokens (`total_gen`, all agents), critical-path tokens
(`cp_tokens`, the longest dependent chain, which tracks latency), uncached
prompt tokens, LM calls and peak context. The eval report reads accuracy
against both `total_gen` and `cp_tokens`.

**Regime for all numbers below:** untrained `tinker:Qwen/Qwen3.5-4B`
([entity](../entities/qwen3-5-4b.md); thinking, renderer `qwen3_5`, T=1,
top_p=1, top_k=−1) on Tinker. Benchmarks are [AIME25](../entities/aime-2025.md)
and [HMMT25](../entities/hmmt-feb-2025.md), n=30 each. G=1, one seed, lockstep,
math-verify, and a 32,768-token episode cap. Source:
[pilot README](../../sources/compute-matched-baselines-pilot.md) and
[report tables](../../sources/compute-matched-baselines-results.md).

## The budget is a cap, not a match [pilot]

All seven cells shared one 32,768-token cap, but realized mean `total_gen`
still differed from `single` by −47% to +20%:

| Cell | AIME25 mean total_gen (vs single) | HMMT25 mean total_gen (vs single) | mean cp_tokens (A / H) |
|---|---|---|---|
| [single](../entities/protocol-single.md) | 25,486 | 28,982 | 25,486 / 28,982 |
| [coordinator](../entities/protocol-coordinator.md) | 15,804 (−38%) ⚑ | 17,683 (−39%) ⚑ | 13,845 / 15,203 |
| [swarm4](../entities/protocol-swarm.md) | 27,176 (+7%) | 29,178 (+1%) | 8,802 / 9,793 |
| [sc4](../entities/protocol-independent.md) | 30,603 (+20%) ⚑ | 30,782 (+6%) | 7,699 / 7,699 |
| [debate3](../entities/protocol-debate.md) | 30,542 (+20%) ⚑ | 31,189 (+8%) | 10,388 / 10,410 |
| [ms3_notes](../entities/protocol-multi-session.md), fixed-harness rerun | 15,300 (−40%) ⚑ | not rerun | 15,300 / — |
| [ms3_compaction](../entities/protocol-multi-session.md), fixed-harness rerun | 13,526 (−47%) ⚑ | not rerun | 13,526 / — |

⚑ = flagged by the report as "Not compute-matched (>10% mean total_gen
difference from baseline)". The rule is in `src/marli/eval/report.py`. The
pilot's own multi-session rows (pre-fix harness) were flagged too; they are
superseded, see [multi-session carry](multi-session-carry.md).

Why realized compute differs under one cap:

- **Cells that stop early spend less.** The coordinator submits well before the
  cap. Multi-session agents submit in session 1 in more than half the episodes.
- **The baseline itself under-spends on easier problems.** On AIME, `single`
  has mean 25.5k but p50 31.2k. Parallel peers almost always run to their
  per-peer cap (sc4 mean 30.6k), so sc4 is flagged at +20% on AIME. The cause is
  that `single` stops early on some problems, not that sc4 exceeds the cap.
  On HMMT, `single` uses more of the budget (mean 29.0k), and the same sc4 cell
  is within 10%.
- **Critical path is a different axis.** Parallel cells have about a third of
  `single`'s critical path (sc4 7.7k, swarm4 8.8–9.8k, debate3 10.4k), so they
  are faster in wall-clock terms. No cell in this pilot is matched on
  `cp_tokens` with `single`.

## How to read comparisons

- A cell flagged ⚑ is **not** a matched-compute comparison, whatever its lift.
  If a cell ties `single` on fewer tokens (coordinator on AIME25, ms3_notes on
  the rerun), that is a point further down the accuracy–compute curve. It is
  not evidence of parity or superiority at equal compute. To make the
  comparison matched, either raise the cell's realized compute (bigger
  per-session budgets, a "use all sessions" instruction) or add a `single` cell
  at the lower budget. [pilot]
- The flag compares means only. Distributions can be bimodal. For example,
  ms3_notes on the rerun has p50 8.9k, mean 15.3k and p90 32.3k.
- Token counts are only comparable within a tokenizer (one model here).

## Tensions

- The pilot's pre-registered falsification test was a paired lift against
  **SC@4** at matched mean `total_gen`. The report's paired lifts are against
  **single**, and the committed tables contain no SC@4-baseline lift, so the
  hypothesis as written was not directly tested. [open]
- Whether a cell counts as compute-matched depends on the benchmark: sc4 is
  flagged on AIME25 (+20%) but not on HMMT25 (+6%) at the same budget, because
  `single`'s realized spend changes.

## Open questions

- Accuracy–compute **curves** (several budgets per protocol, including
  `single` at 8k and 16k) rather than one 32k point. That would give a
  matched-`cp_tokens` comparison for the parallel cells. [open]
- A 64k budget: `tinker_max_ctx` for this model is 65,536, and at 32k `single`
  is itself budget-bound. [open]

See also: [the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md),
[aggregation loss](aggregation-loss.md),
[unanswered episodes](unanswered-episodes.md).
