---
type: source
title: Compute-matched baselines pilot (untrained Qwen3.5-4B, AIME25 + HMMT25)
description: Untrained Qwen3.5-4B (Tinker) on AIME25 and HMMT25, n=30 each, G=1, seven protocol cells at a 32k-token episode budget, plus a multi-session rerun on the fixed harness (AIME25 only); no cell beats single.
resource: https://github.com/SidBaines/marl-investigations/blob/m3-harness/experiments/2026-09-23_compute-matched-baselines/README.md
source_date: 2026-09-24
status: pilot
provenance: copied verbatim on 2026-09-24 from branch m3-harness @ 4317902, experiments/2026-09-23_compute-matched-baselines/README.md (study written 2026-09-23, pilot and multi-session rerun results added 2026-09-24); PR https://github.com/SidBaines/marl-investigations/pull/4 (open). Grid configs are experiments/2026-09-23_compute-matched-baselines/configs/*.yaml @ 7d23f97 (pilot grid @ 95b59ef); full eval-report tables are archived in compute-matched-baselines-results.md.
tags: [experiment, evaluation, compute-matching, math, aime-2025, hmmt-feb-2025, qwen3.5-4b, tinker, untrained, multi-session, swarm, coordinator, debate]
timestamp: 2026-09-24
---

# Compute-matched baselines: agent systems vs a single agent (AIME 2025)

Status: pilot done (2026-09-24). The multi-session cells were rerun on the fixed harness for AIME25 only; the HMMT25 rerun is partial.

## Question

At an equal generated-token budget per episode, do coordinator-free swarms,
coordinator + workers, and multi-session single agents beat a single agent
(and N independent samples + vote) on hard math? This study establishes the
untrained baselines that every later RL study compares against.

## Hypothesis

At a 32k-token episode budget with Qwen3.5-4B, most multi-agent gains over
SC@4 are small or vanish at matched compute. Multi-session notes/compaction
help most on problems whose reasoning exceeds one context. Falsified if a
multi-agent cell beats SC@4 by a paired lift whose CI excludes 0 at matched
mean `total_gen`.

## Design / arms

- **Model / backend:** `tinker:Qwen/Qwen3.5-4B` (thinking), renderer
  `qwen3_5`, sampling T=1, top_p=1, top_k=−1.
- **Tasks:** AIME 2025 (30; MathArena, CC BY-NC-SA — never commit text).
- **Verifier:** math-verify (integer fast path). **G:** 1 episode per task.
  **Schedule:** lockstep.
- **Compute match:** `episode.max_gen_tokens = 32768` in every cell. Per-agent
  budgets split it by protocol shape.

| Cell | Protocol config | Per-agent budget |
|---|---|---|
| `single` (baseline) | `single` | 32768 (call ≤ 16384) |
| `sc4` | `independent_n4` (vote) | 4 × 8192 |
| `swarm4` | `swarm_n4` (notify delivery, vote) | 4 × 8192 |
| `coordinator` | `coordinator_default` | coordinator 16384 + ≤ 4 workers × 4096 |
| `ms3_notes` | `multi_session_s3_notes` | 3 sessions × ~10.7k |
| `ms3_compaction` | `multi_session_s3_compaction` | 3 sessions × ~10.7k |
| `debate3` | `debate_n3_r2` (push, vote) | 3 × 10922 over 2 rounds |

Report accuracy against both `total_gen` and `cp_tokens`, and paired lift vs
`single` (the eval report does both).

## Exact commands run

```bash
./run.sh smoke   # 3 tasks × 7 cells, grid max_usd=1
./run.sh pilot   # 30 tasks × 7 cells, grid max_usd=15
```

On the orchestration box, the same commands were run with the venv python
(`PYTHONPATH=src .venv/bin/python -m marli.cli.main …`) instead of `uv run`,
so the shared venv was not re-synced. The config is `configs/grid.yaml`.

## Results summary

**Pilot:** 30 tasks per benchmark × 7 cells, G=1, $14.00. There were 0 failed
episodes. Accuracy has Wilson 95% CIs. Lift is the paired difference vs
`single` on the same tasks, with wins/losses and an exact McNemar p.

| Cell | AIME25 acc | lift (W/L, p) | HMMT25 acc | lift (W/L, p) | total_gen (A/H) | cp_tokens (A/H) |
|---|---|---|---|---|---|---|
| single | 0.53 [0.36, 0.70] | — | 0.53 [0.36, 0.70] | — | 25.5k / 29.0k | 25.5k / 29.0k |
| coordinator | 0.57 [0.39, 0.73] | +0.03 (2/1, p=1.00) | 0.33 [0.19, 0.51] | −0.20 (2/8, p=0.11) | 15.8k / 17.7k | 13.8k / 15.2k |
| ms3_notes † | 0.47 [0.30, 0.64] | −0.07 (3/5, p=0.73) | 0.33 [0.19, 0.51] | −0.20 (2/8, p=0.11) | 19.0k / 20.3k | 19.0k / 20.3k |
| ms3_compaction † | 0.30 [0.17, 0.48] | −0.23 (1/8, p=0.04) | 0.33 [0.19, 0.51] | −0.20 (2/8, p=0.11) | 16.6k / 14.6k | 16.6k / 14.6k |
| swarm4 | 0.33 [0.19, 0.51] | −0.20 (0/6, p=0.03) | 0.23 [0.12, 0.41] | −0.30 (0/9, p<0.01) | 27.2k / 29.2k | 8.8k / 9.8k |
| sc4 | 0.37 [0.22, 0.54] | −0.17 (1/6, p=0.12) | 0.20 [0.10, 0.37] | −0.33 (0/10, p<0.01) | 30.6k / 30.8k | 7.7k / 7.7k |
| debate3 ‡ | 0.13 [0.05, 0.30] | −0.40 (0/12, p<0.01) | 0.03 [0.01, 0.17] | −0.50 (0/15, p<0.01) | 30.5k / 31.2k | 10.4k / 10.4k |

The full tables are in `results/pilot_{aime25,hmmt25}.RESULTS.md`; the smoke is
in `results/smoke.RESULTS.md`.

**Reading, at n=30:**
- At a 32k-token episode budget, splitting compute across parallel agents hurts
  a 4B thinking model. Each SC@4 or swarm peer gets 8k tokens, and Qwen3.5-4B's
  reasoning routinely runs past that, so every peer's chain is truncated.
- `oracle_any` shows that aggregation loses some correct answers:
  - swarm4 AIME: 0.53 oracle vs 0.33 voted;
  - sc4 HMMT: 0.37 vs 0.20.
- The coordinator roughly matches `single` on AIME, using ~40% fewer tokens.
  It is worse on HMMT.

**Caveats that change the reading:**
- † **Harness bug, fixed after the pilot** (`fix(interact)` 76aa955). Multi-session
  agents whose sessions ended by budget piled up nudge strikes across sessions.
  They were ended in the last session by `max_nudges` with **no answer**:
  ms3_compaction answered only 0.53 (AIME) and 0.70 (HMMT). These cells must be
  rerun before drawing any conclusion.
- ‡ Debate's per-round call cap (5.5k) truncates reasoning before a
  `\boxed{}` answer, so its answered rate was 0.23 / 0.07. That is a
  budget-design problem for thinking models, not evidence against debate.
- The budget is a **cap**. Realized `total_gen` differs by cell, and the report
  flags cells more than 10% off the baseline. Read accuracy against both
  `total_gen` and `cp_tokens`.
- n=30 per benchmark with G=1, so the CIs are wide.

**Multi-session rerun on the fixed harness (AIME25, 2026-09-24).** The cells
now answer every episode (answered 1.00). `single` is reused from the pilot via
`scores:` (same tasks, same seed). Full table:
`results/rerun_ms_aime25.RESULTS.md`.

| Cell | AIME25 acc | lift vs single (W/L, p) | total_gen mean / p50 | sessions used 1/2/3 | acc by sessions used 1/2/3 |
|---|---|---|---|---|---|
| single | 0.53 [0.36, 0.70] | — | 25.5k / 31.2k | — | — |
| ms3_notes | 0.50 [0.33, 0.67] | −0.03 (1/2, p=1.00) | 15.3k / 8.9k | 17 / 4 / 9 | 12/17, 2/4, 1/9 |
| ms3_compaction | 0.40 [0.25, 0.58] | −0.13 (2/6, p=0.29) | 13.5k / 11.4k | 14 / 4 / 12 | 4/14, 4/4, 4/12 |

What the rerun shows:
- Notes-to-self roughly ties `single` (−0.03, CI [−0.17, +0.07]) on about
  60% of the generated tokens.
- More than half the episodes submit in session 1. The 10.7k per-session
  budget ends a chain early, and the agent often answers rather than carry
  over, so realized compute is not matched. The report flags both cells.
- Episodes that spill into session 3 are mostly the hard problems (notes:
  1/9 correct). This is selection, not evidence that later sessions hurt.
- Compaction trails notes by 0.10. 16 of 30 compaction episodes needed a
  forced final, against 8 of 30 for notes. The transcripts have not been read
  yet, so the cause is still open.
- HMMT25 rerun is partial: 7 / 6 of 30 episodes ($0.45) before the $3 rerun
  budget ran out. It is not reported.

**Next:**
- Finish the HMMT25 multi-session rerun (~$2).
- Match realized compute: raise per-session budgets, or add a
  "use all sessions" instruction and compare at equal mean `total_gen`.
- Try a larger budget (64k) or fewer, longer-budget peers.
- Run debate with per-round caps that fit a thinking model.
- Use G≥2 for maj@k and avg@k.

## Deviations from the design

- 2026-09-23: HMMT Feb 2025 ran as a second grid inside the same $15 pilot
  budget (each grid capped at $7, with a $0.35 top-up to finish the HMMT
  debate cell after the per-cell split starved it).
- 2026-09-24: multi-session rerun (`configs/grid_rerun_ms_*.yaml`, Sid approved
  $3). The first attempt stopped because the grid charged the reused `single`
  Scores against its `max_usd` (fixed in 33d86f5). The AIME grid then resumed
  under `parallel_cells=1` so it would finish. HMMT was stopped at 7 / 6
  episodes to stay inside $3.

## Spend

Sid confirmed the smoke and pilot budgets on 2026-09-23.

| Item | Budget | Actual |
|---|---|---|
| Sanity check (single, 2 tasks) | $0.50 | $0.04 |
| Smoke (3 tasks × 7 cells) | $1 | $0.54 |
| Pilot (AIME25 + HMMT25, 30 tasks × 7 cells each) | $15 | $14.00 |
| Multi-session rerun (AIME25 complete, HMMT25 partial) | $3 | ~$2.60 (AIME $2.12, incl. $0.64 in two interrupted attempts; HMMT $0.45) |
