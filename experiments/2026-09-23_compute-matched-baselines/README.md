# Compute-matched baselines: agent systems vs a single agent (AIME 2025)

Status: pilot done (2026-09-24); multi-session cells need a rerun on the fixed harness.

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

**Next:**
- Rerun the † cells on the fixed harness.
- Try a larger budget (64k) or fewer, longer-budget peers.
- Run debate with per-round caps that fit a thinking model.
- Use G≥2 for maj@k and avg@k.

## Deviations from the design

- 2026-09-23: HMMT Feb 2025 ran as a second grid inside the same $15 pilot
  budget (each grid capped at $7, with a $0.35 top-up to finish the HMMT
  debate cell after the per-cell split starved it).

## Spend

Sid confirmed the smoke and pilot budgets on 2026-09-23.

| Item | Budget | Actual |
|---|---|---|
| Sanity check (single, 2 tasks) | $0.50 | $0.04 |
| Smoke (3 tasks × 7 cells) | $1 | $0.54 |
| Pilot (AIME25 + HMMT25, 30 tasks × 7 cells each) | $15 | $14.00 |
