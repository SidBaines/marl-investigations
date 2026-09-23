# Compute-matched baselines: agent systems vs a single agent (AIME 2025)

Status: running (smoke).

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

_Pending._ Deliverables go to `results/` (aggregates only).

## Deviations from the design

- 2026-09-23: HMMT Feb 2025 deferred to a follow-up grid (one TaskSet per grid
  keeps the paired baseline simple).

## Spend

Sid confirmed the smoke and pilot budgets on 2026-09-23.

| Item | Budget | Actual |
|---|---|---|
| Sanity check (single, 2 tasks) | $0.50 | $0.04 |
| Smoke (3 tasks × 7 cells) | $1 | _pending_ |
| Pilot (30 tasks × 7 cells) | $15 | _pending_ |
