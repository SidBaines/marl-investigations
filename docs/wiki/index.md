# Research wiki

The catalog of curated knowledge and its sources; start here when querying past findings.

## Concepts

_none yet_

## Entities

_none yet_

## Syntheses

_none yet_

## Sources

- [scimt GRPOOptions](../sources/scimt-grpo-options.md) — The GRPOOptions dataclass from scimt's hf_grpo backend — its comments record RL-ops lessons (pods are not storage, budget-relative truncation, server-mode vLLM, zero-std/zero-gradient guards).
- [Lessons from scimt prior-latmem](../sources/scimt-prior-latmem-lessons.md) — Ops postmortems from scimt's prior-latmem experiment — reasoning-token budgets, the always-on ChatClient cache collapsing samples, per-batch persistence, kill-testing crash paths, spend control.
- [scimt RLVR throughput matrix](../sources/scimt-rlvr-throughput-matrix.md) — Measured throughput/memory matrix and failure modes from scimt's TRL-GRPO RLVR runs — fp32 logits-upcast OOMs, gradient checkpointing, vLLM engine hangs, ninja on PATH.
