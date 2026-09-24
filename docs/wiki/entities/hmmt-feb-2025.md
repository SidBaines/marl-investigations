---
type: entity
title: HMMT February 2025
description: MathArena/hmmt_feb_2025, 30 LaTeX-answer competition problems (CC BY-NC-SA, contamination-sensitive, never commit text); scored with math-verify.
resource: src/marli/tasks/hmmt_feb_2025.yaml
tags: [benchmark, math, hmmt, gated]
timestamp: 2026-09-24
---

# HMMT February 2025

Registry entry: `src/marli/tasks/hmmt_feb_2025.yaml`.

| Field | Value |
|---|---|
| HF dataset | `MathArena/hmmt_feb_2025`, split `train` |
| Size | 30 rows (fields: `problem_idx` → id, `problem` → prompt, `answer`) |
| Answer format | LaTeX. The verifier is math-verify |
| License | CC BY-NC-SA 4.0 |
| `commit_text` | **false**. Never commit question or transcript text; aggregates only |

**Results in this wiki:** untrained [Qwen3.5-4B](qwen3-5-4b.md) `single`
scored 0.533 [0.361, 0.698] (Tinker, n=30, G=1, 32k cap) [pilot]. That is the
same point estimate as on [AIME25](aime-2025.md), but at a higher realized
spend (mean `total_gen` 29.0k vs 25.5k). Every parallel cell loses more here
than on AIME25 (see [the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md)).
The multi-session cells have no valid HMMT25 numbers yet: the pilot's are
struck through, and the rerun is partial
([multi-session carry](../concepts/multi-session-carry.md)). [open]
