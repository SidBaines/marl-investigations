---
type: entity
title: "Protocol: independent (SC@k)"
description: N independent samples with no cross-peer tools or deliveries, aggregated by verifier-equivalent vote — self-consistency, i.e. the swarm with visibility off.
resource: src/marli/interact/configs/independent_n4.yaml
tags: [protocol, baseline, self-consistency, voting]
timestamp: 2026-09-24
---

# Protocol: `independent` (SC@k)

- **Config:** `independent_n4` (`protocol: independent`, `n_agents: 4`). This
  is four independent peers plus a vote over verifier-equivalent answers, with
  no cross-peer tools or deliveries. It is defined as a fixed
  [swarm](protocol-swarm.md) control in
  `src/marli/interact/protocols/presets.py`.
- **Role:** the "N independent agents + vote" baseline (`CLAUDE.md`). It is
  the [swarm](protocol-swarm.md) with visibility off.
- **In the compute-matched pilot (cell `sc4`):** 4 × 8,192 per agent (call
  ≤ 8,192). On AIME25 it scored 0.367 [0.219, 0.545] (oracle 0.367) with lift
  −0.167 [−0.333, −0.033] (1/6, p=0.125). On HMMT25 it scored 0.200
  [0.095, 0.373] (oracle 0.367) with lift −0.333 [−0.500, −0.167] (0/10,
  p=0.002). Mean `total_gen` was 30.6k / 30.8k and `cp_tokens` 7.7k. Every
  peer ran to its cap. Regime: untrained [Qwen3.5-4B](qwen3-5-4b.md) on
  Tinker, n=30, G=1. See
  [budget splitting](../concepts/budget-splitting-truncation.md) and
  [aggregation loss](../concepts/aggregation-loss.md). [pilot]
