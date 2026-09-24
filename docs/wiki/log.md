# Wiki log

## [2026-09-24] ingest | Compute-matched baselines pilot + Tinker RL smoke

First experiment ingest. Sources added:

- [Compute-matched baselines pilot](../sources/compute-matched-baselines-pilot.md)
  (README @ 4317902, status pilot).
- [Its eval-report tables](../sources/compute-matched-baselines-results.md):
  three RESULTS.md files, verbatim and concatenated.
- [Tinker RL smoke](../sources/tinker-rl-smoke.md) (README @ f62bc7a, status
  pilot; infrastructure validation only).

Both come from branch m3-harness, PR #4 (open).

Headline, recorded in the new synthesis
[agent systems vs single at matched compute](syntheses/agent-systems-vs-single-matched-compute.md):
untrained Qwen3.5-4B at a 32k budget (n=30, G=1) shows no protocol beating
`single` on AIME25/HMMT25. Parallel splitting loses at matched compute.
Coordinator and notes tie on AIME25 at ~60% of the tokens. Everything is
[pilot].

The pilot's multi-session numbers are struck through (nudge-strike harness
bug, 76aa955) and replaced by the AIME25 rerun. HMMT25 multi-session is marked
unmeasured.

Pages created:

- concepts: [compute-matching](concepts/compute-matching.md),
  [budget-splitting-truncation](concepts/budget-splitting-truncation.md),
  [aggregation-loss](concepts/aggregation-loss.md),
  [multi-session-carry](concepts/multi-session-carry.md),
  [unanswered-episodes](concepts/unanswered-episodes.md),
  [on-policy-check](concepts/on-policy-check.md),
  [zero-variance-groups](concepts/zero-variance-groups.md). The last three
  cross-link the scimt sources.
- entities: [qwen3-5-4b](entities/qwen3-5-4b.md), [tinker](entities/tinker.md),
  [aime-2025](entities/aime-2025.md), [hmmt-feb-2025](entities/hmmt-feb-2025.md),
  [polaris-53k](entities/polaris-53k.md), and the protocol cards
  [single](entities/protocol-single.md),
  [independent](entities/protocol-independent.md),
  [swarm](entities/protocol-swarm.md),
  [coordinator](entities/protocol-coordinator.md),
  [multi-session](entities/protocol-multi-session.md),
  [debate](entities/protocol-debate.md).
- synthesis: [agent-systems-vs-single-matched-compute](syntheses/agent-systems-vs-single-matched-compute.md).

Also updated [index.md](index.md) and removed the `.gitkeep` placeholders.

Tensions recorded:

- The pre-registered test (lift vs SC@4) was not computed.
- The rerun's `single` baseline predates the harness fix.
- Compaction's session-1 accuracy (4/14) is far below notes' (12/17), and its
  forced finals (16) outnumber the episodes that reached session 3 (12).
- The fix commit counts 13/30 unanswered; the report implies 14/30.
- The smoke's "both learners stepped" check passed partly on a zero-advantage
  step.

Candidate follow-ups:

- 64k budgets; G≥2 or a second seed.
- A debate budget redesign.
- Finish the HMMT25 multi-session rerun.
- A lift-vs-SC@4 report.
- `single` at 8k/16k budgets, for matched `cp_tokens`.
- An RL pilot at B≥8, G≥4.

## [2026-09-23] schema | wiki created

Adapted the scimt wiki schema for multi-agent RL in [CLAUDE.md](CLAUDE.md)
and created [index.md](index.md) and the empty concept, entity, and synthesis
directories. Three scimt lesson sources were seeded in `docs/sources/`:

- [scimt GRPOOptions](../sources/scimt-grpo-options.md)
- [Lessons from scimt prior-latmem](../sources/scimt-prior-latmem-lessons.md)
- [scimt RLVR throughput matrix](../sources/scimt-rlvr-throughput-matrix.md)

No concept pages yet.
