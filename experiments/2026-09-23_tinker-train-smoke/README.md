# Tinker RL smoke: shared vs per-role LoRA; session credit all vs last

Status: done (2026-09-24).

## Question

Does the M3 training stack run end to end on Tinker? The chain is rollouts,
then credit, then datums, then per-learner `forward_backward` + `optim_step`,
then versioned sampler sync and checkpoints. It must work for the two
headline training axes:
- a coordinator with one shared LoRA vs one LoRA per role;
- a multi-session agent crediting all sessions vs only the last.

This is a plumbing smoke, not a science result: 3 steps, B=2 × G=2, tiny budgets.

## Design / arms

- **Model:** Qwen3.5-4B on Tinker, LoRA r=32, lr 1e-5, importance-sampling
  loss (sum-reduced), no grad clip.
- **Tasks:** POLARIS-53K, 32 shuffled tasks (training data; never commit text).
- **Credit:** team reward, role LOO baseline, `norm=mean`.

| Run | Protocol | Learners | Credit |
|---|---|---|---|
| `coord_shared` | coordinator_default | `shared` for coordinator + workers | all |
| `coord_perrole` | coordinator_default | `coord`, `work` | all |
| `ms_all` | multi_session_s3_notes | `solver` | segment_credit=all (session unit) |
| `ms_last` | multi_session_s3_notes | `solver` | segment_credit=last |

## Checks

For each run:
- metrics.jsonl has 3 steps;
- both learners step in `coord_perrole`;
- `kl_sample_train` ≈ 0 at step 0;
- sampler versions bump each step;
- the checkpoint manifest has state + sampler paths;
- in `ms_last`, datums only come from the last session.

## Exact commands run

`./run.sh`: each run is `marli train rl configs/base.yaml configs/<run>.yaml
tasks=<polaris TaskSet> max_usd=1.25`. On the orchestration box the venv
python was used instead of `uv run`.

## Results summary

All 4 runs completed 3 steps, for **$0.42 total**. The stack works end to end
on Tinker:
- `kl_sample_train` stayed in 2e-4 to 1e-3 at every step, so sampler and
  trainer agree on the exact token ids.
- Sampler versions bumped on every stepped learner.
- Each checkpoint manifest records Tinker state and sampler paths.
- `coord_perrole` stepped both the `coord` and `work` learners at step 0.

With B=2 × G=2, most groups had zero reward variance and were correctly
skipped. `ms_all` never stepped and `ms_last` stepped once, only on
last-session datums. So this smoke cannot compare credit schemes. A pilot
needs B≥8 and G≥4 per step.

The training-loop review found and fixed several issues (M3-5-fix):
- zero-advantage datums were billed and still stepped learners (e.g.
  `coord_perrole` step 0 `work`);
- an all-failed step would pass silently.

| Run | steps | spend |
|---|---|---|
| coord_shared | 3 | $0.134 |
| coord_perrole | 3 | $0.106 |
| ms_all | 3 | $0.082 |
| ms_last | 3 | $0.095 |

## Spend

Sid approved the M3 Tinker smoke on 2026-09-23.

| Item | Budget | Actual |
|---|---|---|
| 4 runs × 3 steps | $5 (4 × $1.25) | $0.42 |
