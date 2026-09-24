# Local learner pilot: per-role LoRA RL on one H100 (PEFT + vLLM)

Status: done (2026-09-24). This validates the infrastructure only; it makes no
claim about learning.

## Question

Does the local backend work end to end on real hardware? That means our PEFT
multi-adapter learner with chunked logprobs, co-located with vLLM multi-LoRA
serving that is updated by hot-load. Specifically:

- Does it train two per-role LoRAs on-policy (IS ratio ≈ 1 per adapter)?
- Does vLLM actually serve the trained adapters (post-hot-load effect check)?
- Does a hard-killed run resume?

This is the M4 gate from `docs/plans/2026-09-23-agent-systems.md`: per-adapter
IS ratio ≈ 1, a 20-step per-role-LoRA run, and kill + resume.

## Hypothesis

Sampler/learner agreement is at the level of the M2 spike, so
`kl_sample_train` is about 1e-3 or less and the mean IS ratio is within 1e-3 of
1 for each adapter. Every hot-loaded adapter passes the effect check. A run
killed with SIGKILL resumes from its last checkpoint and completes.

The hypothesis is falsified by any of: a mean IS ratio off by more than 1e-2,
a hot-load that vLLM ignores, or a resume that fails or silently restarts from
scratch.

## Design / arms

- **Hardware:** pod `qy3ylhqo101lua`, 1× H100 80GB SXM (SECURE, CUDA 13.0 host,
  $3.49/hr).
  - vLLM 0.30.0 runs in its own venv at `gpu_memory_utilization=0.5`, with
    `max_loras=4` and `max_lora_rank=32`.
  - The learner runs in-process with `train rl` on the same GPU (torch 2.14
    cu130, transformers 5.5.4, peft 0.21.0).
- **Model:** `qwen3_5_4b` (Qwen/Qwen3.5-4B, thinking).
  - The learner uses the text-only `Qwen3_5ForCausalLM`.
  - LoRA r=32 on the 12 registry target modules (attention, GatedDeltaNet
    projections, MLP).
  - AdamW at lr 1e-5, no grad clipping, importance-sampling loss, sum-reduced.
- **Protocol:** `coordinator_default`.
  - Per-role LoRA: learner `coord` for the coordinator and learner `work` for
    the workers.
  - Default credit: team reward, role LOO baseline, mean norm.
- **Limits (tiny):**
  - episode: 8192 generated tokens, 24 ticks;
  - coordinator agent: 4096 tokens;
  - each worker: 2048 tokens;
  - spawns: ≤ 2 workers per call and 2 in total;
  - ctx 16k.

  Lockstep, T=1 / top_p=1 / top_k=−1, math env with no python tool.
- **Tasks:** POLARIS-53K train: 256 tasks (`data build source=polaris_53k
  max_n=256 shuffle=true seed=0`), which the loop cycles through with epoch
  shuffles.
- **Runs:**
  - `smoke`: 2 steps, B=2 × G=2.
  - `pilot`: 20 steps, B=8 × G=4 (32 episodes/step), checkpoints every 5.
  - `resume`: 6 steps, B=2 × G=2, checkpoints every 2. It is SIGKILLed after 3
    logged steps (last checkpoint: step 1) and resumed with the same command.
  - `resume_g4`: the same, with CLI overrides B=4 × G=4, 4 steps, checkpoints
    every step, SIGKILLed after 2 steps. The larger groups make sure a learner
    trains after the resume.

## Exact commands run

On the pod, from a clone of branch `m4-harness`, with `uv run --no-sync` so
the pod-train pins survive:

```bash
marli serve vllm model=qwen3_5_4b python=/workspace/venv-vllm/bin/python detach=true \
  max_model_len=32768 gpu_memory_utilization=0.5 max_lora_rank=32 max_loras=4 \
  'learner_ranks=[32,32]' --out out/serve
marli train rl configs/base.yaml configs/<run>.yaml tasks=out/tasks/taskset.json \
  max_usd=1 local_server_json=out/serve/server.json concurrency=32 --out out/<run>
```

`run.sh` records the same sequence. The kill+resume driver was a two-line loop
that ran `kill -9 <train pid>` once `metrics.jsonl` had 3 rows and then re-ran
the identical command. Outputs were rsynced to
`/workspace/marl-investigations/experiments/2026-09-24_local-learner-pilot/out/`
on the orchestration box. They are gitignored and include adapters, learner
states and rollouts.

## Results summary

**Smoke (2 steps):**
- Both learners stepped and hot-loaded; the effect check passed.
- IS ratio mean: coord 0.99998, work 0.99992.
- `kl_sample_train`: 3.6e-4 / 4.6e-4.

**Pilot (20 steps × 32 episodes).** `results/pilot_metrics.txt` holds the
per-step table.

| Metric | coord (coordinator) | work (workers) |
|---|---|---|
| steps trained | 20 / 20 | 19 / 20 (step 5: no informative worker datums) |
| IS ratio mean, over steps: mean [min, max] | 0.99996 [0.99970, 1.00035] | 1.00005 [0.99950, 1.00126] |
| IS ratio max per step (range) | 1.26 – 2.46 | 1.22 – 2.08 |
| `kl_sample_train` mean / max | 3.9e-4 / 6.8e-4 | 3.4e-4 / 9.2e-4 |
| action tokens per step (mean) | 48.3k | 15.8k |
| hot-loads passing the effect check (absolute drift ≤ 0.05 nats: the pilot ran at 2a265c2, before the effect-based check) | 20 | 19 |

- **Episodes:**
  - mean `total_gen` 5.0k, `cp_tokens` 4.5k, 4.6 LM calls;
  - about 1.0 worker spawned per episode;
  - 4.6 of 8 groups per step had zero variance and carried no gradient.
- **Throughput per step** (co-located on one GPU):
  - sampling ≈ 66 s for 32 episodes;
  - training for both learners (fwd/bwd, adapter export, hot-load, effect
    check) ≈ 21 s;
  - step 0 took 83 s of training because of Triton compilation.
- **Accuracy per step:** 0.19–0.66, mean 0.31, with no trend (first half 0.33
  vs second half 0.30). With 8 fresh tasks per step at lr 1e-5, 20 steps are
  not expected to move accuracy, and this run was not designed to detect
  learning.

**Kill + resume**, with `kill -9` on the training process mid-run and then
the same command re-run. Two runs, both on the final code:

| Run | Killed after | Stale adapters at resume | Restored versions (coord/work) | After resume | Status |
|---|---|---|---|---|---|
| `resume` (B=2×G=2, 6 steps, ckpt every 2) | 3 steps logged (ckpt step 1) | 1, evicted | 1 / 0 = recorded | step 2 re-run; steps 2–5 all zero-variance, so no updates | completes, `status: resume` |
| `resume_g4` (B=4×G=4, 4 steps, ckpt every step) | 2 steps logged (ckpt step 1) | 2, evicted | 1 / 1 = recorded | coord trained at steps 2 and 3 (v2, v3), IS ratio 0.9998 / 1.0000, KL 6.1e-4 / 2.5e-4; work had no informative datums | completes, `status: resume` |

- The first resume attempt of `resume` failed loudly on bug 5
  (`restored version 0, expected 1`); it was fixed and re-run.
- On every resume the loop logged a warning that the training commit had
  changed. Provenance is recorded per checkpoint step (`history_state[step]`).
- Sampler names of re-run steps carry a new attempt suffix, so none collided.

**Bugs found on the pod and fixed on `m4-harness`,** each with a regression
test:
1. `serve vllm` started vLLM without its venv's `bin/` on PATH. The flashinfer
   JIT then failed with `ninja` not found (216448e).
2. Learner states were written relative to the CWD, i.e. into the git
   checkout. The next run then refused the dirty tree (2a265c2).
3. A killed run's adapters stayed loaded and held every `max_loras` slot. The
   backend now evicts adapters whose owner process is dead (d86f2a0).
4. A resume re-runs steps after its last checkpoint and reused sampler names.
   Names now carry a per-attempt suffix (d86f2a0).
5. Local `init_from` published restored weights at version 0 instead of the
   recorded version, so the first real resume failed the loop's version
   assertion (8ec026f).
6. From review, not observed on the pod: the effect check now compares
   adapter *effects*, (vLLM adapted − vLLM base) vs (learner adapted − learner
   base). That cancels the ~0.03-nat HF/vLLM kernel mismatch on short probes
   (d86f2a0).

## Deviations from the design

- 2026-09-24: The plan was to kill the 20-step pilot mid-run and resume it.
  Review found two resume bugs while the pilot was running (3 and 4 above), so
  the pilot ran uninterrupted. Kill + resume was tested in a separate 6-step
  run on the fixed code, which exposed bug 5.
- The smoke's learner states landed in the checkout root (bug 2). They were
  kept by a pod-local `.git/info/exclude` so the smoke checkpoint stays valid,
  and copied off-pod with the rest.

## Spend

Sid pre-approved the M4 pod for the local smoke, pilot and debugging on
2026-09-23 (the plan budget is ≤ $25). There was no API or Tinker spend
(`spend.spent_usd` = 0).

| Item | Budget | Actual |
|---|---|---|
| H100 pod `qy3ylhqo101lua`: M2 model spike + M4 smoke, pilot, two resume checks (4h 16m, 2026-09-23 22:36 → 2026-09-24 02:53 UTC) | ≤ $25 | ~$14.92 |

The pod was deleted after its outputs were copied to the orchestration box and
verified: 394 files and 29.25 GB identical, and sha256 matched for 43 key
files (checkpoints, metrics, optimizer states, final adapters). The receipt is
`out/CLEANUP_RECEIPT.md`.
