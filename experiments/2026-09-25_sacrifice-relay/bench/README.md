# Speed and memory benchmark (before the sacrifice-relay runs)

Status: running (2026-09-25). Pod `nu638jwugw3f2i` (`marli-bench`), 2×H200 SXM,
SECURE, CUDA 13.0 host, $9.18/hr.

## Question

How fast can we sample Qwen3.8-27B for this study, and does the learner fit?

1. **Serving config.** Which vLLM setup is fastest? Candidates: the baseline
   (TP1 on GPU 0, bf16), MTP speculative decoding, data-parallel over both
   GPUs (eval phases only) and TP2 (a reference for a future time-shared
   layout). Two workloads are measured:
   - **~16 concurrent agents**, the relay training step: latency-bound,
     because contributors run in turn;
   - **~128 concurrent agents**, filter/gate/solo: throughput-bound.
2. **Learner.** What are peak memory and step time on GPU 1 at our context
   (24k), how long does an adapter sync take, and how much does serving
   through the LoRA adapter slow decoding?
3. **The first 100 filter problems**, kept for the study. This is
   `../run.sh filter 100`: a paused run that `../run.sh filter` later resumes
   for the other 700 without re-sampling these.

## Method

- `bench.sh serve <variant>` starts `serve/<variant>.yaml` and a watcher that
  samples vLLM `/metrics` and `nvidia-smi` every 5 s (`metrics.py`).
- `bench.sh load <variant> <concurrency> <n>` runs throwaway single-agent
  code rollouts (the filter setup) on the first n candidate problems, one
  episode each. It then summarises two things over the run's wall-clock
  window:
  - server metrics: generation tok/s, inter-token latency, prefix-cache hit
    rate, KV usage, preemptions and spec-decode acceptance;
  - rollout facts (`episodes.py`).
- Sampling is T = 1 and top-p 1 in every variant. MTP speculative decoding
  uses rejection sampling, so the sampled distribution is unchanged.

  Before MTP is used for training we still check `kl_sample_train` and the IS
  ratio on a training step, because the returned logprobs must be the target
  model's.
- Learner: 2 `train rl` steps of `relay_team` (4 repos × G = 4) on 4
  unfiltered repos built from the first 16 candidates. The learner runs on
  GPU 1 against the serving config chosen above.

## Deviations

- Install: identical pins, but torch, NVIDIA and triton wheels came from
  `download.pytorch.org/whl/cu130`. PyPI served under 1 MB/s from this
  datacenter. The vLLM wheel was fetched with parallel range requests and its
  sha256 was verified.

## Results

_Pending._
