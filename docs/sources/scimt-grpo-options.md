---
type: source
title: scimt GRPOOptions (TRL GRPO backend knobs, with rationale comments)
description: The GRPOOptions dataclass from scimt's hf_grpo backend — its comments record RL-ops lessons (pods are not storage, budget-relative truncation, server-mode vLLM, zero-std/zero-gradient guards).
resource: https://github.com/ArcadiaImpact/science-of-midtraining (branch sid/dispatch-rlvr-prompt-align-v1) src/scimt/train/__init__.py
source_date: 2026-09-10
status: partial
provenance: copied verbatim on 2026-09-23 from /workspace/scimt-rlvr-prompt-align @ e5f681525162107e728d86d90d0e5d07c1fd4360, src/scimt/train/__init__.py lines 165-294 (dataclass body up to __post_init__)
tags: [lessons, rl, grpo, trl, scimt]
timestamp: 2026-09-23
---

```python
@dataclass(frozen=True)
class GRPOOptions:
    """TRL GRPO controls, with episodes counted as optimized completions."""

    episodes: int
    group_size: int = 16
    max_prompt_length: int = 3072
    max_completion_length: int = 1024
    # Pass enable_thinking=true to chat templates that expose a native
    # reasoning channel (Gemma 4 Unified). False preserves every existing
    # Gemma-3/Qwen prompt byte-for-byte.
    enable_thinking: bool = False
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    steps_per_generation: int | None = None
    # Generate this many times the optimizer batch's groups each update, score
    # them, and optimize only the most informative `1/oversample_factor` of
    # them. 1 disables selection (everything generated is optimized, TRL's own
    # behaviour). The optimizer batch, the loss normalizer and the update count
    # are identical either way -- only the generation batch grows, so the cost
    # is a multiple of generation alone, not of the backward pass.
    oversample_factor: int = 1
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    temperature: float = 1.0
    # Nucleus / top-k truncation of the rollout distribution. TRL's own
    # defaults (top_p 1.0, top_k 0) are NO truncation, i.e. pure temperature
    # scaling, and every run before 2026-09-10 used them implicitly.
    #
    # These are forwarded to GRPOConfig directly rather than through
    # grpo_optional_kwargs: they are core generation controls, so a TRL that
    # does not declare them should raise, not silently sample from a
    # distribution the caller did not ask for.
    #
    # Note the honest cost of truncating: TRL's importance ratio uses
    # full-softmax logprobs, so a truncated rollout distribution is not exactly
    # the policy it is scored against. The mismatch is small at these settings
    # and is the price of matching train-time sampling to the vendor's
    # recommended inference settings, which is what we are scored on.
    top_p: float = 1.0
    top_k: int = 0
    loss_type: str = "dr_grpo"
    scale_rewards: str | bool = "none"
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    vllm: str = "auto"
    # "colocate" runs vLLM inside the trainer process, sharing one GPU with
    # the optimizer; "server" talks to a separate `trl vllm-serve` process,
    # which is what lets generation use GPUs the trainer does not.
    #
    # Server mode keeps the TRAINER at one rank, which matters here: group
    # selection needs a whole group on one rank, so it refuses WORLD_SIZE > 1.
    # Sharding the trainer would forfeit selection; moving generation off-box
    # does not.
    #
    # In server mode the pool is the SERVER's (`trl vllm-serve
    # --gpu-memory-utilization`), so vllm_gpu_memory_utilization below is
    # unused, and sleep mode is meaningless because the server owns its cards
    # for the whole run -- both are refused rather than silently ignored.
    vllm_mode: str = "colocate"
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 1800.0
    vllm_gpu_memory_utilization: float = 0.2
    # Cap colocated vLLM context instead of allocating for a model's full
    # max_position_embeddings when prompts are much shorter.
    vllm_max_model_len: int | None = None
    vllm_enable_sleep_mode: bool = True
    # Sleep level 2 discards colocated vLLM weights each cycle, forcing a
    # full ~49GiB re-push per update on a 26B parent; level 1 offloads them
    # to host RAM (~1s restore) so an attention-only sync stays valid.
    vllm_sleep_level: int = 2
    # "attention_only" pushes only self_attn q/k/v/o tensors after the first
    # full sync — the only tensors an attention-only LoRA merge can change.
    vllm_sync_scope: str = "full"
    # Collapse each group's duplicated prompts into one vLLM request with
    # n=group_size (TRL's own server-mode strategy): prefill once per unique
    # prompt and share its KV across the group's completions.
    vllm_group_n_sampling: bool = False
    # JSONL receiving TRL ProfilingContext spans plus per-micro-step
    # training_step timings; None disables the recorder.
    profile_log_path: str | None = None
    stop_token_ids: tuple[int, ...] = ()
    mask_truncated_completions: bool = True
    log_completions: bool = True
    num_completions_to_print: int = 2
    log_unique_prompts: bool = True
    logging_steps: int = 1
    logging_first_step: bool = True
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    report_to: tuple[str, ...] = ()
    # Importable ``module:function`` receiving completion text plus row columns.
    reward_func: str | None = None
    # True Trainer checkpoint, distinct from the initial model weights in
    # TrainConfig.load_checkpoint_path.
    resume_from_checkpoint: str | None = None
    # Segmented curricula intentionally resume optimizer/model state on a new
    # dataset; opt out of Trainer's same-dataset batch skipping in that case.
    ignore_data_skip: bool = False
    rollout_log_dir: str | None = None
    # Importable ``module:function`` called with each saved checkpoint dir, for
    # copying it somewhere the pod's disk is not. A pod is not storage: deleting
    # one destroys its disk, and on 2026-09-02 a completed RL cell's only copy of
    # checkpoint-16 -- its resume point -- lived on a pod we were about to tear
    # down. At the measured 114.1 s/update a 64-update thinking cell is ~2 h of
    # 1xH200 (~$9 at $4.59/h) and a full 768-update cell ~24 h (~$112); the wall
    # clock is the real loss, not the dollars. The sync is advisory: a failure
    # warns and training continues, because losing the backup is not a reason to
    # lose the run.
    checkpoint_sync_func: str | None = None
    abort_log_path: str | None = None
    validation_dataset_path: str | None = None
    abort_eval_func: str | None = None
    parent_agreement: float | None = None
    parent_reward: float | None = None
    parent_completion_length: float | None = None
    zero_std_warmup_fraction: float = 0.10
    # Online truncation ceiling for the abort gate. Runaway truncation is
    # relative to the generation budget: 5% is alarming for a 512-token direct
    # cell and normal for a 4096-token thinking one.
    abort_truncation_rate: float = 0.05
    completion_length_window: int = 1024
    # Initial zero-gradient batches can be legitimate when a mature policy's
    # usable generation groups are reward-uniform. Keep the guard configurable
    # without changing optimizer math; fresh-policy runs retain the strict
    # three-log default.
    zero_gradient_abort_logs: int = 3

```
