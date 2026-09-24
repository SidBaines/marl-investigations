# Local learner + vLLM on a pod

Before starting a pod or paid run, quote its hourly cost and budget and confirm
with Sid (CLAUDE.md). This milestone's smoke/pilot/debug pods were pre-approved
on **2026-09-23**; that approval does not remove spend limits. Record pod ID,
GPU, hourly rate, start/stop UTC, actual billed USD and the run paths in the
study notebook. The M4 local pilot budget is **$25**. CPU implementation tests
spend **$0** and do not establish GPU or model-family support.

## Layout and installation

Use `/workspace/marl-investigations` for the checkout and its `.venv` for the
learner. Install vLLM into **its own** `/opt/vllm` environment: its torch and
CUDA packages must not replace the learner's pins.

The host must support **CUDA ≥ 13.0**. Select it at pod creation with
`create-pod-cuda.sh … 13.0`; the H100 spike used a CUDA-13.0 host. The learner
uses the default PyPI torch 2.14.0 cu130 wheel; vLLM 0.30.0 brings torch 2.13
cu130 in its separate environment. Do not install `kernels>=0.13` with
transformers 5.5.4. `causal-conv1d` has no cu130 wheel; its torch fallback is
fine. `flash-linear-attention` and `fla-core` are pinned at 0.5.2.

```bash
cd /workspace/marl-investigations
uv sync --extra dev --extra render --extra eval --extra hub
uv pip install --python .venv/bin/python -r requirements/pod-train.txt
python3 -m venv /opt/vllm
/opt/vllm/bin/python -m pip install -r requirements/pod-vllm.txt
```

An activated repo venv can equivalently use
`python -m pip install -r requirements/pod-train.txt` if pip is installed there.
Keep weights/cache, exported adapters, checkpoints and logs on the pod's
workspace disk. Run from the checkout root; commit scientific configuration
before training (dirty training trees require explicit `MARLI_ALLOW_DIRTY=1`).
Keep `~/.triton/cache` on the pod's `/workspace` disk (or set
`TRITON_CACHE_DIR=/workspace/.triton/cache` for the learner). New sequence
lengths incur roughly **25 seconds** of Triton compilation; bucket lengths
to reuse those kernels. vLLM cold startup took roughly **220 seconds** in
the spike; the default 900-second readiness timeout accommodates it.

## GPU placement

For two GPUs, give vLLM physical GPU 0 and the learner physical GPU 1:

```bash
uv run marli serve vllm model=qwen3_5_4b python=/opt/vllm/bin/python detach=true \
  cuda_visible_devices=0 learner_ranks='[32]' --out runs/serve/q4
```

Set `CUDA_VISIBLE_DEVICES=1` for the training process and use backend
`device=cuda` (its visible GPU is numbered 0). Alternatively leave both visible
and use `device=cuda:1`. For co-location on one **80 GB GPU**, models **≤9B**
use `gpu_memory_utilization=0.5` and both processes on GPU 0. Co-location
worked in the H100 spike: Qwen3.5-4B's 32k learner step peaked at **25.7 GB**
alongside vLLM. Measure the intended model and context before a pilot.

The minimal launch command is:

```bash
uv run marli serve vllm model=qwen3_5_4b python=/opt/vllm/bin/python detach=true \
  max_model_len=32768 gpu_memory_utilization=0.5 max_lora_rank=32 max_loras=4 \
  --out runs/serve/q4
```

Without `detach=true`, the verb remains in the foreground until SIGINT/SIGTERM
and stops the entire child process group. Both modes publish
`runs/serve/q4/server.json` after `GET /v1/models` succeeds. Detached servers
survive launcher exit; their stdout/stderr go directly to `server.log`.
Foreground output is also teed to stderr. Never expose the dynamic adapter
API on an untrusted network; the default bind is loopback.

The server uses `--generation-config vllm` and the model registry's context
limit. Override `max_model_len` only downwards. Set `max_lora_rank` at least as
large as every learner's rank; `learner_ranks='[32,32]'` checks planned ranks
and capacity before launch. Set `max_loras` at least to the number of learners
plus one publication slot, with more slots for retained snapshots. Explicit
unloads remove the oldest snapshot of the publishing learner after its new
snapshot is live. Finish rollouts before syncing: eviction is not a lease for
arbitrary asynchronous consumers of an old policy.

`extra_args='[--dtype,bfloat16]'` appends literal argv entries. It is an escape
hatch; overriding managed flags can invalidate manifest limits and parity.
The H100-verified argv is `--generation-config vllm --enable-lora
--max-lora-rank 32 --max-loras 4 --max-model-len 32768
--gpu-memory-utilization 0.5 --seed 0`, with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (set by the launcher).

Qwen3.5 trains with the text-only `Qwen3_5ForCausalLM`: its logprobs matched
the vision-language forward exactly in the spike. Registry LoRA targets cover
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
`in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, and `down_proj` (64.9M trainable
parameters at rank 32 on 4B). Without registry targets the learner adapts all
text-model linear layers. The only loss path projects hidden states in
1024-position chunks with **per-chunk gradient checkpointing**. Chunking alone
retains logits for backward. Measured learner peaks were 13.8/17.5/25.7 GB at
8k/16k/32k, roughly 6k tokens/second; full 32k logits alone needed 30 GiB fp32
and OOMed.

For the registered Qwen3.5-4B/9B and Qwen3.6-35B-A3B export maps, `save_adapter`
rewrites `base_model.model.model.layers.*` to
`base_model.model.model.language_model.layers.*`. vLLM accepted the former
with HTTP 200 but silently ignored it. The adapter's `manifest.json` records
the map; learner state restore reverses it. Mapping metadata does not enable
local training for entries still marked `local: no`.

## Connect training

The backend factory accepts these exact kwargs:

```python
backend = make_backend(
    "local", spend=spend,
    server_json="/workspace/marl-investigations/runs/serve/q4/server.json",
    adapters_dir="/workspace/marl-investigations/runs/train/q4/adapters",
    device="cuda",
    adapter_names="versioned",  # default; "inplace" opts into a stable serving name
    adapter_check_tol=0.05,      # mean absolute logprob drift, in nats
)
```

Each learner shares one base model and publishes a unique name such as
`q4-x-s12`. The adapter directory is never overwritten or deleted when the
server unloads it. Both the adapter list and served model index are updated in
`server.json`; sampling refs have the form
`vllm:@/workspace/marl-investigations/runs/serve/q4/server.json#q4-x-s12`.
`close()` unloads this backend's adapters; keep their files to reload for eval.

Every hot-load scores the same fixed probe ids in the learner (adapted and
base) and in vLLM (adapted and base). The vLLM 0.30 request uses
`/v1/completions`, `echo=true`, `prompt_logprobs=0`, `max_tokens=1`; scores
come from `choices[0].prompt_logprobs`, excluding the null first position.
Publication raises `BackendError` if vLLM returns base-identical scores while
the learner adapter changes them, or mean absolute learner/vLLM drift exceeds
`adapter_check_tol` (default **0.05 nats**, typical spike drift **0.005**).
The initial version-zero base policy needs no check; restoring weights at
version zero does hot-load an adapter and is checked.

`adapter_names="versioned"` keeps unique serving names and is safer for vLLM's
prefix cache. `"inplace"` reuses the first serving name with `load_inplace=true`
on subsequent loads. Continue passing unique snapshot names to `sync_sampler`:
each export gets its own directory in both modes. Inplace saves adapter slots
but old policy refs now resolve to new weights; finish rollouts before syncing.
An uncertain or failed inplace update blocks new `policy()` calls until a
verified sync succeeds.

gpt-oss defaults to attention-only LoRA and `attn_implementation="flex_attention"`;
eager attention OOMed at 8k. Expert LoRA is experimental: PEFT
`target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` showed
about **6% over-application** in vLLM. `LocalLearner` rejects these targets unless
constructed with `allow_unverified_expert_lora=True`; it never enables them by
default. The hot-load parity check still applies. gpt-oss code-tool bodies may
be raw code with `code` or `<|constrain|>code` headers; Harmony binds non-JSON
bodies only when the advertised tool has exactly one string parameter.

The training CLI exposes `local_server_json` and `local_adapters_dir`, and
uses the backend defaults for adapter naming and tolerance. The latter options
are currently available through the Python backend factory above; adding CLI
overrides requires extending `TrainRLConfig` and its backend wiring.

```bash
CUDA_VISIBLE_DEVICES=1 uv run marli train rl experiments/<study>/train.yaml \
  learners.x.backend=local learners.x.base_model=qwen3_5_4b \
  local_server_json=runs/serve/q4/server.json \
  local_adapters_dir=runs/train/q4/adapters \
  max_usd=25 --out runs/train/q4
```

`max_usd` does not itself account for an already rented GPU: monitor actual
pod billing and stop before the approved cap.

## Inspect, stop and restart

```bash
uv run marli serve status server_json=runs/serve/q4/server.json --out runs/serve/q4-status --force
uv run marli serve stop server_json=runs/serve/q4/server.json --out runs/serve/q4-stop --force
```

Status records both PID liveness and HTTP health. Control commands produce
separate observation handles. `--force` on their **separate** output dirs
requests a fresh observation/action instead of reusing a completed run.
Stop sends SIGTERM to the server group, then SIGKILL after `timeout_s` (default
5 seconds). It preserves the source server manifest and logs.

Serving runs obey the shared completed-run no-op contract. A stopped server's
manifest is therefore historical, not a request to relaunch it. Confirm stop
first, preserve needed logs, then launch into a **new output directory**. Never
use `--force` on a live serving directory: that would erase its ownership record.

## Checkpoints off-pod and teardown

For now, rsync results back to the **orchestration box's `/workspace`** before
teardown; do not use HF for this workflow. From the orchestration box:

```bash
rsync -a --checksum pod:/workspace/marl-investigations/runs/ /workspace/marl-investigations/runs/
```

Copy resumable **state** (optimizer plus adapter) and exported sampler
adapters, along with checkpoint manifests and metadata. A sampler alone cannot
resume training. Local absolute input references need rebasing when restoring
on a different pod; keep directory layout and provenance hashes intact.
Verify file inventory and hashes against pod bytes before teardown; test a
state restore and record the destination and checksums in the study notebook.
Keep raw benchmark outputs private; never commit gated prompts or transcripts.

1. Finish or stop training and save its final resumable checkpoint.
2. Rsync state, sampler bytes and raw results to the orchestration box's
   `/workspace`; verify that copy.
3. Save aggregate results, configurations, provenance and spend records.
4. Run `serve stop`, then a fresh `serve status`; verify `running=false`.
5. Stop/delete the pod through the approved pod lifecycle workflow only after
   verifying checkpoint persistence. Record final billed spend. Pods are not storage.
