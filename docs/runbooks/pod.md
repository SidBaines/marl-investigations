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

### Option 2: both GPUs in both phases

The split above leaves each GPU idle for half of every synchronous step:
vLLM samples on GPU 0 while GPU 1 waits, then the learner trains on GPU 1
while GPU 0 waits. Time sharing keeps both busy in both phases:

- **Sampling:** one vLLM server, tensor-parallel over both GPUs, capped at
  45% of each GPU's memory. The learner's model copies stay resident beside it.
- **Training:** vLLM sleeps (weights to CPU, KV cache freed). The learner runs
  data-parallel, with one full base copy per GPU, on token-balanced shards of
  the batch. The shards' LoRA gradients are summed with NCCL, so the update is
  the one-GPU update up to float rounding.
- **Hand-off:** each rank empties its CUDA cache. vLLM wakes, and the new
  adapter is loaded and probe-checked as usual.

Serve with sleep mode (`experiments/2026-09-25_sacrifice-relay/bench/serve/tp2_mtp2_sleep.yaml`):

```yaml
model: qwen3_8_27b
python: /opt/vllm/bin/python
cuda_visible_devices: "0,1"
tensor_parallel_size: 2
gpu_memory_utilization: 0.45
enable_sleep_mode: true        # adds --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1
max_model_len: 32768
max_lora_rank: 32
max_loras: 4
learner_ranks: [32]
extra_args: ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}']
```

```bash
uv run --no-sync marli serve vllm tp2_mtp2_sleep.yaml detach=true --out runs/serve/tp2
```

Train **without** `CUDA_VISIBLE_DEVICES`: both processes see both GPUs.

```bash
uv run --no-sync marli train rl base.yaml arm.yaml tasks=... max_usd=1 \
  local_server_json=runs/serve/tp2/server.json \
  'local_devices=[cuda:0,cuda:1]' local_sleep_sampler=true --out runs/train/arm
```

Rank 0 runs inside the trainer on `cuda:0`. Rank 1 is a spawned worker on
`cuda:1`; its log is `<run>/learner-ranks/rank1.log`, and its tail appears in
any `BackendError` it causes. Both options are
runtime-only, so they do not change a run's identity: a run may resume with or
without them. `metrics.jsonl` gains `backend_metrics.local.{sleep_s,wake_s}`
per step. A failed train step and `close()` both wake the server, so it is
never left asleep.

Measured on 2×H200 with Qwen3.8-27B (see the sacrifice-relay `bench/README.md`):
- sleep: 0.7 s after the first (14 s);
- wake: under 1 s;
- memory: about 115 GiB used on GPU 0 with the server awake beside the learner.

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
the learner adapter changes them, or if the adapter's *effect* disagrees: the
mean over probe positions of |(vLLM adapted − vLLM base) − (learner adapted −
learner base)| exceeds `adapter_check_tol` (default **0.05 nats**). Comparing
effects cancels the ~0.03-nat HF/vLLM kernel mismatch that even the base model
shows on a short context-free probe; both drifts are logged at INFO.
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

Qwen3.6-35B-A3B (Qwen3.5-MoE; `local: unverified` until a pod run passes) trains
through `Qwen3_5MoeForCausalLM` with `experts_implementation="grouped_mm"`. Its
256 routed experts are fused 3-D parameters, so the registry's Linear targets
reach attention, linear attention and the shared expert only; the routed
experts and the router stay frozen (PEFT `target_parameters` adds the expert
update into the bf16 weight on every forward, which rounds small updates away;
gpt-oss expert LoRA showed the 6% vLLM mismatch above). Serve it with `lora_target_modules` equal to those targets so vLLM
does not wrap the experts either; a learner whose targets the server does not
wrap is refused. `experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/`
has the serve config (`configs/serve_a3b.yaml`) and a pod preflight
(`a3b_preflight.py`: memory, logprob agreement, 32k step, hot-load).

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

## Dashboard

`marli dashboard` is a reusable live GUI for any runs: evals, training, data
verbs, servers and GPUs. It only observes run dirs (it never takes a run's lock;
a run counts as running while some process holds its flock, per `/proc/locks`),
and its snapshot holds numbers, agent ids and run metadata only: never prompts,
tool output or answers.

**Per study, a small YAML** sets the title and any charts, e.g.
`experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/dashboard.yaml`:
`grouped` maps a grade component to a chart title (one chart per component,
a line per agent plus the average), `agent_labels` names the agents and
`smooth_steps` sets the rolling mean. `split_by` names a 0/1 grade component
(e.g. `rule_known_at_start`) that splits each agent's turns by its own value:
per other grouped component, a pair of charts (where it is 1, where it is 0,
sharing one y-range) with a line per `split_agents` agent plus all of them
pooled, then the same pair counting turns. Everything is a runtime setting, so
the same `--out` can be reused.

**Run it where the run dirs are** (a pod, the dev box, or a laptop with
synced copies) and leave it running; it refreshes every `watch_s` (default
`refresh_s`, 60 s) and the open page picks up each new snapshot by itself:

```bash
uv run --no-sync marli dashboard <study>/dashboard.yaml 'roots=[<study>/out]' \
  serve_port=8765 --out runs/dashboard
# -> "dashboard live at http://127.0.0.1:8765/" (also in runs/dashboard/serve.json)
```

Ways to open it:
- **Same machine, or an SSH tunnel** (`ssh -L 8765:127.0.0.1:8765 <host>`):
  open `http://127.0.0.1:8765/`. On loopback no key is needed.
- **A RunPod pod's HTTPS proxy, from any browser or phone:** bind every
  interface on a port the pod exposes as `/http` (our pods expose `8888`; stop
  Jupyter first if it holds it), e.g. `serve_host=0.0.0.0 serve_port=8888`, then
  open `https://<pod-id>-8888.proxy.runpod.net/?key=<key>`. Off loopback every
  request needs the access key: the logged URL carries it once, the server
  swaps it for an HttpOnly cookie and drops it from the address bar. Set
  `MARLI_DASHBOARD_KEY` to keep one key (and bookmark) across restarts;
  otherwise each start makes a new random key. The key never enters a config
  or manifest, only `serve.json` (mode 0600) and the log.

Only the page and `snapshot.json` are served. `annotations=<file.json>` puts
off-pod facts at the top (pod id, $/hr, spend). Without `serve_port`, the verb
writes `snapshot.json`, `index.html` (an Artifact fragment) and
`standalone.html` once, or every `watch_s` seconds. Episode files are folded
incrementally: `cache.json` in the out dir keeps each file's offset. SIGINT or
SIGTERM stops watching or serving and records the manifest.

## Continue truncated episodes

`eval continue` extends agents that ran out of budget in a saved `eval rollout`
run, without re-sampling what already happened. It replays each selected
episode's recorded calls exactly: tool calls run again for real to rebuild the
sandbox, but the agent sees the recorded results. The agent that hit a limit
then keeps generating under the new limits, and every agent after it runs
live. The result is distributed as if the source had used the new limits from
the start, because prompts never state the budget.

Only lockstep `single`, `relay` and `multi_session` runs are supported. The
output is a complete EpisodeSet with the same episode ids; untruncated and
not-ok episodes are copied verbatim. `continuations.jsonl` records each
episode's cut, replayed calls, tool-result mismatches and any divergence.

The study's filter run, with the budget raised to 20,480 tokens (run from the
checkout root, with the filter's vLLM server up):

```bash
S=experiments/2026-09-25_sacrifice-relay
uv run --no-sync marli eval continue episodes=$S/out/filter_rollout \
  limits.agent.max_gen_tokens=20480 limits.episode.max_gen_tokens=20480 \
  'policies.q.ref=vllm:@'"$PWD/$S"'/out/serve/server.json#qwen3_8_27b' \
  policies.q.renderer=qwen3_8_medium policies.q.model=qwen3_8_27b \
  concurrency=160 max_usd=1 --out $S/out/filter_rollout_20k
```

The source must be complete (not paused). `policies` defaults to the source's,
so the `policies.q.*` lines are needed only when the server path changed; the
model and renderer must match the source's.

For relay runs, raise `limits.episode.max_gen_tokens` to at least
`n_agents × agent`. Relay adjusts limits and rejects a smaller episode budget.

A change that would alter a replayed call is refused or detected: limits
quoted in a system prompt, the session count, or a larger `call.max_tokens`
where an earlier completion was cut at the cap. `on_divergence=fail` (the
default) marks such an episode not-ok and does not retry it; `live` switches
it to live sampling and flags it.

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

## Kill and resume

A killed run never calls `close()`, so its adapters stay loaded in vLLM and
listed in `server.json`. The next `LocalBackend` unloads adapters whose owner
process is gone (same host, dead pid; records without a pid count as stale)
before counting free `max_loras` slots. Adapters of live runs are untouched.
A resume re-runs the steps after its last checkpoint; sampler names carry a
per-attempt suffix (`<run>-<learner>-s<step>-<attempt>`) so re-run snapshots
never reuse a served name. Learner state is written under `<out>/states/`.
