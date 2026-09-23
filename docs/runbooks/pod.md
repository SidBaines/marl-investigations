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

```bash
cd /workspace/marl-investigations
uv sync --extra dev --extra render --extra eval --extra hub
uv pip install --python .venv/bin/python -r requirements/pod-train.txt
python3 -m venv /opt/vllm
/opt/vllm/bin/python -m pip install -r requirements/pod-vllm.txt
```

`pod-train.txt` comes from M4-1. An activated repo venv can equivalently use
`python -m pip install -r requirements/pod-train.txt` if pip is installed there.
Keep weights/cache, exported adapters, checkpoints and logs on the pod's
workspace disk. Run from the checkout root; commit scientific configuration
before training (dirty training trees require explicit `MARLI_ALLOW_DIRTY=1`).

## GPU placement

For two GPUs, give vLLM physical GPU 0 and the learner physical GPU 1:

```bash
uv run marli serve vllm model=qwen3_5_4b python=/opt/vllm/bin/python detach=true \
  cuda_visible_devices=0 learner_ranks='[32]' --out runs/serve/q4
```

Set `CUDA_VISIBLE_DEVICES=1` for the training process and use backend
`device=cuda` (its visible GPU is numbered 0). Alternatively leave both visible
and use `device=cuda:1`. For co-location on one **80 GB GPU**, models **≤9B**
can start with `gpu_memory_utilization=0.35` and both processes on GPU 0. This
is a starting allocation, not a memory guarantee: measure learner peak memory
at the intended context length before the pilot.

The minimal launch command is:

```bash
uv run marli serve vllm model=qwen3_5_4b python=/opt/vllm/bin/python detach=true --out runs/serve/q4
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
The vLLM 0.30.0 pin and flag table are CPU-tested, not live-verified here.
For Qwen3.5, M4-1 exports text-only `Qwen3_5ForCausalLM` PEFT paths; verify
that the selected vLLM architecture accepts those paths and the `lm_head`
adapter before spending on training. The local-family parity gate remains
required, including prompt ids, stop-token inclusion and raw logprobs.

## Connect training

The backend factory accepts these exact kwargs:

```python
backend = make_backend(
    "local", spend=spend,
    server_json="/workspace/marl-investigations/runs/serve/q4/server.json",
    adapters_dir="/workspace/marl-investigations/runs/train/q4/adapters",
    device="cuda",
)
```

Each learner shares one base model and publishes a unique name such as
`q4-x-s12`. The adapter directory is never overwritten or deleted when the
server unloads it. Both the adapter list and served model index are updated in
`server.json`; sampling refs have the form
`vllm:@/workspace/marl-investigations/runs/serve/q4/server.json#q4-x-s12`.
`close()` unloads this backend's adapters; keep their files to reload for eval.

**Integration required outside M4-2's file ownership:** `TrainRLConfig` does
not yet expose local backend kwargs, and `train/loop.py` still rejects local
learners. The orchestrator must add configuration for `server_json`,
`adapters_dir`, and `device`, pass those kwargs to `make_backend`, and remove
the local-backend rejection. Once that wiring exists, the training invocation
is the following, with those backend kwargs supplied in the study YAML:

```bash
CUDA_VISIBLE_DEVICES=1 uv run marli train rl experiments/<study>/train.yaml \
  learners.x.backend=local learners.x.base_model=qwen3_5_4b \
  max_usd=25 --out runs/train/q4
```

This command is a pending integration step, not a currently runnable CLI
recipe. Do not bypass the shared loop by building a second training runner.
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

Use a **private** Hugging Face repository and a scoped `HF_TOKEN` supplied via
the environment. Never put the token in YAML, manifests, shell history or git.
For example, with the HF CLI installed by the hub extra:

```bash
hf repo create "$HF_REPO" --repo-type model --private
hf upload "$HF_REPO" runs/train/q4 checkpoints/q4 --repo-type model
```

Use an existing private repo instead of creating it again on subsequent syncs.
Upload resumable **state** (optimizer plus adapter) and exported sampler
adapters, along with checkpoint manifests and metadata. A sampler alone cannot
resume training. Local absolute input references need rebasing when restoring
on a different pod; keep directory layout and provenance hashes intact.
Verify uploaded file inventory and hashes against local bytes before teardown;
test a state restore and record the private HF commit ID in the study notebook.
Do not upload gated benchmark prompts or transcripts as part of this checkpoint
sync.

1. Finish or stop training and save its final resumable checkpoint.
2. Sync state and sampler bytes off-pod; verify the private remote copy.
3. Save aggregate results, configurations, provenance and spend records.
4. Run `serve stop`, then a fresh `serve status`; verify `running=false`.
5. Stop/delete the pod through the approved pod lifecycle workflow only after
   verifying checkpoint persistence. Record final billed spend. Pods are not storage.
