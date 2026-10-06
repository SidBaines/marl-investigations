"""Pod preflight for experiment 2 on Qwen3.6-35B-A3B: about 10 minutes, numbers only, no task text.

Run ON THE POD from the checkout root after `MODEL=a3b ./run.sh serve` and before training:
    S=experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule
    uv run --no-sync python $S/a3b_preflight.py [$S/out/serve_a3b/server.json]
One learner copy loads on GPU 0 beside the awake server (training puts one on each GPU). In order:
1. GPU health (clocks, temperature) and memory with vLLM awake and the learner resident.
2. Base agreement before any training: a fixed probe scored by both engines, and T=1 samples from
   vLLM rescored by the learner. mean(sample - train) is step 0's kl_sample_train; and IS ratio.
3. vLLM asleep: train steps on 24,576- and 32,768-token datums (time, peak memory).
4. Hot-load: export the adapter, load it into vLLM, compare its effect in both engines (as the
   backend's own check), sample through it and rescore as in 2, unload it, and generate once more.
Exit code 1 if any check fails; WARN lines need a look but do not block.
"""

import asyncio
import random
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from math import exp

import httpx
import torch

from marli.model import load_model
from marli.policy.base import SamplingSpec
from marli.policy.vllm import VLLMPolicy
from marli.render.base import Msg
from marli.render.registry import get_renderer
from marli.serve.vllm import Server
from marli.train.backends.local.backend import _PROBE_TEXT
from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool
from marli.train.types import LearnerSpec, TrainDatum

SERVER = "experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/out/serve_a3b/server.json"
ADAPTER = "a3b-preflight"
CODE = "".join(
    f"def f{i}(xs):\n    return sorted(x * {i} for x in xs if x % {i + 2})\n\n" for i in range(40)
)
PROMPTS = [
    "Write a Python function that merges two sorted lists, with tests, then explain it briefly.",
    "Explain what a git rebase does and when you would avoid it. Use a short example.",
]
failures: list[str] = []


def check(ok: bool, label: str, *, warn: bool = False) -> None:
    print(("PASS " if ok else "WARN " if warn else "FAIL ") + label, flush=True)
    if not ok and not warn:
        failures.append(label)


def smi(fields: str = "memory.used") -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()


def used_gib() -> list[float]:
    return [round(int(x) / 1024, 1) for x in smi()]


async def prompt_logprobs(client: httpx.AsyncClient, root: str, model: str, ids: list[int]):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "echo": True, "prompt_logprobs": 0}
    response = await client.post(root + "/v1/completions", json=body)
    response.raise_for_status()
    entries = response.json()["choices"][0]["prompt_logprobs"]
    return [float(e[str(t)]["logprob"]) for t, e in zip(ids[1:], entries[1:], strict=True)]


async def sampled_agreement(learner, renderer, server, model: str, label: str) -> None:
    """vLLM samples at T=1; the learner rescores the same ids (step 0's on-policy check)."""
    policy = VLLMPolicy(
        "preflight", server.base_url, model, renderer_name="qwen3_5", trainable=True
    )
    spec = SamplingSpec(max_tokens=2048, stop_token_ids=tuple(renderer.stop_token_ids))
    prompts = [renderer.initial(None, (), [Msg("user", text)]) for text in PROMPTS]
    jobs = [policy.sample(p, spec, seed=s) for p in prompts for s in range(4)]
    samples = await asyncio.gather(*jobs)
    await policy.aclose()
    diffs = []
    for prompt, sample in zip([p for p in prompts for _ in range(4)], samples, strict=True):
        adapted, _ = await learner.probe_logprobs([*prompt, *sample.completion_ids])
        train = adapted[len(prompt) - 1 :]
        diffs += [s - t for s, t in zip(sample.logprobs, train, strict=True)]
    kl = sum(diffs) / len(diffs)
    ratio = sum(exp(-d) for d in diffs) / len(diffs)
    big = sum(abs(d) > 1 for d in diffs) / len(diffs)
    print(
        f"{label}: {len(diffs)} sampled tokens, mean(sample - train) {kl:.2e}, IS ratio mean "
        f"{ratio:.4f}, max |diff| {max(map(abs, diffs)):.2f}, share |diff| > 1: {big:.4f}"
    )
    check(abs(kl) <= 1e-3, f"{label}: |kl_sample_train| <= 1e-3", warn=abs(kl) <= 5e-3)
    check(abs(ratio - 1) <= 0.01, f"{label}: IS ratio mean within 1 +- 0.01")


def long_datum(n: int, version: int) -> TrainDatum:
    rng = random.Random(n)
    mask = tuple(0.0 if i < 4000 else 1.0 for i in range(n))
    return TrainDatum(
        learner="preflight",
        episode_id="e",
        agent_id="a",
        role="contributor",
        segment_id="s",
        session_idx=0,
        policy_version=version,
        tokens=tuple(rng.randrange(1000, 150000) for _ in range(n + 1)),
        logprobs=tuple(-1.0 * m for m in mask),
        mask=mask,
        advantages=mask,
    )


async def main() -> None:
    server = Server.load(sys.argv[1] if len(sys.argv) > 1 else SERVER)
    model = load_model(server.models[0])
    assert model.name == "qwen3_6_35b_a3b", model.name
    renderer = get_renderer(model.renderer, hf_id=model.hf_id)
    root = server.base_url.removesuffix("/v1")
    print(
        "GPU temperature, SM clock, throttle reasons:",
        smi("temperature.gpu,clocks.sm,clocks_event_reasons.active"),
        flush=True,
    )
    t = time.time()
    pool = LocalLearnerPool(model, device="cuda:0")
    # A throwaway adapter; lr 1e-3 so the hot-load check below compares a visible effect.
    spec = LearnerSpec(base_model=model.name, backend="local", rank=32, learning_rate=1e-3)
    learner = LocalLearner("preflight", spec, pool)
    used = used_gib()
    print(f"learner loaded in {time.time() - t:.0f} s; GPU GiB (vLLM awake + learner): {used}")
    check(max(used) <= 132, "co-resident memory <= 132 GiB per GPU")

    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        try:
            probe = renderer.encode_text(_PROBE_TEXT + CODE)
            served = await prompt_logprobs(client, root, model.name, probe)
            _, base = await learner.probe_logprobs(probe)
            drift = sum(abs(s - b) for s, b in zip(served, base, strict=True)) / len(base)
            print(f"base probe: {len(base)} tokens, mean |vLLM - learner| {drift:.4f} nats")
            check(drift <= 0.05, "base probe drift <= 0.05 nats (27B: ~0.03)")
            await sampled_agreement(learner, renderer, server, model.name, "base T=1 samples")

            await client.post(root + "/sleep", params={"level": "1", "mode": "wait"})
            print("vLLM asleep; GPU GiB:", used_gib(), flush=True)
            for n in (24576 - 1, 32768 - 1):
                torch.cuda.reset_peak_memory_stats(0)
                t = time.time()
                result = await learner.train_step([long_datum(n, learner.version)])
                dt = time.time() - t
                peak = torch.cuda.max_memory_allocated(0) / 2**30
                print(
                    f"train_step {n + 1} tokens: {dt:.1f} s = {n / dt:.0f} tok/s, torch peak "
                    f"{peak:.1f} GiB, GPU GiB {used_gib()}, grad_norm {result.grad_norm:.1f}"
                )
                check(peak <= 125, f"{n + 1}-token step peak <= 125 GiB")
            t = time.time()
            await learner.train_step([long_datum(24575, learner.version)])
            rate = 24575 / (time.time() - t)
            # Experiment 2's 27B run trained ~600k datum tokens per step on 2 GPUs (189 s).
            print(
                f"warm 24.6k step: {rate:.0f} tok/s per GPU -> ~{6e5 / (2 * rate):.0f} s per step"
            )
            print(
                "GPU temperature, SM clock, throttle reasons (after load):",
                smi("temperature.gpu,clocks.sm,clocks_event_reasons.active"),
            )
            torch.cuda.empty_cache()
            await client.post(root + "/wake_up")
            print("vLLM awake; GPU GiB:", used_gib(), flush=True)

            with tempfile.TemporaryDirectory(dir="/workspace") as directory:
                path = await learner.save_adapter(directory + "/adapter")
                body = {"lora_name": ADAPTER, "lora_path": path}
                response = await client.post(root + "/v1/load_lora_adapter", json=body)
                check(
                    response.is_success, f"vLLM loads the exported adapter ({response.status_code})"
                )
                if not response.is_success:
                    raise SystemExit(f"NO-GO: adapter load failed: {response.text[:500]}")
                served_a = await prompt_logprobs(client, root, ADAPTER, probe)
                adapted, base = await learner.probe_logprobs(probe)
                n = len(base)
                effect = sum(abs(a - b) for a, b in zip(adapted, base, strict=True)) / n
                served_effect = sum(abs(a - b) for a, b in zip(served_a, served, strict=True)) / n
                drift = (
                    sum(
                        abs((s - sb) - (a - b))
                        for s, sb, a, b in zip(served_a, served, adapted, base, strict=True)
                    )
                    / n
                )
                print(
                    f"adapter effect: learner {effect:.4f}, vLLM {served_effect:.4f}, "
                    f"drift {drift:.4f} nats/token"
                )
                check(effect > 1e-3, "the trained adapter changes the learner's logprobs")
                check(drift <= 0.05, "adapter effect drift <= 0.05 (the backend's hot-load check)")
                check(0.9 <= served_effect / max(effect, 1e-9) <= 1.1, "vLLM applies 90-110% of it")
                await sampled_agreement(learner, renderer, server, ADAPTER, "adapter T=1 samples")
            body = {"model": model.name, "prompt": probe[:16], "max_tokens": 32}
            response = await client.post(root + "/v1/completions", json=body)
            check(response.is_success, "vLLM generates after sleep/wake with the learner resident")
        finally:
            # Never leave the server asleep or holding the probe adapter: training starts next.
            for path, payload in (("/wake_up", None), ("/v1/unload_lora_adapter", ADAPTER)):
                with suppress(httpx.HTTPError):
                    json = None if payload is None else {"lora_name": payload}
                    await client.post(root + path, json=json)
    pool.close()
    print("GO" if not failures else f"NO-GO: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
