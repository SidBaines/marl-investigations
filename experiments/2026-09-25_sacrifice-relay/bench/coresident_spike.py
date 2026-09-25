"""Spike: learner model resident beside awake TP2 vLLM; vLLM sleeps; a 23.6k train step fits."""

import asyncio
import json
import random
import subprocess
import time
import urllib.request

import torch

from marli.model import load_model
from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool
from marli.train.types import LearnerSpec, TrainDatum

BASE = "http://127.0.0.1:8000"


def post(path):
    urllib.request.urlopen(
        urllib.request.Request(BASE + path, data=b"", method="POST"), timeout=600
    ).read()


def smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return [round(int(x) / 1024, 1) for x in out]


async def main():
    t = time.time()
    pool = LocalLearnerPool(load_model("qwen3_8_27b"), device="cuda:0")
    learner = LocalLearner("spike", LearnerSpec(backend="local", rank=32, learning_rate=2e-5), pool)
    print(
        "learner loaded in",
        round(time.time() - t),
        "s; GPU GiB (vLLM awake + learner resident):",
        smi(),
        flush=True,
    )
    n = 23571
    rng = random.Random(0)
    tokens = tuple(rng.randrange(1000, 150000) for _ in range(n + 1))
    mask = tuple(0.0 if i < 4000 else 1.0 for i in range(n))
    datum = TrainDatum(
        learner="spike",
        episode_id="e",
        agent_id="a",
        role="contributor",
        segment_id="s",
        session_idx=0,
        policy_version=0,
        tokens=tokens,
        logprobs=tuple(-1.0 * m for m in mask),
        mask=mask,
        advantages=mask,
    )
    t = time.time()
    post("/sleep?level=1&mode=wait")
    print("vLLM sleep", round(time.time() - t, 1), "s; GPU GiB:", smi(), flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    t = time.time()
    result = await learner.train_step([datum])
    dt = time.time() - t
    print(
        "train_step on",
        n,
        "tokens:",
        round(dt, 1),
        "s =",
        round(n / dt),
        "tok/s; torch peak GiB",
        round(torch.cuda.max_memory_allocated(0) / 2**30, 1),
        "; GPU GiB now:",
        smi(),
        "grad_norm",
        round(result.grad_norm, 2),
        flush=True,
    )
    t = time.time()
    result = await learner.train_step(
        [datum.__class__(**{**datum.__dict__, "policy_version": learner.version})]
    )
    dt = time.time() - t
    print(
        "second train_step (kernels warm):", round(dt, 1), "s =", round(n / dt), "tok/s", flush=True
    )
    torch.cuda.empty_cache()
    t = time.time()
    post("/wake_up")
    print("vLLM wake", round(time.time() - t, 1), "s; GPU GiB:", smi(), flush=True)
    r = urllib.request.urlopen(
        urllib.request.Request(
            BASE + "/v1/completions",
            data=json.dumps(
                {"model": "qwen3_8_27b", "prompt": [151644, 872, 198], "max_tokens": 32}
            ).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=120,
    ).read()
    print(
        "vLLM generates after wake with learner resident:",
        json.loads(r)["usage"]["completion_tokens"],
        "tokens",
        flush=True,
    )


asyncio.run(main())
