"""Spike: vLLM TP2 + MTP + LoRA survives sleep(level 1)/wake with identical logprobs; time it."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
adapters = sorted(Path(sys.argv[1]).iterdir())
adapter = adapters[0]
name = "spike-" + adapter.name


def req(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        text = resp.read().decode()
        return json.loads(text) if text.strip().startswith(("{", "[")) else text


def mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return [round(int(x) / 1024, 1) for x in out]


def plp(model):
    ids = list(range(1000, 1400))
    r = req(
        "POST",
        "/v1/completions",
        {
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            "echo": True,
            "prompt_logprobs": 0,
            "temperature": 0,
        },
    )
    lps = r["choices"][0]["prompt_logprobs"][1:]
    return [
        list(d.values())[0]["logprob"]
        if isinstance(list(d.values())[0], dict)
        else list(d.values())[0]
        for d in lps
    ]


print(
    "load adapter",
    req("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(adapter)}),
)
before_a, before_b = plp(name), plp("qwen3_8_27b")
effect = sum(abs(a - b) for a, b in zip(before_a, before_b, strict=True)) / len(before_a)
print("adapter effect vs base (mean |dlogp|)", round(effect, 4), "mem GiB", mem())
t = time.time()
req("POST", "/sleep?level=1&mode=wait")
t_sleep = time.time() - t
print("sleep_s", round(t_sleep, 2), "is_sleeping", req("GET", "/is_sleeping"), "mem GiB", mem())
t = time.time()
req("POST", "/wake_up")
t_wake = time.time() - t
print("wake_s", round(t_wake, 2), "is_sleeping", req("GET", "/is_sleeping"), "mem GiB", mem())
models = [m["id"] for m in req("GET", "/v1/models")["data"]]
print("adapter still listed:", name in models)
after_a, after_b = plp(name), plp("qwen3_8_27b")
d_a = max(abs(a - b) for a, b in zip(before_a, after_a, strict=True))
d_b = max(abs(a - b) for a, b in zip(before_b, after_b, strict=True))
print("max |dlogp| after wake: adapter", round(d_a, 5), "base", round(d_b, 5))
t = time.time()
r = req(
    "POST",
    "/v1/completions",
    {
        "model": name,
        "prompt": [151644, 872, 198] + list(range(2000, 2050)),
        "max_tokens": 64,
        "temperature": 1.0,
    },
)
print(
    "generate after wake ok, s",
    round(time.time() - t, 2),
    "tokens",
    r["usage"]["completion_tokens"],
)
print("unload", req("POST", "/v1/unload_lora_adapter", {"lora_name": name}))
