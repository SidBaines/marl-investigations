"""Sample vLLM /metrics and nvidia-smi while a benchmark runs, then summarise a window.

  python3 metrics.py watch <base_url> <out.jsonl> [--every S]   # until killed
  python3 metrics.py summary <out.jsonl> [--start T] [--end T]  # unix-time window

Stdlib only (runs with the pod's system python). Counters are summed over label sets
(e.g. the two engines of a data-parallel server); gauges are summed too.
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:request_success_total",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
HISTOGRAMS = ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds")
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")


def scrape(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=10) as response:
        text = response.read().decode()
    values: dict = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        if name in COUNTERS or name in GAUGES:
            values[name] = values.get(name, 0.0) + value
        for hist in HISTOGRAMS:
            if name in (f"{hist}_sum", f"{hist}_count"):
                values[name] = values.get(name, 0.0) + value
            elif name == f"{hist}_bucket":
                le = re.search(r'le="([^"]+)"', labels)
                if le:
                    key = f"{hist}_bucket_{le.group(1)}"
                    values[key] = values.get(key, 0.0) + value
    return values


def gpus() -> list[dict]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        index, memory, util = (part.strip() for part in line.split(","))
        rows.append({"gpu": int(index), "mem_mib": int(memory), "util": int(util)})
    return rows


def watch(base_url: str, path: Path, every: float) -> None:
    with path.open("a") as stream:
        while True:
            row = {"t": time.time(), "gpus": gpus()}
            try:
                row["vllm"] = scrape(base_url)
            except OSError as exc:
                row["error"] = str(exc)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            time.sleep(every)


def quantile(buckets: dict[float, float], q: float) -> float:
    total = max(buckets.values(), default=0.0)
    if total <= 0:
        return float("nan")
    for le in sorted(buckets):
        if buckets[le] >= q * total:
            return le
    return float("inf")


def summary(path: Path, start: float | None, end: float | None) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [
        r
        for r in rows
        if "vllm" in r and (start is None or r["t"] >= start) and (end is None or r["t"] <= end)
    ]
    if len(rows) < 2:
        return {"error": "fewer than two samples in window"}
    first, last = rows[0]["vllm"], rows[-1]["vllm"]
    span = rows[-1]["t"] - rows[0]["t"]

    def delta(name: str) -> float:
        return last.get(name, 0.0) - first.get(name, 0.0)

    out = {
        "window_s": round(span, 1),
        "gen_tok_s": round(delta("vllm:generation_tokens_total") / span, 1),
        "prompt_tok_s": round(delta("vllm:prompt_tokens_total") / span, 1),
        "prefix_hit_rate": round(
            delta("vllm:prefix_cache_hits_total")
            / max(delta("vllm:prefix_cache_queries_total"), 1),
            3,
        ),
        "preemptions": delta("vllm:num_preemptions_total"),
        "requests_done": delta("vllm:request_success_total"),
        "running_mean": round(
            sum(r["vllm"].get("vllm:num_requests_running", 0) for r in rows) / len(rows), 1
        ),
        "running_max": max(r["vllm"].get("vllm:num_requests_running", 0) for r in rows),
        "waiting_max": max(r["vllm"].get("vllm:num_requests_waiting", 0) for r in rows),
        "kv_usage_max": max(
            max(
                r["vllm"].get("vllm:kv_cache_usage_perc", 0),
                r["vllm"].get("vllm:gpu_cache_usage_perc", 0),
            )
            for r in rows
        ),
    }
    drafts = delta("vllm:spec_decode_num_draft_tokens_total")
    if drafts:
        out["spec_accept_rate"] = round(
            delta("vllm:spec_decode_num_accepted_tokens_total") / drafts, 3
        )
        out["spec_mean_accepted_per_draft"] = round(
            delta("vllm:spec_decode_num_accepted_tokens_total")
            / max(delta("vllm:spec_decode_num_drafts_total"), 1),
            2,
        )
    for hist in HISTOGRAMS:
        count = delta(f"{hist}_count")
        if count:
            out[f"{hist.split(':')[1]}_mean_ms"] = round(1000 * delta(f"{hist}_sum") / count, 1)
            buckets = {
                float(key.rsplit("_", 1)[1]): last[key] - first.get(key, 0.0)
                for key in last
                if key.startswith(f"{hist}_bucket_") and key.rsplit("_", 1)[1] != "+Inf"
            }
            out[f"{hist.split(':')[1]}_p50_ms_le"] = round(1000 * quantile(buckets, 0.5), 1)
            out[f"{hist.split(':')[1]}_p90_ms_le"] = round(1000 * quantile(buckets, 0.9), 1)
    by_gpu: dict[int, dict] = {}
    for row in rows:
        for gpu in row["gpus"]:
            item = by_gpu.setdefault(gpu["gpu"], {"mem_max_mib": 0, "util": []})
            item["mem_max_mib"] = max(item["mem_max_mib"], gpu["mem_mib"])
            item["util"].append(gpu["util"])
    out["gpus"] = {
        k: {
            "mem_max_gib": round(v["mem_max_mib"] / 1024, 1),
            "util_mean": round(sum(v["util"]) / len(v["util"])),
        }
        for k, v in by_gpu.items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("base_url")
    w.add_argument("out", type=Path)
    w.add_argument("--every", type=float, default=10.0)
    s = sub.add_parser("summary")
    s.add_argument("path", type=Path)
    s.add_argument("--start", type=float)
    s.add_argument("--end", type=float)
    args = parser.parse_args()
    if args.cmd == "watch":
        watch(args.base_url, args.out, args.every)
    else:
        print(json.dumps(summary(args.path, args.start, args.end), indent=1))


if __name__ == "__main__":
    main()
