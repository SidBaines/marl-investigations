"""Speed facts from saved rollouts: wall clock, tokens, calls and per-call decode rates.

  python3 episodes.py <run_dir>    # eval rollout dir, or train rl dir (one row per step)

Stdlib only. Episode wall = first call start to last call end; per-call rate =
completion tokens / call latency (includes queueing and prefill, so it is a floor on
decode speed), over calls with at least 256 completion tokens.
"""

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def summarise(episodes: list[dict]) -> dict:
    starts, ends, walls, gen, calls, rates, limits = [], [], [], [], [], [], Counter()
    prompt = cached = 0
    metrics: dict[str, list[float]] = {}
    for ep in episodes:
        cs = ep["calls"]
        if not cs:
            continue
        s = [c["timing"]["started_at"] for c in cs]
        e = [c["timing"]["started_at"] + c["timing"]["latency_s"] for c in cs]
        starts.append(min(s))
        ends.append(max(e))
        walls.append(max(e) - min(s))
        gen.append(sum(c["usage"]["completion_tokens"] for c in cs))
        calls.append(len(cs))
        for c in cs:
            prompt += c["usage"]["prompt_tokens"]
            cached += c["usage"].get("cached_prompt_tokens") or 0
            if c["usage"]["completion_tokens"] >= 256 and c["timing"]["latency_s"] > 0:
                rates.append(c["usage"]["completion_tokens"] / c["timing"]["latency_s"])
        limits.update(k for k, v in (ep.get("limits_hit") or {}).items() if v)
        for key, value in (ep.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(value)
    span = max(ends) - min(starts) if starts else float("nan")
    return {
        "episodes": len(episodes),
        "ok": sum(bool(ep.get("ok", True)) for ep in episodes),
        "span_s": round(span, 1),
        "gen_tokens": sum(gen),
        "gen_tok_s_over_span": round(sum(gen) / span, 1) if span else float("nan"),
        "gen_per_episode_mean": round(st.fmean(gen)) if gen else 0,
        "calls_per_episode_mean": round(st.fmean(calls), 1) if calls else 0,
        "episode_wall_s": {
            "mean": round(st.fmean(walls), 1) if walls else 0,
            "p50": round(pct(walls, 0.5), 1),
            "p90": round(pct(walls, 0.9), 1),
            "max": round(max(walls), 1) if walls else 0,
        },
        "call_decode_tok_s": {"p10": round(pct(rates, 0.1), 1), "p50": round(pct(rates, 0.5), 1)},
        "prompt_tokens_per_call": round(prompt / max(sum(calls), 1)),
        "server_cached_prompt_frac": round(cached / max(prompt, 1), 3),
        "limits_hit": dict(limits),
        "metrics_mean": {k: round(st.fmean(v), 1) for k, v in sorted(metrics.items())},
    }


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    run = Path(sys.argv[1])
    steps = sorted((run / "rollouts").glob("step_*/episodes.jsonl"))
    if steps:
        for path in steps:
            print(json.dumps({"step": path.parent.name, **summarise(load(path))}))
    else:
        print(json.dumps(summarise(load(run / "episodes.jsonl")), indent=1))


if __name__ == "__main__":
    main()
