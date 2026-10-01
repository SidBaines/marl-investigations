"""Go/stop numbers for experiment 2, from a gate (`eval rollout`) or training (`train rl`) run dir.

Prints, per block of steps (or once for the gate):
- team score, and the share of playthroughs where anyone scored;
- the share of groups (same repo, same step) whose playthroughs all tied, so training learns nothing
  from them (with team reward every contributor shares the playthrough's score);
- by position: reached CI, chose review (among contributors who started without the rule and ran
  CI), started knowing the rule, and scored;
- hand-off: after a review by contributors 1-3, did the next contributor start knowing the rule;
- redundant reviews: reviews by contributors who already knew the rule.
Then applies the pre-registered rules in README.md and prints GO / STOP / ABORT lines.

Usage: python3 check.py <run dir> [--every 5]
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

N = 4
NO_SIGNAL_MAX = 0.75  # gate and abort: at most this share of groups may tie
ANYONE_SCORED_MIN = 0.10  # gate: at least this share of playthroughs must score
KL_MAX = 5e-3  # abort: sampler/trainer mismatch; experiment 1 stayed below 7.5e-4
CI_MIN = 0.40  # abort: reached-CI rate on each of the last 3 steps; experiment 1 was 61-70%


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(run: Path) -> dict[int, list[dict]]:
    """Episodes by step; a gate is one block at step 0."""
    if (run / "episodes.jsonl").exists():
        return {0: read_jsonl(run / "episodes.jsonl")}
    steps = {}
    for path in sorted((run / "rollouts").glob("step_*/episodes.jsonl")):
        rows = read_jsonl(path)
        if rows:
            steps[int(path.parent.name.removeprefix("step_"))] = rows
    return steps


def pct(k: float, n: int) -> str:
    return f"{k / n:4.0%} ({int(k)}/{n})" if n else "   - (0/0)"


def block(episodes: list[dict]) -> dict:
    eps = [e for e in episodes if e.get("ok", True)]
    groups = defaultdict(list)
    for e in eps:
        groups[e.get("group_id") or e["task_id"]].append(e["grades"]["_system"]["score"])
    tied = [len(set(scores)) == 1 for scores in groups.values() if len(scores) > 1]
    ci, review, knew, scored = ([[0, 0] for _ in range(N)] for _ in range(4))
    hand, redundant = [0, 0], [0, 0]
    for e in eps:
        g = [e["grades"][f"contrib{k}"] for k in range(N)]
        for k in range(N):
            ci[k][0] += g[k]["ran_ci"]
            ci[k][1] += 1
            knew[k][0] += g[k]["rule_known_at_start"]
            knew[k][1] += 1
            scored[k][0] += g[k]["score"] > 0
            scored[k][1] += 1
            if g[k]["ran_ci"]:
                pair = redundant if g[k]["rule_known_at_start"] else review[k]
                pair[0] += g[k]["probed"]
                pair[1] += 1
            if g[k]["probed"] and k < N - 1:
                hand[0] += g[k + 1]["rule_known_at_start"]
                hand[1] += 1
    return {
        "n": len(eps),
        "team": st.fmean(e["grades"]["_system"]["score"] for e in eps) if eps else float("nan"),
        "anyone": sum(e["grades"]["_system"]["score"] > 0 for e in eps),
        "tied": sum(tied),
        "groups": len(tied),
        "ci": ci,
        "review": review,
        "knew": knew,
        "scored": scored,
        "hand": hand,
        "redundant": redundant,
    }


def show(label: str, b: dict) -> None:
    def row(name: str, pairs: list[list[float]]) -> str:
        total = [sum(p[0] for p in pairs), sum(p[1] for p in pairs)]
        cells = "  ".join(pct(*p) for p in pairs)
        return f"  {name:30s} all {pct(*total)} | by position {cells}"

    print(f"{label}: {b['n']} playthroughs, team score {b['team']:.3f}")
    print(f"  {'anyone scored':30s} {pct(b['anyone'], b['n'])}")
    print(f"  {'groups with no learning signal':30s} {pct(b['tied'], b['groups'])}")
    print(row("reached CI", b["ci"]))
    print(row("reviewed (no rule, ran CI)", b["review"]))
    print(row("started knowing the rule", b["knew"]))
    print(row("scored 1", b["scored"]))
    print(f"  {'rule reached the next one':30s} {pct(*b['hand'])} of reviews by contributors 1-3")
    print(f"  {'redundant reviews':30s} {pct(*b['redundant'])} of contributors who knew and ran CI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--every", type=int, default=5, help="steps pooled per row (training)")
    args = parser.parse_args()
    steps = load(args.run_dir)
    if not steps:
        raise SystemExit("no episodes yet")
    gate = (args.run_dir / "episodes.jsonl").exists()
    if gate:
        b = block(steps[0])
        show("gate", b)
        no_signal = b["tied"] / b["groups"] if b["groups"] else 1.0
        ok = no_signal <= NO_SIGNAL_MAX and b["anyone"] >= ANYONE_SCORED_MIN * b["n"]
        print(
            f"GATE: {'GO' if ok else 'STOP'} (no-signal share {no_signal:.0%} vs max "
            f"{NO_SIGNAL_MAX:.0%}; anyone scored {b['anyone'] / b['n']:.0%} vs min "
            f"{ANYONE_SCORED_MIN:.0%})"
        )
        return
    order = sorted(steps)
    for i in range(0, len(order), args.every):
        chunk = order[i : i + args.every]
        show(f"steps {chunk[0]}-{chunk[-1]}", block([e for s in chunk for e in steps[s]]))
    flags = []
    first = [e for s in order[:6] for e in steps[s]]
    if len(order) >= 6:
        b = block(first)
        if b["groups"] and b["tied"] / b["groups"] >= NO_SIGNAL_MAX:
            flags.append(f"steps 0-5: {b['tied']}/{b['groups']} groups had no learning signal")
    metrics = args.run_dir / "metrics.jsonl"
    rows = read_jsonl(metrics) if metrics.exists() else []
    for r in rows:
        for name, learner in (r.get("learners") or {}).items():
            kl = learner.get("kl_sample_train")
            if kl is not None and kl > KL_MAX:
                flags.append(f"step {r['step']} {name}: kl_sample_train {kl:.1e} > {KL_MAX:.0e}")
    last = [block(steps[s]) for s in order[-3:]]
    if len(last) == 3 and all(sum(p[0] for p in b["ci"]) < CI_MIN * b["n"] * N for b in last):
        flags.append(f"reached-CI rate below {CI_MIN:.0%} on each of the last 3 steps")
    print("ABORT: " + "; ".join(flags) if flags else "no abort rule triggered")


if __name__ == "__main__":
    main()
