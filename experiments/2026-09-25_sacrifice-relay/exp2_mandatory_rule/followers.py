"""Experiment 2.1 readouts for contributors 2-4 (slots 1-3), by block of steps.

- Control (started knowing the rule): reached CI, followed the rule, scored 1, reviewed anyway.
- Test (did not know the rule): chose review among those who ran CI, split by what earlier
  contributors did (one ran CI in either mode / none ran CI).
Also the opener's own behaviour (slot 0), which should stay flat if it is frozen.

Usage: python3 followers.py <train rl run dir> [--every 10]
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pct(k: float, n: int) -> str:
    return f"{k / n:4.0%} ({int(k)}/{n})" if n else "   - (0/0)"


def block(episodes: list[dict]) -> dict[str, list[float]]:
    out = {
        k: [0, 0]
        for k in (
            "knew_ci", "knew_rule_met", "knew_scored", "knew_reviewed",
            "test_review_after_ci", "test_review_no_ci",
            "opener_ci", "opener_review", "opener_rule_passed",
        )
    }
    for e in episodes:
        if not e.get("ok", True):
            continue
        g = [e["grades"][f"contrib{k}"] for k in range(4)]
        out["opener_ci"][0] += g[0]["ran_ci"]
        out["opener_ci"][1] += 1
        if g[0]["ran_ci"]:
            out["opener_review"][0] += g[0]["probed"]
            out["opener_review"][1] += 1
        if g[0]["probed"]:
            out["opener_rule_passed"][0] += g[1]["rule_known_at_start"]
            out["opener_rule_passed"][1] += 1
        for k in range(1, 4):
            c = g[k]
            if c["rule_known_at_start"]:
                for key, value in (("knew_ci", c["ran_ci"]), ("knew_rule_met", c["rule_met"]),
                                   ("knew_scored", c["score"] > 0), ("knew_reviewed", c["probed"])):
                    out[key][0] += value
                    out[key][1] += 1
            elif c["ran_ci"]:
                earlier = any(g[j]["ran_ci"] for j in range(k))
                key = "test_review_after_ci" if earlier else "test_review_no_ci"
                out[key][0] += c["probed"]
                out[key][1] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()
    steps = {}
    for path in sorted((args.run_dir / "rollouts").glob("step_*/episodes.jsonl")):
        rows = read_jsonl(path)
        if rows:
            steps[int(path.parent.name.removeprefix("step_"))] = rows
    order = sorted(steps)
    labels = {
        "opener_ci": "opener ran CI",
        "opener_review": "opener reviewed (of CI runs)",
        "opener_rule_passed": "opener review -> contributor 2 knew",
        "knew_ci": "CONTROL knew rule: reached CI",
        "knew_rule_met": "CONTROL knew rule: followed it",
        "knew_scored": "CONTROL knew rule: scored 1",
        "knew_reviewed": "CONTROL knew rule: reviewed anyway",
        "test_review_after_ci": "TEST no rule: reviewed, earlier CI",
        "test_review_no_ci": "TEST no rule: reviewed, no earlier CI",
    }
    for i in range(0, len(order), args.every):
        chunk = order[i : i + args.every]
        b = block([e for s in chunk for e in steps[s]])
        print(f"steps {chunk[0]}-{chunk[-1]}")
        for key, label in labels.items():
            print(f"  {label:40s} {pct(*b[key])}")


if __name__ == "__main__":
    main()
