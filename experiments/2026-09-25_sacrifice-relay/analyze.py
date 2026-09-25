"""Per-step sacrifice metrics from a `train rl` (or `eval rollout`) run dir.

Reads <run>/rollouts/step_*/episodes.jsonl (train rl) or <run>/episodes.jsonl
(eval rollout) and prints, per step:

- first-mover probe rate: P(ci_review | CI run, NOTES.md had no rule) — the primary metric
- redundant probe rate:   P(ci_review | CI run, NOTES.md already had the rule)
- note fidelity:          P(the next contributor's NOTES.md has the rule | a non-last prober)
- follower bonus rate:    P(rule met | ci_submit, NOTES.md had the rule)
- base pass rate, mean own score, mean team score (the `_system` grade)

Usage: python analyze.py <run_dir> [--every K]
"""

import argparse
import json
import statistics as st
from pathlib import Path


def slots(episode: dict) -> list[str]:
    return [a["agent_id"] for a in sorted(episode["agents"], key=lambda a: a["seat_key"])]


def step_metrics(episodes: list[dict]) -> dict[str, float | int]:
    fm = [0, 0]  # probes, CI runs with no rule in notes
    red = [0, 0]  # probes, CI runs with the rule in notes
    fid = [0, 0]  # next contributor saw the rule, non-last probers
    bonus = [0, 0]  # rule met, submits with the rule in notes
    base, own, team = [], [], []
    for ep in episodes:
        if not ep.get("ok", True):
            continue
        ids = slots(ep)
        grades = ep["grades"]
        team.append(grades["_system"]["score"])
        for k, agent in enumerate(ids):
            g = grades.get(agent)
            if g is None:
                continue
            own.append(g["score"])
            ran_ci = g["probed"] + g["submitted"] > 0
            if not ran_ci:
                continue
            bucket = red if g["notes_had_rule"] else fm
            bucket[0] += int(g["probed"])
            bucket[1] += 1
            if g["submitted"]:
                base.append(g["base_pass"])
                if g["notes_had_rule"]:
                    bonus[0] += int(g["rule_met"])
                    bonus[1] += 1
            if g["probed"] and k + 1 < len(ids):
                nxt = grades.get(ids[k + 1])
                if nxt is not None and nxt["probed"] + nxt["submitted"] > 0:
                    fid[0] += int(nxt["notes_had_rule"])
                    fid[1] += 1

    def rate(pair: list[int]) -> float:
        return pair[0] / pair[1] if pair[1] else float("nan")

    return {
        "episodes": len(team),
        "first_mover_probe": rate(fm),
        "n_first_mover": fm[1],
        "redundant_probe": rate(red),
        "n_redundant": red[1],
        "note_fidelity": rate(fid),
        "follower_bonus": rate(bonus),
        "base_pass": st.fmean(base) if base else float("nan"),
        "own_score": st.fmean(own) if own else float("nan"),
        "team_score": st.fmean(team) if team else float("nan"),
    }


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--every", type=int, default=1, help="pool K consecutive steps per row")
    args = parser.parse_args()
    steps = sorted((args.run_dir / "rollouts").glob("step_*/episodes.jsonl"))
    if not steps and (args.run_dir / "episodes.jsonl").exists():
        steps = [args.run_dir / "episodes.jsonl"]
    cols = (
        "first_mover_probe", "n_first_mover", "redundant_probe", "note_fidelity",
        "follower_bonus", "base_pass", "own_score", "team_score", "episodes",
    )
    print("steps      " + " ".join(f"{c:>17}" for c in cols))
    for i in range(0, len(steps), args.every):
        chunk = steps[i : i + args.every]
        m = step_metrics([ep for path in chunk for ep in load(path)])
        label = chunk[0].parent.name.removeprefix("step_")
        if len(chunk) > 1:
            label += "-" + chunk[-1].parent.name.removeprefix("step_")
        print(f"{label:<10} " + " ".join(f"{m[c]:>17.3f}" for c in cols))


if __name__ == "__main__":
    main()
