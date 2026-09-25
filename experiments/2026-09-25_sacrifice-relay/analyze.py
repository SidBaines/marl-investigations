"""Per-step sacrifice metrics from a `train rl` (or `eval rollout`) run dir.

Reads <run>/rollouts/step_*/episodes.jsonl (train rl) or <run>/episodes.jsonl
(eval rollout). "Knew the rule" = the rule's ID was anywhere in the repo when the
contributor started (`rule_known_at_start`; the rule can be passed on in any file,
not only NOTES.md). Per step (or per pooled block of steps):

- first-mover probe rate: P(ci_review | ran CI, didn't know the rule) — the primary metric,
  overall and by slot (the last slot can never help anyone)
- redundant probe rate:   P(ci_review | ran CI, knew the rule)
- note fidelity:          P(the next contributor starts knowing the rule | a non-last prober)
- follower bonus rate:    P(score == bonus | ci_submit, knew the rule)
- pre-edited rate:        P(a later contributor's solution.py was already changed when it started),
                          i.e. an earlier contributor worked on someone else's task
- CI rate, base pass rate, mean own score, mean team score (the `_system` grade)

Usage: python analyze.py <run_dir> [--every K] [--bonus 3]
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def slots(episode: dict) -> list[str]:
    return [a["agent_id"] for a in sorted(episode["agents"], key=lambda a: a["seat_key"])]


def knew(grade: dict) -> bool:
    return bool(grade.get("rule_known_at_start", grade.get("notes_had_rule", 0)))


def ran_ci(grade: dict) -> bool:
    return bool(grade.get("ran_ci", grade["probed"] + grade["submitted"] > 0))


def rate(pair: list[int]) -> float:
    return pair[0] / pair[1] if pair[1] else float("nan")


def step_metrics(episodes: list[dict], bonus: float) -> dict[str, float | int]:
    fm, red, fid, got_bonus, ci = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    pre = [0, 0]
    fm_slot: dict[int, list[int]] = defaultdict(lambda: [0, 0])
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
            if k > 0:
                pre[0] += int(bool(g.get("solution_edited_at_start", 0)))
                pre[1] += 1
            ci[1] += 1
            if not ran_ci(g):
                continue
            ci[0] += 1
            if knew(g):
                red[0] += int(g["probed"])
                red[1] += 1
            else:
                fm[0] += int(g["probed"])
                fm[1] += 1
                fm_slot[k][0] += int(g["probed"])
                fm_slot[k][1] += 1
            if g["submitted"]:
                base.append(g["base_pass"])
                if knew(g):
                    got_bonus[0] += int(g["score"] == bonus)
                    got_bonus[1] += 1
            if g["probed"] and k + 1 < len(ids) and ids[k + 1] in grades:
                fid[0] += int(knew(grades[ids[k + 1]]))
                fid[1] += 1
    row: dict[str, float | int] = {
        "first_mover_probe": rate(fm),
        "n_first_mover": fm[1],
        "redundant_probe": rate(red),
        "note_fidelity": rate(fid),
        "follower_bonus": rate(got_bonus),
        "pre_edited": rate(pre),
        "ci_rate": rate(ci),
        "base_pass": st.fmean(base) if base else float("nan"),
        "own_score": st.fmean(own) if own else float("nan"),
        "team_score": st.fmean(team) if team else float("nan"),
        "episodes": len(team),
    }
    for k in sorted(fm_slot):
        row[f"fm_probe_slot{k}"] = rate(fm_slot[k])
    return row


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--every", type=int, default=1, help="pool K consecutive steps per row")
    parser.add_argument("--bonus", type=float, default=3.0, help="score for base pass + rule")
    args = parser.parse_args()
    steps = sorted((args.run_dir / "rollouts").glob("step_*/episodes.jsonl"))
    if not steps and (args.run_dir / "episodes.jsonl").exists():
        steps = [args.run_dir / "episodes.jsonl"]
    rows = []
    for i in range(0, len(steps), args.every):
        chunk = steps[i : i + args.every]
        label = chunk[0].parent.name.removeprefix("step_")
        if len(chunk) > 1:
            label += "-" + chunk[-1].parent.name.removeprefix("step_")
        rows.append((label, step_metrics([ep for p in chunk for ep in load(p)], args.bonus)))
    cols = sorted({c for _, row in rows for c in row}, key=lambda c: (c.startswith("fm_"), c))
    print("steps      " + " ".join(f"{c:>17}" for c in cols))
    for label, row in rows:
        print(f"{label:<10} " + " ".join(f"{row.get(c, float('nan')):>17.3f}" for c in cols))


if __name__ == "__main__":
    main()
