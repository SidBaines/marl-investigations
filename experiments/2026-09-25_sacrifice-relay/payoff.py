"""Did sacrificing pay off for the team in the overnight trial, and why (not)?

Compares playthroughs of the same repo in the same training step (one group = one repo x
G = 4 playthroughs), so problem difficulty cancels. Readouts:
  1. team score with vs without a sacrifice, overall and by the sacrificer's position;
  2. position-by-position score differences (the cost to the sacrificer, the gain to followers);
  3. the hand-off chain: was the rule passed on, and what did informed followers do with it;
  4. what training actually rewarded: the advantage each choice received under the trial's
     credit settings (team reward, leave-one-out group baseline, no std normalisation), i.e.
     A = team score - mean team score of the other playthroughs in the group;
  5. a back-of-envelope payoff if the rule were always passed on;
  6. what prompts a review (an earlier contributor's note about the extended checks), by step.

Usage: python3 payoff.py <train rl run dir> [first_step last_step]
"""

import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

N = 4
BOOT = 4000
rng = random.Random(0)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def contributor(k: int) -> str:
    return f"contrib{k}"


def load(run: Path, steps: range) -> list[dict]:
    out = []
    for step in steps:
        path = run / "rollouts" / f"step_{step:05d}" / "episodes.jsonl"
        for e in read_jsonl(path) if path.exists() else []:
            if not e.get("ok", True):
                continue
            g = e["grades"]
            probes = [k for k in range(N) if g[contributor(k)]["probed"]]
            out.append(
                {
                    "step": step,
                    "group": e["group_id"],
                    "team": g["_system"]["score"],
                    "scores": [g[contributor(k)]["score"] for k in range(N)],
                    "grades": [g[contributor(k)] for k in range(N)],
                    "first": probes[0] if probes else None,
                    "episode": e,
                }
            )
    return out


def mean_ci(values: list[float]) -> str:
    if not values:
        return "n = 0"
    m = st.fmean(values)
    if len(values) < 2:
        return f"{m:+.3f} (n = 1)"
    boots = sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(BOOT))
    lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {len(values)})"


def rate(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n:.0%}" if n else "0/0"


def groups_of(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    return groups


def team_comparison(groups: dict[str, list[dict]]) -> None:
    print("1. TEAM SCORE: playthroughs with a sacrifice minus same-group playthroughs without")
    print("   (one value per group; mean [95% bootstrap interval])")
    for label, keep in [("any position", lambda f: f is not None)] + [
        (f"first sacrifice by contributor {k + 1}", lambda f, k=k: f == k) for k in range(N)
    ]:
        diffs = []
        for rows in groups.values():
            sac = [r["team"] for r in rows if keep(r["first"])]
            none = [r["team"] for r in rows if r["first"] is None]
            if sac and none:
                diffs.append(st.fmean(sac) - st.fmean(none))
        print(f"   {label:38s} {mean_ci(diffs)}")
    all_rows = [r for rows in groups.values() for r in rows]
    sac = [r["team"] for r in all_rows if r["first"] is not None]
    none = [r["team"] for r in all_rows if r["first"] is None]
    print(
        f"   (unmatched: mean team score {st.fmean(sac):.3f} with a sacrifice, n = {len(sac)}; "
        f"{st.fmean(none):.3f} without, n = {len(none)})"
    )


def by_position(groups: dict[str, list[dict]]) -> None:
    print("\n2. WHERE IT COMES FROM: each position's score, sacrifice playthrough minus the")
    print("   same group's no-sacrifice playthroughs (one value per sacrifice playthrough)")
    header = "".join(f"{'contributor ' + str(j + 1):>30s}" for j in range(N))
    print(f"   {'first sacrifice by':22s}{header}")
    for k in range(N):
        cells = [[] for _ in range(N)]
        for rows in groups.values():
            none = [r for r in rows if r["first"] is None]
            if not none:
                continue
            for r in rows:
                if r["first"] == k:
                    for j in range(N):
                        cells[j].append(r["scores"][j] - st.fmean(x["scores"][j] for x in none))
        line = "".join(f"{mean_ci(c) if c else '-':>30s}" for c in cells)
        print(f"   contributor {k + 1:<10d}{line}")
    print("   Positions before the sacrificer acted before it, so their difference shows how")
    print("   sacrifice playthroughs differed beforehand (selection), not an effect of it.")


def touched_notes_after(episode: dict, agent: str) -> tuple[bool, bool]:
    """(wrote to NOTES.md after its CI review, hit its token budget)."""
    reviewed, wrote = False, False
    for call in sorted(episode["calls"], key=lambda c: c["seq"]):
        if call["agent_id"] != agent:
            continue
        for tool in call["tool_calls"]:
            if tool.get("name") == "ci_review":
                reviewed = True
            elif reviewed and tool.get("name") == "bash":
                command = str((tool.get("arguments") or {}).get("command", ""))
                if "NOTES.md" in command and any(op in command for op in (">", "tee", "write")):
                    wrote = True
    budget = bool(episode["limits_hit"].get(agent))
    return wrote, budget


def follower_stats(followers: list[dict]) -> str:
    n = len(followers)
    if not n:
        return "n = 0"
    ci = sum(g["ran_ci"] for g in followers)
    probed = sum(g["probed"] for g in followers)
    sub = [g for g in followers if g["submitted"]]
    passed = sum(g["base_pass"] for g in sub)
    three = sum(g["score"] == 3 for g in followers)
    return (
        f"n = {n}: reached CI {ci / n:.0%}, reviewed again {probed / n:.0%}, base tests passed "
        f"{rate(int(passed), len(sub))} of submits, scored 3 {three / n:.0%}, "
        f"mean score {st.fmean(g['score'] for g in followers):.2f}"
    )


def hand_off(rows: list[dict]) -> None:
    print("\n3. HAND-OFF: sacrifices by contributors 1-3 (who could help someone)")
    outcome = defaultdict(int)
    informed, uninformed, baseline = [], [], defaultdict(list)
    for r in rows:
        k = r["first"]
        for j in range(N):
            if r["first"] is None:
                baseline[j].append(r["grades"][j])
        if k is None or k == N - 1:
            continue
        passed = bool(r["grades"][k + 1]["rule_known_at_start"])
        wrote, budget = touched_notes_after(r["episode"], contributor(k))
        if passed:
            outcome["rule reached the next contributor"] += 1
        elif wrote:
            outcome["not passed on: wrote to NOTES.md, but without the rule"] += 1
        elif budget:
            outcome["not passed on: ran out of tokens without writing notes"] += 1
        else:
            outcome["not passed on: never wrote to NOTES.md"] += 1
        for j in range(k + 1, N):
            g = r["grades"][j]
            (informed if g["rule_known_at_start"] else uninformed).append(g)
    total = sum(outcome.values())
    for key, value in sorted(outcome.items(), key=lambda kv: -kv[1]):
        print(f"   {key:58s} {rate(value, total)}")
    print("   Followers (later contributors) in those playthroughs:")
    print(f"     started knowing the rule:  {follower_stats(informed)}")
    print(f"     did not know it:           {follower_stats(uninformed)}")
    flat = [g for j in range(1, N) for g in baseline[j]]
    print(f"   Contributors 2-4 in no-sacrifice playthroughs: {follower_stats(flat)}")


def advantages(groups: dict[str, list[dict]]) -> dict[str, float]:
    adv = {}
    for rows in groups.values():
        teams = [r["team"] for r in rows]
        if len(rows) < 2 or len(set(teams)) == 1:  # zero-variance groups are dropped
            for r in rows:
                adv[r["episode"]["episode_id"]] = 0.0
            continue
        for i, r in enumerate(rows):
            others = teams[:i] + teams[i + 1 :]
            adv[r["episode"]["episode_id"]] = r["team"] - st.fmean(others)
    return adv


def check_against_log(run: Path, groups: dict[str, list[dict]], adv: dict[str, float]) -> None:
    """Training logs mean |advantage| over emitted credits; zero-variance groups emit none."""
    logged = {
        r["step"]: r["credit"]["advantage_abs_mean"]["contributor"]
        for r in read_jsonl(run / "metrics.jsonl")
    }
    by_step: dict[int, list[float]] = defaultdict(list)
    for rows in groups.values():
        if len({r["team"] for r in rows}) > 1:
            by_step[rows[0]["step"]] += [abs(adv[r["episode"]["episode_id"]]) for r in rows]
    worst = max(abs(st.fmean(v) - logged[s]) for s, v in by_step.items())
    print(f"   (check: recomputed mean |advantage| matches the training log to within {worst:.1e})")


def credit(groups: dict[str, list[dict]], run: Path) -> None:
    rows = [r for rs in groups.values() for r in rs]
    adv = advantages(groups)
    print("\n4. WHAT TRAINING REWARDED: advantage given to each choice, among contributors who")
    print("   did not know the rule at the start and ran CI (positive = made more likely)")
    check_against_log(run, groups, adv)
    print(f"   {'':16s}{'reviewed (sacrificed)':>34s}{'submitted':>34s}")
    for label, ks in [("all positions", range(N))] + [
        (f"contributor {k + 1}", [k]) for k in range(N)
    ]:
        review, submit = [], []
        for r in rows:
            for k in ks:
                g = r["grades"][k]
                if g["ran_ci"] and not g["rule_known_at_start"]:
                    (review if g["probed"] else submit).append(adv[r["episode"]["episode_id"]])
        print(f"   {label:16s}{mean_ci(review):>34s}{mean_ci(submit):>34s}")
    print("   Training raises the review rate when 'reviewed' is above 'submitted'.")


def counterfactual(rows: list[dict]) -> None:
    print("\n5. IF THE RULE WERE ALWAYS PASSED ON (back-of-envelope from observed rates)")
    knew = [
        g for r in rows for g in r["grades"][1:] if g["rule_known_at_start"] and not g["probed"]
    ]
    none = [r for r in rows if r["first"] is None]
    plain = [[r["grades"][j]["score"] for r in none] for j in range(N)]
    informed = st.fmean(g["score"] for g in knew) if knew else float("nan")
    print(f"   Informed followers' mean score: {informed:.2f} (n = {len(knew)}).")
    print(
        "   No-sacrifice mean score by position: " + ", ".join(f"{st.fmean(p):.2f}" for p in plain)
    )
    for k in range(N):
        cost = st.fmean(plain[k])
        gain = sum(informed - st.fmean(plain[j]) for j in range(k + 1, N))
        print(
            f"   sacrifice by contributor {k + 1}: own cost -{cost:.2f}, followers +{gain:.2f}, "
            f"team average {(gain - cost) / N:+.3f}"
        )


def wrote_about_checks(episode: dict) -> set[str]:
    """Contributors who wrote to NOTES.md with a bash command mentioning the extended checks."""
    out = set()
    for call in episode["calls"]:
        for tool in call["tool_calls"]:
            command = str((tool.get("arguments") or {}).get("command", ""))
            if (
                tool.get("name") == "bash"
                and "NOTES.md" in command
                and re.search("extended", command, re.I)
                and re.search(">|tee|write", command)
            ):
                out.add(call["agent_id"])
    return out


def triggers(rows: list[dict]) -> None:
    print("\n6. WHAT PROMPTS A REVIEW: contributors 2-4 with a real choice (no rule at start, ran")
    print("   CI), split by whether an earlier contributor submitted and wrote about the extended")
    print("   checks in NOTES.md")
    split = defaultdict(lambda: [0, 0])
    for r in rows:
        noted = wrote_about_checks(r["episode"])
        for k in range(1, N):
            g = r["grades"][k]
            if not g["ran_ci"] or g["rule_known_at_start"]:
                continue
            cue = any(r["grades"][j]["submitted"] and contributor(j) in noted for j in range(k))
            split[cue][0] += int(g["probed"])
            split[cue][1] += 1
    print(f"   after such a note: reviewed {rate(*split[True])}")
    print(f"   otherwise:         reviewed {rate(*split[False])}")
    print("   By step block: how often submitters wrote such a note; how often a later")
    print(
        "   contributor reviewed after one; how often contributor 1 (who never sees one) reviewed"
    )
    for lo in range(0, 30, 10):
        block = [r for r in rows if lo <= r["step"] < lo + 10]
        wrote, responded, first = [0, 0], [0, 0], [0, 0]
        for r in block:
            noted = wrote_about_checks(r["episode"])
            for k in range(N - 1):
                g = r["grades"][k]
                if g["submitted"] and not g["rule_known_at_start"]:
                    wrote[0] += contributor(k) in noted
                    wrote[1] += 1
            for k in range(1, N):
                g = r["grades"][k]
                if not g["ran_ci"] or g["rule_known_at_start"]:
                    continue
                if any(r["grades"][j]["submitted"] and contributor(j) in noted for j in range(k)):
                    responded[0] += int(g["probed"])
                    responded[1] += 1
            if r["grades"][0]["ran_ci"]:
                first[0] += int(r["grades"][0]["probed"])
                first[1] += 1
        print(
            f"   steps {lo}-{lo + 9}: wrote a note {rate(*wrote)}; reviewed after one "
            f"{rate(*responded)}; contributor 1 reviewed {rate(*first)}"
        )


def main() -> None:
    run = Path(sys.argv[1])
    first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 29)
    rows = load(run, range(first, last + 1))
    groups = groups_of(rows)
    print(
        f"steps {first}-{last}: {len(rows)} playthroughs in {len(groups)} groups, "
        f"{sum(r['first'] is not None for r in rows)} with a sacrifice\n"
    )
    team_comparison(groups)
    by_position(groups)
    hand_off(rows)
    credit(groups, run)
    counterfactual(rows)
    triggers(rows)


if __name__ == "__main__":
    main()
