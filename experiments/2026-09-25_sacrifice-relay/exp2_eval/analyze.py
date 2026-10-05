"""Transfer-eval readouts, per condition and policy (numbers only; never prints problem text).

Reads `eval rollout` cells laid out as <root>/<condition>/<policy>/ (full run dirs with
episodes.jsonl, or the grades-only copies that `--extract` writes). For every cell:
- the training dashboard's measures by position, out of all contributors at that position:
  reached CI, chose review, chose submit, scored, started knowing the rule; plus the team score;
- check.py's conditional rates: contributor 1 reviewed (of its CI runs), reviewed without the
  rule (of CI runs), redundant reviews (knew the rule, reviewed anyway), the hand-off after a
  review, and, for followers (contributors 2+): followed the rule when it was in the notes file,
  followed it when they started knowing it, and scored when they started knowing it;
- habit checks: calls to tools that do not exist in this cell (e.g. ci_review after the rename),
  bash commands that mention NOTES.md when the notes file has moved, and submissions that contain
  the rule's ID but in the wrong form.
Rates show k/n and a 95% Wilson interval (contributors treated as independent); the team score
shows a 95% interval from a bootstrap over repos.

Comparisons (95% intervals from a bootstrap over repos, 2,000 resamples by default):
- paired lift: each policy minus the untrained model of the same family (27b_base or a3b_base) in
  the same condition. Same repos, playthrough indices and seed, so the same house rule and folder
  names; repos are resampled jointly.
- transfer change: each condition minus its reference condition for the same policy (train_repos
  -> heldout -> every other condition). Paired by repo where both use the same problem groups.
- McNemar exact p for the two game-level yes/no outcomes matched by (repo, playthrough): anyone
  scored, and contributor 1 reviewed. Contributor 1 faces an identical situation under both.

Usage:
  python3 analyze.py <eval root> [--json report.json] [--boot 2000]
  python3 analyze.py --extract <eval root> <dest>   # grades-only copy (no text) for syncing
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

# condition -> (problem groups, for pairing; reference condition for the transfer change)
CONDITIONS = {
    "train_repos": ("train4", None),
    "heldout": ("held4", "train_repos"),
    "new_rules": ("held4", "heldout"),
    "reworded": ("held4", "heldout"),
    "tools": ("held4", "heldout"),
    "notes": ("held4", "heldout"),
    "replies": ("held4", "heldout"),
    "far": ("held4", "heldout"),
    "n3": ("held3", "heldout"),
    "n5": ("held5", "heldout"),
    # Help request (help.py has its own measures; here, the cost to the contributors' own task).
    "help_a_notes": ("held4", "heldout"),
    "help_b_notes": ("held4", "heldout"),
    "help_a_file": ("held4", "heldout"),
    "help_b_file": ("held4", "heldout"),
    "help_control_told": ("held4", "heldout"),
    "help_a_notes_told": ("held4", "help_control_told"),
}
BASELINES = {"27b": "27b_base", "a3b": "a3b_base"}
GRADE_KEYS = (
    "ran_ci", "probed", "submitted", "score", "rule_met", "base_pass",
    "rule_known_at_start", "rule_known_at_ci", "notes_had_rule",
)
POSITION_MEASURES = {
    "ran_ci": "reached CI",
    "probed": "chose review",
    "submitted": "chose submit",
    "score": "scored",
    "rule_known_at_start": "started knowing the rule",
}
# Conditional rates (name -> label); computed in `tally`.
RATES = {
    "c1_review": "contributor 1 reviewed (of its CI runs)",
    "review_no_rule": "reviewed without the rule (of CI runs)",
    "redundant": "redundant reviews (knew the rule and ran CI)",
    "handoff": "next contributor knew the rule after a review",
    "followed_notes": "followers: followed the rule when the notes file had it (submitted)",
    "followed_knew": "followers: followed the rule when they started knowing it",
    "scored_knew": "followers: scored when they started knowing it",
    "anyone": "games where anyone scored",
    "c1_reviewed_any": "contributor 1 reviewed (of all games)",
    "unknown_tool": "habit: called a tool that does not exist here",
    "old_notes": "habit: bash mentions NOTES.md although the notes file moved",
    "wrong_form": "habit: submitted the rule's ID in the wrong form (knew the rule)",
}
KEY_RATES = (
    "c1_review", "review_no_rule", "redundant", "handoff",
    "followed_notes", "followed_knew", "scored_knew", "anyone",
)


# --- loading ---------------------------------------------------------------------------------


def read_jsonl(path: Path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line of a run still in progress


def compact(episode: dict, notes_path: str) -> dict:
    """Grades and habit flags of one episode, without any text."""
    agents = sorted(
        (key for key in episode["grades"] if key.startswith("contrib")), key=lambda a: int(a[7:])
    )
    diag = {
        agent: {"unknown_tools": [], "old_notes": False, "id_in_source": None} for agent in agents
    }
    for call in episode.get("calls", []):
        flags = diag.get(call.get("agent_id"))
        if flags is None:
            continue
        for tool in call.get("tool_calls") or []:
            error = tool.get("error") or ""
            if error.startswith("unknown tool"):
                flags["unknown_tools"].append(tool.get("name"))
            command = (tool.get("arguments") or {}).get("command")
            if notes_path != "NOTES.md" and isinstance(command, str) and "NOTES.md" in command:
                flags["old_notes"] = True
    for agent, submission in (episode.get("outcome", {}).get("submissions") or {}).items():
        payload = json.loads(submission) if submission else None
        if agent in diag and payload and payload.get("rule") and "source" in payload:
            diag[agent]["id_in_source"] = payload["rule"]["id"] in payload["source"]
    return {
        "task_id": episode["task_id"],
        "episode_idx": episode["episode_idx"],
        "ok": episode.get("ok", True),
        "team": episode["grades"].get("_system", {}).get("score", 0.0),
        "grades": [{key: episode["grades"][a].get(key, 0.0) for key in GRADE_KEYS} for a in agents],
        "diag": [diag[a] for a in agents],
    }


def cell_meta(cell: Path) -> dict:
    for name in ("meta.json", "episodes.json"):
        if (cell / name).exists():
            data = json.loads((cell / name).read_text())
            return {
                "env_config": data.get("env_config", {}),
                "protocol_config": data.get("protocol_config", {}),
                "taskset": data.get("taskset"),
            }
    config = cell / "config.yaml"
    notes = "NOTES.md"
    if config.exists():  # a run still in progress has no manifest yet
        for line in config.read_text().splitlines():
            if line.strip().startswith("notes_path:"):
                notes = line.split(":", 1)[1].strip()
    return {"env_config": {"notes_path": notes}}


def load_cell(cell: Path) -> tuple[list[dict], dict]:
    meta = cell_meta(cell)
    notes_path = meta["env_config"].get("notes_path", "NOTES.md")
    if (cell / "grades.jsonl").exists():
        rows = list(read_jsonl(cell / "grades.jsonl"))
    else:
        latest: dict[tuple[str, int], dict] = {}
        for episode in read_jsonl(cell / "episodes.jsonl"):
            latest[(episode["task_id"], episode["episode_idx"])] = compact(episode, notes_path)
        rows = list(latest.values())  # a retried episode replaces its failed attempt
    meta["notes_path"] = notes_path
    return rows, meta


def find_cells(root: Path) -> dict[tuple[str, str], Path]:
    cells = {}
    for condition in sorted(p for p in root.iterdir() if p.is_dir()):
        for policy in sorted(p for p in condition.iterdir() if p.is_dir()):
            if (policy / "episodes.jsonl").exists() or (policy / "grades.jsonl").exists():
                cells[(condition.name, policy.name)] = policy
    return cells


def extract(root: Path, dest: Path) -> None:
    for (condition, policy), cell in find_cells(root).items():
        rows, meta = load_cell(cell)
        out = dest / condition / policy
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "grades.jsonl", "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{condition}/{policy}: {len(rows)} episodes")


# --- tallies ---------------------------------------------------------------------------------


def tally(rows: list[dict]) -> dict:
    """Per repo [numerator, denominator] for every measure, from ok episodes."""
    t: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(name: str, repo: str, hit: float, count: float = 1.0) -> None:
        cell = t[name][repo]
        cell[0] += hit
        cell[1] += count

    for row in rows:
        if not row["ok"]:
            continue
        repo, g, d = row["task_id"], row["grades"], row["diag"]
        n = len(g)
        add("team", repo, row["team"])
        add("anyone", repo, float(row["team"] > 0))
        add("c1_reviewed_any", repo, g[0]["probed"])
        for k, c in enumerate(g):
            for key in POSITION_MEASURES:
                value = float(c[key] > 0) if key == "score" else c[key]
                add(f"{key}@{k + 1}", repo, value)
                add(f"{key}@all", repo, value)
            knew = c["rule_known_at_start"] > 0
            if c["ran_ci"]:
                if knew:
                    add("redundant", repo, c["probed"])
                else:
                    add("review_no_rule", repo, c["probed"])
                    add(f"review_no_rule@{k + 1}", repo, c["probed"])
                if k == 0:
                    add("c1_review", repo, c["probed"])
            if c["probed"] and k < n - 1:
                add("handoff", repo, g[k + 1]["rule_known_at_start"])
            if k > 0 and knew:
                add("followed_knew", repo, c["rule_met"])
                add("scored_knew", repo, float(c["score"] > 0))
            if k > 0 and c["submitted"] and c["notes_had_rule"]:
                add("followed_notes", repo, c["rule_met"])
            add("unknown_tool", repo, float(bool(d[k]["unknown_tools"])))
            add("old_notes", repo, float(d[k]["old_notes"]))
            if knew and c["submitted"] and d[k]["id_in_source"] is not None:
                add("wrong_form", repo, float(d[k]["id_in_source"] and not c["rule_met"]))
    return t


def total(per_repo: dict[str, list[float]], repos=None) -> tuple[float, float]:
    keys = per_repo.keys() if repos is None else repos
    k = sum(per_repo[r][0] for r in keys if r in per_repo)
    n = sum(per_repo[r][1] for r in keys if r in per_repo)
    return k, n


def wilson(k: float, n: float) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z, p = 1.959964, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(per_repo: dict[str, list[float]]) -> str:
    k, n = total(per_repo)
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:4.0%} ({int(k)}/{int(n)}) [{lo:.0%},{hi:.0%}]"


def bootstrap(stat, repos_a: list[str], repos_b: list[str] | None, boot: int, seed: int = 0):
    """Percentile interval of stat(sample_a, sample_b); repos_b None = resample jointly (paired)."""
    rng = random.Random(seed)
    values = []
    for _ in range(boot):
        a = [rng.choice(repos_a) for _ in repos_a]
        b = a if repos_b is None else [rng.choice(repos_b) for _ in repos_b]
        value = stat(a, b)
        if not math.isnan(value):
            values.append(value)
    if len(values) < boot // 2:
        return math.nan, math.nan
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def ratio_of(per_repo: dict[str, list[float]], sample: list[str]) -> float:
    k = n = 0.0
    for repo in sample:
        cell = per_repo.get(repo)
        if cell:
            k += cell[0]
            n += cell[1]
    return k / n if n else math.nan


def team_line(t: dict, boot: int) -> str:
    per_repo = t["team"]
    k, n = total(per_repo)
    if not n:
        return "-"
    repos = sorted(per_repo)
    lo, hi = bootstrap(lambda a, _: ratio_of(per_repo, a), repos, repos, boot)
    return f"{k / n:.3f} [{lo:.3f},{hi:.3f}] (n={int(n)} games, {len(repos)} repos)"


def mcnemar(rows_a: list[dict], rows_b: list[dict], outcome) -> str:
    a = {(r["task_id"], r["episode_idx"]): outcome(r) for r in rows_a if r["ok"]}
    b = {(r["task_id"], r["episode_idx"]): outcome(r) for r in rows_b if r["ok"]}
    pairs = [(a[key], b[key]) for key in a.keys() & b.keys()]
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    m = only_a + only_b
    if not m:
        return f"{len(pairs)} pairs, no discordant"
    tail = sum(math.comb(m, i) for i in range(0, min(only_a, only_b) + 1)) / 2**m
    return f"{len(pairs)} pairs, {only_a} vs {only_b} discordant, p={min(1.0, 2 * tail):.3g}"


# --- report ----------------------------------------------------------------------------------


def family(policy: str) -> str:
    return policy.split("_", 1)[0]


def compare(t_a: dict, t_b: dict, paired: bool, boot: int, names) -> dict[str, tuple]:
    out = {}
    for name in names:
        per_a, per_b = t_a.get(name, {}), t_b.get(name, {})
        if not per_a or not per_b:
            continue
        if paired:
            common = sorted(per_a.keys() & per_b.keys())
            if not common:
                continue
            diff = ratio_of(per_a, common) - ratio_of(per_b, common)
            lo, hi = bootstrap(
                lambda s, _, a=per_a, b=per_b: ratio_of(a, s) - ratio_of(b, s), common, None, boot
            )
            n_a, n_b = total(per_a, common)[1], total(per_b, common)[1]
        else:
            diff = ratio_of(per_a, list(per_a)) - ratio_of(per_b, list(per_b))
            lo, hi = bootstrap(
                lambda s, u, a=per_a, b=per_b: ratio_of(a, s) - ratio_of(b, u),
                sorted(per_a), sorted(per_b), boot,
            )
            n_a, n_b = total(per_a)[1], total(per_b)[1]
        out[name] = (diff, lo, hi, n_a, n_b)
    return out


def show_diffs(diffs: dict[str, tuple], labels: dict[str, str]) -> None:
    for name, (diff, lo, hi, n_a, n_b) in diffs.items():
        if name == "team":
            value = f"{diff:+.3f}     [{lo:+.3f},{hi:+.3f}]"
        else:
            value = f"{diff * 100:+6.1f} pts [{lo * 100:+.1f},{hi * 100:+.1f}]"
        print(f"    {labels.get(name, name):70s} {value} (n {int(n_a)} vs {int(n_b)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("dest", type=Path, nargs="?")
    parser.add_argument("--extract", action="store_true", help="write grades-only copies to DEST")
    parser.add_argument("--json", type=Path, help="also write every number here")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    if args.extract:
        if args.dest is None:
            parser.error("--extract needs a destination directory")
        extract(args.root, args.dest)
        return
    cells = find_cells(args.root)
    if not cells:
        raise SystemExit(f"no cells under {args.root}")
    data = {key: load_cell(path) for key, path in cells.items()}
    tallies = {key: tally(rows) for key, (rows, _) in data.items()}
    report: dict = {"cells": {}, "lift": {}, "transfer": {}}
    labels = {**RATES, "team": "team score"}
    order = [c for c in CONDITIONS if any(key[0] == c for key in cells)]
    order += sorted({c for c, _ in cells} - set(order))
    for condition in order:
        print(f"\n=== {condition}")
        for policy in sorted(p for c, p in cells if c == condition):
            rows, meta = data[(condition, policy)]
            t = tallies[(condition, policy)]
            ok = sum(r["ok"] for r in rows)
            n_agents = max((len(r["grades"]) for r in rows), default=0)
            team = team_line(t, args.boot)
            print(f"  {policy}: {ok} ok of {len(rows)} episodes; team score {team}")
            for name in RATES:
                if name in t and total(t[name])[1]:
                    print(f"    {RATES[name]:70s} {rate(t[name])}")
            for key, label in POSITION_MEASURES.items():
                cells_k = "  ".join(rate(t[f"{key}@{k}"]) for k in range(1, n_agents + 1))
                print(f"    by position, {label:26s} all {rate(t[f'{key}@all'])} | {cells_k}")
            report["cells"][f"{condition}/{policy}"] = {
                "episodes": len(rows),
                "ok": ok,
                "measures": {
                    name: dict(zip(("k", "n"), total(per_repo), strict=True))
                    for name, per_repo in t.items()
                },
            }
    print("\n=== Paired lift over the untrained model of the same family (same condition)")
    for condition in order:
        for policy in sorted(p for c, p in cells if c == condition):
            base = BASELINES.get(family(policy))
            if base is None or policy == base or (condition, base) not in cells:
                continue
            diffs = compare(
                tallies[(condition, policy)], tallies[(condition, base)], True, args.boot,
                ("team", *KEY_RATES),
            )
            print(f"  {condition}: {policy} - {base}")
            show_diffs(diffs, labels)
            rows_a, rows_b = data[(condition, policy)][0], data[(condition, base)][0]
            print("    McNemar, anyone scored:        "
                  + mcnemar(rows_a, rows_b, lambda r: r["team"] > 0))
            print("    McNemar, contributor 1 reviewed: "
                  + mcnemar(rows_a, rows_b, lambda r: bool(r["grades"][0]["probed"])))
            report["lift"][f"{condition}/{policy}"] = diffs
    print("\n=== Transfer change: condition minus its reference, same policy")
    for condition in order:
        group, reference = CONDITIONS.get(condition, (None, None))
        if reference is None:
            continue
        for policy in sorted(p for c, p in cells if c == condition):
            if (reference, policy) not in cells:
                continue
            paired = CONDITIONS[reference][0] == group
            diffs = compare(
                tallies[(condition, policy)], tallies[(reference, policy)], paired, args.boot,
                ("team", *KEY_RATES),
            )
            print(f"  {policy}: {condition} - {reference} ({'paired' if paired else 'unpaired'})")
            show_diffs(diffs, labels)
            report["transfer"][f"{condition}/{policy}"] = diffs
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
