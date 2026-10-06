"""Figures for the study README (`README.md`), re-made from saved data.

Two steps:
- `extract` reads the training runs' saved games (`<run>/rollouts/step_*/episodes.jsonl`, on the
  dev box only, because they hold problem text) and writes numbers only to `results/`:
  - `training_steps.jsonl`: one row per run and step, with counts by position, so any block of
    steps pools exactly;
  - `review_payoff.json`: team score of games with a review minus the same group's games without
    one (steps 0-29; one value per group, as `payoff.py`), with 95% bootstrap intervals.
- `plot` draws `figures/*.png` from `results/` and the eval sessions' committed aggregates
  (`exp2_eval/results/`, `../2026-10-05_coop-evals/results/`).

Usage (from the repo root; `--root` is the study dir that holds the runs' `out/` dirs):
  uv run --with matplotlib python experiments/2026-09-25_sacrifice-relay/plots.py extract \
      [--root DIR] [--jobs 4]
  uv run --with matplotlib python experiments/2026-09-25_sacrifice-relay/plots.py plot
"""

import argparse
import json
import math
import random
import re
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    import orjson

    loads = orjson.loads
except ImportError:  # orjson only makes `extract` faster
    loads = json.loads

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
EVAL = HERE / "exp2_eval" / "results"
COOP = HERE.parent / "2026-10-05_coop-evals" / "results"
N = 4
PAYOFF_STEPS = range(30)
BOOT = 4000
BLOCK = 5  # steps pooled per point in the training curves

# Counts per position (contributor k + 1) and step.
FIELDS = (
    "slots",  # contributors at this position (= games)
    "ci",  # ran CI
    "knew",  # started knowing the rule
    "nr_ci",  # started without the rule and ran CI
    "nr_review",  # ... and reviewed
    "nr_ci_after",  # started without the rule and ran CI after an earlier contributor's CI run
    "nr_review_after",  # ... and reviewed
    "knew_ci",  # knew the rule and ran CI
    "knew_review",  # ... and reviewed anyway (redundant)
    "knew_followed",  # knew the rule and followed it
    "full",  # passed the base tests and followed the rule
    "score",  # sum of own scores
)


@dataclass(frozen=True)
class Run:
    path: str  # run dir, relative to the study dir
    model: str  # "27B" or "A3B"
    label: str  # legend label


RUNS = {
    "exp1_team_27b": Run("out/trial_team", "27B", "27B, team reward (exp 1)"),
    "exp2_team_27b": Run("exp2_mandatory_rule/out/train_team_checks", "27B", "27B, team reward"),
    "exp2_individual_27b": Run(
        "exp2_mandatory_rule/out/train_individual_checks", "27B", "27B, individual reward"
    ),
    "exp2_team_a3b": Run(
        "exp2_mandatory_rule/out/train_team_checks_a3b", "A3B", "A3B, team reward"
    ),
    "exp2_1_27b": Run("exp2_mandatory_rule/out/train_opener_checks", "27B", "27B"),
    "exp2_1_a3b": Run("exp2_mandatory_rule/out/train_opener_checks_a3b", "A3B", "A3B"),
    "exp3_team_a3b": Run("exp3_sacrifice_a3b/out/train_team", "A3B", "A3B, team reward (exp 3)"),
    "exp3_individual_a3b": Run(
        "exp3_sacrifice_a3b/out/train_individual", "A3B", "A3B, individual reward (exp 3)"
    ),
}
COLOURS = {"27B": "#3b6ea8", "A3B": "#d9822b"}


# ---------------------------------------------------------------- extract


def step_counts(path: Path) -> dict:
    """Counts for one step's games, plus (team score, first reviewer) per game by group."""
    pos = [dict.fromkeys(FIELDS, 0) for _ in range(N)]
    games, team, team_sq = 0, 0.0, 0.0
    groups: dict[str, list] = defaultdict(list)
    with open(path, "rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            e = loads(line)
            if not e.get("ok", True):
                continue
            g = [e["grades"][f"contrib{k}"] for k in range(N)]
            t = e["grades"]["_system"]["score"]
            if abs(t - st.fmean(c["score"] for c in g)) > 1e-9:
                raise ValueError(f"{path}: team score is not the contributors' mean")
            games += 1
            team += t
            team_sq += t * t
            for k, c in enumerate(g):
                p = pos[k]
                knew, ci, probed = c["rule_known_at_start"], c["ran_ci"], c["probed"]
                p["slots"] += 1
                p["ci"] += ci
                p["knew"] += knew
                p["score"] += c["score"]
                p["full"] += bool(c["base_pass"] and c["rule_met"])
                if knew:
                    p["knew_followed"] += c["rule_met"]
                    p["knew_ci"] += ci
                    p["knew_review"] += bool(ci and probed)
                elif ci:
                    p["nr_ci"] += 1
                    p["nr_review"] += probed
                    if any(g[j]["ran_ci"] for j in range(k)):
                        p["nr_ci_after"] += 1
                        p["nr_review_after"] += probed
            first = next((k for k in range(N) if g[k]["probed"]), None)
            groups[e["group_id"]].append((t, first))
    return {
        "games": games,
        "team_sum": team,
        "team_sumsq": team_sq,
        "positions": [{k: int(v) for k, v in p.items()} for p in pos],
        "groups": dict(groups),
    }


def bootstrap(values: list[float], seed: int = 0) -> dict:
    rng = random.Random(seed)
    boots = sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(BOOT))
    return {
        "mean": st.fmean(values),
        "low": boots[int(0.025 * BOOT)],
        "high": boots[int(0.975 * BOOT)],
        "groups": len(values),
    }


def payoff(groups: list[list[tuple[float, int | None]]]) -> dict:
    out = {}
    for label, keep in (("any", lambda f: f is not None), ("contributor_1", lambda f: f == 0)):
        diffs = []
        for rows in groups:
            rev = [t for t, f in rows if keep(f)]
            none = [t for t, f in rows if f is None]
            if rev and none:
                diffs.append(st.fmean(rev) - st.fmean(none))
        out[label] = bootstrap(diffs) if diffs else None
    return out


def extract(root: Path, jobs: int) -> None:
    jobs_by_file = []
    for name, run in RUNS.items():
        paths = sorted((root / run.path / "rollouts").glob("step_*/episodes.jsonl"))
        if not paths:
            raise SystemExit(f"{name}: no rollouts under {root / run.path}")
        jobs_by_file += [(name, int(p.parent.name.removeprefix("step_")), p) for p in paths]
    with ProcessPoolExecutor(jobs) as pool:
        counts = list(pool.map(step_counts, [p for _, _, p in jobs_by_file], chunksize=1))
    RESULTS.mkdir(exist_ok=True)
    payoff_groups: dict[str, list] = defaultdict(list)
    with open(RESULTS / "training_steps.jsonl", "w") as out:
        for (name, step, _), c in zip(jobs_by_file, counts, strict=True):
            if not c["games"]:
                continue
            if step in PAYOFF_STEPS:
                payoff_groups[name] += c["groups"].values()
            row = {"run": name, "step": step, **{k: v for k, v in c.items() if k != "groups"}}
            out.write(json.dumps(row) + "\n")
    report = {
        "steps": [PAYOFF_STEPS.start, PAYOFF_STEPS.stop - 1],
        "measure": "team score, games with a review minus the same group's games without one",
        "runs": {name: payoff(groups) for name, groups in payoff_groups.items()},
    }
    (RESULTS / "review_payoff.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {RESULTS / 'training_steps.jsonl'} and {RESULTS / 'review_payoff.json'}")


# ---------------------------------------------------------------- plot helpers


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    if not n:
        return (math.nan, math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, centre - half, centre + half)


def steps_of(rows: list[dict], run: str) -> dict[int, dict]:
    return {r["step"]: r for r in rows if r["run"] == run}


def pool(steps: list[dict]) -> dict:
    """Sum a block of steps' counts."""
    out = {k: sum(s[k] for s in steps) for k in ("games", "team_sum", "team_sumsq")}
    out["positions"] = [
        {f: sum(s["positions"][k][f] for s in steps) for f in FIELDS} for k in range(N)
    ]
    return out


def blocks(by_step: dict[int, dict], size: int = BLOCK) -> list[tuple[float, dict]]:
    """Complete blocks of `size` consecutive steps from step 0, at their mid step."""
    out = []
    for start in range(0, max(by_step) + 1, size):
        chunk = [by_step.get(s) for s in range(start, start + size)]
        if all(chunk):
            out.append((start + (size - 1) / 2, pool(chunk)))
    return out


def team_score(b: dict) -> tuple[float, float, float]:
    n, mean = b["games"], b["team_sum"] / b["games"]
    var = max(b["team_sumsq"] / n - mean * mean, 0.0)
    half = 1.96 * math.sqrt(var / n)
    return (mean, mean - half, mean + half)


def rate(b: dict, num: str, den: str, positions: range) -> tuple[float, float, float]:
    return wilson(
        sum(b["positions"][k][num] for k in positions),
        sum(b["positions"][k][den] for k in positions),
    )


def c1_review(b: dict) -> tuple[float, float, float]:
    """Contributor 1's review rate, of its CI runs (it never starts knowing the rule)."""
    return rate(b, "nr_review", "nr_ci", range(1))


def curve(ax, points: list[tuple[float, tuple]], scale: float, **style) -> None:
    xs = [x for x, _ in points]
    ax.plot(xs, [v[0] * scale for _, v in points], marker="o", markersize=3, **style)
    ax.fill_between(
        xs,
        [v[1] * scale for _, v in points],
        [v[2] * scale for _, v in points],
        color=style["color"],
        alpha=0.15,
        linewidth=0,
    )


def tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------- figures


def fig_training(rows: list[dict], plt) -> None:
    panels = [
        (
            "Experiment 2: reviewing costs the reviewer nothing (0/1 scoring)",
            ["exp2_team_27b", "exp2_individual_27b", "exp2_team_a3b"],
            "Team score (0 to 1)",
        ),
        (
            "Experiments 1 and 3: reviewing costs the reviewer its point (0/1/3 scoring)",
            ["exp1_team_27b", "exp3_team_a3b", "exp3_individual_a3b"],
            "Team score (0 to 3)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    for (title, runs, score_label), (left, right) in zip(panels, axes, strict=True):
        for name in runs:
            run = RUNS[name]
            style = {
                "color": COLOURS[run.model],
                "linestyle": "--" if "individual" in name else "-",
                "label": run.label,
            }
            points = blocks(steps_of(rows, name))
            curve(left, [(x, c1_review(b)) for x, b in points], 100, **style)
            curve(right, [(x, team_score(b)) for x, b in points], 1, **style)
        left.set_title(title, loc="left", fontsize=11, x=0, pad=18)
        left.set_ylabel("Contributor 1 reviewed (% of its CI runs)")
        left.set_ylim(0, 100)
        right.set_ylabel(score_label)
        right.set_ylim(bottom=0)
        left.legend(fontsize=8, frameon=False, loc="upper left")
        for ax in (left, right):
            tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel(f"Training step (points pool {BLOCK} steps; bands are 95% intervals)")
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=150)


def fig_followers(rows: list[dict], plt) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    followers = range(1, N)
    for name in ("exp2_1_27b", "exp2_1_a3b"):
        run = RUNS[name]
        style = {"color": COLOURS[run.model], "label": run.label}
        points = blocks(steps_of(rows, name))
        curve(
            left,
            [(x, rate(b, "knew_followed", "knew", followers)) for x, b in points],
            100,
            **style,
        )
        curve(
            right,
            [(x, rate(b, "nr_review_after", "nr_ci_after", followers)) for x, b in points],
            100,
            **style,
        )
    left.set_title("Followers who started knowing the rule:\nfollowed it", loc="left", fontsize=10)
    right.set_title(
        "Followers without the rule, after an earlier CI run:\nreviewed (% of their CI runs)",
        loc="left",
        fontsize=10,
    )
    left.set_ylabel("%")
    left.set_ylim(0, 100)
    left.legend(frameon=False)
    for ax in (left, right):
        ax.set_xlabel(f"Training step (points pool {BLOCK} steps)")
        tidy(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp2_1_followers.png", dpi=150)


def fig_payoff(rows: list[dict], plt) -> None:
    report = json.loads((RESULTS / "review_payoff.json").read_text())["runs"]
    panels = [
        ("0/1 scoring: a review costs\nthe reviewer nothing", ["exp2_team_27b", "exp2_team_a3b"]),
        (
            "0/1/3 scoring: a review costs\nthe reviewer its point",
            ["exp1_team_27b", "exp3_team_a3b"],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, (title, runs) in zip(axes, panels, strict=True):
        for i, name in enumerate(runs):
            r = report[name]["any"]
            run = RUNS[name]
            ax.bar(i, r["mean"], color=COLOURS[run.model], width=0.6)
            ax.errorbar(
                i,
                r["mean"],
                yerr=[[r["mean"] - r["low"]], [r["high"] - r["mean"]]],
                color="black",
                capsize=4,
            )
            by_step = steps_of(rows, name)
            early = c1_review(pool([by_step[s] for s in range(0, 10)]))[0]
            late = c1_review(pool([by_step[s] for s in range(20, 30)]))[0]
            ax.annotate(
                f"contributor 1 reviewed\n{early:.0%} → {late:.0%}",
                (i, max(r["high"], 0)),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(
            range(len(runs)),
            [RUNS[n].label.replace(", team reward", "\nteam reward") for n in runs],
            fontsize=9,
        )
        ax.set_title(title, loc="left", fontsize=10)
        ax.margins(y=0.25)
        tidy(ax)
    axes[0].set_ylabel("Team score: games with a review\nminus same-repo games without")
    fig.text(
        0.01,
        0.01,
        "Steps 0-29 of each team-reward run. Review rates: steps 0-9 → 20-29. "
        "Bars: 95% bootstrap intervals over groups.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "review_payoff.png", dpi=150)


def transfer_team_scores() -> dict[tuple[str, str], tuple[float, float, float]]:
    """(condition, policy) -> team score [95% interval], from the transfer report's text."""
    out, condition = {}, None
    text = (EVAL / "eval_session_2026-10-05_report.txt").read_text()
    for line in text.splitlines():
        if line.startswith("=== "):
            condition = line.removeprefix("=== ").strip()
        m = re.match(
            r"\s+(\S+): \d+ ok of \d+ episodes; team score ([\d.]+) \[([\d.]+),([\d.]+)\]", line
        )
        if m and condition:
            out[condition, m[1]] = tuple(float(x) for x in m.group(2, 3, 4))
    return out


def fig_transfer(rows: list[dict], plt) -> None:
    report = json.loads((EVAL / "eval_session_2026-10-05_report.json").read_text())["cells"]
    scores = transfer_team_scores()
    groups = [
        ("27B", "heldout", "new problems", "27b"),
        ("27B", "far", "everything\nchanged", "27b"),
        ("A3B", "heldout", "new problems", "a3b"),
        ("A3B", "far", "everything\nchanged", "a3b"),
    ]
    policies = {"27b": ("27b_base", "27b_team_s59"), "a3b": ("a3b_base", "a3b_team_s79")}
    end_of_training = {
        "27b": ("exp2_team_27b", range(50, 60)),
        "a3b": ("exp2_team_a3b", range(70, 80)),
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    width = 0.36
    for i, (model, condition, _label, key) in enumerate(groups):
        for j, policy in enumerate(policies[key]):
            x = i + (j - 0.5) * width
            style = {"color": COLOURS[model], "alpha": 0.45 if j == 0 else 1.0, "width": width}
            m, lo, hi = scores[condition, policy]
            left.bar(x, m, **style)
            left.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=3)
            c = report[f"{condition}/{policy}"]["measures"]["c1_review"]
            p, lo, hi = wilson(c["k"], c["n"])
            right.bar(x, 100 * p, **style)
            right.errorbar(
                x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="black", capsize=3
            )
        name, steps = end_of_training[key]
        b = pool([steps_of(rows, name)[s] for s in steps])
        for ax, value in ((left, team_score(b)[0]), (right, 100 * c1_review(b)[0])):
            ax.hlines(
                value,
                i - width,
                i + width,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="end of training (training repos)" if i == 0 else None,
            )
    for ax in (left, right):
        ax.set_xticks(range(len(groups)), [f"{m}\n{lab}" for m, _, lab, _ in groups], fontsize=9)
        tidy(ax)
    left.set_ylabel("Team score (0 to 1)")
    left.set_ylim(0, 0.5)
    right.set_ylabel("Contributor 1 reviewed (% of its CI runs)")
    right.set_ylim(0, 100)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.45, label="untrained"),
        plt.Rectangle((0, 0), 1, 1, color="grey", label="team-trained"),
        *left.get_legend_handles_labels()[0],
    ]
    left.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.01, 0.01, "25 held-out repos × 2 games = 50 games per bar; 95% intervals.", fontsize=8
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "transfer_eval.png", dpi=150)


def fig_standard(plt) -> None:
    measures = [
        ("Dictator game: share given", "games_{m}.jsonl", "dictator", "give_share"),
        ("Prisoner's dilemma: cooperated", "games_{m}.jsonl", "prisoners_dilemma", "cooperate"),
        ("Public goods: contributed", "games_{m}.jsonl", "public_goods", "contribute"),
        (
            "Volunteer's dilemma: volunteered",
            "volunteer_{m}_report.jsonl",
            "focal",
            "volunteer_rate",
        ),
        ("Volunteer's dilemma: group safe", "volunteer_{m}_report.jsonl", "group", "safe_rate"),
        (
            "HiddenBench: right before discussion",
            "hiddenbench_{m}_report_s3.jsonl",
            "hidden",
            "pre_average",
        ),
        (
            "HiddenBench: right after discussion",
            "hiddenbench_{m}_report_s3.jsonl",
            "hidden",
            "post_average",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (_label, pattern, condition, metric) in enumerate(measures):
        for j, model in enumerate(("27B", "A3B")):
            rows = read_jsonl(COOP / pattern.format(m=model.lower()))
            (gain,) = [
                r["gain"]
                for r in rows
                if r["cell"].startswith("team")
                and r["condition"] == condition
                and r["metric"] == metric
            ]
            y = -i + (0.15 if j == 0 else -0.15)
            d, lo, hi = (100 * gain[k] for k in ("difference", "low", "high"))
            ax.errorbar(
                d,
                y,
                xerr=[[d - lo], [hi - d]],
                fmt="o",
                color=COLOURS[model],
                capsize=3,
                label=model if i == 0 else None,
            )
            if gain["p"] < 0.05:
                ax.annotate(
                    f"p = {gain['p']:.3f}",
                    (hi, y),
                    textcoords="offset points",
                    xytext=(4, -3),
                    fontsize=8,
                )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks([-i for i in range(len(measures))], [m[0] for m in measures], fontsize=9)
    ax.set_xlabel("Team-trained minus untrained (percentage points, 95% interval)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "standard_evals.png", dpi=150)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_jsonl(RESULTS / "training_steps.jsonl")
    FIGURES.mkdir(exist_ok=True)
    fig_training(rows, plt)
    fig_followers(rows, plt)
    fig_payoff(rows, plt)
    fig_transfer(rows, plt)
    fig_standard(plt)
    print(f"wrote {sorted(p.name for p in FIGURES.glob('*.png'))} to {FIGURES}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--root", type=Path, default=HERE, help="study dir holding the runs' out/")
    ext.add_argument("--jobs", type=int, default=4)
    sub.add_parser("plot")
    args = parser.parse_args()
    if args.cmd == "extract":
        extract(args.root, args.jobs)
    else:
        plot()


if __name__ == "__main__":
    main()
