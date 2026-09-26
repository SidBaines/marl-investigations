"""Early (steps 0-9) vs late (steps 20-29) comparison for the overnight trial, with uncertainty.

Usage: python3 trial_stats.py <train rl run dir>
"""

import json
import math
import statistics as st
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def episodes(run: Path, steps: range) -> list[dict]:
    out: list[dict] = []
    for step in steps:
        out += read_jsonl(run / "rollouts" / f"step_{step:05d}" / "episodes.jsonl")
    return [e for e in out if e.get("ok", True)]


def block(eps: list[dict]) -> dict:
    team = [e["grades"]["_system"]["score"] for e in eps]
    contribs = [g for e in eps for k, g in e["grades"].items() if k != "_system"]
    choice = [g for g in contribs if g.get("ran_ci") and not g.get("rule_known_at_start")]
    sub = [g for g in contribs if g.get("submitted")]
    knew_sub = [g for g in sub if g.get("rule_known_at_start")]
    return {
        "relays": len(eps),
        "team_mean": st.fmean(team),
        "team_se": st.stdev(team) / math.sqrt(len(team)),
        "ci_rate": st.fmean(g["ran_ci"] for g in contribs),
        "sacrifice_k": sum(g["probed"] for g in choice),
        "sacrifice_n": len(choice),
        "pass_given_submit": st.fmean(g["base_pass"] for g in sub) if sub else math.nan,
        "bonus_given_knew": st.fmean(g["score"] == 3 for g in knew_sub) if knew_sub else math.nan,
        "n_knew": len(knew_sub),
        "share_3": st.fmean(g["score"] == 3 for g in contribs),
    }


def prop(k: float, n: int) -> tuple[float, float]:
    p = k / n
    return p, math.sqrt(p * (1 - p) / n)


def main() -> None:
    run = Path(sys.argv[1])
    rows = read_jsonl(run / "metrics.jsonl")
    print("TRAINING HEALTH (per step)")
    print("step sample_s train_s kl_sample_train ratio_mean grad_norm")
    for r in rows:
        lrn = r["learners"]["policy"]
        print(
            r["step"],
            round(r["sample_seconds"]),
            round(r["train_seconds"]),
            f"{lrn['kl_sample_train']:.5f}",
            f"{lrn['metrics']['ratio_mean']:.5f}",
            round(lrn["grad_norm"]),
        )
    minutes = st.fmean(r["sample_seconds"] + r["train_seconds"] for r in rows) / 60
    print("mean step minutes:", round(minutes, 1))
    kl_max = max(r["learners"]["policy"]["kl_sample_train"] for r in rows)
    print("max kl_sample_train:", round(kl_max, 5))
    early, late = block(episodes(run, range(0, 10))), block(episodes(run, range(20, 30)))
    print("\nEARLY (steps 0-9) vs LATE (steps 20-29)")
    for name, b in (("early", early), ("late", late)):
        p, se = prop(b["sacrifice_k"], b["sacrifice_n"])
        print(
            f"{name}: relays {b['relays']}, team score {b['team_mean']:.3f} "
            f"± {1.96 * b['team_se']:.3f}, reached CI {b['ci_rate']:.1%}, sacrifice "
            f"{b['sacrifice_k']:.0f}/{b['sacrifice_n']} = {p:.1%} ± {1.96 * se:.1%}, "
            f"pass when submitted {b['pass_given_submit']:.1%}, scored 3 when rule known "
            f"{b['bonus_given_knew']:.1%} (n={b['n_knew']}), scoring 3 {b['share_3']:.1%}"
        )
    d = late["team_mean"] - early["team_mean"]
    z = d / math.hypot(early["team_se"], late["team_se"])
    print(f"team score change {d:+.3f} (z = {z:.2f})")
    p1, s1 = prop(early["sacrifice_k"], early["sacrifice_n"])
    p2, s2 = prop(late["sacrifice_k"], late["sacrifice_n"])
    print(f"sacrifice rate change {p2 - p1:+.1%} (z = {(p2 - p1) / math.hypot(s1, s2):.2f})")


if __name__ == "__main__":
    main()
