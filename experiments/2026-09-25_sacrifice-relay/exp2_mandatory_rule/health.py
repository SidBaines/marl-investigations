"""Training health and timing per step for an experiment 2 `train rl` run dir (numbers only).

Usage: python3 health.py <run dir>
"""

import json
import statistics as st
import sys
from pathlib import Path


def main() -> None:
    run = Path(sys.argv[1])
    with open(run / "metrics.jsonl") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    print("step sample_s train_s team_score no_signal_groups kl_sample_train ratio_mean grad_norm")
    for r in rows:
        learner = (r.get("learners") or {}).get("policy") or {}
        credit = r["credit"]
        kl = learner.get("kl_sample_train")
        ratio = (learner.get("metrics") or {}).get("ratio_mean")
        print(
            r["step"],
            round(r["sample_seconds"]),
            round(r["train_seconds"]),
            f"{r['grades']['_system']['score']:.3f}",
            f"{credit['zero_variance_groups']}/{credit['n_groups']}",
            f"{kl:.5f}" if kl is not None else "skipped",
            f"{ratio:.5f}" if ratio is not None else "-",
            round(learner["grad_norm"]) if "grad_norm" in learner else "-",
        )
    steps = [r["sample_seconds"] + r["train_seconds"] for r in rows]
    kls = [r["learners"]["policy"]["kl_sample_train"] for r in rows if r.get("learners")]
    print(f"steps: {len(rows)}; mean step {st.fmean(steps) / 60:.1f} min; max kl {max(kls):.5f}")


if __name__ == "__main__":
    main()
