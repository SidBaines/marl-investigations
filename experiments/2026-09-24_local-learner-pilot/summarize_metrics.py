"""Summarize a `train rl` metrics.jsonl: per-step accuracy, rewards, and per-learner
IS ratio, KL and grad norm. Usage: python summarize_metrics.py out/pilot/metrics.jsonl"""

import json
import statistics as st
import sys

LEARNERS = ("coord", "work")


def row(d: dict) -> str:
    c = d["credit"]
    rm = c.get("reward_mean", {})
    out = (
        f"{d['step']:>4} {d['accuracy']:.2f}  {rm.get('coordinator', float('nan')):.2f}    "
        f"{rm.get('worker', float('nan')):.2f}   {c['zero_variance_groups']:>2}  "
        f"{d['total_gen']:>5.0f} |"
    )
    for name in LEARNERS:
        learner = d["learners"].get(name)
        if learner:
            m = learner["metrics"]
            out += (
                f" {learner['n_datums']:>3} {m['ratio_mean']:.5f}   {m['ratio_max']:.3f}     "
                f"{m['kl_sample_train']:.2e} {learner['grad_norm']:>6.0f} |"
            )
        else:
            out += "   -   (not stepped)                          |"
    v = d["learner_versions"]
    times = f"{d['sample_seconds']:>5.0f} {d['train_seconds']:>5.0f}"
    return out + f" {v['coord']}/{v['work']}  {times}"


def main(path: str) -> None:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(
        "step acc  r_coord r_work zvar  gen   | coord: n  ratio_mean ratio_max kl       gnorm  "
        "| work: n  ratio_mean ratio_max kl       gnorm | ver  s_samp s_train"
    )
    for d in rows:
        print(row(d))
    for name in LEARNERS:
        steps = [d["learners"][name]["metrics"] for d in rows if name in d["learners"]]
        if steps:
            r = [m["ratio_mean"] for m in steps]
            k = [m["kl_sample_train"] for m in steps]
            print(
                f"{name}: steps={len(r)} ratio_mean mean={st.fmean(r):.5f} min={min(r):.5f} "
                f"max={max(r):.5f}; kl mean={st.fmean(k):.2e} max={max(k):.2e}"
            )
    acc = [d["accuracy"] for d in rows]
    h = len(acc) // 2
    print(
        f"accuracy: first half {st.fmean(acc[:h]):.3f}, second half {st.fmean(acc[h:]):.3f} "
        f"(n_steps={len(acc)})"
    )


if __name__ == "__main__":
    main(sys.argv[1])
