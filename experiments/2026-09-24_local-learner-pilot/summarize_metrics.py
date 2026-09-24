"""Summarize a train rl metrics.jsonl: per-step reward/acc, per-learner IS ratio, KL, grad norm."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("step acc  r_coord r_work zvar  gen   | coord: n  ratio_mean ratio_max kl       gnorm  | work: n  ratio_mean ratio_max kl       gnorm | ver  s_samp s_train")
for d in rows:
    c = d["credit"]
    rm = c.get("reward_mean", {})
    out = f"{d['step']:>4} {d['accuracy']:.2f}  {rm.get('coordinator', float('nan')):.2f}    {rm.get('worker', float('nan')):.2f}   {c['zero_variance_groups']:>2}  {d['total_gen']:>5.0f} |"
    for name in ("coord", "work"):
        l = d["learners"].get(name)
        if l:
            m = l["metrics"]
            out += f" {l['n_datums']:>3} {m['ratio_mean']:.5f}   {m['ratio_max']:.3f}     {m['kl_sample_train']:.2e} {l['grad_norm']:>6.0f} |"
        else:
            out += "   -   (not stepped)                          |"
    v = d["learner_versions"]
    out += f" {v['coord']}/{v['work']}  {d['sample_seconds']:>5.0f} {d['train_seconds']:>5.0f}"
    print(out)
import statistics as st
for name in ("coord", "work"):
    r = [d["learners"][name]["metrics"]["ratio_mean"] for d in rows if name in d["learners"]]
    k = [d["learners"][name]["metrics"]["kl_sample_train"] for d in rows if name in d["learners"]]
    if r:
        print(f"{name}: steps={len(r)} ratio_mean mean={st.fmean(r):.5f} min={min(r):.5f} max={max(r):.5f}; kl mean={st.fmean(k):.2e} max={max(k):.2e}")
acc = [d["accuracy"] for d in rows]
h = len(acc) // 2
print(f"accuracy: first half {st.fmean(acc[:h]):.3f}, second half {st.fmean(acc[h:]):.3f} (n_steps={len(acc)}, 32 episodes/step)")
