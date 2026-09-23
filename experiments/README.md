# Experiments: the lab notebook

Use one directory per study: `experiments/<YYYY-MM-DD>_<slug>/`. Related
sub-studies belong in subdirectories of their parent study instead of
proliferating top-level files.

Each study contains:

- `README.md`: question, hypothesis, design/arms, exact commands run, results
  summary with n and CIs, deviations from the design, and spend.
- `configs/`: YAML passed to `marli` verbs.
- `run.sh`: the canonical chain of `marli` CLI calls, or `run.py` awaiting
  library verbs when the chain needs logic.
- `results/`: small committed deliverables (`results.jsonl`, `RESULTS.md`,
  `figures/`).
- `out/`: gitignored bytes such as episodes and checkpoints.

Results stay as-run: never rewrite outputs; add a dated note instead. Studies
consume `marli` verbs rather than re-implementing runners. Every paid step
states its budget (`max_usd`) and is confirmed with Sid before launch. Ingest
durable findings into `docs/wiki/` at wrap-up, following the
[wiki schema](../docs/wiki/CLAUDE.md). Never commit gated or
contamination-sensitive benchmark question or transcript text (GPQA, AIME,
...); commit aggregate numbers only.

Start from the repo root (replace `<slug>` with the study name):

```bash
cp -r experiments/_template experiments/$(date +%F)_<slug>
```
