# Study title

Status: planned / running / done (choose one).

## Question

State the research question this study will answer.

## Hypothesis

State the expected outcome and what evidence would falsify it.

## Design / arms

List arms, model/backend, protocol, seating/credit, split, n, seeds, verifier,
and compute budgets; specify how comparisons are compute-matched.

## Exact commands run

Record the exact commands, YAML configs, overrides, and run directories used
by `run.sh` (or `run.py`).

## Results summary

Report aggregate results with n and CIs, paired lift, generated and
critical-path tokens per episode, and LM calls; link small deliverables in
`results/`. Raw episodes stay in gitignored `out/`; never commit gated benchmark text.

## Deviations from the design

Record departures and their reasons as dated notes; preserve as-run outputs.

## Spend

State each paid step's `max_usd` budget, record Sid's confirmation before
launch, and fill in actual spend afterward (USD).

| Item | Budget | Actual |
|---|---|---|
| `<step>` | `<max_usd>` | `<actual USD>` |
