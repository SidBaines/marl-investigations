# Cooperation evals in standard harnesses: did team training generalise?

Status: **built and smoke-tested on CPU (2026-10-05); not yet run on a pod.** The run plan
below needs Sid's go-ahead (it costs pod time).

## Question

In the sacrifice relay (`../2026-09-25_sacrifice-relay/`), team reward taught two things:
the first contributor gives up its only scored CI run to learn a hidden rule, and it shares
that rule in `NOTES.md`. Do the trained policies become more cooperative *in general*,
outside our game and outside our harness?

We test with standard external benchmarks run in their own harnesses. Running them in our
interaction layer would reintroduce the cues we want to remove: our system-prompt style,
our tool and turn structure, our message delivery. It would also break comparability with
published numbers.

Every request goes to vLLM's OpenAI-compatible chat endpoint. The model's own chat template
builds the prompt, and vLLM's reasoning parser keeps thinking out of the reply text that
parsers and other agents see.

## What runs

There are three suites, all registered in `src/marli/eval/external/suites/`.

| Suite | Harness | What it measures | Seats |
|---|---|---|---|
| `li_shirado_games` | Inspect task in marli (prompts transcribed verbatim from the paper) | Giving at a personal cost in one-shot games, 100 trials each. Dictator: share of 100 points given. Prisoner's dilemma: give 100 points, which are doubled for the partner. 4-player public goods: contribute. | 1 |
| `hiddenbench` | Upstream code at `3be6ca1` plus `patches/hiddenbench.patch` | Pooling private information. 4 agents each hold one hidden fact and run 15 discussion rounds; we report group and per-seat accuracy before and after discussion. The Full Profile control gives every agent every fact. | 4 |
| `fairgame_volunteer` | FAIRGAME's own runner at `fc302a6` plus `patches/fairgame.patch` | Taking a cost so the group is safe. Volunteer's dilemma, 3 players, 10 known rounds, neutral personalities. Metrics: volunteering rate, round-1 volunteering, and the share of rounds the group is safe. | 3 |

**Cells.** A cell is one policy (games) or one seat assignment (multi-seat suites). The
baseline cell is the untrained model.

- 27B policies: `qwen3_8_27b` (base), `27b_team_s29`, `27b_team_s59`, `27b_indiv_s29`,
  `27b_exp21_s29`.
- A3B policies: `qwen3_6_35b_a3b` (base), `a3b_team_s79`, `a3b_exp21`.
- Mixed cells (multi-seat suites):
  - `mixed_team1`: one `team` seat with untrained seats.
  - `mixed_base1`: the reverse, one untrained seat with `team` seats.

  The seat list rotates with the session or game, so the odd seat plays every position.
- Seat rows are regrouped as **focal** (the odd model's seats) and **others** (the rest). A
  homogeneous cell counts every seat in both groups, so each mixed cell has a like-for-like
  baseline:
  - `focal`: is the trained agent itself right more often?
  - `others`: do untrained teammates do better beside it?

**Report.** Per cell, condition and metric: n, unparsed decisions, harness errors, the mean
with a 95% interval, and the gain over the base cell.

- HiddenBench pairs on its tasks (paired bootstrap; McNemar or sign-flip p).
- The games and FAIRGAME are independent trials (two-sample bootstrap; permutation p).

## Settings

Every setting is recorded in each run's `external.json`.

- **Thinking: on, and the same for every model.** This is both the models' default and how
  training sampled. Thinking moves Li & Shirado's Qwen3-30B from 100/100 to 0/100
  cooperation in the prisoner's dilemma, so it must not differ between cells.
- **Chat template.** The template used on the chat endpoint is the model's own. Checked on
  CPU with `template_parity.py`: for the game prompts, our training renderer's prompt token
  ids equal the HF template's for both models.
  - **One difference, found and fixed.** Qwen3.8's template defaults to
    `reasoning_effort=xhigh`, which adds a "Reasoning effort is set to xhigh…" system
    instruction. Training used `medium`, so `configs/models/27b.yaml` sends
    `chat_template_kwargs.reasoning_effort=medium`.
  - Qwen3.6 has no effort switch, so the A3B needs nothing extra.
  - On the pod, `preflight.py` repeats the check against vLLM's `/tokenize`.
- **Sampling.** None of the three benchmarks fixes sampling: Li & Shirado and HiddenBench used
  provider defaults, and FAIRGAME sets none. So the model-card thinking settings apply.
  - Qwen3.8-27B: T 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
  - Qwen3.6-A3B: the same, but with presence_penalty 1.5.
  - max_tokens is 16384 for both.

  These are sent explicitly on every request (our servers use `--generation-config vllm`,
  so server defaults are vLLM's, not the model's). `marli eval external` refuses upstream
  outputs that were sampled with different settings.
- **Serving.** `configs/serve_*.yaml` is the transfer eval's layout (one engine per GPU, MTP)
  plus `--reasoning-parser qwen3`. The parser does not touch `/v1/completions`, so the
  token-level transfer eval can share the server; `preflight.py` check 4 confirms this.

## Patches (minimal, applied at run time by `run.sh setup`)

- **`patches/hiddenbench.patch`** (+62/−20 lines). Upstream has one client for all agents.
  The patch adds:
  - `--seat-models` (one model name per seat);
  - `--request-kwargs` (extra request fields: top_p, top_k, `chat_template_kwargs`);
  - a record of both in the results;
  - an error record for a scenario that raises, instead of aborting all 65 tasks.

  Prompts, turn order, votes and scoring are untouched.
- **`patches/fairgame.patch`** (+23 lines).
  - `FAIRGAME_LITELLM_KWARGS` (JSON) adds the same request fields to LiteLLM calls.
  - `FAIRGAME_PARSE_FAILURE_LOG` counts the unmatched replies that FAIRGAME re-asks.
  - LiteLLM reaches vLLM unmodified via `hosted_vllm/<name>` and `HOSTED_VLLM_API_BASE`.

## Checks done on CPU (dev box)

- **Parser vs the paper.** On the replies Li & Shirado released (`parity_li_shirado.py`;
  downloaded to /tmp at a pinned revision, not committed), our decision parser reproduces
  their Table 1 to within 2 points in every one of 10 model variants:

  | Model | Dictator share (ours / paper) | PD give | PG contribute | Unparsed |
  | --- | --- | --- | --- | --- |
  | gpt-4o | 0.496 / 0.496 | 95 / 95 | 95 / 96 | 3 / 300 |
  | gpt-o1 | 0.419 / 0.420 | 16 / 16 | 20 / 20 | 1 / 300 |
  | gemini-2.0-flash | 0.473 / 0.473 | 96 / 96 | 100 / 100 | 0 / 300 |
  | gemini-2.0-flash-thinking-exp | 0.297 / 0.297 | 3 / 3 | 2 / 2 | 0 / 300 |
  | deepseek-v3 | 0.486 / 0.488 | 3 / 3 | 25 / 23 | 23 / 300 |
  | deepseek-r1 | 0.277 / 0.276 | 0 / 0 | 0 / 0 | 2 / 300 |
  | claude | 0.410 / 0.410 | 100 / 100 | 99 / 99 | 0 / 300 |
  | claude-thinking | 0.320 / 0.321 | 96 / 96 | 92 / 93 | 1 / 300 |
  | qwen3-32b | 0.500 / 0.500 | 100 / 100 | 64 / 64 | 0 / 300 |
  | qwen3-32b_thinking | 0.099 / 0.099 | 0 / 0 | 0 / 0 | 0 / 300 |

  DeepSeek-V3's 23 unparsed replies are mostly genuine non-answers ("What would you
  choose?").
- **HiddenBench reader vs upstream scoring.** On the same results files, our reader's
  pre/post average- and majority-rule accuracies equal upstream `hiddenbench score`.
  Unit tests pin the formulas on synthetic runs.
- **End to end** (`./run.sh smoke`). All three suites ran against `tests/_fake_openai.py`, a
  vLLM-like stand-in that serves every policy name, with adapters scripted to be selfish:
  - the Inspect games, via `vllm:@server.json#<adapter>` refs;
  - the patched upstream HiddenBench, with rotated mixed seats;
  - the patched FAIRGAME runner, via LiteLLM.

  Ingest checked the pinned commit, the settings sent and the seats. The reports show the
  expected selfish-adapter deficits and n, intervals and gains. The fake server's request log
  confirms what each harness sent: top_k, `chat_template_kwargs`, per-trial seeds and JSON
  mode for votes.

## Not yet verified (pod preflight covers each)

1. vLLM 0.30's `qwen3` reasoning parser on Qwen3.5-family completions, which open with a
   prefilled `<think>`.
2. JSON mode (`response_format=json_object`, used for HiddenBench votes) keeping thinking.
3. `/tokenize` parity with the renderer.
4. Throughput, and therefore the time estimates below.

## Run plan (one 2×H200 pod session, $9.18/hr)

### Commands

Run them in order, after `docs/runbooks/pod.md`'s install. Everything runs on the pod from
the checkout.

```bash
S=experiments/2026-10-05_coop-evals
$S/run.sh setup                              # ~5 min: upstream clones + venvs, Inspect pins
MODEL=27b $S/run.sh serve                    # or: $S/run.sh link <a running server's serve dir>
MODEL=27b $S/run.sh adapters                 # same names as the transfer eval (27b_team_s59, ...)
MODEL=27b $S/run.sh preflight                # must print no FAIL
MODEL=27b $S/run.sh games                    # Li & Shirado, every 27B cell
MODEL=27b SESSIONS=3 $S/run.sh hiddenbench   # upstream HiddenBench, every cell x 3 sessions
MODEL=27b $S/run.sh volunteer                # FAIRGAME, every cell x 50 games
MODEL=27b $S/run.sh report
MODEL=27b $S/run.sh stop; MODEL=a3b $S/run.sh serve; ...   # then the same for the A3B
```

Every phase resumes: rerunning skips finished sessions, games and cells. `CELLS=a,b`
restricts a phase to some cells.

### Grid (27B)

| Suite | Cells | n per cell | Approximate tokens generated |
|---|---|---|---|
| Games | base, team_s29, team_s59, indiv_s29, exp21_s29 | 3 games × 100 trials | ~1.5M |
| HiddenBench, hidden | base, team_s29, team_s59, indiv_s29, exp21_s29, mixed_team1, mixed_base1 | 65 tasks × 3 sessions | ~2.7M per session-cell; 21 × → ~55M |
| HiddenBench, full | base, team_s59 | 65 × 3 | ~16M |
| Volunteer | the same 7 cells | 50 games × ≤30 calls | ~5M |

The A3B uses the same design with cells base, team_s79, exp21 and the two mixed cells.

### Time and cost estimates

These are estimates. Assumptions:

- Measured on this pod type: the 27B in DP2 + MTP gives ~1.5–2.8k generated tok/s on long
  code contexts. These shorter prompts should allow more, so I assume 4k tok/s for the 27B
  and ~12k tok/s for the A3B (3B active).
- Thinking at medium effort: ~1,000 tokens for a game decision and ~600 for a discussion turn.
- HiddenBench sessions are chains of 68 dependent calls, so wall time per session is at
  least ~20 min. Sessions run in parallel.

| Plan | Contents | Pod time | Cost |
|---|---|---|---|
| **Minimal** | 27B only. Games: base, team_s59, indiv_s29. HiddenBench hidden: base, team_s59, indiv_s29, mixed_team1, plus full: base, at SESSIONS=2. Volunteer: base, team_s59, mixed_team1 at GAMES=30. | ~2.5–3 h (incl. ~25 min setup, serve, adapters, preflight) | **~$23–28** |
| Core | Minimal plus all 27B cells at SESSIONS=3, GAMES=50 | ~5–7 h | ~$46–64 |
| Full | Core plus the A3B (~1.5–2 h with the server switch) | ~7–9 h | ~$64–83 |

Run the minimal grid first. Read HiddenBench's throughput from the first session's wall time
(`out/hiddenbench_27b/*/hidden_s0.log`), then choose SESSIONS for the rest.

### Sharing the session with the transfer eval

The transfer eval (branch `exp2-eval`) uses the same pod layout and the same adapter names,
and reads `/v1/completions`, which the reasoning parser leaves alone. So one server serves
both:

1. Start the server with `MODEL=27b $S/run.sh serve` (the transfer eval's dp2+MTP layout
   plus the parser).
2. Point the transfer eval at it: symlink its `out/serve_27b` to ours, or start its own
   server with the parser in `extra_args` and run `$S/run.sh link <its serve dir>`.
3. Load the adapters once; the other side's `adapters` phase just reports them as already
   loaded.

The two evals compete for throughput, so run the transfer eval's grid first (its priority)
and these phases after it, within the same serve/adapters lifetime. That saves one server
start (~10 min) per model.

## Data, licences and contamination

- Li & Shirado's prompts are short paper text, transcribed into
  `src/marli/eval/external/tasks/econ_games.py` with a citation. Their released replies are
  never committed.
- HiddenBench (MIT) and FAIRGAME (Apache-2.0) tasks, configs and templates are read from the
  pinned checkouts at run time. No benchmark text or transcript is copied into this
  repository, and `results/` holds aggregates only.
- HiddenBench tasks have been public since 2025. Cells are compared against the same base
  model, so contamination is shared across cells.

## Exact commands run

So far: `./run.sh setup` and `./run.sh smoke` on the dev box (WORK=/tmp/coop-evals), plus
`parity_li_shirado.py` and `template_parity.py`.

## Results summary

None yet (pod run pending).

## Spend

| Item | Budget | Actual |
|---|---|---|
| Pod session (minimal plan) | ~$28 (3 h × $9.18), to confirm with Sid | — |
