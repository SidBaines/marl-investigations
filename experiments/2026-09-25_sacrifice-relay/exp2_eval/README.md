# Transfer eval: did experiment 2's policies learn the behaviour, or the format?

Status (2026-10-05): **prepared, nothing run on a GPU.** The held-out candidate problems are built on the dev box
(`out/candidates`, not committed). Everything else runs on one 2×H200 pod; see Grid for time and cost.

Part of the sacrifice-relay study (`../README.md`, `../exp2_mandatory_rule/README.md`).

## Question

Every experiment 2 training game used:
- the same 51 repos (reshuffled; each repo came up 4–13 times);
- the exact same first message, tool names, tool descriptions, CI replies and notes file (`NOTES.md`); only the
  random 6-letter folder names changed;
- house rules of only 10 forms (3 kinds × 3–4 names); only the random ID was new each game.

Team training taught contributor 1 to review and write the rule down, and later contributors to read and follow it.
Did the policies learn that behaviour in general (find the rule, share it, use it), or did they learn to react to
this exact format? This eval moves one thing at a time away from training and measures how much of the trained
behaviour survives, against the untrained model on the same games.

## Design

Every cell is a fixed evaluation of one policy under one condition. Each condition is training's setup (the `checks`
prompt, 0/1 scoring, no position in the prompt, random folder names, the same token limits) plus one change. The seed
is the same for every policy, so a repo's playthrough *k* has the same house rule and folder names under every
policy, which makes the comparisons with the untrained model paired.

| Condition | Repos | What changes | What it tests |
|---|---|---|---|
| `train_repos` | the 51 training repos | nothing (new playthroughs only) | the anchor: trained behaviour in its own setting |
| `heldout` | ~51 held-out repos | new problems | **layer 1**: memorised repos vs a general skill |
| `new_rules` | the same held-out repos | the house rule has a form training never used | **layer 2**: reading the rule vs pattern-matching the 10 trained forms |
| `reworded` | held-out | first message and system prompt paraphrased | **layer 3**: the exact instruction text |
| `tools` | held-out | CI tools renamed `grade_solution` / `inspect_checks`, descriptions reworded | layer 3: tool names and descriptions |
| `notes` | held-out | the shared notes file is `docs/handoff.txt`, not `NOTES.md` | layer 3: the notes file name and place |
| `replies` | held-out | CI's reports reworded | layer 3: the wording of CI's replies |
| `n3`, `n5` | held-out problems, 3 or 5 per repo | 3 or 5 contributors instead of 4 | layer 3: the length of the relay |
| `far` | held-out repos, held-out rules | every layer-3 change at once (4 contributors) | the whole package |

Layer 3 is measured against `heldout` (same repos, so paired); `heldout` against `train_repos`.

### Layer 1: new repos, same format

- **Candidates** (`configs/candidates.yaml`, built 2026-10-05 on the dev box, `out/candidates`): 800 DeepCoder train
  problems, drawn the same way as experiment 1's 800 (shuffled with seed 0), from a new task source
  `deepcoder_no_lcb`: DeepCoder train without its LiveCodeBench subset (513 PrimeIntellect, 287 TACO).
  - None of experiment 1's 800 candidates (so none of the 207 trained problems): same normalised text excluded.
  - No near-duplicates either. A first build without that check found 51 of 800 new problems that were the same
    problem reformatted in another subset (5-word-shingle Jaccard ≥ 0.5 to an old candidate; 11 to a trained one).
    The build therefore also drops any problem sharing a 30-word run with an old candidate. Shorter runs catch
    boilerplate: 20 words already matches 571 of 800, e.g. PrimeIntellect's fixed prompt wrapper.
  - Result (`out/candidates_overlap.txt`, from `overlap.py`): 0 shared ids, 0 shared texts, 0 shared 30-word runs,
    no problem with Jaccard ≥ 0.5 to any old candidate (7 between 0.3 and 0.5, 1 of them to a trained problem).
- **Difficulty filter** (pod, `run.sh filter`): experiment 1's filter exactly: the untrained 27B, one agent,
  plain coding task (no house rule), 4 attempts per problem, keep problems solved 1–3 times (`../configs/eval_filter.yaml`,
  `data filter lo=0.25 hi=0.75 inclusive=true`). Experiment 1 kept 207 of 800 (26%), so expect about 210 ± 25
  problems, i.e. about **51 repos of 4** (45–58), as many as training had.
  - **Why 800:** the same size and procedure as the original keeps the difficulty comparable and gives a held-out set
    as large as the training set. About 51 repos × 2 playthroughs is about 100 games per cell, enough to see the large
    trained effects (contributor 1 reviews 17% → 80%; team score 0.09 → 0.27) and drops of about a third of them.
    The same problems regroup into about 70 repos of 3 and 42 of 5.
  - **Filter time**, measured from experiment 1's call timestamps (`../out/filter_rollout`): 3,200 attempts took
    1.7 h of sampling on two GPUs (one engine each), in three sessions (0.28 + 0.52 + 0.88 h; 15.4M tokens generated).
- **Repos** (`run.sh repos`): `repos_n4` (training rule forms), `repos_n4_new` (the same groups of 4 problems, held-out
  rule forms), `repos_n3`, `repos_n5`; all `data repos seed=0`.

### Layer 2: new kinds of rule

Three held-out rule forms, each with three names, an unguessable ID as in training (3 letters + 4 digits), and
checked from `solution.py`'s source alone (CI only keeps `solution.py`). The ID appears in the requirement and in a
following solution, so "started knowing the rule" (the ID anywhere in the repo) works as before. The training forms
(`header`, `constant`, `docstring`) are unchanged: the same 10 forms and, seed for seed, the same rules.

| Form | Requirement, as CI's review prints it | Check |
|---|---|---|
| `footer` | `The last line of solution.py must be exactly '# checksum: ABC-0042'.` | the last non-blank line, exactly (trailing blank lines are ignored) |
| `function` | `solution.py must define a module-level function deploy_token() that returns the string "ABC-0042".` | the last top-level definition of that name is a plain function callable with no arguments, and returns that string literal in its own body (static; nothing is executed) |
| `class_attr` | `solution.py must define a module-level class Manifest with the class attribute version = "ABC-0042".` | the last top-level definition of that name is a class whose body assigns that string |

Names: `footer` signed-off-by / approved-by / checksum; `function` revision_id / artifact_tag / deploy_token;
`class_attr` Manifest.version / BuildInfo.stamp / Provenance.digest. A decorator rule was left out: following it
needs the decorator defined too, otherwise the base tests fail, which makes the rule harder than the trained forms.

### Layer 3: changed surface

New `code_rules` settings, each independent (any combination works); the defaults are the trained surface:

| Setting | Default (training) | Eval value |
|---|---|---|
| `instructions` | `original` | `paraphrase`: the first message in other words and sentence shapes, the same facts (one CI run, two modes and what each does and scores, the checks are undocumented and cannot be worked out, the notes file is shared and writable at any time, call end_session). No sentence of the original is kept word for word. |
| `tool_names` | `original` (`ci_submit`, `ci_review`) | `renamed`: `grade_solution`, `inspect_checks`, in the tool specs, the prompt and CI's replies. The relay must advertise them (`protocol_config.env_tools`). |
| `tool_text` | `original` | `paraphrase`: both descriptions reworded, same facts (`neutral` is the existing symmetric variant). |
| `ci_replies` | `original` | `paraphrase`: e.g. `Dry-run CI finished (unscored, score 0). The extended checks require: …` and `Scored CI finished. Base tests passed: 3 of 3. Extended checks: failed. Your score: 0. Only inspect_checks shows …`. The requirement sentence itself is unchanged (that is layer 2). |
| `notes_path` | `NOTES.md` | `docs/handoff.txt`: created empty at the start, named in the prompt, cleared by hidden notes, and checked for "the notes file had the rule". A habitual root `NOTES.md` still counts for "started knowing the rule", which looks for the ID in any file. |

`reworded` also paraphrases the system prompt (a relay setting). Contributor counts need no new setting: repos of 3
or 5 problems and the new relay configs `relay_n3` / `relay_n5`; the episode limits scale with the count, the
per-contributor budget does not. The prompt never states the count (as in training).

Left the same everywhere: the vocabulary ("CI", "extended checks", "base tests", "score"), the bash tool, the
`end_session` tool, the `tasks/<6 letters>/` layout and the problem statements.

## Policies

| Policy | Adapter on HF `sidbaines/amber-baton` | Note |
|---|---|---|
| `27b_base` | none (untrained Qwen3.8-27B) | the 27B baseline |
| `27b_team_s29` | `exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s29-d81d6d` | team arm after step 29 |
| `27b_team_s59` | `exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s59-6bf5e2` | team arm after step 59 (final) |
| `27b_indiv_s29` | `exp2/train_individual_checks/adapters/sacrifice-relay-exp2-individual-policy-s29-368929` | individual arm after step 29 |
| `27b_exp21_s29` | `exp2/train_opener_checks/adapters/sacrifice-relay-exp2.1-opener-policy-s29-c9483c` | experiment 2.1 after step 29 |
| `a3b_base` | none (untrained Qwen3.6-35B-A3B) | the A3B baseline |
| `a3b_team_s79` | `exp2/train_team_checks_a3b/adapters/sacrifice-relay-exp2-team-a3b-policy-s79-6d9c30` | A3B team arm after step 79 (final) |
| `a3b_exp21` | `exp2/train_opener_checks_a3b/adapters/sacrifice-relay-exp2.1-opener-a3b-policy-s79-*` | A3B experiment 2.1 after step 79; training since 2026-10-05 09:30 UTC (about 14 h); `EXP21_A3B_STEP` picks another step |

- Each adapter is loaded into the eval server under the policy's name (`run.sh adapters`); every contributor seat is
  the evaluated policy.
- **Experiment 2.1 is seated in all four seats too**, although its contributor 1 was a frozen copy of the team policy
  during training. Three reasons:
  - the question is what this policy does; a frozen opener would put the team policy's reviewing into every game;
  - the prompt has no position, so seat 1 is only "nobody has left a note yet", which 2.1's followers met whenever the
    opener did not review (about half the time); whether they review then is exactly what 2.1 tested;
  - its followers' skill (using the rule once someone found it) is still measured, in the games where its own first
    contributor reviews.
  - A frozen-opener variant would take a small config (`protocol_config.opener_role`, as `train_opener.yaml`); it is
    not in the grid.

## Readouts (`analyze.py`)

Per cell, with n and a 95% interval:
- the training dashboard's measures by position, out of all contributors there: reached CI, chose review, chose
  submit, scored, started knowing the rule; and the team score;
- check.py's conditional rates: contributor 1 reviewed (of its CI runs); reviewed without the rule (of CI runs);
  redundant reviews (knew the rule, reviewed anyway); the next contributor knew the rule after a review; followers
  followed the rule when the notes file had it; followed it, and scored, when they started knowing it;
- habit checks (the most direct test of format-matching): calls to tools that do not exist in that condition (e.g.
  `ci_review` after the rename); bash commands that mention `NOTES.md` after the notes moved; submissions that contain
  the rule's ID in the wrong form.

Comparisons, with 95% intervals from a bootstrap over repos:
- **paired lift**: each trained policy minus the untrained model of the same family, same condition, same games;
  plus McNemar tests for "anyone scored" and "contributor 1 reviewed", matched by game;
- **transfer change**: each condition minus its reference for the same policy (paired by repo where the problem
  groups are the same). The size of a trained policy's lift in a condition, against its lift on `train_repos` and
  `heldout`, is the answer to the question.

Note for `n3` / `n5`: the best team score is (n − 1)/n (0.67 and 0.80; 0.75 for 4), so compare those cells through
the rates and the lift over the untrained model rather than the raw team score.

## Grid

All cells use 2 playthroughs per repo, so about 100 games and 400 contributor decisions per cell. Time per cell is
estimated from measured numbers: a game generates about 36k tokens (27B; 34k A3B; from the training rollouts), and two
GPUs with one 27B engine each, MTP and about 32 games per engine generate about 2,250 tokens/s through an adapter
(interpolated from `../bench/README.md`: 900 tokens/s per engine at 16 agents, 1,600 at ~56; LoRA costs about 15%). That is **about 0.45 h per 27B cell**. The A3B sampled 2.4× more games
per hour than the 27B in training; **about 0.23 h per A3B cell** assumes 2×.

| | Cells | Policies × conditions | Sampling | Total with fixed costs* | Cost at $9.18/h |
|---|---|---|---|---|---|
| **minimal** (`GRID=minimal`) | 12 | {27b_base, 27b_team_s59, a3b_base, a3b_team_s79} × {heldout, new_rules, far}, first 24 repos | 27B 1.3 h, A3B 0.65 h | **~5.5 h** | **~$50** |
| **core** (`GRID=core`, recommended) | 32 | all 8 policies × {train_repos, heldout, new_rules, far} | 27B 9 h, A3B 2.8 h | **~15 h** | **~$140** |
| **full** (`GRID=full`) | 56 | core + {base, final team} of each model × {reworded, tools, notes, replies, n3, n5} | 27B 14.4 h, A3B 5.5 h | **~23.5 h** | **~$215** |

\* Fixed costs, about 3.5 h: pod setup and the 27B download (~0.75 h, as in experiment 2), the filter (~1.7 h),
adapters (~0.2 h), switching the server to the A3B (~0.5 h), report and sync (~0.3 h).

- **Recommendation:** run `core` first; it includes `far`. If a trained policy keeps most of its lift under `far`, the
  behaviour transferred and the single-change cells are optional. If it loses it, the `full` cells (about $75 more)
  show which change breaks it. `minimal` first and then `core` is also possible: the same cells resume.
- Without `a3b_exp21` (not trained yet) core is 4 cells fewer (about 0.9 h, $8 less).
- Running the two models on two pods side by side costs the same and halves the wall time.
- The estimates are good to perhaps ±30%. After the first cell, check its time (`progress.json`, the log) and re-plan.

## Commands

On the dev box (done 2026-10-05 for this checkout; rerun only if `out/candidates` is lost):

```bash
E=experiments/2026-09-25_sacrifice-relay/exp2_eval
$E/run.sh candidates   # EXCLUDE=<experiment 1's out/candidates/taskset.json> if not at ../out/candidates
```

Copy to the pod: `../out/repos_n4/` (the training repos) and `$E/out/candidates/`. Then, on the pod, from the
checkout root (install: `docs/runbooks/pod.md`):

```bash
E=experiments/2026-09-25_sacrifice-relay/exp2_eval
$E/run.sh serve                     # 27B, one engine per GPU + MTP, 90% of each GPU
$E/run.sh filter                    # ~1.7 h -> out/filter_rollout, out/pool
$E/run.sh repos                     # out/repos_n4, repos_n4_new, repos_n3, repos_n5
$E/run.sh adapters                  # 27B adapters from HF, loaded and checked with a few requests
GRID=core $E/run.sh grid            # 20 cells; out/eval/<condition>/<policy>; rerun to resume
$E/run.sh stop
MODEL=a3b $E/run.sh serve
MODEL=a3b $E/run.sh adapters
MODEL=a3b GRID=core $E/run.sh grid  # 12 cells (8 until a3b_exp21 exists)
MODEL=a3b $E/run.sh stop
$E/run.sh report                    # out/report.txt and out/report.json
$E/run.sh extract                   # out/grades: numbers only, for the dev box
```

One cell: `$E/run.sh eval 27b_team_s59 far` (extra `key=value` arguments go to `marli eval rollout`). `PARALLEL`
(default 4) cells run at a time with `INFLIGHT` (default 64) games in flight in total: about 32 per engine, within
the 30–40 relay agents an engine handles (the variant test overloaded the server at about 96 per engine).

If loading adapters into the data-parallel server fails its check, serve tensor-parallel instead, the layout every
training run used with adapters (lower throughput):
`SERVE=experiments/2026-09-25_sacrifice-relay/bench/serve/tp2_mtp2_sleep.yaml $E/run.sh serve gpu_memory_utilization=0.9 enable_sleep_mode=false`
(A3B: `SERVE=experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/configs/serve_a3b.yaml`, same overrides).

Persistence: `out/grades` and the reports are small and contain no problem text; sync them to the dev box. The full
rollouts (`out/eval`) contain problem text (the training repos include LiveCodeBench problems): keep them private
(dev box or pod only, never git or HF); they are about 1.3 MB per game, so about 4 GB for core and 7 GB for full.

## Backward compatibility (checked 2026-10-05)

- Every new setting defaults to the trained surface. Tests pin values recorded on the commit before these settings
  existed (m6-harness 2476da3): `tests/test_envs_code_rules_transfer.py` the first message of every seat and every
  tool spec for the experiment 1 and experiment 2 settings and all prompt variants, and the training rules, seed for
  seed; `tests/test_sacrifice_relay_configs.py` the config hashes of the gate, the filter, all five experiment 2
  training configs and the repo and candidate builds.
- The recorded run directories still match their config hashes: train_team_checks, train_individual_checks,
  train_opener_checks, train_team_checks_a3b, variants/checks, filter_rollout, repos_n4, candidates (recomputed from
  each run's saved config with this branch).
- The first messages of 704 recorded training contexts (27B team steps 0, 29, 59; individual and 2.1 steps 0 and 29;
  A3B team steps 0 and 79), rebuilt with this branch from each game's seed, equal the text decoded from the token ids
  the models actually saw; their tool names and descriptions are all present in the decoded system prompts.
- A CPU dry run of all 10 conditions through `marli eval rollout` with scripted contributors (3 synthetic repos each)
  ran every cell; one that reads the prompt reached the best team score in every condition, one that keeps training's
  names failed under `tools` and `far` and lost the notes file under `notes`, as the habit checks reported.

## Files

- `configs/candidates.yaml`: the held-out candidate build.
- `configs/eval.yaml`: shared eval settings (2 playthroughs, seed 0), layered after experiment 2's training setup.
- `configs/conditions/*.yaml`: one per condition.
- `configs/serve_eval_a3b.yaml`: the A3B eval server (one engine per GPU, MTP, the A3B arm's LoRA modules).
- `run.sh`, `analyze.py`, `overlap.py`.
- Library: `code_rules` settings and rule forms (`src/marli/envs/code_rules.py`), `relay_n3` / `relay_n5`,
  the `deepcoder_no_lcb` task source.
