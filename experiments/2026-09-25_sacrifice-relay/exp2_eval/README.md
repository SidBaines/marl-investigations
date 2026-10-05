# Transfer eval: did experiment 2's policies learn the behaviour, or the format?

Status (2026-10-05): **prepared, nothing run on a GPU.** The held-out candidate problems are built on the dev box
(`out/candidates`, not committed): 900 problems that pass a strict independence check against everything the study
used (`results/independence_*.txt`). Everything else runs on one 2×H200 pod; see Grid for time and cost.

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
| `heldout` | 51 held-out repos | new problems, independent of every used one | **layer 1**: memorised repos vs a general skill |
| `new_rules` | the same held-out repos | the house rule has a form training never used | **layer 2**: reading the rule vs pattern-matching the 10 trained forms |
| `reworded` | held-out | first message and system prompt paraphrased | **layer 3**: the exact instruction text |
| `tools` | held-out | CI tools renamed `grade_solution` / `inspect_checks`, descriptions reworded | layer 3: tool names and descriptions |
| `notes` | held-out | the shared notes file is `docs/handoff.txt`, not `NOTES.md` | layer 3: the notes file name and place |
| `replies` | held-out | CI's reports reworded | layer 3: the wording of CI's replies |
| `n3`, `n5` | held-out problems, 3 or 5 per repo | 3 or 5 contributors instead of 4 | layer 3: the length of the relay |
| `far` | held-out repos, held-out rules | every layer-3 change at once (4 contributors) | the whole package |

Layer 3 is measured against `heldout` (same repos, so paired); `heldout` against `train_repos`.

### Layer 1: new repos, same format

The held-out problems must share no problem with anything experiments 1 and 2 used, including reworded copies.
"Used" means all 800 of experiment 1's candidates, not only the 207 trained problems (stricter than needed).
`run.sh candidates` (dev box, CPU, ~25 min) does four steps:

1. **Draw** (`configs/draw.yaml`): 1,600 problems from `deepcoder_no_lcb`, i.e. DeepCoder train without its
   LiveCodeBench subset, shuffled with seed 0 like experiment 1's candidates. Problems whose normalised text equals
   a used one are left out.
2. **Check** (`overlap.py check`; checks in `src/marli/data/overlap.py`). Every drawn problem is compared, in draw
   order, with the 800 used problems and with the drawn problems before it, so of two copies the first is kept.
   A problem is dropped if any check fires. Report: `results/independence_check.txt`.
3. **Build** (`configs/candidates.yaml`): the first 900 drawn problems that passed every check (`out/candidates`).
4. **Verify** (`overlap.py verify`): all checks again on those 900. Result: **none flagged**
   (`results/independence_verify.txt`).

| Check (in this order) | Rule | Removed beyond the checks above it | Caught by no other check |
|---|---|---|---|
| Too little to compare | under 20 words once boilerplate is removed, or under 80% Latin letters (statement-less "Example only" prompts, 3 Russian statements) | 10 | 3 |
| Text | Jaccard ≥ 0.05, or containment ≥ 0.1, on rare 5-word runs of the statement without boilerplate | 208 | 19 |
| Text, wrapper included | Jaccard ≥ 0.3 on 5-word runs of the whole prompt (the first build's measure) | 7 | 6 |
| Tests | complete suites from the raw dataset (public and hidden): one suite inside the other, or a shared pair that is rare (≤ 3 of 21,560 problems) and ≥ 16 characters, or 3+ rare pairs | 4 | 2 |
| Source ids | the same TACO URL or CodeContests problem in PrimeIntellect's upstream data | 0 | 0 |
| Meaning, bge-small-en-v1.5 | cosine ≥ 0.88 | 47 | 31 |
| Meaning, all-MiniLM-L6-v2 | cosine ≥ 0.75 | 19 | 19 |
| **Any** | | **295 of 1,600** (179 against used problems, 116 only as copies of an earlier drawn problem) | |

All 1,305 problems that passed were kept in draw order; the first 900 are the candidates.

**Thresholds and why** (calibrated by reading flagged pairs privately on the dev box; nothing is committed or
reported but counts and scores):
- **Boilerplate.** A line found in at least 50 of the 21,560 DeepCoder problems (408 lines) is removed before any
  text comparison, e.g. PrimeIntellect's fixed prompt wrapper, section headers, sample-data lines. Stock phrases
  inside lines (e.g. PrimeIntellect's function-call sentence) are handled by comparing only rare 5-word runs (found
  in at most 20 problems).
- **Text: 0.05.** With boilerplate removed the scores split cleanly:
  - 1,484 drawn problems score below 0.05 and none score 0.2–0.3;
  - pairs at 0.3 or more are copies;
  - 0.1–0.3: sibling problems (same setting or course template, different question; e.g. an easy and a hard
    version);
  - 0.05–0.1: problems sharing a stock definition (the regular-bracket-sequence definition, the Collatz rule);
  - below 0.05, pairs share only a contest template ("Process Q queries", "perform a sequence of the following
    operations").

  0.05 removes copies, siblings and shared definitions, 6× below the lowest copy score. Containment (shared /
  smaller set) ≥ 0.1 additionally catches a copy inside a longer statement.
- **Text, wrapper included: 0.3.** On the 7 problems between 0.3 and 0.5 (the first build's measure, kept by its
  30-word rule), the reading showed only very short statements that share the wrapper and nothing else (e.g.
  counting the divisors of N! against counting xor pairs). They are dropped anyway, as asked: short statements are
  where the text check has least to go on. 6 more problems.
- **Tests.** Copies share most of their suite; unrelated problems share only generic pairs such as input 1,
  output 1. A shared pair counts only if it is rare (in at most 3 of all 21,560 problems) and at least 16
  characters long. Every rare shared pair read between unrelated problems was 12 characters or shorter (e.g. an
  input of four small numbers with output 2). A whole suite contained in the other's (one statement-less copy:
  101 of 101 pairs) or 3+ rare pairs also count. Public tests: DeepCoder lists none for PrimeIntellect and TACO
  rows (their examples sit in the statement, which the text checks compare).
- **Meaning.** Each statement without boilerplate is embedded (bge: first 512 tokens; MiniLM: first 256) and
  compared by cosine. Read pairs:
  - **bge.** The 51 known copies score at least 0.959. Problems not flagged by text reach 0.91 at most, apart from
    2 statement-less ones at 1.0. Every pair read between 0.84 and 0.91 was a different problem on a similar topic
    (palindromes, permutations, bracket sequences). Threshold 0.88: below the empty 0.92–0.95 band, 0.08 under the
    weakest copy, and it also drops close topical neighbours.
  - **MiniLM.** Scores spread wider (median pair 0.22). Copies score at least 0.774; pairs read between 0.75 and
    0.83 were again topical neighbours (shared characters, two text-editor problems). Threshold 0.75.
- **Source ids.** DeepCoder rows carry no source metadata. PrimeIntellect's upstream dataset does, read by range
  requests with nothing stored: 1,344 of the 2,400 compared problems match an upstream prompt exactly and 883 by
  statement; 173 have no id.
  - Usable ids are TACO's problem URLs and CodeContests names.
  - APPS indices are not ids (APPS's train and test splits both count from 0); all 7 matches they produced were
    unrelated problems on reading.
  - Codeforces rows carry only a contest number.
  - All 91 id matches were caught by another check too (89 by text), a useful cross-check.
- **Upstream rewordings.** PrimeIntellect usually holds about 4 differently worded prompts per source problem, so
  DeepCoder contains rewordings of one problem. That explains the 116 drawn problems that only repeat an earlier
  drawn one.
  - The same holds for the used set: experiment 1's 800 candidates contain 21 copies of each other, and the 207
    trained problems 2.
  - No training repo contains two copies of one problem.

**The 51 known reformatted copies** (found by the first build among the first 800 drawn) are all caught:
- text: 51 (lowest Jaccard 0.558);
- tests: 49;
- source ids: 27;
- bge: 51 (lowest 0.959);
- MiniLM: 51 (lowest 0.774).

For all 51, both models' nearest used problem is the known partner.

**What the checks cannot rule out:**
- **Copies in another language.** 2 used and 3 drawn statements are not in English. The text and meaning checks
  cannot match across languages; only the test check could. The 3 drawn ones are dropped.
- **The same problem with new tests and a wholly new statement.** Only the meaning check would see it, from the
  start of the statement (512 or 256 tokens).
- **Text inside images** (`<image>` placeholders) is invisible to every check.
- **Two different problems built on one idea.** The checks look for repeated problems, not repeated solution ideas.
- **Rows without a source id.** 173 of the compared problems have no PrimeIntellect id (TACO rows with no
  PrimeIntellect twin, and the 20 LiveCodeBench rows among the used).

**Difficulty filter and repos** (pod):
- **Filter** (`run.sh filter`): experiment 1's filter exactly: the untrained 27B, one agent, plain coding task
  (no house rule), 4 attempts per problem, keep problems solved 1–3 times (`../configs/eval_filter.yaml`,
  `data filter lo=0.25 hi=0.75 inclusive=true`). Experiment 1 kept 207 of 800 (26%).
- **Expected yield:** about 233 of the 900, and at least 204 (51 repos of 4) with about 99% probability.
  `run.sh repos` uses the first 204 in its shuffle, so every cell has **51 repos of 4**, as training had: about 100
  games per cell, enough to see the large trained effects (contributor 1 reviews 17% → 80%; team score
  0.09 → 0.27) and drops of about a third of them. The same 204 problems give 68 repos of 3, and 200 of them 40
  repos of 5.
- **Filter time:** 3,600 attempts, about 1.9 h. Experiment 1's 3,200 took 1.7 h of sampling on two GPUs (one
  engine each), measured from its call timestamps (`../out/filter_rollout`).
- **Repos** (`run.sh repos`, all `data repos seed=0`): `repos_n4` (training rule forms), `repos_n4_new` (the same
  groups of 4, held-out rule forms), `repos_n3`, `repos_n5`.

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
| **minimal** (`GRID=minimal`) | 12 | {27b_base, 27b_team_s59, a3b_base, a3b_team_s79} × {heldout, new_rules, far}, first 24 repos | 27B 1.3 h, A3B 0.65 h | **~5.7 h** | **~$52** |
| **core** (`GRID=core`, recommended) | 32 | all 8 policies × {train_repos, heldout, new_rules, far} | 27B 9 h, A3B 2.8 h | **~15.5 h** | **~$142** |
| **full** (`GRID=full`) | 56 | core + {base, final team} of each model × {reworded, tools, notes, replies, n3, n5} | 27B 14.4 h, A3B 5.5 h | **~23.6 h** | **~$217** |

\* Fixed costs, about 3.7 h: pod setup and the 27B download (~0.75 h, as in experiment 2), the filter (~1.9 h for
900 problems), adapters (~0.2 h), switching the server to the A3B (~0.5 h), report and sync (~0.3 h).

- **Recommendation:** run `core` first; it includes `far`. If a trained policy keeps most of its lift under `far`, the
  behaviour transferred and the single-change cells are optional. If it loses it, the `full` cells (about $75 more)
  show which change breaks it. `minimal` first and then `core` is also possible: the same cells resume.
- Without `a3b_exp21` (not trained yet) core is 4 cells fewer (about 0.9 h, $8 less).
- Running the two models on two pods side by side costs the same and halves the wall time.
- The estimates are good to perhaps ±30%. After the first cell, check its time (`progress.json`, the log) and re-plan.

## Commands

On the dev box (done 2026-10-05 for this checkout; rerun only if `out/candidates` is lost; deterministic):

```bash
E=experiments/2026-09-25_sacrifice-relay/exp2_eval
$E/run.sh candidates   # draw, check, build, verify; ~25 min on CPU
# EXCLUDE=<experiment 1's out/candidates/taskset.json> if not at ../out/candidates;
# EMBED_PY=<a python with torch + transformers> (default: the dev box's CPU venv); models go to /tmp/marli-embed
```

Copy to the pod: `../out/repos_n4/` (the training repos) and `$E/out/candidates/`. Then, on the pod, from the
checkout root (install: `docs/runbooks/pod.md`):

```bash
E=experiments/2026-09-25_sacrifice-relay/exp2_eval
$E/run.sh serve                     # 27B, one engine per GPU + MTP, 90% of each GPU
$E/run.sh filter                    # ~1.9 h (900 problems x 4) -> out/filter_rollout, out/pool
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

- `configs/draw.yaml`, `configs/candidates.yaml`: the held-out draw and the final candidate build.
- `results/independence_check.txt`, `results/independence_verify.txt`: the independence reports (numbers only).
- `configs/eval.yaml`: shared eval settings (2 playthroughs, seed 0), layered after experiment 2's training setup.
- `configs/conditions/*.yaml`: one per condition.
- `configs/serve_eval_a3b.yaml`: the A3B eval server (one engine per GPU, MTP, the A3B arm's LoRA modules).
- `run.sh`, `analyze.py`, `overlap.py` (the independence check; its rules are in `src/marli/data/overlap.py`).
- Library: `code_rules` settings and rule forms (`src/marli/envs/code_rules.py`), `relay_n3` / `relay_n5`,
  the `deepcoder_no_lcb` task source.
