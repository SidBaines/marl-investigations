---
type: source
title: "Sacrifice relay: report on experiments 1–3 and the evals (2026-09-25 to 2026-10-05)"
description: "Study-wide report, all training one seed: team reward taught the first contributor to reveal the hidden rule when doing so cost it nothing and paid the team (experiment 2: Qwen3.8-27B 17% → 80% over 60 steps, Qwen3.6-35B-A3B 38% → 77% over 80), but drove reviewing to about 0% when the review cost a point and informed followers rarely applied the rule (experiment 3, A3B); individual reward never taught it; the learned behaviour transferred to held-out problems and a fully changed format, but not to standard cooperation evals (only HiddenBench moved, A3B only)."
resource: https://github.com/SidBaines/marl-investigations/blob/evals-session-1005/experiments/2026-09-25_sacrifice-relay/REPORT.md
source_date: 2026-10-05
status: partial
provenance: "Copied verbatim on 2026-10-05 from branch evals-session-1005 (corrected version @ 1298acf; first copied @ 3d6551f), experiments/2026-09-25_sacrifice-relay/REPORT.md (draft @ 53bbad7, final @ 3d6551f, both 2026-10-05). evals-session-1005 merges m6-harness (training, experiments 1–3; merged @ 5b73c70), exp2-eval (transfer eval, held-out problems, in-relay help request; merged @ 9d0cefc) and evals-coop (external-eval layer and the four standard suites; merged @ 03c3b20), plus the allow_paused filter fix (73c023b, ce65a2a). No PR; none of these branches is merged to main at ingest. Training commits named in the per-experiment READMEs or the log: experiment 2 configs 9cb3c08 and prompt variants d637752 (2026-10-01); team-arm results bc0a658 and continuation to 60 steps ab7edb0, individual arm and 2.1 results 8ded493 (2026-10-02); A3B local support 5ea9b20 and 64dcf29, hot-load tolerance 4c34b0f (2026-10-02); A3B team arm results 7a1a8a0 (2026-10-03); A3B 2.1 resume commit 2476da3 and interim results 517ba1c (2026-10-05); experiment 3 resume commit a4858f3 and interim results f583dc2 (2026-10-05). Eval code: transfer eval 1f4510f, independence check 2cc9a87, help settings 8ca24d0 and 31bf33d (exp2-eval); external layer 8f9f692 and b54635f, planted help 0da14bb (evals-coop). Supporting READMEs (in git at 3d6551f): exp2_mandatory_rule/README.md, exp3_sacrifice_a3b/README.md, exp2_eval/README.md, ../2026-10-05_coop-evals/README.md. Aggregate tables in git: exp2_eval/results/ and ../2026-10-05_coop-evals/results/ (numbers only). Per-run analysis tables (out/analysis/) are not in git; per the report they are on HF sidbaines/amber-baton (public) under exp2/analysis/ and exp3/analysis/. Games, transcripts and eval outputs contain DeepCoder (including LiveCodeBench) problem text and stay private on the dev box. Status is partial: every training run is one seed; there are several arms and two models, and the eval comparisons carry 95% intervals and paired tests."
tags: [experiment, training, rl, evaluation, local-backend, qwen3.8-27b, qwen3.6-35b-a3b, moe, deepcoder, code-rules, relay, team-reward, individual-reward, sacrifice, credit-assignment, information-sharing, transfer, cooperation, hiddenbench, inspect]
timestamp: 2026-10-05
---

# Sacrifice relay: report on experiments 1–3 and the evals (2026-09-25 to 2026-10-05)

**Status: final (2026-10-05, 21:10 UTC).** Training runs finished or paused (resumable); evals done; every pod deleted.

This is the study-wide write-up: what we ran, what we found, and everything we learned about running it, so later
experiments can reuse it. The per-experiment READMEs have the full tables and logs. This report summarises them and
links to them.
- `README.md`: experiment 1.
- `exp2_mandatory_rule/README.md`: experiment 2, 2.1 and the A3B arms.
- `exp3_sacrifice_a3b/README.md`: experiment 3.
- `exp2_eval/README.md`: the transfer eval.
- `../2026-10-05_coop-evals/README.md`: the standard cooperation evals.

## 1. Summary

1. **Team reward taught the first contributor to spend its CI run finding the rule for the others, when that cost it
   nothing. A true sacrifice, one that costs the reviewer its own expected point, has not been learned.** Training
   followed the payoff a review actually delivered (one seed per run).
   - **Experiment 2: learned.** Following the hidden rule was required to score, and the prompt said the checks cannot
     be guessed. So a contributor without the rule scores 0 whether it reviews or submits: reviewing gives up the CI
     run, not a point (experiment 2's README). Team training on Qwen3.8-27B took contributor 1's review rate from 17%
     to 80% over 60 steps, and the team score tripled (0.089 → 0.267).
     - Qwen3.6-35B-A3B learned the same over 80 steps: review 38% → 77%, team score 0.049 → 0.268.
     - The A3B also learned *when* to review: reviews by contributors who already knew the rule fell from 54% to 21%.
   - **Experiment 3: the opposite.** The A3B was given experiment 1's scoring, where passing the tests alone earns a
     point, so a review costs the reviewer that point. Both team and individual reward drove reviewing to about 0%
     within 30 steps.
     - The payoff check shows why: informed followers mostly did not apply the rule (only 14% of them scored 3), so a
       review cost the team more than it gained, and training's signal pointed against reviewing.
   - **Experiment 1** (27B, the same 0/1/3 scoring) is the one case where the costly review did pay off (+0.31 team
     score), yet the model did not learn it in 30 steps.
   - **Confounds:** experiment 2 changed the scoring, the prompt sentence, the position and the folder names at once
     relative to experiment 1, so which change mattered is not isolated. Only experiment 3 vs experiment 2's A3B team
     arm isolates the scoring.
2. **Individual reward did not teach reviewing.**
   - Experiment 2's individual arm was flat over 30 steps.
   - Experiment 2.1 used a frozen, trained first contributor, with the followers trained on individual reward.
     Followers learned to *use* the rule:
     - 27B: followed it 50% → 62%;
     - A3B: followed it 40% → 65%, and stopped wasting reviews.
   - But followers without the rule never learned to review. On the A3B they stopped reviewing altogether
     (35% → 6%).
3. **What was learned transfers.**
   - **New problems:** on 25 repos built from problems never used before, both team-trained models score at least as
     well as at the end of training (27B 0.085 → 0.300; A3B 0.060 → 0.320, untrained → trained). Contributor 1 reviews
     93% / 76% of the time.
   - **A fully changed format** (reworded prompts, renamed tools, moved notes file, new rule kinds):
     - the 27B keeps almost all of its gain (0.115 → 0.275);
     - the A3B keeps reviewing but its followers apply the rule much less (0.050 → 0.140, about a third of its gain),
       probably because of the new rule kinds (the single-change conditions that would show this were not run).
   - **No sign of format-matching habits.** No one called a renamed tool, used the old notes file or wrote the rule in
     the old form.
   - **No memorisation during training either.** The coding pass rate on repeated problems stayed flat, notes were
     never templated, and reviews never became an instant reflex.
4. **But it did not make the models more cooperative in general.**
   - **One-shot giving games and the volunteer's dilemma:** no change for either model.
   - **Help requests from another agent:** never acted on in a standard harness (0 of 480 episodes across both
     models and versions), and team training changed nothing.
   - **One lead, not a finding: HiddenBench** (pooling private facts in a discussion). The team-trained A3B was right
     more often after discussion, 17.8% → 22.7% (+4.8 points, p = 0.025 over 3 discussions per task). But it is the
     one positive result among many tests on two models, the extra discussions were run because the first looked
     promising, the replication alone gave p = 0.077, and the group's majority vote moved less (p = 0.19). The 27B
     showed no effect (19.1% vs 19.2%).
5. **Helping another agent depends strongly on the setting.** The same 27B appended a requested line for another
   agent in 36–38% of games inside our relay, but never in a standard Inspect agent. The two settings differ in several
   ways at once: where the note sits (the shared `NOTES.md` the relay prompt tells it to read and write), how shared
   files are framed, task difficulty and length, how the result is reported, and sampling settings. So which
   difference matters is not isolated.

## 2. The game and what changed between experiments

**The game** (`code_rules` environment, relay protocol, `src/marli/envs/code_rules.py`):
- **Setup:** four contributors, all the same LLM, take turns in one shared folder, each with a fresh context. Each has
  its own coding problem (DeepCoder train). The repo has a hidden "house rule" with a random ID (e.g. first line
  `# release: ABC-1234`).
- **One CI run each, in one of two modes:**
  - `ci_submit`: scored, and reports only PASS/FAIL for the extended checks;
  - `ci_review`: reveals the rule, scores 0. This is "the sacrifice".
- **Sharing:** a shared `NOTES.md` passes knowledge on.
- **Training:**
  - harness: our own token-level relay harness (not Claude Code); the tools are `bash`, `ci_submit`, `ci_review`
    and `end_session`;
  - method: GRPO-style, with a role leave-one-out baseline, a token-sum loss and LoRA rank 32;
  - optimiser: AdamW, flat lr 4e-5;
  - hardware: the local backend (vLLM + PEFT learner) on one 2×H200 pod per run.

| | Exp 1 (09-25) | Exp 2 / 2.1 (10-01/02) | Exp 2 A3B / 2.1 A3B (10-02/05) | Exp 3 (10-05) |
|---|---|---|---|---|
| Model | Qwen3.8-27B | Qwen3.8-27B | Qwen3.6-35B-A3B | Qwen3.6-35B-A3B |
| Scoring | 0 / 1 (tests pass) / 3 (tests + rule) | 0 / 1 only with tests + rule | as exp 2 | as exp 1 |
| Position in the prompt | "contributor k of 4" | not stated | not stated | not stated |
| Task folders | numbered | random names | random names | random names |
| "checks cannot be worked out" sentence | no | yes | yes | yes |
| Batch per step | 4 repos × 4 | 4 × 4 | 8 × 4 | 8 × 4 |
| LoRA targets (rank 32) | attention, linear attention, MLP | same | attention, linear attention, shared expert (routed experts and router frozen) | same as A3B |
| Reward arms | team | team; individual; 2.1 = frozen first contributor + individual | team (80 steps); 2.1 (paused at 31) | team and individual (paused at 31) |

Every run used the same 51 repos (204 DeepCoder problems filtered to the untrained 27B's pass@4 in [¼, ¾]). Each
repo was drawn about 2.4 times per 30 steps at 4 repos per step, or about 4.8 times at 8 repos per step. Every game
draws a fresh rule. The prompt text is otherwise identical in every game.

## 3. Training results

All runs are one seed. Numbers come from `check.py`, `analyze.py`, `followers.py` and `payoff.py`; full tables are in
each experiment's `out/analysis/<run>/` and on HF (`sidbaines/amber-baton`, `exp2/analysis/`, `exp3/analysis/`).

### 3.1 Experiment 1: 27B, 0/1/3 scoring, team reward, 30 steps (`README.md`)
- **Team score:** 0.545 → 0.631 (z = 2.3), mostly from reaching CI more often (61% → 70%).
- **Sacrifice rate** (no rule at start, ran CI): 10.6% → 7.2% (z = −1.7). Contributor 1: 3% → 2%.
- **Payoff check:** sacrificing paid off for the team (+0.31 team score; contributor 1 +0.62), and the advantage
  favoured it. The policy still did not learn it. Reviews were reactive: 90 of 93 came after an earlier note about
  failing checks.

### 3.2 Experiment 2, team arm: 27B, 0/1 scoring, 60 steps (`exp2_mandatory_rule/README.md`)

| Steps | 0–9 | 20–29 | 50–59 |
|---|---|---|---|
| Team score | 0.089 | 0.177 | 0.267 |
| Contributor 1 reviewed (of CI runs) | 17% | 52% | 80% |
| Started knowing the rule | 19% | 38% | 59% |
| Redundant reviews (knew the rule, reviewed anyway) | 18% | 12% | 24% |

- The first mover learned to review. The run had not levelled off at step 59.
- **Late overshoot:** redundant reviews rose again to 24%, so reviewing started to become a habit.
- **The `checks` sentence was decisive at the start.** Without it, the untrained model almost never reviewed
  (2 of 82 CI runs). Its reasoning was that a submit "has a chance".

### 3.3 Experiment 2, individual arm: 27B, 30 steps
- **Flat:** team score 0.078 → 0.081; contributor 1 reviewed 23% → 30%.
- **Why:** under individual reward with 0/1 scoring, neither reviewing nor submitting without the rule ever pays the
  reviewer.

### 3.4 Experiment 2.1: frozen trained opener, individual reward for contributors 2–4, 27B, 30 steps
- **Control** (knew the rule): followed it 50% → 62%, scored 40% → 47%, reviewed anyway 7% → 5%.
- **Test** (did not know it): did not learn to review. Rates stayed at 34% after an earlier CI run, and 10–18%
  otherwise.

### 3.5 Experiment 2, team arm on Qwen3.6-35B-A3B: 80 steps

| Steps | 0–9 | 20–29 | 40–49 | 70–79 |
|---|---|---|---|---|
| Team score | 0.049 | 0.088 | 0.183 | 0.268 |
| Contributor 1 reviewed | 38% | 61% | 73% | 77% |
| Redundant reviews | 54% | 55% | 37% | 21% |

- It learned the same behaviour as the 27B, and also learned when not to review.
- **Per game seen it was slower:** 0.267 after about 960 games for the 27B, against about 2,560 for the A3B. This is
  not a matched comparison (twice the batch, and an adapter without the routed experts).

### 3.6 Experiment 2.1 on the A3B: paused after step 30

| Steps | 0–9 | 20–29 |
|---|---|---|
| Team score | 0.190 | 0.259 |
| Frozen opener reviewed | 87% | 84% |
| Control: followed the rule | 40% | 65% |
| Control: reviewed anyway | 16% | 2% |
| Test: reviewed (after an earlier CI run) | 35% | 6% |

- Followers with the rule learned to use it.
- Followers without it stopped reviewing. Under 0/1 scoring nothing rewards or punishes that choice directly, so this
  is likely spill-over from being punished for reviewing when the rule was already known.

### 3.7 Experiment 3: A3B, 0/1/3 scoring, team and individual, paused after step 30 (`exp3_sacrifice_a3b/README.md`)

| | Team 0–9 | Team 20–29 | Individual 0–9 | Individual 20–29 |
|---|---|---|---|---|
| Team score (0–3 scale) | 0.481 | 0.597 | 0.508 | 0.655 |
| Reviewed (no rule, ran CI) | 21% | 3% | 11% | 0% |
| Contributor 1 reviewed | 19% | 2% | 14% | 0% |
| Scored 3 | 3% | 0% | 1% | 0% |

- **Team-arm payoff check:** games with a review scored 0.12 lower than games on the same repo without one.
  - The rule reached the next contributor 96% of the time.
  - But informed followers scored 3 only 14% of the time (of 375 who started knowing the rule). Of the 178 who
    submitted, 83% passed the base tests, yet only about a third of those (about 52 of 147) also followed the rule;
    the rest took the 1 point without it.
  - The advantage was −0.08 for reviewing and +0.05 for submitting.
- **Reading:** for the A3B under this scoring, the bottleneck is followers applying the rule, not reviewers finding
  it. A hypothesis to check in transcripts: the A3B treats the rule as optional when passing alone earns a point.

### 3.8 Did training learn the behaviour or the format? (checked 2026-10-05, from the saved games)
- **What repeats:**
  - the 51 repos repeat, with each problem fixed to a slot;
  - the prompt, tool names and descriptions, CI wording and `NOTES.md` are identical in every game;
  - rules come in only 10 forms (3 families × 3–4 keys); only the ID is new.
- **No memorisation of problems:** among submitters, the base-test pass rate did not rise with repeats. 27B: 0.81 on
  a repo's first draw, 0.84 on its fifth. A3B: 0.79–0.89 with no trend over 12–13 draws.
- **Notes are not a template:** with IDs, keys and folders masked, the notes written in the last five steps are all
  worded differently (27B: 182 of 182; A3B: 393 of 394).
- **Reviews are not a reflex on the prompt:** the median number of tool calls before a review was 9 at the start and
  the end for the 27B, and 13–14 for the A3B.
- **What this cannot show:** whether the behaviour transfers to a different format. That is what the transfer eval
  is for.

## 4. Evals (2026-10-05)

### 4.1 What we built
- **Transfer eval** (`exp2_eval/`, branch `exp2-eval`), run in our relay game:
  - *Layer 1: new problems.* Held-out DeepCoder problems independent of every problem used before (see 4.3).
  - *Layer 2: new kinds of rule:* a required last line, a function returning the ID, a class attribute.
  - *Layer 3: changed surface:*
    - reworded instructions and system prompt;
    - renamed CI tools (`grade_solution` / `inspect_checks`) and reworded descriptions;
    - reworded CI replies;
    - notes moved to `docs/handoff.txt`;
    - 3 or 5 contributors;
    - `far` combines everything.
  - Every setting defaults to the training behaviour. Tests pin byte-identical prompts and unchanged config hashes for
    every existing config (704 real training prompts reproduced exactly).
- **Planted help request** (in both harnesses, with identical texts):
  - **The note:** a note from "another agent" asks the contributor to append a line about that agent's run to
    `logs/agent_runs.log`, in two phrasings, placed in `NOTES.md` or `HELP_NEEDED.txt`.
  - **Measures:** whether it helped, whether it read the note, whether it told the user (four cells), and the cost to
    its own task.
- **External eval layer** (`src/marli/eval/external/`, verb `marli eval external`, branch `evals-coop`):
  - **Why a standard harness:** these evals use standard harnesses, not ours, so they test whether the behaviour
    generalises beyond our harness (Sid's call).
  - **Registry:** a file-backed list of suites, each pinned to an upstream commit or task version with only its own
    settings.
  - **Checks before sampling:** the server must actually serve every adapter.
  - **Report:** sample sizes, unparsed answers and errors counted separately, 95% intervals and the paired gain over
    the untrained model.
  - **Four suites:**
    - Li & Shirado's one-shot games (Inspect task, the paper's exact prompts);
    - HiddenBench (upstream code at `3be6ca1`, plus a small patch for per-seat models);
    - FAIRGAME's volunteer's dilemma (upstream runner at `fc302a6`);
    - the planted help request (Inspect's stock `react()` agent with bash and submit, in a locked-down sandbox:
      own uid, no network, Landlock, no secrets).
  - **Research:** `/workspace/marli-orchestration/2026-10-05/coop_evals_research.md` explains why these and not others.
    No standard Inspect eval tests cooperation; others need a judge, or reward selfish play.

### 4.2 The session (2026-10-05, 15:20–21:10 UTC)
- **Pods:** one 2×H200 pod per model family. A single data-parallel vLLM server per pod served both the untrained model
  and the team adapter (27B: step 59; A3B: step 79) for both harnesses.
- **Preflight passed on both servers:**
  - the chat template matches the training renderer token for token;
  - thinking is split off;
  - JSON mode keeps thinking;
  - raw completions are unaffected;
  - tool calls are parsed.
- **Run per model:**
  - relay: new problems, everything changed, and help request A in `NOTES.md`; 25 held-out repos × 2 games each;
  - the games;
  - HiddenBench (1, then 3 discussions per task);
  - the volunteer's dilemma (50 games);
  - the planted help request (30 tasks × 5 conditions).
- **How it ran:**
  - The 27B pod first ran the difficulty filter (15:31–17:06); its last step needed the `allow_paused` fix.
  - The A3B pod started at 16:00 and began its relay as soon as the repos arrived (17:17).
  - Late fixes for idle GPUs: more relay games in flight on the A3B (36 → 60); on the 27B, the next phases ran
    alongside one straggling HiddenBench discussion.
  - The planned hand-over of 27B work to the A3B pod was dropped: it would have saved about 15 minutes for about
    0.4 extra pod-hours.
  - A3B pod: 16:00–19:37 (about $33). 27B pod: 15:20–21:10 (about $54).

### 4.3 Building an independent held-out problem set
- **Reworded copies.** DeepCoder's subsets contain the same problem reworded:
  - 51 of 800 fresh candidates were near-copies of an earlier candidate, 11 of them of trained problems;
  - experiment 1's own 800 candidates held 21 duplicates of each other.
- **The final checks** (`src/marli/data/overlap.py`, `exp2_eval/overlap.py`) compare each candidate with all 800 used
  candidates and with the other new candidates:
  - text overlap with boilerplate stripped;
  - whole-prompt similarity;
  - overlapping test cases;
  - upstream source IDs;
  - two sentence-embedding models.
- **Result:** of 1,600 drawn problems, 295 were dropped and 900 kept. Both embedding models caught all 51 known
  copies.
- **Difficulty filter:** the untrained 27B, 4 attempts each, keeping problems solved in 1–3 of 4.
  - It covered the first 600 candidates only. Sid decided 25 repos were enough, so the run was capped with
    `stop_after_tasks`.
  - 156 kept (26%, the same rate as experiment 1); 53% too easy, 21% too hard.
  - 25 repos of 4 were built, in both the training rule forms and the new rule kinds.

### 4.4 Results

All comparisons are untrained vs team-trained of the same model (27B: experiment 2 team arm after step 59; A3B: after
step 79), on the same tasks and seeds where the eval allows pairing. Aggregate tables: `exp2_eval/results/` (relay) and
`../2026-10-05_coop-evals/results/` (standard evals).

**Relay transfer eval** (25 held-out repos × 2 games = 50 games per cell; team score = mean 0/1 score of the four
contributors; p from McNemar on "anyone scored", matched by game):

| | 27B untrained → team | A3B untrained → team |
|---|---|---|
| New problems, same format: team score | 0.085 → **0.300** (+0.215 [+0.125, +0.300]) | 0.060 → **0.320** (+0.26 [+0.19, +0.33]) |
| contributor 1 reviewed (of its CI runs) | 14% → 93% | 36% → 76% |
| Everything changed (`far`): team score | 0.115 → **0.275** (+0.16 [+0.095, +0.235]) | 0.050 → **0.140** (+0.09 [+0.01, +0.175]) |
| contributor 1 reviewed | 37% → 74% | 43% → 70% |
| Help note in NOTES.md: team score | 0.050 → 0.215 | 0.040 → 0.240 |
| p, anyone scored (new problems / far / help) | 3e-6 / 4e-5 / 3e-6 | 1e-7 / 0.017 / 4e-7 |

- **Transfers to new problems, fully.** Both trained models score as well as or better than at the end of training
  (27B 0.300 vs 0.267; A3B 0.320 vs 0.268).
- **Survives a fully changed surface, for the 27B almost entirely.**
  - The `far` condition rewords the instructions and system prompt, renames the CI tools and rewords their
    descriptions, rewords the CI replies, moves the notes to `docs/handoff.txt`, and uses rule kinds never seen in
    training.
  - The 27B team policy's `far` minus new-problems change is −0.025 [−0.11, +0.05].
  - The A3B's is −0.18 [−0.26, −0.10]. It still reviews (contributor 1 −8.5 points, not significant), but its
    followers apply the rule much less (−32 points), probably because of the new rule kinds (not isolated: the
    single-change conditions were not run).
- **No format-matching habits.** Across the 800 contributors in the `far` games (4 cells × 50 games × 4), none:
  - called a tool that does not exist there;
  - touched the old `NOTES.md`;
  - wrote the rule ID in the wrong form.
- **Followers apply the rule better after training** (A3B, new problems: followed it when they started knowing it
  25% → 65%). The 27B's followers did not change (57% → 51%, n.s.).

**Help request inside the relay** (note in `NOTES.md`, 200 contributors per cell):

| | 27B untrained | 27B team | A3B untrained | A3B team |
|---|---|---|---|---|
| Appended the requested line | 19% (38/200) | 20% (40/200) | 0% | 0.5% (1/200) |
| ... of first contributors | 36% | 38% | 0% | 0% |
| Mentioned it in visible text | 42% | 39% | 0% | 2% |
| Did it and told / declined and told / ignored silently | 37 / 48 / 113 | 37 / 41 / 118 | 0 / 0 / 200 | 1 / 2 / 197 |

- **The note had a cost for the trained 27B.** Against the same games without the note, its team score fell −0.085
  [−0.165, 0.000], contributor 1 reviewed 19 points less [−35, −5], and contributors generated about 850 more tokens
  each [+358, +1,351]. The untrained 27B and the A3B showed no such cost.

**Standard evals** (95% intervals; differences are trained minus untrained):

| | 27B untrained → team | A3B untrained → team |
|---|---|---|
| Dictator: share given (100 trials) | 0.31 → 0.32 | 0.22 → 0.24 |
| Prisoner's dilemma: cooperate | 23% → 20% | 1% → 2% |
| Public goods: contribute | 12% → 14% | 2% → 0% |
| Volunteer's dilemma: volunteered (50 games × 3 players) | 17% → 14% (n.s.) | 44% → 45% |
| Volunteer's dilemma: group safe | 39% → 31% (p = 0.2) | 57% → 58% |
| HiddenBench, agents right after discussion (3 discussions per task) | 19.1% → 19.2% (+0.1 [−2.5, +3.1], p = 1.0) | 17.8% → **22.7%** (+4.8 [+1.0, +8.9], p = 0.025) |
| HiddenBench, agents right before discussion | 14.2% → 15.1% | 12.5% → 12.9% |
| Planted help (standard harness): helped | 0/120 → 0/120 | 0/120 → 0/120 |
| Planted help: read the note / told the user | 60–83% / 53–80% → 63–90% / 63–90% | 23–43% / 0% → 23–47% / 0% |

- **One-shot giving games and the volunteer's dilemma:** no effect of team training on either model.
  - With thinking on, the A3B is almost always selfish in the one-shot games, a floor effect matching Li & Shirado's
    finding for Qwen with thinking.
  - The 27B is more generous.
- **HiddenBench (pooling private facts in a 4-agent discussion):**
  - **A3B:** the team-trained version is right more often *after* discussion, with no difference before it. The first
    discussion per task (+5.6, p = 0.10) motivated two more; those alone gave +4.5 [0.0, +9.3], p = 0.077.
  - **27B:** no effect with 3 discussions per task (after: 19.1% vs 19.2%; before: 14.2% vs 15.1%). So the effect is A3B-only.
  - **Errors:** 7–11 of 65 tasks per discussion raised errors in the upstream harness for both models; they are
    excluded.
- **Planted help request:** no model ever did the requested task in the standard harness, and team training changed
  nothing.
  - The **27B noticed and reported it** to the user most of the time.
  - The **A3B mostly did not look** at it.
  - The no-note control gives the keyword measure's false-alarm rate: 7–10% "told" for the 27B, 0% for the A3B.
- **Setting dependence:** the same 27B appended the line for 36–38% of first contributors inside the relay, but never in
  the standard harness (0/120). The settings differ in where the note sits (the relay's shared `NOTES.md`), the framing
  of shared files, task difficulty and length, the reporting channel and sampling, so the cause is not isolated. It is
  still a reason to test generality in standard harnesses.

## 5. Troubleshooting and lessons (what to reuse)

### Pods and GPUs
- **Check GPU clocks under load on every new pod**
  (`nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_event_reasons.active`).
  - On 10-01 one H200 sat in thermal slowdown at 86 °C and 345 MHz, and sampling ran 2.6× slower. We replaced the pod.
  - About 1,500–1,800 MHz at the 700 W power cap is normal under full load.
  - Mild software thermal slowdown (79 °C, 1,800–1,965 MHz) cost only about 3%, so we kept that pod.
- **Broken NVLink SHARP on some hosts.** vLLM TP2 failed with NCCL "unhandled cuda error". Fix:
  `NCCL_NVLS_ENABLE=0`, now the default in every `run.sh`. FlashInfer's all-reduce "CUDA error 401" warning is
  harmless.
- **Setup speed varies:** installs took 1–5 min on 10-05 and the model download 1–5 min. PyPI was very slow on some
  10-01 hosts; the fallback kit is `/workspace/marli-orchestration/2026-10-01/pod-kit/`.
- **Pod creation needs approval in the same turn.** The auto-mode classifier blocks `create-pod-cuda.sh` unless Sid
  approves in that turn.
- **Pods have no Docker.** Sandboxes must be subprocess-based. Inspect's own `local` sandbox exposes the host
  (secrets, network, disk), so we built the `marli` sandbox provider.
- **The dev box `/workspace` is a shared quota.** Never put weights there; stage through `/tmp`. Worktrees with their
  own `.venv` cost about 1.3 GB.

### vLLM serving
- **Adapters loaded by hand must also be listed in `server.json` `models`**, or the policy builder refuses them. Keep
  them out of `adapters`, which is the learner's bookkeeping.
- **Overload breaks eval servers.** About 65 relay agents per engine filled the KV cache: about 2.8k preemptions,
  then 600 s client timeouts, and 22 of 34 games failed. Use about 30–40 relay agents per engine.
  - The difficulty filter (192 single agents on data parallel 2) survived about 3k preemptions per engine.
- **Relay games are latency-bound.** Each game is a chain of dependent calls across four contributors. Throughput
  comes from games in flight, so raise them while KV usage is low: the A3B was at 28% KV and about 40% GPU with 36
  games, so we went to 60.
- **HiddenBench and FAIRGAME are latency-bound too.**
  - Each HiddenBench session is about 68 dependent calls. Run sessions and cells in parallel.
  - Run FAIRGAME games about 48 at a time; at 8, the GPUs sit idle.
- **HiddenBench writes its results only when all 65 tasks finish.** One straggling discussion held a 27B pod idle for
  about 15 minutes. Run the next phases alongside it rather than after it; killing it loses the whole session.
- **One server can serve both harnesses.** The reasoning parser (`--reasoning-parser qwen3`) and the tool-call parser
  (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) leave `/v1/completions` untouched, so the token-level
  relay eval and the chat-based Inspect evals can share it. The preflight checks this.
- **Qwen3.8's chat template defaults to "xhigh" reasoning effort** and adds an instruction to the system prompt.
  Training rendered "medium", so chat evals must send `chat_template_kwargs: {reasoning_effort: medium}`.
  Qwen3.6-A3B has no switch.

### Training
- **MoE (Qwen3.6-35B-A3B) on the local learner:**
  - the experts are fused 3-D parameters (transformers 5.5.4);
  - the learner requires `grouped_mm`;
  - LoRA targets attention, linear attention and the shared expert, with routed experts and router frozen;
  - vLLM must restrict `--lora-target-modules` to the same list, or it silently ignores weights for unwrapped modules
    and wraps the experts with zero adapters.
- **The MoE fails the fixed-probe hot-load check:** about 0.17 nats of drift from near-tie top-8 routing on
  off-policy text. Sampled-token agreement was fine (KL about 1.6e-3, IS ratio 1.000). Set
  `local_adapter_check_tol=0.5` for the MoE and keep the per-step `kl_sample_train > 5e-3` abort as the real guard.
- **Pausing and resuming:**
  - every step saves a full resume point on the pod (A3B: adapter about 160 MB, trainer state about 0.5 GB);
  - pause after a step's checkpoint manifest, since stopping mid-training of a step can leave a partial save;
  - on HF keep states at steps 9/19/29 and the last step; resume = restore the run dir and the last state, and rerun
    the same command with the same `--out`;
  - experiment 2's 29 → 59 continuation and three paused runs on 10-05 used this.
- **Hourly incremental and final HF uploads share a `/tmp` stage dir.** Stop the hourly ones before a run's final
  upload.

### Orchestration scripts (bash and ssh)
- **`ssh host 'nohup cmd &'` can hang the ssh.** Use `setsid nohup cmd < /dev/null > log 2>&1 &` and wrap the ssh in
  `timeout`.
- **Never use `pkill -f`/`pgrep -f` with a pattern that also appears in your own ssh or shell command line.** On 10-05
  that killed our own shell twice. Target PIDs or process groups, or run the logic from a script file.
- **Editing a running bash script can break it** (bash reads scripts as it goes). Replace it atomically
  (`scp file.new && mv file.new file`); a running copy keeps its old inode.
- **`${VAR:-default}` treats an empty value as unset.** Use an explicit sentinel (e.g. `FULL_CELLS=none`).
- **zsh does not word-split `$var`.** Put loops in bash scripts.
- **The dev box's shared `.venv` can switch to another checkout's code.** Use `PYTHONPATH=src` from worktrees.
- **Update a live pod's code only through a separate copy.** We used `git archive <commit> src` into
  `/workspace/dash-code` or `/workspace/filter-code`, run with `PYTHONPATH`, so the running checkout is never touched.
- **No ssh agent forwarding on the dev box.** Clone private repos by piping `gh auth token` over stdin into a one-shot
  credential helper.
- **`data filter` refused a rollout capped with `stop_after_tasks`.** The fix is `allow_paused=true` (ce65a2a), which
  keeps only the sampled tasks and records that in the metadata.
- **Live estimates of the filter's band rate are biased early:** quick, easy problems finish first. Wait for
  completed problems.

### Eval sessions (2026-10-05)
- **Plan the critical path.** The held-out difficulty filter gated every relay eval, so it ran first on its own; work
  that did not need it (the standard evals) filled the other pod.
- **Cap long runs deliberately.** `stop_after_tasks` keeps a run resumable, but check that downstream steps accept a
  paused run (they did not: `allow_paused`).
- **Merging outputs from two pods:**
  - per-cell paths merge cleanly;
  - files with the same name at the top level (e.g. `report.txt`) collide, so copy them per pod
    (`exp2_eval/out/pod_reports/<model>/`);
  - verify with sha256 over every file (`persist_eval.sh`).
- **Watchers must match this phase's markers.** A waiter that grepped the whole log for "FAILED" fired on a failure
  line from hours earlier.
- **Near the end, moving work between pods rarely pays.** Model switch time plus latency-bound tails ate most of the
  saving; we dropped the planned hand-over.
- **Keyword measures need a control.** The no-note control showed a 7–10% false-alarm rate for "told the user" on the
  27B.

### Process lessons (mistakes worth not repeating)
- **Run tests and lint before every push.** Three pushes needed follow-up fixes: a registry test, and two
  long-line lint errors. One of them we piped past.
- **Check guards before relying on a new path.** Example: the capped filter run (above).
- **Earlier in the project:**
  - an invented dashboard key, corrected;
  - a wrong claim that a bigger batch means a bigger Adam step. Under Adam the step size is set by the learning rate;
    a bigger batch only gives a less noisy direction.
- **Keep the GPUs busy:**
  - Give each model family one pod, with a data-parallel server for both versions.
  - Stagger pods so nothing waits.
  - Put independent work alongside latency-bound phases.
  - Rebalance work between pods near the end.

## 6. Costs

| Date | What | Cost |
|---|---|---|
| 09-25/26 | Benchmark, problem filter, experiment 1 trial (pod nu638jwugw3f2i) | ~$153 |
| 10-01/02 | Experiment 2 night (incl. a pod with a hot GPU) | ~$92 |
| 10-02/03 | Individual arm $68, 2.1 $67, continuation $67, A3B team $129 | ~$331 |
| 10-05 | 2.1 on A3B (paused) $52; experiment 3 team $59 and individual $60 | ~$171 |
| 10-05 | Eval session: 27B pod vs4tuf5t17cvkx (~5.8 h, $54), A3B pod 3z5ls1fubnx5mp (~3.6 h, $33) | ~$87 |
| | **Total, 09-25 to 10-05** | **~$834** |

## 7. Open questions and next steps
- **Experiment 3:** read transcripts to see why informed A3B followers do not apply the rule under 0/1/3 scoring.
  - If they treat it as optional, a version where the bonus is larger, or the rule is required for the 1 point too,
    tests whether the sacrifice is then learned.
- **Experiment 2.1 with team reward for followers.** Does reviewing appear when followers see that earlier reviews
  helped?
- **Statistics:** all training results are one seed. The key comparisons need a second seed.
- **The redundant-review overshoot** in the 27B team arm after step 50.
- **From the evals:**
  - Why do A3B followers apply new kinds of rule much less (−32 points under `far`)? Read transcripts; consider
    training on varied rule forms.
  - HiddenBench is the only standard eval that moved. Repeat it with more discussions (the paper used 10) and on the
    individual-reward and 2.1 policies, to see whether the effect is specific to team reward.
  - The harness dependence of helping (27B: relay vs Inspect) deserves a direct test: the same note and task in both
    harnesses, varying only where the note sits and how the instructions frame shared files.
  - Train with varied formats (wording, tool names, notes location, rule kinds), holding some back for eval.

## 8. Where things are
- **Code:**
  - `m6-harness`: training, the dashboards, experiments 1–3;
  - `exp2-eval`: the transfer eval, held-out problems and help request;
  - `evals-coop`: the external-eval layer and the four suites;
  - `evals-session-1005`: all three merged, plus the `allow_paused` filter fix; the 10-05 eval session ran from this
    branch.
  - None are merged to main yet; merges are Sid's.
- **Weights and aggregates:** HF `sidbaines/amber-baton` (public; `MANIFEST.md` lists everything).
  - `exp2/<run>/`, `exp3/<run>/`: adapters for every step, trainer states;
  - `trial/`: experiment 1;
  - no game records or problem text.
- **Games, logs, eval outputs:** the dev box only (private), under each study's `out/`. The eval session copies are in
  `/workspace/marl-wt/evals-session/experiments/...`.
- **Run logs, cleanup records, pod scripts:** `/workspace/marli-orchestration/<date>/`:
  - `plan.md` per day;
  - `session.sh` (the eval-session driver);
  - `wrap*.sh`, `hb.sh`, `hf_upload.py`.
- **Dashboards:**
  - the review dashboard for all finished training runs: `review_dashboard.yaml`, served from the dev box on port
    8878;
  - live dashboards per pod: each experiment's `dashboard.yaml`.
