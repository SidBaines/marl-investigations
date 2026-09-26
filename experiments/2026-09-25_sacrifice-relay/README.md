# Sacrifice relay: do agents learn to give up their own reward for later agents?

Status: overnight trial done (2026-09-26); the full three-arm runs are not started. This is a signs-of-life study: one seed and three
runs.

## Question

A shared repo is worked on by several agents in turn. Each agent can score on
its own. It can also *sacrifice* its own episode reward to obtain hidden
information that lets later agents score higher. Does RL teach agents to do
this, and does it depend on who is rewarded?

- **Individual reward:** each agent gets its own score.
- **Team reward:** every agent gets the mean score of the repo's agents.

## Environment (`code_rules`; see `src/marli/envs/code_rules.py`)

A **repo** holds N coding problems, one per contributor, plus one hidden
**house rule**. The problems come from DeepCoder train and are filtered to the
model's pass@4 in [¼, ¾].

- The rule is sampled fresh per episode from the episode seed, so it can't be
  memorised across repeats.
- It is *static*: a random ID that must appear in a set form in the source,
  e.g. a first line `# release: TCK-4821`. It is unguessable (about 10⁸
  possible IDs) and trivial to follow once known.

Each contributor gets **one CI run**, in one of two modes:

| Action | Contributor's own score |
|---|---|
| `ci_review`: dry run that reveals the rule, unscored (the *sacrifice*) | 0 |
| `ci_submit`, base tests fail | 0 |
| `ci_submit`, base tests pass, rule not followed | 1 |
| `ci_submit`, base tests pass, rule followed | m = 3 |
| no CI run | 0 |

- `ci_submit` reports its score. On extended checks it says only PASS/FAIL,
  never the rule.
- A shared `NOTES.md` persists across contributors. Contributors can read it,
  and can write to it at any time, including after their CI run.
- The prompt states only the mechanics: the contributor's position k of N, the
  two CI modes and the scores above, and that later contributors can read
  `NOTES.md`. It never asks the agent to help anyone. It never says whether
  the reward is individual or team.

**Payoffs.**
- **For the agent itself,** probing always costs its expected pass (p ≈ 0.5).
- **For the team,** with one early prober and followers who apply the rule
  from the notes, probing is worth it only if m(N−1) > N:
  - N = 4, m = 3: 9 > 4, so a first-contributor probe is strongly team-optimal;
  - N = 1: probing is never worth it.

## Runs (one seed each)

| Run | Protocol | N | Reward (`credit.reward_target`) |
|---|---|---|---|
| `solo` (control) | `relay_n1` | 1 | individual (identical to team at N = 1) |
| `relay_individual` | `relay_n4` | 4 | individual |
| `relay_team` | `relay_n4` | 4 | team (every agent gets the repo's mean score) |

**Shared setup:**
- Model: Qwen/Qwen3.8-27B, local backend (PEFT LoRA r = 32 plus vLLM), with the
  renderer's reasoning effort set to `medium`.
- One learner shared by every contributor, lr 2e-5, importance-sampling loss.
- Default credit: role leave-one-out baseline, mean normalisation, token-sum
  aggregation.
- Batches match agent decisions per step:
  - relay: 4 repos × G = 4 → 16 episodes and 64 agent decisions per step;
  - solo: 16 repos × G = 4 → 64 decisions.
- 80 steps, a checkpoint every 10.
- Per-contributor budget: 12,288 generated tokens, ≤ 6,144 per call, context ≤ 24k.

## Pre-registered hypotheses and readouts

**Primary metric:** the **first-mover probe rate**, i.e. the fraction of
contributors who did not know the rule when they started and who chose
`ci_review`. "Knew the rule" means the rule's ID appeared anywhere in the repo
at the contributor's start (`rule_known_at_start`); the rule can be passed on
in any file, not only `NOTES.md`. It is computed by `analyze.py` from saved
rollouts, overall and by slot (the last slot can never help anyone). "Early"
and "late" below mean the mean over steps 0–9 and steps 70–79.

- **H-team:** in `relay_team`, the first-mover probe rate rises from early to
  late. Team return rises with it.
- **H-individual (Sid's hypothesis):** in `relay_individual`, the first-mover
  probe rate also rises. The route would be exposure: later contributors are
  rewarded for acting on rules that earlier contributors' write-ups revealed.
  - The prober itself gets 0 against a positive baseline. So the direct
    gradient *lowers* probing, and the null expectation is a fall.
  - Evidence for exposure is a late rate above the early rate, and above
    `solo`'s late rate.
- **Control:** in `solo`, the probe rate falls toward 0.
- **Understanding vs habit:**
  - redundant probing (probing when the rule is already known) should
    fall in `relay_team`;
  - prober hand-off fidelity (the next contributor starts knowing the rule)
    should rise.

**Secondary metrics:**
- base pass rate (competence drift);
- how often followers who knew the rule score 3;
- individual vs team return;
- `kl_sample_train` and the IS ratio for each run.

With n = 1 seed, "rises" means a clear trend over 80 steps (about 64
decisions per step, so the early and late means each pool about 640). It is
not a significance test. Promising runs get seeds and ablations next:
- hidden notes;
- oracle notes;
- a per-slot LoRA;
- m and N sweeps;
- swarm.

**Gate before training:** run a base-rate eval of the untrained model on the
training repos. This signs-of-life study has no held-out set; the gate is a
pre-training measurement on the same distribution.
- If the first-mover probe rate is below 3%, RL has no probes to reinforce.
  We then stop and discuss. The fix would change prompt salience, not rewards.
- If the base pass rate falls outside [0.25, 0.75], re-filter the problem pool.

## Exact commands run

See `run.sh` (to be filled in as run).

## Results summary

### Overnight trial (team reward, 30 steps, 2026-09-25/26)

Run `out/trial_team`: the team-reward relay, 4 repos × G = 4 per step, flat
lr 4e-5. It took 30 steps in 7 h (14.0 min per step) and stopped cleanly
after the step-29 checkpoint. The table compares steps 0–9 with steps 20–29
(160 relays and 640 contributors in each); ± is a 95% interval
(`out/trial_team_analysis/early_vs_late.txt`).

| | Steps 0–9 | Steps 20–29 | Change |
|---|---|---|---|
| Team score (mean over the 4 contributors) | 0.545 ± 0.054 | 0.631 ± 0.049 | **+0.086 (z = 2.3)** |
| Contributors who reached CI | 61.4% | 70.2% | +8.8 pts |
| Base tests pass when submitted | 83.3% | 85.9% | +2.6 pts |
| Sacrificed, among contributors with a real choice (no rule at start, ran CI) | 38/359 = 10.6% ± 3.2% | 30/419 = 7.2% ± 2.5% | **−3.4 pts (z = −1.7)** |
| Scored 3 when the rule was known at the start and they submitted | 81.8% (n = 33) | 82.1% (n = 28) | — |
| Contributors scoring 3 | 4.2% | 3.6% | — |

Sacrifice rate by position (early → late):

| Contributor 1 | Contributor 2 | Contributor 3 | Contributor 4 (can help no one) |
|---|---|---|---|
| 4/125 = 3% → 2/133 = 2% | 17/98 = 17% → 16/124 = 13% | 14/73 = 19% → 9/80 = 11% | 3/63 = 5% → 3/82 = 4% |

- **Signs of life: yes.** The team reward rose clearly within 30 steps. Most of
  the gain came from contributors reaching CI more often: the model spends
  fewer turns before running CI, and the 12,288-token budget is the bottleneck.
- **H-team: not supported so far.** The sacrifice rate fell slightly rather
  than rising; the decline is suggestive but not conclusive at n = 1 seed.
  Learning to reveal the rule for others did not show up in 30 steps.
  - The first contributor, who could help the most people, almost never
    sacrifices. Contributors 2 and 3 do so most often; one possibility is that
    they react to an earlier contributor's notes about failed extended checks.
    That is a hypothesis to check in transcripts.
- **Followers use the notes:** when the rule was known and they submitted,
  they scored 3 about 82% of the time.
- **Training health:** `kl_sample_train` ≤ 7.5e-4 and a mean IS ratio of
  1.0000 ± 0.0001 at every step. Adapter hot-load drift stayed at 0.016–0.028
  nats (tolerance 0.05). There were no errors, restarts or stalls.
- **Caveats:**
  - One seed and one arm.
  - The 51 relay repos repeat (each drawn about 2.4 times over 30 steps).
  - The token budget still binds: about 30% of contributors never reach CI.
  - One step's gradient mixes 16 relays, so a single sacrifice decision is a
    weak signal under team reward.

Artefacts:
- Adapters for every step, trainer states every 5 steps, metrics and analysis:
  HF `sidbaines/amber-baton` (public; no prompts or transcripts).
- Rollouts: dev box `out/trial_team/rollouts` (private; they contain problem
  text).
- To resume: restore the run dir, with its states and checkpoints, and rerun
  the same `train rl` command with the same `--out`.

**Data, as run 2026-09-25** (untrained Qwen3.8-27B; single agent; 12,288
tokens per episode; 6,144 per turn):
- The filter covered 800 DeepCoder train problems × 4 attempts: 3,200
  episodes, 0 failed. It was paused after the first 100 problems and resumed
  for the rest; the run's identity was unchanged.
- **Kept 207 problems** (26%): solved in 1–3 of 4 attempts. Dropped 385 as
  too easy (4 of 4) and 208 as too hard (0 of 4).
- This makes **51 relay repos** of 4 problems (3 problems left over) and
  **207 solo repos**.
- With 80 steps, each relay repo is drawn about 6.3 times and each solo repo
  about 6.2 times. The problems repeat; only the rule is resampled.
- **22% of filter episodes ran out of budget** (702 of 3,200; 23% among the
  kept problems). 95% of those had at least one turn stopped at the 6,144-token
  per-turn limit mid-thought, and 74% had two or more.
- A trial `eval continue` on 16 of them at a 20,480-token budget
  (`out/bench/continue_test`) replayed every earlier call exactly and joined
  the cut call correctly. 1 of the 16 then passed; 15 ran out again, mostly on
  more turns at the per-turn limit. The per-turn limit (or reasoning effort),
  not the total budget, is the main constraint.

## Deviations from the design

- The pool is 207 problems (51 relay repos), not the ~320 extrapolated from the
  first 100 problems.
- **Overnight trial** (`configs/trial_team.yaml`, run `out/trial_team`): only the
  team-reward relay, for about 30 of 80 steps, on the two-GPU layout (option 2).
  - Learning rate is a flat **4e-5**: twice the pre-registered 2e-5, because
    only ~30 steps fit in one night.
  - A checkpoint is written after every step, and the run resumes with the same
    `--out`.
  - The token limits are unchanged (12,288 per contributor, 6,144 per turn),
    matching the filter.

## Spend

| Item | Budget | Actual |
|---|---|---|
| 2×H200 pod `nu638jwugw3f2i`: benchmark, filter (800 problems), option-2 test, continue test, overnight trial (30 steps) | approved by Sid step by step | about $153 (2026-09-25 14:22 → 2026-09-26 07:03 UTC; pod deleted after HF + dev box persistence was verified; see `/workspace/marli-orchestration/2026-09-25/CLEANUP_RECEIPT.md`) |
