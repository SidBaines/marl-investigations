# Sacrifice relay: do agents learn to give up their own reward for later agents?

Status: planned (2026-09-25). This is a signs-of-life study: one seed and three
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
contributors whose `NOTES.md` held no rule at their CI time and who chose
`ci_review`. It is logged every step (`grades/…/probed`, split by slot and by
`notes_had_rule`). "Early" and "late" below mean the mean over steps 0–9 and
steps 70–79.

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
  - redundant probing (probing when the rule is already in the notes) should
    fall in `relay_team`;
  - prober write-up fidelity (the rule appears in the next contributor's
    `NOTES.md`) should rise.

**Secondary metrics:**
- base pass rate (competence drift);
- how often followers get the bonus when the notes hold the rule;
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

**Gate before training:** run a base-rate eval of the untrained model on held-out
repos.
- If the first-mover probe rate is below 3%, RL has no probes to reinforce.
  We then stop and discuss. The fix would change prompt salience, not rewards.
- If the base pass rate falls outside [0.25, 0.75], re-filter the problem pool.

## Exact commands run

See `run.sh` (to be filled in as run).

## Results summary

_Pending._

## Deviations from the design

_None yet._

## Spend

| Item | Budget | Actual |
|---|---|---|
| 2×H200 pod: spike, filter/base-rate eval, 3 runs | to be confirmed with Sid | — |
