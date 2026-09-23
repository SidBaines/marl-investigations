Generated: `uv run marli describe --all --markdown > docs/cli.md`

# marli CLI reference

## Built-in commands

| Command | Purpose |
| --- | --- |
| `marli --version` | Report the package version. |
| `marli list [KIND]` | List all registries, or the entries in one registry. |
| `marli describe VERB... [--markdown]` | Describe one verb and its config fields. |
| `marli describe --all [--markdown]` | Describe every registered verb. |
| `marli inspect PATH` | Inspect a manifest file or directory and its input chain. |
| `marli status OUT` | Read saved run and progress records without taking a lock. |

## Output and exit codes

Commands emit exactly one compact JSON line on stdout; logs and stray verb
prints (including subprocess output) go to stderr. Success includes `ok: true`;
errors include `ok: false`,
`error`, `message`, and `exit_code`. `describe --markdown` emits Markdown instead
of JSON. `--help` prints normal argparse help to stdout and exits with code 0.
A verb's `SystemExit` is an unexpected failure (exit 1); `KeyboardInterrupt` exits 130.

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | `MarliError`: An unexpected or otherwise unclassified marli failure.; `RunDirLockedError`: Another owner holds this run directory's exclusive lock. |
| 2 | `ConfigError`: Invalid configuration or usage.; `DirtyTreeError`: Training provenance cannot be tied to a clean commit. |
| 3 | `HashMismatchError`: An existing run directory belongs to a different configuration. |
| 4 | `BudgetExceededError`: A configured spending limit was exceeded. |
| 5 | `BackendError`: A backend failed or cannot support the requested operation. |
| 130 | Interrupted (`KeyboardInterrupt`) |

## Verbs

Usage: `marli VERB... [CONFIG.yaml ...] [key=value ...] --out DIR|auto [--force]`.
YAML files merge left to right, followed by dotted overrides. `--out auto` uses
`$MARLI_RUNS/<verb-with-hyphens>/<hash12>` (default root: `runs`). Matching completed
runs are reused; incomplete runs resume. `--force` replaces existing run output.

### data build

Build a taskset from a source, optionally excluding overlapping prompts.

Manifest produced: `taskset.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| source | str | aime_2025 | false | false | false | task source registry name (marli list tasks) |
| split | str \| None | null | false | false | false |  |
| max_n | int \| None | null | false | false | false |  |
| seed | int | 0 | false | false | false |  |
| shuffle | bool | false | false | false | false |  |
| exclude | str \| None | null | false | false | true | TaskSet whose prompts are removed (decontamination by exact/normalised match) |
| ngram_exclude | int | 0 | false | false | false | if >0, also drop tasks sharing any n-gram of this length (words) with `exclude` |

### data filter

Filter a taskset by pass rates from saved rollouts.

Manifest produced: `taskset.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| tasks | str \| None | null | false | false | true | TaskSet to filter |
| episodes | str \| None | null | false | false | true | episodes dir of rollouts of that TaskSet (eval rollout output) |
| lo | float | 0.0 | false | false | false |  |
| hi | float | 1.0 | false | false | false |  |
| inclusive | bool | false | false | false | false |  |
| min_episodes | int | 2 | false | false | false |  |
| metric | str | correct | false | false | false |  |

### eval grid

Run labelled rollout/score cells and combine their compute-aware report.

Manifest produced: `report.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| cells | list[CellSpec] | [] | false | false | false |  |
| common.tasks | str \| None | null | false | false | false |  |
| common.protocol | str | single | false | false | false |  |
| common.protocol_config | dict[str, Any] | {} | false | false | false |  |
| common.env | str | math | false | false | false |  |
| common.env_config | dict[str, Any] | {} | false | false | false |  |
| common.policies | dict[str, PolicySpec] | {} | false | false | false |  |
| common.seating | dict[str, str] | {} | false | false | false |  |
| common.limits.call.max_tokens | int | 4096 | false | false | false | cap on tokens sampled by a single call |
| common.limits.call.min_call_tokens | int | 16 | false | false | false | below this allocation a call is not made (limit exhausted) |
| common.limits.agent.max_gen_tokens | int | 32768 | false | false | false | total generated tokens for one agent instance |
| common.limits.agent.max_calls | int | 64 | false | false | false | max LLM calls for one agent instance |
| common.limits.agent.final_reserve | int | 512 | false | false | false | tokens held back for the forced final/report call |
| common.limits.worker.max_gen_tokens | int | 8192 | false | false | false | total generated tokens for one agent instance |
| common.limits.worker.max_calls | int | 24 | false | false | false | max LLM calls for one agent instance |
| common.limits.worker.final_reserve | int | 512 | false | false | false | tokens held back for the forced final/report call |
| common.limits.session.max_sessions | int | 1 | false | false | false | sessions per multi-session agent (1 = single-session) |
| common.limits.session.max_gen_tokens | int | 16384 | false | false | false | generated tokens per session (incl. compaction calls) |
| common.limits.session.carry_reserve | int | 2048 | false | false | false | tokens held back for the end-of-session carry call (must cover the summary) |
| common.limits.session.carry_max_tokens | int | 1024 | false | false | false | cap on the carried summary/notes length |
| common.limits.episode.max_gen_tokens | int | 131072 | false | false | false | generated tokens for the whole episode (all agents) |
| common.limits.episode.max_ticks | int | 64 | false | false | false | lockstep ticks before the episode is forced to finish |
| common.limits.episode.max_wall_s | float | 1800.0 | false | false | false | wall-clock cap (straggler guard; logged, not used for matching) |
| common.limits.spawn.max_per_call | int | 4 | false | false | false | workers one spawn_workers call may start |
| common.limits.spawn.max_total | int | 8 | false | false | false | workers per coordinator per episode |
| common.limits.spawn.max_depth | int | 1 | false | false | false | spawn depth (v1: workers cannot spawn) |
| common.limits.ctx.max_ctx | int | 32768 | false | false | false | max prompt+completion tokens in one segment; <= model and backend max_seq_len |
| common.limits.on_exhaust | str | force_final | false | false | false | force_final \| none |
| common.limits.on_no_tool_call | str | nudge | false | false | false | no-tool turn: nudge \| end_agent \| final_text_as_answer \| final_text_continue |
| common.limits.max_nudges | int | 2 | false | false | false | consecutive nudges before the agent is ended |
| common.limits.tool_output_chars | int | 8000 | false | false | false | tool results are truncated (head+tail) to this many chars |
| common.schedule | str | lockstep | false | false | false |  |
| common.episodes_per_task | int | 1 | false | false | false |  |
| common.run_seed | int | 0 | false | false | false |  |
| common.max_tasks | int \| None | null | false | false | false |  |
| common.record_tokens | bool | false | false | false | false |  |
| common.concurrency | int | 8 | false | true | false | concurrent episodes |
| common.max_usd | float \| None | null | false | true | false | spend guard for this run |
| baseline | str \| None | null | false | false | false |  |
| parallel_cells | int | 1 | false | true | false | concurrent cells; each has its own spend guard |

### eval report

Report one Scores input with compute and paired task statistics.

Manifest produced: `report.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| scores | str \| None | null | false | false | true | One Scores manifest or dir; use eval grid to combine labelled cells |
| baseline | str \| None | null | false | false | false |  |
| group_by | list[str] | ["protocol", "policy", "taskset"] | false | false | false |  |

### eval rollout

Sample resumable episodes with bounded concurrency and a spend guard.

Manifest produced: `episodes.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| tasks | str \| None | null | false | false | true | TaskSet manifest or dir |
| protocol | str | single | false | false | false |  |
| protocol_config | dict[str, Any] | {} | false | false | false |  |
| env | str | math | false | false | false |  |
| env_config | dict[str, Any] | {} | false | false | false |  |
| policies | dict[str, PolicySpec] | {} | false | false | false |  |
| seating | dict[str, str] | {} | false | false | false |  |
| limits.call.max_tokens | int | 4096 | false | false | false | cap on tokens sampled by a single call |
| limits.call.min_call_tokens | int | 16 | false | false | false | below this allocation a call is not made (limit exhausted) |
| limits.agent.max_gen_tokens | int | 32768 | false | false | false | total generated tokens for one agent instance |
| limits.agent.max_calls | int | 64 | false | false | false | max LLM calls for one agent instance |
| limits.agent.final_reserve | int | 512 | false | false | false | tokens held back for the forced final/report call |
| limits.worker.max_gen_tokens | int | 8192 | false | false | false | total generated tokens for one agent instance |
| limits.worker.max_calls | int | 24 | false | false | false | max LLM calls for one agent instance |
| limits.worker.final_reserve | int | 512 | false | false | false | tokens held back for the forced final/report call |
| limits.session.max_sessions | int | 1 | false | false | false | sessions per multi-session agent (1 = single-session) |
| limits.session.max_gen_tokens | int | 16384 | false | false | false | generated tokens per session (incl. compaction calls) |
| limits.session.carry_reserve | int | 2048 | false | false | false | tokens held back for the end-of-session carry call (must cover the summary) |
| limits.session.carry_max_tokens | int | 1024 | false | false | false | cap on the carried summary/notes length |
| limits.episode.max_gen_tokens | int | 131072 | false | false | false | generated tokens for the whole episode (all agents) |
| limits.episode.max_ticks | int | 64 | false | false | false | lockstep ticks before the episode is forced to finish |
| limits.episode.max_wall_s | float | 1800.0 | false | false | false | wall-clock cap (straggler guard; logged, not used for matching) |
| limits.spawn.max_per_call | int | 4 | false | false | false | workers one spawn_workers call may start |
| limits.spawn.max_total | int | 8 | false | false | false | workers per coordinator per episode |
| limits.spawn.max_depth | int | 1 | false | false | false | spawn depth (v1: workers cannot spawn) |
| limits.ctx.max_ctx | int | 32768 | false | false | false | max prompt+completion tokens in one segment; <= model and backend max_seq_len |
| limits.on_exhaust | str | force_final | false | false | false | force_final \| none |
| limits.on_no_tool_call | str | nudge | false | false | false | no-tool turn: nudge \| end_agent \| final_text_as_answer \| final_text_continue |
| limits.max_nudges | int | 2 | false | false | false | consecutive nudges before the agent is ended |
| limits.tool_output_chars | int | 8000 | false | false | false | tool results are truncated (head+tail) to this many chars |
| schedule | str | lockstep | false | false | false |  |
| episodes_per_task | int | 1 | false | false | false |  |
| run_seed | int | 0 | false | false | false |  |
| max_tasks | int \| None | null | false | false | false |  |
| record_tokens | bool | false | false | false | false |  |
| concurrency | int | 8 | false | true | false | concurrent episodes |
| max_usd | float \| None | null | false | true | false | spend guard for this run |

### eval score

Score saved episodes; optionally regrade with the environment verifier.

Manifest produced: `scores.json`.

| Name | Type | Default | Required | Runtime | Input | Help |
| --- | --- | --- | --- | --- | --- | --- |
| episodes | str \| None | null | false | false | true | EpisodeSet manifest or dir |
| regrade | bool | false | false | false | false |  |
