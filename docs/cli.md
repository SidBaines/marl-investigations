<!-- Generated: uv run marli describe --all --markdown > docs/cli.md -->

# marli CLI reference

## Built-in commands

| Command | Purpose |
| --- | --- |
| `marli --version` | Report the package version. |
| `marli list [KIND]` | List all registries, or the entries in one registry. |
| `marli describe VERB... [--markdown]` | Describe one verb and its config fields. |
| `marli describe --all [--markdown]` | Describe every registered verb. |
| `marli inspect PATH` | Inspect a manifest and its input provenance chain. |
| `marli status OUT` | Read saved run and progress records without taking a lock. |

## Output and exit codes

Commands emit exactly one compact JSON line on stdout; logs and stray verb
prints go to stderr. Success includes `ok: true`; errors include `ok: false`,
`error`, `message`, and `exit_code`. `describe --markdown` emits Markdown instead
of JSON. `--help` prints normal argparse help to stdout and exits with code 0.

| Exit code | Meaning (`marli.errors`) |
| --- | --- |
| 0 | Success |
| 1 | Unexpected failure (`MarliError`) |
| 2 | Usage/config error (`ConfigError`, `DirtyTreeError`) |
| 3 | Config-hash mismatch (`HashMismatchError`) |
| 4 | Budget exceeded (`BudgetExceededError`) |
| 5 | Backend error (`BackendError`) |

## Verbs

Usage: `marli VERB... [CONFIG.yaml ...] [key=value ...] --out DIR|auto [--force]`.
YAML files merge left to right, followed by dotted overrides. `--out auto` uses
`$MARLI_RUNS/<verb-with-hyphens>/<hash12>` (default root: `runs`). Matching completed
runs are reused; incomplete runs resume. `--force` replaces existing run output.

No verbs registered yet.
