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

No verbs registered yet.
