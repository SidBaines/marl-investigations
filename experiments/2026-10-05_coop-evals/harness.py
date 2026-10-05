"""Glue between run.sh and the upstream harnesses; the study configs stay the one source.

    python harness.py cells <config.yaml> [--only a,b]          # label<TAB>seats<TAB>results
    python harness.py subset <config.yaml> <out.yaml> --only a,b
    python harness.py request <suite> <model.yaml>              # resolved request fields (JSON)
    python harness.py sidecar <dir> --repo R --commit C --patch P [--request JSON] [--extra JSON]
    python harness.py fairgame-config <checkout> <resources> <name> --seats a,b,c --seed K

``request`` applies marli's own resolution (model card < benchmark < overrides) so the
harness is launched with exactly the settings ``marli eval external`` later checks.
``fairgame-config`` builds a FAIRGAME runner config from the pinned checkout's starter
library (seed_cfg_volunteer + its English template) with neutral personalities: no
FAIRGAME text is copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from marli.eval.external.serving import request_fields, resolve_settings
from marli.eval.external.spec import SUITES

VOLUNTEER_CONFIG = "starter_library/configurations/18-seed_cfg_volunteer.json"
VOLUNTEER_TEMPLATE = "starter_library/templates/13-seed_tpl_gt_volunteer_conventional_en.md"


def _cells(config: Path, only: list[str] | None) -> list[dict]:
    cells = yaml.safe_load(config.read_text())["cells"]
    if only:
        unknown = set(only) - {cell["label"] for cell in cells}
        if unknown:
            sys.exit(f"unknown cells {sorted(unknown)} in {config}")
        cells = [cell for cell in cells if cell["label"] in only]
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cells")
    p.add_argument("config", type=Path)
    p.add_argument("--only")
    p = sub.add_parser("subset")
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--only", required=True)
    p = sub.add_parser("request")
    p.add_argument("suite")
    p.add_argument("model", type=Path)
    p = sub.add_parser("sidecar")
    p.add_argument("dir", type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--request", default="null")
    p.add_argument("--extra", default="{}")
    p = sub.add_parser("fairgame-config")
    p.add_argument("checkout", type=Path)
    p.add_argument("resources", type=Path)
    p.add_argument("name")
    p.add_argument("--seats", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prefix", default="litellm:hosted_vllm/")
    args = parser.parse_args()

    if args.command == "cells":
        for cell in _cells(args.config, args.only.split(",") if args.only else None):
            print(f"{cell['label']}\t{','.join(cell['seats'])}\t{cell['results']}")
    elif args.command == "subset":
        data = yaml.safe_load(args.config.read_text())
        data["cells"] = _cells(args.config, args.only.split(","))
        if data.get("baseline") not in {cell["label"] for cell in data["cells"]}:
            data.pop("baseline", None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(data, sort_keys=False))
    elif args.command == "request":
        model = yaml.safe_load(args.model.read_text()).get("model_generation", {})
        settings = resolve_settings(SUITES.load(args.suite), model, {})
        print(json.dumps(request_fields(settings), sort_keys=True))
    elif args.command == "sidecar":
        args.dir.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "repo": args.repo,
            "commit": args.commit,
            "patch": str(args.patch),
            "patch_sha256": hashlib.sha256(args.patch.read_bytes()).hexdigest(),
            "request": json.loads(args.request),
            **json.loads(args.extra),
        }
        (args.dir / "harness.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    elif args.command == "fairgame-config":
        seats = args.seats.split(",")
        source = json.loads((args.checkout / VOLUNTEER_CONFIG).read_text())
        config = dict(source["game_config"])
        n_agents = len(config["agents"]["names"])
        if len(seats) != n_agents:
            sys.exit(f"{len(seats)} seats for a {n_agents}-agent game")
        config["agents"] = {
            **config["agents"],
            "personalities": {"en": ["None"] * n_agents},  # neutral: the intro block is dropped
            "opponentPersonalityProb": [0] * n_agents,
        }
        config["llms"] = [args.prefix + seat for seat in seats]
        config["seed"] = args.seed
        config.update(name=args.name, languages=["en"])
        template = (args.checkout / VOLUNTEER_TEMPLATE).read_text()
        body = (
            template.split("\n---\n", 1)[1].lstrip("\n") if template.startswith("---") else template
        )
        (args.resources / "config/volunteer").mkdir(parents=True, exist_ok=True)
        (args.resources / "game_templates").mkdir(parents=True, exist_ok=True)
        (args.resources / f"config/volunteer/{args.name}.json").write_text(
            json.dumps(config, indent=1)
        )
        (args.resources / "game_templates/volunteer_en.txt").write_text(body)
        strategies = config["payoffMatrix"]["strategies"]["en"]
        print(
            json.dumps(
                {"do_nothing": strategies["strategy1"], "volunteer": strategies["strategy2"]}
            )
        )


if __name__ == "__main__":
    main()
