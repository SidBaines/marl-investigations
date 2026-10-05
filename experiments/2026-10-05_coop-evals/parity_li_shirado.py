"""Check marli's game-decision parser against Li & Shirado's published Table 1 (CPU, ~1 min).

    python parity_li_shirado.py [--cache /tmp/li_shirado_replies]

Downloads the authors' released replies (HF YuxuanLi1225/UncooperativeReasoning at a pinned
revision; no licence is stated, so they are cached under /tmp and never committed), parses
every dictator / prisoner's dilemma / public goods reply with marli.eval.external.parsing and
prints our rates beside the paper's (arXiv:2502.17720v4, Table 1). Exit 1 if any rate is off
by more than 3 points (or 0.02 of the dictator share).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from marli.eval.external.parsing import parse_amount, parse_option, strip_reasoning

REVISION = "c014aa20fc1c6c3fc7f5cfc44d0dcd5585df4917"
URL = "https://huggingface.co/datasets/YuxuanLi1225/UncooperativeReasoning/resolve/{rev}/data/{f}/single_shot_games.json"
FILES = ("OpenAI", "Gemini", "DeepSeek", "Claude", "Qwen")
# Table 1: dictator mean share, prisoner's dilemma and public goods cooperation out of 100.
PAPER = {
    "gpt-4o": (0.496, 95, 96),
    "gpt-o1": (0.420, 16, 20),
    "gemini-2.0-flash": (0.473, 96, 100),
    "gemini-2.0-flash-thinking-exp": (0.297, 3, 2),
    "deepseek-v3": (0.488, 3, 23),
    "deepseek-r1": (0.276, 0, 0),
    "claude": (0.410, 100, 99),
    "claude-thinking": (0.321, 96, 93),
    "qwen3-32b": (0.500, 100, 64),  # the paper's "Qwen3-30B"; the released files say qwen3-32b
    "qwen3-32b_thinking": (0.099, 0, 0),
}
GAMES = ("dictator_game", "prisoner_dilemma_game", "public_goods_game")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/li_shirado_replies"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)
    for name in FILES:
        path = args.cache / f"{name}.json"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(rev=REVISION, f=name), path)
        for reply in json.loads(path.read_text()):
            if reply["game"] not in GAMES:
                continue
            text, _ = strip_reasoning(reply["response"])
            if reply["game"] == "dictator_game":
                amount = parse_amount(text)
                value = None if amount is None else amount / 100
            else:
                option = parse_option(text)
                value = None if option is None else float(option == "A")
            if value is None:
                failures[reply["model"]] += 1
            else:
                values[(reply["model"], reply["game"])].append(value)
    worst = 0.0
    print("| Model | Dictator share (ours / paper) | PD give | PG contribute | Unparsed |")
    print("| --- | --- | --- | --- | --- |")
    for model, (dictator, pd, pg) in PAPER.items():
        share = sum(values[(model, GAMES[0])]) / len(values[(model, GAMES[0])])
        ours_pd, ours_pg = (int(sum(values[(model, game)])) for game in GAMES[1:])
        worst = max(worst, abs(share - dictator) / 0.02 * 3, abs(ours_pd - pd), abs(ours_pg - pg))
        print(
            f"| {model} | {share:.3f} / {dictator:.3f} | {ours_pd} / {pd} | {ours_pg} / {pg} | "
            f"{failures[model]} / 300 |"
        )
    sys.exit(1 if worst > 3 else 0)


if __name__ == "__main__":
    main()
