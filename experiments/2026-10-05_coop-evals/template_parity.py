"""Our training renderer vs the model's own HF chat template, which vLLM's chat endpoint uses.

    python template_parity.py      # CPU; downloads the two tokenizers (set HF_HOME under /tmp)

For each model, the game prompts rendered by the training renderer must equal the HF template
rendered with the study's chat_template_kwargs; it also shows what the template renders
without them (a stock client), which for Qwen3.8 adds the default xhigh reasoning instruction.
Needs the [render] extra plus jinja2.
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

from marli.eval.external.tasks.econ_games import GAMES, SYSTEM_PROMPT
from marli.render.base import Msg
from marli.render.registry import get_renderer

CASES = {
    "qwen3_8_medium": ("Qwen/Qwen3.8-27B", {"enable_thinking": True, "reasoning_effort": "medium"}),
    "qwen3_5": ("Qwen/Qwen3.6-35B-A3B", {"enable_thinking": True}),
}


def main() -> None:
    ok = True
    for renderer_name, (hf_id, kwargs) in CASES.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        renderer = get_renderer(renderer_name, hf_id=hf_id)
        for game in ("dictator", "prisoners_dilemma"):
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": GAMES[game].prompt},
            ]
            ours = list(renderer.initial(SYSTEM_PROMPT, [], [Msg("user", GAMES[game].prompt)]))
            render = dict(add_generation_prompt=True, tokenize=True, return_dict=False)
            served = list(tokenizer.apply_chat_template(chat, **render, **kwargs))
            stock = list(tokenizer.apply_chat_template(chat, **render))
            ok &= ours == served
            print(
                f"{renderer_name:16s} {game:18s} renderer == template(kwargs): {ours == served} "
                f"({len(ours)} ids); template without kwargs differs: {stock != served}"
            )
            if stock != served:
                print("    without kwargs it renders:", repr(tokenizer.decode(stock)[:160]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
