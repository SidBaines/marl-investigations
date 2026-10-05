"""Private audit of one planted-help sample: what the agent saw, did and reported.

    python audit_help.py <eval external out dir> <cell> <sample id, e.g. fizzbuzz/A_notes>

Prints the condition, the note, every bash command with the start of its output, the
visible assistant text, the submitted report and the scores, from the cell's Inspect log.
Transcripts stay on the pod/dev box (out/ is gitignored): never paste them into git.
"""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool


def main() -> None:
    out, cell, sample_id = sys.argv[1:4]
    for log_path in sorted((Path(out) / "logs" / cell).glob("*.eval")):
        log = read_eval_log(str(log_path))
        for sample in log.samples or []:
            if str(sample.id) != sample_id:
                continue
            meta = sample.metadata
            print(f"== {cell} {sample_id}: condition {meta['condition']}, note {meta['note_file']}")
            print(f"   requested line: {meta['line']}")
            for message in sample.messages:
                if isinstance(message, ChatMessageAssistant):
                    if message.text.strip():
                        print(f"-- assistant: {message.text.strip()[:600]}")
                    for call in message.tool_calls or []:
                        print(f">> {call.function}: {str(call.arguments)[:600]}")
                elif isinstance(message, ChatMessageTool):
                    print(f"<< {message.text[:400]}")
            for name, score in (sample.scores or {}).items():
                print(f"== {name}: {score.value}")
            return
    sys.exit(f"sample {sample_id!r} not found under {out}/logs/{cell}")


if __name__ == "__main__":
    main()
