"""marli — multi-agent RL investigations on LLMs.

Importing this package must stay cheap and CPU-only: heavy dependencies
(torch, tinker, vllm, datasets, transformers) are imported lazily inside the
modules that need them. ``tests/test_contracts.py`` enforces this.
"""

__version__ = "0.0.1"
