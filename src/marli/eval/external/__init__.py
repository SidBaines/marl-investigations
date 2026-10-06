"""Standard external evals run against our served policies, not through our own harness.

Our interaction layer renders prompts with our renderers, our system prompts and our
turn/delivery structure. Evals of *transfer* must not reintroduce those cues, and must
stay comparable with published numbers, so this package runs benchmarks in their own
harnesses against a vLLM server's OpenAI-compatible chat endpoint (the model's own chat
template, thinking split off by vLLM's reasoning parser):

- ``inspect`` suites are UK AISI Inspect tasks, driven through Inspect's Python API
  inside the ``eval external`` verb (``inspect_ai`` is imported lazily; ``[external]``).
- ``upstream`` suites are a benchmark's own code at a pinned commit. They need their own
  process, so a study's ``run.sh`` runs them (library subprocesses are reserved for
  supervised servers and git) and ``eval external`` ingests their outputs with the
  suite's reader.

Each suite is one YAML file in ``suites/`` (:mod:`marli.eval.external.spec`). Both kinds
end in the same rows (one per sample and condition) and the same report: n, parse
failures, 95% intervals and the gain over a baseline cell (the untrained model).
"""

from __future__ import annotations
