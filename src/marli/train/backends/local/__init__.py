"""Local training stays importable without installing a GPU software stack."""

from __future__ import annotations

from marli.train.backends.local.learner import LocalLearner, LocalLearnerPool

__all__ = ["LocalLearner", "LocalLearnerPool"]
