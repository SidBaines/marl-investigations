"""Deterministic seed derivation.

Every sampling call gets a seed derived from stable identifiers (run seed,
task id, episode index, agent id, call index, sample index) — never from
wall-clock order — so lockstep episodes with deterministic policies replay
byte-identically and parallel peers with identical prompts still get distinct
samples. The same parts also build API ``cache_salt`` strings.
"""

from __future__ import annotations

import hashlib

_MASK63 = (1 << 63) - 1


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from ``parts`` (order-sensitive; str() of each part)."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        b = str(p).encode("utf-8")
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
    return int.from_bytes(h.digest(), "big") & _MASK63


def cache_salt(*parts: object) -> str:
    """Per-call salt for API policies' request cache (see CLAUDE.md, ChatClient)."""
    return f"{derive_seed(*parts):016x}"
