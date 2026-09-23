"""Viewer outputs participate in the same provenance and recovery as other verbs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from marli.config import input_field
from marli.errors import ConfigError
from marli.eval.rollout import EpisodeSet
from marli.handles import Handle, InputRef, atomic_write_text, register_handle
from marli.interact import records
from marli.interact.types import Episode
from marli.rundir import RunDir
from marli.viewer.html import render_html


@dataclass
class ViewConfig:
    episodes: str | None = input_field(
        None, help="episodes dir (eval rollout output) or its manifest"
    )
    episode_ids: list[str] = field(default_factory=list)
    max_episodes: int = 20
    max_chars_per_call: int = 20000
    include_thinking: bool = True

    def __post_init__(self) -> None:
        for name in ("max_episodes", "max_chars_per_call"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")


@register_handle
@dataclass(frozen=True)
class View(Handle):
    KIND: ClassVar[str] = "view"
    MANIFEST: ClassVar[str] = "view.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("html",)

    html: str


async def view(cfg: ViewConfig, run: RunDir) -> View:
    """Render selected rows in store order, stopping as soon as selection is complete."""
    if cfg.episodes is None:
        raise ConfigError("view requires episodes")
    source = EpisodeSet.load(cfg.episodes)
    remaining = set(cfg.episode_ids)
    selected: list[Episode] = []
    episodes = records.read_episodes(source.root, with_tokens=False)
    try:
        for episode, _ in episodes:
            if cfg.episode_ids:
                if episode.episode_id not in remaining:
                    continue
                remaining.remove(episode.episode_id)
            selected.append(episode)
            if (cfg.episode_ids and not remaining) or (
                not cfg.episode_ids and len(selected) == cfg.max_episodes
            ):
                break
    finally:
        episodes.close()
    if remaining:
        raise ConfigError(f"episode_ids not found: {sorted(remaining)}")
    atomic_write_text(
        run.path("index.html"),
        render_html(
            selected,
            max_chars_per_call=cfg.max_chars_per_call,
            include_thinking=cfg.include_thinking,
        ),
    )
    return View(root=run.out, inputs=(InputRef.of(source),), html="index.html")
