"""Episode and token generations stay aligned across retries and interrupted compaction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_eval_rollout import json_rows, make_taskset, rollout_config

from marli.eval import store
from marli.interact import records
from marli.rundir import RunDir
from marli.verbs import run_verb


async def test_compaction_recovers_between_store_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "rollout"
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1), record_tokens=True)
    first = await run_verb("eval rollout", cfg, out=out)
    episode, buffers = next(records.read_episodes(out, with_tokens=True))
    first.manifest.unlink()
    with RunDir(
        out, kind="eval rollout", manifest_name="episodes.json", config_hash=first.config_hash
    ) as run:
        records.write_episode(
            run.append_row,
            replace(episode, ok=False, errors=("first failed retry",)),
            buffers,
            record_tokens=True,
        )
        records.write_episode(run.append_row, episode, buffers, record_tokens=True)
        # An uncommitted retry token append must never supersede the saved attempt.
        run.append_row("tokens.jsonl", {"episode_id": episode.episode_id, "segments": {}})
        original = store.atomic_write_text

        def crash(path: Path, text: str) -> None:
            if path.name == "tokens.jsonl":
                raise OSError("interrupted between renames")
            original(path, text)

        with monkeypatch.context() as patch:
            patch.setattr(store, "atomic_write_text", crash)
            with pytest.raises(OSError, match="between renames"):
                store.compact(run, record_tokens=True)
        assert run.path(".episode-compaction.json").exists()
    resumed = await run_verb("eval rollout", cfg, out=out)
    assert resumed.handle.n == 1 and resumed.handle.n_failed == 0
    assert len(json_rows(out / "episodes.jsonl")) == len(json_rows(out / "tokens.jsonl")) == 1
    assert next(records.read_episodes(out, with_tokens=True)) == (episode, buffers)
    assert not (out / ".episode-compaction.json").exists()
