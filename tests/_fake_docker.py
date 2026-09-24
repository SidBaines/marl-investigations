#!/usr/bin/env python3
"""Emulate Docker transport, never Docker isolation, for offline lifecycle tests."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import BinaryIO


def stop(root: Path) -> None:
    for entry in root.glob("process-*"):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(int(entry.name.removeprefix("process-")), signal.SIGKILL)
        entry.unlink(missing_ok=True)


def main(args: list[str]) -> int:
    state = Path(os.environ["FAKE_DOCKER_STATE"])
    state.mkdir(exist_ok=True)
    with (state / "argv.jsonl").open("a") as log:
        log.write(json.dumps(args) + "\n")
    verb, *args = args
    if (state / f"fail-{verb}").exists():
        sys.stderr.write(f"fake {verb} failure\n")
        return 1
    if verb == "run":
        cid = uuid.uuid4().hex
        root = state / cid
        root.mkdir()
        (root / "workspace").mkdir()
        (root / "tmp").mkdir()
        metadata = {
            "workdir": args[args.index("--workdir") + 1],
            "labels": [args[i + 1] for i, arg in enumerate(args) if arg == "--label"],
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
        delay = state / "delay-run"
        if delay.exists():
            import time

            time.sleep(float(delay.read_text()))
        print(cid)
        return 0
    if verb == "ps":
        label = args[args.index("--filter") + 1].removeprefix("label=")
        for root in state.iterdir():
            if (
                root.is_dir()
                and label in json.loads((root / "metadata.json").read_text())["labels"]
            ):
                print(root.name)
        return 0
    if verb == "rm":
        for cid in args[1:]:
            root = state / cid
            if root.exists():
                stop(root)
                shutil.rmtree(root)
        return 0
    if verb == "inspect":
        root = state / args[-1]
        if not root.exists():
            return 1
        print("true")
        return 0
    if verb == "exec":
        interactive = args[0] == "-i"
        if interactive:
            args = args[1:]
        cid, *command = args
        root = state / cid
        if not root.exists():
            sys.stderr.write("No such container\n")
            return 1
        if command[-1] == "marli-start":
            print("2")
            return 0
        if len(command) >= 5 and command[4].startswith("# marli-stop"):
            stop(root)
            if command[-1]:
                marker = root / "tmp" / Path(command[-1]).name
                if marker.exists():
                    print(marker.read_text())
                    marker.unlink()
            return 0
        if (state / "hang-exec").exists() and command[0] == "timeout":
            import time

            (state / "hung-client").write_text(str(os.getpid()))
            time.sleep(30)
        workdir = json.loads((root / "metadata.json").read_text())["workdir"]
        # Only translate container paths in argv; tests use relative paths in
        # model programs. No chroot/root privileges are needed by the fake.
        command = [
            arg.replace(workdir, str(root / "workspace")).replace(
                "/tmp/marli-", str(root / "tmp" / "marli-")
            )
            for arg in command
        ]
        process = subprocess.Popen(
            command,
            cwd=root / "workspace",
            start_new_session=True,
            stdin=None if interactive else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        marker = root / f"process-{process.pid}"
        marker.touch()

        def forward(source: BinaryIO, target: BinaryIO) -> None:
            while data := source.read1(65536):
                target.write(data)
                target.flush()

        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=forward, args=(process.stdout, sys.stdout.buffer)),
            threading.Thread(target=forward, args=(process.stderr, sys.stderr.buffer)),
        ]
        for thread in threads:
            thread.start()
        rc = process.wait()
        for thread in threads:
            thread.join()
        # Keep the pgid until explicit cleanup: a shell can exit leaving children.
        return rc if rc >= 0 else 128 - rc
    if verb == "cp":
        source, destination = args

        def path(value: str) -> Path:
            if ":" not in value:
                return Path(value)
            cid, container_path = value.split(":", 1)
            root = state / cid
            workdir = json.loads((root / "metadata.json").read_text())["workdir"]
            if container_path.startswith("/tmp/"):
                return root / "tmp" / str(Path(container_path).relative_to("/tmp"))
            return root / "workspace" / str(Path(container_path).relative_to(workdir))

        shutil.copy2(path(source), path(destination), follow_symlinks=False)
        return 0
    raise ValueError(f"unhandled fake docker command: {verb}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
