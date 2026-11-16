"""Classify exact event commit trees without the GitHub pull-request diff API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from collections.abc import Mapping
from typing import Any


def _revision(value: object) -> str:
    """Require a full event SHA before using it as a Git revision."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value
    ):
        raise ValueError("change detection requires full base and head commit SHAs")
    return value


def _ensure_commit(repository: Path, revision: str) -> None:
    """Fetch one exact event commit only when the checkout lacks it."""
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
            cwd=repository,
            check=True,
        )


def probes_required(
    repository: Path, event_name: str, event: Mapping[str, Any]
) -> bool:
    """Enable probes when any changed path is not Markdown, or on manual runs.

    Compare the exact event trees, rather than a moving branch or a merge-base.
    Disabling rename detection includes both old and new paths, so renaming a
    source file to Markdown still requires probes. NUL-delimited Git output
    retains filenames containing whitespace, newlines, or non-UTF-8 bytes.
    Failure to obtain a tree raises instead of silently skipping probes.
    """
    if event_name == "workflow_dispatch":
        return True
    if event_name == "pull_request":
        pull_request = event["pull_request"]
        base = _revision(pull_request["base"]["sha"])
        head = _revision(pull_request["head"]["sha"])
    elif event_name == "push":
        base, head = _revision(event["before"]), _revision(event["after"])
    else:
        raise ValueError(f"unsupported change-detection event: {event_name!r}")
    _ensure_commit(repository, head)
    if set(base) == {"0"}:
        command = ["git", "ls-tree", "--name-only", "-r", "-z", head]
    else:
        _ensure_commit(repository, base)
        command = ["git", "diff", "--no-renames", "--name-only", "-z", base, head, "--"]
    paths = subprocess.check_output(command, cwd=repository)
    return any(not path.endswith(b".md") for path in paths.split(b"\0") if path)


def main() -> None:
    """Write the probe decision to the current Actions step's output file."""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    required = probes_required(Path.cwd(), os.environ["GITHUB_EVENT_NAME"], event)
    decision = str(required).lower()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"probes={decision}\n")
    print(f"Probe tests required: {decision}")


if __name__ == "__main__":
    main()
