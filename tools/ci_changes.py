"""Read the paths an event changes from its exact commit trees."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

type EventValue = str | int | bool | None | list[EventValue] | dict[str, EventValue]
"""A value of the JSON payload of an event."""


def _revision(value: EventValue) -> str:
    """Require a full commit name before using it as a Git revision.

    Parameters
    ----------
    value : EventValue
        The value an event carries.

    Returns
    -------
    str
        The commit name.

    Raises
    ------
    ValueError
        If the value is not a full SHA-1 or SHA-256 name.
    """
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value
    ):
        raise ValueError("change detection requires full base and head commit SHAs")
    return value


def _ensure_commit(repository: Path, revision: str) -> None:
    """Fetch one exact commit only when the checkout lacks it.

    Parameters
    ----------
    repository : Path
        The checkout.
    revision : str
        The commit.

    Raises
    ------
    subprocess.CalledProcessError
        If the origin has no such commit.
    """
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


def tree_difference(repository: Path, base: str, head: str) -> list[str]:
    """The paths that differ between the trees of two commits.

    Rename detection is off, so a renamed file gives both of its names, and
    names are read NUL-delimited, so one with whitespace, newlines, or bytes
    that are not UTF-8 is kept.

    Parameters
    ----------
    repository : Path
        The checkout.
    base : str
        The first commit.
    head : str
        The second commit.

    Returns
    -------
    list[str]
        Every path that exists in one tree only, or differs.

    Raises
    ------
    subprocess.CalledProcessError
        If a commit can be neither found nor fetched.
    ValueError
        If a commit name is not full.
    """
    for revision in (base, head):
        _ensure_commit(repository, _revision(revision))
    paths = subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", "-z", base, head, "--"],
        cwd=repository,
    )
    return [
        path.decode(errors="surrogateescape") for path in paths.split(b"\0") if path
    ]


def _parents(repository: Path, revision: str) -> list[str]:
    """The parents a commit names, whether or not the checkout has them.

    Parameters
    ----------
    repository : Path
        The checkout.
    revision : str
        The commit.

    Returns
    -------
    list[str]
        The parents, in order.

    Raises
    ------
    subprocess.CalledProcessError
        If the commit can be neither found nor fetched.
    """
    _ensure_commit(repository, revision)
    text = subprocess.check_output(
        ["git", "cat-file", "commit", revision], cwd=repository
    )
    header = text.partition(b"\n\n")[0].decode(errors="replace")
    return re.findall(r"^parent (\w+)$", header, re.MULTILINE)


def changed_paths(
    repository: Path,
    event_name: str,
    event: Mapping[str, EventValue],
    revision: str,
) -> list[str] | None:
    """The paths a run's event changes against a commit of the default branch.

    A pull request is tested at the merge of its head into its base, so what
    it changes is the difference of that merge from its first parent. A run
    that certifies a commit by itself has no base and changes everything.

    Parameters
    ----------
    repository : Path
        The checkout.
    event_name : str
        The event that started the run.
    event : Mapping[str, EventValue]
        The event's payload.
    revision : str
        The commit the run checks out.

    Returns
    -------
    list[str] | None
        The changed paths, or ``None`` when everything counts as changed: for
        a push, a manual or a scheduled run, and for a change whose base is
        not the default branch.

    Raises
    ------
    ValueError
        If the event is of another kind, carries a name that is not a full
        commit name, or its commit is not the merge of its head.
    subprocess.CalledProcessError
        If a commit can be neither found nor fetched.
    """
    if event_name in {"push", "schedule", "workflow_dispatch"}:
        return None
    repository_data = event.get("repository")
    default = (
        repository_data.get("default_branch")
        if isinstance(repository_data, dict)
        else None
    )
    if event_name == "pull_request":
        request = event["pull_request"]
        assert isinstance(request, dict)
        base, head = request["base"], request["head"]
        assert isinstance(base, dict) and isinstance(head, dict)
        tip = _revision(head["sha"])
        if base["ref"] != default:
            return None
        parents = _parents(repository, _revision(revision))
        if len(parents) != 2 or parents[1] != tip:
            raise ValueError(f"{revision} is not the merge of the head {tip}")
        return tree_difference(repository, parents[0], revision)
    if event_name == "merge_group":
        group = event["merge_group"]
        assert isinstance(group, dict)
        if group["base_ref"] != f"refs/heads/{default}":
            return None
        return tree_difference(
            repository, _revision(group["base_sha"]), _revision(group["head_sha"])
        )
    raise ValueError(f"unsupported change-detection event: {event_name!r}")


__all__ = ["changed_paths", "tree_difference"]
