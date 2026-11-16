"""Probe selection uses exact Git trees, independently of PR diff size."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.ci_changes import probes_required


TOOL = Path(__file__).resolve().parents[1] / "tools" / "ci_changes.py"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True, stderr=subprocess.PIPE
    ).strip()


def _commit(repository: Path) -> str:
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "--allow-empty", "-m", "fixture")
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "CI fixture")
    (repo / "README.md").write_text("initial\n")
    _commit(repo)
    return repo


def _pr(base: str, head: str) -> dict:
    return {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}


@pytest.mark.parametrize("path", ["guide.md", "docs/nested.md", "docs/line\nbreak.md"])
def test_markdown_only_changes_skip_probes(repository: Path, path: str) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    head = _commit(repository)
    assert not probes_required(repository, "pull_request", _pr(base, head))


@pytest.mark.parametrize(
    "path", ["module.py", "line\nbreak.py", ".github/workflows/ci.yml"]
)
def test_non_markdown_changes_enable_probes(repository: Path, path: str) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    head = _commit(repository)
    assert probes_required(repository, "push", {"before": base, "after": head})


@pytest.mark.parametrize("suffix,required", [("md", False), ("py", True)])
def test_deleted_files_keep_their_original_classification(
    repository: Path, suffix: str, required: bool
) -> None:
    target = repository / f"deleted.{suffix}"
    target.write_text("deleted\n")
    base = _commit(repository)
    target.unlink()
    head = _commit(repository)
    assert probes_required(repository, "pull_request", _pr(base, head)) is required


@pytest.mark.parametrize(
    "before,after,required",
    [
        ("source.py", "guide.md", True),
        ("guide.md", "source.py", True),
        ("guide.md", "docs\nnew.md", False),
    ],
)
def test_renames_consider_both_original_and_destination_paths(
    repository: Path, before: str, after: str, required: bool
) -> None:
    (repository / before).write_text("unchanged contents\n")
    base = _commit(repository)
    (repository / before).rename(repository / after)
    head = _commit(repository)
    assert probes_required(repository, "pull_request", _pr(base, head)) is required


def test_identical_trees_skip_probes_even_for_distinct_commits(
    repository: Path,
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    head = _commit(repository)
    assert not probes_required(repository, "pull_request", _pr(base, head))
    assert not probes_required(repository, "pull_request", _pr(head, head))


def test_exact_event_revisions_ignore_newer_checkout_changes(repository: Path) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("documentation change\n")
    head = _commit(repository)
    (repository / "unrelated.py").write_text("not in this event\n")
    _commit(repository)
    assert not probes_required(repository, "pull_request", _pr(base, head))


def test_first_push_classifies_the_full_head_tree(repository: Path) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    event = {"before": "0" * 40, "after": head}
    assert not probes_required(repository, "push", event)
    (repository / "source.py").write_text("source\n")
    event["after"] = _commit(repository)
    assert probes_required(repository, "push", event)


def test_large_path_list_is_not_truncated(repository: Path) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    docs = repository / "docs"
    docs.mkdir()
    for index in range(4096):
        (docs / f"page-{index:05}.md").write_text("documentation\n")
    head = _commit(repository)
    assert not probes_required(repository, "pull_request", _pr(base, head))
    (repository / "zz-last-source.py").write_text("source\n")
    head = _commit(repository)
    assert probes_required(repository, "pull_request", _pr(base, head))


@pytest.mark.parametrize("missing_role", ["base", "head"])
def test_missing_exact_event_commit_is_fetched_from_shallow_origin(
    repository: Path, tmp_path: Path, missing_role: str
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("docs only\n")
    head = _commit(repository)
    remote = tmp_path / "origin.git"
    _git(repository, "clone", "--quiet", "--bare", str(repository), str(remote))
    checkout = tmp_path / "shallow"
    _git(repository, "clone", "--quiet", "--depth=1", remote.as_uri(), str(checkout))
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            cwd=checkout,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )
    event = _pr(base, head) if missing_role == "base" else _pr(head, base)
    assert not probes_required(checkout, "pull_request", event)
    assert _git(checkout, "rev-parse", f"{base}^{{commit}}") == base


def test_missing_commits_fail_instead_of_skipping_probes(repository: Path) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(subprocess.CalledProcessError):
        probes_required(repository, "pull_request", _pr("a" * 40, head))


def test_manual_dispatch_always_enables_probes_without_commits(tmp_path: Path) -> None:
    assert probes_required(tmp_path, "workflow_dispatch", {})


def test_actions_entry_point_appends_boolean_output(
    repository: Path, tmp_path: Path
) -> None:
    revision = _git(repository, "rev-parse", "HEAD")
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(_pr(revision, revision)))
    output = tmp_path / "output"
    output.write_text("existing=value\n")
    env = dict(
        os.environ,
        GITHUB_EVENT_PATH=str(payload),
        GITHUB_EVENT_NAME="pull_request",
        GITHUB_OUTPUT=str(output),
    )
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert output.read_text() == "existing=value\nprobes=false\n"
    assert result.stdout.strip() == "Probe tests required: false"


@pytest.mark.parametrize("bad", ["main", "--help", "", None])
def test_event_revisions_must_be_full_commit_ids(repository: Path, bad: object) -> None:
    with pytest.raises(ValueError, match="full base and head commit SHAs"):
        probes_required(repository, "push", {"before": bad, "after": bad})
