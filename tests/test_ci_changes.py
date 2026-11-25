"""Change detection reads exact Git trees, independently of pull-request diff size."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.ci_changes import changed_paths, tree_difference


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
    _git(repo, "init", "--quiet", "--initial-branch=main")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "CI fixture")
    (repo / "README.md").write_text("initial\n")
    _commit(repo)
    return repo


def _merge(repository: Path, base: str, head: str) -> str:
    """Merge a head into a base as a pull request's test merge does."""
    tree = _git(repository, "merge-tree", "--write-tree", base, head)
    return _git(repository, "commit-tree", tree, "-p", base, "-p", head, "-m", "merge")


def _pr(head: str, ref: str = "main") -> dict:
    return {
        "pull_request": {"base": {"ref": ref}, "head": {"sha": head}},
        "repository": {"default_branch": "main"},
    }


def _changed(repository: Path, base: str, head: str) -> list[str] | None:
    return changed_paths(
        repository, "pull_request", _pr(head), _merge(repository, base, head)
    )


@pytest.mark.parametrize(
    "path",
    [
        "guide.md",
        "docs/nested.md",
        "docs/line\nbreak.md",
        "module.py",
        "line\nbreak.py",
        ".github/workflows/ci.yml",
    ],
)
def test_an_added_file_is_a_changed_path(repository: Path, path: str) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    head = _commit(repository)
    assert _changed(repository, base, head) == [path]


@pytest.mark.parametrize("suffix", ["md", "py"])
def test_a_deleted_file_is_a_changed_path(repository: Path, suffix: str) -> None:
    target = repository / f"deleted.{suffix}"
    target.write_text("deleted\n")
    base = _commit(repository)
    target.unlink()
    head = _commit(repository)
    assert _changed(repository, base, head) == [f"deleted.{suffix}"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("source.py", "guide.md"),
        ("guide.md", "source.py"),
        ("guide.md", "docs\nnew.md"),
    ],
)
def test_a_rename_changes_both_its_names(
    repository: Path, before: str, after: str
) -> None:
    (repository / before).write_text("unchanged contents\n")
    base = _commit(repository)
    (repository / before).rename(repository / after)
    head = _commit(repository)
    assert _changed(repository, base, head) == sorted([before, after])


def test_identical_trees_change_nothing_even_for_distinct_commits(
    repository: Path,
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    head = _commit(repository)
    assert _changed(repository, base, head) == []
    assert tree_difference(repository, head, head) == []


def test_changes_the_base_gained_meanwhile_are_not_the_pull_requests(
    repository: Path,
) -> None:
    fork = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("pull request\n")
    head = _commit(repository)
    _git(repository, "checkout", "--quiet", "--detach", fork)
    (repository / "source.py").write_text("main moved on\n")
    base = _commit(repository)
    assert _changed(repository, base, head) == ["README.md"]


def test_exact_event_revisions_ignore_newer_checkout_changes(repository: Path) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("documentation change\n")
    head = _commit(repository)
    (repository / "unrelated.py").write_text("not in this event\n")
    _commit(repository)
    assert _changed(repository, base, head) == ["README.md"]


def test_a_large_path_list_is_not_truncated(repository: Path) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    docs = repository / "docs"
    docs.mkdir()
    for index in range(4096):
        (docs / f"page-{index:05}.md").write_text("documentation\n")
    (repository / "zz-last-source.py").write_text("source\n")
    head = _commit(repository)
    changed = _changed(repository, base, head)
    assert changed is not None
    assert len(changed) == 4097
    assert changed[-1] == "zz-last-source.py"


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
    old, new = (base, head) if missing_role == "base" else (head, base)
    assert tree_difference(checkout, old, new) == ["README.md"]
    assert _git(checkout, "rev-parse", f"{base}^{{commit}}") == base


def test_missing_commits_fail_instead_of_counting_as_unchanged(
    repository: Path,
) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(subprocess.CalledProcessError):
        tree_difference(repository, "a" * 40, head)
    with pytest.raises(subprocess.CalledProcessError):
        changed_paths(repository, "pull_request", _pr(head), "a" * 40)


@pytest.mark.parametrize("name", ["push", "schedule", "workflow_dispatch"])
def test_a_run_that_certifies_a_commit_changes_everything(
    tmp_path: Path, name: str
) -> None:
    assert changed_paths(tmp_path, name, {}, "") is None


def test_a_change_to_another_branch_than_the_default_changes_everything(
    repository: Path,
) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    event = _pr(head, ref="feature")
    assert changed_paths(repository, "pull_request", event, head) is None


def test_a_merge_group_changes_what_its_commits_differ_in(repository: Path) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "source.py").write_text("source\n")
    head = _commit(repository)
    event = {
        "merge_group": {
            "base_ref": "refs/heads/main",
            "base_sha": base,
            "head_sha": head,
        },
        "repository": {"default_branch": "main"},
    }
    assert changed_paths(repository, "merge_group", event, head) == ["source.py"]
    event["merge_group"]["base_ref"] = "refs/heads/other"
    assert changed_paths(repository, "merge_group", event, head) is None


def test_a_commit_that_is_not_the_merge_of_the_head_is_refused(
    repository: Path,
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "source.py").write_text("source\n")
    head = _commit(repository)
    (repository / "source.py").write_text("newer\n")
    newer = _commit(repository)
    with pytest.raises(ValueError, match="is not the merge of the head"):
        changed_paths(repository, "pull_request", _pr(head), head)
    with pytest.raises(ValueError, match="is not the merge of the head"):
        changed_paths(
            repository, "pull_request", _pr(head), _merge(repository, base, newer)
        )


@pytest.mark.parametrize("bad", ["main", "--help", "", None])
def test_event_revisions_must_be_full_commit_ids(
    repository: Path, bad: str | None
) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    event = _pr(head)
    event["pull_request"]["head"]["sha"] = bad
    with pytest.raises(ValueError, match="full base and head commit SHAs"):
        changed_paths(repository, "pull_request", event, head)


def test_an_unknown_event_is_refused(repository: Path) -> None:
    with pytest.raises(ValueError, match="unsupported change-detection event"):
        changed_paths(repository, "issue_comment", {}, "")
