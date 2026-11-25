"""The pytest plugin runs a job's tests, all of them, and reports what it read."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.verification import Suite, coverage, find_job
from tools.verification_plugin import ReadAudit, outside_closure

ROOT = Path(__file__).resolve().parents[1]

MODULE = "tests/test_verification_plugin.py"


@pytest.fixture
def opened(tmp_path: Path) -> tuple[set[str], set[str]]:
    """What an audit of a small repository sees."""
    (tmp_path / "docs").mkdir()
    for name in ("docs/page.md", "module.py", "untracked.txt"):
        (tmp_path / name).write_text("x = 1\n")
    (tmp_path / "__pycache__").mkdir()
    compiled = tmp_path / "__pycache__" / "module.cpython-314-pytest-9.2.0.pyc"
    compiled.write_text("")
    audit = ReadAudit()
    (tmp_path / "docs/page.md").read_text()
    compiled.read_bytes()
    (tmp_path / "untracked.txt").read_text()
    os.listdir(tmp_path / "docs")
    return audit.within(tmp_path, frozenset({"docs/page.md", "module.py"}))


def test_the_audit_sees_tracked_files_and_listed_directories(
    opened: tuple[set[str], set[str]],
) -> None:
    files, listed = opened
    assert files == {"docs/page.md", "module.py"}
    assert "docs" in listed


def test_a_gate_that_reads_past_its_closure_is_reported() -> None:
    suite = Suite.load()
    part = suite.parts["gallery pages"]
    inside = {"docs/examples/pcfg.md", "src/quivers/dsl/__init__.py"}
    assert outside_closure(suite, part, inside, {"docs/examples"}) == []
    assert outside_closure(suite, part, {"src/quivers/lsp/server.py"}, set()) == [
        "src/quivers/lsp/server.py"
    ]
    assert outside_closure(suite, part, {"docs/guides/programs.md"}, set()) == [
        "docs/guides/programs.md"
    ]
    assert outside_closure(suite, part, set(), {"docs/guides"}) == ["docs/guides/"]


def test_a_part_every_change_affects_may_read_anything() -> None:
    suite = Suite.load()
    for name in ("library", "documentation blocks"):
        part = suite.parts[name]
        assert outside_closure(suite, part, {"docs/guides/programs.md"}, set()) == []


@pytest.mark.parametrize("workers", ["0", "2"])
def test_a_job_runs_its_tests_and_reports_the_rest(
    tmp_path: Path, workers: str
) -> None:
    suite = Suite.load()
    reports = []
    for index in range(1, suite.parts["library"].jobs + 1):
        job = f"library {index}/{suite.parts['library'].jobs}"
        part, _ = find_job(suite, job)
        done = subprocess.run(
            [
                *(sys.executable, "-m", "pytest", "tests/test_ci_changes.py"),
                *("-p", "tools.verification_plugin", "--verification-job", job),
                *("-p", "no:cacheprovider", "-q", "-n", workers),
                *("-m", "", "--dist", "loadscope"),
            ],
            cwd=ROOT,
            env={**os.environ, "VERIFICATION_REPORTS": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        # pytest exits with 5 when a job holds none of the file's tests.
        assert done.returncode in {0, 5}, done.stdout + done.stderr
        reports = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    owners = reports[0]["owners"]
    assert len(owners) > 20
    assert all(report["owners"] == owners for report in reports)
    ran = [name for report in reports for name in report["ran"]]
    assert sorted(ran) == sorted(owners)
    for report in reports:
        assert all(owners[name] == report["job"] for name in report["ran"])
    problems = coverage(suite, reports, [report["job"] for report in reports])
    assert not [problem for problem in problems if "test_ci_changes" in problem]
