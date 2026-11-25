"""The pytest plugin that runs one job of the declared verification.

Loaded with ``-p tools.verification_plugin --verification-job JOB``, it keeps
the tests the job owns and deselects the rest, hands them to the workers unit
by unit, longest first, and writes what it collected, what ran, what was
skipped, and which files of the repository a gate read outside its closure. The
`coverage` command of `tools.verification` checks the reports of a run together.
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from xdist.remote import Producer
from xdist.scheduler import LoadScopeScheduling
from xdist.workermanage import WorkerController

from tools.verification import ROOT, Part, Suite, find_job, matches

REPORT_DIRECTORY = "VERIFICATION_REPORTS"
"""The environment variable naming the directory the report is written to."""


class ReadAudit:
    """The files of the repository a process opens, and the directories it lists.

    Installed as an interpreter audit hook, which sees every ``open`` of the
    process, whichever library makes it.
    """

    def __init__(self) -> None:
        self.opened: set[str] = set()
        self.listed: set[str] = set()
        sys.addaudithook(self._record)

    def _record(self, event: str, arguments: tuple) -> None:
        """Keep the path of one event.

        Parameters
        ----------
        event : str
            The name of the audit event.
        arguments : tuple
            Its arguments.
        """
        if event == "open":
            kept = self.opened
        elif event in {"os.listdir", "os.scandir"}:
            kept = self.listed
        else:
            return
        path = arguments[0] if arguments else None
        if isinstance(path, str | bytes | os.PathLike):
            kept.add(os.path.abspath(os.fsdecode(path)))

    def within(self, root: Path, files: frozenset[str]) -> tuple[set[str], set[str]]:
        """The repository's part of what was recorded.

        Parameters
        ----------
        root : Path
            The repository.
        files : frozenset[str]
            The files it tracks.

        Returns
        -------
        tuple[set[str], set[str]]
            The tracked files opened, a compiled module counting as its
            source, and the directories listed, relative to the root.
        """
        prefix = f"{root}{os.sep}"
        opened: set[str] = set()
        for path in self.opened:
            if path.startswith(prefix):
                name = path.removeprefix(prefix).replace(os.sep, "/")
                head, _, tail = name.rpartition("/")
                if head.rpartition("/")[2] == "__pycache__":
                    folder = head.rpartition("/")[0]
                    stem = tail.partition(".cpython-")[0]
                    name = f"{folder}/{stem}.py" if folder else f"{stem}.py"
                if name in files:
                    opened.add(name)
        listed = {
            path.removeprefix(prefix).replace(os.sep, "/")
            for path in self.listed
            if path.startswith(prefix)
        }
        return opened, listed


def outside_closure(
    suite: Suite, part: Part, opened: set[str], listed: set[str]
) -> list[str]:
    """What a gate read that a change to could pass it by.

    Parameters
    ----------
    suite : Suite
        The suite.
    part : Part
        The part whose tests ran.
    opened : set[str]
        The tracked files they opened.
    listed : set[str]
        The directories they listed.

    Returns
    -------
    list[str]
        Each file outside the gate's closure, and each directory in which a
        new file would be inert and would not be one the gate reads. Empty for
        a bucket, which runs on every change.
    """
    if not part.gate or part.everything:
        return []
    outside = sorted(opened - suite.closure(part))
    for directory in sorted(listed):
        new = f"{directory}/new-file"
        if matches(new, suite.patterns("inert")) and not part.declares(new):
            outside.append(f"{directory}/")
    return outside


class UnitScheduling(LoadScopeScheduling):
    """Hand out whole units, in the order of the collection.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration.
    log : Producer | None
        The scheduler's log.
    suite : Suite
        The suite.
    part : Part
        The part whose tests run.
    """

    def __init__(
        self, config: pytest.Config, log: Producer | None, suite: Suite, part: Part
    ) -> None:
        super().__init__(config, log)
        self._suite = suite
        self._part = part
        config.option.loadscopereorder = False

    def _split_scope(self, nodeid: str) -> str:
        """The unit of a test.

        Parameters
        ----------
        nodeid : str
            The test.

        Returns
        -------
        str
            Its unit.
        """
        return self._suite.unit_of(self._part.name, nodeid)


class JobPlugin:
    """Select, schedule, and report the tests of one job.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration.
    suite : Suite
        The suite.
    name : str
        The job.
    audit : ReadAudit
        The record of what the process opened.
    """

    def __init__(
        self, config: pytest.Config, suite: Suite, name: str, audit: ReadAudit
    ) -> None:
        self._config = config
        self._suite = suite
        self._name = name
        self._audit = audit
        self._part, self._index = find_job(suite, name)
        self._owners: dict[str, str] = {}
        self._ran: OrderedDict[str, float] = OrderedDict()
        self._skipped: dict[str, str] = {}
        self._outside: set[str] = set()

    # -- every process ------------------------------------------------------

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        """Keep the job's tests, its longest units first.

        Parameters
        ----------
        config : pytest.Config
            The session's configuration.
        items : list[pytest.Item]
            The collected tests, replaced in place.

        Raises
        ------
        tools.verification.UnclaimedTest
            If no part claims a collected test.
        """
        kept: list[pytest.Item] = []
        dropped: list[pytest.Item] = []
        for item in items:
            markers = frozenset(mark.name for mark in item.iter_markers())
            owner = self._suite.owner(item.nodeid, markers)
            self._owners[item.nodeid] = owner
            (kept if owner == self._name else dropped).append(item)
        order: dict[str, int] = {}
        units = [self._suite.unit_of(self._part.name, item.nodeid) for item in kept]
        for unit in units:
            order.setdefault(unit, len(order))
        ranked = sorted(
            range(len(kept)),
            key=lambda position: (
                -self._suite.seconds(self._part.name, units[position]),
                order[units[position]],
                position,
            ),
        )
        items[:] = [kept[position] for position in ranked]
        config.hook.pytest_deselected(items=dropped)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self) -> None:
        """Report what this process saw: to the controller, or to the file."""
        opened, listed = self._audit.within(ROOT, self._suite.files)
        for name, module in list(sys.modules.items()):
            if name.partition(".")[0] == "quivers":
                origin = getattr(module, "__file__", None)
                if isinstance(origin, str) and origin.startswith(f"{ROOT}{os.sep}"):
                    opened.add(origin.removeprefix(f"{ROOT}{os.sep}"))
        output = getattr(self._config, "workeroutput", None)
        if output is not None:
            output["verification"] = json.dumps(
                {
                    "owners": self._owners,
                    "outside": outside_closure(self._suite, self._part, opened, listed),
                }
            )
            return
        if self._config.getoption("numprocesses") is None:
            self._outside.update(
                outside_closure(self._suite, self._part, opened, listed)
            )
        self._write()

    # -- the controller -----------------------------------------------------

    def pytest_xdist_make_scheduler(
        self, config: pytest.Config, log: Producer
    ) -> UnitScheduling:
        """Schedule the job by units.

        Parameters
        ----------
        config : pytest.Config
            The session's configuration.
        log : Producer
            The scheduler's log.

        Returns
        -------
        UnitScheduling
            The scheduler.
        """
        return UnitScheduling(config, log, self._suite, self._part)

    def pytest_testnodedown(self, node: WorkerController) -> None:
        """Take in what a worker saw.

        Parameters
        ----------
        node : WorkerController
            The worker that finished.

        Raises
        ------
        RuntimeError
            If two workers placed one test differently.
        """
        output = cast("dict[str, str]", getattr(node, "workeroutput", {}))
        if "verification" not in output:
            return
        seen = json.loads(output["verification"])
        for nodeid, owner in seen["owners"].items():
            if self._owners.setdefault(nodeid, owner) != owner:
                raise RuntimeError(f"two workers place {nodeid} differently")
        self._outside.update(seen["outside"])

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count one phase of a test.

        Parameters
        ----------
        report : pytest.TestReport
            The phase.
        """
        spent = self._ran.get(report.nodeid, 0.0)
        self._ran[report.nodeid] = round(spent + report.duration, 3)
        if report.skipped and not hasattr(report, "wasxfail"):
            reason = report.longrepr
            self._skipped[report.nodeid] = (
                str(reason[2]) if isinstance(reason, tuple) else str(reason)
            )

    def _write(self) -> None:
        """Write the job's report, when a directory for it is named."""
        directory = os.environ.get(REPORT_DIRECTORY)
        if not directory:
            return
        Path(directory).mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "-" for c in self._name.lower())
        (Path(directory) / f"{slug}.json").write_text(
            json.dumps(
                {
                    "job": self._name,
                    "files": self._suite.files_of(self._part, self._index),
                    "owners": self._owners,
                    "ran": self._ran,
                    "skipped": self._skipped,
                    "outside": sorted(self._outside),
                }
            )
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the option that names the job.

    Parameters
    ----------
    parser : pytest.Parser
        The option parser.
    """
    parser.addoption("--verification-job", help="the declared job to run")


_AUDIT = ReadAudit()


def pytest_configure(config: pytest.Config) -> None:
    """Install the job's plugin.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration.

    Raises
    ------
    pytest.UsageError
        If no job is named.
    """
    name = config.getoption("--verification-job")
    if not name:
        raise pytest.UsageError("tools.verification_plugin needs --verification-job")
    config.pluginmanager.register(JobPlugin(config, Suite.load(), name, _AUDIT))


__all__: Sequence[str] = [
    "JobPlugin",
    "ReadAudit",
    "UnitScheduling",
    "outside_closure",
]
