"""Derive CI's test jobs from ``.config/verification.toml`` and check them.

::

    verification.py plan               job matrices of a run, for the event it
                                       answers, as Actions outputs
    verification.py run JOB [ARG...]   run the tests of one job with pytest
    verification.py check              the declaration fits the repository
    verification.py coverage DIR JOB.. every test ran once in the job owning it
    verification.py record DIR         rewrite the recorded durations from the
                                       reports of a run
    verification.py prune              drop the recorded durations of tests
                                       that are gone

A *part* is a gate or a bucket of the declaration, and a *job* is one of the
pieces a part is divided into, named ``part i/n``. A *unit* is the group of
tests of a part that is never divided further: a whole test file, one function
of a long file, one hash class of a long parametrized function, one long test,
or one page of a sequence. Which job holds a test is a function of the test's
name and of two committed files alone, so every job of a run divides the suite
the same way without exchanging anything.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
import zlib
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ci_changes import changed_paths  # noqa: E402
from tools.import_graph import ImportGraph, module_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ".config/verification.toml"
DURATIONS = ".config/test-durations.json"

LONG_TEST = 20.0
"""The seconds from which one test is a unit of its own."""

LONG_GROUP = 60.0
"""The seconds from which a function, or a file, is divided into units."""

TOOL_USE = re.compile(r"\brequires?_(?:tool|scheme)\b|\b_webppl\b")
"""How a test file says it needs an external language tool."""

type Table = dict[str, str | int | bool | list[str]]
"""One table of the declaration."""


class UnclaimedTest(Exception):
    """A test carries a gate's marker and no gate claims it."""


def matches(path: str, patterns: Iterable[str]) -> bool:
    """Whether a path fits one of some patterns.

    Parameters
    ----------
    path : str
        The path, relative to the repository.
    patterns : Iterable[str]
        Shell patterns, in which ``*`` also crosses directories.

    Returns
    -------
    bool
        Whether any pattern fits the whole path.
    """
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def test_file(nodeid: str) -> str:
    """The file of a test.

    Parameters
    ----------
    nodeid : str
        The test's pytest name.

    Returns
    -------
    str
        The path before the first ``::``.
    """
    return nodeid.partition("::")[0]


def test_function(nodeid: str) -> str:
    """The function of a test, without its parameters.

    Parameters
    ----------
    nodeid : str
        The test's pytest name.

    Returns
    -------
    str
        The name up to the first bracket.
    """
    return nodeid.partition("[")[0]


def parameter_words(nodeid: str) -> frozenset[str]:
    """The words among a test's parameters.

    Parameters
    ----------
    nodeid : str
        The test's pytest name.

    Returns
    -------
    frozenset[str]
        The dash-separated parts of the bracketed parameters.
    """
    return frozenset(nodeid.partition("[")[2].removesuffix("]").split("-"))


def _under(nodeid: str, selectors: Iterable[str]) -> bool:
    """Whether a test stands under one of some selectors.

    Parameters
    ----------
    nodeid : str
        The test's pytest name.
    selectors : Iterable[str]
        Files, directories, classes, or functions.

    Returns
    -------
    bool
        Whether a selector is the test's function or holds it.
    """
    function = test_function(nodeid)
    return any(
        function == selector or function.startswith((f"{selector}::", f"{selector}/"))
        for selector in selectors
    )


class Part:
    """A gate or a bucket of the declaration.

    Parameters
    ----------
    table : Table
        The declared table.
    gate : bool
        Whether the part is a gate.
    groups : Mapping[str, Sequence[str]]
        The files of each declared group, by name.
    others : Sequence[str]
        The patterns the other buckets list.
    """

    def __init__(
        self,
        table: Table,
        *,
        gate: bool,
        groups: Mapping[str, Sequence[str]] = MappingProxyType({}),
        others: Sequence[str] = (),
    ) -> None:
        self.others = tuple(others)
        self.gate = gate
        self.name = str(table["name"])
        self.jobs = int(str(table["jobs"]))
        self.workers = str(table.get("workers", 4))
        self.tools = str(table.get("tools", ""))
        self.marker = str(table.get("marker", ""))
        self.everything = bool(table.get("everything", False))
        self.tests = tuple(
            test
            for selector in self._strings(table, "tests")
            for test in (
                groups[selector[1:]] if selector.startswith("@") else (selector,)
            )
        )
        self.files = self._strings(table, "files")
        self.parameters = self._strings(table, "parameters")
        self.also = self._strings(table, "also")
        self.reads = self._strings(table, "reads")
        self.runs = self._strings(table, "runs")
        self.unread = self._strings(table, "unread")
        self.images = self._strings(table, "images")

    @staticmethod
    def _strings(table: Table, key: str) -> tuple[str, ...]:
        """A list of strings of a declared table.

        Parameters
        ----------
        table : Table
            The table.
        key : str
            The key.

        Returns
        -------
        tuple[str, ...]
            The strings, none when the table has no such key.
        """
        value = table.get(key, [])
        return tuple(value) if isinstance(value, list) else ()

    def declares(self, path: str) -> bool:
        """Whether the part declares that it reads a path.

        Parameters
        ----------
        path : str
            The path, relative to the repository.

        Returns
        -------
        bool
            Whether the path fits ``reads`` or ``runs`` and not ``unread``.
        """
        return matches(path, (*self.reads, *self.runs)) and not matches(
            path, self.unread
        )

    def claims(self, nodeid: str, markers: frozenset[str]) -> bool:
        """Whether this part claims a test no earlier part claimed.

        Parameters
        ----------
        nodeid : str
            The test's pytest name.
        markers : frozenset[str]
            The test's markers.

        Returns
        -------
        bool
            For a gate, whether the test has its marker, stands under one of
            its selectors, and has one of its parameter words or stands under
            one of the selectors that need none. For a bucket,
            whether the test's file is listed, or the bucket lists none.
        """
        if not self.gate:
            return self.may_hold(test_file(nodeid))
        if self.marker and self.marker not in markers:
            return False
        if self.parameters and not parameter_words(nodeid) & set(self.parameters):
            return _under(nodeid, self.also)
        return _under(nodeid, self.tests)

    def may_hold(self, path: str) -> bool:
        """Whether a test file can hold a test of this part.

        Parameters
        ----------
        path : str
            The test file.

        Returns
        -------
        bool
            Whether a selector of a gate stands in the file or above it, or a
            bucket lists the file. The bucket that lists no files holds those
            that no other bucket lists.
        """
        if not self.gate:
            listed = self.files or self.others
            return matches(path, listed) == bool(self.files)
        return any(
            test_file(selector) == path or path.startswith(f"{selector}/")
            for selector in self.tests
        )


class Suite:
    """The declared parts, the recorded durations, and the test files.

    Parameters
    ----------
    declaration : Mapping[str, list[Table] | dict[str, list[str]]]
        The parsed declaration.
    durations : Mapping[str, dict[str, dict[str, float]]]
        For each part, ``units``, the recorded seconds of each unit, and
        ``chunks``, how many hash classes each long function has.
    files : Iterable[str]
        Every file of the repository, relative to its root.
    """

    def __init__(
        self,
        declaration: Mapping[str, list[Table] | dict[str, list[str]]],
        durations: Mapping[str, dict[str, dict[str, float]]],
        files: Iterable[str],
    ) -> None:
        self.declaration = declaration
        self.durations = durations
        self.files = frozenset(files)
        groups = {
            str(table["name"]): table["files"]
            for table in self._tables("group")
            if isinstance(table["files"], list)
        }
        self.gates = [
            Part(table, gate=True, groups=groups) for table in self._tables("gate")
        ]
        listed = [
            pattern
            for table in self._tables("bucket")
            if isinstance(patterns := table.get("files", []), list)
            for pattern in patterns
        ]
        self.buckets = [
            Part(table, gate=False, others=listed) for table in self._tables("bucket")
        ]
        self.parts = {part.name: part for part in (*self.gates, *self.buckets)}
        self.sequences = [
            (str(table["tests"]), re.compile(str(table["key"])))
            for table in self._tables("sequence")
        ]

    @classmethod
    def load(cls, root: Path = ROOT) -> Suite:
        """Read the suite of a checkout.

        Parameters
        ----------
        root : Path
            The checkout.

        Returns
        -------
        Suite
            Its suite.
        """
        with (root / DECLARATION).open("rb") as stream:
            declaration = tomllib.load(stream)
        recorded = root / DURATIONS
        durations = json.loads(recorded.read_text()) if recorded.exists() else {}
        listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, text=True)
        files = [path for path in listed.split("\0") if path and (root / path).exists()]
        return cls(declaration, durations, files)

    def _tables(self, key: str) -> list[Table]:
        """The declared tables under one key.

        Parameters
        ----------
        key : str
            The key.

        Returns
        -------
        list[Table]
            The tables, in declaration order.
        """
        value = self.declaration.get(key, [])
        return value if isinstance(value, list) else []

    def patterns(self, section: str, key: str = "files") -> tuple[str, ...]:
        """The strings listed by one section.

        Parameters
        ----------
        section : str
            The section.
        key : str
            The key of the list.

        Returns
        -------
        tuple[str, ...]
            The strings.
        """
        value = self.declaration.get(section, {})
        return tuple(value.get(key, [])) if isinstance(value, dict) else ()

    @cached_property
    def test_files(self) -> tuple[str, ...]:
        """The files pytest collects tests from.

        Returns
        -------
        tuple[str, ...]
            Every ``test_*.py`` and ``*_test.py`` under ``tests``, sorted.
        """
        return tuple(
            sorted(
                path
                for path in self.files
                if path.startswith("tests/")
                and fnmatch.fnmatchcase(Path(path).name, "test_*.py")
                or path.startswith("tests/")
                and fnmatch.fnmatchcase(Path(path).name, "*_test.py")
            )
        )

    @cached_property
    def markers(self) -> frozenset[str]:
        """The markers gates claim tests by.

        Returns
        -------
        frozenset[str]
            Each declared marker.
        """
        return frozenset(gate.marker for gate in self.gates if gate.marker)

    # -- which job holds a test ---------------------------------------------

    def part_of(self, nodeid: str, markers: frozenset[str]) -> Part:
        """The part that owns a test.

        Parameters
        ----------
        nodeid : str
            The test's pytest name.
        markers : frozenset[str]
            The test's markers.

        Returns
        -------
        Part
            The first gate that claims it, or else the first bucket.

        Raises
        ------
        UnclaimedTest
            If the test carries a gate's marker and no gate claims it, or no
            bucket lists its file.
        """
        for gate in self.gates:
            if gate.claims(nodeid, markers):
                return gate
        stray = sorted(markers & self.markers)
        if stray:
            raise UnclaimedTest(f"{nodeid} is marked {stray} and no gate claims it")
        for bucket in self.buckets:
            if bucket.claims(nodeid, markers):
                return bucket
        raise UnclaimedTest(f"no gate and no bucket claims {nodeid}")

    def unit_of(self, part: str, nodeid: str) -> str:
        """The unit of a part that holds a test.

        Parameters
        ----------
        part : str
            The part owning the test.
        nodeid : str
            The test's pytest name.

        Returns
        -------
        str
            The page of a sequence, or the longest recorded unit the test's
            name falls under, or its file.
        """
        path = test_file(nodeid)
        for tests, key in self.sequences:
            if tests == path:
                found = key.search(nodeid)
                return f"{path}[{found.group(1)}]" if found else path
        recorded = self.durations.get(part, {})
        units = recorded.get("units", {})
        function = test_function(nodeid)
        chunks = int(recorded.get("chunks", {}).get(function, 0))
        if nodeid in units:
            return nodeid
        if chunks:
            return f"{function}#{zlib.crc32(nodeid.encode()) % chunks}"
        return function if function in units else path

    @cached_property
    def _placed(self) -> dict[str, dict[str, int]]:
        """The job of every recorded unit, by part.

        Returns
        -------
        dict[str, dict[str, int]]
            For each part, each recorded unit's job, counted from one. The
            longest unit goes first, each to the job with the least time.
        """
        placed: dict[str, dict[str, int]] = {}
        for part in self.parts.values():
            load = [0.0] * part.jobs
            jobs: dict[str, int] = {}
            for unit, seconds in sorted(
                self.durations.get(part.name, {}).get("units", {}).items(),
                key=lambda entry: (-entry[1], entry[0]),
            ):
                lightest = min(range(part.jobs), key=lambda index: (load[index], index))
                load[lightest] += seconds
                jobs[unit] = lightest + 1
            placed[part.name] = jobs
        return placed

    def job_of(self, part: str, unit: str) -> int:
        """The job of a part that runs a unit.

        Parameters
        ----------
        part : str
            The part.
        unit : str
            The unit.

        Returns
        -------
        int
            The job, counted from one: the balanced one for a recorded unit,
            and one fixed by the unit's name for any other.
        """
        placed = self._placed[part].get(unit)
        if placed is not None:
            return placed
        return zlib.crc32(unit.encode()) % self.parts[part].jobs + 1

    def owner(self, nodeid: str, markers: frozenset[str]) -> str:
        """The name of the job that runs a test.

        Parameters
        ----------
        nodeid : str
            The test's pytest name.
        markers : frozenset[str]
            The test's markers.

        Returns
        -------
        str
            The job's name.

        Raises
        ------
        UnclaimedTest
            If no part claims the test.
        """
        part = self.part_of(nodeid, markers)
        index = self.job_of(part.name, self.unit_of(part.name, nodeid))
        return job_name(part, index)

    def seconds(self, part: str, unit: str) -> float:
        """The recorded seconds of a unit.

        Parameters
        ----------
        part : str
            The part.
        unit : str
            The unit.

        Returns
        -------
        float
            The recorded time, zero for a unit without one.
        """
        return float(self.durations.get(part, {}).get("units", {}).get(unit, 0.0))

    def files_of(self, part: Part, index: int) -> list[str]:
        """The test files that can hold a test of one job.

        Parameters
        ----------
        part : Part
            The part.
        index : int
            The job, counted from one.

        Returns
        -------
        list[str]
            The files to collect, sorted.
        """
        ordered = {tests for tests, _ in self.sequences}
        recorded = self.durations.get(part.name, {})
        units = {
            *recorded.get("units", {}),
            *(
                f"{function}#{chunk}"
                for function, count in recorded.get("chunks", {}).items()
                for chunk in range(int(count))
            ),
        }
        held = {
            test_file(unit).partition("[")[0]
            for unit in units
            if self.job_of(part.name, unit) == index
        }
        return [
            path
            for path in self.test_files
            if part.may_hold(path)
            and (
                path in held or path in ordered or self.job_of(part.name, path) == index
            )
        ]

    # -- what a change affects ----------------------------------------------

    @cached_property
    def graph(self) -> ImportGraph:
        """The repository's import graph.

        Returns
        -------
        ImportGraph
            The graph over the suite's files.
        """
        return ImportGraph(ROOT, self.files)

    def closure(self, gate: Part) -> frozenset[str]:
        """The files that can affect a gate.

        Parameters
        ----------
        gate : Part
            The gate.

        Returns
        -------
        frozenset[str]
            The files its tests can import, with the package data beside them;
            the files it declares that it reads or runs, with the modules that
            the text of those it runs names; and the files that affect every
            gate.
        """
        if gate.everything:
            return self.files
        read = {path for path in self.files if gate.declares(path)}
        names = {
            name
            for path in self.test_files
            if gate.may_hold(path) and (name := module_name(path)) is not None
        }
        for path in read:
            if matches(path, gate.runs):
                names |= self.graph.named_in((ROOT / path).read_text(encoding="utf-8"))
        shared = {
            path for path in self.files if matches(path, self.patterns("everything"))
        }
        return self.graph.closure_files(names) | read | shared

    def affected(self, changed: Iterable[str]) -> set[str]:
        """The gates a change affects.

        Parameters
        ----------
        changed : Iterable[str]
            The changed paths, with both names of a renamed file.

        Returns
        -------
        set[str]
            The names of the gates to run. A path that is in no gate's
            closure and fits no line of ``[inert]`` affects every gate.
        """
        closures = {gate.name: self.closure(gate) for gate in self.gates}
        everything = {gate.name for gate in self.gates}
        affected: set[str] = set()
        for path in changed:
            # A removed file is in no closure computed from what remains.
            reaching = {
                gate.name
                for gate in self.gates
                if path in closures[gate.name] or gate.declares(path)
            }
            attributed = path in self.files and bool(reaching)
            if matches(path, self.patterns("everything")):
                return everything
            if not attributed and not matches(path, self.patterns("inert")):
                if not self._only_ordinary(path):
                    return everything
            affected |= reaching
        return affected

    def _only_ordinary(self, path: str) -> bool:
        """Whether only ordinary tests can depend on a changed file.

        Parameters
        ----------
        path : str
            A changed path in no gate's closure.

        Returns
        -------
        bool
            Whether the path is a module the import graph holds, which every
            gate's closure was computed over and left out.
        """
        return path in self.files and (
            module_name(path) is not None or self.graph.owner(path) is not None
        )


def job_name(part: Part, index: int) -> str:
    """The name of one job of a part.

    Parameters
    ----------
    part : Part
        The part.
    index : int
        The job, counted from one.

    Returns
    -------
    str
        The name.
    """
    return f"{part.name} {index}/{part.jobs}"


def find_job(suite: Suite, name: str) -> tuple[Part, int]:
    """Read a job's name.

    Parameters
    ----------
    suite : Suite
        The suite.
    name : str
        The name.

    Returns
    -------
    tuple[Part, int]
        The part and the job's number.

    Raises
    ------
    SystemExit
        If the suite has no such job.
    """
    part_name, _, fraction = name.rpartition(" ")
    index, _, count = fraction.partition("/")
    part = suite.parts.get(part_name)
    if part is None or count != str(part.jobs) or not 1 <= int(index) <= part.jobs:
        sys.exit(f"no job {name!r}; the declaration has {list(suite.parts)}")
    return part, int(index)


def matrix(suite: Suite, parts: Iterable[Part]) -> list[dict[str, str]]:
    """The job matrix of some parts.

    Parameters
    ----------
    suite : Suite
        The suite.
    parts : Iterable[Part]
        The parts.

    Returns
    -------
    list[dict[str, str]]
        One entry per job: its name, a name usable in a file, the tools and
        the images it needs.
    """
    return [
        {
            "name": job_name(part, index),
            "slug": re.sub(r"[^a-z0-9]+", "-", job_name(part, index).lower()),
            "tools": part.tools,
            "images": " ".join(part.images),
        }
        for part in parts
        for index in range(1, part.jobs + 1)
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def plan(suite: Suite, selected: set[str] | None) -> dict[str, str]:
    """The outputs of the job that plans a run.

    Parameters
    ----------
    suite : Suite
        The suite.
    selected : set[str] | None
        The gates to run, or ``None`` for all of them.

    Returns
    -------
    dict[str, str]
        ``tests``, the matrix of the jobs that need no container; ``probes``,
        the matrix of those that do; and ``jobs``, the name of every job.
    """
    gates = [gate for gate in suite.gates if selected is None or gate.name in selected]
    tests = matrix(suite, [*suite.buckets, *(g for g in gates if not g.images)])
    probes = matrix(suite, [gate for gate in gates if gate.images])
    return {
        "tests": json.dumps(tests),
        "probes": json.dumps(probes),
        "jobs": json.dumps([entry["name"] for entry in (*tests, *probes)]),
    }


def run(suite: Suite, name: str, arguments: Sequence[str]) -> int:
    """Run the tests of one job.

    Parameters
    ----------
    suite : Suite
        The suite.
    name : str
        The job's name.
    arguments : Sequence[str]
        Further arguments for pytest.

    Returns
    -------
    int
        pytest's exit code.
    """
    part, index = find_job(suite, name)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *suite.files_of(part, index),
        # The plugin keeps the job's tests; a marker expression would hide the
        # others from the report of what was collected.
        *("-m", ""),
        *("-p", "tools.verification_plugin", "--verification-job", name),
        *("-n", part.workers, "--dist", "loadscope", "--max-worker-restart", "0"),
        *("-q", "-rsxX", "--durations", "15"),
        *arguments,
    ]
    print(f"{name}: {len(suite.files_of(part, index))} test files", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def check(suite: Suite) -> list[str]:
    """Check that the declaration fits the repository.

    Parameters
    ----------
    suite : Suite
        The suite.

    Returns
    -------
    list[str]
        Each problem found.
    """
    problems: list[str] = []
    names = [part.name for part in (*suite.gates, *suite.buckets)]
    problems += [
        f"two parts are named {name!r}" for name in set(names) if names.count(name) > 1
    ]
    rests = [bucket.name for bucket in suite.buckets if not bucket.files]
    if rests != [bucket.name for bucket in suite.buckets[-1:]]:
        problems.append("the last bucket, and no other, lists no files")
    for gate in suite.gates:
        problems += [
            f"gate {gate.name!r} claims {selector}, which is not in the repository"
            for selector in (*gate.tests, *gate.also)
            if not _selector_exists(suite, selector)
        ]
        problems += [
            f"gate {gate.name!r} reads {pattern}, which fits no file"
            for pattern in (*gate.reads, *gate.runs, *gate.unread)
            if not any(matches(path, (pattern,)) for path in suite.files)
        ]
        problems += [
            f"gate {gate.name!r} loads image {image}, which has no Dockerfile"
            for image in gate.images
            if f"tests/transpile/docker/{image}/Dockerfile" not in suite.files
        ]
    for bucket in suite.buckets:
        problems += [
            f"bucket {bucket.name!r} lists {pattern}, which fits no test file"
            for pattern in bucket.files
            if not any(matches(path, (pattern,)) for path in suite.test_files)
        ]
    for section in ("everything", "inert"):
        problems += [
            f"[{section}] lists {pattern}, which fits no file"
            for pattern in suite.patterns(section)
            if not any(matches(path, (pattern,)) for path in suite.files)
        ]
    problems += [
        f"{path} is both inert and in [everything]"
        for path in sorted(suite.files)
        if matches(path, suite.patterns("inert"))
        and matches(path, suite.patterns("everything"))
    ]
    problems += [
        f"[[sequence]] names {tests}, which is not a test file"
        for tests, _ in suite.sequences
        if tests not in suite.test_files
    ]
    problems += [
        f"durations are recorded for {entry}, which is not a test of {name!r}; "
        "run `python tools/verification.py prune`"
        for name, recorded in suite.durations.items()
        for entry in (*recorded.get("units", {}), *recorded.get("chunks", {}))
        if not recorded_entry_exists(suite, name, entry)
    ]
    tooled = {
        path
        for path in suite.test_files
        if TOOL_USE.search((ROOT / path).read_text(encoding="utf-8"))
    }
    for path in sorted(tooled):
        problems += [
            f"{path} needs the language tools, which part {part.name!r} does not install"
            for part in suite.parts.values()
            if part.may_hold(path)
            and not part.tools
            and not (part.gate and part.marker == "slow")
            and (
                part.gate or part is next(b for b in suite.buckets if b.may_hold(path))
            )
        ]
    return problems


def recorded_entry_exists(suite: Suite, part: str, entry: str) -> bool:
    """Whether the suite still has the tests of a recorded entry.

    Parameters
    ----------
    suite : Suite
        The suite.
    part : str
        The part the entry is recorded for.
    entry : str
        The unit, or the divided function.

    Returns
    -------
    bool
        Whether the part exists, can hold the entry's file, and the file has
        the entry's function.
    """
    return (
        part in suite.parts
        and suite.parts[part].may_hold(test_file(entry).partition("[")[0])
        and _selector_exists(suite, _selector(entry))
    )


def prune(suite: Suite) -> dict[str, dict[str, dict]]:
    """The recorded durations without the entries of tests that are gone.

    Parameters
    ----------
    suite : Suite
        The suite.

    Returns
    -------
    dict[str, dict[str, dict]]
        The durations of the parts and the tests the suite still has.
    """
    return {
        part: {
            kind: {
                entry: value
                for entry, value in recorded.get(kind, {}).items()
                if recorded_entry_exists(suite, part, entry)
            }
            for kind in ("chunks", "units")
        }
        for part, recorded in suite.durations.items()
        if part in suite.parts
    }


def _selector(unit: str) -> str:
    """The file or function a recorded unit stands under.

    Parameters
    ----------
    unit : str
        The unit.

    Returns
    -------
    str
        The unit without its parameters, its hash class, or its page.
    """
    return unit.partition("[")[0].partition("#")[0]


def _selector_exists(suite: Suite, selector: str) -> bool:
    """Whether a file, a directory, or a function of a file exists.

    Parameters
    ----------
    suite : Suite
        The suite.
    selector : str
        The file, the directory, or ``file::function``.

    Returns
    -------
    bool
        Whether the repository has it.
    """
    path, _, inside = selector.partition("::")
    if path not in suite.files:
        return not inside and any(f.startswith(f"{path}/") for f in suite.test_files)
    if not inside:
        return True
    source = (ROOT / path).read_text(encoding="utf-8")
    return all(
        re.search(rf"^\s*(?:async def|def|class) {re.escape(name)}\b", source, re.M)
        for name in inside.split("::")
    )


def coverage(
    suite: Suite, reports: Sequence[Mapping], jobs: Sequence[str]
) -> list[str]:
    """Check that a run left out no test of the jobs it planned.

    Parameters
    ----------
    suite : Suite
        The suite.
    reports : Sequence[Mapping]
        The report of each job, as `tools.verification_plugin` writes it.
    jobs : Sequence[str]
        The jobs the run planned.

    Returns
    -------
    list[str]
        Each problem found: a job without a report, a test file nothing
        collected, a test two jobs place differently, a test that did not run
        once in the job owning it, a test that ran elsewhere, a skip for no
        declared reason, and a file opened outside a gate's closure.
    """
    problems: list[str] = []
    by_job: dict[str, Mapping] = {}
    for report in reports:
        if report["job"] in by_job:
            problems.append(f"two reports of job {report['job']!r}")
        by_job[report["job"]] = report
    problems += [f"no report of job {job!r}" for job in jobs if job not in by_job]
    problems += [f"job {job!r} was not planned" for job in by_job if job not in jobs]

    owners: dict[str, str] = {}
    for report in by_job.values():
        for nodeid, owner in report["owners"].items():
            if owners.setdefault(nodeid, owner) != owner:
                problems.append(
                    f"{nodeid} is placed in {owners[nodeid]!r} and {owner!r}"
                )

    planned = {find_job(suite, job)[0].name for job in jobs}
    collected = {test_file(nodeid) for nodeid in owners}
    empty = {path for report in by_job.values() for path in report["files"]}
    for path in suite.test_files:
        holders = {part.name for part in suite.parts.values() if part.may_hold(path)}
        if path not in collected | empty and holders & planned:
            problems.append(f"no job collected {path}")

    runs: dict[str, list[str]] = {}
    for job, report in by_job.items():
        for nodeid in report["ran"]:
            runs.setdefault(nodeid, []).append(job)
    for nodeid, owner in owners.items():
        if owner in jobs and runs.get(nodeid, []) != [owner]:
            problems.append(
                f"{nodeid} belongs to {owner!r} and ran in {runs.get(nodeid, [])}"
            )
    problems += [
        f"{nodeid} ran in {ran} and no job collected it"
        for nodeid, ran in runs.items()
        if nodeid not in owners
    ]
    problems += [
        f"{nodeid} ran in {ran}, and {owners[nodeid]!r} owns it"
        for nodeid, ran in runs.items()
        if nodeid in owners and owners[nodeid] not in jobs
    ]
    reasons = suite.patterns("skips", "reasons")
    for report in by_job.values():
        problems += [
            f"{nodeid} was skipped for no declared reason: {reason}"
            for nodeid, reason in report["skipped"].items()
            if not any(allowed in reason for allowed in reasons)
        ]
        problems += [
            f"job {report['job']!r} read {path}, which is outside its closure"
            for path in report["outside"]
        ]
    return problems


def record(suite: Suite, reports: Sequence[Mapping]) -> dict[str, dict[str, dict]]:
    """The durations to record from the reports of a run.

    Parameters
    ----------
    suite : Suite
        The suite.
    reports : Sequence[Mapping]
        The report of each job.

    Returns
    -------
    dict[str, dict[str, dict]]
        For each part, its units and their seconds, and how many hash classes
        each long function has. A part no report mentions keeps what was
        recorded.
    """
    tests: dict[str, dict[str, float]] = {}
    for report in reports:
        # A run may have divided the part into another number of jobs.
        part = report["job"].rpartition(" ")[0]
        if part not in suite.parts:
            sys.exit(f"the declaration has no part {part!r}")
        tests.setdefault(part, {}).update(report["ran"])
    recorded = dict(suite.durations)
    ordered = {path: key for path, key in suite.sequences}
    for part, seconds in tests.items():
        units: dict[str, float] = {}
        chunks: dict[str, int] = {}
        files: dict[str, float] = {}
        functions: dict[str, float] = {}
        rest: dict[str, float] = {}
        for nodeid, spent in seconds.items():
            files[test_file(nodeid)] = files.get(test_file(nodeid), 0.0) + spent
            if test_file(nodeid) in ordered:
                unit = suite.unit_of(part, nodeid)
                units[unit] = units.get(unit, 0.0) + spent
            elif spent >= LONG_TEST:
                units[nodeid] = spent
            else:
                rest[nodeid] = spent
                function = test_function(nodeid)
                functions[function] = functions.get(function, 0.0) + spent
        for nodeid, spent in rest.items():
            function = test_function(nodeid)
            path = test_file(nodeid)
            if functions[function] >= LONG_GROUP:
                count = int(functions[function] // LONG_TEST)
                chunks[function] = count
                unit = f"{function}#{zlib.crc32(nodeid.encode()) % count}"
            else:
                unit = function if files[path] >= LONG_GROUP else path
            units[unit] = units.get(unit, 0.0) + spent
        recorded[part] = {
            "chunks": dict(sorted(chunks.items())),
            "units": {
                unit: round(spent, 1)
                for unit, spent in sorted(units.items())
                if spent >= 1.0 or "#" in unit
            },
        }
    return dict(sorted(recorded.items()))


def _reports(directory: str) -> list[Mapping]:
    """Read the reports of a run.

    Parameters
    ----------
    directory : str
        The directory that holds them.

    Returns
    -------
    list[Mapping]
        The parsed reports.
    """
    return [
        json.loads(path.read_text()) for path in sorted(Path(directory).rglob("*.json"))
    ]


def _store(durations: dict[str, dict[str, dict]]) -> int:
    """Write the recorded durations.

    Parameters
    ----------
    durations : dict[str, dict[str, dict]]
        The durations.

    Returns
    -------
    int
        The exit code.
    """
    (ROOT / DURATIONS).write_text(json.dumps(durations, indent=1) + "\n")
    return 0


def _fail(problems: Sequence[str]) -> int:
    """Print problems.

    Parameters
    ----------
    problems : Sequence[str]
        The problems.

    Returns
    -------
    int
        The exit code: one when there is any.
    """
    for problem in problems[:200]:
        print(f"::error::{problem}" if os.environ.get("CI") else f"error: {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _plan_command(suite: Suite) -> int:
    """Plan the run the environment describes.

    Parameters
    ----------
    suite : Suite
        The suite.

    Returns
    -------
    int
        The exit code.
    """
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    changed = changed_paths(
        ROOT, os.environ["GITHUB_EVENT_NAME"], event, os.environ["GITHUB_SHA"]
    )
    selected = None if changed is None else suite.affected(changed)
    outputs = plan(suite, selected)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")
    left = [
        gate.name
        for gate in suite.gates
        if selected is not None and gate.name not in selected
    ]
    print("changed paths: " + ("all" if changed is None else str(len(changed))))
    print(f"gates left out: {left or 'none'}")
    print("\n".join(json.loads(outputs["jobs"])))
    return 0


def main(arguments: Sequence[str]) -> int:
    """Run one command.

    Parameters
    ----------
    arguments : Sequence[str]
        The command and its arguments.

    Returns
    -------
    int
        The exit code.
    """
    match list(arguments):
        case ["plan"]:
            return _plan_command(Suite.load())
        case ["run", name, *further]:
            return run(Suite.load(), name, further)
        case ["check"]:
            return _fail(check(Suite.load()))
        case ["coverage", directory, jobs]:
            return _fail(coverage(Suite.load(), _reports(directory), json.loads(jobs)))
        case ["record", directory]:
            return _store(record(Suite.load(), _reports(directory)))
        case ["prune"]:
            return _store(prune(Suite.load()))
        case _:
            print(__doc__)
            return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
