"""The verification script loses no test, and no reason to run one.

Most of these tests are mutations: each breaks one thing about a small suite
or a small import graph, a lost test, a missing dependency, a stale entry, and
requires the check that guards it to say so.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pytest

from tools import verification
from tools.import_graph import ImportGraph, module_name
from tools.verification import (
    Suite,
    UnclaimedTest,
    check,
    coverage,
    job_name,
    plan,
    prune,
    record,
    test_file as file_of,
)

ROOT = Path(__file__).resolve().parents[1]

DECLARATION: dict = {
    "bucket": [
        {
            "name": "languages",
            "files": ["tests/lang/test_*.py"],
            "jobs": 2,
            "tools": "x",
        },
        {"name": "library", "jobs": 3},
    ],
    "gate": [
        {
            "name": "probes of a",
            "marker": "probe",
            "tests": ["@probes"],
            "parameters": ["a"],
            "also": ["tests/lang/test_probe.py::test_reference"],
            "images": ["a"],
            "jobs": 1,
        },
        {"name": "probes", "marker": "probe", "tests": ["@probes"], "jobs": 2},
        {
            "name": "slow",
            "marker": "slow",
            "tests": ["tests/test_slow.py", "tests/test_mixed.py::test_long"],
            "reads": ["docs/pages/*"],
            "unread": ["docs/pages/skipped.md"],
            "jobs": 2,
        },
        {"name": "pages", "tests": ["tests/test_pages.py"], "jobs": 2},
    ],
    "group": [{"name": "probes", "files": ["tests/lang/test_probe.py"]}],
    "sequence": [{"tests": "tests/test_pages.py", "key": r"\[([^\]:]+):"}],
    "everything": {"files": ["uv.lock"]},
    "inert": {"files": ["docs/*"]},
    "skips": {"reasons": ["no accelerator"]},
}

NAMES: dict[str, frozenset[str]] = {
    **{
        f"tests/lang/test_probe.py::test_cell[{target}-{index}]": frozenset({"probe"})
        for target in ("a", "b", "ab")
        for index in range(5)
    },
    "tests/lang/test_probe.py::test_reference": frozenset({"probe"}),
    "tests/lang/test_render.py::test_text[a]": frozenset(),
    **{f"tests/test_slow.py::test_{index}": frozenset({"slow"}) for index in range(9)},
    "tests/test_slow.py::test_quick": frozenset(),
    "tests/test_mixed.py::test_long[x]": frozenset({"slow"}),
    "tests/test_mixed.py::test_short": frozenset(),
    **{
        f"tests/test_pages.py::test_block[{page}.md:{block}]": frozenset()
        for page in ("one", "two", "three", "four")
        for block in range(4)
    },
    **{f"tests/test_unit.py::test_grid[{index}]": frozenset() for index in range(60)},
    **{
        f"tests/test_unit.py::TestClass::test_{index}": frozenset()
        for index in range(4)
    },
    "tests/test_other.py::test_it": frozenset(),
}

DURATIONS: dict = {
    "library": {
        "units": {
            "tests/test_unit.py::test_grid[7]": 40.0,
            **{f"tests/test_unit.py::test_grid#{index}": 20.0 for index in range(4)},
            "tests/test_unit.py::TestClass::test_1": 5.0,
            "tests/test_other.py": 9.0,
        },
        "chunks": {"tests/test_unit.py::test_grid": 4},
    },
    "slow": {"units": {"tests/test_slow.py::test_3": 300.0}},
}

FILES = [
    *sorted({file_of(name) for name in NAMES}),
    "docs/pages/one.md",
    "docs/pages/skipped.md",
    "tests/conftest.py",
    "uv.lock",
]


def _suite(**changes: list[dict]) -> Suite:
    """The small suite, with sections of its declaration replaced."""
    declaration = {**copy.deepcopy(DECLARATION), **changes}
    return Suite(declaration, copy.deepcopy(DURATIONS), FILES)


def _reports(suite: Suite) -> list[dict]:
    """The reports of a run of every job that lost nothing."""
    reports = []
    for part in suite.parts.values():
        for index in range(1, part.jobs + 1):
            job = job_name(part, index)
            files = suite.files_of(part, index)
            owners = {
                name: suite.owner(name, markers)
                for name, markers in NAMES.items()
                if file_of(name) in files
            }
            reports.append(
                {
                    "job": job,
                    "files": files,
                    "owners": owners,
                    "ran": {
                        name: 1.0 for name, owner in owners.items() if owner == job
                    },
                    "skipped": {},
                    "outside": [],
                }
            )
    return reports


def _jobs(suite: Suite) -> list[str]:
    return [
        job_name(part, index)
        for part in suite.parts.values()
        for index in range(1, part.jobs + 1)
    ]


# ---------------------------------------------------------------------------
# Every test has one job, and that job collects it
# ---------------------------------------------------------------------------


def test_every_test_is_collected_and_owned_by_exactly_one_job() -> None:
    suite = _suite()
    jobs = _jobs(suite)
    for name, markers in NAMES.items():
        owner = suite.owner(name, markers)
        assert jobs.count(owner) == 1
        part, index = verification.find_job(suite, owner)
        assert file_of(name) in suite.files_of(part, index)


@pytest.mark.parametrize("jobs", range(1, 7))
@pytest.mark.parametrize("recorded", [True, False])
def test_the_partition_is_total_whatever_is_recorded(jobs: int, recorded: bool) -> None:
    declaration = copy.deepcopy(DECLARATION)
    for table in (*declaration["bucket"], *declaration["gate"]):
        table["jobs"] = jobs
    suite = Suite(declaration, DURATIONS if recorded else {}, FILES)
    assert coverage(suite, _reports(suite), _jobs(suite)) == []


def test_the_declared_partition_of_this_repository_is_total() -> None:
    suite = Suite.load()
    for path in suite.test_files:
        for markers in (frozenset(), frozenset({"slow"}), frozenset({"probe"})):
            for name in (f"{path}::test_new", f"{path}::test_new[a-b:1]"):
                try:
                    owner = suite.owner(name, markers)
                except UnclaimedTest:
                    assert markers
                    continue
                part, index = verification.find_job(suite, owner)
                assert path in suite.files_of(part, index)


def test_an_unrecorded_test_of_a_divided_file_still_has_a_job() -> None:
    suite = _suite()
    name = "tests/test_unit.py::test_added_later"
    part, index = verification.find_job(suite, suite.owner(name, frozenset()))
    assert part.name == "library"
    assert "tests/test_unit.py" in suite.files_of(part, index)


def test_gates_claim_before_buckets_and_in_order() -> None:
    suite = _suite()
    probe = frozenset({"probe"})
    path = "tests/lang/test_probe.py"
    assert suite.part_of(f"{path}::test_cell[a-1]", probe).name == "probes of a"
    assert suite.part_of(f"{path}::test_cell[ab-1]", probe).name == "probes"
    assert suite.part_of(f"{path}::test_reference", probe).name == "probes of a"
    assert suite.part_of(f"{path}::test_reference_two", probe).name == "probes"
    assert suite.part_of(f"{path}::test_cell[a-1]", frozenset()).name == "languages"
    slow = frozenset({"slow"})
    assert suite.part_of("tests/test_mixed.py::test_long[x]", slow).name == "slow"
    assert (
        suite.part_of("tests/test_slow.py::test_quick", frozenset()).name == "library"
    )


def test_no_job_of_one_bucket_collects_a_file_of_another() -> None:
    for suite in (_suite(), Suite.load()):
        *listed, rest = suite.buckets
        for path in suite.test_files:
            holders = [bucket.name for bucket in suite.buckets if bucket.may_hold(path)]
            assert len(holders) == 1, (path, holders)
        for index in range(1, rest.jobs + 1):
            for path in suite.files_of(rest, index):
                assert not any(bucket.may_hold(path) for bucket in listed)


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("tests/test_other.py::test_it", "slow"),
        ("tests/test_mixed.py::test_longer", "slow"),
        ("tests/lang/test_render.py::test_text[a]", "probe"),
    ],
)
def test_a_marked_test_no_gate_claims_is_refused(name: str, marker: str) -> None:
    with pytest.raises(UnclaimedTest, match="no gate claims it"):
        _suite().owner(name, frozenset({marker}))


def test_the_blocks_of_one_page_share_a_unit_and_a_job() -> None:
    suite = _suite()
    units = {
        page: {
            (suite.unit_of("pages", name), suite.owner(name, frozenset()))
            for name in NAMES
            if f"[{page}.md:" in name
        }
        for page in ("one", "two", "three", "four")
    }
    assert all(len(pages) == 1 for pages in units.values())
    assert len(set(itertools.chain(*units.values()))) == 4


def test_the_longest_units_are_spread_over_the_jobs() -> None:
    suite = _suite()
    units = DURATIONS["library"]["units"]
    load = [0.0] * 3
    for unit, seconds in units.items():
        load[suite.job_of("library", unit) - 1] += seconds
    assert max(load) - min(load) <= max(units.values())


# ---------------------------------------------------------------------------
# Coverage of a run
# ---------------------------------------------------------------------------


def test_a_complete_run_covers_the_suite() -> None:
    suite = _suite()
    assert coverage(suite, _reports(suite), _jobs(suite)) == []


def test_a_run_without_an_unaffected_gate_covers_what_it_planned() -> None:
    suite = _suite()
    kept = [job for job in _jobs(suite) if not job.startswith("slow")]
    reports = [item for item in _reports(suite) if item["job"] in kept]
    assert coverage(suite, reports, kept) == []


def _drop_test(reports: list[dict]) -> None:
    next(item for item in reports if item["ran"])["ran"].popitem()


def _run_twice(reports: list[dict]) -> None:
    first, second = [item for item in reports if item["ran"]][:2]
    second["ran"].update(first["ran"])


def _lose_report(reports: list[dict]) -> None:
    reports.remove(next(item for item in reports if item["ran"]))


def _duplicate_report(reports: list[dict]) -> None:
    reports.append(copy.deepcopy(reports[0]))


def _move_test(reports: list[dict]) -> None:
    first, second = [item for item in reports if item["ran"]][:2]
    name, seconds = first["ran"].popitem()
    second["ran"][name] = seconds


def _lose_file(reports: list[dict]) -> None:
    for item in reports:
        item["files"] = [path for path in item["files"] if "other" not in path]
        for name in [name for name in item["owners"] if "other" in name]:
            item["owners"].pop(name)
            item["ran"].pop(name, None)


def _disagree(reports: list[dict]) -> None:
    holders = [item for item in reports if "tests/test_unit.py" in item["files"]]
    name = next(iter(holders[0]["owners"]))
    holders[1]["owners"][name] = "library 9/9"


def _skip(reports: list[dict]) -> None:
    item = next(item for item in reports if item["ran"])
    item["skipped"][next(iter(item["ran"]))] = "could not import a tool"


def _read_outside(reports: list[dict]) -> None:
    reports[0]["outside"].append("docs/other.md")


def _run_unknown(reports: list[dict]) -> None:
    reports[0]["ran"]["tests/test_unit.py::test_unknown"] = 1.0


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_test,
        _run_twice,
        _lose_report,
        _duplicate_report,
        _move_test,
        _lose_file,
        _disagree,
        _skip,
        _read_outside,
        _run_unknown,
    ],
)
def test_a_run_that_lost_something_is_not_covered(mutate) -> None:
    suite = _suite()
    reports = _reports(suite)
    mutate(reports)
    assert coverage(suite, reports, _jobs(suite)) != []


def test_a_skip_for_a_declared_reason_is_covered() -> None:
    suite = _suite()
    reports = _reports(suite)
    item = next(item for item in reports if item["ran"])
    item["skipped"][next(iter(item["ran"]))] = "Skipped: no accelerator here"
    assert coverage(suite, reports, _jobs(suite)) == []


def test_the_plan_names_every_job_and_leaves_out_only_unselected_gates() -> None:
    suite = _suite()
    everything = plan(suite, None)
    assert sorted(verification.json.loads(everything["jobs"])) == sorted(_jobs(suite))
    probes = verification.json.loads(everything["probes"])
    assert [entry["name"] for entry in probes] == ["probes of a 1/1"]
    some = verification.json.loads(plan(suite, {"slow"})["jobs"])
    assert some == [
        *(f"languages {index}/2" for index in (1, 2)),
        *(f"library {index}/3" for index in (1, 2, 3)),
        *(f"slow {index}/2" for index in (1, 2)),
    ]


def test_recorded_durations_divide_what_is_long_and_keep_every_test() -> None:
    suite = Suite(copy.deepcopy(DECLARATION), {}, FILES)
    reports = _reports(suite)
    for item in reports:
        for name in item["ran"]:
            item["ran"][name] = 50.0 if name.endswith("test_3") else 2.0
    durations = record(suite, reports)
    recorded = Suite(copy.deepcopy(DECLARATION), durations, FILES)
    units = durations["library"]["units"]
    assert "tests/test_unit.py::TestClass::test_3" in units
    assert durations["library"]["chunks"] == {"tests/test_unit.py::test_grid": 6}
    assert durations["pages"]["units"]["tests/test_pages.py[one.md]"] == 8.0
    assert coverage(recorded, _reports(recorded), _jobs(recorded)) == []


# ---------------------------------------------------------------------------
# What a change affects
# ---------------------------------------------------------------------------

TREE = {
    "src/quivers/__init__.py": "import quivers.core\n",
    "src/quivers/core.py": "from .data import table\n",
    "src/quivers/data/__init__.py": "",
    "src/quivers/data/table.json": "{}",
    "src/quivers/lazy/__init__.py": (
        "import importlib\n"
        "_EXPORTS = {'thing': 'quivers.lazy.thing', 'near': '.near'}\n"
        "def __getattr__(name):\n"
        "    return importlib.import_module(_EXPORTS[name], __name__)\n"
    ),
    "src/quivers/lazy/thing.py": "",
    "src/quivers/lazy/near.py": "",
    "src/quivers/plain.py": "",
    "src/quivers/text.py": "",
    "src/quivers/text_fragment_one.py": "",
    "src/quivers/text_spoken_two.py": "",
    "src/quivers/shown.py": "",
    "src/quivers/opaque.py": "import importlib\nimportlib.import_module(input())\n",
    "src/quivers/formatted.py": (
        "import importlib\n"
        "def load(name):\n"
        "    return importlib.import_module(f'quivers.family.{name}')\n"
    ),
    "src/quivers/family/__init__.py": "",
    "src/quivers/family/one.py": "",
    "src/quivers/script.py": "",
    "src/quivers/typed.py": "",
    "src/quivers/inside.py": "",
    "src/quivers/unused.py": '"""Mentions quivers.plain in prose."""\n',
    "tests/conftest.py": "import quivers\n",
    "tests/test_lazy.py": "from quivers.lazy import thing\n",
    "tests/test_formatted.py": "import quivers.formatted\n",
    "tests/test_opaque.py": "import quivers.opaque\n",
    "tests/test_data.py": "from quivers import data\n",
    "tests/test_prose.py": "import quivers.unused\n",
    "tests/test_process.py": "CODE = 'import sys; import quivers.script'\n",
    "tests/test_typed.py": (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from quivers.typed import x\n"
    ),
    "tests/test_inside.py": "def test():\n    from quivers import inside\n",
    "tests/test_text.py": (
        "def test(kind, end):\n"
        "    return (open('text.py'), open(f'text_fragment_{kind}.py'),\n"
        "            open(f'text_spoken_{kind}.{end}'), f'{kind}.{end}')\n"
    ),
    "tests/test_page.py": "import pathlib\n",
    "tests/test_text_only.py": "import pathlib\n",
    "docs/page.md": "```python\nfrom quivers.shown import x\n```\n",
    "docs/other.md": "",
    "README.md": "",
    "uv.lock": "",
    "notes.bin": "",
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for path, text in TREE.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    return tmp_path


def _tree_suite(monkeypatch: pytest.MonkeyPatch, tree: Path) -> Suite:
    monkeypatch.setattr(verification, "ROOT", tree)
    gates = [
        {"name": path.removesuffix(".py"), "tests": [path], "jobs": 1}
        for path in TREE
        if path.startswith("tests/test_")
    ]
    next(gate for gate in gates if gate["name"] == "tests/test_page")["runs"] = [
        "docs/page.md"
    ]
    next(gate for gate in gates if gate["name"] == "tests/test_text_only")["reads"] = [
        "docs/page.md"
    ]
    return Suite(
        {
            "bucket": [{"name": "rest", "jobs": 1}],
            "gate": gates,
            "everything": {"files": ["uv.lock"]},
            "inert": {"files": ["docs/*", "README.md"]},
        },
        {},
        TREE,
    )


@pytest.mark.parametrize(
    ("changed", "affected"),
    [
        # The packages above a module run before it.
        ("src/quivers/core.py", "every"),
        ("src/quivers/data/__init__.py", "every"),
        # Package data counts with the package.
        ("src/quivers/data/table.json", "every"),
        ("src/quivers/lazy/thing.py", {"lazy", "opaque"}),
        ("src/quivers/lazy/near.py", {"lazy", "opaque"}),
        ("src/quivers/family/one.py", {"formatted", "opaque"}),
        ("src/quivers/script.py", {"process", "opaque"}),
        ("src/quivers/typed.py", {"typed", "opaque"}),
        ("src/quivers/inside.py", {"inside", "opaque"}),
        ("src/quivers/text.py", {"text", "opaque"}),
        ("src/quivers/text_fragment_one.py", {"text", "opaque"}),
        ("src/quivers/text_spoken_two.py", {"text", "opaque"}),
        ("src/quivers/shown.py", {"page", "opaque"}),
        ("src/quivers/plain.py", {"opaque"}),
        ("docs/page.md", {"page", "text_only"}),
        ("docs/other.md", set()),
        ("README.md", set()),
        ("tests/test_lazy.py", {"lazy", "opaque"}),
        ("tests/conftest.py", "every"),
        ("uv.lock", "every"),
        # Nothing accounts for these.
        ("notes.bin", "every"),
        ("src/quivers/removed.py", "every"),
        ("added.txt", "every"),
    ],
)
def test_a_change_affects_the_gates_that_can_run_it(
    monkeypatch: pytest.MonkeyPatch,
    tree: Path,
    changed: str,
    affected: set[str] | str,
) -> None:
    suite = _tree_suite(monkeypatch, tree)
    everything = {gate.name.removeprefix("tests/test_") for gate in suite.gates}
    found = {name.removeprefix("tests/test_") for name in suite.affected([changed])}
    assert found == (everything if affected == "every" else affected)


def test_a_new_dependency_edge_widens_the_affected_gates(
    monkeypatch: pytest.MonkeyPatch, tree: Path
) -> None:
    before = _tree_suite(monkeypatch, tree).affected(["src/quivers/plain.py"])
    assert "tests/test_prose" not in before
    (tree / "src/quivers/unused.py").write_text("import quivers.plain\n")
    after = _tree_suite(monkeypatch, tree).affected(["src/quivers/plain.py"])
    assert after == before | {"tests/test_prose"}


def test_relative_imports_resolve_against_their_package(tree: Path) -> None:
    (tree / "src/quivers/lazy/thing.py").write_text(
        "from . import near\nfrom .. import plain\nfrom ..family.one import x\n"
    )
    graph = ImportGraph(tree, TREE)
    assert {
        "quivers.lazy.near",
        "quivers.plain",
        "quivers.family.one",
    } <= graph.edges["quivers.lazy.thing"]


def test_module_names() -> None:
    assert module_name("src/quivers/a/__init__.py") == "quivers.a"
    assert module_name("tests/a/b.py") == "tests.a.b"
    assert module_name("docs/a.py") is None
    assert module_name("src/quivers/a.json") is None


def test_the_graph_contains_every_module_a_process_imports() -> None:
    """The static graph reaches what running a real module loads."""
    suite = Suite.load()
    code = (
        "import json, sys; import quivers.formulas, quivers.modeling;"
        "print(json.dumps([m for m in sys.modules if m.split('.')[0] == 'quivers']))"
    )
    printed = verification.subprocess.check_output(
        [verification.sys.executable, "-c", code],
        cwd=ROOT,
        env={**verification.os.environ, "PYTHONPATH": "src"},
        text=True,
    )
    loaded = set(verification.json.loads(printed.splitlines()[-1]))
    reached = suite.graph.reachable(["quivers.formulas", "quivers.modeling"])
    assert loaded - reached == set()
    assert len(loaded) > 300


def test_every_gate_of_this_repository_depends_on_the_library_core() -> None:
    suite = Suite.load()
    for path in ("src/quivers/dsl/compiler/core.py", "src/quivers/qiec/evaluator.py"):
        assert suite.affected([path]) == {gate.name for gate in suite.gates}


# ---------------------------------------------------------------------------
# The declaration fits the repository
# ---------------------------------------------------------------------------


def test_the_declaration_fits_this_repository() -> None:
    assert check(Suite.load()) == []


def _declared() -> dict:
    with (ROOT / verification.DECLARATION).open("rb") as stream:
        return verification.tomllib.load(stream)


def _mutated(change) -> list[str]:
    suite = Suite.load()
    declaration, durations = _declared(), copy.deepcopy(dict(suite.durations))
    change(declaration, durations)
    return check(Suite(declaration, durations, suite.files))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda d, t: d["gate"][0]["tests"].append("tests/test_gone.py"),
            "not in the repository",
        ),
        (
            lambda d, t: d["gate"][0].update(
                tests=["tests/test_gallery_sweep.py::test_gone"]
            ),
            "not in the repository",
        ),
        (lambda d, t: d["gate"][0]["runs"].append("docs/gone/*"), "fits no file"),
        (lambda d, t: d["gate"][-1].update(images=["gone"]), "has no Dockerfile"),
        (lambda d, t: d["gate"].append(dict(d["gate"][0])), "two parts are named"),
        (lambda d, t: d["bucket"].reverse(), "lists no files"),
        (
            lambda d, t: d["bucket"][0]["files"].append("tests/gone_*.py"),
            "fits no test file",
        ),
        (lambda d, t: d["inert"]["files"].append("gone.txt"), "fits no file"),
        (lambda d, t: d["inert"]["files"].append("uv.lock"), "both inert and"),
        (
            lambda d, t: d["sequence"][0].update(tests="tests/gone.py"),
            "not a test file",
        ),
        (
            lambda d, t: t.update(gone={"units": {"tests/test_dsl.py": 2.0}}),
            "which is not a test",
        ),
        (
            lambda d, t: t["library"]["units"].update({"tests/test_gone.py": 2.0}),
            "which is not a test",
        ),
        (
            lambda d, t: t["library"]["units"].update(
                {"tests/test_dsl.py::test_gone": 2.0}
            ),
            "which is not a test",
        ),
        (lambda d, t: d["bucket"][0].pop("tools"), "needs the language tools"),
    ],
)
def test_a_declaration_that_does_not_fit_is_reported(change, message: str) -> None:
    assert any(message in problem for problem in _mutated(change))


def test_pruning_drops_the_durations_of_tests_that_are_gone() -> None:
    suite = Suite.load()
    durations = copy.deepcopy(dict(suite.durations))
    durations["gone"] = {"units": {"tests/test_dsl.py": 2.0}, "chunks": {}}
    durations["library"]["units"]["tests/test_gone.py"] = 2.0
    durations["library"]["chunks"]["tests/test_dsl.py::test_gone"] = 3
    stale = Suite(_declared(), durations, suite.files)
    assert check(stale) != []
    pruned = prune(stale)
    assert pruned == suite.durations
    assert check(Suite(_declared(), pruned, suite.files)) == []


@pytest.mark.parametrize("changed", [None, ["README.md"]])
def test_the_plan_command_appends_its_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: list[str] | None
) -> None:
    event, output = tmp_path / "event.json", tmp_path / "output"
    event.write_text("{}")
    output.write_text("existing=value\n")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    if changed is not None:
        monkeypatch.setattr(verification, "changed_paths", lambda *_: changed)
    assert verification.main(["plan"]) == 0
    first, *rest = output.read_text().splitlines()
    assert first == "existing=value"
    outputs = dict(line.split("=", 1) for line in rest)
    suite = Suite.load()
    selected = None if changed is None else {"documentation blocks"}
    assert outputs == plan(suite, selected)


def test_durations_of_a_run_with_another_number_of_jobs_are_recorded() -> None:
    suite = _suite()
    report = {"job": "slow 1/9", "ran": {"tests/test_slow.py::test_3": 400.0}}
    recorded = record(suite, [report])
    assert recorded["slow"]["units"] == {"tests/test_slow.py::test_3": 400.0}
    assert recorded["library"] == DURATIONS["library"]
