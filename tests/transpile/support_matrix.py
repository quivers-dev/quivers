"""Measure backend support and generate `docs/transpile-support.md`.

The measurement calls [`transpile`][quivers.transpile.transpile] for
each backend over two corpora: the gallery programs under
`docs/examples/source/` and the single-construct fixtures under
`tests/transpile/fixtures/`. Each cell records either emitted source or
an [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with its structured kinds and runtime message. Other exceptions
propagate with the cell name.

The page groups refusals by the identifier prefix of each reported
kind, such as `family:LKJCholesky` or `param-source:mlp`. Run
`python -m tests.transpile.support_matrix` to regenerate it. The
`--check` option compares the committed page with a fresh measurement
without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import Module
from quivers.dsl.parser import parse
from quivers.transpile import (
    UnsupportedConstruct,
    available_targets,
    transpile,
)
from tests.transpile.fixtures import _load


_REPO_ROOT = Path(__file__).resolve().parents[2]

DOCS_DIR = _REPO_ROOT / "docs"
"""Root of the mkdocs source tree."""

GALLERY_DIR = DOCS_DIR / "examples" / "source"
"""Directory holding the documentation gallery's QVR programs."""

DOC_PATH = DOCS_DIR / "transpile-support.md"
"""The page this module owns end to end."""

_CONSTRUCT_CATEGORIES: tuple[str, ...] = (
    "statements",
    "steps",
    "let_expressions",
    "options",
    "axes",
)
"""Fixture categories that isolate one surface construct each.

`families` and `compositions` are deliberately absent: the family
corpus is a distribution inventory, matrixed by
`tests/transpile/test_family_matrix.py` and summarised per target by
the transpilation-correctness pages, and the composition corpus holds
whole benchmark models rather than isolated constructs.
"""

_CATEGORY_TITLES: dict[str, str] = {
    "statements": "Declarations",
    "steps": "Program steps",
    "let_expressions": "Let-expressions",
    "options": "Option values",
    "axes": "Axis specifications",
}
"""Reader-facing heading for each construct category."""


class Cell(dx.Model):
    """One measured (program, backend) outcome."""

    group: Literal["gallery", "construct"]
    """Which corpus the program came from."""

    category: str
    """`"gallery"` for a gallery program, else the fixture category."""

    name: str
    """The program's file stem."""

    backend: str
    """The backend key passed to `transpile`."""

    renders: bool
    """True when the call returned bytes."""

    kinds: tuple[str, ...]
    """`UnsupportedConstruct.kinds` when refused; empty when rendered."""

    message: str
    """The refusal's user-facing text; empty when rendered."""


# A program's identity in the measurement: corpus, category, stem.
ProgramKey = tuple[str, str, str]


def _key(cell: Cell) -> ProgramKey:
    """The program a cell measured."""
    return (str(cell.group), str(cell.category), str(cell.name))


def _kinds(cell: Cell) -> tuple[str, ...]:
    """The cell's reported kinds as plain strings."""
    return tuple(str(kind) for kind in cell.kinds)


def _label(key: ProgramKey) -> str:
    """Return the stem, qualified by category for construct fixtures."""
    group, category, name = key
    if group == "gallery":
        return name
    return f"{category}/{name}"


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


def gallery_programs() -> list[tuple[str, str]]:
    """Every gallery program as `(stem, source text)`, stem-sorted."""
    return [(path.stem, path.read_text()) for path in sorted(GALLERY_DIR.glob("*.qvr"))]


def construct_fixtures() -> list[_load.Fixture]:
    """Every construct fixture, category-major then name-sorted."""
    loaders = {
        "statements": _load.load_statements,
        "steps": _load.load_steps,
        "let_expressions": _load.load_let_expressions,
        "options": _load.load_options,
        "axes": _load.load_axes,
    }
    fixtures: list[_load.Fixture] = []
    for category in _CONSTRUCT_CATEGORIES:
        fixtures.extend(loaders[category]())
    return fixtures


def _measure_program(
    *,
    group: Literal["gallery", "construct"],
    category: str,
    name: str,
    module: Module,
    backends: list[str],
) -> list[Cell]:
    """Transpile one parsed program to every backend."""
    cells: list[Cell] = []
    for backend in backends:
        try:
            transpile(module, target=backend)
        except UnsupportedConstruct as refusal:
            cells.append(
                Cell(
                    group=group,
                    category=category,
                    name=name,
                    backend=backend,
                    renders=False,
                    kinds=tuple(refusal.kinds),
                    message=str(refusal),
                )
            )
            continue
        except Exception as error:
            raise RuntimeError(
                f"transpiling {category}/{name} to {backend!r} raised "
                f"{type(error).__name__}, which is neither a rendered "
                f"program nor a declared refusal. The support page "
                f"publishes what the pipeline emits and what it refuses; "
                f"an unclassified failure has to be fixed, or turned "
                f"into an UnsupportedConstruct naming the construct, "
                f"before it can be published."
            ) from error
        cells.append(
            Cell(
                group=group,
                category=category,
                name=name,
                backend=backend,
                renders=True,
                kinds=(),
                message="",
            )
        )
    return cells


def measure() -> list[Cell]:
    """Transpile every program in both corpora to every backend.

    Each source is parsed once and its read-only `Module` is reused
    across backends because parsing dominates the cost of a cell.
    """
    backends = available_targets()
    cells: list[Cell] = []
    for name, source in gallery_programs():
        cells.extend(
            _measure_program(
                group="gallery",
                category="gallery",
                name=name,
                module=parse(source),
                backends=backends,
            )
        )
    for fixture in construct_fixtures():
        cells.extend(
            _measure_program(
                group="construct",
                category=fixture.category,
                name=fixture.name,
                module=parse(fixture.source),
                backends=backends,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# Views over the measurement.
# ---------------------------------------------------------------------------


def _by_program(cells: list[Cell]) -> dict[ProgramKey, dict[str, Cell]]:
    """Index the measurement as `program -> backend -> cell`.

    Insertion order follows corpus order and thus determines page order.
    """
    indexed: dict[ProgramKey, dict[str, Cell]] = {}
    for cell in cells:
        indexed.setdefault(_key(cell), {})[str(cell.backend)] = cell
    return indexed


def _backends(cells: list[Cell]) -> list[str]:
    """Every backend named by the measurement, sorted."""
    return sorted({str(cell.backend) for cell in cells})


def _leading_identifier_path(kind: str) -> str:
    """The identifier-shaped prefix of one reported kind.

    Keep leading colon-separated identifiers and discard a prose tail.
    A kind beginning with prose yields the empty string.
    """
    kept: list[str] = []
    for segment in kind.split(":"):
        if not segment or segment.strip() != segment or " " in segment:
            break
        kept.append(segment)
    return ":".join(kept)


def construct_labels(kinds: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable construct labels for a refusal's reported kinds.

    Raise when every kind begins with prose because such a refusal has
    no programmatically matchable construct.
    """
    labels = {path for kind in kinds if (path := _leading_identifier_path(kind))}
    if not labels:
        raise RuntimeError(
            f"refusal reported no identifier-shaped kind: {list(kinds)!r}. "
            f"Every refusal has to name at least one construct kind a "
            f"caller can match on programmatically; prose alone leaves "
            f"the support page with nothing to group or index."
        )
    return tuple(sorted(labels))


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table, columns padded to a common width.

    The padding serves the committed file's readability under `diff`;
    it does not change the rendered page.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _line(values: list[str]) -> str:
        padded = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return f"| {' | '.join(padded)} |"

    lines = [_line(headers)]
    lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    lines.extend(_line(row) for row in rows)
    return lines


def _program_link(key: ProgramKey) -> str:
    """A program's label, linked to its gallery page when it has one."""
    group, _, name = key
    label = _label(key)
    if group != "gallery":
        return f"`{label}`"
    page = f"examples/{name.replace('_', '-')}.md"
    if not (DOCS_DIR / page).is_file():
        raise RuntimeError(
            f"gallery program {name!r} has no page at docs/{page}; "
            f"either the page was renamed or the program was added to "
            f"docs/examples/source/ without one."
        )
    return f"[`{label}`]({page})"


def _correctness_link(backend: str) -> str:
    """A backend's name, linked to its transpile-correctness page."""
    page = f"semantics/transpile-correctness/{backend}.md"
    if not (DOCS_DIR / page).is_file():
        raise RuntimeError(
            f"backend {backend!r} has no correctness page at "
            f"docs/{page}; every registered backend carries one."
        )
    return f"[{backend}]({page})"


def _matrix(cells: list[Cell], *, group: str, category: str | None) -> list[str]:
    """One (program x backend) table."""
    backends = _backends(cells)
    rows: list[list[str]] = []
    for key, per_backend in _by_program(cells).items():
        row_group, row_category, _ = key
        if row_group != group:
            continue
        if category is not None and row_category != category:
            continue
        row = [_program_link(key)]
        row.extend(
            "yes" if per_backend[backend].renders else "no" for backend in backends
        )
        rows.append(row)
    return _table(["Program", *backends], rows)


def _counts(cells: list[Cell], *, backend: str, group: str) -> tuple[int, int]:
    """`(rendered, total)` for one backend over one corpus."""
    selected = [
        cell
        for cell in cells
        if str(cell.backend) == backend and str(cell.group) == group
    ]
    return sum(1 for cell in selected if cell.renders), len(selected)


def _quote(message: str) -> list[str]:
    """The runtime's own message, fenced so it survives verbatim."""
    return ["```text", *message.splitlines(), "```"]


def _group_by_label(
    entries: list[tuple[ProgramKey, tuple[str, ...]]],
) -> dict[tuple[str, ...], list[ProgramKey]]:
    """Group program/kind pairs by construct label in stable order."""
    grouped: dict[tuple[str, ...], list[ProgramKey]] = {}
    for key, kinds in entries:
        grouped.setdefault(construct_labels(kinds), []).append(key)
    return {labels: sorted(grouped[labels]) for labels in sorted(grouped)}


def _label_heading(labels: tuple[str, ...]) -> str:
    """The construct labels of one group, as a heading fragment."""
    return ", ".join(f"`{label}`" for label in labels)


def _reported_kinds_note(
    keys: list[ProgramKey],
    reported: dict[ProgramKey, tuple[str, ...]],
) -> list[str]:
    """Render the structured kinds that callers can match.

    Use a fence because prose tails may contain backticks. Omit entries
    that begin with prose and have no matchable prefix.
    """
    kinds = sorted(
        {
            kind
            for key in keys
            for kind in reported[key]
            if _leading_identifier_path(kind)
        }
    )
    return ["Reported kinds:", "", "```text", *kinds, "```", ""]


def _universal_gaps(cells: list[Cell]) -> list[str]:
    """Render constructs refused by every backend and their reasons.

    Attribute only kinds shared by all backends to a language-level
    gap; backend-specific kinds remain target limits.
    """
    backends = _backends(cells)
    programs = _by_program(cells)
    agreed_kinds: dict[ProgramKey, tuple[str, ...]] = {}
    divergent: list[ProgramKey] = []
    for key, per_backend in programs.items():
        if any(per_backend[backend].renders for backend in backends):
            continue
        agreed = set(_kinds(per_backend[backends[0]]))
        for backend in backends[1:]:
            agreed &= set(_kinds(per_backend[backend]))
        if not agreed:
            divergent.append(key)
            continue
        agreed_kinds[key] = tuple(sorted(agreed))

    if not agreed_kinds and not divergent:
        return ["Every program in both corpora renders on at least one backend."]

    lines = [
        f"Each construct below is refused by all {len(backends)} "
        f"backends, so it marks the boundary of what QVR exports at all "
        f"rather than a gap in one target. Reaching a target means "
        f"writing the model differently, not switching language.",
    ]
    entries = [(key, kinds) for key, kinds in agreed_kinds.items()]
    for labels, keys in _group_by_label(entries).items():
        lines.append("")
        lines.append(f"### {_label_heading(labels)}")
        lines.append("")
        listed = ", ".join(_program_link(key) for key in keys)
        lines.append(f"Refused for: {listed}.")
        lines.append("")
        lines.extend(_reported_kinds_note(keys, agreed_kinds))
        representative = keys[0]
        backend = backends[0]
        variants = {str(cell.message) for cell in programs[representative].values()}
        agreement = (
            "Every backend reports it in the same words."
            if len(variants) == 1
            else f"Each backend words it differently; this is {backend}'s."
        )
        lines.append(f"{agreement} `{backend}` on `{_label(representative)}` reports:")
        lines.append("")
        lines.extend(_quote(str(programs[representative][backend].message)))
    if divergent:
        lines.append("")
        lines.append("### Refused everywhere, for unrelated reasons")
        lines.append("")
        listed = ", ".join(_program_link(key) for key in sorted(divergent))
        lines.append(
            f"No backend renders {listed}, and no kind is reported by "
            f"all of them, so the refusals are per-target limits that "
            f"happen to overlap. Section 3 carries each backend's own "
            f"reason."
        )
    return lines


def _backend_gaps(cells: list[Cell], *, backend: str) -> list[str]:
    """Section: what one backend refuses that another backend renders."""
    backends = _backends(cells)
    programs = _by_program(cells)
    reported: dict[ProgramKey, tuple[str, ...]] = {}
    for key, per_backend in programs.items():
        cell = per_backend[backend]
        if cell.renders:
            continue
        if not any(
            per_backend[other].renders for other in backends if other != backend
        ):
            continue
        reported[key] = _kinds(cell)

    if not reported:
        return [
            f"{backend} renders every program any other backend "
            f"renders. Its remaining refusals are the language-level "
            f"gaps of section 2."
        ]
    lines = [
        "Every program below renders on at least one other backend and is refused here."
    ]
    entries = [(key, kinds) for key, kinds in reported.items()]
    for labels, keys in _group_by_label(entries).items():
        lines.append("")
        lines.append(f"**{_label_heading(labels)}**")
        lines.append("")
        listed = ", ".join(_program_link(key) for key in keys)
        lines.append(f"Refused for: {listed}.")
        accepting = [
            other
            for other in backends
            if other != backend and all(programs[key][other].renders for key in keys)
        ]
        if accepting:
            names = ", ".join(f"`{name}`" for name in accepting)
            lines.append("")
            lines.append(f"Renders on: {names}.")
        lines.append("")
        lines.extend(_reported_kinds_note(keys, reported))
        representative = keys[0]
        lines.append(f"`{backend}` on `{_label(representative)}` reports:")
        lines.append("")
        lines.extend(_quote(str(programs[representative][backend].message)))
    return lines


# ---------------------------------------------------------------------------
# Page assembly.
# ---------------------------------------------------------------------------


def render_page(cells: list[Cell]) -> str:
    """Render the whole support page from a measurement."""
    backends = _backends(cells)
    programs = _by_program(cells)
    gallery_total = sum(1 for key in programs if key[0] == "gallery")
    construct_total = sum(1 for key in programs if key[0] == "construct")

    lines: list[str] = [
        "<!--",
        "  Generated by `python -m tests.transpile.support_matrix`.",
        "  Every row is a measurement. Hand edits are overwritten, and",
        "  tests/transpile/test_support_matrix_docs.py fails until the",
        "  page is regenerated.",
        "-->",
        "",
        "# Transpilation support",
        "",
        '!!! note "This page is measured, not written"',
        "",
        "    `python -m tests.transpile.support_matrix` calls",
        "    `quivers.transpile.transpile` once per (program, backend)",
        "    pair and writes this file from the outcomes. Every refusal",
        "    quoted below is the runtime's own message, copied verbatim",
        "    from the `UnsupportedConstruct` it raised. A drift test",
        "    regenerates the page and fails when the result differs from",
        "    what is committed.",
        "",
        f"`transpile(module, target=...)` has {len(backends)} registered "
        f"backends. Given a program it either returns target source bytes "
        f"or raises `UnsupportedConstruct` naming the constructs it "
        f"cannot represent. This page records which of the two happens, "
        f"for the {gallery_total} programs of the "
        f"[examples gallery](examples/index.md) and for "
        f"{construct_total} construct fixtures, each isolating a single "
        f"QVR surface construct.",
        "",
        "Support is thus stated as a refusal boundary rather than as a "
        "feature list. A construct no backend accepts is a limit of the "
        "export surface itself, and reaching any target means writing "
        "the model differently; a construct one backend alone refuses is "
        "a limit of that target, and another target may take the program "
        "unchanged. Sections 2 and 3 separate the two, since the "
        "remedies differ.",
        "",
        "Refusals are grouped by construct, not by program: the heading "
        "of each group is the identifier prefix of the reported "
        "`UnsupportedConstruct.kinds`, which is what a reader asking "
        "whether a feature of their own model is supported wants to "
        "match on.",
        "",
        "QIEC uses the same measured renderer boundary as probabilistic programs. "
        "Its complete checked module is retained as typed `IRQiecModule` nodes. "
        "Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church emit named "
        "QIEC functions through a shared stable-ID runtime ABI. Stan, BUGS, and "
        "JAGS emit the closed, monomorphic, effect-free scalar `Return`/`Bind` "
        "fragment and report unsupported forms as "
        "`qiec:capability:<feature>:<computation>`.",
        "",
        "What this page does not cover is whether a rendered program's "
        "density agrees with QVR's own. That is the subject of the "
        "[transpilation-correctness contract]"
        "(semantics/transpile-correctness/index.md), which states the "
        "evidence available for the programs that do render, and of the "
        "[transpilation architecture](semantics/transpile-architecture.md), "
        "which describes how a program reaches a target at all.",
        "",
        "## Reading the measurements",
        "",
        "Transpilation translates a checked QVR program into source for another "
        "probabilistic programming system. A `yes` in the matrix means source "
        "bytes were produced; it does not mean that system executed them, that "
        "inference converged, or that a density-equivalence test was run for "
        "this cell. Those are separate checks.",
        "",
        "Start with a gallery model similar to yours, then inspect the construct "
        "rows for the features you need. A fixture isolates one feature; your "
        "own program can combine several. Run the target check and transpilation "
        "on that complete program before relying on a measured neighboring cell.",
        "",
        "For instance, `qiec:capability:search:lp_parser__run` says that the target "
        "cannot preserve the search capability required by the named parser "
        "entry. The identifier points to a missing execution feature, not a "
        "statistical failure of the parser. A refusal shared by every backend "
        "requires a supported reformulation or execution in Quivers itself.",
        "",
        "## Coverage at a glance",
        "",
    ]
    summary_rows: list[list[str]] = []
    for backend in backends:
        rendered_gallery, total_gallery = _counts(
            cells, backend=backend, group="gallery"
        )
        rendered_construct, total_construct = _counts(
            cells, backend=backend, group="construct"
        )
        summary_rows.append(
            [
                _correctness_link(backend),
                f"{rendered_gallery} / {total_gallery}",
                f"{rendered_construct} / {total_construct}",
            ]
        )
    lines.extend(_table(["Backend", "Gallery programs", "Constructs"], summary_rows))
    lines.extend(
        [
            "",
            "Each backend links to its transpilation-correctness page, "
            "which documents the structure it emits, the parameter "
            "conversions it applies, and the evidence exercised for it.",
            "",
            "## 1. Can this backend take my program?",
            "",
            "`yes` means `transpile` returned bytes; `no` means it "
            "raised `UnsupportedConstruct`. Sections 2 and 3 give the "
            "reason behind every `no`.",
            "",
            "### 1.1 Gallery programs",
            "",
        ]
    )
    lines.extend(_matrix(cells, group="gallery", category=None))
    lines.extend(
        [
            "",
            "### 1.2 Constructs",
            "",
            "One minimal program per surface construct, so a `no` here "
            "isolates the construct rather than the model that used it.",
        ]
    )
    for index, category in enumerate(_CONSTRUCT_CATEGORIES):
        lines.append("")
        lines.append(f"#### 1.2.{index + 1} {_CATEGORY_TITLES[category]}")
        lines.append("")
        lines.extend(_matrix(cells, group="construct", category=category))
    lines.extend(["", "## 2. What no backend supports", ""])
    lines.extend(_universal_gaps(cells))
    lines.extend(
        [
            "",
            "## 3. What each backend alone refuses",
            "",
            "These are target gaps rather than language gaps: another "
            "backend renders the same program. Each quoted message is "
            "the one that backend raises, including whatever alternative "
            "it offers.",
        ]
    )
    for backend in backends:
        lines.append("")
        lines.append(f"### {backend}")
        lines.append("")
        lines.extend(_backend_gaps(cells, backend=backend))
    return "\n".join(lines).rstrip() + "\n"


def generate() -> str:
    """Measure both corpora and render the page."""
    return render_page(measure())


def main(argv: list[str] | None = None) -> int:
    """Write the page, or check the committed page against a fresh
    measurement."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.transpile.support_matrix",
        description=(
            "Measure transpile support across every backend and write "
            "docs/transpile-support.md."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the committed page against a fresh measurement, "
            "exiting non-zero on a difference, without writing"
        ),
    )
    args = parser.parse_args(argv)
    page = generate()
    if args.check:
        if not DOC_PATH.is_file():
            print(f"missing: {DOC_PATH}", file=sys.stderr)
            return 1
        if DOC_PATH.read_text() != page:
            print(
                f"out of date: {DOC_PATH} does not match the current "
                f"measurement; run `python -m "
                f"tests.transpile.support_matrix`",
                file=sys.stderr,
            )
            return 1
        print(f"up to date: {DOC_PATH.relative_to(_REPO_ROOT)}")
        return 0
    DOC_PATH.write_text(page)
    print(f"wrote {DOC_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
