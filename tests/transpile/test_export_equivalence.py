"""Compare QVR return values with transpiled program exports.

Log-density agreement does not determine a program return value.
For gallery programs with a `return` clause, this tier evaluates the
QVR program and each supported backend at the same points, then
compares returned values recursively by name, shape, and numeric value.

Each backend exposes returns through its native surface: model
function returns, Stan generated quantities, BUGS/JAGS monitored
aliases, or Gen assessed returns. The tests also check that declared
return names and probe payloads have matching arity and that return
mutations are detected independently of log density.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import didactic.api as dx
import pytest
import torch

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.lower import exported_return_names
from tests.transpile import _docker, _gallery_data, _mutations
from tests.transpile import test_gallery_numeric_equivalence as _gallery_tier
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import (
    assert_all_latents_clamped,
    reference_traces,
)


# The export tier reuses the gallery tier's backend table, shape
# table, dtype table, and cell registries rather than restating them.
# A private copy would drift, and the export check is only a statement
# about the cells the density check covers if it runs on exactly those
# cells.
_BACKENDS = _gallery_tier._BACKENDS_WITH_IMAGES

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE_DIR = _REPO_ROOT / "docs" / "examples" / "source"
_SCRATCH_ROOT = pathlib.Path("/tmp") / "qvr_export_equivalence"

_PointSection = dict[str, float | int | list[float] | list[int]]
_ProbePoint = dict[str, _PointSection]

#: One exported value as it crosses the probe boundary: a JSON
#: scalar, an arbitrarily nested list of the same, or `None` for a
#: target whose return surface yielded nothing (a JavaScript
#: `undefined`, a JSON `null`). The `None` arm is what lets a dropped
#: export reach a named assertion instead of a `TypeError`.
_ExportValue = float | int | list["_ExportValue"] | None

#: One point's exported values, one entry per export name.
_ExportRow = list[_ExportValue]


# ---------------------------------------------------------------------
# Tolerance.
# ---------------------------------------------------------------------

_EXPORT_ULP_BUDGET = 8
"""Round-off budget for
[`export_atol`][tests.transpile.test_export_equivalence.export_atol],
in units of the float32 grid spacing at the compared magnitude.

The reference export is a `torch` tensor under the default float32
dtype, and every backend recomputes the same function from the same
clamped inputs in its own arithmetic (float64 for Stan, JAX, torch,
Julia, WebPPL and JAGS; float32 for TensorFlow). Two evaluations of
the same deterministic function therefore differ only by the float32
representation of the reference's intermediates, which is a relative
error at the float32 grid.

The budget is calibrated against a direct measurement rather than a
guess: across every live `(backend, example)` cell, every point of
the six-point set, and every element of every exported vector, the
largest disagreement is
`_MEASURED_WORST_ULP_RATIO` float32 ULPs. Eight ULPs clears that
worst case by a factor of 1.29, and
[`test_export_tolerance_is_tight`][tests.transpile.test_export_equivalence.test_export_tolerance_is_tight]
pins both numbers so the headroom cannot quietly grow.

The factor is that small because one gallery export runs a link
function. `kumaraswamy_bounded_outcome` returns
`a = exp(gamma_0 + gamma_1 * x)`, and `exp` turns the reference's
float32 error on the exponent into a *relative* error on the result,
so its worst element misses by six ULPs of a 99-magnitude value
where every export without a link misses by less than one. Those
cells sit an order of magnitude inside the budget: `beta_regression`,
`pcfg` and `ccg` top out at 0.81, 0.73 and 0.63 ULPs, and 53 cells
reproduce the reference bit for bit.

This is a headroom figure, not a necessity. Widening it is never the
fix for a failure: a real export defect moves the value by a
multiple of its own magnitude, not by bits."""

_MEASURED_WORST_ULP_RATIO = 6.18
"""Largest measured reference-versus-backend disagreement, in float32
ULPs at the compared element's magnitude.

Measured over every `(backend, example)` cell whose probe run
completes, 250 of them at the time of the measurement, all six
points of each cell's set, and every element of every exported
vector: the worst is `pyro@kumaraswamy_bounded_outcome`, 4.71431e-05
absolute at a reference magnitude of 99.0359, which is 6.179 float32
ULPs. 53 of the 250 cells agree to the last bit.

The worst case is a link function rather than a renderer, and the
attribution is derived rather than assumed. That program returns
`a = exp(gamma_0 + gamma_1 * x)`; recomputing the exponent in
float64 from the point's own clamped `gamma_0 = 0.0594303198158741`,
`gamma_1 = 1.3300246000289917` and `x[59] = 3.4105026721954346`
gives `exp(gamma_0 + gamma_1 * x[59]) = 99.03593581497408`, which is
what pyro reports to the last digit, while the reference rounds the
same quantity onto the float32 grid at 99.035888671875. The gap is
the reference's own rounding scaled by the exported magnitude, so no
change to any emission would close it.

Recorded so the budget above reads as a ratio rather than as a bare
constant. Re-measure before changing it; a growth here is a change in
some renderer's arithmetic, not noise."""


def export_atol(reference: float) -> float:
    """Absolute tolerance for one exported element at `reference`.

    [`_EXPORT_ULP_BUDGET`][tests.transpile.test_export_equivalence._EXPORT_ULP_BUDGET]
    float32 ULPs at the element's own magnitude, floored at the same
    budget taken at magnitude 1.

    The floor is what keeps an exported element near zero from
    demanding bit equality it cannot have. Such an element is still
    the output of a computation over order-one intermediates (a
    `sigmoid` of a difference, a residual, a gathered coefficient),
    and the float32 grid at 1 is the finest resolution those
    intermediates carry, so it is the right unit for the error they
    propagate.
    """
    return _EXPORT_ULP_BUDGET * max(
        _gallery_tier._float32_ulp(reference),
        _gallery_tier._float32_ulp(1.0),
    )


# ---------------------------------------------------------------------
# Cell registries.
# ---------------------------------------------------------------------

# Gallery examples that export nothing this tier could compare, each
# with the reason. `test_export_registry_is_total` checks the claim
# against the `.qvr` text rather than taking it on its word, so an
# entry cannot outlive the gap it describes.
_NO_EXPORTED_PROGRAM: dict[str, str] = {
    # No `program` block at all: the module exports a `define`d
    # contraction morphism (`bilinear_score(...)`), which denotes a
    # linear map rather than a Markov kernel. There is no `return`
    # clause, and `exported_return_names` raises rather than inventing
    # one. It carries a synthetic-data block and a pinned joint, so the
    # density tier scores it; what it lacks is a codomain for this tier
    # to compare, and every one of its `(backend, example)` cells sits
    # in the gallery tier's `_EXPECTED_TRANSPILE_RAISES` under
    # `composition_decl`.
    "tensor_contraction": ("structural: contraction morphism, no program block"),
}

# `(backend, example)` cells the gallery tier scores numerically but
# whose exported value cannot yet be compared, each with the single
# defect that blocks it and the measurement the container produced.
#
# One row. Every other cell whose probe run completes has its
# exported value compared at every point and elementwise, and the
# widest sits at 0.77 of the round-off budget. A cell reaches this
# registry only when its density is scored and its export is not,
# which is why the alternative, a `pytest.skip` reached by a bare
# `except`, is not used: that is how a tier stops asserting anything
# without anyone noticing, and
# `test_export_skip_registry_is_disjoint_from_the_gallery_skips`
# keeps each row honest.
#
# Closure path: fix the named defect, re-measure the cell, drop the
# row.
_SKIP_EXPORT_INCOMPATIBLE: dict[tuple[str, str], str] = {}


def _cell_skip_reason(backend: str, example: str) -> str | None:
    """Why the gallery tier does not numerically score this cell, or
    `None` when it does.

    Delegates every judgement to the gallery tier's own registries so
    the export check covers the density check's live set exactly. A
    cell that becomes live there becomes live here on the same run.
    """
    cell = (backend, example)
    if cell in _gallery_tier._EXPECTED_TRANSPILE_RAISES:
        return "the transpile is a pinned `UnsupportedConstruct` raise"
    if example in _gallery_tier._SKIP_DATASET_LOAD_FAILED:
        return "the example's synthetic-data block does not load"
    if example in _gallery_tier._SKIP_QVR_INCOMPATIBLE:
        return "the QVR probe cannot score the example"
    if cell in _gallery_tier._SKIP_PROBE_INCOMPATIBLE:
        return "the in-container probe has no shape registration"
    return None


def _gallery_examples() -> list[pathlib.Path]:
    return _gallery_data.gallery_examples_with_data()


def _scorable_examples() -> list[pathlib.Path]:
    """Gallery examples that have an exported value the QVR reference
    can trace to a deterministic one.

    Two conditions, and both are read off registries rather than
    restated. The reference has to score the example to a
    deterministic joint, which is the precondition for a deterministic
    export; and the example has to declare a probabilistic program at
    all, since a module whose only export is a composition morphism
    has no `return` clause to trace. The second exclusion is not a
    hole: `test_export_registry_is_total` re-derives it from the
    `.qvr` text for every entry of `_NO_EXPORTED_PROGRAM`, so an
    example that grows a program fails there rather than quietly
    dropping out here.
    """
    return [
        example
        for example in _gallery_examples()
        if example.stem not in _gallery_tier._SKIP_DATASET_LOAD_FAILED
        and example.stem not in _gallery_tier._SKIP_QVR_INCOMPATIBLE
        and example.stem not in _NO_EXPORTED_PROGRAM
    ]


# ---------------------------------------------------------------------
# The in-container export helper, loaded here for direct testing.
#
# `probes/_scripts/_reshape.py` is written to be imported inside a
# container as a bare top-level module (`from _reshape import ...`),
# so it is not reachable through the test package's import path. It is
# plain stdlib, though, and the arity guard in its `export_payload` is
# the code that catches a renderer dropping *part* of a multi-name
# `return` clause. Loading the module here tests that guard directly,
# beside the one gallery example that exports two names
# (`parametric_pooling`).
# ---------------------------------------------------------------------

_RESHAPE_PATH = pathlib.Path(__file__).parent / "probes" / "_scripts" / "_reshape.py"
_RESHAPE_SPEC = importlib.util.spec_from_file_location(
    "tests_transpile_probe_reshape",
    _RESHAPE_PATH,
)
if _RESHAPE_SPEC is None or _RESHAPE_SPEC.loader is None:
    raise ImportError(
        f"cannot load the in-container reshape helper from "
        f"{_RESHAPE_PATH}; the probe scripts and this tier would then "
        f"disagree about the export payload's shape."
    )
_probe_reshape = importlib.util.module_from_spec(_RESHAPE_SPEC)
_RESHAPE_SPEC.loader.exec_module(_probe_reshape)


# ---------------------------------------------------------------------
# Reference side.
# ---------------------------------------------------------------------


def export_names_for(example: pathlib.Path) -> tuple[str, ...]:
    """The names the example's exported program returns."""
    return exported_return_names(parse(example.read_text()))


def _flatten(value: _ExportValue, into: list[float]) -> None:
    """Append every scalar leaf of a nested list to `into`, row-major.

    Both sides of the comparison are produced row-major (torch's
    `tolist`, numpy's `tolist`, the Julia helper's explicit row-major
    nesting), so flattening is order-preserving and an element the
    backend placed in the wrong slot lands in the wrong position here.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, into)
        return
    if isinstance(value, bool):
        into.append(float(value))
        return
    if isinstance(value, (int, float)):
        into.append(float(value))
        return
    raise AssertionError(
        f"exported payload carries a {type(value).__name__} leaf "
        f"({value!r}); the export channel is strictly numeric, and a "
        f"`null` leaf is a target that returned nothing where a "
        f"number belongs."
    )


def flat_export(value: _ExportValue) -> list[float]:
    """Row-major flattening of one exported value."""
    out: list[float] = []
    _flatten(value, out)
    return out


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether two tensors carry identical dtype, shape, and raw bytes.

    Reinterpreting through `uint8` sidesteps floating-point equality
    entirely, which matters in both directions here: `torch.equal`
    calls two identical `nan` bit patterns unequal, and it calls two
    distinct subnormals equal to zero. Determinism of an exported
    value is exact or it is absent, so the comparison is on bytes.

    The `clone` permits the `view`: a tensor sliced out of
    a larger buffer carries a storage offset, and reinterpreting the
    dtype of such a view is rejected unless the offset happens to
    divide evenly into the target element size.
    """
    if left.dtype is not right.dtype or left.shape != right.shape:
        return False
    left_bytes = left.detach().cpu().reshape(-1).clone().view(torch.uint8)
    right_bytes = right.detach().cpu().reshape(-1).clone().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def reference_export_row(
    dataset: _gallery_data.GalleryDataset,
    point: Point,
    names: tuple[str, ...],
    example: str,
) -> list[list[float]]:
    """The QVR reference's exported value at `point`, per name.

    Traces the program once per entry of the probe's determinism seed
    tuple and requires the exported value to agree bit for bit across
    them before returning it. A value that moved between the two runs
    is a draw rather than a function of the point, and comparing a
    draw against a backend's independent draw is not a test; it fails
    here instead, naming the site.
    """
    monadic = dataset.monadic
    if monadic is None:
        raise AssertionError(
            f"{example!r}: the gallery dataset carries no compiled "
            f"`MonadicProgram`, so there is no program to trace for a "
            f"reference export. The example's synthetic-data block "
            f"stopped binding one."
        )
    traces = reference_traces(
        monadic,
        point,
        x_input=dataset.x_input,
        observations=_gallery_data.observations_for_point(dataset, point),
    )
    assert_all_latents_clamped(traces[0], example)
    row: list[list[float]] = []
    for name in names:
        recorded = [trace.sites.get(name) for trace in traces]
        if recorded[0] is None:
            raise AssertionError(
                f"{example!r}: the program returns {name!r} but the "
                f"reference trace records no site under that name "
                f"(recorded: {sorted(traces[0].sites)}). An export "
                f"naming something the trace never binds has no "
                f"reference value, so the cell cannot be compared "
                f"against a backend."
            )
        first = recorded[0].value
        for index in range(1, len(recorded)):
            other = recorded[index]
            if other is None or not _bitwise_equal(first, other.value):
                raise AssertionError(
                    f"{example!r}: the exported value of {name!r} is "
                    f"not deterministic. Tracing the program under two "
                    f"distinct global torch RNG seeds produced "
                    f"different values, so the export is a draw the "
                    f"point does not clamp and no backend comparison "
                    f"against it means anything. Bind the site's "
                    f"ground truth in the example's synthetic-data "
                    f"block, or exclude the cell with the reason."
                )
        row.append(flat_export(first.detach().cpu().tolist()))
    return row


# ---------------------------------------------------------------------
# Backend side.
# ---------------------------------------------------------------------


def _require_image(backend: str) -> tuple[str, str, str]:
    """The `(image, extension, script)` triple for `backend`, or a
    configuration error when its image is missing."""
    image, ext, script_name = _BACKENDS[backend]
    if not _docker.docker_available():
        raise RuntimeError(
            "docker daemon not reachable; the session-scope "
            "`_ensure_docker_environment` autouse fixture should have "
            "started it"
        )
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it"
        )
    return image, ext, script_name


def run_export_probe(
    *,
    backend: str,
    source_text: str,
    export_names: tuple[str, ...],
    points: list[_ProbePoint],
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
    scratch_name: str,
) -> list[_ExportRow]:
    """Run `backend`'s probe on `source_text` and return its exports.

    The export names travel to the container in `export_names.json`
    beside the point set, and each probe reads them to decide which
    values its target's own return surface has to yield. A probe that
    cannot produce one raises inside the container, which surfaces
    here as a failed run rather than as a silently short vector.
    """
    image, ext, script_name = _require_image(backend)
    scratch = _SCRATCH_ROOT / scratch_name
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "export_names.json").write_text(json.dumps(list(export_names)))
    script_path = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script_path,
        source=source_text.encode("utf-8"),
        source_ext=ext,
        points=list(points),
        scratch=scratch,
        shapes=shapes,
        dtypes=dtypes,
        timeout=600.0,
    )
    exports = raw.get("exports")
    if exports is None:
        raise AssertionError(
            f"{backend!r} probe returned no `exports` key even though "
            f"the harness shipped export_names.json with "
            f"{list(export_names)!r}. The probe script does not "
            f"implement the export channel."
        )
    if len(exports) != len(points):
        raise AssertionError(
            f"{backend!r} probe returned {len(exports)} export row(s) "
            f"for {len(points)} point(s)."
        )
    return exports


def export_deviation(
    reference: list[list[float]],
    measured: _ExportRow,
    *,
    context: str,
    names: tuple[str, ...],
) -> tuple[float, float, str]:
    """Largest tolerance-relative deviation between two export rows.

    Returns `(worst_ratio, worst_abs, detail)` where `worst_ratio` is
    `|measured - reference| / export_atol(reference)` maximised over
    every element of every exported name. A ratio at or above 1 is a
    rejection; the number itself is what lets the mutation tier report
    a margin instead of only a verdict.

    An arity or length disagreement is not folded into the ratio: it
    means the emitted program returns a different *object*, not a
    numerically close one, and it raises here so the failure names the
    shape rather than a meaningless magnitude.
    """
    if len(measured) != len(names):
        raise AssertionError(
            f"{context}: the probe reported {len(measured)} exported "
            f"value(s) for {len(names)} exported name(s) {names!r}."
        )
    worst_ratio = 0.0
    worst_abs = 0.0
    detail = "no element deviated"
    for name, expected, actual in zip(names, reference, measured):
        flat_actual = flat_export(actual)
        if len(flat_actual) != len(expected):
            raise AssertionError(
                f"{context}: exported {name!r} carries "
                f"{len(flat_actual)} element(s) where the QVR "
                f"reference carries {len(expected)}. The emitted "
                f"program returns an object of a different shape, so "
                f"it denotes a kernel into a different codomain."
            )
        for index, (want, got) in enumerate(zip(expected, flat_actual)):
            if not math.isfinite(got):
                raise AssertionError(
                    f"{context}: exported {name!r}[{index}] is "
                    f"{got!r}. A non-finite export is a defect, not a "
                    f"tolerance question."
                )
            gap = abs(got - want)
            ratio = gap / export_atol(want)
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_abs = gap
                detail = (
                    f"{name}[{index}]: reference {want!r} against "
                    f"backend {got!r}, a gap of {gap:.6g} at a "
                    f"{export_atol(want):.6g} tolerance"
                )
    return worst_ratio, worst_abs, detail


def _probe_points(points: list[Point]) -> list[_ProbePoint]:
    return [{"params": point.params, "data": point.data} for point in points]


# ---------------------------------------------------------------------
# Structural tests. No container: these guard the registries, so a
# coverage regression surfaces even where the probe images are absent.
# ---------------------------------------------------------------------


def test_export_registry_is_total() -> None:
    """Every gallery example either exports a returned value or is a
    justified structural exemption, and the exemption is checked.

    The failure this prevents is silent. A `try / except` around
    `exported_return_names` with a `pytest.skip` in the handler turns
    every example whose program stopped exporting anything into a
    cell that passes while asserting nothing. Requiring the two
    outcomes to partition the gallery makes both directions loud: an
    example that loses its `return` clause fails here, and an
    exemption whose module grew a program fails here too.
    """
    gallery = {example.stem for example in _gallery_examples()}
    exempt = set(_NO_EXPORTED_PROGRAM)

    stale = sorted(exempt - gallery)
    assert not stale, (
        f"{stale!r} are registered as exporting no program but are no "
        f"longer gallery examples with synthetic data. Drop the rows."
    )

    for example in _gallery_examples():
        stem = example.stem
        reason = _NO_EXPORTED_PROGRAM.get(stem)
        if reason is None:
            names = export_names_for(example)
            assert names, (
                f"{stem!r}: the exported program declares no `return` "
                f"clause, so this tier has nothing to compare and the "
                f"program's codomain is unvalidated on every backend. "
                f"Give the program a `return`, or record why it has "
                f"none in `_NO_EXPORTED_PROGRAM`."
            )
            continue
        assert reason.strip(), (
            f"{stem!r}: exempt with an empty reason. An exemption "
            f"without a stated cause is an unexplained hole."
        )
        with pytest.raises(UnsupportedConstruct) as exc_info:
            export_names_for(example)
        assert "no program_decl" in str(exc_info.value), (
            f"{stem!r}: claims to declare no probabilistic program "
            f"({reason}), but `exported_return_names` did not reject "
            f"it for that cause: {exc_info.value!r}. Either the module "
            f"grew a program and the row is stale, or the exemption "
            f"names the wrong cause."
        )


def test_export_skip_registry_is_disjoint_from_the_gallery_skips() -> None:
    """An export skip may only name a cell the density tier scores.

    A cell the gallery tier already skips is covered by that tier's
    own registry and its reason; repeating it here would double-count
    the gap and let a closed gallery skip leave a stale export skip
    behind, reading as coverage. Requiring disjointness means every
    row in `_SKIP_EXPORT_INCOMPATIBLE` is a defect the export channel
    found and the density channel did not, which is the only thing
    this registry should ever hold.
    """
    for (backend, example), reason in sorted(_SKIP_EXPORT_INCOMPATIBLE.items()):
        assert backend in _BACKENDS, (
            f"({backend!r}, {example!r}) names a backend outside the "
            f"gallery matrix {sorted(_BACKENDS)!r}."
        )
        assert reason.strip(), (
            f"({backend!r}, {example!r}) skips the export check with an empty reason."
        )
        gallery_reason = _cell_skip_reason(backend, example)
        assert gallery_reason is None, (
            f"({backend!r}, {example!r}) is skipped by the export tier "
            f"({reason}) and by the density tier ({gallery_reason}). "
            f"Drop the export row: the cell is already uncovered for a "
            f"recorded reason, and a second row makes the gap look "
            f"bigger than it is."
        )


@pytest.mark.parametrize("example", _scorable_examples(), ids=lambda p: p.stem)
def test_export_reference_is_deterministic(example: pathlib.Path) -> None:
    """The reference's exported value is a function of the point.

    This is the precondition that makes an elementwise comparison
    well-posed at all. It is asserted per example and per point rather
    than assumed, because the failure it guards against produces a
    finite number on both sides: an export naming a site the point
    leaves free is redrawn on every call, so the "comparison" is
    between two independent draws and its verdict is a coin flip.

    Bit equality across two distinct global RNG seeds is the test, not
    a tolerance. A tolerance here would be a second, silent threshold
    underneath the one the tier asserts on, and a resampled export
    whose spread happened to land inside it would pass.
    """
    stem = example.stem
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{stem!r}: `load_gallery_data` returned None even though the "
        f"example is not in `_SKIP_DATASET_LOAD_FAILED`."
    )
    names = export_names_for(example)
    assert names, f"{stem!r}: no `return` clause; see `test_export_registry_is_total`."
    points = _gallery_data.points_from_dataset(dataset)
    for point in points:
        row = reference_export_row(dataset, point, names, stem)
        assert len(row) == len(names)
        for name, values in zip(names, row):
            assert values, (
                f"{stem!r}: exported {name!r} carries no elements, so "
                f"a backend returning anything at all would compare "
                f"equal to it."
            )
            for value in values:
                assert math.isfinite(value), (
                    f"{stem!r}: exported {name!r} contains {value!r}. "
                    f"A non-finite reference export asserts nothing: "
                    f"every comparison against it is `nan`, which no "
                    f"`<=` rejects."
                )


@pytest.mark.parametrize("example", _scorable_examples(), ids=lambda p: p.stem)
def test_export_reference_varies_across_points(
    example: pathlib.Path,
) -> None:
    """The reference export takes at least two values across the set.

    Elementwise equality is not vacuous the way a spread is, but it
    has one degenerate case: an export frozen at every point is
    matched by a backend that returns a hard-coded constant. Requiring
    the reference to move removes it, and the requirement is
    satisfiable for every gallery example because the point set
    perturbs the latents and the observations both.
    """
    stem = example.stem
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{stem!r}: `load_gallery_data` returned None even though the "
        f"example is not in `_SKIP_DATASET_LOAD_FAILED`."
    )
    names = export_names_for(example)
    points = _gallery_data.points_from_dataset(dataset)
    seen = {
        json.dumps(reference_export_row(dataset, point, names, stem))
        for point in points
    }
    assert len(seen) > 1, (
        f"{stem!r}: the exported value is identical at every point of "
        f"the set, so a backend that returned a frozen constant would "
        f"pass this tier. Either the point set stopped perturbing the "
        f"sites the export reads, or the program's `return` names "
        f"something no perturbation reaches."
    )


# ---------------------------------------------------------------------
# The export equivalence tier.
# ---------------------------------------------------------------------


def _export_cells() -> list[tuple[pathlib.Path, str]]:
    """Every ``(example, backend)`` cell the export comparison is a test
    case for: the cells the gallery tier scores and the cells whose
    transpile is a pinned refusal. A cell the gallery tier does not
    score, or one in ``_SKIP_EXPORT_INCOMPATIBLE``, is not a test case;
    the registry tests keep those registries honest.
    """
    cells: list[tuple[pathlib.Path, str]] = []
    for example in _gallery_examples():
        for backend in sorted(_BACKENDS):
            cell = (backend, example.stem)
            if cell in _gallery_tier._EXPECTED_TRANSPILE_RAISES:
                cells.append((example, backend))
                continue
            if _cell_skip_reason(backend, example.stem) is not None:
                continue
            if cell in _SKIP_EXPORT_INCOMPATIBLE:
                continue
            cells.append((example, backend))
    return cells


@pytest.mark.parametrize(
    ("example", "backend"),
    _export_cells(),
    ids=lambda value: value.stem if isinstance(value, pathlib.Path) else value,
)
def test_gallery_backend_export_matches_qvr(
    example: pathlib.Path, backend: str
) -> None:
    """The backend's exported value equals the QVR reference's, at
    every point of the multi-point set and elementwise.

    This is the tier's central assertion. It says the emitted program
    denotes the same kernel as the QVR program, including the same
    joint: the density comparison fixes the measure over
    `(latents, observations)`, and this fixes the value the kernel
    carries into its codomain.
    """
    stem = example.stem
    expected_raise = _gallery_tier._EXPECTED_TRANSPILE_RAISES.get((backend, stem))
    if expected_raise is not None:
        module = parse(example.read_text())
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_raise) for k in kinds), (
            f"{backend!r} on {stem!r}: expected the transpile to "
            f"refuse with a {expected_raise!r} kind and it reported "
            f"{list(kinds)!r}. Update the entry in "
            f"`_EXPECTED_TRANSPILE_RAISES`, or find the gap that fired."
        )
        return
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{stem!r}: `load_gallery_data` returned None even though the "
        f"example is not in `_SKIP_DATASET_LOAD_FAILED`."
    )
    names = export_names_for(example)
    assert names, f"{stem!r}: no `return` clause; see `test_export_registry_is_total`."

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    emitted = transpile(parse(example.read_text()), target=backend).decode("utf-8")
    measured = run_export_probe(
        backend=backend,
        source_text=emitted,
        export_names=names,
        points=_probe_points(points),
        shapes=_gallery_tier._shapes_from_dataset(dataset),
        dtypes=_gallery_tier._dtypes_from_dataset(dataset),
        scratch_name=f"{stem}_{backend}",
    )

    for index, point in enumerate(points):
        reference = reference_export_row(dataset, point, names, stem)
        ratio, gap, detail = export_deviation(
            reference,
            measured[index],
            context=f"{backend}@{stem} point {index} ({labels[index]})",
            names=names,
        )
        assert ratio < 1.0, (
            f"{backend}@{stem}: the exported value at point {index} "
            f"({labels[index]}) disagrees with the QVR reference by "
            f"{ratio:.3g} times the round-off budget ({gap:.6g} "
            f"absolute). {detail}. The emitted program computes a "
            f"different value for the program's `return` clause, so it "
            f"denotes a different Markov kernel even where its joint "
            f"log-density matches. Fix the renderer's export emission; "
            f"widening `_EXPORT_ULP_BUDGET` would restore exactly the "
            f"blindness this tier exists to remove."
        )


# ---------------------------------------------------------------------
# Sensitivity: what the export check rejects.
# ---------------------------------------------------------------------


class ExportMutation(dx.Model):
    """One export defect, applied to one gallery example on one target.

    Attributes
    ----------
    name
        Catalogue key; also part of the pytest parameter id.
    defect_class
        Short slug naming the kind of kernel error the mutant carries.
    example
        Stem of the gallery example the rewrite is anchored to.
    backend
        Target whose emitted source the rewrite edits.
    min_margin
        Lower bound on the tolerance-relative deviation the mutant
        must produce, in multiples of
        [`export_atol`][tests.transpile.test_export_equivalence.export_atol].
        A mutant that merely clears 1.0 still passes the reject test
        while the point set or the export quietly loses its content,
        so the floor asserts the mutant stays as loud as it was when
        the catalogue was built.
    probe_refuses
        True when the target's probe cannot even produce a value for
        the mutant, which is the shape a *dropped* export takes: there
        is nothing to compare, and the rejection is the probe's
        refusal rather than a numeric gap.
    rewrites
        The literal-text substitutions, each with a pinned occurrence
        count.
    """

    name: str
    defect_class: str
    example: str
    backend: str
    min_margin: float
    probe_refuses: bool
    rewrites: tuple[_mutations.SourceRewrite, ...]


class AcceptedExportRewrite(dx.Model):
    """A rewrite the export check must **not** reject.

    Without these the reject tests would be satisfied by a check that
    fails on everything. Each entry changes the emitted text while
    leaving the exported value bit-identical, so a tier that rejected
    it would be measuring the source rather than the kernel.
    """

    name: str
    example: str
    backend: str
    rationale: str
    rewrites: tuple[_mutations.SourceRewrite, ...]


def _negation(backend: str, old: str, new: str) -> ExportMutation:
    """An `exported_value_negated` mutant for one target.

    The headline defect, and the one already pinned as a measured
    blind spot of the density tier in `tests/transpile/_mutations.py`:
    negating what a program returns leaves its joint untouched and
    changes the kernel it denotes at every point.
    """
    return ExportMutation(
        name="exported_value_negated",
        defect_class="export-sign-flipped",
        example="ar1",
        backend=backend,
        min_margin=_NEGATION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(_mutations.SourceRewrite(old=old, new=new),),
    )


def _dropped(backend: str, old: str, new: str) -> ExportMutation:
    """An `exported_value_dropped` mutant for one target.

    The emit keeps the program's whole joint and stops exposing what
    the program returns. On the seven targets with a program-level
    return the rewrite deletes the `return`; on Stan, PyMC, and the
    BUGS family it deletes the alias those targets expose a quantity
    through. Either way the probe has nothing to report and refuses.
    """
    return ExportMutation(
        name="exported_value_dropped",
        defect_class="export-dropped",
        example="ar1",
        backend=backend,
        min_margin=0.0,
        probe_refuses=True,
        rewrites=(_mutations.SourceRewrite(old=old, new=new),),
    )


_NEGATION_MIN_MARGIN = 1.0e6
"""Floor on the deviation an `exported_value_negated` mutant produces,
in multiples of the export tolerance.

`ar1` exports `phi`, whose magnitude across the point set runs from
0.585 to 0.700, so negation moves the value by up to 1.400 while the
tolerance at that magnitude is 9.537e-07. Every one of the ten
targets measures a worst-point ratio of 1.468e6; the floor sits at
1.0e6, under the measurement and six orders of magnitude above the
1.0 rejection threshold."""

_SUBSTITUTION_MIN_MARGIN = 3.0e5
"""Floor on the deviation an `exported_site_substituted` mutant
produces.

`ar1`'s `sigma` differs from its `phi` by between 0.070 and 0.491
across the point set, against the same 9.537e-07 tolerance, so the
worst-point ratio measures 5.15e5. This is the quietest defect class
in the catalogue, because the substituted site happens to sit near
the exported one; the floor is set below the measurement with room
for the point set's own arithmetic to shift."""

_PERMUTATION_MIN_MARGIN = 5.0e5
"""Floor on the deviation an `exported_vector_permuted` mutant
produces.

`irt_2pl` exports the 64-element response-probability vector `p`;
reversing it moves individual entries by up to 0.988, so the
worst-point ratio measures 1.036e6. The margin is bounded by how far
apart the vector's extremes sit rather than by the vector's own
magnitude, which is why a permutation of a nearly-constant vector
would be a far quieter mutant than this one."""


# The catalogue. Every anchor below is the literal text the renderer
# emits today; read the emit before editing an entry:
#
#     transpile(parse(Path("docs/examples/source/ar1.qvr").read_text()),
#               target=backend)
EXPORT_MUTATIONS: tuple[ExportMutation, ...] = (
    # --- negate the exported value, on all ten targets -------------
    _negation("numpyro", "  return phi", "  return -phi"),
    _negation("pyro", "  return phi", "  return -phi"),
    _negation("edward2", "  return phi", "  return -phi"),
    _negation("turing", "  return phi", "  return -phi"),
    _negation("gen", "  return phi", "  return -phi"),
    _negation("webppl", "  return phi ;", "  return -phi ;"),
    _negation(
        "stan",
        "real <lower = -1 , upper = 1> phi_value = phi;",
        "real <lower = -1 , upper = 1> phi_value = -phi;",
    ),
    _negation(
        "pymc",
        'pymc.Deterministic("phi_value" ,pymc.math.as_tensor(phi))',
        'pymc.Deterministic("phi_value" ,pymc.math.as_tensor(-phi))',
    ),
    _negation("jags", "phi_value <- phi", "phi_value <- -1*phi"),
    _negation("bugs", "phi_value <- phi", "phi_value <- -1*phi"),
    # --- drop the exported value, on all ten targets ---------------
    _dropped("numpyro", "  return phi", "  pass"),
    _dropped("pyro", "  return phi", "  pass"),
    _dropped("edward2", "  return phi", "  pass"),
    _dropped("turing", "  return phi", "  return nothing"),
    _dropped("gen", "  return phi", "  return nothing"),
    _dropped("webppl", "  return phi ;", "  return null ;"),
    _dropped(
        "stan",
        "real <lower = -1 , upper = 1> phi_value = phi;",
        "",
    ),
    _dropped(
        "pymc",
        'pymc.Deterministic("phi_value" ,pymc.math.as_tensor(phi))',
        "phi",
    ),
    _dropped("jags", "phi_value <- phi", ""),
    _dropped("bugs", "phi_value <- phi", ""),
    # --- return a different site -----------------------------------
    # One target per export surface family: a program-level `return`
    # (numpyro, turing), a typed generated quantity (stan), and a
    # deterministic relation (jags). The defect is spelled identically
    # within a family, so a fourth member would re-measure the same
    # rewrite rather than a new one.
    ExportMutation(
        name="exported_site_substituted",
        defect_class="export-wrong-site",
        example="ar1",
        backend="numpyro",
        min_margin=_SUBSTITUTION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="  return phi",
                new="  return sigma",
            ),
        ),
    ),
    ExportMutation(
        name="exported_site_substituted",
        defect_class="export-wrong-site",
        example="ar1",
        backend="turing",
        min_margin=_SUBSTITUTION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="  return phi",
                new="  return sigma",
            ),
        ),
    ),
    ExportMutation(
        name="exported_site_substituted",
        defect_class="export-wrong-site",
        example="ar1",
        backend="stan",
        min_margin=_SUBSTITUTION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            # The declared bounds go with the substituted site: a
            # `sigma` draw outside [-1, 1] would make CmdStan reject
            # the generated quantity's own constraint, and the mutant
            # has to be a *valid* Stan program whose measure is right
            # and whose export is wrong.
            _mutations.SourceRewrite(
                old="real <lower = -1 , upper = 1> phi_value = phi;",
                new="real phi_value = sigma;",
            ),
        ),
    ),
    ExportMutation(
        name="exported_site_substituted",
        defect_class="export-wrong-site",
        example="ar1",
        backend="jags",
        min_margin=_SUBSTITUTION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="phi_value <- phi",
                new="phi_value <- sigma",
            ),
        ),
    ),
    # --- permute a vector export -----------------------------------
    # `irt_2pl` exports the 64-element `p`. A permutation leaves every
    # element of the exported multiset in place and every log-density
    # term untouched, so it is invisible to any check that reads only
    # the joint or only the export's summary statistics.
    ExportMutation(
        name="exported_vector_permuted",
        defect_class="export-reordered",
        example="irt_2pl",
        backend="numpyro",
        min_margin=_PERMUTATION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="  return p",
                new="  return p[::-1]",
            ),
        ),
    ),
    ExportMutation(
        name="exported_vector_permuted",
        defect_class="export-reordered",
        example="irt_2pl",
        backend="pymc",
        min_margin=_PERMUTATION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old='pymc.Deterministic("p_value" ,pymc.math.as_tensor(p))',
                new=('pymc.Deterministic("p_value" ,pymc.math.as_tensor(p[::-1]))'),
            ),
        ),
    ),
    ExportMutation(
        name="exported_vector_permuted",
        defect_class="export-reordered",
        example="irt_2pl",
        backend="stan",
        min_margin=_PERMUTATION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="array [64] real p_value = p;",
                new="array [64] real p_value = reverse(p);",
            ),
        ),
    ),
    ExportMutation(
        name="exported_vector_permuted",
        defect_class="export-reordered",
        example="irt_2pl",
        backend="jags",
        min_margin=_PERMUTATION_MIN_MARGIN,
        probe_refuses=False,
        rewrites=(
            _mutations.SourceRewrite(
                old="p_value[m_Resp] <- p[m_Resp]",
                new="p_value[m_Resp] <- p[65-m_Resp]",
            ),
        ),
    ),
)


ACCEPTED_EXPORT_REWRITES: tuple[AcceptedExportRewrite, ...] = (
    AcceptedExportRewrite(
        name="exported_value_reassociated",
        example="ar1",
        backend="numpyro",
        rationale=(
            "Adding a literal zero to a float is the identity on every "
            "value the point set carries, so the emitted program "
            "denotes the same kernel and the tier has to accept it. "
            "Without an accepted control the reject tests would be "
            "satisfied by a check that fails on every rewrite."
        ),
        rewrites=(
            _mutations.SourceRewrite(
                old="  return phi",
                new="  return phi + 0.0",
            ),
        ),
    ),
    AcceptedExportRewrite(
        name="exported_value_reassociated",
        example="ar1",
        backend="stan",
        rationale=(
            "The same identity on a target whose export is a typed "
            "generated quantity rather than a function return, so the "
            "control covers both export surfaces."
        ),
        rewrites=(
            _mutations.SourceRewrite(
                old="real <lower = -1 , upper = 1> phi_value = phi;",
                new="real <lower = -1 , upper = 1> phi_value = phi + 0;",
            ),
        ),
    ),
)


def _mutation_id(mutation: ExportMutation) -> str:
    return f"{mutation.name}-{mutation.backend}"


def _accepted_id(rewrite: AcceptedExportRewrite) -> str:
    return f"{rewrite.name}-{rewrite.backend}"


def _worst_export_ratio(
    *,
    backend: str,
    example: str,
    source_text: str,
    scratch_name: str,
) -> tuple[float, str]:
    """Largest tolerance-relative export deviation of `source_text`.

    Runs the target's probe over the example's own point set and
    compares against the QVR reference exactly as the tier does, then
    reports the worst ratio instead of only a verdict, so a mutant's
    margin is a number the catalogue can pin.
    """
    source_path = _SOURCE_DIR / f"{example}.qvr"
    dataset = _gallery_data.load_gallery_data(source_path)
    assert dataset is not None, (
        f"{example!r}: `load_gallery_data` returned None, so the "
        f"sensitivity grid has no point set to mutate against."
    )
    names = export_names_for(source_path)
    points = _gallery_data.points_from_dataset(dataset)
    measured = run_export_probe(
        backend=backend,
        source_text=source_text,
        export_names=names,
        points=_probe_points(points),
        shapes=_gallery_tier._shapes_from_dataset(dataset),
        dtypes=_gallery_tier._dtypes_from_dataset(dataset),
        scratch_name=scratch_name,
    )
    worst = 0.0
    detail = "no element deviated"
    for index, point in enumerate(points):
        reference = reference_export_row(dataset, point, names, example)
        ratio, _, point_detail = export_deviation(
            reference,
            measured[index],
            context=f"{backend}@{example} point {index}",
            names=names,
        )
        if ratio > worst:
            worst, detail = ratio, point_detail
    return worst, detail


@pytest.mark.parametrize(
    "mutation",
    EXPORT_MUTATIONS,
    ids=_mutation_id,
)
def test_export_mutant_is_rejected(mutation: ExportMutation) -> None:
    """Each catalogued export defect is caught, with a pinned margin.

    A mutant is a *valid program in the target language* whose joint
    density is untouched and whose exported value is wrong, which is
    precisely the class the density tier cannot see. Rejecting it is
    the whole content of this module's claim.

    A `probe_refuses` mutant is rejected differently and just as
    hard: the emit no longer exposes the export at all, so the probe
    raises inside the container instead of reporting a wrong number.
    That is a stronger outcome than a numeric gap, not a weaker one,
    and the test requires it rather than accepting silence.
    """
    skip_reason = _cell_skip_reason(mutation.backend, mutation.example)
    assert skip_reason is None, (
        f"{mutation.backend!r} on {mutation.example!r} is not a "
        f"numerically-scored gallery cell ({skip_reason}), so a mutant "
        f"of it measures nothing. Anchor the mutation to a live cell."
    )
    source_path = _SOURCE_DIR / f"{mutation.example}.qvr"
    emitted = transpile(
        parse(source_path.read_text()),
        target=mutation.backend,
    ).decode("utf-8")
    context = f"{mutation.name}@{mutation.backend}/{mutation.example}"
    mutated = _mutations.apply_rewrites(
        emitted,
        mutation.rewrites,
        context=context,
    )

    if mutation.probe_refuses:
        with pytest.raises((RuntimeError, AssertionError)) as exc_info:
            _worst_export_ratio(
                backend=mutation.backend,
                example=mutation.example,
                source_text=mutated,
                scratch_name=f"mutant_{mutation.name}_{mutation.backend}",
            )
        message = str(exc_info.value)
        assert message.strip(), (
            f"{context}: the run failed with an empty message, so "
            f"nothing distinguishes a detected dropped export from an "
            f"unrelated container crash."
        )
        return

    worst, detail = _worst_export_ratio(
        backend=mutation.backend,
        example=mutation.example,
        source_text=mutated,
        scratch_name=f"mutant_{mutation.name}_{mutation.backend}",
    )
    assert worst >= 1.0, (
        f"{context}: the mutant's exported value stayed within the "
        f"round-off budget (worst deviation {worst:.6g} times the "
        f"tolerance; {detail}). The export check does not reject this "
        f"defect class on this target, which means the target's "
        f"exported value is unvalidated. Do not weaken the catalogue "
        f"entry; fix the probe's export surface or the renderer's "
        f"emission."
    )
    assert worst >= mutation.min_margin, (
        f"{context}: rejected, but at {worst:.6g} times the tolerance "
        f"against a pinned floor of {mutation.min_margin:.6g}. The "
        f"mutant got quieter, which means the point set, the exported "
        f"quantity, or the tolerance moved under the catalogue. "
        f"Re-derive the margin from a measurement before re-pinning."
    )


@pytest.mark.parametrize(
    "rewrite",
    ACCEPTED_EXPORT_REWRITES,
    ids=_accepted_id,
)
def test_accepted_export_rewrite_is_not_rejected(
    rewrite: AcceptedExportRewrite,
) -> None:
    """A rewrite that leaves the exported value alone must pass.

    The negative controls for the reject tests. A tier that flagged
    these would be comparing source text rather than kernels, and
    every rejection it reported would be uninformative.
    """
    source_path = _SOURCE_DIR / f"{rewrite.example}.qvr"
    emitted = transpile(
        parse(source_path.read_text()),
        target=rewrite.backend,
    ).decode("utf-8")
    context = f"{rewrite.name}@{rewrite.backend}/{rewrite.example}"
    rewritten = _mutations.apply_rewrites(
        emitted,
        rewrite.rewrites,
        context=context,
    )
    worst, detail = _worst_export_ratio(
        backend=rewrite.backend,
        example=rewrite.example,
        source_text=rewritten,
        scratch_name=f"accepted_{rewrite.name}_{rewrite.backend}",
    )
    assert worst < 1.0, (
        f"{context}: rejected at {worst:.6g} times the tolerance "
        f"({detail}), but the rewrite is value-preserving "
        f"({rewrite.rationale}). The tier is rejecting something other "
        f"than a changed exported value."
    )


# ---------------------------------------------------------------------
# Strength: the tolerance and the catalogue cannot quietly weaken.
# ---------------------------------------------------------------------


def test_export_tolerance_is_tight() -> None:
    """The export tolerance is pinned, bounded, and above the measured
    noise floor by a stated, small factor.

    Three separate claims, each of which a future edit could break on
    its own:

    1. The budget is the pinned integer. A silent widening is the
       cheapest way to make a real export defect pass, so the constant
       is asserted rather than merely defined.
    2. The tolerance at magnitude 1 stays under 1e-06. Every defect
       class in the catalogue moves the exported value by an order-one
       amount, so a tolerance anywhere near that scale would start
       admitting them.
    3. The budget exceeds the measured worst-case disagreement, and by
       no more than a factor of 16. The lower bound is what keeps the
       tier from failing on arithmetic noise; the upper bound is what
       keeps the headroom from growing into a hiding place.

    The third claim is the loose one today, and knowing by how much
    matters: the measured worst case is 6.18 ULPs against a budget of
    8, so the real headroom is a factor of 1.29 while the ceiling
    would tolerate 16. The ceiling bites only once the measurement
    drops under half an ULP, a little below where the widest
    link-free export sits today (`stan@beta_regression`, 0.81); a
    budget of 8 over a half-ULP worst case would be headroom rather
    than round-off, and the assertion would then demand it be
    re-derived.
    """
    assert _EXPORT_ULP_BUDGET == 8, (
        f"the export round-off budget is {_EXPORT_ULP_BUDGET}, not the "
        f"pinned 8. Changing it changes what every cell in this module "
        f"accepts; re-derive the worst-case measurement first."
    )
    unit = export_atol(1.0)
    assert unit <= 1.0e-6, (
        f"the export tolerance at magnitude 1 is {unit:.6g}, past the "
        f"1e-06 ceiling. Every catalogued export defect moves the "
        f"value by an order-one amount, so a tolerance at this scale "
        f"is on its way to admitting them."
    )
    assert _EXPORT_ULP_BUDGET > _MEASURED_WORST_ULP_RATIO, (
        f"the budget ({_EXPORT_ULP_BUDGET}) does not cover the "
        f"measured worst-case disagreement "
        f"({_MEASURED_WORST_ULP_RATIO}), so live cells fail on "
        f"arithmetic noise rather than on defects."
    )
    assert _EXPORT_ULP_BUDGET <= 16 * _MEASURED_WORST_ULP_RATIO, (
        f"the budget ({_EXPORT_ULP_BUDGET}) sits more than 16 times "
        f"above the measured worst case "
        f"({_MEASURED_WORST_ULP_RATIO}). Headroom that large is a "
        f"place for a small real defect to hide; re-measure and "
        f"tighten."
    )
    # The tolerance scales with the compared magnitude, and the scaling
    # is what keeps a large exported value from demanding bit equality
    # while a small one keeps a meaningful floor.
    assert export_atol(1024.0) > export_atol(1.0)
    assert export_atol(0.001) == export_atol(1.0)


def test_export_payload_rejects_a_partial_return() -> None:
    """The probes' shared arity guard catches a truncated return.

    Seven of the nine probe scripts route their target's return value
    through `export_payload`, whose job is to split one return into
    one entry per exported name and to refuse anything else. Three
    refusals matter, and none of them is reachable from a gallery
    cell, because every scorable example exports exactly one name:

    1. a return of `None`, which is what a dropped `return` looks like
       in Python and what a `null` looks like coming back from WebPPL;
    2. a single value where the program exports several, which is a
       renderer that emitted only the first component;
    3. a tuple of the wrong arity, which is a renderer that dropped a
       component from the middle.

    Each is a program with the right joint and the wrong codomain, so
    a guard that let one through would leave the tier reporting a
    comparison it never made. The test runs in-process and needs no
    container.
    """
    payload = _probe_reshape.export_payload
    assert payload(["a"], 1.5) == [1.5]
    assert payload(["a", "b"], (1.5, [2.0, 3.0])) == [1.5, [2.0, 3.0]]

    with pytest.raises(ValueError, match="returns nothing"):
        payload(["a"], None)
    with pytest.raises(ValueError, match="single value"):
        payload(["a", "b"], 1.5)
    with pytest.raises(ValueError, match="value"):
        payload(["a", "b", "c"], (1.5, 2.5))
    with pytest.raises(ValueError, match="no export names"):
        payload([], 1.5)


def test_export_mutation_catalogue_covers_every_target() -> None:
    """The catalogue's coverage is itself asserted.

    Two claims. First, every backend in the gallery matrix carries at
    least one export mutant, so no target's export surface is
    unmeasured: a probe that silently reported the reference back
    would pass the live tier on that target and fail here. Second,
    every declared defect class is represented, so dropping a class
    from the catalogue is a test failure rather than a quiet loss of
    sensitivity.
    """
    covered = {mutation.backend for mutation in EXPORT_MUTATIONS}
    missing = sorted(set(_BACKENDS) - covered)
    assert not missing, (
        f"{missing!r} carry no export mutant, so nothing demonstrates "
        f"that their exported value is actually read. A probe that "
        f"echoed the reference back would pass every live cell on "
        f"those targets."
    )
    classes = {mutation.defect_class for mutation in EXPORT_MUTATIONS}
    expected_classes = {
        "export-sign-flipped",
        "export-dropped",
        "export-wrong-site",
        "export-reordered",
    }
    assert classes == expected_classes, (
        f"the catalogue covers {sorted(classes)!r} against the "
        f"declared {sorted(expected_classes)!r}. A dropped class is a "
        f"dropped capability claim."
    )
    for mutation in EXPORT_MUTATIONS:
        if mutation.probe_refuses:
            continue
        assert mutation.min_margin >= 1.0e4, (
            f"{_mutation_id(mutation)}: pinned margin "
            f"{mutation.min_margin:.6g} is within four orders of "
            f"magnitude of the tolerance. A margin that small no "
            f"longer separates a defect from round-off, and the "
            f"catalogue entry has stopped being a demonstration."
        )
    assert ACCEPTED_EXPORT_REWRITES, (
        "the accepted-rewrite controls are empty, so nothing rules "
        "out a check that rejects every rewrite it is handed."
    )
