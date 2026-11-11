"""Mutation tests for numeric-equivalence sensitivity.

Each catalogue entry rewrites emitted source for a passing gallery
cell so the target program denotes a different measure. The mutated
source must remain valid in its backend, and its log-density difference
from QVR must exceed both the equivalence tolerance and the mutation
minimum spread.

The tests verify exact rewrite occurrence counts, catalogue coverage,
and acceptance of the unmodified baseline. This distinguishes a weak
point set from a syntax or probe failure.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.inference.trace import trace
from quivers.transpile import transpile
from tests.transpile import _docker, _equivalence, _gallery_data, _mutations
from tests.transpile import test_gallery_numeric_equivalence as _gallery_tier
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


# The sensitivity suite reuses the gallery tier's own backend table,
# shape table, dtype table, and cell registries rather than restating
# them. A mutant is a negative control for *this* check only if it is
# scored through exactly the harness the check uses; a private copy of
# the shape / dtype contract would drift and start measuring the
# sensitivity of a harness nobody runs.
_BACKENDS = _gallery_tier._BACKENDS_WITH_IMAGES

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE_DIR = _REPO_ROOT / "docs" / "examples" / "source"
_SCRATCH_ROOT = pathlib.Path("/tmp") / "qvr_equivalence_sensitivity"

_PointSection = dict[str, float | int | list[float] | list[int]]
_ProbePoint = dict[str, _PointSection]


# ---------------------------------------------------------------------
# QVR reference side, cached per example.
#
# The reference log-density vector depends on the example alone, not
# on the backend or the mutation, and every mutant of a given example
# is compared against the same vector. Computing it once per session
# keeps the suite's cost proportional to the number of container runs
# rather than to the number of QVR traces.
# ---------------------------------------------------------------------


class _Reference:
    """One example's dataset, point set, QVR log-densities, and
    observed-data count, memoised for the session.

    A plain holder rather than a `dx.Model`: its fields are a
    [`GalleryDataset`][tests.transpile._gallery_data.GalleryDataset]
    and a list of
    [`Point`][tests.transpile.probes._protocol.Point]s carrying torch
    tensors, none of which survive didactic's encode / decode round
    trip. Nothing here is serialised or compared structurally; it
    exists to keep one expensive computation from being repeated once
    per mutant.
    """

    def __init__(
        self,
        dataset: _gallery_data.GalleryDataset,
        points: list[Point],
        log_densities: list[float],
        n_obs: int,
    ) -> None:
        self.dataset = dataset
        self.points = points
        self.log_densities = log_densities
        self.n_obs = n_obs

    @property
    def atol(self) -> float:
        """The tolerance the equivalence check applies to this
        example, from the same adaptive estimator the gallery tier's
        contract is written against."""
        return _equivalence.adaptive_atol(n_obs=self.n_obs)

    @property
    def labels(self) -> list[str]:
        """Per-point perturbation labels, so a failure names which
        section moved."""
        return _gallery_data.perturbation_labels(len(self.points))


_REFERENCE_CACHE: dict[str, _Reference] = {}


def _gallery_reference(example: str) -> _Reference:
    """Build (or return the memoised) QVR reference for `example`.

    The QVR side is evaluated one point at a time with that point's
    own pre-shaped observations. The probe's `observations` keyword
    overrides the flat per-point payload, so handing it the dataset's
    ground-truth observations once for the whole set would score the
    reference at unperturbed data while the container scored the
    perturbed data, manufacturing a spread out of the harness rather
    than out of the program.
    """
    cached = _REFERENCE_CACHE.get(example)
    if cached is not None:
        return cached

    source_path = _SOURCE_DIR / f"{example}.qvr"
    dataset = _gallery_data.load_gallery_data(source_path)
    if dataset is None:
        raise AssertionError(
            f"{example!r}: `load_gallery_data` returned None, so the "
            f"sensitivity grid has no point set to mutate against. "
            f"Either the example's `.md` synthetic-data block broke, or "
            f"the grid names an example that never had one."
        )

    points = _gallery_data.points_from_dataset(dataset)
    if len(points) < 2:
        raise AssertionError(
            f"{example!r}: {len(points)} evaluation point(s). The "
            f"constant-spread contract is a statement about how the "
            f"difference varies across points, so no mutant can be "
            f"rejected on a point set this small."
        )

    probe = QvrProbe()
    scratch = _SCRATCH_ROOT / f"reference_{example}"
    scratch.mkdir(parents=True, exist_ok=True)
    source = source_path.read_bytes()
    log_densities: list[float] = []
    for point in points:
        log_densities.extend(
            probe.evaluate(
                source,
                example,
                [point],
                scratch=scratch,
                monadic=dataset.monadic,
                x_input=dataset.x_input,
                observations=_gallery_data.observations_for_point(
                    dataset,
                    point,
                ),
            ).log_densities
        )
    n_obs = sum(
        int(dataset.observations[name].numel())
        for name in _gallery_data.observed_data_names(dataset)
    )
    reference = _Reference(dataset, points, log_densities, max(n_obs, 1))
    _REFERENCE_CACHE[example] = reference
    return reference


def _emit(example_source: str, backend: str) -> str:
    """Transpile QVR source text to `backend` and decode it."""
    return transpile(parse(example_source), target=backend).decode("utf-8")


def _gallery_emit(example: str, backend: str) -> str:
    """Transpile the gallery example `example` to `backend`."""
    return _emit((_SOURCE_DIR / f"{example}.qvr").read_text(), backend)


def _require_image(backend: str) -> tuple[str, str, str]:
    """The `(image, extension, script)` triple for `backend`, or a
    configuration error when its image is missing."""
    image, ext, script_name = _BACKENDS[backend]
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it"
        )
    return image, ext, script_name


def _run_probe(
    *,
    backend: str,
    points: list[_ProbePoint],
    source_text: str,
    scratch_name: str,
    shapes: dict[str, list[int]] | None = None,
    dtypes: dict[str, str] | None = None,
) -> list[float]:
    """Run `backend`'s probe container on `source_text` at `points`."""
    image, ext, script_name = _require_image(backend)
    scratch = _SCRATCH_ROOT / scratch_name
    scratch.mkdir(parents=True, exist_ok=True)
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
        timeout=300.0,
    )
    return [float(value) for value in raw["log_densities"]]


def _gallery_probe(
    *,
    backend: str,
    example: str,
    reference: _Reference,
    source_text: str,
    tag: str,
) -> list[float]:
    """Score `source_text` on the gallery example's own point set."""
    return _run_probe(
        backend=backend,
        points=[
            {"params": point.params, "data": point.data} for point in reference.points
        ],
        source_text=source_text,
        scratch_name=f"{example}_{backend}_{tag}",
        shapes=_gallery_tier._shapes_from_dataset(reference.dataset),
        dtypes=_gallery_tier._dtypes_from_dataset(reference.dataset),
    )


def _spread(reference: list[float], target: list[float]) -> float:
    """`max_i |d_i - mean(d)|` for `d_i = target_i - reference_i`.

    The same quantity
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    bounds, recomputed here because the sensitivity suite needs the
    number itself (to report a margin and to compare it against a
    pinned floor) in addition to the pass/fail verdict. Every test that
    uses it also calls the real helper, so a divergence between the
    two surfaces rather than hiding.
    """
    if len(reference) != len(target):
        raise AssertionError(
            f"length mismatch: {len(reference)} reference values vs "
            f"{len(target)} target values"
        )
    diffs = [t - r for r, t in zip(reference, target)]
    for index, diff in enumerate(diffs):
        if not math.isfinite(diff):
            raise AssertionError(
                f"non-finite difference at point {index}: "
                f"reference={reference[index]!r} target={target[index]!r}. "
                f"A mutant that sends the log-density to -inf left the "
                f"model's support, which is a different failure from "
                f"denoting a different measure and is not what this "
                f"suite measures."
            )
    mean = sum(diffs) / len(diffs)
    return max(abs(diff - mean) for diff in diffs)


def _cell_skip_reason(backend: str, example: str) -> str | None:
    """Why the gallery tier does not numerically check this cell, or
    `None` when it does."""
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


# ---------------------------------------------------------------------
# Structural tests. No container, no Docker: these guard the grid
# itself, so a coverage regression surfaces even on a run where the
# probe images are unavailable.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation_name", "example", "backend"),
    _mutations.gallery_cells(),
    ids=str,
)
def test_mutated_cell_is_one_the_gallery_tier_checks(
    mutation_name: str, example: str, backend: str
) -> None:
    """Every mutated cell is a cell the equivalence tier numerically
    checks today.

    A negative control is evidence only about a check that actually
    runs on that cell. If the gallery tier starts skipping a cell the
    catalogue mutates (a new probe-shape gap, a newly pinned
    `UnsupportedConstruct`), then the mutant does not test the
    live suite, and this test says so rather than letting the
    rejection test pass on a cell nobody checks.
    """
    del mutation_name
    reason = _cell_skip_reason(backend, example)
    assert reason is None, (
        f"{backend!r} on {example!r} is mutated by the sensitivity "
        f"catalogue but the gallery equivalence tier no longer checks "
        f"it: {reason}. Either restore the cell in the gallery tier, or "
        f"move the mutation in `tests/transpile/_mutations.py` to a "
        f"cell that is still checked. Leaving it here would report a "
        f"sensitivity the live suite does not have."
    )


def test_catalogue_covers_every_backend_and_defect_class() -> None:
    """Every probe-imaged backend carries at least four mutations, and
    every defect class is exercised on at least four backends.

    The suite's claim is about the equivalence check as a whole, not
    about one lucky target. A backend with no negative control has
    never been shown to reject anything; a defect class pinned on a
    single backend says nothing about whether the other nine renderers
    would be caught making the same mistake.
    """
    per_backend: dict[str, set[str]] = {name: set() for name in _BACKENDS}
    per_class: dict[str, set[str]] = {}
    for mutation in (*_mutations.CATALOGUE, _mutations.MARGINALIZE_MUTATION):
        for mutant in mutation.mutants:
            per_backend.setdefault(mutant.backend, set()).add(mutation.name)
    for mutation in _mutations.CATALOGUE:
        for mutant in mutation.mutants:
            per_class.setdefault(mutation.defect_class, set()).add(mutant.backend)

    thin_backends = sorted(
        name for name, names in per_backend.items() if len(names) < 4
    )
    assert not thin_backends, (
        f"backends with fewer than four negative controls: "
        f"{thin_backends}. Each is a target whose rejection behaviour "
        f"is largely unmeasured; add mutations for it in "
        f"`tests/transpile/_mutations.py`."
    )

    thin_classes = sorted(
        name for name, backends in per_class.items() if len(backends) < 4
    )
    assert not thin_classes, (
        f"defect classes exercised on fewer than four backends: "
        f"{thin_classes}. A class pinned this narrowly does not support "
        f"a claim about the check."
    )

    # The marginalize class is scoped separately because it is bounded
    # by the renderers, not by the catalogue: only the targets that
    # lower the block to an explicit enumeration have a correct
    # baseline for a "replace the integral by one component" mutant to
    # deviate from.
    marginalize_backends = {
        mutant.backend for mutant in _mutations.MARGINALIZE_MUTATION.mutants
    }
    declared = {
        str(entry.backend) for entry in _mutations.MARGINALIZE_ENUMERATION_MARKERS
    }
    assert marginalize_backends == declared, (
        f"the marginalize mutation covers {sorted(marginalize_backends)} "
        f"but `MARGINALIZE_ENUMERATION_MARKERS` declares "
        f"{sorted(declared)}. Keep the two in step: the declaration is "
        f"what documents why the other targets are out."
    )


_TIGHTEST_MUTATION = "prior_term_flattened"
"""The catalogue's least-loud mutant, and therefore the number that
states the equivalence check's real sensitivity.

Flattening `ar1`'s `alpha` prior moves the log-density by only ~6.8e-3
nats across the point set, roughly 13x the 5e-4 tolerance floor. The
reason is structural rather than incidental: a weakly informative
`Normal(0, 5)` prior on a latent whose ground truth sits near 1
contributes about `-alpha^2 / 50`, and the point set perturbs `alpha`
by a fifth of its scale, so the term moves by hundredths of a nat
while a likelihood defect on the same example moves by tens. Any
dropped prior on a diffuse, small-magnitude latent lands in this
regime, and a tolerance an order of magnitude looser than today's
would stop catching it."""


def test_tightest_catalogue_margin_is_declared() -> None:
    """Every pinned floor clears the tolerance, and the tightest one
    is the mutation this suite declares as its sensitivity limit.

    Reporting a suite's sensitivity means reporting its *worst* case.
    A catalogue full of 100-nat mutants detects only a gross
    defect is caught; the number that matters is the smallest spread
    any modelled defect produces, because that is where the check
    stops being able to tell a wrong program from a right one. Adding
    a quieter mutation than the current tightest is a real change to
    that claim, so it has to move this declaration too.
    """
    atol_floor = _equivalence.adaptive_atol(n_obs=0)
    entries = [
        (str(mutation.name), float(mutation.min_spread))
        for mutation in (
            *_mutations.CATALOGUE,
            _mutations.MARGINALIZE_MUTATION,
        )
    ]
    under_tolerance = sorted(name for name, floor in entries if floor <= atol_floor)
    assert not under_tolerance, (
        f"mutations whose pinned floor sits at or below the "
        f"equivalence tolerance {atol_floor:.6e}: {under_tolerance}. A "
        f"floor that low cannot distinguish a rejected mutant from "
        f"round-off, so the mutation needs a louder realisation rather "
        f"than a lower floor."
    )

    tightest = min(entries, key=lambda entry: entry[1])
    assert tightest[0] == _TIGHTEST_MUTATION, (
        f"the catalogue's tightest margin is now {tightest[0]!r} at "
        f"{tightest[1]:.6f} nats, not {_TIGHTEST_MUTATION!r}. The "
        f"suite's declared sensitivity is its worst case, so update "
        f"`_TIGHTEST_MUTATION` and its rationale to describe the new "
        f"limiting defect."
    )


def test_catalogue_names_are_unique() -> None:
    """Catalogue entries are addressed by name, so the names must be
    distinct or a lookup silently resolves to the wrong mutation."""
    names = [
        mutation.name
        for mutation in (
            *_mutations.CATALOGUE,
            _mutations.MARGINALIZE_MUTATION,
        )
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate mutation names in the catalogue: {duplicates}"


# ---------------------------------------------------------------------
# Rejection: the catalogue's mutants must all fail the check.
# ---------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    ("mutation_name", "example", "backend"),
    _mutations.gallery_cells(),
    ids=str,
)
def test_mutant_is_rejected(mutation_name: str, example: str, backend: str) -> None:
    """The equivalence check rejects the mutant, by a measured margin.

    Three assertions, in order of strength:

    1. The observed spread exceeds the equivalence tolerance, so the
       check the gallery tier runs would fail on this program.
    2. The real
       [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
       helper does raise on the mutant's vectors, so the rejection is
       the live check's verdict and not this module's arithmetic.
    3. The observed spread exceeds the mutation's pinned
       `min_spread`. This sensitivity floor detects a point set that loses
       its variation shrinks every mutant's spread long before any of
       them drops under the tolerance, and the floor trips first.
    """
    mutation = _mutations.mutation_by_name(mutation_name)
    context = f"{mutation_name}/{backend}@{example}"
    reference = _gallery_reference(example)

    mutant_source = _mutations.apply_rewrites(
        _gallery_emit(example, backend),
        _mutations.rewrites_for(mutation, backend),
        context=context,
    )
    target = _gallery_probe(
        backend=backend,
        example=example,
        reference=reference,
        source_text=mutant_source,
        tag=mutation_name,
    )

    spread = _spread(reference.log_densities, target)
    atol = reference.atol

    assert spread > atol, (
        f"{context}: the equivalence check ACCEPTS this mutant (spread "
        f"{spread:.6e} <= atol {atol:.6e}). This is a hole in the "
        f"check, not a flaw in the mutation: the emitted program "
        f"denotes a different measure ({mutation.defect_class}; "
        f"{mutation.provenance}) and the constant-spread contract "
        f"cannot see it. Record it in `_mutations.BLIND_SPOTS` with the "
        f"reason it is invisible, and treat the gap as a finding about "
        f"the check."
    )
    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            reference.log_densities,
            target,
            atol=atol,
            context=context,
            labels=reference.labels,
            min_points=2,
        )
    assert spread >= mutation.min_spread, (
        f"{context}: the mutant is still rejected but only by "
        f"{spread:.6f} nats, under the pinned floor of "
        f"{mutation.min_spread:.6f}. The mutation did not change; the "
        f"evaluation points did. A point set whose latents or data "
        f"stopped moving shrinks every mutant's spread toward zero, and "
        f"this floor trips while the mutants are still nominally "
        f"rejected. Confirm the example's point set still perturbs both "
        f"sections before touching this number."
    )


# ---------------------------------------------------------------------
# Acceptance: a constant offset denotes the same measure.
# ---------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "control",
    _mutations.ACCEPTED,
    ids=lambda control: f"{control.name}-{control.backend}",
)
def test_constant_offset_is_accepted(
    control: _mutations.AcceptedRewrite,
) -> None:
    """A point-independent shift of the log-density passes.

    This is the positive control that keeps every rejection above
    meaningful. A check that failed on all input would satisfy the
    rejection tests and be useless. The equivalence contract quotients
    by exactly this constant, and a check that rejected it would fire
    on every benign base-measure difference between two targets.
    """
    context = f"{control.name}/{control.backend}@{control.example}"
    reference = _gallery_reference(control.example)

    shifted_source = _mutations.apply_rewrites(
        _gallery_emit(control.example, control.backend),
        control.rewrites,
        context=context,
    )
    target = _gallery_probe(
        backend=control.backend,
        example=control.example,
        reference=reference,
        source_text=shifted_source,
        tag=f"accepted_{control.name}",
    )
    _equivalence.assert_log_density_match(
        reference.log_densities,
        target,
        atol=reference.atol,
        context=f"{context} ({control.rationale})",
        labels=reference.labels,
        min_points=2,
    )


# ---------------------------------------------------------------------
# Blind spots: rewrites the check provably misses.
# ---------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "blind_spot",
    _mutations.BLIND_SPOTS,
    ids=lambda spot: f"{spot.name}-{spot.backend}",
)
def test_known_blind_spot_stays_invisible(
    blind_spot: _mutations.BlindSpot,
) -> None:
    """A registered blind spot still produces no spread.

    These are not mutations the check ought to catch and misses by
    accident; each is a defect outside what a log-density comparison
    can observe, and the registry says which. Pinning them as
    assertions turns three claims about the check's reach into
    measured facts, and a failure here means the check gained a
    capability (or the harness gained a coupling it should not have)
    and the registry needs re-deriving.
    """
    context = f"{blind_spot.name}/{blind_spot.backend}@{blind_spot.example}"
    reference = _gallery_reference(blind_spot.example)

    rewritten_source = _mutations.apply_rewrites(
        _gallery_emit(blind_spot.example, blind_spot.backend),
        blind_spot.rewrites,
        context=context,
    )
    target = _gallery_probe(
        backend=blind_spot.backend,
        example=blind_spot.example,
        reference=reference,
        source_text=rewritten_source,
        tag=f"blind_{blind_spot.name}",
    )

    spread = _spread(reference.log_densities, target)
    atol = reference.atol
    assert spread <= atol, (
        f"{context}: this rewrite is registered as invisible to the "
        f"constant-spread check ({blind_spot.why_invisible}) but it now "
        f"produces a spread of {spread:.6e}, above atol {atol:.6e}. "
        f"Either the check reaches further than the registry claims, or "
        f"the harness picked up a coupling that makes the rewrite "
        f"observable for the wrong reason. Re-derive the entry before "
        f"deleting it."
    )
    _equivalence.assert_log_density_match(
        reference.log_densities,
        target,
        atol=atol,
        context=context,
        labels=reference.labels,
        min_points=2,
    )


# ---------------------------------------------------------------------
# The marginalize leg.
#
# No gallery example with live cells carries a `marginalize` block, so
# the "integrated marginal replaced by a live draw" defect is measured
# on a dedicated mixture fixture whose Stan emit enumerates the
# discrete class with `log_sum_exp`.
# ---------------------------------------------------------------------


def _mixture_probe_points() -> list[_ProbePoint]:
    """The mixture point set, in QVR's zero-based coding on every target.

    Every target reads the grouping fibration `idx` as QVR holds it;
    a one-based target's program reads it one higher itself.
    """
    return [
        {
            "params": {
                "probs": [float(p) for p in point.probs],
                "mu_low": float(point.mu_low),
                "mu_diff": float(point.mu_diff),
            },
            "data": {
                "y": [float(v) for v in point.response],
                "idx": list(range(len(point.response))),
            },
        }
        for point in _mutations.MARGINALIZE_POINTS
    ]


_MIXTURE_REFERENCE_CACHE: list[float] = []


def _mixture_reference() -> list[float]:
    """QVR marginal joint at each mixture point, under float64.

    The shared [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] lifts
    every scalar parameter to a shape-`(1,)` tensor, but `mu_low` and
    `mu_diff` feed a `factor c : Cls in ...` expression and must arrive
    rank-0 so the resulting `(2,)` factor does not pick up a phantom
    trailing axis that
    [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]
    rejects. The coercion below is shape-aware for that reason, and the
    grouping fibration `idx` rides on `torch.long` so its `index_add`
    sees an integer index.
    """
    if _MIXTURE_REFERENCE_CACHE:
        return _MIXTURE_REFERENCE_CACHE
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        program = Compiler(parse(_mutations.MARGINALIZE_SOURCE)).compile()
        monadic = program.morphism
        if not isinstance(monadic, MonadicProgram):
            raise AssertionError(
                f"marginalize fixture compiled to "
                f"{type(monadic).__name__}, not a MonadicProgram; the "
                f"reference trace has nothing to walk."
            )
        log_densities: list[float] = []
        for point in _mutations.MARGINALIZE_POINTS:
            response = [float(v) for v in point.response]
            observations: dict[str, torch.Tensor] = {
                "probs": torch.tensor(
                    [float(p) for p in point.probs],
                    dtype=torch.float64,
                ),
                "mu_low": torch.tensor(
                    float(point.mu_low),
                    dtype=torch.float64,
                ),
                "mu_diff": torch.tensor(
                    float(point.mu_diff),
                    dtype=torch.float64,
                ),
                "y": torch.tensor(response, dtype=torch.float64),
                "idx": torch.tensor(
                    list(range(len(response))),
                    dtype=torch.long,
                ),
            }
            result = trace(
                monadic,
                torch.zeros(1, 1, dtype=torch.float64),
                observations=observations,
            )
            if result.log_joint is None:
                raise AssertionError(
                    "marginalize fixture: the reference trace returned a None log_joint"
                )
            log_densities.append(float(result.log_joint.item()))
    finally:
        torch.set_default_dtype(prior_dtype)
    _MIXTURE_REFERENCE_CACHE.extend(log_densities)
    return log_densities


@pytest.mark.parametrize(
    "entry",
    _mutations.MARGINALIZE_ENUMERATION_MARKERS,
    ids=lambda entry: str(entry.backend),
)
def test_marginalize_fixture_is_transpiled_as_an_enumeration(
    entry: _mutations.EnumerationMarker,
) -> None:
    """The mixture fixture's emit really enumerates the discrete
    latent on every target the marginalize mutation is measured on.

    The rejection test below mutates the enumeration out of the emit.
    If a renderer ever stopped emitting one, the anchor would vanish
    and [`apply_rewrites`][tests.transpile._mutations.apply_rewrites]
    would fail with a message that reads like catalogue drift. This
    test names the real cause instead: the marginalize lowering itself
    regressed to a sampled discrete latent, which is the defect the
    mutation exists to model.
    """
    backend = str(entry.backend)
    marker = str(entry.marker)
    emitted = _emit(_mutations.MARGINALIZE_SOURCE, backend)
    assert marker in emitted, (
        f"the {backend} emit for the marginalize fixture no longer "
        f"contains {marker!r}, so the discrete latent is not being "
        f"enumerated. The transpiled program is then the very defect "
        f"the `integrated_marginal_replaced_by_draw` mutation models, "
        f"and the equivalence tier should be failing on it."
    )


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "backend",
    [mutant.backend for mutant in _mutations.MARGINALIZE_MUTATION.mutants],
    ids=str,
)
def test_marginalize_fixture_baseline_is_accepted(backend: str) -> None:
    """The unmutated marginalize fixture passes the check.

    The mixture fixture is not a gallery example, so the gallery tier
    never scores it and the rejection test below would otherwise have
    no positive control: a mutant that "fails" a comparison whose
    baseline also fails says nothing. This asserts the enumerated emit
    agrees with the QVR marginal at every point of the mixture grid.
    """
    reference = _mixture_reference()
    target = _run_probe(
        backend=backend,
        points=_mixture_probe_points(),
        source_text=_emit(_mutations.MARGINALIZE_SOURCE, backend),
        scratch_name=f"marginalize_{backend}_baseline",
    )
    _equivalence.assert_log_density_match(
        reference,
        target,
        atol=_equivalence.adaptive_atol(
            n_obs=len(_mutations.MARGINALIZE_POINTS[0].response)
        ),
        context=f"marginalize baseline/{backend}@normal_mix",
        min_points=2,
    )


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "backend",
    [mutant.backend for mutant in _mutations.MARGINALIZE_MUTATION.mutants],
    ids=str,
)
def test_integrated_marginal_replaced_by_draw_is_rejected(
    backend: str,
) -> None:
    """Replacing a marginalize block's enumeration with one
    component's contribution is rejected.

    This is the shipped defect where a `marginalize` lowered to a live
    sample of the discrete latent: the emitted program then scores the
    joint at one class rather than the sum over classes. The difference
    is constant in neither the latents nor the data, so the
    constant-spread contract sees it, but only on a fixture that
    carries a marginalize block at all, which no gallery example with
    live cells does.
    """
    mutation = _mutations.MARGINALIZE_MUTATION
    context = f"{mutation.name}/{backend}@{mutation.example}"

    reference = _mixture_reference()
    mutant_source = _mutations.apply_rewrites(
        _emit(_mutations.MARGINALIZE_SOURCE, backend),
        _mutations.rewrites_for(mutation, backend),
        context=context,
    )
    target = _run_probe(
        backend=backend,
        points=_mixture_probe_points(),
        source_text=mutant_source,
        scratch_name=f"marginalize_{backend}_{mutation.name}",
    )

    spread = _spread(reference, target)
    atol = _equivalence.adaptive_atol(
        n_obs=len(_mutations.MARGINALIZE_POINTS[0].response)
    )
    assert spread > atol, (
        f"{context}: the equivalence check ACCEPTS a marginalize block "
        f"lowered to a single component (spread {spread:.6e} <= atol "
        f"{atol:.6e}). The emitted program computes a conditional joint "
        f"where the QVR reference computes a marginal, so this is a "
        f"hole in the check."
    )
    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            reference,
            target,
            atol=atol,
            context=context,
            min_points=2,
        )
    assert spread >= mutation.min_spread, (
        f"{context}: rejected by only {spread:.6f} nats, under the "
        f"pinned floor of {mutation.min_spread:.6f}. The mixture "
        f"fixture's point set lost variation; re-derive the floor only "
        f"after confirming the grid still moves both sections."
    )
