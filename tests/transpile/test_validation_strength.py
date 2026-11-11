"""Tests for the sensitivity of the transpile equivalence check.

The operational check evaluates QVR and backend log densities on a
finite point set, subtracts their mean difference, and bounds the
remaining spread. This module checks the parts of that approximation
that can weaken independently:

1. tolerance constants and the adaptive-tolerance ceiling;
2. point count and deterministic defaults;
3. coordinate variation across several fixed seeds;
4. finite, varying QVR reference values;
5. rejection of a planted defect on every non-exempt coordinate;
6. the added sensitivity of a wider excursion to nonlinear defects;
7. the spread statistic on sequences with known behavior.

The coordinate checks cover latent sites and observed arrays. Frozen
structural coordinates require entries in
`_UNPERTURBABLE_COORDINATES`. The excursion tests measure realized
point distances because support-preserving redraws may reduce the
requested scale.
"""

from __future__ import annotations

import inspect
import math
import pathlib
from collections.abc import Sequence
from typing import Literal

import didactic.api as dx
import pytest

from quivers.continuous import Draw
from tests.transpile import (
    _equivalence,
    _gallery_data,
    test_gallery_numeric_equivalence as _gallery_equivalence,
)
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


# ---------------------------------------------------------------------------
# Pinned values. Every constant here mirrors a value the equivalence
# check reads at run time; the tests assert the mirror still matches.
# ---------------------------------------------------------------------------

_PINNED_DEFAULT_ATOL = 5e-4
"""Mirror of
[`_DEFAULT_ATOL`][tests.transpile._equivalence._DEFAULT_ATOL]."""

_PINNED_PER_OBS_ROUNDOFF_ESTIMATE = 5e-16
"""Mirror of
[`_PER_OBS_ROUNDOFF_ESTIMATE`][tests.transpile._equivalence._PER_OBS_ROUNDOFF_ESTIMATE]."""

_PINNED_TOLERANCE_HEADROOM = 100.0
"""Mirror of
[`_TOLERANCE_HEADROOM`][tests.transpile._equivalence._TOLERANCE_HEADROOM]."""

_ADAPTIVE_ATOL_CEILING = 1e-3
"""Hard ceiling the adaptive tolerance must stay under for every
fixture size the gallery can realistically reach.

The round-off term of
[`adaptive_atol`][tests.transpile._equivalence.adaptive_atol] is
``n_obs * condition_number * 5e-16 * 100``, so at
``condition_number = 1`` it only overtakes the 5e-4 floor once
``n_obs`` passes 1e10. Well-conditioned fixtures therefore evaluate
at the floor, and the ceiling asserts exactly that: no realistic
observation count may buy a fixture a looser tolerance."""

_LARGEST_REALISTIC_N_OBS = 100_000
"""Two orders of magnitude above the largest gallery fixture
(``zip_regression`` scores 1200 observed entries), so the ceiling is
asserted well past any size the corpus can grow into."""

_SMALLEST_SEMANTIC_DISCREPANCY = 1e-2
"""Per-point log-density discrepancy of the cheapest real bug the
check must catch: a swapped distribution argument on the gallery's
parameter ranges. The tolerance model is only meaningful while it
sits orders of magnitude below this, which
`test_adaptive_atol_stays_under_ceiling` asserts."""

_BUG_DETECTION_MARGIN = 10.0
"""Required ratio between
[`_SMALLEST_SEMANTIC_DISCREPANCY`][tests.transpile.test_validation_strength._SMALLEST_SEMANTIC_DISCREPANCY]
and the tolerance ceiling. A ratio near 1 would mean the tolerance
and the smallest real bug are the same size, so a rounding
difference and a semantic error would be indistinguishable."""

_MIN_GALLERY_POINTS = 6
"""Smallest point count the gallery equivalence check may run on.

Point 0 is the ground truth and the schedule cycles latents-only,
data-only, latents+data. Six points give two latents-only, two
data-only, and one joint perturbation, so a broken constancy
localises to the section that moved instead of merely reporting that
some point disagreed. Below six the schedule stops covering each
mode twice, and at one point the assertion is a tautology."""

_MIN_JOINT_VARIATION_IN_ATOL = 100.0
"""Required ratio between the range of the reference joint across the
point set and the equivalence tolerance.

A point set whose reference measure moves by less than the tolerance
cannot separate a correct backend from a wrong one: any discrepancy
the perturbation could expose is smaller than the noise the check
already forgives. Requiring two orders of magnitude of headroom keeps
"the points moved" from degenerating into "the points moved within
round-off"."""

_MIN_SWEPT_SEEDS = 4
"""Smallest number of independent draws the per-cell claims must be
proved on.

One draw cannot distinguish a property of the perturbation design from
an accident of where that draw landed, which is the seed-locality gap
in one sentence. Two draws agreeing is weak evidence, since the second
is one coin flip; four is the point at which a coordinate that moves
far enough to expose a defect at every seed is doing so because the
perturber steps it, not because a draw was kind. The floor is pinned
rather than read from
[`GALLERY_SEEDS`][tests.transpile._gallery_data.GALLERY_SEEDS] so that
shrinking the sweep back toward a single seed fails here instead of
silently narrowing every per-cell claim in this module at once."""

_DEFAULT_GALLERY_SEED = 0
"""The seed every gallery caller takes by default.

Reproducibility and coverage pull in opposite directions here. A seed
drawn from the clock would sweep the support over a run history but
would turn a genuine regression into an intermittent one, so the
default stays fixed and the *sweep* is what covers the support."""

_WIDE_EXCURSION_GAIN = 1.5
"""How much further the wide point set must travel before this module
holds it to the nonlinear-sensitivity claim.

The perturbation scale only nominally controls the excursion: the
redraw ladder halves it whenever a draw leaves the support, and a
coordinate whose step is an integer clamped to its attested range does
not move further however large the scale is. So the claim is gated on
the excursion the set *measurably achieved*, read back off the points
by
[`point_excursion`][tests.transpile._gallery_data.point_excursion],
rather than on the scale that was requested.

Half again as far is where the gate sits because the measured
relationship is clean there: across the corpus at sixteen seeds, every
cell whose wide set travelled at least 1.5x further detected a
quadratic defect at least 2.4x smaller, while cells gated in at 1.05x
included one whose quadratic sensitivity was marginally *worse* (a
beta-binomial whose excursion is dominated by integer counts that the
wider scale merely reshuffles inside their attested window)."""

_NONLINEAR_DEFECT_HEADROOM = 0.9
"""Fraction of the tolerance the planted nonlinear defect is
calibrated to occupy on the *default* point set.

The defect has to be genuinely invisible at the default excursion for
the wide set's rejection of it to mean anything, and it has to be as
close to the boundary as possible for the demonstration to be about
the excursion rather than about a defect chosen absurdly small. 0.9
sits just inside the accept region, so the rejection at the wider
excursion is attributable to the excursion alone."""

_MIN_WIDE_GAIN_FRACTION = 0.5
"""Fraction of the swept (example, seed) pairs whose wide point set
must clear
[`_WIDE_EXCURSION_GAIN`][tests.transpile.test_validation_strength._WIDE_EXCURSION_GAIN].

`test_wider_excursion_detects_a_nonlinear_defect` is a conditional,
and a conditional whose antecedent stops holding anywhere is a test
that passes by never running. The measured fraction is about
two-thirds; requiring half keeps the demonstration alive across corpus
growth without pinning a number that a single new discrete-support
example would break."""


# ---------------------------------------------------------------------------
# Registry of coordinates that are outside the quantified space.
# ---------------------------------------------------------------------------


class UnperturbableCoordinate(dx.Model):
    """One (example, coordinate) pair the point set holds fixed, with
    the reason the coordinate is not a dimension Theorem 4.1
    quantifies over.

    `kind` is the structural class of the coordinate and
    `justification` is the written argument for the exemption. An
    entry is a claim the suite checks in both directions: a
    coordinate listed here must be frozen across the whole point set,
    and a coordinate not listed here must move. An unexplained frozen
    coordinate fails.
    """

    example: str
    coordinate: str
    kind: Literal["plate-subscript"]
    justification: str


_PLATE_SUBSCRIPT_ARGUMENT = (
    "A plate subscript is fixed by the experimental design rather "
    "than drawn from the model, so it is a structural input on the "
    "same footing as a FinSet size: it is not a coordinate of the "
    "(theta, y) support Theorem 4.1 quantifies over, and the "
    "constant c is allowed to depend on it. Freezing it does not "
    "blind the check to a mis-gathered index, because the gathered "
    "latent and the scored response both move across the point set, "
    "so a permuted or off-by-one gather shifts the per-row mean by "
    "an amount that varies with the latents and breaks constancy."
)

_UNPERTURBABLE_COORDINATES: tuple[UnperturbableCoordinate, ...] = tuple(
    UnperturbableCoordinate(
        example=_example,
        coordinate=_coordinate,
        kind="plate-subscript",
        justification=f"{_detail}. {_PLATE_SUBSCRIPT_ARGUMENT}",
    )
    for _example, _coordinate, _detail in (
        (
            "beta_binomial_ab_test",
            "arm_idx",
            "Gathers conc1 and conc0 over the Arm plate",
        ),
        (
            "beta_regression",
            "out_idx",
            "Gathers beta_0, beta_1 and phi over the Out plate",
        ),
        (
            "factor_analysis",
            "item_idx",
            "Gathers the per-item factor-score row of Z over the Item plate",
        ),
        (
            "factor_analysis",
            "obs_idx",
            "Gathers the loading row of W and the uniqueness psi over the ObsDim plate",
        ),
        (
            "gamma_regression",
            "cat_idx",
            "Gathers beta_0, beta_1 and shape over the Cat plate",
        ),
        (
            "half_student_t_hierarchical",
            "group_idx",
            "Gathers the per-group offset u over the Group plate",
        ),
        (
            "irt_2pl",
            "item_idx",
            "Gathers difficulty and discrim over the Item plate",
        ),
        (
            "irt_2pl",
            "person_idx",
            "Gathers ability over the Person plate",
        ),
        (
            "lda",
            "word_idx",
            "Names the fibration from each word position into its "
            "document, selecting the theta row the per-word mixture "
            "marginal is scored under",
        ),
        (
            "negbin_regression",
            "out_idx",
            "Gathers beta_0, beta_1 and dispersion over the Out plate",
        ),
        (
            "pmf",
            "movie_idx",
            "Gathers the movie factor row of V over the Movie plate",
        ),
        (
            "pmf",
            "user_idx",
            "Gathers the user factor row of U over the User plate",
        ),
        (
            "ppca",
            "item_idx",
            "Gathers the per-item latent code row of Z over the Item plate",
        ),
        (
            "ppca",
            "obs_idx",
            "Gathers the loading row of W over the ObsDim plate",
        ),
        (
            "tree_categorical",
            "verb_idx",
            "Gathers each response's row of leaf probabilities over the Verb plate",
        ),
        (
            "zip_regression",
            "out_idx",
            "Gathers alpha_zero, beta_zero, alpha_rate and beta_rate "
            "over the Out plate",
        ),
    )
)
"""Every coordinate the gallery point set holds fixed on purpose.

Populate an entry only for a coordinate that is genuinely outside the
quantified space. A covariate whose value enters the density is
inside it: `x` is perturbed in every regression example, so any other
covariate that the harness merely fails to move is a coverage gap to
close in the perturber, not an entry to add here.

Which of the two a given coordinate is, is not left to the entry's
prose. `test_every_quantified_coordinate_varies` checks every
`plate-subscript` entry against
[`structural_subscript_names`][tests.transpile._gallery_data.structural_subscript_names],
the classifier the point builder itself consults, so the registry can
only ever exempt a coordinate the perturber independently reads as a
plate subscript. A frozen coordinate the classifier does not name,
`n_trials` being the count parameter of a Beta-Binomial rather than
an index into a plate, cannot be registered at all and has to be
fixed in the perturber."""


_MIN_JUSTIFICATION_CHARS = 80
"""Length floor on an exemption's written justification. A one-word
reason is how a registry of arguments decays into a list of names."""


# ---------------------------------------------------------------------------
# Cached per-example evaluation.
# ---------------------------------------------------------------------------


class _EvaluatedPointSet(dx.Model):
    """Cached point set and reference densities for one example, seed, and scale.

    The cache key includes all three inputs. Point generation uses a local
    deterministic RNG, so cached and uncached evaluations agree.
    """

    stem: str
    seed: int
    scale: float
    dataset: _gallery_data.GalleryDataset = dx.field(opaque=True)
    points: tuple[Point, ...] = dx.field(opaque=True)
    log_densities: tuple[float, ...]


_EVALUATION_CACHE: dict[tuple[str, int, float], _EvaluatedPointSet] = {}


def _gallery_cells() -> list[pathlib.Path]:
    """Every gallery example whose `.md` ships a synthetic-data block
    and whose point set the reference oracle can score."""
    return [
        example
        for example in _gallery_data.gallery_examples_with_data()
        if example.stem not in _gallery_equivalence._SKIP_DATASET_LOAD_FAILED
        and example.stem not in _gallery_equivalence._SKIP_QVR_INCOMPATIBLE
    ]


def _seeded_cells() -> list[tuple[pathlib.Path, int]]:
    """Return every `(example, seed)` pair used by per-cell checks."""
    return [
        (example, seed)
        for example in _gallery_cells()
        for seed in _gallery_data.GALLERY_SEEDS
    ]


def _cell_id(value: pathlib.Path | int) -> str:
    """Parameter id for a `(example, seed)` pytest parametrisation."""
    if isinstance(value, pathlib.Path):
        return value.stem
    return f"seed{value}"


def _evaluate(
    example: pathlib.Path,
    seed: int = _DEFAULT_GALLERY_SEED,
    scale: float = _gallery_data.PERTURBATION_SCALE,
) -> _EvaluatedPointSet:
    """Load `example`, build its point set at `seed` and `scale`, and
    score the reference joint at every point."""
    key = (example.stem, seed, scale)
    cached = _EVALUATION_CACHE.get(key)
    if cached is not None:
        return cached
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example is in neither "
        f"`_SKIP_DATASET_LOAD_FAILED` nor `_SKIP_QVR_INCOMPATIBLE`."
    )
    points = _gallery_data.points_from_dataset(
        dataset,
        seed=seed,
        scale=scale,
    )
    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_strength_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    log_densities: list[float] = []
    for point in points:
        # One probe call per point. The probe's `observations` keyword
        # overrides the flat per-point payload, so a perturbed point
        # needs its own pre-shaped observation dict; passing the
        # dataset's ground-truth observations for the whole set would
        # score the reference at unperturbed data.
        log_densities.extend(
            probe.evaluate(
                source,
                example.stem,
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
    evaluated = _EvaluatedPointSet(
        stem=example.stem,
        seed=seed,
        scale=scale,
        dataset=dataset,
        points=tuple(points),
        log_densities=tuple(float(lp) for lp in log_densities),
    )
    _EVALUATION_CACHE[key] = evaluated
    return evaluated


def _latent_site_names(
    dataset: _gallery_data.GalleryDataset,
) -> tuple[str, ...]:
    """Post-inline names of the compiled program's latent sample
    sites.

    Derived here rather than imported so this module measures the
    quantified latent dimensions independently of the helper the
    point builder uses to fill them: a bug that dropped a site from
    the ground-truth capture would otherwise also drop it from the
    coverage check.
    """
    monadic = dataset.monadic
    if monadic is None:
        return ()
    names: list[str] = []
    for spec in monadic.steps:
        if isinstance(spec, Draw):
            names.extend(spec.names)
    return tuple(names)


def _coordinate_key(
    value: float | int | list[float] | list[int],
) -> tuple[float, ...]:
    """Hashable identity of one point-section entry, for counting
    distinct values across the point set."""
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(entry) for entry in value)


def _distinct_values(
    points: tuple[Point, ...],
    name: str,
    *,
    latent: bool,
) -> int:
    """Number of distinct values `name` takes across `points`."""
    seen: set[tuple[float, ...]] = set()
    for point in points:
        section = point.params if latent else point.data
        if name in section:
            seen.add(_coordinate_key(section[name]))
    return len(seen)


def _registered(stem: str) -> dict[str, UnperturbableCoordinate]:
    """Registry entries for one example, keyed by coordinate name."""
    return {
        entry.coordinate: entry
        for entry in _UNPERTURBABLE_COORDINATES
        if entry.example == stem
    }


def _diff_pair(
    diffs: list[float],
    base: float = -137.25,
) -> tuple[list[float], list[float]]:
    """A (qvr, target) sequence pair realising `diffs` exactly.

    The reference side walks a spread of realistic joint values so the
    constructed case exercises the same subtraction the gallery check
    performs, rather than differencing two constants.
    """
    qvr = [base - 3.5 * index for index in range(len(diffs))]
    return qvr, [q + d for q, d in zip(qvr, diffs)]


def _spread(values: Sequence[float]) -> float:
    """Compute the maximum deviation of a sequence from its mean.

    This mirrors the statistic used by `assert_log_density_match` and lets
    the tests calibrate synthetic defects analytically.
    """
    if not values:
        raise ValueError("the spread of an empty sequence is undefined")
    mean = sum(values) / len(values)
    return max(abs(value - mean) for value in values)


# ---------------------------------------------------------------------------
# 1. The tolerance model cannot be silently loosened.
# ---------------------------------------------------------------------------


def test_tolerance_constants_are_pinned() -> None:
    """Check the three derived constants in the tolerance model.

    `_DEFAULT_ATOL` is 5e-4, `_PER_OBS_ROUNDOFF_ESTIMATE` is 5e-16,
    and `_TOLERANCE_HEADROOM` is 100. Changes require updating the
    independently recorded values in this module.
    """
    assert _equivalence._DEFAULT_ATOL == _PINNED_DEFAULT_ATOL, (
        f"`_DEFAULT_ATOL` moved to "
        f"{_equivalence._DEFAULT_ATOL!r} from the pinned "
        f"{_PINNED_DEFAULT_ATOL!r}. The floor is derived from measured "
        f"cross-backend agreement, so re-derive it before changing it "
        f"and update this mirror in the same commit."
    )
    assert (
        _equivalence._PER_OBS_ROUNDOFF_ESTIMATE == _PINNED_PER_OBS_ROUNDOFF_ESTIMATE
    ), (
        f"`_PER_OBS_ROUNDOFF_ESTIMATE` moved to "
        f"{_equivalence._PER_OBS_ROUNDOFF_ESTIMATE!r} from the pinned "
        f"{_PINNED_PER_OBS_ROUNDOFF_ESTIMATE!r}. This is a measured "
        f"per-log_prob float64 round-off; re-measure it across the "
        f"backends before changing it."
    )
    assert _equivalence._TOLERANCE_HEADROOM == _PINNED_TOLERANCE_HEADROOM, (
        f"`_TOLERANCE_HEADROOM` moved to "
        f"{_equivalence._TOLERANCE_HEADROOM!r} from the pinned "
        f"{_PINNED_TOLERANCE_HEADROOM!r}. The multiplier is the "
        f"headroom above per-point round-off, not a dial to turn when "
        f"a fixture fails."
    )


def test_adaptive_atol_stays_under_ceiling() -> None:
    """Bound adaptive tolerances for realistic observation counts.

    At unit conditioning, the round-off term remains below the 5e-4 floor
    through the tested range. Returned tolerances must equal the floor, be
    monotone, and stay below the 1e-3 ceiling.
    """
    assert _ADAPTIVE_ATOL_CEILING >= _PINNED_DEFAULT_ATOL, (
        "the ceiling must sit at or above the floor, otherwise it "
        "asserts nothing about the adaptive term."
    )
    assert (
        _SMALLEST_SEMANTIC_DISCREPANCY >= _BUG_DETECTION_MARGIN * _ADAPTIVE_ATOL_CEILING
    ), (
        f"the tolerance ceiling {_ADAPTIVE_ATOL_CEILING!r} leaves less "
        f"than a factor of {_BUG_DETECTION_MARGIN!r} below the "
        f"cheapest semantic discrepancy "
        f"{_SMALLEST_SEMANTIC_DISCREPANCY!r}; at that ratio a real bug "
        f"and float round-off are the same size."
    )

    sizes = [1, 8, 64, 200, 1_200, 10_000, _LARGEST_REALISTIC_N_OBS]
    previous = 0.0
    for n_obs in sizes:
        atol = _equivalence.adaptive_atol(n_obs=n_obs)
        assert atol <= _ADAPTIVE_ATOL_CEILING, (
            f"adaptive_atol(n_obs={n_obs}) = {atol!r} exceeds the "
            f"ceiling {_ADAPTIVE_ATOL_CEILING!r}. Either the round-off "
            f"estimate or the headroom grew; re-derive the round-off "
            f"argument rather than raising the ceiling."
        )
        assert atol == _PINNED_DEFAULT_ATOL, (
            f"adaptive_atol(n_obs={n_obs}) = {atol!r} but the "
            f"round-off term at this size is "
            f"{n_obs * _PINNED_PER_OBS_ROUNDOFF_ESTIMATE * _PINNED_TOLERANCE_HEADROOM!r}, "
            f"far under the {_PINNED_DEFAULT_ATOL!r} floor, so the "
            f"floor must dominate."
        )
        assert atol >= previous, (
            f"adaptive_atol is not monotone in n_obs: {atol!r} at "
            f"{n_obs} follows {previous!r}. A non-monotone estimator "
            f"tightens the bug-detection threshold as coverage grows."
        )
        previous = atol

    for degenerate in (0, -1):
        assert _equivalence.adaptive_atol(n_obs=degenerate) == _PINNED_DEFAULT_ATOL, (
            "a non-positive observation count must fall back to the floor."
        )
    assert (
        _equivalence.adaptive_atol(n_obs=64, condition_number=0.0)
        == _PINNED_DEFAULT_ATOL
    ), "a degenerate condition number must never drop below the floor."


# ---------------------------------------------------------------------------
# 2. The point set cannot collapse.
# ---------------------------------------------------------------------------


def test_gallery_point_set_size_is_pinned() -> None:
    """Require at least six gallery points.

    Six points cover two latent-only, two data-only, and one joint
    perturbation in addition to the ground truth.
    """
    default = (
        inspect.signature(
            _gallery_data.points_from_dataset,
        )
        .parameters["n_points"]
        .default
    )
    assert default >= _MIN_GALLERY_POINTS, (
        f"`points_from_dataset` defaults to {default!r} points, below "
        f"the pinned minimum {_MIN_GALLERY_POINTS!r}. Every gallery "
        f"caller takes this default, so lowering it weakens every "
        f"equivalence cell at once."
    )

    labels = _gallery_data.perturbation_labels(default)
    assert labels[0] == _gallery_data.PERTURB_GROUND_TRUTH
    for mode in (
        _gallery_data.PERTURB_LATENTS,
        _gallery_data.PERTURB_DATA,
        _gallery_data.PERTURB_BOTH,
    ):
        assert labels.count(mode) >= 1, (
            f"the {default}-point schedule {labels!r} never runs the "
            f"{mode!r} perturbation, so that section of the point "
            f"space is untested."
        )
    assert labels.count(_gallery_data.PERTURB_LATENTS) >= 2, (
        f"the schedule {labels!r} carries fewer than two latents-only "
        f"points; one gives no cross-check on the prior-scoring path."
    )
    assert labels.count(_gallery_data.PERTURB_DATA) >= 2, (
        f"the schedule {labels!r} carries fewer than two data-only "
        f"points; one gives no cross-check on the dropped-data-term "
        f"path."
    )

    cells = _gallery_cells()
    assert cells, (
        "no gallery example is available to measure a real point set "
        "against; the corpus or its skip registries collapsed."
    )
    evaluated = _evaluate(cells[0])
    assert len(evaluated.points) >= _MIN_GALLERY_POINTS, (
        f"{evaluated.stem!r}: the realised point set holds "
        f"{len(evaluated.points)} points, below the pinned minimum "
        f"{_MIN_GALLERY_POINTS!r}."
    )


def test_point_set_defaults_are_pinned() -> None:
    """Check the reproducible seed, point count, scale, and seed sweep."""
    parameters = inspect.signature(
        _gallery_data.points_from_dataset,
    ).parameters
    assert parameters["seed"].default == _DEFAULT_GALLERY_SEED, (
        f"`points_from_dataset` defaults to seed "
        f"{parameters['seed'].default!r} rather than the pinned "
        f"{_DEFAULT_GALLERY_SEED!r}. Every gallery cell takes this "
        f"default, so a default that moves run to run turns a real "
        f"transpile regression into an intermittent one."
    )
    assert parameters["scale"].default == _gallery_data.PERTURBATION_SCALE, (
        f"`points_from_dataset` defaults to scale "
        f"{parameters['scale'].default!r} rather than "
        f"{_gallery_data.PERTURBATION_SCALE!r}. The excursion every "
        f"displacement is proportional to is not a knob: shortening it "
        f"weakens every equivalence cell at once, and a defect "
        f"nonlinear in the point is what goes quiet first."
    )

    seeds = _gallery_data.GALLERY_SEEDS
    assert len(seeds) >= _MIN_SWEPT_SEEDS, (
        f"`GALLERY_SEEDS` sweeps {len(seeds)} seed(s) {seeds!r}, below "
        f"the pinned minimum {_MIN_SWEPT_SEEDS!r}. Narrowing the sweep "
        f"narrows every per-cell claim in this module at once, and at "
        f"one seed the sensitivity claim stops being about the "
        f"perturbation design and becomes a statement about one draw."
    )
    assert len(set(seeds)) == len(seeds), (
        f"`GALLERY_SEEDS` repeats a seed ({seeds!r}). A repeated seed "
        f"yields a byte-identical point set, so the sweep would report "
        f"one draw as several and overstate its own coverage."
    )
    assert _DEFAULT_GALLERY_SEED in seeds, (
        f"`GALLERY_SEEDS` {seeds!r} omits the default seed "
        f"{_DEFAULT_GALLERY_SEED!r}, so the one draw every gallery "
        f"equivalence cell actually runs on is the one draw this "
        f"module never checks."
    )

    assert _gallery_data.WIDE_PERTURBATION_SCALE > _gallery_data.PERTURBATION_SCALE, (
        f"`WIDE_PERTURBATION_SCALE` "
        f"({_gallery_data.WIDE_PERTURBATION_SCALE!r}) does not exceed "
        f"the base scale ({_gallery_data.PERTURBATION_SCALE!r}), so "
        f"the excursion-magnitude measurement compares a point set "
        f"against itself."
    )


def test_single_point_comparison_is_vacuous() -> None:
    """Show that a one-point constant-spread comparison always passes.

    With one difference, subtracting its mean produces zero regardless of
    the two input values.
    """
    qvr, target = _diff_pair([1.0e6])
    constant = _equivalence.assert_log_density_match(
        qvr,
        target,
        context="vacuity-demonstration",
    )
    assert constant == pytest.approx(1.0e6), (
        "the one-point comparison should absorb the entire "
        "discrepancy into the constant, which is precisely why it "
        "cannot reject anything."
    )


def test_min_points_rejects_a_collapsed_point_set() -> None:
    """Reject a one-point comparison when `min_points=2`."""
    qvr, target = _diff_pair([1.0e6])
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr,
            target,
            context="collapsed",
            min_points=2,
        )
    message = str(exc_info.value)
    assert "at least 2" in message, (
        f"the rejection must name the required point count so the "
        f"failure is actionable; got {message!r}"
    )

    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            [],
            [],
            context="empty",
            min_points=2,
        )

    # Pin the gallery call site, not just the helper's capability: the
    # `min_points` guard only protects a caller that asks for it, and
    # the gallery equivalence cell is the caller whose point set the
    # whole multi-point design exists to keep honest. Reading the
    # module source rather than one function's keeps the pin alive
    # across a refactor that moves the call between functions.
    source = inspect.getsource(_gallery_equivalence)
    assert "min_points=2" in source, (
        "the gallery equivalence module no longer passes "
        "`min_points=2` to `assert_log_density_match`. Without it a "
        "collapsed point set passes unconditionally, which is exactly "
        "how this check decayed before."
    )


# ---------------------------------------------------------------------------
# 3. Coverage of the quantified dimensions.
# ---------------------------------------------------------------------------


def _point_signature(
    point: Point,
) -> tuple[
    tuple[tuple[str, tuple[float, ...]], ...],
    tuple[tuple[str, tuple[float, ...]], ...],
]:
    """The full identity of a point, both sections, for equality
    comparison against another point of the same set."""
    return (
        tuple(
            sorted(
                (name, _coordinate_key(value)) for name, value in point.params.items()
            )
        ),
        tuple(
            sorted((name, _coordinate_key(value)) for name, value in point.data.items())
        ),
    )


def test_seed_sweep_draws_independent_point_sets() -> None:
    """Reproduce equal seeds and distinguish different seeds.

    The comparison excludes point 0, which is the fixed ground truth in
    every set.
    """
    cells = _gallery_cells()
    assert cells, "no gallery example is available to draw a seed sweep from."
    example = cells[0]
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None
    sweep = _gallery_data.points_across_seeds(dataset)
    assert set(sweep) == set(_gallery_data.GALLERY_SEEDS)

    repeat = _gallery_data.points_from_dataset(
        dataset,
        seed=_DEFAULT_GALLERY_SEED,
    )
    assert [_point_signature(point) for point in repeat] == [
        _point_signature(point) for point in sweep[_DEFAULT_GALLERY_SEED]
    ], (
        f"{example.stem!r}: two draws at seed "
        f"{_DEFAULT_GALLERY_SEED!r} gave different point sets, so the "
        f"builder is reading a generator it does not own and no "
        f"failure in this suite is reproducible."
    )

    seeds = sorted(sweep)
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            first = sweep[seeds[left]]
            second = sweep[seeds[right]]
            assert _point_signature(first[0]) == _point_signature(second[0]), (
                f"{example.stem!r}: the ground-truth point differs "
                f"between seeds {seeds[left]} and {seeds[right]}. "
                f"Point 0 is the captured fixture, not a draw; a seed "
                f"that moves it has leaked into the data loader."
            )
            shared = [
                index
                for index in range(1, min(len(first), len(second)))
                if _point_signature(first[index]) == _point_signature(second[index])
            ]
            assert not shared, (
                f"{example.stem!r}: seeds {seeds[left]} and "
                f"{seeds[right]} produced identical perturbed points "
                f"at indices {shared!r}, so the sweep is reporting one "
                f"draw as two and the coverage it claims is smaller "
                f"than it looks."
            )

    with pytest.raises(ValueError):
        _gallery_data.points_across_seeds(dataset, seeds=())
    with pytest.raises(ValueError):
        _gallery_data.points_across_seeds(dataset, seeds=(0, 1, 0))


@pytest.mark.parametrize("example", _gallery_cells(), ids=_cell_id)
def test_seed_sweep_is_independent_for_every_cell(
    example: pathlib.Path,
) -> None:
    """Require distinct swept point sets for every gallery cell.

    Latent-only points for programs without latent sites may coincide by
    construction; other duplicate draws are rejected.
    """
    sets = {
        seed: _evaluate(example, seed=seed).points
        for seed in _gallery_data.GALLERY_SEEDS
    }
    latents = _latent_site_names(_evaluate(example).dataset)
    seeds = sorted(sets)
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            first = sets[seeds[left]]
            second = sets[seeds[right]]
            labels = _gallery_data.perturbation_labels(min(len(first), len(second)))
            for index in range(1, min(len(first), len(second))):
                if _point_signature(first[index]) != _point_signature(second[index]):
                    continue
                assert labels[index] == _gallery_data.PERTURB_LATENTS and not latents, (
                    f"{example.stem!r}: seeds {seeds[left]} and "
                    f"{seeds[right]} drew the identical point at index "
                    f"{index} ({labels[index]}), and the program "
                    f"declares {len(latents)} latent site(s), so this "
                    f"is a coordinate that stopped responding to the "
                    f"seed rather than a section with nothing to move. "
                    f"Every per-cell claim in this module is proved "
                    f"once per seed, so a draw that repeats is a claim "
                    f"reported more times than it was proved: the "
                    f"sweep still reads as {len(seeds)} independent "
                    f"draws while covering fewer. Widen the step the "
                    f"perturber takes for the coordinate that froze; "
                    f"do not drop the seed from the sweep."
                )


@pytest.mark.parametrize("example, seed", _seeded_cells(), ids=_cell_id)
def test_no_point_collapses_onto_the_ground_truth(
    example: pathlib.Path,
    seed: int,
) -> None:
    """Require each scheduled perturbation to differ from ground truth.

    A latent-only point may coincide when the program declares no latent
    site. This exemption is derived from the compiled program.
    """
    evaluated = _evaluate(example, seed=seed)
    points = evaluated.points
    labels = _gallery_data.perturbation_labels(len(points))
    ground_truth = _point_signature(points[0])
    latents = _latent_site_names(evaluated.dataset)

    for index in range(1, len(points)):
        if _point_signature(points[index]) != ground_truth:
            continue
        assert labels[index] == _gallery_data.PERTURB_LATENTS and not latents, (
            f"{example.stem!r} at seed {seed}: point {index} "
            f"({labels[index]}) is byte-identical to the ground-truth "
            f"point, so the {len(points)}-point schedule realises only "
            f"{len({_point_signature(p) for p in points})} distinct "
            f"points and the {labels[index]!r} mode is covered one "
            f"time fewer than the schedule claims. The program "
            f"declares {len(latents)} latent site(s), so this is a "
            f"perturbation that moved nothing rather than a section "
            f"with nothing to move: fix the step the perturber takes "
            f"for the coordinate that froze."
        )


@pytest.mark.parametrize("example, seed", _seeded_cells(), ids=_cell_id)
def test_every_quantified_coordinate_varies(
    example: pathlib.Path,
    seed: int,
) -> None:
    """Vary every latent and observed coordinate across each point set.

    Frozen structural subscripts require a matching entry in
    `_UNPERTURBABLE_COORDINATES`. The registry is checked for missing,
    stale, and misclassified entries at every seed.
    """
    evaluated = _evaluate(example, seed=seed)
    dataset = evaluated.dataset
    points = evaluated.points
    registry = _registered(example.stem)

    latents = _latent_site_names(dataset)
    observed = sorted(_gallery_data.observed_data_names(dataset))
    known = frozenset(latents) | frozenset(observed)
    unknown = sorted(name for name in registry if name not in known)
    assert not unknown, (
        f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` names "
        f"{unknown!r}, which is neither a latent site nor an observed "
        f"array of this example. Drop the stale entry; a registry of "
        f"names that no longer exist stops describing the gap it "
        f"claims to."
    )

    subscripts = _gallery_data.structural_subscript_names(dataset)
    misfiled = sorted(
        name
        for name, entry in registry.items()
        if entry.kind == "plate-subscript" and name not in subscripts
    )
    assert not misfiled, (
        f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` files "
        f"{misfiled!r} as a plate subscript, but "
        f"`structural_subscript_names` does not classify it as one. "
        f"A subscript indexes a plate, so it is an integer vector "
        f"with one entry per row; anything else the harness leaves "
        f"frozen is a coordinate of the support whose value enters "
        f"the density, and the exemption would hide a coverage gap "
        f"instead of recording a structural one. Close the gap in "
        f"the perturber."
    )

    for name in latents:
        present = sum(1 for point in points if name in point.params)
        assert present == len(points), (
            f"{example.stem!r} at seed {seed}: latent site {name!r} is "
            f"carried by "
            f"{present} of {len(points)} points. A site the point set "
            f"does not pin is a coordinate the equivalence check never "
            f"fixes, so the two evaluators are free to score it at "
            f"different values and the difference is not a function of "
            f"the point at all."
        )
        distinct = _distinct_values(points, name, latent=True)
        entry = registry.get(name)
        if entry is None:
            assert distinct >= 2, (
                f"{example.stem!r} at seed {seed}: latent site {name!r} takes "
                f"a single value across all {len(points)} points, so the "
                f"constant-spread check says nothing about how the "
                f"backend scores that coordinate. Give the site a "
                f"support the perturber can step in, or, if it truly "
                f"cannot move, record the argument in "
                f"`_UNPERTURBABLE_COORDINATES`."
            )
        else:
            assert distinct < 2, (
                f"{example.stem!r} at seed {seed}: "
                f"`_UNPERTURBABLE_COORDINATES` claims latent site "
                f"{name!r} cannot move "
                f"({entry.justification}), but it takes {distinct} "
                f"distinct values. Drop the entry: the coordinate is "
                f"covered and the claim is stale."
            )

    for name in observed:
        distinct = _distinct_values(points, name, latent=False)
        entry = registry.get(name)
        if entry is None:
            assert distinct >= 2, (
                f"{example.stem!r} at seed {seed}: observed array {name!r} is "
                f"byte-identical at every one of the {len(points)} "
                f"points, so the check is blind to any backend error "
                f"that depends on {name!r} alone: such an error is a "
                f"constant across the set and the mean subtraction "
                f"absorbs it. Give the array a support the perturber "
                f"can step in, or, if it is a structural index rather "
                f"than a coordinate of the support, record the "
                f"argument in `_UNPERTURBABLE_COORDINATES`."
            )
        else:
            assert distinct < 2, (
                f"{example.stem!r} at seed {seed}: "
                f"`_UNPERTURBABLE_COORDINATES` claims observed array "
                f"{name!r} cannot move "
                f"({entry.justification}), but it takes {distinct} "
                f"distinct values. Drop the entry: the coordinate is "
                f"covered and the claim is stale."
            )


@pytest.mark.parametrize("example, seed", _seeded_cells(), ids=_cell_id)
def test_point_set_exposes_a_planted_coordinate_defect(
    example: pathlib.Path,
    seed: int,
) -> None:
    """Reject a defect attached to each non-exempt coordinate.

    The planted term activates where that coordinate differs from ground
    truth. A constant offset alone remains accepted. The check repeats at
    every gallery seed.
    """
    evaluated = _evaluate(example, seed=seed)
    points = evaluated.points
    labels = _gallery_data.perturbation_labels(len(points))
    reference = list(evaluated.log_densities)
    registry = _registered(example.stem)
    constant = 4.25

    coordinates = [(name, True) for name in _latent_site_names(evaluated.dataset)] + [
        (name, False)
        for name in sorted(_gallery_data.observed_data_names(evaluated.dataset))
    ]
    planted_any = False
    for name, latent in coordinates:
        if name in registry:
            continue
        keys = [
            _coordinate_key((point.params if latent else point.data)[name])
            for point in points
        ]
        moved_at = [key != keys[0] for key in keys]
        planted = [
            value + constant + (_SMALLEST_SEMANTIC_DISCREPANCY if moved else 0.0)
            for value, moved in zip(reference, moved_at)
        ]
        planted_any = True
        with pytest.raises(AssertionError) as exc_info:
            _equivalence.assert_log_density_match(
                reference,
                planted,
                context=f"planted@{example.stem}:s{seed}:{name}",
                labels=labels,
                min_points=2,
            )
        assert "spread" in str(exc_info.value), (
            f"{example.stem!r} at seed {seed}: the rejection of the "
            f"defect planted on {name!r} does not report the spread "
            f"statistic, so a real failure would not be diagnosable; "
            f"got {str(exc_info.value)!r}"
        )
        # The same sequence without the defect must pass, so the
        # rejection above is attributable to the planted term and not
        # to the additive constant the contract allows.
        assert _equivalence.assert_log_density_match(
            reference,
            [value + constant for value in reference],
            context=f"constant-only@{example.stem}:s{seed}:{name}",
            labels=labels,
            min_points=2,
        ) == pytest.approx(constant, abs=1e-9)

    assert planted_any, (
        f"{example.stem!r} at seed {seed}: every coordinate of this "
        f"example is registered as unperturbable, so no defect can be "
        f"planted anywhere and the equivalence cell for it tests "
        f"nothing."
    )


def test_unperturbable_registry_is_well_formed() -> None:
    """Require every frozen-coordinate exemption to name a live example and reason."""
    stems = {example.stem for example in _gallery_data.gallery_examples_with_data()}
    seen: set[tuple[str, str]] = set()
    for entry in _UNPERTURBABLE_COORDINATES:
        assert entry.example in stems, (
            f"`_UNPERTURBABLE_COORDINATES` names example "
            f"{entry.example!r}, which is not a gallery example with "
            f"synthetic data. Drop the entry."
        )
        key = (entry.example, entry.coordinate)
        assert key not in seen, (
            f"duplicate registry entry for {key!r}; two arguments for "
            f"one exemption means one of them is unreviewed."
        )
        seen.add(key)
        assert len(entry.justification) >= _MIN_JUSTIFICATION_CHARS, (
            f"the exemption for {key!r} carries a "
            f"{len(entry.justification)}-character justification, "
            f"under the {_MIN_JUSTIFICATION_CHARS}-character floor. "
            f"State why the coordinate is outside the (theta, y) "
            f"support Theorem 4.1 quantifies over."
        )


# ---------------------------------------------------------------------------
# 4. Every point is in the support, and the reference measure moves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example, seed", _seeded_cells(), ids=_cell_id)
def test_reference_joint_is_in_support_and_varies(
    example: pathlib.Path,
    seed: int,
) -> None:
    """Require finite and varying QVR joints at every swept seed.

    The joint range must exceed the equivalence tolerance by the recorded
    margin.
    """
    evaluated = _evaluate(example, seed=seed)
    labels = _gallery_data.perturbation_labels(len(evaluated.points))
    log_densities = list(evaluated.log_densities)

    for index, lp in enumerate(log_densities):
        assert math.isfinite(lp), (
            f"{example.stem!r} at seed {seed}: point {index} "
            f"({labels[index]}) scores a non-finite reference joint "
            f"({lp!r}), so it left the model's support and cannot "
            f"witness the equivalence at any tolerance."
        )

    span = max(log_densities) - min(log_densities)
    required = _MIN_JOINT_VARIATION_IN_ATOL * _PINNED_DEFAULT_ATOL
    assert span >= required, (
        f"{example.stem!r} at seed {seed}: the reference joint spans "
        f"only {span:.6e} nats across {len(log_densities)} points, under "
        f"the {required:.6e} nats this check requires "
        f"({_MIN_JOINT_VARIATION_IN_ATOL!r} times the "
        f"{_PINNED_DEFAULT_ATOL!r} tolerance). A point set that moves "
        f"the measure by less than the tolerance cannot reject a "
        f"wrong backend. Per-point values: "
        f"{[round(lp, 6) for lp in log_densities]!r}"
    )


# ---------------------------------------------------------------------------
# 5. The excursion magnitude is a coverage parameter, not a cosmetic one.
# ---------------------------------------------------------------------------


def _quadratic_defect(points: Sequence[Point]) -> list[float]:
    """The per-point discrepancy of a defect whose error grows with
    distance from the ground truth.

    Every point of the set is a displacement `r` away from point 0 in
    the flat wire coordinate, and this returns `r**2` at each point.
    Squared distance is the leading term of any discrepancy that
    vanishes at the ground truth and is smooth around it, which is the
    generic shape of a defect the harness would otherwise miss: a
    backend that expands a density around the ground-truth parameters
    and drops the second-order term, a link function inverted to first
    order, a variance whose reparameterisation agrees to first order
    with the reference. Each of those is exactly right at the point the
    fixture was generated at and wrong in proportion to how far the
    evaluation strays.

    That is why the excursion magnitude is a coverage parameter. A
    discrepancy linear in `r` contributes a spread proportional to the
    scale, so widening the excursion buys detection in proportion; a
    discrepancy quadratic in `r` contributes a spread proportional to
    the square, so it can sit under tolerance at one excursion and be
    rejected outright at a wider one. The first kind is a matter of
    degree, the second is a matter of what the check can see at all.
    """
    return [
        displacement**2 for displacement in _gallery_data.point_displacements(points)
    ]


def _linear_defect(points: Sequence[Point]) -> list[float]:
    """The per-point discrepancy of a defect whose error grows in
    direct proportion to distance from the ground truth.

    The control against which the quadratic defect is measured. A
    discrepancy linear in the displacement is the shape a mis-scaled
    coefficient produces: wrong everywhere except at the ground truth,
    and wrong in proportion to how far the evaluation strays. Widening
    the excursion buys detection against it too, which is exactly why
    it is the control. Showing that a wider point set rejects a
    *quadratic* defect alone does not establish that the excursion
    being a coverage parameter, since a longer excursion makes every
    defect that vanishes at the ground truth louder. The claim only
    has content if the wider set gains *more* against the nonlinear
    shape than against the linear one, and that comparison needs both.
    """
    return list(_gallery_data.point_displacements(points))


def _detection_threshold(
    profile: Sequence[float],
    atol: float,
) -> float:
    """The smallest coefficient at which a defect of shape `profile` is
        rejected on the point set that produced it.

        A defect contributing `c * profile[i]` at point `i` has spread
        `c * spread(profile)`, since the spread statistic is positively
        homogeneous, and the assertion rejects once that exceeds `atol`.
    The threshold is thus `atol / spread(profile)` exactly, with
        no search required, and it is the quantity that answers what a
        point set can *detect* rather than how loudly it complains about
        one defect that was planted.

        Returns infinity for a profile that is constant across the set:
        such a defect is absorbed into the additive constant Theorem 4.1
        permits, at any coefficient whatever, so no magnitude of it is
        detectable.
    """
    reach = _spread(profile)
    return atol / reach if reach > 0.0 else math.inf


def _wide_excursion_gain(
    example: pathlib.Path,
    seed: int,
) -> tuple[_EvaluatedPointSet, _EvaluatedPointSet, float]:
    """The default and wide point sets of one cell, and the ratio of
    the distances they actually travelled.

    The ratio is read off the realised points rather than off the two
    `scale` arguments, because the requested scale is not what a point
    set achieves. Two mechanisms drive them apart, and both are
    intended: the redraw ladder halves the scale whenever a draw leaves
    the support, and an integer coordinate steps inside its attested
    range whatever the scale asks for, so a cell whose coordinates are
    all integer-clamped travels the same distance at any scale.
    """
    default = _evaluate(example, seed=seed)
    wide = _evaluate(
        example,
        seed=seed,
        scale=_gallery_data.WIDE_PERTURBATION_SCALE,
    )
    reach = _gallery_data.point_excursion(default.points)
    assert reach > 0.0, (
        f"{example.stem!r} at seed {seed}: the default point set never "
        f"leaves the ground truth, so it has no excursion to widen and "
        f"its spread statistic is identically zero."
    )
    return default, wide, (_gallery_data.point_excursion(wide.points) / reach)


@pytest.mark.parametrize("example, seed", _seeded_cells(), ids=_cell_id)
def test_wider_excursion_detects_a_nonlinear_defect(
    example: pathlib.Path,
    seed: int,
) -> None:
    """A wider excursion changes *what the check can detect*, not
    merely how loudly it detects it.

    The six points the gallery evaluates move each coordinate a fixed
    fraction of its own scale. That is enough for a defect whose
    per-point discrepancy is constant on a region, which is the shape
    every historical bug in this repository had. It is not enough for a
    defect that vanishes at the ground truth and grows with distance
    from it, because such a defect contributes a spread proportional to
    the *square* of the excursion: shrink the excursion and it goes
    quiet quadratically.

    This test exhibits one. It plants
    [`_quadratic_defect`][tests.transpile.test_validation_strength._quadratic_defect]
    on the default point set, scaled so its spread is
    `_NONLINEAR_DEFECT_HEADROOM` of the tolerance, and asserts two
    things about the same defect:

    1. The default point set **accepts** it. This is the gap, stated
       as a measurement rather than as a worry: a real defect of this
       shape and this magnitude passes the gallery equivalence cell
       today.
    2. The wide point set **rejects** it, with the constant-only
       sequence still accepted so the rejection is attributable to the
       planted term.

    The second claim is conditional on the wide set having actually
    travelled `_WIDE_EXCURSION_GAIN` further, and the condition is
    measured from the realised points rather than assumed from the
    scale argument. A cell whose coordinates are integer steps clamped
    to their attested range does not travel further at a wider scale,
    however large the scale is, so for such a cell the excursion is not
    the lever and the claim is not made. That is not an exemption to be
    granted by hand: the ratio is recomputed every run, and
    `test_wide_excursion_gain_covers_the_corpus` fails if the
    conditional ever stops firing across the corpus.
    """
    default, wide, gain = _wide_excursion_gain(example, seed)
    constant = 4.25
    atol = _PINNED_DEFAULT_ATOL

    default_profile = _quadratic_defect(default.points)
    default_spread = _spread(default_profile)
    assert default_spread > 0.0, (
        f"{example.stem!r} at seed {seed}: the squared displacement is "
        f"the same at every point, so no defect that grows with "
        f"distance from the ground truth can be planted here at all."
    )
    # Calibrate the defect to sit just inside the accept region on the
    # default set. Its magnitude is therefore derived from that set's
    # own geometry, not chosen to make either half of the test come
    # out: the accept below is what fixes it, and the reject above is
    # what the wider excursion then has to earn.
    magnitude = _NONLINEAR_DEFECT_HEADROOM * atol / default_spread

    reference = list(default.log_densities)
    planted = [
        value + constant + magnitude * profile
        for value, profile in zip(reference, default_profile)
    ]
    assert _equivalence.assert_log_density_match(
        reference,
        planted,
        context=f"nonlinear-invisible@{example.stem}:s{seed}",
        labels=_gallery_data.perturbation_labels(len(default.points)),
        min_points=2,
    ) == pytest.approx(constant, abs=atol), (
        f"{example.stem!r} at seed {seed}: the quadratic defect "
        f"calibrated to {_NONLINEAR_DEFECT_HEADROOM!r} of the "
        f"{atol!r} tolerance was rejected on the default point set, so "
        f"the calibration is wrong rather than the check being blind. "
        f"Squared-displacement spread: {default_spread:.6e}."
    )

    if gain < _WIDE_EXCURSION_GAIN:
        return

    wide_profile = _quadratic_defect(wide.points)
    wide_reference = list(wide.log_densities)
    wide_planted = [
        value + constant + magnitude * profile
        for value, profile in zip(wide_reference, wide_profile)
    ]
    wide_labels = _gallery_data.perturbation_labels(len(wide.points))
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            wide_reference,
            wide_planted,
            context=f"nonlinear-detected@{example.stem}:s{seed}",
            labels=wide_labels,
            min_points=2,
        )
    assert "spread" in str(exc_info.value), (
        f"{example.stem!r} at seed {seed}: the wide point set rejected "
        f"the quadratic defect without reporting the spread "
        f"statistic; got {str(exc_info.value)!r}"
    )
    # The rejection has to be attributable to the planted term. The
    # wide set's own reference joints are far larger in magnitude than
    # the default set's, so a statistic that grew with the values being
    # differenced would reject the constant-only sequence too, and the
    # demonstration would be about magnitude rather than about the
    # defect.
    assert _equivalence.assert_log_density_match(
        wide_reference,
        [value + constant for value in wide_reference],
        context=f"nonlinear-constant-only@{example.stem}:s{seed}",
        labels=wide_labels,
        min_points=2,
    ) == pytest.approx(constant, abs=1e-9)

    # The rejection above says the wider set catches this one defect.
    # On its own that is unremarkable: a longer excursion makes *every*
    # discrepancy that vanishes at the ground truth louder, the linear
    # ones included, so a set that travels further would reject a
    # calibrated defect of any such shape. The excursion is a coverage
    # parameter rather than a volume knob only if the nonlinear shape
    # gains strictly more from the widening than the linear control
    # does, and that is a ratio of detection thresholds rather than a
    # statement about one planted magnitude.
    #
    # The floor is derived, not fitted. Were every displacement scaled
    # by exactly `g`, the linear profile would scale by `g` and the
    # quadratic one by `g**2`, so the quadratic threshold would improve
    # by `g` times as much as the linear one: the advantage is the
    # excursion gain itself, and it exceeds 1 whenever the set travels
    # further at all. Real sets are mixtures, since an integer
    # coordinate steps inside its attested window whatever the scale
    # asks, which pulls the achieved advantage below `g` without
    # touching the direction of the inequality.
    quadratic_advantage = _detection_threshold(
        default_profile, atol
    ) / _detection_threshold(wide_profile, atol)
    linear_advantage = _detection_threshold(
        _linear_defect(default.points), atol
    ) / _detection_threshold(_linear_defect(wide.points), atol)
    assert quadratic_advantage > linear_advantage, (
        f"{example.stem!r} at seed {seed}: widening the excursion by "
        f"{gain:.3f}x improved the detection threshold for a quadratic "
        f"defect by {quadratic_advantage:.3f}x and for a linear one by "
        f"{linear_advantage:.3f}x, so the wider set is not "
        f"preferentially better against the nonlinear shape. Under "
        f"uniform scaling the quadratic advantage is the linear one "
        f"times the excursion gain, so an advantage that failed to "
        f"exceed it means the set did not really travel further in the "
        f"coordinates the defect grows in: the excursion is being "
        f"spent on coordinates clamped to a fixed window, and for this "
        f"cell the scale is not the lever this test claims it is."
    )


def test_wide_excursion_gain_covers_the_corpus() -> None:
    """The nonlinear-defect demonstration is made on a real share of
    the corpus, and the wider scale never yields a narrower set.

    `test_wider_excursion_detects_a_nonlinear_defect` is a conditional,
    and a conditional is the quietest way for a measurement to stop
    measuring: if the antecedent ever held nowhere, every one of its
    parametrisations would still pass while demonstrating nothing.
    This test is the guard, and it makes two claims.

    First, monotonicity. A larger requested scale must never produce a
    point set that travels less than the smaller one does. The redraw
    ladder is the mechanism that could break this: a wide draw that
    leaves the support is halved, and halved again, so a scale asked to
    reach further can come back having reached less. That would make
    the wide set strictly worse than the default one, and the gallery's
    own excursion would be sitting at the edge of the support rather
    than inside it.

    Second, coverage. At least `_MIN_WIDE_GAIN_FRACTION` of the swept
    (example, seed) pairs must clear `_WIDE_EXCURSION_GAIN`, so the
    conditional demonstrably fires. The pairs that do not clear it are
    the ones whose excursion is dominated by integer coordinates
    stepping inside an attested range, where a wider scale reshuffles
    which admissible integer each draw lands on rather than reaching
    further; those cells are covered by the constant-magnitude planted
    defect instead, which is the shape their discrete coordinates
    admit.
    """
    pairs = _seeded_cells()
    assert pairs, (
        "no (example, seed) pair is available, so the excursion "
        "measurement has no corpus to run on."
    )
    gains: list[tuple[float, str, int]] = []
    for example, seed in pairs:
        _, _, gain = _wide_excursion_gain(example, seed)
        gains.append((gain, example.stem, seed))

    narrowed = [entry for entry in gains if entry[0] < 1.0]
    assert not narrowed, (
        f"the wide excursion travelled *less* far than the default one "
        f"for {[(round(g, 3), stem, seed) for g, stem, seed in narrowed]!r}. "
        f"A larger scale that reaches less means the redraw ladder "
        f"halved that draw more than once, so the wide excursion is "
        f"sitting outside the support this module derives for some "
        f"coordinate of the cell. Fix the support the perturber "
        f"derives for the coordinate that left it; do not lower "
        f"`WIDE_PERTURBATION_SCALE` to make this pass."
    )

    cleared = [entry for entry in gains if entry[0] >= _WIDE_EXCURSION_GAIN]
    fraction = len(cleared) / len(gains)
    assert fraction >= _MIN_WIDE_GAIN_FRACTION, (
        f"only {len(cleared)} of {len(gains)} (example, seed) pairs "
        f"reach {_WIDE_EXCURSION_GAIN!r} times the default excursion, "
        f"a fraction of {fraction:.3f} under the required "
        f"{_MIN_WIDE_GAIN_FRACTION!r}. "
        f"`test_wider_excursion_detects_a_nonlinear_defect` is "
        f"conditional on that ratio, so at this coverage it is close "
        f"to demonstrating nothing. Widest gains: "
        f"{sorted(gains, reverse=True)[:3]!r}."
    )


# ---------------------------------------------------------------------------
# 6. The spread statistic itself behaves.
# ---------------------------------------------------------------------------


def test_spread_accepts_a_genuinely_constant_offset() -> None:
    """A constant additive offset is accepted, and reported as `c`.

    This is the permissive half of Theorem 4.1 and it is not
    negotiable: a backend may legitimately differ from the reference
    by a fixed constant (a Jacobian term from a different
    parameterisation, a normaliser one framing drops), and the
    equivalence relation quotients that out. A check that rejected a
    constant offset would fail correct transpilations, which is the
    mirror-image failure of the vacuity this module guards against.
    """
    offset = 12.3456789
    qvr, target = _diff_pair([offset] * 6)
    constant = _equivalence.assert_log_density_match(
        qvr,
        target,
        context="constant-offset",
        min_points=2,
    )
    assert constant == pytest.approx(offset, abs=1e-12)

    # The same offset must still be accepted when the reference joints
    # are far apart: the statistic is a deviation from the mean, so it
    # may not grow with the magnitude of the values being differenced.
    wide_qvr = [-1.0e4, -1.0, 5.0e3, 12.5, -7.5e3, 0.25]
    wide_target = [value + offset for value in wide_qvr]
    assert _equivalence.assert_log_density_match(
        wide_qvr,
        wide_target,
        context="constant-offset-wide",
        min_points=2,
    ) == pytest.approx(offset, abs=1e-9)


def test_spread_rejects_a_non_constant_offset() -> None:
    """A difference that drifts by the smallest real-bug magnitude is
    rejected.

    The documented detection threshold is 1e-2 nats per point: a
    swapped distribution argument on the gallery's parameter ranges
    moves the log-density by at least that much. A drift of that size
    across a six-point set produces a spread of roughly 2.5e-2, fifty
    times the 5e-4 tolerance, so the check has to reject it with
    plenty of margin. If this test ever passes, the tolerance has
    grown past the point where a real bug is distinguishable from
    round-off.
    """
    drift = _SMALLEST_SEMANTIC_DISCREPANCY
    diffs = [7.0 + index * drift for index in range(6)]
    qvr, target = _diff_pair(diffs)
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr,
            target,
            context="drifting-offset",
            min_points=2,
        )
    message = str(exc_info.value)
    assert "spread" in message, (
        f"the rejection must report the spread statistic so the "
        f"failure is diagnosable; got {message!r}"
    )

    mean = sum(diffs) / len(diffs)
    spread = max(abs(d - mean) for d in diffs)
    assert spread >= 10.0 * _PINNED_DEFAULT_ATOL, (
        f"a {drift!r}-per-point drift produces a spread of "
        f"{spread:.6e}, less than ten times the {_PINNED_DEFAULT_ATOL!r} "
        f"tolerance. The margin between the cheapest real bug and the "
        f"tolerance has collapsed."
    )


def test_spread_rejects_a_single_deviating_point() -> None:
    """A difference that is constant on every point but one is
    rejected.

    This is the shape a partially-correct backend produces: it agrees
    with the reference wherever the perturbation did not reach the
    broken term, and disagrees at the one point that moved the
    coordinate the term depends on. Because the mean absorbs `1/n` of
    a lone deviation, the surviving spread is `(n-1)/n` of it, so a
    1e-2 outlier over six points still leaves 8.3e-3, more than an
    order of magnitude above tolerance. A check that smoothed a lone
    outlier away would pass exactly the backends whose error is
    confined to one section of the point space.
    """
    outlier = _SMALLEST_SEMANTIC_DISCREPANCY
    for outlier_index in (0, 3, 5):
        diffs = [-2.5] * 6
        diffs[outlier_index] += outlier
        qvr, target = _diff_pair(diffs)
        with pytest.raises(AssertionError) as exc_info:
            _equivalence.assert_log_density_match(
                qvr,
                target,
                context=f"outlier-{outlier_index}",
                labels=_gallery_data.perturbation_labels(6),
                min_points=2,
            )
        message = str(exc_info.value)
        assert f"index {outlier_index}" in message, (
            f"the rejection must localise the worst point so the "
            f"failure names the perturbation that broke constancy; "
            f"got {message!r}"
        )


def test_spread_boundary_matches_the_pinned_tolerance() -> None:
    """The accept / reject boundary sits at the pinned tolerance.

    The statistic is `max_i |d_i - mean(d)|` compared against `atol`,
    so a sequence engineered to a known spread must be accepted just
    below the floor and rejected just above it. Pinning both sides
    keeps a change to the comparison itself, a `>=` turned into a `>`,
    a spread computed before the mean subtraction, from passing as a
    refactor.
    """
    n = 6
    for factor, should_pass in ((0.5, True), (0.9, True), (2.0, False)):
        # A lone deviation of `delta` over `n` points leaves a spread
        # of `delta * (n - 1) / n` after the mean subtraction.
        target_spread = factor * _PINNED_DEFAULT_ATOL
        delta = target_spread * n / (n - 1)
        diffs = [4.0] * n
        diffs[2] += delta
        qvr, target = _diff_pair(diffs)
        if should_pass:
            _equivalence.assert_log_density_match(
                qvr,
                target,
                context=f"boundary-{factor}",
                min_points=2,
            )
        else:
            with pytest.raises(AssertionError):
                _equivalence.assert_log_density_match(
                    qvr,
                    target,
                    context=f"boundary-{factor}",
                    min_points=2,
                )


def test_spread_rejects_degenerate_sequences() -> None:
    """Non-finite differences and mismatched lengths are errors, not
    passes.

    Two `-inf` log-densities differ by `nan`, and `nan` compares false
    against any tolerance, so a naive spread check would report
    agreement for a pair of points that both left the support. A
    length mismatch is the other degenerate shape: it means the two
    evaluators were scored on different point sets, and pairing them
    by position compares unrelated points.
    """
    qvr, target = _diff_pair([1.0] * 4)
    target[2] = math.inf
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr,
            target,
            context="non-finite",
            min_points=2,
        )
    assert "non-finite" in str(exc_info.value)

    both_infinite = [-math.inf, -math.inf, -math.inf]
    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            both_infinite,
            list(both_infinite),
            context="both-outside-support",
            min_points=2,
        )

    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
            context="length",
            min_points=2,
        )
    assert "length mismatch" in str(exc_info.value)
