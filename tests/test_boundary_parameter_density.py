"""A family parameter at a boundary its family admits is scored there.

A Poisson ``rate`` of zero is the point mass at zero, and a Geometric
``probs`` of one is the point mass at zero failures: a positive count has
log probability ``-inf`` under either. The compiled runtime's inline
clamp holds every other parameter strictly inside its boundary, so it
must leave these two in place. Both execution paths are checked against
`torch.distributions` and a float64 closed form, at the boundary, on a
zero-inflated Poisson whose structural branch gates the rate to zero,
and at interior points, where the densities must not move.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.continuous.families import _make_family
from quivers.continuous.family_spec import get_family_spec
from quivers.continuous.param_transforms import closed_interval_clamp
from quivers.core._util import EPS
from quivers.dsl import loads

#: A float32 joint of a few rows against its float64 closed form.
_ATOL = 1e-4

_POISSON = """\
object Resp : FinSet 2
object Val : Real 1

program p : Resp -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let gated = scale * exp(a)
    observe y : Resp <- Poisson(rate=gated)
    return a

export p
"""

_GEOMETRIC = """\
object Resp : FinSet 2
object Val : Real 1

program g : Resp -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let p = 1.0 - shrink * sigmoid(a)
    observe y : Resp <- Geometric(probs=p)
    return a

export g
"""

_ZIP = """\
object Resp : FinSet 3
object Val : Real 1

program zip : Resp -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let pi_z = sigmoid(a + x)
    let rate = exp(0.5 * x - 2.0)
    marginalize z : Resp <- Bernoulli(probs=pi_z)
        let gated_rate = real(z) * rate
        observe y : Resp <- Poisson(rate=gated_rate)
    return a

export zip
"""

#: The middle row's count of one is unlikely under its counting branch
#: (a small gate and a small rate), so a structural branch that leaks
#: mass onto positive counts shows in the joint.
_ZIP_X = [-1.0, -10.0, 2.0]
_ZIP_Y = [0, 1, 3]


def _f64(value: float | list[float]) -> torch.Tensor:
    """A host value in double precision.

    Parameters
    ----------
    value : float | list[float]
        The value.

    Returns
    -------
    torch.Tensor
        A float64 tensor.
    """
    return torch.tensor(value, dtype=torch.float64)


def _compiled(source: str, values: dict[str, torch.Tensor]) -> float:
    """The compiled runtime's one-run log joint.

    Parameters
    ----------
    source : str
        The QVR source.
    values : dict[str, torch.Tensor]
        Data and latents.

    Returns
    -------
    float
        The log joint.
    """
    return float(loads(source).morphism.log_joint(torch.zeros(1, 1), values)[0])


def _reference(
    source: str, data: dict[str, float | list[float]], sites: dict[str, float]
) -> float:
    """The reference machine's log joint.

    Parameters
    ----------
    source : str
        The QVR source.
    data : dict[str, float | list[float] | list[int]]
        The host data, observed site included; counts as integers.
    sites : dict[str, float]
        The latents.

    Returns
    -------
    float
        The log joint.
    """
    program = loads(source)
    entry = program.entry_points()[0]
    run = program.run(
        entry.name,
        data={
            # The observed counts `y` are integers to the checked machine.
            name: tuple(int(v) if name == "y" else v for v in value)
            if isinstance(value, list)
            else value
            for name, value in data.items()
        },
        sites=sites,
    )
    return float(run.log_joint)


def _poisson_joint(scale: float, a: float, y: list[float]) -> float:
    """The Poisson program's joint in float64.

    Parameters
    ----------
    scale : float
        The rate's multiplier.
    a : float
        The latent.
    y : list[float]
        The counts.

    Returns
    -------
    float
        The prior plus the counts' log masses.
    """
    rate = scale * math.exp(a)
    prior = D.Normal(_f64(0.0), _f64(1.0)).log_prob(_f64(a))
    return float(prior + D.Poisson(_f64(rate)).log_prob(_f64(y)).sum())


def _geometric_joint(shrink: float, a: float, y: list[float]) -> float:
    """The Geometric program's joint in float64.

    Parameters
    ----------
    shrink : float
        How far the success probability sits below one.
    a : float
        The latent.
    y : list[float]
        The failure counts.

    Returns
    -------
    float
        The prior plus the counts' log masses.
    """
    p = 1.0 - shrink / (1.0 + math.exp(-a))
    prior = D.Normal(_f64(0.0), _f64(1.0)).log_prob(_f64(a))
    return float(prior + D.Geometric(probs=_f64(p)).log_prob(_f64(y)).sum())


def _zip_joint(a: float, *, structural_rate: float) -> float:
    """The zero-inflated Poisson program's joint in float64.

    Parameters
    ----------
    a : float
        The latent.
    structural_rate : float
        The rate the structural-zero branch scores its counts under:
        zero for the model, a positive floor for the clamped variant.

    Returns
    -------
    float
        The prior plus each row's two-branch marginal.
    """
    x, y = _f64(_ZIP_X), _f64([float(count) for count in _ZIP_Y])
    pi = torch.sigmoid(a + x)
    rate = torch.exp(0.5 * x - 2.0)
    structural = D.Poisson(torch.full_like(x, structural_rate)).log_prob(y)
    counting = D.Poisson(rate).log_prob(y)
    rows = torch.logaddexp(torch.log1p(-pi) + structural, torch.log(pi) + counting)
    return float(D.Normal(_f64(0.0), _f64(1.0)).log_prob(_f64(a)) + rows.sum())


# -- the reproducer ---------------------------------------------------------


def test_a_positive_count_under_a_zero_rate_scores_minus_infinity() -> None:
    values = {
        "a": torch.tensor([0.0]),
        "y": torch.tensor([1.0, 0.0]),
        "scale": torch.tensor([0.0, 0.0]),
    }
    assert _compiled(_POISSON, values) == -math.inf
    assert _reference(_POISSON, {"y": [1.0, 0.0], "scale": [0.0, 0.0]}, {"a": 0.0}) == (
        -math.inf
    )


@pytest.mark.parametrize("a", [-0.7, 0.0, 1.3])
def test_zero_counts_under_a_zero_rate_score_the_prior_alone(a: float) -> None:
    expected = _poisson_joint(0.0, a, [0.0, 0.0])
    assert expected == pytest.approx(-0.5 * math.log(2 * math.pi) - 0.5 * a * a)
    values = {
        "a": torch.tensor([a]),
        "y": torch.tensor([0.0, 0.0]),
        "scale": torch.tensor([0.0, 0.0]),
    }
    assert _compiled(_POISSON, values) == pytest.approx(expected, abs=_ATOL)
    reference = _reference(_POISSON, {"y": [0.0, 0.0], "scale": [0.0, 0.0]}, {"a": a})
    assert reference == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("scale", "a", "y"),
    [
        (1e-9, 0.0, [0.0, 2.0]),
        (0.3, -1.0, [1.0, 0.0]),
        (2.5, 0.4, [3.0, 5.0]),
        (40.0, 1.1, [91.0, 130.0]),
    ],
)
def test_an_interior_rate_scores_the_poisson_density(
    scale: float, a: float, y: list[float]
) -> None:
    values = {
        "a": torch.tensor([a]),
        "y": torch.tensor(y),
        "scale": torch.tensor([scale, scale]),
    }
    expected = _poisson_joint(scale, a, y)
    assert _compiled(_POISSON, values) == pytest.approx(expected, rel=1e-5)
    reference = _reference(_POISSON, {"y": y, "scale": [scale, scale]}, {"a": a})
    assert reference == pytest.approx(expected, rel=1e-9)


def test_a_rate_below_the_old_floor_is_scored_where_it_stands() -> None:
    # A rate of 1e-9 is an ordinary member of the family: a floor at 1e-7
    # would score its count of two about 9.2 nats higher.
    floored = _poisson_joint(1e-7, 0.0, [0.0, 2.0])
    exact = _poisson_joint(1e-9, 0.0, [0.0, 2.0])
    assert floored - exact == pytest.approx(2 * math.log(100.0), abs=1e-6)


# -- the zero-inflated Poisson ----------------------------------------------


@pytest.mark.parametrize("a", [-2.0, 0.0, 1.5])
def test_the_structural_zero_branch_puts_no_mass_on_a_positive_count(
    a: float,
) -> None:
    expected = _zip_joint(a, structural_rate=0.0)
    values = {
        "a": torch.tensor([a]),
        "x": torch.tensor(_ZIP_X),
        "y": torch.tensor(_ZIP_Y, dtype=torch.float32),
    }
    assert _compiled(_ZIP, values) == pytest.approx(expected, abs=_ATOL)
    reference = _reference(_ZIP, {"x": _ZIP_X, "y": _ZIP_Y}, {"a": a})
    assert reference == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("a", [-2.0, 0.0, 1.5])
def test_the_zip_oracle_rejects_a_floored_structural_rate(a: float) -> None:
    exact = _zip_joint(a, structural_rate=0.0)
    floored = _zip_joint(a, structural_rate=EPS)
    assert floored - exact > 100 * _ATOL


# -- the Geometric ------------------------------------------------------------


def test_a_failure_under_a_certain_success_scores_minus_infinity() -> None:
    values = {
        "a": torch.tensor([0.0]),
        "y": torch.tensor([0.0, 2.0]),
        "shrink": torch.tensor([0.0, 0.0]),
    }
    assert _compiled(_GEOMETRIC, values) == -math.inf
    reference = _reference(
        _GEOMETRIC, {"y": [0.0, 2.0], "shrink": [0.0, 0.0]}, {"a": 0.0}
    )
    assert reference == -math.inf


def test_no_failure_under_a_certain_success_scores_the_prior_alone() -> None:
    values = {
        "a": torch.tensor([0.4]),
        "y": torch.tensor([0.0, 0.0]),
        "shrink": torch.tensor([0.0, 0.0]),
    }
    expected = _geometric_joint(0.0, 0.4, [0.0, 0.0])
    assert _compiled(_GEOMETRIC, values) == pytest.approx(expected, abs=_ATOL)
    reference = _reference(
        _GEOMETRIC, {"y": [0.0, 0.0], "shrink": [0.0, 0.0]}, {"a": 0.4}
    )
    assert reference == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("shrink", "a", "y"),
    [(0.5, 0.0, [0.0, 3.0]), (0.9, 1.2, [4.0, 1.0]), (0.2, -0.8, [0.0, 0.0])],
)
def test_an_interior_success_probability_scores_the_geometric_density(
    shrink: float, a: float, y: list[float]
) -> None:
    values = {
        "a": torch.tensor([a]),
        "y": torch.tensor(y),
        "shrink": torch.tensor([shrink, shrink]),
    }
    expected = _geometric_joint(shrink, a, y)
    assert _compiled(_GEOMETRIC, values) == pytest.approx(expected, rel=1e-5)
    reference = _reference(_GEOMETRIC, {"y": y, "shrink": [shrink, shrink]}, {"a": a})
    assert reference == pytest.approx(expected, rel=1e-9)


# -- the clamps themselves ----------------------------------------------------


def test_the_poisson_rate_clamp_projects_onto_the_nonnegative_reals() -> None:
    spec = get_family_spec("Poisson")
    assert spec is not None
    (rate,) = spec.params
    moved = rate.inline_clamp(_f64([-1.0, 0.0, 1e-9, 3.0]))
    assert moved.tolist() == pytest.approx([0.0, 0.0, 1e-9, 3.0], abs=0.0)


def test_the_geometric_probs_clamp_admits_one_and_not_zero() -> None:
    spec = get_family_spec("Geometric")
    assert spec is not None
    (probs,) = spec.params
    moved = probs.inline_clamp(_f64([-0.5, 0.0, 0.25, 1.0, 1.5]))
    assert moved.tolist() == pytest.approx([EPS, EPS, 0.25, 1.0, 1.0], abs=0.0)


@pytest.mark.parametrize("family", ["Normal", "Exponential", "Gamma", "Bernoulli"])
def test_a_family_with_an_open_boundary_keeps_its_floor(family: str) -> None:
    spec = get_family_spec(family)
    assert spec is not None
    for param in spec.params:
        if param.transform_name in ("softplus", "softplus_shifted", "exp"):
            assert float(param.inline_clamp(torch.tensor(0.0))) == pytest.approx(EPS)
        if param.transform_name == "sigmoid":
            assert float(param.inline_clamp(torch.tensor(1.0))) == pytest.approx(
                1.0 - EPS
            )


def test_a_closed_interval_clamp_refuses_an_empty_interval() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        closed_interval_clamp(1.0, 0.0)


def test_a_family_refuses_bounds_for_a_parameter_it_lacks() -> None:
    with pytest.raises(ValueError, match="no parameter named"):
        _make_family(
            "ConditionalBoundedPoisson",
            D.Poisson,
            [("rate", "softplus")],
            "A Poisson whose bounds name a missing parameter.",
            dsl_name="BoundedPoissonUnregistered",
            inline_bounds={"scale": (0.0, math.inf)},
        )
