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
from collections.abc import Callable

import pytest
import torch
import torch.distributions as D

from quivers.continuous._zip_hurdle import ZeroInflatedPoisson
from quivers.continuous.families import _make_family
from quivers.continuous.family_spec import get_family_spec
from quivers.continuous.measure import Mixture, PointMass
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


# -- the probability and trial-count boundaries of the counting families ------

_PRIOR = -0.5 * math.log(2 * math.pi)
"""The log density of the latent ``a = 0`` under its standard normal prior."""


def _observed_program(family_call: str) -> str:
    """A program scoring two observations under one family call.

    Parameters
    ----------
    family_call : str
        The family and its arguments, which read the data by name.

    Returns
    -------
    str
        The QVR source.
    """
    return (
        "object Resp : FinSet 2\n"
        "object Val : Real 1\n\n"
        "program q : Resp -> Val\n"
        "    sample a <- Normal(loc=0.0, scale=1.0)\n"
        f"    observe y : Resp <- {family_call}\n"
        "    return a\n\n"
        "export q\n"
    )


def _both_paths(
    family_call: str,
    y: list[int] | list[bool],
    data: dict[str, list[float] | list[int]],
) -> tuple[float, float]:
    """The compiled runtime's and the reference machine's joint at ``a = 0``.

    Parameters
    ----------
    family_call : str
        The family and its arguments.
    y : list[int] | list[bool]
        The two observations.
    data : dict[str, list[float] | list[int]]
        The other data, two values each.

    Returns
    -------
    tuple[float, float]
        The compiled and the reference log joint.
    """
    source = _observed_program(family_call)
    values = {
        "a": torch.tensor([0.0]),
        "y": torch.tensor([float(v) for v in y]),
        **{name: torch.tensor(value) for name, value in data.items()},
    }
    compiled = _compiled(source, values)
    program = loads(source)
    entry = program.entry_points()[0]
    run = program.run(
        entry.name,
        data={"y": tuple(y), **{name: tuple(value) for name, value in data.items()}},
        sites={"a": 0.0},
    )
    return compiled, float(run.log_joint)


#: Each family at a boundary its parameters admit, the observations, the
#: data, and the exact log mass of the two observations together.
_BOUNDARY_CASES: dict[
    str, tuple[str, list[int] | list[bool], dict[str, list[float] | list[int]], float]
] = {
    "bernoulli-success-under-zero": (
        "Bernoulli(probs=p)",
        [True, False],
        {"p": [0.0, 0.0]},
        -math.inf,
    ),
    "bernoulli-failure-under-one": (
        "Bernoulli(probs=p)",
        [True, False],
        {"p": [1.0, 1.0]},
        -math.inf,
    ),
    "bernoulli-point-masses": (
        "Bernoulli(probs=p)",
        [True, False],
        {"p": [1.0, 0.0]},
        0.0,
    ),
    "binomial-all-successes-under-one": (
        "Binomial(total_count=n, probs=p)",
        [3, 3],
        {"n": [3, 3], "p": [1.0, 1.0]},
        0.0,
    ),
    "binomial-a-failure-under-one": (
        "Binomial(total_count=n, probs=p)",
        [3, 2],
        {"n": [3, 3], "p": [1.0, 1.0]},
        -math.inf,
    ),
    "binomial-a-success-under-zero": (
        "Binomial(total_count=n, probs=p)",
        [0, 1],
        {"n": [3, 3], "p": [0.0, 0.0]},
        -math.inf,
    ),
    "binomial-zero-trials": (
        "Binomial(total_count=n, probs=p)",
        [0, 0],
        {"n": [0, 0], "p": [0.4, 0.4]},
        0.0,
    ),
    "binomial-a-success-in-zero-trials": (
        "Binomial(total_count=n, probs=p)",
        [0, 1],
        {"n": [0, 0], "p": [0.4, 0.4]},
        -math.inf,
    ),
    "negative-binomial-no-success-under-zero": (
        "NegativeBinomial(total_count=r, probs=p)",
        [0, 0],
        {"r": [2.5, 2.5], "p": [0.0, 0.0]},
        0.0,
    ),
    "negative-binomial-a-success-under-zero": (
        "NegativeBinomial(total_count=r, probs=p)",
        [0, 2],
        {"r": [2.5, 2.5], "p": [0.0, 0.0]},
        -math.inf,
    ),
    "negative-binomial-zero-failures-waited-for": (
        "NegativeBinomial(total_count=r, probs=p)",
        [0, 0],
        {"r": [0.0, 0.0], "p": [0.4, 0.4]},
        0.0,
    ),
    "negative-binomial-a-success-with-none-waited-for": (
        "NegativeBinomial(total_count=r, probs=p)",
        [0, 1],
        {"r": [0.0, 0.0], "p": [0.4, 0.4]},
        -math.inf,
    ),
    "beta-binomial-zero-trials": (
        "BetaBinomial(total_count=n, concentration1=2.0, concentration0=3.0)",
        [0, 0],
        {"n": [0, 0]},
        0.0,
    ),
    "beta-binomial-a-success-in-zero-trials": (
        "BetaBinomial(total_count=n, concentration1=2.0, concentration0=3.0)",
        [0, 1],
        {"n": [0, 0]},
        -math.inf,
    ),
    "zip-a-count-under-certain-inflation": (
        "ZeroInflatedPoisson(zero_prob=z, rate=2.0)",
        [0, 1],
        {"z": [1.0, 1.0]},
        -math.inf,
    ),
    "zip-zeros-under-certain-inflation": (
        "ZeroInflatedPoisson(zero_prob=z, rate=2.0)",
        [0, 0],
        {"z": [1.0, 1.0]},
        0.0,
    ),
    "hurdle-a-count-under-a-certain-zero": (
        "HurdlePoisson(zero_prob=z, rate=2.0)",
        [0, 1],
        {"z": [1.0, 1.0]},
        -math.inf,
    ),
    "hurdle-a-zero-under-no-zero": (
        "HurdlePoisson(zero_prob=z, rate=2.0)",
        [0, 1],
        {"z": [0.0, 0.0]},
        -math.inf,
    ),
}


@pytest.mark.parametrize("case", sorted(_BOUNDARY_CASES))
def test_a_counting_family_at_its_boundary_scores_the_point_mass(case: str) -> None:
    family_call, y, data, log_mass = _BOUNDARY_CASES[case]
    compiled, reference = _both_paths(family_call, y, data)
    expected = _PRIOR + log_mass
    if math.isinf(expected):
        assert compiled == expected
        assert reference == expected
    else:
        assert compiled == pytest.approx(expected, abs=_ATOL)
        assert reference == pytest.approx(expected, abs=1e-12)


def test_the_hurdle_without_zeros_scores_the_truncated_poisson() -> None:
    # A zero probability of zero leaves the zero-truncated Poisson.
    compiled, reference = _both_paths(
        "HurdlePoisson(zero_prob=z, rate=2.0)", [1, 3], {"z": [0.0, 0.0]}
    )
    counts = _f64([1.0, 3.0])
    truncated = D.Poisson(_f64(2.0)).log_prob(counts) - math.log(-math.expm1(-2.0))
    expected = _PRIOR + float(truncated.sum())
    assert compiled == pytest.approx(expected, abs=_ATOL)
    assert reference == pytest.approx(expected, abs=1e-12)


#: Interior parameters, with the `torch.distributions` construction that
#: scores them, for each counting family.
_INTERIOR_CASES: dict[
    str,
    tuple[
        str,
        list[int] | list[bool],
        dict[str, list[float] | list[int]],
        Callable[[], D.Distribution],
    ],
] = {
    "bernoulli": (
        "Bernoulli(probs=p)",
        [True, False],
        {"p": [0.2, 0.9]},
        lambda: D.Bernoulli(probs=_f64([0.2, 0.9])),
    ),
    "binomial": (
        "Binomial(total_count=n, probs=p)",
        [2, 5],
        {"n": [4, 7], "p": [0.3, 0.85]},
        lambda: D.Binomial(total_count=_f64([4.0, 7.0]), probs=_f64([0.3, 0.85])),
    ),
    "negative-binomial": (
        "NegativeBinomial(total_count=r, probs=p)",
        [0, 6],
        {"r": [1.5, 4.0], "p": [0.25, 0.6]},
        lambda: D.NegativeBinomial(
            total_count=_f64([1.5, 4.0]), probs=_f64([0.25, 0.6])
        ),
    ),
}


@pytest.mark.parametrize("case", sorted(_INTERIOR_CASES))
def test_an_interior_counting_parameter_scores_the_torch_density(case: str) -> None:
    family_call, y, data, reference_law = _INTERIOR_CASES[case]
    compiled, reference = _both_paths(family_call, y, data)
    expected = _PRIOR + float(
        reference_law().log_prob(_f64([float(v) for v in y])).sum()
    )
    assert compiled == pytest.approx(expected, rel=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("case", sorted(_INTERIOR_CASES))
def test_an_interior_counting_parameter_keeps_the_torch_gradient(case: str) -> None:
    # The closed-interval scoring differentiates as torch's logit path does
    # wherever the probability is strictly inside the unit interval.
    family = {
        "bernoulli": "Bernoulli",
        "binomial": "Binomial",
        "negative-binomial": "NegativeBinomial",
    }[case]
    spec = get_family_spec(family)
    assert spec is not None
    _, y, data, _ = _INTERIOR_CASES[case]
    counts = _f64([float(v) for v in y])
    probs = _f64(data["p"]).requires_grad_(True)
    torch_probs = _f64(data["p"]).requires_grad_(True)
    extra = {name: _f64(value) for name, value in data.items() if name != "p"}
    torch_class = {
        "bernoulli": D.Bernoulli,
        "binomial": D.Binomial,
        "negative-binomial": D.NegativeBinomial,
    }[case]
    totals = {"total_count": next(iter(extra.values()))} if extra else {}
    closed = spec.dist_class(probs=probs, **totals).log_prob(counts).sum()
    plain = torch_class(probs=torch_probs, **totals).log_prob(counts).sum()
    (closed_grad,) = torch.autograd.grad(closed, probs)
    (plain_grad,) = torch.autograd.grad(plain, torch_probs)
    assert float(closed) == pytest.approx(float(plain), rel=1e-12)
    assert closed_grad.tolist() == pytest.approx(plain_grad.tolist(), rel=1e-10)


def test_torch_scores_a_boundary_probability_through_its_clamp() -> None:
    # The oracle above rejects torch's own probability path: it scores a
    # success under probs = 0 at about -15.9 in single precision.
    clamped = float(D.Bernoulli(probs=torch.tensor(0.0)).log_prob(torch.tensor(1.0)))
    assert -17.0 < clamped < -15.0


# -- the zero-inflated Poisson's weights ---------------------------------------


def test_a_certain_inflation_puts_no_mass_on_a_positive_count() -> None:
    zip_law = ZeroInflatedPoisson(torch.tensor(1.0), torch.tensor(2.0))
    assert float(zip_law.log_prob(torch.tensor(1.0))) == -math.inf
    assert float(zip_law.log_prob(torch.tensor(0.0))) == 0.0


@pytest.mark.parametrize("zero_prob", [0.05, 0.4, 0.93])
def test_an_interior_inflation_scores_and_differentiates_the_closed_form(
    zero_prob: float,
) -> None:
    counts = _f64([0.0, 1.0, 4.0])
    rate = _f64(1.7)
    pi = _f64(zero_prob).requires_grad_(True)
    exact_pi = _f64(zero_prob).requires_grad_(True)
    mixture = ZeroInflatedPoisson(pi, rate).log_prob(counts).sum()
    poisson = D.Poisson(rate).log_prob(counts)
    exact = torch.where(
        counts == 0,
        torch.log(exact_pi + (1.0 - exact_pi) * torch.exp(-rate)),
        torch.log1p(-exact_pi) + poisson,
    ).sum()
    (grad,) = torch.autograd.grad(mixture, pi)
    (exact_grad,) = torch.autograd.grad(exact, exact_pi)
    assert float(mixture) == pytest.approx(float(exact), rel=1e-12)
    assert float(grad) == pytest.approx(float(exact_grad), rel=1e-10)


def test_a_zero_mixture_weight_has_a_finite_gradient() -> None:
    weights = _f64([0.0, 1.0]).requires_grad_(True)
    mixture = Mixture(weights, [PointMass(0.0), D.Poisson(_f64(2.0))])
    log_mass = mixture.log_prob(_f64(1.0))
    (grad,) = torch.autograd.grad(log_mass, weights)
    assert float(log_mass) == pytest.approx(
        float(D.Poisson(_f64(2.0)).log_prob(_f64(1.0)))
    )
    assert bool(torch.isfinite(grad).all())


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_a_saturated_probability_has_a_finite_gradient_where_the_mass_is(
    probability: float,
) -> None:
    # A sigmoid that rounds to exactly zero or one scores the outcome it
    # is certain of at log mass zero, and the gradient through it stays
    # finite rather than the 0 * inf of differentiating 0 log 0.
    spec = get_family_spec("Bernoulli")
    assert spec is not None
    probs = torch.tensor([probability], requires_grad=True)
    outcome = torch.tensor([probability])
    log_mass = spec.dist_class(probs=probs).log_prob(outcome).sum()
    (grad,) = torch.autograd.grad(log_mass, probs)
    assert float(log_mass) == 0.0
    assert bool(torch.isfinite(grad).all())


_GATED = """\
object Resp : FinSet 3
object Val : Real 1
program zip : Resp -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let pi_z = sigmoid(a + x)
    let rate = exp(b)
    marginalize z : Resp <- Bernoulli(probs=pi_z)
        let gated_rate = real(z) * rate
        observe y : Resp <- Poisson(rate=gated_rate)
    return a
export zip
"""


def test_the_gated_rate_s_off_branch_leaves_the_gradient_finite() -> None:
    # The off branch scores a positive count under a rate of zero, -inf
    # with weight zero in the reduction: its gradient is zero, so the
    # gradient in the rate's coefficient is the active branch's alone,
    # against the closed form differentiated in double precision.
    x, y = [0.0, 1.0, -1.0], [0.0, 2.0, 1.0]
    a = torch.tensor([0.3], requires_grad=True)
    b = torch.tensor([0.2], requires_grad=True)
    values = {"a": a, "b": b, "x": torch.tensor(x), "y": torch.tensor(y)}
    loads(_GATED).morphism.log_joint(torch.zeros(1, 1), values).sum().backward()
    a64 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    b64 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    pi = torch.sigmoid(a64 + _f64(x))
    counts = _f64(y)
    on = torch.log(pi) + D.Poisson(torch.exp(b64)).log_prob(counts)
    off = torch.where(counts == 0, torch.log1p(-pi), _f64(-math.inf))
    rows = torch.where(counts == 0, torch.logaddexp(on, off), on)
    exact = (
        rows.sum()
        + D.Normal(_f64(0.0), _f64(1.0)).log_prob(a64)
        + D.Normal(_f64(0.0), _f64(1.0)).log_prob(b64)
    )
    grad_a, grad_b = torch.autograd.grad(exact, (a64, b64))
    assert float(a.grad[0]) == pytest.approx(float(grad_a), rel=1e-4)
    assert float(b.grad[0]) == pytest.approx(float(grad_b), rel=1e-4)
