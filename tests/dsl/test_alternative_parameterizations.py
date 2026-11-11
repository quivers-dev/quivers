"""Alternative parameterizations score as `torch.distributions` does on both paths.

A multivariate normal or a Wishart written with a covariance or a
precision matrix, and a Bernoulli, Binomial, Geometric, or negative
binomial written with logits, are each scored on the compiled runtime and
the reference machine against `torch.distributions` built with the same
parameterization in double precision. Each family's alternatives are
checked to be distinct (a precision read as a covariance, or logits read
as probabilities, scores another density), logits stay finite where the
probability they name rounds to zero or one, and a draw naming two
parameterizations is refused at the draw.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.family_schemas import family_parameterizations
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.inference.trace import trace
from quivers.qiec.families import FAMILIES

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number, a vector, or a matrix as a case writes it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a case's joint, in double precision."""

_HEADER = (
    "object One : FinSet 1\nobject Two : FinSet 2\nobject Row : FinSet 4\n"
    "object D : Real 2\n\n"
)

_COVARIANCE = [[1.0, 0.5], [0.5, 1.25]]
_PRECISION = [[1.25, -0.5], [-0.5, 1.0]]
_SCALE = [[2.0, 0.3], [0.3, 0.9]]


def _t(value: torch.Tensor | Host) -> torch.Tensor:
    """A value as a float64 tensor.

    Parameters
    ----------
    value : torch.Tensor | Host
        The value.

    Returns
    -------
    torch.Tensor
        The value in double precision.
    """
    return torch.as_tensor(value, dtype=torch.float64)


def _normal(value: torch.Tensor) -> torch.Tensor:
    """The standard normal log-density summed over a value's entries.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    torch.Tensor
        The summed log-density.
    """
    return D.Normal(_t(0.0), _t(1.0)).log_prob(_t(value)).sum()


#: Each case: its family and parameterization, the program, its data, its
#: site values, and the oracle.
_CASES: dict[str, tuple[str, str, str, dict[str, Host], dict[str, Host], Oracle]] = {
    "mvn-covariance-literal": (
        "MultivariateNormal",
        "covariance_matrix",
        "program p : One -> D\n"
        "    sample m <- Normal(loc=0.0, scale=1.0) [over=Two]\n"
        f"    sample v <- MultivariateNormal(loc=m, covariance_matrix={_COVARIANCE})\n"
        "    return v\n",
        {},
        {"m": [0.3, -0.2], "v": [0.9, 0.4]},
        lambda s: (
            _normal(s["m"])
            + D.MultivariateNormal(
                _t(s["m"]), covariance_matrix=_t(_COVARIANCE)
            ).log_prob(_t(s["v"]))
        ),
    ),
    "mvn-covariance-host": (
        "MultivariateNormal",
        "covariance_matrix",
        "program p : One -> D\n"
        "    sample v <- MultivariateNormal(loc=m0, covariance_matrix=S)\n"
        "    return v\n",
        {"m0": [0.1, -0.3], "S": _COVARIANCE},
        {"v": [0.9, 0.4]},
        lambda s: D.MultivariateNormal(
            _t(s["m0"]), covariance_matrix=_t(s["S"])
        ).log_prob(_t(s["v"])),
    ),
    "mvn-precision-literal": (
        "MultivariateNormal",
        "precision_matrix",
        "program p : One -> D\n"
        "    sample m <- Normal(loc=0.0, scale=1.0) [over=Two]\n"
        f"    sample v <- MultivariateNormal(loc=m, precision_matrix={_PRECISION})\n"
        "    return v\n",
        {},
        {"m": [0.3, -0.2], "v": [0.9, 0.4]},
        lambda s: (
            _normal(s["m"])
            + D.MultivariateNormal(
                _t(s["m"]), precision_matrix=_t(_PRECISION)
            ).log_prob(_t(s["v"]))
        ),
    ),
    "mvn-precision-host": (
        "MultivariateNormal",
        "precision_matrix",
        "program p : One -> D\n"
        "    sample v <- MultivariateNormal(loc=m0, precision_matrix=P)\n"
        "    observe y : Row <- MultivariateNormal(loc=v, precision_matrix=P) "
        "[over=Two]\n"
        "    return v\n",
        {
            "m0": [0.1, -0.3],
            "P": _PRECISION,
            "y": [[0.2, 0.1], [-0.5, 0.7], [1.1, 0.0], [0.3, -0.9]],
        },
        {"v": [0.9, 0.4]},
        lambda s: (
            D.MultivariateNormal(_t(s["m0"]), precision_matrix=_t(s["P"])).log_prob(
                _t(s["v"])
            )
            + D.MultivariateNormal(_t(s["v"]), precision_matrix=_t(s["P"]))
            .log_prob(_t(s["y"]))
            .sum()
        ),
    ),
    "wishart-covariance": (
        "Wishart",
        "covariance_matrix",
        "program p : One -> D\n"
        f"    sample W <- Wishart(df=4.0, covariance_matrix={_COVARIANCE})\n"
        "    return W\n",
        {},
        {"W": _SCALE},
        lambda s: D.Wishart(_t(4.0), covariance_matrix=_t(_COVARIANCE)).log_prob(
            _t(s["W"])
        ),
    ),
    "wishart-precision": (
        "Wishart",
        "precision_matrix",
        "program p : One -> D\n"
        "    sample W <- Wishart(df=4.0, precision_matrix=P)\n"
        "    return W\n",
        {"P": _PRECISION},
        {"W": _SCALE},
        lambda s: D.Wishart(_t(4.0), precision_matrix=_t(s["P"])).log_prob(_t(s["W"])),
    ),
    "bernoulli-logits": (
        "Bernoulli",
        "logits",
        "program p : One -> One\n"
        "    sample eta <- Normal(loc=0.0, scale=2.0)\n"
        "    observe y : Row <- Bernoulli(logits=eta)\n"
        "    return eta\n",
        {"y": [True, False, True, True]},
        {"eta": 0.7},
        lambda s: (
            D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
            + D.Bernoulli(logits=_t(s["eta"])).log_prob(_t(s["y"])).sum()
        ),
    ),
    "binomial-logits": (
        "Binomial",
        "logits",
        "program p : One -> One\n"
        "    sample eta <- Normal(loc=0.0, scale=2.0)\n"
        "    observe y : Row <- Binomial(total_count=10, logits=eta)\n"
        "    return eta\n",
        {"y": [3, 7, 5, 9]},
        {"eta": -0.4},
        lambda s: (
            D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
            + D.Binomial(10, logits=_t(s["eta"])).log_prob(_t(s["y"])).sum()
        ),
    ),
    "geometric-logits": (
        "Geometric",
        "logits",
        "program p : One -> One\n"
        "    sample eta <- Normal(loc=0.0, scale=2.0)\n"
        "    observe y : Row <- Geometric(logits=eta)\n"
        "    return eta\n",
        {"y": [0, 2, 5, 1]},
        {"eta": -0.6},
        lambda s: (
            D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
            + D.Geometric(logits=_t(s["eta"])).log_prob(_t(s["y"])).sum()
        ),
    ),
    "negative-binomial-logits": (
        "NegativeBinomial",
        "logits",
        "program p : One -> One\n"
        "    sample eta <- Normal(loc=0.0, scale=2.0)\n"
        "    observe y : Row <- NegativeBinomial(total_count=3.0, logits=eta)\n"
        "    return eta\n",
        {"y": [0, 4, 2, 7]},
        {"eta": 0.5},
        lambda s: (
            D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
            + D.NegativeBinomial(_t(3.0), logits=_t(s["eta"]))
            .log_prob(_t(s["y"]))
            .sum()
        ),
    ),
}


def _tensor(value: Host) -> torch.Tensor:
    """A value as the compiled runtime takes it: integers and Booleans as
    written, reals in single precision.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    tensor = torch.as_tensor(value)
    return tensor.float() if tensor.is_floating_point() else tensor


def _nested(value: Host) -> float | int | tuple[object, ...]:
    """A value as the reference machine's nested tuples.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    float | int | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    return value


def _both_paths(case: str, sites: Mapping[str, Host]) -> tuple[float, float, float]:
    """The compiled runtime's, the reference machine's, and the oracle's joint.

    Parameters
    ----------
    case : str
        The case.
    sites : Mapping[str, Host]
        The site values to score at.

    Returns
    -------
    tuple[float, float, float]
        The three joints.
    """
    _, _, body, data, _, oracle = _CASES[case]
    program = loads(_HEADER + body + "export p\n")
    values = {name: _tensor(value) for name, value in {**data, **sites}.items()}
    compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
    reference = program.run(
        "p",
        data={name: _nested(value) for name, value in data.items()},
        sites={name: _nested(value) for name, value in sites.items()},
    ).log_joint
    exact = {name: _t(value) for name, value in {**data, **sites}.items()}
    return compiled, reference, float(oracle(exact))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_paths_score_the_torch_density_of_the_written_parameterization(
    case: str,
) -> None:
    compiled, reference, expected = _both_paths(case, _CASES[case][4])
    assert compiled == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9, abs=1e-9)


#: The case each alternative is told apart from: the same values read
#: under the canonical parameterization score another density.
_MISREAD: dict[str, Oracle] = {
    "mvn-precision-literal": lambda s: (
        _normal(s["m"])
        + D.MultivariateNormal(_t(s["m"]), covariance_matrix=_t(_PRECISION)).log_prob(
            _t(s["v"])
        )
    ),
    "wishart-precision": lambda s: D.Wishart(
        _t(4.0), covariance_matrix=_t(s["P"])
    ).log_prob(_t(s["W"])),
    "bernoulli-logits": lambda s: (
        D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
        + D.Bernoulli(probs=_t(s["eta"])).log_prob(_t(s["y"])).sum()
    ),
    "binomial-logits": lambda s: (
        D.Normal(_t(0.0), _t(2.0)).log_prob(_t(s["eta"]))
        + D.Binomial(10, probs=torch.sigmoid(-_t(s["eta"]))).log_prob(_t(s["y"])).sum()
    ),
}


@pytest.mark.parametrize("case", sorted(_MISREAD))
def test_a_misread_parameterization_scores_another_density(case: str) -> None:
    _, _, _, data, sites, _ = _CASES[case]
    compiled, reference, _ = _both_paths(case, sites)
    values = {name: _tensor(value) for name, value in {**data, **sites}.items()}
    wrong = float(_MISREAD[case](values))
    assert abs(compiled - wrong) > 1e-2
    assert abs(reference - wrong) > 1e-2


@pytest.mark.parametrize("eta", [-30.0, -12.0, 12.0, 30.0])
@pytest.mark.parametrize(
    "case", ["bernoulli-logits", "binomial-logits", "geometric-logits"]
)
def test_logits_score_finitely_where_the_probability_rounds_off(
    case: str, eta: float
) -> None:
    # At |eta| = 30 the probability 1 / (1 + exp(-eta)) rounds to 0 or 1 in
    # float32, so its probability form would score -inf for some outcome.
    compiled, reference, expected = _both_paths(case, {"eta": eta})
    if case == "geometric-logits":
        # `torch.distributions.Geometric` scores through the probability
        # even in double precision, losing digits of log(1 - p) once the
        # logit passes about 20; the closed form in log space is exact.
        logit = _t(eta)
        counts = _t(_CASES[case][3]["y"])
        expected = float(
            D.Normal(_t(0.0), _t(2.0)).log_prob(logit)
            + (
                counts * -torch.nn.functional.softplus(logit)
                - torch.nn.functional.softplus(-logit)
            ).sum()
        )
    assert math.isfinite(compiled) and math.isfinite(reference)
    assert compiled == pytest.approx(expected, rel=1e-4)
    assert reference == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_forward_draw_is_scored_at_its_own_density(case: str) -> None:
    _, _, body, data, _, _ = _CASES[case]
    program = loads(_HEADER + body + "export p\n")
    observed = {name: _tensor(value) for name, value in data.items()}
    torch.manual_seed(13)
    recorded = trace(program.morphism, torch.zeros(1, 1), observations=observed)
    values = {name: site.value for name, site in recorded.sites.items()}
    rescored = float(
        program.morphism.log_joint(torch.zeros(1, 1), {**observed, **values}).sum()
    )
    assert float(recorded.log_joint.sum()) == pytest.approx(rescored, rel=1e-5)


@pytest.mark.parametrize(
    ("family", "call"),
    [
        (
            "MultivariateNormal",
            f"MultivariateNormal(loc=[0.0, 0.0], scale_tril={_COVARIANCE}, "
            f"precision_matrix={_PRECISION}) [over=Two]",
        ),
        ("Bernoulli", "Bernoulli(probs=0.3, logits=0.2)"),
        ("Binomial", "Binomial(total_count=4, probs=0.3, logits=0.2)"),
    ],
)
def test_two_parameterizations_in_one_draw_are_refused_at_the_draw(
    family: str, call: str
) -> None:
    source = (
        _HEADER
        + f"program p : One -> One\n    sample v <- {call}\n    return v\nexport p\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _ = loads(source).qiec
    message = str(caught.value)
    assert ":7:" in message
    assert f"family {family!r} requires exactly one complete parameter schema" in (
        message
    )


def test_every_surface_alternative_has_a_case() -> None:
    covered = {(family, parameter) for family, parameter, *_ in _CASES.values()}
    for record in FAMILIES.values():
        schemas = family_parameterizations(record.name, record.parameter_names)
        if len(schemas) < 2 or record.compositional:
            continue
        canonical = set(schemas[0])
        for schema in schemas[1:]:
            for parameter in set(schema) - canonical:
                if record.name == "Categorical":
                    continue
                assert (record.name, parameter) in covered, (record.name, parameter)
