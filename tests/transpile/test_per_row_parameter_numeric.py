"""A per-row value filling a family parameter is read at its row on every target.

Three fixtures put one value per plate row into a family parameter of a
plated observe: a host datum read directly as a Normal scale, a let read as
a Gamma rate (which a scale-parameterized target reciprocates), and a
`factor` let inside a marginalize scope whose axis shares its cardinality
with the marginalized latent's. Each is scored on the compiled runtime and
the reference machine, both pinned to an independent `torch.distributions`
joint, and on every target that renders it. A target that cannot express a
fixture refuses it under a stated kind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a list of numbers."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_SCALE = "steps/observe_per_row_datum_scale"
_RATE = "steps/observe_per_row_gamma_rate"
_CHANGE = "steps/marginalize_factor_let_rows"


def _f(value: float) -> torch.Tensor:
    """A constant in double precision, so the oracle stays float64.

    Parameters
    ----------
    value : float
        The constant.

    Returns
    -------
    torch.Tensor
        A float64 scalar.
    """
    return torch.tensor(value, dtype=torch.float64)


def _scale(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y, se = v["mu"], v["y"], v["se"]
    return D.Normal(_f(0.0), _f(10.0)).log_prob(mu) + D.Normal(mu, se).log_prob(y).sum()


def _rate(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    nu, y, x = v["nu"], v["y"], v["x"]
    return (
        D.Gamma(_f(2.0), _f(0.5)).log_prob(nu) + D.Gamma(nu, nu / x).log_prob(y).sum()
    )


def _change(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    before, after, y = v["rate_before"], v["rate_after"], v["y"]
    steps = torch.arange(y.shape[0])
    terms = []
    for tau in range(4):
        rate = torch.where(steps > tau, after, before)
        terms.append(math.log(0.25) + D.Poisson(rate).log_prob(y).sum())
    prior = D.Gamma(_f(2.0), _f(1.0))
    return (
        prior.log_prob(before)
        + prior.log_prob(after)
        + torch.logsumexp(torch.stack(terms), dim=0)
    )


#: Each case: its fixture, its data, its clamped points, and its oracle.
_CASES: dict[str, tuple[str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "datum-scale": (
        _SCALE,
        {"y": [28.0, 8.0, -3.0, 7.0], "se": [15.0, 10.0, 16.0, 11.0]},
        [{"mu": -2.0}, {"mu": 4.5}, {"mu": 11.0}],
        _scale,
    ),
    "gamma-rate": (
        _RATE,
        {"y": [0.8, 2.5, 1.1, 4.0], "x": [0.5, 2.0, 1.0, 3.5]},
        [{"nu": 0.7}, {"nu": 2.2}, {"nu": 5.0}],
        _rate,
    ),
    "changepoint": (
        _CHANGE,
        {"y": [1, 0, 6, 9]},
        [
            {"rate_before": 0.6, "rate_after": 7.0},
            {"rate_before": 2.0, "rate_after": 2.5},
            {"rate_before": 5.0, "rate_after": 0.4},
        ],
        _change,
    ),
}

#: The program each fixture exports.
_PROGRAMS: dict[str, str] = {_SCALE: "pool", _RATE: "regress", _CHANGE: "change"}

#: The data names each fixture reads as integers.
_INTEGER_DATA: dict[str, frozenset[str]] = {
    _SCALE: frozenset(),
    _RATE: frozenset(),
    _CHANGE: frozenset({"y"}),
}

#: The refusals each target states for a fixture it cannot render. A
#: marginalize with no index over a plated observe shares one latent
#: across the rows, which these targets score the other way round.
_REFUSED: dict[tuple[str, str], str] = {
    (_CHANGE, target): "marginalize:ungrouped-over-plate:tau"
    for target in ("bugs", "church", "edward2", "gen", "jags", "pymc", "turing")
}

_TARGETS: tuple[str, ...] = tuple(sorted(_BACKENDS_WITH_IMAGES))


def _source(fixture: str) -> str:
    """A fixture's QVR source.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    str
        The source text.
    """
    return (_FIXTURES / f"{fixture}.qvr").read_text()


def _tensor(fixture: str, name: str, value: Host) -> torch.Tensor:
    """A host value as a tensor of its dtype.

    Parameters
    ----------
    fixture : str
        The fixture, which decides which names are integers.
    name : str
        The value's name.
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    dtype = torch.long if name in _INTEGER_DATA[fixture] else torch.float32
    return torch.as_tensor(value, dtype=dtype)


def _reference(case: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    list[float]
        The compiled runtime's log joint at each point, each equal to the
        reference machine's and to the case's oracle.
    """
    fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    entry = next(
        item for item in program.entry_points() if item.name == _PROGRAMS[fixture]
    )
    densities: list[float] = []
    for point in points:
        merged = {**data, **point}
        values = {name: _tensor(fixture, name, value) for name, value in merged.items()}
        expected = float(oracle({k: v.double() for k, v in values.items()}))
        joint = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={name: _nested(values[name]) for name in data},
            sites={name: _nested(values[name]) for name in point},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(joint)
    return densities


def _nested(value: torch.Tensor) -> float | int | tuple[float | int, ...]:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        A scalar or a vector.

    Returns
    -------
    float | int | tuple[float | int, ...]
        A number, or a tuple of numbers.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(item.item() for item in value)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(case: str) -> None:
    assert len(_reference(case)) == len(_CASES[case][2])


def test_the_changepoint_oracle_separates_rows_from_a_shared_rate() -> None:
    # Reading one rate for every row (the regime before the change) in
    # place of each row's own scores a joint the oracle tells apart.
    _, data, points, oracle = _CASES["changepoint"]
    values = {
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in {**data, **points[0]}.items()
    }
    prior = D.Gamma(_f(2.0), _f(1.0))
    shared = (
        prior.log_prob(values["rate_before"])
        + prior.log_prob(values["rate_after"])
        + D.Poisson(values["rate_before"]).log_prob(values["y"]).sum()
    )
    assert abs(float(oracle(values)) - float(shared)) > 1.0


@pytest.mark.parametrize(
    ("case", "backend"),
    [
        (case, backend)
        for case in sorted(_CASES)
        for backend in _TARGETS
        if (_CASES[case][0], backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    case: str, backend: str, scratch: pathlib.Path
) -> None:
    fixture, data, points, _ = _CASES[case]
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    shapes: dict[str, list[int]] = {
        name: list(torch.as_tensor(value).shape) for name, value in data.items()
    }
    shapes.update({name: [] for name in points[0]})
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": dict(point), "data": dict(data)} for point in points],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={
            name: "int" if name in _INTEGER_DATA[fixture] else "float"
            for name in shapes
        },
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=math.prod(shapes["y"])),
        context=f"{backend}@{case}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize(
    "case",
    [case for case in sorted(_CASES) if (_CASES[case][0], "church") not in _REFUSED],
)
def test_church_scores_the_reference_joint(case: str, scratch: pathlib.Path) -> None:
    fixture, data, points, _ = _CASES[case]
    source = transpile(parse(_source(fixture)), target="church")
    result = ChurchProbe().evaluate(
        source,
        case,
        [_protocol.Point(params=dict(point), data=dict(data)) for point in points],
        scratch=scratch / case / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=torch.as_tensor(data["y"]).numel()),
        context=f"church@{case}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )
