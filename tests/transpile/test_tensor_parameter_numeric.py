"""Vector and matrix family parameters score one joint on every path.

Each fixture fills a multivariate family's vector or matrix parameter from
host data or from the program's own values. It is scored on the compiled
runtime and the reference machine, both pinned to an independent
`torch.distributions` joint computed in double precision, and on every
target that renders it, against the same joint. A target with no spelling
for a fixture refuses it under a stated kind.
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
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number, a list of numbers, or a list of rows."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_HOST_LOC = "steps/sample_mvn_host_loc"
_HOST_FACTOR = "steps/observe_mvn_host_factor"
_WISHART_HOST = "steps/sample_wishart_host_factor"

_L_PRIOR = [[1.0, 0.0], [0.5, 1.0]]
_L_OBS = [[1.5, 0.0], [-0.3, 0.8]]
_L_HOST = [[1.2, 0.0], [0.4, 0.7]]


def _t(value: torch.Tensor | Host) -> torch.Tensor:
    """A value in double precision, at which every oracle is computed.

    Parameters
    ----------
    value : torch.Tensor | Host
        The value.

    Returns
    -------
    torch.Tensor
        The value as a float64 tensor.
    """
    return torch.as_tensor(value, dtype=torch.float64)


def _host_loc(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu = _t(v["mu"])
    return D.MultivariateNormal(_t(v["m0"]), scale_tril=_t(_L_PRIOR)).log_prob(
        mu
    ) + D.MultivariateNormal(mu, scale_tril=_t(_L_OBS)).log_prob(_t(v["y"]))


def _host_factor(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, factor = _t(v["mu"]), _t(v["L"])
    return (
        D.MultivariateNormal(_t(v["m0"]), scale_tril=factor).log_prob(mu)
        + D.MultivariateNormal(mu, scale_tril=factor).log_prob(_t(v["y"])).sum()
    )


def _wishart_host(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return D.Wishart(df=_t(4.0), scale_tril=_t(v["L"])).log_prob(_t(v["W"]))


#: Each case: its fixture, its data, its clamped points, and its oracle.
_CASES: dict[str, tuple[str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "host-loc": (
        _HOST_LOC,
        {"m0": [0.3, -0.4], "y": [0.9, -0.6]},
        [{"mu": [0.2, 0.1]}, {"mu": [-1.0, 0.7]}, {"mu": [1.5, -0.2]}],
        _host_loc,
    ),
    "host-factor": (
        _HOST_FACTOR,
        {
            "m0": [0.3, -0.4],
            "L": _L_HOST,
            "y": [[0.9, -0.6], [0.1, 0.4], [-1.3, 0.2]],
        },
        [{"mu": [0.2, 0.1]}, {"mu": [-1.0, 0.7]}, {"mu": [1.5, -0.2]}],
        _host_factor,
    ),
    "wishart-host": (
        _WISHART_HOST,
        {"L": _L_HOST},
        [
            {"W": [[2.0, 0.3], [0.3, 1.1]]},
            {"W": [[0.7, -0.2], [-0.2, 0.9]]},
            {"W": [[4.1, 1.0], [1.0, 2.2]]},
        ],
        _wishart_host,
    ),
}

_EVERY_TARGET: tuple[str, ...] = (
    "bugs",
    "church",
    "edward2",
    "gen",
    "jags",
    "numpyro",
    "pymc",
    "pyro",
    "stan",
    "turing",
    "webppl",
)

#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    (_HOST_LOC, "bugs"): "arg:matrix-literal",
    (_HOST_LOC, "church"): "arg:matrix-literal",
    (_HOST_LOC, "jags"): "family:MultivariateNormal:non-ref-covariance",
    **{
        (_HOST_FACTOR, target): "family:MultivariateNormal:scale_tril"
        for target in _EVERY_TARGET
    },
    **{(_WISHART_HOST, target): "family:Wishart:scale_tril" for target in _EVERY_TARGET},
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


def _nested(value: torch.Tensor) -> float | tuple[object, ...]:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    float | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return float(value.item())
    return tuple(_nested(item) for item in value)


def _reference(case: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    list[float]
        The reference machine's log joint at each point, equal to the
        compiled runtime's and to the case's oracle.
    """
    fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    densities: list[float] = []
    for point in points:
        values = {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in {**data, **point}.items()
        }
        expected = float(oracle(values))
        joint = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={name: _nested(values[name].double()) for name in data},
            sites={name: _nested(values[name].double()) for name in point},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(run.log_joint)
    return densities


def _shapes(case: str) -> dict[str, list[int]]:
    """The shape of every name a target reads.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    dict[str, list[int]]
        Each datum's and site's shape.
    """
    _, data, points, _ = _CASES[case]
    return {
        name: list(torch.as_tensor(value).shape)
        for name, value in {**data, **points[0]}.items()
    }


def _flat(value: Host) -> Host:
    """A host value flattened to a list in row-major order.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    Host
        A number, or the flat list of its entries.
    """
    tensor = torch.as_tensor(value, dtype=torch.float64)
    return tensor.item() if tensor.dim() == 0 else tensor.reshape(-1).tolist()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(case: str) -> None:
    assert len(_reference(case)) == len(_CASES[case][2])


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
    shapes = _shapes(case)
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {
                "params": {name: _flat(value) for name, value in point.items()},
                "data": {name: _flat(value) for name, value in data.items()},
            }
            for point in points
        ],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={name: "float" for name in shapes},
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=math.prod(shapes.get("y", [1]))),
        context=f"{backend}@{case}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )


def test_every_target_renders_or_refuses_every_fixture() -> None:
    for fixture in {fixture for fixture, _, _, _ in _CASES.values()}:
        for target in _EVERY_TARGET:
            if (fixture, target) in _REFUSED:
                continue
            assert target in _TARGETS, (fixture, target)
            transpile(parse(_source(fixture)), target=target)
