"""Let reductions and host data score one joint on every target.

A reduction builtin collapses the last axis of its argument, and a free
host name takes the shape of the plated values it is combined with. Each
fixture here isolates one case where a target could render a program that
runs and still reduce a different axis or read a datum at a different
shape: a vector over a plate, a plate of event vectors, a table a factor
builds over two plates, reductions nested in one let, and a host weight a
`score` reads elementwise. Every target that renders a fixture is scored
against the reference machine at points that vary every site, and the
reference is pinned to an independent `torch.distributions` joint. A
target with no spelling for a case refuses it under a stated kind.
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
"""A number or a flat list of numbers, as a point carries it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint at shaped values."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _reduce_plate(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    theta, y = v["theta"], v["y"]
    return (
        D.Normal(0.0, 1.0).log_prob(theta).sum()
        + D.Normal(theta.sum(), 1.0).log_prob(y).sum()
    )


def _reduce_event(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    z, y = v["z"], v["y"]
    return (
        D.Normal(0.0, 1.0).log_prob(z).sum()
        + D.Normal(z.sum(-1), 1.0).log_prob(y).sum()
    )


def _reduce_matrix(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    x, w, y = v["x"], v["w"], v["y"]
    mu = torch.outer(x, w).sum(-1)
    return (
        D.Normal(0.0, 1.0).log_prob(x).sum()
        + D.Normal(0.0, 1.0).log_prob(w).sum()
        + D.Normal(mu, 1.0).log_prob(y).sum()
    )


def _reduce_nested(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    w, y = v["w"], v["y"]
    table = w[:, None, None] * w[None, :, None] * w[None, None, :]
    s = table.sum(-1).sum(-1)
    return D.Normal(0.0, 1.0).log_prob(w).sum() + D.Normal(s, 1.0).log_prob(y).sum()


def _score_host_weight(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    theta, y, weight = v["theta"], v["y"], v["weight"]
    return (
        D.Normal(0.0, 1.0).log_prob(theta).sum()
        + D.Normal(theta, 1.0).log_prob(y).sum()
        + (weight * theta).sum()
    )


#: Each fixture's data inputs, as flat lists.
_DATA: dict[str, dict[str, Host]] = {
    "let_expressions/let_reduce_plate": {"y": [0.4, -1.1, 2.3]},
    "let_expressions/let_reduce_event": {"y": [0.4, -1.1, 2.3]},
    "let_expressions/let_reduce_matrix": {"y": [0.4, -1.1, 2.3]},
    "let_expressions/let_reduce_nested": {"y": [0.6, -0.9]},
    "let_expressions/let_score_host_weight": {
        "y": [0.5, -1.0, 2.0, 0.3],
        "weight": [1.0, 0.0, 1.0, 0.5],
    },
}

#: Each fixture's clamped points; every site varies across them.
_POINTS: dict[str, list[dict[str, Host]]] = {
    "let_expressions/let_reduce_plate": [
        {"theta": [0.3, -0.2, 1.1, 0.4]},
        {"theta": [-1.0, 0.8, 0.0, -0.5]},
        {"theta": [1.4, 0.1, -0.7, 0.9]},
    ],
    "let_expressions/let_reduce_event": [
        {"z": [0.3, -0.2, 1.1, 0.4, -0.6, 0.9]},
        {"z": [-1.0, 0.8, 0.0, -0.5, 1.2, 0.3]},
        {"z": [1.4, 0.1, -0.7, 0.9, 0.2, -1.3]},
    ],
    "let_expressions/let_reduce_matrix": [
        {"x": [0.3, -0.2, 1.1], "w": [0.4, -0.6]},
        {"x": [-1.0, 0.8, 0.0], "w": [-0.5, 1.2]},
        {"x": [1.4, 0.1, -0.7], "w": [0.9, 0.2]},
    ],
    "let_expressions/let_reduce_nested": [
        {"w": [0.3, -0.8]},
        {"w": [-1.0, 0.5]},
        {"w": [0.7, 0.9]},
    ],
    "let_expressions/let_score_host_weight": [
        {"theta": [0.3, -0.2, 1.1, 0.4]},
        {"theta": [-1.0, 0.8, 0.0, -0.5]},
        {"theta": [1.4, 0.1, -0.7, 0.9]},
    ],
}

#: The shape of every name a point carries; a scalar's is empty.
_SHAPES: dict[str, dict[str, list[int]]] = {
    "let_expressions/let_reduce_plate": {"theta": [4], "y": [3]},
    "let_expressions/let_reduce_event": {"z": [3, 2], "y": [3]},
    "let_expressions/let_reduce_matrix": {"x": [3], "w": [2], "y": [3]},
    "let_expressions/let_reduce_nested": {"w": [2], "y": [2]},
    "let_expressions/let_score_host_weight": {"theta": [4], "y": [4], "weight": [4]},
}

#: Each fixture's independent joint.
_ORACLES: dict[str, Oracle] = {
    "let_expressions/let_reduce_plate": _reduce_plate,
    "let_expressions/let_reduce_event": _reduce_event,
    "let_expressions/let_reduce_matrix": _reduce_matrix,
    "let_expressions/let_reduce_nested": _reduce_nested,
    "let_expressions/let_score_host_weight": _score_host_weight,
}

#: The refusals each target states for a fixture it cannot render, by
#: the kind prefix the refusal names.
_REFUSED: dict[tuple[str, str], str] = {
    ("let_expressions/let_reduce_nested", "bugs"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_nested", "church"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_nested", "gen"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_nested", "jags"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_nested", "stan"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_nested", "turing"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_matrix", "bugs"): "let-expr:axis-reduction",
    ("let_expressions/let_reduce_matrix", "jags"): "let-expr:axis-reduction",
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


def _shaped(fixture: str, values: Mapping[str, Host]) -> dict[str, torch.Tensor]:
    """A point's values as tensors of their declared shapes.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.
    values : Mapping[str, Host]
        Flat values by name.

    Returns
    -------
    dict[str, torch.Tensor]
        Each value reshaped.
    """
    return {
        name: torch.as_tensor(value, dtype=torch.float32).reshape(
            _SHAPES[fixture][name]
        )
        for name, value in values.items()
    }


def _points(fixture: str) -> list[_protocol.Point]:
    """The fixture's clamped points with its data.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    list[_protocol.Point]
        One point per entry of the fixture's point set.
    """
    return [
        _protocol.Point(params=dict(point), data=dict(_DATA[fixture]))
        for point in _POINTS[fixture]
    ]


def _reference(fixture: str) -> list[float]:
    """The compiled runtime's joint at each point, checked twice over.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    list[float]
        The log joint at each point, each equal to the reference
        machine's and to the fixture's independent oracle.
    """
    program = loads(_source(fixture))
    densities: list[float] = []
    for point in _POINTS[fixture]:
        values = _shaped(fixture, {**_DATA[fixture], **point})
        joint = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        oracle = float(_ORACLES[fixture](values))
        assert joint == pytest.approx(oracle, rel=1e-5)
        run = program.run(
            "prog",
            data={name: _nested(values[name]) for name in _DATA[fixture]},
            sites={name: _nested(values[name]) for name in point},
        )
        assert run.log_joint == pytest.approx(oracle, rel=1e-5)
        densities.append(joint)
    return densities


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
        return float(value)
    return tuple(_nested(item) for item in value)


def _n_obs(fixture: str) -> int:
    """The number of observed rows a fixture's joint sums over.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    int
        The size of the observation `y`.
    """
    return math.prod(_SHAPES[fixture]["y"])


@pytest.mark.parametrize("fixture", sorted(_ORACLES))
def test_reference_matches_the_independent_joint(fixture: str) -> None:
    assert len(_reference(fixture)) == len(_POINTS[fixture])


@pytest.mark.parametrize(
    ("fixture", "backend"),
    [
        (fixture, backend)
        for fixture in sorted(_ORACLES)
        for backend in _TARGETS
        if (fixture, backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    fixture: str, backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    points = _points(fixture)
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": pt.params, "data": pt.data} for pt in points],
        scratch=scratch / fixture.replace("/", "-") / backend,
        shapes=_SHAPES[fixture],
        dtypes={name: "float" for name in _SHAPES[fixture]},
    )
    _equivalence.assert_log_density_match(
        _reference(fixture),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture)),
        context=f"{backend}@{fixture}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize(
    "fixture",
    [fixture for fixture in sorted(_ORACLES) if (fixture, "church") not in _REFUSED],
)
def test_church_scores_the_reference_joint(fixture: str, scratch: pathlib.Path) -> None:
    source = transpile(parse(_source(fixture)), target="church")
    result = ChurchProbe().evaluate(
        source,
        fixture.replace("/", "-"),
        _points(fixture),
        scratch=scratch / fixture.replace("/", "-") / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(fixture),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture)),
        context=f"church@{fixture}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(expected in kind for kind in caught.value.kinds), caught.value.kinds
