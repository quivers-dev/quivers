"""Sites drawn over more `over` axes than their family's event rank, on every target.

A family of event rank ``r`` drawn ``[over=[X1, ..., Xn]]`` with
``n > r`` is an independent array over the leading ``n - r`` axes of
one variate on the trailing ``r``, after the plate of an index
annotation. Each fixture reads the array entrywise by literal indices,
so a target that lays the axes out in another order, or flattens them,
scores a different joint. Every rendering target is scored against the
reference machine, and the reference is pinned to an independent
``torch.distributions`` computation of the joint.
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

type Host = float | list[float]
"""A number or a flat list of numbers, as a point carries it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint at shaped values."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_Y = [0.4, -0.3, 0.8, 0.1]


def _scalar_array(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    w, y = v["w"], v["y"]
    t = w[1, 2, 0] - w[0, 1, 1]
    return D.Normal(0.0, 1.0).log_prob(w).sum() + D.Normal(t, 1.0).log_prob(y).sum()


def _vector_array(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    p, y = v["p"], v["y"]
    t = p[1, 0, 2] * 3.0 - p[0, 1, 0]
    prior = D.Dirichlet(torch.full((3,), 2.0)).log_prob(p).sum()
    return prior + D.Normal(t, 1.0).log_prob(y).sum()


def _simplex_rows(seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.softmax(torch.randn(2, 2, 3, generator=generator), dim=-1)
    return [float(value) for value in rows.reshape(-1)]


def _normal_cells(seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    return [float(value) for value in torch.randn(12, generator=generator)]


_DATA: dict[str, dict[str, Host]] = {
    "axes/scalar_independent_array": {"y": _Y},
    "axes/vector_independent_array": {"y": _Y},
}

_POINTS: dict[str, list[dict[str, Host]]] = {
    "axes/scalar_independent_array": [{"w": _normal_cells(seed)} for seed in range(3)],
    "axes/vector_independent_array": [{"p": _simplex_rows(seed)} for seed in range(3)],
}

_SHAPES: dict[str, dict[str, list[int]]] = {
    "axes/scalar_independent_array": {"w": [2, 3, 2], "y": [4]},
    "axes/vector_independent_array": {"p": [2, 2, 3], "y": [4]},
}

_ORACLES: dict[str, Oracle] = {
    "axes/scalar_independent_array": _scalar_array,
    "axes/vector_independent_array": _vector_array,
}


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
        Each value reshaped to its declared shape.
    """
    return {
        name: torch.tensor(value).reshape(_SHAPES[fixture][name])
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
    """The torch runtime's joint at each point, held to the oracle.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    list[float]
        The log joint at each point, each checked against the
        fixture's independent oracle.
    """
    model = loads(_source(fixture)).morphism
    densities: list[float] = []
    for point in _POINTS[fixture]:
        values = _shaped(fixture, {**_DATA[fixture], **point})
        joint = float(model.log_joint(torch.zeros(1, 1), values).sum())
        assert joint == pytest.approx(float(_ORACLES[fixture](values)), rel=1e-5)
        densities.append(joint)
    return densities


def _n_obs(fixture: str) -> int:
    """The number of observed rows a fixture's joint sums over.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    int
        The size of the observation ``y``.
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
        for backend in sorted(_BACKENDS_WITH_IMAGES)
    ],
)
def test_target_scores_the_reference_joint(
    fixture: str, backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    try:
        source = transpile(parse(_source(fixture)), target=backend)
    except UnsupportedConstruct as refusal:
        assert refusal.kinds, refusal
        return
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": pt.params, "data": pt.data} for pt in _points(fixture)],
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
@pytest.mark.parametrize("fixture", sorted(_ORACLES))
def test_church_scores_the_reference_joint(fixture: str, scratch: pathlib.Path) -> None:
    try:
        source = transpile(parse(_source(fixture)), target="church")
    except UnsupportedConstruct as refusal:
        assert refusal.kinds, refusal
        return
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
