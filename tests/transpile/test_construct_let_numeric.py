"""Let-expression construct fixtures, scored on every target against the reference.

Each fixture isolates a let-expression construct whose rendering a
syntax check cannot judge: a ``fold`` whose body reads its accumulator
several times, which every target states as one shared binding per
step, and a draw from a program template whose lets index a renamed
local, build a ``factor`` over an object parameter, and shadow a
parameter with a lambda. Every rendering target is scored against the reference machine
at points that vary every site, and the reference itself is pinned to
an independent ``torch`` computation of the joint.
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
from quivers.transpile import transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a flat list of numbers, as a point carries it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint at shaped values."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _fold_shared_accumulator(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y = v["mu"], v["y"]
    xs = (0.4, -0.3, 0.8)
    total = mu
    for x in xs:
        total = total * total * 0.1 + total * 0.2 + x
    message = [torch.tensor(0.3), torch.tensor(0.5)]
    for x in xs:
        message = [
            message[0] * 0.5 + message[1] * 0.25 + message[j] * x for j in range(2)
        ]
    rows = []
    for x in xs:
        acc = torch.tensor(x * 0.7)
        for z in xs:
            acc = acc * acc * 0.1 + acc * 0.3 + z
        rows.append(acc)
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + total
        + mu * sum(message)
        + mu * mu * sum(rows)
    )


def _binary_min_max(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y = v["mu"], v["y"]
    ys = (1.0, 2.0, 3.0, 4.0)
    highs = sum(max(a, b) for a, b in zip(ys, (2.5, 0.5, 3.5, 1.0), strict=True))
    lag = sum(ys[max(t - 1, 0)] for t in range(4))
    lead = sum(ys[min(t + 1, 3)] for t in range(4))
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + torch.minimum(mu, torch.tensor(1.0))
        + torch.maximum(mu, torch.tensor(0.0))
        + mu * highs
        + mu * mu * lag
        + mu * mu * mu * lead
    )


def _observed_read(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y = v["mu"], v["y"]
    resid = y - mu
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + (resid * resid).sum() * 0.5
    )


def _pow(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y = v["mu"], v["y"]
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + mu**2 * 0.5
        + 1.5**mu
    )


def _real_literals(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mu, y = v["mu"], v["y"]
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + mu * (12.0 + 3.5)
    )


#: Each fixture's data inputs, as flat lists.
def _template_substitution(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    effect, y = v["a$effect"], v["y"]
    shifted = effect[1] * 0.5 * 2.0 - effect[0]
    return (
        D.Normal(0.0, 0.5).log_prob(effect).sum()
        + D.Normal(shifted, 1.0).log_prob(y).sum()
    )


_DATA: dict[str, dict[str, Host]] = {
    "let_expressions/let_fold_shared_accumulator": {"y": [0.4, -0.3, 0.8]},
    "let_expressions/let_binary_min_max": {"y": [0.2, -0.5, 1.1, 0.4]},
    "let_expressions/let_observed_read": {"y": [0.5, -1.0, 2.0, 0.3]},
    "let_expressions/let_template_substitution": {"y": [0.4, -0.3, 0.8]},
    "let_expressions/let_pow": {"y": [0.4, -0.3, 0.8]},
    "let_expressions/let_real_literals": {"y": [0.2, -0.1, 0.6]},
}

#: Each fixture's clamped points; every site varies across them.
_POINTS: dict[str, list[dict[str, Host]]] = {
    "let_expressions/let_fold_shared_accumulator": [
        {"mu": -0.6},
        {"mu": 0.2},
        {"mu": 1.1},
    ],
    "let_expressions/let_binary_min_max": [{"mu": -0.6}, {"mu": 0.4}, {"mu": 1.7}],
    "let_expressions/let_observed_read": [{"mu": -0.4}, {"mu": 0.2}, {"mu": 1.1}],
    "let_expressions/let_template_substitution": [
        {"a$effect": [-0.6, 0.3]},
        {"a$effect": [0.2, -1.1]},
        {"a$effect": [1.4, 0.7]},
    ],
    "let_expressions/let_pow": [{"mu": -0.7}, {"mu": 0.3}, {"mu": 1.4}],
    "let_expressions/let_real_literals": [{"mu": -0.4}, {"mu": 0.1}, {"mu": 0.9}],
}

#: The shape of every name a point carries; a scalar's is empty.
_SHAPES: dict[str, dict[str, list[int]]] = {
    "let_expressions/let_fold_shared_accumulator": {"mu": [], "y": [3]},
    "let_expressions/let_binary_min_max": {"mu": [], "y": [4]},
    "let_expressions/let_observed_read": {"mu": [], "y": [4]},
    "let_expressions/let_template_substitution": {"a$effect": [2], "y": [3]},
    "let_expressions/let_pow": {"mu": [], "y": [3]},
    "let_expressions/let_real_literals": {"mu": [], "y": [3]},
}

#: The integer-valued names of each fixture; every other name is real.
_INTEGERS: dict[str, frozenset[str]] = {}

#: Each fixture's independent joint.
_ORACLES: dict[str, Oracle] = {
    "let_expressions/let_fold_shared_accumulator": _fold_shared_accumulator,
    "let_expressions/let_binary_min_max": _binary_min_max,
    "let_expressions/let_observed_read": _observed_read,
    "let_expressions/let_template_substitution": _template_substitution,
    "let_expressions/let_pow": _pow,
    "let_expressions/let_real_literals": _real_literals,
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


def _dtypes(fixture: str) -> dict[str, str]:
    """Each name's wire dtype.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    dict[str, str]
        ``"int"`` for an integer-valued name, ``"float"`` otherwise.
    """
    integers = _INTEGERS.get(fixture, frozenset())
    return {name: "int" if name in integers else "float" for name in _SHAPES[fixture]}


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
        Each value reshaped, integer names as `torch.long`.
    """
    integers = _INTEGERS.get(fixture, frozenset())
    shaped: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        dtype = torch.long if name in integers else torch.get_default_dtype()
        shaped[name] = torch.tensor(value, dtype=dtype).reshape(_SHAPES[fixture][name])
    return shaped


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
        sites = {
            name: value.reshape(1) if value.dim() == 0 else value
            for name, value in values.items()
        }
        joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
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
        for backend in sorted(_BACKENDS_WITH_IMAGES)
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
        dtypes=_dtypes(fixture),
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
