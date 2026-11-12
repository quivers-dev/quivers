"""The statistical fixtures of the QIEC corpus score the same joint everywhere.

Two fixtures under `tests/fixtures/qiec/` carry a joint density:

* `effectful_probabilistic_helper` calls an indexed helper that case
  analyses a GADT, scoring a `Present` measurement and drawing an
  `Absent` one under a site name. Every dynamic target renders the
  call through its host runtime; the static targets refuse the call
  or the helper's effects.
* `grouped_marginalization` integrates a per-item class out of a
  Normal mixture whose rows fibre into the items. Every target that
  can add a per-group log-density to its joint scores the block;
  BUGS cannot and refuses it.

Each admitting target's log-density at a clamped point must agree with
the reference machine's up to a constant, at every point of a set that
varies every site, so a target that scores a different measure (a row
reduced on its own, a constructor's branch dropped, a helper's draw
left out) is caught by the spread of the offsets rather than by one
coincidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import pathlib

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program
from quivers.transpile import UnsupportedConstruct, transpile
from tests.test_qiec_fixture_corpus import (
    _C_DATA,
    _C_POINTS,
    _D_DATA,
    _D_POINTS,
    _c_joint,
    _d_joint,
    _sites,
    _source,
)
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

#: The targets whose emission runs the fixture through a host runtime.
_DYNAMIC = ("pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl")

type Host = float | int | list[float] | list[int]
"""A number or a list of numbers, as a point carries it."""

type Oracle = Callable[[Mapping[str, Host]], float]
"""An independent computation of a fixture's joint at a point."""

#: Each fixture's program.
_PROGRAMS: dict[str, str] = {"C": "hierarchy", "D": "mixture"}

#: Each fixture's data arguments.
_DATA: dict[str, dict[str, list[float] | list[int]]] = {"C": _C_DATA, "D": _D_DATA}

#: Each fixture's clamped points, every site set at each.
_POINTS: dict[str, list[dict[str, Host]]] = {"C": _C_POINTS, "D": _D_POINTS}

#: Each fixture's independent joint.
_ORACLES: dict[str, Oracle] = {"C": _c_joint, "D": _d_joint}

#: The image-backed targets that score each fixture.
_BACKENDS: dict[str, tuple[str, ...]] = {
    "C": _DYNAMIC,
    "D": (*_DYNAMIC, "stan", "jags"),
}

#: The shape of every name a point carries, for the probes that
#: rebuild arrays from flat lists.
_SHAPES: dict[str, dict[str, list[int]]] = {
    "C": {
        "mu": [],
        "tau": [],
        "theta": [3],
        "predicted": [],
        "idx": [6],
        "y": [6],
    },
    "D": {"probs": [2], "mu": [2], "idx": [6], "y": [6]},
}

#: The dtype of every name a point carries; `idx` is an index.
_DTYPES: dict[str, dict[str, str]] = {
    "C": {
        "mu": "float",
        "tau": "float",
        "theta": "float",
        "predicted": "float",
        "idx": "int",
        "y": "float",
    },
    "D": {"probs": "float", "mu": "float", "idx": "int", "y": "float"},
}

#: The kind each static target refuses a fixture under.
_REFUSED: dict[str, dict[str, str]] = {
    "C": {
        "stan": "qiec:capability:case:account",
        "bugs": "call:graph:account",
        "jags": "call:graph:account",
    },
    "D": {"bugs": "marginalize:grouped-index-map"},
}


def _points(letter: str) -> list[_protocol.Point]:
    """The fixture's clamped points.

    Parameters
    ----------
    letter : str
        The fixture's letter.

    Returns
    -------
    list[_protocol.Point]
        One point per entry of the fixture's point set.
    """
    return [
        _protocol.Point(params=dict(point), data=dict(_DATA[letter]))
        for point in _POINTS[letter]
    ]


def _reference(letter: str, points: list[_protocol.Point]) -> list[float]:
    """The reference machine's joint at each point, every site replayed.

    Parameters
    ----------
    letter : str
        The fixture's letter.
    points : list[_protocol.Point]
        The points.

    Returns
    -------
    list[float]
        The log joint at each point, checked against the fixture's
        independent oracle.
    """
    module = lower_qvr_to_qiec(parse(_source(letter)), file_path=f"{letter}.qvr")
    densities: list[float] = []
    for point in points:
        result = run_program(
            module,
            _PROGRAMS[letter],
            data={name: tuple(value) for name, value in _DATA[letter].items()},
            sites=_sites(dict(point.params)),
        )
        assert float(result.log_joint) == pytest.approx(
            _ORACLES[letter](dict(point.params)), rel=1e-6
        )
        densities.append(float(result.log_joint))
    return densities


@pytest.mark.parametrize(
    ("letter", "backend"),
    [
        (letter, backend)
        for letter, backends in _BACKENDS.items()
        for backend in backends
    ],
)
def test_fixture_scores_the_reference_joint(
    letter: str, backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    points = _points(letter)
    source = transpile(parse(_source(letter)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": pt.params, "data": pt.data} for pt in points],
        scratch=scratch / letter / backend,
        shapes=_SHAPES[letter],
        dtypes=_DTYPES[letter],
    )
    _equivalence.assert_log_density_match(
        _reference(letter, points),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=len(_DATA[letter]["y"])),
        context=f"{backend}@{letter}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize("letter", sorted(_PROGRAMS))
def test_church_scores_the_reference_joint(letter: str, scratch: pathlib.Path) -> None:
    points = _points(letter)
    source = transpile(parse(_source(letter)), target="church")
    result = ChurchProbe().evaluate(
        source, letter, points, scratch=scratch / letter / "church"
    )
    _equivalence.assert_log_density_match(
        _reference(letter, points),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=len(_DATA[letter]["y"])),
        context=f"church@{letter}",
    )


@pytest.mark.parametrize(
    ("letter", "backend"),
    [(letter, backend) for letter, refused in _REFUSED.items() for backend in refused],
)
def test_static_targets_refuse_under_their_kind(letter: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(letter)), target=backend)
    expected = _REFUSED[letter][backend]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )
