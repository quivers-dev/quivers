"""Inline OrderedProbit, InverseWishart, and MatrixNormal score one joint everywhere.

An ordinal response under a probit link, an inverse Wishart drawn as the
inverse of a Wishart with a literal scale factor, and a matrix normal with
distinct row and column covariances are each scored on the compiled
runtime and the reference machine, both pinned to an independent SciPy
density in double precision, and on every target that renders the
fixture. Every other target refuses the fixture under a stated kind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib

import pytest
import scipy.stats
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number, a vector, or a matrix."""

type Oracle = Callable[[Mapping[str, Host]], float]
"""An independent computation of a fixture's joint, in double precision."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_CUTS = [-1.0, 0.0, 1.5]
_FACTOR = [[1.0, 0.0], [0.3, 0.8]]
_LOC = [[0.0, 1.0, 0.0], [0.5, 0.0, -0.5]]
_ROWS = [[1.0, 0.3], [0.3, 2.0]]
_COLS = [[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.7]]


def _ordered_probit(v: Mapping[str, Host]) -> float:
    eta = float(v["eta"])  # type: ignore[arg-type]
    bounds = [-math.inf, *_CUTS, math.inf]
    total = float(scipy.stats.norm.logpdf(eta))
    for category in v["y"]:  # type: ignore[union-attr]
        k = int(category)
        total += math.log(
            scipy.stats.norm.cdf(bounds[k + 1] - eta)
            - scipy.stats.norm.cdf(bounds[k] - eta)
        )
    return total


def _inverse_wishart(v: Mapping[str, Host]) -> float:
    factor = torch.tensor(_FACTOR, dtype=torch.float64)
    scale = torch.linalg.inv(factor @ factor.T).numpy()
    return float(scipy.stats.invwishart(df=5.0, scale=scale).logpdf(v["sigma"]))


def _matrix_normal(v: Mapping[str, Host]) -> float:
    return float(
        scipy.stats.matrix_normal(mean=_LOC, rowcov=_ROWS, colcov=_COLS).logpdf(v["m"])
    )


#: Each case: its fixture, its data, its points, and its oracle.
_CASES: dict[str, tuple[str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "ordered-probit": (
        "steps/observe_ordered_probit",
        {"y": [0, 1, 2, 3, 1, 2]},
        [{"eta": -1.3}, {"eta": 0.4}, {"eta": 2.2}],
        _ordered_probit,
    ),
    "inverse-wishart": (
        "steps/sample_inverse_wishart_factor",
        {},
        [
            {"sigma": [[2.0, 0.3], [0.3, 1.0]]},
            {"sigma": [[0.4, -0.1], [-0.1, 0.9]]},
            {"sigma": [[1.2, 0.6], [0.6, 3.0]]},
        ],
        _inverse_wishart,
    ),
    "matrix-normal": (
        "steps/sample_matrix_normal",
        {},
        [
            {"m": [[0.3, 0.9, -0.2], [0.1, 0.4, -1.0]]},
            {"m": [[-1.0, 2.0, 0.5], [1.5, -0.3, 0.0]]},
            {"m": [[0.0, 1.0, 0.0], [0.5, 0.0, -0.5]]},
        ],
        _matrix_normal,
    ),
}

_INTEGERS = frozenset({"y"})

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
    ("steps/observe_ordered_probit", "bugs"): "arg:list-literal",
    ("steps/observe_ordered_probit", "church"): "family:OrderedProbit",
    ("steps/observe_ordered_probit", "edward2"): "family:OrderedProbit",
    ("steps/observe_ordered_probit", "gen"): "family:OrderedProbit",
    ("steps/observe_ordered_probit", "jags"): "family:no-target-name:OrderedProbit",
    ("steps/observe_ordered_probit", "turing"): "family:OrderedProbit",
    ("steps/observe_ordered_probit", "webppl"): "family:no-webppl-target:OrderedProbit",
    ("steps/sample_inverse_wishart_factor", "bugs"): "arg:matrix-literal",
    ("steps/sample_inverse_wishart_factor", "church"): "family:InverseWishart",
    ("steps/sample_inverse_wishart_factor", "edward2"): "family:InverseWishart",
    ("steps/sample_inverse_wishart_factor", "gen"): "family:InverseWishart",
    ("steps/sample_inverse_wishart_factor", "jags"): (
        "family:no-target-name:InverseWishart"
    ),
    ("steps/sample_inverse_wishart_factor", "pymc"): "family:InverseWishart",
    ("steps/sample_inverse_wishart_factor", "webppl"): (
        "family:no-webppl-target:InverseWishart"
    ),
    ("steps/sample_matrix_normal", "bugs"): "arg:matrix-literal",
    ("steps/sample_matrix_normal", "church"): "arg:matrix-literal",
    ("steps/sample_matrix_normal", "jags"): "family:no-target-name:MatrixNormal",
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


def _tensor(name: str, value: Host) -> torch.Tensor:
    """A value as the compiled runtime takes it.

    Parameters
    ----------
    name : str
        Its name, which decides integer against real.
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    return torch.as_tensor(
        value, dtype=torch.long if name in _INTEGERS else torch.float32
    )


def _nested(value: Host, boolean: bool) -> float | int | bool | tuple[object, ...]:
    """A value as the reference machine's nested tuples.

    Parameters
    ----------
    value : Host
        The value.
    boolean : bool
        Whether its entries are the Booleans a Bernoulli draws.

    Returns
    -------
    float | int | bool | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if isinstance(value, list):
        return tuple(_nested(item, boolean) for item in value)
    return bool(value) if boolean else value


def _reference(case: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    case : str
        The case.

    Returns
    -------
    list[float]
        The reference machine's joint at each point, equal to the
        compiled runtime's and to the oracle.
    """
    fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    densities: list[float] = []
    for point in points:
        values = {
            name: _tensor(name, value) for name, value in {**data, **point}.items()
        }
        expected = oracle({**data, **point})
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={name: _nested(value, False) for name, value in data.items()},
            sites={name: _nested(value, False) for name, value in point.items()},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(run.log_joint)
    return densities


def _flat(value: Host) -> Host:
    """A value flattened in row-major order.

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
    shapes = {
        name: list(torch.as_tensor(value).shape)
        for name, value in {**data, **points[0]}.items()
    }
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {
                "params": {name: _flat(value) for name, value in point.items()},
                "data": {
                    name: [int(item) for item in value]
                    if isinstance(value, list) and name in _INTEGERS
                    else _flat(value)
                    for name, value in data.items()
                },
            }
            for point in points
        ],
        scratch=scratch / case / backend,
        shapes=shapes,
        # A matrix site is typed `float32` so Pyro scores it in the dtype of
        # the model's literal matrix; every other probe reads `float32` as a
        # float.
        dtypes={
            name: "int"
            if name in _INTEGERS
            else "float32"
            if len(shapes[name]) == 2
            else "float"
            for name in shapes
        },
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
    for fixture, _, _, _ in _CASES.values():
        for target in _EVERY_TARGET:
            if (fixture, target) not in _REFUSED:
                assert target in _TARGETS, (fixture, target)
                transpile(parse(_source(fixture)), target=target)
