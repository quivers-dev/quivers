"""Host data filling a vector-valued family parameter score one joint.

A free name that fills a family parameter of a plated draw directly holds
one value per row: OrderedLogistic cutpoints, Categorical logits, and
Categorical probabilities each take one vector per row of the observed
plate, and a datum without the row axis is one vector every row shares.
Each fixture is scored with per-row and with shared data on the compiled
runtime and the reference machine, both pinned to an independent
`torch.distributions` joint, and on every target that renders it. A shared
datum reaches a target spread across the plate by
[`broadcast_shared_data`][quivers.qiec.broadcast_shared_data], since every
target declares the per-row shape. A target with no spelling for a
fixture refuses it under a stated kind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.continuous._ordered import OrderedLogistic
from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.qiec import broadcast_shared_data
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number, a list of numbers, or a list of rows."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_ORDERED = "steps/observe_vector_parameter_data"
_LOGITS = "steps/observe_logits_data"
_PROBS = "steps/observe_probs_data"


def _ordered(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    eta, y, cuts = v["eta"], v["y"], v["cuts"]
    return D.Normal(0.0, 1.0).log_prob(eta) + (
        OrderedLogistic(eta.expand(y.shape[0]), cuts).log_prob(y).sum()
    )


def _logits(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    bias, choice, scores = v["bias"], v["choice"], v["scores"]
    return (
        D.Normal(0.0, 1.0).log_prob(bias)
        + D.Categorical(logits=scores).log_prob(choice).sum()
    )


def _probs(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    bias, choice, weights = v["bias"], v["choice"], v["weights"]
    return (
        D.Normal(0.0, 1.0).log_prob(bias)
        + D.Categorical(probs=weights).log_prob(choice).sum()
    )


#: Each case: its fixture, its data, its clamped points, its oracle, and
#: the data names that are integers.
_CASES: dict[str, tuple[str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "ordered-per-row": (
        _ORDERED,
        {
            "y": [0, 1, 2, 3, 1, 2],
            "cuts": [[-2.0, 0.0, 2.0]] * 3 + [[-1.0, 0.5, 1.5]] * 3,
        },
        [{"eta": -0.4}, {"eta": 0.3}, {"eta": 1.2}],
        _ordered,
    ),
    "ordered-shared": (
        _ORDERED,
        {"y": [0, 1, 2, 3, 1, 2], "cuts": [-1.5, 0.0, 1.5]},
        [{"eta": -0.4}, {"eta": 0.3}, {"eta": 1.2}],
        _ordered,
    ),
    "logits-per-row": (
        _LOGITS,
        {
            "choice": [0, 2, 1, 2, 0],
            "scores": [
                [0.2, -0.4, 1.0],
                [1.5, 0.0, -1.0],
                [0.0, 0.3, 0.3],
                [-0.7, 0.9, 0.1],
                [2.0, -1.0, 0.5],
            ],
        },
        [{"bias": -0.4}, {"bias": 0.3}, {"bias": 1.2}],
        _logits,
    ),
    "logits-shared": (
        _LOGITS,
        {"choice": [0, 2, 1, 2, 0], "scores": [0.2, -0.4, 1.0]},
        [{"bias": -0.4}, {"bias": 0.3}, {"bias": 1.2}],
        _logits,
    ),
    "probs-per-row": (
        _PROBS,
        {
            "choice": [0, 2, 1, 2, 0],
            "weights": [
                [0.2, 0.3, 0.5],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.3, 0.4],
                [0.5, 0.25, 0.25],
            ],
        },
        [{"bias": -0.4}, {"bias": 0.3}, {"bias": 1.2}],
        _probs,
    ),
    "probs-shared": (
        _PROBS,
        {"choice": [0, 2, 1, 2, 0], "weights": [0.2, 0.3, 0.5]},
        [{"bias": -0.4}, {"bias": 0.3}, {"bias": 1.2}],
        _probs,
    ),
}

_INTEGERS = frozenset({"y", "choice"})

#: The refusals each target states for a fixture it cannot render: a
#: target whose library cannot read one parameter vector per row refuses
#: the per-row datum rather than score a different model.
_REFUSED: dict[tuple[str, str], str] = {
    **{
        (fixture, target): "data:per-row-vector-parameter"
        for fixture in (_ORDERED, _PROBS)
        for target in (
            "bugs",
            "church",
            "gen",
            "jags",
            "pymc",
            "stan",
            "turing",
            "webppl",
        )
    },
    (_ORDERED, "edward2"): "family:OrderedLogistic",
    (_LOGITS, "pymc"): "data:per-row-vector-parameter",
    (_LOGITS, "stan"): "data:per-row-vector-parameter",
    **{
        (_LOGITS, target): "family:Categorical:logits"
        for target in ("bugs", "church", "gen", "jags", "turing", "webppl")
    },
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


def _tensor(name: str, value: object) -> torch.Tensor:
    """A host value as a tensor of its dtype.

    Parameters
    ----------
    name : str
        The value's name, which decides integer against real.
    value : object
        The value: a number or nested lists of numbers.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    dtype = torch.long if name in _INTEGERS else torch.float32
    return torch.as_tensor(value, dtype=dtype)


def _nested(value: torch.Tensor) -> float | int | tuple[object, ...]:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    float | int | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return value.item()
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
        The compiled runtime's log joint at each point, each equal to
        the reference machine's and to the case's oracle.
    """
    fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    densities: list[float] = []
    for point in points:
        values = {
            name: _tensor(name, value) for name, value in {**data, **point}.items()
        }
        expected = float(oracle(values))
        joint = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={name: _nested(values[name]) for name in data},
            sites={name: _nested(values[name]) for name in point},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-5)
        densities.append(joint)
    return densities


def _target_data(case: str) -> dict[str, Host]:
    """The case's data as a target reads it, shared values spread.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    dict[str, Host]
        Each datum at its per-row shape.
    """
    fixture, data, _, _ = _CASES[case]
    module = loads(_source(fixture)).qiec
    assert module is not None
    (entry,) = module.entries
    spread = broadcast_shared_data(entry, data)
    return {
        name: _tensor(name, _flat_rows(value)).reshape(-1).tolist()
        for name, value in spread.items()
    }


def _shapes(case: str) -> dict[str, list[int]]:
    """The per-row shape of every name a point carries.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    dict[str, list[int]]
        Each name's shape as a target reads it.
    """
    fixture, data, points, _ = _CASES[case]
    module = loads(_source(fixture)).qiec
    assert module is not None
    (entry,) = module.entries
    spread = broadcast_shared_data(entry, data)
    shapes = {
        name: list(torch.as_tensor(_flat_rows(value)).shape)
        for name, value in spread.items()
    }
    shapes.update({name: [] for name in points[0]})
    return shapes


def _flat_rows(value: object) -> object:
    """A host value with tuples of rows as lists of rows.

    Parameters
    ----------
    value : object
        A number, a list, or a tuple of rows.

    Returns
    -------
    object
        The value as nested lists.
    """
    if isinstance(value, tuple):
        return [_flat_rows(item) for item in value]
    return value


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
    fixture, _, points, _ = _CASES[case]
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    data = _target_data(case)
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    shapes = _shapes(case)
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": dict(point), "data": data} for point in points],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={name: "int" if name in _INTEGERS else "float" for name in shapes},
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(
            n_obs=math.prod(shapes.get("y", shapes.get("choice", [1])))
        ),
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


def test_a_datum_of_neither_admissible_shape_is_refused() -> None:
    program = loads(_source(_ORDERED))
    with pytest.raises(
        ValueError,
        match=(
            r"host datum 'cuts' has shape \(2, 3\), but the checked program reads "
            r"it at type Tensor\[Real\]\(\[6, cuts_extent\]\), of shape "
            r"\(6, cuts_extent\) or \(cuts_extent,\), the second a value every "
            r"row of the plate shares"
        ),
    ):
        program.morphism.log_joint(
            torch.zeros(1, 1),
            {
                "eta": torch.tensor(0.3),
                "y": torch.tensor([0, 1, 2, 3, 1, 2]),
                "cuts": torch.zeros(2, 3),
            },
        )
