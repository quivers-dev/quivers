"""A vector filling a rank-0 family parameter is read entry by entry on every target.

A scalar family whose parameter is one vector, such as ``Normal(loc=z,
scale=0.5)`` with ``z`` drawn over ``Feat``, is one draw per entry of
``z``, each at its own entry. A target that visits the draw one position
at a time must read ``z`` at that position: the whole vector in a scalar
slot scores each entry of the observation against every entry of ``z``
(Stan sums the vectorized density), which is a different joint. Each
fixture here places such a vector in a different position (an observe's
location read off a let or a latent, an observe replicated with
``[over=...]``, a latent's location, a scale and a rate, a plated
observe) and one fixture keeps a vector in a slot that reads a whole
event, which no target may index. Each is scored on the compiled runtime
and the reference machine, both pinned to an independent
`torch.distributions` joint, and on every target that renders it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
"""A number or a flat list of numbers."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_LET = "steps/observe_vector_location_let"
_LATENT = "steps/observe_vector_location_latent"
_OVER = "steps/observe_over_vector_location"
_SAMPLE = "steps/sample_vector_location"
_SCALE_RATE = "steps/observe_vector_scale_rate"
_PLATED = "steps/observe_plated_vector_location"
_SHARED = "steps/observe_shared_probs_vector"

_HALF = 0.5


def _standard_prior(a: torch.Tensor) -> torch.Tensor:
    """The standard normal prior every fixture draws ``a`` from.

    Parameters
    ----------
    a : torch.Tensor
        The drawn vector.

    Returns
    -------
    torch.Tensor
        Its log density.
    """
    return D.Normal(0.0, 1.0).log_prob(a).sum()


def _let(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y = v["a"], v["y"]
    return _standard_prior(a) + D.Normal(torch.tanh(a), _HALF).log_prob(y).sum()


def _latent(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y = v["a"], v["y"]
    return _standard_prior(a) + D.Normal(a, _HALF).log_prob(y).sum()


def _sample(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, b, y = v["a"], v["b"], v["y"]
    return (
        _standard_prior(a)
        + D.Normal(torch.tanh(a), _HALF).log_prob(b).sum()
        + D.Normal(b, _HALF).log_prob(y).sum()
    )


def _scale_rate(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y, c = v["a"], v["y"], v["c"]
    s = torch.exp(a)
    return (
        _standard_prior(a)
        + D.Normal(1.0, s).log_prob(y).sum()
        + D.Poisson(s).log_prob(c).sum()
    )


def _shared(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    w, y = v["w"], v["y"].long()
    prior = D.Dirichlet(torch.ones(3, dtype=torch.float64)).log_prob(w)
    return prior + D.Categorical(probs=w).log_prob(y).sum()


#: Each case: its fixture, its data, its clamped points, and its oracle.
_CASES: dict[str, tuple[str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "let-location": (
        _LET,
        {"y": [0.9, -1.2, 0.3]},
        [{"a": [0.4, -0.8, 1.5]}, {"a": [-1.1, 0.2, 0.6]}, {"a": [2.0, -2.0, 0.1]}],
        _let,
    ),
    "latent-location": (
        _LATENT,
        {"y": [0.9, -1.2, 0.3]},
        [{"a": [0.4, -0.8, 1.5]}, {"a": [-1.1, 0.2, 0.6]}, {"a": [2.0, -2.0, 0.1]}],
        _latent,
    ),
    "over-location": (
        _OVER,
        {"y": [0.9, -1.2, 0.3]},
        [{"a": [0.4, -0.8, 1.5]}, {"a": [-1.1, 0.2, 0.6]}, {"a": [2.0, -2.0, 0.1]}],
        _let,
    ),
    "latent-site-location": (
        _SAMPLE,
        {"y": [0.9, -1.2, 0.3]},
        [
            {"a": [0.4, -0.8, 1.5], "b": [0.5, -0.4, 1.0]},
            {"a": [-1.1, 0.2, 0.6], "b": [-0.7, 0.9, 0.2]},
            {"a": [2.0, -2.0, 0.1], "b": [1.3, -1.6, -0.3]},
        ],
        _sample,
    ),
    "scale-and-rate": (
        _SCALE_RATE,
        {"y": [0.9, -1.2, 2.3], "c": [1, 0, 4]},
        [{"a": [0.4, -0.8, 1.5]}, {"a": [-1.1, 0.2, 0.6]}, {"a": [1.2, -0.5, 0.1]}],
        _scale_rate,
    ),
    "plated-location": (
        _PLATED,
        {"y": [0.9, -1.2, 0.3, 0.0]},
        [
            {"a": [0.4, -0.8, 1.5, -0.2]},
            {"a": [-1.1, 0.2, 0.6, 1.4]},
            {"a": [2.0, -2.0, 0.1, 0.7]},
        ],
        _let,
    ),
    "shared-probs": (
        _SHARED,
        {"y": [0, 2, 2]},
        [
            {"w": [0.5, 0.25, 0.25]},
            {"w": [0.125, 0.125, 0.75]},
            {"w": [0.25, 0.5, 0.25]},
        ],
        _shared,
    ),
}

#: The program every fixture exports.
_PROGRAM = "p"

#: The observed sites each fixture scores, whose entries size the tolerance.
_OBSERVED: dict[str, tuple[str, ...]] = {
    _LET: ("y",),
    _LATENT: ("y",),
    _OVER: ("y",),
    _SAMPLE: ("y",),
    _SCALE_RATE: ("y", "c"),
    _PLATED: ("y",),
    _SHARED: ("y",),
}

#: The data names each fixture reads as integers.
_INTEGER_DATA: dict[str, frozenset[str]] = {
    _SCALE_RATE: frozenset({"c"}),
    _SHARED: frozenset({"y"}),
}

#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    # BUGS has no data block to shift an observed zero-based class code
    # to its one-based `dcat`.
    (_SHARED, "bugs"): "observe:class-code:y",
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
    integers = _INTEGER_DATA.get(fixture, frozenset())
    dtype = torch.long if name in integers else torch.float32
    return torch.as_tensor(value, dtype=dtype)


type Nested = float | int | tuple["Nested", ...]
"""A number, or nested tuples of numbers."""


def _nested(value: torch.Tensor) -> Nested:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    Nested
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
        The compiled runtime's log joint at each point, each equal to the
        reference machine's and to the case's oracle.
    """
    fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    densities: list[float] = []
    for point in points:
        merged = {**data, **point}
        values = {name: _tensor(fixture, name, value) for name, value in merged.items()}
        expected = float(oracle({k: v.double() for k, v in values.items()}))
        joint = float(
            program.morphism.log_joint(torch.zeros(1, dtype=torch.long), values).sum()
        )
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            _PROGRAM,
            data={name: _nested(values[name]) for name in data},
            sites={name: _nested(values[name]) for name in point},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(joint)
    return densities


def _n_obs(fixture: str, data: Mapping[str, Host]) -> int:
    """The number of scored observation entries of a fixture.

    Parameters
    ----------
    fixture : str
        The fixture.
    data : Mapping[str, Host]
        Its data.

    Returns
    -------
    int
        The entries of every observed site together.
    """
    return sum(torch.as_tensor(data[name]).numel() for name in _OBSERVED[fixture])


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(case: str) -> None:
    assert len(_reference(case)) == len(_CASES[case][2])


def test_the_oracle_separates_entrywise_from_whole_vector_scoring() -> None:
    # Scoring each observed entry against every entry of the location,
    # the density a whole vector in a looped scalar slot computes, is a
    # joint the oracle tells apart at every point.
    _, data, points, oracle = _CASES["let-location"]
    for point in points:
        values = {
            name: torch.as_tensor(value, dtype=torch.float64)
            for name, value in {**data, **point}.items()
        }
        a, y = values["a"], values["y"]
        whole = (
            _standard_prior(a)
            + D.Normal(torch.tanh(a)[None, :], _HALF).log_prob(y[:, None]).sum()
        )
        assert abs(float(oracle(values)) - float(whole)) > 1.0


def test_the_shared_oracle_separates_one_event_from_one_entry_per_row() -> None:
    # Scoring each row at one entry of `w`, the density an index into the
    # whole-event slot computes, is a joint the oracle tells apart.
    _, data, points, oracle = _CASES["shared-probs"]
    values = {
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in {**data, **points[1]}.items()
    }
    w = values["w"]
    per_row = (
        D.Dirichlet(torch.ones(3, dtype=torch.float64)).log_prob(w) + torch.log(w).sum()
    )
    assert abs(float(oracle(values)) - float(per_row)) > 1.0


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
        name: list(torch.as_tensor(value).shape)
        for name, value in {**data, **points[0]}.items()
    }
    integers = _INTEGER_DATA.get(fixture, frozenset())
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": dict(point), "data": dict(data)} for point in points],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={name: "int" if name in integers else "float" for name in shapes},
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture, data)),
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
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture, data)),
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
