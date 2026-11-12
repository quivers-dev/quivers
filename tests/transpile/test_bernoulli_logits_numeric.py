"""A Bernoulli written with logits scores one joint on every target.

Three fixtures write a Bernoulli through its logits: an observation at one
sampled log-odds every row shares, a logistic regression whose linear
predictor is one log-odds per row, and a latent outcome drawn at a
sampled log-odds. Each is evaluated at log-odds from -30 to 30, where the
probability rounds to within an ulp of zero or one, so a target that
scored through the probability would lose the complement's digits. Each
is scored on the compiled runtime and the reference machine, both pinned
to `torch.distributions.Bernoulli(logits=...)` in double precision, and on
every target. BUGS, whose `dbern` takes a probability and which has no
data block for a zeros-trick carrier, refuses the fixtures under
`family:Bernoulli:logits` rather than score them through the probability.
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
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a list of numbers."""

type Oracle = Callable[[Mapping[str, Host]], torch.Tensor]
"""An independent computation of a fixture's joint, in double precision."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_SHARED = "steps/observe_bernoulli_logits"
_ROWS = "steps/observe_bernoulli_logits_rows"
_DRAWN = "steps/sample_bernoulli_logits"


def _t(value: Host) -> torch.Tensor:
    """A value as a float64 tensor.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The value in double precision.
    """
    return torch.as_tensor(value, dtype=torch.float64)


def _shared(v: Mapping[str, Host]) -> torch.Tensor:
    eta = _t(v["eta"])
    return (
        D.Normal(_t(0.0), _t(2.0)).log_prob(eta)
        + D.Bernoulli(logits=eta).log_prob(_t(v["y"])).sum()
    )


def _rows(v: Mapping[str, Host]) -> torch.Tensor:
    a, b = _t(v["a"]), _t(v["b"])
    prior = D.Normal(_t(0.0), _t(5.0))
    eta = a + b * _t(v["x"])
    return (
        prior.log_prob(a)
        + prior.log_prob(b)
        + D.Bernoulli(logits=eta).log_prob(_t(v["y"])).sum()
    )


def _drawn(v: Mapping[str, Host]) -> torch.Tensor:
    eta, z = _t(v["eta"]), _t(v["z"])
    return (
        D.Normal(_t(0.0), _t(10.0)).log_prob(eta)
        + D.Bernoulli(logits=eta).log_prob(z).sum()
        + D.Normal(2.0 * z - 1.0, _t(1.0)).log_prob(_t(v["y"])).sum()
    )


#: Each case: its fixture, its program, its data, its points, its oracle.
_CASES: dict[str, tuple[str, str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "shared": (
        _SHARED,
        "bernoulli_logits",
        {"y": [1, 0, 1, 1]},
        [
            {"eta": -30.0},
            {"eta": -1.3},
            {"eta": 0.4},
            {"eta": 7.5},
            {"eta": 15.0},
            {"eta": 30.0},
        ],
        _shared,
    ),
    "rows": (
        _ROWS,
        "logit",
        {"x": [-3.0, -1.0, 0.0, 1.0, 3.0], "y": [0, 0, 1, 1, 0]},
        [
            {"a": 0.5, "b": 5.0},
            {"a": -1.0, "b": 1.25},
            {"a": 2.0, "b": -4.0},
            {"a": 0.0, "b": 10.0},
            {"a": 0.0, "b": -10.0},
        ],
        _rows,
    ),
    "drawn": (
        _DRAWN,
        "draw_logits",
        {"y": [0.8, -1.1, 1.3]},
        [
            {"eta": -30.0, "z": [0, 0, 1]},
            {"eta": 30.0, "z": [1, 0, 1]},
            {"eta": 0.7, "z": [1, 1, 0]},
            {"eta": -12.0, "z": [0, 1, 0]},
        ],
        _drawn,
    ),
}

#: The names each fixture reads as integer outcomes.
_INTEGERS: dict[str, frozenset[str]] = {
    _SHARED: frozenset({"y"}),
    _ROWS: frozenset({"y"}),
    _DRAWN: frozenset({"z"}),
}

#: BUGS's `dbern` takes a probability, and BUGS has no data block for
#: the zeros-trick carrier the mass written from the logits needs.
_REFUSED: dict[tuple[str, str], str] = {
    (fixture, "bugs"): "family:Bernoulli:logits" for fixture in (_SHARED, _ROWS, _DRAWN)
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
    """A host value as the compiled runtime reads it.

    Parameters
    ----------
    fixture : str
        The fixture, which decides which names are integers.
    name : str
        The value's name, which decides integer against real.
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    dtype = torch.long if name in _INTEGERS[fixture] else torch.float32
    return torch.as_tensor(value, dtype=dtype)


def _nested(
    fixture: str, name: str, value: Host
) -> float | int | bool | tuple[float | bool, ...]:
    """A host value as the reference machine reads it.

    Parameters
    ----------
    fixture : str
        The fixture, which decides which names are outcomes.
    name : str
        The value's name; an observed Bernoulli outcome is a Boolean.
    value : Host
        The value.

    Returns
    -------
    float | int | bool | tuple[float | bool, ...]
        A number, or a tuple of numbers or Booleans.
    """
    if isinstance(value, list):
        return tuple(
            bool(item) if name in _INTEGERS[fixture] else item for item in value
        )
    return value


def _reference(case: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    list[float]
        The reference machine's joint at each point, equal to the compiled
        runtime's and to the oracle.
    """
    fixture, program_name, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    densities: list[float] = []
    for point in points:
        merged = {**data, **point}
        expected = float(oracle(merged))
        values = {name: _tensor(fixture, name, value) for name, value in merged.items()}
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=1e-5)
        run = program.run(
            program_name,
            data={name: _nested(fixture, name, value) for name, value in data.items()},
            sites={
                name: _nested(fixture, name, value) for name, value in point.items()
            },
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(run.log_joint)
    return densities


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(case: str) -> None:
    assert len(_reference(case)) == len(_CASES[case][3])


def test_a_log_odds_read_as_a_probability_scores_a_different_joint() -> None:
    # The oracle separates the logits from the probability they would be
    # misread as: at every point a target that read `eta` as `probs` would
    # score a joint a hundred probe tolerances away (or none at all, `eta`
    # outside [0, 1]).
    tolerance = _equivalence.adaptive_atol(n_obs=4)
    _, _, data, points, oracle = _CASES["shared"]
    for point in points:
        eta = _t(point["eta"])
        misread = (
            D.Normal(_t(0.0), _t(2.0)).log_prob(eta)
            + D.Bernoulli(probs=eta.clamp(1e-6, 1 - 1e-6)).log_prob(_t(data["y"])).sum()
        )
        assert abs(float(oracle({**data, **point})) - float(misread)) > 100 * tolerance


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
    fixture, _, data, points, _ = _CASES[case]
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
        points=[{"params": dict(point), "data": dict(data)} for point in points],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={
            name: "int" if name in _INTEGERS[fixture] else "float" for name in shapes
        },
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=math.prod(shapes["y"])),
        context=f"{backend}@{case}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_church_scores_the_reference_joint(case: str, scratch: pathlib.Path) -> None:
    fixture, _, data, points, _ = _CASES[case]
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


def test_the_probability_route_loses_the_complement_at_large_log_odds() -> None:
    # Scoring an outcome of zero at log-odds 30 through the probability,
    # `log(1 - sigmoid(30))` in double precision, misses the exact
    # `-softplus(30)` by more than the probe tolerance, so the cells above
    # reject a target that spells the family through its probability.
    eta = _t(30.0)
    exact = float(D.Bernoulli(logits=eta).log_prob(_t(0.0)))
    through_probability = float(torch.log1p(-torch.sigmoid(eta)))
    assert abs(exact - through_probability) > _equivalence.adaptive_atol(n_obs=1)


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )


def test_every_target_renders_or_refuses_every_fixture() -> None:
    for fixture, _, _, _, _ in _CASES.values():
        for target in available_targets():
            if (fixture, target) in _REFUSED:
                continue
            assert transpile(parse(_source(fixture)), target=target)
