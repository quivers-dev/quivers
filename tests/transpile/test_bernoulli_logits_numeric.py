"""A Bernoulli written with logits scores one joint on every target.

Two fixtures write a Bernoulli likelihood through its logits: one sampled
log-odds every row shares, and a logistic regression whose linear
predictor is one log-odds per row, evaluated where a row saturates (a
log-odds past 15, whose probability rounds to one in single precision).
Each is scored on the compiled runtime and the reference machine, both
pinned to `torch.distributions.Bernoulli(logits=...)` in double precision,
and on every target whose library has a logit-parameterized Bernoulli. The
targets without one refuse the fixture under `family:Bernoulli:logits`
rather than score the log-odds as a probability.
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
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a list of numbers."""

type Oracle = Callable[[Mapping[str, Host]], torch.Tensor]
"""An independent computation of a fixture's joint, in double precision."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_SHARED = "steps/observe_bernoulli_logits"
_ROWS = "steps/observe_bernoulli_logits_rows"


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


#: Each case: its fixture, its program, its data, its points, its oracle.
_CASES: dict[str, tuple[str, str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "shared": (
        _SHARED,
        "bernoulli_logits",
        {"y": [1, 0, 1, 1]},
        [{"eta": -1.3}, {"eta": 0.4}, {"eta": 2.2}, {"eta": 7.5}],
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
        ],
        _rows,
    ),
}

_INTEGERS = frozenset({"y"})

#: The targets whose library has no logit-parameterized Bernoulli.
_REFUSING: tuple[str, ...] = ("bugs", "church", "gen", "jags", "webppl")

_REFUSED: dict[tuple[str, str], str] = {
    (fixture, target): "family:Bernoulli:logits"
    for fixture in (_SHARED, _ROWS)
    for target in _REFUSING
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
    """A host value as the compiled runtime reads it.

    Parameters
    ----------
    name : str
        The value's name, which decides integer against real.
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    dtype = torch.long if name in _INTEGERS else torch.float32
    return torch.as_tensor(value, dtype=dtype)


def _nested(name: str, value: Host) -> float | int | bool | tuple[float | bool, ...]:
    """A host value as the reference machine reads it.

    Parameters
    ----------
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
        return tuple(bool(item) if name in _INTEGERS else item for item in value)
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
        values = {name: _tensor(name, value) for name, value in merged.items()}
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=1e-5)
        run = program.run(
            program_name,
            data={name: _nested(name, value) for name, value in data.items()},
            sites={name: _nested(name, value) for name, value in point.items()},
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
        dtypes={name: "int" if name in _INTEGERS else "float" for name in shapes},
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=math.prod(shapes["y"])),
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
    for fixture, _, _, _, _ in _CASES.values():
        for target in available_targets():
            if (fixture, target) in _REFUSED:
                continue
            assert transpile(parse(_source(fixture)), target=target)
