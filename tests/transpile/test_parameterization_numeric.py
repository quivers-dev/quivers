"""Alternative parameterizations score one joint on every target.

A multivariate normal or a Wishart written with a covariance or a
precision matrix, and a Bernoulli, continuous Bernoulli, Binomial,
Geometric, or negative binomial written with logits, are each scored on the compiled runtime and
the reference machine, both pinned to `torch.distributions` built with the
same parameterization in double precision, and on every target that
renders the fixture: natively where the target's library has the
parameterization, with a literal precision matrix inverted into the
covariance where it has not. Every other target refuses the fixture under
a stated kind, and every alternative the surface accepts has a fixture
here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.family_schemas import family_parameterizations
from quivers.dsl.parser import parse
from quivers.qiec.families import FAMILIES
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number, a vector, or a matrix."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint, in double precision."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_COVARIANCE = [[1.0, 0.5], [0.5, 1.25]]
_PRECISION = [[1.25, -0.5], [-0.5, 1.0]]


def _t(value: torch.Tensor | Host) -> torch.Tensor:
    """A value as a float64 tensor.

    Parameters
    ----------
    value : torch.Tensor | Host
        The value.

    Returns
    -------
    torch.Tensor
        The value in double precision.
    """
    return torch.as_tensor(value, dtype=torch.float64)


def _eta(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return D.Normal(_t(0.0), _t(2.0)).log_prob(_t(v["eta"]))


def _mean(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return D.Normal(_t(0.0), _t(1.0)).log_prob(_t(v["m"])).sum()


_ETAS: list[dict[str, Host]] = [{"eta": -1.3}, {"eta": 0.4}, {"eta": 2.2}]
_MEANS: list[dict[str, Host]] = [
    {"m": [0.5, -1.0], "v": [0.3, 0.4]},
    {"m": [-0.2, 1.5], "v": [-1.0, 2.5]},
    {"m": [1.0, 0.25], "v": [1.2, -0.7]},
]
_SIGMAS: list[dict[str, Host]] = [
    {"sigma": [[2.0, 0.3], [0.3, 1.0]]},
    {"sigma": [[1.0, 0.5], [0.5, 1.25]]},
    {"sigma": [[4.0, -1.0], [-1.0, 3.0]]},
]

#: Each case: the family and parameter it covers, its fixture, its data,
#: its points, and its oracle.
_CASES: dict[
    str, tuple[str, str, str, dict[str, Host], list[dict[str, Host]], Oracle]
] = {
    "mvn-covariance": (
        "MultivariateNormal",
        "covariance_matrix",
        "steps/sample_mvn_covariance",
        {},
        _MEANS,
        lambda v: (
            _mean(v)
            + D.MultivariateNormal(
                _t(v["m"]), covariance_matrix=_t(_COVARIANCE)
            ).log_prob(_t(v["v"]))
        ),
    ),
    "mvn-precision": (
        "MultivariateNormal",
        "precision_matrix",
        "steps/sample_mvn_precision",
        {},
        _MEANS,
        lambda v: (
            _mean(v)
            + D.MultivariateNormal(
                _t(v["m"]), precision_matrix=_t(_PRECISION)
            ).log_prob(_t(v["v"]))
        ),
    ),
    "wishart-covariance": (
        "Wishart",
        "covariance_matrix",
        "steps/sample_wishart_covariance",
        {},
        _SIGMAS,
        lambda v: D.Wishart(_t(4.0), covariance_matrix=_t(_COVARIANCE)).log_prob(
            _t(v["sigma"])
        ),
    ),
    "wishart-precision": (
        "Wishart",
        "precision_matrix",
        "steps/sample_wishart_precision",
        {},
        _SIGMAS,
        lambda v: D.Wishart(_t(4.0), precision_matrix=_t(_PRECISION)).log_prob(
            _t(v["sigma"])
        ),
    ),
    "bernoulli-logits": (
        "Bernoulli",
        "logits",
        "steps/observe_bernoulli_logits",
        {"y": [1, 0, 1, 1]},
        _ETAS,
        lambda v: _eta(v) + D.Bernoulli(logits=_t(v["eta"])).log_prob(_t(v["y"])).sum(),
    ),
    "binomial-logits": (
        "Binomial",
        "logits",
        "steps/observe_binomial_logits",
        {"y": [3, 7, 5, 9]},
        _ETAS,
        lambda v: (
            _eta(v) + D.Binomial(10, logits=_t(v["eta"])).log_prob(_t(v["y"])).sum()
        ),
    ),
    "continuous-bernoulli-logits": (
        "ContinuousBernoulli",
        "logits",
        "steps/observe_continuous_bernoulli_logits",
        {"u": [0.1, 0.45, 0.8, 0.97]},
        _ETAS,
        lambda v: (
            _eta(v)
            + D.ContinuousBernoulli(logits=_t(v["eta"])).log_prob(_t(v["u"])).sum()
        ),
    ),
    "geometric-logits": (
        "Geometric",
        "logits",
        "steps/observe_geometric_logits",
        {"y": [0, 2, 5, 1]},
        _ETAS,
        lambda v: _eta(v) + D.Geometric(logits=_t(v["eta"])).log_prob(_t(v["y"])).sum(),
    ),
    "negative-binomial-logits": (
        "NegativeBinomial",
        "logits",
        "steps/observe_negative_binomial_logits",
        {"y": [0, 4, 2, 7]},
        _ETAS,
        lambda v: (
            _eta(v)
            + D.NegativeBinomial(_t(3.0), logits=_t(v["eta"]))
            .log_prob(_t(v["y"]))
            .sum()
        ),
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


def _logits_refused(fixture: str, family: str, targets: tuple[str, ...]) -> dict:
    """The refusal of logits on each target that has no logit spelling.

    Parameters
    ----------
    fixture : str
        The fixture.
    family : str
        The family.
    targets : tuple[str, ...]
        The targets that refuse.

    Returns
    -------
    dict
        The refusals keyed by fixture and target.
    """
    return {(fixture, target): f"family:{family}:logits" for target in targets}


#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    ("steps/sample_mvn_covariance", "bugs"): "arg:matrix-literal",
    ("steps/sample_mvn_covariance", "church"): "arg:matrix-literal",
    ("steps/sample_mvn_covariance", "jags"): (
        "family:MultivariateNormal:non-ref-covariance"
    ),
    ("steps/sample_mvn_precision", "bugs"): "arg:matrix-literal",
    ("steps/sample_mvn_precision", "church"): "arg:matrix-literal",
    ("steps/sample_mvn_precision", "jags"): "arg:matrix-literal",
    **{
        (fixture, target): kind
        for fixture in (
            "steps/sample_wishart_covariance",
            "steps/sample_wishart_precision",
        )
        for target, kind in (
            ("bugs", "arg:matrix-literal"),
            ("jags", "arg:matrix-literal"),
            ("church", "family:Wishart"),
            ("edward2", "family:Wishart"),
            ("gen", "family:Wishart"),
            ("webppl", "family:no-webppl-target:Wishart"),
        )
    },
    **_logits_refused(
        "steps/observe_bernoulli_logits",
        "Bernoulli",
        ("bugs", "church", "gen", "jags", "webppl"),
    ),
    **_logits_refused(
        "steps/observe_binomial_logits",
        "Binomial",
        ("bugs", "church", "gen", "jags", "webppl"),
    ),
    **_logits_refused(
        "steps/observe_continuous_bernoulli_logits",
        "ContinuousBernoulli",
        _EVERY_TARGET,
    ),
    **_logits_refused(
        "steps/observe_geometric_logits",
        "Geometric",
        ("bugs", "church", "gen", "jags", "pymc", "stan", "turing", "webppl"),
    ),
    **_logits_refused(
        "steps/observe_negative_binomial_logits",
        "NegativeBinomial",
        (
            "bugs",
            "church",
            "gen",
            "jags",
            "numpyro",
            "pymc",
            "stan",
            "turing",
            "webppl",
        ),
    ),
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
    family, _, fixture, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    densities: list[float] = []
    for point in points:
        values = {
            name: _tensor(name, value) for name, value in {**data, **point}.items()
        }
        expected = float(oracle({**data, **point}))
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={
                name: _nested(value, family == "Bernoulli")
                for name, value in data.items()
            },
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
    assert len(_reference(case)) == len(_CASES[case][4])


@pytest.mark.parametrize(
    ("case", "backend"),
    [
        (case, backend)
        for case in sorted(_CASES)
        for backend in _TARGETS
        if (_CASES[case][2], backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    case: str, backend: str, scratch: pathlib.Path
) -> None:
    _, _, fixture, data, points, _ = _CASES[case]
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
    for _, _, fixture, _, _, _ in _CASES.values():
        for target in _EVERY_TARGET:
            if (fixture, target) not in _REFUSED:
                assert target in _TARGETS, (fixture, target)
                transpile(parse(_source(fixture)), target=target)


def test_every_surface_alternative_has_a_fixture() -> None:
    covered = {(family, parameter) for family, parameter, *_ in _CASES.values()}
    for record in FAMILIES.values():
        schemas = family_parameterizations(record.name, record.parameter_names)
        if len(schemas) < 2 or record.compositional or record.name == "Categorical":
            continue
        for schema in schemas[1:]:
            for parameter in set(schema) - set(schemas[0]):
                assert (record.name, parameter) in covered, (record.name, parameter)
