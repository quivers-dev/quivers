"""A marginalized gate weighted from its logits scores one joint on every target.

The fixture is a zero-inflated Poisson whose gate is
``marginalize z <- Bernoulli(logits=eta)``: each row is a mixture of
a point mass at zero and ``Poisson(r)`` at a per-row rate ``r``, weighted by ``sigmoid(-l)`` and
``sigmoid(l)`` for the row's logit ``l = eta + x``. It is scored at
predictors from -30 to 30, where the
probability ``sigmoid(eta)`` is within an ulp of zero or one and a weight
formed as ``1 - sigmoid(eta)`` loses its digits. The compiled runtime and the
reference machine are pinned to an exact double-precision computation of the
mixture, and every target that renders the fixture is held to the reference
machine's joints, each of which weighs the atoms in log space (or, where the
target scores the mixture on the probability scale, by the logistic function
of each sign of the logit, which no complement cancels). BUGS has no
statement that adds an integrated log-density to the joint and refuses the
block by name.

A continuous Bernoulli drawn or observed with logits is scored by both
execution paths, and refused by name on every target, none of which spells
the family on the logit scale.
"""

from __future__ import annotations

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

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_MARGINAL = "steps/marginalize_bernoulli_logits"
_OBSERVED = "steps/observe_continuous_bernoulli_logits"

_COUNTS = [0, 3, 0, 5]
_OFFSETS = [0.0, 0.5, -0.5, 1.0]
_RATES = [3.0, 2.0, 4.0, 3.5]
_ETAS = [-30.0, -15.0, 0.0, 15.0, 30.0]
_POINTS = [0.1, 0.45, 0.8, 0.97]

#: The compiled runtime's relative tolerance against the exact joint: 64
#: float32 ulps, fixed from single-precision resolution.
_FLOAT32_TOLERANCE = 2.0**-18

#: The reference machine's relative tolerance against the exact joint.
_FLOAT64_TOLERANCE = 1e-12

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

#: The refusal each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    (_MARGINAL, "bugs"): "marginalize:no-collapse",
    **{
        (_OBSERVED, target): "family:ContinuousBernoulli:logits"
        for target in _EVERY_TARGET
    },
}


def _f64(value: float | list[float] | list[int]) -> torch.Tensor:
    """A value as a double-precision tensor.

    Parameters
    ----------
    value : float | list[float] | list[int]
        The value.

    Returns
    -------
    torch.Tensor
        The value in double precision.
    """
    return torch.as_tensor(value, dtype=torch.float64)


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


def _exact_marginal(eta: float) -> float:
    """The zero-inflated Poisson joint in double precision.

    Parameters
    ----------
    eta : float
        The shared predictor; row ``n``'s gate has logit
        ``eta + _OFFSETS[n]``.

    Returns
    -------
    float
        The prior's log density plus each row's log marginal.
    """
    logit = _f64(eta) + _f64(_OFFSETS)
    counts = _f64(_COUNTS)
    active = torch.nn.functional.logsigmoid(logit) + D.Poisson(_f64(_RATES)).log_prob(
        counts
    )
    inactive = torch.where(
        counts == 0, torch.nn.functional.logsigmoid(-logit), _f64(-math.inf)
    )
    prior = D.Normal(_f64(0.0), _f64(10.0)).log_prob(_f64(eta))
    return float(prior + torch.logaddexp(active, inactive).sum())


def _exact_observed(eta: float) -> float:
    """The continuous Bernoulli joint in double precision, from the logit.

    Parameters
    ----------
    eta : float
        The logit.

    Returns
    -------
    float
        The prior's log density plus the observations' log densities,
        with the normalizer ``l / tanh(l / 2)`` (two at ``l = 0``).
    """
    logit = _f64(eta)
    points = _f64(_POINTS)
    normalizer = (
        math.log(2.0)
        if eta == 0.0
        else float(torch.log(logit.abs() / torch.tanh(logit.abs() / 2.0)))
    )
    return float(
        D.Normal(_f64(0.0), _f64(2.0)).log_prob(logit)
        + (
            points * torch.nn.functional.logsigmoid(logit)
            + (1.0 - points) * torch.nn.functional.logsigmoid(-logit)
            + normalizer
        ).sum()
    )


def _reference(fixture: str) -> list[float]:
    """The reference machine's joint at each predictor, checked exact.

    Both execution paths are pinned to the exact joint first: the
    compiled runtime to single-precision resolution, the reference
    machine to double.

    Parameters
    ----------
    fixture : str
        The fixture.

    Returns
    -------
    list[float]
        The reference machine's joint at each predictor.
    """
    program = loads(_source(fixture))
    observed = fixture == _OBSERVED
    entry, name, data, exact = (
        ("continuous_bernoulli_logits", "u", _POINTS, _exact_observed)
        if observed
        else ("zip_logits", "y", _COUNTS, _exact_marginal)
    )
    joints: list[float] = []
    for eta in _ETAS:
        expected = exact(eta)
        values = {"eta": torch.tensor([[eta]]), name: torch.tensor(data).float()}
        host = {name: tuple(data)}
        if not observed:
            values["x"] = torch.tensor(_OFFSETS)
            values["r"] = torch.tensor(_RATES)
            host["x"] = tuple(_OFFSETS)
            host["r"] = tuple(_RATES)
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=_FLOAT32_TOLERANCE), eta
        run = program.run(entry, data=host, sites={"eta": eta})
        assert run.log_joint == pytest.approx(expected, rel=_FLOAT64_TOLERANCE), eta
        joints.append(run.log_joint)
    return joints


@pytest.mark.parametrize("fixture", [_MARGINAL, _OBSERVED])
def test_both_execution_paths_score_the_exact_joint(fixture: str) -> None:
    assert len(_reference(fixture)) == len(_ETAS)


def test_a_complement_weight_would_fail_the_target_tolerance() -> None:
    # The weight a renderer that formed `1 - sigmoid(eta)` in double
    # precision would score: at eta = 30 the complement keeps three digits,
    # which moves the joint by more than the tolerance the targets are held
    # to.
    logit = _f64(30.0)
    complement = torch.log1p(-torch.sigmoid(logit))
    exact = torch.nn.functional.logsigmoid(-logit)
    shift = 2.0 * float((complement - exact).abs())
    assert shift > _equivalence.adaptive_atol(n_obs=len(_COUNTS))


@pytest.mark.parametrize(
    "backend",
    sorted(
        backend
        for backend in _BACKENDS_WITH_IMAGES
        if (_MARGINAL, backend) not in _REFUSED
    ),
)
def test_target_scores_the_reference_joint(backend: str, scratch: pathlib.Path) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_source(_MARGINAL)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {"params": {"eta": eta}, "data": {"y": _COUNTS, "x": _OFFSETS, "r": _RATES}}
            for eta in _ETAS
        ],
        scratch=scratch / backend,
        shapes={
            "eta": [],
            "y": [len(_COUNTS)],
            "x": [len(_OFFSETS)],
            "r": [len(_RATES)],
        },
        dtypes={"eta": "float", "y": "int", "x": "float", "r": "float"},
    )
    _equivalence.assert_log_density_match(
        _reference(_MARGINAL),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=len(_COUNTS)),
        context=f"{backend}@{_MARGINAL}",
    )


@pytest.mark.requires_probe("church")
def test_church_scores_the_reference_joint(scratch: pathlib.Path) -> None:
    source = transpile(parse(_source(_MARGINAL)), target="church")
    result = ChurchProbe().evaluate(
        source,
        _MARGINAL.replace("/", "-"),
        [
            _protocol.Point(
                params={"eta": eta},
                data={"y": _COUNTS, "x": _OFFSETS, "r": _RATES},
            )
            for eta in _ETAS
        ],
        scratch=scratch / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(_MARGINAL),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=len(_COUNTS)),
        context=f"church@{_MARGINAL}",
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
    for fixture in (_MARGINAL, _OBSERVED):
        for target in _EVERY_TARGET:
            if (fixture, target) not in _REFUSED:
                transpile(parse(_source(fixture)), target=target)
