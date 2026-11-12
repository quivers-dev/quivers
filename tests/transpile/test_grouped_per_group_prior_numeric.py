"""A grouped mixture with a per-group prior shares one class within a group.

A block ``marginalize c : K <- Categorical(probs=pi) [over=Grp]`` declares
one latent class per group of ``Grp``. When the prior ``pi`` is allocated
per group, group ``g``'s class is drawn from ``pi[g]``, and every row the
observation's ``via`` map sends to ``g`` is scored under that one class:

    log p(y | pi, mu) = sum_g log sum_k pi[g, k] prod_{n : idx(n) = g} p(y_n | k).

The compiled runtime, Stan and the Python-family renderers instead gave each
row its own class drawn from its group's prior, which differs from the
grouped mixture whenever the map sends several rows to one group. The
fixture's map ``[0, 0, 1]`` does. The compiled runtime and the reference
machine are pinned to a double-precision computation of the grouped
mixture, a test shows the oracle rejects the class-per-row joint, and
every target that renders the fixture is held to the reference machine.
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

_FIXTURE = (
    pathlib.Path(__file__).parent
    / "fixtures"
    / "steps"
    / "marginalize_per_group_prior_shared_class.qvr"
)

_IDX = [0, 0, 1]
_Y = [0.5, 2.5, -1.0]

#: Points: each group's class weights and the class means.
_POINTS: list[tuple[list[list[float]], list[float]]] = [
    ([[0.3, 0.7], [0.6, 0.4]], [0.0, 2.0]),
    ([[0.9, 0.1], [0.2, 0.8]], [-1.0, 1.5]),
    ([[0.5, 0.5], [0.5, 0.5]], [0.4, 3.0]),
    ([[0.05, 0.95], [0.7, 0.3]], [2.0, -2.0]),
]

#: The compiled runtime's relative tolerance against the exact joint: 64
#: float32 ulps, fixed from single-precision resolution.
_FLOAT32_TOLERANCE = 2.0**-18

#: The reference machine's relative tolerance against the exact joint.
_FLOAT64_TOLERANCE = 1e-12

#: The refusal each target states for the fixture.
_REFUSED: dict[str, str] = {
    "bugs": "marginalize:grouped-fibration",
    "jags": "let-expr:elementwise-axis-operator",
}

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


def _row_log_likelihood(mu: list[float], row: int, k: int) -> float:
    """``log p(y_row | class k)``.

    Parameters
    ----------
    mu : list[float]
        The class means.
    row : int
        The row.
    k : int
        The class.

    Returns
    -------
    float
        The Normal log density.
    """
    return float(
        D.Normal(torch.tensor(mu[k], dtype=torch.float64), 1.0).log_prob(
            torch.tensor(_Y[row], dtype=torch.float64)
        )
    )


def _prior(pi: list[list[float]], mu: list[float]) -> float:
    """The latents' prior log density.

    Parameters
    ----------
    pi : list[list[float]]
        Each group's class weights.
    mu : list[float]
        The class means.

    Returns
    -------
    float
        The Dirichlet and Normal log densities.
    """
    dirichlet = D.Dirichlet(torch.ones(2, dtype=torch.float64))
    normal = D.Normal(torch.tensor(0.0, dtype=torch.float64), 5.0)
    return sum(
        float(dirichlet.log_prob(torch.tensor(row, dtype=torch.float64))) for row in pi
    ) + sum(float(normal.log_prob(torch.tensor(m, dtype=torch.float64))) for m in mu)


def _grouped(pi: list[list[float]], mu: list[float]) -> float:
    """The grouped mixture's joint: one class per group.

    Parameters
    ----------
    pi : list[list[float]]
        Each group's class weights.
    mu : list[float]
        The class means.

    Returns
    -------
    float
        The joint.
    """
    total = _prior(pi, mu)
    for group in range(2):
        total += math.log(
            sum(
                pi[group][k]
                * math.exp(
                    sum(
                        _row_log_likelihood(mu, row, k)
                        for row in range(3)
                        if _IDX[row] == group
                    )
                )
                for k in range(2)
            )
        )
    return total


def _class_per_row(pi: list[list[float]], mu: list[float]) -> float:
    """The joint of a class per row drawn from its group's prior.

    Parameters
    ----------
    pi : list[list[float]]
        Each group's class weights.
    mu : list[float]
        The class means.

    Returns
    -------
    float
        The joint the fixture does not denote.
    """
    return _prior(pi, mu) + sum(
        math.log(
            sum(
                pi[_IDX[row]][k] * math.exp(_row_log_likelihood(mu, row, k))
                for k in range(2)
            )
        )
        for row in range(3)
    )


def _reference() -> list[float]:
    """The reference machine's joint at each point, checked exact.

    Returns
    -------
    list[float]
        The reference machine's joints.
    """
    program = loads(_FIXTURE.read_text())
    joints: list[float] = []
    for pi, mu in _POINTS:
        expected = _grouped(pi, mu)
        values = {
            "pi": torch.tensor([pi]),
            "mu": torch.tensor([mu]),
            "y": torch.tensor(_Y),
            "idx": torch.tensor(_IDX),
        }
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=_FLOAT32_TOLERANCE)
        run = program.run(
            "grouped",
            data={"y": tuple(_Y), "idx": tuple(_IDX)},
            sites={"pi": tuple(tuple(row) for row in pi), "mu": tuple(mu)},
        )
        assert run.log_joint == pytest.approx(expected, rel=_FLOAT64_TOLERANCE)
        joints.append(run.log_joint)
    return joints


def test_both_execution_paths_score_the_grouped_mixture() -> None:
    assert len(_reference()) == len(_POINTS)


@pytest.mark.parametrize(("pi", "mu"), _POINTS)
def test_the_oracle_rejects_a_class_per_row(
    pi: list[list[float]], mu: list[float]
) -> None:
    gap = abs(_grouped(pi, mu) - _class_per_row(pi, mu))
    assert gap > _equivalence.adaptive_atol(n_obs=len(_Y))


@pytest.mark.parametrize(
    "backend",
    sorted(backend for backend in _BACKENDS_WITH_IMAGES if backend not in _REFUSED),
)
def test_target_scores_the_reference_joint(backend: str, scratch: pathlib.Path) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_FIXTURE.read_text()), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {
                "params": {"pi": [w for row in pi for w in row], "mu": mu},
                "data": {"y": _Y, "idx": _IDX},
            }
            for pi, mu in _POINTS
        ],
        scratch=scratch / backend,
        shapes={"pi": [2, 2], "mu": [2], "y": [len(_Y)], "idx": [len(_IDX)]},
        dtypes={"pi": "float", "mu": "float", "y": "float", "idx": "int"},
    )
    _equivalence.assert_log_density_match(
        _reference(),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=len(_Y)),
        context=f"{backend}@{_FIXTURE.stem}",
    )


@pytest.mark.requires_probe("church")
def test_church_scores_the_reference_joint(scratch: pathlib.Path) -> None:
    source = transpile(parse(_FIXTURE.read_text()), target="church")
    result = ChurchProbe().evaluate(
        source,
        _FIXTURE.stem,
        [
            _protocol.Point(params={"pi": pi, "mu": mu}, data={"y": _Y, "idx": _IDX})
            for pi, mu in _POINTS
        ],
        scratch=scratch / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=len(_Y)),
        context=f"church@{_FIXTURE.stem}",
    )


@pytest.mark.parametrize("backend", sorted(_REFUSED))
def test_target_refuses_under_its_kind(backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_FIXTURE.read_text()), target=backend)
    expected = _REFUSED[backend]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )


def test_every_target_renders_or_refuses_the_fixture() -> None:
    for target in _EVERY_TARGET:
        if target not in _REFUSED:
            transpile(parse(_FIXTURE.read_text()), target=target)
