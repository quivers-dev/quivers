"""A per-row gate under one shared prior scores one joint on every target.

The fixture draws a Bernoulli gate for each row from one probability
``p = sigmoid(eta)`` that every row shares, and reads it through a scope
``let`` that, at each atom, holds a single number rather than one value
per row. Each row is then an independent two-component mixture, a point
mass at zero with weight ``1 - p`` and ``Poisson(3)`` with weight ``p``,
so the joint is the prior plus the sum of the rows' log marginals.

Two renderer defects broke it. Stan seeded the accumulator in a loop
over the gate's own index and then added each observation's term at
that index from inside the observation's loop, where it is not in scope,
so stanc rejected the program. WebPPL read the scope let at each row,
``gated_rate[n]``, though at an atom it is one number, so every score
was ``NaN``.

The compiled runtime and the reference machine are pinned to an exact
double-precision computation first, and every target that renders the
fixture is held to the reference machine's joints. BUGS has no
statement that adds an integrated log-density to the joint and refuses
the block by name.
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
    / "marginalize_shared_prior_scalar_let.qvr"
)

_COUNTS = [0, 2, 0, 5]
_ETAS = [-2.0, -0.5, 0.0, 1.0, 2.5]
_RATE = 3.0

#: The compiled runtime's relative tolerance against the exact joint: 64
#: float32 ulps, fixed from single-precision resolution.
_FLOAT32_TOLERANCE = 2.0**-18

#: The reference machine's relative tolerance against the exact joint.
_FLOAT64_TOLERANCE = 1e-12

#: The refusal each target states for the fixture.
_REFUSED: dict[str, str] = {"bugs": "marginalize:no-collapse"}

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


def _exact(eta: float) -> float:
    """The joint in double precision.

    Parameters
    ----------
    eta : float
        The shared logit.

    Returns
    -------
    float
        The prior's log density plus each row's log marginal.
    """
    logit = _f64(eta)
    counts = _f64(_COUNTS)
    on = torch.nn.functional.logsigmoid(logit) + D.Poisson(_f64(_RATE)).log_prob(counts)
    off = torch.where(
        counts == 0, torch.nn.functional.logsigmoid(-logit), _f64(-math.inf)
    )
    prior = D.Normal(_f64(0.0), _f64(2.0)).log_prob(logit)
    return float(prior + torch.logaddexp(on, off).sum())


def _reference() -> list[float]:
    """The reference machine's joint at each logit, checked exact.

    Returns
    -------
    list[float]
        The reference machine's joints.
    """
    program = loads(_FIXTURE.read_text())
    joints: list[float] = []
    for eta in _ETAS:
        expected = _exact(eta)
        values = {"eta": torch.tensor([[eta]]), "y": torch.tensor(_COUNTS).float()}
        compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert compiled == pytest.approx(expected, rel=_FLOAT32_TOLERANCE), eta
        run = program.run("gate_shared", data={"y": tuple(_COUNTS)}, sites={"eta": eta})
        assert run.log_joint == pytest.approx(expected, rel=_FLOAT64_TOLERANCE), eta
        joints.append(run.log_joint)
    return joints


def test_both_execution_paths_score_the_exact_joint() -> None:
    assert len(_reference()) == len(_ETAS)


def test_one_gate_shared_by_every_row_would_score_another_joint() -> None:
    # The oracle tells the per-row gate from a single gate shared by the
    # rows, which the fixture does not denote: one draw for all four rows
    # puts the two positive counts and the zeros under one branch.
    logit = _f64(0.0)
    counts = _f64(_COUNTS)
    on = (
        torch.nn.functional.logsigmoid(logit)
        + D.Poisson(_f64(_RATE)).log_prob(counts).sum()
    )
    off = torch.nn.functional.logsigmoid(-logit) + torch.where(
        (counts == 0).all(), _f64(0.0), _f64(-math.inf)
    )
    shared = float(
        D.Normal(_f64(0.0), _f64(2.0)).log_prob(logit) + torch.logaddexp(on, off)
    )
    assert abs(shared - _exact(0.0)) > _equivalence.adaptive_atol(n_obs=len(_COUNTS))


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
        points=[{"params": {"eta": eta}, "data": {"y": _COUNTS}} for eta in _ETAS],
        scratch=scratch / backend,
        shapes={"eta": [], "y": [len(_COUNTS)]},
        dtypes={"eta": "float", "y": "int"},
    )
    _equivalence.assert_log_density_match(
        _reference(),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=len(_COUNTS)),
        context=f"{backend}@{_FIXTURE.stem}",
    )


@pytest.mark.requires_probe("church")
def test_church_scores_the_reference_joint(scratch: pathlib.Path) -> None:
    source = transpile(parse(_FIXTURE.read_text()), target="church")
    result = ChurchProbe().evaluate(
        source,
        _FIXTURE.stem,
        [_protocol.Point(params={"eta": eta}, data={"y": _COUNTS}) for eta in _ETAS],
        scratch=scratch / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=len(_COUNTS)),
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


def test_stan_indexes_each_row_s_accumulator_by_the_row() -> None:
    emitted = transpile(parse(_FIXTURE.read_text()), target="stan").decode()
    compact = "".join(emitted.split())
    assert "lps_z[n_Row,1]+=poisson_lpmf(y[n_Row]|" in compact
    assert "lps_z[g_Row,1]+=" not in compact


def test_webppl_reads_the_atom_s_scope_let_as_one_number() -> None:
    emitted = transpile(parse(_FIXTURE.read_text()), target="webppl").decode()
    assert "gated_rate[n]" not in emitted
