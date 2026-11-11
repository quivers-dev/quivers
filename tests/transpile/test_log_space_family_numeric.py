"""The log-space families score one joint on every target that renders them.

A program draws a `LogGamma` and a `LogitBeta` latent at small shapes and
observes a plate through both. Its joint is pinned to an independent SciPy
float64 closed form on the compiled runtime and the reference machine, at
points that include the far tails, where a Gamma or Beta density composed
with an exponential or a sigmoid is no longer finite. Stan, NumPyro, and
Pyro render both families through their runtime helpers and must score the
same joint, constant included, since both densities are normalized; every
other target refuses the families by name.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch
from scipy import special, stats

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

_SOURCE = """\
object Item : FinSet 4

program log_space : Item -> Item
    sample log_rate <- LogGamma(concentration=0.002, rate=1.5)
    sample weight <- LogitBeta(concentration1=0.01, concentration0=2.0)
    let centre = sigmoid(weight)
    let spread = exp(log_rate) + 1.0
    observe y : Item <- Normal(loc=centre, scale=spread)
    return centre

export log_space
"""

_Y = [0.3, -1.2, 0.8, 0.1]

#: Latent points: the bulk, and the far tails the families exist for.
_POINTS: list[dict[str, float]] = [
    {"log_rate": -0.4, "weight": 0.2},
    {"log_rate": 1.1, "weight": -2.0},
    {"log_rate": -650.0, "weight": -480.0},
    {"log_rate": -90.0, "weight": -35.0},
]

#: The targets that render both families.
_RENDERING = frozenset({"stan", "numpyro", "pyro"})


def _closed_form(point: dict[str, float]) -> float:
    """The joint from SciPy's float64 special functions.

    Parameters
    ----------
    point : dict[str, float]
        The latent values.

    Returns
    -------
    float
        ``log p(log_rate) + log p(weight) + sum log p(y | ...)``.
    """
    log_rate, weight = point["log_rate"], point["weight"]
    prior = float(stats.loggamma.logpdf(log_rate + math.log(1.5), 0.002)) + float(
        0.01 * special.log_expit(weight)
        + 2.0 * special.log_expit(-weight)
        - special.betaln(0.01, 2.0)
    )
    likelihood = np.sum(
        stats.norm.logpdf(
            np.array(_Y), loc=special.expit(weight), scale=math.exp(log_rate) + 1.0
        )
    )
    return prior + float(likelihood)


def _reference() -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Returns
    -------
    list[float]
        The closed-form joint at each point, after checking the compiled
        runtime and the reference machine against it.
    """
    program = loads(_SOURCE)
    (entry,) = program.entry_points()
    densities = []
    for point in _POINTS:
        expected = _closed_form(point)
        assert math.isfinite(expected)
        values = {
            "y": torch.tensor(_Y, dtype=torch.float64),
            **{
                name: torch.tensor(value, dtype=torch.float64)
                for name, value in point.items()
            },
        }
        joint = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
        assert joint == pytest.approx(expected, rel=1e-6)
        run = program.run(entry.name, data={"y": tuple(_Y)}, sites=dict(point))
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(expected)
    return densities


def test_both_execution_paths_score_the_closed_form_joint() -> None:
    assert len(_reference()) == len(_POINTS)


@pytest.mark.parametrize("backend", sorted(_RENDERING))
def test_target_scores_the_closed_form_joint(
    backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_SOURCE), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    shapes = {"y": [len(_Y)], "log_rate": [], "weight": []}
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": dict(point), "data": {"y": _Y}} for point in _POINTS],
        scratch=scratch / backend,
        shapes=shapes,
        dtypes={name: "float" for name in shapes},
    )
    scored = [float(value) for value in raw["log_densities"]]
    reference = _reference()
    atol = _equivalence.adaptive_atol(n_obs=len(_Y))
    _equivalence.assert_log_density_match(
        reference, scored, atol=atol, context=f"{backend}@log_space"
    )
    # Both densities are normalized, so the target carries no offset.
    for got, want in zip(scored, reference):
        assert got == pytest.approx(want, abs=atol), (backend, got, want)


@pytest.mark.parametrize(
    "backend", sorted(set(available_targets()) - _RENDERING), ids=str
)
def test_other_targets_refuse_the_families_by_name(backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_SOURCE), target=backend)
    # Each target spells its family refusal its own way; every one is a
    # `family:` kind naming the family.
    assert any(
        kind.startswith("family:") and ("LogGamma" in kind or "LogitBeta" in kind)
        for kind in caught.value.kinds
    ), caught.value.kinds
