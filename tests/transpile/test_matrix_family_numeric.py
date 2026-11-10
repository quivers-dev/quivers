"""Vector- and matrix-valued draws score one joint on every path.

A multivariate normal and a Wishart each written with a Cholesky factor,
a one-hot draw from a probability vector, and a relaxed one-hot draw
whose event extent comes from its probabilities are each scored on the
compiled runtime and the reference machine, both pinned to an independent
`torch.distributions` joint, and on every target that renders them. A
target with no spelling for a fixture refuses it under a stated kind.
"""

from __future__ import annotations

from collections.abc import Callable
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.effects.checked_program import CheckedProgram
from quivers.qiec.entries import HostValue
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = int | float | list[int] | list[float] | list[list[float]]
"""A vector or a matrix, as a point carries a site's value."""

type Oracle = Callable[[torch.Tensor], torch.Tensor]
"""An independent computation of a fixture's joint at its one site."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_WISHART = "steps/sample_wishart_factor"
_MVN = "steps/sample_mvn_factor"
_ONEHOT = "steps/sample_onehot_probs"
_RELAXED = "steps/sample_relaxed_onehot_probs"

_FACTOR = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
_PROBS = torch.tensor([0.2, 0.3, 0.5])

#: Each fixture: its site, the site's points, the site's element dtype,
#: and the oracle.
_CASES: dict[str, tuple[str, list[Host], torch.dtype, Oracle]] = {
    _MVN: (
        "v",
        [[0.3, 0.4], [-1.0, 2.5], [1.2, -0.7]],
        torch.float32,
        lambda v: D.MultivariateNormal(
            torch.tensor([0.0, 1.0]), scale_tril=_FACTOR
        ).log_prob(v),
    ),
    _WISHART: (
        "sigma",
        [
            [[2.0, 0.3], [0.3, 1.0]],
            [[1.0, 0.5], [0.5, 1.25]],
            [[4.0, -1.0], [-1.0, 3.0]],
        ],
        torch.float32,
        lambda v: D.Wishart(df=torch.tensor(4.0), scale_tril=_FACTOR).log_prob(v),
    ),
    _ONEHOT: (
        "v",
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        torch.long,
        lambda v: D.OneHotCategorical(probs=_PROBS).log_prob(v.float()),
    ),
    _RELAXED: (
        "z",
        [[0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]],
        torch.float32,
        lambda v: D.RelaxedOneHotCategorical(torch.tensor(0.5), probs=_PROBS).log_prob(
            v
        ),
    ),
}

#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    (_MVN, "bugs"): "arg:list-literal",
    (_MVN, "church"): "arg:matrix-literal",
    (_MVN, "jags"): "family:MultivariateNormal",
    (_WISHART, "bugs"): "arg:matrix-literal",
    (_WISHART, "jags"): "arg:matrix-literal",
    (_WISHART, "church"): "family:Wishart:no-church-target",
    (_WISHART, "edward2"): "family:Wishart:no-edward2-target",
    (_WISHART, "gen"): "family:Wishart:no-gen-target",
    (_WISHART, "webppl"): "family:no-webppl-target:Wishart",
    (_ONEHOT, "bugs"): "arg:list-literal",
    (_ONEHOT, "church"): "family:OneHotCategorical:no-church-target",
    (_ONEHOT, "gen"): "family:OneHotCategorical:no-gen-target",
    (_ONEHOT, "jags"): "family:no-target-name:OneHotCategorical",
    (_ONEHOT, "pymc"): "family:OneHotCategorical",
    (_ONEHOT, "stan"): "family:no-stan-target:OneHotCategorical",
    (_ONEHOT, "turing"): "family:OneHotCategorical",
    (_ONEHOT, "webppl"): "family:no-webppl-target:OneHotCategorical",
    (_RELAXED, "bugs"): "arg:list-literal",
    (_RELAXED, "church"): "family:RelaxedOneHotCategorical:no-church-target",
    (_RELAXED, "gen"): "family:RelaxedOneHotCategorical:no-gen-target",
    (_RELAXED, "jags"): "family:no-target-name:RelaxedOneHotCategorical",
    (_RELAXED, "numpyro"): "family:no-target-name:RelaxedOneHotCategorical",
    (_RELAXED, "pymc"): "family:RelaxedOneHotCategorical",
    (_RELAXED, "stan"): "declare:simplex:event-rank:0",
    (_RELAXED, "turing"): "family:RelaxedOneHotCategorical",
    (_RELAXED, "webppl"): "family:no-webppl-target:RelaxedOneHotCategorical",
}


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


def _nested(value: Host) -> HostValue:
    """A site value as the reference machine's nested tuples.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    HostValue
        The value with every list a tuple.
    """
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    return value


def _reference(fixture: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    fixture : str
        The key of the case in `_CASES`.

    Returns
    -------
    list[float]
        The compiled runtime's log joint at each point, each equal to
        the reference machine's and to the case's oracle.
    """
    site, points, dtype, oracle = _CASES[fixture]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    model = program.morphism
    assert isinstance(model, MonadicProgram | CheckedProgram)
    densities: list[float] = []
    for point in points:
        value = torch.tensor(point, dtype=dtype)
        expected = float(oracle(value))
        joint = float(
            model.log_joint(torch.zeros(1, 1), {site: value.unsqueeze(0)}).sum()
        )
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(entry.name, data={}, sites={site: _nested(point)})
        assert run.log_joint == pytest.approx(expected, rel=1e-5)
        densities.append(joint)
    return densities


@pytest.mark.parametrize("fixture", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(fixture: str) -> None:
    assert len(_reference(fixture)) == len(_CASES[fixture][1])


@pytest.mark.parametrize(
    ("fixture", "backend"),
    [
        (fixture, backend)
        for fixture in sorted(_CASES)
        for backend in sorted(_BACKENDS_WITH_IMAGES)
        if (fixture, backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    fixture: str, backend: str, scratch: pathlib.Path
) -> None:
    site, points, dtype, _ = _CASES[fixture]
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    shape = list(torch.tensor(points[0]).shape)
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {"params": {site: torch.tensor(point).reshape(-1).tolist()}, "data": {}}
            for point in points
        ],
        scratch=scratch / fixture.replace("/", "_") / backend,
        shapes={site: shape},
        # A float site is typed `float32` so Pyro scores it in the dtype
        # of the model's literal parameters; every other probe reads
        # `float32` as a float.
        dtypes={site: "int" if dtype == torch.long else "float32"},
    )
    _equivalence.assert_log_density_match(
        _reference(fixture),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=1),
        context=f"{backend}@{fixture}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )


@pytest.mark.parametrize("family", ["MultivariateNormal", "Wishart"])
@pytest.mark.parametrize("backend", ["numpyro", "stan"])
def test_a_factor_known_only_at_run_time_is_refused_by_its_keyword(
    family: str, backend: str
) -> None:
    draw = (
        "MultivariateNormal(loc=[0.0, 1.0], scale_tril=L) [over=Dim]"
        if family == "MultivariateNormal"
        else "Wishart(df=4.0, scale_tril=L) [over=[Dim, Dim]]"
    )
    source = (
        "object Dim : FinSet 2\nobject One : FinSet 1\n"
        "program p : One -> One\n"
        "    sample L : Dim <- LKJCholesky(concentration=2.0)\n"
        f"    sample v <- {draw}\n    return v\nexport p\n"
    )
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(source), target=backend)
    assert list(caught.value.kinds) == [
        f"family:{family}:scale_tril: the {family}'s scale_tril is a Cholesky "
        "factor known only at run time; write it as a matrix literal",
    ], caught.value.kinds
