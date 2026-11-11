"""Vector- and matrix-valued draws score one joint on every path.

A multivariate normal and a Wishart each written with a Cholesky factor,
a multivariate normal whose mean is sampled, a Dirichlet whose
concentration is sampled, a one-hot draw from a literal and from a
sampled probability vector, and a relaxed one-hot draw whose event
extent comes from its probabilities are each scored on the
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

type Point = dict[str, Host]
"""One value at each of a fixture's sites."""

type Oracle = Callable[[dict[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint at its sites."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_WISHART = "steps/sample_wishart_factor"
_MVN = "steps/sample_mvn_factor"
_MVN_SAMPLED_LOC = "steps/sample_mvn_sampled_loc"
_DIRICHLET_SAMPLED = "steps/sample_dirichlet_sampled_concentration"
_ONEHOT = "steps/sample_onehot_probs"
_ONEHOT_SAMPLED = "steps/sample_onehot_sampled_probs"
_RELAXED = "steps/sample_relaxed_onehot_probs"

_FACTOR = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
_PROBS = torch.tensor([0.2, 0.3, 0.5])
_CONCENTRATION = torch.tensor([1.0, 2.0, 3.0])

#: Each fixture: its points, each site's element dtype, and the oracle.
_CASES: dict[str, tuple[list[Point], dict[str, torch.dtype], Oracle]] = {
    _MVN: (
        [{"v": [0.3, 0.4]}, {"v": [-1.0, 2.5]}, {"v": [1.2, -0.7]}],
        {"v": torch.float32},
        lambda s: D.MultivariateNormal(
            torch.tensor([0.0, 1.0]), scale_tril=_FACTOR
        ).log_prob(s["v"]),
    ),
    # The mean is a vector site: a target that reads it as a number
    # fills a vector from it and scores a different density.
    _MVN_SAMPLED_LOC: (
        [
            {"m": [0.5, -1.0], "v": [0.3, 0.4]},
            {"m": [-0.2, 1.5], "v": [-1.0, 2.5]},
            {"m": [1.0, 0.25], "v": [1.2, -0.7]},
        ],
        {"m": torch.float32, "v": torch.float32},
        lambda s: (
            D.Normal(0.0, 1.0).log_prob(s["m"]).sum()
            + D.MultivariateNormal(s["m"], scale_tril=_FACTOR).log_prob(s["v"])
        ),
    ),
    _DIRICHLET_SAMPLED: (
        [
            {"a": [0.5, 2.0, 4.0], "v": [0.2, 0.3, 0.5]},
            {"a": [3.0, 1.0, 0.7], "v": [0.6, 0.3, 0.1]},
            {"a": [1.5, 1.5, 2.5], "v": [0.1, 0.1, 0.8]},
        ],
        {"a": torch.float32, "v": torch.float32},
        lambda s: (
            D.Gamma(2.0, 1.0).log_prob(s["a"]).sum()
            + D.Dirichlet(s["a"]).log_prob(s["v"])
        ),
    ),
    _WISHART: (
        [
            {"sigma": [[2.0, 0.3], [0.3, 1.0]]},
            {"sigma": [[1.0, 0.5], [0.5, 1.25]]},
            {"sigma": [[4.0, -1.0], [-1.0, 3.0]]},
        ],
        {"sigma": torch.float32},
        lambda s: D.Wishart(df=torch.tensor(4.0), scale_tril=_FACTOR).log_prob(
            s["sigma"]
        ),
    ),
    _ONEHOT: (
        [{"v": [1, 0, 0]}, {"v": [0, 1, 0]}, {"v": [0, 0, 1]}],
        {"v": torch.long},
        lambda s: D.OneHotCategorical(probs=_PROBS).log_prob(s["v"].float()),
    ),
    _ONEHOT_SAMPLED: (
        [
            {"q": [0.2, 0.3, 0.5], "h": [1, 0, 0]},
            {"q": [0.6, 0.3, 0.1], "h": [0, 1, 0]},
            {"q": [0.1, 0.1, 0.8], "h": [0, 0, 1]},
        ],
        {"q": torch.float32, "h": torch.long},
        lambda s: (
            D.Dirichlet(_CONCENTRATION).log_prob(s["q"])
            + D.OneHotCategorical(probs=s["q"]).log_prob(s["h"].float())
        ),
    ),
    _RELAXED: (
        [
            {"z": [0.2, 0.5, 0.3]},
            {"z": [0.6, 0.3, 0.1]},
            {"z": [0.1, 0.1, 0.8]},
        ],
        {"z": torch.float32},
        lambda s: D.RelaxedOneHotCategorical(torch.tensor(0.5), probs=_PROBS).log_prob(
            s["z"]
        ),
    ),
}

#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    (_MVN, "bugs"): "arg:list-literal",
    (_MVN, "church"): "arg:matrix-literal",
    (_MVN, "jags"): "family:MultivariateNormal",
    (_MVN_SAMPLED_LOC, "bugs"): "arg:matrix-literal",
    (_MVN_SAMPLED_LOC, "church"): "arg:matrix-literal",
    (_MVN_SAMPLED_LOC, "jags"): "family:MultivariateNormal:non-ref-covariance",
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
    (_ONEHOT_SAMPLED, "bugs"): "arg:list-literal",
    (_ONEHOT_SAMPLED, "church"): "family:OneHotCategorical:no-church-target",
    (_ONEHOT_SAMPLED, "gen"): "family:OneHotCategorical:no-gen-target",
    (_ONEHOT_SAMPLED, "jags"): "arg:list-literal",
    (_ONEHOT_SAMPLED, "pymc"): "family:OneHotCategorical",
    (_ONEHOT_SAMPLED, "stan"): "family:no-stan-target:OneHotCategorical",
    (_ONEHOT_SAMPLED, "turing"): "family:OneHotCategorical",
    (_ONEHOT_SAMPLED, "webppl"): "family:no-webppl-target:OneHotCategorical",
    (_RELAXED, "bugs"): "arg:list-literal",
    (_RELAXED, "church"): "family:RelaxedOneHotCategorical:no-church-target",
    (_RELAXED, "gen"): "family:RelaxedOneHotCategorical:no-gen-target",
    (_RELAXED, "jags"): "family:no-target-name:RelaxedOneHotCategorical",
    (_RELAXED, "numpyro"): "family:no-target-name:RelaxedOneHotCategorical",
    (_RELAXED, "pymc"): "family:RelaxedOneHotCategorical",
    (_RELAXED, "stan"): "family:no-stan-target:RelaxedOneHotCategorical",
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


def _tensors(fixture: str, point: Point) -> dict[str, torch.Tensor]:
    """A point's site values as tensors of each site's dtype.

    Parameters
    ----------
    fixture : str
        The key of the case in `_CASES`.
    point : Point
        The point.

    Returns
    -------
    dict[str, torch.Tensor]
        Each site's value.
    """
    dtypes = _CASES[fixture][1]
    return {
        site: torch.tensor(value, dtype=dtypes[site]) for site, value in point.items()
    }


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
    points, _, oracle = _CASES[fixture]
    program = loads(_source(fixture))
    (entry,) = program.entry_points()
    model = program.morphism
    assert isinstance(model, MonadicProgram | CheckedProgram)
    densities: list[float] = []
    for point in points:
        values = _tensors(fixture, point)
        expected = float(oracle(values))
        joint = float(
            model.log_joint(
                torch.zeros(1, 1),
                {site: value.unsqueeze(0) for site, value in values.items()},
            ).sum()
        )
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry.name,
            data={},
            sites={site: _nested(value) for site, value in point.items()},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-5)
        densities.append(joint)
    return densities


@pytest.mark.parametrize("fixture", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(fixture: str) -> None:
    assert len(_reference(fixture)) == len(_CASES[fixture][0])


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
    points, dtypes, _ = _CASES[fixture]
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    first = _tensors(fixture, points[0])
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {
                "params": {
                    site: value.reshape(-1).tolist()
                    for site, value in _tensors(fixture, point).items()
                },
                "data": {},
            }
            for point in points
        ],
        scratch=scratch / fixture.replace("/", "_") / backend,
        shapes={site: list(value.shape) for site, value in first.items()},
        # A float site is typed `float32` so Pyro scores it in the dtype
        # of the model's literal parameters; every other probe reads
        # `float32` as a float.
        dtypes={
            site: "int" if dtype == torch.long else "float32"
            for site, dtype in dtypes.items()
        },
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
