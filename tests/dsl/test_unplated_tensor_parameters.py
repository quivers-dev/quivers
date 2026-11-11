"""Host tensors filling the vector and matrix parameters of an unplated draw.

A free name filling a tensor parameter of a draw with no batch axes is one
tensor over the parameter's own axes. The family ties those axes to its
sample's: a multivariate normal's mean is a vector over its event axis and
its Cholesky factor a square matrix over it, a Wishart's scale factor a
matrix over its two. The draw fixes each tied axis from its plate's event
axes or its other arguments; an axis nothing in the draw fixes is one open
extent shared by every name tied to it, which a later use fixes: an event
axis an ``[over=...]`` plate names, a ``Real N`` codomain the draw is
returned through, or another reading of the same datum. The compiled
runtime and the reference machine then score the same joint, pinned here to
`torch.distributions`, at vector and matrix data, for sampled and observed
sites; data of another shape, and a datum read at two different sizes, are
refused by name.
"""

from __future__ import annotations

from collections.abc import Mapping
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.kinds import NAT
from quivers.qiec.types import IndexLiteral

_FIXTURES = pathlib.Path(__file__).parents[1] / "transpile" / "fixtures" / "steps"

_L_PRIOR = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
_L_OBS = torch.tensor([[1.5, 0.0], [-0.3, 0.8]])
_L_HOST = torch.tensor([[1.2, 0.0], [0.4, 0.7]])


def _d(value: torch.Tensor) -> torch.Tensor:
    """A value in double precision, at which every oracle is computed.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    torch.Tensor
        The value as float64.
    """
    return value.to(torch.float64)


def _source(stem: str) -> str:
    """A fixture's QVR source.

    Parameters
    ----------
    stem : str
        The fixture's file stem.

    Returns
    -------
    str
        The source text.
    """
    return (_FIXTURES / f"{stem}.qvr").read_text()


def _nested(value: torch.Tensor) -> float | tuple[object, ...]:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    float | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return float(value.item())
    return tuple(_nested(item) for item in value)


def _both_paths(
    stem: str,
    data: Mapping[str, torch.Tensor],
    sites: Mapping[str, torch.Tensor],
) -> tuple[float, float]:
    """The joint the compiled runtime and the reference machine score.

    Parameters
    ----------
    stem : str
        The fixture.
    data : Mapping[str, torch.Tensor]
        The host data.
    sites : Mapping[str, torch.Tensor]
        The latent sites' values.

    Returns
    -------
    tuple[float, float]
        The compiled runtime's joint and the reference machine's.
    """
    program = loads(_source(stem))
    (entry,) = program.entry_points()
    compiled = float(
        program.morphism.log_joint(torch.zeros(1, 1), {**data, **sites}).sum()
    )
    reference = program.run(
        entry.name,
        data={name: _nested(value) for name, value in data.items()},
        sites={name: _nested(value) for name, value in sites.items()},
    ).log_joint
    return compiled, reference


_MEANS = (torch.tensor([0.3, -0.4]), torch.tensor([-1.1, 0.8]))


@pytest.mark.parametrize("m0", _MEANS)
def test_a_host_mean_vector_scores_the_torch_joint(m0: torch.Tensor) -> None:
    mu = torch.tensor([0.2, 0.1])
    y = torch.tensor([0.9, -0.6])
    expected = float(
        D.MultivariateNormal(_d(m0), scale_tril=_d(_L_PRIOR)).log_prob(_d(mu))
        + D.MultivariateNormal(_d(mu), scale_tril=_d(_L_OBS)).log_prob(_d(y))
    )
    compiled, reference = _both_paths(
        "sample_mvn_host_loc", {"m0": m0, "y": y}, {"mu": mu}
    )
    assert compiled == pytest.approx(expected, rel=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("m0", _MEANS)
def test_a_host_mean_and_factor_score_the_plated_torch_joint(m0: torch.Tensor) -> None:
    mu = torch.tensor([0.2, 0.1])
    y = torch.tensor([[0.9, -0.6], [0.1, 0.4], [-1.3, 0.2]])
    expected = float(
        D.MultivariateNormal(_d(m0), scale_tril=_d(_L_HOST)).log_prob(_d(mu))
        + D.MultivariateNormal(_d(mu), scale_tril=_d(_L_HOST)).log_prob(_d(y)).sum()
    )
    compiled, reference = _both_paths(
        "observe_mvn_host_factor", {"m0": m0, "L": _L_HOST, "y": y}, {"mu": mu}
    )
    assert compiled == pytest.approx(expected, rel=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9)


def test_a_host_wishart_factor_scores_the_torch_density() -> None:
    w = torch.tensor([[2.0, 0.3], [0.3, 1.1]])
    expected = float(
        D.Wishart(df=torch.tensor(4.0, dtype=torch.float64), scale_tril=_d(_L_HOST))
        .log_prob(_d(w))
    )
    compiled, reference = _both_paths(
        "sample_wishart_host_factor", {"L": _L_HOST}, {"W": w}
    )
    assert compiled == pytest.approx(expected, rel=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9)


def test_a_wrong_factor_is_told_apart() -> None:
    w = torch.tensor([[2.0, 0.3], [0.3, 1.1]])
    transposed = D.Wishart(df=torch.tensor(4.0), scale_tril=_L_PRIOR).log_prob(w)
    compiled, reference = _both_paths(
        "sample_wishart_host_factor", {"L": _L_HOST}, {"W": w}
    )
    assert abs(compiled - float(transposed)) > 1e-2
    assert abs(reference - float(transposed)) > 1e-2


@pytest.mark.parametrize(
    ("stem", "data"),
    [
        ("sample_mvn_host_loc", {"m0": torch.tensor([0.3, -0.4])}),
        ("sample_wishart_host_factor", {"L": _L_HOST}),
        (
            "observe_mvn_host_factor",
            {"m0": torch.tensor([0.3, -0.4]), "L": _L_HOST},
        ),
    ],
)
def test_a_forward_draw_is_scored_at_the_density_it_was_drawn_from(
    stem: str, data: dict[str, torch.Tensor]
) -> None:
    program = loads(_source(stem))
    torch.manual_seed(5)
    recorded = trace(program.morphism, torch.zeros(1, 1), observations=dict(data))
    values = {name: site.value for name, site in recorded.sites.items()}
    rescored = float(program.morphism.log_joint(torch.zeros(1, 1), {**data, **values}))
    assert float(recorded.log_joint.sum()) == pytest.approx(rescored, rel=1e-5)
    if "mu" in values:
        assert values["mu"].reshape(-1).shape == (2,)
        expected = D.MultivariateNormal(
            data["m0"], scale_tril=data.get("L", _L_PRIOR)
        ).log_prob(values["mu"].reshape(-1))
        assert float(recorded.sites["mu"].log_prob.sum()) == pytest.approx(
            float(expected), rel=1e-5
        )
    if "W" in values:
        assert values["W"].reshape(-1, 2, 2).shape == (1, 2, 2)


def test_every_tied_extent_is_fixed_and_left_out_of_the_telescope() -> None:
    module = loads(_source("observe_mvn_host_factor")).qiec
    assert module is not None
    (entry,) = module.entries
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    for name, extents in (("m0", (2,)), ("L", (2, 2))):
        shape = tensor_shape(types[name])
        assert shape is not None
        assert shape[1] == tuple(IndexLiteral(value, NAT) for value in extents)
    assert entry.telescope == ()


_UNSIZED = (
    "object One : FinSet 1\n"
    "program p : One -> One\n"
    "    sample mu <- MultivariateNormal(loc=m0, scale_tril=L)\n"
    "    return mu\n"
    "export p\n"
)


def test_the_mean_and_factor_share_one_open_extent_when_nothing_fixes_it() -> None:
    module = lower_qvr_to_qiec(parse(_UNSIZED))
    (entry,) = module.entries
    types = {
        parameter.name: tensor_shape(parameter.type) for parameter in entry.parameters
    }
    loc, factor = types["m0"], types["L"]
    assert loc is not None and factor is not None
    assert factor[1] == (loc[1][0], loc[1][0])
    assert [binder.name for binder in entry.telescope] == ["m0_extent"]


def test_the_compiled_runtime_refuses_a_factor_only_the_data_size() -> None:
    with pytest.raises(CompileError, match=r"fixes no event size the compiled runtime"):
        _ = loads(_UNSIZED).morphism


def test_data_of_another_size_is_refused_on_both_paths() -> None:
    program = loads(_source("sample_mvn_host_loc"))
    data = {"m0": torch.tensor([0.3, -0.4, 0.2]), "y": torch.tensor([0.9, -0.6])}
    with pytest.raises(
        ValueError,
        match=r"host datum 'm0' has shape \(3,\), but the checked program reads "
        r"it at type Tensor\[Real\]\(\[2\]\)",
    ):
        program.morphism.log_joint(
            torch.zeros(1, 1), {**data, "mu": torch.tensor([0.2, 0.1])}
        )
    with pytest.raises(Exception, match=r"m0"):
        program.run(
            "mvn_host_loc",
            data={name: _nested(value) for name, value in data.items()},
            sites={"mu": (0.2, 0.1)},
        )


def test_a_datum_read_at_two_sizes_is_refused_naming_both() -> None:
    source = (
        "object One : FinSet 1\n"
        "object Two : FinSet 2\n"
        "object Three : FinSet 3\n"
        "program p : One -> One\n"
        "    sample a <- MultivariateNormal(loc=m0, scale_tril=L) [over=Two]\n"
        "    sample b <- MultivariateNormal(loc=m1, scale_tril=L) [over=Three]\n"
        "    return a\n"
        "export p\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _ = loads(source).qiec
    message = str(caught.value)
    assert "reads its 'scale_tril' from 'L', an axis of extent 2" in message
    assert "the sample's axis it spans has extent 3" in message
    assert ":6:" in message


def test_an_open_extent_meeting_another_event_size_is_refused() -> None:
    source = (
        "object One : FinSet 1\n"
        "object Two : FinSet 2\n"
        "object Row : FinSet 4\n"
        "program p : One -> One\n"
        "    sample mu <- MultivariateNormal(loc=m0, scale_tril=L) [over=Two]\n"
        "    observe y : Row <- MultivariateNormal(loc=mu, scale_tril=[[1.0, 0.0, "
        "0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])\n"
        "    return mu\n"
        "export p\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _ = loads(source).qiec
    assert ":6:" in str(caught.value)
