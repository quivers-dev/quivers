"""All guides share flat coordinates and structured natural-scale events."""

import math

import pytest
import torch

from quivers.continuous import Draw, Euclidean, MonadicProgram, PlateDraw
from quivers.continuous.families import ConditionalNormal
from quivers.dsl import loads
from quivers.effects import subsample
from quivers.inference import AmortizedGuide, AutoNormalGuide
from quivers.inference.guides.auto_structured import AutoStructured
from quivers.inference.guides.delta import AutoDeltaGuide
from quivers.inference.guides.flow import AutoIAFGuide, AutoNeuralSplineGuide
from quivers.inference.guides.laplace import AutoLaplaceApproximation
from quivers.inference.guides.multivariate_normal import (
    AutoLowRankMultivariateNormalGuide,
    AutoMultivariateNormalGuide,
)


@pytest.fixture(params=[(2, 3, False), (1, 3, False), (1, 1, False), (2, 3, True)])
def matrix_model(request):
    rows, columns, plate = request.param
    binding = " : Batch" if plate else ""
    source = f"""
object Rows : FinSet {rows}
object Columns : FinSet {columns}
object Batch : FinSet 2
object One : FinSet 1
program matrices : One -> One
    sample value{binding} <- MatrixNormal(loc={torch.zeros(rows, columns).tolist()}, row_covariance={torch.eye(rows).tolist()}, col_covariance={torch.eye(columns).tolist()}) [over=[Rows, Columns]]
    return value
export matrices
"""
    shape = (2, rows, columns) if plate else (rows, columns)
    return loads(source).morphism, shape, plate


_GUIDES = [
    AutoNormalGuide,
    AutoDeltaGuide,
    AutoMultivariateNormalGuide,
    lambda m, o: AutoLowRankMultivariateNormalGuide(m, o, rank=1),
    AutoLaplaceApproximation,
    lambda m, o: AutoStructured(m, o, conditionals="normal"),
    lambda m, o: AutoStructured(m, o, conditionals="mvn"),
    lambda m, o: AutoStructured(m, o, conditionals="delta"),
    lambda m, o: AutoIAFGuide(m, o, num_flows=1),
    lambda m, o: AutoNeuralSplineGuide(m, o, num_flows=1, num_bins=4),
]


@pytest.mark.parametrize("factory", _GUIDES)
def test_every_guide_matrix_event_density_and_coordinate_gradient(
    matrix_model, factory
):
    model, shape, plate = matrix_model
    if math.prod(shape) == 1 and factory in _GUIDES[-2:]:
        with pytest.raises(ValueError, match=">= 2.*unconstrained"):
            factory(model, set())
        return
    guide = factory(model, set())
    x = torch.zeros(1, 1)
    values = guide.rsample(x)
    assert values["value"].shape == (shape if plate else (1, *shape))
    density = guide.log_prob(x, values)
    assert density.shape == (1,)
    assert torch.isfinite(density).all()
    value = values["value"].detach().requires_grad_()
    actual = model.log_joint(x, {"value": value}).sum()
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(value).sum()
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual, value)
    torch.testing.assert_close(gradient, -value)
    (density.sum() + values["value"].square().sum()).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in guide.parameters()
    )


def test_mean_field_matrix_density_and_initial_center(matrix_model):
    model, shape, plate = matrix_model
    center = torch.linspace(-0.3, 0.4, math.prod(shape)).reshape(shape)
    guide = AutoNormalGuide(
        model, set(), init_scale=0.4, initial_values={"value": center}
    )
    values = center if plate else center.unsqueeze(0)
    actual = guide.log_prob(torch.zeros(1, 1), {"value": values})
    expected = torch.distributions.Normal(center, 0.4).log_prob(center).sum().reshape(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(guide._loc("value").reshape(shape), center)


def test_matrix_laplace_precision_is_full_coordinate_identity(matrix_model):
    model, shape, _ = matrix_model
    guide = AutoLaplaceApproximation(model, set())
    guide.fit_hessian(model, torch.zeros(1, 1), {}, jitter=1e-5)
    expected = torch.eye(math.prod(shape)) * (1.0 + 1e-5)
    torch.testing.assert_close(guide._scale_tril @ guide._scale_tril.T, expected)
    values = guide.rsample(torch.zeros(1, 1))
    expected_density = torch.distributions.MultivariateNormal(
        guide.map_z, covariance_matrix=expected
    ).log_prob(values["value"].reshape(-1))
    torch.testing.assert_close(
        guide.log_prob(torch.zeros(1, 1), values), expected_density.reshape(1)
    )


def test_amortized_matrix_rows_and_independent_density(matrix_model):
    model, shape, plate = matrix_model
    width = math.prod(shape[1:] if plate else shape)
    encoder = ConditionalNormal(
        Euclidean(name="Datum", dim=1), Euclidean(name="Flat", dim=width)
    )
    with torch.no_grad():
        encoder.param_source.linear.weight.zero_()
        encoder.param_source.linear.bias[:width].fill_(0.2)
        encoder.param_source.linear.bias[width:].fill_(math.log(0.7))
    leading = (2,) if plate else (1,)
    data = {"datum": torch.zeros(*leading, 1)}
    guide = AmortizedGuide(
        model, set(), {"value": (encoder, "datum")}, observations=data
    )
    x = torch.zeros(1, 1)
    values = guide.rsample(x)
    assert values["value"].shape == (shape if plate else (1, *shape))
    actual = guide.log_prob(x, values)
    expected = (
        torch.distributions.Normal(0.2, 0.7).log_prob(values["value"]).sum().reshape(1)
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()
    )


@pytest.mark.parametrize("per_run", [False, True])
def test_amortized_structured_positive_plates_preserve_jacobian_and_subsampling(
    per_run,
):
    model = loads("""
object Row : FinSet 2
object Column : FinSet 3
object One : FinSet 1
program p : One -> One
    sample value : Row <- HalfNormal(scale=1.0) [over=[Column,One]]
    return value
export p
""").morphism
    encoder = ConditionalNormal(
        Euclidean(name="Datum", dim=1), Euclidean(name="Z", dim=1)
    )
    with torch.no_grad():
        encoder.param_source.linear.weight.zero_()
        encoder.param_source.linear.bias.copy_(torch.tensor([0.2, math.log(0.7)]))
    leading = (2, 2, 3, 1) if per_run else (2, 3, 1)
    guide = AmortizedGuide(
        model,
        set(),
        {"value": (encoder, "datum")},
        observations={"datum": torch.zeros(*leading, 1)},
    )
    x = torch.zeros(2, 1)
    values = guide.rsample(x)
    assert values["value"].shape == leading
    natural = values["value"]
    element_scores = (
        torch.distributions.Normal(0.2, 0.7).log_prob(natural.log()) - natural.log()
    )
    expected = element_scores.reshape(-1, 6).sum(-1).expand(2)
    torch.testing.assert_close(guide.log_prob(x, values), expected)
    rows, columns = torch.tensor([1]), torch.tensor([0, 2])
    selected = element_scores.reshape(-1, 2, 3, 1)[:, rows][:, :, columns]
    expected_subset = 3 * selected.reshape(-1, 2).sum(-1).expand(2)
    with subsample("Row", rows), subsample("Column", columns):
        actual = guide.log_prob(x, values)
    torch.testing.assert_close(actual, expected_subset)
    actual.sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()
    )


@pytest.mark.parametrize("factory", _GUIDES[:-2])
@pytest.mark.parametrize(
    "law",
    [
        "Dirichlet(concentration=[1.0,2.0,3.0]) [over=Component]",
        "Wishart(df=4.0,scale_tril=[[1.0,0.0],[0.0,1.0]]) [over=[Dim,Dim]]",
        "LKJCholesky(concentration=2.0) [over=[Dim,Dim]]",
    ],
)
def test_nonlinear_support_events_have_finite_density_and_parameter_gradients(
    factory, law
):
    model = loads(f"""
object One : FinSet 1
object Dim : FinSet 2
object Component : FinSet 3
program p : One -> One
    sample value <- {law}
    return value
export p
""").morphism
    guide = factory(model, set())
    x = torch.zeros(1, 1)
    values = guide.rsample(x)
    site = guide._registry["value"]
    assert values["value"].shape == (1, *site.constrained_shape)
    assert site.support.check(values["value"]).all()
    density = guide.log_prob(x, values)
    assert torch.isfinite(density).all()
    (density.sum() + values["value"].square().sum()).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in guide.parameters()
    )


@pytest.mark.parametrize("plate", [False, True])
def test_positive_definite_density_reduces_event_jacobian_once_and_keeps_runs(plate):
    model = loads("""
object Row : FinSet 2
object Dim : FinSet 2
object One : FinSet 1
program p : One -> One
    sample value <- Wishart(df=4.0,scale_tril=[[1.0,0.0],[0.0,1.0]]) [over=[Dim,Dim]]
    return value
export p
""").morphism
    if plate:
        step = model.steps[0]
        assert isinstance(step, Draw)
        model = MonadicProgram(
            model.domain,
            model.codomain,
            steps=[Draw(names=("value",), morphism=PlateDraw(2, step.morphism))],
            return_vars=("value",),
        )
    guide = AutoNormalGuide(model, set(), init_scale=0.4)
    z = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]], requires_grad=True)
    zero = torch.zeros_like(z[:, 0])
    factor = torch.stack((z[:, 0].exp(), zero, z[:, 1], z[:, 2].exp()), -1).reshape(
        2, 2, 2
    )
    value = factor @ factor.transpose(-1, -2)
    actual = guide.log_prob(torch.zeros(2, 1), {"value": value})
    # L -> LL' contributes 2 log2 + 2 logL00 + logL11;
    # exponentiating the diagonal adds logL00+logL11.
    jacobian = 2 * math.log(2) + 3 * z[:, 0] + 2 * z[:, 2]
    per_event = torch.distributions.Normal(0.0, 0.4).log_prob(z).sum(-1) - jacobian
    expected = per_event.sum().expand(2) if plate else per_event
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual.sum(), z, create_graph=True)
    multiplicity = 2 if plate else 1
    expected_gradient = multiplicity * (-z / 0.4**2 - torch.tensor([3.0, 0.0, 2.0]))
    torch.testing.assert_close(gradient, expected_gradient)
    (curvature,) = torch.autograd.grad(gradient.sum(), z)
    torch.testing.assert_close(curvature, torch.full_like(z, -multiplicity / 0.4**2))
