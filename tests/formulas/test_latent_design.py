"""Retained latent designs preserve penalty laws, geometry, and sparse maps."""

import pytest
import torch

from quivers.formulas._latent_design import DenseDesign, IndexedDesign, PenaltyBasis
from quivers.modeling import Model, MCMCMethod, VariationalMethod
from quivers.inference import HMCKernel, MCMC, NUTSKernel


def _training_basis():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    return torch.column_stack((torch.ones_like(x), x, x.square() - x.square().mean()))


def test_penalty_nullspace_and_centering_have_independent_closed_form_covariance():
    design = _training_basis()
    basis = PenaltyBasis.fit(
        design, torch.diag(torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64))
    )
    fixed, random = basis.evaluate(design)
    x = design[:, 1]
    q = design[:, 2]
    f = torch.tensor(fixed.values)
    z = torch.tensor(random.values)
    torch.testing.assert_close(f.double() @ f.double().T, x[:, None] * x[None, :])
    torch.testing.assert_close(z.double() @ z.double().T, q[:, None] * q[None, :] / 4)
    assert fixed.width == random.width == 1
    assert basis.constraint == (1.0, 0.0, 0.0)
    fresh = torch.tensor([[1.0, 2.0, 3.5], [1.0, -2.0, 3.5]], dtype=torch.float64)
    xf, zf = basis.evaluate(fresh)
    # Reusing fitted centering permits nonzero fresh means; recentering would
    # erase this legitimate quadratic extrapolation.
    assert abs(torch.tensor(zf.values).mean().item()) == pytest.approx(1.75)
    assert basis.evaluate(design) == (fixed, random)
    assert xf.width == 1


def test_dense_and_indexed_contractions_preserve_chains_draws_and_all_derivatives():
    membership = IndexedDesign.fit(
        (("a", "b"), ("c", "a"), ("b", "b")),
        torch.tensor([[1.0, 3.0], [2.0, 1.0], [4.0, 5.0]], dtype=torch.float64),
    )
    dense = DenseDesign(((0.25, 0.75, 0.0), (1 / 3, 0.0, 2 / 3), (0.0, 1.0, 0.0)))
    coefficients = (
        torch.arange(12, dtype=torch.float64).reshape(2, 2, 3).requires_grad_()
    )
    actual = membership.apply(coefficients)
    expected = dense.apply(coefficients)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), coefficients)[0],
        torch.autograd.grad(expected.square().sum(), coefficients)[0],
    )
    assert torch.autograd.gradcheck(
        lambda x: membership.apply(x).square(), (coefficients,)
    )
    assert torch.autograd.gradgradcheck(
        lambda x: dense.apply(x).square(), (coefficients,)
    )
    empty = DenseDesign((), coordinate_width=3)
    assert empty.apply(coefficients).shape == (2, 2, 0)
    with pytest.raises(ValueError, match="real design coordinates"):
        dense.apply(torch.tensor([1, 2, 3]))


def test_membership_new_rows_reuse_union_and_preserve_authored_signed_weights():
    training = IndexedDesign.fit((("b", "a"), ("c", "b")))
    fresh = IndexedDesign.fit(
        (("c", "a"),), torch.tensor([[2.0, -1.0]]), scale=False, levels=training.levels
    )
    assert fresh.levels == ("a", "b", "c")
    assert fresh.apply(torch.tensor([1.0, 10.0, 4.0])).item() == 7.0
    with pytest.raises(ValueError, match="unknown membership"):
        IndexedDesign.fit((("d", "a"),), levels=training.levels)
    with pytest.raises(ValueError, match="negative membership"):
        IndexedDesign.fit((("c", "a"),), torch.tensor([[2.0, -1.0]]))
    with pytest.raises(ValueError, match="positive row totals"):
        IndexedDesign.fit((("c", "a"),), torch.zeros(1, 2))
    huge = IndexedDesign.fit(
        (("c", "a"),), torch.tensor([[1e308, 1e308]], dtype=torch.float64)
    )
    assert huge.weights == ((0.5, 0.5),)


@pytest.mark.parametrize(
    "bad",
    [torch.tensor([[0.0, 1.0], [0.0, 0.0]]), torch.diag(torch.tensor([-1.0, 1.0]))],
)
def test_penalty_rejects_nonquadratic_or_negative_geometry(bad):
    with pytest.raises(ValueError, match="symmetric|semidefinite"):
        PenaltyBasis.fit(torch.eye(2), bad, center=False)


def test_penalty_has_no_fake_regularization_of_unidentified_null_directions():
    with pytest.raises(ValueError, match="unpenalized.*not identified"):
        PenaltyBasis.fit(torch.ones(3, 2), torch.zeros(2, 2), center=False)
    with pytest.raises(ValueError, match="require gradients"):
        PenaltyBasis.fit(torch.eye(2, requires_grad=True), torch.eye(2))


SOURCE = """object Row:FinSet _
object Basis:FinSet 2
object One:FinSet 1
object Out:Real 1
program p(Z:Tensor[Real]([Row,Basis]),y:Tensor[Real]([Row])) : One -> Out
    sample scale <- LogNormal(loc=-0.2,scale=0.5)
    sample z:Basis <- Normal(loc=0.0,scale=1.0)
    let coefficients = scale*z
    let eta = matmul(Z,coefficients)
    observe y:Row <- Normal(loc=eta,scale=0.7)
    return scale
export p
"""


def test_gaussian_design_reuses_source_joint_and_all_head_gradients():
    matrix = torch.tensor([[1.0, -0.2], [0.1, 0.7], [-0.5, 0.4]], dtype=torch.float64)
    y = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    bound = Model(SOURCE).bind({"Z": matrix, "y": y})
    point = torch.tensor([0.8, 0.2, -0.3], dtype=torch.float64, requires_grad=True)

    def actual(v):
        return bound.program.log_joint(
            bound.input, {**bound.data, "scale": v[0], "z": v[1:]}
        ).sum()

    def oracle(v):
        return (
            torch.distributions.LogNormal(v.new_tensor(-0.2), 0.5).log_prob(v[0])
            + torch.distributions.Normal(0.0, 1.0).log_prob(v[1:]).sum()
            + torch.distributions.Normal(matrix @ (v[0] * v[1:]), 0.7).log_prob(y).sum()
        )

    torch.testing.assert_close(actual(point), oracle(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
    )


@pytest.mark.parametrize("algorithm", ["svi", "hmc", "nuts"])
def test_existing_inference_and_fresh_design_prediction_need_no_term_specific_loop(
    algorithm,
):
    matrix = torch.tensor([[1.0, -0.2], [0.1, 0.7], [-0.5, 0.4]])
    model = Model(SOURCE)
    if algorithm == "svi":
        method = VariationalMethod(num_steps=2, seed=4)
    else:
        kernel = (
            HMCKernel(step_size=0.03, num_steps=2)
            if algorithm == "hmc"
            else NUTSKernel(step_size=0.03, max_tree_depth=2)
        )
        method = MCMCMethod(
            sampler=MCMC(kernel, num_warmup=1, num_samples=2, num_chains=1, seed=4)
        )
    fitted = model.fit(
        {"Z": matrix, "y": torch.tensor([0.3, -0.2, 0.1])}, method=method
    )
    fresh = {"Z": matrix[:2]}
    predicted = fitted.predict(data=fresh, num_samples=2, seed=8)
    assert torch.isfinite(predicted["y"]).all()
    assert predicted["y"].shape[-1] == 2
