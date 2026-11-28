"""Independent moment, KL, and gradient oracles for fixed n-gram projection."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.relations import NgramFamily

_SUPPORT = ((), (0,), (1,), (0, 1))


def test_unigram_projection_has_the_independent_bernoulli_distribution():
    family = NgramFamily(_SUPPORT, order=1)
    source = nn.Parameter(
        torch.tensor([1 / 2, 1 / 8, 1 / 8, 1 / 4], dtype=torch.float64).log()
    )
    projected = family.project(source, tolerance=1e-12)
    assert projected.status == "converged"
    assert family.features == ((0,), (1,))
    probabilities = projected.log_probabilities.exp()
    assert probabilities.tolist() == pytest.approx([25 / 64, 15 / 64, 15 / 64, 9 / 64])
    assert projected.target_moments.tolist() == pytest.approx([3 / 8, 3 / 8])
    assert projected.fitted_moments.tolist() == pytest.approx([3 / 8, 3 / 8])
    assert projected.moment_residual <= 1e-12
    target = source.softmax(dim=0)
    (actual_gradient,) = torch.autograd.grad(probabilities[3], source)
    # Union of independent Bernoulli indicators gives an independent
    # closed form, with no use of the runtime feature matrix or solver.
    entry = (target[1] + target[3]) * (target[2] + target[3])
    (expected_gradient,) = torch.autograd.grad(entry, source)
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10)
    expected_kl = (
        target
        * (
            target.log()
            - torch.tensor(
                [25 / 64, 15 / 64, 15 / 64, 9 / 64], dtype=torch.float64
            ).log()
        )
    ).sum()
    assert projected.divergence.item() == pytest.approx(expected_kl.item())


def test_projection_with_a_cavity_recovers_its_factorized_moment_fit():
    family = NgramFamily(_SUPPORT, order=1)
    cavity = torch.tensor([4 / 9, 2 / 9, 2 / 9, 1 / 9], dtype=torch.float64).log()
    # Multiplication by a separable site has exact unit-gram log odds,
    # and should be recovered independently from the supplied cavity.
    delta = torch.tensor([0.0, 0.25, -0.75, -0.5], dtype=torch.float64)
    fit = family.project(cavity + delta, base_log_weights=cavity, tolerance=1e-12)
    assert fit.status == "converged"
    assert fit.coefficients.tolist() == pytest.approx([0.25, -0.75], abs=1e-10)
    assert fit.divergence.item() == pytest.approx(0.0, abs=1e-14)


def test_exact_affine_basis_removes_redundant_count_columns():
    family = NgramFamily(((0,), (1,)), order=3)
    assert len(family.features) == 1
    source = torch.tensor(
        [float(Fraction(1, 4)), float(Fraction(3, 4))], dtype=torch.float64
    ).log()
    fitted = family.project(source, tolerance=1e-12)
    assert fitted.status == "converged"
    assert torch.allclose(fitted.log_probabilities, source)

    singleton = NgramFamily(((),), order=1).project(torch.zeros(1, dtype=torch.float64))
    assert singleton.coefficients.shape == (0,)
    assert singleton.moment_residual == 0
    assert singleton.log_probabilities.tolist() == [0.0]


def test_iteration_limit_and_unsupported_boundary_are_visible():
    family = NgramFamily(_SUPPORT, order=1)
    source = torch.tensor([1.0, -1.0, -1.0, 0.0], dtype=torch.float64)
    result = family.project(source, max_iterations=0, tolerance=1e-10)
    assert result.status == "limit"
    assert result.moment_residual > 0
    assert result.log_probabilities.exp().tolist() == [0.25] * 4
    source[0] = -torch.inf
    with pytest.raises(ValueError, match="full support"):
        family.project(source)
    with pytest.raises(TypeError, match="float64"):
        family.project(torch.zeros(4))
