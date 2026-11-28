"""Independent KKT checks for tree-group penalized n-gram projection."""

from math import sqrt

import pytest
import torch
from torch import nn

from quivers.core.relations import NgramFamily
from quivers.core.relations.projection import _prefix_indices, _prefix_prox


def test_binary_prefix_regularizer_matches_the_composite_kkt_solution_and_derivative():
    family = NgramFamily(((), (0,)), order=1)
    logits = nn.Parameter(torch.tensor([0.25, 0.75], dtype=torch.float64).log())
    result = family.project_penalized(logits, penalty=0.05, tolerance=1e-12)
    assert result.status == "converged"
    assert result.optimality_residual <= 1e-12
    assert result.features == ((0,),)
    # Both the empty prefix and the single-symbol prefix contain this
    # coefficient. The stationarity equation is p - 3/4 + 2*penalty = 0.
    fitted = result.log_probabilities.exp()
    assert fitted.tolist() == pytest.approx([0.35, 0.65], abs=2e-12)
    assert result.regularization.item() == pytest.approx(
        0.10 * result.coefficients.abs().item()
    )
    (actual,) = torch.autograd.grad(fitted[1], logits)
    (expected,) = torch.autograd.grad(logits.softmax(dim=0)[1] - 0.10, logits)
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    collapsed = family.project_penalized(logits, penalty=0.20)
    assert collapsed.status == "converged"
    assert collapsed.coefficients.tolist() == [0.0]
    assert collapsed.log_probabilities.exp().tolist() == [0.5, 0.5]
    limited = family.project_penalized(logits, penalty=0.05, max_iterations=0)
    assert limited.status == "limit" and limited.optimality_residual > 0


def test_nested_group_proximal_map_matches_one_exact_hand_shrink():
    features = ((0,), (0, 1), (1,))
    groups = tuple(torch.tensor(group) for group in _prefix_indices(features))
    origin = torch.tensor([3.0, 4.0, 12.0], dtype=torch.float64)
    actual = _prefix_prox(origin, groups, 1.0)
    small = 3 * (1 - 1 / sqrt(18))
    root_scale = 1 - 1 / sqrt(2 * small * small + 121)
    assert actual.tolist() == pytest.approx(
        [root_scale * small, root_scale * small, root_scale * 11]
    )


def test_penalty_uses_the_full_ngram_coordinates_even_if_the_family_has_a_gauge():
    family = NgramFamily(((0,), (1,)), order=1)
    assert len(family.features) == 1
    source = torch.tensor([0.25, 0.75], dtype=torch.float64).log()
    result = family.project_penalized(source, penalty=0.05, tolerance=1e-11)
    assert result.status == "converged"
    assert result.features == ((0,), (1,))
    assert result.coefficients[0].item() == pytest.approx(
        -result.coefficients[1].item()
    )
