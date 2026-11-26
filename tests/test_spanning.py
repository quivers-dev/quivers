"""Directed spanning trees: conventions, masks, batches, and the closed form."""

from __future__ import annotations

import math
from fractions import Fraction

import didactic.api as dx
import pytest
import torch

from quivers.core import REAL, directed_tree, tree_marginals, tree_partition
from quivers.core.algebras import BOOLEAN, LOG_PROB, MARKOV
from tests.evidence.oracles import trees
from tests.evidence.spanning_controls import WEIGHTS, rational_weights, tensor_of

ROOT = 1


def _matrix() -> torch.Tensor:
    """The asymmetric weights of the checks, in float64.

    Returns
    -------
    torch.Tensor
        The ``4 x 4`` weights.
    """
    return tensor_of(rational_weights(WEIGHTS), 4)


def test_orientation_is_parent_then_child() -> None:
    """Require ``W[x, y]`` to weight the edge from ``x`` to ``y``."""
    weights = torch.full((2, 2), -math.inf, dtype=torch.float64)
    weights[0, 1] = -2.0
    assert float(tree_partition(weights, 0)) == -2.0
    assert math.isinf(float(tree_partition(weights, 1)))


def test_masks_remove_edges() -> None:
    """Require a mask to remove exactly its edges."""
    rationals = rational_weights(WEIGHTS)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0, 2] = mask[3, 0] = False
    kept = {edge: value for edge, value in rationals.items() if mask[edge]}
    expected = trees.partition(
        [t for t in trees.spanning_trees(4, ROOT) if all(e in kept for e in t)],
        kept,
        Fraction(0),
        Fraction(1),
    )
    found = tree_partition(_matrix(), ROOT, algebra=MARKOV, mask=mask)
    assert float(found) == pytest.approx(float(expected), rel=1e-13)


def test_batches_and_single_nodes() -> None:
    """Require a batch to give one partition per graph, and one node the unit."""
    batch = torch.stack((_matrix().log(), _matrix().t().log()))
    found = tree_partition(batch, ROOT)
    assert found.shape == (2,)
    assert float(found[0]) == pytest.approx(float(tree_partition(batch[0], ROOT)))
    assert float(tree_partition(torch.zeros(3, 1, 1), 0).abs().sum()) == 0.0


def test_refusals() -> None:
    """Require a semifield with zero sums, a bad root, and a bad shape to be refused."""
    with pytest.raises(TypeError):
        tree_partition(_matrix(), ROOT, algebra=REAL)
    with pytest.raises(ValueError):
        tree_partition(_matrix(), 4)
    with pytest.raises(ValueError):
        tree_partition(torch.zeros(3, 4), 0)
    with pytest.raises(dx.ValidationError):
        directed_tree(_matrix(), root=7)


def test_boolean_support() -> None:
    """Require the Boolean partition to say whether a tree exists."""
    weights = _matrix() > 0
    assert bool(tree_partition(weights, ROOT, algebra=BOOLEAN))
    weights[:, 3] = False
    assert not bool(tree_partition(weights, ROOT, algebra=BOOLEAN))


def test_the_determinant_cancels_where_elimination_does_not() -> None:
    """Require the documented ill-scaled graph: -59.6867 by elimination, -inf by slogdet."""
    weights = torch.full((3, 3), -math.inf, dtype=torch.float64)
    weights[0, 1], weights[0, 2] = -60.0, -61.0
    weights[1, 2], weights[2, 1] = 0.0, 0.0
    assert round(float(tree_partition(weights, 0)), 4) == -59.6867
    linear = weights.exp()
    laplacian = torch.diag(linear.sum(dim=0)) - linear
    assert math.isinf(float(torch.linalg.slogdet(laplacian[1:, 1:]).logabsdet))


def _closed_form(weights: torch.Tensor, root: int, *, printed: bool) -> torch.Tensor:
    """Edge marginals by the inverse of the Laplacian without the root.

    The Laplacian is ``L[y, y] = sum_x W[x, y]`` and ``L[x, y] = -W[x, y]``.
    The derivative of ``log det`` of it without the root is
    ``W[x, y] (inv[y, y] - inv[y, x])``; the printed equation 14 of
    Andrews, Eisner and Dredze (2012) has ``inv[x, x] - inv[x, y]``.

    Parameters
    ----------
    weights : torch.Tensor
        Linear weights, ``n x n``.
    root : int
        The root.
    printed : bool
        Whether to use the indices as printed.

    Returns
    -------
    torch.Tensor
        The marginals.
    """
    size = weights.shape[0]
    rest = [node for node in range(size) if node != root]
    laplacian = torch.diag(weights.sum(dim=0)) - weights
    inverse = torch.linalg.inv(laplacian[rest][:, rest])
    position = {node: index for index, node in enumerate(rest)}
    marginals = torch.zeros_like(weights)
    for x in range(size):
        for y in rest:
            if x == y:
                continue
            j = position[y]
            if x == root:
                marginals[x, y] = weights[x, y] * inverse[j, j]
                continue
            i = position[x]
            if printed:
                marginals[x, y] = weights[x, y] * (inverse[i, i] - inverse[i, j])
            else:
                marginals[x, y] = weights[x, y] * (inverse[j, j] - inverse[j, i])
    return marginals


def test_the_derivative_of_equation_12_and_the_printed_equation_14() -> None:
    """Require the marginals to be the derivative of eq. 12, not eq. 14 as printed."""
    weights = _matrix()
    rationals = rational_weights(WEIGHTS)
    enumerated = trees.edge_marginals(trees.spanning_trees(4, ROOT), rationals)
    expected = torch.zeros(4, 4, dtype=torch.float64)
    for (x, y), value in enumerated.items():
        expected[x, y] = float(value)
    found = tree_marginals(weights.log(), ROOT)
    assert torch.allclose(found, expected, atol=1e-13)
    assert torch.allclose(
        _closed_form(weights, ROOT, printed=False), expected, atol=1e-12
    )
    printed = _closed_form(weights, ROOT, printed=True)
    assert float((printed - expected).abs().max()) > 0.1


def test_marginals_sum_to_one_per_child_and_differentiate_again() -> None:
    """Require one parent per child, and a second derivative through the marginals."""
    weights = _matrix().log().requires_grad_()
    marginals = directed_tree(weights, root=ROOT).edge_marginals()
    sums = marginals.sum(dim=0)
    assert torch.allclose(sums[[0, 2, 3]], torch.ones(3, dtype=torch.float64))
    assert float(sums[ROOT]) == 0.0
    (second,) = torch.autograd.grad(marginals[0, 2], weights)
    assert torch.isfinite(second).all()
    assert float(directed_tree(weights, root=ROOT).partition()) == pytest.approx(
        float(tree_partition(weights, ROOT, algebra=LOG_PROB))
    )
