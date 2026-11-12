"""Tests for shape-aware change-of-base via :class:`MorphismTransformation`.

Verifies:

* ``Softmax(axis)`` produces a row-stochastic Markov kernel.
* ``L1Normalize`` produces unit-sum rows over the chosen axis.
* ``L2Normalize`` produces unit-L2-norm rows.
* ``BayesInvert(prior)`` produces a kernel whose rows sum to 1 and
  satisfies the disintegration equation
  ``f^{-1}_pi(a | b) * marginal_b(b) = f(b | a) * prior(a)``.
* ``Morphism.change_base`` dispatches between
  ``CarrierMap`` (pointwise) and
  ``MorphismTransformation`` (shape-aware) correctly.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.morphism_transformations import (
    BayesInvert,
    L1Normalize,
    L2Normalize,
    MorphismTransformation,
    Softmax,
)
from quivers.core.morphisms import observed
from quivers.core.objects import FinSet
from quivers.core.algebras import (
    PRODUCT_FUZZY,
    REAL,
)
from quivers.core.algebras import MARKOV


def _real_morphism(A: FinSet, B: FinSet) -> object:
    tensor = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [2.0, 3.0, 1.0, 0.5],
        ]
    )
    assert tensor.shape == (A.cardinality, B.cardinality)
    return observed(A, B, tensor, algebra=REAL)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def test_softmax_produces_row_stochastic_kernel() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    g = f.change_base(Softmax(B, source=REAL))
    assert torch.allclose(g.tensor.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert (g.tensor > 0).all()


def test_softmax_target_algebra_is_markov() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    g = f.change_base(Softmax(B, source=REAL))
    assert g.algebra is MARKOV


# ---------------------------------------------------------------------------
# L1 / L2 normalisation
# ---------------------------------------------------------------------------


def test_l1_normalize_row_sums_to_one() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    g = f.change_base(L1Normalize(B, source=REAL))
    assert torch.allclose(g.tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_l2_normalize_row_norm_one() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    g = f.change_base(L2Normalize(B, source=REAL))
    norms = g.tensor.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_l2_normalize_preserves_source_algebra() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    g = f.change_base(L2Normalize(B, source=REAL))
    assert type(g.algebra) is type(REAL)


# ---------------------------------------------------------------------------
# Bayes inversion
# ---------------------------------------------------------------------------


def _markov_kernel(A: FinSet) -> object:
    k = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )
    return observed(A, A, k, algebra=MARKOV)


def test_bayes_invert_rows_sum_to_one() -> None:
    A = FinSet(name="A", cardinality=3)
    f = _markov_kernel(A)
    prior = torch.tensor([0.5, 0.3, 0.2])
    g = f.change_base(BayesInvert(prior))
    assert torch.allclose(g.tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_bayes_invert_swaps_domain_and_codomain() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=3)
    k = torch.eye(3)  # identity kernel; doesn't matter
    f = observed(A, B, k, algebra=MARKOV)
    prior = torch.tensor([0.5, 0.3, 0.2])
    g = f.change_base(BayesInvert(prior))
    assert g.domain is B
    assert g.codomain is A


def test_bayes_invert_satisfies_disintegration_equation() -> None:
    """f^{-1}_pi(a | b) * sum_a' pi(a') f(b | a') = pi(a) * f(b | a)."""
    A = FinSet(name="A", cardinality=3)
    f = _markov_kernel(A)
    prior = torch.tensor([0.5, 0.3, 0.2])
    g = f.change_base(BayesInvert(prior))

    f_kernel = f.tensor  # shape (a, b)
    inv = g.tensor  # shape (b, a)
    joint = prior.unsqueeze(-1) * f_kernel  # (a, b)
    marg_b = joint.sum(dim=0)  # (b,)
    rhs = joint  # (a, b) -- prior(a) * f(b | a)
    lhs = inv.t() * marg_b.unsqueeze(0)  # (a, b)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_bayes_invert_rejects_unnormalized_prior() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        BayesInvert(torch.tensor([0.5, 0.3]))


def test_bayes_invert_rejects_negative_prior() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BayesInvert(torch.tensor([-0.1, 0.6, 0.5]))


def test_bayes_invert_rejects_multivariate_prior() -> None:
    with pytest.raises(ValueError, match="1-D"):
        BayesInvert(torch.tensor([[0.5, 0.3], [0.1, 0.1]]))


# ---------------------------------------------------------------------------
# Dispatch in change_base
# ---------------------------------------------------------------------------


def test_change_base_dispatches_on_algebra_homomorphism() -> None:
    """Pointwise CarrierMap still works through
    change_base."""
    from quivers.core.algebra_morphisms import EXPECTATION

    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    # A Markov kernel (rows sum to 1) becomes a fuzzy morphism via Expectation.
    tensor = torch.tensor(
        [
            [0.5, 0.2, 0.2, 0.1],
            [0.1, 0.5, 0.3, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    f = observed(A, B, tensor, algebra=MARKOV)
    g = f.change_base(EXPECTATION)
    assert g.algebra is PRODUCT_FUZZY
    assert torch.allclose(g.tensor, tensor.clamp(0, 1))


def test_change_base_rejects_wrong_source() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)  # algebra=REAL
    # Softmax with source=PRODUCT_FUZZY, but f is REAL → should fail.
    with pytest.raises(TypeError, match="algebra"):
        f.change_base(Softmax(B, source=PRODUCT_FUZZY))


def test_change_base_rejects_non_transformation() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = _real_morphism(A, B)
    with pytest.raises(TypeError, match="MorphismTransformation"):
        f.change_base("not a transformation")


def test_morphism_transformation_is_abc() -> None:
    """Direct instantiation should fail."""
    with pytest.raises(TypeError):
        MorphismTransformation()  # type: ignore[abstract]
