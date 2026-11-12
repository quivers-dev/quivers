"""Sum-product semiring family: Real, Probability, Counting.

These three algebras share the categorical structure of a
sum-product semiring (``⊕ = +``, ``⊗ = ·``) but differ in their
underlying lattice and clamping behavior:

* :class:`RealAlgebra`: entries in :math:`\\mathbb{R}` with no
  clamping. The canonical numeric semiring.
* :class:`BoundedSumProductAlgebra`: entries in ``[0, 1]`` with clamp
  on every op so the unit-interval invariant is preserved.
* :class:`CountingAlgebra`: entries in the non-negative
  integers (held as float for autograd compatibility) with
  ``+`` and ``·`` operations.

These three are distinct from the existing ``MarkovAlgebra``
(which constrains rows to sum to 1), :class:`ProductFuzzyAlgebra` (whose
join is noisy-OR), and :class:`LogProbAlgebra` (whose
computation lives in log-space).

The tests verify:

1. Each algebra's tensor / join / meet / negate / identity-tensor
   operations satisfy their documented contract.
2. Composition under a module-level
   ``composition <name> [level=algebra]`` declaration works
   end-to-end: ``>>`` composes in the declared algebra.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.core.algebras import (
    COUNTING,
    LOG_PROB,
    PROBABILITY,
    REAL,
)
from quivers.core.morphisms import LatentMorphism, ObservedMorphism
from quivers.core.objects import FinSet


class TestLogProbImpossibleReduction:
    def test_all_negative_infinity_has_zero_gradient(self) -> None:
        values = torch.full((2, 3), -float("inf"), requires_grad=True)

        result = LOG_PROB.join(values, dim=1)
        result.sum().backward()

        assert torch.isneginf(result).all()
        torch.testing.assert_close(values.grad, torch.zeros_like(values))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_mixed_rows_match_logsumexp_and_keep_finite_gradients(
        self, dtype: torch.dtype
    ) -> None:
        values = torch.tensor(
            [[0.0, -1.0, -float("inf")], [-float("inf")] * 3],
            dtype=dtype,
            requires_grad=True,
        )

        result = LOG_PROB.join(values, dim=1)
        result.sum().backward()

        torch.testing.assert_close(result[0], torch.logsumexp(values.detach()[0], 0))
        assert torch.isneginf(result[1])
        torch.testing.assert_close(
            values.grad[0], torch.softmax(values.detach()[0], dim=0)
        )
        torch.testing.assert_close(values.grad[1], torch.zeros(3, dtype=dtype))
        assert torch.isfinite(values.grad).all()

    def test_tuple_dimensions_preserve_unreduced_shape(self) -> None:
        values = torch.full((2, 3, 4), -float("inf"), requires_grad=True)
        values.data[1, 0, 0] = 0.0

        result = LOG_PROB.join(values, dim=(0, 2))
        result.sum().backward()

        assert result.shape == (3,)
        assert result[0] == 0.0
        assert torch.isneginf(result[1:]).all()
        assert torch.isfinite(values.grad).all()


# ---------------------------------------------------------------------------
# Algebraic contract for RealAlgebra
# ---------------------------------------------------------------------------


class TestRealAlgebra:
    def test_name_is_real(self) -> None:
        assert REAL.name == "Real"

    def test_tensor_op_is_product(self) -> None:
        a = torch.tensor([1.0, -2.0, 3.0])
        b = torch.tensor([2.0, 0.5, -1.0])
        assert torch.allclose(REAL.tensor_op(a, b), a * b)

    def test_join_is_sum(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(REAL.join(t, dim=0), t.sum(dim=0))
        assert torch.allclose(REAL.join(t, dim=1), t.sum(dim=1))
        # Multi-axis join.
        assert torch.allclose(REAL.join(t, dim=(0, 1)), t.sum())

    def test_meet_is_min(self) -> None:
        t = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        assert torch.allclose(REAL.meet(t, dim=0), torch.tensor([1.0, -2.0]))

    def test_negate_is_additive_inverse(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.0])
        assert torch.allclose(REAL.negate(t), -t)

    def test_unit_and_zero(self) -> None:
        assert REAL.unit == 1.0
        assert REAL.zero == 0.0

    def test_identity_tensor_is_eye(self) -> None:
        I = REAL.identity_tensor((4,))
        assert torch.allclose(I, torch.eye(4))

    def test_identity_tensor_multidim(self) -> None:
        I = REAL.identity_tensor((2, 3))
        # 4-axis tensor with 1.0 at (i, j, i, j).
        expected = torch.zeros(2, 3, 2, 3)
        for i in range(2):
            for j in range(3):
                expected[i, j, i, j] = 1.0
        assert torch.allclose(I, expected)


# ---------------------------------------------------------------------------
# Algebraic contract for BoundedSumProductAlgebra
# ---------------------------------------------------------------------------


class TestBoundedSumProductAlgebra:
    def test_name_is_probability(self) -> None:
        assert PROBABILITY.name == "Probability"

    def test_tensor_op_clamps_to_unit_interval(self) -> None:
        a = torch.tensor([0.5, 0.8, 1.0])
        b = torch.tensor([1.5, 0.3, 0.0])  # 1.5 would push out of unit interval
        out = PROBABILITY.tensor_op(a, b)
        # 0.5 * 1.5 = 0.75 (in range); 0.8 * 0.3 = 0.24; 1.0 * 0.0 = 0
        assert torch.allclose(out, torch.tensor([0.75, 0.24, 0.0]))

    def test_join_clamps_after_summation(self) -> None:
        t = torch.tensor([[0.7, 0.8], [0.6, 0.9]])
        # Sum along axis 0: [1.3, 1.7] → clamped to [1.0, 1.0]
        out = PROBABILITY.join(t, dim=0)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= 0.0)
        assert torch.allclose(out, torch.tensor([1.0, 1.0]))

    def test_negate_is_complement(self) -> None:
        t = torch.tensor([0.3, 0.7, 1.0, 0.0])
        out = PROBABILITY.negate(t)
        assert torch.allclose(out, torch.tensor([0.7, 0.3, 0.0, 1.0]))

    def test_unit_and_zero(self) -> None:
        assert PROBABILITY.unit == 1.0
        assert PROBABILITY.zero == 0.0

    def test_identity_tensor_is_eye(self) -> None:
        I = PROBABILITY.identity_tensor((3,))
        assert torch.allclose(I, torch.eye(3))


# ---------------------------------------------------------------------------
# Algebraic contract for CountingAlgebra
# ---------------------------------------------------------------------------


class TestCountingAlgebra:
    def test_name_is_counting(self) -> None:
        assert COUNTING.name == "Counting"

    def test_tensor_op_is_product(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(COUNTING.tensor_op(a, b), a * b)

    def test_join_is_sum(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(COUNTING.join(t, dim=0), torch.tensor([4.0, 6.0]))

    def test_negate_raises(self) -> None:
        t = torch.tensor([1.0, 2.0])
        with pytest.raises(NotImplementedError, match="counting"):
            COUNTING.negate(t)

    def test_unit_and_zero(self) -> None:
        assert COUNTING.unit == 1.0
        assert COUNTING.zero == 0.0


# ---------------------------------------------------------------------------
# Composition through the new operators
# ---------------------------------------------------------------------------


def test_real_compose_via_rshift_uses_sum_product() -> None:
    """``f >> g`` over two RealAlgebra morphisms produces a
    ComposedMorphism whose tensor is the matrix product (sum-
    product semiring on ℝ)."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g_data = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    f = ObservedMorphism(A, B, f_data, algebra=REAL)
    g = ObservedMorphism(B, C, g_data, algebra=REAL)
    chain = f >> g
    assert chain.algebra.name == "Real"
    # Materialise the chain's tensor and compare against the
    # matrix product.
    expected = f_data @ g_data
    assert torch.allclose(chain.tensor, expected)


def test_probability_compose_clamps_to_unit_interval() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = ObservedMorphism(
        A, B, torch.tensor([[0.5, 0.5], [0.5, 0.5]]), algebra=PROBABILITY
    )
    g = ObservedMorphism(
        B, C, torch.tensor([[0.5, 0.5], [0.5, 0.5]]), algebra=PROBABILITY
    )
    chain = f >> g
    assert chain.algebra.name == "Probability"
    # Every entry must be in [0, 1].
    assert torch.all(chain.tensor >= 0.0)
    assert torch.all(chain.tensor <= 1.0)


def test_counting_compose_counts_paths() -> None:
    """The counting algebra composition counts the number of
    paths from A through B to C. For two identity-1 matrices the
    result is 1 per direct path; for higher entries the path
    count scales linearly."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    # Two "fan-out" matrices: each A maps to both B; each B maps
    # to both C. The chain has 2 paths from each A to each C.
    f = ObservedMorphism(A, B, torch.ones(2, 2), algebra=COUNTING)
    g = ObservedMorphism(B, C, torch.ones(2, 2), algebra=COUNTING)
    chain = f >> g
    expected = torch.ones(2, 2) @ torch.ones(2, 2)  # [[2, 2], [2, 2]]
    assert torch.allclose(chain.tensor, expected)


# ---------------------------------------------------------------------------
# DSL surface: composition in the declared algebra
# ---------------------------------------------------------------------------


def test_real_composition_uses_declared_algebra() -> None:
    from quivers.dsl import loads

    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    object C : FinSet 3

    morphism f : A -> B [role=latent]
    morphism g : B -> C [role=latent]
    define chain = f >> g
    export chain
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Real"


def test_probability_composition_uses_declared_algebra() -> None:
    from quivers.dsl import loads

    src = """
    composition probability [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    object C : FinSet 3

    morphism f : A -> B [role=latent]
    morphism g : B -> C [role=latent]
    define chain = f >> g
    export chain
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Probability"


# ---------------------------------------------------------------------------
# Module-level composition declarations
# ---------------------------------------------------------------------------


def test_algebra_real_declaration_compiles() -> None:
    from quivers.dsl import loads

    src = """
    composition real [level=algebra]
    object A : FinSet 4
    morphism f : A -> A [role=latent]
    export f
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Real"


def test_algebra_probability_declaration_compiles() -> None:
    from quivers.dsl import loads

    src = """
    composition probability [level=algebra]
    object A : FinSet 4
    morphism f : A -> A [role=latent]
    export f
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Probability"


def test_algebra_counting_declaration_compiles() -> None:
    from quivers.dsl import loads

    src = """
    composition counting [level=algebra]
    object A : FinSet 4
    morphism f : A -> A [role=latent]
    export f
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Counting"


def test_algebra_max_plus_declaration_compiles() -> None:
    """The ``max_plus`` algebra name resolves at module level."""
    from quivers.dsl import loads

    src = """
    composition max_plus [level=algebra]
    object A : FinSet 4
    morphism f : A -> A [role=latent]
    export f
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "MaxPlus"


def test_algebra_log_prob_declaration_compiles() -> None:
    from quivers.dsl import loads

    src = """
    composition log_prob [level=algebra]
    object A : FinSet 4
    morphism f : A -> A [role=latent]
    export f
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "LogProb"


# ---------------------------------------------------------------------------
# Gradient flow through sum-product semiring composition
# ---------------------------------------------------------------------------


def test_real_chain_gradients_flow() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=REAL)
    g = LatentMorphism(B, C, algebra=REAL)
    chain = f >> g
    loss = chain.tensor.sum()
    loss.backward()
    assert f.raw.grad is not None
    assert g.raw.grad is not None
    assert torch.isfinite(f.raw.grad).all()
    assert torch.isfinite(g.raw.grad).all()


def test_probability_chain_gradients_flow() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=PROBABILITY)
    g = LatentMorphism(B, C, algebra=PROBABILITY)
    chain = f >> g
    loss = chain.tensor.sum()
    loss.backward()
    assert f.raw.grad is not None
    assert g.raw.grad is not None
    assert torch.isfinite(f.raw.grad).all()
    assert torch.isfinite(g.raw.grad).all()


def test_counting_chain_gradients_flow() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=COUNTING)
    g = LatentMorphism(B, C, algebra=COUNTING)
    chain = f >> g
    loss = chain.tensor.sum()
    loss.backward()
    assert f.raw.grad is not None
    assert g.raw.grad is not None
    assert torch.isfinite(f.raw.grad).all()
    assert torch.isfinite(g.raw.grad).all()


# ---------------------------------------------------------------------------
# Cross-algebra change-of-base interactions
# ---------------------------------------------------------------------------


def test_change_base_real_to_log_prob_via_custom_homomorphism() -> None:
    """Apply ``log`` from Real to LogProb via a one-off
    homomorphism; the result is a LogProb-algebra morphism."""
    from quivers.core.algebras import LOG_PROB
    from quivers.core.algebra_morphisms import CarrierMap

    class _RealToLogProb(CarrierMap):
        @property
        def source(self):
            return REAL

        @property
        def target(self):
            return LOG_PROB

        def apply(self, t):
            return torch.log(t.clamp(min=1e-30))

    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=3)
    data = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    f = ObservedMorphism(A, B, data, algebra=REAL)
    g = f.change_base(_RealToLogProb())
    assert g.algebra.name == "LogProb"
    assert torch.allclose(g.tensor, torch.log(data))


# ---------------------------------------------------------------------------
# Algebra homomorphisms between Real / Probability / Counting
# ---------------------------------------------------------------------------


def test_probability_clamp_homomorphism_squeezes_into_unit_interval() -> None:
    from quivers.core.algebra_morphisms import PROBABILITY_CLAMP

    t = torch.tensor([-0.5, 0.3, 0.8, 1.7])
    out = PROBABILITY_CLAMP.apply(t)
    assert torch.allclose(out, torch.tensor([0.0, 0.3, 0.8, 1.0]))
    assert PROBABILITY_CLAMP.source.name == "Real"
    assert PROBABILITY_CLAMP.target.name == "Probability"


def test_counting_from_real_floors_and_clamps_negative() -> None:
    from quivers.core.algebra_morphisms import COUNTING_FROM_REAL

    t = torch.tensor([-1.7, 0.3, 1.7, 3.0])
    out = COUNTING_FROM_REAL.apply(t)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0, 3.0]))


def test_probability_to_real_is_inclusion() -> None:
    from quivers.core.algebra_morphisms import PROBABILITY_TO_REAL

    t = torch.tensor([0.1, 0.5, 0.9])
    out = PROBABILITY_TO_REAL.apply(t)
    assert torch.allclose(out, t)
    assert PROBABILITY_TO_REAL.source.name == "Probability"
    assert PROBABILITY_TO_REAL.target.name == "Real"


def test_counting_to_real_is_inclusion() -> None:
    from quivers.core.algebra_morphisms import COUNTING_TO_REAL

    t = torch.tensor([0.0, 1.0, 2.0, 5.0])
    out = COUNTING_TO_REAL.apply(t)
    assert torch.allclose(out, t)


def test_lookup_carrier_map_real_to_probability() -> None:
    from quivers.core.algebra_morphisms import lookup_carrier_map

    phi = lookup_carrier_map(REAL, PROBABILITY)
    assert phi is not None
    assert phi.source.name == "Real"
    assert phi.target.name == "Probability"


def test_lookup_carrier_map_real_to_counting() -> None:
    from quivers.core.algebra_morphisms import lookup_carrier_map

    phi = lookup_carrier_map(REAL, COUNTING)
    assert phi is not None
    assert phi.target.name == "Counting"


def test_lookup_carrier_map_probability_to_real() -> None:
    from quivers.core.algebra_morphisms import lookup_carrier_map

    phi = lookup_carrier_map(PROBABILITY, REAL)
    assert phi is not None


def test_change_base_real_to_probability_via_named_homomorphism() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=3)
    data = torch.tensor([[-0.5, 0.3, 1.7], [0.0, 0.5, 1.0], [2.0, -1.0, 0.9]])
    f = ObservedMorphism(A, B, data, algebra=REAL)
    from quivers.core.algebra_morphisms import PROBABILITY_CLAMP

    g = f.change_base(PROBABILITY_CLAMP)
    assert g.algebra.name == "Probability"
    expected = data.clamp(min=0.0, max=1.0)
    assert torch.allclose(g.tensor, expected)


def test_dsl_change_base_to_probability_clamp() -> None:
    from quivers.dsl import loads

    src = """
    composition real [level=algebra]
    object A : FinSet 3

    morphism f : A -> A [role=latent]
    define g = f.change_base(probability_clamp)
    export g
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Probability"


def test_dsl_change_base_to_counting_from_real() -> None:
    from quivers.dsl import loads

    src = """
    composition real [level=algebra]
    object A : FinSet 3

    morphism f : A -> A [role=latent]
    define g = f.change_base(counting_from_real)
    export g
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism.algebra.name == "Counting"


# ---------------------------------------------------------------------------
# Compact-closed surface on the new algebras
# ---------------------------------------------------------------------------


def test_real_dagger_swaps_axes() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    data = torch.randn(3, 4)
    f = ObservedMorphism(A, B, data, algebra=REAL)
    g = f.dagger
    assert g.domain is B
    assert g.codomain is A
    assert g.algebra.name == "Real"
    assert torch.allclose(g.tensor, data.t())


def test_probability_cup_returns_identity() -> None:
    from quivers.core.morphisms import cup

    A = FinSet(name="A", cardinality=3)
    eta = cup(A, algebra=PROBABILITY)
    assert eta.algebra.name == "Probability"
    assert torch.allclose(eta.tensor.squeeze(0), torch.eye(3))


def test_counting_cap_returns_identity() -> None:
    from quivers.core.morphisms import cap

    A = FinSet(name="A", cardinality=3)
    eps = cap(A, algebra=COUNTING)
    assert eps.algebra.name == "Counting"
    assert torch.allclose(eps.tensor.squeeze(-1), torch.eye(3))
