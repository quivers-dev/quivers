"""Tests for monoidal structures."""

import torch
from quivers.core.objects import FinSet, ProductSet, CoproductSet, Unit
from quivers.core.morphisms import identity
from quivers.categorical.monoidal import (
    CartesianProductMonoidal,
    CoproductMonoidal,
    EmptySet,
)


class TestCartesianProductMonoidal:
    def test_product(self):
        m = CartesianProductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        result = m.product(a, b)
        assert isinstance(result, ProductSet)
        assert result.shape == (2, 3)

    def test_unit(self):
        m = CartesianProductMonoidal()
        assert m.unit == Unit

    def test_associator_is_identity(self):
        """Since ProductSet flattens, associator is identity."""
        m = CartesianProductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        assoc = m.associator(a, b, c)
        flat = ProductSet(components=(a, b, c))
        expected = identity(flat).tensor
        torch.testing.assert_close(assoc.tensor, expected)

    def test_left_unitor(self):
        """I × A → A should be an isomorphism."""
        m = CartesianProductMonoidal()
        a = FinSet(name="A", cardinality=3)
        lu = m.left_unitor(a)
        assert lu.tensor.shape == (1, 3, 3)
        for i in range(3):
            assert lu.tensor[0, i, i].item() == 1.0

    def test_right_unitor(self):
        """A × I → A should be an isomorphism."""
        m = CartesianProductMonoidal()
        a = FinSet(name="A", cardinality=3)
        ru = m.right_unitor(a)
        assert ru.tensor.shape == (3, 1, 3)
        for i in range(3):
            assert ru.tensor[i, 0, i].item() == 1.0

    def test_braiding_is_involution(self):
        """σ_{B,A} ∘ σ_{A,B} = id."""
        m = CartesianProductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        sigma_ab = m.braiding(a, b)
        sigma_ba = m.braiding(b, a)
        result = (sigma_ab >> sigma_ba).tensor
        expected = identity(ProductSet(components=(a, b))).tensor
        torch.testing.assert_close(result, expected, atol=1e-05, rtol=1e-05)


class TestCoproductMonoidal:
    def test_product(self):
        m = CoproductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        result = m.product(a, b)
        assert isinstance(result, CoproductSet)
        assert result.size == 5

    def test_unit_is_empty(self):
        m = CoproductMonoidal()
        assert isinstance(m.unit, EmptySet)
        assert m.unit.size == 0

    def test_associator_is_identity(self):
        """Since CoproductSet flattens, associator is identity."""
        m = CoproductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        assoc = m.associator(a, b, c)
        flat = CoproductSet(components=(a, b, c))
        expected = identity(flat).tensor
        torch.testing.assert_close(assoc.tensor, expected)

    def test_braiding_swap(self):
        """Braiding swaps the two coproduct blocks."""
        m = CoproductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        sigma = m.braiding(a, b)
        assert sigma.tensor[0, 3].item() == 1.0
        assert sigma.tensor[1, 4].item() == 1.0
        assert sigma.tensor[2, 0].item() == 1.0

    def test_braiding_involution(self):
        """σ_{B,A} ∘ σ_{A,B} = id."""
        m = CoproductMonoidal()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        sigma_ab = m.braiding(a, b)
        sigma_ba = m.braiding(b, a)
        result = (sigma_ab >> sigma_ba).tensor
        expected = identity(CoproductSet(components=(a, b))).tensor
        torch.testing.assert_close(result, expected, atol=1e-05, rtol=1e-05)
