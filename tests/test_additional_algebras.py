"""Tests for additional algebras."""

import torch
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed, identity
from quivers.core.algebras import (
    LUKASIEWICZ,
    GODEL,
    TROPICAL,
)


class TestStandardMVAlgebra:
    def test_tensor_op(self):
        """Łukasiewicz t-norm: max(a + b - 1, 0)."""
        q = LUKASIEWICZ
        a = torch.tensor([0.8, 0.3, 1.0, 0.0])
        b = torch.tensor([0.5, 0.4, 1.0, 1.0])
        expected = torch.tensor([0.3, 0.0, 1.0, 0.0])
        torch.testing.assert_close(q.tensor_op(a, b), expected)

    def test_join_bounded_sum(self):
        """Bounded sum: min(1, ∑ x_i)."""
        q = LUKASIEWICZ
        t = torch.tensor([0.3, 0.4, 0.5])
        result = q.join(t, dim=0)
        assert result.item() == pytest.approx(1.0)  # 1.2 clamped to 1.0

    def test_join_small(self):
        """Bounded sum below 1."""
        q = LUKASIEWICZ
        t = torch.tensor([0.2, 0.3])
        result = q.join(t, dim=0)
        assert result.item() == pytest.approx(0.5)

    def test_meet_is_min(self):
        """Meet = min."""
        q = LUKASIEWICZ
        t = torch.tensor([0.8, 0.3, 0.5])
        result = q.meet(t, dim=0)
        assert result.item() == pytest.approx(0.3)

    def test_unit_and_zero(self):
        q = LUKASIEWICZ
        assert q.unit == 1.0
        assert q.zero == 0.0

    def test_identity_composition(self):
        """id >> id = id for Łukasiewicz."""
        q = LUKASIEWICZ
        a = FinSet(name="A", cardinality=3)
        id_a = identity(a, algebra=q)
        result = (id_a >> id_a).tensor
        expected = q.identity_tensor((3,))

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_composition(self):
        """V-enriched composition with Łukasiewicz algebra."""
        q = LUKASIEWICZ
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)

        f_data = torch.tensor([[0.8, 0.3], [0.5, 0.9]])
        g_data = torch.tensor([[0.7, 0.2], [0.4, 0.8]])
        f = observed(a, b, f_data, algebra=q)
        g = observed(b, b, g_data, algebra=q)

        result = (f >> g).tensor
        assert result.shape == (2, 2)

        # values should be in [0, 1]
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


class TestGodelAlgebra:
    def test_tensor_op(self):
        """Gödel t-norm: min(a, b)."""
        q = GODEL
        a = torch.tensor([0.8, 0.3, 1.0])
        b = torch.tensor([0.5, 0.7, 0.0])
        expected = torch.tensor([0.5, 0.3, 0.0])
        torch.testing.assert_close(q.tensor_op(a, b), expected)

    def test_join_is_max(self):
        """Join = max."""
        q = GODEL
        t = torch.tensor([0.3, 0.8, 0.5])
        result = q.join(t, dim=0)
        assert result.item() == pytest.approx(0.8)

    def test_meet_is_min(self):
        """Meet = min."""
        q = GODEL
        t = torch.tensor([0.8, 0.3, 0.5])
        result = q.meet(t, dim=0)
        assert result.item() == pytest.approx(0.3)

    def test_negate(self):
        """Gödel negation: ¬0 = 1, ¬x = 0 for x > 0."""
        q = GODEL
        t = torch.tensor([0.0, 0.5, 1.0])
        expected = torch.tensor([1.0, 0.0, 0.0])
        torch.testing.assert_close(q.negate(t), expected)

    def test_identity_composition(self):
        """id >> id = id for Gödel."""
        q = GODEL
        a = FinSet(name="A", cardinality=3)
        id_a = identity(a, algebra=q)
        result = (id_a >> id_a).tensor
        expected = q.identity_tensor((3,))

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_minimax_composition(self):
        """Gödel composition is minimax."""
        q = GODEL
        # f(a, b) = [[0.8, 0.3], [0.5, 0.9]]
        # g(b, c) = [[0.7, 0.2], [0.4, 0.6]]
        # (f >> g)(a, c) = max_b min(f(a,b), g(b,c))
        # (0, 0) = max(min(0.8, 0.7), min(0.3, 0.4)) = max(0.7, 0.3) = 0.7
        # (0, 1) = max(min(0.8, 0.2), min(0.3, 0.6)) = max(0.2, 0.3) = 0.3
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)

        f = observed(a, b, torch.tensor([[0.8, 0.3], [0.5, 0.9]]), algebra=q)
        g = observed(b, c, torch.tensor([[0.7, 0.2], [0.4, 0.6]]), algebra=q)

        result = (f >> g).tensor
        expected = torch.tensor([[0.7, 0.3], [0.5, 0.6]])

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestTropicalAlgebra:
    def test_tensor_op_addition(self):
        """Tropical tensor: a + b."""
        q = TROPICAL
        a = torch.tensor([1.0, 2.0, 0.0])
        b = torch.tensor([3.0, 0.5, 0.0])
        expected = torch.tensor([4.0, 2.5, 0.0])
        torch.testing.assert_close(q.tensor_op(a, b), expected)

    def test_join_is_min(self):
        """Tropical join = inf = min."""
        q = TROPICAL
        t = torch.tensor([3.0, 1.0, 5.0])
        result = q.join(t, dim=0)
        assert result.item() == pytest.approx(1.0)

    def test_meet_is_max(self):
        """Tropical meet = sup = max."""
        q = TROPICAL
        t = torch.tensor([3.0, 1.0, 5.0])
        result = q.meet(t, dim=0)
        assert result.item() == pytest.approx(5.0)

    def test_identity_tensor(self):
        """Identity has 0 on diagonal, ∞ elsewhere."""
        q = TROPICAL
        t = q.identity_tensor((3,))
        assert t.shape == (3, 3)

        for i in range(3):
            for j in range(3):
                if i == j:
                    assert t[i, j].item() == 0.0
                else:
                    assert t[i, j].item() == float("inf")

    def test_negate_raises(self):
        """Tropical negation should raise."""
        q = TROPICAL
        with pytest.raises(NotImplementedError):
            q.negate(torch.tensor([1.0]))

    def test_shortest_path_composition(self):
        """Tropical composition is (min, +) = shortest path."""
        q = TROPICAL
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)

        # distances
        f_data = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
        g_data = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
        f = observed(a, b, f_data, algebra=q)
        g = observed(b, c, g_data, algebra=q)

        result = (f >> g).tensor

        # (0,0) = min(1+2, 3+1) = min(3, 4) = 3
        # (0,1) = min(1+4, 3+3) = min(5, 6) = 5
        # (1,0) = min(2+2, 1+1) = min(4, 2) = 2
        # (1,1) = min(2+4, 1+3) = min(6, 4) = 4
        expected = torch.tensor([[3.0, 5.0], [2.0, 4.0]])

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_identity_composition(self):
        """id >> f = f for tropical."""
        q = TROPICAL
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)

        f_data = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
        f = observed(a, b, f_data, algebra=q)
        id_a = identity(a, algebra=q)

        result = (id_a >> f).tensor
        torch.testing.assert_close(result, f_data, atol=1e-5, rtol=1e-5)
