"""Tests for Kan extensions."""

import itertools

import pytest
import torch

from quivers.core.algebras import BOOLEAN, GODEL, MAX_PLUS
from quivers.core.morphisms import observed
from quivers.core.objects import CoproductSet, FinSet, ProductSet
from quivers.enriched.kan_extensions import Inclusion, Projection, left_kan, right_kan


class TestProjection:
    def test_source_target(self):
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        p = ProductSet(components=(a, b, c))
        proj = Projection(p, keep_indices=(0, 2))
        assert proj.source == p
        assert proj.target.shape == (2, 4)

    def test_apply(self):
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        p = ProductSet(components=(a, b))
        proj = Projection(p, keep_indices=(0,))
        assert proj.apply((1, 2)) == (1,)

    def test_fiber_indices(self):
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        p = ProductSet(components=(a, b))
        proj = Projection(p, keep_indices=(0,))
        fiber = proj.fiber_indices((1,))
        assert len(fiber) == 3
        assert (1, 0) in fiber
        assert (1, 1) in fiber
        assert (1, 2) in fiber


class TestLeftKan:
    def test_domain_reduction(self):
        """left_kan with Projection joins over domain fiber."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        dom = ProductSet(components=(a, b))
        f_data = torch.rand(*dom.shape, *c.shape)
        f = observed(dom, c, f_data, algebra=GODEL)
        proj = Projection(dom, keep_indices=(0,))
        kan_result = left_kan(f, along=proj)
        assert kan_result.tensor.shape == (2, 4)
        expected = torch.zeros(2, 4)
        for a_i in range(2):
            for c_i in range(4):
                expected[a_i, c_i] = max(
                    float(f_data[a_i, b_i, c_i]) for b_i in range(3)
                )
        torch.testing.assert_close(kan_result.tensor, expected, atol=1e-05, rtol=1e-05)

    def test_matches_codomain_marginalize(self):
        """left_kan on codomain product matches .marginalize()."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        cod = ProductSet(components=(b, c))
        f_data = torch.rand(*a.shape, *cod.shape)
        f = observed(a, cod, f_data)
        marginalized = f.marginalize(b)
        Projection(cod, keep_indices=(1,))
        assert marginalized.tensor.shape == (2, 4)

    def test_left_kan_boolean(self):
        """Left Kan with Boolean algebra uses OR."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)
        dom = ProductSet(components=(a, b))
        f_data = torch.zeros(2, 2, 2)
        f_data[0, 0, 0] = 1.0
        f_data[0, 1, 1] = 1.0
        f = observed(dom, c, f_data, algebra=BOOLEAN)
        proj = Projection(dom, keep_indices=(0,))
        result = left_kan(f, along=proj, algebra=BOOLEAN)
        assert result.tensor[0, 0].item() == pytest.approx(1.0)
        assert result.tensor[0, 1].item() == pytest.approx(1.0)


class TestRightKan:
    def test_right_kan_uses_meet(self):
        """Right Kan should use meet (AND for boolean)."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)
        dom = ProductSet(components=(a, b))
        f_data = torch.zeros(2, 2, 2)
        f_data[0, 0, 0] = 1.0
        f_data[0, 1, 1] = 1.0
        f = observed(dom, c, f_data, algebra=BOOLEAN)
        proj = Projection(dom, keep_indices=(0,))
        result = right_kan(f, along=proj, algebra=BOOLEAN)
        assert result.tensor[0, 0].item() == pytest.approx(0.0)
        assert result.tensor[0, 1].item() == pytest.approx(0.0)

    def test_right_kan_shape(self):
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        dom = ProductSet(components=(a, b))
        f_data = torch.rand(2, 3, 4)
        f = observed(dom, c, f_data, algebra=GODEL)
        proj = Projection(dom, keep_indices=(0,))
        result = right_kan(f, along=proj)
        assert result.tensor.shape == (2, 4)


class TestKanAdjunctions:
    """``Lan_p ⊣ p* ⊣ Ran_p``: for every ``S : A' → B`` (here, every
    Gödel matrix over a grid), ``Lan_p R ≤ S ⟺ R ≤ S ∘ p`` and
    ``S ∘ p ≤ R ⟺ S ≤ Ran_p R``, entrywise."""

    @staticmethod
    def _setup():
        a = FinSet(name="A", cardinality=3)
        a2 = FinSet(name="A2", cardinality=2)
        b = FinSet(name="B", cardinality=1)
        cop = CoproductSet(components=(a2, a))
        # p : A → A2 + A is the second injection: A2's elements have empty fibres.
        p = Inclusion(cop, 1)
        return a, cop, b, p

    def test_adjunctions_hold_including_empty_fibres(self):
        a, cop, b, p = self._setup()
        grid = [0.0, 0.5, 1.0]
        for r_vals in itertools.product(grid, repeat=a.size):
            r = observed(a, b, torch.tensor(r_vals).reshape(3, 1), algebra=GODEL)
            lan = left_kan(r, along=p).tensor.reshape(-1)
            ran = right_kan(r, along=p).tensor.reshape(-1)
            for s_vals in itertools.product(grid, repeat=cop.size):
                s_t = torch.tensor(s_vals)
                s_p = torch.tensor([s_vals[2 + i] for i in range(3)])
                r_t = torch.tensor(r_vals)
                assert bool((lan <= s_t).all()) == bool((r_t <= s_p).all())
                assert bool((s_p <= r_t).all()) == bool((s_t <= ran).all())

    def test_empty_fibres_give_bottom_and_top(self):
        a, cop, b, p = self._setup()
        r = observed(a, b, torch.full((3, 1), 0.5), algebra=MAX_PLUS)
        assert left_kan(r, along=p).tensor[0, 0].item() == float("-inf")
        assert right_kan(r, along=p).tensor[0, 0].item() == float("inf")

    def test_refuses_an_algebra_whose_join_is_not_a_lattice_join(self):
        a, _, b, p = self._setup()
        r = observed(a, b, torch.full((3, 1), 0.5))
        with pytest.raises(TypeError, match="IdempotentSemiring"):
            left_kan(r, along=p)
        with pytest.raises(TypeError, match="IdempotentSemiring"):
            right_kan(r, along=p)
