"""Tests for Day convolution."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet
from quivers.enriched.profunctors import Profunctor
from quivers.categorical.monoidal import CartesianMonoidal
from quivers.core.algebras import GODEL, PRODUCT_FUZZY
from quivers.enriched.day_convolution import (
    day_convolution,
    day_convolution_profunctors,
    day_unit,
)


class TestDayUnit:
    def test_unit_presheaf(self):
        """Unit presheaf has I at the unit index, ⊥ elsewhere."""
        objects = [
            FinSet(name="A", cardinality=2),
            FinSet(name="B", cardinality=3),
            FinSet(name="I", cardinality=1),
        ]
        result = day_unit(objects, unit_index=2, algebra=GODEL)

        assert result.shape == (3,)
        assert result[0].item() == 0.0
        assert result[1].item() == 0.0
        assert result[2].item() == 1.0


class TestDayConvolutionProfunctors:
    def test_external_tensor_product(self):
        """(P ⊛ Q)(ac, bd) = P(a, b) ⊗ Q(c, d)."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=2)
        d = FinSet(name="D", cardinality=2)

        p_data = torch.rand(2, 3)
        q_data = torch.rand(2, 2)
        p = Profunctor(contra=a, co=b, tensor=p_data, algebra=GODEL)
        q = Profunctor(contra=c, co=d, tensor=q_data, algebra=GODEL)

        monoidal = CartesianMonoidal()
        result = day_convolution_profunctors(p, q, monoidal, GODEL)

        # result should be (A×C) ↛ (B×D), shape (2, 2, 3, 2)
        assert result.tensor.shape == (2, 2, 3, 2)

        # check a specific entry
        # result[a, c, b, d] = p[a, b] ⊗ q[c, d] (min under Gödel)
        for ai in range(2):
            for ci in range(2):
                for bi in range(3):
                    for di in range(2):
                        expected = torch.minimum(p_data[ai, bi], q_data[ci, di])
                        actual = result.tensor[ai, ci, bi, di]
                        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_profunctor_objects(self):
        """Day convolution produces correct contra/co objects."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=4)
        d = FinSet(name="D", cardinality=5)

        p = Profunctor(contra=a, co=b, tensor=torch.rand(2, 3), algebra=GODEL)
        q = Profunctor(contra=c, co=d, tensor=torch.rand(4, 5), algebra=GODEL)

        monoidal = CartesianMonoidal()
        result = day_convolution_profunctors(p, q, monoidal, GODEL)

        assert isinstance(result.contra, ProductSet)
        assert isinstance(result.co, ProductSet)
        assert result.contra.shape == (2, 4)
        assert result.co.shape == (3, 5)


class TestDayConvolutionMonoidalLaws:
    """Over the discrete monoidal category of the group Z/3, Day
    convolution is ``(F ⊛ G)(m) = ⋁_{a + b = m} F(a) ⊗ G(b)``; it is
    associative and the representable at the unit is its unit."""

    objects = [FinSet(name=f"g{i}", cardinality=1) for i in range(3)]
    table = {(i, j): ((i + j) % 3, torch.tensor(1.0)) for i in range(3) for j in range(3)}

    def _conv(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return day_convolution(
            f, g, self.objects, CartesianMonoidal(), GODEL, product_table=self.table
        )

    @staticmethod
    def _reference(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(3)
        for a in range(3):
            for b in range(3):
                m = (a + b) % 3
                out[m] = max(out[m], min(f[a], g[b]))
        return out

    def test_matches_the_convolution_and_satisfies_the_laws(self):
        gen = torch.Generator().manual_seed(5)
        f, g, h = (torch.rand(3, generator=gen) for _ in range(3))
        torch.testing.assert_close(self._conv(f, g), self._reference(f, g))
        torch.testing.assert_close(
            self._conv(self._conv(f, g), h), self._conv(f, self._conv(g, h))
        )
        unit = day_unit(self.objects, unit_index=0, algebra=GODEL)
        torch.testing.assert_close(self._conv(unit, f), f)
        torch.testing.assert_close(self._conv(f, unit), f)

    def test_discrete_case_refuses_objects_not_closed_under_the_product(self):
        objects = [FinSet(name="A", cardinality=2), FinSet(name="I", cardinality=1)]
        with pytest.raises(ValueError, match="closed under"):
            day_convolution(
                torch.rand(2), torch.rand(2), objects, CartesianMonoidal(), GODEL
            )

    def test_refuses_an_algebra_that_is_not_a_lattice(self):
        with pytest.raises(TypeError, match="IdempotentSemiring"):
            day_unit(self.objects, unit_index=0, algebra=PRODUCT_FUZZY)  # type: ignore[arg-type]
