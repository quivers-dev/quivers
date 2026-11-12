"""Tests for profunctors."""

import pytest
import torch

from quivers.core.algebras import GODEL, PRODUCT_FUZZY

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.enriched.profunctors import Profunctor
from quivers.enriched.yoneda import representable_profunctor


class TestProfunctor:
    def test_from_morphism(self):
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data, algebra=GODEL)

        p = Profunctor.from_morphism(f)
        assert p.contra == a
        assert p.co == b
        torch.testing.assert_close(p.tensor, f_data)

    def test_compose_matches_rshift(self):
        """Profunctor composition should match morphism >>."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)

        f_data = torch.rand(3, 4)
        g_data = torch.rand(4, 2)
        f = observed(a, b, f_data, algebra=GODEL)
        g = observed(b, c, g_data, algebra=GODEL)

        pf = Profunctor.from_morphism(f)
        pg = Profunctor.from_morphism(g)
        composed = pf.compose(pg)

        expected = (f >> g).tensor
        torch.testing.assert_close(composed.tensor, expected, atol=1e-5, rtol=1e-5)

    def test_to_morphism(self):
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        data = torch.rand(3, 4)
        p = Profunctor(a, b, data, GODEL)
        m = p.to_morphism()

        assert m.domain == a
        assert m.codomain == b
        torch.testing.assert_close(m.tensor, data)

    def test_shape_validation(self):
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        with pytest.raises(ValueError, match="tensor shape"):
            Profunctor(a, b, torch.rand(3, 5), GODEL)

    def test_compose_mismatch_raises(self):
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)
        d = FinSet(name="D", cardinality=5)

        p1 = Profunctor(a, b, torch.rand(3, 4), GODEL)
        p2 = Profunctor(c, d, torch.rand(2, 5), GODEL)

        with pytest.raises(TypeError, match="cannot compose"):
            p1.compose(p2)


class TestProfunctorBicategoryLaws:
    """Composition (the coend) is associative and the hom profunctors
    are its units, checked against a max-min product written out here."""

    @staticmethod
    def _maxmin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x[:, :, None], y[None, :, :]).amax(dim=1)

    def test_associative_and_unital(self):
        gen = torch.Generator().manual_seed(3)
        a, b, c, d = (
            FinSet(name=n, cardinality=k) for n, k in zip("ABCD", (2, 3, 2, 3))
        )
        r = Profunctor(a, b, torch.rand(2, 3, generator=gen), GODEL)
        s = Profunctor(b, c, torch.rand(3, 2, generator=gen), GODEL)
        t = Profunctor(c, d, torch.rand(2, 3, generator=gen), GODEL)
        left = r.compose(s).compose(t).tensor
        right = r.compose(s.compose(t)).tensor
        torch.testing.assert_close(left, right)
        torch.testing.assert_close(
            left, self._maxmin(self._maxmin(r.tensor, s.tensor), t.tensor)
        )
        torch.testing.assert_close(
            representable_profunctor(a, GODEL).compose(r).tensor, r.tensor
        )
        torch.testing.assert_close(
            r.compose(representable_profunctor(b, GODEL)).tensor, r.tensor
        )

    def test_refuses_an_algebra_that_is_not_a_lattice(self):
        a = FinSet(name="A", cardinality=2)
        with pytest.raises(TypeError, match="IdempotentSemiring"):
            Profunctor(a, a, torch.rand(2, 2), PRODUCT_FUZZY)  # type: ignore[arg-type]
