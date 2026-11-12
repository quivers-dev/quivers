"""Tests for Yoneda embedding and representable profunctors."""

import itertools

import torch

from quivers.core.algebras import GODEL

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.enriched.profunctors import Profunctor
from quivers.enriched.yoneda import (
    Presheaf,
    representable_profunctor,
    yoneda_lemma,
    yoneda_embedding,
    yoneda_density,
    verify_yoneda_preserves_composition,
)


class TestRepresentableProfunctor:
    def test_representable_is_identity(self):
        """y(A) should be the identity profunctor."""
        a = FinSet(name="A", cardinality=3)
        p = representable_profunctor(a, GODEL)

        expected = torch.eye(3)
        torch.testing.assert_close(p.tensor, expected)

    def test_contra_and_co(self):
        a = FinSet(name="A", cardinality=3)
        p = representable_profunctor(a, GODEL)

        assert p.contra == a
        assert p.co == a


class TestYonedaEmbedding:
    def test_embedding_equals_profunctor_view(self):
        """y(f) should equal Profunctor.from_morphism(f)."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data, algebra=GODEL)

        y_f = yoneda_embedding(f)
        p_f = Profunctor.from_morphism(f)

        torch.testing.assert_close(y_f.tensor, p_f.tensor)

    def test_preserves_composition(self):
        """y(f >> g) ≈ y(f) ; y(g)."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)

        f = observed(a, b, torch.rand(3, 4), algebra=GODEL)
        g = observed(b, c, torch.rand(4, 2), algebra=GODEL)

        assert verify_yoneda_preserves_composition(f, g)


class TestYonedaDensity:
    def test_density_recovers_morphism(self):
        """Yoneda density: id_A >> f = f."""
        torch.manual_seed(42)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data, algebra=GODEL)

        result = yoneda_density(f, GODEL)

        torch.testing.assert_close(result, f.tensor, atol=1e-4, rtol=1e-4)


class TestYonedaLemma:
    """``∫_X [C(X, A), F(X)] = F(A)`` for a Gödel-enriched preorder ``C``
    on three elements and every V-presheaf ``F`` on it over a grid."""

    def test_end_recovers_the_presheaf_value(self):
        x = FinSet(name="X", cardinality=3)
        # hom(i, j): a chain 0 ≤ 1 ≤ 2 with graded degrees, reflexive and
        # transitive: hom(i, k) ≥ min(hom(i, j), hom(j, k)).
        hom = torch.tensor([[1.0, 0.5, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        for i, j, k in itertools.product(range(3), repeat=3):
            assert hom[i, k] >= min(hom[i, j], hom[j, k])
        grid = [0.0, 0.25, 0.5, 1.0]
        checked = 0
        for vals in itertools.product(grid, repeat=3):
            f = torch.tensor(vals)
            # F : C^op → V is a V-presheaf when min(hom(i, j), F(j)) ≤ F(i).
            if not all(
                min(hom[i, j], f[j]) <= f[i] for i in range(3) for j in range(3)
            ):
                continue
            presheaf = Presheaf(objects=[x], values={0: f}, algebra=GODEL)
            got = yoneda_lemma(presheaf, 0, [hom], GODEL)
            torch.testing.assert_close(got, f)
            checked += 1
        assert checked > 5
