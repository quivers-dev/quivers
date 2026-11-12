"""Tests for algebras and coalgebras."""

import pytest
import torch
from quivers.core.objects import FinSet, ProductSet
from quivers.core.algebras import GODEL
from quivers.core.morphisms import identity
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.monadic.comonads import CofreeComonad
from quivers.monadic.algebras import (
    FreeAlgebra,
    ObservedAlgebra,
    CofreeCoalgebra,
    EilenbergMooreCategory,
    ObservedCoalgebra,
)


class TestFreeAlgebra:
    def test_fuzzy_powerset_free_algebra(self):
        """Free algebra for FuzzyPowerset has carrier A and μ = id."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        alg = FreeAlgebra(m, a)
        assert alg.carrier == a
        alpha = alg.structure_map()
        torch.testing.assert_close(alpha.tensor, torch.eye(3))

    def test_free_algebra_unit_law(self):
        """α ∘ η_A = id for free algebra."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        alg = FreeAlgebra(m, a)
        assert alg.verify_unit_law()

    def test_free_algebra_associativity(self):
        """α ∘ μ = α ∘ T(α) for free algebra."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        alg = FreeAlgebra(m, a)
        assert alg.verify_associativity()


class TestObservedAlgebra:
    def test_identity_algebra(self):
        """The identity morphism is an algebra for FuzzyPowerset."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        alg = ObservedAlgebra(m, a, torch.eye(3), algebra=GODEL)
        assert alg.verify_unit_law()
        assert alg.verify_associativity()

    def test_refuses_a_structure_map_that_breaks_the_unit_law(self):
        """A constant map is not an algebra: α ∘ η ≠ id."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=2)
        constant = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="unit law"):
            ObservedAlgebra(m, a, constant, algebra=GODEL)


class TestCofreeCoalgebra:
    def test_cofree_coalgebra_structure(self):
        """Cofree coalgebra has carrier W(A) = A × S and γ = δ_A."""
        s = FinSet(name="S", cardinality=2)
        w = CofreeComonad(s, GODEL)
        a = FinSet(name="A", cardinality=2)
        coalg = CofreeCoalgebra(w, a)
        assert coalg.carrier == ProductSet(components=(a, s))

    def test_cofree_coalgebra_laws(self):
        """ε ∘ γ = id and δ ∘ γ = W(γ) ∘ γ for the cofree coalgebra."""
        w = CofreeComonad(FinSet(name="S", cardinality=2), GODEL)
        coalg = CofreeCoalgebra(w, FinSet(name="A", cardinality=2))
        assert coalg.verify_counit_law()
        assert coalg.verify_coassociativity()

    def test_observed_coalgebra_accepts_a_lawful_map_and_refuses_another(self):
        """``a ↦ (a, s₀)`` is a coalgebra of the environment comonad;
        a map that sends every ``a`` to the same pair is not."""
        s = FinSet(name="S", cardinality=2)
        a = FinSet(name="A", cardinality=2)
        w = CofreeComonad(s, GODEL)
        lawful = torch.zeros(2, 2, 2)
        lawful[0, 0, 0] = 1.0
        lawful[1, 1, 0] = 1.0
        coalg = ObservedCoalgebra(w, a, lawful, algebra=GODEL)
        assert coalg.verify_coassociativity()
        broken = torch.zeros(2, 2, 2)
        broken[:, 0, 0] = 1.0
        with pytest.raises(ValueError, match="counit law"):
            ObservedCoalgebra(w, a, broken, algebra=GODEL)


class TestEilenbergMooreCategory:
    def test_free_algebra_creation(self):
        """EM category can create free algebras."""
        m = FuzzyPowersetMonad(GODEL)
        em = EilenbergMooreCategory(m)
        a = FinSet(name="A", cardinality=3)
        alg = em.free_algebra(a)
        assert alg.carrier == a
        assert alg.verify_unit_law()

    def test_identity_is_homomorphism(self):
        """The identity morphism should be an algebra homomorphism."""
        m = FuzzyPowersetMonad(GODEL)
        em = EilenbergMooreCategory(m)
        a = FinSet(name="A", cardinality=3)
        alg = em.free_algebra(a)
        id_a = identity(a, algebra=GODEL)
        assert em.is_homomorphism(id_a, alg, alg)
