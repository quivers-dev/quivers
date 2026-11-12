"""Tests for monads and Kleisli categories."""

import pytest
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.monadic.monads import FuzzyPowersetMonad, KleisliCategory
from quivers.core.algebras import (
    BOOLEAN,
    GODEL,
    LOG_PROB,
    PRODUCT_FUZZY,
    REAL,
    TROPICAL,
)


class TestFuzzyPowersetMonad:
    def test_unit_is_identity(self):
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        eta = m.unit(a)
        torch.testing.assert_close(eta.tensor, torch.eye(3))

    def test_multiply_is_identity(self):
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        mu = m.multiply(a)
        torch.testing.assert_close(mu.tensor, torch.eye(3))

    def test_kleisli_compose_matches_rshift(self):
        """f >=> g should equal f >> g for FuzzyPowerset."""
        torch.manual_seed(42)
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)
        f_data = torch.rand(3, 4)
        g_data = torch.rand(4, 2)
        f = observed(a, b, f_data, algebra=GODEL)
        g = observed(b, c, g_data, algebra=GODEL)
        kleisli = m.kleisli_compose(f, g)
        rshift = (f >> g).tensor
        torch.testing.assert_close(kleisli.tensor, rshift, atol=1e-05, rtol=1e-05)

    def test_boolean_kleisli(self):
        m = FuzzyPowersetMonad(algebra=BOOLEAN)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)
        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        f = observed(a, b, f_data, algebra=BOOLEAN)
        g = observed(b, c, g_data, algebra=BOOLEAN)
        result = m.kleisli_compose(f, g)
        torch.testing.assert_close(result.tensor, g_data, atol=1e-05, rtol=1e-05)


class TestKleisliCategory:
    def test_identity(self):
        m = FuzzyPowersetMonad(GODEL)
        cat = KleisliCategory(m)
        a = FinSet(name="A", cardinality=3)
        eta = cat.identity(a)
        torch.testing.assert_close(eta.tensor, torch.eye(3))

    def test_compose(self):
        torch.manual_seed(42)
        m = FuzzyPowersetMonad(GODEL)
        cat = KleisliCategory(m)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)
        f = observed(a, b, torch.rand(3, 4), algebra=GODEL)
        g = observed(b, c, torch.rand(4, 2), algebra=GODEL)
        result = cat.compose(f, g)
        expected = (f >> g).tensor
        torch.testing.assert_close(result.tensor, expected, atol=1e-05, rtol=1e-05)


class TestMonadLaws:
    def test_fuzzy_powerset_left_unit(self):
        """μ ∘ η_T = id (left unit)."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        eta = m.unit(a)
        mu = m.multiply(a)
        result = (eta >> mu).tensor
        torch.testing.assert_close(result, torch.eye(3), atol=1e-05, rtol=1e-05)

    def test_fuzzy_powerset_right_unit(self):
        """μ ∘ T(η) = id (right unit)."""
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=3)
        eta = m.unit(a)
        mu = m.multiply(a)
        result = (eta >> mu).tensor
        torch.testing.assert_close(result, torch.eye(3), atol=1e-05, rtol=1e-05)


class TestFuzzyPowersetLaws:
    """The Kleisli category of the fuzzy powerset monad is ``V``-Rel, so
    Kleisli composition must be associative and unital. These checks
    compare against max-min (Gödel) and min-plus (tropical) matrix
    products written out here."""

    @staticmethod
    def _godel(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f[:, :, None], g[None, :, :]).amax(dim=1)

    @staticmethod
    def _tropical(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return (f[:, :, None] + g[None, :, :]).amin(dim=1)

    def test_refuses_an_algebra_that_is_not_an_idempotent_semiring(self):
        for algebra in (PRODUCT_FUZZY, LOG_PROB, REAL):
            with pytest.raises(TypeError, match="IdempotentSemiring"):
                FuzzyPowersetMonad(algebra)  # type: ignore[arg-type]

    @pytest.mark.parametrize("which", ["godel", "tropical"])
    def test_kleisli_composition_is_associative_and_unital(self, which):
        algebra = GODEL if which == "godel" else TROPICAL
        reference = self._godel if which == "godel" else self._tropical
        gen = torch.Generator().manual_seed(7)
        m = FuzzyPowersetMonad(algebra)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        c = FinSet(name="C", cardinality=2)
        d = FinSet(name="D", cardinality=3)
        scale = 1.0 if which == "godel" else 5.0
        ft = torch.rand(3, 4, generator=gen) * scale
        gt = torch.rand(4, 2, generator=gen) * scale
        ht = torch.rand(2, 3, generator=gen) * scale
        f = observed(a, b, ft, algebra=algebra)
        g = observed(b, c, gt, algebra=algebra)
        h = observed(c, d, ht, algebra=algebra)
        left = m.kleisli_compose(m.kleisli_compose(f, g), h).tensor
        right = m.kleisli_compose(f, m.kleisli_compose(g, h)).tensor
        torch.testing.assert_close(left, right)
        torch.testing.assert_close(left, reference(reference(ft, gt), ht))
        torch.testing.assert_close(m.kleisli_compose(m.pure(a), f).tensor, ft)
        torch.testing.assert_close(m.kleisli_compose(f, m.pure(b)).tensor, ft)

    def test_noisy_or_composition_would_fail_associativity(self):
        """The check above rejects the default algebra the monad used to
        accept: noisy-OR composition depends on bracketing."""
        gen = torch.Generator().manual_seed(7)
        a = FinSet(name="A", cardinality=3)
        f = observed(a, a, torch.rand(3, 3, generator=gen))
        g = observed(a, a, torch.rand(3, 3, generator=gen))
        h = observed(a, a, torch.rand(3, 3, generator=gen))
        left = ((f >> g) >> h).tensor
        right = (f >> (g >> h)).tensor
        assert (left - right).abs().max() > 1e-2

    def test_refuses_a_morphism_over_another_algebra(self):
        m = FuzzyPowersetMonad(GODEL)
        a = FinSet(name="A", cardinality=2)
        f = observed(a, a, torch.rand(2, 2))
        with pytest.raises(ValueError, match="cannot compose"):
            m.kleisli_compose(f, f)
