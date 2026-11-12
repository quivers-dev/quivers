"""The runtime law checks of `quivers.monadic.laws`.

Each check passes on lawful instances and, so that a passing check
means something, fails on an instance built to break the law it
tests.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core import GODEL, FinSet, Morphism, SetObject, observed
from quivers.monadic import (
    Alternative_,
    FuzzyPowersetMonad,
    Identity,
    Maybe,
    check_functor_laws,
    check_monad_laws,
)

A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
C = FinSet(name="C", cardinality=2)


@pytest.mark.parametrize(
    "monad",
    [Identity(), Maybe(), Alternative_(), FuzzyPowersetMonad(GODEL)],
    ids=["identity", "maybe", "alternative", "fuzzy-powerset"],
)
def test_lawful_monads_pass(monad) -> None:
    check_monad_laws(monad, A)


@pytest.mark.parametrize(
    "functor",
    [Identity(), Maybe(), FuzzyPowersetMonad(GODEL)],
    ids=["identity", "maybe", "fuzzy-powerset"],
)
def test_lawful_functors_pass(functor) -> None:
    torch.manual_seed(0)
    f = observed(A, B, torch.rand(2, 3))
    g = observed(B, C, torch.rand(3, 2))
    check_functor_laws(functor, A, B, C, f, g)


class _ScaledJoin(FuzzyPowersetMonad):
    """A powerset monad whose ``join`` halves every entry."""

    def join(self, A: SetObject) -> Morphism:
        lawful = super().join(A)
        return observed(
            lawful.domain, lawful.codomain, lawful.tensor * 0.5, algebra=lawful.algebra
        )


class _ScaledFmap(FuzzyPowersetMonad):
    """A powerset functor whose ``fmap`` halves every entry."""

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        lawful = super().fmap(A, B, f)
        return observed(
            lawful.domain, lawful.codomain, lawful.tensor * 0.5, algebra=lawful.algebra
        )


def test_monad_check_rejects_a_broken_join() -> None:
    with pytest.raises(AssertionError, match="left unit"):
        check_monad_laws(_ScaledJoin(GODEL), A)


def test_functor_check_rejects_a_broken_fmap() -> None:
    torch.manual_seed(0)
    f = observed(A, B, torch.rand(2, 3))
    g = observed(B, C, torch.rand(3, 2))
    with pytest.raises(AssertionError, match="identity law"):
        check_functor_laws(_ScaledFmap(GODEL), A, B, C, f, g)
