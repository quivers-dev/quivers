"""Tests for :class:`TransformedMorphism` autograd safety.

Cover the four morphism transforms that return a derived-from-source
morphism (``change_base``, ``.dagger``, ``.trace``, ``.refactor``).
Each must support:

* Multi-step ``loss.backward()`` over fresh ``.tensor`` accesses.
* Source parameter exposure via ``.module().parameters()``.
* End-to-end Adam optimisation that actually moves the source's
  learnable parameters.
"""

from __future__ import annotations

import torch

from quivers.core.morphisms import (
    LatentMorphism,
    ObservedMorphism,
    TransformedMorphism,
    identity,
)
from quivers.core.objects import FinSet, ProductSet
from quivers.core.algebra_morphisms import PRODUCT_FUZZY_TO_GODEL
from quivers.core.algebras import MARKOV


class TestChangeBaseAutograd:
    def _setup(self):
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B)
        return A, B, f

    def test_returns_transformed_morphism(self):
        A, B, f = self._setup()
        g = f.change_base(PRODUCT_FUZZY_TO_GODEL)
        assert isinstance(g, TransformedMorphism)

    def test_exposes_source_parameters(self):
        A, B, f = self._setup()
        g = f.change_base(PRODUCT_FUZZY_TO_GODEL)
        params = list(g.module().parameters())
        assert sum(p.numel() for p in params) == 12  # 3 * 4

    def test_multi_step_backward(self):
        A, B, f = self._setup()
        g = f.change_base(PRODUCT_FUZZY_TO_GODEL)
        g.tensor.sum().backward()
        f.raw.grad = None
        # Second backward must not raise.
        g.tensor.sum().backward()
        assert f.raw.grad is not None

    def test_adam_moves_source(self):
        A, B, f = self._setup()
        g = f.change_base(PRODUCT_FUZZY_TO_GODEL)
        initial = f.raw.detach().clone()
        opt = torch.optim.Adam(g.module().parameters(), lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            loss = (g.tensor - 0.5).pow(2).sum()
            loss.backward()
            opt.step()
        assert (f.raw.detach() - initial).norm().item() > 0.5


class TestDaggerAutograd:
    def test_returns_transformed_morphism(self):
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B, algebra=MARKOV)
        g = f.dagger
        assert isinstance(g, TransformedMorphism)
        assert g.domain == B
        assert g.codomain == A

    def test_multi_step_backward(self):
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B, algebra=MARKOV)
        g = f.dagger
        g.tensor.sum().backward()
        f.raw.grad = None
        g.tensor.sum().backward()
        assert f.raw.grad is not None

    def test_adam_moves_source(self):
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B, algebra=MARKOV)
        g = f.dagger
        initial = f.raw.detach().clone()
        opt = torch.optim.Adam(g.module().parameters(), lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            loss = (g.tensor - 0.5).pow(2).sum()
            loss.backward()
            opt.step()
        assert (f.raw.detach() - initial).norm().item() > 0.3


class TestTraceAutograd:
    def _setup(self):
        A = FinSet(name="A", cardinality=3)
        X = FinSet(name="X", cardinality=2)
        Y = FinSet(name="Y", cardinality=2)
        dom = ProductSet(components=(A, X))
        cod = ProductSet(components=(A, Y))
        f = LatentMorphism(dom, cod)
        return A, X, Y, f

    def test_returns_transformed_morphism(self):
        A, X, Y, f = self._setup()
        g = f.trace(A)
        assert isinstance(g, TransformedMorphism)
        assert g.domain == X
        assert g.codomain == Y

    def test_multi_step_backward(self):
        A, X, Y, f = self._setup()
        g = f.trace(A)
        g.tensor.sum().backward()
        f.raw.grad = None
        g.tensor.sum().backward()
        assert f.raw.grad is not None

    def test_adam_moves_source(self):
        A, X, Y, f = self._setup()
        g = f.trace(A)
        initial = f.raw.detach().clone()
        opt = torch.optim.Adam(g.module().parameters(), lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            loss = (g.tensor - 0.5).pow(2).sum()
            loss.backward()
            opt.step()
        assert (f.raw.detach() - initial).norm().item() > 0.3


class TestRefactorAutograd:
    def test_returns_transformed_morphism(self):
        A = FinSet(name="A", cardinality=6)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B)
        A2 = FinSet(name="A2_a", cardinality=2)
        A3 = FinSet(name="A2_b", cardinality=3)
        prod = ProductSet(components=(A2, A3))
        g = f.refactor(domain=prod)
        assert isinstance(g, TransformedMorphism)

    def test_multi_step_backward(self):
        A = FinSet(name="A", cardinality=6)
        B = FinSet(name="B", cardinality=4)
        f = LatentMorphism(A, B)
        A2 = FinSet(name="A2_a", cardinality=2)
        A3 = FinSet(name="A2_b", cardinality=3)
        prod = ProductSet(components=(A2, A3))
        g = f.refactor(domain=prod)
        g.tensor.sum().backward()
        f.raw.grad = None
        g.tensor.sum().backward()
        assert f.raw.grad is not None


class TestComposedChain:
    def test_composed_chain_exposes_both_sources(self):
        # f.change_base(phi) >> g.change_base(psi) must expose both
        # f.raw and g.raw under chain.module().parameters().
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        C = FinSet(name="C", cardinality=2)
        f = LatentMorphism(A, B)
        g = LatentMorphism(B, C)
        chain = f.change_base(PRODUCT_FUZZY_TO_GODEL) >> g.change_base(
            PRODUCT_FUZZY_TO_GODEL
        )
        params = list(chain.module().parameters())
        # f has 3*4 = 12 params; g has 4*2 = 8 params; total = 20.
        assert sum(p.numel() for p in params) == 20

    def test_composed_chain_multi_step_backward(self):
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        C = FinSet(name="C", cardinality=2)
        f = LatentMorphism(A, B)
        g = LatentMorphism(B, C)
        chain = f.change_base(PRODUCT_FUZZY_TO_GODEL) >> g.change_base(
            PRODUCT_FUZZY_TO_GODEL
        )
        chain.tensor.sum().backward()
        f.raw.grad = None
        g.raw.grad = None
        # Second backward must not raise.
        chain.tensor.sum().backward()
        assert f.raw.grad is not None
        assert g.raw.grad is not None


class TestObservedMorphismUnchanged:
    """The fix must not regress the behaviour of truly-frozen
    ``ObservedMorphism`` tensors built from ``from_data`` style
    fixed buffers.
    """

    def test_observed_morphism_still_buffer_backed(self):
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=2)
        data = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
        f = ObservedMorphism(A, B, data)
        # Same buffer reference across accesses.
        assert f.tensor is f.tensor
        # No parameters.
        assert sum(p.numel() for p in f.module().parameters()) == 0

    def test_identity_morphism_unchanged(self):
        A = FinSet(name="A", cardinality=3)
        I = identity(A)
        assert isinstance(I, ObservedMorphism)
        # Identity tensor has the expected diagonal shape.
        assert I.tensor.shape == (3, 3)
