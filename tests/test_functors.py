"""Tests for Functor ABC and FreeMonoidFunctor."""

import torch
import pytest
from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import morphism, observed, FunctorMorphism
from quivers.categorical.functors import FreeMonoidFunctor
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY, REAL


class TestFreeMonoidFunctorObject:
    def test_map_object(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet(name="A", cardinality=3)
        result = fm.map_object(a)
        assert isinstance(result, FreeMonoid)
        assert result.generators == a
        assert result.max_length == 2
        assert result.size == 13

    def test_map_object_accepts_non_finset(self):
        """FreeMonoidFunctor accepts any SetObject via proxy FinSet."""
        fm = FreeMonoidFunctor(max_length=1)
        inner = FreeMonoid(generators=FinSet(name="A", cardinality=2), max_length=1)
        result = fm.map_object(inner)
        assert isinstance(result, FreeMonoid)
        assert result.size == 4

    def test_max_length_zero(self):
        fm = FreeMonoidFunctor(max_length=0)
        a = FinSet(name="A", cardinality=5)
        result = fm.map_object(a)
        assert result.size == 1


class TestFreeMonoidFunctorMorphism:
    def test_map_morphism_returns_functor_morphism(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        f = morphism(a, b, algebra=REAL)
        result = fm.map_morphism(f)
        assert isinstance(result, FunctorMorphism)

    def test_map_morphism_domain_codomain(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = morphism(a, b, algebra=REAL)
        result = fm.map_morphism(f)
        assert isinstance(result.domain, FreeMonoid)
        assert isinstance(result.codomain, FreeMonoid)
        assert result.domain.generators == a
        assert result.codomain.generators == b

    def test_map_morphism_tensor_shape(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = morphism(a, b, algebra=REAL)
        result = fm.map_morphism(f)
        assert result.tensor.shape == (7, 13)

    def test_block_diagonal_structure(self):
        """The functor morphism should have block-diagonal structure."""
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f_data = torch.tensor([[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]])
        f = observed(a, b, f_data, algebra=REAL)
        result = fm.map_morphism(f)
        t = result.tensor
        assert t[0, 0].item() == pytest.approx(1.0, abs=1e-06)
        assert t[0, 1].item() == pytest.approx(0.0, abs=1e-06)
        assert t[0, 2].item() == pytest.approx(0.0, abs=1e-06)
        assert t[0, 3].item() == pytest.approx(0.0, abs=1e-06)
        torch.testing.assert_close(t[1:3, 1:4], f_data, atol=1e-06, rtol=1e-06)
        assert t[1, 0].item() == pytest.approx(0.0, abs=1e-06)
        assert t[2, 0].item() == pytest.approx(0.0, abs=1e-06)


class TestFunctorPreservesComposition:
    """F(f >> g) ≈ F(f) >> F(g) on boolean inputs."""

    def test_preserves_composition_boolean(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        c = FinSet(name="C", cardinality=2)
        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        f = observed(a, b, f_data, algebra=REAL)
        g = observed(b, c, g_data, algebra=REAL)
        fg = f >> g
        fm_fg = fm.map_morphism(fg)
        fm_f = fm.map_morphism(f)
        fm_g = fm.map_morphism(g)
        composed = fm_f >> fm_g
        torch.testing.assert_close(
            fm_fg.tensor, composed.tensor, atol=0.0001, rtol=0.0001
        )


class TestGradientFlowThroughFunctor:
    def test_gradient_flows(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = morphism(a, b, algebra=REAL)
        result = fm.map_morphism(f)
        loss = result.tensor.sum()
        loss.backward()
        assert f.raw.grad is not None
        assert torch.isfinite(f.raw.grad).all()

    def test_module_delegates_to_inner(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = morphism(a, b, algebra=REAL)
        result = fm.map_morphism(f)
        inner_params = list(f.module().parameters())
        functor_params = list(result.module().parameters())
        assert len(inner_params) == len(functor_params)
        for ip, fp in zip(inner_params, functor_params):
            assert ip.data_ptr() == fp.data_ptr()


class TestAlgebraRespected:
    """FreeMonoidFunctor should use the morphism's algebra."""

    def test_boolean_functor(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f = observed(a, b, f_data, algebra=BOOLEAN)
        result = fm.map_morphism(f)
        assert result.algebra is BOOLEAN
        t = result.tensor
        assert t[0, 0].item() == pytest.approx(1.0, abs=1e-06)
        torch.testing.assert_close(t[1:3, 1:3], f_data, atol=1e-06, rtol=1e-06)


class TestFreeMonoidFunctorLaws:
    """``F(f >> g) = F(f) >> F(g)`` and ``F(id) = id`` for random real
    matrices with words up to length 2."""

    def test_functor_laws_over_the_reals(self):
        gen = torch.Generator().manual_seed(11)
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        c = FinSet(name="C", cardinality=2)
        f = observed(a, b, torch.rand(2, 3, generator=gen), algebra=REAL)
        g = observed(b, c, torch.rand(3, 2, generator=gen), algebra=REAL)
        torch.testing.assert_close(
            fm.map_morphism(f >> g).tensor,
            (fm.map_morphism(f) >> fm.map_morphism(g)).tensor,
        )
        id_a = observed(a, a, torch.eye(2), algebra=REAL)
        n = fm.map_object(a).size
        torch.testing.assert_close(fm.map_morphism(id_a).tensor, torch.eye(n))

    def test_refuses_an_algebra_that_is_not_a_semiring(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet(name="A", cardinality=2)
        with pytest.raises(TypeError, match="Semiring"):
            fm.map_morphism(observed(a, a, torch.eye(2), algebra=PRODUCT_FUZZY))
