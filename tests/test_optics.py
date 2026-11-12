"""Tests for optics (lenses, prisms, adapters)."""

import torch
import pytest
from quivers.core.objects import FinSet, ProductSet, CoproductSet
from quivers.core.morphisms import observed, identity
from quivers.enriched.optics import Adapter, ComposedOptic, Lens, Prism, compose_optics


class TestLens:
    def test_lens_get(self):
        """Lens get should project the focus component."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = ProductSet(components=(a, b))
        lens = Lens(whole, focus_index=0)
        get_morph = lens.forward()
        assert get_morph.domain.shape == whole.shape
        assert get_morph.codomain.shape == a.shape
        for ai in range(2):
            for bi in range(3):
                assert get_morph.tensor[ai, bi, ai].item() == 1.0
                for ao in range(2):
                    if ao != ai:
                        assert get_morph.tensor[ai, bi, ao].item() == 0.0

    def test_lens_put(self):
        """Lens put should embed the focus back."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = ProductSet(components=(a, b))
        lens = Lens(whole, focus_index=0)
        put_morph = lens.backward()
        assert put_morph.domain.shape == a.shape
        assert put_morph.codomain.shape == whole.shape

    def test_lens_get_put_law(self):
        """get >> put should be idempotent (in the fuzzy sense)."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        whole = ProductSet(components=(a, b))
        lens = Lens(whole, focus_index=0)
        get_m = lens.forward()
        put_m = lens.backward()
        roundtrip = (get_m >> put_m).tensor
        assert roundtrip.shape == (*whole.shape, *whole.shape)

    def test_lens_requires_product(self):
        """Lens should require ProductSet."""
        with pytest.raises(TypeError, match="ProductSet"):
            Lens(FinSet(name="A", cardinality=3), focus_index=0)

    def test_lens_second_component(self):
        """Lens focusing on second component of a product."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = ProductSet(components=(a, b))
        lens = Lens(whole, focus_index=1)
        get_morph = lens.forward()
        assert get_morph.codomain.shape == b.shape
        for ai in range(2):
            for bi in range(3):
                assert get_morph.tensor[ai, bi, bi].item() == 1.0


class TestPrism:
    def test_prism_match(self):
        """Prism match should extract the focus component."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = CoproductSet(components=(a, b))
        prism = Prism(whole, focus_index=0)
        match_morph = prism.forward()
        assert match_morph.domain.shape == (5,)
        assert match_morph.codomain.shape == (2,)
        assert match_morph.tensor[0, 0].item() == 1.0
        assert match_morph.tensor[1, 1].item() == 1.0
        for i in range(2, 5):
            for j in range(2):
                assert match_morph.tensor[i, j].item() == 0.0

    def test_prism_build(self):
        """Prism build should inject the focus into the coproduct."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = CoproductSet(components=(a, b))
        prism = Prism(whole, focus_index=0)
        build_morph = prism.backward()
        assert build_morph.domain.shape == (2,)
        assert build_morph.codomain.shape == (5,)
        assert build_morph.tensor[0, 0].item() == 1.0
        assert build_morph.tensor[1, 1].item() == 1.0

    def test_prism_build_match_roundtrip(self):
        """build >> match = id for the focus."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = CoproductSet(components=(a, b))
        prism = Prism(whole, focus_index=0)
        build = prism.backward()
        match = prism.forward()
        roundtrip = (build >> match).tensor
        expected = identity(a).tensor
        torch.testing.assert_close(roundtrip, expected, atol=1e-05, rtol=1e-05)

    def test_prism_requires_coproduct(self):
        with pytest.raises(TypeError, match="CoproductSet"):
            Prism(FinSet(name="A", cardinality=3), focus_index=0)


class TestAdapter:
    def test_adapter_isomorphism(self):
        """Adapter with identity morphisms is an isomorphism."""
        a = FinSet(name="A", cardinality=3)
        fwd = identity(a)
        bwd = identity(a)
        adapter = Adapter(fwd, bwd)
        assert adapter.verify_isomorphism()

    def test_adapter_permutation(self):
        """Adapter with a permutation isomorphism."""
        a = FinSet(name="A", cardinality=2)
        swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fwd = observed(a, a, swap)
        bwd = observed(a, a, swap)
        adapter = Adapter(fwd, bwd)
        assert adapter.verify_isomorphism()


class TestComposeOptics:
    def test_compose_lenses(self):
        """Composing two lenses should produce a valid optic."""
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        FinSet(name="C", cardinality=2)
        whole = ProductSet(components=(a, b))
        lens1 = Lens(whole, focus_index=0)
        id_a = identity(a)
        adapter = Adapter(id_a, id_a)
        composed = compose_optics(lens1, adapter)
        fwd = composed.forward()
        composed.backward()
        assert fwd.codomain.shape == a.shape


class TestLensLaws:
    """GetPut, PutGet and PutPut for `Lens.forward` and `Lens.put`, as
    equations between functions on ``S = A × B``."""

    @staticmethod
    def _setup():
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = ProductSet(components=(a, b))
        return a, b, whole, Lens(whole, focus_index=0)

    def test_lens_laws(self):
        a, b, whole, lens = self._setup()
        get = lens.forward().tensor.reshape(6, 2)
        put = lens.put().tensor.reshape(6, 2, 6)
        get_f = get.argmax(dim=1)
        put_f = put.argmax(dim=2)
        assert torch.equal(get.sum(dim=1), torch.ones(6))
        assert torch.equal(put.sum(dim=2), torch.ones(6, 2))
        for s in range(6):
            assert put_f[s, get_f[s]] == s  # GetPut
            for x in range(2):
                assert get_f[put_f[s, x]] == x  # PutGet
                for y in range(2):
                    assert put_f[put_f[s, x], y] == put_f[s, y]  # PutPut

    def test_backward_is_the_converse_of_get(self):
        _, _, _, lens = self._setup()
        get = lens.forward().tensor.reshape(6, 2)
        back = lens.backward().tensor.reshape(2, 6)
        torch.testing.assert_close(back, get.T)


class TestAdapterAndComposition:
    def test_adapter_refuses_a_pair_that_is_not_an_isomorphism(self):
        a = FinSet(name="A", cardinality=2)
        collapse = observed(a, a, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        with pytest.raises(ValueError, match="not mutually inverse"):
            Adapter(collapse, collapse)

    def test_composed_lenses_are_not_adapters(self):
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        whole = ProductSet(components=(a, b))
        composed = compose_optics(Lens(whole, 0), Adapter(identity(a), identity(a)))
        assert isinstance(composed, ComposedOptic)
        assert not isinstance(composed, Adapter)
