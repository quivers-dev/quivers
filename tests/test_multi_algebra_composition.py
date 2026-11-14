"""Multi-algebra composition and change-of-base for V-Cat
morphisms.

A morphism in :mod:`quivers.core.morphisms` carries an enrichment
algebra. Composing two morphisms over a shared algebra uses that
algebra's monoidal structure; composing across algebras requires
an algebra homomorphism (a lax monoidal poset functor) applied via
:meth:`Morphism.change_base`.

The DSL exposes two sequential composition operators: ``>>``
composes in the operands' shared algebra, and ``<<`` is reversed
``>>`` (``g << f`` compiles as ``f >> g``). Neither operator
auto-base-changes: operands whose declared algebras differ must be
brought into a common algebra with an explicit ``.change_base(φ)``
first.

This module verifies:

1. The Python-level ``change_base`` adapter: applying each
   canonical homomorphism to a morphism's tensor produces a new
   morphism over the target algebra with the expected
   per-entry mapping.
2. Composition over each non-default algebra (Markov, LogProb,
   Gödel, Viterbi, Boolean, Łukasiewicz) produces a
   :class:`ComposedMorphism` over the right algebra.
3. The DSL operators ``>>`` and ``<<`` compose in the declared
   algebra, and cross-algebra composition without an explicit
   base change raises a typed error.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.core.algebras import (
    GODEL,
    LOG_PROB,
    LUKASIEWICZ,
    MAX_PLUS,
    TROPICAL,
)
from quivers.core.morphisms import (
    ComposedMorphism,
    LatentMorphism,
    ObservedMorphism,
)
from quivers.core.objects import FinSet
from quivers.core.algebra_morphisms import (
    IdentityHom,
    LOG_PROB_MAP,
    PRODUCT_FUZZY_TO_GODEL,
    MAX_PLUS_MAP,
    Threshold,
    embedding,
    lookup_carrier_map,
    threshold,
)
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY
from quivers.core.algebras import MARKOV


# ---------------------------------------------------------------------------
# change_base on Morphism
# ---------------------------------------------------------------------------


def test_change_base_identity_preserves_tensor() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=3)
    f = LatentMorphism(A, B)
    g = f.change_base(IdentityHom(PRODUCT_FUZZY))
    assert g.algebra.name == "ProductFuzzy"
    assert torch.allclose(g.tensor, f.tensor)


def test_change_base_to_log_prob_returns_log_of_tensor() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(LOG_PROB_MAP)
    expected = torch.log(data.clamp(min=1e-30))
    assert torch.allclose(g.tensor, expected)


def test_change_base_to_boolean_via_threshold() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.7], [0.3, 0.9]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(threshold(0.5))
    expected = (data > 0.5).to(dtype=torch.float32)
    assert torch.allclose(g.tensor, expected)
    assert g.algebra.name == "Boolean"


def test_change_base_to_godel_via_material_implication() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(PRODUCT_FUZZY_TO_GODEL)
    assert g.algebra.name == "Godel"
    # Material implication clamps to [0, 1] but otherwise preserves
    # entry values; the carry-through is exact for inputs already
    # in [0, 1].
    assert torch.allclose(g.tensor, data)


def test_change_base_to_max_plus_via_log() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(MAX_PLUS_MAP)
    expected = torch.log(data.clamp(min=1e-30))
    assert torch.allclose(g.tensor, expected)


def test_change_base_rejects_wrong_source_algebra() -> None:
    """A homomorphism's source must match the morphism's algebra;
    applying ``LOG_PROB_MAP`` (which expects ``ProductFuzzyAlgebra``) to
    a morphism declared over Markov raises an algebra-specific error."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    f = ObservedMorphism(A, B, data, algebra=MARKOV)
    with pytest.raises(TypeError, match="does not match"):
        f.change_base(LOG_PROB_MAP)


def test_change_base_rejects_non_homomorphism() -> None:
    A = FinSet(name="A", cardinality=2)
    f = LatentMorphism(A, A)
    with pytest.raises(TypeError, match="CarrierMap"):
        f.change_base("not a homomorphism")


def test_change_base_chain_through_two_homomorphisms() -> None:
    """Chaining ``change_base`` calls realises composition of
    algebra homomorphisms. Going ProductFuzzyAlgebra → Boolean (via
    threshold) → ProductFuzzyAlgebra (via embedding) recovers a Boolean-
    valued tensor embedded back in [0, 1]."""
    A = FinSet(name="A", cardinality=2)
    data = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    f = ObservedMorphism(A, A, data)
    g = f.change_base(threshold(0.5)).change_base(embedding(BOOLEAN, PRODUCT_FUZZY))
    expected = (data > 0.5).to(dtype=torch.float32)
    assert torch.allclose(g.tensor, expected)
    assert g.algebra.name == "ProductFuzzy"


def test_change_base_preserves_gradients_to_morphism_parameters() -> None:
    """A learnable LatentMorphism's parameters remain gradient-
    connected through the base change."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    f = LatentMorphism(A, B)
    g = f.change_base(LOG_PROB_MAP)
    loss = g.tensor.sum()
    loss.backward()
    raw = f.raw
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


# ---------------------------------------------------------------------------
# Composition operator dispatch (DSL surface)
# ---------------------------------------------------------------------------


def test_default_compose_uses_product_fuzzy() -> None:
    from quivers.dsl import loads

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    object C : FinSet 3
    morphism f : A -> B [role=parameter]
    morphism g : B -> C [role=parameter]
    define chain = f >> g
    export chain
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None
    assert m.morphism.algebra.name == "ProductFuzzy"


def test_reverse_compose_matches_forward_compose() -> None:
    """``g << f`` compiles as ``f >> g``: same signature, same
    algebra, same tensor."""
    from quivers.dsl import loads

    header = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    object C : FinSet 2
    morphism f : A -> B [role=observed] ~ from_data("F")
    morphism g : B -> C [role=observed] ~ from_data("G")
    """
    torch.manual_seed(0)
    data = {"F": torch.rand(3, 4), "G": torch.rand(4, 2)}
    forward = loads(
        textwrap.dedent(header + "    define chain = f >> g\n    export chain\n"),
        data=data,
    )
    reverse = loads(
        textwrap.dedent(header + "    define chain = g << f\n    export chain\n"),
        data=data,
    )
    fwd = forward.morphism
    rev = reverse.morphism
    assert isinstance(fwd, ComposedMorphism)
    assert isinstance(rev, ComposedMorphism)
    assert rev.algebra.name == "ProductFuzzy"
    assert torch.allclose(rev.tensor, fwd.tensor)


def test_cross_algebra_compose_without_change_base_errors() -> None:
    """``>>`` composes in the operands' shared algebra; operands
    whose declared algebras differ raise a typed error unless an
    explicit base change brings them into a common algebra."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B)
    g = LatentMorphism(B, C, algebra=MARKOV)
    with pytest.raises(TypeError, match="incompatible algebras"):
        _ = f >> g


# ---------------------------------------------------------------------------
# Composition over each non-default algebra
# ---------------------------------------------------------------------------


def _markov_chain(A_size: int, B_size: int, C_size: int) -> tuple:
    A = FinSet(name="A", cardinality=A_size)
    B = FinSet(name="B", cardinality=B_size)
    C = FinSet(name="C", cardinality=C_size)
    f = LatentMorphism(A, B, algebra=MARKOV)
    g = LatentMorphism(B, C, algebra=MARKOV)
    return A, B, C, f, g


def test_markov_compose_produces_markov_composed_morphism() -> None:
    A, B, C, f, g = _markov_chain(3, 3, 3)
    chain = f >> g
    assert isinstance(chain, ComposedMorphism)
    assert chain.algebra.name == "Markov"
    assert chain.domain is A
    assert chain.codomain is C


def test_log_prob_compose_produces_log_prob_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=LOG_PROB)
    g = LatentMorphism(B, C, algebra=LOG_PROB)
    chain = f >> g
    assert chain.algebra.name == "LogProb"


def test_godel_compose_produces_godel_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=GODEL)
    g = LatentMorphism(B, C, algebra=GODEL)
    chain = f >> g
    assert chain.algebra.name == "Godel"


def test_viterbi_compose_produces_max_plus_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=MAX_PLUS)
    g = LatentMorphism(B, C, algebra=MAX_PLUS)
    chain = f >> g
    assert chain.algebra.name == "MaxPlus"


def test_boolean_compose_produces_boolean_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=BOOLEAN)
    g = LatentMorphism(B, C, algebra=BOOLEAN)
    chain = f >> g
    assert chain.algebra.name == "Boolean"


def test_lukasiewicz_compose_produces_lukasiewicz_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=LUKASIEWICZ)
    g = LatentMorphism(B, C, algebra=LUKASIEWICZ)
    chain = f >> g
    assert chain.algebra.name == "Lukasiewicz"


def test_tropical_min_plus_compose() -> None:
    """The existing shortest-path Tropical algebra also composes;
    no dedicated operator but the existing ``>>`` works once both
    operands are declared over it."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, algebra=TROPICAL)
    g = LatentMorphism(B, C, algebra=TROPICAL)
    chain = f >> g
    assert chain.algebra.name == "Tropical"


# ---------------------------------------------------------------------------
# Homomorphism registry lookup
# ---------------------------------------------------------------------------


def test_homomorphism_registry_returns_expectation_for_markov_to_pf() -> None:
    phi = lookup_carrier_map(MARKOV, PRODUCT_FUZZY)
    assert phi is not None
    assert phi.source.name == "Markov"
    assert phi.target.name == "ProductFuzzy"


def test_homomorphism_registry_returns_identity_for_same_algebra() -> None:
    phi = lookup_carrier_map(PRODUCT_FUZZY, PRODUCT_FUZZY)
    assert phi is not None
    assert isinstance(phi, IdentityHom)


def test_homomorphism_registry_returns_none_for_unknown_pair() -> None:
    """Unknown pairs return None so the caller can decide whether
    to raise or to construct a custom homomorphism."""
    from quivers.core.algebras import ProductFuzzyAlgebra

    class _BogusAlgebra(ProductFuzzyAlgebra):
        @property
        def name(self):
            return "BogusUnregistered"

    phi = lookup_carrier_map(_BogusAlgebra(), PRODUCT_FUZZY)
    assert phi is None


def test_threshold_homomorphism_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau must be in"):
        Threshold(tau=1.5)
    with pytest.raises(ValueError, match="tau must be in"):
        Threshold(tau=-0.1)


# ---------------------------------------------------------------------------
# Pythonic interplay with change_base + compose
# ---------------------------------------------------------------------------


def test_change_base_then_compose_in_target_algebra() -> None:
    """Bring a ProductFuzzyAlgebra morphism into the Markov algebra
    via a custom homomorphism, then compose with a Markov-native
    morphism."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    # Construct a "noop" homomorphism from ProductFuzzyAlgebra to Markov.
    from quivers.core.algebra_morphisms import CarrierMap

    class ProductToMarkov(CarrierMap):
        @property
        def source(self):
            return PRODUCT_FUZZY

        @property
        def target(self):
            return MARKOV

        def apply(self, t):
            # Row-normalize.
            return t / t.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    f_pf = LatentMorphism(A, B)
    f_markov = f_pf.change_base(ProductToMarkov())
    assert f_markov.algebra.name == "Markov"
    g_markov = LatentMorphism(B, C, algebra=MARKOV)
    chain = f_markov >> g_markov
    assert chain.algebra.name == "Markov"
