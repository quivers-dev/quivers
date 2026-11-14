"""Data-derived, expression-derived, and hyperparameter-dependent
initializers for ``observed`` / ``latent`` morphism declarations.

Three kinds of initializers are tested end-to-end:

* ``observed f : A -> B = from_data("KEY")`` — runtime-supplied
  frozen tensor. The compiler resolves the string key against the
  ``data=`` dictionary passed to :func:`quivers.dsl.loads` and
  binds the resulting tensor as the morphism's data buffer.
* ``observed f : A -> B = (g >> h).freeze`` — expression-derived
  initializer whose constituent morphisms' parameters are pinned;
  gradient flow stops at the freeze.
* ``latent f : A -> B [scale=0.1]`` — hyperparameter-dependent
  initialisation already supported via the option block; tested
  for regression.
"""

from __future__ import annotations

import pytest
import torch
import textwrap

from quivers.dsl import loads


# ---------------------------------------------------------------------------
# from_data initializer
# ---------------------------------------------------------------------------


def test_from_data_binds_supplied_tensor() -> None:
    """``observed f : A -> B = from_data("KEY")`` binds the
    supplied tensor as the morphism's data buffer."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4

    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    data_tensor = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]]
    )
    m = loads(textwrap.dedent(src), data={"H": data_tensor})
    assert m.morphism.tensor.shape == (3, 4)
    assert torch.allclose(m.morphism.tensor, data_tensor)
    assert m.morphism.domain.cardinality == 3
    assert m.morphism.codomain.cardinality == 4


def test_from_data_unknown_key_errors() -> None:
    from quivers.dsl.compiler import CompileError

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("MISSING_KEY")
    export h
    """
    with pytest.raises(CompileError, match="unknown data key"):
        loads(textwrap.dedent(src), data={"OTHER_KEY": torch.zeros(2, 2)})


def test_from_data_without_data_dict_errors() -> None:
    from quivers.dsl.compiler import CompileError

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("KEY")
    export h
    """
    with pytest.raises(CompileError, match="unknown data key"):
        loads(textwrap.dedent(src))


def test_from_data_shape_mismatch_with_declared_types_errors() -> None:
    """A data tensor whose shape doesn't match the declared
    domain/codomain shapes must be rejected with a shape-specific error."""
    from quivers.dsl.compiler import CompileError

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    # Wrong shape: (2, 4) instead of (3, 4).
    with pytest.raises(CompileError):
        loads(textwrap.dedent(src), data={"H": torch.zeros(2, 4)})


def test_from_data_does_not_register_parameters() -> None:
    """A ``from_data``-initialized morphism is structural / frozen;
    its tensor is a buffer, not an ``nn.Parameter``."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.tensor([[0.1, 0.2], [0.3, 0.4]])})
    params = list(m.parameters())
    assert len(params) == 0


# ---------------------------------------------------------------------------
# .freeze postfix
# ---------------------------------------------------------------------------


def test_freeze_materialises_composition_tensor() -> None:
    """``(f >> g).freeze`` produces an :class:`ObservedMorphism`
    whose tensor equals the composition's materialised tensor."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    object C : FinSet 2

    morphism f : A -> B [role=parameter]
    morphism g : B -> C [role=parameter]

    define chain = f >> g
    define frozen = chain.freeze
    export frozen
    """
    m = loads(textwrap.dedent(src))
    # Frozen morphism has no learnable parameters.
    params = list(m.parameters())
    assert len(params) == 0
    # Tensor shape matches the composition.
    assert m.morphism.tensor.shape == (2, 2)


def test_freeze_detaches_gradients() -> None:
    """Gradients do not propagate through a freeze boundary."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    morphism f : A -> B [role=parameter]
    define frozen = f.freeze
    export frozen
    """
    m = loads(textwrap.dedent(src))
    loss = m.morphism.tensor.sum()
    # ``frozen`` carries a detached tensor; ``loss.backward()``
    # has no grad-requiring leaves, so the call raises rather
    # than mutating anyone's .grad.
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


# ---------------------------------------------------------------------------
# Expression-derived initializer (no freeze): parameters propagate
# ---------------------------------------------------------------------------


def test_expression_initializer_propagates_parameters() -> None:
    """``observed h : A -> B = f`` (or any other expression) is an
    alias bound through the compiler; the underlying morphism's
    parameters are reachable via the alias. Without the
    ``.freeze`` modifier the binding does NOT detach."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    morphism f : A -> B [role=parameter]
    morphism h : A -> B [role=observed] ~ f
    export h
    """
    m = loads(textwrap.dedent(src))
    # ``h`` is bound to ``f``'s underlying LatentMorphism; its
    # tensor has gradient lineage.
    assert m.morphism.tensor.requires_grad


# ---------------------------------------------------------------------------
# Hyperparameter-dependent initialisation (option block)
# ---------------------------------------------------------------------------


def test_latent_morphism_accepts_scale_option() -> None:
    """The ``[scale=...]`` option block already supports
    hyperparameter-dependent init for ``latent`` morphisms; this
    is a regression test for the surface."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 5
    object B : FinSet 5
    morphism f : A -> B [role=parameter, scale=0.01]
    export f
    """
    m = loads(textwrap.dedent(src))
    # With a tiny scale the raw parameter starts near zero, so
    # sigmoid(raw) starts near 0.5.
    t = m.morphism.tensor
    assert t.shape == (5, 5)
    # All entries should be in [0, 1] (sigmoid output).
    assert torch.all((t >= 0) & (t <= 1))
    # Mean should be close to 0.5 because scale is tiny.
    assert abs(float(t.mean()) - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Combination: from_data with a downstream learnable morphism
# ---------------------------------------------------------------------------


def test_from_data_composed_with_latent_yields_learnable_chain() -> None:
    """A frozen data-derived morphism composed with a learnable
    one yields a chain whose only learnable parameters come from
    the learnable side."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    object C : FinSet 5

    morphism h : A -> B [role=observed] ~ from_data("H")
    morphism g : B -> C [role=parameter]
    define chain = h >> g
    export chain
    """
    H = torch.rand(3, 4)
    m = loads(textwrap.dedent(src), data={"H": H})
    # The chain's parameters come exclusively from ``g`` (since
    # ``h`` is a frozen data-derived morphism).
    params = list(m.parameters())
    assert len(params) == 1
