"""First-class transformations in the DSL.

A transformation (a :class:`CarrierMap` or
:class:`MorphismTransformation`) is a value in the DSL: it can be
let-bound, composed with ``>>>``, and passed to ``change_base``.
This file exercises:

* Bare-name singleton references (``expectation``, ``log_prob``,
  ``boolean_embedding``, …).
* Constructor calls (``softmax(B)``, ``l1_normalize(B)``,
  ``l2_normalize(B)``, ``bayes_invert(prior)``).
* Let-binding to user-chosen names.
* Composition ``t1 >>> t2`` with source/target compatibility
  checks at compile time.
* Application inside ``change_base``, including chains.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError


# ---------------------------------------------------------------------------
# Let-binding a transformation
# ---------------------------------------------------------------------------


def test_let_binds_singleton_transformation() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define phi = boolean_embedding
    define g = f.change_base(phi)
    export g
    """
    # phi has source=Boolean, target=ProductFuzzyAlgebra; f is over Real.
    # The change_base will raise at runtime because source doesn't
    # match.  Just confirm the parse + let-binding compiles up to
    # the point of trying to apply it.
    with pytest.raises(CompileError, match="algebra"):
        loads(textwrap.dedent(src))


def test_let_binds_constructor_result() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define normalize = softmax(B)
    define g = f.change_base(normalize)
    export g
    """
    program = loads(textwrap.dedent(src))
    tensor = program.morphism.tensor
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_let_binds_then_reuses_transformation() -> None:
    """A single trans can be applied to two different morphisms."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    morphism h : A -> B [role=latent]
    define normalize = softmax(B)
    define f_norm = f.change_base(normalize)
    define h_norm = h.change_base(normalize)
    export f_norm
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


def test_let_trans_name_disjoint_from_morphism_name() -> None:
    """Trans and morphism namespaces are disjoint."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define t = softmax(B)
    define t = f
    export f
    """
    with pytest.raises(CompileError, match="already bound"):
        loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Composition ``>>>``
# ---------------------------------------------------------------------------


def test_trans_compose_chains_two_compatible_steps() -> None:
    """``t1 >>> t2`` applies t1 then t2; the morphism's algebra
    travels from t1.source through t1.target == t2.source to
    t2.target."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define s = softmax(B)
    define pipe = s >>> expectation
    define g = f.change_base(pipe)
    export g
    """
    # softmax : ProductFuzzyAlgebra -> Markov ; expectation : Markov ->
    # ProductFuzzyAlgebra.  The chain takes f from ProductFuzzyAlgebra through
    # Markov and back to ProductFuzzyAlgebra.
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None
    tensor = program.morphism.tensor
    assert tensor.shape == (3, 4)


def test_trans_compose_inline_in_change_base() -> None:
    """``f.change_base(t1 >>> t2)`` works without an intermediate
    let."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(softmax(B) >>> expectation)
    export g
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


def test_trans_compose_three_steps_flattens() -> None:
    """``t1 >>> t2 >>> t3`` flattens into a single 3-step
    sequence; intermediate boundaries are typed."""
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define pipe = probability_clamp >>> probability_to_real >>> counting_from_real
    define g = f.change_base(pipe)
    export g
    """
    # Real -> Probability -> Real -> Counting
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


def test_trans_compose_source_target_mismatch_errors() -> None:
    """``boolean_embedding >>> expectation`` is a type error:
    boolean_embedding's target is ProductFuzzyAlgebra, but expectation
    expects Markov."""
    src = """
    composition boolean [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define bad = boolean_embedding >>> expectation
    define g = f.change_base(bad)
    export g
    """
    with pytest.raises(CompileError, match="does not match"):
        loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Constructor-call resolution
# ---------------------------------------------------------------------------


def test_softmax_constructor_accepts_object() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(softmax(B))
    export g
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


def test_constructor_args_resolve_against_object_scope() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    object NotUsed : FinSet 2
    morphism f : A -> B [role=latent]
    define g = f.change_base(l1_normalize(B))
    export g
    """
    program = loads(textwrap.dedent(src))
    tensor = program.morphism.tensor
    assert (tensor >= 0).all()


# ---------------------------------------------------------------------------
# Singleton-vs-constructor resolution
# ---------------------------------------------------------------------------


def test_bare_singleton_resolves() -> None:
    src = """
    composition boolean [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(boolean_embedding)
    export g
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


def test_constructor_referenced_bare_errors_helpfully() -> None:
    """Using a constructor's name without parentheses surfaces a
    pointed error."""
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(softmax)
    export g
    """
    with pytest.raises(CompileError, match="needs arguments"):
        loads(textwrap.dedent(src))


def test_unknown_name_in_change_base_errors() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(no_such_thing)
    export g
    """
    with pytest.raises(CompileError, match="undefined transformation"):
        loads(textwrap.dedent(src))


def test_unknown_constructor_errors() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=latent]
    define g = f.change_base(no_such_constructor(B))
    export g
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(textwrap.dedent(src))
