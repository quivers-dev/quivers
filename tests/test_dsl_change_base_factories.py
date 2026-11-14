"""DSL surface for parametric morphism transformations.

Exercises the constructor-call form of :class:`MorphismTransformation`
values inside ``change_base``:

* ``f.change_base(softmax(B))``: constructor with an object argument
  resolved at compile time.
* ``f.change_base(l1_normalize(B))``: same shape.
* ``f.change_base(l2_normalize(B))``: same shape, identity target
  algebra.
* ``f.change_base(bayes_invert(prior))``: constructor with a morphism
  argument; the prior's tensor is read when the constructor runs to
  produce the :class:`BayesInvert`.
* Bare-name lookup paths still resolve singletons.
* Helpful error messages for misuse (bare constructor without
  args, unknown constructor, unknown argument).
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError


# ---------------------------------------------------------------------------
# Object-argument constructors
# ---------------------------------------------------------------------------


def test_softmax_compiles() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(softmax(B))
    export g
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None
    # The result tensor should be row-stochastic along B.
    tensor = program.morphism.tensor
    assert tensor.shape == (3, 4)
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_l1_normalize_compiles() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(l1_normalize(B))
    export g
    """
    program = loads(textwrap.dedent(src))
    tensor = program.morphism.tensor
    # All entries non-negative and row-summing to 1.
    assert (tensor >= 0).all()
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_l2_normalize_compiles() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(l2_normalize(B))
    export g
    """
    program = loads(textwrap.dedent(src))
    tensor = program.morphism.tensor
    norms = tensor.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Morphism-argument constructor
# ---------------------------------------------------------------------------


def test_bayes_invert_with_morphism_prior() -> None:
    src = """
    composition markov [level=algebra]
    object Unit : FinSet 1
    object A : FinSet 3
    morphism prior : Unit -> A [role=observed] ~ from_data("PRIOR")
    morphism f : A -> A [role=parameter]
    define g = f.change_base(bayes_invert(prior))
    export g
    """
    prior = torch.tensor([[0.5, 0.3, 0.2]])
    program = loads(textwrap.dedent(src), data={"PRIOR": prior})
    assert program.morphism is not None
    out = program.morphism.tensor
    # BayesInvert returns a Markov kernel (rows sum to 1).
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Bare-name lookups still work
# ---------------------------------------------------------------------------


def test_bare_name_homomorphism_still_resolves() -> None:
    src = """
    composition boolean [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(boolean_embedding)
    export g
    """
    program = loads(textwrap.dedent(src))
    assert program.morphism is not None


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


def test_bare_constructor_without_args_errors() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(softmax)
    export g
    """
    with pytest.raises(CompileError, match="needs arguments"):
        loads(textwrap.dedent(src))


def test_unknown_constructor_errors() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(not_a_real_constructor(B))
    export g
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(textwrap.dedent(src))


def test_unknown_constructor_arg_errors() -> None:
    src = """
    composition real [level=algebra]
    object A : FinSet 3
    object B : FinSet 4
    morphism f : A -> B [role=parameter]
    define g = f.change_base(softmax(NotAnObject))
    export g
    """
    with pytest.raises(CompileError, match="unresolved"):
        loads(textwrap.dedent(src))
