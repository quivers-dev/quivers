"""End-to-end DSL coverage for the composition-rule hierarchy and the
operadic contraction surface.

Covers:

* ``composition NAME [level=LEVEL]`` keyword + level matching against
  the registry (``algebra`` / ``semigroupoid`` / ``bilinear_form`` /
  ``rule``).
* Inline-bodied composition rules: ``composition NAME [level=LEVEL]``
  with an indented body of ``key (args) = expr`` entries.
* ``contraction NAME (inputs) : DOM -> COD [rule=..., wiring=...]``.
* ``define z = op(arg1, arg2, ...)`` invoking a contraction or a
  parametric program template.

Also confirms that Algebra-only operations (``identity``, ``dagger``,
``trace``, ``cup``, ``cap``) raise a typed ``CompileError`` when the
active rule is not an Algebra.
"""

from __future__ import annotations

import textwrap

import pytest
import torch

from quivers.core.algebras import (
    Algebra,
    Magmoid,
    CompositionRule,
    Semigroupoid,
)
from quivers.dsl import loads
from quivers.dsl.compiler import CompileError


# ---------------------------------------------------------------------------
# Keyword + registry resolution
# ---------------------------------------------------------------------------


def test_algebra_keyword_resolves_algebra() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    assert isinstance(m.morphism.algebra, Algebra)


def test_bilinear_form_level_resolves_material_implication() -> None:
    """``material_impl`` is a magmoid: neither associative nor unital,
    so it sits at the ``bilinear_form`` level, the DSL's no-law level."""
    src = """
    composition material_impl [level=bilinear_form]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    assert isinstance(m.morphism.algebra, Magmoid)
    assert not isinstance(m.morphism.algebra, Algebra)


def test_semigroupoid_level_rejects_material_implication() -> None:
    """Material implication composes non-associatively, so declaring it
    at the ``semigroupoid`` level is a level mismatch."""
    src = """
    composition material_impl [level=semigroupoid]
    object A : FinSet 2
    """
    with pytest.raises(CompileError, match="not at level 'semigroupoid'"):
        loads(textwrap.dedent(src))


def test_semigroupoid_level_rejects_product_fuzzy() -> None:
    """The default algebra fails distributivity, so it is not offered
    at the ``semigroupoid`` level."""
    src = """
    composition product_fuzzy [level=semigroupoid]
    object A : FinSet 2
    """
    with pytest.raises(CompileError, match="not at level 'semigroupoid'"):
        loads(textwrap.dedent(src))


def test_semigroupoid_level_accepts_a_semiring() -> None:
    src = """
    composition boolean [level=semigroupoid]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    assert isinstance(m.morphism.algebra, Semigroupoid)


def test_algebra_keyword_rejects_semigroupoid() -> None:
    """Declaring ``material_impl [level=algebra]`` is a level mismatch."""
    src = """
    composition material_impl [level=algebra]
    object A : FinSet 2
    export A
    """
    with pytest.raises(CompileError, match="not at level"):
        loads(textwrap.dedent(src))


def test_composition_rule_keyword_accepts_any_rule() -> None:
    src = """
    composition material_impl [level=rule]
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    assert isinstance(m.morphism.algebra, CompositionRule)


def test_unknown_rule_name_errors() -> None:
    src = """
    composition not_a_real_algebra [level=algebra]
    object A : FinSet 2
    export A
    """
    with pytest.raises(CompileError, match="unknown algebra"):
        loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Inline user-defined bodies
# ---------------------------------------------------------------------------


def test_inline_algebra_body_builds_custom_rule() -> None:
    """Define a Goedel-like algebra inline."""
    src = """
    composition my_godel [level=algebra]
        tensor_op(a, b) = a * b
        join(t) = sum(t)
        unit = 1.0
        zero = 0.0
    object A : FinSet 3
    object B : FinSet 3
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(3, 3)})
    rule = m.morphism.algebra
    assert isinstance(rule, Algebra)
    assert rule.name == "my_godel"
    assert rule.unit == 1.0
    assert rule.zero == 0.0


def test_inline_body_missing_required_entry_errors() -> None:
    src = """
    composition broken [level=algebra]
        tensor_op(a, b) = a * b
        join(t) = sum(t)
    object A : FinSet 2
    export A
    """
    with pytest.raises(CompileError, match="missing required entries"):
        loads(textwrap.dedent(src))


def test_inline_semigroupoid_body() -> None:
    src = """
    composition my_semi [level=semigroupoid]
        tensor_op(a, b) = a * b
        join(t) = sum(t)
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    rule = m.morphism.algebra
    assert isinstance(rule, Semigroupoid)
    assert not isinstance(rule, Algebra)
    assert rule.name == "my_semi"


def test_inline_bilinear_form_body() -> None:
    src = """
    composition my_bf [level=bilinear_form]
        tensor_op(a, b) = (a + b) * 0.5
        join(t) = sum(t)
    object A : FinSet 2
    object B : FinSet 2
    morphism h : A -> B [role=observed] ~ from_data("H")
    export h
    """
    m = loads(textwrap.dedent(src), data={"H": torch.zeros(2, 2)})
    rule = m.morphism.algebra
    assert isinstance(rule, Magmoid)
    assert not isinstance(rule, Semigroupoid)


def test_inline_body_duplicate_entry_errors() -> None:
    src = """
    composition dup [level=algebra]
        tensor_op(a, b) = a * b
        tensor_op(a, b) = a + b
        join(t) = sum(t)
        unit = 1.0
        zero = 0.0
    object A : FinSet 2
    export A
    """
    with pytest.raises(CompileError, match="duplicate entry"):
        loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Compile-time refusal of Algebra-only operations
# ---------------------------------------------------------------------------


_NON_ALGEBRA_HEADER = """
composition material_impl [level=bilinear_form]
object A : FinSet 3
object B : FinSet 3
morphism f : A -> B [role=latent]
"""


@pytest.mark.parametrize(
    "expr,op_name",
    [
        ("identity(A)", "identity"),
        ("f.dagger", "dagger"),
        ("f.trace(A)", "trace"),
        ("cup(A)", "cup"),
        ("cap(A)", "cap"),
    ],
)
def test_algebra_only_ops_rejected_under_magmoid(
    expr: str,
    op_name: str,
) -> None:
    src = (
        _NON_ALGEBRA_HEADER
        + f"""
    define x = {expr}
    export x
    """
    )
    with pytest.raises(CompileError, match=op_name):
        loads(textwrap.dedent(src))


def test_algebra_ops_accepted_under_algebra() -> None:
    """The same operations compile cleanly when the rule is an
    Algebra."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    morphism f : A -> B [role=latent]
    define i = identity(A)
    define d = f.dagger
    define c = cup(A)
    define cc = cap(A)
    export i
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None


# ---------------------------------------------------------------------------
# Contraction declarations + invocations
# ---------------------------------------------------------------------------


def test_binary_contraction_matches_composition() -> None:
    """A binary contraction with the ``ij, jk -> ik`` wiring spec
    reproduces normal composition under the same rule."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 3
    object C : FinSet 3

    morphism f : A -> B [role=observed] ~ from_data("F")
    morphism g : B -> C [role=observed] ~ from_data("G")

    contraction binop (
        x : A -> B,
        y : B -> C,
    ) : A -> C [rule=product_fuzzy, wiring="ij, jk -> ik"]

    define composed = binop(f, g)
    export composed
    """
    torch.manual_seed(0)
    f = torch.rand(3, 3)
    g = torch.rand(3, 3)
    m = loads(textwrap.dedent(src), data={"F": f, "G": g})
    # Compare to normal composition.
    from quivers.core.algebras import PRODUCT_FUZZY

    expected = PRODUCT_FUZZY.compose(f, g, n_contract=1)
    assert torch.allclose(m.morphism.tensor, expected, atol=1e-6)


def test_ternary_contraction_via_dsl() -> None:
    """An operadic 3-input contraction folds two argument tensors
    and a kernel under a shared reduction; the surviving axes
    define the output."""
    src = """
    composition product_fuzzy [level=algebra]
    object S : FinSet 4
    object P : FinSet 3
    object Q : FinSet 5

    morphism arg1 : S -> P [role=observed] ~ from_data("A1")
    morphism arg2 : S -> Q [role=observed] ~ from_data("A2")
    morphism kernel : P -> Q [role=observed] ~ from_data("K")

    contraction triop (
        a : S -> P,
        b : S -> Q,
        k : P -> Q,
    ) : S -> Q [rule=product_fuzzy, wiring="sp, sq, pq -> sq"]

    define combined = triop(arg1, arg2, kernel)
    export combined
    """
    torch.manual_seed(1)
    a1 = torch.rand(4, 3)
    a2 = torch.rand(4, 5)
    k = torch.rand(3, 5)
    m = loads(textwrap.dedent(src), data={"A1": a1, "A2": a2, "K": k})
    assert tuple(m.morphism.tensor.shape) == (4, 5)


def test_contraction_with_input_shape_mismatch_errors() -> None:
    """Numel-level check on each input argument's domain/codomain
    pair against the contraction's declared input typing."""
    src = """
    composition product_fuzzy [level=algebra]
    object S : FinSet 4
    object P : FinSet 3
    object Q : FinSet 5
    object Wrong : FinSet 7

    morphism arg1 : S -> Wrong [role=observed] ~ from_data("A1")
    morphism arg2 : S -> Q     [role=observed] ~ from_data("A2")
    morphism kernel : P -> Q   [role=observed] ~ from_data("K")

    contraction triop (
        a : S -> P,
        b : S -> Q,
        k : P -> Q,
    ) : S -> Q [rule=product_fuzzy, wiring="sp, sq, pq -> sq"]

    define combined = triop(arg1, arg2, kernel)
    export combined
    """
    a1 = torch.rand(4, 7)
    a2 = torch.rand(4, 5)
    k = torch.rand(3, 5)
    with pytest.raises(CompileError, match="declares"):
        loads(textwrap.dedent(src), data={"A1": a1, "A2": a2, "K": k})


def test_contraction_with_magmoid_rule() -> None:
    """A contraction can use any registered rule, including a
    magmoid such as ``material_impl``."""
    src = """
    composition material_impl [level=bilinear_form]
    object A : FinSet 2
    object B : FinSet 2
    object C : FinSet 2

    morphism f : A -> B [role=observed] ~ from_data("F")
    morphism g : B -> C [role=observed] ~ from_data("G")

    contraction binop (
        x : A -> B,
        y : B -> C,
    ) : A -> C [rule=material_impl, wiring="ij, jk -> ik"]

    define composed = binop(f, g)
    export composed
    """
    torch.manual_seed(3)
    f = torch.rand(2, 2)
    g = torch.rand(2, 2)
    m = loads(textwrap.dedent(src), data={"F": f, "G": g})
    assert tuple(m.morphism.tensor.shape) == (2, 2)


def test_contraction_arity_mismatch_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    contraction binop (
        x : A -> B,
        y : A -> B,
    ) : A -> B [rule=product_fuzzy, wiring="ij, jk, kl -> il"]
    export A
    """
    with pytest.raises(CompileError, match="arity|3 inputs"):
        loads(textwrap.dedent(src))


def test_contraction_unknown_rule_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    contraction binop (
        x : A -> B,
        y : A -> B,
    ) : A -> B [rule=nonexistent, wiring="ij, jk -> ik"]
    export A
    """
    with pytest.raises(CompileError, match="unknown rule"):
        loads(textwrap.dedent(src))


def test_contraction_call_wrong_argument_count_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    morphism f : A -> B [role=latent]
    morphism g : A -> B [role=latent]

    contraction binop (
        x : A -> B,
        y : A -> B,
    ) : A -> B [rule=product_fuzzy, wiring="ij, ij -> ij"]

    define bad = binop(f)
    export bad
    """
    with pytest.raises(CompileError, match="expected 2 arguments"):
        loads(textwrap.dedent(src))


def test_contraction_invocation_argument_not_a_morphism_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    morphism f : A -> B [role=latent]

    contraction binop (
        x : A -> B,
        y : A -> B,
    ) : A -> B [rule=product_fuzzy, wiring="ij, ij -> ij"]

    define bad = binop(f, undefined_morph)
    export bad
    """
    with pytest.raises(CompileError, match="not a declared morphism"):
        loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Parametric program template invocation at let-binding
# ---------------------------------------------------------------------------


def test_parametric_template_call_from_let() -> None:
    """``define applied = p(f)`` builds a synthetic non-parametric
    program by substituting ``f`` for the formal ``k`` and
    compiles it to a runtime morphism."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object B : FinSet 3

    morphism f : A -> B [role=latent]

    program p (k : Mor[A, B]) : A -> B
        sample out <- k
        return out

    define applied = p(f)
    export applied
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None


def test_parametric_template_call_wrong_arity_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    morphism f : A -> B [role=latent]

    program p (k : Mor[A, B], j : Mor[A, B]) : A -> B
        sample out <- k
        return out

    define bad = p(f)
    export bad
    """
    with pytest.raises(CompileError, match="expects 2 arguments"):
        loads(textwrap.dedent(src))


def test_parametric_template_call_undefined_morphism_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    program p (k : Mor[A, B]) : A -> B
        sample out <- k
        return out

    define bad = p(undefined)
    export bad
    """
    with pytest.raises(CompileError, match="not declared"):
        loads(textwrap.dedent(src))


def test_morphism_call_undefined_callee_errors() -> None:
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 2
    object B : FinSet 2

    morphism f : A -> B [role=latent]

    define bad = nothing(f)
    export bad
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(textwrap.dedent(src))
