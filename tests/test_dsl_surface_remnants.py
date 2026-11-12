"""End-to-end tests for DSL surface constructs that lack positive
coverage elsewhere in the suite:

* ``lexicon from "file"``: a deduction whose lexicon is loaded from a
  TSV file (``word\\tcategory\\tlf`` rows) compiles and parses a toy
  string through the agenda engine.
* Plural lexicon entries: ``"a", "an" : Det = lf`` expands to one
  axiom row (and one learnable weight) per word, matching the
  separately written entries exactly.
* ``define NAME = EXPR where ...``: nested define bindings feed the
  outer right-hand side; the outer name lands in the compile
  environment with the composite's shape.
* ``#!`` doc comments: the parser attaches stripped doc lines to the
  ``docs`` tuple of the following declaration.
* ``.curry_right`` / ``.curry_left`` / ``.trace(A)`` postfix methods:
  positive compilations with forward-shape assertions.
* ``from_data("KEY")`` initializers: the bound tensor's values flow
through a forward pass as well as parameter registration.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import cast


import torch
import torch.nn as nn

from quivers.core.morphisms import CurriedMorphism, Morphism
from quivers.core.objects import FinSet
from quivers.dsl import loads
from quivers.dsl.ast_nodes import (
    DefineDecl,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
)
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.program import Program
from quivers.stochastic.deduction import DeductionSystem

_LEXICON_TSV = Path(__file__).resolve().parent / "data" / "lexicon_toy.tsv"


def _deduction(prog: Program, name: str) -> DeductionSystem:
    """Fetch a compiled deduction system with a concrete type."""
    systems = cast(dict[str, DeductionSystem], prog.deductions)
    return systems[name]


# ---------------------------------------------------------------------------
# lexicon from "file"
# ---------------------------------------------------------------------------


def test_lexicon_from_file_compiles_and_parses_toy_string() -> None:
    """A deduction whose lexicon comes from a TSV file compiles, and
    the resulting system parses a two-word string to the start
    symbol with a finite weight."""
    src = f"""
    object Term : FinSet 8

    deduction ToyFile : Term -> Term [semiring=LogProb, start=S]
        atoms S, Det, N
        rule combine : span(I, K, Det), span(K, J, N) |- span(I, J, S)
        lexicon from "{_LEXICON_TSV}"
    """
    mod = parse(textwrap.dedent(src))
    prog = Compiler(mod).compile()
    ded = _deduction(prog, "ToyFile")

    chart = ded(["the", "dog"])
    goals = list(chart.goal_items)
    assert len(goals) == 1, f"expected one goal item, got {goals}"
    item, weight = goals[0]
    assert item == ("span", 0, 2, ("atom", "S"))
    assert torch.isfinite(weight)

    # A string built from different rows of the same file parses too.
    assert len(list(ded(["a", "cat"]).goal_items)) == 1
    # The ungrammatical order N-then-Det derives no goal item.
    assert list(ded(["dog", "the"]).goal_items) == []


def test_file_lexicon_normalizes_constants_and_binders(tmp_path: Path) -> None:
    """TSV logical forms use the same term algebra as inline entries."""
    lexicon = tmp_path / "logical_forms.tsv"
    lexicon.write_text(
        "every-dog\tS\tClaim(forall_t, Lam(x, App(dog_p, Var(x))))\n",
        encoding="utf-8",
    )
    src = f"""
    object Term : FinSet 8

    deduction FileLF : Term -> Term [semiring=Boolean, start=Accepted]
        atoms S, Accepted, span, Claim, forall_t, App, Var, dog_p
        binders Lam
        rule accept : span(I, J, S, Claim(forall_t, P)) |- Accepted
        lexicon from "{lexicon}"
    """
    prog = Compiler(parse(textwrap.dedent(src))).compile()
    ded = _deduction(prog, "FileLF")

    axioms = list(ded.axiom_injector(["every-dog"]))
    lf = axioms[0][0][4]
    assert lf[1] == ("atom", "forall_t")
    assert lf[2][1][0] == "var"
    assert lf[2][2] == (
        "App",
        ("atom", "dog_p"),
        ("Var", lf[2][1]),
    )
    assert [item for item, _weight in ded(["every-dog"]).goal_items] == [
        ("atom", "Accepted")
    ]


# ---------------------------------------------------------------------------
# Plural lexicon entries
# ---------------------------------------------------------------------------


_PLURAL_VS_SEPARATE_SRC = """
object Term : FinSet 8

deduction Plural : Term -> Term [semiring=LogProb, start=S]
    atoms S, Det, N, a_lf, dog_lf
    rule combine : span(I, K, Det), span(K, J, N) |- span(I, J, S)
    lexicon
        "a", "an" : Det = a_lf #[learnable]
        "dog"     : N   = dog_lf #[learnable]

deduction Separate : Term -> Term [semiring=LogProb, start=S]
    atoms S, Det, N, a_lf, dog_lf
    rule combine : span(I, K, Det), span(K, J, N) |- span(I, J, S)
    lexicon
        "a"   : Det = a_lf #[learnable]
        "an"  : Det = a_lf #[learnable]
        "dog" : N   = dog_lf #[learnable]
"""


def test_plural_lexicon_entries_match_separate_entries() -> None:
    """``"a", "an" : Det = a_lf`` expands per word: the injected
    axioms, the learnable weight count, and the parse result all
    match the same lexicon written as separate entries."""
    mod = parse(textwrap.dedent(_PLURAL_VS_SEPARATE_SRC))
    prog = Compiler(mod).compile()
    plural = _deduction(prog, "Plural")
    separate = _deduction(prog, "Separate")

    tokens = ["a", "an", "dog"]
    plural_axioms = plural.axiom_injector(tokens)
    separate_axioms = separate.axiom_injector(tokens)
    assert len(plural_axioms) == 3
    assert len(plural_axioms) == len(separate_axioms)
    assert [item for item, _ in plural_axioms] == [item for item, _ in separate_axioms]

    # Each expanded word carries its own learnable weight.
    plural_module = getattr(plural, "_axiom_module")
    separate_module = getattr(separate, "_axiom_module")
    assert isinstance(plural_module, nn.Module)
    assert isinstance(separate_module, nn.Module)
    plural_params = list(plural_module.parameters())
    separate_params = list(separate_module.parameters())
    assert len(plural_params) == 3
    assert len(plural_params) == len(separate_params)

    # Both systems parse a string using the second plural word.
    assert len(list(plural(["an", "dog"]).goal_items)) == 1
    assert len(list(separate(["an", "dog"]).goal_items)) == 1


# ---------------------------------------------------------------------------
# define ... where
# ---------------------------------------------------------------------------


_DEFINE_WHERE_SRC = """
composition product_fuzzy [level=algebra]
object A : FinSet 2
object B : FinSet 3
object C : FinSet 4
morphism g : A -> B [role=latent]
morphism h : B -> C [role=latent]
define f = gg >> hh where
    define gg = g
    define hh = h
export f
"""


def test_define_where_compiles_and_wires_bindings() -> None:
    """The where-bound names feed the outer right-hand side: the
    exported composite has the domain of ``g`` and the codomain of
    ``h``."""
    m = loads(textwrap.dedent(_DEFINE_WHERE_SRC))
    morph = m.morphism
    assert isinstance(morph, Morphism)
    assert isinstance(morph.domain, FinSet)
    assert isinstance(morph.codomain, FinSet)
    assert morph.domain.cardinality == 2
    assert morph.codomain.cardinality == 4
    assert m.forward().shape == (2, 4)


def test_define_where_outer_binding_lands_in_env() -> None:
    """The outer define name is bound in the compile environment and
    carries the composite morphism; the where-bound names are scoped
    to the binding and do not leak into the module namespace."""
    mod = parse(textwrap.dedent(_DEFINE_WHERE_SRC))
    env = Compiler(mod).compile_env()
    assert "f" in env
    bound = env["f"]
    assert isinstance(bound, Morphism)
    assert bound.tensor.shape == (2, 4)
    assert "gg" not in env
    assert "hh" not in env


# ---------------------------------------------------------------------------
# #! doc comments
# ---------------------------------------------------------------------------


def test_doc_comments_attach_to_declarations() -> None:
    """``#!`` lines populate the ``docs`` tuple on the following
    declaration, stripped of the marker, for object, morphism,
    define, and program declarations alike; undocumented
    declarations keep an empty tuple."""
    src = """
    #! The response space.
    object A : FinSet 3

    object B : FinSet 2

    #! Prior over responses.
    #! Second doc line.
    morphism g : A -> A [role=latent]

    #! Composite binding.
    define h = g >> g

    #! A toy program.
    program p : A -> A
        sample x <- g
        return x
    """
    mod = parse(textwrap.dedent(src))
    obj_a, obj_b, morph_g, define_h, program_p = mod.statements

    assert isinstance(obj_a, ObjectDecl)
    assert obj_a.docs == ("The response space.",)

    assert isinstance(obj_b, ObjectDecl)
    assert obj_b.docs == ()

    assert isinstance(morph_g, MorphismDecl)
    assert morph_g.docs == ("Prior over responses.", "Second doc line.")

    assert isinstance(define_h, DefineDecl)
    assert define_h.docs == ("Composite binding.",)

    assert isinstance(program_p, ProgramDecl)
    assert program_p.docs == ("A toy program.",)


# ---------------------------------------------------------------------------
# .curry_right / .curry_left / .trace
# ---------------------------------------------------------------------------


_CURRY_SRC = """
composition product_fuzzy [level=algebra]
object A : FinSet 2
object B : FinSet 3
object C : FinSet 4
morphism f : A * B -> C [role=latent]
define fc = f.{method}
export fc
"""


def test_curry_right_compiles_with_expected_shape() -> None:
    """``f.curry_right`` on ``f : A * B -> C`` keeps the first
    factor as the domain and reads the same tensor as ``A → B × C``."""
    m = loads(textwrap.dedent(_CURRY_SRC.format(method="curry_right")))
    morph = m.morphism
    assert isinstance(morph, CurriedMorphism)
    assert morph.direction == "right"
    assert isinstance(morph.domain, FinSet)
    assert morph.domain.cardinality == 2
    assert m.forward().shape == (2, 3, 4)


def test_curry_left_compiles_with_expected_shape() -> None:
    """``f.curry_left`` on ``f : A * B -> C`` keeps the second
    factor as the domain, so the tensor is re-indexed to ``B × (A × C)``."""
    m = loads(textwrap.dedent(_CURRY_SRC.format(method="curry_left")))
    morph = m.morphism
    assert isinstance(morph, CurriedMorphism)
    assert morph.direction == "left"
    assert isinstance(morph.domain, FinSet)
    assert morph.domain.cardinality == 3
    assert m.forward().shape == (3, 2, 4)


def test_trace_compiles_with_expected_shape() -> None:
    """``f.trace(A)`` on ``f : A * X -> A * Y`` contracts the shared
    factor, leaving an ``X -> Y`` morphism whose forward tensor has
    shape ``(|X|, |Y|)`` and participates in autograd."""
    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 3
    object X : FinSet 2
    object Y : FinSet 5
    morphism f : A * X -> A * Y [role=latent]
    define g = f.trace(A)
    export g
    """
    m = loads(textwrap.dedent(src))
    morph = m.morphism
    assert isinstance(morph, Morphism)
    assert isinstance(morph.domain, FinSet)
    assert isinstance(morph.codomain, FinSet)
    assert morph.domain.cardinality == 2
    assert morph.codomain.cardinality == 5
    out = m.forward()
    assert out.shape == (2, 5)
    assert out.requires_grad


# ---------------------------------------------------------------------------
# from_data through a forward pass
# ---------------------------------------------------------------------------


_FROM_DATA_SRC = """
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism h : A -> B [role=observed] ~ from_data("H")
define chain = h >> identity(B)
export chain
"""


def test_from_data_tensor_flows_through_forward() -> None:
    """The tensor bound via ``loads(..., data={...})`` reaches the
    forward pass: composing the data-derived morphism with the
    identity materializes exactly the bound values, and rebinding a
    different tensor changes the output accordingly."""
    h1 = torch.tensor(
        [[0.1, 0.9, 0.2, 0.4], [0.7, 0.3, 0.8, 0.5], [0.6, 0.2, 0.1, 0.9]]
    )
    m1 = loads(textwrap.dedent(_FROM_DATA_SRC), data={"H": h1})
    out1 = m1.forward()
    assert out1.shape == (3, 4)
    assert torch.allclose(out1, h1)

    h2 = torch.tensor(
        [[0.5, 0.4, 0.6, 0.1], [0.2, 0.8, 0.3, 0.7], [0.9, 0.1, 0.5, 0.2]]
    )
    m2 = loads(textwrap.dedent(_FROM_DATA_SRC), data={"H": h2})
    out2 = m2.forward()
    assert torch.allclose(out2, h2)
    assert not torch.allclose(out1, out2)
