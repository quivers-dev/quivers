"""End-to-end tests for ``schema`` / ``bundle`` declarations and the
``parser(...)`` / ``chart_fold(...)`` expressions.

The positive path drives the shipped gallery example
``docs/examples/source/schema_chart_parser.qvr`` from source text to a
scoring call: ``schema`` declarations compile to pattern schemas,
``bundle`` splices its members into the ``parser(...)`` rule list, and
the exported morphism is a ``ChartParser`` whose forward pass scores
token sequences differentiably. ``chart_fold(...)`` is exercised in its
primitive form, compiling to an ``InsideAlgorithm`` over user-declared
kernels. The negative path checks that a bundle member naming nothing
raises the compiler's unknown-rule diagnostic at the ``parser(...)``
use site.

Run with::

    pytest tests/test_dsl_schema_bundle_parser.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch


from quivers.dsl import CompileError, loads
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.stochastic.categories import AtomicCategory
from quivers.stochastic.inside import InsideAlgorithm
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.schema import PatternBinarySchema, PatternUnarySchema

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "schema_chart_parser.qvr"
)


def _example_source() -> str:
    return _EXAMPLE.read_text()


def _load_example_parser() -> ChartParser:
    prog = loads(_example_source())
    morphism = prog.morphism
    assert isinstance(morphism, ChartParser)
    return morphism


# ---------------------------------------------------------------------------
# schema / bundle declarations
# ---------------------------------------------------------------------------


def test_schema_declarations_compile_to_pattern_schemas() -> None:
    env = Compiler(parse(_example_source())).compile_env()
    assert isinstance(env["fwd_app"], PatternBinarySchema)
    assert isinstance(env["bwd_app"], PatternBinarySchema)
    assert isinstance(env["perm"], PatternUnarySchema)


def test_bundle_binds_member_names_in_order() -> None:
    env = Compiler(parse(_example_source())).compile_env()
    assert env["lp_rules"] == ("fwd_app", "bwd_app", "perm")


# ---------------------------------------------------------------------------
# parser(...) over a bundle
# ---------------------------------------------------------------------------


def test_example_compiles_to_chart_parser() -> None:
    parser = _load_example_parser()
    # 3 atoms plus every depth-1 slash category: 3 + 3 * 3 * 2 = 21.
    assert parser.rule_system.n_categories == 21
    # fwd_app / bwd_app instantiate over the 9 atomic (X, Y) pairs
    # each; perm contributes one unary firing per pair.
    assert parser.n_rules == 18
    assert parser.n_unary_rules == 9
    category_system = parser.category_system
    assert category_system is not None
    assert AtomicCategory(name="S") in category_system


def test_bundle_reference_splices_like_verbatim_rule_list() -> None:
    spliced = _load_example_parser()
    verbatim_src = _example_source().replace(
        "rules=[lp_rules]",
        "rules=[fwd_app, bwd_app, perm]",
    )
    prog = loads(verbatim_src)
    verbatim = prog.morphism
    assert isinstance(verbatim, ChartParser)
    assert verbatim.rule_system.n_categories == spliced.rule_system.n_categories
    assert verbatim.n_rules == spliced.n_rules
    assert verbatim.n_unary_rules == spliced.n_unary_rules


def test_example_forward_scores_and_backpropagates() -> None:
    parser = _load_example_parser()
    # token indices follow Token's declaration order:
    # the=0, dog=1, cat=2, sleeps=3.
    single = parser(torch.tensor([0, 1, 3]))
    assert single.shape == ()
    assert torch.isfinite(single)
    batch = parser(torch.tensor([[0, 1, 3], [0, 2, 3]]))
    assert batch.shape == (2,)
    assert torch.isfinite(batch).all()
    batch.sum().backward()
    grads = [p.grad for p in parser.parameters() if p.grad is not None]
    assert grads
    assert any(bool(g.abs().sum() > 0) for g in grads)


# ---------------------------------------------------------------------------
# bundle diagnostics
# ---------------------------------------------------------------------------


def test_bundle_with_unknown_member_raises_at_parser_use_site() -> None:
    src = textwrap.dedent(
        """
        object Atoms : {NP, S}
        object Cat : FreeResiduated(Atoms)
        object Token : FinSet 2

        schema fwd_app [X, Y : Cat] : (X / Y) * Y -> X

        bundle broken : [fwd_app, missing_rule]

        define p = parser(rules=[broken], terminal=Token, start=S)

        export p
        """
    )
    with pytest.raises(CompileError, match="unknown rule 'missing_rule'"):
        loads(src)


# ---------------------------------------------------------------------------
# chart_fold(...) primitive form
# ---------------------------------------------------------------------------


def test_chart_fold_direct_form_compiles_and_scores() -> None:
    src = textwrap.dedent(
        """
        object NT : FinSet 3
        object Tok : FinSet 5

        morphism grow : NT -> NT * NT
        morphism emit : NT -> Tok

        define inside = chart_fold(lex=emit, binary=grow, start=0)

        export inside
        """
    )
    prog = loads(src)
    inside = prog.morphism
    assert isinstance(inside, InsideAlgorithm)
    assert inside.n_nonterminals == 3
    assert inside.n_terminals == 5
    assert inside.start == 0
    scores = inside(torch.tensor([[0, 1, 2, 4]]))
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
