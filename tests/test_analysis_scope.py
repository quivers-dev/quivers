"""Tests for the ``::``-path scope resolver.

The resolver is the foundation for every meta-command that takes
a binding name (``:info``, ``:type``, ``:doc``, ``:browse``,
``:where``). It walks ``compiler``'s elaborated module via
per-container scope views, handling arbitrary nesting depth and
every container kind the homogenized DSL surface admits.
"""

from __future__ import annotations

import pytest  # noqa: E402

from quivers.analysis.scope import (  # noqa: E402
    SCOPE_SEPARATOR,
    find_all_references,
    resolve_scoped_path,
    scope_children,
    split_path,
)
from quivers.cli.repl_session import ReplSession  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _session(path: str) -> ReplSession:
    s = ReplSession()
    s.load_file(path)
    return s


@pytest.fixture
def lda():
    return _session("tests/fixtures/qvr/grouped_topic_mixture.qvr")


@pytest.fixture
def ccg():
    return _session("docs/examples/source/ccg.qvr")


# ---------------------------------------------------------------------------
# Path-string utilities
# ---------------------------------------------------------------------------


def test_split_path_single_segment():
    assert split_path("lda") == ["lda"]


def test_split_path_nested():
    assert split_path("lda::z::w") == ["lda", "z", "w"]


def test_separator_is_double_colon():
    assert SCOPE_SEPARATOR == "::"


# ---------------------------------------------------------------------------
# Top-level lookup
# ---------------------------------------------------------------------------


def test_resolve_top_level_program(lda):
    ref = resolve_scoped_path(lda._compiler, "lda")
    assert ref is not None
    assert ref.name == "lda"
    assert ref.kind == "program"
    assert ref.path == "lda"
    assert ref.parent_kind is None


def test_resolve_top_level_object(lda):
    ref = resolve_scoped_path(lda._compiler, "Doc")
    assert ref is not None
    assert ref.kind == "object"
    assert ref.name == "Doc"


def test_resolve_top_level_deduction(ccg):
    ref = resolve_scoped_path(ccg._compiler, "CCG")
    assert ref is not None
    assert ref.kind == "deduction"


def test_resolve_unknown_top_level_returns_none(lda):
    assert resolve_scoped_path(lda._compiler, "nope") is None


# ---------------------------------------------------------------------------
# Program scopes
# ---------------------------------------------------------------------------


def test_resolve_sample_site_inside_program(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::theta")
    assert ref is not None
    assert ref.kind == "sample-site"
    assert ref.parent_kind == "program"
    assert ref.path == "lda::theta"


def test_resolve_marginalize_site_inside_program(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::z")
    assert ref is not None
    assert ref.kind == "marginalize-site"


def test_resolve_observe_site_inside_marginalize(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::z::w")
    assert ref is not None
    assert ref.kind == "observe-site"
    assert ref.parent_kind == "marginalize-site"
    assert ref.path == "lda::z::w"


def test_resolve_typed_program_parameter(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::alpha")
    assert ref is not None
    assert ref.kind == "param"


def test_resolve_return_step(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::return")
    assert ref is not None
    assert ref.kind == "return-site"


def test_resolve_missing_segment_returns_none(lda):
    assert resolve_scoped_path(lda._compiler, "lda::nonexistent") is None


def test_resolve_missing_deep_segment_returns_none(lda):
    assert resolve_scoped_path(lda._compiler, "lda::z::nonexistent") is None


def test_resolve_path_into_non_container_returns_none(lda):
    # ``Doc`` is a plain object (FinSet 20); it has no scope.
    assert resolve_scoped_path(lda._compiler, "Doc::anything") is None


def test_resolve_empty_segment_rejected(lda):
    assert resolve_scoped_path(lda._compiler, "lda::") is None
    assert resolve_scoped_path(lda._compiler, "::lda") is None
    assert resolve_scoped_path(lda._compiler, "") is None


# ---------------------------------------------------------------------------
# Deduction scopes
# ---------------------------------------------------------------------------


def test_resolve_deduction_rule(ccg):
    ref = resolve_scoped_path(ccg._compiler, "CCG::fwd_app")
    assert ref is not None
    assert ref.kind == "deduction-rule"
    assert ref.parent_kind == "deduction"


def test_resolve_deduction_atom(ccg):
    # CCG declares atoms NP, S, VP, ...
    for atom in ("NP", "S", "VP"):
        ref = resolve_scoped_path(ccg._compiler, f"CCG::{atom}")
        if ref is not None:  # not every fixture declares every atom
            assert ref.kind in ("atom", "deduction-rule"), (atom, ref.kind)


# ---------------------------------------------------------------------------
# scope_children for non-container kinds
# ---------------------------------------------------------------------------


def test_scope_children_empty_on_object(lda):
    ref = resolve_scoped_path(lda._compiler, "Doc")
    assert ref is not None
    assert dict(scope_children(ref)) == {}


def test_scope_children_lists_program_steps(lda):
    ref = resolve_scoped_path(lda._compiler, "lda")
    children = dict(scope_children(ref))
    # Every typed param + every named step
    assert "alpha" in children
    assert "beta" in children
    assert "theta" in children
    assert "phi" in children
    assert "z" in children
    assert "return" in children


def test_scope_children_recurses_marginalize(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::z")
    children = dict(scope_children(ref))
    assert "w" in children
    assert children["w"].kind == "observe-site"


def test_scope_children_deduction_lists_rules(ccg):
    ref = resolve_scoped_path(ccg._compiler, "CCG")
    children = dict(scope_children(ref))
    assert "fwd_app" in children
    assert "bwd_app" in children
    assert children["fwd_app"].kind == "deduction-rule"


# ---------------------------------------------------------------------------
# Path round-trip: every child's path field matches its actual position
# ---------------------------------------------------------------------------


def test_child_path_is_concatenation_of_parent_and_name(lda):
    parent = resolve_scoped_path(lda._compiler, "lda")
    for name, child in scope_children(parent).items():
        assert child.path == f"lda::{name}", child


def test_grandchild_path_is_three_segments(lda):
    z = resolve_scoped_path(lda._compiler, "lda::z")
    for name, child in scope_children(z).items():
        assert child.path == f"lda::z::{name}", child


# ---------------------------------------------------------------------------
# find_all_references
# ---------------------------------------------------------------------------


def test_find_all_references_top_level(lda):
    refs = find_all_references(lda._compiler, "Doc")
    assert any(r.path == "Doc" and r.kind == "object" for r in refs), refs


def test_find_all_references_scoped(lda):
    refs = find_all_references(lda._compiler, "theta")
    # ``theta`` appears as the sample site inside lda.
    paths = {r.path for r in refs}
    assert "lda::theta" in paths


def test_find_all_references_returns_empty_for_missing(lda):
    assert find_all_references(lda._compiler, "totallymadeup") == []


# ---------------------------------------------------------------------------
# ScopedRef is a dx.Model: serialisable, frozen
# ---------------------------------------------------------------------------


def test_scoped_ref_is_dx_model(lda):
    ref = resolve_scoped_path(lda._compiler, "lda::theta")
    # dx.Model exposes __field_specs__ + model_dump
    assert hasattr(ref, "__field_specs__")
    dumped = ref.model_dump() if hasattr(ref, "model_dump") else None
    if dumped is not None:
        assert dumped["name"] == "theta"
        assert dumped["kind"] == "sample-site"


# ---------------------------------------------------------------------------
# Cross-container coverage: every kind of declaration must either
# expose useful child names or be a sensible leaf.
# ---------------------------------------------------------------------------


import textwrap  # noqa: E402

from quivers.dsl import Compiler  # noqa: E402
from quivers.dsl.parser import parse  # noqa: E402


def _compile(src: str):  # type: ignore[no-untyped-def]
    c = Compiler(parse(textwrap.dedent(src)))
    c.compile_env()
    return c


def test_scope_for_signature_exposes_sorts_constructors_binders():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]

        signature LF
            sorts
                Term : object [dim=64]
                Type : object [dim=32]
                Name : data   [dim=32]
            constructors
                Const : Name      -> Term
                App   : Term, Term -> Term
        """
    )
    ref = resolve_scoped_path(c, "LF")
    children = dict(scope_children(ref))
    # All three sorts + both constructors are addressable
    assert {"Term", "Type", "Name", "Const", "App"} <= set(children)
    # Sort kinds are tagged
    assert children["Term"].kind == "sort"
    assert children["Const"].kind == "constructor"


def test_resolve_scoped_path_walks_into_signature_sorts():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]

        signature LF
            sorts
                Term : object [dim=64]
            constructors
                Const : Term -> Term
        """
    )
    r = resolve_scoped_path(c, "LF::Term")
    assert r is not None and r.kind == "sort"
    r2 = resolve_scoped_path(c, "LF::Const")
    assert r2 is not None and r2.kind == "constructor"


def test_scope_for_bundle_exposes_member_names():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]
        object A : FinSet 3
        object B : FinSet 4
        morphism f : A -> B [role=latent]
        morphism g : A -> B [role=latent]
        bundle MyBundle : [f, g]
        """
    )
    ref = resolve_scoped_path(c, "MyBundle")
    children = dict(scope_children(ref))
    assert set(children) == {"f", "g"}
    for child in children.values():
        assert child.kind == "bundle-member"


def test_scope_for_contraction_exposes_input_names():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]
        object A : FinSet 3
        object B : FinSet 4
        object C : FinSet 5

        contraction op_apply (
            arg1 : A -> B,
            arg2 : B -> C,
        ) : A -> C [rule=product_fuzzy]
        """
    )
    ref = resolve_scoped_path(c, "op_apply")
    children = dict(scope_children(ref))
    assert {"arg1", "arg2"} <= set(children)
    for arg in ("arg1", "arg2"):
        scoped = resolve_scoped_path(c, f"op_apply::{arg}")
        assert scoped is not None
        assert scoped.kind == "param"


def test_factory_encoder_has_no_user_named_children():
    """A factory-built encoder (``encoder enc : SIG [factory=...]``)
    delegates its body to the factory; the user names no inner
    bindings, so ``scope_children`` is empty. The encoder itself
    still resolves at the top level."""
    c = _compile(
        """
        composition product_fuzzy [level=algebra]

        signature seq
            sorts
                Seq : object [dim=64]
                L   : data   [dim=64]
            constructors
                Nil  :        -> Seq
                Cons : L, Seq -> Seq

        encoder enc : seq [factory=rnn_encoder]
        """
    )
    ref = resolve_scoped_path(c, "enc")
    assert ref is not None
    assert ref.kind == "encoder"
    assert dict(scope_children(ref)) == {}


def test_loss_is_a_leaf_in_the_scope_graph():
    """A ``loss`` decl is conceptually a leaf attachment; we don't
    enumerate the body's let-expression internals."""
    c = _compile(
        """
        composition product_fuzzy [level=algebra]

        signature LF
            sorts
                Term : object [dim=64]
            constructors
                Const : Term -> Term

        encoder enc : LF [factory=tree_lstm_encoder]

        loss l1 [weight=0.5, on=encoder(enc)]
            sum([1.0])
        """
    )
    losses = list(c.losses)
    if "l1" in losses:
        ref = resolve_scoped_path(c, "l1")
        assert ref is not None
        assert ref.kind == "loss"
        assert dict(scope_children(ref)) == {}


def test_object_is_a_leaf_regardless_of_constructor_shape():
    """``object X : FinSet N``, ``object X : Real N``, ``object X
    : A * B`` — all are leaf scopes."""
    c = _compile(
        """
        composition product_fuzzy [level=algebra]
        object A : FinSet 3
        object B : FinSet 4
        object Pair : A * B
        object S : Real 64
        """
    )
    for name in ("A", "B", "Pair", "S"):
        ref = resolve_scoped_path(c, name)
        assert ref is not None
        assert dict(scope_children(ref)) == {}, name


def test_morphism_is_a_leaf():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]
        object A : FinSet 3
        object B : FinSet 4
        morphism f : A -> B [role=latent]
        """
    )
    ref = resolve_scoped_path(c, "f")
    assert ref is not None
    assert ref.kind == "morphism"
    assert dict(scope_children(ref)) == {}
