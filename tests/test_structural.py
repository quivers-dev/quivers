"""Tests for the structural-compression substrate: signatures,
encoders, decoders, losses, and the canonical shape sugar.
"""

from __future__ import annotations

import copy
import pickle
import textwrap

import didactic.api as dx
import pytest
import torch

from quivers.dsl import CompileError, loads
from quivers.structural import (
    Term,
    bound_var,
    make_term,
)
from quivers.structural.shapes import (
    ar_decoder,
    bow_encoder,
    gnn_encoder,
    graph_signature,
    rnn_encoder,
    seq_signature,
    transformer_encoder,
    tree_decoder,
    tree_lstm_encoder,
    tree_signature,
)
from quivers.structural.shapes.seq import list_to_term, term_to_list


# ---------------------------------------------------------------------------
# DSL surface
# ---------------------------------------------------------------------------


def test_signature_block_parses_and_compiles():
    src = """
    signature LF
        sorts
            Term : object [dim=32]
            Type : object [dim=16]
            Name : data   [dim=16]
        constructors
            Const : Name -> Term
            App   : Term, Term -> Term
            Base  : Name -> Type
        binders
            Lam : binds (ty : Type) in (body : Term) -> Term
    """
    prog = loads(textwrap.dedent(src))
    assert "LF" in prog.signatures
    sig = prog.signatures["LF"]
    assert set(sig.sorts) == {"Term", "Type", "Name"}
    assert sig.sorts["Term"].kind == "object"
    assert sig.sorts["Name"].kind == "data"
    assert "App" in sig.constructors
    assert sig.constructors["App"].domain == ("Term", "Term")
    assert "Lam" in sig.binders
    assert sig.binders["Lam"].binds[0].sort == "Type"


def test_encoder_default_scaffold_compiles():
    src = """
    signature S
        sorts
            Term : object [dim=16]
        constructors
            A : -> Term
            B : Term -> Term
            C : Term, Term -> Term

    encoder MyComp : S
        dim Term = 16
    """
    prog = loads(textwrap.dedent(src))
    C = prog.encoders["MyComp"]
    v = C(make_term("C", make_term("A"), make_term("B", make_term("A"))))
    assert v.shape == (16,)
    assert v.requires_grad


def test_decoder_default_scaffold_samples_and_scores():
    src = """
    signature S
        sorts
            Term : object [dim=16]
        constructors
            A : -> Term
            B : Term -> Term

    encoder C : S
        dim Term = 16

    decoder D : S [depth=4]
        body |-> recursive
    """
    prog = loads(textwrap.dedent(src))
    C, D = prog.encoders["C"], prog.decoders["D"]
    torch.manual_seed(0)
    sample = D(torch.randn(16))
    assert isinstance(sample, Term)
    t = make_term("B", make_term("A"))
    v = C(t)
    lp = D.log_prob(t, v)
    assert torch.is_tensor(lp)
    lp.backward()


def test_loss_decl_registers_and_evaluates():
    src = """
    signature S
        sorts
            Term : object [dim=8]
        constructors
            A : -> Term
            B : Term -> Term

    encoder C : S
        dim Term = 8

    decoder D : S [depth=3]
        body |-> recursive

    loss zero [weight=2.0, on=encoder(C)]
        1.0 + 2.0
    """
    prog = loads(textwrap.dedent(src))
    assert prog.losses is not None
    assert len(prog.losses.entries) == 1
    e = prog.losses.entries[0]
    assert e.name == "zero"
    assert e.attachment_kind == "encoder"
    assert e.target == "C"
    total = prog.losses.evaluate({})
    assert float(total) == 6.0  # (1+2) * 2


def test_rule_attached_loss_fires_on_each_rule_application():
    src = """
    signature Items
        sorts
            Item : object [dim=4]
        constructors
            NP : -> Item
            S : -> Item

    encoder C : Items
        dim Item = 4

    deduction D : Item -> Item [semiring=LogProb, signature=Items, encoder=C]
        atoms NP, S
        rule app : NP, NP |- S

    loss per_app [weight=1.0, on=rule(app, D)]
        0.5
    """
    prog = loads(textwrap.dedent(src))
    D = prog.deductions["D"]
    chart = D([(("atom", "NP"), torch.tensor(0.0))])
    # The rule `app : NP, NP |- S` matches twice on the single-NP
    # axiom: the agenda's semi-naive loop fires the rule once when
    # the popped item is premise 0 and once when it's premise 1,
    # each with the same single sibling; each firing accumulates
    # 0.5.
    assert chart.attached_loss is not None
    assert float(chart.attached_loss) == 1.0


def test_chart_attached_loss_fires_after_deduction_completes():
    src = """
    signature Items
        sorts
            Item : object [dim=4]
        constructors
            NP : -> Item
            S : -> Item

    encoder C : Items
        dim Item = 4

    deduction D : Item -> Item [semiring=LogProb, signature=Items, encoder=C]
        atoms NP, S
        rule app : NP, NP |- S

    loss completed [weight=1.0, on=chart(D)]
        7.0
    """
    prog = loads(textwrap.dedent(src))
    D = prog.deductions["D"]
    chart = D([(("atom", "NP"), torch.tensor(0.0))])
    assert chart.attached_loss is not None
    assert float(chart.attached_loss) == 7.0


def test_multiple_losses_attached_at_different_sites():
    src = """
    signature S
        sorts
            Term : object [dim=8]
        constructors
            A : -> Term
            B : Term -> Term

    encoder C : S
        dim Term = 8

    decoder D : S [depth=3]
        body |-> recursive

    loss a [weight=1.0, on=encoder(C)]
        1.0

    loss b [weight=3.0, on=decoder(D)]
        2.0

    loss c [global]
        5.0
    """
    prog = loads(textwrap.dedent(src))
    assert prog.losses is not None
    kinds = sorted(e.attachment_kind for e in prog.losses.entries)
    assert kinds == sorted(["encoder", "decoder", "global"])
    # 1*1 + 3*2 + 5 = 1 + 6 + 5 = 12
    assert float(prog.losses.evaluate({})) == 12.0


# ---------------------------------------------------------------------------
# Binder discipline (de Bruijn + type tracking)
# ---------------------------------------------------------------------------


def test_undeclared_sort_in_constructor_domain_raises():
    """Strict rule: constructors may only mention sorts already
    declared in the signature's `sorts` block."""
    src = """
    signature S
        sorts
            Term : object [dim=16]
        constructors
            Const : Name -> Term
    """
    with pytest.raises(CompileError, match="undeclared sort 'Name'"):
        loads(textwrap.dedent(src))


def test_undeclared_binder_var_sort_raises():
    """Strict rule: a binder's introduced variable sort must be
    declared in the signature's `sorts` block."""
    src = """
    signature S
        sorts
            Term : object [dim=16]
        binders
            Lam : binds (x : Unknown) in (body : Term) -> Term
    """
    with pytest.raises(CompileError, match="undeclared sort"):
        loads(textwrap.dedent(src))


def test_data_sort_vocab_declared_in_signature_block():
    """The signature surface declares a closed vocabulary on a
    data sort; the runtime ``Sort.vocab`` carries the decoded
    Python values in declaration order, and the decoder samples
    from exactly those tokens."""
    src = """
    signature LM
        sorts
            Tok    : data   [dim=16, vocab=["a", "b", "c", "d"]]
            Phrase : object [dim=16]
        constructors
            Word : Tok -> Phrase

    encoder C : LM
        dim Tok = 16
        dim Phrase = 16

    decoder D : LM [depth=3]
        body |-> recursive
    """
    prog = loads(textwrap.dedent(src))
    sig = prog.signatures["LM"]
    # The vocab survives the parser → compiler → runtime pipeline.
    assert sig.sorts["Tok"].vocab_values == ("a", "b", "c", "d")
    # The decoder's data_vocab is non-empty, so sampling a Word
    # term from a random vector yields a Word(<a known token>).
    D = prog.decoders["D"]
    torch.manual_seed(0)
    out = D(torch.randn(16))
    # Decoder's `sample` produces a Term whose `Word` child is a
    # token drawn from the declared vocabulary.
    assert isinstance(out, Term)
    assert out.op == "Word"
    assert out.args[0] in ("a", "b", "c", "d")
    # log_prob over an observed term with a known vocab token
    # returns a finite, autograd-flowing tensor.
    obs = make_term("Word", "c")
    v = torch.randn(16)
    lp = D.log_prob(obs, v)
    assert torch.is_tensor(lp)
    assert torch.isfinite(lp)


def test_data_sort_vocab_supports_integer_and_float_literals():
    """A vocabulary may mix string, integer, and float literals,
    each decoded into its canonical Python value."""
    src = """
    signature S
        sorts
            Item : object [dim=8]
            Tag  : data   [dim=8, vocab=["ok", 0, 1, 3.14]]
        constructors
            Mark : Tag -> Item
    """
    prog = loads(textwrap.dedent(src))
    sig = prog.signatures["S"]
    assert sig.sorts["Tag"].vocab_values == ("ok", 0, 1, 3.14)


def test_vocab_clause_only_valid_on_data_sorts():
    src = """
    signature S
        sorts
            Term : object [dim=8, vocab=["x"]]
    """
    with pytest.raises(CompileError, match="vocab clause is only valid"):
        loads(textwrap.dedent(src))


def test_duplicate_vocab_entry_rejected():
    src = """
    signature S
        sorts
            Tok : data [dim=8, vocab=["a", "b", "a"]]
    """
    with pytest.raises(CompileError, match="duplicate entry"):
        loads(textwrap.dedent(src))


def test_reserved_op_name_rejected():
    """Strict rule: `BoundVar` and `Data` are framework-reserved."""
    src = """
    signature S
        sorts
            Term : object [dim=8]
        constructors
            BoundVar : -> Term
    """
    with pytest.raises(CompileError, match="reserved"):
        loads(textwrap.dedent(src))


def test_recurrent_mode_binds_state_to_recursive_child():
    src = """
    signature Seq
        sorts
            Seq : object [dim=8]
            A   : data   [dim=8]
        constructors
            Nil  :        -> Seq
            Cons : A, Seq -> Seq

    encoder C : Seq
        dim Seq = 8
        op Nil                              |-> 0.0
        op Cons(head, tail) recurrent state |-> head + state
    """
    prog = loads(textwrap.dedent(src))
    C = prog.encoders["C"]
    rule = C.op_fns["Cons"]
    assert rule.mode == "recurrent"
    assert rule.state_var == "state"
    # And it actually compresses a real sequence.
    t = make_term("Cons", "a", make_term("Cons", "b", make_term("Nil")))
    v = C(t)
    assert v.shape == (8,)


def test_attention_mode_threads_prefix_list():
    src = """
    signature Seq
        sorts
            Seq : object [dim=8]
            A   : data   [dim=8]
        constructors
            Nil  :        -> Seq
            Cons : A, Seq -> Seq

    encoder C : Seq
        dim Seq = 8
        op Nil                               |-> 0.0
        op Cons(head, tail) attention prefix |-> head + tail
    """
    prog = loads(textwrap.dedent(src))
    C = prog.encoders["C"]
    rule = C.op_fns["Cons"]
    assert rule.mode == "attention"
    assert rule.prefix_var == "prefix"
    # Compress a 3-element sequence; the framework walks the chain
    # and threads the prefix.
    t = make_term(
        "Cons",
        "a",
        make_term("Cons", "b", make_term("Cons", "c", make_term("Nil"))),
    )
    v = C(t)
    assert v.shape == (8,)


def test_per_pair_var_init_overrides():
    src = """
    signature S
        sorts
            Term : object [dim=16]
            Type : object [dim=8]
            Name : data   [dim=8]
        constructors
            Base : Name -> Type
        binders
            Lam : binds (x : Term : ty : Type) in (body : Term) -> Term
            All : binds (a : Type) in (body : Term) -> Term

    encoder C : S
        dim Term = 16
        dim Type = 8
        var_init Term from Type as ty |-> ty
        var_init Type                  |-> 0.0
    """
    prog = loads(textwrap.dedent(src))
    C = prog.encoders["C"]
    # Both pairs are populated.
    assert ("Term", "Type") in C.var_init_fns
    assert "Type" in C.var_init_fns


def test_binder_threads_de_bruijn_context():
    src = """
    signature STLC
        sorts
            Term : object [dim=16]
            Type : object [dim=8]
            Name : data   [dim=8]
        constructors
            Const : Name -> Term
            App   : Term, Term -> Term
            Base  : Name -> Type
        binders
            # `Lam` binds a Term-sorted variable annotated by a Type.
            Lam : binds (x : Term : ty : Type) in (body : Term) -> Term

    encoder C : STLC
        dim Term = 16
        dim Type = 8
    """
    prog = loads(textwrap.dedent(src))
    C = prog.encoders["C"]
    sig = prog.signatures["STLC"]
    # Verify the binder's annot_sort plumbing.
    assert sig.binders["Lam"].binds[0].annot_sort == "Type"
    # \x:A. App(x, x): App takes two BoundVar(0)
    term = make_term(
        "Lam",
        make_term("Base", "A"),
        make_term("App", bound_var(0), bound_var(0)),
    )
    v = C(term)
    assert v.shape == (16,)
    assert v.requires_grad


# ---------------------------------------------------------------------------
# Sequence shapes
# ---------------------------------------------------------------------------


def test_seq_rnn_compresses_a_list():
    sig = seq_signature(dim=12)
    C = rnn_encoder(sig, dim=12)
    t = list_to_term(["a", "b", "c"])
    v = C(t)
    assert v.shape == (12,)
    # round-trip the list helper
    assert term_to_list(t) == ["a", "b", "c"]


def test_seq_transformer_and_bow_share_dim():
    sig = seq_signature(dim=8)
    for factory in (transformer_encoder, bow_encoder):
        C = factory(sig, dim=8)
        v = C(list_to_term(["x", "y"]))
        assert v.shape == (8,)


def test_seq_ar_decoder_samples_and_scores():
    sig = seq_signature(dim=8)
    C = rnn_encoder(sig, dim=8)
    D = ar_decoder(sig, dim=8, vocab=["a", "b"])
    torch.manual_seed(0)
    sample = D(torch.randn(8))
    assert isinstance(sample, Term)
    obs = list_to_term(["a", "b"])
    lp = D.log_prob(obs, C(obs))
    assert torch.is_tensor(lp)


# ---------------------------------------------------------------------------
# Tree shapes
# ---------------------------------------------------------------------------


def test_tree_lstm_compresses_a_tree():
    sig = tree_signature(dim=8)
    C = tree_lstm_encoder(sig, dim=8)
    t = make_term(
        "Node",
        "*",
        make_term("Leaf", "x"),
        make_term("Node", "+", make_term("Leaf", "y"), make_term("Leaf", "z")),
    )
    v = C(t)
    assert v.shape == (8,)
    assert v.requires_grad


def test_tree_decoder_terminates_at_depth():
    sig = tree_signature(dim=8)
    D = tree_decoder(sig, dim=8, leaf_vocab=["x"], label_vocab=["*"], depth=3)
    torch.manual_seed(0)
    sample = D(torch.randn(8))
    assert isinstance(sample, Term)


# ---------------------------------------------------------------------------
# Graph shapes
# ---------------------------------------------------------------------------


def test_gnn_compresses_a_graph():
    sig = graph_signature(
        "Mol",
        {"Atom": 12},
        {"bonded": ("Atom", "Atom", False)},
    )
    C = gnn_encoder(sig, iterations=2, dim=12)
    vertices = [("Atom", "C"), ("Atom", "H"), ("Atom", "H")]
    edges = [("bonded", 0, 1), ("bonded", 0, 2)]
    v = C.forward_graph(vertices, edges)
    assert v.shape == (12,)
    assert v.requires_grad


# ---------------------------------------------------------------------------
# Deduction integration
# ---------------------------------------------------------------------------


def test_deduction_with_attached_encoder_exposes_embeddings():
    src = """
    signature Items
        sorts
            Item : object [dim=8]
        constructors
            S   : -> Item
            NP  : -> Item
            App : Item, Item -> Item

    encoder C : Items
        dim Item = 8

    deduction Parse : Item -> Item [semiring=LogProb, signature=Items, encoder=C]
        atoms S, NP
        rule combine : NP, NP |- S
    """
    prog = loads(textwrap.dedent(src))
    D = prog.deductions["Parse"]
    assert getattr(D, "_item_signature", None) is not None
    assert getattr(D, "_item_encoder", None) is not None
    chart = D(
        [
            (("atom", "NP"), torch.tensor(0.0)),
            (("atom", "S"), torch.tensor(0.0)),
        ]
    )
    # The chart-as-presheaf works as before.
    assert chart.chart.get(("atom", "S")) is not None


# ---------------------------------------------------------------------------
# Term equality and hashing
# ---------------------------------------------------------------------------


def _lam(annotation: str, body: Term) -> Term:
    """``λx : annotation. body`` over the STLC-style signature."""
    return make_term("Lam", make_term("Base", annotation), body)


def test_terms_with_different_data_leaves_are_distinct():
    f = make_term("Const", "f")
    plus = make_term("Const", "plus")
    assert f != plus
    assert not f == plus
    assert hash(f) != hash(plus)
    assert len({f, plus}) == 2


def test_equal_terms_built_separately_are_equal_and_hash_alike():
    def build() -> Term:
        return _lam("base", make_term("App", make_term("Const", "plus"), bound_var(0)))

    left, right = build(), build()
    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


@pytest.mark.parametrize(
    "other",
    [
        # differing leaf two levels down
        _lam("base", make_term("App", make_term("Const", "f"), bound_var(0))),
        # differing nested subterm shape
        _lam("base", make_term("App", bound_var(0), make_term("Const", "plus"))),
        # differing binder annotation
        _lam("other", make_term("App", make_term("Const", "plus"), bound_var(0))),
        # differing de-Bruijn index
        _lam("base", make_term("App", make_term("Const", "plus"), bound_var(1))),
        # differing binder operator over the same arguments
        make_term(
            "All",
            make_term("Base", "base"),
            make_term("App", make_term("Const", "plus"), bound_var(0)),
        ),
        # differing arity
        _lam("base", make_term("App", make_term("Const", "plus"))),
    ],
    ids=["leaf", "nested", "annotation", "index", "binder", "arity"],
)
def test_terms_differing_below_the_root_are_distinct(other: Term):
    base = _lam("base", make_term("App", make_term("Const", "plus"), bound_var(0)))
    assert base != other
    assert other != base
    assert hash(base) != hash(other)
    assert len({base, other}) == 2


def test_data_leaves_compare_by_type_and_value():
    as_int = make_term("C", 1)
    as_float = make_term("C", 1.0)
    as_bool = make_term("C", True)
    as_str = make_term("C", "1")
    as_bytes = make_term("C", b"1")
    variants = [as_int, as_float, as_bool, as_str, as_bytes]
    for i, left in enumerate(variants):
        for j, right in enumerate(variants):
            assert (left == right) is (i == j)
    assert len(set(variants)) == len(variants)


def test_bound_variable_is_distinct_from_an_int_data_leaf():
    assert bound_var(0) != make_term("Const", 0)
    assert bound_var(0) == make_term("BoundVar", 0)


def test_hash_is_consistent_with_equality_across_a_term_population():
    leaves = ("f", "x", "plus")
    population = [
        _lam(ann, make_term("App", make_term("Const", leaf), bound_var(idx)))
        for ann in ("base", "other")
        for leaf in leaves
        for idx in (0, 1)
    ]
    rebuilt = [
        _lam(ann, make_term("App", make_term("Const", leaf), bound_var(idx)))
        for ann in ("base", "other")
        for leaf in leaves
        for idx in (0, 1)
    ]
    for left in population + rebuilt:
        for right in population + rebuilt:
            if left == right:
                assert hash(left) == hash(right)
    assert len(set(population)) == len(population)
    assert set(population) == set(rebuilt)


def test_terms_work_as_dict_keys():
    f = make_term("Const", "f")
    plus = make_term("Const", "plus")
    table = {f: "f", plus: "plus"}
    assert len(table) == 2
    assert table[make_term("Const", "f")] == "f"
    assert table[make_term("Const", "plus")] == "plus"
    assert make_term("Const", "x") not in table


def test_terms_never_equal_non_terms():
    term = make_term("Const", "f")
    assert term != ("Const", "f")
    assert term != term.to_tuple()
    assert term != "f"


def test_deep_terms_compare_and_hash_without_recursion():
    depth = 20_000
    left = list_to_term(list(range(depth)))
    right = list_to_term(list(range(depth)))
    changed = list_to_term([*range(depth - 1), -1])
    assert left == right
    assert hash(left) == hash(right)
    assert left != changed


def test_copied_and_pickled_terms_keep_their_arguments():
    term = _lam("base", make_term("App", make_term("Const", b"f"), bound_var(0)))
    for clone in (
        copy.copy(term),
        copy.deepcopy(term),
        pickle.loads(pickle.dumps(term)),
    ):
        assert clone.args == term.args
        assert clone == term
        assert hash(clone) == hash(term)


def test_with_produces_a_term_with_its_own_structural_hash():
    term = make_term("Const", "f")
    hash(term)
    renamed = term.with_(args=("plus",))
    assert renamed == make_term("Const", "plus")
    assert renamed != term
    assert hash(renamed) == hash(make_term("Const", "plus"))


@pytest.mark.parametrize(
    "args",
    [[make_term("A")], (make_term("A"), [1]), (None,), (torch.tensor(1.0),)],
    ids=["list-args", "list-element", "none", "tensor"],
)
def test_term_rejects_arguments_outside_term_arg(args):
    with pytest.raises(dx.ValidationError, match="Term 'C'"):
        Term(op="C", args=args)
