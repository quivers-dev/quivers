"""End-to-end tests for the shipped term-autoencoder example.

Compiles ``docs/examples/source/term_autoencoder.qvr``, the gallery
example for the combined signature / encoder / decoder / loss
surface, and exercises the full round trip: fold a term into a code
vector, sample and score terms from the code, and evaluate the
attached reconstruction loss. Negative cases pin the structural
diagnostics for unknown option keys, ops outside the signature, and
unresolved sort dims.

"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import cast


import pytest
import torch

from quivers.dsl import CompileError, load, loads, parse
from quivers.dsl.ast_nodes import (
    DecoderDecl,
    EncoderDecl,
    LossDecl,
    SignatureDecl,
)
from quivers.program import Program
from quivers.qiec import QiecModule, dumps, loads as load_qiec
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.module import validate_module
from quivers.qiec.structural_runtime import run_structural
from quivers.structural import Term, bound_var, make_term
from quivers.structural.decoder import Decoder
from quivers.structural.encoder import Encoder
from quivers.structural.losses import LossRegistry, TrainEnv
from quivers.structural.signature import Signature

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "term_autoencoder.qvr"
)


@pytest.fixture(scope="module")
def program() -> Program:
    return load(_EXAMPLE)


def _artifacts(program: Program) -> tuple[Encoder, Decoder, LossRegistry]:
    """Pull the example's compiled artifacts off the Program container
    with runtime type verification (the container exposes them as
    dynamically attached attributes)."""
    encoders = program.encoders
    decoders = program.decoders
    losses = program.losses
    assert isinstance(encoders, dict)
    assert isinstance(decoders, dict)
    assert isinstance(losses, LossRegistry)
    enc = cast("dict[str, Encoder]", encoders)["Enc"]
    dec = cast("dict[str, Decoder]", decoders)["Dec"]
    assert isinstance(enc, Encoder)
    assert isinstance(dec, Decoder)
    return enc, dec, losses


def _loss_env(term: Term) -> Mapping[str, TrainEnv]:
    """Wrap a term observation as a loss-evaluation environment."""
    return cast("Mapping[str, TrainEnv]", {"term": term})


def _example_term() -> Term:
    # \x : base. plus x
    return make_term(
        "Lam",
        make_term("Base", "base"),
        make_term("App", make_term("Const", "plus"), bound_var(0)),
    )


# ---------------------------------------------------------------------------
# Compilation and registries
# ---------------------------------------------------------------------------


def test_example_compiles_and_registers_artifacts(program: Program) -> None:
    signatures = program.signatures
    assert isinstance(signatures, dict)
    assert set(signatures) == {"STLC"}
    sig = cast("dict[str, Signature]", signatures)["STLC"]
    assert isinstance(sig, Signature)
    assert set(sig.sorts) == {"Term", "Type", "Name"}
    assert sig.sorts["Name"].kind == "data"
    assert sig.sorts["Name"].vocab_values == ("f", "x", "plus", "base")
    assert sig.constructors["App"].domain == ("Term", "Term")
    assert sig.binders["Lam"].binds[0].annot_sort == "Type"

    enc, dec, losses = _artifacts(program)
    assert enc.name == "Enc"
    assert dec.name == "Dec"
    assert len(losses.entries) == 1
    entry = losses.entries[0]
    assert entry.name == "reconstruct"
    assert entry.attachment_kind == "decoder"
    assert entry.target == "Dec"
    assert entry.weight is not None


def test_doc_comments_attach_to_every_declaration() -> None:
    module = parse(_EXAMPLE.read_text())
    structural = [
        stmt
        for stmt in module.statements
        if isinstance(stmt, (SignatureDecl, EncoderDecl, DecoderDecl, LossDecl))
    ]
    assert len(structural) == 4
    for decl in structural:
        assert decl.docs, f"{decl.kind} carries no #! doc comment"


def test_structural_declarations_lower_to_a_checked_qiec_graph(
    program: Program,
) -> None:
    module = program.qiec
    assert isinstance(module, QiecModule)
    validate_module(module)
    assert load_qiec(dumps(module)) == module
    assert [(sort.name, sort.constructors) for sort in module.index_sorts] == [
        ("STLC__Sort", ("Term", "Type", "Name"))
    ]
    assert [(family.name, family.closed) for family in module.families] == [
        ("STLC__Term", False)
    ]
    assert [computation.name for computation in module.computations] == [
        "Enc",
        "Dec",
        "Dec__nll",
        "reconstruct",
    ]


# ---------------------------------------------------------------------------
# Encoder / decoder round trip
# ---------------------------------------------------------------------------


def test_encoder_compresses_a_term(program: Program) -> None:
    enc, _, _ = _artifacts(program)
    # The explicit ``op`` rule is installed for App; scaffolded
    # defaults cover the remaining operators.
    rule = enc.op_fns["App"]
    assert rule.mode == "plain"
    assert rule.args == ("fun", "arg")
    assert set(enc.op_fns) == {"Const", "App", "Base", "Lam"}
    # The var_init override is keyed by the (variable, annotation)
    # sort pair the Lam binder declares.
    assert ("Term", "Type") in enc.var_init_fns

    code = enc(_example_term())
    assert code.shape == (24,)
    assert code.requires_grad


def test_decoder_samples_and_scores(program: Program) -> None:
    enc, dec, _ = _artifacts(program)
    torch.manual_seed(0)
    sample = dec(torch.randn(24))
    assert isinstance(sample, Term)

    term = _example_term()
    lp = dec.log_prob(term, enc(term))
    assert torch.is_tensor(lp)
    assert lp.shape == ()
    assert torch.isfinite(lp)
    lp.backward()


def _replace_first_leaf(term: Term, vocab: tuple[str, ...]) -> Term | None:
    """Swap the first string leaf (pre-order) for another vocab entry.

    Returns ``None`` when the term carries no string leaf.
    """
    args = list(term.args)
    for position, arg in enumerate(args):
        if isinstance(arg, str):
            args[position] = next(v for v in vocab if v != arg)
            return make_term(term.op, *args)
        if isinstance(arg, Term):
            replaced = _replace_first_leaf(arg, vocab)
            if replaced is not None:
                args[position] = replaced
                return make_term(term.op, *args)
    return None


def test_reconstruction_equality_rejects_a_wrong_leaf_below_the_root(
    program: Program,
) -> None:
    enc, dec, _ = _artifacts(program)
    vocab = ("f", "x", "plus", "base")
    decoded: Term | None = None
    redecoded: Term | None = None
    wrong: Term | None = None
    for seed in range(50):
        torch.manual_seed(seed)
        code = torch.randn(24)
        torch.manual_seed(seed)
        decoded = dec(code)
        torch.manual_seed(seed)
        redecoded = dec(code)
        wrong = _replace_first_leaf(decoded, vocab)
        if wrong is not None:
            break
    if decoded is None or redecoded is None or wrong is None:
        pytest.fail("no decoded sample carried a data leaf")
    # The same code and seed reproduce the same term.
    assert redecoded == decoded
    # A reconstruction that differs from the decoded term only in one
    # leaf keeps the root operator, and is a different term.
    assert wrong.op == decoded.op
    assert wrong.to_tuple() != decoded.to_tuple()
    assert wrong != decoded
    assert hash(wrong) != hash(decoded)
    assert len({wrong, decoded}) == 2
    # The wrong reconstruction is a well-formed term the decoder scores.
    assert torch.isfinite(dec.log_prob(wrong, enc(wrong)))


def test_decoder_samples_data_leaves_from_declared_vocab(
    program: Program,
) -> None:
    _, dec, _ = _artifacts(program)
    vocab = {"f", "x", "plus", "base"}

    def leaves(term: Term) -> list[str]:
        if term.op == "Data":
            return [str(term.args[0])]
        out: list[str] = []
        for arg in term.args:
            if isinstance(arg, Term):
                out.extend(leaves(arg))
            elif isinstance(arg, str):
                out.append(arg)
        return out

    torch.manual_seed(1)
    for _ in range(5):
        sample = dec(torch.randn(24))
        assert set(leaves(sample)) <= vocab


# ---------------------------------------------------------------------------
# Loss evaluation
# ---------------------------------------------------------------------------


def test_loss_evaluates_reconstruction_nll(program: Program) -> None:
    enc, dec, losses = _artifacts(program)
    term = _example_term()

    total = losses.evaluate(_loss_env(term))
    assert torch.is_tensor(total)
    assert total.requires_grad
    # weight=1.0, so the registry total is exactly the negative
    # round-trip log-likelihood.
    expected = -dec.log_prob(term, enc(term))
    torch.testing.assert_close(total, expected)

    attached = losses.evaluate_on("decoder", "Dec", _loss_env(term))
    torch.testing.assert_close(attached, expected)
    total.backward()


def test_qiec_loss_matches_classic_and_preserves_autograd(program: Program) -> None:
    enc, dec, losses = _artifacts(program)
    term = _example_term()
    for parameter in (*enc.parameters(), *dec.parameters()):
        parameter.grad = None

    expected = losses.evaluate(_loss_env(term))
    module = program.qiec
    assert isinstance(module, QiecModule)
    run = run_structural(module, program, "reconstruct", (term,))
    assert isinstance(run.value, torch.Tensor)
    torch.testing.assert_close(run.value, expected)
    assert [event.event for event in run.trace].count("operation.handled") == 2

    run.value.backward()
    parameters = (*enc.parameters(), *dec.parameters())
    assert any(parameter.grad is not None for parameter in parameters)


def test_standard_program_entry_runs_the_structural_loss(program: Program) -> None:
    """Structural entries use the same public invocation API as QIEC entries."""

    term = _example_term()
    expected = program.losses.evaluate(_loss_env(term))
    run = program.run("reconstruct", term)
    assert isinstance(run.value, torch.Tensor)
    torch.testing.assert_close(run.value, expected)
    assert run.result.runtime == "core+structural"


def test_program_registers_structural_parameters(program: Program) -> None:
    """Optimizers and checkpoints see every attached neural parameter."""

    names = dict(program.named_parameters())
    assert any(name.startswith("_qvr_encoder_Enc.") for name in names)
    assert any(name.startswith("_qvr_decoder_Dec.") for name in names)
    assert set(names) <= set(program.state_dict())


def test_structural_attachment_result_is_runtime_checked() -> None:
    """A provider cannot return a code with the wrong checked width."""

    class WrongWidth(torch.nn.Module):
        def forward(self, _term: object) -> torch.Tensor:
            return torch.zeros(3)

    compiled = load(_EXAMPLE)
    compiled.encoders["Enc"] = WrongWidth()
    with pytest.raises(ExecutionFailure) as caught:
        compiled.run("reconstruct", _example_term())
    assert caught.value.diagnostic.code == "qiec-run-evaluation"
    assert "does not inhabit the checked type" in str(caught.value)
    assert "name='Tensor'" in str(caught.value)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_BASE = """
signature S
    sorts
        Term : object [dim=8]
    constructors
        A : -> Term
        B : Term -> Term
"""


def test_unknown_encoder_option_key_is_rejected() -> None:
    src = (
        _BASE
        + """
encoder E : S [facotry=bow_encoder]
    dim Term = 8
"""
    )
    with pytest.raises(
        CompileError,
        match=r"encoder 'E': unknown option 'facotry'; did you mean 'factory'\?",
    ):
        loads(textwrap.dedent(src))


def test_unknown_decoder_option_key_is_rejected() -> None:
    src = (
        _BASE
        + """
decoder D : S [depht=4]
    body |-> recursive
"""
    )
    with pytest.raises(
        CompileError,
        match=r"decoder 'D': unknown option 'depht'; did you mean 'depth'\?",
    ):
        loads(textwrap.dedent(src))


def test_encoder_op_outside_signature_is_rejected() -> None:
    src = (
        _BASE
        + """
encoder E : S
    op Zap(x) |-> x
"""
    )
    with pytest.raises(
        CompileError,
        match=r"encoder 'E': op 'Zap' is not in signature 'S'",
    ):
        loads(textwrap.dedent(src))


def test_unresolved_sort_dim_is_rejected() -> None:
    # A sort with no dim on the signature and no ``dim`` override in
    # the decoder block has no resolvable embedding width.
    src = """
signature S2
    sorts
        Term : object
    constructors
        A : -> Term

decoder D : S2 [depth=3]
    body |-> recursive
"""
    with pytest.raises(CompileError, match=r"decoder 'D': sort 'Term' has no dim"):
        loads(textwrap.dedent(src))
