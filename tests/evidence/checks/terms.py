"""Checks of structural encoders and decoders against enumerated terms."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction
from functools import partial

import torch

from quivers.structural.decoder import Decoder
from quivers.structural.encoder import Encoder, PerOpFn
from quivers.structural.signature import Constructor, Signature, Sort, Term
from tests.evidence.checks._registry import add
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_number,
)
from tests.evidence.oracles import derivations
from tests.evidence.oracles.derivations import Tree

LEAF, LEFT, RIGHT, SHIFT = (Fraction(v) for v in ("3/2", "1/4", "5/8", "1/3"))
"""The leaf value, the two child coefficients, and the constant of a node."""


def signature(extra: int = 0) -> Signature:
    """A signature of binary trees with optional unused constructors.

    Parameters
    ----------
    extra : int
        How many unused constructors to declare.

    Returns
    -------
    Signature
        The signature.
    """
    return Signature(
        name="Tree",
        sorts_t=(Sort(name="T", kind="object", dim=1),),
        constructors_t=(
            Constructor(name="Leaf", domain=(), codomain="T"),
            Constructor(name="Node", domain=("T", "T"), codomain="T"),
            *(
                Constructor(name=f"Unused{i}", domain=("T",), codomain="T")
                for i in range(extra)
            ),
        ),
    )


def leaf() -> Term:
    """The term without children.

    Returns
    -------
    Term
        The leaf.
    """
    return Term(op="Leaf", args=())


def node(left: Term, right: Term) -> Term:
    """A term with two children.

    Parameters
    ----------
    left : Term
        The first child.
    right : Term
        The second child.

    Returns
    -------
    Term
        The node.
    """
    return Term(op="Node", args=(left, right))


def encoder(parameters: dict[str, torch.Tensor], extra: int = 0) -> Encoder:
    """An encoder whose node value is a linear form of its children.

    Parameters
    ----------
    parameters : dict of str to torch.Tensor
        The leaf value, the coefficients, and the constant.
    extra : int
        How many unused constructors the signature declares.

    Returns
    -------
    Encoder
        The encoder.
    """

    def combine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """The value of a node.

        Parameters
        ----------
        left : torch.Tensor
            The first child's value.
        right : torch.Tensor
            The second child's value.

        Returns
        -------
        torch.Tensor
            The linear form.
        """
        return (
            parameters["left"] * left
            + parameters["right"] * right
            + parameters["shift"]
        )

    return Encoder(
        name="fold",
        signature=signature(extra),
        sort_dims={"T": 1},
        op_fns={
            "Leaf": PerOpFn(
                op="Leaf", mode="plain", args=(), fn=lambda: parameters["leaf"]
            ),
            "Node": PerOpFn(op="Node", mode="plain", args=("l", "r"), fn=combine),
        },
        var_init_fns={},
        data_embedders={},
    )


def as_tree(term: Term) -> Tree:
    """Read a term as an oracle derivation.

    Parameters
    ----------
    term : Term
        The term.

    Returns
    -------
    Tree
        The derivation.
    """
    children = [child for child in term.args if isinstance(child, Term)]
    return (term.op, ("T",), None, tuple(as_tree(child) for child in children))


def term_value(term: Term) -> Fraction:
    """The value of a term by an oracle fold.

    Parameters
    ----------
    term : Term
        The term.

    Returns
    -------
    Fraction
        The value.
    """

    def algebra(name: str, children: tuple[Fraction, ...]) -> Fraction:
        """The value of one operation.

        Parameters
        ----------
        name : str
            The operation.
        children : tuple of Fraction
            The children's values.

        Returns
        -------
        Fraction
            The value.
        """
        if name == "Leaf":
            return LEAF
        return LEFT * children[0] + RIGHT * children[1] + SHIFT

    return derivations.fold(as_tree(term), algebra)


def asymmetric(depth: int) -> Term:
    """A term whose two sides differ at every node.

    Parameters
    ----------
    depth : int
        The number of nodes.

    Returns
    -------
    Term
        The term.
    """
    term = leaf()
    for position in range(depth):
        term = node(term, leaf()) if position % 2 else node(leaf(), term)
    return term


def parameters(requires_grad: bool = False) -> dict[str, torch.Tensor]:
    """The encoder's parameters as float64 tensors.

    Parameters
    ----------
    requires_grad : bool
        Whether they are differentiated.

    Returns
    -------
    dict of str to torch.Tensor
        The parameters.
    """
    values = {"leaf": LEAF, "left": LEFT, "right": RIGHT, "shift": SHIFT}
    return {
        name: torch.tensor(
            [float(value)], dtype=torch.float64, requires_grad=requires_grad
        )
        for name, value in values.items()
    }


def check_encoder() -> None:
    """Compare an encoder's fold with the oracle's, one term at a time."""
    encode = encoder(parameters())
    for depth in (0, 1, 2, 5, 12):
        term = asymmetric(depth)
        compare_number(
            float(encode(term)),
            term_value(term),
            Accuracy(domain="linear", operations=4 * (depth + 1)),
            f"fold of depth {depth}",
        )


def check_encoder_gradient() -> None:
    """Compare an encoder's gradient with the derivative of the fold.

    On one node the value is ``left * leaf + right * leaf + shift``.
    """
    leaves = parameters(requires_grad=True)
    value = encoder(leaves)(node(leaf(), leaf())).sum()
    found = torch.autograd.grad(value, list(leaves.values()), allow_unused=True)
    compare_gradients(
        dict(zip(leaves, found, strict=True)),
        {"leaf": LEFT + RIGHT, "left": LEAF, "right": LEAF, "shift": Fraction(1)},
        Accuracy(domain="linear", operations=16),
        "fold gradient",
    )


def check_encoder_update() -> None:
    """Compare a fold before and after its parameters change."""
    leaves = parameters()
    encode = encoder(leaves)
    for factor in (1.0, 0.5):
        with torch.no_grad():
            leaves["shift"].mul_(factor)
        compare_number(
            float(encode(node(leaf(), leaf()))),
            LEFT * LEAF + RIGHT * LEAF + SHIFT * Fraction(factor),
            Accuracy(domain="linear", operations=16),
            f"fold after an update by {factor}",
        )


def check_encoder_refusal() -> None:
    """Require a term with an undeclared operation to be refused by name.

    Raises
    ------
    Mismatch
        If a value is returned or the diagnostic differs.
    """
    wanted = (
        "encoder 'fold': op 'Other' is not a constructor / binder of signature "
        "'Tree' and is not the reserved BoundVar"
    )
    try:
        value = encoder(parameters())(Term(op="Other", args=()))
    except RuntimeError as refusal:
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"an undeclared operation gave {value}")


DEPTH = 2
"""The decoder's depth bound."""

#: The structure logits of leaf and node.
LOGITS = (Fraction(1, 3), Fraction(5, 4))


def decoder(
    logits: torch.Tensor, structure: Callable[[torch.Tensor], torch.Tensor] | None = None
) -> Decoder:
    """A decoder of binary trees with constant structure logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits of leaf and node.
    structure : Callable or None
        Another producer of logits.

    Returns
    -------
    Decoder
        The decoder.
    """
    return Decoder(
        name="trees",
        signature=signature(),
        sort_dims={"T": 1},
        depth=DEPTH,
        structure_fns={"T": structure or (lambda vector: logits)},
        primitive_fns={},
        factor_fns={"T": {2: lambda vector: (vector, vector)}},
        binder_select_fn=lambda vector, variables: torch.zeros(len(variables)),
        data_vocab={},
    )


def terms_within(budget: int) -> list[Term]:
    """Every tree a decoder with a budget can produce.

    Parameters
    ----------
    budget : int
        The remaining depth.

    Returns
    -------
    list of Term
        The trees.
    """
    if budget <= 0:
        return [leaf()]
    below = terms_within(budget - 1)
    return [leaf(), *(node(a, b) for a, b in itertools.product(below, below))]


def term_log_probability(term: Term, budget: int) -> float:
    """The log probability of a tree from its definition.

    Parameters
    ----------
    term : Term
        The tree.
    budget : int
        The remaining depth.

    Returns
    -------
    float
        The log probability: one choice per node above the depth bound.
    """
    if budget <= 0:
        return 0.0
    scale = math.log(math.fsum(math.exp(float(value)) for value in LOGITS))
    if term.op == "Leaf":
        return float(LOGITS[0]) - scale
    answer = float(LOGITS[1]) - scale
    for child in term.args:
        assert isinstance(child, Term)
        answer += term_log_probability(child, budget - 1)
    return answer


def decoder_accuracy() -> Accuracy:
    """The accuracy of a decoder's score.

    Returns
    -------
    Accuracy
        The accuracy: at most seven choices, each one normalized
        exponential of two entries.
    """
    return Accuracy(domain="log", operations=7 * 6 * ELEMENTARY, magnitude=8.0)


def check_decoder() -> None:
    """Compare a decoder's scores with their definition, and their mass with one."""
    decode = decoder(torch.tensor([float(v) for v in LOGITS], dtype=torch.float64))
    vector = torch.zeros(1, dtype=torch.float64)
    mass = 0.0
    for term in terms_within(DEPTH):
        score = decode.log_prob(term, vector)
        compare_number(
            float(score),
            term_log_probability(term, DEPTH),
            decoder_accuracy(),
            f"score of {term.to_tuple()}",
        )
        mass += math.exp(float(score))
    compare_number(
        mass,
        1,
        Accuracy(
            domain="linear",
            operations=decoder_accuracy().operations * 8,
            condition=1.0,
        ),
        "mass of all trees",
    )


def check_decoder_gradient() -> None:
    """Compare a decoder's gradient with the one of a normalized exponential.

    A tree with one node above the depth bound chooses a node and then two
    leaves, so the derivative is the count of each choice minus its
    expected count.
    """
    logits = torch.tensor(
        [float(v) for v in LOGITS], dtype=torch.float64, requires_grad=True
    )
    score = decoder(logits).log_prob(
        node(leaf(), leaf()), torch.zeros(1, dtype=torch.float64)
    )
    (found,) = torch.autograd.grad(score, logits)
    weights = [math.exp(float(value)) for value in LOGITS]
    chance = [weight / math.fsum(weights) for weight in weights]
    reference = {"leaf": 2 - 3 * chance[0], "node": 1 - 3 * chance[1]}
    smallest = min(abs(value) for value in reference.values())
    compare_gradients(
        {"leaf": found[0], "node": found[1]},
        reference,
        Accuracy(
            domain="linear",
            operations=5 * decoder_accuracy().operations,
            condition=3 / smallest,
        ),
        "decoder gradient",
    )


def check_decoder_support() -> None:
    """Require a tree below the depth bound to be refused by name.

    Raises
    ------
    Mismatch
        If a score is returned or the diagnostic differs.
    """
    decode = decoder(torch.zeros(2, dtype=torch.float64))
    deep = leaf()
    for _ in range(DEPTH + 1):
        deep = node(deep, leaf())
    wanted = (
        "decoder 'trees': op 'Node' not in candidate set at sort 'T'; the term "
        "may violate the depth budget or reference an unknown op"
    )
    try:
        score = decode.log_prob(deep, torch.zeros(1, dtype=torch.float64))
    except RuntimeError as refusal:
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"a tree that is too deep scored {score}")


def check_decoder_replay() -> None:
    """Require every sampled tree to be scored at its sampling probability."""
    decode = decoder(torch.tensor([float(v) for v in LOGITS], dtype=torch.float64))
    vector = torch.zeros(1, dtype=torch.float64)
    support = {term.to_tuple() for term in terms_within(DEPTH)}
    torch.manual_seed(0)
    for _ in range(20):
        term = decode(vector)
        if term.to_tuple() not in support:
            raise Mismatch("support", math.inf, f"{term.to_tuple()} is not a tree")
        compare_number(
            float(decode.log_prob(term, vector)),
            term_log_probability(term, DEPTH),
            decoder_accuracy(),
            "replayed tree",
        )


def _register() -> None:
    """Register every check of this module."""
    add("terms/encoder", "encoder", "Q24", check_encoder)
    add("terms/encoder_gradient", "encoder", "Q1g", check_encoder_gradient)
    add("terms/encoder_update", "encoder", "Q24", check_encoder_update)
    add("terms/encoder_refusal", "encoder", "Q24", check_encoder_refusal)
    add("terms/decoder", "decoder", "Q14", check_decoder)
    add("terms/decoder_gradient", "decoder", "Q1g", check_decoder_gradient)
    add("terms/decoder_support", "decoder", "Q24", check_decoder_support)
    add("terms/decoder_replay", "decoder", "Q1", check_decoder_replay)


_register()
