"""Sequence-shaped signatures and canonical encoders / decoders.

Surface algebra::

    Seq[A] = Nil | Cons(A, Seq[A])

The principal shape behind autoregressive LMs (the "compress a
prefix to a vector, decode next-token from the vector" loop). All
encoders and decoders below are realised on top of the generic
[`quivers.structural.Encoder`][quivers.structural.Encoder] and
[`quivers.structural.Decoder`][quivers.structural.Decoder] runtimes, with sequence-
specific per-op functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from ..encoder import Encoder, PerOpFn
from ..decoder import Decoder
from ..signature import (
    DataLeaf,
    Constructor,
    Signature,
    Sort,
    Term,
)


def seq_signature(name: str = "Seq", dim: int = 64) -> Signature:
    """Return a cons-list signature with a data element sort.

    The object sort ``Seq`` has the constructors ``Nil`` and ``Cons``,
    the latter over an element of data sort ``A`` and a tail.

    Parameters
    ----------
    name : str
        The signature's name.
    dim : int
        The embedding dimension of every sort.

    Returns
    -------
    Signature
        The sequence signature.
    """
    sorts = {
        "Seq": Sort(name="Seq", kind="object", dim=dim),
        "A": Sort(name="A", kind="data", dim=dim),
    }
    constructors = {
        "Nil": Constructor(name="Nil", domain=(), codomain="Seq"),
        "Cons": Constructor(
            name="Cons",
            domain=("A", "Seq"),
            codomain="Seq",
        ),
    }
    return Signature(
        name=name,
        sorts_t=tuple(sorts.values()),
        constructors_t=tuple(constructors.values()),
    )


def _data_embedder(
    dim: int,
) -> tuple[nn.ParameterDict, Callable[[DataLeaf], torch.Tensor]]:
    table = nn.ParameterDict()

    def embed(key: DataLeaf, table=table, dim=dim) -> torch.Tensor:
        skey = str(key).replace(".", "_")
        if skey not in table:
            table[skey] = nn.Parameter(torch.randn(dim) * 0.1)
        return table[skey]

    return table, embed


def _learnable_nil(dim: int) -> tuple[nn.Module, torch.Tensor]:
    """A learnable zero-state vector for the `Nil` constructor.

    Returned as a `(module, parameter)` pair: ``module`` owns the
    parameter so that the enclosing `Encoder` picks it up
    through ``modules_owned`` and exposes it to optimizers via
    ``.parameters()``.
    """
    holder = nn.Module()
    p = nn.Parameter(torch.zeros(dim))
    holder.register_parameter("nil", p)
    return holder, p


def rnn_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """A GRU-cell encoder: ``Cons(head, tail)`` updates a hidden
    state from the tail's compressed vector and the head's element
    embedding.

    Parameters
    ----------
    sig : Signature or None
        A sequence signature with object sort ``Seq`` and data sort
        ``A``; `seq_signature` at dimension ``dim`` when omitted.
    dim : int
        The embedding dimension of both sorts.

    Returns
    -------
    Encoder
        The sequence encoder.
    """
    sig = sig or seq_signature(dim=dim)
    cell = nn.GRUCell(dim, dim)
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(a: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return cell(a.reshape(1, dim), tail.reshape(1, dim)).reshape(dim)

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": PerOpFn("Cons", "plain", ("head", "tail"), cons_fn),
    }
    return Encoder(
        name="RNN",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[cell, table, nil_mod],
    )


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """The sinusoidal position encodings of Vaswani et al. (2017).

    Parameters
    ----------
    length : int
        The number of positions.
    dim : int
        The model width.

    Returns
    -------
    torch.Tensor
        A ``(length, dim)`` tensor whose row ``p`` holds
        ``sin(p / 10000^(2i/dim))`` at even columns ``2i`` and the
        matching cosine at odd columns.
    """
    position = torch.arange(length, dtype=torch.get_default_dtype()).unsqueeze(1)
    rate = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.get_default_dtype())
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * rate)
    encoding[:, 1::2] = torch.cos(position * rate[: dim // 2])
    return encoding


def transformer_encoder(
    sig: Signature | None = None,
    dim: int = 64,
    heads: int = 4,
    layers: int = 1,
) -> Encoder:
    """A transformer encoder over the elements of a sequence.

    The element embeddings of ``Cons(h_0, Cons(h_1, …, Nil))``, with
    sinusoidal position encodings added, pass through ``layers``
    transformer encoder layers, each multi-head scaled dot-product
    self-attention followed by a position-wise feed-forward network,
    with residual connections and layer normalization [Vaswani et al.
    (2017)]. The sequence embedding is the mean of the outputs over
    positions; the empty sequence embeds as a learned vector.

    The encoder uses the ``attention`` per-operation mode: the step at
    the innermost ``Cons`` receives every element's embedding and
    attends over all of them, so each position attends to the whole
    sequence.

    Parameters
    ----------
    sig : Signature or None
        A sequence signature with object sort ``Seq`` and data sort
        ``A``; `seq_signature` at dimension ``dim`` when omitted.
    dim : int
        The embedding dimension of both sorts, the model width.
    heads : int
        The number of attention heads; it must divide ``dim``.
    layers : int
        The number of encoder layers.

    Returns
    -------
    Encoder
        The sequence encoder.

    Raises
    ------
    ValueError
        If ``heads`` or ``layers`` is not positive, or ``heads`` does
        not divide ``dim``.

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
    Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is
    all you need. *Advances in Neural Information Processing Systems*,
    30. [doi:10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
    """
    if heads < 1 or layers < 1:
        raise ValueError(
            f"transformer_encoder takes positive heads and layers, got "
            f"heads={heads}, layers={layers}"
        )
    if dim % heads != 0:
        raise ValueError(
            f"transformer_encoder: {heads} heads do not divide the width {dim}"
        )
    sig = sig or seq_signature(dim=dim)
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=4 * dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
    )
    stack = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(
        head: torch.Tensor, prefix: list[torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        elements = torch.stack([*prefix, head], dim=0).reshape(-1, dim)
        positioned = elements + _sinusoidal_positions(elements.shape[0], dim)
        return stack(positioned.unsqueeze(0)).squeeze(0).mean(dim=0)

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": PerOpFn(
            "Cons", "attention", ("head", "tail"), cons_fn, prefix_var="prefix"
        ),
    }
    return Encoder(
        name="Transformer",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[stack, table, nil_mod],
    )


def bow_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """An order-independent sum-of-element-embeddings encoder.

    Each ``Cons(head, tail)`` step adds the head's element embedding
    to the running tail embedding; ``Nil`` is a learnable zero.
    Equivalent to a bag-of-tokens fixed-length representation.

    Parameters
    ----------
    sig : Signature or None
        A sequence signature with object sort ``Seq`` and data sort
        ``A``; `seq_signature` at dimension ``dim`` when omitted.
    dim : int
        The embedding dimension of both sorts.

    Returns
    -------
    Encoder
        The sequence encoder.
    """
    sig = sig or seq_signature(dim=dim)
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(a: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return a + tail

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": PerOpFn("Cons", "plain", ("head", "tail"), cons_fn),
    }
    return Encoder(
        name="BoW",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[table, nil_mod],
    )


def ar_decoder(
    sig: Signature | None = None,
    dim: int = 64,
    vocab: list[DataLeaf] | None = None,
    depth: int = 64,
) -> Decoder:
    """An autoregressive decoder over a sequence signature.

    ``vocab`` is the closed token set the primitive head selects
    from. Required; the runtime raises on an empty vocabulary.

    Parameters
    ----------
    sig : Signature or None
        A sequence signature with object sort ``Seq`` and data sort
        ``A``; `seq_signature` at dimension ``dim`` when omitted.
    dim : int
        The embedding dimension of both sorts.
    vocab : list[DataLeaf] or None
        The closed set of element values the decoder emits.
    depth : int
        The recursion-depth bound, which caps the sampled length.

    Returns
    -------
    Decoder
        The sequence decoder.

    Raises
    ------
    ValueError
        If ``vocab`` is missing or empty.
    """
    if not vocab:
        raise ValueError("ar_decoder requires a non-empty vocabulary")
    sig = sig or seq_signature(dim=dim)
    # Two structural choices: Nil, Cons. (Seq has no binders and so
    # no BoundVar candidate is ever activated.)
    structure_head = nn.Linear(dim, 2)
    primitive = nn.Linear(dim, len(vocab))
    # Factor for Cons (arity 2 over sort Seq): split into head + tail.
    factor_lin = nn.Linear(dim, 2 * dim)
    # Bilinear binder-select. Seq itself never invokes this, but
    # signatures that compose Seq into a larger algebra (with a
    # binder reaching into Seq scope) do; the scorer is a real
    # learned bilinear key/query pair.
    bs_q = nn.Linear(dim, dim)
    bs_k = nn.Linear(dim, dim)

    def structure_fn(v: torch.Tensor) -> torch.Tensor:
        return structure_head(v.reshape(-1))

    def primitive_fn(v: torch.Tensor) -> torch.Tensor:
        return primitive(v.reshape(-1))

    def factor_2(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = factor_lin(v.reshape(-1))
        return out[:dim], out[dim:]

    def binder_select_fn(v: torch.Tensor, embeds: list[torch.Tensor]) -> torch.Tensor:
        qv = bs_q(v.reshape(-1))
        keys = torch.stack([bs_k(e.reshape(-1)) for e in embeds], dim=0)
        return keys @ qv

    return Decoder(
        name="AR",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        depth=depth,
        structure_fns={"Seq": structure_fn},
        primitive_fns={"A": primitive_fn},
        factor_fns={"Seq": {2: factor_2}},
        binder_select_fn=binder_select_fn,
        data_vocab={"A": vocab},
        modules_owned=[structure_head, primitive, factor_lin, bs_q, bs_k],
    )


def list_to_term(elements: list[DataLeaf]) -> Term:
    """Convert a Python list to a right-nested ``Cons(..., Nil)`` term.

    Parameters
    ----------
    elements : list[DataLeaf]
        The sequence elements, in order.

    Returns
    -------
    Term
        The cons-list term over `seq_signature`'s constructors.
    """
    term = Term(op="Nil", args=())
    for e in reversed(elements):
        term = Term(op="Cons", args=(e, term))
    return term


def term_to_list(term: Term) -> list[DataLeaf]:
    """Inverse of `list_to_term`.

    Parameters
    ----------
    term : Term
        A cons-list term.

    Returns
    -------
    list[DataLeaf]
        The heads of the ``Cons`` chain, outermost first; the walk
        stops at the first term that is not a ``Cons``.
    """
    out: list[DataLeaf] = []
    cur = term
    while isinstance(cur, Term) and cur.op == "Cons":
        head, tail = cur.args
        out.append(head)
        cur = tail
    return out


__all__ = [
    "seq_signature",
    "rnn_encoder",
    "transformer_encoder",
    "bow_encoder",
    "ar_decoder",
    "list_to_term",
    "term_to_list",
]
