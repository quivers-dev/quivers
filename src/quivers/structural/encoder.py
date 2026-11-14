"""Encoder runtime: F-algebra homomorphisms T_Σ -> Vec_D.

A `Encoder` is a ``torch.nn.Module`` that, given a closed
`Term` over a signature, returns its vector embedding. The
recursion structure is supplied by the framework (algebra
homomorphism); per-operation parametric functions are supplied by
the user (declared in a ``encoder { … }`` DSL block) or by the
default scaffolding.

Canonical term form
-------------------

A term over Σ has exactly one of three argument shapes at each
position, determined by the codomain sort's kind:

* **object**: the argument is a `Term` whose ``op`` is a
  constructor of Σ, a binder of Σ, or the reserved built-in
  ``"BoundVar"``.
* **data**: the argument is a raw Python value (string, int,
  float, …) consumed by the encoder's per-data-sort embedder.
* **index**: the argument is a non-negative integer denoting a
  de-Bruijn index into the current scope context Γ.

No ``Term("Data", …)`` or other wrappings are accepted at runtime;
the canonical form is the only form.

The reserved ``"BoundVar"`` op is the built-in de-Bruijn reference:
``Term("BoundVar", (i,))`` at an object position reads the i-th
in-scope variable's embedding from Γ.

Binders thread a `Context` of in-scope variables: each entry
records the variable's sort, its embedding, and the type term
captured at binding. Compressing a binder constructor:

1. Compresses each of the binder's ``binds`` arguments in the
   OUTER context.
2. Synthesises a fresh variable embedding from each via the
   encoder's ``var_init`` function.
3. Pushes the new entries onto Γ and recurses on the ``scoped``
   arguments under the extended context.

Graph-shaped signatures are compressed by `forward_graph`:
per-vertex-kind ``init`` seeds initial embeddings, finitely many
``message_passing`` rounds alternate per-edge-kind ``message`` and
per-vertex-kind ``update`` functions, and a ``readout`` reduces
the final per-vertex embeddings to one fixed-length vector.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import didactic.api as dx
import torch
import torch.nn as nn

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

from .signature import (
    Constructor,
    Context,
    DataLeaf,
    EMPTY_CONTEXT,
    Signature,
    Term,
    TermArg,
)


BOUND_VAR_OP = "BoundVar"
"""The reserved op name of a bound-variable term.

A `Term` whose ``op`` is this name refers to a variable a binder
introduced. The signature compiler forbids it as a user-declared
constructor or binder name.
"""


type PerOpMode = Literal["plain", "recurrent", "attention"]
"""How a `PerOpFn` combines its children: ``"plain"`` applies the
function once, ``"recurrent"`` left-folds it across a list-shaped
signature, and ``"attention"`` gives each step the running prefix."""


class PerOpFn(ValueModel):
    """A per-operation parametric function.

    ``mode`` is ``"plain"`` (algebra-hom), ``"recurrent"``
    (left-fold across a list-shaped signature) or ``"attention"``
    (each step sees a running prefix list). For ``plain`` rules,
    ``fn`` is called with one positional argument per child
    embedding; for ``recurrent`` / ``attention`` the framework
    threads ``state`` / ``prefix`` arguments alongside the children.

    Function identity participates in equality and hashing. Copying
    or updating the record retains the callable and its live parameters.

    Parameters
    ----------
    op : str
        The constructor or binder name the function interprets.
    mode : PerOpMode
        How the function combines its children.
    args : tuple[str, ...]
        The names of the operation's arguments, in positional order.
    fn : Callable[..., torch.Tensor]
        The function mapping child embeddings to the operation's
        embedding.
    state_var : str or None
        The name of the threaded state argument in ``"recurrent"``
        mode.
    prefix_var : str or None
        The name of the running-prefix argument in ``"attention"``
        mode.
    """

    op: str
    mode: PerOpMode
    args: tuple[str, ...]
    fn: Callable[..., torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    state_var: str | None = None
    prefix_var: str | None = None

    @dx.model_validator()
    def _check_callable(self) -> None:
        """Require a callable operation implementation.

        Raises
        ------
        TypeError
            If the operation implementation is not callable.
        """
        if not callable(self.fn):
            raise TypeError("PerOpFn.fn must be callable")


def make_default_op_fn(
    op: str,
    arg_dims: tuple[int, ...],
    out_dim: int,
) -> tuple[nn.Module, Callable[..., torch.Tensor]]:
    """Build a parametric per-op function as a 2-layer MLP over
    concatenated child embeddings.

    ``arg_dims`` is the tuple of per-argument dimensions in the
    order in which the children are passed to the function. For
    binders this is ``(annot_dim_1, …, annot_dim_k, scoped_dim_1,
    …, scoped_dim_m)``. A nullary operation gets a learned constant
    vector instead of an MLP.

    Parameters
    ----------
    op : str
        The operation name, used in parameter names and diagnostics.
    arg_dims : tuple[int, ...]
        The embedding dimension of each child, in call order.
    out_dim : int
        The dimension of the operation's output embedding.

    Returns
    -------
    tuple[nn.Module, Callable[..., torch.Tensor]]
        The module owning the parameters, and the function that maps
        the child embeddings to the output embedding.

    Raises
    ------
    RuntimeError
        Raised by the returned function when it is called with a
        number of children other than ``len(arg_dims)``.
    """
    arity = len(arg_dims)
    if arity == 0:
        param = nn.Parameter(torch.randn(out_dim) * 0.1)
        mod = nn.Module()
        mod.register_parameter(f"const_{op}", param)

        def call() -> torch.Tensor:
            return param

        return mod, call

    in_dim = sum(arg_dims)
    hidden = max(out_dim, 64)
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )

    def call(*children: torch.Tensor) -> torch.Tensor:
        if len(children) != arity:
            raise RuntimeError(
                f"op {op!r}: expected {arity} children, got {len(children)}"
            )
        cat = torch.cat([c.reshape(-1) for c in children], dim=-1)
        return mlp(cat)

    return mlp, call


def make_default_var_init(
    in_dim: int, out_dim: int
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """A parametric `var_init(ty)` function: an MLP mapping a type
    embedding to a fresh variable embedding.

    Parameters
    ----------
    in_dim : int
        The dimension of the type-annotation embedding.
    out_dim : int
        The dimension of the variable embedding.

    Returns
    -------
    tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        The MLP owning the parameters, and the function that maps a
        type embedding to a variable embedding.
    """
    mlp = nn.Sequential(
        nn.Linear(in_dim, max(out_dim, 64)),
        nn.Tanh(),
        nn.Linear(max(out_dim, 64), out_dim),
    )

    def call(ty: torch.Tensor) -> torch.Tensor:
        return mlp(ty.reshape(-1))

    return mlp, call


class Encoder(nn.Module):
    """An F-algebra homomorphism T_Σ -> Vec_D realised as an
    ``nn.Module``.

    The graph-specific parameters (``iterations``, ``init_fns``,
    ``message_fns``, ``update_fns``, ``readout``) are used only by
    `forward_graph` on a graph-shaped signature.

    Parameters
    ----------
    name : str
        Identifier used in diagnostics.
    signature : Signature
        The Σ whose terms this encoder consumes.
    sort_dims : dict[str, int]
        Per-sort embedding dimension.
    op_fns : dict[str, PerOpFn]
        One entry per constructor and binder of Σ. The DSL compiler
        scaffolds defaults for omitted ops.
    var_init_fns : dict[tuple[str, str] | str, Callable]
        Variable initialisers called at every binder to mint a fresh
        embedding for each introduced variable. An annotated variable
        is keyed by ``(var_sort, annot_sort)`` and receives the
        annotation's embedding; an unannotated one is keyed by
        ``var_sort`` and receives ``None``.
    data_embedders : dict[str, Callable[[DataLeaf], torch.Tensor]]
        One entry per data sort: ``(raw_value) -> embedding``.
    modules_owned : list[nn.Module] or None
        Modules holding the parameters of the per-op functions,
        registered as submodules so their parameters train.
    iterations : int
        Number of message-passing rounds in `forward_graph`.
    init_fns : dict[str, Callable[[DataLeaf], torch.Tensor]] or None
        Per-vertex-kind initial embedders.
    message_fns : dict[str, Callable[..., torch.Tensor]] or None
        Per-edge-kind ``(src_e, tgt_e) -> message`` functions.
    update_fns : dict[str, Callable[..., torch.Tensor]] or None
        Per-vertex-kind ``(self_e, aggregated_msg) -> next_e``.
    readout : Callable[[list[torch.Tensor]], torch.Tensor] or None
        ``(list[Vec_D]) -> Vec_D`` reducer.
    """

    def __init__(
        self,
        name: str,
        signature: Signature,
        sort_dims: dict[str, int],
        op_fns: dict[str, PerOpFn],
        var_init_fns: dict[
            tuple[str, str] | str,
            Callable[[torch.Tensor | None], torch.Tensor],
        ],
        data_embedders: dict[str, Callable[[DataLeaf], torch.Tensor]],
        modules_owned: list[nn.Module] | None = None,
        iterations: int = 0,
        init_fns: dict[str, Callable[[DataLeaf], torch.Tensor]] | None = None,
        message_fns: dict[str, Callable[..., torch.Tensor]] | None = None,
        update_fns: dict[str, Callable[..., torch.Tensor]] | None = None,
        readout: Callable[[list[torch.Tensor]], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.signature = signature
        self.sort_dims = dict(sort_dims)
        self.op_fns = dict(op_fns)
        # Keyed by `(var_sort, annot_sort)` for annotated binders or
        # by `var_sort` (a plain string) for unannotated ones.
        self.var_init_fns: dict[
            tuple[str, str] | str,
            Callable[[torch.Tensor | None], torch.Tensor],
        ] = dict(var_init_fns)
        self.data_embedders = dict(data_embedders)
        self.iterations = iterations
        self.init_fns = dict(init_fns or {})
        self.message_fns = dict(message_fns or {})
        self.update_fns = dict(update_fns or {})
        self.readout = readout
        for i, m in enumerate(modules_owned or []):
            self.add_module(f"_op_{i}", m)

    # -----------------------------------------------------------------
    # Forward (inductive)
    # -----------------------------------------------------------------

    def forward(self, term: Term, ctx: Context | None = None) -> torch.Tensor:
        """Encode a term to its embedding by structural recursion.

        Parameters
        ----------
        term : Term
            The term to encode.
        ctx : Context or None
            The scope the term's free bound variables refer to;
            `EMPTY_CONTEXT` when omitted.

        Returns
        -------
        torch.Tensor
            The embedding of ``term``.

        Raises
        ------
        TypeError
            If ``term`` is not a `Term`, or an argument does not match
            the kind of its sort.
        RuntimeError
            If the term uses an operation, sort, or bound-variable
            index the encoder cannot interpret.
        """
        if not isinstance(term, Term):
            raise TypeError(
                f"encoder {self.name!r}: expected a Term at the root, got "
                f"{type(term).__name__} ({term!r}); construct one with "
                f"`make_term` or `bound_var`"
            )
        return self._compress_object(term, ctx or EMPTY_CONTEXT)

    def __call__(self, term: Term, ctx: Context | None = None) -> torch.Tensor:  # noqa: D401
        return super().__call__(term, ctx)

    def _compress_object(self, term: Term, ctx: Context) -> torch.Tensor:
        op = term.op
        sig = self.signature

        if op == BOUND_VAR_OP:
            if len(term.args) != 1 or not isinstance(term.args[0], int):
                raise RuntimeError(
                    f"encoder {self.name!r}: BoundVar must carry a single "
                    f"non-negative integer argument, got {term.args!r}"
                )
            idx = term.args[0]
            if idx < 0 or idx >= ctx.depth():
                raise RuntimeError(
                    f"encoder {self.name!r}: BoundVar({idx}) out of scope "
                    f"(context depth {ctx.depth()})"
                )
            return ctx.var(idx)

        if op in sig.binders:
            return self._compress_binder(term, ctx)
        if op in sig.constructors:
            return self._compress_constructor(term, ctx)

        raise RuntimeError(
            f"encoder {self.name!r}: op {op!r} is not a constructor / "
            f"binder of signature {sig.name!r} and is not the reserved "
            f"BoundVar"
        )

    def _compress_constructor(self, term: Term, ctx: Context) -> torch.Tensor:
        sig = self.signature
        cons = sig.constructors[term.op]
        if len(term.args) != cons.arity:
            raise RuntimeError(
                f"encoder {self.name!r}: constructor {term.op!r} expects "
                f"{cons.arity} arguments, got {len(term.args)}"
            )
        rule = self._require_op_fn(term.op)
        if rule.mode == "attention":
            return self._compress_attention_chain(term, ctx)
        children: list[torch.Tensor] = []
        for sort, arg in zip(cons.domain, term.args):
            children.append(self._compress_arg(arg, sort, ctx))
        return rule.fn(*children)

    def _compress_attention_chain(
        self,
        term: Term,
        ctx: Context,
    ) -> torch.Tensor:
        """Iteratively walk a chain of recursive applications of the
        same constructor, threading a *prefix* list of the
        non-recursive children's embeddings collected outside-in.

        For a `Cons(h_0, Cons(h_1, …, Cons(h_n, Nil)))` chain this
        yields, at step `i`, ``cons_fn(*non_rec_children_at_i,
        prefix=[h_0_emb, …, h_{i-1}_emb])``. The encoder's final
        embedding is the output of the deepest (innermost) step.
        """
        sig = self.signature
        op = term.op
        cons = sig.constructors[op]
        rec_idx = self._recursive_child_index(op, cons)
        rule = self._require_op_fn(op)

        # Walk the chain outside-in. Each entry records the
        # non-recursive child args (raw) and their sorts.
        chain: list[tuple[tuple, tuple[str, ...]]] = []
        current: Term = term
        while isinstance(current, Term) and current.op == op:
            non_rec_args = tuple(a for i, a in enumerate(current.args) if i != rec_idx)
            non_rec_sorts = tuple(s for i, s in enumerate(cons.domain) if i != rec_idx)
            chain.append((non_rec_args, non_rec_sorts))
            current = current.args[rec_idx]

        # Compress the chain's base case (innermost term).
        if not isinstance(current, Term):
            raise RuntimeError(
                f"encoder {self.name!r}: attention chain base of "
                f"{op!r} must be a Term, got {type(current).__name__}"
            )
        base_embed = self._compress_object(current, ctx)

        prefix: list[torch.Tensor] = []
        step_output = base_embed
        for non_rec_args, non_rec_sorts in chain:
            head_embeds = [
                self._compress_arg(a, s, ctx)
                for a, s in zip(non_rec_args, non_rec_sorts)
            ]
            # The per-op function receives the non-recursive child
            # embeddings followed by the prefix list (a fresh list
            # to prevent in-place mutation from leaking).
            step_output = rule.fn(*head_embeds, list(prefix), step_output)
            prefix = prefix + head_embeds
        return step_output

    def _recursive_child_index(self, op: str, cons: "Constructor") -> int:
        codomain = cons.codomain
        for i, s in enumerate(cons.domain):
            if s == codomain:
                return i
        raise RuntimeError(
            f"encoder {self.name!r}: constructor {op!r} declared with "
            f"`attention` mode requires at least one recursive child "
            f"(a position whose sort matches the codomain {codomain!r})"
        )

    def _compress_binder(self, term: Term, ctx: Context) -> torch.Tensor:
        sig = self.signature
        binder = sig.binders[term.op]
        if len(term.args) != binder.arity:
            raise RuntimeError(
                f"encoder {self.name!r}: binder {term.op!r} expects "
                f"{binder.arity} arguments, got {len(term.args)}"
            )

        # The binder's positional args are: one annotation per
        # annotated bound variable (outer-context), followed by the
        # scoped args (extended-context). Walk them in this order.
        annotated_binds = [b for b in binder.binds if b.annot_sort is not None]
        n_annots = len(annotated_binds)
        annot_args = term.args[:n_annots]
        scoped_args = term.args[n_annots:]

        # Compress the type annotations in the outer context.
        annot_embeds: list[torch.Tensor] = []
        for spec, arg in zip(annotated_binds, annot_args):
            assert spec.annot_sort is not None
            annot_embeds.append(self._compress_arg(arg, spec.annot_sort, ctx))

        # Mint a fresh variable embedding for each bound variable.
        # Annotated vars consult `var_init_fns[(annot_sort, var_sort)]`
        # to map the annotation embedding to the new variable's
        # embedding. Unannotated vars use `var_init_fns[var_sort]`
        # applied to a fresh zero vector of the var's dim.
        ctx_ext = ctx
        annotated_iter = iter(zip(annotated_binds, annot_embeds, annot_args))
        for spec in binder.binds:
            if spec.annot_sort is not None:
                ann_spec, e_annot, raw_annot = next(annotated_iter)
                e_var = self._make_var(spec.sort, ann_spec.annot_sort, e_annot)
                type_term = raw_annot if isinstance(raw_annot, Term) else None
                ctx_ext = ctx_ext.push(spec.sort, e_var, type_term)
            else:
                e_var = self._make_var(spec.sort, None, None)
                ctx_ext = ctx_ext.push(spec.sort, e_var, None)

        # Compress the scoped args in the extended context.
        scoped_embeds: list[torch.Tensor] = []
        for spec, arg in zip(binder.scoped, scoped_args):
            scoped_embeds.append(self._compress_arg(arg, spec.sort, ctx_ext))

        # The per-op function takes a flat list of child embeddings
        # in the order [annot_1, …, annot_k, scoped_1, …, scoped_m].
        children = annot_embeds + scoped_embeds
        rule = self._require_op_fn(term.op)
        return rule.fn(*children)

    def _make_var(
        self,
        var_sort: str,
        annot_sort: str | None,
        annot_embed: torch.Tensor | None,
    ) -> torch.Tensor:
        if annot_sort is not None:
            if annot_embed is None:
                raise RuntimeError(
                    f"encoder {self.name!r}: missing annotation embedding "
                    f"for var of sort {var_sort!r} annotated by {annot_sort!r}"
                )
            fn = self.var_init_fns.get((var_sort, annot_sort))
            if fn is None:
                raise RuntimeError(
                    f"encoder {self.name!r}: no var_init for variable "
                    f"sort {var_sort!r} annotated by {annot_sort!r}"
                )
            return fn(annot_embed)
        fn = self.var_init_fns.get(var_sort)
        if fn is None:
            raise RuntimeError(
                f"encoder {self.name!r}: no var_init for unannotated "
                f"variable sort {var_sort!r}"
            )
        return fn(None)

    def _compress_arg(self, arg: TermArg, sort: str, ctx: Context) -> torch.Tensor:
        sig = self.signature
        sort_decl = sig.sorts.get(sort)
        if sort_decl is None:
            raise RuntimeError(
                f"encoder {self.name!r}: arg sort {sort!r} is not in "
                f"signature {sig.name!r}"
            )
        kind = sort_decl.kind
        if kind == "data":
            if isinstance(arg, Term):
                raise TypeError(
                    f"encoder {self.name!r}: data-sorted arg at sort "
                    f"{sort!r} must be a raw Python value, got "
                    f"Term({arg.op!r}, …)"
                )
            embed = self.data_embedders.get(sort)
            if embed is None:
                raise RuntimeError(
                    f"encoder {self.name!r}: no embedder configured for "
                    f"data sort {sort!r}"
                )
            return embed(arg)
        if kind == "index":
            if not isinstance(arg, int):
                raise TypeError(
                    f"encoder {self.name!r}: index-sorted arg at sort "
                    f"{sort!r} must be an int, got {type(arg).__name__}"
                )
            if arg < 0 or arg >= ctx.depth():
                raise RuntimeError(
                    f"encoder {self.name!r}: index {arg} out of scope "
                    f"(context depth {ctx.depth()})"
                )
            return ctx.var(arg)
        if kind == "object":
            if not isinstance(arg, Term):
                raise TypeError(
                    f"encoder {self.name!r}: object-sorted arg at sort "
                    f"{sort!r} must be a Term, got {type(arg).__name__}"
                )
            return self._compress_object(arg, ctx)
        raise RuntimeError(f"encoder {self.name!r}: unknown sort kind {kind!r}")

    def _require_op_fn(self, op: str) -> PerOpFn:
        rule = self.op_fns.get(op)
        if rule is None:
            raise RuntimeError(f"encoder {self.name!r}: no per-op function for {op!r}")
        return rule

    # -----------------------------------------------------------------
    # Forward (graph)
    # -----------------------------------------------------------------

    def forward_graph(
        self,
        vertices: list[tuple[str, DataLeaf]],
        edges: list[tuple[str, int, int]],
    ) -> torch.Tensor:
        """Encode a graph by message passing followed by a readout.

        Each vertex is initialised by its kind's initial embedder;
        each of ``iterations`` rounds sends a message along every edge
        (both ways for an undirected edge kind), averages each
        vertex's inbox, and applies its kind's update function.

        Parameters
        ----------
        vertices : list[tuple[str, DataLeaf]]
            Each vertex's kind and payload.
        edges : list[tuple[str, int, int]]
            Each edge's kind, source index, and target index.

        Returns
        -------
        torch.Tensor
            The readout of the final vertex embeddings.

        Raises
        ------
        RuntimeError
            If the signature is not graph-shaped, ``iterations`` is not
            positive, an edge references an out-of-range vertex or an
            unknown edge kind, or a needed initial, message, update, or
            readout function is missing.
        """
        if not self.signature.is_graph():
            raise RuntimeError(
                f"encoder {self.name!r}: forward_graph requires a graph signature"
            )
        if self.iterations <= 0:
            raise RuntimeError(
                f"encoder {self.name!r}: graph encoder requires `iterations` > 0"
            )

        embeds: list[torch.Tensor] = []
        for vkind, payload in vertices:
            init = self.init_fns.get(vkind)
            if init is None:
                raise RuntimeError(
                    f"encoder {self.name!r}: no init for vertex_kind {vkind!r}"
                )
            embeds.append(init(payload))

        for _ in range(self.iterations):
            inboxes: list[list[torch.Tensor]] = [[] for _ in vertices]
            for ekind, src, tgt in edges:
                if src < 0 or tgt < 0 or src >= len(vertices) or tgt >= len(vertices):
                    raise RuntimeError(
                        f"encoder {self.name!r}: edge {(ekind, src, tgt)} "
                        f"references out-of-range vertex"
                    )
                edge_spec = self.signature.edge_kinds.get(ekind)
                if edge_spec is None:
                    raise RuntimeError(
                        f"encoder {self.name!r}: unknown edge_kind {ekind!r}"
                    )
                m_fn = self.message_fns.get(ekind)
                if m_fn is None:
                    raise RuntimeError(
                        f"encoder {self.name!r}: no message fn for edge_kind {ekind!r}"
                    )
                inboxes[tgt].append(m_fn(embeds[src], embeds[tgt]))
                if not edge_spec.directed:
                    inboxes[src].append(m_fn(embeds[tgt], embeds[src]))

            new_embeds = list(embeds)
            for i, (vkind, _) in enumerate(vertices):
                upd = self.update_fns.get(vkind)
                if upd is None:
                    raise RuntimeError(
                        f"encoder {self.name!r}: no update fn for vertex_kind {vkind!r}"
                    )
                msgs = inboxes[i]
                if msgs:
                    agg = torch.stack(msgs, dim=0).mean(dim=0)
                else:
                    agg = torch.zeros_like(embeds[i])
                new_embeds[i] = upd(embeds[i], agg)
            embeds = new_embeds

        if self.readout is None:
            raise RuntimeError(
                f"encoder {self.name!r}: graph encoder requires a readout function"
            )
        return self.readout(embeds)


def encoder_draw_message(name: str) -> str:
    """The refusal of a program ``sample`` or ``observe`` from an encoder.

    An encoder is a deterministic map from the terms of its signature to
    code vectors, so no program step draws from it.

    Parameters
    ----------
    name : str
        The encoder's name.

    Returns
    -------
    str
        The message.
    """
    return (
        f"{name!r} is an encoder, a deterministic map from terms of its "
        "signature to code vectors, so no program step draws from it; call it "
        f"on a term in a loss body, as in `{name}(term)`"
    )


__all__ = [
    "BOUND_VAR_OP",
    "PerOpMode",
    "PerOpFn",
    "make_default_op_fn",
    "make_default_var_init",
    "Encoder",
]
