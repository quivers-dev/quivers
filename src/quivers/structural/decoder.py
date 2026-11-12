"""Decoder runtime: Markov kernels from code vectors to terms.

A `Decoder` is a Markov kernel ``R^D -> T_Σ`` from code vectors to the
terms of a signature Σ, that is, a morphism ``R^D -> T_Σ`` of the Kleisli
category of the distribution monad. It is a ``torch.nn.Module`` exposing
two operations:

* `sample`: draws a single `Term` from the
  distribution induced by an input vector.
* `log_prob`: scores an observed term under the same
  distribution.

The corecursion structure over a signature Σ is:

1. At each sort position, the decoder produces logits over its
   *choice set*: every constructor and binder whose codomain is
   that sort, plus the built-in `BOUND_VAR_OP` whenever the
   context Γ contains at least one in-scope variable of that sort.
2. For the chosen op, the parent vector is split into per-child
   sub-vectors by the per-(sort, arity) ``factor`` function, and
   the decoder recurses on each child under the same canonical
   form used by the encoder.
3. Data-sorted children are sampled from a closed vocabulary via
   the per-sort ``primitive`` head; index-sorted children are
   sampled via `binder_select` over the in-scope variables.
4. Binder ops extend Γ before recursing on their scoped arguments,
   exactly mirroring the encoder.

Termination is depth-bounded at construction. At the budget limit
the choice set is restricted to recursion-terminating ops (every
child sort is data / index, never object); if no such op exists at
a sort, the decoder raises.

No silent type coercion or sentinel value is ever emitted: an
observed term whose shape doesn't match the canonical form raises.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .encoder import BOUND_VAR_OP
from .signature import (
    Context,
    DataLeaf,
    EMPTY_CONTEXT,
    Signature,
    Term,
    TermArg,
    bound_var,
)


def _categorical(logits: torch.Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=logits)


class Decoder(nn.Module):
    """A Kleisli coalgebraic decoder over a signature.

    Parameters
    ----------
    name : str
        Identifier used in diagnostics.
    signature : Signature
        The Σ whose terms this decoder generates.
    sort_dims : dict[str, int]
        Per-sort embedding dimension.
    depth : int
        Hard upper bound on recursion depth. Sampling beyond this
        depth is restricted to recursion-terminating ops at each
        sort.
    structure_fns : dict[str, callable]
        Per-sort logit producers ``(vec) -> Tensor``. Indexed by
        sort name, plus a shared ``"*"`` entry consulted when no
        sort-specific entry exists.
    primitive_fns : dict[str, callable]
        Per-data-sort logit producers over the closed token vocab.
    factor_fns : dict[str, dict[int, callable]]
        Per-sort, per-arity child-vector projections
        ``(parent_vec) -> tuple[Tensor, …]`` of ``n`` sub-vectors.
    binder_select_fn : callable
        ``(parent_vec, list_of_var_embeddings) -> Tensor`` of
        logits over the in-scope variables of a sort, used by
        `BOUND_VAR_OP` and by index-sorted child positions.
    data_vocab : dict[str, list]
        Per-data-sort closed vocabulary aligned with the column
        order of the corresponding ``primitive_fns`` output.
    modules_owned : list[nn.Module] or None
        Modules holding the parameters of the logit producers and
        projections, registered as submodules so their parameters
        train.

    Raises
    ------
    ValueError
        If ``depth`` is not positive.
    """

    def __init__(
        self,
        name: str,
        signature: Signature,
        sort_dims: dict[str, int],
        depth: int,
        structure_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
        primitive_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
        factor_fns: dict[
            str, dict[int, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]]
        ],
        binder_select_fn: Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor],
        data_vocab: dict[str, list[DataLeaf]],
        modules_owned: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.signature = signature
        self.sort_dims = dict(sort_dims)
        if depth <= 0:
            raise ValueError(f"decoder {name!r}: depth must be positive, got {depth}")
        self.depth = depth
        self.structure_fns = dict(structure_fns)
        self.primitive_fns = dict(primitive_fns)
        self.factor_fns = {s: dict(fs) for s, fs in factor_fns.items()}
        self.binder_select_fn = binder_select_fn
        self.data_vocab = dict(data_vocab)
        self._candidates_by_sort = self._collect_candidates()
        for i, m in enumerate(modules_owned or []):
            self.add_module(f"_dec_{i}", m)

    def _collect_candidates(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, c in self.signature.constructors.items():
            out.setdefault(c.codomain, []).append(name)
        for name, b in self.signature.binders.items():
            out.setdefault(b.codomain, []).append(name)
        return out

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    def sample(
        self,
        vec: torch.Tensor,
        ctx: Context | None = None,
        sort: str | None = None,
    ) -> Term:
        """Draw one term from the decoder's distribution.

        Parameters
        ----------
        vec : torch.Tensor
            The embedding to decode from.
        ctx : Context or None
            The scope of in-scope bound variables; `EMPTY_CONTEXT`
            when omitted.
        sort : str or None
            The sort of the term to draw; the first declared object
            sort when omitted.

        Returns
        -------
        Term
            The sampled term.

        Raises
        ------
        RuntimeError
            If no sort is given and the signature has no object sort,
            or the decoder lacks a function, vocabulary, or candidate
            it needs at some position.
        """
        return self._decode_object(
            vec,
            ctx or EMPTY_CONTEXT,
            self._resolve_sort(sort),
            self.depth,
        )

    def __call__(
        self,
        vec: torch.Tensor,
        ctx: Context | None = None,
        sort: str | None = None,
    ) -> Term:  # noqa: D401
        return self.sample(vec, ctx, sort)

    def _resolve_sort(self, sort: str | None) -> str:
        if sort is not None:
            return sort
        sig = self.signature
        # Principal sort: the first declared object sort.
        for s in sig.sorts.values():
            if s.kind == "object":
                return s.name
        raise RuntimeError(
            f"decoder {self.name!r}: signature {sig.name!r} has no object "
            f"sort; specify `sort=…` explicitly"
        )

    # ---- per-sort dispatchers ----

    def _decode_object(
        self,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
        budget: int,
    ) -> Term:
        candidates = self._object_candidates(ctx, sort, budget)
        logits = self._structure_logits(vec, sort, candidates)
        idx = int(_categorical(logits).sample().item())
        choice = candidates[idx]
        if choice == BOUND_VAR_OP:
            return self._decode_bound_var(vec, ctx, sort)
        return self._decode_op(choice, vec, ctx, budget)

    def _decode_data(self, vec: torch.Tensor, sort: str) -> DataLeaf:
        vocab = self.data_vocab.get(sort)
        if not vocab:
            raise RuntimeError(
                f"decoder {self.name!r}: data sort {sort!r} has no vocabulary; "
                f"populate `data_vocab[{sort!r}]` before sampling"
            )
        fn = self.primitive_fns.get(sort) or self.primitive_fns.get("*")
        if fn is None:
            raise RuntimeError(
                f"decoder {self.name!r}: no primitive head for data sort {sort!r}"
            )
        logits = fn(vec)
        if logits.shape[-1] < len(vocab):
            raise RuntimeError(
                f"decoder {self.name!r}: primitive head for {sort!r} produces "
                f"{logits.shape[-1]} logits but the vocabulary has "
                f"{len(vocab)} tokens"
            )
        idx = int(_categorical(logits[..., : len(vocab)]).sample().item())
        return vocab[idx]

    def _decode_index(self, vec: torch.Tensor, ctx: Context, sort: str) -> int:
        in_scope = ctx.by_sort(sort)
        if not in_scope:
            raise RuntimeError(
                f"decoder {self.name!r}: no in-scope variable of sort {sort!r}"
            )
        embeds = [e.embedding for _, e in in_scope]
        logits = self.binder_select_fn(vec, embeds).reshape(-1)
        if logits.shape[0] != len(in_scope):
            raise RuntimeError(
                f"decoder {self.name!r}: binder_select returned "
                f"{logits.shape[0]} logits for {len(in_scope)} in-scope "
                f"variables at sort {sort!r}"
            )
        idx = int(_categorical(logits).sample().item())
        return in_scope[idx][0]

    def _decode_bound_var(
        self,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
    ) -> Term:
        return bound_var(self._decode_index(vec, ctx, sort))

    def _decode_op(
        self,
        op: str,
        vec: torch.Tensor,
        ctx: Context,
        budget: int,
    ) -> Term:
        sig = self.signature
        if op in sig.constructors:
            cons = sig.constructors[op]
            if cons.arity == 0:
                return Term(op=op, args=())
            sub_vecs = self._factor(vec, op, cons.codomain, cons.arity)
            children = tuple(
                self._decode_child(sub_vecs[i], ctx, cons.domain[i], budget - 1)
                for i in range(cons.arity)
            )
            return Term(op=op, args=children)
        if op in sig.binders:
            b = sig.binders[op]
            n_annots = sum(1 for s in b.binds if s.annot_sort is not None)
            sub_vecs = self._factor(vec, op, b.codomain, b.arity)
            binder_children: list[TermArg] = []
            # 1. Decode each type annotation in the outer context.
            annot_terms: list[Term | None] = []
            annot_vecs: list[torch.Tensor | None] = []
            ai = 0
            for spec in b.binds:
                if spec.annot_sort is not None:
                    ann_vec = sub_vecs[ai]
                    ann = self._decode_child(
                        ann_vec,
                        ctx,
                        spec.annot_sort,
                        budget - 1,
                    )
                    binder_children.append(ann)
                    annot_terms.append(ann if isinstance(ann, Term) else None)
                    annot_vecs.append(ann_vec)
                    ai += 1
                else:
                    annot_terms.append(None)
                    annot_vecs.append(None)
            # 2. Extend the context with one entry per bound variable.
            ctx_ext = ctx
            for spec, ann_term, ann_vec in zip(
                b.binds,
                annot_terms,
                annot_vecs,
            ):
                slot_vec = (
                    ann_vec
                    if ann_vec is not None
                    else torch.zeros(self.sort_dims[spec.sort])
                )
                ctx_ext = ctx_ext.push(spec.sort, slot_vec, ann_term)
            # 3. Decode each scoped argument in the extended context.
            for j, spec in enumerate(b.scoped):
                child = self._decode_child(
                    sub_vecs[n_annots + j],
                    ctx_ext,
                    spec.sort,
                    budget - 1,
                )
                binder_children.append(child)
            return Term(op=op, args=tuple(binder_children))
        raise RuntimeError(
            f"decoder {self.name!r}: op {op!r} is not a constructor / binder"
        )

    def _decode_child(
        self,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
        budget: int,
    ) -> TermArg:
        sort_decl = self.signature.sorts.get(sort)
        if sort_decl is None:
            raise RuntimeError(f"decoder {self.name!r}: unknown sort {sort!r}")
        if sort_decl.kind == "data":
            return self._decode_data(vec, sort)
        if sort_decl.kind == "index":
            return self._decode_index(vec, ctx, sort)
        return self._decode_object(vec, ctx, sort, budget)

    def _object_candidates(
        self,
        ctx: Context,
        sort: str,
        budget: int,
    ) -> list[str]:
        cands = list(self._candidates_by_sort.get(sort, []))
        if ctx.by_sort(sort):
            cands.append(BOUND_VAR_OP)
        if not cands:
            raise RuntimeError(
                f"decoder {self.name!r}: sort {sort!r} has no constructors, "
                f"binders, or in-scope variables"
            )
        if budget <= 0:
            terminating = [
                c
                for c in cands
                if c == BOUND_VAR_OP or self._is_recursion_terminating(c)
            ]
            if not terminating:
                raise RuntimeError(
                    f"decoder {self.name!r}: depth budget exhausted at sort "
                    f"{sort!r}; signature has no recursion-terminating op at "
                    f"this sort. Increase `depth` or add a nullary / data-"
                    f"only constructor."
                )
            cands = terminating
        return cands

    def _is_recursion_terminating(self, op: str) -> bool:
        sig = self.signature
        if op in sig.constructors:
            domain = sig.constructors[op].domain
        elif op in sig.binders:
            domain = sig.binders[op].domain()
        else:
            return False
        for s in domain:
            sd = sig.sorts.get(s)
            if sd is None:
                raise RuntimeError(
                    f"decoder {self.name!r}: op {op!r} mentions undeclared sort {s!r}"
                )
            if sd.kind == "object":
                return False
        return True

    def _structure_logits(
        self,
        vec: torch.Tensor,
        sort: str,
        candidates: list[str],
    ) -> torch.Tensor:
        fn = self.structure_fns.get(sort) or self.structure_fns.get("*")
        if fn is None:
            raise RuntimeError(
                f"decoder {self.name!r}: no structure logit producer for sort {sort!r}"
            )
        logits = fn(vec)
        if logits.shape[-1] < len(candidates):
            raise RuntimeError(
                f"decoder {self.name!r}: structure head for {sort!r} produces "
                f"{logits.shape[-1]} logits but the candidate set has "
                f"{len(candidates)}"
            )
        return logits[..., : len(candidates)]

    def _factor(
        self,
        vec: torch.Tensor,
        op: str,
        sort: str,
        n: int,
    ) -> tuple[torch.Tensor, ...]:
        per_sort = self.factor_fns.get(sort) or self.factor_fns.get("*")
        if per_sort is None:
            raise RuntimeError(
                f"decoder {self.name!r}: no factor function for sort {sort!r}"
            )
        fn = per_sort.get(n)
        if fn is None:
            raise RuntimeError(
                f"decoder {self.name!r}: no factor function for sort {sort!r} "
                f"at arity {n} (required by op {op!r})"
            )
        result = fn(vec)
        if not isinstance(result, tuple) or len(result) != n:
            raise RuntimeError(
                f"decoder {self.name!r}: factor for sort {sort!r} arity {n} "
                f"returned {type(result).__name__} of length "
                f"{len(result) if isinstance(result, (tuple, list)) else '?'}"
            )
        return result

    # -----------------------------------------------------------------
    # log_prob
    # -----------------------------------------------------------------

    def log_prob(
        self,
        term: Term,
        vec: torch.Tensor,
        ctx: Context | None = None,
        sort: str | None = None,
    ) -> torch.Tensor:
        """Score an observed term under the decoder's distribution.

        Parameters
        ----------
        term : Term
            The observed term.
        vec : torch.Tensor
            The embedding the term is decoded from.
        ctx : Context or None
            The scope of in-scope bound variables; `EMPTY_CONTEXT`
            when omitted.
        sort : str or None
            The sort of ``term``; the first declared object sort when
            omitted.

        Returns
        -------
        torch.Tensor
            The scalar log-probability of ``term``.

        Raises
        ------
        TypeError
            If ``term`` is not a `Term`, or an argument does not match
            the kind of its sort.
        RuntimeError
            If the term's shape is outside the decoder's support, for
            instance an operation not in the choice set of its sort or
            a data value outside the closed vocabulary.
        """
        if not isinstance(term, Term):
            raise TypeError(
                f"decoder {self.name!r}: log_prob expects a Term, got "
                f"{type(term).__name__}"
            )
        return self._logp_object(
            term,
            vec,
            ctx or EMPTY_CONTEXT,
            self._resolve_sort(sort),
            self.depth,
        )

    def _logp_object(
        self,
        term: Term,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
        budget: int,
    ) -> torch.Tensor:
        cands = self._object_candidates(ctx, sort, budget)
        if term.op == BOUND_VAR_OP:
            if BOUND_VAR_OP not in cands:
                raise RuntimeError(
                    f"decoder {self.name!r}: BoundVar at sort {sort!r} but no "
                    f"in-scope variable of that sort"
                )
            structure_lp = torch.log_softmax(
                self._structure_logits(vec, sort, cands),
                dim=-1,
            )[cands.index(BOUND_VAR_OP)]
            if len(term.args) != 1 or not isinstance(term.args[0], int):
                raise RuntimeError(
                    f"decoder {self.name!r}: malformed BoundVar args {term.args!r}"
                )
            var_lp = self._logp_index(vec, ctx, sort, term.args[0])
            return structure_lp + var_lp

        if term.op not in cands:
            raise RuntimeError(
                f"decoder {self.name!r}: op {term.op!r} not in candidate set "
                f"at sort {sort!r}; the term may violate the depth budget or "
                f"reference an unknown op"
            )
        structure_lp = torch.log_softmax(
            self._structure_logits(vec, sort, cands),
            dim=-1,
        )[cands.index(term.op)]

        sig = self.signature
        if term.op in sig.constructors:
            cons = sig.constructors[term.op]
            if len(term.args) != cons.arity:
                raise RuntimeError(
                    f"decoder {self.name!r}: op {term.op!r} expects "
                    f"{cons.arity} args, got {len(term.args)}"
                )
            if cons.arity == 0:
                return structure_lp
            sub_vecs = self._factor(vec, term.op, cons.codomain, cons.arity)
            total = structure_lp
            for i, child_sort in enumerate(cons.domain):
                total = total + self._logp_child(
                    term.args[i],
                    sub_vecs[i],
                    ctx,
                    child_sort,
                    budget - 1,
                )
            return total

        b = sig.binders[term.op]
        if len(term.args) != b.arity:
            raise RuntimeError(
                f"decoder {self.name!r}: binder {term.op!r} expects "
                f"{b.arity} args, got {len(term.args)}"
            )
        sub_vecs = self._factor(vec, term.op, b.codomain, b.arity)
        total = structure_lp
        # Score annotation children in outer ctx, then build extended
        # ctx (so type/var info lines up with the encoder's path),
        # then score scoped children.
        annot_terms: list[Term | None] = []
        annot_vecs: list[torch.Tensor | None] = []
        ai = 0
        for spec in b.binds:
            if spec.annot_sort is not None:
                child = term.args[ai]
                total = total + self._logp_child(
                    child,
                    sub_vecs[ai],
                    ctx,
                    spec.annot_sort,
                    budget - 1,
                )
                annot_terms.append(child if isinstance(child, Term) else None)
                annot_vecs.append(sub_vecs[ai])
                ai += 1
            else:
                annot_terms.append(None)
                annot_vecs.append(None)
        ctx_ext = ctx
        for spec, ann_term, ann_vec in zip(b.binds, annot_terms, annot_vecs):
            slot_vec = (
                ann_vec
                if ann_vec is not None
                else torch.zeros(self.sort_dims[spec.sort])
            )
            ctx_ext = ctx_ext.push(spec.sort, slot_vec, ann_term)
        n_annots = ai
        for j, spec in enumerate(b.scoped):
            idx = n_annots + j
            total = total + self._logp_child(
                term.args[idx],
                sub_vecs[idx],
                ctx_ext,
                spec.sort,
                budget - 1,
            )
        return total

    def _logp_child(
        self,
        child: TermArg,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
        budget: int,
    ) -> torch.Tensor:
        sd = self.signature.sorts.get(sort)
        if sd is None:
            raise RuntimeError(f"decoder {self.name!r}: unknown sort {sort!r}")
        if sd.kind == "data":
            if isinstance(child, Term):
                raise TypeError(
                    f"decoder {self.name!r}: data-sorted arg at sort "
                    f"{sort!r} must be a raw Python value, got Term"
                )
            return self._logp_data(child, vec, sort)
        if sd.kind == "index":
            if not isinstance(child, int):
                raise TypeError(
                    f"decoder {self.name!r}: index-sorted arg at sort "
                    f"{sort!r} must be an int, got {type(child).__name__}"
                )
            return self._logp_index(vec, ctx, sort, child)
        if not isinstance(child, Term):
            raise TypeError(
                f"decoder {self.name!r}: object-sorted arg at sort {sort!r} "
                f"must be a Term, got {type(child).__name__}"
            )
        return self._logp_object(child, vec, ctx, sort, budget)

    def _logp_data(
        self,
        token: DataLeaf,
        vec: torch.Tensor,
        sort: str,
    ) -> torch.Tensor:
        vocab = self.data_vocab.get(sort)
        if not vocab:
            raise RuntimeError(
                f"decoder {self.name!r}: data sort {sort!r} has no vocabulary"
            )
        if token not in vocab:
            raise RuntimeError(
                f"decoder {self.name!r}: token {token!r} not in vocabulary "
                f"for data sort {sort!r}"
            )
        fn = self.primitive_fns.get(sort) or self.primitive_fns.get("*")
        if fn is None:
            raise RuntimeError(
                f"decoder {self.name!r}: no primitive head for data sort {sort!r}"
            )
        logits = fn(vec)
        if logits.shape[-1] < len(vocab):
            raise RuntimeError(
                f"decoder {self.name!r}: primitive head for {sort!r} produces "
                f"{logits.shape[-1]} logits but the vocabulary has "
                f"{len(vocab)} tokens"
            )
        return torch.log_softmax(logits[..., : len(vocab)], dim=-1)[vocab.index(token)]

    def _logp_index(
        self,
        vec: torch.Tensor,
        ctx: Context,
        sort: str,
        index: int,
    ) -> torch.Tensor:
        in_scope = ctx.by_sort(sort)
        if not in_scope:
            raise RuntimeError(
                f"decoder {self.name!r}: no in-scope variable of sort {sort!r}"
            )
        choices = [i for i, _ in in_scope]
        if index not in choices:
            raise RuntimeError(
                f"decoder {self.name!r}: index {index} not in scope at sort "
                f"{sort!r}; in-scope indices: {choices}"
            )
        embeds = [e.embedding for _, e in in_scope]
        logits = self.binder_select_fn(vec, embeds).reshape(-1)
        if logits.shape[0] != len(choices):
            raise RuntimeError(
                f"decoder {self.name!r}: binder_select produced "
                f"{logits.shape[0]} logits for {len(choices)} in-scope "
                f"variables"
            )
        return torch.log_softmax(logits, dim=-1)[choices.index(index)]


def decoder_draw_message(name: str) -> str:
    """The refusal of a program ``sample`` or ``observe`` from a decoder.

    A decoder's values are structured terms, not the tensors a program's
    sites hold, so no program step draws from one; a loss scores a term
    under it instead.

    Parameters
    ----------
    name : str
        The decoder's name.

    Returns
    -------
    str
        The message.
    """
    return (
        f"{name!r} is a decoder, whose values are terms of its signature, and "
        "a program step draws only tensor values; score a term under it in a "
        f"loss, as in `loss l [on=decoder({name})]` with body "
        f"`-{name}.log_prob(term, code)`, or sample terms in Python with "
        f"`{name}.sample(code)`"
    )


__all__ = [
    "Decoder",
]
