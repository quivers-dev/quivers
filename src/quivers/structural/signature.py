"""Signatures: typed multi-sorted algebras with binders.

A `Signature` is the runtime form of a ``signature { … }`` DSL
declaration. It carries the sort table, constructor table, binder
table, and (for graph-shaped signatures) vertex / edge kind tables.

Terms over a signature are first-class Python values
(`Term` instances): each carries the constructor / binder
name and its positional arguments. The framework's encoder and
decoder runtimes walk these records uniformly.

The de-Bruijn discipline is enforced structurally: a ``BoundVar``
term carries an integer index; binders push a fresh entry onto an
implicit context Γ tracked by the encoder / decoder runtime.

All structured value types in this module are didactic
`didactic.api.Model` records, matching the quivers
convention for schema-bearing data. Recursive term arguments and scope
entries use opaque value fields to preserve nested storage. Runtime
embeddings use opaque identity fields to preserve tensors and autograd.
`quivers._values.ValueModel` includes these fields in equality, hashing,
copying, and pickling.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel

type DataLeaf = str | int | float | bytes | bool
"""A raw value at a data-sorted argument position of a `Term`.

Values outside this union are rejected at runtime with a typed
error rather than coerced.
"""

type TermArg = Term | DataLeaf | int
"""A positional argument of a `Term`: a subterm at an object
position, a `DataLeaf` at a data position, or an ``int`` at an index
position."""

type SortKind = Literal["object", "index", "data"]
"""The kinds of sort a `Signature` can declare."""


# ---------------------------------------------------------------------------
# Sort, constructor, binder, edge specs (all schema-bearing)
# ---------------------------------------------------------------------------


class SortVocabEntry(dx.Model):
    """One closed-vocabulary entry on a data sort.

    Each entry is a tagged Python value: ``kind`` is the leaf
    type (``"string" | "integer" | "float"``), ``value`` is the
    decoded Python value (``str``, ``int``, or ``float``).
    """

    kind: Literal["string", "integer", "float"]
    value: DataLeaf


class Sort(dx.Model):
    """A sort declaration: name, kind, optional dim, optional
    closed vocabulary.

    A data sort may carry a ``vocab`` tuple of
    `SortVocabEntry` records. Object / index sorts must
    have an empty vocab (the framework checks at runtime).
    """

    name: str
    kind: SortKind
    dim: int | None = None
    vocab: tuple[SortVocabEntry, ...] = ()

    @property
    def vocab_values(self) -> tuple[DataLeaf, ...]:
        """The decoded Python values in declaration order."""
        return tuple(e.value for e in self.vocab)


class Constructor(dx.Model):
    """A typed constructor symbol."""

    name: str
    domain: tuple[str, ...] = ()
    codomain: str = ""

    @property
    def arity(self) -> int:
        """The number of arguments the constructor takes."""
        return len(self.domain)


class BinderVarSpec(dx.Model):
    """A scoped variable introduced by a binder.

    ``sort`` is the variable's own sort. ``annot_sort`` is the sort
    of an optional *type annotation*, a sibling argument in the
    enclosing Term that travels alongside the variable's embedding
    in Γ. Used to track per-variable type information.
    """

    var: str
    sort: str
    annot_sort: str | None = None


class BinderArgSpec(dx.Model):
    """A scoped argument of a binder."""

    arg: str
    sort: str


class Binder(dx.Model):
    """A binder constructor: introduces variables of given sorts into
    the scope of given arguments."""

    name: str
    binds: tuple[BinderVarSpec, ...] = ()
    scoped: tuple[BinderArgSpec, ...] = ()
    codomain: str = ""

    @property
    def arity(self) -> int:
        """The number of positional arguments of the binder in a `Term`.

        Each annotated bound variable contributes one annotation
        argument, and each scoped argument contributes one more.
        """
        n_annots = sum(1 for b in self.binds if b.annot_sort is not None)
        return n_annots + len(self.scoped)

    def domain(self) -> tuple[str, ...]:
        """Positional sort sequence of the binder's arguments in the
        enclosing `Term`: type-annotations for each annotated
        bound variable (in declaration order, outer-context-evaluated),
        followed by the scoped arguments (extended-context-evaluated).

        Returns
        -------
        tuple[str, ...]
            The sort name of each positional argument.
        """
        return tuple(
            b.annot_sort for b in self.binds if b.annot_sort is not None
        ) + tuple(a.sort for a in self.scoped)


class VertexKind(dx.Model):
    """A vertex kind in a graph-shaped signature."""

    name: str
    kind: SortKind
    dim: int | None = None


class EdgeKind(dx.Model):
    """An edge kind in a graph-shaped signature."""

    name: str
    src: str
    tgt: str
    directed: bool = True


# ---------------------------------------------------------------------------
# Signature (whole-algebra record)
# ---------------------------------------------------------------------------


class Signature(dx.Model):
    """A multi-sorted algebra signature with optional binders / graph
    structure.

    Sort, constructor, binder, vertex_kind and edge_kind tables are
    represented as tuples of records (rather than dicts) so the
    enclosing `dx.Model` keeps a schema-bearing layout. The
    runtime exposes O(1) name-keyed lookup methods (``sort(name)``,
    ``constructor(name)``, etc.) on top of those tuples.
    """

    name: str
    params: tuple[str, ...] = ()
    sorts_t: tuple[Sort, ...] = ()
    constructors_t: tuple[Constructor, ...] = ()
    binders_t: tuple[Binder, ...] = ()
    vertex_kinds_t: tuple[VertexKind, ...] = ()
    edge_kinds_t: tuple[EdgeKind, ...] = ()

    # ---- dict-like lookups ----

    @property
    def sorts(self) -> dict[str, Sort]:
        """The sort declarations keyed by sort name."""
        return {s.name: s for s in self.sorts_t}

    @property
    def constructors(self) -> dict[str, Constructor]:
        """The constructor declarations keyed by constructor name."""
        return {c.name: c for c in self.constructors_t}

    @property
    def binders(self) -> dict[str, Binder]:
        """The binder declarations keyed by binder name."""
        return {b.name: b for b in self.binders_t}

    @property
    def vertex_kinds(self) -> dict[str, VertexKind]:
        """The graph vertex kinds keyed by kind name."""
        return {v.name: v for v in self.vertex_kinds_t}

    @property
    def edge_kinds(self) -> dict[str, EdgeKind]:
        """The graph edge kinds keyed by kind name."""
        return {e.name: e for e in self.edge_kinds_t}

    # ---- shape queries ----

    def is_inductive(self) -> bool:
        """Whether the signature declares any constructor or binder.

        Returns
        -------
        bool
            ``True`` when terms over the signature are inductive.
        """
        return bool(self.constructors_t) or bool(self.binders_t)

    def is_graph(self) -> bool:
        """Whether the signature declares both vertex and edge kinds.

        Returns
        -------
        bool
            ``True`` when the signature is graph-shaped.
        """
        return bool(self.vertex_kinds_t) and bool(self.edge_kinds_t)

    def all_ops(self) -> Iterable[str]:
        """Iterate over every operation name of the signature.

        Yields
        ------
        str
            Each constructor name in declaration order, then each
            binder name in declaration order.
        """
        for c in self.constructors_t:
            yield c.name
        for b in self.binders_t:
            yield b.name

    def codomain_of(self, op: str) -> str:
        """Return the codomain sort of a constructor or binder.

        Parameters
        ----------
        op : str
            The constructor or binder name.

        Returns
        -------
        str
            The name of the sort the operation produces.

        Raises
        ------
        KeyError
            If ``op`` is neither a constructor nor a binder.
        """
        for c in self.constructors_t:
            if c.name == op:
                return c.codomain
        for b in self.binders_t:
            if b.name == op:
                return b.codomain
        raise KeyError(
            f"signature {self.name!r}: op {op!r} not a constructor or binder"
        )

    def domain_of(self, op: str) -> tuple[str, ...]:
        """Return the positional argument sorts of a constructor or binder.

        Parameters
        ----------
        op : str
            The constructor or binder name.

        Returns
        -------
        tuple[str, ...]
            The sort of each positional argument; for a binder, as
            given by `Binder.domain`.

        Raises
        ------
        KeyError
            If ``op`` is neither a constructor nor a binder.
        """
        for c in self.constructors_t:
            if c.name == op:
                return c.domain
        for b in self.binders_t:
            if b.name == op:
                return b.domain()
        raise KeyError(
            f"signature {self.name!r}: op {op!r} not a constructor or binder"
        )

    def is_binder(self, op: str) -> bool:
        """Whether an operation name is a binder of the signature.

        Parameters
        ----------
        op : str
            The operation name.

        Returns
        -------
        bool
            ``True`` when a binder of that name is declared.
        """
        return any(b.name == op for b in self.binders_t)

    def sort_dim(self, sort: str) -> int | None:
        """Return the declared embedding dimension of a sort or vertex kind.

        Parameters
        ----------
        sort : str
            The sort or vertex-kind name.

        Returns
        -------
        int or None
            The declared dimension, or ``None`` when the name is not
            declared or declares no dimension.
        """
        for s in self.sorts_t:
            if s.name == sort:
                return s.dim
        for v in self.vertex_kinds_t:
            if v.name == sort:
                return v.dim
        return None


# ---------------------------------------------------------------------------
# Term representation
# ---------------------------------------------------------------------------


_DATA_LEAF_TYPES: tuple[type, ...] = (str, int, float, bytes, bool)
"""The Python types admitted as non-`Term` arguments of a `Term`."""


class Term(ValueModel):
    """A closed term over a signature.

    ``op`` is the constructor name, the binder name, or the
    framework-reserved ``"BoundVar"`` for a de-Bruijn variable
    reference.

    ``args`` are positional children. Object positions carry a
    `Term`; data positions carry a `DataLeaf` raw
    value; index positions carry a non-negative ``int``. ``args``
    is held opaque (its element type is the open union
    `TermArg`, which didactic cannot encode because one of its arms
    is `Term` itself); construction checks that ``args`` is a tuple
    whose elements are `Term` or `DataLeaf` values, and the encoder /
    decoder enforce sort agreement at use time.

    ``args`` compares by value, so equality and hashing are
    structural (see `quivers._values.ValueModel`): two terms are equal
    exactly when their operators agree and their arguments agree
    position by position, recursively, so terms that differ in a data
    leaf, a nested subterm, a binder, or a de-Bruijn index are
    distinct. Data leaves compare by type as well as value, so ``1``,
    ``1.0``, and ``True`` at the same position are three different
    terms. Both operations walk the term with an explicit stack, so
    deep terms (long ``Cons`` lists) do not exhaust the interpreter's
    recursion limit; a term's hash is computed once and cached, and
    copies and pickles keep ``args``.

    No wrapping such as ``Term("Data", …)`` is used at any
    position; raw values appear directly at data positions.
    """

    op: str
    args: tuple = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _check_args(self) -> None:
        """Reject arguments outside the `TermArg` union.

        Raises
        ------
        TypeError
            If ``args`` is not a tuple, or if one of its elements is
            neither a `Term` nor a `DataLeaf`.
        """
        if not isinstance(self.args, tuple):
            raise TypeError(
                f"Term {self.op!r}: args must be a tuple, got {type(self.args).__name__}"
            )
        for position, arg in enumerate(self.args):
            if not isinstance(arg, Term) and not isinstance(arg, _DATA_LEAF_TYPES):
                raise TypeError(
                    f"Term {self.op!r}: argument {position} must be a Term or a "
                    f"data leaf (str, int, float, bytes, bool), got "
                    f"{type(arg).__name__}"
                )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if not self.args:
            return self.op
        inside = ", ".join(repr(a) for a in self.args)
        return f"{self.op}({inside})"

    def to_tuple(self) -> tuple:
        """A serialisable form matching the agenda's item-tuple
        convention: ``(op, *args_serialised)``. Children that are
        `Term`s are recursively serialised; data leaves and
        indices pass through.

        Returns
        -------
        tuple
            The nested tuple ``(op, *args_serialised)``.
        """
        out: list = [self.op]
        for a in self.args:
            if isinstance(a, Term):
                out.append(a.to_tuple())
            else:
                out.append(a)
        return tuple(out)


def bound_var(index: int) -> Term:
    """A de-Bruijn variable reference.

    Parameters
    ----------
    index : int
        The de-Bruijn index: 0 refers to the innermost bound variable.

    Returns
    -------
    Term
        The term ``BoundVar(index)``.

    Raises
    ------
    TypeError
        If ``index`` is not a non-negative ``int``.
    """
    if not isinstance(index, int) or index < 0:
        raise TypeError(f"bound_var requires a non-negative int, got {index!r}")
    return Term(op="BoundVar", args=(index,))


def make_term(op: str, *args) -> Term:
    """Construct a term over a signature.

    Each ``arg`` must be a `Term`, a `DataLeaf` raw
    value, or a non-negative int (for index-sorted positions). The
    encoder / decoder runtime validate sort agreement at use
    time.

    Parameters
    ----------
    op : str
        The constructor or binder name.
    *args : TermArg
        The positional children.

    Returns
    -------
    Term
        The term ``op(*args)``.

    Raises
    ------
    TypeError
        If ``op`` is not a string.
    didactic.api.ValidationError
        If an argument is neither a `Term` nor a `DataLeaf`.
    """
    if not isinstance(op, str):
        raise TypeError(f"make_term requires a string op, got {type(op).__name__}")
    return Term(op=op, args=tuple(args))


# ---------------------------------------------------------------------------
# de-Bruijn context
# ---------------------------------------------------------------------------


class ContextEntry(ValueModel):
    """One scope entry on the binder stack.

    ``embedding`` is a runtime tensor; ``type_term`` is the binder's
    annotation term captured at scope-extension time. Tensor identity
    participates in equality and hashing, preserving live autograd state.
    """

    var_sort: str
    embedding: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    type_term: Term | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _check_runtime_fields(self) -> None:
        """Require a tensor embedding and an optional annotation term.

        Raises
        ------
        TypeError
            If an opaque field has an incompatible runtime type.
        """
        if not isinstance(self.embedding, torch.Tensor):
            raise TypeError("ContextEntry.embedding must be a torch.Tensor")
        if self.type_term is not None and not isinstance(self.type_term, Term):
            raise TypeError("ContextEntry.type_term must be a Term or None")


class Context(ValueModel):
    """The de-Bruijn scope context threaded through encoder and
    decoder runtimes.

    Position 0 is the most recently bound variable; lookup ``var(i)``
    returns the i-th entry from the top.
    """

    entries: tuple[ContextEntry, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _check_entries(self) -> None:
        """Require an immutable tuple of scope entries.

        Raises
        ------
        TypeError
            If entries are not a tuple of ContextEntry models.
        """
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, ContextEntry) for entry in self.entries
        ):
            raise TypeError("Context.entries must be a tuple of ContextEntry models")

    def push(
        self,
        var_sort: str,
        embedding: torch.Tensor,
        type_term: Term | None = None,
    ) -> "Context":
        """Return the context extended by one innermost bound variable.

        Parameters
        ----------
        var_sort : str
            The sort of the new variable.
        embedding : torch.Tensor
            The variable's embedding.
        type_term : Term or None
            The binder's annotation term for the variable, if any.

        Returns
        -------
        Context
            A new context whose index 0 is the new variable; ``self``
            is unchanged.
        """
        return Context(
            entries=(
                ContextEntry(
                    var_sort=var_sort,
                    embedding=embedding,
                    type_term=type_term,
                ),
            )
            + self.entries,
        )

    def var(self, index: int) -> torch.Tensor:
        """Return the embedding of a bound variable.

        Parameters
        ----------
        index : int
            The de-Bruijn index, counted from the innermost variable.

        Returns
        -------
        torch.Tensor
            The variable's embedding.

        Raises
        ------
        IndexError
            If ``index`` is negative or not below the context depth.
        """
        if index < 0 or index >= len(self.entries):
            raise IndexError(
                f"Context.var({index}): out of range "
                f"(context depth {len(self.entries)})"
            )
        return self.entries[index].embedding

    def type_of(self, index: int) -> Term | None:
        """Return the annotation term of a bound variable.

        Parameters
        ----------
        index : int
            The de-Bruijn index, counted from the innermost variable.

        Returns
        -------
        Term or None
            The annotation captured when the variable was bound, or
            ``None`` when it carries none.
        """
        return self.entries[index].type_term

    def depth(self) -> int:
        """Return the number of variables in scope.

        Returns
        -------
        int
            The context depth.
        """
        return len(self.entries)

    def by_sort(self, sort: str) -> list[tuple[int, ContextEntry]]:
        """All entries whose ``var_sort`` matches; returned as
        ``(depth-index, entry)`` pairs for the decoder's
        categorical-over-in-scope-variables.

        Parameters
        ----------
        sort : str
            The variable sort to select.

        Returns
        -------
        list[tuple[int, ContextEntry]]
            The matching entries with their de-Bruijn indices, innermost
            first.
        """
        return [(i, e) for i, e in enumerate(self.entries) if e.var_sort == sort]


EMPTY_CONTEXT = Context()
"""The context with no bound variables."""


__all__ = [
    "DataLeaf",
    "TermArg",
    "SortKind",
    "SortVocabEntry",
    "Sort",
    "Constructor",
    "BinderVarSpec",
    "BinderArgSpec",
    "Binder",
    "VertexKind",
    "EdgeKind",
    "Signature",
    "Term",
    "bound_var",
    "make_term",
    "ContextEntry",
    "Context",
    "EMPTY_CONTEXT",
]
