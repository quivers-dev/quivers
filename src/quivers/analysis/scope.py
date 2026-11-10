"""Scope-path resolution for compiled QVR modules.

Every meta-command in the REPL that takes a binding name (``:info``,
``:type``, ``:doc``, ``:browse``, ``:where``, ...) routes through
the resolver in this module. The grammar of a scope path is::

    path     ::= name ('::' name)*
    name     ::= a valid Python-style identifier

The leftmost segment is looked up against the module's top-level
declarations (objects / spaces / morphisms / rules / programs /
deductions / signatures / encoders / decoders / losses / bundles /
contractions / categories / compositions). Each subsequent segment
is looked up against the previous binding's *scope view*, a
``Mapping[str, ScopedRef]`` derived from its AST shape:

* A ``program lda(α, β) : ...`` exposes its typed parameters and
  every step (``sample`` / ``observe`` / ``let`` / ``marginalize`` /
  ``score`` / ``return``) by the names they bind.
* A ``marginalize z : T <- F [over=G]`` exposes its inner body
  steps the same way, so ``lda::z::w`` works the same as
  ``lda::w`` would at top level.
* A ``deduction CCG`` exposes ``rules`` and ``atoms`` as
  sub-bindings; each rule under ``rules`` exposes ``premises`` and
  ``conclusion`` for further drill-down.
* A ``signature LF`` exposes ``sorts`` / ``constructors`` /
  ``binders`` / ``vertex_kinds`` / ``edge_kinds`` as sub-maps.
* ``encoder`` / ``decoder`` / ``bundle`` / inline-bodied
  ``composition`` declarations expose the named entries from their
  bodies (``op_rules``, member rule names, ``tensor_op`` / ``join``
  / ``unit`` / ``zero``).

Container kinds that do not introduce names (``rule`` premise
patterns, ``let`` value expressions, leaf morphisms) return an
empty scope; the resolver stops walking at them.

The resolver is purely structural. It does *not* recompile the
program; it reads the already-elaborated AST nodes from the
``Compiler`` instance the REPL hands it. Consumers that need
runtime values (a sample site's actual tensor, a deduction rule's
weight) should look those up via the runtime accessors after
locating the binding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.dsl.ast_nodes import (
    BundleDecl,
    LetStep,
    MarginalizeStep,
    ObserveStep,
    ReturnStep,
    SampleStep,
)

ScopeKind = Literal[
    "object",
    "space",
    "morphism",
    "rule",
    "program",
    "deduction",
    "signature",
    "encoder",
    "decoder",
    "loss",
    "bundle",
    "contraction",
    "composition",
    "category",
    # Sub-scopes
    "param",  # program parameter
    "sample-site",
    "observe-site",
    "let-site",
    "marginalize-site",
    "score-site",
    "return-site",
    "atom",
    "deduction-rule",
    "lexicon-entry",
    "sort",
    "constructor",
    "binder",
    "vertex-kind",
    "edge-kind",
    "op-rule",
    "init-rule",
    "message-rule",
    "update-rule",
    "var-init",
    "decoder-head",
    "composition-entry",
    "bundle-member",
]
"""The kind tag carried by every `ScopedRef`, used for pretty-printing,
click-target classification, and the ``:browse`` section grouping."""

TOP_LEVEL_KINDS: tuple[ScopeKind, ...] = (
    "object",
    "space",
    "morphism",
    "rule",
    "program",
    "deduction",
    "signature",
    "encoder",
    "decoder",
    "loss",
    "bundle",
    "contraction",
    "composition",
    "category",
)
"""The top-level container kinds, those reachable directly from the
compiler's public accessors, in the order ``ReplSession.browse()``
walks them for output."""


class ScopedRef(ValueModel):
    """One resolved binding.

    Refs compare by value over their names, kinds, and paths, and by
    identity over ``node``: two refs are equal exactly when they
    resolve the same path to the same node object.

    Attributes
    ----------
    name : str
        Last path segment. ``"lda"`` for ``lda``, ``"w"`` for
        ``lda::z::w``.
    kind : ScopeKind
        Categorical tag indicating which declaration / step kind
        this ref points at.
    path : str
        The full ``::``-separated path from the module root.
    parent_kind : ScopeKind | None
        ``None`` for a top-level binding; otherwise the
        enclosing container's kind (useful for context-aware
        rendering: a ``sample-site`` belongs to a ``program`` or a
        ``marginalize-site``).
    """

    name: str
    kind: ScopeKind
    path: str
    parent_kind: ScopeKind | None = None
    # The underlying AST node / runtime object. Stored opaquely, and
    # compared by identity, because the union of all possible node
    # types is wide; callers downcast based on ``kind``.
    node: object = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


# ---------------------------------------------------------------------------
# Top-level lookup
# ---------------------------------------------------------------------------


def _top_level_lookup(compiler, name: str) -> ScopedRef | None:  # type: ignore[no-untyped-def]
    """Find ``name`` in ``compiler``'s public buckets in priority
    order. The order resolves shadowing: a morphism shadows a rule
    of the same name shadows a deduction, and so on. In practice
    QVR forbids same-name collisions across buckets, so the order
    only matters when the same name legitimately appears twice
    (the canonical case is an inline-bodied ``composition`` whose
    name also appears as a referenced algebra)."""
    accessor_map: tuple[tuple[str, ScopeKind], ...] = (
        ("objects", "object"),
        ("spaces", "space"),
        ("morphisms", "morphism"),
        ("rules", "rule"),
        ("programs", "program"),
        ("deductions", "deduction"),
        ("signatures", "signature"),
        ("encoders", "encoder"),
        ("decoders", "decoder"),
        ("losses", "loss"),
        ("bundles", "bundle"),
        ("contractions", "contraction"),
    )
    for attr, kind in accessor_map:
        mapping = getattr(compiler, attr, None) or {}
        if name in mapping:
            return ScopedRef(
                name=name,
                kind=kind,
                path=name,
                parent_kind=None,
                node=mapping[name],
            )
    return None


# ---------------------------------------------------------------------------
# Scope views per container kind
# ---------------------------------------------------------------------------


def _step_kind(step: object) -> ScopeKind | None:
    if isinstance(step, SampleStep):
        return "sample-site"
    if isinstance(step, ObserveStep):
        return "observe-site"
    if isinstance(step, LetStep):
        return "let-site"
    if isinstance(step, MarginalizeStep):
        return "marginalize-site"
    if isinstance(step, ReturnStep):
        return "return-site"
    # ScoreStep is the remaining case; classify by class name to
    # avoid pulling the symbol in for an isinstance check.
    if type(step).__name__ == "ScoreStep":
        return "score-site"
    return None


def _step_bound_name(step: object) -> str | None:
    """Return the name a step binds, or ``None`` for steps that
    bind multiple names or none."""
    if isinstance(step, SampleStep):
        vars_ = getattr(step, "vars", ()) or ()
        return vars_[0] if len(vars_) == 1 else None
    if isinstance(step, ObserveStep):
        return step.vars[0] if len(step.vars) == 1 else None
    if isinstance(step, MarginalizeStep):
        return step.var
    if isinstance(step, LetStep):
        return getattr(step, "name", None)
    if isinstance(step, ReturnStep):
        # The return step doesn't bind a name; we expose it under
        # ``return`` in the scope so users can address it.
        return "return"
    if type(step).__name__ == "ScoreStep":
        return getattr(step, "name", None)
    return None


def _scope_for_program(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """Children of a ``program`` (or a ``marginalize-site`` whose
    AST shape is the same: an indented body of program steps).

    Includes every typed parameter (``param``) and every body step
    keyed by the name it binds.
    """
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    # Type parameters: alpha : Real, beta : Real, etc.
    for p in getattr(decl, "type_params", None) or ():
        pname = getattr(p, "name", None)
        if isinstance(pname, str) and pname:
            out[pname] = ScopedRef(
                name=pname,
                kind="param",
                path=f"{ref.path}::{pname}",
                parent_kind=ref.kind,
                node=p,
            )
    # Bare-name params (the param list shape on programs that
    # don't carry type annotations).
    for n in getattr(decl, "params", None) or ():
        if isinstance(n, str) and n and n not in out:
            out[n] = ScopedRef(
                name=n,
                kind="param",
                path=f"{ref.path}::{n}",
                parent_kind=ref.kind,
                node=n,
            )
    # Body steps.
    steps = getattr(decl, "draws", None) or getattr(decl, "scope", ()) or ()
    for step in steps:
        kind = _step_kind(step)
        if kind is None:
            continue
        bound = _step_bound_name(step)
        if bound is None:
            continue
        out[bound] = ScopedRef(
            name=bound,
            kind=kind,
            path=f"{ref.path}::{bound}",
            parent_kind=ref.kind,
            node=step,
        )
    # ``ProgramDecl.return_vars`` stores the return clause as a
    # plain tuple of names rather than a ``ReturnStep`` AST node.
    # Surface it under ``return`` so the scope path
    # ``PROG::return`` resolves to the program's terminal step.
    # ``MarginalizeStep`` has no return clause, so the attribute
    # access below silently no-ops on inner scopes.
    ret_vars = getattr(decl, "return_vars", ()) or ()
    if ret_vars and "return" not in out:
        out["return"] = ScopedRef(
            name="return",
            kind="return-site",
            path=f"{ref.path}::return",
            parent_kind=ref.kind,
            node=tuple(ret_vars),
        )
    return out


def _scope_for_deduction(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """A ``deduction`` exposes named children for ``rules`` (one
    per declared rule) and ``atoms`` (each atom as its own ref)."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    for rule in getattr(decl, "rules", ()) or ():
        rname = getattr(rule, "name", None)
        if isinstance(rname, str) and rname:
            out[rname] = ScopedRef(
                name=rname,
                kind="deduction-rule",
                path=f"{ref.path}::{rname}",
                parent_kind=ref.kind,
                node=rule,
            )
    for atom in getattr(decl, "atoms", ()) or ():
        if isinstance(atom, str) and atom and atom not in out:
            out[atom] = ScopedRef(
                name=atom,
                kind="atom",
                path=f"{ref.path}::{atom}",
                parent_kind=ref.kind,
                node=atom,
            )
    for entry in getattr(decl, "lexicon", ()) or ():
        word = getattr(entry, "word", None)
        if isinstance(word, str) and word and word not in out:
            out[word] = ScopedRef(
                name=word,
                kind="lexicon-entry",
                path=f"{ref.path}::{word}",
                parent_kind=ref.kind,
                node=entry,
            )
    return out


def _scope_for_signature(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """A ``signature`` exposes ``sorts`` / ``constructors`` /
    ``binders`` / ``vertex_kinds`` / ``edge_kinds`` as named
    sub-maps."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    for src_attr, kind in (
        ("sorts", "sort"),
        ("sorts_t", "sort"),
        ("constructors", "constructor"),
        ("constructors_t", "constructor"),
        ("binders", "binder"),
        ("binders_t", "binder"),
        ("vertex_kinds", "vertex-kind"),
        ("vertex_kinds_t", "vertex-kind"),
        ("edge_kinds", "edge-kind"),
        ("edge_kinds_t", "edge-kind"),
    ):
        entries = getattr(decl, src_attr, None) or ()
        for entry in entries:
            ename = getattr(entry, "name", None)
            if isinstance(ename, str) and ename and ename not in out:
                out[ename] = ScopedRef(
                    name=ename,
                    kind=kind,  # type: ignore[arg-type]
                    path=f"{ref.path}::{ename}",
                    parent_kind=ref.kind,
                    node=entry,
                )
    return out


def _scope_for_encoder(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """An ``encoder`` exposes its named per-op rules + per-init /
    -message / -update rules + var inits. Anonymous bodies (the
    ``readout`` expression) are not exposed; users address them via
    the encoder's ``:info`` output."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    for attr, kind in (
        ("op_rules", "op-rule"),
        ("init_rules", "init-rule"),
        ("message_rules", "message-rule"),
        ("update_rules", "update-rule"),
        ("var_inits", "var-init"),
    ):
        for entry in getattr(decl, attr, ()) or ():
            ename = getattr(entry, "name", None) or getattr(entry, "constructor", None)
            if isinstance(ename, str) and ename and ename not in out:
                out[ename] = ScopedRef(
                    name=ename,
                    kind=kind,  # type: ignore[arg-type]
                    path=f"{ref.path}::{ename}",
                    parent_kind=ref.kind,
                    node=entry,
                )
    return out


def _scope_for_decoder(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """A ``decoder`` exposes the per-constructor heads it
    declares. Anonymous lambda bodies (``structure``, ``primitive``,
    ``factor``, ``binder_select``) are not exposed."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    # Decoders may store per-constructor heads under any of a few
    # attribute names; check the common shapes.
    for attr in ("heads", "primitives", "constructor_heads"):
        entries = getattr(decl, attr, None)
        if entries is None:
            continue
        if isinstance(entries, Mapping):
            for ename, body in entries.items():
                if isinstance(ename, str) and ename and ename not in out:
                    out[ename] = ScopedRef(
                        name=ename,
                        kind="decoder-head",
                        path=f"{ref.path}::{ename}",
                        parent_kind=ref.kind,
                        node=body,
                    )
        else:
            for entry in entries or ():
                ename = getattr(entry, "name", None) or getattr(
                    entry, "constructor", None
                )
                if isinstance(ename, str) and ename and ename not in out:
                    out[ename] = ScopedRef(
                        name=ename,
                        kind="decoder-head",
                        path=f"{ref.path}::{ename}",
                        parent_kind=ref.kind,
                        node=entry,
                    )
    return out


def _scope_for_bundle(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """A ``bundle`` exposes each member rule name. The value of
    each child is the bare member name string; consumers wanting
    the underlying rule should resolve ``BUNDLE::MEMBER`` then
    re-look-up ``MEMBER`` at top level."""
    decl = ref.node
    members: tuple[str, ...] = ()
    if isinstance(decl, BundleDecl):
        members = tuple(getattr(decl, "rules", ()) or ())
    elif isinstance(decl, tuple):
        members = tuple(m for m in decl if isinstance(m, str))
    out: dict[str, ScopedRef] = {}
    for m in members:
        if isinstance(m, str) and m and m not in out:
            out[m] = ScopedRef(
                name=m,
                kind="bundle-member",
                path=f"{ref.path}::{m}",
                parent_kind=ref.kind,
                node=m,
            )
    return out


def _scope_for_composition(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """An inline-bodied ``composition`` exposes its entries
    (``tensor_op``, ``join``, ``unit``, ``zero``). A
    ``composition NAME as algebra`` with no body has empty scope."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    for entry in getattr(decl, "entries", ()) or ():
        ename = getattr(entry, "key", None) or getattr(entry, "name", None)
        if isinstance(ename, str) and ename and ename not in out:
            out[ename] = ScopedRef(
                name=ename,
                kind="composition-entry",
                path=f"{ref.path}::{ename}",
                parent_kind=ref.kind,
                node=entry,
            )
    return out


def _scope_for_contraction(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """A ``contraction NAME(arg1 : A -> B, arg2 : ...)`` exposes
    its declared inputs by parameter name. The compiled record
    stores ``input_types`` either as a dict
    ``name -> ContractionInput`` (from the AST) or as a tuple of
    ``(name, input_dom, input_cod)`` triples (from
    ``_CompiledContraction``)."""
    decl = ref.node
    out: dict[str, ScopedRef] = {}
    input_types = getattr(decl, "input_types", None)
    if isinstance(input_types, Mapping):
        for arg_name, arg_type in input_types.items():
            if isinstance(arg_name, str) and arg_name and arg_name not in out:
                out[arg_name] = ScopedRef(
                    name=arg_name,
                    kind="param",
                    path=f"{ref.path}::{arg_name}",
                    parent_kind=ref.kind,
                    node=arg_type,
                )
    elif isinstance(input_types, (tuple, list)):
        for entry in input_types:
            if not isinstance(entry, tuple) or not entry:
                continue
            arg_name = entry[0]
            if isinstance(arg_name, str) and arg_name and arg_name not in out:
                out[arg_name] = ScopedRef(
                    name=arg_name,
                    kind="param",
                    path=f"{ref.path}::{arg_name}",
                    parent_kind=ref.kind,
                    node=entry,
                )
    return out


# Dispatcher keyed on the ScopedRef.kind.
_SCOPE_DISPATCH: dict[str, object] = {
    "program": _scope_for_program,
    "marginalize-site": _scope_for_program,
    "deduction": _scope_for_deduction,
    "signature": _scope_for_signature,
    "encoder": _scope_for_encoder,
    "decoder": _scope_for_decoder,
    "bundle": _scope_for_bundle,
    "composition": _scope_for_composition,
    "contraction": _scope_for_contraction,
}


def scope_children(ref: ScopedRef) -> Mapping[str, ScopedRef]:
    """Return the named children of ``ref``'s scope, or an empty
    mapping if ``ref.kind`` introduces no further bindings.

    Idempotent + pure; callers can cache. Reads only the AST
    fields exposed on each declaration shape, so adding a new
    declaration kind requires registering one entry in
    ``_SCOPE_DISPATCH``.

    Parameters
    ----------
    ref : ScopedRef
        The binding whose scope is listed.

    Returns
    -------
    Mapping[str, ScopedRef]
        The child bindings, keyed by name.
    """
    builder = _SCOPE_DISPATCH.get(ref.kind)
    if builder is None:
        return {}
    return builder(ref)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


SCOPE_SEPARATOR = "::"
"""The separator between the segments of a scoped path such as
``"prog::x"``."""


def split_path(path: str) -> list[str]:
    """Split ``"a::b::c"`` into ``["a", "b", "c"]``. Leading or
    trailing separators yield empty segments, which the resolver
    rejects.

    Parameters
    ----------
    path : str
        A `SCOPE_SEPARATOR`-separated path.

    Returns
    -------
    list[str]
        The path's segments, in order.
    """
    return path.split(SCOPE_SEPARATOR)


def resolve_scoped_path(compiler, path: str) -> ScopedRef | None:  # type: ignore[no-untyped-def]
    """Resolve a ``::``-separated path against ``compiler``'s
    elaborated module. Returns ``None`` if any segment fails to
    resolve.

    Parameters
    ----------
    compiler
        A ``quivers.dsl.compiler.Compiler`` instance whose
        ``compile_env()`` has run.
    path
        A non-empty ``::``-separated string.

    Returns
    -------
    ScopedRef | None
        The terminal binding, or ``None`` if any segment doesn't
        exist in its parent's scope.
    """
    segments = split_path(path)
    if not segments or any(not s for s in segments):
        return None
    current = _top_level_lookup(compiler, segments[0])
    if current is None:
        return None
    for seg in segments[1:]:
        children = scope_children(current)
        if seg not in children:
            return None
        current = children[seg]
    return current


def find_all_references(compiler, name: str) -> list[ScopedRef]:  # type: ignore[no-untyped-def]
    """Find every binding in ``compiler``'s module whose final
    path segment equals ``name``.

    Returns refs sorted by depth (top-level first) then by path
    lexicographic order. Used by ``:where NAME`` to surface every
    scope a bare name appears in.

    Parameters
    ----------
    compiler : Compiler
        A `quivers.dsl.Compiler` whose module has been compiled.
    name : str
        The bare name to search for.

    Returns
    -------
    list[ScopedRef]
        Every binding whose final path segment is ``name``.
    """
    hits: list[ScopedRef] = []
    # Walk every top-level binding and recursively visit scopes.
    # Bounded by the AST size; QVR programs are small.
    seen: set[str] = set()
    for attr, kind in (
        ("objects", "object"),
        ("spaces", "space"),
        ("morphisms", "morphism"),
        ("rules", "rule"),
        ("programs", "program"),
        ("deductions", "deduction"),
        ("signatures", "signature"),
        ("encoders", "encoder"),
        ("decoders", "decoder"),
        ("losses", "loss"),
        ("bundles", "bundle"),
        ("contractions", "contraction"),
    ):
        mapping = getattr(compiler, attr, None) or {}
        for n, node in mapping.items():
            top = ScopedRef(name=n, kind=kind, path=n, parent_kind=None, node=node)
            _collect_named(top, name, hits, seen)
    hits.sort(key=lambda r: (r.path.count(SCOPE_SEPARATOR), r.path))
    return hits


def _collect_named(  # type: ignore[no-untyped-def]
    ref: ScopedRef, target: str, hits: list[ScopedRef], seen: set[str]
) -> None:
    if ref.path in seen:
        return
    seen.add(ref.path)
    if ref.name == target:
        hits.append(ref)
    for child in scope_children(ref).values():
        _collect_named(child, target, hits, seen)


__all__ = [
    "SCOPE_SEPARATOR",
    "ScopedRef",
    "ScopeKind",
    "TOP_LEVEL_KINDS",
    "find_all_references",
    "resolve_scoped_path",
    "scope_children",
    "split_path",
]
