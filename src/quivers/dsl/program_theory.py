"""Program Theory: a panproto protocol for the resolved (post-compilation) DSL.

The `QVR_PROGRAM_PROTOCOL` protocol describes the static structure of
a compiled `.qvr` program as a panproto `panproto.Schema`: every object,
space, morphism, and output the program declares, together with the
checked computation and effect graph the QIEC elaboration produced for
it. This sits one layer above the syntactic ``qvr`` protocol compiled
from the in-tree QVR grammar: the syntactic protocol carries the AST as
parsed, while this protocol carries the AST after the resolution layer
(``_resolve_type``, ``_resolve_space``) and the kernel check have run.

Why have it
-----------
Once two programs share this protocol, panproto's structural diff,
auto-lens-generation, and breaking-change detection apply to compiled
programs as a whole. Two `.qvr` files that compile to structurally
equivalent programs produce equal Schemas; two that diverge surface
their divergence through `panproto.diff_schemas`.

Vertex kinds
------------

Top-level container:
    ``program``
        the root vertex; one per compiled module.

Discrete objects (mirrors [`quivers.core.objects`][quivers.core.objects]):
    ``finset`` ``product_set`` ``coproduct_set`` ``free_monoid`` ``empty_set``

Continuous spaces (mirrors [`quivers.continuous.spaces`][quivers.continuous.spaces]):
    ``euclidean`` ``simplex`` ``positive_reals`` ``product_space``

Top-level declarations:
    ``object_decl`` ``space_decl`` ``morphism_decl``
    ``kernel_decl`` ``discretize_decl`` ``embed_decl`` ``output_decl``

Kernel declarations (mirrors [`quivers.qiec.module`][quivers.qiec.module]):
    ``effect_decl`` ``operation`` ``instance_decl`` ``handler_decl``
    ``handler_clause`` ``handler_parameter`` ``computation_decl``
    ``computation_parameter`` ``program_entry`` ``program_parameter``
    ``program_site``

Edge kinds
----------

``decl``
    ``program -> *_decl``: each declaration, source or kernel, is a child
    of the root program.
``binds_to``
    ``object_decl -> set_object`` / ``space_decl -> space``.
``component``
    ``product_set | coproduct_set -> set_object``;
    ``product_space -> space``: structural recursion into composite types.
``domain`` / ``codomain``
    ``morphism_decl | kernel_decl | discretize_decl | embed_decl ->
    set_object | space``.
``output``
    ``program -> output_decl``.
``operation``
    ``effect_decl -> operation``; ``handler_clause -> operation``: the
    operations an interface declares and the one a clause answers.
``instance_of``
    ``instance_decl -> effect_decl``: the interface an instance applies.
``handles``
    ``handler_decl -> effect_decl``.
``clause``
    ``handler_decl -> handler_clause``.
``parameter``
    ``handler_decl -> handler_parameter``;
    ``computation_decl -> computation_parameter``;
    ``program_entry -> program_parameter``.
``row``
    ``computation_decl -> instance_decl``: the instances a computation's
    declared row names.
``calls``
    ``computation_decl | handler_clause -> computation_decl``: the
    computations a body calls.
``performs``
    ``computation_decl | handler_clause -> instance_decl``: the instances
    a body performs requests on.
``installs``
    ``computation_decl | handler_clause -> handler_decl``: the handlers a
    body installs.
``entry``
    ``program -> program_entry``.
``body``
    ``program_entry -> computation_decl``.
``site``
    ``program_entry -> program_site``.
``random`` / ``score``
    ``program_entry -> instance_decl``: the lexical instances the entry's
    samples and scores address.

Constraint sorts carry the per-vertex scalar metadata: ``name``,
``cardinality``, ``dim``, ``low``, ``high``, ``family``, ``modality``,
``morphism_kind``, ``replicate``, ``n_bins``, and for kernel vertices
``identity``, ``type``, ``effects``, ``role``, ``site_kind``, ``grade``,
``implementation``, ``total``, ``batch``, ``event``, and ``gap``.
"""

from __future__ import annotations

from collections.abc import Iterator
import panproto

from quivers.continuous.boundaries import Discretize, Embed
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)
from quivers.categorical.monoidal import EmptySet
from quivers.core.objects import (
    CoproductSet,
    FinSet,
    EnumSet,
    FreeMonoid,
    FreeResiduated,
    ProductSet,
    SetObject,
)
from quivers.dsl.ast_nodes import ExprIdent, ExprIdentity
from quivers.dsl.compiler import Compiler
from quivers.qiec.effects import EffectRow, HandlerDef
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    Computation,
    Handle,
    If,
    NewInstance,
    Perform,
    PlateAxis,
)
from quivers.qiec.types import render_static
from quivers.stochastic.morphisms import StochasticMorphism


# ---------------------------------------------------------------------------
# protocol definition
# ---------------------------------------------------------------------------

_KERNEL_DECL_KINDS = [
    "effect_decl",
    "instance_decl",
    "handler_decl",
    "computation_decl",
]

_KERNEL_KINDS = [
    *_KERNEL_DECL_KINDS,
    "operation",
    "handler_clause",
    "handler_parameter",
    "computation_parameter",
    "program_entry",
    "program_parameter",
    "program_site",
]

_BODY_KINDS = ["computation_decl", "handler_clause"]

_OBJECT_KINDS = [
    "program",
    # discrete
    "finset",
    "product_set",
    "coproduct_set",
    "free_monoid",
    "empty_set",
    "enum_set",
    "free_residuated",
    # continuous
    "euclidean",
    "simplex",
    "positive_reals",
    "product_space",
    # declarations
    "object_decl",
    "space_decl",
    "morphism_decl",
    "kernel_decl",
    "discretize_decl",
    "embed_decl",
    "output_decl",
    "schema_decl",
    # kernel
    *_KERNEL_KINDS,
]

_SET_OBJECT_KINDS = [
    "finset",
    "product_set",
    "coproduct_set",
    "free_monoid",
    "empty_set",
    "enum_set",
    "free_residuated",
]
_SPACE_KINDS = ["euclidean", "simplex", "positive_reals", "product_space"]
_DOMAIN_KINDS = _SET_OBJECT_KINDS + _SPACE_KINDS
_DECL_KINDS = [
    "object_decl",
    "space_decl",
    "morphism_decl",
    "kernel_decl",
    "discretize_decl",
    "embed_decl",
    "output_decl",
]

_MORPHISM_DECL_KINDS = [
    "morphism_decl",
    "kernel_decl",
    "discretize_decl",
    "embed_decl",
]

_EDGE_RULES = [
    {
        "edge_kind": "decl",
        "src_kinds": ["program"],
        "tgt_kinds": _DECL_KINDS + _KERNEL_DECL_KINDS,
    },
    # ``binds_to`` covers both object_decl to set_object and space_decl to
    # space:
    # panproto requires a unique edge_kind label, so the two cases share one
    # rule whose src/tgt kinds are unioned. The Schema-level validator then
    # accepts any valid pairing across the union.
    {
        "edge_kind": "binds_to",
        "src_kinds": ["object_decl", "space_decl"],
        "tgt_kinds": _SET_OBJECT_KINDS + _SPACE_KINDS,
    },
    # ``component`` covers ProductSet/CoproductSet → set-object children
    # and ProductSpace → space|set-object children (mixed-domain products
    # land here too).
    {
        "edge_kind": "component",
        "src_kinds": ["product_set", "coproduct_set", "product_space"],
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {
        "edge_kind": "generators",
        "src_kinds": ["free_monoid", "free_residuated"],
        "tgt_kinds": ["finset", "enum_set"],
    },
    {
        "edge_kind": "domain",
        "src_kinds": _MORPHISM_DECL_KINDS,
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {
        "edge_kind": "codomain",
        "src_kinds": _MORPHISM_DECL_KINDS,
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {"edge_kind": "output", "src_kinds": ["program"], "tgt_kinds": ["output_decl"]},
    {
        "edge_kind": "operation",
        "src_kinds": ["effect_decl", "handler_clause"],
        "tgt_kinds": ["operation"],
    },
    {
        "edge_kind": "instance_of",
        "src_kinds": ["instance_decl"],
        "tgt_kinds": ["effect_decl"],
    },
    {
        "edge_kind": "handles",
        "src_kinds": ["handler_decl"],
        "tgt_kinds": ["effect_decl"],
    },
    {
        "edge_kind": "clause",
        "src_kinds": ["handler_decl"],
        "tgt_kinds": ["handler_clause"],
    },
    {
        "edge_kind": "parameter",
        "src_kinds": ["handler_decl", "computation_decl", "program_entry"],
        "tgt_kinds": [
            "handler_parameter",
            "computation_parameter",
            "program_parameter",
        ],
    },
    {
        "edge_kind": "row",
        "src_kinds": ["computation_decl"],
        "tgt_kinds": ["instance_decl"],
    },
    {
        "edge_kind": "calls",
        "src_kinds": _BODY_KINDS,
        "tgt_kinds": ["computation_decl"],
    },
    {
        "edge_kind": "performs",
        "src_kinds": _BODY_KINDS,
        "tgt_kinds": ["instance_decl"],
    },
    {
        "edge_kind": "installs",
        "src_kinds": _BODY_KINDS,
        "tgt_kinds": ["handler_decl"],
    },
    {"edge_kind": "entry", "src_kinds": ["program"], "tgt_kinds": ["program_entry"]},
    {
        "edge_kind": "body",
        "src_kinds": ["program_entry"],
        "tgt_kinds": ["computation_decl"],
    },
    {
        "edge_kind": "site",
        "src_kinds": ["program_entry"],
        "tgt_kinds": ["program_site"],
    },
    {
        "edge_kind": "random",
        "src_kinds": ["program_entry"],
        "tgt_kinds": ["instance_decl"],
    },
    {
        "edge_kind": "score",
        "src_kinds": ["program_entry"],
        "tgt_kinds": ["instance_decl"],
    },
]

_CONSTRAINT_SORTS = [
    "value_constraint",
    "name",
    "cardinality",
    "max_length",
    "dim",
    "low",
    "high",
    "family",
    "modality",
    "morphism_kind",
    "replicate",
    "n_bins",
    "algebra",
    "element",
    "op",
    "depth",
    "identity",
    "type",
    "effects",
    "role",
    "site_kind",
    "grade",
    "implementation",
    "total",
    "batch",
    "event",
    "axes",
    "gap",
]

QVR_PROGRAM_PROTOCOL: panproto.Protocol = panproto.define_protocol(
    {
        "name": "qvr_program",
        # Reuse the brat schema/instance theories: both shape graphs with
        # vertices, edges, and constraint metadata; the kinds and rules above
        # are what specialize them to the compiled-quivers shape.
        "schema_theory": "ThBratSchema",
        "instance_theory": "ThBratInstance",
        "edge_rules": _EDGE_RULES,
        "obj_kinds": _OBJECT_KINDS,
        "constraint_sorts": _CONSTRAINT_SORTS,
    }
)
"""Panproto protocol for a compiled QVR program.

`extract_program_schema` writes schemas in this protocol. The vertex
kinds, edge rules, and constraint sorts are the ones the module
docstring lists.
"""


# ---------------------------------------------------------------------------
# extraction: compiled program -> panproto.Schema
# ---------------------------------------------------------------------------


class _SchemaWriter:
    """Helper that emits set-object / space subgraphs into a SchemaBuilder.

    Each emitted SetObject / ContinuousSpace instance gets its own vertex,
    keyed in the cache by Python object identity. Structural deduplication
    (one vertex per ``__eq__``-equivalent value) would collapse the
    components of e.g. ``ProductSet(components=(N, N))`` into a single
    target, and ``component`` edges between the same source and target
    can't repeat under panproto's edge-set semantics, so we keep distinct
    occurrences distinct.

    Parameters
    ----------
    builder : panproto.SchemaBuilder
        The builder the vertices and edges are written into.
    """

    def __init__(self, builder: panproto.SchemaBuilder) -> None:
        self._builder = builder
        self._object_ids: dict[int, str] = {}
        self._counter = 0

    def _fresh(self, prefix: str) -> str:
        """Return a vertex id no earlier vertex of the writer has.

        Parameters
        ----------
        prefix : str
            The vertex kind the id is for.

        Returns
        -------
        str
            The prefix with the writer's next counter value.
        """
        self._counter += 1
        return f"{prefix}_{self._counter}"

    # -- discrete -------------------------------------------------------

    def write_set_object(self, obj: SetObject) -> str:
        """Emit vertices and constraints for a set object.

        Parameters
        ----------
        obj : SetObject
            The object to write; a composite writes its components too.

        Returns
        -------
        str
            The root vertex id, which repeats for the same object.

        Raises
        ------
        TypeError
            If the object is of a variant the protocol has no kind for.
        """
        cached = self._object_ids.get(id(obj))
        if cached is not None:
            return cached
        if isinstance(obj, FinSet):
            vid = self._fresh("finset")
            self._builder.vertex(vid, "finset")
            self._builder.constraint(vid, "name", obj.name)
            self._builder.constraint(vid, "cardinality", str(obj.cardinality))
        elif isinstance(obj, ProductSet):
            vid = self._fresh("product_set")
            self._builder.vertex(vid, "product_set")
            for child in obj.components:
                cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        elif isinstance(obj, CoproductSet):
            vid = self._fresh("coproduct_set")
            self._builder.vertex(vid, "coproduct_set")
            for child in obj.components:
                cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        elif isinstance(obj, FreeMonoid):
            vid = self._fresh("free_monoid")
            self._builder.vertex(vid, "free_monoid")
            self._builder.constraint(vid, "max_length", str(obj.max_length))
            gvid = self.write_set_object(obj.generators)
            self._builder.edge(vid, gvid, "generators")
        elif isinstance(obj, EmptySet):
            vid = self._fresh("empty_set")
            self._builder.vertex(vid, "empty_set")
        elif isinstance(obj, EnumSet):
            vid = self._fresh("enum_set")
            self._builder.vertex(vid, "enum_set")
            self._builder.constraint(vid, "name", obj.name)
            for elem in obj.elements:
                self._builder.constraint(vid, "element", elem)
        elif isinstance(obj, FreeResiduated):
            vid = self._fresh("free_residuated")
            self._builder.vertex(vid, "free_residuated")
            self._builder.constraint(vid, "depth", str(obj.depth))
            for op in obj.ops:
                self._builder.constraint(vid, "op", op)
            gvid = self.write_set_object(obj.generators)
            self._builder.edge(vid, gvid, "generators")
        else:
            raise TypeError(f"unsupported SetObject variant: {type(obj).__name__}")
        self._object_ids[id(obj)] = vid
        return vid

    # -- continuous -----------------------------------------------------

    def write_space(self, space: ContinuousSpace) -> str:
        """Emit vertices and constraints for a continuous space.

        Parameters
        ----------
        space : ContinuousSpace
            The space to write; a product writes its components too.

        Returns
        -------
        str
            The root vertex id, which repeats for the same space.

        Raises
        ------
        TypeError
            If the space is of a variant the protocol has no kind for.
        """
        cached = self._object_ids.get(id(space))
        if cached is not None:
            return cached
        if isinstance(space, Euclidean):
            vid = self._fresh("euclidean")
            self._builder.vertex(vid, "euclidean")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
            if space.low is not None:
                self._builder.constraint(vid, "low", str(space.low))
            if space.high is not None:
                self._builder.constraint(vid, "high", str(space.high))
        elif isinstance(space, Simplex):
            vid = self._fresh("simplex")
            self._builder.vertex(vid, "simplex")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
        elif isinstance(space, PositiveReals):
            vid = self._fresh("positive_reals")
            self._builder.vertex(vid, "positive_reals")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
        elif isinstance(space, ProductSpace):
            vid = self._fresh("product_space")
            self._builder.vertex(vid, "product_space")
            for child in space.components:
                if isinstance(child, ContinuousSpace):
                    cvid = self.write_space(child)
                else:
                    cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        else:
            raise TypeError(
                f"unsupported ContinuousSpace variant: {type(space).__name__}"
            )
        self._object_ids[id(space)] = vid
        return vid

    def write_any(self, target: object) -> str:
        """Emit a morphism's domain or codomain, whichever kind it is.

        Parameters
        ----------
        target : object
            A set object or a continuous space.

        Returns
        -------
        str
            The root vertex id.

        Raises
        ------
        TypeError
            If the target is neither.
        """
        if isinstance(target, ContinuousSpace):
            return self.write_space(target)
        if isinstance(target, SetObject):
            return self.write_set_object(target)
        raise TypeError(f"unsupported domain/codomain: {type(target).__name__}")


def _subcomputations(computation: Computation) -> Iterator[Computation]:
    """Yield a computation and every computation nested in it.

    Parameters
    ----------
    computation : Computation
        The root.

    Yields
    ------
    Computation
        The root first, then its parts in evaluation order. Values are
        not entered: a value never holds a computation.
    """
    yield computation
    if isinstance(computation, Bind):
        yield from _subcomputations(computation.first)
        yield from _subcomputations(computation.then)
    elif isinstance(computation, Handle):
        yield from _subcomputations(computation.computation)
    elif isinstance(computation, Case):
        for branch in computation.branches:
            yield from _subcomputations(branch.body)
    elif isinstance(computation, If):
        yield from _subcomputations(computation.then)
        yield from _subcomputations(computation.otherwise)
    elif isinstance(computation, NewInstance):
        yield from _subcomputations(computation.body)


def _row_text(row: EffectRow) -> str:
    """Render an effect row for a constraint.

    Parameters
    ----------
    row : EffectRow
        The row.

    Returns
    -------
    str
        The instance names of the entries in order, with the tail
        variable after a ``|`` when the row is open.
    """
    entries = ", ".join(entry.instance.digest[:12] for entry in row.entries)
    if row.tail is None:
        return f"{{{entries}}}"
    return f"{{{entries} | {row.tail.name}}}"


def _axes_text(axes: tuple[PlateAxis, ...]) -> str:
    """Render plate axes for a constraint.

    Parameters
    ----------
    axes : tuple[PlateAxis, ...]
        The axes.

    Returns
    -------
    str
        The axis names with their extents, comma separated.
    """
    return ", ".join(f"{axis.name}:{render_static(axis.size)}" for axis in axes)


class _KernelWriter:
    """Emit a checked kernel module's declaration graph into a builder.

    Every kernel declaration becomes one vertex keyed by its stable
    identity, so the same computation called from two places is one
    target, and a body's calls, requests, and handler installations
    become edges from the vertex holding the body.

    Parameters
    ----------
    builder : panproto.SchemaBuilder
        The builder the vertices and edges are written into.
    module : QiecModule
        The checked module.
    """

    def __init__(self, builder: panproto.SchemaBuilder, module: QiecModule) -> None:
        self._builder = builder
        self._module = module
        self._effects: dict[str, str] = {}
        self._operations: dict[str, str] = {}
        self._instances: dict[str, str] = {}
        self._handlers: dict[str, str] = {}
        self._computations: dict[str, str] = {}

    def write(self) -> None:
        """Write the module's effects, instances, handlers, computations, and entries."""
        if self._module.gap is not None:
            self._builder.constraint("program", "gap", self._module.gap)
        for effect in self._module.effects:
            vid = f"effect_decl::{effect.ref.name}"
            self._builder.vertex(vid, "effect_decl")
            self._builder.constraint(vid, "name", effect.ref.name)
            self._builder.constraint(vid, "identity", effect.ref.id.digest)
            self._builder.edge("program", vid, "decl")
            self._effects[effect.ref.id.digest] = vid
            for operation in effect.operations:
                ovid = f"{vid}/operation::{operation.name}"
                self._builder.vertex(ovid, "operation")
                self._builder.constraint(ovid, "name", operation.name)
                self._builder.constraint(ovid, "identity", operation.id.digest)
                self._builder.constraint(
                    ovid, "type", render_static(operation.result_type)
                )
                self._builder.edge(vid, ovid, "operation")
                self._operations[operation.id.digest] = ovid
        for instance in self._module.instances:
            vid = f"instance_decl::{instance.name}"
            self._builder.vertex(vid, "instance_decl")
            self._builder.constraint(vid, "name", instance.name)
            self._builder.constraint(vid, "identity", instance.entry.instance.digest)
            self._builder.constraint(vid, "type", render_static(instance.entry.effect))
            self._builder.edge("program", vid, "decl")
            self._instances[instance.entry.instance.digest] = vid
            effect_vid = self._effects.get(instance.entry.effect.id.digest)
            if effect_vid is not None:
                self._builder.edge(vid, effect_vid, "instance_of")
        for handler in self._module.handlers:
            vid = f"handler_decl::{handler.name}"
            self._builder.vertex(vid, "handler_decl")
            self._builder.constraint(vid, "name", handler.name)
            self._builder.constraint(vid, "identity", handler.id.digest)
            self._builder.constraint(vid, "implementation", handler.implementation)
            self._builder.constraint(vid, "total", str(handler.total).lower())
            self._builder.constraint(
                vid,
                "type",
                f"{render_static(handler.input_type)} => "
                f"{render_static(handler.output_type)}",
            )
            self._builder.constraint(vid, "effects", _row_text(handler.introduced))
            self._builder.edge("program", vid, "decl")
            self._handlers[handler.id.digest] = vid
            effect_vid = self._effects.get(handler.effect.id.digest)
            if effect_vid is not None:
                self._builder.edge(vid, effect_vid, "handles")
        for computation in self._module.computations:
            vid = f"computation_decl::{computation.name}"
            self._builder.vertex(vid, "computation_decl")
            self._builder.constraint(vid, "name", computation.name)
            self._builder.constraint(vid, "identity", computation.id.digest)
            self._builder.constraint(
                vid, "type", render_static(computation.type.result)
            )
            self._builder.constraint(
                vid, "effects", _row_text(computation.type.effects)
            )
            self._builder.edge("program", vid, "decl")
            self._computations[computation.id.digest] = vid
        for handler in self._module.handlers:
            self._write_clauses(handler)
        for computation in self._module.computations:
            self._write_computation(computation)
        for entry in self._module.entries:
            self._write_entry(entry)

    def _write_clauses(self, handler: HandlerDef) -> None:
        """Write a handler's parameters, its clauses, and their bodies' edges.

        Parameters
        ----------
        handler : HandlerDef
            The handler.
        """
        vid = self._handlers[handler.id.digest]
        for parameter in handler.parameters:
            pvid = f"{vid}/parameter::{parameter.name}"
            self._builder.vertex(pvid, "handler_parameter")
            self._builder.constraint(pvid, "name", parameter.name)
            self._builder.constraint(pvid, "type", render_static(parameter.type))
            self._builder.edge(vid, pvid, "parameter")
        for clause in handler.clauses:
            operation_vid = self._operations.get(clause.operation.digest)
            name = (
                operation_vid.rsplit("::", 1)[1]
                if operation_vid is not None
                else clause.operation.digest[:12]
            )
            cvid = f"{vid}/clause::{name}"
            self._builder.vertex(cvid, "handler_clause")
            self._builder.constraint(cvid, "name", name)
            self._builder.constraint(cvid, "grade", clause.grade.value)
            self._builder.edge(vid, cvid, "clause")
            if operation_vid is not None:
                self._builder.edge(cvid, operation_vid, "operation")
            if clause.body is not None:
                self._write_body_edges(cvid, clause.body)
        if handler.return_clause is not None:
            cvid = f"{vid}/clause::return"
            self._builder.vertex(cvid, "handler_clause")
            self._builder.constraint(cvid, "name", "return")
            self._builder.edge(vid, cvid, "clause")
            self._write_body_edges(cvid, handler.return_clause.body)

    def _write_computation(self, computation: NamedComputation) -> None:
        """Write a computation's parameters, row, and body edges.

        Parameters
        ----------
        computation : NamedComputation
            The computation.
        """
        vid = self._computations[computation.id.digest]
        for parameter in computation.parameters:
            pvid = f"{vid}/parameter::{parameter.name}"
            self._builder.vertex(pvid, "computation_parameter")
            self._builder.constraint(pvid, "name", parameter.name)
            self._builder.constraint(pvid, "type", render_static(parameter.type))
            self._builder.edge(vid, pvid, "parameter")
        for entry in computation.type.effects.entries:
            instance_vid = self._instances.get(entry.instance.digest)
            if instance_vid is not None:
                self._builder.edge(vid, instance_vid, "row")
        self._write_body_edges(vid, computation.body)

    def _write_body_edges(self, source: str, body: Computation) -> None:
        """Write the calls, requests, and installations a body makes.

        Parameters
        ----------
        source : str
            The vertex holding the body.
        body : Computation
            The body.
        """
        calls: set[str] = set()
        performs: set[str] = set()
        installs: set[str] = set()
        for part in _subcomputations(body):
            if isinstance(part, Call):
                target = self._computations.get(part.callee.digest)
                if target is not None:
                    calls.add(target)
            elif isinstance(part, Perform):
                target = self._instances.get(part.request.instance.digest)
                if target is not None:
                    performs.add(target)
            elif isinstance(part, Handle):
                target = self._handlers.get(part.handler.digest)
                if target is not None:
                    installs.add(target)
        for target in sorted(calls):
            self._builder.edge(source, target, "calls")
        for target in sorted(performs):
            self._builder.edge(source, target, "performs")
        for target in sorted(installs):
            self._builder.edge(source, target, "installs")

    def _write_entry(self, entry: ProgramEntry) -> None:
        """Write a program entry point with its parameters and sites.

        Parameters
        ----------
        entry : ProgramEntry
            The entry.
        """
        vid = f"program_entry::{entry.name}"
        self._builder.vertex(vid, "program_entry")
        self._builder.constraint(vid, "name", entry.name)
        self._builder.constraint(vid, "identity", entry.computation.digest)
        self._builder.edge("program", vid, "entry")
        body = self._computations.get(entry.computation.digest)
        if body is not None:
            self._builder.edge(vid, body, "body")
        for parameter in entry.parameters:
            pvid = f"{vid}/parameter::{parameter.name}"
            self._builder.vertex(pvid, "program_parameter")
            self._builder.constraint(pvid, "name", parameter.name)
            self._builder.constraint(pvid, "role", parameter.role)
            self._builder.constraint(pvid, "type", render_static(parameter.type))
            if parameter.constraint is not None:
                self._builder.constraint(pvid, "value_constraint", parameter.constraint)
            if parameter.axes:
                self._builder.constraint(pvid, "axes", ", ".join(parameter.axes))
            self._builder.edge(vid, pvid, "parameter")
        for site in entry.sites:
            svid = f"{vid}/site::{site.name}"
            self._builder.vertex(svid, "program_site")
            self._builder.constraint(svid, "name", site.name)
            self._builder.constraint(svid, "site_kind", site.kind)
            self._builder.constraint(svid, "family", site.family)
            if site.batch:
                self._builder.constraint(svid, "batch", _axes_text(site.batch))
            if site.event:
                self._builder.constraint(svid, "event", _axes_text(site.event))
            self._builder.edge(vid, svid, "site")
        random = self._instances.get(entry.random_instance.digest)
        if random is not None:
            self._builder.edge(vid, random, "random")
        score = self._instances.get(entry.score_instance.digest)
        if score is not None:
            self._builder.edge(vid, score, "score")


# ---------------------------------------------------------------------------
# the public extractor
# ---------------------------------------------------------------------------


def extract_program_schema(compiler: Compiler) -> panproto.Schema:
    """Produce a `panproto.Schema` for a compiled program.

    Walks the compiler's resolved environment (objects, spaces, morphisms)
    and its checked kernel module (effects, instances, handlers,
    computations, and program entry points) and emits a graph of
    vertices and edges in the `QVR_PROGRAM_PROTOCOL` protocol. The
    returned schema validates against that protocol and is suitable for
    `panproto.diff_schemas`, `panproto.auto_generate_lens`, and the rest
    of panproto's schema-level operations.

    Parameters
    ----------
    compiler : Compiler
        A [`quivers.dsl.compiler.Compiler`][quivers.dsl.compiler.Compiler] after `compile_env`
        (or `compile`) has populated the resolved environments.

    Returns
    -------
    panproto.Schema
        A program-level Schema in the ``qvr_program`` protocol.
    """
    builder = QVR_PROGRAM_PROTOCOL.schema()
    writer = _SchemaWriter(builder)

    builder.vertex("program", "program")
    if compiler._algebra is not None:
        builder.constraint("program", "algebra", type(compiler._algebra).__name__)

    # object decls
    for name, obj in compiler._objects.items():
        decl_vid = f"object_decl::{name}"
        builder.vertex(decl_vid, "object_decl")
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")
        target_vid = writer.write_set_object(obj)
        builder.edge(decl_vid, target_vid, "binds_to")

    # space decls
    for name, space in compiler._spaces.items():
        decl_vid = f"space_decl::{name}"
        builder.vertex(decl_vid, "space_decl")
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")
        target_vid = writer.write_space(space)
        builder.edge(decl_vid, target_vid, "binds_to")

    # morphism decls: the compiler's _morphisms env holds named primitive
    # morphisms; we record them as morphism_decl vertices with domain/codomain.
    # Composite morphisms (let-bindings, output) are derived rather than
    # recorded directly; the structural Diff-able layer is the named decls.
    for name, morphism in compiler._morphisms.items():
        kind = _classify_morphism_kind(morphism)

        decl_vid = f"{kind}::{name}"
        builder.vertex(decl_vid, kind)
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")

        dom = getattr(morphism, "domain", None)
        cod = getattr(morphism, "codomain", None)
        if dom is not None:
            dom_vid = writer.write_any(dom)
            builder.edge(decl_vid, dom_vid, "domain")
        if cod is not None:
            cod_vid = writer.write_any(cod)
            builder.edge(decl_vid, cod_vid, "codomain")

    # output decl: the compiler's `_output_expr` holds the AST expression
    # whose compilation produces the program's root morphism. We record an
    # output_decl vertex carrying the expression's source-text form (when
    # available) as a name constraint; the structural diff cares about
    # presence/absence of an output, not its detailed shape.
    if compiler._output_expr is not None:
        out_vid = "output_decl"
        builder.vertex(out_vid, "output_decl")
        # ExprIdent / ExprIdentity carry a single name; composite expressions
        # don't have a single canonical name, so mark them as "<composite>".
        expr = compiler._output_expr
        if isinstance(expr, ExprIdent):
            label = expr.name
        elif isinstance(expr, ExprIdentity):
            label = f"identity({expr.object_name})"
        else:
            label = "<composite>"
        builder.constraint(out_vid, "name", label)
        builder.edge("program", out_vid, "output")

    if compiler.qiec_module is not None:
        _KernelWriter(builder, compiler.qiec_module).write()

    return builder.build()


def _classify_morphism_kind(morphism: object) -> str:
    """Classify a runtime morphism into the program-theory vertex kind.

    The compiler's ``_morphisms`` env holds primitive morphisms produced
    by every `MorphismDecl` lowering (one per role: latent,
    observed, kernel, embed, discretize, let). Classification routes
    through ``isinstance`` rather than module/class name string-matching
    so the boundaries are explicit.

    Parameters
    ----------
    morphism : object
        The compiled morphism.

    Returns
    -------
    str
        The vertex kind: ``discretize_decl``, ``embed_decl``,
        ``kernel_decl``, or ``morphism_decl``.
    """
    if isinstance(morphism, Discretize):
        return "discretize_decl"
    if isinstance(morphism, Embed):
        return "embed_decl"
    if isinstance(morphism, StochasticMorphism):
        return "kernel_decl"
    if isinstance(morphism, ContinuousMorphism):
        return "kernel_decl"
    return "morphism_decl"


# ---------------------------------------------------------------------------
# Deduction-system protocol
# ---------------------------------------------------------------------------


_DEDUCTION_OBJECT_KINDS = [
    "deduction_system",
    "deduction_rule",
    "deduction_atom",
    "deduction_premise",
    "deduction_conclusion",
]

_DEDUCTION_EDGE_RULES = [
    {
        "edge_kind": "decl",
        "src_kinds": ["deduction_system"],
        "tgt_kinds": ["deduction_rule"],
    },
    {
        "edge_kind": "atom",
        "src_kinds": ["deduction_system"],
        "tgt_kinds": ["deduction_atom"],
    },
    {
        "edge_kind": "premise",
        "src_kinds": ["deduction_rule"],
        "tgt_kinds": ["deduction_premise"],
    },
    {
        "edge_kind": "conclusion",
        "src_kinds": ["deduction_rule"],
        "tgt_kinds": ["deduction_conclusion"],
    },
]

_DEDUCTION_CONSTRAINT_SORTS = [
    "name",
    "semiring",
    "start",
    "depth",
    "pattern",
    "identity",
    "weight",
    "judgment",
    "law",
]


QVR_DEDUCTION_PROTOCOL: panproto.Protocol = panproto.define_protocol(
    {
        "name": "qvr_deduction",
        "schema_theory": "ThBratSchema",
        "instance_theory": "ThBratInstance",
        "edge_rules": _DEDUCTION_EDGE_RULES,
        "obj_kinds": _DEDUCTION_OBJECT_KINDS,
        "constraint_sorts": _DEDUCTION_CONSTRAINT_SORTS,
    }
)
"""Panproto protocol for a weighted deductive system.

Vertex kinds:
    * ``deduction_system``: the top-level declaration.
    * ``deduction_atom``: an atom of the item algebra.
    * ``deduction_rule``: a named sequent-style inference rule.
    * ``deduction_premise``: one premise pattern of a rule.
    * ``deduction_conclusion``: a rule's conclusion pattern.

Edges:
    * ``decl``: system :math:`\\to` rule.
    * ``atom``: system :math:`\\to` atom.
    * ``premise``: rule :math:`\\to` premise.
    * ``conclusion``: rule :math:`\\to` conclusion.

Constraint sorts carry the system's semiring and the status of each
law the semiring claims, its start and depth, each pattern's textual
form, and, from the rule presentation the rules are derived from, each
rule's identity and weight parameter and each pattern's judgment. Schema morphisms over this protocol
correspond to specializations of deduction systems
(e.g., :math:`\\mathsf{CCG} \\subset \\mathsf{Lambek} \\subset \\mathsf{MultimodalLambek}`).
"""


def extract_deduction_schema(compiler: Compiler) -> panproto.Schema:
    """Produce a `panproto.Schema` for the compiler's
    deduction-system environment.

    Walks the compiler's ``_deductions`` registry and emits one
    panproto vertex per deduction system, one per rule, one per
    atom, and per-premise / per-conclusion pattern vertices. The
    returned schema validates against
    `QVR_DEDUCTION_PROTOCOL` and is suitable for
    `panproto.diff_schemas` and
    `panproto.auto_generate_lens` operations over deduction
    systems.

    Parameters
    ----------
    compiler : Compiler
        A compiler whose ``_deductions`` registry is populated.

    Returns
    -------
    panproto.Schema
        A schema in the ``qvr_deduction`` protocol.
    """
    builder = QVR_DEDUCTION_PROTOCOL.schema()
    deductions = getattr(compiler, "_deductions", {})
    for name, system in deductions.items():
        sys_vid = f"deduction:{name}"
        builder.vertex(sys_vid, "deduction_system")
        builder.constraint(sys_vid, "name", name)
        builder.constraint(sys_vid, "semiring", system.semiring.__class__.__name__)
        for claim in system.semiring.claims():
            builder.constraint(sys_vid, "law", f"{claim.law.name}:{claim.status.value}")
        presentation = system.presentation
        presented = presentation.rules if presentation is not None else ()
        for rule_idx, rule in enumerate(system.rules):
            rule_vid = f"{sys_vid}/rule:{rule.name}"
            builder.vertex(rule_vid, "deduction_rule")
            builder.constraint(rule_vid, "name", rule.name)
            source = presented[rule_idx] if presented else None
            if source is not None:
                builder.constraint(rule_vid, "identity", str(source.id))
                if source.weight is not None:
                    builder.constraint(rule_vid, "weight", str(source.weight))
            builder.edge(sys_vid, rule_vid, "decl")
            for prem_idx, premise in enumerate(rule.premises):
                p_vid = f"{rule_vid}/premise:{prem_idx}"
                builder.vertex(p_vid, "deduction_premise")
                builder.constraint(p_vid, "pattern", repr(premise))
                if source is not None:
                    builder.constraint(
                        p_vid,
                        "judgment",
                        repr(source.premises[prem_idx].judgment),
                    )
                builder.edge(rule_vid, p_vid, "premise")
            conc_vid = f"{rule_vid}/conclusion"
            builder.vertex(conc_vid, "deduction_conclusion")
            builder.constraint(conc_vid, "pattern", repr(rule.conclusion))
            if source is not None:
                builder.constraint(conc_vid, "judgment", repr(source.conclusion))
            builder.edge(rule_vid, conc_vid, "conclusion")
    return builder.build()


__all__ = [
    "QVR_DEDUCTION_PROTOCOL",
    "QVR_PROGRAM_PROTOCOL",
    "extract_deduction_schema",
    "extract_program_schema",
]
