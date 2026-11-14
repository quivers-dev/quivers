"""Typed, target-independent IR for the Quivers Indexed Effect Core.

The QIEC kernel uses immutable recursive models for checking and evaluation.
This module projects them into fully encoded structural transpiler IR. Every stable QIEC
construct has a corresponding :mod:`didactic` model, so a backend can inspect
effects, values, evidence, resumptions, and source provenance without parsing
an opaque JSON side channel.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, cast

import didactic.api as dx

from quivers.qiec import QiecModule
from quivers.qiec._models import QiecModel
from quivers.qiec import declarations as d
from quivers.qiec import effects as e
from quivers.qiec import evidence as ev
from quivers.qiec import identifiers as ids
from quivers.qiec import kinds as k
from quivers.qiec import module as m
from quivers.qiec import programs as pr
from quivers.qiec import terms as tm
from quivers.qiec import types as ty
from quivers.qiec.identifiers import SourceOrigin, StableId
from quivers.qiec.primitives import PRIMITIVES


type IRScalar = None | bool | int | float | str
type IRPathPart = str | int


class IRQiecId(dx.Model):
    """A stable QIEC identifier, split into namespace and digest."""

    namespace: str
    digest: str

    @property
    def text(self) -> str:
        """The identifier's ``qiec:<namespace>:<digest>`` spelling."""
        return f"qiec:{self.namespace}:{self.digest}"


class IRQiecSourceOrigin(dx.Model):
    """Mirror of a kernel source origin: a site's structural location.

    Line and column are diagnostic coordinates; the structural path alone
    locates the site.

    Parameters
    ----------
    module
        The module the site belongs to.
    structural_path
        The path of names and positions from the module root to the site.
    role
        What the site is, such as ``"sample"`` or ``"call"``.
    source_protocol
        The source language or protocol the path is expressed in.
    file
        The file the site was read from, if any.
    line
        The one-based line of the site, if known.
    column
        The one-based column of the site, if known.
    """

    module: str
    structural_path: tuple[IRPathPart, ...]
    role: str
    source_protocol: str
    file: str | None = None
    line: int | None = None
    column: int | None = None


class IRQiecDynamicAddressFrame(dx.Model):
    """One component of a dynamic effect address.

    Parameters
    ----------
    scope
        What kind of frame this is, such as a call, a handler clause, a
        resumption, or a local instance.
    key
        The frame's position within its scope: a name or an ordinal.
    """

    scope: str
    key: str | int


class IRQiecSiteProvenance(dx.Model):
    """Static and dynamic provenance of an effect request.

    Parameters
    ----------
    origin
        The source site the request was written at.
    dynamic_path
        The address frames entered to reach this occurrence of the site,
        outermost first.
    resumption_path
        The ordinal of each resumption taken to reach this occurrence.
    relation
        How this request relates to the source site: ``"preserve"`` for the
        request as written, ``"split"`` or ``"duplicate"`` for a request a
        handler derived from it, and ``"eliminate"`` for one it discharged.
    parents
        The static sites a derived request was generated from.
    """

    origin: IRQiecSourceOrigin
    dynamic_path: tuple[IRQiecDynamicAddressFrame, ...] = ()
    resumption_path: tuple[int, ...] = ()
    relation: Literal["preserve", "split", "duplicate", "eliminate"] = "preserve"
    parents: tuple[IRQiecId, ...] = ()


class IRQiecKind(dx.TaggedUnion, discriminator="kind"):
    """Kind in the QIEC static language."""


class IRQiecTypeKind(IRQiecKind):
    """The kind of types.

    Parameters
    ----------
    kind
        The discriminator; always ``"type"``.
    """

    kind: Literal["type"] = "type"


class IRQiecEffectKind(IRQiecKind):
    """The kind of effect interfaces.

    Parameters
    ----------
    kind
        The discriminator; always ``"effect"``.
    """

    kind: Literal["effect"] = "effect"


class IRQiecRowKind(IRQiecKind):
    """The kind of effect rows.

    Parameters
    ----------
    kind
        The discriminator; always ``"row"``.
    """

    kind: Literal["row"] = "row"


class IRQiecArrowKind(IRQiecKind):
    """The kind of a constructor from one kind to another.

    Parameters
    ----------
    domain
        The kind of the argument.
    codomain
        The kind of the result.
    kind
        The discriminator; always ``"arrow"``.
    """

    domain: IRQiecKind
    codomain: IRQiecKind
    kind: Literal["arrow"] = "arrow"


class IRQiecIndexSort(dx.TaggedUnion, discriminator="kind"):
    """Closed index sort."""


class IRQiecNatSort(IRQiecIndexSort):
    """The sort of natural-number indices.

    Parameters
    ----------
    kind
        The discriminator; always ``"nat"``.
    """

    kind: Literal["nat"] = "nat"


class IRQiecShapeSort(IRQiecIndexSort):
    """The sort of shapes, optionally restricted to a fixed rank.

    Parameters
    ----------
    rank
        The number of dimensions, or ``None`` for any rank.
    kind
        The discriminator; always ``"shape"``.
    """

    rank: int | None = None
    kind: Literal["shape"] = "shape"


class IRQiecContextSort(IRQiecIndexSort):
    """The sort of logic-program contexts over a named signature.

    Parameters
    ----------
    signature
        The name of the signature whose contexts inhabit this sort.
    kind
        The discriminator; always ``"context"``.
    """

    signature: str
    kind: Literal["context"] = "context"


class IRQiecUserIndexSort(IRQiecIndexSort):
    """A closed user-defined index datatype.

    Parameters
    ----------
    name
        The sort's name, part of its identity.
    constructors
        The constructor names, part of the identity.
    arities
        The argument count of each constructor, aligned with
        ``constructors``.
    kind
        The discriminator; always ``"user"``.
    """

    name: str
    constructors: tuple[str, ...]
    arities: tuple[int, ...]
    kind: Literal["user"] = "user"


class IRQiecBinder(dx.TaggedUnion, discriminator="kind"):
    """A telescope binder for a static argument.

    The subclasses distinguish type, index, and effect binders by the
    ``kind`` discriminator.

    Parameters
    ----------
    name
        The binder's display name.
    refinable
        Whether case analysis may refine the bound argument; true for a
        family's indices and false for its uniform parameters.
    """

    name: str
    refinable: bool = False


class IRQiecTypeBinder(IRQiecBinder):
    """A telescope binder for a type-kinded static argument.

    Inherits ``name`` and ``refinable`` from
    [`IRQiecBinder`][quivers.transpile.qiec_ir.IRQiecBinder].

    Parameters
    ----------
    type_kind
        The kind of the bound argument.
    kind
        The discriminator; always ``"type"``.
    """

    type_kind: IRQiecKind
    kind: Literal["type"] = "type"


class IRQiecIndexBinder(IRQiecBinder):
    """A telescope binder for an index argument.

    Inherits ``name`` and ``refinable`` from
    [`IRQiecBinder`][quivers.transpile.qiec_ir.IRQiecBinder].

    Parameters
    ----------
    sort
        The closed index sort of the bound argument.
    kind
        The discriminator; always ``"index"``.
    """

    sort: IRQiecIndexSort
    kind: Literal["index"] = "index"


class IRQiecEffectBinder(IRQiecBinder):
    """A telescope binder for an effect argument.

    Inherits ``name`` and ``refinable`` from
    [`IRQiecBinder`][quivers.transpile.qiec_ir.IRQiecBinder].

    Parameters
    ----------
    kind
        The discriminator; always ``"effect"``.
    """

    kind: Literal["effect"] = "effect"


class IRQiecStatic(dx.TaggedUnion, discriminator="kind"):
    """Type, index, or effect argument in the QIEC static language."""


class IRQiecIndexVariable(IRQiecStatic):
    """A variable standing for an index term.

    Parameters
    ----------
    name
        The variable's display name.
    sort
        The closed index sort of the terms it stands for.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    kind
        The discriminator; always ``"index-variable"``.
    """

    name: str
    sort: IRQiecIndexSort
    identity: IRQiecId | None = None
    kind: Literal["index-variable"] = "index-variable"


class IRQiecIndexLiteral(IRQiecStatic):
    """A closed index literal.

    Parameters
    ----------
    value
        A natural number for the nat sort, or a nullary constructor name for
        a user-defined index sort.
    sort
        The sort the literal inhabits.
    kind
        The discriminator; always ``"index-literal"``.
    """

    value: int | str
    sort: IRQiecIndexSort
    kind: Literal["index-literal"] = "index-literal"


class IRQiecIndexConstructor(IRQiecStatic):
    """A fully applied constructor of a user-defined index sort.

    Parameters
    ----------
    name
        The constructor's name, which belongs to ``sort``.
    arguments
        The constructor's index arguments, in order.
    sort
        The sort the constructor builds.
    kind
        The discriminator; always ``"index-constructor"``.
    """

    name: str
    arguments: tuple[IRQiecStatic, ...]
    sort: IRQiecIndexSort
    kind: Literal["index-constructor"] = "index-constructor"


class IRQiecShapeIndex(IRQiecStatic):
    """A shape given by its dimensions.

    Parameters
    ----------
    dimensions
        One index term per dimension, outermost first.
    kind
        The discriminator; always ``"shape-index"``.
    """

    dimensions: tuple[IRQiecStatic, ...]
    kind: Literal["shape-index"] = "shape-index"


class IRQiecTypeVariable(IRQiecStatic):
    """A variable standing for a type-kinded static argument.

    Parameters
    ----------
    name
        The variable's display name.
    type_kind
        The kind of the arguments it stands for.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    kind
        The discriminator; always ``"type-variable"``.
    """

    name: str
    type_kind: IRQiecKind
    identity: IRQiecId | None = None
    kind: Literal["type-variable"] = "type-variable"


class IRQiecTypeConstructor(dx.Model):
    """A fully qualified type constructor and its kinding telescope.

    Parameters
    ----------
    id
        The constructor's stable identity, which alone determines equality.
    name
        The constructor's display name.
    telescope
        The binders the constructor's arguments instantiate, in order.
    """

    id: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...] = ()


class IRQiecTypeApplication(IRQiecStatic):
    """A type constructor applied to static arguments.

    Parameters
    ----------
    constructor
        The constructor applied.
    arguments
        The arguments instantiating the constructor's telescope, in order;
        empty for a nullary constructor.
    kind
        The discriminator; always ``"type-application"``.
    """

    constructor: IRQiecTypeConstructor
    arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["type-application"] = "type-application"


class IRQiecFunctionType(IRQiecStatic):
    """A pure function type.

    An effectful codomain is a
    [`IRQiecComputationType`][quivers.transpile.qiec_ir.IRQiecComputationType],
    which keeps the value and computation strata apart.

    Parameters
    ----------
    parameter
        The argument type.
    result
        The result type.
    kind
        The discriminator; always ``"function-type"``.
    """

    parameter: IRQiecStatic
    result: IRQiecStatic
    kind: Literal["function-type"] = "function-type"


class IRQiecEqualityType(IRQiecStatic):
    """The proposition that two static arguments are equal.

    Parameters
    ----------
    classifier
        The kind or index sort both sides inhabit.
    left
        The left side of the equation.
    right
        The right side of the equation.
    kind
        The discriminator; always ``"equality-type"``.
    """

    classifier: IRQiecKind | IRQiecIndexSort
    left: IRQiecStatic
    right: IRQiecStatic
    kind: Literal["equality-type"] = "equality-type"


class IRQiecEffectVariable(IRQiecStatic):
    """A variable standing for an effect interface application.

    Parameters
    ----------
    name
        The variable's display name.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    kind
        The discriminator; always ``"effect-variable"``.
    """

    name: str
    identity: IRQiecId | None = None
    kind: Literal["effect-variable"] = "effect-variable"


class IRQiecEffectRef(IRQiecStatic):
    """One closed effect interface application.

    Parameters
    ----------
    id
        The stable identity of the interface declaration.
    name
        The interface's display name.
    arguments
        The arguments instantiating the interface's telescope, in order.
    kind
        The discriminator; always ``"effect-ref"``.
    """

    id: IRQiecId
    name: str
    arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["effect-ref"] = "effect-ref"


class IRQiecLocal(dx.Model):
    """A typed binder for a runtime value.

    Parameters
    ----------
    name
        The binder's display name.
    type
        The type of every value bound to it.
    """

    name: str
    type: IRQiecStatic


class IRQiecRowVariable(dx.Model):
    """The open tail of an effect row, with the instances it excludes.

    Parameters
    ----------
    name
        The display name, used in diagnostics only.
    identity
        The stable identity, which equality is taken from.
    lacks
        The instances the tail is proven not to contain.
    """

    name: str
    identity: IRQiecId
    lacks: tuple[IRQiecId, ...] = ()


class IRQiecRowEntry(dx.Model):
    """One lexical effect instance and the interface it implements.

    Parameters
    ----------
    instance
        The lexical instance, which is the row's key.
    effect
        The interface application the instance implements.
    """

    instance: IRQiecId
    effect: IRQiecEffectRef


class IRQiecEffectRow(dx.Model):
    """A finite map from lexical instances to interfaces, plus a tail.

    Parameters
    ----------
    entries
        The instances the row names, each with its interface.
    tail
        A row variable standing for further entries, or ``None`` for a
        closed row.
    """

    entries: tuple[IRQiecRowEntry, ...] = ()
    tail: IRQiecRowVariable | None = None


class IRQiecComputationType(dx.Model):
    """The type of a computation: its effect row and result type.

    Parameters
    ----------
    effects
        The row of effect instances the computation may perform.
    result
        The type of the value it returns.
    """

    effects: IRQiecEffectRow
    result: IRQiecStatic


class IRQiecEvidence(dx.TaggedUnion, discriminator="kind"):
    """Kernel-issued equality evidence."""


class IRQiecReflexivity(IRQiecEvidence):
    """Evidence that an equality holds because both sides are the same term.

    Parameters
    ----------
    equality
        The proposition witnessed.
    kind
        The discriminator; always ``"reflexivity"``.
    """

    equality: IRQiecEqualityType
    kind: Literal["reflexivity"] = "reflexivity"


class IRQiecBranchGiven(IRQiecEvidence):
    """Evidence granted by a case branch's constructor refinement.

    Parameters
    ----------
    id
        The stable identity of the given, derived from the branch scope.
    equality
        The proposition the branch's constructor makes available.
    kind
        The discriminator; always ``"branch-given"``.
    """

    id: IRQiecId
    equality: IRQiecEqualityType
    kind: Literal["branch-given"] = "branch-given"


class IRQiecLiteral(dx.TaggedUnion, discriminator="kind"):
    """JSON-safe literal data, including bytes and nested tuples."""


class IRQiecNullLiteral(IRQiecLiteral):
    """The null literal.

    Parameters
    ----------
    kind
        The discriminator; always ``"null"``.
    """

    kind: Literal["null"] = "null"


class IRQiecBoolLiteral(IRQiecLiteral):
    """A Boolean literal.

    Parameters
    ----------
    value
        The Boolean.
    kind
        The discriminator; always ``"bool"``.
    """

    value: bool
    kind: Literal["bool"] = "bool"


class IRQiecIntLiteral(IRQiecLiteral):
    """An integer literal.

    Parameters
    ----------
    value
        The integer.
    kind
        The discriminator; always ``"int"``.
    """

    value: int
    kind: Literal["int"] = "int"


class IRQiecFloatLiteral(IRQiecLiteral):
    """A floating-point literal.

    Parameters
    ----------
    value
        The number.
    kind
        The discriminator; always ``"float"``.
    """

    value: float
    kind: Literal["float"] = "float"


class IRQiecStringLiteral(IRQiecLiteral):
    """A string literal.

    Parameters
    ----------
    value
        The string.
    kind
        The discriminator; always ``"string"``.
    """

    value: str
    kind: Literal["string"] = "string"


class IRQiecBytesLiteral(IRQiecLiteral):
    """A byte-string literal.

    Parameters
    ----------
    value
        The bytes, written as a hexadecimal string.
    kind
        The discriminator; always ``"bytes"``.
    """

    value: str
    kind: Literal["bytes"] = "bytes"


class IRQiecTupleLiteral(IRQiecLiteral):
    """A tuple of literals.

    Parameters
    ----------
    items
        The component literals, in order.
    kind
        The discriminator; always ``"tuple"``.
    """

    items: tuple[IRQiecLiteral, ...]
    kind: Literal["tuple"] = "tuple"


class IRQiecValue(dx.TaggedUnion, discriminator="kind"):
    """Stable value node consumed by QIEC-capable renderers."""


class IRQiecVar(IRQiecValue):
    """Reference to a bound local.

    Parameters
    ----------
    local
        The binder referenced.
    kind
        The discriminator; always ``"var"``.
    """

    local: IRQiecLocal
    kind: Literal["var"] = "var"


class IRQiecLiteralValue(IRQiecValue):
    """A literal host value at a primitive or tuple type.

    Parameters
    ----------
    value
        The literal's data.
    type
        The literal's type.
    kind
        The discriminator; always ``"literal"``.
    """

    value: IRQiecLiteral
    type: IRQiecStatic
    kind: Literal["literal"] = "literal"


class IRQiecConstructorValue(IRQiecValue):
    """Application of a declared constructor to static and value arguments.

    Parameters
    ----------
    constructor
        The stable identity of the constructor applied.
    static_arguments
        The constructor's telescope instantiation, in binder order.
    fields
        The field values, in declaration order.
    result_type
        The indexed family type the application inhabits.
    kind
        The discriminator; always ``"constructor"``.
    """

    constructor: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    fields: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    kind: Literal["constructor"] = "constructor"


class IRQiecEvidenceValue(IRQiecValue):
    """Kernel-checked equality evidence used as a value.

    Parameters
    ----------
    evidence
        The evidence term.
    kind
        The discriminator; always ``"evidence"``.
    """

    evidence: IRQiecEvidence
    kind: Literal["evidence"] = "evidence"


class IRQiecAttachmentRef(IRQiecValue):
    """A stable reference to a runtime-owned host value.

    Parameters
    ----------
    attachment
        The identity under which a runtime provider binds the host value.
    type
        The type the bound value inhabits.
    kind
        The discriminator; always ``"attachment"``.
    """

    attachment: IRQiecId
    type: IRQiecStatic
    kind: Literal["attachment"] = "attachment"


class IRQiecPrimitiveApplication(IRQiecValue):
    """Application of a pure primitive from the closed registry.

    Parameters
    ----------
    primitive
        The primitive's stable identity.
    name
        The primitive's nominal name.
    arguments
        The value arguments, in order.
    result_type
        The primitive's result type.
    origin
        The application's source location.
    kind
        The discriminator; always ``"primitive"``.
    """

    primitive: IRQiecId
    name: str
    arguments: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    origin: IRQiecSourceOrigin
    kind: Literal["primitive"] = "primitive"


class IRQiecTupleValue(IRQiecValue):
    """Construction of a finite product from its components.

    Parameters
    ----------
    items
        The component values, in order.
    result_type
        The product type.
    kind
        The discriminator; always ``"tuple"``.
    """

    items: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    kind: Literal["tuple"] = "tuple"


class IRQiecTensorValue(IRQiecValue):
    """Construction of a tensor from its entries along the outermost axis.

    Parameters
    ----------
    items
        The entries, each an element or a tensor one rank lower.
    result_type
        The ``Tensor`` type constructed.
    kind
        The discriminator; always ``"tensor"``.
    """

    items: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    kind: Literal["tensor"] = "tensor"


class IRQiecProjection(IRQiecValue):
    """Selection of one component of a finite product.

    Parameters
    ----------
    value
        The product value.
    position
        The zero-based component selected.
    result_type
        The component's type.
    kind
        The discriminator; always ``"projection"``.
    """

    value: IRQiecValue
    position: int
    result_type: IRQiecStatic
    kind: Literal["projection"] = "projection"


class IRQiecNamedArgument(dx.Model):
    """One named argument of a distribution construction.

    Parameters
    ----------
    name
        The family parameter supplied.
    value
        The value supplied for it.
    """

    name: str
    value: IRQiecValue


class IRQiecPlateAxis(dx.Model):
    """One axis of a plated distribution.

    Parameters
    ----------
    name
        The axis's source name.
    size
        Its extent, an index term.
    """

    name: str
    size: IRQiecStatic


class IRQiecPlateShape(dx.Model):
    """The plate a distribution is constructed over.

    Parameters
    ----------
    batch
        The independent replication axes, outermost first.
    event
        The joint axes, outermost first.
    """

    batch: tuple[IRQiecPlateAxis, ...] = ()
    event: tuple[IRQiecPlateAxis, ...] = ()


class IRQiecDistributionValue(IRQiecValue):
    """Construction of a distribution from a family of the registry.

    Parameters
    ----------
    family
        The family's stable identity.
    name
        The family's source name.
    arguments
        The named parameters, in source order.
    result_type
        The ``Sampleable`` type constructed.
    origin
        The construction's source location.
    plate
        The plate the construction ranges over; empty for one draw.
    kind
        The discriminator; always ``"distribution"``.
    """

    family: IRQiecId
    name: str
    arguments: tuple[IRQiecNamedArgument, ...]
    result_type: IRQiecStatic
    origin: IRQiecSourceOrigin
    plate: IRQiecPlateShape = IRQiecPlateShape()
    kind: Literal["distribution"] = "distribution"


class IRQiecLogDensity(IRQiecValue):
    """Evaluation of a distribution's log density at a value.

    Parameters
    ----------
    sampleable
        The distribution.
    value
        The point evaluated.
    origin
        The evaluation's source location.
    batch
        The batch axes the weights are kept apart over; empty for one
        total weight.
    kind
        The discriminator; always ``"log_density"``.
    """

    sampleable: IRQiecValue
    value: IRQiecValue
    origin: IRQiecSourceOrigin
    batch: tuple[IRQiecPlateAxis, ...] = ()
    kind: Literal["log_density"] = "log_density"


class IRQiecGather(IRQiecValue):
    """Selection along a tensor's outermost axis.

    Parameters
    ----------
    value
        The tensor selected from.
    index
        An integer or a tensor of integers.
    result_type
        The selection's type.
    kind
        The discriminator; always ``"gather"``.
    """

    value: IRQiecValue
    index: IRQiecValue
    result_type: IRQiecStatic
    kind: Literal["gather"] = "gather"


class IRQiecWeightSum(IRQiecValue):
    """The total of a tensor of log weights.

    Parameters
    ----------
    value
        The weights.
    kind
        The discriminator; always ``"weight_sum"``.
    """

    value: IRQiecValue
    kind: Literal["weight_sum"] = "weight_sum"


class IRQiecSegmentSum(IRQiecValue):
    """Per-group totals of a vector of log weights.

    Parameters
    ----------
    value
        The weights.
    index
        Each entry's group.
    groups
        The number of groups.
    result_type
        The totals' type.
    kind
        The discriminator; always ``"segment_sum"``.
    """

    value: IRQiecValue
    index: IRQiecValue
    groups: IRQiecStatic
    result_type: IRQiecStatic
    kind: Literal["segment_sum"] = "segment_sum"


class IRQiecKernelMatrix(IRQiecValue):
    """A covariance matrix over input locations.

    Parameters
    ----------
    inputs
        The locations.
    kernel
        The kernel's name.
    length_scale
        Its length scale.
    jitter
        The diagonal jitter.
    result_type
        The matrix type.
    kind
        The discriminator; always ``"kernel_matrix"``.
    """

    inputs: IRQiecValue
    kernel: str
    length_scale: float
    jitter: float
    result_type: IRQiecStatic
    kind: Literal["kernel_matrix"] = "kernel_matrix"


class IRQiecAffineMap(IRQiecValue):
    """One head of an affine parameter map.

    Parameters
    ----------
    weight
        The weight matrix.
    bias
        The bias vector.
    sources
        The conditioning row's factors in order.
    row_offset
        The first row of the head's block.
    rows
        The block's height.
    transform
        ``"identity"``, ``"exp"``, or ``"exp_floor"``.
    result_type
        The head's type.
    kind
        The discriminator; always ``"affine_map"``.
    """

    weight: IRQiecValue
    bias: IRQiecValue
    sources: tuple[IRQiecValue, ...]
    row_offset: int
    rows: int
    transform: Literal["identity", "exp", "exp_floor"]
    result_type: IRQiecStatic
    kind: Literal["affine_map"] = "affine_map"


class IRQiecTableMap(IRQiecValue):
    """One head of a table-indexed parameter map.

    Parameters
    ----------
    table
        The parameter table, one row per element of the domain.
    index
        The element of the domain.
    row_offset
        The first column of the head's block.
    rows
        The block's width.
    transform
        ``"identity"``, ``"exp"``, or ``"exp_floor"``.
    result_type
        The head's type.
    kind
        The discriminator; always ``"table_map"``.
    """

    table: IRQiecValue
    index: IRQiecValue
    row_offset: int
    rows: int
    transform: Literal["identity", "exp", "exp_floor"]
    result_type: IRQiecStatic
    kind: Literal["table_map"] = "table_map"


class IRQiecSiteValue(IRQiecValue):
    """A named sample site.

    Parameters
    ----------
    label
        The site's source name.
    result_type
        The ``Site`` type of the value it produces.
    kind
        The discriminator; always ``"site"``.
    """

    label: str
    result_type: IRQiecStatic
    kind: Literal["site"] = "site"


class IRQiecReduction(IRQiecValue):
    """A number summarizing every entry of a tensor.

    Parameters
    ----------
    operator
        The reduction.
    value
        The tensor.
    result_type
        The element type.
    kind
        The discriminator; always ``"reduction"``.
    """

    operator: str
    value: IRQiecValue
    result_type: IRQiecStatic
    kind: Literal["reduction"] = "reduction"


class IRQiecRowwise(IRQiecValue):
    """An operation along a tensor's last axis keeping its shape.

    Parameters
    ----------
    operator
        The operation.
    value
        The tensor.
    result_type
        The tensor's type.
    kind
        The discriminator; always ``"rowwise"``.
    """

    operator: str
    value: IRQiecValue
    result_type: IRQiecStatic
    kind: Literal["rowwise"] = "rowwise"


class IRQiecComprehension(IRQiecValue):
    """A tensor built by evaluating a body at every index of an axis.

    Parameters
    ----------
    binder
        The index local.
    extent
        The axis's extent.
    body
        The entry at each index.
    result_type
        The tensor type.
    kind
        The discriminator; always ``"comprehension"``.
    """

    binder: IRQiecLocal
    extent: IRQiecStatic
    body: IRQiecValue
    result_type: IRQiecStatic
    kind: Literal["comprehension"] = "comprehension"


class IRQiecLetBinding(dx.Model):
    """One binding of an :class:`IRQiecLetValue`.

    Parameters
    ----------
    binder
        The local the bound value is read through.
    value
        The value bound, evaluated once.
    """

    binder: IRQiecLocal
    value: IRQiecValue


class IRQiecLetValue(IRQiecValue):
    """A pure value shared by name through sequential bindings.

    Parameters
    ----------
    bindings
        The bindings in evaluation order; each is in scope in the later
        ones and in the body.
    body
        The value the term denotes.
    result_type
        The body's type.
    kind
        The discriminator; always ``"let"``.
    """

    bindings: tuple[IRQiecLetBinding, ...]
    body: IRQiecValue
    result_type: IRQiecStatic
    kind: Literal["let"] = "let"


class IRQiecTransportValue(IRQiecValue):
    """Transport of a value along kernel-checked equality evidence.

    Parameters
    ----------
    evidence
        The equality the value is carried across.
    value
        The value at the equality's left type.
    target_type
        The type the transported value inhabits, which the equality's right
        side determines.
    kind
        The discriminator; always ``"transport"``.
    """

    evidence: IRQiecEvidence
    value: IRQiecValue
    target_type: IRQiecStatic
    kind: Literal["transport"] = "transport"


class IRQiecEffectRequest(dx.Model):
    """A typed request for one effect operation.

    Parameters
    ----------
    instance
        The effect instance the request is addressed to.
    effect
        The instance's interface application.
    operation
        The operation requested.
    static_arguments
        The operation's telescope instantiation, in binder order.
    arguments
        The value arguments, one per operation argument.
    result_type
        The type of the value the operation returns under the instantiation.
    origin
        Where the request was written and how it was reached.
    """

    instance: IRQiecId
    effect: IRQiecEffectRef
    operation: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    arguments: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    origin: IRQiecSiteProvenance


class IRQiecComputation(dx.TaggedUnion, discriminator="kind"):
    """Stable computation node consumed by QIEC-capable renderers."""


class IRQiecReturn(IRQiecComputation):
    """Return of a value with no effects.

    Parameters
    ----------
    value
        The value produced.
    kind
        The discriminator; always ``"return"``.
    """

    value: IRQiecValue
    kind: Literal["return"] = "return"


class IRQiecBindStep(dx.Model):
    """One step of a bind sequence: a computation and the local it binds.

    Parameters
    ----------
    binder
        The local the step's result is bound to.
    first
        The computation run at the step.
    """

    binder: IRQiecLocal
    first: IRQiecComputation


class IRQiecBind(IRQiecComputation):
    """A straight-line sequence of binds ending in a computation.

    A kernel ``Bind`` chain nests one bind inside the next; the mirror
    keeps the chain flat so a long straight-line body is a tuple of
    steps rather than a term as deep as it is long.

    Parameters
    ----------
    steps
        The binds, in evaluation order; each may read the binders before
        it.
    then
        The computation after the last step, which may read every binder.
    kind
        The discriminator; always ``"bind"``.
    """

    steps: tuple[IRQiecBindStep, ...]
    then: IRQiecComputation
    kind: Literal["bind"] = "bind"


class IRQiecPerform(IRQiecComputation):
    """Performance of one effect operation.

    Parameters
    ----------
    request
        The operation, its instance, arguments, and result type.
    kind
        The discriminator; always ``"perform"``.
    """

    request: IRQiecEffectRequest
    kind: Literal["perform"] = "perform"


class IRQiecHandle(IRQiecComputation):
    """A computation run under a handler installed for one effect instance.

    Parameters
    ----------
    instance
        The effect instance whose operations the handler intercepts.
    handler
        The stable identity of the handler installed.
    computation
        The handled computation.
    static_arguments
        The handler's telescope instantiation, in binder order.
    kind
        The discriminator; always ``"handle"``.
    """

    instance: IRQiecId
    handler: IRQiecId
    computation: IRQiecComputation
    static_arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["handle"] = "handle"


class IRQiecCaseMotive(dx.Model):
    """The result family of an indexed case expression.

    Parameters
    ----------
    indices
        The binders abstracting the scrutinee's indices in ``result_type``.
    result_type
        The type every branch produces, over ``indices``.
    """

    indices: tuple[IRQiecBinder, ...]
    result_type: IRQiecStatic


class IRQiecCaseBranch(dx.Model):
    """One branch of an indexed case expression.

    Parameters
    ----------
    constructor
        The constructor this branch matches.
    static_arguments
        The constructor's static arguments as bound by the pattern, in
        binder order.
    fields
        The locals binding the constructor's fields, in declaration order.
    body
        The computation run when the branch is selected.
    scope
        The static scope identity under which the branch's skolems and
        equality evidence are minted.
    """

    constructor: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    fields: tuple[IRQiecLocal, ...]
    body: IRQiecComputation
    scope: IRQiecId


class IRQiecCase(IRQiecComputation):
    """Case analysis over an indexed family value.

    Parameters
    ----------
    scrutinee
        The value analyzed.
    motive
        The result family, over the scrutinee's indices.
    branches
        One branch per constructor considered.
    kind
        The discriminator; always ``"case"``.
    """

    scrutinee: IRQiecValue
    motive: IRQiecCaseMotive
    branches: tuple[IRQiecCaseBranch, ...]
    kind: Literal["case"] = "case"


class IRQiecIf(IRQiecComputation):
    """Branch on a Boolean value.

    Parameters
    ----------
    condition
        The Boolean value branched on.
    then
        The computation run when the condition holds.
    otherwise
        The computation run when it does not.
    kind
        The discriminator; always ``"if"``.
    """

    condition: IRQiecValue
    then: IRQiecComputation
    otherwise: IRQiecComputation
    kind: Literal["if"] = "if"


class IRQiecCall(IRQiecComputation):
    """Application of a named computation by stable identity.

    Parameters
    ----------
    callee
        The identity of the computation called.
    name
        The callee's display name.
    static_arguments
        The callee's telescope instantiation, in binder order.
    arguments
        The value arguments, one per callee parameter.
    result_type
        The callee's result type under the instantiation.
    effects
        The callee's effect row under the instantiation.
    origin
        The call site's source location.
    kind
        The discriminator; always ``"call"``.
    """

    callee: IRQiecId
    name: str
    static_arguments: tuple[IRQiecStatic, ...]
    arguments: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    effects: IRQiecEffectRow
    origin: IRQiecSourceOrigin
    kind: Literal["call"] = "call"


class IRQiecResume(IRQiecComputation):
    """Invocation of the enclosing handler clause's continuation.

    Parameters
    ----------
    value
        The value returned to the suspended computation.
    origin
        The resumption's source location.
    kind
        The discriminator; always ``"resume"``.
    """

    value: IRQiecValue
    origin: IRQiecSourceOrigin
    kind: Literal["resume"] = "resume"


class IRQiecNewInstance(IRQiecComputation):
    """Lexically scoped allocation of an effect instance.

    Parameters
    ----------
    instance
        The scope-derived identity of the allocated instance.
    effect
        The applied interface the instance provides.
    body
        The computation within which the instance is in scope.
    origin
        The allocation's source location.
    kind
        The discriminator; always ``"new_instance"``.
    """

    instance: IRQiecId
    effect: IRQiecEffectRef
    body: IRQiecComputation
    origin: IRQiecSourceOrigin
    kind: Literal["new_instance"] = "new_instance"


class IRQiecFieldDef(dx.Model):
    """One named, typed field of a constructor.

    Parameters
    ----------
    name
        The field's display name.
    type
        The field's type, which may mention the constructor's telescope.
    """

    name: str
    type: IRQiecStatic


class IRQiecFamilyDecl(dx.Model):
    """A closed or opaque indexed data family.

    Parameters
    ----------
    id
        The family's stable identity.
    name
        The family's source name.
    parameters
        The binders uniform across every constructor.
    indices
        The binders a constructor result may refine.
    constructors
        The constructors the family declares.
    closed
        Whether the constructor list is exhaustive, which admits coverage
        checking.
    """

    id: IRQiecId
    name: str
    parameters: tuple[IRQiecBinder, ...]
    indices: tuple[IRQiecBinder, ...]
    constructors: tuple[IRQiecId, ...]
    closed: bool = True


class IRQiecConstructorDecl(dx.Model):
    """One constructor of an indexed data family.

    Parameters
    ----------
    id
        The constructor's stable identity.
    family
        The family the constructor belongs to.
    name
        The constructor's source name.
    telescope
        The binders local to the constructor, which are rigid while a
        branch matching it is checked.
    fields
        The values the constructor carries.
    result_indices
        What the constructor fixes the family's indices to, one per family
        index.
    """

    id: IRQiecId
    family: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...]
    fields: tuple[IRQiecFieldDef, ...]
    result_indices: tuple[IRQiecStatic, ...]


class IRQiecArgumentDef(dx.Model):
    """One named, typed value argument of an operation.

    Parameters
    ----------
    name
        The argument's display name.
    type
        The argument's type, which may mention the operation's telescope.
    """

    name: str
    type: IRQiecStatic


class IRQiecOperationDef(dx.Model):
    """One result-indexed request constructor of an effect interface.

    Parameters
    ----------
    id
        The operation's stable identity.
    name
        The operation's source name.
    telescope
        The static binders the operation adds to the interface's own.
    arguments
        The value arguments the request carries.
    result_type
        What resuming the request supplies.
    """

    id: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...]
    arguments: tuple[IRQiecArgumentDef, ...]
    result_type: IRQiecStatic


class IRQiecEffectDef(dx.Model):
    """An effect interface: its telescope and operations.

    Parameters
    ----------
    ref
        The declaration's own reference.
    telescope
        The static binders the interface takes.
    operations
        The declared operations.
    """

    ref: IRQiecEffectRef
    telescope: tuple[IRQiecBinder, ...]
    operations: tuple[IRQiecOperationDef, ...]


class IRQiecHandlerClauseDef(dx.Model):
    """One operation a handler covers, and how it may resume.

    Parameters
    ----------
    operation
        The operation the clause handles.
    grade
        How often the clause may invoke its continuation: never (``"0"``),
        at most once (``"aff"``), exactly once (``"1"``), or any number of
        times (``"omega"``).
    parameters
        The binders for the operation's value arguments, in declaration
        order.
    body
        What the clause does, or ``None`` for a clause whose implementation
        a runtime provider supplies.
    """

    operation: IRQiecId
    grade: Literal["0", "aff", "1", "omega"]
    parameters: tuple[IRQiecLocal, ...] = ()
    body: IRQiecComputation | None = None


class IRQiecHandlerReturnClauseDef(dx.Model):
    """What a handler does with a value its computation returns.

    Parameters
    ----------
    binder
        The local naming the returned value inside the body.
    body
        What the handler produces from it.
    """

    binder: IRQiecLocal
    body: IRQiecComputation


class IRQiecHandlerDef(dx.Model):
    """The type-and-coverage signature of a handler.

    Parameters
    ----------
    id
        The handler's stable identity.
    name
        The handler's display name.
    effect
        The interface application the handler handles.
    clauses
        One clause per operation the handler covers.
    input_type
        The result type of the computation the handler accepts.
    output_type
        The type of the handler's answer.
    introduced
        The effects the handler's own clauses perform.
    total
        Whether the clauses cover every operation of ``effect``.
    forwards_unknown
        Whether an operation no clause covers is forwarded outward rather
        than rejected.
    telescope
        The static parameters the handler abstracts over, in order.
    return_clause
        How the handler answers the computation's return, or ``None`` to
        return the value unchanged.
    implementation
        ``"authored"`` when every clause carries a body, ``"foreign"`` when
        a runtime provider supplies the behavior.
    """

    id: IRQiecId
    name: str
    effect: IRQiecEffectRef
    clauses: tuple[IRQiecHandlerClauseDef, ...]
    input_type: IRQiecStatic
    output_type: IRQiecStatic
    introduced: IRQiecEffectRow
    total: bool
    forwards_unknown: bool
    telescope: tuple[IRQiecBinder, ...]
    return_clause: IRQiecHandlerReturnClauseDef | None = None
    implementation: Literal["authored", "foreign"] = "foreign"


class IRQiecNamedEffectInstance(dx.Model):
    """A source name for one lexical effect instance.

    Parameters
    ----------
    name
        The name requests resolve through.
    entry
        The allocated instance and its interface application.
    origin
        Where the instance was declared.
    """

    name: str
    entry: IRQiecRowEntry
    origin: IRQiecSourceOrigin


class IRQiecNamedComputation(dx.Model):
    """One named computation and its checked type.

    Parameters
    ----------
    id
        The computation's stable identity, which calls refer to.
    name
        The display name, unique within the module.
    telescope
        The static parameters the computation abstracts over, in order.
    parameters
        The value parameters, in order.
    body
        The computation's body, with the parameters in scope.
    type
        The checked effect row and result type of ``body``.
    origin
        Where the computation was declared.
    """

    id: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...]
    parameters: tuple[IRQiecLocal, ...]
    body: IRQiecComputation
    type: IRQiecComputationType
    origin: IRQiecSourceOrigin


class IRQiecProgramParameter(dx.Model):
    """One parameter of an elaborated program.

    Parameters
    ----------
    name
        The parameter's name.
    role
        How it is supplied.
    type
        Its type.
    axes
        The name of each dimension of a tensor-typed parameter.
    per_row
        Whether its leading axis is the plate of the draw it fills, so
        that a value without that axis is shared by every row.
    """

    name: str
    role: str
    type: IRQiecStatic
    axes: tuple[str, ...] = ()
    per_row: bool = False


class IRQiecProgramSite(dx.Model):
    """One probabilistic step of an elaborated program.

    Parameters
    ----------
    name
        The site's label.
    kind
        ``"sample"``, ``"observe"``, or ``"marginal"``.
    family
        The family drawn from.
    batch
        The plate's batch axes.
    event
        The plate's event axes.
    """

    name: str
    kind: str
    family: str
    batch: tuple[IRQiecPlateAxis, ...] = ()
    event: tuple[IRQiecPlateAxis, ...] = ()


class IRQiecProgramEntry(dx.Model):
    """The typed entry point a program declaration elaborates to.

    Parameters
    ----------
    name
        The program's name.
    computation
        The identity of the computation holding its body.
    parameters
        The computation's parameters with their roles.
    return_names
        The names the program returns.
    return_labels
        The labels of a labelled return, or ``None``.
    sites
        The program's probabilistic steps.
    random_instance
        The ``Random`` instance the program samples on.
    score_instance
        The ``Score`` instance the program scores on.
    telescope
        The static index binders of the open input extents.
    """

    name: str
    computation: IRQiecId
    parameters: tuple[IRQiecProgramParameter, ...]
    return_names: tuple[str, ...]
    return_labels: tuple[str, ...] | None
    sites: tuple[IRQiecProgramSite, ...]
    random_instance: IRQiecId
    score_instance: IRQiecId
    telescope: tuple[IRQiecBinder, ...] = ()


class IRQiecModule(dx.Model):
    """Lossless structural projection of one checked :class:`QiecModule`."""

    module: str
    source_protocol: str
    index_sorts: tuple[IRQiecUserIndexSort, ...] = ()
    families: tuple[IRQiecFamilyDecl, ...] = ()
    constructors: tuple[IRQiecConstructorDecl, ...] = ()
    effects: tuple[IRQiecEffectDef, ...] = ()
    instances: tuple[IRQiecNamedEffectInstance, ...] = ()
    handlers: tuple[IRQiecHandlerDef, ...] = ()
    computations: tuple[IRQiecNamedComputation, ...] = ()
    entries: tuple[IRQiecProgramEntry, ...] = ()
    programs: tuple[IRQiecId, ...] = ()
    gap: str = ""
    abi: str

    def program_computations(self) -> frozenset[str]:
        """The computations that hold programs and their marginal helpers.

        Returns
        -------
        frozenset[str]
            Identity texts of every entry point's computation and of
            every helper it calls, transitively, as the kernel module
            recorded them at lowering.
        """
        return frozenset(identity.text for identity in self.programs)


def _callees(node: IRQiecComputation) -> list[str]:
    """The computations a body calls.

    Parameters
    ----------
    node
        The body.

    Returns
    -------
    list[str]
        Identity texts of every callee.
    """
    found: list[str] = []

    def visit(item: IRQiecComputation) -> None:
        if isinstance(item, IRQiecCall):
            found.append(item.callee.text)
        elif isinstance(item, IRQiecBind):
            for step in item.steps:
                visit(step.first)
            visit(item.then)
        elif isinstance(item, IRQiecHandle):
            visit(item.computation)
        elif isinstance(item, IRQiecCase):
            for branch in item.branches:
                visit(branch.body)
        elif isinstance(item, IRQiecIf):
            visit(item.then)
            visit(item.otherwise)
        elif isinstance(item, IRQiecNewInstance):
            visit(item.body)

    visit(node)
    return found


def _id(value: StableId) -> IRQiecId:
    return IRQiecId(namespace=value.namespace, digest=value.digest)


def _literal(value: object) -> IRQiecLiteral:
    if value is None:
        return IRQiecNullLiteral()
    if isinstance(value, bool):
        return IRQiecBoolLiteral(value=value)
    if isinstance(value, int):
        return IRQiecIntLiteral(value=value)
    if isinstance(value, float):
        return IRQiecFloatLiteral(value=value)
    if isinstance(value, str):
        return IRQiecStringLiteral(value=value)
    if isinstance(value, bytes):
        return IRQiecBytesLiteral(value=value.hex())
    if isinstance(value, tuple):
        return IRQiecTupleLiteral(items=tuple(_literal(item) for item in value))
    raise TypeError(f"unsupported QIEC literal {value!r}")


def _convert(value: object) -> object:  # noqa: C901, PLR0911, PLR0912
    """Recursively project a checked kernel record into structural IR."""
    if isinstance(value, StableId):
        return _id(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    if isinstance(value, ids.SourceOrigin):
        return IRQiecSourceOrigin(
            module=value.module,
            structural_path=value.structural_path,
            role=value.role,
            source_protocol=value.source_protocol,
            file=value.file,
            line=value.line,
            column=value.column,
        )
    if isinstance(value, ids.DynamicAddressFrame):
        return IRQiecDynamicAddressFrame(scope=value.scope, key=value.key)
    if isinstance(value, ids.SiteProvenance):
        return IRQiecSiteProvenance(
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
            dynamic_path=cast(
                tuple[IRQiecDynamicAddressFrame, ...], _convert(value.dynamic_path)
            ),
            resumption_path=value.resumption_path,
            relation=value.relation,
            parents=cast(tuple[IRQiecId, ...], _convert(value.parents)),
        )
    kind_map = {
        k.TypeKind: IRQiecTypeKind,
        k.EffectKind: IRQiecEffectKind,
        k.RowKind: IRQiecRowKind,
    }
    if type(value) in kind_map:
        return kind_map[type(value)]()
    if isinstance(value, k.ArrowKind):
        return IRQiecArrowKind(
            domain=cast(IRQiecKind, _convert(value.domain)),
            codomain=cast(IRQiecKind, _convert(value.codomain)),
        )
    sort_map = {k.NatSort: IRQiecNatSort}
    if type(value) in sort_map:
        return sort_map[type(value)]()
    if isinstance(value, k.ShapeSort):
        return IRQiecShapeSort(rank=value.rank)
    if isinstance(value, k.ContextSort):
        return IRQiecContextSort(signature=value.signature)
    if isinstance(value, k.UserIndexSort):
        return IRQiecUserIndexSort(
            name=value.name, constructors=value.constructors, arities=value.arities
        )
    if isinstance(value, k.TypeBinder):
        return IRQiecTypeBinder(
            name=value.name,
            type_kind=cast(IRQiecKind, _convert(value.kind)),
            refinable=value.refinable,
        )
    if isinstance(value, k.IndexBinder):
        return IRQiecIndexBinder(
            name=value.name,
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
            refinable=value.refinable,
        )
    if isinstance(value, k.EffectBinder):
        return IRQiecEffectBinder(name=value.name, refinable=value.refinable)
    if isinstance(value, ty.IndexVariable):
        return IRQiecIndexVariable(
            name=value.name,
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
            identity=cast(IRQiecId | None, _convert(value.identity)),
        )
    if isinstance(value, ty.IndexLiteral):
        return IRQiecIndexLiteral(
            value=value.value, sort=cast(IRQiecIndexSort, _convert(value.sort))
        )
    if isinstance(value, ty.IndexConstructor):
        return IRQiecIndexConstructor(
            name=value.name,
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
        )
    if isinstance(value, ty.ShapeIndex):
        return IRQiecShapeIndex(
            dimensions=cast(tuple[IRQiecStatic, ...], _convert(value.dimensions))
        )
    if isinstance(value, ty.TypeVariable):
        return IRQiecTypeVariable(
            name=value.name,
            type_kind=cast(IRQiecKind, _convert(value.kind)),
            identity=cast(IRQiecId | None, _convert(value.identity)),
        )
    if isinstance(value, ty.TypeConstructorRef):
        return IRQiecTypeConstructor(
            id=_id(value.id),
            name=value.name,
            telescope=cast(tuple[IRQiecBinder, ...], _convert(value.telescope)),
        )
    if isinstance(value, ty.TypeApplication):
        return IRQiecTypeApplication(
            constructor=cast(IRQiecTypeConstructor, _convert(value.constructor)),
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
        )
    if isinstance(value, ty.FunctionType):
        return IRQiecFunctionType(
            parameter=cast(IRQiecStatic, _convert(value.parameter)),
            result=cast(IRQiecStatic, _convert(value.result)),
        )
    if isinstance(value, ty.EqualityType):
        return IRQiecEqualityType(
            classifier=cast(IRQiecKind | IRQiecIndexSort, _convert(value.kind)),
            left=cast(IRQiecStatic, _convert(value.left)),
            right=cast(IRQiecStatic, _convert(value.right)),
        )
    if isinstance(value, ty.EffectVariable):
        return IRQiecEffectVariable(
            name=value.name, identity=cast(IRQiecId | None, _convert(value.identity))
        )
    if isinstance(value, ty.EffectRef):
        return IRQiecEffectRef(
            id=_id(value.id),
            name=value.name,
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
        )
    if isinstance(value, tm.Local):
        return IRQiecLocal(
            name=value.name, type=cast(IRQiecStatic, _convert(value.type))
        )
    if isinstance(value, e.RowVariable):
        return IRQiecRowVariable(
            name=value.name,
            identity=_id(value.identity),
            lacks=cast(tuple[IRQiecId, ...], _convert(value.lacks)),
        )
    if isinstance(value, e.RowEntry):
        return IRQiecRowEntry(
            instance=_id(value.instance),
            effect=cast(IRQiecEffectRef, _convert(value.effect)),
        )
    if isinstance(value, e.EffectRow):
        return IRQiecEffectRow(
            entries=cast(tuple[IRQiecRowEntry, ...], _convert(value.entries)),
            tail=cast(IRQiecRowVariable | None, _convert(value.tail)),
        )
    if isinstance(value, e.ComputationType):
        return IRQiecComputationType(
            effects=cast(IRQiecEffectRow, _convert(value.effects)),
            result=cast(IRQiecStatic, _convert(value.result)),
        )
    if isinstance(value, ev.Reflexivity):
        return IRQiecReflexivity(
            equality=cast(IRQiecEqualityType, _convert(value.equality))
        )
    if isinstance(value, ev.BranchGiven):
        return IRQiecBranchGiven(
            id=_id(value.id),
            equality=cast(IRQiecEqualityType, _convert(value.equality)),
        )
    if isinstance(value, tm.Var):
        return IRQiecVar(local=cast(IRQiecLocal, _convert(value.local)))
    if isinstance(value, tm.LiteralValue):
        return IRQiecLiteralValue(
            value=_literal(value.value), type=cast(IRQiecStatic, _convert(value.type))
        )
    if isinstance(value, tm.ConstructorValue):
        return IRQiecConstructorValue(
            constructor=_id(value.constructor),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            fields=cast(tuple[IRQiecValue, ...], _convert(value.fields)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.EvidenceValue):
        return IRQiecEvidenceValue(
            evidence=cast(IRQiecEvidence, _convert(value.evidence))
        )
    if isinstance(value, tm.AttachmentRef):
        return IRQiecAttachmentRef(
            attachment=_id(value.attachment),
            type=cast(IRQiecStatic, _convert(value.type)),
        )
    if isinstance(value, tm.TransportValue):
        return IRQiecTransportValue(
            evidence=cast(IRQiecEvidence, _convert(value.evidence)),
            value=cast(IRQiecValue, _convert(value.value)),
            target_type=cast(IRQiecStatic, _convert(value.target_type)),
        )
    if isinstance(value, tm.PlateAxis):
        return IRQiecPlateAxis(
            name=value.name, size=cast(IRQiecStatic, _convert(value.size))
        )
    if isinstance(value, tm.PlateShape):
        return IRQiecPlateShape(
            batch=cast(tuple[IRQiecPlateAxis, ...], _convert(value.batch)),
            event=cast(tuple[IRQiecPlateAxis, ...], _convert(value.event)),
        )
    if isinstance(value, tm.DistributionValue):
        return IRQiecDistributionValue(
            family=_id(value.family),
            name=value.name,
            arguments=tuple(
                IRQiecNamedArgument(name=name, value=cast(IRQiecValue, _convert(item)))
                for name, item in value.arguments
            ),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
            plate=cast(IRQiecPlateShape, _convert(value.plate)),
        )
    if isinstance(value, tm.LogDensity):
        return IRQiecLogDensity(
            sampleable=cast(IRQiecValue, _convert(value.sampleable)),
            value=cast(IRQiecValue, _convert(value.value)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
            batch=cast(tuple[IRQiecPlateAxis, ...], _convert(value.batch)),
        )
    if isinstance(value, tm.Gather):
        return IRQiecGather(
            value=cast(IRQiecValue, _convert(value.value)),
            index=cast(IRQiecValue, _convert(value.index)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.WeightSum):
        return IRQiecWeightSum(value=cast(IRQiecValue, _convert(value.value)))
    if isinstance(value, tm.SegmentSum):
        return IRQiecSegmentSum(
            value=cast(IRQiecValue, _convert(value.value)),
            index=cast(IRQiecValue, _convert(value.index)),
            groups=cast(IRQiecStatic, _convert(value.groups)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.KernelMatrix):
        return IRQiecKernelMatrix(
            inputs=cast(IRQiecValue, _convert(value.inputs)),
            kernel=value.kernel,
            length_scale=value.length_scale,
            jitter=value.jitter,
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.Reduction):
        return IRQiecReduction(
            operator=value.operator,
            value=cast(IRQiecValue, _convert(value.value)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.Rowwise):
        return IRQiecRowwise(
            operator=value.operator,
            value=cast(IRQiecValue, _convert(value.value)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.Comprehension):
        return IRQiecComprehension(
            binder=cast(IRQiecLocal, _convert(value.binder)),
            extent=cast(IRQiecStatic, _convert(value.extent)),
            body=cast(IRQiecValue, _convert(value.body)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.LetValue):
        return IRQiecLetValue(
            bindings=tuple(
                IRQiecLetBinding(
                    binder=cast(IRQiecLocal, _convert(binding.binder)),
                    value=cast(IRQiecValue, _convert(binding.value)),
                )
                for binding in value.bindings
            ),
            body=cast(IRQiecValue, _convert(value.body)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.AffineMap):
        return IRQiecAffineMap(
            weight=cast(IRQiecValue, _convert(value.weight)),
            bias=cast(IRQiecValue, _convert(value.bias)),
            sources=cast(tuple[IRQiecValue, ...], _convert(value.sources)),
            row_offset=value.row_offset,
            rows=value.rows,
            transform=value.transform,
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.TableMap):
        return IRQiecTableMap(
            table=cast(IRQiecValue, _convert(value.table)),
            index=cast(IRQiecValue, _convert(value.index)),
            row_offset=value.row_offset,
            rows=value.rows,
            transform=value.transform,
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.SiteValue):
        return IRQiecSiteValue(
            label=value.label,
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.PrimitiveApplication):
        return IRQiecPrimitiveApplication(
            primitive=_id(value.primitive),
            name=value.name,
            arguments=cast(tuple[IRQiecValue, ...], _convert(value.arguments)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
        )
    if isinstance(value, tm.TupleValue):
        return IRQiecTupleValue(
            items=cast(tuple[IRQiecValue, ...], _convert(value.items)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.TensorValue):
        return IRQiecTensorValue(
            items=cast(tuple[IRQiecValue, ...], _convert(value.items)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.Projection):
        return IRQiecProjection(
            value=cast(IRQiecValue, _convert(value.value)),
            position=value.position,
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, e.EffectRequest):
        return IRQiecEffectRequest(
            instance=_id(value.instance),
            effect=cast(IRQiecEffectRef, _convert(value.effect)),
            operation=_id(value.operation),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            arguments=cast(tuple[IRQiecValue, ...], _convert(value.arguments)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
            origin=cast(IRQiecSiteProvenance, _convert(value.origin)),
        )
    if isinstance(value, tm.Return):
        return IRQiecReturn(value=cast(IRQiecValue, _convert(value.value)))
    if isinstance(value, tm.Bind):
        steps: list[IRQiecBindStep] = []
        current: tm.Computation = value
        while isinstance(current, tm.Bind):
            steps.append(
                IRQiecBindStep(
                    binder=cast(IRQiecLocal, _convert(current.binder)),
                    first=cast(IRQiecComputation, _convert(current.first)),
                )
            )
            current = current.then
        return IRQiecBind(
            steps=tuple(steps),
            then=cast(IRQiecComputation, _convert(current)),
        )
    if isinstance(value, tm.Perform):
        return IRQiecPerform(request=cast(IRQiecEffectRequest, _convert(value.request)))
    if isinstance(value, tm.Handle):
        return IRQiecHandle(
            instance=_id(value.instance),
            handler=_id(value.handler),
            computation=cast(IRQiecComputation, _convert(value.computation)),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
        )
    if isinstance(value, tm.CaseMotive):
        return IRQiecCaseMotive(
            indices=cast(tuple[IRQiecBinder, ...], _convert(value.indices)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.CaseBranch):
        return IRQiecCaseBranch(
            constructor=_id(value.constructor),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            fields=cast(tuple[IRQiecLocal, ...], _convert(value.fields)),
            body=cast(IRQiecComputation, _convert(value.body)),
            scope=_id(value.scope),
        )
    if isinstance(value, tm.Case):
        return IRQiecCase(
            scrutinee=cast(IRQiecValue, _convert(value.scrutinee)),
            motive=cast(IRQiecCaseMotive, _convert(value.motive)),
            branches=cast(tuple[IRQiecCaseBranch, ...], _convert(value.branches)),
        )
    if isinstance(value, tm.If):
        return IRQiecIf(
            condition=cast(IRQiecValue, _convert(value.condition)),
            then=cast(IRQiecComputation, _convert(value.then)),
            otherwise=cast(IRQiecComputation, _convert(value.otherwise)),
        )
    if isinstance(value, tm.Call):
        return IRQiecCall(
            callee=_id(value.callee),
            name=value.name,
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            arguments=cast(tuple[IRQiecValue, ...], _convert(value.arguments)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
            effects=cast(IRQiecEffectRow, _convert(value.effects)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
        )
    if isinstance(value, tm.Resume):
        return IRQiecResume(
            value=cast(IRQiecValue, _convert(value.value)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
        )
    if isinstance(value, tm.NewInstance):
        return IRQiecNewInstance(
            instance=_id(value.instance),
            effect=cast(IRQiecEffectRef, _convert(value.effect)),
            body=cast(IRQiecComputation, _convert(value.body)),
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
        )
    if not isinstance(value, QiecModel):
        raise TypeError(f"unsupported QIEC kernel value {value!r}")
    model_map: dict[type[object], type[dx.Model]] = {
        d.FieldDef: IRQiecFieldDef,
        d.FamilyDecl: IRQiecFamilyDecl,
        d.ConstructorDecl: IRQiecConstructorDecl,
        e.ArgumentDef: IRQiecArgumentDef,
        e.OperationDef: IRQiecOperationDef,
        e.EffectDef: IRQiecEffectDef,
        e.HandlerClauseDef: IRQiecHandlerClauseDef,
        e.HandlerReturnClauseDef: IRQiecHandlerReturnClauseDef,
        e.HandlerDef: IRQiecHandlerDef,
        m.NamedEffectInstance: IRQiecNamedEffectInstance,
        m.NamedComputation: IRQiecNamedComputation,
        m.QiecModule: IRQiecModule,
        pr.ProgramParameter: IRQiecProgramParameter,
        pr.ProgramSite: IRQiecProgramSite,
        pr.ProgramEntry: IRQiecProgramEntry,
    }
    model = model_map.get(type(value))
    if model is None:
        raise TypeError(f"unsupported QIEC kernel record {type(value).__qualname__}")
    values = {
        name: _convert(getattr(value, name))
        for name in type(value).__field_specs__
        if name != "tag"
    }
    return model(**values)


def lower_qiec_ir(module: QiecModule) -> IRQiecModule:
    """Project an already checked kernel module into structural IR.

    Parameters
    ----------
    module : QiecModule
        The checked module.

    Returns
    -------
    IRQiecModule
        The mirror, with the program computations recorded so a renderer
        need not walk every body to find them.
    """
    reachable = m.reachable_computations(module)
    standalone_deduction_roots = (
        {
            computation.id
            for computation in module.computations
            if _is_deduction_origin(computation.origin)
            and computation.name.endswith("__run")
        }
        if not module.entries
        else set()
    )
    values = {
        name: _convert(getattr(module, name))
        for name in type(module).__field_specs__
        if name not in ("tag", "computations")
    }
    # A deduction's computations serve the reference machine; a target
    # carries them only when a program the module transpiles calls one,
    # and then refuses the search they perform.
    values["computations"] = _convert(
        tuple(
            computation
            for computation in module.computations
            if not _is_deduction_origin(computation.origin)
            or computation.id in reachable
            or computation.id in standalone_deduction_roots
        )
    )
    values["programs"] = tuple(
        _id(identity)
        for identity in sorted(
            m.program_computations(module), key=lambda identity: identity.digest
        )
    )
    return IRQiecModule(**values)


def _is_deduction_origin(origin: SourceOrigin | IRQiecSourceOrigin) -> bool:
    """Whether an origin lies under a deduction's elaboration.

    Parameters
    ----------
    origin : SourceOrigin | IRQiecSourceOrigin
        The origin.

    Returns
    -------
    bool
        ``True`` when the structural path starts at ``deductions``.
    """
    path = origin.structural_path
    return bool(path) and path[0] == "deductions"


type QiecFeature = Literal[
    "declarations",
    "return",
    "bind",
    "perform",
    "handle",
    "case",
    "if",
    "call",
    "recursion",
    "resume",
    "local-instance",
    "authored-handler",
    "arithmetic",
    "comparison",
    "boolean",
    "string",
    "conversion",
    "math",
    "tuple",
    "tensor",
    "distribution",
    "plate",
    "log-density",
    "site",
    "gather",
    "segment-sum",
    "kernel-matrix",
    "affine-map",
    "table-map",
    "neural-attachment",
    "search",
    "reduction",
    "rowwise",
    "comprehension",
    "let",
    "special",
    "activation",
    "weight",
    "evidence",
    "transport",
    "attachment",
    "structured-value",
    "effectful-row",
    "open-effect-row",
    "type-polymorphism",
    "index-polymorphism",
    "effect-polymorphism",
    "named-parameter",
    "non-scalar-result",
    "non-scalar-parameter",
    "zero-resumption",
    "affine-resumption",
    "linear-resumption",
    "unrestricted-resumption",
]
"""One QIEC construct a target may or may not preserve.

[`analyze_qiec_capabilities`][quivers.transpile.qiec_ir.analyze_qiec_capabilities]
names the features a module needs, and a
[`QiecTargetCapabilities`][quivers.transpile.qiec_ir.QiecTargetCapabilities]
names the ones a target has.
"""


class QiecTargetCapabilities(dx.Model):
    """Feature set one target can preserve without semantic erasure.

    Parameters
    ----------
    features
        The features the target preserves.
    """

    features: frozenset[QiecFeature] = frozenset({"declarations"})

    def supports(self, feature: QiecFeature) -> bool:
        """Whether the target preserves one feature.

        Parameters
        ----------
        feature
            The feature asked about.

        Returns
        -------
        bool
            ``True`` when ``feature`` is in the set.
        """
        return feature in self.features


class QiecCapabilityDiagnostic(dx.Model):
    """One precise target mismatch discovered before renderer dispatch.

    Parameters
    ----------
    code
        The diagnostic code; always ``"qiec-capability"``.
    target
        The target that lacks the feature.
    feature
        The feature the target lacks.
    computation
        The computation that needs the feature, or ``None`` for the module.
    origin
        Where the computation was declared, when known.
    detail
        Why the target cannot preserve the feature, phrased to follow the
        target's name, as in ``"has no semantics-preserving lowering"``.
    """

    code: Literal["qiec-capability"] = "qiec-capability"
    target: str
    feature: QiecFeature
    computation: str | None = None
    origin: IRQiecSourceOrigin | None = None
    detail: str

    @property
    def kind(self) -> str:
        """The ``qiec:capability:<feature>:<subject>`` refusal identifier."""
        subject = self.computation or "module"
        return f"qiec:capability:{self.feature}:{subject}"

    @property
    def message(self) -> str:
        """The diagnostic as one user-facing sentence."""
        subject = (
            f"QIEC computation `{self.computation}`"
            if self.computation is not None
            else "the QIEC module"
        )
        return f"{subject} requires `{self.feature}`, but {self.target} {self.detail}"


_DECLARATION_ONLY = QiecTargetCapabilities()
_GENERIC_RUNTIME = QiecTargetCapabilities(
    features=frozenset(
        {
            "declarations",
            "return",
            "bind",
            "perform",
            "handle",
            "case",
            "if",
            "call",
            "recursion",
            "resume",
            "local-instance",
            "authored-handler",
            "arithmetic",
            "comparison",
            "boolean",
            "string",
            "conversion",
            "math",
            "tuple",
            "tensor",
            "distribution",
            "log-density",
            "site",
            "special",
            "activation",
            "weight",
            "gather",
            "segment-sum",
            "reduction",
            "rowwise",
            "comprehension",
            "let",
            "evidence",
            "transport",
            "attachment",
            "structured-value",
            "effectful-row",
            "open-effect-row",
            "type-polymorphism",
            "index-polymorphism",
            "effect-polymorphism",
            "named-parameter",
            "non-scalar-result",
            "non-scalar-parameter",
            "zero-resumption",
            "affine-resumption",
            "linear-resumption",
            "unrestricted-resumption",
        }
    )
)
_STAN_STATIC = QiecTargetCapabilities(
    features=frozenset(
        {
            "declarations",
            "return",
            "bind",
            "named-parameter",
            "arithmetic",
            "comparison",
            "boolean",
            "conversion",
            "math",
            "special",
            "activation",
            "if",
            "call",
            "recursion",
            "let",
        }
    )
)
_GRAPH_STATIC = QiecTargetCapabilities(
    features=frozenset({"declarations", "return", "bind"})
)


def capabilities_for_target(target: str) -> QiecTargetCapabilities:
    """Return the registered QIEC capability set for ``target``.

    The eight host-language backends lower the full stable core through the
    QIEC runtime ABI. Stan admits closed, monomorphic, effect-free scalar
    computations as user-defined functions: binds, shared value bindings,
    scalar primitives, conditionals, calls, and recursion. BUGS and JAGS admit only
    ``return``/``bind`` computations, which they inline. Unknown targets
    safely retain declaration metadata only.

    Parameters
    ----------
    target
        The target name, with or without the ``qvr-`` prefix.

    Returns
    -------
    QiecTargetCapabilities
        The features the target preserves.
    """
    normalized = target.removeprefix("qvr-").lower()
    if normalized in {
        "pyro",
        "numpyro",
        "pymc",
        "edward2",
        "turing",
        "gen",
        "webppl",
        "church",
    }:
        return _GENERIC_RUNTIME
    if normalized == "stan":
        return _STAN_STATIC
    if normalized in {"bugs", "jags"}:
        return _GRAPH_STATIC
    return _DECLARATION_ONLY


def _required_features(computation: IRQiecNamedComputation) -> set[QiecFeature]:
    """The features a computation's signature and body need a target to have.

    Parameters
    ----------
    computation
        The named computation to inspect.

    Returns
    -------
    set[QiecFeature]
        Every feature the signature or body uses. Recursion is a property
        of the call graph rather than of one body, so it is added by
        :func:`analyze_qiec_capabilities`, which sees the whole module.
    """
    required: set[QiecFeature] = {"return"}
    if computation.parameters:
        required.add("named-parameter")
    if computation.type.effects.entries:
        required.add("effectful-row")
    if computation.type.effects.tail is not None:
        required.add("open-effect-row")
    for binder in computation.telescope:
        if isinstance(binder, IRQiecTypeBinder):
            required.add("type-polymorphism")
        elif isinstance(binder, IRQiecIndexBinder):
            required.add("index-polymorphism")
        else:
            required.add("effect-polymorphism")
    if not _is_scalar_type(computation.type.result):
        required.add("non-scalar-result")
    if any(not _is_scalar_type(parameter.type) for parameter in computation.parameters):
        required.add("non-scalar-parameter")
    required.update(_body_features(computation.body))
    return required


def _body_features(node: IRQiecComputation) -> set[QiecFeature]:
    """The features one computation body needs a target to have.

    Parameters
    ----------
    node
        The body to walk.

    Returns
    -------
    set[QiecFeature]
        Every feature a term in the body uses.
    """
    required: set[QiecFeature] = set()

    def value(item: IRQiecValue) -> None:
        if isinstance(item, IRQiecLiteralValue) and isinstance(
            item.value, IRQiecTupleLiteral
        ):
            required.add("structured-value")
        elif isinstance(item, IRQiecEvidenceValue):
            required.add("evidence")
        elif isinstance(item, IRQiecTransportValue):
            required.update(("evidence", "transport"))
            value(item.value)
        elif isinstance(item, IRQiecAttachmentRef):
            required.add("attachment")
        elif isinstance(item, IRQiecConstructorValue):
            required.add("structured-value")
            for field in item.fields:
                value(field)
        elif isinstance(item, IRQiecPrimitiveApplication):
            required.add(_primitive_feature(item.name))
            for argument in item.arguments:
                value(argument)
        elif isinstance(item, IRQiecTupleValue):
            required.add("tuple")
            for component in item.items:
                value(component)
        elif isinstance(item, IRQiecProjection):
            required.add("tuple")
            value(item.value)
        elif isinstance(item, IRQiecTensorValue):
            required.add("tensor")
            for entry in item.items:
                value(entry)
        elif isinstance(item, IRQiecDistributionValue):
            required.add("distribution")
            if item.plate.batch or item.plate.event:
                required.add("plate")
            for argument in item.arguments:
                value(argument.value)
        elif isinstance(item, IRQiecLogDensity):
            required.add("log-density")
            if item.batch:
                required.add("plate")
            value(item.sampleable)
            value(item.value)
        elif isinstance(item, IRQiecSiteValue):
            required.add("site")
        elif isinstance(item, IRQiecGather):
            required.add("gather")
            value(item.value)
            value(item.index)
        elif isinstance(item, IRQiecWeightSum):
            required.add("log-density")
            value(item.value)
        elif isinstance(item, IRQiecSegmentSum):
            required.add("segment-sum")
            value(item.value)
            value(item.index)
        elif isinstance(item, IRQiecKernelMatrix):
            required.add("kernel-matrix")
            value(item.inputs)
        elif isinstance(item, IRQiecAffineMap):
            required.add("affine-map")
            value(item.weight)
            value(item.bias)
            for source in item.sources:
                value(source)
        elif isinstance(item, IRQiecTableMap):
            required.add("table-map")
            value(item.table)
            value(item.index)
        elif isinstance(item, IRQiecReduction):
            required.add("reduction")
            value(item.value)
        elif isinstance(item, IRQiecRowwise):
            required.add("rowwise")
            value(item.value)
        elif isinstance(item, IRQiecComprehension):
            required.add("comprehension")
            value(item.body)
        elif isinstance(item, IRQiecLetValue):
            required.add("let")
            for binding in item.bindings:
                if not _is_scalar_type(binding.binder.type):
                    required.add("non-scalar-parameter")
                value(binding.value)
            value(item.body)

    def visit(item: IRQiecComputation) -> None:
        if isinstance(item, IRQiecReturn):
            value(item.value)
        elif isinstance(item, IRQiecBind):
            required.add("bind")
            for step in item.steps:
                if not _is_scalar_type(step.binder.type):
                    required.add("non-scalar-parameter")
                visit(step.first)
            visit(item.then)
        elif isinstance(item, IRQiecPerform):
            required.add("perform")
            if item.request.effect.name == "Compute":
                required.add("neural-attachment")
            for argument in item.request.arguments:
                value(argument)
        elif isinstance(item, IRQiecHandle):
            required.add("handle")
            visit(item.computation)
        elif isinstance(item, IRQiecCase):
            required.add("case")
            value(item.scrutinee)
            for branch in item.branches:
                visit(branch.body)
        elif isinstance(item, IRQiecIf):
            required.add("if")
            value(item.condition)
            visit(item.then)
            visit(item.otherwise)
        elif isinstance(item, IRQiecCall):
            required.add("call")
            for argument in item.arguments:
                value(argument)
        elif isinstance(item, IRQiecResume):
            required.add("resume")
            value(item.value)
        elif isinstance(item, IRQiecNewInstance):
            required.add("local-instance")
            visit(item.body)

    visit(node)
    return required


def _primitive_feature(name: str) -> QiecFeature:
    """The capability feature a primitive's registry tag names.

    Parameters
    ----------
    name
        The primitive's nominal name.

    Returns
    -------
    QiecFeature
        The registry's capability for the primitive. An unknown name is
        reported as ``arithmetic`` here only so the analysis can name the
        computation; the kernel has already rejected the module if the
        name is not in the registry.
    """
    signature = PRIMITIVES.get(name)
    return cast(QiecFeature, signature.capability if signature else "arithmetic")


def _callees(node: IRQiecComputation) -> set[str]:
    """The identities of every computation a body calls.

    Parameters
    ----------
    node
        The body to walk.

    Returns
    -------
    set[str]
        The callee identities, as :attr:`IRQiecId.text`.
    """
    out: set[str] = set()
    if isinstance(node, IRQiecCall):
        out.add(node.callee.text)
    elif isinstance(node, IRQiecBind):
        for step in node.steps:
            out.update(_callees(step.first))
        out.update(_callees(node.then))
    elif isinstance(node, IRQiecHandle):
        out.update(_callees(node.computation))
    elif isinstance(node, IRQiecCase):
        for branch in node.branches:
            out.update(_callees(branch.body))
    elif isinstance(node, IRQiecIf):
        out.update(_callees(node.then))
        out.update(_callees(node.otherwise))
    elif isinstance(node, IRQiecNewInstance):
        out.update(_callees(node.body))
    return out


def _recursive_computations(module: IRQiecModule) -> frozenset[str]:
    """The computations on a cycle of the module's call graph.

    Parameters
    ----------
    module
        The module whose named computations form the graph.

    Returns
    -------
    frozenset[str]
        The identity of every computation that can reach itself through
        calls, directly or through other computations. Tarjan's algorithm
        finds the strongly connected components; a component is recursive
        when it has more than one member or its single member calls itself.
    """
    graph = {
        computation.id.text: _callees(computation.body)
        for computation in module.computations
    }
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    recursive: set[str] = set()
    counter = 0

    def strongconnect(root: str) -> None:
        nonlocal counter
        # An explicit work list rather than recursion, so a long call chain
        # is bounded by memory rather than by the host's call depth.
        work: list[tuple[str, list[str]]] = [(root, sorted(graph.get(root, ())))]
        index[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            if pending:
                successor = pending.pop(0)
                if successor not in graph:
                    continue
                if successor not in index:
                    index[successor] = lowlink[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, sorted(graph[successor])))
                elif successor in on_stack:
                    lowlink[node] = min(lowlink[node], index[successor])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in graph[node]:
                    recursive.update(component)

    for name in graph:
        if name not in index:
            strongconnect(name)
    return frozenset(recursive)


def _is_scalar_type(type_: IRQiecStatic) -> bool:
    """Whether the static PPL subset can bind ``type_`` as one scalar."""
    return (
        isinstance(type_, IRQiecTypeApplication)
        and not type_.arguments
        and type_.constructor.name in {"Bool", "Int", "Real"}
    )


def analyze_qiec_capabilities(
    module_or_ir: QiecModule | IRQiecModule | None,
    target: str,
    *,
    capabilities: QiecTargetCapabilities | None = None,
) -> tuple[QiecCapabilityDiagnostic, ...]:
    """Report every QIEC feature ``target`` cannot preserve.

    Parameters
    ----------
    module_or_ir
        The checked module or its IR projection; ``None`` reports nothing.
    target
        The target name.
    capabilities
        The feature set to check against, else the target's registered
        one.

    Returns
    -------
    tuple[QiecCapabilityDiagnostic, ...]
        One diagnostic per feature a computation or handler needs that
        the target lacks. Program computations and their marginal
        helpers are excluded: a renderer emits them from the program's
        target plan rather than through the host runtime.
    """
    if module_or_ir is None:
        return ()
    module = (
        module_or_ir
        if isinstance(module_or_ir, IRQiecModule)
        else lower_qiec_ir(module_or_ir)
    )
    supported = capabilities or capabilities_for_target(target)
    diagnostics: list[QiecCapabilityDiagnostic] = list(
        host_runtime_capability_diagnostics(
            module,
            target,
            capabilities=supported,
        )
    )
    if not supported.supports("declarations") and (
        module.index_sorts or module.families or module.effects or module.handlers
    ):
        diagnostics.append(
            QiecCapabilityDiagnostic(
                target=target,
                feature="declarations",
                detail="cannot preserve its typed declarations",
            )
        )
    grade_features = {
        "0": "zero-resumption",
        "aff": "affine-resumption",
        "1": "linear-resumption",
        "omega": "unrestricted-resumption",
    }
    recursive = _recursive_computations(module)
    # A program's computation and its marginal helpers are rendered from
    # the target plan the renderer builds for the program, and a program
    # another program draws from is planned in place, so the host
    # runtime is never asked to carry them.
    programs = module.program_computations()
    for computation in module.computations:
        if _is_deduction_origin(computation.origin) or _is_structural_origin(
            computation.origin
        ):
            # These computations have one canonical entry-level diagnostic,
            # selected by ``host_runtime_capability_diagnostics``. Walking
            # every generated helper would make the editor and transpiler
            # disagree about which source construct is unsupported.
            continue
        if computation.id.text in programs:
            continue
        required = _required_features(computation)
        if computation.id.text in recursive:
            required.add("recursion")
        handler_ids = _handled_ids(computation.body)
        for handler in module.handlers:
            if handler.id.text not in handler_ids:
                continue
            required.update(
                cast(QiecFeature, grade_features[clause.grade])
                for clause in handler.clauses
            )
            if handler.implementation == "authored":
                required.add("authored-handler")
                for clause in handler.clauses:
                    if clause.body is not None:
                        required.update(_body_features(clause.body))
                if handler.return_clause is not None:
                    required.update(_body_features(handler.return_clause.body))
        for feature in sorted(required):
            if supported.supports(feature):
                continue
            diagnostics.append(
                QiecCapabilityDiagnostic(
                    target=target,
                    feature=feature,
                    computation=computation.name,
                    origin=computation.origin,
                    detail="has no semantics-preserving lowering for that feature",
                )
            )
    return tuple(diagnostics)


def _is_structural_origin(origin: SourceOrigin | IRQiecSourceOrigin) -> bool:
    """Whether an origin belongs to a structural neural declaration."""

    path = origin.structural_path
    return bool(path) and path[0] == "structural"


def _reachable_ir_computations(
    module: IRQiecModule,
    roots: set[str],
) -> frozenset[str]:
    """Close ``roots`` over the IR computation call graph."""

    graph = {
        computation.id.text: _callees(computation.body)
        for computation in module.computations
    }
    found: set[str] = set()
    pending = list(roots)
    while pending:
        identity = pending.pop()
        if identity in found or identity not in graph:
            continue
        found.add(identity)
        pending.extend(graph[identity])
    return frozenset(found)


def _host_runtime_root_ids(module: IRQiecModule) -> frozenset[str]:
    """Select the search and neural entries a target must preserve.

    Program modules retain every structural declaration because those
    declarations are independently executable. Standalone structural modules
    diagnose their loss entries, falling back to their components when no loss
    exists. Standalone deductions diagnose only their public ``__run`` entry.
    """

    structural = {
        computation.id.text
        for computation in module.computations
        if _is_structural_origin(computation.origin)
    }
    if module.entries:
        programs = {identity.text for identity in module.programs}
        roots = _reachable_ir_computations(module, programs)
        return frozenset((*roots, *structural))
    deduction_roots = {
        computation.id.text
        for computation in module.computations
        if _is_deduction_origin(computation.origin)
        and computation.name.endswith("__run")
    }
    structural_losses = {
        computation.id.text
        for computation in module.computations
        if computation.origin.structural_path[:2] == ("structural", "loss")
    }
    return frozenset(
        (*deduction_roots, *(structural_losses if structural_losses else structural))
    )


def _core_host_runtime_root_ids(module: QiecModule) -> frozenset[StableId]:
    """Select host-runtime roots without projecting the core into IR."""

    structural = {
        computation.id
        for computation in module.computations
        if _is_structural_origin(computation.origin)
    }
    if module.entries:
        return frozenset((*m.reachable_computations(module), *structural))
    deduction_roots = {
        computation.id
        for computation in module.computations
        if _is_deduction_origin(computation.origin)
        and computation.name.endswith("__run")
    }
    structural_losses = {
        computation.id
        for computation in module.computations
        if computation.origin.structural_path[:2] == ("structural", "loss")
    }
    return frozenset(
        (*deduction_roots, *(structural_losses if structural_losses else structural))
    )


def host_runtime_capability_diagnostics(
    module_or_ir: QiecModule | IRQiecModule,
    target: str,
    *,
    capabilities: QiecTargetCapabilities | None = None,
) -> tuple[QiecCapabilityDiagnostic, ...]:
    """Report unavailable search and neural runtime boundaries.

    This is the shared entry-root policy used by transpilation, the CLI, the
    REPL, the TUI, and the language server. Keeping it beside the general
    capability analyzer prevents those surfaces from diagnosing different
    generated helpers for the same source declaration.

    Parameters
    ----------
    module_or_ir
        The checked module or its IR projection.
    target
        The target name.
    capabilities
        The feature set to check against, else the target's registered
        one.

    Returns
    -------
    tuple[QiecCapabilityDiagnostic, ...]
        One diagnostic per search or neural host-runtime boundary an entry
        root reaches that the target cannot provide.
    """

    supported = capabilities or capabilities_for_target(target)
    if isinstance(module_or_ir, QiecModule):
        candidates = tuple(
            computation
            for computation in module_or_ir.computations
            if _is_deduction_origin(computation.origin)
            or _is_structural_origin(computation.origin)
        )
        if not candidates:
            return ()
        roots = _core_host_runtime_root_ids(module_or_ir)
        diagnostics: list[QiecCapabilityDiagnostic] = []
        for computation in candidates:
            if computation.id not in roots:
                continue
            feature: QiecFeature | None = None
            detail = "has no semantics-preserving lowering for that feature"
            if _is_deduction_origin(computation.origin) and computation.name.endswith(
                "__run"
            ):
                feature = "search"
                detail = "has no search runtime to enumerate its derivations"
            elif _is_structural_origin(computation.origin):
                feature = "neural-attachment"
            if feature is None or supported.supports(feature):
                continue
            diagnostics.append(
                QiecCapabilityDiagnostic(
                    target=target,
                    feature=feature,
                    computation=computation.name,
                    origin=cast(IRQiecSourceOrigin, _convert(computation.origin)),
                    detail=detail,
                )
            )
        return tuple(diagnostics)

    module = module_or_ir
    candidates = tuple(
        computation
        for computation in module.computations
        if _is_deduction_origin(computation.origin)
        or _is_structural_origin(computation.origin)
    )
    # Nearly every transpiled module has only ordinary QIEC computations.
    # Avoid decoding its (potentially large) Didactic term graph merely to
    # discover that no host-runtime boundary exists. This guard keeps the
    # shared CLI/LSP/transpiler check constant-time on that common path.
    if not candidates:
        return ()
    roots = _host_runtime_root_ids(module)
    diagnostics: list[QiecCapabilityDiagnostic] = []
    for computation in candidates:
        if computation.id.text not in roots:
            continue
        feature: QiecFeature | None = None
        detail = "has no semantics-preserving lowering for that feature"
        if _is_deduction_origin(computation.origin) and computation.name.endswith(
            "__run"
        ):
            feature = "search"
            detail = "has no search runtime to enumerate its derivations"
        elif _is_structural_origin(computation.origin):
            feature = "neural-attachment"
        if feature is None or supported.supports(feature):
            continue
        diagnostics.append(
            QiecCapabilityDiagnostic(
                target=target,
                feature=feature,
                computation=computation.name,
                origin=computation.origin,
                detail=detail,
            )
        )
    return tuple(diagnostics)


def _handled_ids(node: IRQiecComputation) -> set[str]:
    out: set[str] = set()
    if isinstance(node, IRQiecHandle):
        out.add(node.handler.text)
        out.update(_handled_ids(node.computation))
    elif isinstance(node, IRQiecBind):
        for step in node.steps:
            out.update(_handled_ids(step.first))
        out.update(_handled_ids(node.then))
    elif isinstance(node, IRQiecCase):
        for branch in node.branches:
            out.update(_handled_ids(branch.body))
    elif isinstance(node, IRQiecIf):
        out.update(_handled_ids(node.then))
        out.update(_handled_ids(node.otherwise))
    elif isinstance(node, IRQiecNewInstance):
        out.update(_handled_ids(node.body))
    return out


__all__ = [
    "IRQiecId",
    "IRQiecSourceOrigin",
    "IRQiecDynamicAddressFrame",
    "IRQiecSiteProvenance",
    "IRQiecKind",
    "IRQiecTypeKind",
    "IRQiecEffectKind",
    "IRQiecRowKind",
    "IRQiecArrowKind",
    "IRQiecIndexSort",
    "IRQiecNatSort",
    "IRQiecShapeSort",
    "IRQiecContextSort",
    "IRQiecUserIndexSort",
    "IRQiecBinder",
    "IRQiecTypeBinder",
    "IRQiecIndexBinder",
    "IRQiecEffectBinder",
    "IRQiecStatic",
    "IRQiecIndexVariable",
    "IRQiecIndexLiteral",
    "IRQiecIndexConstructor",
    "IRQiecShapeIndex",
    "IRQiecTypeVariable",
    "IRQiecTypeConstructor",
    "IRQiecTypeApplication",
    "IRQiecFunctionType",
    "IRQiecEqualityType",
    "IRQiecEffectVariable",
    "IRQiecEffectRef",
    "IRQiecLocal",
    "IRQiecRowVariable",
    "IRQiecRowEntry",
    "IRQiecEffectRow",
    "IRQiecComputationType",
    "IRQiecEvidence",
    "IRQiecReflexivity",
    "IRQiecBranchGiven",
    "IRQiecLiteral",
    "IRQiecNullLiteral",
    "IRQiecBoolLiteral",
    "IRQiecIntLiteral",
    "IRQiecFloatLiteral",
    "IRQiecStringLiteral",
    "IRQiecBytesLiteral",
    "IRQiecTupleLiteral",
    "IRQiecValue",
    "IRQiecVar",
    "IRQiecLiteralValue",
    "IRQiecConstructorValue",
    "IRQiecEvidenceValue",
    "IRQiecAttachmentRef",
    "IRQiecPrimitiveApplication",
    "IRQiecTupleValue",
    "IRQiecTensorValue",
    "IRQiecProjection",
    "IRQiecNamedArgument",
    "IRQiecPlateAxis",
    "IRQiecPlateShape",
    "IRQiecDistributionValue",
    "IRQiecLogDensity",
    "IRQiecGather",
    "IRQiecWeightSum",
    "IRQiecSegmentSum",
    "IRQiecKernelMatrix",
    "IRQiecAffineMap",
    "IRQiecTableMap",
    "IRQiecSiteValue",
    "IRQiecReduction",
    "IRQiecRowwise",
    "IRQiecComprehension",
    "IRQiecLetBinding",
    "IRQiecLetValue",
    "IRQiecTransportValue",
    "IRQiecEffectRequest",
    "IRQiecComputation",
    "IRQiecReturn",
    "IRQiecBindStep",
    "IRQiecBind",
    "IRQiecPerform",
    "IRQiecHandle",
    "IRQiecCaseMotive",
    "IRQiecCaseBranch",
    "IRQiecCase",
    "IRQiecIf",
    "IRQiecCall",
    "IRQiecResume",
    "IRQiecNewInstance",
    "IRQiecFieldDef",
    "IRQiecFamilyDecl",
    "IRQiecConstructorDecl",
    "IRQiecArgumentDef",
    "IRQiecOperationDef",
    "IRQiecEffectDef",
    "IRQiecHandlerClauseDef",
    "IRQiecHandlerReturnClauseDef",
    "IRQiecHandlerDef",
    "IRQiecNamedEffectInstance",
    "IRQiecNamedComputation",
    "IRQiecProgramParameter",
    "IRQiecProgramSite",
    "IRQiecProgramEntry",
    "IRQiecModule",
    "QiecFeature",
    "QiecTargetCapabilities",
    "QiecCapabilityDiagnostic",
    "analyze_qiec_capabilities",
    "capabilities_for_target",
    "host_runtime_capability_diagnostics",
    "lower_qiec_ir",
]
