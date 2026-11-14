"""Stable value and computation terms for QIEC.

Terms contain no Python callables.  Runtime values and handler bodies cross
the boundary only through stable attachment and handler identifiers.
"""

from __future__ import annotations

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx

from typing import Literal

from quivers.qiec.effects import EffectRequest, EffectRow
from quivers.qiec.evidence import EqualityEvidence
from quivers.qiec.identifiers import (
    AttachmentId,
    ComputationId,
    ConstructorId,
    DistributionId,
    EffectInstanceId,
    HandlerId,
    PrimitiveId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.kinds import Telescope
from quivers.qiec.types import EffectRef, IndexTerm, StaticArgument, TypeExpr


class Local(QiecModel):
    """A typed binder for a runtime value.

    Locals are compared by identity of name and type together, so two
    binders of the same name at different types are distinct.

    Parameters
    ----------
    name
        The binder's display name.
    type
        The type of every value bound to it.
    """

    name: str
    type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class Var(QiecModel):
    """Reference to a bound local.

    Parameters
    ----------
    local
        The binder being referenced.
    tag
        The serialization discriminator; always ``"var"``.
    """

    local: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["var"] = "var"


type LiteralData = None | bool | int | float | str | bytes | tuple[LiteralData, ...]
"""The host data a literal carries: a scalar, bytes, or a tuple of literal data."""


class LiteralValue(QiecModel):
    """A literal host value at a primitive or tuple type.

    Parameters
    ----------
    value
        The literal's data: ``None``, a bool, int, float, str, bytes, or a
        tuple of literal data.
    type
        The literal's type.
    tag
        The serialization discriminator; always ``"literal"``.
    """

    value: LiteralData = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["literal"] = "literal"


class ConstructorValue(QiecModel):
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
    tag
        The serialization discriminator; always ``"constructor"``.
    """

    constructor: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    static_arguments: tuple[StaticArgument, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    fields: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["constructor"] = "constructor"


class EvidenceValue(QiecModel):
    """Kernel-checked equality evidence used as a value.

    Parameters
    ----------
    evidence
        The evidence term.
    tag
        The serialization discriminator; always ``"evidence"``.
    """

    evidence: EqualityEvidence = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["evidence"] = "evidence"


class AttachmentRef(QiecModel):
    """A stable reference to a runtime-owned host value.

    Parameters
    ----------
    attachment
        The identity under which a runtime provider binds the host value.
    type
        The type the bound value must inhabit.
    tag
        The serialization discriminator; always ``"attachment"``.
    """

    attachment: AttachmentId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["attachment"] = "attachment"


class TransportValue(QiecModel):
    """Transport a value along kernel-checked equality evidence.

    Parameters
    ----------
    evidence
        The equality the value is carried across.
    value
        The value at the equality's left type.
    target_type
        The type the transported value inhabits, which the equality's right
        side determines.
    tag
        The serialization discriminator; always ``"transport"``.
    """

    evidence: EqualityEvidence = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    target_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["transport"] = "transport"


class PrimitiveApplication(QiecModel):
    """Application of a pure primitive from the closed registry.

    The primitive is named nominally and its signature is fixed, so the
    checker validates the arguments against the registry and a backend
    renders the name it knows or refuses the capability; nothing is
    resolved by inspecting a runtime value.

    Parameters
    ----------
    primitive
        The stable identity of the primitive applied.
    name
        The primitive's nominal name, as the registry spells it.
    arguments
        The value arguments, one per parameter.
    result_type
        The primitive's result type.
    origin
        The application's source location.
    tag
        The serialization discriminator; always ``"primitive"``.
    """

    primitive: PrimitiveId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    arguments: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["primitive"] = "primitive"


class TupleValue(QiecModel):
    """Construction of a finite product from its components.

    Parameters
    ----------
    items
        The component values, in order.
    result_type
        The product type of the components.
    tag
        The serialization discriminator; always ``"tuple"``.
    """

    items: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["tuple"] = "tuple"


class TensorValue(QiecModel):
    """Construction of a tensor from its entries along the outermost axis.

    A rank-one tensor lists its elements; a tensor of higher rank lists
    the tensors one rank lower that are its slices, all of one shape, so
    a matrix literal is a tensor of row tensors.

    Parameters
    ----------
    items
        The entries along the outermost axis, in order.
    result_type
        The ``Tensor`` type constructed, whose leading dimension is the
        number of entries.
    tag
        The serialization discriminator; always ``"tensor"``.
    """

    items: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["tensor"] = "tensor"


class Projection(QiecModel):
    """Selection of one component of a finite product.

    Parameters
    ----------
    value
        The product value projected from.
    position
        The zero-based component selected.
    result_type
        The selected component's type.
    tag
        The serialization discriminator; always ``"projection"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    position: int
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["projection"] = "projection"


class PlateAxis(QiecModel):
    """One axis of a plated distribution.

    Parameters
    ----------
    name
        The axis's source name, the object it ranges over.
    size
        The axis's extent, an index term of the nat sort.
    tag
        The serialization discriminator; always ``"plate_axis"``.
    """

    name: str
    size: IndexTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["plate_axis"] = "plate_axis"


class PlateShape(QiecModel):
    """The plate a distribution is constructed over.

    Batch axes replicate the distribution independently, one draw per
    position, each scored on its own; event axes join the draws along
    them into one event, scored together. The trailing event axes are
    the family's own event dimensions when it has any; any leading ones
    extend a scalar family's event.

    Parameters
    ----------
    batch
        The independent replication axes, outermost first.
    event
        The joint axes, outermost first.
    tag
        The serialization discriminator; always ``"plate_shape"``.
    """

    batch: tuple[PlateAxis, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    event: tuple[PlateAxis, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["plate_shape"] = "plate_shape"

    @property
    def empty(self) -> bool:
        """Whether the plate has no axes at all.

        Returns
        -------
        bool
            ``True`` when neither batch nor event axes are declared.
        """
        return not self.batch and not self.event


class DistributionValue(QiecModel):
    """Construction of a distribution from a family of the semantic registry.

    The family is named nominally and its parameters are supplied by name,
    so the term records which parameterization the source used and a
    backend renders the spelling it knows without inspecting values. A
    plate replicates the family over batch axes and joins draws along
    event axes; a parameter may then be a tensor whose leading dimensions
    follow a suffix of the plate's axes and broadcast over the rest.

    Parameters
    ----------
    family
        The stable identity of the family applied.
    name
        The family's source name, as the registry spells it.
    arguments
        The parameters supplied, each named, in source order.
    result_type
        The ``Sampleable`` type of the constructed distribution.
    origin
        The construction's source location.
    plate
        The plate the distribution is constructed over; empty for one
        draw of the family.
    tag
        The serialization discriminator; always ``"distribution"``.
    """

    family: DistributionId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    arguments: tuple[tuple[str, Value], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    plate: PlateShape = dx.field(
        default=PlateShape(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["distribution"] = "distribution"


class Gather(QiecModel):
    """Selection along the outermost axis of a tensor.

    Parameters
    ----------
    value
        The tensor selected from.
    index
        An ``Int`` selecting one slice, or a ``Tensor[Int]`` selecting a
        slice per entry.
    result_type
        The selection's type: the slice type for an integer index, or
        a tensor of slices shaped like the index tensor.
    tag
        The serialization discriminator; always ``"gather"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    index: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["gather"] = "gather"


class WeightSum(QiecModel):
    """The total of a tensor of log weights.

    Parameters
    ----------
    value
        A ``Tensor[LogWeight]`` of any shape, or a ``LogWeight``.
    tag
        The serialization discriminator; always ``"weight_sum"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["weight_sum"] = "weight_sum"


class SegmentSum(QiecModel):
    """Per-group totals of a vector of log weights.

    Parameters
    ----------
    value
        A ``Tensor[LogWeight]([n])``.
    index
        A ``Tensor[Int]([n])`` naming each entry's group.
    groups
        The number of groups, an index term of the nat sort.
    result_type
        ``Tensor[LogWeight]([groups])``.
    tag
        The serialization discriminator; always ``"segment_sum"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    index: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    groups: IndexTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["segment_sum"] = "segment_sum"


class KernelMatrix(QiecModel):
    """A positive-definite covariance over input locations.

    Parameters
    ----------
    inputs
        A ``Tensor[Real]([n])`` of locations.
    kernel
        The kernel's name; ``"rbf"`` is the squared-exponential kernel.
    length_scale
        The kernel's length scale.
    jitter
        The diagonal added for numerical positive definiteness.
    result_type
        ``Tensor[Real]([n, n])``.
    tag
        The serialization discriminator; always ``"kernel_matrix"``.
    """

    inputs: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    kernel: str
    length_scale: float
    jitter: float
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["kernel_matrix"] = "kernel_matrix"


type ReductionOperator = Literal["sum", "mean", "max", "min", "logsumexp", "prod"]
"""The reductions of a whole tensor to one number."""

type RowwiseOperator = Literal["softmax", "log_softmax", "cumsum", "sort", "normalize"]
"""The operations along a tensor's last axis that keep its shape."""


class Reduction(QiecModel):
    """A number summarizing every entry of a tensor.

    Parameters
    ----------
    operator
        The reduction.
    value
        A tensor of reals, or of integers for ``sum``, ``max``, ``min``,
        and ``prod``.
    result_type
        The element type.
    tag
        The serialization discriminator; always ``"reduction"``.
    """

    operator: ReductionOperator = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["reduction"] = "reduction"


class Rowwise(QiecModel):
    """An operation along a tensor's last axis keeping its shape.

    Parameters
    ----------
    operator
        The operation.
    value
        A tensor of reals.
    result_type
        The tensor's own type.
    tag
        The serialization discriminator; always ``"rowwise"``.
    """

    operator: RowwiseOperator = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["rowwise"] = "rowwise"


class Comprehension(QiecModel):
    """A tensor built by evaluating a body at every index of an axis.

    Parameters
    ----------
    binder
        The ``Int`` local the body reads the index through.
    extent
        The axis's extent, an index term of the nat sort.
    body
        The entry at each index.
    result_type
        ``Tensor[T]([extent])`` for the body's type ``T``, or, when the
        body is itself a tensor, the tensor with ``extent`` prepended.
    tag
        The serialization discriminator; always ``"comprehension"``.
    """

    binder: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    extent: IndexTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["comprehension"] = "comprehension"


class LetBinding(QiecModel):
    """One binding of a :class:`LetValue`.

    Parameters
    ----------
    binder
        The local the bound value is read through.
    value
        The value bound, evaluated once.
    """

    binder: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class LetValue(QiecModel):
    """A pure value shared by name through sequential bindings.

    Each binding's value is evaluated once, in order, and is in scope in
    every later binding and in the body, so a term that reads one
    intermediate value several times holds it once. The bindings bind
    sequentially, as ``let*`` does: binding ``k`` may read binders
    ``0`` through ``k - 1``.

    Parameters
    ----------
    bindings
        The bindings, in evaluation order; at least one.
    body
        The value the term denotes, read with every binder in scope.
    result_type
        The body's type.
    tag
        The serialization discriminator; always ``"let"``.
    """

    bindings: tuple[LetBinding, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["let"] = "let"


SCALE_FLOOR = 1e-7
"""The floor an exponentiated scale head of an :class:`AffineMap` or
:class:`TableMap` is clamped at, which every backend applies alike.
"""


class AffineMap(QiecModel):
    """One head of an affine parameter map, ``W x + b`` on a row block.

    For ``i`` below ``rows`` the head's coordinate ``i`` is the sum over
    ``j`` of ``weight[row_offset + i, j] * x[j]`` plus
    ``bias[row_offset + i]``, where ``x`` concatenates the sources in
    order; ``exp`` exponentiates every coordinate afterwards, and
    ``exp_floor`` exponentiates and clamps each at the scale floor.

    Parameters
    ----------
    weight
        A ``Tensor[Real]([total_rows, columns])``.
    bias
        A ``Tensor[Real]([total_rows])``.
    sources
        The conditioning row's factors in order, each a ``Real`` or a
        ``Tensor[Real]([width])``.
    row_offset
        The first row of the head's block.
    rows
        The block's height.
    transform
        ``"identity"``, ``"exp"``, or ``"exp_floor"``.
    result_type
        ``Tensor[Real]([rows])``, or ``Real`` for a one-row head.
    tag
        The serialization discriminator; always ``"affine_map"``.
    """

    weight: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    bias: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sources: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    row_offset: int
    rows: int
    transform: Literal["identity", "exp", "exp_floor"]
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["affine_map"] = "affine_map"


class TableMap(QiecModel):
    """One head of a table-indexed parameter map, a row block of ``T[i]``.

    A kernel over a finite domain reads its parameters from a table
    with one row per element of the domain: for ``i`` below ``rows``
    the head's coordinate ``i`` is ``table[index, row_offset + i]``,
    with ``exp`` exponentiating every coordinate afterwards and
    ``exp_floor`` exponentiating and clamping each at the scale floor.

    Parameters
    ----------
    table
        A ``Tensor[Real]([entries, total_rows])``.
    index
        An ``Int`` naming the row, the element of the domain.
    row_offset
        The first column of the head's block.
    rows
        The block's width.
    transform
        ``"identity"``, ``"exp"``, or ``"exp_floor"``.
    result_type
        ``Tensor[Real]([rows])``, or ``Real`` for a one-column head.
    tag
        The serialization discriminator; always ``"table_map"``.
    """

    table: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    index: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    row_offset: int
    rows: int
    transform: Literal["identity", "exp", "exp_floor"]
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["table_map"] = "table_map"


class SiteValue(QiecModel):
    """A named sample site.

    A site is the address a probabilistic step is observed and traced
    under. Its label is the source name of the step, which is stable data;
    the dynamic address a run adds comes from the request that carries it.

    Parameters
    ----------
    label
        The site's source name.
    result_type
        The ``Site`` type of the value the site produces.
    tag
        The serialization discriminator; always ``"site"``.
    """

    label: str
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["site"] = "site"


class LogDensity(QiecModel):
    """Evaluation of a distribution's log density at a value.

    Parameters
    ----------
    sampleable
        The distribution, of type ``Sampleable[A]``.
    value
        The point evaluated, of type ``A``.
    origin
        The evaluation's source location.
    batch
        The batch axes the density is kept apart over: empty for one
        ``LogWeight`` totalling every draw, else the batch axes of the
        plated construction ``sampleable`` must be, for one weight per
        position.
    tag
        The serialization discriminator; always ``"log_density"``.
    """

    sampleable: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    batch: tuple[PlateAxis, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["log_density"] = "log_density"


type Value = (
    Var
    | LiteralValue
    | ConstructorValue
    | EvidenceValue
    | AttachmentRef
    | TransportValue
    | PrimitiveApplication
    | TupleValue
    | TensorValue
    | Projection
    | DistributionValue
    | LogDensity
    | SiteValue
    | Gather
    | WeightSum
    | SegmentSum
    | KernelMatrix
    | AffineMap
    | TableMap
    | Reduction
    | Rowwise
    | Comprehension
    | LetValue
)
"""A value term: every pure, effect-free term a computation may mention."""


class Return(QiecModel):
    """Return a value with no effects.

    Parameters
    ----------
    value
        The value produced.
    tag
        The serialization discriminator; always ``"return"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["return"] = "return"


class Bind(QiecModel):
    """Sequence two computations, binding the first's result in the second.

    Parameters
    ----------
    binder
        The local the first computation's result is bound to.
    first
        The computation run first.
    then
        The computation run second, with ``binder`` in scope.
    tag
        The serialization discriminator; always ``"bind"``.
    """

    binder: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    first: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    then: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["bind"] = "bind"


class Perform(QiecModel):
    """Perform one effect operation.

    Parameters
    ----------
    request
        The operation, its instance, static arguments, value arguments, and
        result type.
    tag
        The serialization discriminator; always ``"perform"``.
    """

    request: EffectRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["perform"] = "perform"


class Handle(QiecModel):
    """Run a computation under a handler installed for one effect instance.

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
    tag
        The serialization discriminator; always ``"handle"``.
    """

    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    handler: HandlerId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    computation: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    static_arguments: tuple[StaticArgument, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["handle"] = "handle"


class CaseMotive(QiecModel):
    """The result family of an indexed case expression.

    Parameters
    ----------
    indices
        The binders abstracting the scrutinee's indices in ``result_type``.
    result_type
        The type every branch must produce, over ``indices``.
    """

    indices: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class CaseBranch(QiecModel):
    """One stable GADT branch with no host-language closure.

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

    constructor: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    static_arguments: tuple[StaticArgument, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    fields: tuple[Local, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    scope: StaticScopeId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class Case(QiecModel):
    """Case analysis over an indexed family value.

    Parameters
    ----------
    scrutinee
        The value analyzed.
    motive
        The result family, over the scrutinee's indices.
    branches
        One branch per constructor considered; coverage is checked against
        the exact applied family.
    tag
        The serialization discriminator; always ``"case"``.
    """

    scrutinee: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    motive: CaseMotive = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    branches: tuple[CaseBranch, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["case"] = "case"


class Call(QiecModel):
    """Application of a named computation.

    The callee is a stable identifier rather than an inlined body, which
    is what lets a recursive or mutually recursive call graph serialize
    as a finite tree.

    Parameters
    ----------
    callee
        The stable identity of the computation called.
    name
        The callee's display name, for diagnostics and traces.
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
    tag
        The serialization discriminator; always ``"call"``.
    """

    callee: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    static_arguments: tuple[StaticArgument, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    arguments: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effects: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["call"] = "call"


class Resume(QiecModel):
    """Invocation of the enclosing handler clause's continuation.

    The resumption is not a value and cannot be stored, so it is a term
    of its own rather than a call to a bound name. That is what makes a
    clause's grade checkable by counting invocations along the paths
    through its body.

    Parameters
    ----------
    value
        The value returned to the suspended computation as the operation's
        result.
    origin
        The resumption's source location.
    tag
        The serialization discriminator; always ``"resume"``.
    """

    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["resume"] = "resume"


class NewInstance(QiecModel):
    """Lexically scoped allocation of an effect instance.

    The identity is derived from the module, the enclosing computation,
    the lexical path, and the applied interface, so the same allocation
    site yields the same instance on every run while two sites of the
    same interface stay distinct.

    Parameters
    ----------
    instance
        The scope-derived stable identity of the allocated instance.
    effect
        The applied interface the instance provides.
    body
        The computation within which the instance is in scope.
    origin
        The allocation's source location.
    tag
        The serialization discriminator; always ``"new_instance"``.
    """

    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effect: EffectRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["new_instance"] = "new_instance"


class If(QiecModel):
    """Branch on a Boolean value.

    Both branches must produce the same type; the row of the whole is
    the union of the branches' rows. This is what lets a recursive
    computation stop, since a value-level select would evaluate both
    branches and never terminate.

    Parameters
    ----------
    condition
        The Boolean value branched on.
    then
        The computation run when the condition holds.
    otherwise
        The computation run when it does not.
    tag
        The serialization discriminator; always ``"if"``.
    """

    condition: Value = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    then: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    otherwise: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["if"] = "if"


type Computation = (
    Return | Bind | Perform | Handle | Case | If | Call | Resume | NewInstance
)
"""A computation term: every term that may perform effects or control flow."""


__all__ = [
    "AffineMap",
    "AttachmentRef",
    "Bind",
    "Call",
    "Case",
    "CaseBranch",
    "CaseMotive",
    "Comprehension",
    "Computation",
    "ConstructorValue",
    "DistributionValue",
    "EvidenceValue",
    "Gather",
    "Handle",
    "If",
    "KernelMatrix",
    "LetBinding",
    "LetValue",
    "LiteralData",
    "LiteralValue",
    "Local",
    "LogDensity",
    "NewInstance",
    "Perform",
    "PlateAxis",
    "PlateShape",
    "PrimitiveApplication",
    "Projection",
    "Reduction",
    "ReductionOperator",
    "Resume",
    "Return",
    "Rowwise",
    "RowwiseOperator",
    "SCALE_FLOOR",
    "SegmentSum",
    "SiteValue",
    "TableMap",
    "TensorValue",
    "TransportValue",
    "TupleValue",
    "Value",
    "Var",
    "WeightSum",
]
