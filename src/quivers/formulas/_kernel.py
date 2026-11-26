"""The response kernel: a program body with named parameter slots.

The likelihood of a response is a *kernel*: a body that reads named parameter
slots (``mu``, ``sigma``, ``disc``) and ends in the observation of the
response. A built-in family is one way to write a kernel
(`quivers.formulas._family_kernel.family_kernel`); a custom family is another.

Each slot is fed by a `ParameterSource`: a linear predictor, a site with its
own prior, or a constant. A nonlinear expression, a stack of per-category
parameters, or a declared data column feeds a slot by *precomposition*: the
slot is removed from the slots with sources, the body computes it from new
slots and data, and the rest of the body is unchanged.

The body is a sequence of `KernelPart` values. One composer, `compose`, emits
every kernel by emitting its parts in order; it never asks which family the
kernel was built from. Each part states one construction as data: emit or bind
a slot, literal steps, or a family construction (ordinal thresholds,
category masses, the native arguments of the observation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    SampleStep,
    Statement,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import (
    _add,
    _bind,
    _factor,
    _index,
    _let,
    _var,
    row_zero,
    stack_rows,
)
from quivers.formulas._observation import Observation
from quivers.formulas._predictor import Predictor
from quivers.formulas._predictor_graph import PredictorGraph
from quivers.formulas._residual_law import ResidualLaw
from quivers.formulas._term import AxisDecl, InputDecl, OutputAxis, SiteDecl
from quivers.formulas._term_emission import ProgramEmission, emit_axis_part, emit_terms
from quivers.formulas.family import Family, Link
from quivers.formulas.formula import Formula

__all__: list[str] = []


class Completion(ValueModel):
    """The map from the free coordinates of a vector value to all of them.

    A categorical predictor has one coordinate per category but the
    reference, which is zero; a parameter without a reference has one per
    coordinate. The value over the full axis is bound before the link.

    Parameters
    ----------
    axis : str
        Full coordinate axis.
    size : int
        Number of coordinates of the full axis.
    reference : int or None
        Coordinate fixed at zero, or None.
    coordinates : tuple[int, ...]
        Full coordinate of each free coordinate, in free order.
    prefix : str
        Namespace of the intermediate bindings.
    """

    axis: str
    size: int
    reference: int | None = None
    coordinates: tuple[int, ...] = ()
    prefix: str = ""

    def steps(self, free: str, result: str) -> tuple[ProgramStep, ...]:
        """Bind the value over the full axis.

        Parameters
        ----------
        free : str
            Binding of the free coordinates over ``(Resp, free axis)``.
        result : str
            Binding of the value over ``(Resp, axis)``.

        Returns
        -------
        tuple[ProgramStep, ...]
            The zero row of the reference and the stack by coordinate.
        """
        zero = f"{self.prefix}categorical_zero_logit"
        cases: tuple[tuple[int, LetExprNode], ...] = (
            () if self.reference is None else ((self.reference, _var(zero)),)
        ) + tuple(
            (
                category,
                _factor(
                    (("response", "Resp"),),
                    body=_index(
                        free,
                        _var("response"),
                        LetExprLiteral(value=float(coordinate), integral=True),
                    ),
                ),
            )
            for coordinate, category in enumerate(self.coordinates)
        )
        return (
            row_zero(zero),
            *stack_rows(
                result,
                cases,
                ("category", self.axis),
                stem=f"{self.prefix}categorical_logits_by_category",
            ),
        )


class ValueShape(ValueModel):
    """Where a bound slot value varies.

    Parameters
    ----------
    rows : bool
        Whether the value is indexed by the response rows first.
    coordinates : bool
        Whether the value is indexed by a coordinate axis (after the rows when
        both).
    """

    rows: bool = False
    coordinates: bool = False

    def at(
        self, name: str, row: LetExprNode, coordinate: LetExprNode | None = None
    ) -> LetExprNode:
        """Read the value at one row and coordinate.

        Parameters
        ----------
        name : str
            Binding of the value.
        row : LetExprNode
            Row index.
        coordinate : LetExprNode or None
            Coordinate index; None for a scalar-event read.

        Returns
        -------
        LetExprNode
            The binding indexed by the axes it varies over.
        """
        indices = (
            *((row,) if self.rows else ()),
            *((coordinate,) if self.coordinates and coordinate is not None else ()),
        )
        return _index(name, *indices) if indices else _var(name)


class KernelArguments(ValueModel):
    """The native law of the observation and what is passed to it.

    Parameters
    ----------
    kernel : str
        Native or compilation-local law.
    values : tuple[str or int or float, ...]
        Bindings in the order of the law's parameters.
    parameters : tuple[str, ...] or None
        Parameter each value is passed to, when it is not the law's schema
        order; None for that order.
    event : str or None
        Event axis of a vector response.
    log_location : str or None
        Parameter that takes the linear predictor on the log scale in place
        of the location; None when the location is passed as it is.
    """

    kernel: str
    values: tuple[str | int | float, ...]
    parameters: tuple[str, ...] | None = None
    event: str | None = None
    log_location: str | None = None


class KernelPart(dx.TaggedUnion, discriminator="kind"):
    """One construction of a kernel body."""

    def emit(self, emission: KernelEmission) -> None:
        """Emit the part.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        raise NotImplementedError


class EmitSlot(KernelPart):
    """Emit the source or precomposition of a slot, binding it later.

    Parameters
    ----------
    slot : str
        Slot name.
    """

    kind: Literal["emit_slot"] = "emit_slot"
    slot: str

    def emit(self, emission: KernelEmission) -> None:
        """Emit the slot's source.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.emit_slot(self.slot)


class BindSlot(KernelPart):
    """Bind a slot, emitting its source first if it has not been.

    Parameters
    ----------
    slot : str
        Slot name.
    """

    kind: Literal["bind_slot"] = "bind_slot"
    slot: str

    def emit(self, emission: KernelEmission) -> None:
        """Bind the slot.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.bind_slot(self.slot)


class EmitSlotAxis(KernelPart):
    """Emit the part of a slot's predictor over another output axis.

    Parameters
    ----------
    slot : str
        Slot name.
    output : OutputAxis
        Axis of the part.
    binding : str
        Binding of the part over ``(Resp, axis)``.
    """

    kind: Literal["emit_slot_axis"] = "emit_slot_axis"
    slot: str
    output: OutputAxis
    binding: str

    def emit(self, emission: KernelEmission) -> None:
        """Emit the part.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        source = emission.sources.get(self.slot)
        if not isinstance(source, PredictorSource):
            raise ValueError(
                f"parameter {self.slot!r} has no predictor over {self.output.name!r}"
            )
        emit_axis_part(
            source.predictor, self.output, emission.terms, emission.steps, self.binding
        )


class Steps(KernelPart):
    """Literal steps, after the axes they range over.

    Parameters
    ----------
    objects : tuple[AxisDecl, ...]
        Axes declared first.
    steps : tuple[ProgramStep, ...]
        Steps in order.
    """

    kind: Literal["steps"] = "steps"
    objects: tuple[AxisDecl, ...] = ()
    steps: tuple[ProgramStep, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Declare the axes and append the steps.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        for axis in self.objects:
            emission.declare(axis)
        emission.steps.extend(self.steps)


class Precomposition(ValueModel):
    """A slot computed by the body from other slots and data.

    Parameters
    ----------
    parts : tuple[KernelPart, ...]
        Parts computing the value.
    value : LetExprNode or None
        Value the parts compute, on the stage named by ``stage``; None when
        the parts bind the binding of that stage themselves.
    stage : {"free", "eta", "name"}
        Stage of the slot the value enters at: the free coordinates of a
        vector value, the full value on the link scale, or the value itself.
    shape : ValueShape
        Where the computed value varies.
    """

    parts: tuple[KernelPart, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: LetExprNode | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    stage: Literal["free", "eta", "name"] = "eta"
    shape: ValueShape = ValueShape(rows=True)


class ParameterSlot(ValueModel):
    """A named parameter the kernel body reads.

    A slot's value passes through stages: the free coordinates of a vector
    value (``free``), their completion over the full axis, the value on the
    link scale (``eta``), and the value itself (``name``), bound by the
    inverse link or materialized over the rows.

    Parameters
    ----------
    name : str
        Binding the body reads.
    eta : str or None
        Binding of the value on the link scale; None for a slot fed only by
        a site or a constant.
    link : Link or None
        Inverse link from ``eta`` to ``name``; None when the body applies it.
    free : str or None
        Binding of the free coordinates of a vector value.
    completion : Completion or None
        Map from the free coordinates to the full axis; None binds ``eta``
        to the free coordinates.
    axis : str or None
        Axis a site of this slot is drawn over.
    row_binder : str or None
        Binder of a value materialized over the rows (``name`` indexed by
        the row); None for a linked value.
    precomposition : Precomposition or None
        How the body computes the slot, when no source feeds it.
    """

    name: str
    eta: str | None = None
    link: Link | None = None
    free: str | None = None
    completion: Completion | None = None
    axis: str | None = None
    row_binder: str | None = None
    precomposition: Precomposition | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class ParameterSource(dx.TaggedUnion, discriminator="kind"):
    """What feeds a parameter slot."""

    def emit(self, slot: ParameterSlot, emission: KernelEmission) -> None:
        """Emit the source of a slot, keeping a value to bind later.

        Parameters
        ----------
        slot : ParameterSlot
            Slot fed by the source.
        emission : KernelEmission
            Program being emitted.
        """
        raise NotImplementedError


class PredictorSource(ParameterSource):
    """A linear predictor.

    Parameters
    ----------
    predictor : Predictor
        Planned predictor of the slot.
    """

    kind: Literal["predictor"] = "predictor"
    predictor: Predictor = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    def emit(self, slot: ParameterSlot, emission: KernelEmission) -> None:
        """Emit the terms of the predictor.

        A vector predictor binds its free coordinates; a scalar one keeps the
        sum of its contributions to bind with the slot.

        Parameters
        ----------
        slot : ParameterSlot
            Slot fed by the predictor.
        emission : KernelEmission
            Program being emitted.
        """
        predictor = self.predictor
        output = predictor.output
        if output is not None:
            if slot.free is None:
                raise ValueError(
                    f"parameter {slot.name!r} has a vector predictor and no "
                    "binding of its free coordinates"
                )
            if slot.completion is not None:
                emission.declare(
                    AxisDecl(name=slot.completion.axis, size=slot.completion.size)
                )
            emission.declare(AxisDecl(name=output.name, size=output.size))
            emit_axis_part(predictor, output, emission.terms, emission.steps, slot.free)
            emission.pending(slot.name, None, ValueShape(rows=True, coordinates=True))
            return
        summands = emit_terms(predictor.part(None), emission.terms, emission.steps)
        emission.pending(
            slot.name,
            _add(*(_var(summand.name) for summand in summands)),
            ValueShape(rows=predictor.rowwise),
        )


class SiteSource(ParameterSource):
    """A site with its own prior.

    Parameters
    ----------
    site : SiteDecl
        The site, drawn under the slot's name.
    floor : float or None
        Lower bound the prior is restricted to, unless a configured prior
        already restricts it.
    """

    kind: Literal["site"] = "site"
    site: SiteDecl
    floor: float | None = None

    def emit(self, slot: ParameterSlot, emission: KernelEmission) -> None:
        """Draw the site.

        Parameters
        ----------
        slot : ParameterSlot
            Slot fed by the site.
        emission : KernelEmission
            Program being emitted.
        """
        emission.steps.append(emission.host.draw_site(self.site, self.floor))
        emission.bound(slot.name, ValueShape(coordinates=self.site.index is not None))


class ConstantSource(ParameterSource):
    """A constant.

    Parameters
    ----------
    value : float
        The value on the natural scale.
    """

    kind: Literal["constant"] = "constant"
    value: float

    def emit(self, slot: ParameterSlot, emission: KernelEmission) -> None:
        """Bind the constant.

        Parameters
        ----------
        slot : ParameterSlot
            Slot fed by the constant.
        emission : KernelEmission
            Program being emitted.
        """
        emission.steps.append(_let(slot.name, LetExprLiteral(value=float(self.value))))
        emission.bound(slot.name, ValueShape())


class ResponseKernel(ValueModel):
    """A program body with named parameter slots, ending in an observation.

    Parameters
    ----------
    slots : tuple[ParameterSlot, ...]
        Parameters the body reads, including those it computes by
        precomposition.
    objects : tuple[AxisDecl, ...]
        Axes declared before any group block or predictor.
    data : tuple[InputDecl, ...]
        Columns the body reads by name.
    head : tuple[KernelPart, ...]
        Parts emitted before the group blocks shared by the predictors.
    body : tuple[KernelPart, ...]
        Parts emitted after them, ending in the observation.
    response : str
        Observed site.
    row_axis : str
        Axis of the response rows.
    """

    slots: tuple[ParameterSlot, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    objects: tuple[AxisDecl, ...] = ()
    data: tuple[InputDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    head: tuple[KernelPart, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    body: tuple[KernelPart, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    response: str
    row_axis: str = "Resp"

    def slot(self, name: str) -> ParameterSlot:
        """Return one slot.

        Parameters
        ----------
        name : str
            Slot name.

        Returns
        -------
        ParameterSlot
            The slot.

        Raises
        ------
        KeyError
            If the kernel has no such slot.
        """
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise KeyError(f"the response kernel has no parameter {name!r}")

    def has_slot(self, name: str) -> bool:
        """Report whether the kernel reads a slot.

        Parameters
        ----------
        name : str
            Slot name.

        Returns
        -------
        bool
            True when the kernel has the slot.
        """
        return any(slot.name == name for slot in self.slots)

    @property
    def free_slots(self) -> tuple[ParameterSlot, ...]:
        """The slots a source feeds.

        Returns
        -------
        tuple[ParameterSlot, ...]
            Slots without a precomposition, in order.
        """
        return tuple(slot for slot in self.slots if slot.precomposition is None)

    def precompose(
        self,
        slot: str,
        precomposition: Precomposition,
        free: Sequence[ParameterSlot] = (),
        data: Sequence[InputDecl] = (),
    ) -> ResponseKernel:
        """Compute one slot by the body.

        The slot stops being fed by a source; ``precomposition`` gives its
        value from the slots in ``free`` (bound by its own parts) and the
        columns in ``data``.

        Parameters
        ----------
        slot : str
            Slot computed.
        precomposition : Precomposition
            Parts and value computing it.
        free : Sequence[ParameterSlot]
            New slots the value reads, bound by the precomposition's parts.
        data : Sequence[InputDecl]
            Columns the value reads.

        Returns
        -------
        ResponseKernel
            The kernel with the slot computed.

        Raises
        ------
        ValueError
            If the slot is unknown or already computed, or a new slot exists.
        """
        target = self.slot(slot)
        if target.precomposition is not None:
            raise ValueError(
                f"parameter {slot!r} has more than one predictor definition"
            )
        clashes = {item.name for item in free} & {item.name for item in self.slots}
        if clashes:
            raise ValueError(f"parameters {sorted(clashes)} are defined twice")
        return self.with_(
            slots=(
                *(
                    item.with_(precomposition=precomposition)
                    if item.name == slot
                    else item
                    for item in self.slots
                ),
                *free,
            ),
            data=_joined_inputs(self.data, data),
        )

    def with_prelude(
        self,
        free: Sequence[ParameterSlot],
        parts: Sequence[KernelPart],
        data: Sequence[InputDecl] = (),
    ) -> ResponseKernel:
        """Add slots and parts the body emits first.

        Parameters
        ----------
        free : Sequence[ParameterSlot]
            New slots, bound by ``parts``.
        parts : Sequence[KernelPart]
            Parts placed before the body.
        data : Sequence[InputDecl]
            Columns the parts read.

        Returns
        -------
        ResponseKernel
            The kernel with the prelude.

        Raises
        ------
        ValueError
            If a new slot exists.
        """
        clashes = {item.name for item in free} & {item.name for item in self.slots}
        if clashes:
            raise ValueError(f"parameters {sorted(clashes)} are defined twice")
        return self.with_(
            slots=(*self.slots, *free),
            body=(*parts, *self.body),
            data=_joined_inputs(self.data, data),
        )


def _joined_inputs(
    first: Sequence[InputDecl], second: Sequence[InputDecl]
) -> tuple[InputDecl, ...]:
    """Join two input lists, keeping the first declaration of a name.

    Parameters
    ----------
    first, second : Sequence[InputDecl]
        Inputs in order.

    Returns
    -------
    tuple[InputDecl, ...]
        Every name once.
    """
    seen = {item.name for item in first}
    return (*first, *(item for item in second if item.name not in seen))


class KernelInputs(ValueModel):
    """What a family kernel is built from, besides the family's declarations.

    Parameters
    ----------
    family : Family
        Response family.
    response : str
        Observed site.
    labels : tuple[str, ...]
        Component or category labels of a categorical or vector response;
        empty otherwise.
    categories : int
        Number of response categories; zero for a scalar response.
    predictors : tuple[Predictor, ...]
        Planned predictors of the location, the distributional parameters,
        and the nonlinear parameters.
    distributional : frozenset[str]
        Parameters with a formula of their own.
    fixed : dict[str, float]
        Parameters bound to constants.
    priors : dict[str, str]
        Configured priors by site.
    graph : PredictorGraph or None
        Nonlinear parameters, expressions, and roots.
    residual : ResidualLaw or None
        Residual law correlated across the rows.
    host : str or None
        Input holding a host contribution to the location.
    category_parameters : frozenset[str]
        Categories with a predictor of their own.
    trials : str or float or None
        Binding or constant of the trial counts.
    response_values : np.ndarray
        Training responses.
    """

    family: Family
    response: str
    labels: tuple[str, ...] = ()
    categories: int = 0
    predictors: tuple[Predictor, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    distributional: frozenset[str] = frozenset()
    fixed: dict[str, float] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    priors: dict[str, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    graph: PredictorGraph | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    residual: ResidualLaw | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    host: str | None = None
    category_parameters: frozenset[str] = frozenset()
    trials: str | float | None = None
    response_values: np.ndarray = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def predictor(self, prefix: str) -> Predictor:
        """Return the planned predictor of one namespace.

        Parameters
        ----------
        prefix : str
            Namespace: empty for the location.

        Returns
        -------
        Predictor
            The predictor.

        Raises
        ------
        KeyError
            If no predictor has the namespace.
        """
        for item in self.predictors:
            if item.prefix == prefix:
                return item
        raise KeyError(f"no predictor has the namespace {prefix!r}")


class KernelHost(Protocol):
    """What the parts of a kernel ask of the compiler emitting it."""

    def terms(
        self,
        statements: list[Statement],
        gathered: Mapping[tuple[str, str, str], LetExprNode],
    ) -> ProgramEmission:
        """Return the program the terms of the predictors are emitted into.

        Parameters
        ----------
        statements : list[Statement]
            Module declarations receiving axes.
        gathered : Mapping[tuple[str, str, str], LetExprNode]
            Group-level coefficients emitted ahead of the body.

        Returns
        -------
        ProgramEmission
            The program as the terms see it.
        """
        ...

    def declare(self, statements: list[Statement], axis: AxisDecl) -> None:
        """Declare a finite axis once.

        Parameters
        ----------
        statements : list[Statement]
            Module declarations.
        axis : AxisDecl
            Axis and its size.
        """
        ...

    def draw_site(self, site: SiteDecl, floor: float | None) -> SampleStep:
        """Draw a site from its configured or default prior.

        Parameters
        ----------
        site : SiteDecl
            Site with a default prior.
        floor : float or None
            Lower bound the prior is restricted to unless it already is.

        Returns
        -------
        SampleStep
            The draw.
        """
        ...

    def ordinal_thresholds(
        self, formula: Formula, statements: list[Statement], steps: list[ProgramStep]
    ) -> str:
        """Emit the ordinal thresholds.

        Parameters
        ----------
        formula : Formula
            Response categories and group coding.
        statements : list[Statement]
            Declarations receiving the threshold axes.
        steps : list[ProgramStep]
            Steps receiving the threshold draws and transformations.

        Returns
        -------
        str
            Binding of the thresholds; empty without a boundary.
        """
        ...

    def ordinal_contrasts(
        self,
        formula: Formula,
        cutpoints: str,
        steps: list[ProgramStep],
        emission: KernelEmission,
    ) -> None:
        """Bind the contrasts of each row and transition.

        Parameters
        ----------
        formula : Formula
            Response and predictors.
        cutpoints : str
            Binding of the thresholds.
        steps : list[ProgramStep]
            Steps receiving the contrasts.
        emission : KernelEmission
            Kernel being emitted.
        """
        ...

    def ordinal_logits(
        self,
        formula: Formula,
        cutpoints: str,
        statements: list[Statement],
        steps: list[ProgramStep],
        emission: KernelEmission,
    ) -> None:
        """Bind the category log masses of an ordinal law.

        Parameters
        ----------
        formula : Formula
            Response and predictors.
        cutpoints : str
            Binding of the thresholds.
        statements : list[Statement]
            Declarations receiving the category axis.
        steps : list[ProgramStep]
            Steps receiving the log masses.
        emission : KernelEmission
            Kernel being emitted.
        """
        ...

    def known_error_scale(
        self, formula: Formula, steps: list[ProgramStep], emission: KernelEmission
    ) -> None:
        """Bind the response scale under known errors.

        Parameters
        ----------
        formula : Formula
            Formula with an ``se()`` term.
        steps : list[ProgramStep]
            Steps receiving the scale.
        emission : KernelEmission
            Kernel being emitted.
        """
        ...

    def residual_correlation(
        self, steps: list[ProgramStep], axis: str, width: int, result: str
    ) -> None:
        """Draw a correlation factor over an event axis.

        Parameters
        ----------
        steps : list[ProgramStep]
            Steps receiving the draw and binding.
        axis : str
            Axis of the factor.
        width : int
            Number of coordinates.
        result : str
            Binding of the factor.
        """
        ...

    def baseline(
        self, formula: Formula, statements: list[Statement], steps: list[ProgramStep]
    ) -> None:
        """Emit the baseline hazard of a proportional-hazards family.

        Parameters
        ----------
        formula : Formula
            Formula with the event times.
        statements : list[Statement]
            Declarations receiving the basis axes.
        steps : list[ProgramStep]
            Steps receiving the baseline draws.
        """
        ...

    def observed(self, observation: Observation) -> None:
        """Record the observation of the response.

        Parameters
        ----------
        observation : Observation
            The observation written.
        """
        ...

    @property
    def response_laws(self) -> FamilyCatalog | None:
        """The compilation-local laws of the response.

        Returns
        -------
        FamilyCatalog or None
            The catalog, or None when the law is built in.
        """
        ...


class KernelEmission:
    """The program a kernel is emitted into.

    Parameters
    ----------
    host : KernelHost
        Compiler carrying the prior configuration.
    inputs : KernelInputs
        What the kernel was built from.
    kernel : ResponseKernel
        Kernel being emitted.
    sources : Mapping[str, ParameterSource]
        Source of each slot a source feeds.
    statements : list[Statement]
        Module declarations receiving axes.
    steps : list[ProgramStep]
        Program steps receiving the body.
    gathered : Mapping[tuple[str, str, str], LetExprNode]
        Group-level coefficients emitted ahead of the body.
    formula : Formula
        Response, its rows, and its modifiers.
    """

    def __init__(
        self,
        host: KernelHost,
        inputs: KernelInputs,
        kernel: ResponseKernel,
        sources: Mapping[str, ParameterSource],
        statements: list[Statement],
        steps: list[ProgramStep],
        gathered: Mapping[tuple[str, str, str], LetExprNode],
        formula: Formula,
    ) -> None:
        self.host = host
        self.inputs = inputs
        self.formula = formula
        self.kernel = kernel
        self.sources = dict(sources)
        self.statements = statements
        self.steps = steps
        self.terms = host.terms(statements, gathered)
        self._emitted: set[str] = set()
        self._values: dict[str, LetExprNode | None] = {}
        self._shapes: dict[str, ValueShape] = {}
        self.buffers: dict[str, list[ProgramStep]] = {}
        self.names: dict[str, str] = {}
        self.arguments: KernelArguments | None = None

    def declare(self, axis: AxisDecl) -> None:
        """Declare a finite axis once.

        Parameters
        ----------
        axis : AxisDecl
            Axis and its size.
        """
        self.host.declare(self.statements, axis)

    def pending(self, slot: str, value: LetExprNode | None, shape: ValueShape) -> None:
        """Keep the value of an emitted source until the slot is bound.

        Parameters
        ----------
        slot : str
            Slot name.
        value : LetExprNode or None
            Scalar value on the link scale; None for a vector value bound to
            the slot's free coordinates.
        shape : ValueShape
            Where the value varies.
        """
        self._values[slot] = value
        self._shapes[slot] = shape

    def bound(self, slot: str, shape: ValueShape) -> None:
        """Record a slot bound under its own name by its source.

        Parameters
        ----------
        slot : str
            Slot name.
        shape : ValueShape
            Where the value varies.
        """
        self._shapes[slot] = shape
        self._values.pop(slot, None)

    def shape(self, slot: str) -> ValueShape:
        """Return where a bound slot varies.

        Parameters
        ----------
        slot : str
            Slot name.

        Returns
        -------
        ValueShape
            Its shape.

        Raises
        ------
        KeyError
            If the slot is not bound yet.
        """
        if slot not in self._shapes:
            raise KeyError(f"parameter {slot!r} is read before it is bound")
        return self._shapes[slot]

    def value_at(
        self, slot: str, row: LetExprNode, coordinate: LetExprNode | None = None
    ) -> LetExprNode:
        """Read a bound slot at one row and coordinate.

        Parameters
        ----------
        slot : str
            Slot name.
        row : LetExprNode
            Row index.
        coordinate : LetExprNode or None
            Coordinate index.

        Returns
        -------
        LetExprNode
            The slot's binding indexed by the axes it varies over.
        """
        return self.shape(slot).at(slot, row, coordinate)

    def emit_slot(self, name: str) -> None:
        """Emit the source or precomposition of a slot once.

        Parameters
        ----------
        name : str
            Slot name.
        """
        if name in self._emitted:
            return
        self._emitted.add(name)
        slot = self.kernel.slot(name)
        if slot.precomposition is not None:
            for part in slot.precomposition.parts:
                part.emit(self)
            self._values[name] = slot.precomposition.value
            self._shapes[name] = slot.precomposition.shape
            return
        source = self.sources.get(name)
        if source is not None:
            source.emit(slot, self)

    def bind_slot(self, name: str) -> None:
        """Bind a slot through the stages after its source.

        Parameters
        ----------
        name : str
            Slot name.
        """
        self.emit_slot(name)
        slot = self.kernel.slot(name)
        precomposition = slot.precomposition
        if precomposition is not None and precomposition.stage == "name":
            return
        if name not in self._values:
            # A site or constant bound the slot, or a modifier feeds it.
            return
        value = self._values.pop(name)
        vector = (
            precomposition.stage == "free"
            if precomposition is not None
            else value is None
        )
        if vector:
            if slot.free is None or slot.eta is None:
                raise ValueError(f"parameter {name!r} has no vector stages")
            if precomposition is not None and value is not None:
                if value != _var(slot.free):
                    self.steps.append(_let(slot.free, value))
            if slot.completion is not None:
                self.steps.extend(slot.completion.steps(slot.free, slot.eta))
            else:
                self.steps.append(_let(slot.eta, _var(slot.free)))
        else:
            if slot.eta is None:
                raise ValueError(f"parameter {name!r} has no link-scale binding")
            assert value is not None
            if value != _var(slot.eta):
                self.steps.append(_let(slot.eta, value))
        if slot.row_binder is not None:
            row = _var(slot.row_binder)
            self.steps.append(
                _let(
                    slot.name,
                    _factor(
                        ((slot.row_binder, self.kernel.row_axis),),
                        body=self._shapes[name].at(slot.eta, row),
                    ),
                )
            )
            self._shapes[name] = ValueShape(rows=True)
        elif slot.link is not None:
            append_link(self.steps, slot.name, _var(slot.eta), slot.link)


def append_link(
    steps: list[ProgramStep], name: str, eta: LetExprNode, link: Link
) -> None:
    """Bind a parameter to its declared inverse link of a linear predictor.

    Parameters
    ----------
    steps : list[ProgramStep]
        Program body receiving the link's intermediates and the parameter.
    name : str
        Parameter being defined.
    eta : LetExprNode
        Linear predictor.
    link : Link
        Inverse link, emitted exactly as it declares itself.
    """
    for binding in link.intermediates:
        steps.append(
            _let(
                binding.name.format(name=name),
                _bind(binding.expr.format(eta="eta", name=name), eta=eta),
            )
        )
    steps.append(
        _let(name, _bind(link.inverse_expr.format(eta="eta", name=name), eta=eta))
    )


def compose(
    host: KernelHost,
    inputs: KernelInputs,
    kernel: ResponseKernel,
    sources: Mapping[str, ParameterSource],
    statements: list[Statement],
    steps: list[ProgramStep],
    gathered: Mapping[tuple[str, str, str], LetExprNode],
    formula: Formula,
) -> KernelEmission:
    """Emit the body of a kernel.

    Parameters
    ----------
    host : KernelHost
        Compiler carrying the prior configuration.
    inputs : KernelInputs
        What the kernel was built from.
    kernel : ResponseKernel
        Kernel to emit.
    sources : Mapping[str, ParameterSource]
        Source of each slot a source feeds.
    statements : list[Statement]
        Module declarations receiving axes.
    steps : list[ProgramStep]
        Program steps receiving the body.
    gathered : Mapping[tuple[str, str, str], LetExprNode]
        Group-level coefficients emitted ahead of the body.
    formula : Formula
        Response, its rows, and its modifiers.

    Returns
    -------
    KernelEmission
        The finished emission.

    Raises
    ------
    ValueError
        If a source names no slot of the kernel.
    """
    unknown = set(sources) - {slot.name for slot in kernel.free_slots}
    if unknown:
        raise ValueError(
            f"sources name no free parameter of the kernel: {sorted(unknown)}"
        )
    emission = KernelEmission(
        host, inputs, kernel, sources, statements, steps, gathered, formula
    )
    for part in kernel.body:
        part.emit(emission)
    return emission


def emit_head(
    host: KernelHost,
    inputs: KernelInputs,
    kernel: ResponseKernel,
    statements: list[Statement],
    steps: list[ProgramStep],
    formula: Formula,
) -> None:
    """Emit the parts of a kernel that come before the group blocks.

    Parameters
    ----------
    host : KernelHost
        Compiler carrying the prior configuration.
    inputs : KernelInputs
        What the kernel was built from.
    kernel : ResponseKernel
        Kernel to emit.
    statements : list[Statement]
        Module declarations receiving axes.
    steps : list[ProgramStep]
        Program steps receiving the parts.
    formula : Formula
        Response, its rows, and its modifiers.
    """
    emission = KernelEmission(host, inputs, kernel, {}, statements, steps, {}, formula)
    for part in kernel.head:
        part.emit(emission)
    for axis in kernel.objects:
        emission.declare(axis)
