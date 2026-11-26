"""Formulas on the parameters of a named QVR program.

A program whose observation scores one response is a response kernel: its
sample sites and runtime inputs are the kernel's parameters, and its steps are
the kernel's body. `ProgramKernelSpec` is the `KernelLaw` of such a program.
The parameters given a formula are slots fed by a predictor on the link scale
their support requires; every other site keeps the program's own draw. The
program's row axis (the domain of the program and the plate of the response)
becomes the rows of the response.

The law is a value: the program's text, its entry, its response, and the
names of the parameters the formulas replace. `program_analysis` checks the
text once, with the row axis left symbolic, and reads every site, input and
support from the checked program; `ProgramKernelSpec.kernel` builds the
kernel from that analysis, so nothing rewrites program text.

Links by support: a real parameter is its predictor; one on ``(a, inf)`` is
``a + exp(eta)`` (``exp(eta)`` for ``a = 0``); one on ``(-inf, b)`` is
``b - exp(eta)``; one on ``(a, b)`` is ``a + (b - a) sigmoid(eta)``; a simplex
is the softmax of a predictor with the first coordinate fixed at zero. A
replaced site becomes a deterministic binding, so no density term comes from
the link. A discrete, ordered, positive definite, Cholesky or improper site
has no such link and is refused by name.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Literal

import didactic.api as dx
import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl import Compiler, CompileError, ParseError, QiecDiagnosticError, parse
from quivers.dsl.ast_nodes import (
    AxisSpec,
    DiscreteConstructor,
    DrawArg,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    ExportDecl,
    ExprIdent,
    LetExprFactor,
    MarginalizeStep,
    Module,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecValueParameter,
    SampleStep,
    Statement,
    TypeFromExpr,
    TypeName,
)
from quivers.dsl.emit import draw_call_to_source
from quivers.dsl.program_templates import _bound_names
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.step_resolution import _draw_arg_value_names, _step_value_names
from quivers.formulas._blocks import Namespace, rename_program
from quivers.formulas._kernel import (
    BindSlot,
    Completion,
    ConstantSource,
    KernelEmission,
    KernelInputs,
    KernelPart,
    ParameterSlot,
    ParameterSource,
    PredictorSource,
    ResponseKernel,
    Steps,
)
from quivers.formulas._kernel_law import KernelDatum, KernelLaw
from quivers.formulas._modifiers import response_modifiers
from quivers.formulas._observation import Observation
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas._brms_priors import flat_event_rank
from quivers.formulas._term import AxisDecl, InputDecl
from quivers.formulas.family import (
    AuxParam,
    Domain,
    Family,
    Link,
    ParameterCoordinates,
    links,
)
from quivers.formulas.formula import Formula
from quivers.inference import LatentRegistry
from quivers.qiec import (
    IndexLiteral,
    IndexVariable,
    ProgramParameter,
    ShapeIndex,
    TypeApplication,
)
from quivers.qiec.families import family as semantic_family

__all__: list[str] = []

type Node = dx.Model | tuple[Node, ...] | str | int | float | bool | None
"""A value an AST node holds: a node, a tuple of values, or a leaf."""

ROWS = "Resp"
"""The axis of the response rows in an emitted model."""

_CARRIERS: dict[str, Literal["Real", "Int", "Bool"]] = {
    "Real": "Real",
    "Int": "Int",
    "Nat": "Int",
    "Bool": "Bool",
}
"""Element carriers of a program input, by the name of its checked type."""


class SiteLaw(ValueModel):
    """The link a parameter's support requires, or why it has none.

    Parameters
    ----------
    link : Link or None
        Inverse link from the real line onto the support of each coordinate;
        None when the support has no such link.
    domain : Domain
        The interval of each coordinate.
    reference : int or None
        Coordinate of a simplex fixed at zero on the link scale.
    obstacle : str or None
        Why the parameter cannot be given a formula; None when it can.
    """

    link: Link | None = None
    domain: Domain = Domain(lower=None)
    reference: int | None = None
    obstacle: str | None = None


class KernelSite(ValueModel):
    """One site of a checked program.

    Parameters
    ----------
    name : str
        Site name.
    kind : str
        Kind of the site: ``sample``, ``marginal``, ``observe``, ...
    law : str
        The law the site is drawn from or scored under.
    axes : tuple[str, ...]
        The plates of the site, batch plates first.
    sizes : tuple[int, ...]
        The size of each plate other than the row axis.
    top_level : bool
        Whether the program draws the site at its top level, outside every
        scope.
    prior : str
        The site's law as QVR source; empty for a site that is not a top-level
        draw.
    law_of_support : SiteLaw
        The link of a top-level draw, or why it has none.
    """

    name: str
    kind: str
    law: str
    axes: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    top_level: bool = False
    prior: str = ""
    law_of_support: SiteLaw = SiteLaw()


class KernelInput(ValueModel):
    """One external value a checked program reads.

    Parameters
    ----------
    name : str
        Input name.
    role : str
        Checked role: ``data``, ``index_map``, ``scalar`` or ``observation``.
    element : {"Real", "Int", "Bool"}
        Element carrier.
    rows : bool
        Whether the input has one value per row, the row axis first.
    axes : tuple[str, ...]
        The names of its other axes, as the program declares them; an axis
        the checker infers without a name is ``""``.
    sizes : tuple[int or None, ...]
        The size of each other axis; None for an extent the data decides.
    declared : bool
        Whether the program declares the input among its parameters.
    """

    name: str
    role: str
    element: Literal["Real", "Int", "Bool"]
    rows: bool = False
    axes: tuple[str, ...] = ()
    sizes: tuple[int | None, ...] = ()
    declared: bool = False


class ProgramAnalysis(ValueModel):
    """A program checked with its row axis left symbolic.

    Parameters
    ----------
    entry : str
        The program read.
    row_axis : str
        The program's domain, whose elements are the rows of the response.
    objects : tuple[AxisDecl, ...]
        The finite objects other than the row axis.
    module : Module
        The parsed source.
    program : ProgramDecl
        The entry program.
    statements : tuple[Statement, ...]
        Declarations the entry reads besides its objects: other programs and
        morphisms.
    sites : tuple[KernelSite, ...]
        The program's sites, in checked order.
    inputs : tuple[KernelInput, ...]
        The program's external values, in checked order.
    """

    entry: str
    row_axis: str
    objects: tuple[AxisDecl, ...] = ()
    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    program: ProgramDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    statements: tuple[Statement, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    sites: tuple[KernelSite, ...] = ()
    inputs: tuple[KernelInput, ...] = ()

    def site(self, name: str) -> KernelSite | None:
        """Return the site of a name.

        Parameters
        ----------
        name : str
            Site name.

        Returns
        -------
        KernelSite or None
            The site, or None when the program has none of that name.
        """
        return next((site for site in self.sites if site.name == name), None)

    def input(self, name: str) -> KernelInput | None:
        """Return the external value of a name.

        Parameters
        ----------
        name : str
            Input name.

        Returns
        -------
        KernelInput or None
            The input, or None when the program reads none of that name.
        """
        return next((item for item in self.inputs if item.name == name), None)

    @property
    def observations(self) -> tuple[str, ...]:
        """The sites the program observes.

        Returns
        -------
        tuple[str, ...]
            Observed site names.
        """
        return tuple(site.name for site in self.sites if site.kind == "observe")

    @property
    def parameters(self) -> tuple[str, ...]:
        """The names a formula may replace: top-level draws and real inputs.

        Returns
        -------
        tuple[str, ...]
            Sample sites drawn at the top level, then inputs that are not
            observations, in checked order.
        """
        return (
            *(site.name for site in self.sites if site.top_level),
            *(item.name for item in self.inputs if item.role != "observation"),
        )


def _entry(module: Module) -> ProgramDecl:
    """Select the entry program of a module.

    Parameters
    ----------
    module : Module
        Parsed source.

    Returns
    -------
    ProgramDecl
        The exported program, or the only program.

    Raises
    ------
    ValueError
        If the source has no program, or several candidates.
    """
    programs = [item for item in module.statements if isinstance(item, ProgramDecl)]
    if not programs:
        raise ValueError("the program source declares no program")
    names = [program.name for program in programs]
    exported = [
        item.expr.name
        for item in module.statements
        if isinstance(item, ExportDecl)
        and isinstance(item.expr, ExprIdent)
        and item.expr.name in names
    ]
    if len(exported) > 1:
        raise ValueError(
            f"the program source exports the programs {exported}; export only the "
            "one whose parameters the formulas give"
        )
    if not exported and len(programs) > 1:
        raise ValueError(
            f"the program source declares the programs {names} and exports none; "
            "export the one whose parameters the formulas give"
        )
    chosen = exported[0] if exported else names[0]
    return next(program for program in programs if program.name == chosen)


def _finite_size(item: ObjectDecl) -> str | None:
    """Read the size of a finite object declaration.

    Parameters
    ----------
    item : ObjectDecl
        Object declaration.

    Returns
    -------
    str or None
        The size as written, or None for an object that is not a finite set.
    """
    init = item.init
    if not isinstance(init, TypeFromExpr):
        return None
    expression = init.expr
    if not isinstance(expression, DiscreteConstructor):
        return None
    if expression.constructor != "FinSet" or len(expression.args) != 1:
        return None
    return expression.args[0]


def _symbolic_rows(module: Module, row_axis: str) -> Module:
    """Leave the size of the row axis to the data.

    Parameters
    ----------
    module : Module
        Parsed source.
    row_axis : str
        The program's row axis.

    Returns
    -------
    Module
        The source with ``object <row_axis> : FinSet _``.
    """
    statements: list[Statement] = []
    for item in module.statements:
        if isinstance(item, ObjectDecl) and row_axis in item.names:
            init = item.init
            assert isinstance(init, TypeFromExpr)
            expression = init.expr
            assert isinstance(expression, DiscreteConstructor)
            item = item.with_(init=init.with_(expr=expression.with_(args=("_",))))
        statements.append(item)
    return module.with_(statements=tuple(statements))


def _bound(value: torch.Tensor | float, name: str, end: str) -> float | None:
    """Read one end of a support as a constant.

    Parameters
    ----------
    value : torch.Tensor or float
        The end, possibly one value per coordinate.
    name : str
        Site name.
    end : str
        ``lower`` or ``upper``.

    Returns
    -------
    float or None
        The finite end, or None for an infinite one.

    Raises
    ------
    ValueError
        If the end differs between coordinates.
    """
    values = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or not bool((values == values[0]).all()):
        raise ValueError(
            f"the {end} end of the support of {name!r} differs between its "
            "coordinates; a formula needs one link for every coordinate"
        )
    number = float(values[0])
    return None if math.isinf(number) else number


def _number(value: float) -> str:
    """Write a constant as a QVR literal.

    Parameters
    ----------
    value : float
        The constant.

    Returns
    -------
    str
        A literal that reads back to the same float.
    """
    return repr(float(value))


def support_law(name: str, support: constraints.Constraint) -> SiteLaw:
    """Choose the link a support requires.

    Parameters
    ----------
    name : str
        Site name.
    support : torch.distributions.constraints.Constraint
        The site's support, as the inference layer reads it.

    Returns
    -------
    SiteLaw
        The link and domain, or the obstacle.
    """
    if support.is_discrete:
        return SiteLaw(
            obstacle=f"{name!r} is a discrete site; a formula gives a parameter on "
            "a continuous scale"
        )
    if type(support) is type(constraints.simplex):
        return SiteLaw(
            link=links["softmax"],
            domain=Domain(lower=0.0, upper=1.0),
            reference=0,
        )
    if support.event_dim:
        return SiteLaw(
            obstacle=f"{name!r} has the support {support!r}, whose coordinates are "
            "not free (an ordered vector, a positive definite matrix or a "
            "Cholesky factor); no link maps a predictor onto it"
        )
    if type(support) is type(constraints.real):
        return SiteLaw(link=links["identity"])
    if not hasattr(support, "lower_bound") and not hasattr(support, "upper_bound"):
        return SiteLaw(
            obstacle=f"{name!r} has the support {support!r}, which states no "
            "constant interval"
        )
    try:
        lower = _bound(getattr(support, "lower_bound", -math.inf), name, "lower")
        upper = _bound(getattr(support, "upper_bound", math.inf), name, "upper")
    except ValueError as error:
        return SiteLaw(obstacle=str(error))
    domain = Domain(lower=lower, upper=upper)
    if lower is None and upper is None:
        return SiteLaw(link=links["identity"], domain=domain)
    if upper is None:
        assert lower is not None
        if lower == 0.0:
            return SiteLaw(link=links["log"], domain=domain)
        return SiteLaw(
            link=Link(
                name=f"log_above_{_number(lower)}",
                inverse_expr=f"{_number(lower)} + exp({{eta}})",
                image=domain,
            ),
            domain=domain,
        )
    if lower is None:
        return SiteLaw(
            link=Link(
                name=f"log_below_{_number(upper)}",
                inverse_expr=f"{_number(upper)} - exp({{eta}})",
                image=domain,
            ),
            domain=domain,
        )
    if (lower, upper) == (0.0, 1.0):
        return SiteLaw(link=links["logit"], domain=domain)
    return SiteLaw(
        link=Link(
            name=f"logit_between_{_number(lower)}_{_number(upper)}",
            inverse_expr=(
                f"{_number(lower)} + {_number(upper - lower)} * sigmoid({{eta}})"
            ),
            image=domain,
        ),
        domain=domain,
    )


def _carrier(name: str, element: str) -> Literal["Real", "Int", "Bool"]:
    """Read the carrier of an input's element type.

    Parameters
    ----------
    name : str
        Input name.
    element : str
        Name of the checked element type.

    Returns
    -------
    {"Real", "Int", "Bool"}
        The carrier.

    Raises
    ------
    ValueError
        If the element is not a real, integer or Boolean number.
    """
    if element not in _CARRIERS:
        raise ValueError(
            f"the program input {name!r} holds {element} values; a formula binds "
            "real, integer and Boolean data"
        )
    return _CARRIERS[element]


def _checked_input(
    parameter: ProgramParameter,
    row_axis: str,
    declared: QiecValueParameter | None,
) -> KernelInput:
    """Read one checked external value.

    Parameters
    ----------
    parameter : ProgramParameter
        Checked program parameter.
    row_axis : str
        The symbolic row axis.
    declared : QiecValueParameter or None
        The program's declaration of the input, if it declares one.

    Returns
    -------
    KernelInput
        The input.

    Raises
    ------
    ValueError
        If the input has the row axis twice or not first, or a type that is
        not a tensor of numbers.
    """
    name = parameter.name
    kind = parameter.type
    if isinstance(kind, TypeApplication) and kind.constructor.name == "Tensor":
        element, shape = kind.arguments[0], kind.arguments[1]
        if not isinstance(element, TypeApplication) or not isinstance(
            shape, ShapeIndex
        ):
            raise ValueError(f"the program input {name!r} has the type {kind!r}")
        dimensions = shape.dimensions
        element_name = element.constructor.name
    elif isinstance(kind, TypeApplication):
        dimensions = ()
        element_name = kind.constructor.name
    else:
        raise ValueError(f"the program input {name!r} has the type {kind!r}")
    rows = [
        index
        for index, dimension in enumerate(dimensions)
        if isinstance(dimension, IndexVariable) and dimension.name == row_axis
    ]
    if len(rows) > 1 or rows and rows[0] != 0:
        raise ValueError(
            f"the program input {name!r} uses the row axis {row_axis!r} "
            + (
                "more than once, which couples rows"
                if len(rows) > 1
                else "after another axis; the rows of a data column come first"
            )
            + "; resizing the rows to the data would change what it means"
        )
    others = dimensions[1:] if rows else dimensions
    names: tuple[str, ...] = ("",) * len(others)
    if declared is not None and isinstance(declared.type_expr, QiecTypeApplication):
        written = tuple(
            dimension.name if isinstance(dimension, QiecIndexName) else ""
            for index in declared.type_expr.indices
            if isinstance(index, QiecShapeIndex)
            for dimension in index.dimensions
        )
        names = written[1:] if rows else written
    return KernelInput(
        name=name,
        role=parameter.role,
        element=_carrier(name, element_name),
        rows=bool(rows),
        axes=names,
        sizes=tuple(
            int(dimension.value) if isinstance(dimension, IndexLiteral) else None
            for dimension in others
        ),
        declared=declared is not None,
    )


def _row_couplings(node: Node, row_axis: str) -> Iterator[str]:
    """Find the places of a program that use the row axis twice at once.

    Parameters
    ----------
    node : Node
        An AST node, a tuple of them, or a leaf.
    row_axis : str
        The row axis.

    Yields
    ------
    str
        The kind of each node that uses the row axis twice.
    """
    if isinstance(node, tuple):
        for item in node:
            yield from _row_couplings(item, row_axis)
        return
    if not isinstance(node, dx.Model):
        return
    axes: tuple[str, ...] = ()
    if isinstance(node, QiecShapeIndex):
        axes = tuple(
            item.name for item in node.dimensions if isinstance(item, QiecIndexName)
        )
    elif isinstance(node, LetExprFactor):
        axes = tuple(
            binder.index.name
            for binder in node.binders
            if isinstance(binder.index, TypeName)
        )
    elif isinstance(node, AxisSpec):
        axes = (*node.over, *node.iid_over)
    if axes.count(row_axis) > 1:
        yield type(node).__name__
    for key in type(node).__field_specs__:
        yield from _row_couplings(getattr(node, key), row_axis)


@functools.lru_cache(maxsize=32)
def program_analysis(source: str) -> ProgramAnalysis:
    """Check a program with its row axis left symbolic and read its parts.

    Parameters
    ----------
    source : str
        QVR source declaring the program.

    Returns
    -------
    ProgramAnalysis
        The entry, its row axis, objects, sites, inputs, and supports.

    Raises
    ------
    ValueError
        If the source does not parse or check, its entry is ambiguous or has
        static parameters, its domain is not a finite object, or it couples
        its rows.
    """
    try:
        module = parse(source)
    except ParseError as error:
        raise ValueError(f"the program source does not parse: {error}") from error
    program = _entry(module)
    if program.params or program.type_params or program.binders:
        raise ValueError(
            f"the program {program.name!r} has static parameters; a formula block "
            "reads a program whose parameters are all values"
        )
    domain = program.domain
    objects = {
        name: item
        for item in module.statements
        if isinstance(item, ObjectDecl)
        for name in item.names
    }
    if not isinstance(domain, TypeName) or domain.name not in objects:
        raise ValueError(
            f"the domain of the program {program.name!r} is not a declared object; "
            "its elements are the rows of the response, so it must be a finite set"
        )
    row_axis = domain.name
    if _finite_size(objects[row_axis]) is None:
        raise ValueError(
            f"the domain {row_axis!r} of the program {program.name!r} is not a "
            "finite set; its elements are the rows of the response"
        )
    couplings = sorted(
        set(_row_couplings((program.value_parameters, program.draws), row_axis))
    )
    if couplings:
        raise ValueError(
            f"the program {program.name!r} uses its row axis {row_axis!r} twice in "
            f"one place ({', '.join(couplings)}), which couples rows; resizing the "
            "rows to the data would change what the program means"
        )
    sizes: dict[str, int] = {}
    for name, item in objects.items():
        written = _finite_size(item)
        if written is None or name == row_axis:
            continue
        if not written.isdigit():
            raise ValueError(
                f"the object {name!r} of the program has no size; give it one, as "
                f"'object {name} : FinSet 3'"
            )
        sizes[name] = int(written)
    try:
        checked = lower_qvr_to_qiec(_symbolic_rows(module, row_axis))
    except (QiecDiagnosticError, CompileError) as error:
        raise ValueError(f"the program does not check: {error}") from error
    entry = next((item for item in checked.entries if item.name == program.name), None)
    if entry is None:
        raise ValueError(f"the program {program.name!r} has no checked contract")
    declared = {parameter.name: parameter for parameter in program.value_parameters}
    inputs = tuple(
        _checked_input(parameter, row_axis, declared.get(parameter.name))
        for parameter in entry.parameters
        if parameter.role in {"data", "index_map", "scalar", "observation"}
    )
    top = {step.vars[0]: step for step in program.draws if isinstance(step, SampleStep)}
    compiler = Compiler(module)
    try:
        compiler.compile()
    except CompileError as error:
        raise ValueError(f"the program does not compile: {error}") from error
    registry = LatentRegistry.from_model(
        compiler.morphisms[program.name],
        observed_names={item.name for item in inputs},
    )
    sites: list[KernelSite] = []
    for site in entry.sites:
        plates = (*site.batch, *site.event)
        names = tuple(plate.name for plate in plates)
        step = top.get(site.name)
        law = SiteLaw(obstacle=f"{site.name!r} is not drawn at the program's top level")
        prior = ""
        if step is not None and site.kind == "sample":
            prior = draw_call_to_source(
                step.morphism, step.args, static_args=step.static_args
            )
            if flat_event_rank(step.morphism) is not None:
                law = SiteLaw(
                    obstacle=f"{site.name!r} is drawn from the improper law "
                    f"{step.morphism!r}, which states no support to read a link from"
                )
            elif site.name in registry.names:
                law = support_law(site.name, registry[site.name].support)
        sites.append(
            KernelSite(
                name=site.name,
                kind=site.kind,
                law=site.family,
                axes=names,
                sizes=tuple(sizes.get(name, 0) for name in names if name != row_axis),
                top_level=step is not None and site.kind == "sample",
                prior=prior,
                law_of_support=law,
            )
        )
    return ProgramAnalysis(
        entry=program.name,
        row_axis=row_axis,
        objects=tuple(AxisDecl(name=name, size=size) for name, size in sizes.items()),
        module=module,
        program=program,
        statements=tuple(
            item
            for item in module.statements
            if not isinstance(item, (ObjectDecl, ExportDecl))
            and not (isinstance(item, ProgramDecl) and item.name == program.name)
        ),
        sites=tuple(sites),
        inputs=inputs,
    )


class ParameterDecl(ValueModel):
    """One parameter a formula replaces, as the kernel reads it.

    Parameters
    ----------
    name : str
        Site or input name.
    origin : {"site", "input"}
        Whether the program draws it or reads it.
    link : Link
        Inverse link from the predictor.
    domain : Domain
        Interval of each coordinate.
    rows : bool
        Whether the program reads one value per row.
    coordinates : ParameterCoordinates or None
        The axis of a vector parameter.
    prior : str
        The program's law of a site; empty for an input.
    """

    name: str
    origin: Literal["site", "input"]
    link: Link
    domain: Domain
    rows: bool = False
    coordinates: ParameterCoordinates | None = None
    prior: str = ""


def parameter_decl(
    analysis: ProgramAnalysis, name: str, response: str
) -> ParameterDecl:
    """Read one parameter a formula replaces.

    Parameters
    ----------
    analysis : ProgramAnalysis
        The checked program.
    name : str
        Parameter name.
    response : str
        The observed site.

    Returns
    -------
    ParameterDecl
        The parameter.

    Raises
    ------
    ValueError
        If the name is the response, another observation, an unknown name, a
        site the program does not draw at its top level, or a value without
        a link or with more than one axis besides the rows.
    """
    entry = analysis.entry
    if name == response:
        raise ValueError(
            f"{name!r} is the response the program {entry!r} observes; a formula "
            "gives a parameter of the program, not its response"
        )
    site = analysis.site(name)
    if site is not None:
        if site.kind == "observe":
            raise ValueError(
                f"{name!r} is observed by the program {entry!r}; a formula gives "
                "a parameter of the program"
            )
        law = site.law_of_support
        if law.obstacle is not None or law.link is None:
            raise ValueError(
                f"the program {entry!r} cannot take a formula for {law.obstacle}"
            )
        if analysis.row_axis in site.axes:
            raise ValueError(
                f"the program {entry!r} draws {name!r} once per row; a formula "
                "gives a parameter that the program draws once"
            )
        coordinates = _coordinates(entry, name, site.axes, site.sizes, law.reference)
        return ParameterDecl(
            name=name,
            origin="site",
            link=law.link,
            domain=law.domain,
            coordinates=coordinates,
            prior=site.prior,
        )
    item = analysis.input(name)
    if item is None:
        raise ValueError(
            f"the program {entry!r} has no parameter {name!r}; its sample sites "
            f"and inputs are {list(analysis.parameters)}"
        )
    if item.role == "observation":
        raise ValueError(
            f"{name!r} is observed by the program {entry!r}; a formula gives a "
            "parameter of the program"
        )
    if item.element != "Real":
        raise ValueError(
            f"the program input {name!r} holds {item.element} values; a formula "
            "gives a parameter on a continuous scale"
        )
    if any(not axis for axis in item.axes):
        raise ValueError(
            f"the program {entry!r} does not declare the axes of {name!r}; declare "
            f"it among the program's parameters, as '{name} : Tensor[Real]([Res])', "
            "so the formula knows its coordinates"
        )
    sizes = tuple(
        next((axis.size for axis in analysis.objects if axis.name == written), 0)
        for written in item.axes
    )
    return ParameterDecl(
        name=name,
        origin="input",
        link=links["identity"],
        domain=Domain(lower=None),
        rows=item.rows,
        coordinates=_coordinates(entry, name, item.axes, sizes, None),
    )


def _coordinates(
    entry: str,
    name: str,
    axes: Sequence[str],
    sizes: Sequence[int],
    reference: int | None,
) -> ParameterCoordinates | None:
    """Read the coordinates of a parameter with at most one axis.

    Parameters
    ----------
    entry : str
        Program name.
    name : str
        Parameter name.
    axes : Sequence[str]
        The parameter's axes other than the rows.
    sizes : Sequence[int]
        Their sizes.
    reference : int or None
        Coordinate fixed at zero on the link scale.

    Returns
    -------
    ParameterCoordinates or None
        The coordinates of a vector parameter; None for a scalar one.

    Raises
    ------
    ValueError
        If the parameter has more than one axis or an axis without a size.
    """
    if not axes:
        return None
    if len(axes) > 1:
        raise ValueError(
            f"the program {entry!r} reads {name!r} over the axes {list(axes)}; a "
            "formula gives a parameter with at most one axis besides the rows"
        )
    if sizes[0] < 1:
        raise ValueError(
            f"the axis {axes[0]!r} of {name!r} in the program {entry!r} is not a "
            "finite object with a size"
        )
    return ParameterCoordinates(axis=axes[0], size=sizes[0], reference=reference)


def _response_site(analysis: ProgramAnalysis, response: str) -> KernelSite:
    """Check the response of a program.

    Parameters
    ----------
    analysis : ProgramAnalysis
        The checked program.
    response : str
        The observed site.

    Returns
    -------
    KernelSite
        The response's site.

    Raises
    ------
    ValueError
        If the program does not observe the response, observes another site,
        or observes the response over other plates than its rows.
    """
    entry = analysis.entry
    observed = analysis.observations
    if response not in observed:
        raise ValueError(
            f"the program {entry!r} does not observe {response!r}; it observes "
            f"{list(observed)}"
        )
    others = [name for name in observed if name != response]
    if others:
        raise ValueError(
            f"the program {entry!r} observes {others} besides the response "
            f"{response!r}; a program block has one response, so state the others "
            "as blocks of their own"
        )
    site = analysis.site(response)
    assert site is not None
    if site.axes != (analysis.row_axis,):
        raise ValueError(
            f"the program {entry!r} observes {response!r} over {list(site.axes)}; "
            f"the response has one value per row, so its plate is the row axis "
            f"{analysis.row_axis!r} alone"
        )
    return site


class ProgramParameters(KernelPart):
    """Declare the runtime parameters the program declares itself.

    Parameters
    ----------
    parameters : tuple[QiecValueParameter, ...]
        The declarations, the row axis renamed to the rows.
    """

    kind: Literal["program_parameters"] = "program_parameters"
    parameters: tuple[QiecValueParameter, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Declare the parameters.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        for parameter in self.parameters:
            emission.parameters.setdefault(parameter.name, parameter)


class ProgramDeclarations(KernelPart):
    """Place the declarations the program reads besides its objects.

    Parameters
    ----------
    statements : tuple[Statement, ...]
        Other programs and morphisms of the source.
    """

    kind: Literal["program_declarations"] = "program_declarations"
    statements: tuple[Statement, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Append the declarations.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.statements.extend(self.statements)


class ObserveResponse(KernelPart):
    """Record the observation of the response the program's steps write.

    Parameters
    ----------
    observation : Observation or None
        The observation, when its arguments are names and constants; None
        when the program passes an indexed or composite argument, which no
        `Observation` states.
    """

    kind: Literal["program_observation"] = "program_observation"
    observation: Observation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Record the observation.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        if self.observation is not None:
            emission.host.observed(self.observation)


def _enclosing(
    steps: Sequence[ProgramStep], response: str
) -> tuple[ObserveStep, tuple[MarginalizeStep, ...]] | None:
    """Find the observation of the response and the blocks around it.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        Program steps.
    response : str
        Observed site.

    Returns
    -------
    tuple[ObserveStep, tuple[MarginalizeStep, ...]] or None
        The observation and the marginalizations enclosing it, outermost
        first; None when the steps do not observe the response.
    """
    for step in steps:
        if isinstance(step, ObserveStep) and response in step.vars:
            return step, ()
        if isinstance(step, MarginalizeStep):
            found = _enclosing(step.scope, response)
            if found is not None:
                return found[0], (step, *found[1])
    return None


def _observation(step: ObserveStep, factorized: bool) -> Observation | None:
    """State a program's observation of the response as a value.

    Parameters
    ----------
    step : ObserveStep
        The observation, in the model's names.
    factorized : bool
        Whether the rows are independent given the parameters.

    Returns
    -------
    Observation or None
        The observation, or None when an argument is not a name or constant.
    """
    schema = [parameter.name for parameter in semantic_family(step.morphism).parameters]
    parameters: list[str] = []
    values: list[str | int | float] = []
    for position, argument in enumerate(step.args or ()):
        parameter = schema[position] if position < len(schema) else ""
        value: DrawArg = argument
        if isinstance(argument, DrawArgNamed):
            parameter, value = argument.parameter, argument.value
        if isinstance(value, DrawArgName):
            values.append(value.text)
        elif isinstance(value, DrawArgScalar):
            values.append(value.value)
        else:
            return None
        parameters.append(parameter)
    return Observation(
        site=step.vars[0],
        kernel=step.morphism,
        parameters=tuple(parameters),
        values=tuple(values),
        plate=ROWS,
        factorized=factorized,
    )


def _row_varying_laws(
    steps: Sequence[ProgramStep], varying: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    """Find the marginalized latents whose law varies over the rows.

    A latent marginalized without ``over=`` the rows is one draw shared by
    every row, so its law cannot read a value that varies over them.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        Program steps, in the model's names.
    varying : Mapping[str, frozenset[str]]
        Bindings that vary over the rows, with the parameters they read.

    Returns
    -------
    dict[str, frozenset[str]]
        Each such latent, with the parameters its law reads.
    """
    reads = dict(varying)
    found: dict[str, frozenset[str]] = {}

    def walk(items: Sequence[ProgramStep]) -> None:
        for step in items:
            if isinstance(step, MarginalizeStep):
                read = frozenset().union(
                    *(_draw_arg_value_names(argument) for argument in step.args or ())
                )
                sources = frozenset().union(
                    *(reads[name] for name in read if name in reads)
                )
                if sources and step.over != ROWS:
                    found[step.var] = sources
                walk(step.scope)
                continue
            sources = frozenset().union(
                *(reads[name] for name in _step_value_names(step) if name in reads)
            )
            if sources:
                for name in _bound_names((step,)):
                    reads[name] = reads.get(name, frozenset()) | sources

    walk(steps)
    return found


class ProgramKernelSpec(KernelLaw):
    """The kernel law of a named QVR program.

    Parameters
    ----------
    source : str
        QVR source declaring the program.
    entry : str
        The program read: the exported program, or the only one.
    response : str
        The observed site.
    parameters : tuple[str, ...]
        The sites and inputs a formula or a fixed value replaces, in program
        order.
    """

    kind: Literal["program"] = "program"
    source: str
    entry: str
    response: str
    parameters: tuple[str, ...] = ()

    @property
    def analysis(self) -> ProgramAnalysis:
        """The program checked with a symbolic row axis.

        Returns
        -------
        ProgramAnalysis
            The checked program.
        """
        return program_analysis(self.source)

    @property
    def row_columns(self) -> tuple[str, ...]:
        """The inputs the program reads at each row.

        Returns
        -------
        tuple[str, ...]
            Names of row inputs no formula replaces.
        """
        return tuple(
            item.name
            for item in self.analysis.inputs
            if item.rows
            and item.role != "observation"
            and item.name not in self.parameters
        )

    def bindings(
        self,
        frame: pd.DataFrame,
        data2: Mapping[str, KernelDatum] | None,
    ) -> dict[str, torch.Tensor]:
        """Bind the program's data by name.

        Data at each row are columns of ``frame``; data on another axis and
        scalars come from ``data2``.

        Parameters
        ----------
        frame : pandas.DataFrame
            The response rows, in row order.
        data2 : Mapping[str, KernelDatum] or None
            Values that are not indexed by the rows.

        Returns
        -------
        dict[str, torch.Tensor]
            Values by input name.

        Raises
        ------
        ValueError
            If a value is missing, given in the wrong place, or does not fit
            its checked type.
        """
        extra = dict(data2 or {})
        bound: dict[str, torch.Tensor] = {}
        for item in self.analysis.inputs:
            if item.role == "observation" or item.name in self.parameters:
                continue
            if item.rows:
                if item.name not in frame.columns:
                    place = (
                        "; it is given in data2, but it has one value per row, so "
                        "it is a data column"
                        if item.name in extra
                        else ""
                    )
                    raise ValueError(
                        f"the program {self.entry!r} reads {item.name!r} at each "
                        f"row, and the data has no column {item.name!r}{place}"
                    )
                bound[item.name] = _tensor(
                    item, np.asarray(frame[item.name].to_numpy()), len(frame)
                )
                continue
            if item.name not in extra:
                place = (
                    "; it is a column of the data, but the program does not read it "
                    "by row"
                    if item.name in frame.columns
                    else ""
                )
                raise ValueError(
                    f"the program {self.entry!r} reads {item.name!r}, which is not "
                    f"indexed by its rows; give it in data2{place}"
                )
            value = extra[item.name]
            array = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value.to_numpy()
                if isinstance(value, pd.DataFrame)
                else np.asarray(value)
            )
            bound[item.name] = _tensor(item, array, None)
        return bound

    def kernel(
        self, inputs: KernelInputs, formula: Formula
    ) -> tuple[ResponseKernel, dict[str, ParameterSource]]:
        """Build the kernel of the program.

        Parameters
        ----------
        inputs : KernelInputs
            The program's family, the planned predictors, and fixed values.
        formula : Formula
            Formula with the response and its modifiers.

        Returns
        -------
        tuple[ResponseKernel, dict[str, ParameterSource]]
            The kernel and the source of each replaced parameter.

        Raises
        ------
        ValueError
            If the response carries a term that changes its law, the model
            adds to a location the program does not have, a vector or row
            parameter is fixed, or a parameter that varies over the rows
            feeds the law of a latent the rows share.
        """
        analysis = self.analysis
        entry = analysis.entry
        modifiers = response_modifiers(formula.modifiers, inputs.residual)
        if modifiers:
            raise ValueError(
                f"the program {entry!r} scores its response by its own observation; "
                f"the response terms {[modifier.kind for modifier in modifiers]} "
                "change that law, so the program must state them itself"
            )
        if inputs.host is not None or inputs.graph is not None:
            raise ValueError(
                f"the program {entry!r} has no location for a neural predictor or a "
                "nonlinear formula to feed; give its parameters formulas instead"
            )
        site = _response_site(analysis, self.response)
        declarations = {
            name: parameter_decl(analysis, name, self.response)
            for name in self.parameters
        }
        objects = {analysis.row_axis: ROWS} | {
            axis.name: axis.name for axis in analysis.objects
        }
        renamed = rename_program(
            Module(
                statements=(
                    *(
                        ObjectDecl(
                            names=(name,),
                            init=TypeFromExpr(
                                expr=DiscreteConstructor(
                                    constructor="FinSet", args=("1",)
                                )
                            ),
                        )
                        for name in objects
                    ),
                    analysis.program,
                )
            ),
            Namespace(prefix="", shared_objects=objects),
        )
        slots: list[ParameterSlot] = []
        sources: dict[str, ParameterSource] = {}
        varying: dict[str, frozenset[str]] = {}
        for name, declaration in declarations.items():
            prefix = f"dpar_{name}_"
            coordinates = declaration.coordinates
            if name in inputs.distributional:
                predictor = inputs.predictor(prefix)
                sources[name] = PredictorSource(predictor=predictor)
                if predictor.rowwise or declaration.rows:
                    varying[name] = frozenset({name})
            else:
                if coordinates is not None or declaration.rows:
                    raise ValueError(
                        f"a fixed value of {name!r} is one number, and the program "
                        f"{entry!r} reads {name!r} "
                        + ("over an axis" if coordinates is not None else "per row")
                        + "; give it a formula such as '~ 1'"
                    )
                sources[name] = ConstantSource(value=float(inputs.fixed[name]))
            slots.append(
                ParameterSlot(
                    name=name,
                    eta=f"{prefix}eta",
                    link=declaration.link,
                    free=None if coordinates is None else f"{prefix}free",
                    completion=None
                    if coordinates is None or coordinates.reference is None
                    else Completion(
                        axis=coordinates.axis,
                        size=coordinates.size,
                        reference=coordinates.reference,
                        coordinates=coordinates.free,
                        prefix=prefix,
                    ),
                    row_binder=f"{prefix}row"
                    if declaration.rows and coordinates is None
                    else None,
                    constant_rows=not declaration.rows,
                )
            )
        shared = _row_varying_laws(renamed.program.draws, varying)
        if shared:
            latent, sources_read = next(iter(shared.items()))
            raise ValueError(
                f"the law of the latent {latent!r} that the program {entry!r} "
                f"marginalizes reads {sorted(sources_read)}, whose formulas vary "
                f"over the rows; the marginalization draws one {latent!r} for "
                f"every row, so its law cannot vary by row. To give each row its "
                f"own {latent!r}, state the block per row in the program: "
                f"'marginalize {latent} : ... [over={analysis.row_axis}, ...]' "
                f"with '[via=<index of each row>]' on the observation"
            )
        found = _enclosing(renamed.program.draws, self.response)
        assert found is not None
        observed, enclosing = found
        factorized = all(block.over == ROWS for block in enclosing)
        replaced_inputs = {
            name for name, item in declarations.items() if item.origin == "input"
        }
        body: list[KernelPart] = [
            ProgramDeclarations(statements=analysis.statements),
            ProgramParameters(
                parameters=tuple(
                    parameter
                    for parameter in renamed.program.value_parameters
                    if parameter.name not in replaced_inputs
                )
            ),
            *(BindSlot(slot=name) for name in replaced_inputs),
        ]
        literal: list[ProgramStep] = []
        for step in renamed.program.draws:
            if isinstance(step, SampleStep) and step.vars[0] in declarations:
                if literal:
                    body.append(Steps(steps=tuple(literal)))
                    literal = []
                body.append(BindSlot(slot=step.vars[0]))
                continue
            literal.append(step)
        if literal:
            body.append(Steps(steps=tuple(literal)))
        body.append(ObserveResponse(observation=_observation(observed, factorized)))
        data = tuple(
            InputDecl(
                name=item.name,
                element=item.element,
                axes=((ROWS,) if item.rows else ()) + item.axes,
                declared=False,
            )
            for item in analysis.inputs
            if item.role != "observation" and item.name not in declarations
        )
        assert site.name == self.response
        return (
            ResponseKernel(
                slots=tuple(slots),
                objects=analysis.objects,
                data=data,
                body=tuple(body),
                response=self.response,
            ),
            sources,
        )


def _tensor(item: KernelInput, array: np.ndarray, rows: int | None) -> torch.Tensor:
    """Convert one value to its input's carrier and check its shape.

    Parameters
    ----------
    item : KernelInput
        The input.
    array : numpy.ndarray
        The value.
    rows : int or None
        The number of rows of a row input; None for any other input.

    Returns
    -------
    torch.Tensor
        The value as a tensor.

    Raises
    ------
    ValueError
        If the value is not numeric, has the wrong carrier or shape, or is
        not finite.
    """
    name = item.name
    if array.dtype.kind not in "biuf":
        raise ValueError(f"the program input {name!r} requires numeric values")
    rank = (1 if item.rows else 0) + len(item.sizes)
    if array.ndim != rank:
        raise ValueError(
            f"the program input {name!r} has {rank} axes, and its value has "
            f"{array.ndim}"
        )
    expected = ((rows,) if rows is not None else ()) + item.sizes
    for position, (size, wanted) in enumerate(zip(array.shape, expected)):
        if wanted is not None and size != wanted:
            raise ValueError(
                f"axis {position} of the program input {name!r} has {wanted} "
                f"elements, and its value has {size}"
            )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"the program input {name!r} requires finite values")
    if item.element == "Real":
        return torch.as_tensor(array.astype(np.float64))
    integral = array.dtype.kind in "biu" or bool(np.equal(array, np.round(array)).all())
    if not integral:
        raise ValueError(
            f"the program input {name!r} holds {item.element} values, and its data "
            "are not whole numbers"
        )
    if item.element == "Bool":
        if not np.isin(array, (0, 1)).all():
            raise ValueError(
                f"the program input {name!r} holds Boolean values, and its data are "
                "not zero or one"
            )
        return torch.as_tensor(array.astype(bool))
    return torch.as_tensor(array.astype(np.int64))


def program_family(spec: ProgramKernelSpec) -> Family:
    """State the response family of a program.

    The family observes the response under the law of the program's
    observation, has one auxiliary parameter per replaced site or input, and
    builds its kernel through the program's law.

    Parameters
    ----------
    spec : ProgramKernelSpec
        The program, its response, and the parameters replaced.

    Returns
    -------
    Family
        The family.

    Raises
    ------
    ValueError
        If the response or a parameter is not one the program can take.
    """
    analysis = spec.analysis
    site = _response_site(analysis, spec.response)
    declarations = [
        parameter_decl(analysis, name, spec.response) for name in spec.parameters
    ]
    return Family(
        name=analysis.entry,
        location_link=links["identity"],
        observe_family=site.law,
        aux_params=tuple(
            AuxParam(
                name=item.name,
                prior=item.prior,
                link=item.link,
                domain=item.domain,
                coordinates=item.coordinates,
            )
            for item in declarations
        ),
        kernel_law=spec,
    )


def program_spec(
    program: str,
    response: str,
    parameters: Collection[str],
) -> ProgramKernelSpec:
    """Read the kernel law of a program from the arguments of a formula call.

    Parameters
    ----------
    program : str
        QVR source declaring the program.
    response : str
        The observed site.
    parameters : Collection[str]
        The sites and inputs given a formula or a fixed value.

    Returns
    -------
    ProgramKernelSpec
        The law, its parameters in program order.

    Raises
    ------
    TypeError
        If ``program`` is not source text.
    ValueError
        If a parameter is not one the program has.
    """
    if not isinstance(program, str):
        raise TypeError(
            "program= takes the QVR source text of the program; read a file with "
            "pathlib.Path(path).read_text(). A saved fit keeps the text, not a path"
        )
    analysis = program_analysis(program)
    order = {name: index for index, name in enumerate(analysis.parameters)}
    unknown = sorted(set(parameters) - order.keys())
    for name in unknown:
        parameter_decl(analysis, name, response)
    return ProgramKernelSpec(
        source=program,
        entry=analysis.entry,
        response=response,
        parameters=tuple(sorted(parameters, key=lambda name: order[name])),
    )


def program_arguments[Authored](
    formula: Authored | None,
    *,
    program: str | None,
    parameters: Mapping[str, str] | None,
    response: str | None,
    family: str | Family,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None,
    fixed_parameters: Mapping[str, float] | None,
    caller: str,
) -> tuple[
    Authored | str, str | Family, Mapping[str, str | NonlinearExpression] | None
]:
    """Read the response, family and parameter formulas of a formula call.

    A program call names the program, its response, and the formulas of its
    parameters; it becomes the formula ``<response> ~ 0`` (a program has no
    location design) under the program's family, with one distributional
    formula per parameter.

    Parameters
    ----------
    formula : Authored or None
        The formula of an ordinary call, as authored; None for a program
        call.
    program : str or None
        QVR source declaring the program.
    parameters : Mapping[str, str] or None
        Formulas of the program's parameters, by name.
    response : str or None
        The observed site, with any response terms.
    family : str or Family
        The family keyword of the call.
    distributional, nonlinear : Mapping or None
        The parameter formulas of an ordinary call.
    fixed_parameters : Mapping[str, float] or None
        Parameters held at constants.
    caller : str
        Entry point, named in errors.

    Returns
    -------
    tuple[Authored or str, str or Family, Mapping or None]
        The formula, the family, and the distributional formulas.

    Raises
    ------
    ValueError
        If the call mixes a program with a formula, a family, or ordinary
        parameter formulas, or gives program keywords without a program.
    """
    if program is None:
        if parameters is not None or response is not None:
            raise ValueError(
                f"{caller}: parameters= and response= describe a program; pass its "
                "source as program="
            )
        if formula is None:
            raise ValueError(f"{caller}: give a formula, or a program= and response=")
        return formula, family, distributional
    if formula is not None:
        raise ValueError(
            f"{caller}: give a formula or a program, not both; the program states "
            "the response law"
        )
    if family != "gaussian":
        raise ValueError(
            f"{caller}: a program states its own response law; family= does not "
            "apply to it"
        )
    if distributional is not None or nonlinear is not None:
        raise ValueError(
            f"{caller}: the formulas of a program's parameters are given as "
            "parameters=, not distributional= or nonlinear="
        )
    if response is None:
        raise ValueError(
            f"{caller}: a program fit needs response=, the site the program observes"
        )
    site = response.partition("|")[0].strip()
    formulas = dict(parameters or {})
    spec = program_spec(program, site, {*formulas, *(fixed_parameters or {})})
    return f"{response} ~ 0", program_family(spec), formulas or None
