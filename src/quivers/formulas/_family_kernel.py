"""The response kernel of a built-in or custom family.

`family_kernel` states, as a `ResponseKernel`, the body a family declares:
the slots of its location and auxiliary parameters, the parts that compute
the arguments of its native law (ordinal thresholds and masses, category
completion, vector concentration, the map from the formula parameterization
to the native one), and the observation. A mixture declares its components,
which its kernel emits as nested blocks (`quivers.formulas._mixture`). `slot_sources` gives each slot the
source the formula, the fixed values, or the priors choose, and
`precompose_graph` folds the nonlinear predictor graph into the kernel.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
)
from quivers.dsl.let_expr_traversal import free_let_names, substitute_let_expr
from quivers.formulas._ast import (
    _add,
    _bind,
    _factor,
    _index,
    _let,
    _mul,
    _var,
    row_zero,
    stack_rows,
)
from quivers.formulas._category_formulas import (
    check_scale_formulas,
    free_categories,
    free_scales,
    is_scale_parameter,
)
from quivers.formulas._kernel import (
    BindSlot,
    Completion,
    ConstantSource,
    EmitSlot,
    EmitSlotAxis,
    KernelArguments,
    KernelEmission,
    KernelInputs,
    KernelPart,
    ParameterSlot,
    ParameterSource,
    Precomposition,
    PredictorSource,
    ResponseKernel,
    SiteSource,
    Steps,
    ValueShape,
)
from quivers.formulas._mixture import mixture_kernel
from quivers.formulas._observation import observation
from quivers.formulas._predictor import response_axis
from quivers.formulas._predictor_graph import ComponentExpression, PredictorGraph
from quivers.formulas._modifiers import response_modifiers
from quivers.formulas._term import AxisDecl, InputDecl, SiteDecl
from quivers.formulas.family import AuxParam, links
from quivers.formulas.formula import Formula, _qvr_name

__all__: list[str] = []

GRAPH_ROW = "graph_row"
"""Row binder of the nonlinear predictor graph."""


class BaselineHazard(KernelPart):
    """The normalized spline baseline of a proportional-hazards family."""

    kind: Literal["baseline_hazard"] = "baseline_hazard"

    def emit(self, emission: KernelEmission) -> None:
        """Draw the baseline simplex and bind the hazard basis.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.host.baseline(emission.formula, emission.statements, emission.steps)


class OrdinalThresholds(KernelPart):
    """Declare the ordinal thresholds, keeping their steps for later.

    The axes are declared at this point, the steps placed by
    `OrdinalThresholdSteps`.
    """

    kind: Literal["ordinal_thresholds"] = "ordinal_thresholds"

    def emit(self, emission: KernelEmission) -> None:
        """Declare the threshold axes and build the threshold steps.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        steps: list[ProgramStep] = []
        emission.names["cutpoints"] = emission.host.ordinal_thresholds(
            emission.formula, emission.statements, steps
        )
        emission.buffers["cutpoints"] = steps


class OrdinalThresholdSteps(KernelPart):
    """Place the steps of the ordinal thresholds."""

    kind: Literal["ordinal_threshold_steps"] = "ordinal_threshold_steps"

    def emit(self, emission: KernelEmission) -> None:
        """Append the threshold steps.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.steps.extend(emission.buffers.pop("cutpoints"))


class OrdinalMasses(KernelPart):
    """The arguments of an ordinal law: cutpoints, contrasts, or log masses.

    Parameters
    ----------
    cutpoint_law : bool
        Whether the native law takes a predictor and cutpoints.
    unit_discrimination : bool
        Whether the discrimination is its fixed unit and no transition
        varies, so the law takes the location and cutpoints directly.
    """

    kind: Literal["ordinal_masses"] = "ordinal_masses"
    cutpoint_law: bool
    unit_discrimination: bool

    def emit(self, emission: KernelEmission) -> None:
        """Bind the masses and state the native arguments.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        inputs, host = emission.inputs, emission.host
        cutpoints = emission.names["cutpoints"]
        kernel = inputs.family.observe_family
        if self.cutpoint_law and self.unit_discrimination:
            emission.arguments = KernelArguments(
                kernel=kernel, values=("mu", cutpoints)
            )
            return
        if self.cutpoint_law:
            host.ordinal_contrasts(
                emission.formula, cutpoints, emission.steps, emission
            )
            emission.steps.append(_let("ordinal_scaled_eta", LetExprLiteral(value=0.0)))
            emission.arguments = KernelArguments(
                kernel=kernel, values=("ordinal_scaled_eta", "ordinal_contrasts")
            )
            return
        host.ordinal_logits(
            emission.formula, cutpoints, emission.statements, emission.steps, emission
        )
        emission.arguments = KernelArguments(
            kernel="Categorical", values=("ordinal_logits",), parameters=("logits",)
        )


class NativeArguments(KernelPart):
    """The map from the formula parameterization to the native law's.

    The family states it as data: an event formed from a decision, bindings
    derived from the parameters, and the native arguments.
    """

    kind: Literal["native_arguments"] = "native_arguments"

    def emit(self, emission: KernelEmission) -> None:
        """State the native arguments and the bindings they read.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        inputs = emission.inputs
        family = inputs.family
        decision = (
            ()
            if family.decision_event is None
            else (_let(inputs.response, _bind(family.decision_event)),)
        )
        emission.arguments = KernelArguments(
            kernel=family.observe_family,
            values=tuple(
                _native_argument(argument, inputs) for argument in family.observe_args
            ),
            log_location=family.log_location_argument
            if family.retains_log_location
            else None,
            lets=(
                *decision,
                *(
                    _let(binding.name, _bind(binding.expr))
                    for binding in family.observe_lets
                ),
            ),
        )


def _native_argument(argument: str | float, inputs: KernelInputs) -> str | float:
    """Resolve one declared native constructor argument.

    Parameters
    ----------
    argument : str or float
        A declared `Family.observe_args` entry.
    inputs : KernelInputs
        Trial counts of the response.

    Returns
    -------
    str or float
        The bound program name or literal constant.
    """
    if argument == "trials":
        assert inputs.trials is not None
        return inputs.trials
    return argument


class CategoricalArguments(KernelPart):
    """The category probabilities of a categorical law."""

    kind: Literal["categorical_arguments"] = "categorical_arguments"

    def emit(self, emission: KernelEmission) -> None:
        """State the native arguments.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.arguments = KernelArguments(kernel="Categorical", values=("mu",))


class VectorArguments(KernelPart):
    """The arguments of a law over a vector response event.

    Parameters
    ----------
    law : {"multinomial", "dirichlet", "dirichlet_multinomial", "dirichlet2", "logistic_normal"}
        Construction of the arguments.
    width : int
        Number of free coordinates.
    reference : int or None
        Reference component of a log-ratio law.
    """

    kind: Literal["vector_arguments"] = "vector_arguments"
    law: Literal[
        "multinomial",
        "dirichlet",
        "dirichlet_multinomial",
        "dirichlet2",
        "logistic_normal",
    ]
    width: int = 0
    reference: int | None = None

    def emit(self, emission: KernelEmission) -> None:
        """Bind the derived quantities and state the native arguments.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.

        Raises
        ------
        ValueError
            If a residual-correlation prior is not an LKJ Cholesky law.
        """
        inputs, host = emission.inputs, emission.host
        family = inputs.family
        steps = emission.steps
        response = _var("response")
        kernel = family.observe_family
        if self.law == "multinomial":
            emission.arguments = KernelArguments(
                kernel=kernel,
                values=("formula_trials", "eta"),
                parameters=("total_count", "logits"),
                event="FormulaCategory",
            )
            return
        if self.law in ("dirichlet", "dirichlet_multinomial"):
            concentration = _factor(
                (("response", "Resp"), ("component", "FormulaCategory")),
                body=_mul(
                    _index("mu", response, _var("component")),
                    emission.value_at("phi", response),
                ),
            )
            steps.append(_let("formula_concentration", concentration))
            emission.arguments = KernelArguments(
                kernel=kernel,
                values=("formula_trials", "formula_concentration")
                if self.law == "dirichlet_multinomial"
                else ("formula_concentration",),
                event="FormulaCategory",
            )
            return
        if self.law == "dirichlet2":
            emission.arguments = KernelArguments(
                kernel=kernel, values=("mu",), event="FormulaCategory"
            )
            return
        host.residual_correlation(
            steps, "FormulaCategoryLogit", self.width, result="formula_alr_correlation"
        )
        component = family.component_scale
        assert component is not None and self.reference is not None
        steps.append(
            _let(
                "formula_alr_scale",
                _factor(
                    (
                        ("response", "Resp"),
                        ("i", "FormulaCategoryLogit"),
                        ("j", "FormulaCategoryLogit"),
                    ),
                    body=_mul(
                        emission.value_at(component.name, response, _var("i")),
                        _index("formula_alr_correlation", _var("i"), _var("j")),
                    ),
                ),
            )
        )
        emission.arguments = KernelArguments(
            kernel=kernel,
            values=("mu", "formula_alr_scale", self.reference),
            event="FormulaCategory",
        )


class Observe(KernelPart):
    """Observe the response under the native law the body states.

    The response's modifiers are folded over the law in their fixed order
    (exposure, known errors, residual law, recorded events, weights).
    """

    kind: Literal["observe"] = "observe"

    def emit(self, emission: KernelEmission) -> None:
        """Write the bindings of the native law and the observation step.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.

        Raises
        ------
        ValueError
            If no part stated the native arguments.
        """
        inputs, host = emission.inputs, emission.host
        stated = emission.arguments
        if stated is None:
            raise ValueError("the response kernel states no native arguments")
        modifiers = response_modifiers(emission.formula.modifiers, inputs.residual)
        for modifier in modifiers:
            stated = modifier.apply(stated, emission)
        emission.steps.extend(stated.lets)
        observed = observation(
            inputs.response,
            stated.kernel,
            stated.values,
            event=stated.event,
            catalog=host.response_laws,
        ).with_(factorized=stated.factorized)
        if stated.log_location is not None:
            # The kernel takes the linear predictor on the log scale.
            observed = observed.with_arguments(
                tuple(
                    (stated.log_location, "eta")
                    if bound == "mu"
                    else (parameter, bound)
                    for parameter, bound in observed.arguments
                )
            )
        if stated.parameters is not None:
            observed = observed.with_arguments(
                tuple(zip(stated.parameters, observed.values, strict=True))
            )
        for modifier in modifiers:
            observed = modifier.observe(observed)
        host.observed(observed)
        emission.steps.append(observed.step())


def graph_expression_at(
    graph: PredictorGraph,
    expression: LetExprNode,
    row: LetExprNode,
    component: str | None = None,
) -> LetExprNode:
    """Project the graph's inputs and parameters to one row and coordinate.

    Parameters
    ----------
    graph : PredictorGraph
        Nonlinear parameters, expressions, and data columns.
    expression : LetExprNode
        Scalar expression at one result coordinate.
    row : LetExprNode
        Row index.
    component : str or None
        Coordinate label, required for vector dependencies.

    Returns
    -------
    LetExprNode
        The expression over the bindings at the row.

    Raises
    ------
    ValueError
        If a vector dependency has no compatible labelled coordinate.
    """
    substitution: dict[str, LetExprNode] = {
        column.qvr_name: _index(column.qvr_name, row)
        for column in graph.inputs.values()
    }
    for name in (*graph.leaves, *graph.expressions):
        value = graph.expressions.get(name)
        if isinstance(value, ComponentExpression):
            if component is None:
                # Only reject an actually used vector, not unrelated nodes.
                if f"nlpar_{name}" in free_let_names(expression):
                    raise ValueError(
                        f"nonlinear vector {name!r} requires explicit component expressions"
                    )
                continue
            if component not in value.components:
                if f"nlpar_{name}" in free_let_names(expression):
                    raise ValueError(
                        f"nonlinear vector {name!r} has no component {component!r}"
                    )
                continue
            substitution[f"nlpar_{name}"] = _index(
                f"nlpar_{name}",
                row,
                LetExprLiteral(value=value.components.index(component), integral=True),
            )
        else:
            substitution[f"nlpar_{name}"] = _index(f"nlpar_{name}", row)
    return substitute_let_expr(expression, substitution)


def graph_value(
    graph: PredictorGraph,
    name: str,
    value: LetExprNode | ComponentExpression,
    *,
    axis: str | None = None,
    components: tuple[str, ...] | None = None,
) -> Steps:
    """Bind a graph value over the rows and, for a vector, its coordinates.

    Parameters
    ----------
    graph : PredictorGraph
        Nonlinear parameters, expressions, and data columns.
    name : str
        Binding.
    value : LetExprNode or ComponentExpression
        Scalar expression or labelled coordinate expressions.
    axis : str or None
        Coordinate axis of a vector value; None for the node's own axis.
    components : tuple[str, ...] or None
        Coordinate labels the value must have, in order; None for a value
        that names its own.

    Returns
    -------
    Steps
        The axis of a vector value and the bindings.

    Raises
    ------
    ValueError
        If the labels or the scalar or vector carrier do not match.
    """
    row = _var(GRAPH_ROW)
    if not isinstance(value, ComponentExpression):
        if components is not None:
            raise ValueError(
                f"nonlinear {name!r} requires explicit component expressions for {components}"
            )
        return Steps(
            steps=(
                _let(
                    name,
                    _factor(
                        ((GRAPH_ROW, "Resp"),),
                        body=graph_expression_at(graph, value, row),
                    ),
                ),
            )
        )
    labels = value.components if components is None else components
    if set(labels) != set(value.components):
        raise ValueError(
            f"nonlinear {name!r} component labels {value.components} do not match expected {labels}"
        )
    axis = axis or f"FormulaGraphCoordinate_{name}"
    return Steps(
        objects=(AxisDecl(name=axis, size=len(labels)),),
        steps=stack_rows(
            name,
            (
                (
                    index,
                    _factor(
                        ((GRAPH_ROW, "Resp"),),
                        body=graph_expression_at(
                            graph,
                            value.values[value.components.index(label)],
                            row,
                            label,
                        ),
                    ),
                )
                for index, label in enumerate(labels)
            ),
            ("graph_coordinate", axis),
            stem=f"{name}_by_coordinate",
            row=(GRAPH_ROW, "Resp"),
        ),
    )


def _auxiliary_slot(aux: AuxParam, inputs: KernelInputs) -> ParameterSlot:
    """State the slot of one auxiliary parameter.

    Parameters
    ----------
    aux : AuxParam
        Declared parameter.
    inputs : KernelInputs
        The planned predictors.

    Returns
    -------
    ParameterSlot
        The slot, with vector levels for a parameter with one value per
        component.
    """
    prefix = f"dpar_{aux.name}_"
    completion = None
    if aux.per_component:
        output = next(
            (
                predictor.output
                for predictor in inputs.predictors
                if predictor.prefix == prefix
            ),
            None,
        )
        if output is not None:
            completion = Completion(
                axis=f"{prefix}FormulaCategory",
                size=output.size,
                coordinates=tuple(range(output.size)),
                prefix=prefix,
            )
    return ParameterSlot(
        name=aux.name,
        eta=f"{prefix}eta",
        link=aux.link,
        free=f"{prefix}eta_nonbaseline" if aux.per_component else None,
        completion=completion,
        axis="FormulaCategoryLogit" if aux.per_component else None,
    )


def _auxiliary_source(
    aux: AuxParam, inputs: KernelInputs, formula: Formula
) -> ParameterSource | None:
    """Choose the source of one auxiliary parameter without a precomposition.

    Parameters
    ----------
    aux : AuxParam
        Declared parameter.
    inputs : KernelInputs
        The formulas, fixed values, and priors.
    formula : Formula
        Formula with the response and its modifiers.

    Returns
    -------
    ParameterSource or None
        A constant, a predictor, or a site; None when the known errors of an
        ``se()`` term give the parameter.
    """
    error = formula.modifiers.standard_error
    if aux.name == inputs.family.error_scale and error is not None and not error.sigma:
        return None
    if aux.name in inputs.fixed:
        return ConstantSource(value=float(inputs.fixed[aux.name]))
    if aux.name in inputs.distributional:
        return PredictorSource(predictor=inputs.predictor(f"dpar_{aux.name}_"))
    if aux.fixed_default is not None and aux.name not in inputs.priors:
        return ConstantSource(value=aux.fixed_default)
    prior = (
        inputs.priors[aux.name]
        if aux.name in inputs.priors
        else aux.with_response_minimum(aux.prior, np.asarray(formula.response_values))
    )
    return SiteSource(
        site=SiteDecl(
            name=aux.name,
            class_=aux.parameter_class,
            default=prior,
            axes=("FormulaCategoryLogit",) if aux.per_component else (),
        ),
        floor=aux.domain.lower if aux.domain.lower else None,
    )


def _scale_stack(
    aux: AuxParam,
    inputs: KernelInputs,
    labels: tuple[str, ...],
    sources: dict[str, ParameterSource],
) -> tuple[Precomposition, tuple[ParameterSlot, ...]]:
    """Compute a per-component scale from one scale per free category.

    A category with a ``sigma<c>`` formula has a predictor under the scale
    link, a fixed ``sigma<c>`` a constant, and any other category a positive
    site; the scale is their ``(Resp, K - 1)`` stack.

    Parameters
    ----------
    aux : AuxParam
        The per-component scale.
    inputs : KernelInputs
        The per-category formulas and values.
    labels : tuple[str, ...]
        Component labels.
    sources : dict[str, ParameterSource]
        Sources, receiving those of the per-category scales.

    Returns
    -------
    tuple[Precomposition, tuple[ParameterSlot, ...]]
        The stack, and the per-category slots it reads.
    """
    family = inputs.family
    check_scale_formulas(family, labels, inputs.distributional | inputs.fixed.keys())
    parts: list[KernelPart] = [Steps(steps=(row_zero("scale_row_zero"),))]
    slots: list[ParameterSlot] = []
    values: list[LetExprNode] = []
    for _, name in free_scales(family, labels):
        slots.append(ParameterSlot(name=name, eta=f"dpar_{name}_eta", link=aux.link))
        if name in inputs.fixed:
            sources[name] = ConstantSource(value=float(inputs.fixed[name]))
        elif name in inputs.distributional:
            sources[name] = PredictorSource(predictor=inputs.predictor(f"dpar_{name}_"))
        else:
            sources[name] = SiteSource(
                site=SiteDecl(name=name, class_=aux.parameter_class, default=aux.prior)
            )
        parts.append(BindSlot(slot=name))
        values.append(_add(_var("scale_row_zero"), _var(name)))
    parts.append(
        Steps(
            steps=stack_rows(
                aux.name,
                enumerate(values),
                ("category_logit", "FormulaCategoryLogit"),
                stem=f"{aux.name}_by_category",
            )
        )
    )
    return (
        Precomposition(
            parts=tuple(parts),
            level="name",
            shape=ValueShape(rows=True, coordinates=True),
        ),
        tuple(slots),
    )


def _category_stack(
    inputs: KernelInputs,
    formula: Formula,
    sources: dict[str, ParameterSource],
) -> tuple[Precomposition, tuple[ParameterSlot, ...]]:
    """Compute the free category logits from one predictor per category.

    Each ``mu<c>`` has an ordinary predictor with the ``dpar_mu<c>_`` prefix
    or, without a formula, its own or the main nonlinear expression; the free
    logits are their ``(Resp, K - 1)`` stack.

    Parameters
    ----------
    inputs : KernelInputs
        The per-category formulas and expressions.
    formula : Formula
        Formula with the response.
    sources : dict[str, ParameterSource]
        Sources, receiving those of the categories.

    Returns
    -------
    tuple[Precomposition, tuple[ParameterSlot, ...]]
        The stack, and the per-category slots it reads.

    Raises
    ------
    ValueError
        If the family has no reference category or a category has no
        predictor.
    """
    family = inputs.family
    assert family.response is not None
    graph = inputs.graph
    roots = {} if graph is None else graph.roots
    main = roots.get("mu")
    labels = inputs.labels
    if family.response.reference_index(labels) is None:
        raise ValueError(
            "category-specific formulas require a reference category (refcat)"
        )
    categories = free_categories(family, labels)
    missing = [
        name
        for _, name in categories
        if name not in inputs.category_parameters and main is None
    ]
    if missing:
        raise ValueError(
            f"category-specific formulas require every free category: missing {missing}"
        )
    parts: list[KernelPart] = []
    slots: list[ParameterSlot] = []
    for index, name in categories:
        eta = f"dpar_{name}_eta"
        slot = ParameterSlot(name=name, eta=eta, link=links["identity"])
        if name in inputs.distributional:
            sources[name] = PredictorSource(predictor=inputs.predictor(f"dpar_{name}_"))
        else:
            assert graph is not None
            expression = roots.get(name, main)
            component = None
            if isinstance(expression, ComponentExpression):
                component = str(labels[index])
                expression = expression.at(component)
            assert expression is not None
            slot = slot.with_(
                precomposition=Precomposition(
                    parts=(
                        Steps(
                            steps=(
                                _let(
                                    eta,
                                    _factor(
                                        ((GRAPH_ROW, "Resp"),),
                                        body=graph_expression_at(
                                            graph,
                                            expression,
                                            _var(GRAPH_ROW),
                                            component,
                                        ),
                                    ),
                                ),
                            )
                        ),
                    ),
                    value=_var(eta),
                )
            )
        slots.append(slot)
        parts.append(BindSlot(slot=name))
    parts.append(
        Steps(
            steps=(
                row_zero("category_row_zero"),
                *stack_rows(
                    "eta_nonbaseline",
                    (
                        (index, _add(_var("category_row_zero"), _var(slot.eta or "")))
                        for index, slot in enumerate(slots)
                    ),
                    ("category_logit", "FormulaCategoryLogit"),
                    stem="eta_by_category",
                ),
            )
        )
    )
    return (
        Precomposition(
            parts=tuple(parts),
            level="free",
            shape=ValueShape(rows=True, coordinates=True),
        ),
        tuple(slots),
    )


def precompose_graph(
    kernel: ResponseKernel,
    inputs: KernelInputs,
    formula: Formula,
    sources: dict[str, ParameterSource],
) -> ResponseKernel:
    """Fold the nonlinear predictor graph into the kernel.

    The graph's parameters become slots with predictors, bound first with
    the intermediate expressions; each root computes its slot. A root of a
    slot the kernel computes already (the categories of a categorical law)
    is read there.

    Parameters
    ----------
    kernel : ResponseKernel
        Kernel with every family slot.
    inputs : KernelInputs
        The graph and the planned predictors.
    formula : Formula
        Formula with the response.
    sources : dict[str, ParameterSource]
        Sources, receiving those of the graph's parameters.

    Returns
    -------
    ResponseKernel
        The kernel computing the graph's roots.

    Raises
    ------
    ValueError
        If a root's carrier does not match its slot.
    """
    graph = inputs.graph
    if graph is None:
        return kernel
    leaves: list[ParameterSlot] = []
    parts: list[KernelPart] = []
    for name in graph.leaves:
        leaf = f"nlpar_{name}"
        leaves.append(ParameterSlot(name=leaf, eta=f"{leaf}_raw", row_binder=GRAPH_ROW))
        sources[leaf] = PredictorSource(predictor=inputs.predictor(f"nlpar_{name}_"))
        parts.append(BindSlot(slot=leaf))
    for name, expression in graph.expressions.items():
        parts.append(graph_value(graph, f"nlpar_{name}", expression))
    data = tuple(
        InputDecl(
            name=column.qvr_name,
            element="Bool"
            if column.data.dtype.kind == "b"
            else "Int"
            if column.data.dtype.kind in "iu"
            else "Real",
            axes=("Resp",),
            declared=True,
        )
        for column in graph.inputs.values()
    )
    family = inputs.family
    labels = inputs.labels if family.response is not None else ()
    for name, value in graph.roots.items():
        if not kernel.has_slot(name) or kernel.slot(name).precomposition is not None:
            continue
        slot = kernel.slot(name)
        if slot.free is not None and name == "mu":
            # The free logits of a categorical law.
            response = family.response
            assert response is not None
            coordinates = response.coordinate_indices(labels)
            step = graph_value(
                graph,
                slot.free,
                value,
                axis="FormulaCategoryLogit",
                components=tuple(labels[index] for index in coordinates),
            )
            precomposition = Precomposition(
                parts=(step,),
                level="free",
                shape=ValueShape(rows=True, coordinates=True),
            )
        elif name == "mu":
            if isinstance(value, ComponentExpression):
                raise ValueError(
                    "a scalar response requires a scalar nonlinear mu expression"
                )
            root = "graph_mu_eta"
            total: LetExprNode = _var(root)
            if inputs.host is not None:
                total = _add(total, _var(inputs.host))
            precomposition = Precomposition(
                parts=(graph_value(graph, root, value),), value=total
            )
        else:
            assert slot.eta is not None
            components = None
            shape = ValueShape(rows=True)
            if isinstance(value, ComponentExpression):
                if slot.free is None:
                    raise ValueError(
                        f"family parameter {name!r} requires a scalar nonlinear root"
                    )
                response = family.response
                assert response is not None
                components = tuple(
                    labels[index] for index in response.coordinate_indices(labels)
                )
                shape = ValueShape(rows=True, coordinates=True)
            precomposition = Precomposition(
                parts=(
                    graph_value(
                        graph,
                        slot.eta,
                        value,
                        axis=f"dpar_{name}_FormulaCategoryLogit",
                        components=components,
                    ),
                ),
                value=_var(slot.eta),
                shape=shape,
            )
        kernel = kernel.precompose(name, precomposition)
        sources.pop(name, None)
    return kernel.with_prelude(leaves, parts, data)


def family_kernel(
    inputs: KernelInputs, formula: Formula
) -> tuple[ResponseKernel, dict[str, ParameterSource]]:
    """State the kernel of a family and the sources of its slots.

    Parameters
    ----------
    inputs : KernelInputs
        The family, the planned predictors, the formulas,
        fixed values, priors, and residual law.
    formula : Formula
        Formula with the response.

    Returns
    -------
    tuple[ResponseKernel, dict[str, ParameterSource]]
        The kernel and the source of each slot a source feeds. A mixture's
        kernel emits its components as nested blocks and has no slot.
    """
    family = inputs.family
    if family.mixture is not None:
        return mixture_kernel(inputs, formula), {}
    location = inputs.predictor("")
    graph = inputs.graph
    roots = {} if graph is None else graph.roots
    sources: dict[str, ParameterSource] = {}
    slots: list[ParameterSlot] = []
    head: list[KernelPart] = []
    objects: list[AxisDecl] = []
    body: list[KernelPart] = []
    if family.baseline is not None:
        head.append(BaselineHazard())
    response = family.response
    labels = inputs.labels if response is not None else ()
    if response is not None:
        reference = response.reference_index(labels)
        objects.append(
            AxisDecl(
                name="FormulaCategory",
                size=inputs.categories,
            )
        )
        axis = response_axis(family, inputs.labels)
        assert axis is not None
        objects.append(AxisDecl(name=axis.name, size=axis.size))
        mu = ParameterSlot(
            name="mu",
            eta="eta",
            link=family.location_link,
            free="eta_nonbaseline",
            completion=None
            if response.kind == "logistic_normal"
            else Completion(
                axis="FormulaCategory",
                size=len(labels),
                reference=reference,
                coordinates=tuple(
                    index for index in range(len(labels)) if index != reference
                ),
            ),
        )
        main = roots.get("mu")
        if inputs.category_parameters or (
            main is not None and not isinstance(main, ComponentExpression)
        ):
            precomposition, categories = _category_stack(inputs, formula, sources)
            mu = mu.with_(precomposition=precomposition)
            slots.extend(categories)
        elif main is None:
            sources["mu"] = PredictorSource(predictor=location)
        body.append(BindSlot(slot="mu"))
    else:
        mu = ParameterSlot(
            name="mu",
            eta="eta",
            link=None
            if family.retains_log_location
            else links["identity"]
            if family.ordinal is not None
            else family.location_link,
        )
        if "mu" not in roots:
            sources["mu"] = PredictorSource(predictor=location)
        body.append(EmitSlot(slot="mu"))
    slots.insert(0, mu)
    auxiliaries = (
        family.aux_params if response is None or family.vector_response else ()
    )
    for aux in auxiliaries:
        slot = _auxiliary_slot(aux, inputs)
        if (
            aux.per_component
            and aux.name not in inputs.fixed
            and aux.name not in roots
            and any(
                is_scale_parameter(family, name)
                for name in inputs.distributional | inputs.fixed.keys()
            )
        ):
            precomposition, scales = _scale_stack(aux, inputs, labels, sources)
            slot = slot.with_(precomposition=precomposition)
            slots.append(slot)
            slots.extend(scales)
        else:
            slots.append(slot)
            if aux.name not in roots or aux.name in inputs.fixed:
                source = _auxiliary_source(aux, inputs, formula)
                if source is not None:
                    sources[aux.name] = source
        body.append(BindSlot(slot=aux.name))
    residual = inputs.residual
    for site in residual.sites if residual is not None else ():
        slots.append(ParameterSlot(name=site.name))
        sources[site.name] = (
            ConstantSource(value=float(inputs.fixed[site.name]))
            if site.name in inputs.fixed
            else SiteSource(site=site)
        )
        body.append(BindSlot(slot=site.name))
    if response is not None:
        if family.vector_response:
            assert response.kind in (
                "multinomial",
                "dirichlet",
                "dirichlet_multinomial",
                "dirichlet2",
                "logistic_normal",
            )
            body.append(
                VectorArguments(
                    law=response.kind,
                    width=len(labels) - 1,
                    reference=response.reference_index(labels)
                    if response.kind == "logistic_normal"
                    else None,
                )
            )
        else:
            body.append(CategoricalArguments())
    elif family.ordinal is not None:
        transitions = tuple(
            output for output, _ in location.parts() if output != location.output
        )
        body.append(OrdinalThresholds())
        body.extend(
            EmitSlotAxis(slot="mu", output=output, binding="eta_nonbaseline")
            for output in transitions
            if output is not None
        )
        body.append(OrdinalThresholdSteps())
        body.append(BindSlot(slot="mu"))
        body.append(
            OrdinalMasses(
                # Grouped thresholds are scored on category log masses.
                cutpoint_law=family.scores_cutpoints
                and formula.modifiers.threshold_groups is None,
                unit_discrimination=inputs.fixed.get("disc", 1.0) == 1.0
                and "disc" not in inputs.distributional
                and "disc" not in inputs.priors
                and not transitions,
            )
        )
    else:
        body.append(BindSlot(slot="mu"))
        body.append(NativeArguments())
    body.append(Observe())
    kernel = ResponseKernel(
        slots=tuple(slots),
        objects=tuple(objects),
        head=tuple(head),
        body=tuple(body),
        response=_qvr_name(formula.response_name),
    )
    return precompose_graph(kernel, inputs, formula, sources), sources
