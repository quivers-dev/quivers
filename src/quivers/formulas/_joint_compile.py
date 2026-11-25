"""Compose response-indexed uses of the existing formula-to-QVR emitter."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import TypeVar, cast

import didactic.api as dx
import torch
from quivers.family_catalog import FamilyCatalog

from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
    LetExprLiteral,
    LetExprNode,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecValueParameter,
    SampleStep,
    Statement,
    TypeFromExpr,
    LetStep,
    ObserveStep,
    OptionEntry,
    OptionName,
    DrawArgNamed,
    TypeName,
)
from quivers.dsl.program_templates import (
    TemplateBindings,
    _bound_names,
    _rename_step,
    substitute_program_object,
    draw_argument_expression,
)
from quivers.dsl.step_resolution import _step_value_names
from quivers.formulas._autocor import autocor_binding
from quivers.formulas._ast import _var, _let, _factor, _index, _add, _mul
from quivers.formulas._group_law import (
    GroupBlockLaw,
    IdentityCovariance,
    LevelSite,
    law_inputs,
    renamed_law,
)
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._group_emission import draw_level, emit_group_block
from quivers.formulas._group_plan import check_names, level_law, plan_joint_block
from quivers.formulas._predictor import (
    Predictor,
    group_term,
    response_axis,
    slot_link,
)
from quivers.formulas._term import OutputAxis, bind_inputs, declare_inputs
from quivers.formulas._brms_priors import FLAT_CATALOG, improper_sites
from quivers.formulas._prior_emission import (
    apply_prior_plan,
    build_prior_targets,
    ordinary_prior_step,
    order_prior_dependencies,
)
from quivers.formulas._priors import (
    PriorInput,
    PriorSet,
    PriorSpec,
    PriorTarget,
    normalize_priors,
    resolve_priors,
)
from quivers.formulas._response import response_tensor
from quivers.formulas.compile import FormulaToQVRModule, _draw
from quivers.formulas.formula import Formula, FormulaData, RandomTerm, _qvr_name

_Value = TypeVar("_Value")

#: Response label to its prepared lens and formula.
type _Components = Mapping[str, tuple[FormulaToQVRModule, Formula]]

#: Shared group coefficients keyed by local prefix, group, and slope.
type _GroupValues = dict[tuple[str, str, str], LetExprNode]


@dataclass(frozen=True)
class _ResponseEmission:
    """One response emission renamed into its joint namespace.

    Parameters
    ----------
    name : str
        Authored response label.
    prefix : str
        Namespace prefix applied to the response's local bindings.
    lens : FormulaToQVRModule
        Lens that emitted the response.
    formula : Formula
        Response formula with reconciled joint grouping codes.
    module : Module
        Namespaced response objects and program.
    complement : FormulaData
        Data complement retained by the lens.
    names : Mapping[str, str]
        Local binding names and their namespaced names.
    objects : Mapping[str, str]
        Local object names and their namespaced or shared names.
    families : FamilyCatalog or None
        Laws of the response and, under namespaced names, of its terms.
    """

    name: str
    prefix: str
    lens: FormulaToQVRModule
    formula: Formula
    module: Module
    complement: FormulaData
    names: Mapping[str, str]
    objects: Mapping[str, str]
    families: FamilyCatalog | None


@dataclass(frozen=True)
class _JointEmission:
    """One composed joint program with its response and latent metadata.

    Parameters
    ----------
    module : Module
        The composed QVR module, exporting one program.
    responses : Mapping[str, _ResponseEmission]
        Namespaced response emissions, in authored order.
    group_laws : tuple[GroupBlockLaw, ...]
        Laws of the shared and response-local group blocks.
    level_sites : tuple[LevelSite, ...]
        Sites holding values for every level of a grouping factor.
    prior_targets : tuple[PriorTarget, ...]
        Every latent site eligible for prior rules.
    fixed_inputs : Mapping[str, torch.Tensor]
        Runtime inputs bound by the joint program itself.
    families : FamilyCatalog
        Catalog extended with every custom response family.
    """

    module: Module
    responses: Mapping[str, _ResponseEmission]
    group_laws: tuple[GroupBlockLaw, ...]
    level_sites: tuple[LevelSite, ...]
    prior_targets: tuple[PriorTarget, ...]
    fixed_inputs: Mapping[str, torch.Tensor]
    families: FamilyCatalog

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Return the coordinate layout of each group block.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Raw sites and coefficient slices derived from the block laws.
        """
        return tuple(law.layout for law in self.group_laws)


_LevelFactors = tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]


def _declared_level_factors(components: _Components) -> _LevelFactors:
    """Collect the level factors every response declares for its levels.

    A ``by`` partition and a prior weight are properties of a grouping
    level, so responses that declare them must agree on every level they
    share.

    Parameters
    ----------
    components : Mapping[str, tuple[FormulaToQVRModule, Formula]]
        Response labels and their prepared lens/formula pairs, each coded on
        its own levels.

    Returns
    -------
    tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]
        The declared ``by`` level and the declared prior weight of each
        labeled level, by grouping factor.

    Raises
    ------
    ValueError
        If two declarations give one level of a grouping factor different
        ``by`` levels or different prior weights.
    """
    by: dict[str, dict[str, str]] = {}
    weights: dict[str, dict[str, float]] = {}
    for lens, formula in components.values():
        for _, _, predictor in lens._predictor_items(formula):
            for option in predictor.grouping.options:
                group = option.group
                labels = predictor.group_levels[group]
                if option.by_levels:
                    for label, code in zip(labels, option.by_codes, strict=True):
                        value = option.by_levels[code]
                        if by.setdefault(group, {}).setdefault(label, value) != value:
                            raise ValueError(
                                f"gr({group}, by = ...): a by partition assigns "
                                f"each level of {group!r} one by level, and the "
                                f"responses give level {label!r} both "
                                f"{by[group][label]!r} and {value!r}"
                            )
                if option.prior_weights:
                    for label, weight in zip(labels, option.prior_weights, strict=True):
                        known = weights.setdefault(group, {}).setdefault(label, weight)
                        if known != weight:
                            raise ValueError(
                                f"gr({group}, pw = ...): a prior weight belongs to "
                                f"a level of {group!r}, and the responses give "
                                f"level {label!r} both {known!r} and {weight!r}"
                            )
    return by, weights


def _recode(
    formula: Formula, levels: Mapping[str, tuple[str, ...]], factors: _LevelFactors
) -> Formula:
    """Retain group identities while reconciling response-specific subsets.

    Parameters
    ----------
    formula : Formula
        Response formula carrying its own group levels and codes.
    levels : Mapping[str, tuple[str, ...]]
        Joint levels for every group, a superset of each response's levels.
    factors : tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]
        The ``by`` level and the prior weight declared for each labeled
        level, by grouping factor.

    Returns
    -------
    Formula
        The formula with group codes and level-coded ``gr()`` options over
        the joint levels.

    Raises
    ------
    ValueError
        If the formula declares a ``by`` partition or prior weights for a
        grouping factor and a joint level of that factor has no declared
        value.
    """
    indices = {}
    for group, original in formula.group_levels.items():
        positions = {name: index for index, name in enumerate(levels[group])}
        indices[group] = tuple(
            positions[original[index]] for index in formula.group_indices[group]
        )
    options = []
    for option in formula.grouping.options:
        group = option.group
        joint = levels[group]
        if option.by_levels:
            declared = factors[0][group]
            missing = [label for label in joint if label not in declared]
            if missing:
                raise ValueError(
                    f"gr({group}, by = {option.options.by}): every level of "
                    f"{group!r} has a coefficient with the scale of its by level, "
                    f"and levels {missing!r} have no by level in the rows of a "
                    "response that declares the partition"
                )
            by_levels = tuple(sorted(set(declared.values())))
            position = {label: index for index, label in enumerate(by_levels)}
            option = option.with_(
                by_levels=by_levels,
                by_codes=tuple(position[declared[label]] for label in joint),
            )
        if option.prior_weights:
            known = factors[1][group]
            missing = [label for label in joint if label not in known]
            if missing:
                raise ValueError(
                    f"gr({group}, pw = {option.options.prior_weight}): every "
                    f"level of {group!r} has innovations with a weighted prior, "
                    f"and levels {missing!r} have no weight in the rows of a "
                    "response that declares the weights"
                )
            option = option.with_(prior_weights=tuple(known[label] for label in joint))
        options.append(option)
    return formula.with_(
        group_levels={group: levels[group] for group in formula.group_levels},
        group_indices=indices,
        grouping=formula.grouping.with_(options=tuple(options)),
    )


def _normalized_components(
    components: _Components,
) -> tuple[dict[str, tuple[FormulaToQVRModule, Formula]], dict[str, tuple[str, ...]]]:
    """Reconcile actual group labels once across all retained predictors.

    Parameters
    ----------
    components : Mapping[str, tuple[FormulaToQVRModule, Formula]]
        Response labels and their prepared lens/formula pairs.

    Returns
    -------
    tuple[dict, dict]
        Copied lens/formula pairs recoded to the joint levels, and the joint
        levels of every group in first-appearance order.

    Raises
    ------
    ValueError
        If two grouping factors share a QVR name, a known covariance names no
        grouping factor, or the responses declare inconsistent or incomplete
        ``by`` partitions or prior weights.
    """
    levels: dict[str, tuple[str, ...]] = {}
    tokens: dict[str, str] = {}
    for lens, formula in components.values():
        for _, _, predictor in lens._predictor_items(formula):
            for group, names in predictor.group_levels.items():
                token = _qvr_name(group)
                if tokens.setdefault(token, group) != group:
                    raise ValueError(
                        "joint grouping factors collide after QVR normalization"
                    )
                current = levels.get(group, ())
                levels[group] = (
                    *current,
                    *(name for name in names if name not in current),
                )
    unknown = {
        group for lens, _ in components.values() for group in lens._group_covariances
    } - levels.keys()
    if unknown:
        raise ValueError(
            f"group_covariances names unknown grouping factors {sorted(unknown)!r}"
        )
    factors = _declared_level_factors(components)
    result: dict[str, tuple[FormulaToQVRModule, Formula]] = {}
    for name, (original, formula) in components.items():
        lens = copy(original)
        lens._distributional = {
            key: _recode(value, levels, factors)
            for key, value in original._distributional.items()
        }
        if original._predictor_graph is not None:
            lens._predictor_graph = original._predictor_graph.with_(
                leaves={
                    key: _recode(value, levels, factors)
                    for key, value in original._predictor_graph.leaves.items()
                }
            )
        lens._emitted_groups = []
        recoded = _recode(formula, levels, factors)
        used = {
            group
            for _, _, predictor in lens._predictor_items(recoded)
            for group in predictor.group_levels
        }
        lens._group_covariances = {
            group: covariance
            for group, covariance in original._group_covariances.items()
            if group in used
        }
        result[name] = (lens, recoded)
    return result, levels


def _indices(value: _Value, objects: Mapping[str, str]) -> _Value:
    """Rename nominal indices throughout a generated runtime parameter type.

    Parameters
    ----------
    value : _Value
        A type expression, a tuple of them, or a leaf field value.
    objects : Mapping[str, str]
        Local object names and their namespaced names.

    Returns
    -------
    _Value
        The value with every nominal index renamed, of the same type.
    """
    if isinstance(value, QiecIndexName):
        return value.with_(name=objects.get(value.name, value.name))
    if isinstance(value, tuple):
        return cast(_Value, tuple(_indices(item, objects) for item in value))
    if isinstance(value, dx.Model):
        return value.with_(
            **{
                key: _indices(getattr(value, key), objects)
                for key in type(value).__field_specs__
            }
        )
    return value


def _namespace(
    name: str,
    lens: FormulaToQVRModule,
    formula: Formula,
    *,
    shared_objects: Mapping[str, str],
    global_names: set[str],
    supplied_groups: Mapping[tuple[str, str, str], LetExprNode] | None = None,
    supplied_sites: Collection[str] = (),
    supplied_inputs: Mapping[str, LetExprNode] | None = None,
    local_module: Callable[[str, Module], Module] | None = None,
) -> _ResponseEmission:
    """Apply the existing program-template substitution to one emitted response.

    Parameters
    ----------
    name : str
        Authored response label, which determines the namespace prefix.
    lens : FormulaToQVRModule
        Lens that emits the response.
    formula : Formula
        Response formula with joint grouping codes.
    shared_objects : Mapping[str, str]
        Object names kept shared across responses, with their joint names.
    global_names : set[str]
        Bindings and inputs owned by the joint program, which are neither
        renamed nor declared again by the response.
    supplied_groups : Mapping or None
        Shared group coefficients replacing the response's own group blocks.
    supplied_sites : Collection[str]
        Level hyperparameters of grouping factors, which the response reads
        from the joint program.
    supplied_inputs : Mapping[str, LetExprNode] or None
        Explicit expressions for response inputs over joint bindings.
    local_module : Callable[[str, Module], Module] or None
        Rewrite of the response's local module before namespacing, used by
        prediction to resize the row axis and select varying effects.

    Returns
    -------
    _ResponseEmission
        The namespaced response emission.

    Raises
    ------
    ValueError
        If an explicit input conflicts with its symbolic expression or names
        no input of the response.
    TypeError
        If a formula object lacks its finite declaration.
    """
    module, complement = lens._forward(
        formula,
        supplied_groups=supplied_groups,
        supplied_sites=supplied_sites,
        check=False,
    )
    if local_module is not None:
        module = local_module(name, module)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    prefix = "response_" + _qvr_name(name) + "__"
    objects = {
        value: shared_objects.get(value, prefix + value)
        for item in module.statements
        if isinstance(item, ObjectDecl)
        for value in item.names
    }
    variables = set(_bound_names(program.draws)) | {
        parameter.name for parameter in program.value_parameters
    }
    variables.update(
        value for step in program.draws for value in _step_value_names(step)
    )
    variables.update(
        law.support_site for law in lens.group_laws if law.support_site is not None
    )
    names = {value: prefix + value for value in variables - global_names}
    explicit_inputs = supplied_inputs or {}
    supplied_inputs = lens.symbolic_column_expressions(formula)
    for value, expression in explicit_inputs.items():
        if value in supplied_inputs and supplied_inputs[value] != expression:
            raise ValueError(
                f"supplied input {value!r} conflicts with its symbolic expression"
            )
        supplied_inputs[value] = expression
    known_inputs = {parameter.name for parameter in program.value_parameters}
    # Fixed columns may be inferred inputs of an ordinary emission, so use the
    # actual retained input metadata as well as explicit runtime parameters.
    known_inputs.update(lens.fixed_column_observations(formula))
    unknown = set(supplied_inputs) - known_inputs
    if unknown:
        raise ValueError(f"unknown supplied response inputs: {sorted(unknown)!r}")
    for value in supplied_inputs:
        names.setdefault(value, prefix + value)
    # Two responses may draw from one law over different coordinates.
    laws = lens.term_laws
    morphisms = (
        {}
        if laws is None
        else {record.name: prefix + record.name for record in laws.definitions}
    )
    families = lens._family.custom.catalog if lens._family.custom is not None else None
    if laws is not None:
        laws = laws.renamed(morphisms)
        families = laws if families is None else families.with_catalog(laws)
    bindings = TemplateBindings(objects=objects, values={}, morphisms=morphisms)
    parameters = tuple(
        parameter.with_(
            name=names.get(parameter.name, parameter.name),
            type_expr=_indices(parameter.type_expr, objects),
        )
        for parameter in program.value_parameters
        if parameter.name not in supplied_inputs and parameter.name not in global_names
    )
    statements = []
    for item in module.statements:
        if isinstance(item, ObjectDecl):
            if not isinstance(item.init, TypeFromExpr):
                raise TypeError(
                    "formula objects require their retained finite declaration"
                )
            statements.append(
                item.with_(
                    names=tuple(objects[value] for value in item.names),
                    init=item.init.with_(
                        expr=substitute_program_object(
                            item.init.expr,
                            {
                                key: TypeName(name=value)
                                for key, value in objects.items()
                            },
                        )
                    ),
                )
            )
        elif isinstance(item, ProgramDecl):
            statements.append(
                item.with_(
                    name=prefix + "model",
                    value_parameters=parameters,
                    domain=substitute_program_object(
                        item.domain,
                        {key: TypeName(name=value) for key, value in objects.items()},
                    ),
                    codomain=substitute_program_object(
                        item.codomain,
                        {key: TypeName(name=value) for key, value in objects.items()},
                    ),
                    draws=(
                        *(
                            LetStep(name=names[value], value=expression)
                            for value, expression in supplied_inputs.items()
                        ),
                        *(_rename_step(step, bindings, names) for step in item.draws),
                    ),
                    return_vars=tuple(
                        names.get(value, value) for value in item.return_vars
                    ),
                )
            )
    return _ResponseEmission(
        name,
        prefix,
        lens,
        formula,
        Module(statements=tuple(statements)),
        complement,
        names,
        objects,
        families,
    )


def _renamed(names: Mapping[str, str], value: str | None) -> str | None:
    """Rename an optional binding, keeping names outside the namespace.

    Parameters
    ----------
    names : Mapping[str, str]
        Local names and their namespaced names.
    value : str or None
        Optional local name.

    Returns
    -------
    str or None
        The namespaced name, the name itself when it is not renamed, or None.
    """
    return None if value is None else names.get(value, value)


def _law(
    law: GroupBlockLaw,
    names: Mapping[str, str],
    objects: Mapping[str, str],
    prefix: str,
) -> GroupBlockLaw:
    """Carry a response-local group block into the joint namespace.

    Parameters
    ----------
    law : GroupBlockLaw
        Response-local group block law.
    names : Mapping[str, str]
        Local binding names and their namespaced names.
    objects : Mapping[str, str]
        Local object names and their namespaced names.
    prefix : str
        Response namespace prefix.

    Returns
    -------
    GroupBlockLaw
        The law with every identity, axis, and site renamed.
    """
    return renamed_law(
        law, names, objects, lambda local: prefix + local, prefix + law.identity
    )


def _response_parts(
    responses: Mapping[str, _ResponseEmission],
    initial_objects: Sequence[Statement],
    global_steps: Sequence[ProgramStep],
    global_parameters: Sequence[QiecValueParameter],
    global_laws: Sequence[GroupBlockLaw],
) -> tuple[
    list[Statement],
    list[ProgramStep],
    list[QiecValueParameter],
    list[str],
    list[PriorTarget],
    list[GroupBlockLaw],
]:
    """Collect namespaced responses without changing their conditional prior laws.

    Parameters
    ----------
    responses : Mapping[str, _ResponseEmission]
        Namespaced response emissions, in authored order.
    initial_objects : Sequence[Statement]
        Joint object declarations preceding the response objects.
    global_steps : Sequence[ProgramStep]
        Joint program steps preceding the response steps.
    global_parameters : Sequence[QiecValueParameter]
        Joint runtime parameters preceding the response parameters.
    global_laws : Sequence[GroupBlockLaw]
        Shared group-block laws preceding the response laws.

    Returns
    -------
    tuple
        Object declarations, program steps, runtime parameters, returned
        names, prior targets, and group-block laws of the joint program.

    Raises
    ------
    ValueError
        If two responses declare one nominal group axis differently, or two
        group blocks would own the same name.
    """
    statements = list(initial_objects)
    steps, parameters = list(global_steps), list(global_parameters)
    returned: list[str] = []
    targets: list[PriorTarget] = []
    laws = list(global_laws)
    for name, response in responses.items():
        for statement in response.module.statements:
            if isinstance(statement, ObjectDecl):
                existing = next(
                    (
                        item
                        for item in statements
                        if isinstance(item, ObjectDecl)
                        and item.names == statement.names
                    ),
                    None,
                )
                if existing is None:
                    statements.append(statement)
                elif existing != statement:
                    raise ValueError(
                        "joint response graphs disagree on a nominal group axis"
                    )
            elif isinstance(statement, ProgramDecl):
                steps.extend(statement.draws)
                parameters.extend(statement.value_parameters)
                returned.extend(statement.return_vars)
        targets.extend(
            target.with_(
                site=response.names[target.site],
                axes=tuple(response.objects.get(axis, axis) for axis in target.axes),
                response=name,
            )
            for target in response.lens.prior_targets
        )
        laws.extend(
            _law(law, response.names, response.objects, response.prefix)
            for law in response.lens.group_laws
        )
    check_names(laws)
    return statements, steps, parameters, returned, targets, laws


def _member_axis(
    name: str,
    lens: FormulaToQVRModule,
    formula: Formula,
    prefix: str,
    term: RandomTerm,
) -> tuple[str, int] | None:
    """Read an explicit response-component or ordinal-transition coordinate.

    Parameters
    ----------
    name : str
        Authored response label.
    lens : FormulaToQVRModule
        Lens of the response.
    formula : Formula
        Response formula.
    prefix : str
        Local coefficient prefix of the predictor.
    term : RandomTerm
        Shared group term.

    Returns
    -------
    tuple[str, int] or None
        The namespaced coordinate axis and its width, or None for a scalar
        member.
    """
    namespace = "response_" + _qvr_name(name) + "__"
    if lens._family.response is not None and (
        prefix == ""
        or lens._family.component_scale is not None
        and prefix == f"dpar_{lens._family.component_scale.name}_"
    ):
        labels = lens._component_labels(formula)
        width = len(lens._family.response.coordinate_indices(labels))
        axis = (
            "FormulaCategoryLogit" if prefix == "" else prefix + "FormulaCategoryLogit"
        )
        return namespace + axis, width
    if term.column is not None and term.column.category_specific:
        count = lens._category_count(formula, family=lens._family.name) - 1
        if (
            lens._family.ordinal is not None
            and lens._family.ordinal.kind == "hurdle_cumulative"
        ):
            count -= 1
        return namespace + "FormulaCutpoint", count
    return None


def _validate_residual_rows(components: _Components) -> None:
    """Require an actual common observation domain for a joint residual event.

    Parameters
    ----------
    components : Mapping[str, tuple[FormulaToQVRModule, Formula]]
        Response labels and their recoded lens/formula pairs.

    Raises
    ------
    ValueError
        If the families do not declare one common elliptical law, a
        response event is not scalar, the rows, subsets, or weights differ,
        a response is censored or truncated, or Student responses do not
        share one ``nu``.
    """
    laws = {lens._family.elliptical for lens, _ in components.values()}
    # Residual autocorrelation terms replace a response's law. brms refuses
    # its covariance terms beside estimated residual correlations; it admits
    # the conditional ARMA formulation there, which this emission lacks.
    for label, (lens, _) in components.items():
        binding = autocor_binding(lens._family)
        if binding is None:
            continue
        if binding.term.cov:
            raise ValueError(
                "Explicit covariance terms cannot be modeled when 'rescor' is "
                "estimated at the same time."
            )
        raise ValueError(
            f"response {label!r}: {binding.term.label} beside estimated residual "
            "correlations is not supported; use set_rescor(False)"
        )
    elliptical = next(iter(laws))
    if len(laws) != 1 or elliptical is None:
        raise ValueError(
            "residual correlations require homogeneous Gaussian or Student responses, "
            "families declaring one elliptical residual law (Family.elliptical)"
        )
    df = elliptical.df
    first = next(iter(components.values()))[1]
    first_weights = first.modifiers.weights
    for lens, formula in components.values():
        if formula.response_layout.event_shape:
            raise ValueError(
                "joint residual coordinates require scalar response events"
            )
        if formula.modifiers.subset != first.modifiers.subset or len(
            formula.response_values
        ) != len(first.response_values):
            raise ValueError(
                "residual correlations require identical original response rows"
            )
        weights = formula.modifiers.weights
        if (weights is None) != (first_weights is None) or (
            weights is not None
            and first_weights is not None
            and not torch.equal(
                torch.as_tensor(weights.data), torch.as_tensor(first_weights.data)
            )
        ):
            raise ValueError(
                "one joint residual event requires identical response weights"
            )
        modifiers = formula.modifiers
        if modifiers.censoring is not None or modifiers.truncation is not None:
            raise ValueError(
                "joint residual censoring and truncation require a multivariate recording law"
            )
        if df is not None and (
            df in lens._distributional
            or lens._predictor_graph is not None
            and df in lens._predictor_graph.roots
        ):
            raise ValueError(
                f"correlated Student responses use one shared {df} parameter"
            )
    if df is not None:
        values = {lens._fixed_parameters.get(df) for lens, _ in components.values()}
        if len(values) != 1:
            raise ValueError(
                f"correlated Student responses require one common fixed or sampled {df}"
            )


def _correlated_residual(
    program: ProgramDecl,
    responses: Mapping[str, _ResponseEmission],
    emitter: FormulaToQVRModule,
    statements: list[Statement],
    targets: Sequence[PriorTarget],
    fixed_inputs: Mapping[str, torch.Tensor],
) -> tuple[ProgramDecl, list[PriorTarget], dict[str, torch.Tensor]]:
    """Join existing scalar location/scale bindings under one vector residual law.

    Parameters
    ----------
    program : ProgramDecl
        Joint program whose response observations are replaced.
    responses : Mapping[str, _ResponseEmission]
        Namespaced scalar Gaussian or Student responses.
    emitter : FormulaToQVRModule
        Lens used to append the joint axis and correlation steps.
    statements : list[Statement]
        Module declarations, extended in place with the joint response axis.
    targets : Sequence[PriorTarget]
        Prior targets of the joint program.
    fixed_inputs : Mapping[str, torch.Tensor]
        Runtime inputs bound by the joint program.

    Returns
    -------
    tuple[ProgramDecl, list[PriorTarget], dict[str, torch.Tensor]]
        The program with one multivariate observation, its prior targets
        with the residual correlation and shared ``nu``, and the inputs with
        the stacked joint response.
    """
    names = tuple(responses)
    width = len(names)
    axis, row = "FormulaJointResponse", "FormulaJointRow"
    emitter._append_finite_object(statements, axis, width)
    removed = {
        response.names[_qvr_name(response.formula.response_name)]
        for response in responses.values()
    }
    arguments_by_response = {}
    for name, response in responses.items():
        observation = next(
            step
            for step in program.draws
            if isinstance(step, ObserveStep)
            and step.vars[0]
            == response.names[_qvr_name(response.formula.response_name)]
        )
        arguments_by_response[name] = {
            argument.parameter: draw_argument_expression(argument.value)
            for argument in observation.args or ()
            if isinstance(argument, DrawArgNamed)
        }
    shared = next(iter(responses.values())).lens
    elliptical = shared._family.elliptical
    assert elliptical is not None
    df = elliptical.df
    nu_sites = (
        {response.names.get(df) for response in responses.values()}
        if df is not None
        else set()
    )
    nu_prior = next(
        (
            step
            for step in program.draws
            if isinstance(step, SampleStep) and step.vars[0] in nu_sites
        ),
        None,
    )
    steps = [
        step
        for step in program.draws
        if not (
            isinstance(step, ObserveStep)
            and step.vars[0] in removed
            or isinstance(step, SampleStep)
            and step.vars[0] in nu_sites
        )
    ]
    targets = [target for target in targets if target.site not in nu_sites]
    emitter._append_residual_correlation(
        steps, axis, width, result="formula_joint_correlation"
    )
    targets.append(
        PriorTarget(
            site="rescor",
            class_="cor",
            event_rank=2,
            component_labels=names,
        )
    )
    if df is not None:
        fixed_nu = shared._fixed_parameters.get(df)
        if fixed_nu is None:
            assert nu_prior is not None
            steps.append(nu_prior.with_(vars=(df,)))
            original = next(
                target
                for response in responses.values()
                for target in response.lens.prior_targets
                if target.site == df
            )
            targets.append(original.with_(site=df, response=None))
        else:
            steps.append(_let(df, LetExprLiteral(value=float(fixed_nu))))
    steps.append(
        _let(
            "formula_joint_zeros",
            _factor((("row", row),), body=LetExprLiteral(value=0.0)),
        )
    )
    for result, quantity in (
        ("formula_joint_means_by_response", "mu"),
        ("formula_joint_scales_by_response", elliptical.scale),
    ):
        # Scalar-to-row broadcasting uses the ordinary typed binary operation;
        # nominal row ownership never depends on coincident axis cardinalities.
        cases = tuple(
            (
                index,
                _add(
                    arguments_by_response[name][
                        response.lens._family.native_parameter(quantity)
                    ],
                    _var("formula_joint_zeros"),
                ),
            )
            for index, (name, response) in enumerate(responses.items())
        )
        steps.append(_let(result, _factor((("component", axis),), cases=cases)))
    steps.append(
        _let(
            "formula_joint_mu",
            _factor(
                (("row", row), ("component", axis)),
                body=_index(
                    "formula_joint_means_by_response", _var("component"), _var("row")
                ),
            ),
        )
    )
    steps.append(
        _let(
            "formula_joint_scale",
            _factor(
                (("row", row), ("i", axis), ("j", axis)),
                body=_mul(
                    _index("formula_joint_scales_by_response", _var("i"), _var("row")),
                    _index("formula_joint_correlation", _var("i"), _var("j")),
                ),
            ),
        )
    )
    first = next(iter(responses.values()))
    weights = (
        first.names.get("formula_weights")
        if first.formula.modifiers.weights is not None
        else None
    )
    location = ("formula_joint_mu", "formula_joint_scale")
    observation = _draw(
        "formula_joint_response",
        elliptical.kernel,
        location if df is None else (df, *location),
        index=TypeName(name=row),
        mode="score",
        weights=weights,
        parameterization=elliptical.parameterization,
    )
    observation = observation.with_(
        options=(
            *observation.options,
            OptionEntry(key="over", value=OptionName(value=axis)),
        )
    )
    steps.append(observation)
    fixed_inputs = dict(fixed_inputs) | {
        "formula_joint_response": torch.stack(
            tuple(
                response_tensor(response.formula, response.lens._family)
                for response in responses.values()
            ),
            dim=-1,
        )
    }
    return (
        program.with_(draws=tuple(steps), return_vars=("formula_joint_response",)),
        targets,
        fixed_inputs,
    )


def _shared_groups(
    components: _Components, levels: Mapping[str, tuple[str, ...]]
) -> tuple[
    FormulaToQVRModule,
    Formula,
    list[Statement],
    list[ProgramStep],
    tuple[QiecValueParameter, ...],
    dict[str, torch.Tensor],
    dict[str, _GroupValues],
    list[Predictor],
    dict[str, dict[str, str]],
]:
    """Emit one existing Gaussian block for each cross-response shared ID.

    Parameters
    ----------
    components : Mapping[str, tuple[FormulaToQVRModule, Formula]]
        Response labels and their recoded lens/formula pairs.
    levels : Mapping[str, tuple[str, ...]]
        Joint levels of every group.

    Returns
    -------
    tuple
        The emitting lens, the grouping formula, object declarations, program
        steps, runtime parameters and their bound inputs, shared coefficients
        for each response, namespaced predictors, and their response scopes.
    """
    blocks: dict[tuple[str, str], list[tuple[str, str, str, Formula, RandomTerm]]] = {}
    for name, (lens, formula) in components.items():
        for parameter, prefix, predictor in lens._predictor_items(formula):
            for term in predictor.random_terms:
                if term.shared and term.correlation_id is not None:
                    blocks.setdefault((term.group, term.correlation_id), []).append(
                        (name, parameter, prefix, predictor, term)
                    )
    first_lens, first_formula = next(iter(components.values()))
    emitter = copy(first_lens)
    emitter._emitted_groups = []
    emitter._drawn_sites = frozenset()
    emitter._group_covariances = {
        group: covariance
        for lens, _ in components.values()
        for group, covariance in lens._group_covariances.items()
    }
    emitter._set_prior_configuration(None)
    counts = len(first_formula.response_values)
    grouping = first_formula.with_(
        group_levels=dict(levels),
        group_indices={group: (0,) * counts for group in levels},
    )
    statements: list[Statement] = []
    steps: list[ProgramStep] = []
    scopes: dict[str, dict[str, str]] = {}
    planned: dict[str, Predictor] = {}
    supplied: dict[str, _GroupValues] = {name: {} for name in components}
    for (group, identity), members in blocks.items():
        if len({item[0] for item in members}) < 2:
            continue
        actual: list[tuple[str, RandomTerm]] = []
        axes: list[tuple[str, int] | None] = []
        for name, parameter, prefix, predictor, term in members:
            lens, formula = components[name]
            actual_prefix = "response_" + _qvr_name(name) + "__" + prefix
            actual.append((actual_prefix, term))
            axis = _member_axis(name, lens, formula, prefix, term)
            axes.append(axis)
            if axis is not None:
                emitter._append_finite_object(statements, *axis)
            role = (
                "nlpar"
                if lens._predictor_graph is not None
                and parameter in lens._predictor_graph.leaves
                else "dpar"
            )
            scopes[actual_prefix] = {"response": name, role: parameter}
            planned.setdefault(
                actual_prefix,
                Predictor(
                    parameter=parameter,
                    role=role,
                    prefix=actual_prefix,
                    link=slot_link(lens._family, parameter),
                ),
            )
        law = plan_joint_block(
            group,
            "joint_shared_" + identity,
            [
                (actual_prefix, term, predictor)
                for (actual_prefix, term), (_, _, _, predictor, _) in zip(
                    actual, members, strict=True
                )
            ],
            axes,
            basis=emitter._level_basis(group, grouping.group_levels)
            if group in emitter._group_covariances
            else None,
            correlate=True,
            parameterization=emitter._reparameterize,
        )
        values = emit_group_block(emitter, law, statements, steps)
        for (name, _, prefix, predictor, term), axis in zip(members, axes, strict=True):
            actual_prefix = "response_" + _qvr_name(name) + "__" + prefix
            lens, formula = components[name]
            response = response_axis(
                lens._family,
                lens._component_labels(formula)
                if lens._family.response is not None
                else (),
            )
            planned[actual_prefix] = planned[actual_prefix].with_(
                terms=(
                    *planned[actual_prefix].terms,
                    group_term(
                        predictor,
                        actual_prefix,
                        term,
                        OutputAxis(
                            name=axis[0],
                            size=axis[1],
                            labels=response.labels if response is not None else (),
                        )
                        if axis is not None
                        else None,
                        {(actual_prefix, group, term.slope): law},
                    ),
                )
            )
            supplied[name][(prefix, group, term.slope)] = values[
                (actual_prefix, group, term.slope)
            ]
    # A grouping factor has one Student-t mixing law in the joint program, read
    # by the blocks each response emits for its unshared coefficients.
    for lens, formula in components.values():
        for _, _, predictor in lens._predictor_items(formula):
            for term in predictor.random_terms:
                option = predictor.grouping.option(term.group)
                if option is not None and option.options.student:
                    draw_level(
                        emitter,
                        level_law(
                            term.group,
                            len(levels[term.group]),
                            option.with_(by_levels=(), by_codes=(), prior_weights=()),
                            IdentityCovariance(),
                        ),
                        statements,
                        steps,
                    )
    # The responses bind the row data of multi-membership factors.
    inputs = bind_inputs(law_inputs(emitter.group_laws))
    parameters = tuple(
        parameter
        for parameter in declare_inputs(law_inputs(emitter.group_laws))
        if parameter.name in inputs
    )
    return (
        emitter,
        grouping,
        statements,
        steps,
        parameters,
        inputs,
        supplied,
        list(planned.values()),
        scopes,
    )


def emit_joint(
    components: Mapping[str, tuple[FormulaToQVRModule, Formula]],
    *,
    priors: PriorInput | None = None,
    supplied_inputs: Mapping[str, Mapping[str, LetExprNode]] | None = None,
    rescor: bool = False,
    preamble: tuple[ProgramStep, ...] = (),
    external_inputs: tuple[QiecValueParameter, ...] = (),
    external_objects: tuple[ObjectDecl, ...] = (),
    local_module: Callable[[str, Module], Module] | None = None,
) -> _JointEmission:
    """Compose existing response emissions, shared groups, and one prior plan.

    Parameters
    ----------
    components : Mapping
        Ordered response labels and their ordinary prepared lens/formula pairs.
    priors : PriorInput or None
        One catalogue of rules, including optional response selectors.
    supplied_inputs : Mapping or None
        Existing response input names and typed expressions over joint bindings.
        Replaced inputs become local deterministic bindings and are omitted from
        the external runtime telescope.
    rescor : bool
        Join scalar Gaussian or Student responses under one correlated
        residual law.
    preamble : tuple[ProgramStep, ...]
        Joint steps preceding every response step.
    external_inputs : tuple[QiecValueParameter, ...]
        Joint runtime parameters preceding every response parameter.
    external_objects : tuple[ObjectDecl, ...]
        Joint object declarations preceding every response object.
    local_module : Callable[[str, Module], Module] or None
        Rewrite applied to each response's local module before namespacing.

    Returns
    -------
    _JointEmission
        One QVR program and explicit response/latent coordinate metadata.

    Raises
    ------
    ValueError
        If no response is given, explicit inputs name an unknown
        response, a response lens carries its own prior rules, or the response
        graphs, residual rows, or prior selections are inconsistent.
    """
    if not components:
        raise ValueError("joint emission requires at least one response graph")
    components, levels = _normalized_components(components)
    if rescor:
        _validate_residual_rows(components)
    supplied_inputs = supplied_inputs or {}
    if set(supplied_inputs) - set(components):
        raise ValueError("supplied inputs name an unknown response")
    for lens, _ in components.values():
        if lens.prior_set.rules:
            raise ValueError(
                "joint priors must be resolved once on the joint catalogue"
            )
    shared_objects = {_qvr_name(group): _qvr_name(group) for group in levels}
    if rescor:
        shared_objects["Resp"] = "FormulaJointRow"
    (
        emitter,
        grouping,
        global_objects,
        global_steps,
        global_parameters,
        fixed_inputs,
        supplied,
        predictors,
        scopes,
    ) = _shared_groups(components, levels)
    global_steps = [*preamble, *global_steps]
    global_parameters = (*external_inputs, *global_parameters)
    global_objects = (*external_objects, *global_objects)
    global_names = set(_bound_names(tuple(global_steps))) | {
        p.name for p in global_parameters
    }
    responses = {
        name: _namespace(
            name,
            lens,
            formula,
            shared_objects=shared_objects,
            global_names=global_names,
            supplied_groups=supplied[name],
            supplied_sites=emitter._drawn_sites,
            supplied_inputs=supplied_inputs.get(name),
            local_module=local_module,
        )
        for name, (lens, formula) in components.items()
    }
    statements = list(global_objects)
    for group, names in levels.items():
        emitter._append_finite_object(statements, _qvr_name(group), len(names))
    emitter._append_finite_object(statements, "FormulaJointUnit", 1)
    initial_objects = tuple(statements)
    statements, steps, parameters, returned, targets, layouts = _response_parts(
        responses,
        initial_objects,
        global_steps,
        global_parameters,
        emitter.group_laws,
    )
    global_module = Module(
        statements=(
            *statements,
            ProgramDecl(
                name="model",
                params=None,
                domain=TypeName(name="FormulaJointUnit"),
                codomain=TypeName(name="FormulaJointUnit"),
                draws=tuple(global_steps),
                return_vars=(),
            ),
        )
    )
    global_targets = build_prior_targets(
        global_module,
        grouping,
        emitter._family,
        predictors,
        layouts,
        draw=emitter._parse_prior,
        predictor_scopes=scopes,
    )
    targets.extend(global_targets)
    program = ProgramDecl(
        name="model",
        params=None,
        domain=TypeName(name="FormulaJointUnit"),
        codomain=TypeName(name="FormulaJointUnit"),
        value_parameters=tuple(parameters),
        draws=tuple(steps),
        return_vars=tuple(returned),
    )
    if rescor:
        program, targets, fixed_inputs = _correlated_residual(
            program, responses, emitter, statements, targets, fixed_inputs
        )
    module = Module(
        statements=(*statements, program, ExportDecl(expr=ExprIdent(name="model")))
    )
    plan = normalize_priors(priors)
    if plan.rules:
        selected = resolve_priors(plan, targets)
        ordinary = {
            site: rule
            for site, rule in selected.items()
            if isinstance(rule.distribution, str)
        }
        by_site = {target.site: target for target in targets}
        # Re-enter the ordinary emitter after global selector resolution so its
        # existing dynamic support restrictions remain conditional and normalized.
        # This is source emission only; the native program is compiled once.
        for name, response in tuple(responses.items()):
            local_rules = []
            reverse = {
                global_name: local for local, global_name in response.names.items()
            }
            for site, rule in ordinary.items():
                if by_site[site].response == name and site in reverse:
                    local_rules.append(
                        PriorSpec(
                            distribution=rule.distribution,
                            class_=None,
                            site=reverse[site],
                            lower=rule.lower,
                            upper=rule.upper,
                        )
                    )
            if local_rules:
                lens = copy(response.lens)
                lens._set_prior_configuration(PriorSet(rules=tuple(local_rules)))
                responses[name] = _namespace(
                    name,
                    lens,
                    response.formula,
                    shared_objects=shared_objects,
                    global_names=global_names,
                    supplied_groups=supplied[name],
                    supplied_sites=emitter._drawn_sites,
                    supplied_inputs=supplied_inputs.get(name),
                    local_module=local_module,
                )
        statements, steps, parameters, returned, _, layouts = _response_parts(
            responses,
            initial_objects,
            global_steps,
            global_parameters,
            emitter.group_laws,
        )
        program = program.with_(
            draws=tuple(steps),
            value_parameters=tuple(parameters),
            return_vars=tuple(returned),
        )
        if rescor:
            program, _, fixed_inputs = _correlated_residual(
                program, responses, emitter, statements, targets, fixed_inputs
            )
        replacements = {
            site: ordinary_prior_step(by_site[site], rule, draw=emitter._parse_prior)
            for site, rule in ordinary.items()
            if by_site[site] in global_targets or by_site[site].response is None
        }
        program = program.with_(
            draws=tuple(
                step.with_(
                    morphism=replacements[step.vars[0]].morphism,
                    args=replacements[step.vars[0]].args,
                    static_args=replacements[step.vars[0]].static_args,
                )
                if isinstance(step, SampleStep) and step.vars[0] in replacements
                else step
                for step in program.draws
            )
        )
        module = module.with_(
            statements=(*statements, program, ExportDecl(expr=ExprIdent(name="model")))
        )
        special = {
            site: rule
            for site, rule in selected.items()
            if not isinstance(rule.distribution, str)
        }
        if special:
            row_counts = {
                target.site: len(responses[target.response].formula.response_values)
                for target in targets
                if target.response is not None
            }
            sigmas = {}
            for target in targets:
                if target.response is None:
                    continue
                response = responses[target.response]
                residual = response.lens._family.residual_scale
                if (
                    residual is not None
                    and residual.name in response.names
                    and residual.name not in response.lens._distributional
                ):
                    sigmas[target.site] = response.names[residual.name]
            module = apply_prior_plan(
                module,
                targets,
                special,
                draw=emitter._parse_prior,
                training_rows=row_counts,
                sigma=sigmas,
            )
        else:
            module = order_prior_dependencies(module)
    module = order_prior_dependencies(module)
    catalog = FamilyCatalog()
    for response in responses.values():
        if response.families is not None:
            catalog = catalog.with_catalog(response.families)
    # Improper flat priors are compilation-local native laws.
    if improper_sites(module):
        catalog = catalog.with_catalog(FLAT_CATALOG)
    level_sites = {site.site: site for law in layouts for site in law.level_sites}
    for response in responses.values():
        for site in response.lens.level_sites:
            name = response.names.get(site.site, site.site)
            level_sites.setdefault(name, site.with_(site=name))
    return _JointEmission(
        module,
        responses,
        tuple(layouts),
        tuple(level_sites.values()),
        tuple(targets),
        fixed_inputs,
        catalog,
    )


__all__: list[str] = []
