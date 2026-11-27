"""Compose response-indexed uses of the existing formula-to-QVR emitter."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass

import didactic.api as dx
import torch
from quivers.family_catalog import FamilyCatalog

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    QiecValueParameter,
    SampleStep,
    ScoreStep,
    Statement,
    LetStep,
    ObserveStep,
    TypeName,
)
from quivers.dsl.emit import draw_call_to_source
from quivers.dsl.program_templates import (
    _bound_names,
)
from quivers.formulas._blocks import Namespace, emit_block, emit_model, place_block
from quivers.formulas._ast import (
    _add,
    _call,
    _div,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
    row_zero,
    stack_rows,
)
from quivers.formulas._group_law import (
    GroupBlockLaw,
    IdentityCovariance,
    KnownCovariance,
    LevelSite,
    law_inputs,
    renamed_law,
)
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._kernel_law import elliptical_family
from quivers.formulas._group_emission import draw_level, emit_group_block
from quivers.formulas._group_plan import (
    block_covariance,
    check_names,
    level_law,
    plan_joint_block,
)
from quivers.formulas._predictor import (
    Predictor,
    group_term,
    slot_link,
)
from quivers.formulas._row_source import RowSource
from quivers.formulas._term import OutputAxis, bind_inputs, declare_inputs
from quivers.formulas._brms_priors import FLAT_CATALOG, improper_sites
from quivers.formulas._prior_emission import (
    apply_prior_plan,
    separate_components,
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
    component_rules,
    resolve_priors,
)
from quivers.dsl.step_resolution import _step_value_names
from quivers.formulas._response import response_tensor
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas._observation import Observation, observation
from quivers.formulas._joint_residual import (
    RecursiveResidualFactory,
    recursive_residual_record,
)
from quivers.formulas.formula import Formula, FormulaData, RandomTerm, _qvr_name

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
    joint : JointElliptical or None
        The law of responses with correlated residuals; None when each
        response is observed on its own.
    """

    module: Module
    responses: Mapping[str, _ResponseEmission]
    group_laws: tuple[GroupBlockLaw, ...]
    level_sites: tuple[LevelSite, ...]
    prior_targets: tuple[PriorTarget, ...]
    fixed_inputs: Mapping[str, torch.Tensor]
    families: FamilyCatalog
    joint: JointElliptical | None = None

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Return the coordinate layout of each group block.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Raw sites and coefficient slices derived from the block laws.
        """
        return tuple(law.layout for law in self.group_laws)


type _ByLevels = dict[tuple[str, str | None], dict[str, str]]
type _PriorWeights = dict[tuple[str, str | None], dict[str, float]]
_LevelFactors = tuple[_ByLevels, _PriorWeights]


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
    tuple
        The declared ``by`` level and the declared prior weight of each
        labeled level, by grouping factor and declaring column.

    Raises
    ------
    ValueError
        If two declarations give one level of a grouping factor different
        ``by`` levels or different prior weights.
    """
    by: _ByLevels = {}
    weights: _PriorWeights = {}
    for lens, formula in components.values():
        for _, _, predictor in lens._predictor_items(formula):
            for option in predictor.grouping.options:
                group = option.group
                labels = predictor.group_levels[group]
                if option.by_levels:
                    partition = by.setdefault((group, option.options.by), {})
                    for label, code in zip(labels, option.by_codes, strict=True):
                        value = option.by_levels[code]
                        if partition.setdefault(label, value) != value:
                            raise ValueError(
                                f"gr({group}, by = {option.options.by}): a by "
                                f"partition assigns each level of {group!r} one by "
                                f"level, and the responses give level {label!r} both "
                                f"{partition[label]!r} and {value!r}"
                            )
                if option.prior_weights:
                    declared = weights.setdefault(
                        (group, option.options.prior_weight), {}
                    )
                    for label, weight in zip(labels, option.prior_weights, strict=True):
                        known = declared.setdefault(label, weight)
                        if known != weight:
                            raise ValueError(
                                f"gr({group}, pw = {option.options.prior_weight}): a "
                                f"prior weight belongs to a level of {group!r}, and "
                                f"the responses give level {label!r} both {known!r} "
                                f"and {weight!r}"
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
    factors : tuple
        The ``by`` level and the prior weight declared for each labeled
        level, by grouping factor and declaring column.

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
            declared = factors[0][(group, option.options.by)]
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
            known = factors[1][(group, option.options.prior_weight)]
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
    rows: RowSource | None = None,
    resolved_priors: Mapping[str, PriorSpec] | None = None,
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
    rows : RowSource or None
        Rows the response covers; None for its training rows.
    resolved_priors : Mapping[str, PriorSpec] or None
        Joint prior selections keyed by their local sites in this response.

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
    module, complement = emit_block(
        lens,
        formula,
        supplied_groups=supplied_groups,
        supplied_sites=supplied_sites,
        rows=rows,
        resolved_priors=resolved_priors,
    )
    return _place_response(
        name,
        lens,
        formula,
        module,
        complement,
        Namespace(
            prefix="response_" + _qvr_name(name) + "__",
            shared=frozenset(global_names),
            shared_objects=dict(shared_objects),
        ),
        supplied_inputs=supplied_inputs,
    )


def _place_response(
    name: str,
    lens: FormulaToQVRModule,
    formula: Formula,
    module: Module,
    complement: FormulaData,
    namespace: Namespace,
    *,
    supplied_inputs: Mapping[str, LetExprNode] | None = None,
) -> _ResponseEmission:
    """Place an emitted response and substitute its symbolic inputs."""
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    explicit_inputs = supplied_inputs or {}
    supplied_inputs = lens.symbolic_column_expressions(formula)
    symbolic = set(supplied_inputs)
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
    unknown = set(explicit_inputs) - known_inputs - symbolic
    if unknown:
        raise ValueError(f"unknown supplied response inputs: {sorted(unknown)!r}")
    # A group-level coefficient the request leaves out reads no column.
    supplied_inputs = {
        value: expression
        for value, expression in supplied_inputs.items()
        if value in known_inputs
    }
    renamed = place_block(
        lens,
        module,
        namespace,
        inputs=supplied_inputs,
        omitted=supplied_inputs.keys() | namespace.shared,
    )
    names = renamed.names
    families = lens.response_laws
    if renamed.families is not None:
        families = (
            renamed.families
            if families is None
            else families.with_catalog(renamed.families)
        )
    statements = (
        *renamed.objects,
        renamed.program.with_(
            draws=(
                *(
                    LetStep(name=names[value], value=expression)
                    for value, expression in supplied_inputs.items()
                ),
                *renamed.program.draws,
            )
        ),
    )
    objects = renamed.object_names
    return _ResponseEmission(
        name,
        namespace.prefix,
        lens,
        formula,
        Module(statements=tuple(statements)),
        complement,
        names,
        objects,
        families,
    )


def place_single_response(
    name: str,
    lens: FormulaToQVRModule,
    formula: Formula,
    module: Module,
    complement: FormulaData,
) -> _JointEmission:
    """Provide an identity-namespaced response to the missing-data rewrite."""
    response = _place_response(
        name, lens, formula, module, complement, Namespace(prefix="")
    )
    program = next(
        item for item in response.module.statements if isinstance(item, ProgramDecl)
    )
    return _JointEmission(
        module=module.with_(
            statements=tuple(
                program if isinstance(item, ProgramDecl) else item
                for item in module.statements
            )
        ),
        responses={name: response},
        group_laws=complement.group_laws,
        level_sites=complement.level_sites,
        prior_targets=complement.prior_targets,
        fixed_inputs={},
        families=response.families
        if response.families is not None
        else FamilyCatalog(),
    )


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
) -> tuple[Module, list[PriorTarget], list[GroupBlockLaw]]:
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
    tuple[Module, list[PriorTarget], list[GroupBlockLaw]]
        The assembled joint module, its prior targets and its group laws.

    Raises
    ------
    ValueError
        If two responses declare one nominal group axis differently, or two
        group blocks would own the same name.
    """
    module = emit_model(
        initial_objects,
        global_steps,
        global_parameters,
        (),
        axis="FormulaJointUnit",
        blocks=(response.module for response in responses.values()),
    )
    targets: list[PriorTarget] = []
    laws = list(global_laws)
    for name, response in responses.items():
        targets.extend(
            target.with_(
                site=response.names[target.site],
                axes=tuple(response.objects.get(axis, axis) for axis in target.axes),
                whole=None
                if target.whole is None
                else response.names.get(target.whole, target.whole),
                partition_axis=None
                if target.partition_axis is None
                else response.objects.get(target.partition_axis, target.partition_axis),
                response=name,
            )
            for target in response.lens.prior_targets
        )
        laws.extend(
            _law(law, response.names, response.objects, response.prefix)
            for law in response.lens.group_laws
        )
    check_names(laws)
    return module, targets, laws


def _member_axis(
    name: str,
    lens: FormulaToQVRModule,
    formula: Formula,
    prefix: str,
    term: RandomTerm,
) -> OutputAxis | None:
    """Locate the coordinate axis of one member of a shared block.

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
    OutputAxis or None
        The namespaced coordinate axis, or None for a scalar member.
    """
    axis = lens._output_axes(formula).get(prefix)
    if axis is None and term.category_specific:
        axis = lens._transition_axis(formula)
    if axis is None:
        return None
    return axis.with_(name="response_" + _qvr_name(name) + "__" + axis.name)


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
    laws = {
        elliptical_family(lens._family).elliptical for lens, _ in components.values()
    }
    # A conditional ARMA recursion moves each row's location and keeps the law
    # of the row's residual vector (RecursiveResiduals). A covariance across
    # rows beside one across responses has no single composition: a Kronecker
    # product, a sum and a nesting all fit the syntax, and brms refuses it.
    for lens, _ in components.values():
        residual = lens._residual
        if residual is not None and not residual.factorized:
            raise ValueError(
                "Explicit covariance terms cannot be modeled when 'rescor' is "
                "estimated at the same time."
            )
    elliptical = next(iter(laws))
    if len(laws) != 1 or elliptical is None:
        names = sorted({lens._family.name for lens, _ in components.values()})
        raise ValueError(
            "rescor requires homogeneous Gaussian or Student responses, "
            "families declaring one elliptical residual law "
            f"(Family.elliptical); got families {names!r}"
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


@dataclass(frozen=True, slots=True)
class _SharedDegrees:
    """Bindings for the common degrees of freedom of Student responses.

    Parameters
    ----------
    name : str
        Binding passed to the vector observation.
    steps : tuple[ProgramStep, ...]
        Bindings added before the vector observation.
    replaced : frozenset[str]
        Scalar degrees-of-freedom draws replaced by the shared draw.
    target : PriorTarget or None
        Formula prior target carried to the shared draw, if there is one.
    """

    name: str
    steps: tuple[ProgramStep, ...] = ()
    replaced: frozenset[str] = frozenset()
    target: PriorTarget | None = None


def _shared_student_degrees(
    df: str,
    arguments: tuple[LetExprNode, ...],
    steps: tuple[ProgramStep, ...],
    observations: set[str],
    targets: Sequence[PriorTarget],
) -> _SharedDegrees:
    """Unify the scalar degrees of freedom of one Student residual event.

    Parameters
    ----------
    df : str
        Name of the shared binding.
    arguments : tuple[LetExprNode, ...]
        Degrees of freedom passed to each scalar observation, in response order.
    steps : tuple[ProgramStep, ...]
        Existing steps of the composed body.
    observations : set[str]
        Scalar observations replaced by the joint event.
    targets : Sequence[PriorTarget]
        Existing prior targets of the composed body.

    Returns
    -------
    _SharedDegrees
        Common constant, existing binding, or common draw and its prior target.

    Raises
    ------
    ValueError
        If the responses cannot share one scalar degrees of freedom argument.
    """
    error = f"correlated Student responses require one common fixed or sampled {df}"
    lets = {step.name: step.value for step in steps if isinstance(step, LetStep)}

    def resolve(expression: LetExprNode) -> LetExprNode:
        """Follow aliases to a literal or one program binding."""
        seen: set[str] = set()
        while (
            isinstance(expression, LetExprVar)
            and expression.name in lets
            and expression.name not in seen
        ):
            seen.add(expression.name)
            expression = lets[expression.name]
        return expression

    resolved = tuple(resolve(argument) for argument in arguments)
    constants = {
        float(value.value) for value in resolved if isinstance(value, LetExprLiteral)
    }
    if len(constants) == 1 and all(
        isinstance(value, LetExprLiteral) for value in resolved
    ):
        (value,) = constants
        return _SharedDegrees(df, steps=(_let(df, LetExprLiteral(value=value)),))
    bindings = tuple(value.name for value in resolved if isinstance(value, LetExprVar))
    if len(bindings) != len(arguments):
        raise ValueError(error)
    sites = frozenset(bindings)
    if len(sites) == 1:
        return _SharedDegrees(bindings[0])
    draws = {
        step.vars[0]: step
        for step in steps
        if isinstance(step, SampleStep) and step.vars[0] in sites
    }
    if draws.keys() != sites or any(step.index is not None for step in draws.values()):
        raise ValueError(error)
    prior_targets = {target.site: target for target in targets if target.site in sites}
    if (
        sites - prior_targets.keys()
        and len(
            {
                draw_call_to_source(
                    step.morphism, step.args, static_args=step.static_args
                )
                for step in draws.values()
            }
        )
        != 1
    ):
        raise ValueError(error + " with the same authored prior")
    if any(
        _step_value_names(step) & sites
        for step in steps
        if not isinstance(step, ObserveStep) or step.vars[0] not in observations
    ):
        raise ValueError(error + " used only by the correlated observations")
    original = next(
        (prior_targets[name] for name in bindings if name in prior_targets), None
    )
    return _SharedDegrees(
        df,
        steps=(next(iter(draws.values())).with_(vars=(df,)),),
        replaced=sites,
        target=None if original is None else original.with_(site=df, response=None),
    )


def _correlated_residual(
    program: ProgramDecl,
    responses: Mapping[str, _ResponseEmission],
    emitter: FormulaToQVRModule,
    statements: list[Statement],
    targets: Sequence[PriorTarget],
    fixed_inputs: Mapping[str, torch.Tensor],
) -> tuple[
    ProgramDecl,
    list[PriorTarget],
    dict[str, torch.Tensor],
    FamilyCatalog | None,
    JointElliptical,
]:
    """Join existing scalar location/scale bindings under one vector residual law.

    Responses with a conditional ARMA recursion are observed through
    `RecursiveResiduals`, whose locations follow each response's own past.

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
    tuple
        The program with one multivariate observation, its prior targets
        with the residual correlation and shared ``nu``, the inputs with the
        stacked joint response, the catalog of a recursive joint law or
        None, and the joint law.
    """
    names = tuple(responses)
    width = len(names)
    axis, row = "FormulaJointResponse", "FormulaJointRow"
    emitter._append_finite_object(statements, axis, width)
    removed = {
        response.names[_qvr_name(response.formula.response_name)]
        for response in responses.values()
    }
    # Each response's own observation, in the joint program's names.
    observations = {
        name: response_observation(response) for name, response in responses.items()
    }
    arguments_by_response = {
        name: {
            parameter: observed.argument(parameter)
            for parameter, _ in observed.arguments
        }
        for name, observed in observations.items()
    }
    declarations = {
        name: elliptical_family(response.lens._family)
        for name, response in responses.items()
    }
    elliptical = declarations[names[0]].elliptical
    assert elliptical is not None
    direct = {
        step.vars[0]: step
        for step in program.draws
        if isinstance(step, ObserveStep) and step.vars[0] in removed
    }
    if direct.keys() != removed:
        raise ValueError(
            "rescor requires response observations in the same program scope"
        )
    for observed in observations.values():
        allowed = {"weights"} if observed.weights is not None else set()
        if {option.key for option in direct[observed.site].options} - allowed:
            raise ValueError(
                "rescor requires one joint recording law for the response observations"
            )
    df = elliptical.df
    degrees = (
        None
        if df is None
        else _shared_student_degrees(
            df,
            tuple(
                arguments_by_response[name][
                    "nu"
                    if responses[name].lens._residual is not None
                    else declarations[name].native_parameter(df)
                ]
                for name in names
            ),
            program.draws,
            removed,
            targets,
        )
    )
    df_binding = None if degrees is None else degrees.name
    nu_sites = frozenset() if degrees is None else degrees.replaced
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
    if degrees is not None:
        steps.extend(degrees.steps)
        if degrees.target is not None:
            targets.append(degrees.target)
    rows, zero = ("row", row), "formula_joint_zeros"

    def stacked(quantity: str) -> Iterator[tuple[int, LetExprNode]]:
        # Scalar-to-row broadcasting uses the ordinary typed binary operation;
        # nominal row ownership never depends on coincident axis cardinalities.
        for index, name in enumerate(responses):
            argument = declarations[name].native_parameter(quantity)
            yield index, _add(arguments_by_response[name][argument], _var(zero))

    component = ("component", axis)
    means, locations = stack_rows(
        "formula_joint_mu",
        stacked("mu"),
        component,
        stem="formula_joint_means_by_response",
        row=rows,
    )
    scales = _let(
        "formula_joint_scales_by_response",
        _factor((component,), cases=tuple(stacked(elliptical.scale))),
    )
    steps.extend((row_zero(zero, rows), means, scales, locations))
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
    weights = next(iter(observations.values())).weights
    location = ("formula_joint_mu", "formula_joint_scale")
    recursions = recursion_kernel(
        responses, arguments_by_response, student=df_binding is not None
    )
    joint = (
        observation(
            "formula_joint_response",
            elliptical.kernel,
            location if df_binding is None else (df_binding, *location),
            plate=row,
            event=axis,
            parameterization=elliptical.parameterization,
        )
        if recursions is None
        else observation(
            "formula_joint_response",
            recursions[0].definitions[0].name,
            (
                *((df_binding,) if df_binding is not None else ()),
                *location,
                *recursions[1],
            ),
            plate=row,
            event=axis,
            catalog=recursions[0],
        )
    ).with_(weights=weights)
    completed = (
        None
        if recursions is None
        else observation(
            "formula_joint_response",
            recursions[0].definitions[1].name,
            (
                *((df_binding,) if df_binding is not None else ()),
                *location,
                *recursions[1],
                COMPLETION,
            ),
            plate=row,
            event=axis,
            catalog=recursions[0],
        ).with_(weights=weights)
    )
    steps.append(joint.step())
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
        None if recursions is None else recursions[0],
        JointElliptical(responses=names, observation=joint, completed=completed),
    )


COMPLETION = "formula_joint_completion"
"""Binding of the completed response vectors of a joint law."""


_LOG_TWO_PI = 1.8378770664093453
_LOG_PI = 1.1447298858494002


class JointElliptical(ValueModel):
    """Residuals correlated across the responses of each row.

    The vector of one row's residuals has a multivariate Normal or Student
    law; with conditional recursions the locations follow the responses'
    own past and the law of each row's innovation vector is unchanged. One
    value states the observation of the recorded responses and the score of
    completed responses, so both read the same stacks of locations, scales
    and recursion coefficients.

    Parameters
    ----------
    responses : tuple[str, ...]
        Responses in coordinate order.
    observation : Observation
        Observation of the recorded response vectors.
    completed : Observation or None
        Observation of the recorded response vectors by the recursive law
        over completed responses, which it takes as its ``completion``
        argument; None for a law without recursions.
    """

    responses: tuple[str, ...]
    observation: Observation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    completed: Observation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def observes_recorded(self) -> bool:
        """Whether completed responses are scored by observing the recorded ones.

        Returns
        -------
        bool
            True for the recursive law, whose observation takes the completed
            responses as an argument; False when the score of the completed
            vectors replaces the observation.
        """
        return self.completed is not None

    def score(self, values: tuple[str, ...]) -> tuple[ProgramStep, ...]:
        """Score completed response vectors under the joint law.

        Without recursions, the multivariate Normal or Student density of
        each row is written by forward substitution through the row's
        lower-triangular scale factor, so the factor is never inverted or
        multiplied out; cost O(rows * responses^2). With recursions the
        recorded vectors are observed by the recursive law over the
        completed responses: the map from the completed responses to the
        innovations is unit lower triangular in each series' order, so the
        joint density of the recorded and missing responses is the product
        of each row's innovation density, which the law evaluates.

        Parameters
        ----------
        values : tuple[str, ...]
            Row vector of each response in coordinate order, completed where
            a response is missing.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps of the score.
        """
        observation = self.observation
        if self.completed is not None:
            row, axis = observation.plate, observation.event
            assert axis is not None
            return (
                *stack_rows(
                    COMPLETION,
                    enumerate(_var(value) for value in values),
                    ("component", axis),
                    stem=f"{COMPLETION}_by_response",
                    row=("row", row),
                ),
                self.completed.step(),
            )
        arguments = {
            parameter: observation.argument(parameter)
            for parameter in observation.parameters
        }
        row = observation.plate
        width = len(values)
        at = _var("row")

        def entry(parameter: str, *components: int) -> LetExprNode:
            """Select one row's coordinate of a joint argument.

            Parameters
            ----------
            parameter : str
                ``loc`` or ``scale_tril``.
            *components : int
                Response coordinates.

            Returns
            -------
            LetExprNode
                The indexed expression at the scoped row.
            """
            return LetExprIndex(
                array=arguments[parameter],
                indices=(
                    at,
                    *(
                        LetExprLiteral(value=index, integral=True)
                        for index in components
                    ),
                ),
            )

        steps: list[ProgramStep] = []
        innovations = tuple(
            f"formula_joint_innovation_{index}" for index in range(width)
        )
        for index, value in enumerate(values):
            residual = _sub(_index(value, at), entry("loc", index))
            for earlier in range(index):
                residual = _sub(
                    residual,
                    _mul(
                        entry("scale_tril", index, earlier),
                        _index(innovations[earlier], at),
                    ),
                )
            steps.append(
                _let(
                    innovations[index],
                    _factor(
                        (("row", row),),
                        body=_div(residual, entry("scale_tril", index, index)),
                    ),
                )
            )
        log_scale = _add(
            *(_call("log", entry("scale_tril", index, index)) for index in range(width))
        )
        squares = _add(
            *(
                _mul(_index(innovation, at), _index(innovation, at))
                for innovation in innovations
            )
        )
        if observation.kernel == "MultivariateNormal":
            density = _sub(
                _sub(LetExprLiteral(value=-0.5 * width * _LOG_TWO_PI), log_scale),
                _mul(LetExprLiteral(value=0.5), squares),
            )
        else:
            nu = arguments["df"]
            half = _mul(
                LetExprLiteral(value=0.5), _add(nu, LetExprLiteral(value=float(width)))
            )
            steps.append(
                _let(
                    "formula_joint_normalizer",
                    _sub(
                        _sub(
                            _call("lgamma", half),
                            _call("lgamma", _mul(LetExprLiteral(value=0.5), nu)),
                        ),
                        _mul(
                            LetExprLiteral(value=0.5 * width),
                            _add(_call("log", nu), LetExprLiteral(value=_LOG_PI)),
                        ),
                    ),
                )
            )
            density = _sub(
                _sub(_var("formula_joint_normalizer"), log_scale),
                _mul(half, _call("log1p", _div(squares, nu))),
            )
        weights = observation.weights
        if weights is not None:
            density = _mul(_index(weights, at), density)
        steps.append(
            ScoreStep(
                name="formula_joint_score",
                value=_call("sum", _factor((("row", row),), body=density)),
            )
        )
        return tuple(steps)


def response_observation(response: _ResponseEmission) -> Observation:
    """Return the observation of one response in the joint program's names.

    Parameters
    ----------
    response : _ResponseEmission
        Namespaced response.

    Returns
    -------
    Observation
        The response's observation with namespaced bindings.

    Raises
    ------
    ValueError
        If the response's lens has emitted no observation.
    """
    observed = response.lens._observation
    if observed is None:
        raise ValueError(
            f"response {response.formula.response_name!r} has no observation"
        )
    return observed.renamed(response.names, objects=response.objects)


def recursion_kernel(
    responses: Mapping[str, _ResponseEmission],
    arguments: Mapping[str, Mapping[str, LetExprNode]],
    *,
    student: bool,
) -> tuple[FamilyCatalog, tuple[str, ...]] | None:
    """Declare the joint residual law of responses with conditional recursions.

    Parameters
    ----------
    responses : Mapping[str, _ResponseEmission]
        Namespaced responses in coordinate order.
    arguments : Mapping[str, Mapping[str, LetExprNode]]
        Arguments of each response's own observation, by kernel parameter.
    student : bool
        Student vectors with shared degrees of freedom.

    Returns
    -------
    tuple[FamilyCatalog, tuple[str, ...]] or None
        The catalog registering the law, first, and the law over completed
        responses, and the bindings of the responses' structural parameters
        in signature order; None when no response has a residual law.
    """
    laws = {name: response.lens._residual for name, response in responses.items()}
    if all(law is None for law in laws.values()):
        return None
    names: list[str] = []
    bindings: list[str] = []
    coefficients: list[tuple[tuple[str, str], ...]] = []
    for label, law in laws.items():
        pairs = []
        for coefficient in () if law is None else law.coefficients:
            value = arguments[label][coefficient]
            if not isinstance(value, LetExprVar):
                raise TypeError("a structural parameter is passed by its binding")
            kernel = f"{coefficient}_{_qvr_name(label)}"
            names.append(kernel)
            bindings.append(value.name)
            pairs.append((kernel, coefficient))
        coefficients.append(tuple(pairs))
    name = "JointResidualStudentT" if student else "JointResidualNormal"
    factory = RecursiveResidualFactory(
        layouts=tuple(None if law is None else law.layout for law in laws.values()),
        coefficients=tuple(coefficients),
        student=student,
    )
    catalog = FamilyCatalog()
    for record in (
        recursive_residual_record(name, names, student=student),
        recursive_residual_record(
            f"{name}Completed", names, student=student, completion=True
        ),
    ):
        catalog = catalog.with_family(record, factory)
    return catalog, tuple(bindings)


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
        axes: list[OutputAxis | None] = []
        for name, parameter, prefix, predictor, term in members:
            lens, formula = components[name]
            actual_prefix = "response_" + _qvr_name(name) + "__" + prefix
            actual.append((actual_prefix, term))
            axis = _member_axis(name, lens, formula, prefix, term)
            axes.append(axis)
            if axis is not None:
                emitter._append_finite_object(statements, axis.name, axis.size)
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
        matrix_name, matrix = block_covariance(
            group, [term for _, term in actual], emitter._group_covariances
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
            [None if axis is None else (axis.name, axis.size) for axis in axes],
            covariance=IdentityCovariance()
            if matrix is None
            else KnownCovariance(
                basis=matrix.basis(grouping.group_levels[group]), name=matrix_name
            ),
            correlate=True,
            components=1,
            vectors={
                actual_prefix
                for (actual_prefix, _), axis in zip(actual, axes, strict=True)
                if axis is not None
            },
            parameterization=emitter._reparameterize,
        )
        values = emit_group_block(emitter, law, statements, steps)
        for (name, _, prefix, predictor, term), axis in zip(members, axes, strict=True):
            actual_prefix = "response_" + _qvr_name(name) + "__" + prefix
            planned[actual_prefix] = planned[actual_prefix].with_(
                terms=(
                    *planned[actual_prefix].terms,
                    group_term(
                        predictor,
                        actual_prefix,
                        term,
                        axis,
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
                option = predictor.grouping.option(term.options)
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
    inputs = bind_inputs(law_inputs(emitter.group_laws))
    parameters = declare_inputs(law_inputs(emitter.group_laws))
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
    rows: Mapping[str, RowSource] | None = None,
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
    rows : Mapping[str, RowSource] or None
        Rows each response covers; None for the training rows.

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
            rows=(rows or {}).get(name),
        )
        for name, (lens, formula) in components.items()
    }
    statements = list(global_objects)
    for group, names in levels.items():
        emitter._append_finite_object(statements, _qvr_name(group), len(names))
    emitter._append_finite_object(statements, "FormulaJointUnit", 1)
    initial_objects = tuple(statements)
    plan = normalize_priors(priors)

    def shared_targets(
        separated: Sequence[PriorTarget],
    ) -> tuple[list[ProgramStep], tuple[PriorTarget, ...]]:
        """Draw the shared group blocks, with some sites by component."""
        module = separate_components(
            Module(
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
            ),
            separated,
            {},
        )
        targets = build_prior_targets(
            module,
            grouping,
            emitter._family,
            predictors,
            layouts,
            draw=emitter._parse_prior,
            predictor_scopes=scopes,
            separated=frozenset(target.site for target in separated),
        )
        program = next(
            item for item in module.statements if isinstance(item, ProgramDecl)
        )
        return list(program.draws), targets

    def parts() -> tuple[Module, list[PriorTarget], list[GroupBlockLaw]]:
        """Join the responses' programs."""
        return _response_parts(
            responses,
            initial_objects,
            global_steps,
            global_parameters,
            emitter.group_laws,
        )

    module, targets, layouts = parts()
    statements = list(module.statements[:-2])
    _, global_targets = shared_targets(())
    separated = component_rules(plan, [*targets, *global_targets])
    if separated:
        sites = {target.site for target in separated}
        for name, response in tuple(responses.items()):
            local = frozenset(
                local for local, site in response.names.items() if site in sites
            )
            if local:
                lens = copy(response.lens)
                lens._component_sites = lens._component_sites | local
                responses[name] = _namespace(
                    name,
                    lens,
                    response.formula,
                    shared_objects=shared_objects,
                    global_names=global_names,
                    supplied_groups=supplied[name],
                    supplied_sites=emitter._drawn_sites,
                    supplied_inputs=supplied_inputs.get(name),
                    rows=(rows or {}).get(name),
                )
        global_steps, global_targets = shared_targets(
            [target for target in separated if target in global_targets]
        )
        module, targets, layouts = parts()
        statements = list(module.statements[:-2])
    targets.extend(global_targets)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    joint_laws = None
    joint: JointElliptical | None = None
    if rescor:
        program, targets, fixed_inputs, joint_laws, joint = _correlated_residual(
            program, responses, emitter, statements, targets, fixed_inputs
        )
    shared_by_response: dict[str | None, list[PriorTarget]] = {}
    for target in global_targets:
        shared_by_response.setdefault(target.response, []).append(target)
    selected = {}
    for name, shared in shared_by_response.items():
        lens, formula = (emitter, grouping) if name is None else components[name]
        selected.update(lens._brms_defaults(formula, tuple(shared)))
    if plan.rules:
        selected.update(resolve_priors(plan, targets))
    ordinary = {
        site: rule
        for site, rule in selected.items()
        if isinstance(rule.distribution, str)
    }
    by_site = {target.site: target for target in targets}
    shared_sites = {target.site for target in global_targets}
    responses_changed = False
    if plan.rules:
        # Re-enter the ordinary emitter after global selector resolution so its
        # existing dynamic support restrictions remain conditional and normalized.
        # This is source emission only; the native program is compiled once.
        for name, response in tuple(responses.items()):
            local_rules = {}
            reverse = {
                global_name: local for local, global_name in response.names.items()
            }
            for site, rule in ordinary.items():
                if (
                    site not in shared_sites
                    and by_site[site].response == name
                    and site in reverse
                ):
                    local = reverse[site]
                    local_rules[local] = PriorSpec(
                        distribution=rule.distribution,
                        class_=None,
                        site=local,
                        lower=rule.lower,
                        upper=rule.upper,
                    )
            if local_rules:
                lens = copy(response.lens)
                lens._set_prior_configuration(
                    PriorSet(rules=tuple(local_rules.values()))
                )
                responses[name] = _namespace(
                    name,
                    lens,
                    response.formula,
                    shared_objects=shared_objects,
                    global_names=global_names,
                    supplied_groups=supplied[name],
                    supplied_sites=emitter._drawn_sites,
                    supplied_inputs=supplied_inputs.get(name),
                    rows=(rows or {}).get(name),
                    resolved_priors=local_rules,
                )
                responses_changed = True
    if responses_changed:
        module, _, layouts = parts()
        statements = list(module.statements[:-2])
        program = next(
            item for item in module.statements if isinstance(item, ProgramDecl)
        )
        if rescor:
            program, _, fixed_inputs, joint_laws, joint = _correlated_residual(
                program, responses, emitter, statements, targets, fixed_inputs
            )
    replacements = {
        site: ordinary_prior_step(by_site[site], rule, draw=emitter._parse_prior)
        for site, rule in ordinary.items()
        if site in shared_sites or by_site[site].response is None
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
    module = module.with_(statements=(*statements, program, module.statements[-1]))
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
    module = order_prior_dependencies(module)
    catalog = FamilyCatalog() if joint_laws is None else joint_laws
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
        joint,
    )


__all__: list[str] = []
