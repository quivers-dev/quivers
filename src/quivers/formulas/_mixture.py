"""Finite mixtures of response families: the brms ``mixture()`` family.

A finite mixture with components ``k = 1, ..., K`` has the density
``p(y) = sum_k theta_k p_k(y | mu_k, ...)``. Each component is an ordinary
response family whose distributional parameters carry the component number
as a suffix (``mu1``, ``sigma1``, ``mu2``, ``nu2``, ...), and the mixing
proportions ``theta_k`` either share one Dirichlet simplex, are fixed, or are
predicted through a softmax over component linear predictors with one
reference component held at zero.

The likelihood is a response kernel (`mixture_kernel`) whose parts emit
every component and every predicted mixing proportion as a nested response
block (`quivers.formulas._blocks.ResponseBlock`): the ordinary single-family
compiler emits the block, which is placed in the component's
``dpar_<parameter><k>_`` namespace (the namespace of the mixture's
distributional parameter of the same name), and the component's observation
becomes one component of a native QVR ``Mixture``. Every component thus
inherits the complete single-family predictor surface, priors, censoring
checks and parameter conversions. A correlation that crosses the
components' predictors, ``(1 | p | g)``, is a group block the model plans
and emits with its own group blocks, and the components read it.

Ordering (``order = "mu"``) identifies the components by strictly increasing
location intercepts. The first intercept keeps its prior; later intercepts
are the first plus cumulative sums of positive gaps, the same exact
positive-coordinate transform used for ordinal thresholds, so the mirrored
labelings are excluded from the parameter space rather than penalized.

A mixture of ordinal families has no location intercepts to order. brms
identifies it under ``order = "mu"`` by one threshold vector shared by every
component, and leaves each component its own thresholds under
``order = "none"``, its default for ordinal mixtures.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import math
import re
from typing import Literal, TypeVar

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_VALUE
from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    LetStep,
    ProgramStep,
    SampleStep,
)
from quivers.dsl.program_templates import _bound_names
from quivers.dsl.step_resolution import _step_value_names
from quivers.formulas._ast import (
    _add,
    _call,
    _factor,
    _index,
    _let,
    _var,
    row_zero,
    stack_rows,
)
from quivers.formulas._blocks import NestedBlocks, ResponseBlock
from quivers.formulas._design import SymbolicColumn
from quivers.formulas._kernel import (
    KernelEmission,
    KernelInputs,
    KernelPart,
    ResponseKernel,
)
from quivers.formulas._modifiers import response_modifiers
from quivers.formulas._observation import Observation
from quivers.formulas._priors import PriorTarget
from quivers.formulas._response_modifiers import response_input_name
from quivers.formulas._term import AxisDecl, InputDecl, SiteDecl
from quivers.formulas._threshold_groups import (
    LOCATION_SITE,
    MATRIX_SITE,
    THRESHOLD_GROUP_NAMES,
)
from quivers.formulas.family import AuxParam, Family, MixtureSpec, families, links
from quivers.formulas.formula import Formula, _qvr_name


_Value = TypeVar("_Value")

#: Object carrying the component axis of weights and stacked predictors.
COMPONENT_AXIS = "FormulaMixtureComponent"
#: Object carrying the K - 1 positive gaps between ordered intercepts.
SPACING_AXIS = "FormulaMixtureSpacing"
#: Site of the positive gaps between ordered component intercepts.
SPACING_SITE = "mixture_intercept_spacing"
#: Default prior of each gap between consecutive ordered intercepts.
SPACING_PRIOR = "HalfNormal(5.0)"
#: Let binding of the mixing proportions passed to the native mixture.
WEIGHTS = "mixture_weights"
#: Name, among an emission's names, of the first ordered intercept.
FIRST_INTERCEPT = "mixture_first_intercept"


def _component_family(value: str | Family) -> Family:
    """Resolve and check one mixture component.

    Parameters
    ----------
    value : str or Family
        Registered family name or a configured family value.

    Returns
    -------
    Family
        The component family.

    Raises
    ------
    ValueError
        If the family is unknown, is excluded from mixtures by brms, or is
        not supported as a mixture component.
    """
    if isinstance(value, str):
        if value not in families:
            raise ValueError(
                f"mixture: unknown family {value!r}; choices are {sorted(families)}"
            )
        value = families[value]
    if not isinstance(value, Family):
        raise TypeError("mixture components must be family names or Family values")
    if value.mixture is not None:
        raise ValueError("mixture: a component cannot itself be a mixture")
    if value.decision_event is not None:
        raise ValueError(
            f"mixture: family {value.name!r} is not supported as a mixture "
            "component: its signed decision response is not a scalar mixture event"
        )
    if value.baseline is not None:
        raise ValueError(
            f"mixture: family {value.name!r} is not supported as a mixture "
            "component: its baseline hazard is shared by construction; brms's "
            "Stan program for two Cox components declares the baseline data "
            "twice and does not compile"
        )
    if value.custom is not None:
        raise ValueError(
            "mixture: custom families are not supported as mixture components"
        )
    if value.response is not None:
        raise ValueError(
            f"mixture: family {value.name!r} is not allowed in mixture models: "
            "a component must have a scalar response event"
        )
    if any(aux.name == "theta" for aux in value.aux_params):
        raise ValueError("mixture: parameter name 'theta' is reserved in mixtures")
    return value


def mixture(
    *components: str | Family,
    nmix: int | Sequence[int] = 1,
    order: Literal["mu", "none"] | bool | None = None,
    equal: Mapping[str, str] | None = None,
) -> Family:
    """Build a finite mixture family, brms ``mixture()``.

    The distributional parameters of component ``k`` are those of its family
    with ``k`` appended (``mu1``, ``sigma1``, ``mu2``, ...), plus a mixing
    proportion ``theta<k>``. The main formula predicts every ``mu<k>``
    that has no formula of its own; each component receives its own
    coefficients.

    Parameters
    ----------
    *components : str or Family
        Component families, as registered names or configured family values
        (for instance with links set through `Family.with_links`).
    nmix : int or Sequence[int]
        Number of repetitions of each component, brms ``nmix``: one integer
        for every component or one per component.
    order : {"mu", "none"} or bool or None
        Identification by ordered location intercepts. ``"mu"`` or True orders
        the intercepts of ``mu1, ..., muK``; ``"none"`` or False applies no
        constraint. None selects ``"mu"`` when all components share one
        family and ``"none"`` otherwise, as brms does. For ordinal components
        ``"mu"`` gives every component one shared threshold vector, and None
        selects ``"none"``, where each component has its own thresholds.
    equal : Mapping[str, str] or None
        Parameters set equal to another parameter of the same class, brms
        ``bf(..., sigma2 = "sigma1")``; keys are the equated parameters.

    Returns
    -------
    Family
        The mixture family with suffixed auxiliary parameters.

    Raises
    ------
    ValueError
        If fewer than two components result, families mix real and integer
        support or ordinal and other families, ``nmix`` or ``order`` is
        invalid, a component is excluded, an equality is invalid, or shared
        thresholds are requested for components without flexible thresholds.
    """
    resolved = tuple(_component_family(component) for component in components)
    counts = (nmix,) * len(resolved) if isinstance(nmix, int) else tuple(nmix)
    if len(counts) != len(resolved):
        raise ValueError(
            "mixture: the length of 'nmix' should be the same as the number of "
            "mixture components"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts
    ):
        raise ValueError("mixture: nmix must contain positive integers")
    expanded = tuple(
        family
        for family, count in zip(resolved, counts, strict=True)
        for _ in range(count)
    )
    if len(expanded) < 2:
        raise ValueError("mixture: expecting at least 2 mixture components")
    integer = {family.native.discrete for family in expanded}
    if len(integer) > 1:
        raise ValueError("mixture: cannot mix families with real and integer support")
    ordinal = {family.ordinal is not None for family in expanded}
    if len(ordinal) > 1:
        raise ValueError("mixture: cannot mix ordinal and non-ordinal families")
    if order is None:
        chosen: Literal["mu", "none"] = (
            "mu"
            if len({family.name for family in expanded}) == 1 and ordinal == {False}
            else "none"
        )
    elif isinstance(order, bool):
        chosen = "mu" if order else "none"
    elif order in ("mu", "none"):
        chosen = order
    else:
        raise ValueError("mixture: argument 'order' must be 'mu', 'none', or a bool")
    if (
        chosen == "mu"
        and ordinal == {True}
        and any(
            family.ordinal is None
            or family.ordinal.kind == "hurdle_cumulative"
            or family.ordinal.thresholds != "flexible"
            for family in expanded
        )
    ):
        raise ValueError(
            "mixture: thresholds shared across ordinal components (order = 'mu') "
            "require flexible thresholds and no hurdle component; use order = 'none'"
        )
    aux_params: list[AuxParam] = []
    trials = None
    for index, family in enumerate(expanded, start=1):
        aux_params.append(
            AuxParam(
                name=f"mu{index}",
                prior="Normal(0.0, 5.0)",
                link=family.location_link,
            )
        )
        aux_params.extend(
            aux.with_(name=f"{aux.name}{index}") for aux in family.aux_params
        )
        if family.trials_parameter is not None:
            trials = family.trials_parameter
    aux_params.extend(
        AuxParam(name=f"theta{index}", prior="Normal(0.0, 5.0)", link=links["identity"])
        for index in range(1, len(expanded) + 1)
    )
    equalities = tuple(sorted((equal or {}).items()))
    _validate_equalities(equalities, {aux.name for aux in aux_params})
    return Family(
        name="mixture",
        location_link=links["identity"],
        observe_family="Mixture",
        aux_params=tuple(aux_params),
        trials_parameter=trials,
        recorded_events=all(family.recorded_events for family in expanded),
        mixture=MixtureSpec(components=expanded, order=chosen, equal=equalities),
    )


def _dpar_class(name: str) -> str:
    """Strip the component number from a mixture parameter name.

    Parameters
    ----------
    name : str
        Suffixed parameter name such as ``sigma2``.

    Returns
    -------
    str
        The parameter class, such as ``sigma``.
    """
    return re.sub(r"\d+$", "", name)


def _dpar_id(name: str) -> int:
    """Read the component number of a mixture parameter name.

    Parameters
    ----------
    name : str
        Suffixed parameter name such as ``sigma2``.

    Returns
    -------
    int
        The one-based component number.

    Raises
    ------
    ValueError
        If the name carries no component number.
    """
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"mixture parameter {name!r} has no component number")
    return int(match.group(1))


def equated_sites(family: Family, sampled: Collection[str]) -> list[str]:
    """List mixture parameters equated to a sampled parameter.

    brms declares ``sigma2 = sigma1`` as a transformed parameter, so the
    equated parameter is reported with the draws under its own name. A
    parameter equated to a predicted one is a row quantity and is not listed.

    Parameters
    ----------
    family : Family
        Response family.
    sampled : Collection[str]
        Sites the posterior draws hold.

    Returns
    -------
    list[str]
        Equated parameter names whose source is a sampled site.
    """
    if family.mixture is None:
        return []
    return [target for target, source in family.mixture.equal if source in sampled]


def _validate_equalities(
    equalities: tuple[tuple[str, str], ...], names: set[str]
) -> None:
    """Check brms dpar equality constraints.

    Parameters
    ----------
    equalities : tuple[tuple[str, str], ...]
        (equated, source) parameter pairs.
    names : set[str]
        Auxiliary parameter names of the mixture.

    Raises
    ------
    ValueError
        If a name is unknown, equates a parameter with itself, equates a
        location or mixing proportion, or crosses parameter classes.
    """
    targets = {target for target, _ in equalities}
    for target, source in equalities:
        if target not in names or source not in names:
            raise ValueError(
                f"mixture: invalid equality {target!r} = {source!r}; "
                f"parameters are {sorted(names)}"
            )
        if target == source:
            raise ValueError(
                f"mixture: equating {target!r} with itself is not meaningful"
            )
        if _dpar_class(target) in {"mu", "theta"}:
            raise ValueError(
                f"mixture: equating parameters of class {_dpar_class(target)!r} "
                "is not allowed"
            )
        if _dpar_class(target) != _dpar_class(source):
            raise ValueError("mixture: can only equate parameters of the same class")
        if source in targets:
            raise ValueError(
                f"mixture: {source!r} is itself equated; equate to a free parameter"
            )


def expand_mixture(family: Family, components: int) -> Family:
    """Repeat the component of a mixture template.

    Parameters
    ----------
    family : Family
        A response family.
    components : int
        Component count of a mixture template.

    Returns
    -------
    Family
        The mixture of ``components`` copies of the one component of a
        template, as the registered ``"mixture"`` family is of the Gaussian,
        otherwise ``family`` unchanged.

    Raises
    ------
    ValueError
        If ``components`` is not an integer of at least two.
    """
    if family.mixture is None or len(family.mixture.components) != 1:
        return family
    if (
        isinstance(components, bool)
        or not isinstance(components, int)
        or components < 2
    ):
        raise ValueError("mixture_components must be an integer of at least two")
    return mixture(*family.mixture.components, nmix=components)


def mixture_distributional(
    family: Family,
    formula: str,
    distributional: Mapping[str, _Value] | None,
    components: int,
) -> tuple[Family, dict[str, _Value | str] | None]:
    """Copy the main formula to every component location without a formula.

    brms combines the main formula with each ``mu<k>`` formula; an explicit
    ``mu<k> ~ ...`` replaces it for that component.

    Parameters
    ----------
    family : Family
        Response family, possibly the bare registered mixture.
    formula : str
        Authored main formula.
    distributional : Mapping or None
        Authored distributional formulas.
    components : int
        Component count of the bare registered mixture family.

    Returns
    -------
    tuple[Family, dict or None]
        The expanded family and the distributional formulas including every
        component location.

    Raises
    ------
    ValueError
        If a mixture lacks a main formula right-hand side or names ``mu``.
    """
    family = expand_mixture(family, components)
    if family.mixture is None:
        return family, None if distributional is None else dict(distributional)
    if "~" not in formula:
        raise ValueError("mixture models require a formula with '~'")
    authored: dict[str, _Value | str] = dict(distributional or {})
    if "mu" in authored:
        raise ValueError(
            "mixture location formulas are named mu1, mu2, ...; 'mu' is the main formula"
        )
    rhs = formula.split("~", 1)[1]
    for index in range(1, len(family.mixture.components) + 1):
        authored.setdefault(f"mu{index}", f"mu{index} ~ {rhs}")
    return family, authored


def validate_mixture_options(
    family: Family,
    distributional: Mapping[str, Formula],
    fixed_parameters: Mapping[str, float],
    *,
    predictor_name: str | None,
    nonlinear: bool,
    thresholds_by: str | None,
) -> None:
    """Check mixing proportions, fixed values and refused options of a mixture.

    Parameters
    ----------
    family : Family
        The mixture family.
    distributional : Mapping[str, Formula]
        Distributional formulas by parameter name.
    fixed_parameters : Mapping[str, float]
        Fixed parameter values by name.
    predictor_name : str or None
        External neural predictor binding.
    nonlinear : bool
        Whether a nonlinear predictor graph is configured.
    thresholds_by : str or None
        Grouped ordinal thresholds.

    Raises
    ------
    ValueError
        If proportions are partly predicted or partly fixed (brms requires
        all but one predicted, or all fixed), a fixed proportion is negative,
        an equated parameter also has a formula or a fixed value, or a
        refused option is set.
    """
    spec = family.mixture
    assert spec is not None
    count = len(spec.components)
    if predictor_name is not None:
        raise ValueError("mixture models do not support a neural predictor")
    if nonlinear:
        raise ValueError("mixture models do not support nonlinear formulas")
    if thresholds_by is not None:
        raise ValueError("mixture models do not support grouped ordinal thresholds")
    predicted = [name for name in distributional if _dpar_class(name) == "theta"]
    fixed = {
        name: value
        for name, value in fixed_parameters.items()
        if _dpar_class(name) == "theta"
    }
    if predicted and fixed:
        raise ValueError(
            "mixture: mixing proportions are either predicted or fixed, not both"
        )
    if predicted and len(predicted) != count - 1:
        raise ValueError("mixture: can only predict all but one mixing proportion")
    if fixed:
        if len(fixed) != count:
            raise ValueError("mixture: can only fix no or all mixing proportions")
        values = list(fixed.values())
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            )
            or not sum(values) > 0
        ):
            raise ValueError(
                "mixture: fixed mixing proportions must be finite, nonnegative, "
                "and not all zero"
            )
    for target, _ in spec.equal:
        if target in distributional or target in fixed_parameters:
            raise ValueError(
                f"mixture: equated parameter {target!r} cannot also be predicted or fixed"
            )


def _shared_names(formula: Formula, trials: int | str | None) -> frozenset[str]:
    """Collect inputs and objects shared by every component of one response.

    Parameters
    ----------
    formula : Formula
        Main formula with the response and its modifiers.
    trials : int or str or None
        Trial count or trial column of finite-trial components.

    Returns
    -------
    frozenset[str]
        Response, modifier, trial, weight and group names.
    """
    names = {
        "Resp",
        response_input_name(formula),
        _qvr_name(formula.response_name),
        "formula_weights",
        "formula_decision",
        "formula_trials",
    }
    names.update(column.qvr_name for _, column in formula.modifiers.columns())
    # brms thres(n, gr): the components read one group axis and its selectors.
    names.update(THRESHOLD_GROUP_NAMES)
    if isinstance(trials, str):
        names.add(_qvr_name(trials))
    return frozenset(names)


def _row_stack(
    name: str, values: Sequence[LetExprNode], steps: list[ProgramStep]
) -> None:
    """Bind per-component row values as one ``(Resp, component)`` matrix.

    Parameters
    ----------
    name : str
        Binding of the stacked matrix.
    values : Sequence[LetExprNode]
        One scalar or row-vector value per component.
    steps : list[ProgramStep]
        Program steps receiving the bindings.
    """
    zero = "mixture_row_zero"
    if not any(isinstance(step, LetStep) and step.name == zero for step in steps):
        steps.append(row_zero(zero))
    steps.extend(
        stack_rows(
            name,
            ((index, _add(_var(zero), value)) for index, value in enumerate(values)),
            ("component", COMPONENT_AXIS),
            stem=f"{name}_by_component",
        )
    )


def _shared_thresholds(
    steps: Sequence[ProgramStep],
    index: int,
    source: int,
    cutpoints: int,
    intercept: str | None,
    grouped: bool = False,
) -> list[ProgramStep]:
    """Give one ordinal component the thresholds of the source component.

    brms fixes the thresholds of an ordered ordinal mixture to one vector.
    The emitted threshold of transition ``k`` is the centered cutpoint minus
    the location intercept, so the component reads both from the source.
    With threshold groups (``thres(n, gr)``) brms fixes each group's vector,
    and the component reads the source's group-by-threshold matrix.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        Renamed steps of component ``index``.
    index : int
        One-based number of the component receiving the thresholds.
    source : int
        One-based number of the component whose thresholds are sampled.
    cutpoints : int
        Number of thresholds.
    intercept : str or None
        Sampled location intercept of the component; None without one.
    grouped : bool
        Whether the response has threshold groups.

    Returns
    -------
    list[ProgramStep]
        Steps without the component's own threshold and intercept sites,
        whose bindings read the source component's values.
    """
    stem = f"dpar_mu{index}_cutpoint"
    shared_intercept = (intercept or "").replace(
        f"dpar_mu{index}_", f"dpar_mu{source}_", 1
    )
    if grouped:
        matrix = f"dpar_mu{index}_{MATRIX_SITE}"
        own = (
            stem,
            f"dpar_mu{index}_threshold_shape_",
            f"dpar_mu{index}_threshold_center_",
            f"dpar_mu{index}_threshold_values_by_position",
            f"dpar_mu{index}_{LOCATION_SITE}",
        )
        shared: list[ProgramStep] = []
        for step in steps:
            name = (
                step.vars[0]
                if isinstance(step, SampleStep)
                else step.name
                if isinstance(step, LetStep)
                else None
            )
            if intercept is not None and name == intercept:
                shared.append(_let(intercept, _var(shared_intercept)))
            elif name == matrix:
                shared.append(_let(matrix, _var(f"dpar_mu{source}_{MATRIX_SITE}")))
            elif name is None or not name.startswith(own):
                shared.append(step)
        return shared
    centered = f"dpar_mu{index}_cutpoints"
    result: list[ProgramStep] = []
    placed = False
    for step in steps:
        name = (
            step.vars[0]
            if isinstance(step, SampleStep)
            else step.name
            if isinstance(step, LetStep)
            else None
        )
        if intercept is not None and name == intercept:
            result.append(_let(intercept, _var(shared_intercept)))
            continue
        if name is None or not name.startswith(stem):
            result.append(step)
            continue
        if not placed:
            placed = True
            result.append(
                _let(
                    centered,
                    _factor(
                        (("cutpoint", f"dpar_mu{index}_FormulaCutpoint"),),
                        cases=tuple(
                            (
                                position,
                                _index(
                                    f"dpar_mu{source}_cutpoints",
                                    LetExprLiteral(
                                        value=float(position), integral=True
                                    ),
                                ),
                            )
                            for position in range(cutpoints)
                        ),
                    ),
                )
            )
    return result


def response_blocks(
    family: Family,
    formula: Formula,
    distributional: Mapping[str, Formula],
    fixed: Mapping[str, float],
    trials: int | str | None,
) -> NestedBlocks:
    """State the response laws a family emits as nested blocks.

    A mixture emits one block per component, reading ``mu<k>`` and the
    component's auxiliaries ``<aux><k>`` over the shared response, and one
    per predicted mixing proportion ``theta<k>``, a Gaussian location of unit
    scale. The component locations replace the main design, whose
    right-hand side brms copies into every ``mu<k>`` without a formula. Any
    other family emits no block.

    Parameters
    ----------
    family : Family
        Response family.
    formula : Formula
        Main formula with the response and its modifiers.
    distributional : Mapping[str, Formula]
        Distributional formulas by model parameter.
    fixed : Mapping[str, float]
        Fixed parameter values by model parameter.
    trials : int or str or None
        Trial count or trial column of finite-trial components.

    Returns
    -------
    NestedBlocks
        The blocks, components first and in component order.
    """
    spec = family.mixture
    if spec is None:
        return NestedBlocks()
    blocks: list[ResponseBlock] = []
    for index, component in enumerate(spec.components, start=1):
        auxiliaries = tuple(aux.name for aux in component.aux_params)
        location = distributional.get(f"mu{index}")
        blocks.append(
            ResponseBlock(
                location="mu",
                index=index,
                auxiliaries=auxiliaries,
                family=component,
                formula=None
                if location is None
                else location.with_(
                    response_name=formula.response_name,
                    response_values=formula.response_values,
                    response_layout=formula.response_layout,
                    modifiers=formula.modifiers,
                ),
                distributional={
                    aux: distributional[f"{aux}{index}"]
                    for aux in auxiliaries
                    if f"{aux}{index}" in distributional
                },
                fixed={
                    aux: fixed[f"{aux}{index}"]
                    for aux in auxiliaries
                    if f"{aux}{index}" in fixed
                },
                trials=trials,
            )
        )
    for index in range(1, len(spec.components) + 1):
        predictor = distributional.get(f"theta{index}")
        if predictor is None:
            continue
        blocks.append(
            ResponseBlock(
                location="theta",
                index=index,
                family=families["gaussian"],
                # The proportion predictor reads no response categories.
                formula=predictor.with_(
                    response_layout=predictor.response_layout.with_(category_levels=())
                ),
                fixed={"sigma": 1.0},
            )
        )
    return NestedBlocks(blocks=tuple(blocks), replaces_location=True)


def _model_drawn(emission: KernelEmission) -> frozenset[str]:
    """List the level hyperparameters the model's blocks draw or read.

    Parameters
    ----------
    emission : KernelEmission
        Program being emitted.

    Returns
    -------
    frozenset[str]
        Sites drawn by the model and the Student-t degrees of freedom of the
        nested blocks' laws.
    """
    return emission.host.drawn_sites | {
        law.level.mixing.df_site
        for law in emission.laws
        if law.level.mixing is not None
    }


class ModelNames(KernelPart):
    """Fix the names the nested blocks keep: those the model already owns.

    Parameters
    ----------
    names : frozenset[str]
        Response, modifier, trial, weight and threshold-group names every
        component reads.
    """

    kind: Literal["mixture_model_names"] = "mixture_model_names"
    names: frozenset[str] = frozenset()

    def emit(self, emission: KernelEmission) -> None:
        """Record the shared names: the declared ones, the model's bindings,
        and the inputs of the group coefficients another program emits.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.shared = (
            self.names
            | {name for step in emission.steps for name in _bound_names((step,))}
            | {
                name
                for value in emission.host.supplied.values()
                for name in _step_value_names(LetStep(name="supplied", value=value))
            }
        )


class InterceptSpacing(KernelPart):
    """Draw the positive gaps between ordered component intercepts.

    Parameters
    ----------
    gaps : int
        Number of gaps, one fewer than the components.
    """

    kind: Literal["mixture_intercept_spacing"] = "mixture_intercept_spacing"
    gaps: int

    def emit(self, emission: KernelEmission) -> None:
        """Declare the gap axis and draw the gaps.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.declare(AxisDecl(name=SPACING_AXIS, size=self.gaps))
        emission.steps.append(
            emission.host.draw_site(
                SiteDecl(
                    name=SPACING_SITE,
                    class_=SPACING_SITE,
                    default=SPACING_PRIOR,
                    axes=(SPACING_AXIS,),
                ),
                None,
            )
        )
        emission.targets.append(
            PriorTarget(
                site=SPACING_SITE,
                class_=SPACING_SITE,
                axes=(SPACING_AXIS,),
                shape=(self.gaps,),
                lower=0.0,
            )
        )


def _drop_targets(emission: KernelEmission, sites: Collection[str]) -> None:
    """Remove the prior targets of sites a part binds deterministically.

    Parameters
    ----------
    emission : KernelEmission
        Program being emitted.
    sites : Collection[str]
        Sites no longer drawn.
    """
    emission.targets[:] = [
        target for target in emission.targets if target.site not in sites
    ]


class ComponentBlock(KernelPart):
    """Emit one mixture component and identify it.

    The component is emitted as a nested block. Under ordering its sampled
    location intercept becomes the first component's plus cumulative
    positive gaps; under shared ordinal thresholds it reads the threshold
    source's thresholds and intercept; an equated parameter reads its
    source. The steps are kept for `PlaceComponents`.

    Parameters
    ----------
    block : ResponseBlock
        The component.
    ordered : bool
        Whether the components are identified by ordered intercepts.
    thresholds_from : int or None
        Component whose thresholds this one reads; None for its own.
    shared_thresholds : bool
        Whether the components share one threshold vector.
    intercepts : frozenset[bool]
        Whether each component location formula keeps its intercept.
    equal : tuple[tuple[str, str], ...]
        Parameters of this component equated to another, with their sources.
    """

    kind: Literal["mixture_component"] = "mixture_component"
    block: ResponseBlock = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    ordered: bool = False
    thresholds_from: int | None = None
    shared_thresholds: bool = False
    intercepts: frozenset[bool] = frozenset()
    equal: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Emit the component and keep its identified steps.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.

        Raises
        ------
        ValueError
            If the location has no formula or reads latent-dependent inputs,
            ordering lacks a location intercept, or the location formulas
            disagree on their intercepts under shared thresholds.
        """
        block = self.block
        index = block.index
        location = block.formula
        if location is None:
            raise ValueError(
                f"mixture component {index} requires its location formula "
                f"'{block.label}'"
            )
        if any(isinstance(column, SymbolicColumn) for column in location.fixed_columns):
            raise ValueError(
                "mixture components do not support latent-dependent inputs"
            )
        if self.ordered and not any(
            column.is_intercept for column in location.fixed_columns
        ):
            raise ValueError(
                "identifying mixture components via ordering requires population-level "
                "intercepts to be present; try order = 'none'"
            )
        placed = emission.host.emit_block(
            block, emission.shared, emission.gathered, _model_drawn(emission)
        )
        emission.absorb(block.label, placed)
        steps = list(placed.steps)
        sampled = {step.vars[0] for step in steps if isinstance(step, SampleStep)}
        # The sampled intercept: brms's centered one when the component centers.
        intercept = next(
            (
                name
                for name in (
                    f"dpar_mu{index}_intercept_centered",
                    f"dpar_mu{index}_intercept",
                )
                if name in sampled
            ),
            None,
        )
        if self.ordered and intercept is not None:
            if index == 1:
                emission.names[FIRST_INTERCEPT] = intercept
            else:
                first = emission.names.get(FIRST_INTERCEPT, "dpar_mu1_intercept")
                # Cumulative positive gaps, written as an affine sum so that
                # derived coefficient draws evaluate it like the program does.
                gaps = [
                    _index(
                        SPACING_SITE, LetExprLiteral(value=float(gap), integral=True)
                    )
                    for gap in range(index - 1)
                ]
                steps = [
                    _let(intercept, _add(_var(first), *gaps))
                    if isinstance(step, SampleStep) and step.vars == (intercept,)
                    else step
                    for step in steps
                ]
                _drop_targets(emission, (intercept,))
        if self.shared_thresholds:
            if len(self.intercepts) > 1:
                raise ValueError(
                    "mixture: thresholds shared across ordinal components require "
                    "every location formula to keep or drop its intercept alike"
                )
            if self.thresholds_from is not None:
                dropped = {
                    step.vars[0]
                    for step in steps
                    if isinstance(step, SampleStep)
                    and (
                        step.vars[0].startswith(f"dpar_mu{index}_cutpoint")
                        or step.vars[0] == intercept
                        or step.vars[0] == f"dpar_mu{index}_{LOCATION_SITE}"
                    )
                }
                steps = _shared_thresholds(
                    steps,
                    index,
                    self.thresholds_from,
                    emission.host.category_count(emission.formula, block.family) - 1,
                    intercept if self.intercepts == {True} else None,
                    grouped=emission.formula.modifiers.threshold_groups is not None,
                )
                _drop_targets(emission, dropped)
        for target, source in self.equal:
            steps = [
                _let(target, _var(source))
                if isinstance(step, SampleStep) and step.vars == (target,)
                else step
                for step in steps
            ]
            _drop_targets(emission, (target,))
        emission.buffers[block.label] = steps


class PlaceComponents(KernelPart):
    """Place the components' steps and stack their predictors over the rows.

    Parameters
    ----------
    labels : tuple[str, ...]
        Component locations, in component order.
    order : tuple[str, ...]
        Component locations in placement order: a component whose
        thresholds others read precedes them.
    equal : tuple[tuple[str, str], ...]
        Equated parameters and their sources.
    """

    kind: Literal["mixture_place_components"] = "mixture_place_components"
    labels: tuple[str, ...]
    order: tuple[str, ...]
    equal: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Append the steps and bind ``eta`` and ``mu`` by row and component.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.

        Raises
        ------
        ValueError
            If an equated parameter's source is not a parameter of the model.
        """
        etas: list[LetExprNode] = []
        means: list[LetExprNode] = []
        for label in self.labels:
            bound = {
                step.name
                for step in emission.buffers[label]
                if isinstance(step, LetStep)
            }
            etas.append(_var(f"dpar_{label}_eta"))
            means.append(
                _var(label)
                if label in bound
                else _call("exp", _var(f"dpar_{label}_eta"))
            )
        for label in self.order:
            emission.steps.extend(emission.buffers.pop(label))
        for target, source in self.equal:
            if not any(
                isinstance(step, (SampleStep, LetStep))
                and (step.vars[0] if isinstance(step, SampleStep) else step.name)
                == source
                for step in emission.steps
            ):
                raise ValueError(
                    f"mixture: {target!r} is equated to {source!r}, which is not a "
                    "parameter of this model"
                )
        _row_stack("eta", etas, emission.steps)
        _row_stack("mu", means, emission.steps)


class PredictedWeights(KernelPart):
    """Mixing proportions as the softmax of component predictors.

    Parameters
    ----------
    count : int
        Number of components.
    blocks : tuple[ResponseBlock, ...]
        Predictors of the proportions that have one; the others have a zero
        logit.
    """

    kind: Literal["mixture_predicted_weights"] = "mixture_predicted_weights"
    count: int
    blocks: tuple[ResponseBlock, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def emit(self, emission: KernelEmission) -> None:
        """Emit the predictors and bind the weights.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        predicted = {block.index: block for block in self.blocks}
        logits: list[LetExprNode] = []
        for index in range(1, self.count + 1):
            block = predicted.get(index)
            if block is None:
                logits.append(LetExprLiteral(value=0.0))
                continue
            placed = emission.host.emit_block(
                block, emission.shared, emission.gathered, _model_drawn(emission)
            )
            emission.absorb(block.label, placed)
            emission.steps.extend(placed.steps)
            logits.append(_var(f"dpar_{block.label}_eta"))
        _row_stack("mixture_logits", logits, emission.steps)
        emission.steps.append(_let(WEIGHTS, _call("softmax", _var("mixture_logits"))))


class FixedWeights(KernelPart):
    """Fixed mixing proportions, normalized to sum to one.

    Parameters
    ----------
    values : tuple[float, ...]
        Fixed proportions in component order.
    total : float
        Their sum.
    """

    kind: Literal["mixture_fixed_weights"] = "mixture_fixed_weights"
    values: tuple[float, ...]
    total: float

    def emit(self, emission: KernelEmission) -> None:
        """Bind the weights.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        emission.steps.append(
            _let(
                WEIGHTS,
                _factor(
                    (("component", COMPONENT_AXIS),),
                    cases=tuple(
                        (index, LetExprLiteral(value=value / self.total))
                        for index, value in enumerate(self.values)
                    ),
                ),
            )
        )


class DirichletWeights(KernelPart):
    """Mixing proportions drawn from a uniform Dirichlet.

    Parameters
    ----------
    count : int
        Number of components.
    """

    kind: Literal["mixture_dirichlet_weights"] = "mixture_dirichlet_weights"
    count: int

    def emit(self, emission: KernelEmission) -> None:
        """Draw the simplex and bind the weights.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        concentration = ", ".join("1.0" for _ in range(self.count))
        emission.steps.append(
            emission.host.draw_site(
                SiteDecl(
                    name="theta",
                    class_="theta",
                    default=f"Dirichlet(concentration=[{concentration}])",
                ),
                None,
            )
        )
        emission.steps.append(_let(WEIGHTS, _var("theta")))
        emission.targets.append(PriorTarget(site="theta", class_="theta", event_rank=1))


class ObserveMixture(KernelPart):
    """Observe the response under the mixture of the components' laws.

    The components' arguments are stacked by component for a native mixture
    law the components share, and otherwise their observations become the
    components of ``Mixture``. The response's modifiers are folded over the
    observation. Inputs read only as component arguments carry no plate
    context for type inference, so each row-aligned numeric one is declared
    as a ``Resp``-indexed tensor.

    Parameters
    ----------
    labels : tuple[str, ...]
        Component locations, in component order.
    native : str or None
        Native mixture law of the components' family; None for ``Mixture``.
    """

    kind: Literal["mixture_observe"] = "mixture_observe"
    labels: tuple[str, ...]
    native: str | None = None

    def emit(self, emission: KernelEmission) -> None:
        """Write the observation and declare the inputs it reads.

        Parameters
        ----------
        emission : KernelEmission
            Program being emitted.
        """
        steps = emission.steps
        formula = emission.formula
        observations = [emission.placed[label].observation for label in self.labels]
        site = _qvr_name(formula.response_name)
        if self.native is not None:
            # The native law of the mixture takes the stacked component arguments.
            stacked = observations[0].parameters
            for parameter in stacked:
                _row_stack(
                    f"mixture_{parameter}",
                    [component.argument(parameter) for component in observations],
                    steps,
                )
            mixed = Observation(
                site=site,
                kernel=self.native,
                parameters=("weights", *stacked),
                values=(WEIGHTS, *(f"mixture_{parameter}" for parameter in stacked)),
            )
        else:
            mixed = Observation(
                site=site,
                kernel="Mixture",
                parameters=("weights",),
                values=(WEIGHTS,),
                components=tuple(observations),
            )
        for modifier in response_modifiers(formula.modifiers, emission.inputs.residual):
            mixed = modifier.observe(mixed)
        emission.host.observed(mixed)
        steps.append(mixed.step())
        rows = int(formula.response_values.shape[0])
        bound = set(_bound_names(tuple(steps))) | set(emission.parameters)
        free = {name for step in steps for name in _step_value_names(step)} - bound
        values = emission.host.column_values(formula)
        for name in sorted(free):
            value = values.get(name)
            if value is None or value.ndim != 1 or value.shape[0] != rows:
                continue
            emission.data.append(
                InputDecl(
                    name=name,
                    element="Real" if value.is_floating_point() else "Int",
                    axes=("Resp",),
                    declared=True,
                )
            )


def mixture_kernel(inputs: KernelInputs, formula: Formula) -> ResponseKernel:
    """State the kernel of a finite mixture.

    The body emits every component as a nested block, identifies the
    components (ordered intercepts, shared ordinal thresholds, equated
    parameters), binds the mixing proportions (predicted, fixed, or a
    Dirichlet simplex), and observes ``Mixture(weights, components)`` or the
    components' native mixture law. Scoring costs O(n K) beyond the
    component predictors.

    Parameters
    ----------
    inputs : KernelInputs
        The mixture family, its nested blocks and fixed values.
    formula : Formula
        Main formula with the response and modifiers.

    Returns
    -------
    ResponseKernel
        The kernel; it has no slot of its own.

    Raises
    ------
    ValueError
        If integer-support components receive a response that is not a
        nonnegative integer.
    """
    family = inputs.family
    spec = family.mixture
    assert spec is not None and inputs.blocks is not None
    count = len(spec.components)
    if spec.components[0].native.discrete:
        values = np.asarray(formula.response_values, dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.equal(values, np.round(values)).all()
        ):
            raise ValueError(
                "a mixture of integer-support families requires nonnegative "
                f"integer responses ({', '.join(c.name for c in spec.components)})"
            )
    components = tuple(
        block for block in inputs.blocks.blocks if block.location == "mu"
    )
    proportions = tuple(
        block for block in inputs.blocks.blocks if block.location == "theta"
    )
    ordinal = spec.components[0].ordinal is not None
    # brms: an ordered ordinal mixture shares one threshold vector, ordered
    # when any component has ordered thresholds.
    shared_thresholds = ordinal and spec.order == "mu"
    ordered = spec.order == "mu" and not ordinal
    threshold_source = next(
        (
            index
            for index, component in enumerate(spec.components, start=1)
            if component.ordinal is not None and component.ordinal.kind == "cumulative"
        ),
        1,
    )
    intercepts = frozenset(
        any(column.is_intercept for column in block.formula.fixed_columns)
        for block in components
        if block.formula is not None
    )
    labels = tuple(block.label for block in components)
    body: list[KernelPart] = [
        ModelNames(
            names=_shared_names(formula, components[0].trials if components else None)
        )
    ]
    if ordered:
        body.append(InterceptSpacing(gaps=count - 1))
    body.extend(
        ComponentBlock(
            block=block,
            ordered=ordered,
            thresholds_from=threshold_source
            if shared_thresholds and block.index != threshold_source
            else None,
            shared_thresholds=shared_thresholds,
            intercepts=intercepts,
            equal=tuple(
                (target, source)
                for target, source in spec.equal
                if _dpar_id(target) == block.index
            ),
        )
        for block in components
    )
    body.append(
        PlaceComponents(
            labels=labels,
            # The component whose thresholds are shared precedes its readers.
            order=tuple(
                sorted(
                    labels,
                    key=lambda label: (
                        not shared_thresholds or label != f"mu{threshold_source}"
                    ),
                )
            ),
            equal=spec.equal,
        )
    )
    fixed = {
        name: value
        for name, value in inputs.fixed.items()
        if _dpar_class(name) == "theta"
    }
    if proportions:
        body.append(PredictedWeights(count=count, blocks=proportions))
    elif fixed:
        body.append(
            FixedWeights(
                values=tuple(
                    float(fixed[f"theta{index}"]) for index in range(1, count + 1)
                ),
                total=sum(float(value) for value in fixed.values()),
            )
        )
    else:
        body.append(DirichletWeights(count=count))
    native = {component.mixture_law for component in spec.components}
    body.append(
        ObserveMixture(labels=labels, native=native.pop() if len(native) == 1 else None)
    )
    return ResponseKernel(
        objects=(AxisDecl(name=COMPONENT_AXIS, size=count),),
        body=tuple(body),
        response=_qvr_name(formula.response_name),
    )


def mixture_component_count(family: Family) -> int:
    """Read the number of mixture components.

    Parameters
    ----------
    family : Family
        Response family.

    Returns
    -------
    int
        Component count, or zero for a family that is not a mixture.
    """
    return 0 if family.mixture is None else len(family.mixture.components)


__all__ = [
    "COMPONENT_AXIS",
    "expand_mixture",
    "mixture",
    "mixture_component_count",
    "mixture_distributional",
    "mixture_kernel",
    "response_blocks",
    "validate_mixture_options",
]
