"""Finite mixtures of response families: the brms ``mixture()`` family.

A finite mixture with components ``k = 1, ..., K`` has the density
``p(y) = sum_k theta_k p_k(y | mu_k, ...)``. Each component is an ordinary
response family whose distributional parameters carry the component number
as a suffix (``mu1``, ``sigma1``, ``mu2``, ``nu2``, ...), and the mixing
proportions ``theta_k`` either share one Dirichlet simplex, are fixed, or are
predicted through a softmax over component linear predictors with one
reference component held at zero.

Emission reuses the ordinary single-family compiler once per component and
once per predicted mixing proportion: each sub-emission is renamed into the
component's ``dpar_<parameter><k>_`` namespace (the namespace of the
mixture's distributional parameter of the same name) and its observation
kernel becomes one component of a native QVR ``Mixture``. Every component
thus inherits the complete single-family predictor surface, priors,
censoring checks and parameter conversions.

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
from copy import copy
import math
import re
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import didactic.api as dx
from functools import reduce

import numpy as np

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    LetStep,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
    SampleStep,
    Statement,
    TypeFromExpr,
    TypeName,
    DiscreteConstructor,
    ExportDecl,
    ExprIdent,
)
from quivers.dsl.program_templates import (
    TemplateBindings,
    _bound_names,
    _rename_step,
    substitute_program_object,
)
from quivers.dsl.step_resolution import _step_value_names
from quivers.family_catalog import FamilyCatalog
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
from quivers.formulas._group_law import (
    GroupBlockLaw,
    law_inputs,
    level_names,
    renamed_law,
)
from quivers.formulas._group_plan import check_names
from quivers.formulas._prior_emission import build_prior_targets
from quivers.formulas._term import ColumnName, ProgramNames, declare_inputs
from quivers.formulas._priors import PriorTarget
from quivers.formulas._response_modifiers import response_input_name
from quivers.formulas._threshold_groups import (
    LOCATION_SITE,
    MATRIX_SITE,
    THRESHOLD_GROUP_NAMES,
)
from quivers.formulas._modifiers import response_modifiers
from quivers.formulas._observation import Observation
from quivers.formulas.family import AuxParam, Family, MixtureSpec, families, links
from quivers.formulas._design import SymbolicColumn
from quivers.formulas.formula import Formula, RandomTerm, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule


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


class MixtureEmission(ValueModel):
    """One mixture program with its prior catalogue and group-block laws.

    Parameters
    ----------
    module : Module
        The emitted QVR module.
    prior_targets : tuple[PriorTarget, ...]
        Every overrideable latent site with mixture parameter scopes.
    group_laws : tuple[GroupBlockLaw, ...]
        Group-block laws renamed into component namespaces.
    families : FamilyCatalog or None
        Laws the terms of the sub-emissions draw from.
    names : ProgramNames
        Names of the sub-emissions' columns and bindings.
    """

    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    families: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    names: ProgramNames = dx.field(
        default_factory=ProgramNames, opaque=True, extras=OPAQUE_BY_VALUE
    )


class _Namespace(ValueModel):
    """Rename one sub-emission into its mixture parameter namespace.

    Parameters
    ----------
    location : str
        Mixture parameter class of the sub-emission's location, ``mu`` for a
        component and ``theta`` for a predicted mixing proportion.
    index : int
        One-based component number.
    auxiliaries : tuple[str, ...]
        Auxiliary parameter names of the component family, renamed with the
        component number; empty for a mixing-proportion predictor.
    shared : frozenset[str]
        Data inputs and objects shared by every component.
    """

    location: str
    index: int
    auxiliaries: tuple[str, ...] = ()
    shared: frozenset[str] = frozenset()

    def name(self, value: str) -> str:
        """Rename one value or object name.

        Parameters
        ----------
        value : str
            Local name in the sub-emission.

        Returns
        -------
        str
            The mixture name; shared names are unchanged.
        """
        if value in self.shared:
            return value
        for auxiliary in self.auxiliaries:
            if value == auxiliary:
                return f"{auxiliary}{self.index}"
            stem = f"dpar_{auxiliary}_"
            if value.startswith(stem):
                return f"dpar_{auxiliary}{self.index}_" + value[len(stem) :]
        if value == "mu":
            return f"{self.location}{self.index}"
        return f"dpar_{self.location}{self.index}_{value}"

    def prefix(self, value: str) -> str:
        """Rename a predictor coefficient prefix.

        Parameters
        ----------
        value : str
            Local prefix: empty for the location, ``dpar_<aux>_`` otherwise.

        Returns
        -------
        str
            The mixture parameter's prefix.
        """
        if not value:
            return f"dpar_{self.location}{self.index}_"
        return self.name(value)

    def dpar(self, value: str | None) -> str | None:
        """Rename a predictor scope.

        Parameters
        ----------
        value : str or None
            Local distributional parameter.

        Returns
        -------
        str or None
            The suffixed mixture parameter.
        """
        if value is None:
            return None
        if value == "mu":
            return f"{self.location}{self.index}"
        return f"{value}{self.index}"


def _indices(value: _Value, objects: Mapping[str, str]) -> _Value:
    """Rename nominal indices throughout a runtime parameter type.

    Parameters
    ----------
    value : _Value
        A type expression, a tuple of them, or a leaf field value.
    objects : Mapping[str, str]
        Local object names and their mixture names.

    Returns
    -------
    _Value
        The value with every nominal index renamed.
    """
    if isinstance(value, QiecIndexName):
        return cast(_Value, value.with_(name=objects.get(value.name, value.name)))
    if isinstance(value, tuple):
        return cast(_Value, tuple(_indices(item, objects) for item in value))
    if isinstance(value, dx.Model):
        return cast(
            _Value,
            value.with_(
                **{
                    key: _indices(getattr(value, key), objects)
                    for key in type(value).__field_specs__
                }
            ),
        )
    return value


class _SubEmission(ValueModel):
    """One renamed sub-emission.

    Parameters
    ----------
    objects : tuple[Statement, ...]
        Renamed object declarations.
    steps : tuple[ProgramStep, ...]
        Renamed program steps without the observation.
    observation : Observation or None
        The renamed observation of a component, without the response terms
        of the mixture; None for a proportion.
    parameters : tuple[QiecValueParameter, ...]
        Renamed runtime parameters.
    prior_targets : tuple[PriorTarget, ...]
        Renamed prior catalogue entries with mixture scopes.
    group_laws : tuple[GroupBlockLaw, ...]
        Renamed group-block laws.
    correlation_ids : frozenset[str]
        Group-level correlation identifiers used by the sub-emission.
    families : FamilyCatalog or None
        Laws its terms draw from, under namespaced names.
    names : ProgramNames
        Names of its columns and bindings, under namespaced names.
    """

    objects: tuple[Statement, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    steps: tuple[ProgramStep, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observation: Observation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    parameters: tuple[QiecValueParameter, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    correlation_ids: frozenset[str] = frozenset()
    families: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    names: ProgramNames = dx.field(
        default_factory=ProgramNames, opaque=True, extras=OPAQUE_BY_VALUE
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


def _local_priors(
    lens: FormulaToQVRModule, namespace: _Namespace, sites: Sequence[str]
) -> dict[str, str]:
    """Map composite exact-site priors into one sub-emission.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The mixture lens with its current site priors.
    namespace : _Namespace
        The sub-emission namespace.
    sites : Sequence[str]
        Local sample sites of the sub-emission.

    Returns
    -------
    dict[str, str]
        Local site names and their prior sources.
    """
    priors = dict(lens._user_priors)
    return {
        site: priors[namespace.name(site)]
        for site in sites
        if namespace.name(site) in priors
    }


def _sample_sites(module: Module) -> tuple[str, ...]:
    """Read every sampled latent site of an emitted module.

    Parameters
    ----------
    module : Module
        Emitted single-program module.

    Returns
    -------
    tuple[str, ...]
        Sampled site names.
    """
    return tuple(
        step.vars[0]
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, SampleStep)
    )


def _sub_emission(
    lens: FormulaToQVRModule,
    family: Family,
    formula: Formula,
    namespace: _Namespace,
    *,
    distributional: Mapping[str, Formula],
    fixed_parameters: Mapping[str, float],
    binomial_trials: int | str | None,
    default_priors: Mapping[str, str] | None = None,
    supplied: Mapping[tuple[str, str, str], LetExprNode] | None = None,
    supplied_sites: Collection[str] = (),
) -> _SubEmission:
    """Emit one component or proportion predictor through the ordinary compiler.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The mixture lens, whose options and priors the sub-emission inherits.
    family : Family
        Component family; Gaussian with unit scale for a proportion predictor.
    formula : Formula
        Location design with the shared response and modifiers.
    namespace : _Namespace
        Renaming into the mixture parameter namespace.
    distributional : Mapping[str, Formula]
        Component auxiliary formulas under their local names.
    fixed_parameters : Mapping[str, float]
        Component fixed values under their local names.
    binomial_trials : int or str or None
        Trials of a finite-trial component.
    default_priors : Mapping[str, str] or None
        Local default priors replacing the compiler defaults when the user
        gave none.
    supplied : Mapping or None
        Group coefficients of blocks shared with other mixture parameters,
        keyed by local predictor prefix, group, and slope.
    supplied_sites : Collection[str]
        Level hyperparameters of grouping factors drawn for other mixture
        parameters.

    Returns
    -------
    _SubEmission
        The renamed program parts.

    Raises
    ------
    ValueError
        If the component rejects its formula, modifiers or options.
    """
    groups = {
        group
        for predictor in (formula, *distributional.values())
        for group in predictor.group_levels
    }
    covariances = {
        group: covariance
        for group, covariance in lens._group_covariances.items()
        if group in groups
    }

    def build(priors: Mapping[str, str]) -> tuple[FormulaToQVRModule, Module]:
        sub = type(lens)(
            family,
            fixed_prior=lens._fixed_prior,
            random_scale_prior=lens._random_scale_prior,
            correlation_prior=lens._correlation_prior,
            user_priors=dict(priors) or None,
            reparameterize=lens._reparameterize,
            default_priors=lens._default_priors,
            center=lens._center,
            decomp=lens._decomp,
            binomial_trials=binomial_trials
            if family.trials_parameter is not None
            else None,
            distributional=distributional,
            fixed_parameters=fixed_parameters,
            group_covariances=covariances or None,
        )
        module, _ = sub._forward(
            formula,
            supplied_groups=supplied,
            supplied_sites=supplied_sites,
            check=False,
            rows=lens._rows.within(namespace.prefix)
            if lens._rows is not None
            else None,
        )
        return sub, module

    sub, module = build(dict(default_priors or {}))
    # The components of a response share the levels of a grouping factor.
    namespace = namespace.with_(
        shared=namespace.shared | level_names(sub.group_laws) | set(supplied_sites)
    )
    priors = dict(default_priors or {}) | _local_priors(
        lens, namespace, _sample_sites(module)
    )
    if priors != dict(default_priors or {}):
        sub, module = build(priors)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    objects = {
        name: namespace.name(name)
        for item in module.statements
        if isinstance(item, ObjectDecl)
        for name in item.names
    }
    for group in groups:
        objects[_qvr_name(group)] = _qvr_name(group)
    variables = set(_bound_names(program.draws)) | {
        parameter.name for parameter in program.value_parameters
    }
    variables.update(
        value for step in program.draws for value in _step_value_names(step)
    )
    variables.update(
        law.support_site for law in sub.group_laws if law.support_site is not None
    )
    names = {value: namespace.name(value) for value in variables}
    for group in groups:
        names[f"{_qvr_name(group)}_idx"] = f"{_qvr_name(group)}_idx"
    # Two components may draw from one law over different coordinates.
    term_laws = sub.term_laws
    morphisms = (
        {}
        if term_laws is None
        else {
            record.name: namespace.prefix("") + record.name
            for record in term_laws.definitions
        }
    )
    bindings = TemplateBindings(objects=objects, values={}, morphisms=morphisms)
    renamed_objects = tuple(
        item.with_(
            names=tuple(objects[value] for value in item.names),
            init=item.init.with_(
                expr=substitute_program_object(
                    item.init.expr,
                    {key: TypeName(name=value) for key, value in objects.items()},
                )
            ),
        )
        for item in module.statements
        if isinstance(item, ObjectDecl) and isinstance(item.init, TypeFromExpr)
    )
    steps = [_rename_step(step, bindings, names) for step in program.draws]
    observed = sub._observation
    if observed is None:
        raise ValueError("a mixture component emitted no observation")
    observation = observed.renamed(names, morphisms, objects)
    # The component's own observation is the last step it writes; the
    # mixture scores the component law instead.
    written = steps.pop()
    if written != observation.step():
        raise ValueError("a mixture component wrote a step after its observation")
    observation = observation.with_events(()).with_(weights=None)
    parameters = tuple(
        parameter.with_(
            name=names.get(parameter.name, parameter.name),
            type_expr=_indices(parameter.type_expr, objects),
        )
        for parameter in program.value_parameters
    )
    targets = tuple(
        target.with_(
            site=names.get(target.site, namespace.name(target.site)),
            axes=tuple(objects.get(axis, axis) for axis in target.axes),
            class_=f"{target.class_}{namespace.index}"
            if target.class_ in namespace.auxiliaries
            else target.class_,
            dpar=namespace.dpar(target.dpar),
        )
        for target in sub.prior_targets
    )
    laws = tuple(
        renamed_law(
            law, names, objects, namespace.prefix, namespace.prefix("") + law.identity
        )
        for law in sub.group_laws
    )
    correlation_ids = frozenset(
        term.correlation_id
        for predictor in (formula, *distributional.values())
        for term in predictor.random_terms
        if term.correlation_id is not None and term.shared
    )
    return _SubEmission(
        objects=renamed_objects,
        steps=tuple(steps),
        observation=observation,
        parameters=parameters,
        prior_targets=targets,
        group_laws=laws,
        correlation_ids=correlation_ids,
        families=term_laws.renamed(morphisms) if term_laws is not None else None,
        names=ProgramNames(
            inputs={
                ColumnName(
                    prefix=namespace.prefix(key.partition(":")[0]),
                    column=key.partition(":")[2],
                ).key: namespace.name(value)
                for key, value in sub._names.inputs.items()
            },
            bindings={
                f"{_namespaced_owner(namespace, owner)}:{name}": namespace.name(value)
                for key, value in sub._names.bindings.items()
                for owner, _, name in (key.rpartition(":"),)
            },
        ),
    )


def _namespaced_owner(namespace: _Namespace, owner: str) -> str:
    """Rename the owner of a private binding into a mixture parameter.

    Parameters
    ----------
    namespace : _Namespace
        Names of the mixture parameter.
    owner : str
        Predictor namespace, followed by the parts of a group coefficient.

    Returns
    -------
    str
        The owner in the mixture program.
    """
    prefix, separator, rest = owner.partition("|")
    return namespace.prefix(prefix) + separator + rest


def _component_formula(main: Formula, location: Formula) -> Formula:
    """Attach the shared response and modifiers to a component location design.

    Parameters
    ----------
    main : Formula
        Main formula carrying the response and its modifiers.
    location : Formula
        Evaluated ``mu<k>`` design.

    Returns
    -------
    Formula
        The component's location formula over the shared response.

    Raises
    ------
    ValueError
        If the design carries latent-dependent inputs.
    """
    if any(isinstance(column, SymbolicColumn) for column in location.fixed_columns):
        raise ValueError("mixture components do not support latent-dependent inputs")
    return location.with_(
        response_name=main.response_name,
        response_values=main.response_values,
        response_layout=main.response_layout,
        modifiers=main.modifiers,
    )


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


def row_count(lens: FormulaToQVRModule, formula: Formula) -> int:
    """Count the rows the program covers.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The mixture lens.
    formula : Formula
        Main formula.

    Returns
    -------
    int
        Fresh rows of the request, or the training rows.
    """
    fresh = lens._rows.size if lens._rows is not None else None
    return int(formula.response_values.shape[0]) if fresh is None else fresh


def _row_inputs(
    lens: FormulaToQVRModule,
    formula: Formula,
    steps: Sequence[ProgramStep],
    declared: Mapping[str, QiecValueParameter],
) -> tuple[QiecValueParameter, ...]:
    """Declare the row-aligned data inputs read inside mixture components.

    Inputs read only as component arguments carry no plate context for type
    inference, so every row-aligned numeric input is declared explicitly as a
    ``Resp``-indexed tensor.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The mixture lens, whose bound inputs are listed.
    formula : Formula
        Main formula.
    steps : Sequence[ProgramStep]
        The composed program steps.
    declared : Mapping[str, QiecValueParameter]
        Runtime parameters already declared by the components.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        One ``Tensor[Real; Resp]`` or ``Tensor[Int; Resp]`` per free input.
    """
    rows = int(formula.response_values.shape[0])
    bound = set(_bound_names(tuple(steps))) | set(declared)
    free = {name for step in steps for name in _step_value_names(step)} - bound
    inputs = lens.fixed_column_observations(formula)
    result = []
    for name in sorted(free):
        value = inputs.get(name)
        if value is None or value.ndim != 1 or value.shape[0] != rows:
            continue
        element = "Real" if value.is_floating_point() else "Int"
        result.append(
            QiecValueParameter(
                name=name,
                type_expr=QiecTypeApplication(
                    constructor="Tensor",
                    static_arguments=(QiecTypeName(name=element),),
                    indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
                ),
            )
        )
    return tuple(result)


def _shared_blocks(
    lens: FormulaToQVRModule,
    formula: Formula,
    statements: list[Statement],
    steps: list[ProgramStep],
    targets: list[PriorTarget],
    laws: list[GroupBlockLaw],
    external: Mapping[tuple[str, str, str], LetExprNode],
) -> dict[tuple[str, str, str], LetExprNode]:
    """Emit one group block for each identifier shared by mixture parameters.

    brms correlates group effects with the same identifier, ``(1 | p | g)``,
    across every distributional parameter, including different mixture
    components. Such blocks are emitted once by the ordinary correlation
    pass over the mixture parameters and supplied to the component
    emissions, which then reuse them.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The mixture lens.
    formula : Formula
        Main formula.
    statements, steps : list
        Declarations and program steps receiving the blocks.
    targets : list[PriorTarget]
        Prior catalogue receiving the blocks' scale and correlation sites.
    laws : list[GroupBlockLaw]
        Group-block laws receiving the blocks.
    external : Mapping
        Coefficients a joint multivariate program already supplies; their
        terms are not emitted again.

    Returns
    -------
    dict[tuple[str, str, str], LetExprNode]
        Coefficients keyed by mixture predictor prefix, group, and slope.
    """
    owners: dict[str, set[str]] = {}

    def pending(name: str, term: RandomTerm) -> bool:
        return (
            term.shared
            and term.correlation_id is not None
            and (f"dpar_{name}_", term.group, term.slope) not in external
        )

    for name, predictor in lens._distributional.items():
        for term in predictor.random_terms:
            if pending(name, term) and term.correlation_id is not None:
                owners.setdefault(term.correlation_id, set()).add(name)
    cross = {identifier for identifier, names in owners.items() if len(names) > 1}
    if not cross:
        return {}
    helper = copy(lens)
    helper._emitted_groups = []
    helper._distributional = {
        name: predictor.with_(
            fixed_columns=(),
            design_terms=(),
            offsets=(),
            monotonic_terms=(),
            random_terms=tuple(
                term
                for term in predictor.random_terms
                if pending(name, term) and term.correlation_id in cross
            ),
        )
        for name, predictor in lens._distributional.items()
    }
    bare = formula.with_(
        fixed_columns=(),
        design_terms=(),
        offsets=(),
        monotonic_terms=(),
        random_terms=(),
    )
    block_steps: list[ProgramStep] = []
    supplied = helper._append_correlations(
        helper._plan_groups(bare, ()), statements, block_steps, {}
    )
    steps.extend(block_steps)
    laws.extend(helper.group_laws)
    module = Module(
        statements=(
            *statements,
            ProgramDecl(
                name="model",
                params=None,
                value_parameters=(),
                domain=TypeName(name="Resp"),
                codomain=TypeName(name="Resp"),
                draws=tuple(block_steps),
                return_vars=(),
            ),
        )
    )
    targets.extend(
        target
        for target in build_prior_targets(
            module,
            bare,
            lens._family,
            helper._plan(bare, helper._group_plan, None),
            helper.group_laws,
            draw=lens._parse_prior,
        )
        if target.group is not None
    )
    return dict(supplied)


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


def emit_mixture(
    lens: FormulaToQVRModule,
    formula: Formula,
    supplied_groups: Mapping[tuple[str, str, str], LetExprNode] | None = None,
) -> MixtureEmission:
    """Emit a finite mixture by composing ordinary component emissions.

    Every component and every predicted mixing proportion is compiled by the
    ordinary single-family emitter and renamed into its ``dpar_<name><k>_``
    namespace. The observation is ``Mixture(weights, components)`` with
    per-row or shared weights. Scoring costs O(n K) beyond the component
    predictors.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Mixture lens with its formulas, fixed values, options and priors.
    formula : Formula
        Main formula with the response and modifiers.
    supplied_groups : Mapping or None
        Group coefficients supplied by a joint multivariate program, keyed by
        mixture predictor prefix, group, and slope.

    Returns
    -------
    MixtureEmission
        Module, prior catalogue and group-block laws.

    Raises
    ------
    ValueError
        If a location formula is missing, a component rejects its formula, a
        group-level identifier is shared across mixture parameters, or
        ordering is requested without location intercepts.
    """
    family = lens._family
    spec = family.mixture
    assert spec is not None
    count = len(spec.components)
    trials = lens._binomial_trials
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
    shared = _shared_names(formula, trials)
    statements: list[Statement] = [
        ObjectDecl(
            names=("Resp",),
            init=TypeFromExpr(
                expr=DiscreteConstructor(
                    constructor="FinSet",
                    args=(str(row_count(lens, formula)),),
                )
            ),
        )
    ]
    groups = {
        _qvr_name(group): len(levels)
        for _, _, predictor in lens._predictor_items(formula)
        for group, levels in predictor.group_levels.items()
    }
    for name, size in groups.items():
        lens._append_finite_object(statements, name, size)
    lens._append_finite_object(statements, COMPONENT_AXIS, count)
    steps: list[ProgramStep] = []
    parameters: dict[str, QiecValueParameter] = {}
    targets: list[PriorTarget] = []
    laws: list[GroupBlockLaw] = []
    term_laws: list[FamilyCatalog] = []
    inputs: dict[str, str] = {}
    bindings: dict[str, str] = {}
    ordinal = spec.components[0].ordinal is not None
    # brms: an ordered ordinal mixture shares one threshold vector, ordered
    # when any component has ordered thresholds.
    fixed_thresholds = ordinal and spec.order == "mu"
    ordered = spec.order == "mu" and not ordinal
    threshold_source = next(
        (
            index
            for index, component in enumerate(spec.components, start=1)
            if component.ordinal is not None and component.ordinal.kind == "cumulative"
        ),
        1,
    )
    equal = dict(spec.equal)

    def absorb(emission: _SubEmission) -> None:
        for item in emission.objects:
            assert isinstance(item, ObjectDecl)
            # The components declare the axes they share once.
            if any(
                isinstance(declared, ObjectDecl) and declared.names == item.names
                for declared in statements
            ):
                continue
            statements.append(item)
        for parameter in emission.parameters:
            parameters.setdefault(parameter.name, parameter)
        targets.extend(emission.prior_targets)
        laws.extend(emission.group_laws)
        inputs.update(emission.names.inputs)
        bindings.update(emission.names.bindings)
        if emission.families is not None:
            term_laws.append(emission.families)

    external = dict(supplied_groups or {})
    supplied = _shared_blocks(lens, formula, statements, steps, targets, laws, external)
    supplied.update(external)
    shared = (
        shared
        | {name for step in steps for name in _bound_names((step,))}
        | {
            name
            for value in external.values()
            for name in _step_value_names(LetStep(name="supplied", value=value))
        }
    )

    def drawn() -> frozenset[str]:
        return lens._drawn_sites | {
            law.level.mixing.df_site for law in laws if law.level.mixing is not None
        }

    def local(prefixes: Mapping[str, str]) -> dict[tuple[str, str, str], LetExprNode]:
        return {
            (prefixes[prefix], group, slope): value
            for (prefix, group, slope), value in supplied.items()
            if prefix in prefixes
        }

    if ordered:
        lens._append_finite_object(statements, SPACING_AXIS, count - 1)
        steps.append(
            lens._prior_draw(
                SPACING_SITE, SPACING_PRIOR, index=TypeName(name=SPACING_AXIS)
            )
        )
        targets.append(
            PriorTarget(
                site=SPACING_SITE,
                class_=SPACING_SITE,
                axes=(SPACING_AXIS,),
                shape=(count - 1,),
                lower=0.0,
            )
        )
    first = "dpar_mu1_intercept"
    emitted: dict[int, list[ProgramStep]] = {}
    observations: list[Observation] = []
    etas: list[LetExprNode] = []
    means: list[LetExprNode] = []
    for index, component in enumerate(spec.components, start=1):
        location = lens._distributional.get(f"mu{index}")
        if location is None:
            raise ValueError(
                f"mixture component {index} requires its location formula 'mu{index}'"
            )
        auxiliaries = tuple(aux.name for aux in component.aux_params)
        namespace = _Namespace(
            location="mu", index=index, auxiliaries=auxiliaries, shared=shared
        )
        local_formulas = {
            aux: lens._distributional[f"{aux}{index}"]
            for aux in auxiliaries
            if f"{aux}{index}" in lens._distributional
        }
        local_fixed = {
            aux: lens._fixed_parameters[f"{aux}{index}"]
            for aux in auxiliaries
            if f"{aux}{index}" in lens._fixed_parameters
        }
        component_formula = _component_formula(formula, location)
        if ordered and not any(
            column.is_intercept for column in location.fixed_columns
        ):
            raise ValueError(
                "identifying mixture components via ordering requires population-level "
                "intercepts to be present; try order = 'none'"
            )
        emission = _sub_emission(
            lens,
            component,
            component_formula,
            namespace,
            distributional=local_formulas,
            fixed_parameters=local_fixed,
            binomial_trials=trials,
            supplied_sites=drawn(),
            supplied=local(
                {f"dpar_mu{index}_": ""}
                | {f"dpar_{aux}{index}_": f"dpar_{aux}_" for aux in auxiliaries}
            ),
        )
        absorb(emission)
        component_steps = list(emission.steps)
        sampled = {
            step.vars[0] for step in component_steps if isinstance(step, SampleStep)
        }
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
        if ordered and intercept is not None:
            if index == 1:
                first = intercept
            else:
                # Cumulative positive gaps, written as an affine sum so that
                # derived coefficient draws evaluate it like the program does.
                gaps = [
                    _index(
                        SPACING_SITE, LetExprLiteral(value=float(gap), integral=True)
                    )
                    for gap in range(index - 1)
                ]
                component_steps = [
                    _let(intercept, _add(_var(first), *gaps))
                    if isinstance(step, SampleStep) and step.vars == (intercept,)
                    else step
                    for step in component_steps
                ]
                targets[:] = [target for target in targets if target.site != intercept]
        if fixed_thresholds:
            intercepts = {
                any(column.is_intercept for column in predictor.fixed_columns)
                for name, predictor in lens._distributional.items()
                if _dpar_class(name) == "mu"
            }
            if len(intercepts) > 1:
                raise ValueError(
                    "mixture: thresholds shared across ordinal components require "
                    "every location formula to keep or drop its intercept alike"
                )
            if index != threshold_source:
                dropped = {
                    step.vars[0]
                    for step in component_steps
                    if isinstance(step, SampleStep)
                    and (
                        step.vars[0].startswith(f"dpar_mu{index}_cutpoint")
                        or step.vars[0] == intercept
                        or step.vars[0] == f"dpar_mu{index}_{LOCATION_SITE}"
                    )
                }
                component_steps = _shared_thresholds(
                    component_steps,
                    index,
                    threshold_source,
                    lens._category_count(formula, family=component.name) - 1,
                    intercept if intercepts == {True} else None,
                    grouped=formula.modifiers.threshold_groups is not None,
                )
                targets[:] = [
                    target for target in targets if target.site not in dropped
                ]
        for target_name, source in equal.items():
            if _dpar_id(target_name) != index:
                continue
            component_steps = [
                _let(target_name, _var(source))
                if isinstance(step, SampleStep) and step.vars == (target_name,)
                else step
                for step in component_steps
            ]
            targets[:] = [target for target in targets if target.site != target_name]
        emitted[index] = component_steps
        assert emission.observation is not None
        observations.append(emission.observation)
        etas.append(_var(f"dpar_mu{index}_eta"))
        bound = {step.name for step in component_steps if isinstance(step, LetStep)}
        means.append(
            _var(f"mu{index}")
            if f"mu{index}" in bound
            else _call("exp", _var(f"dpar_mu{index}_eta"))
        )
    # The component whose thresholds are shared precedes its readers.
    for index in sorted(
        emitted, key=lambda item: not fixed_thresholds or item != threshold_source
    ):
        steps.extend(emitted[index])
    for target_name, source in equal.items():
        if not any(
            isinstance(step, (SampleStep, LetStep))
            and (step.vars[0] if isinstance(step, SampleStep) else step.name) == source
            for step in steps
        ):
            raise ValueError(
                f"mixture: {target_name!r} is equated to {source!r}, which is not a "
                "parameter of this model"
            )
    _row_stack("eta", etas, steps)
    _row_stack("mu", means, steps)
    predicted = {
        name: value
        for name, value in lens._distributional.items()
        if _dpar_class(name) == "theta"
    }
    fixed = {
        name: value
        for name, value in lens._fixed_parameters.items()
        if _dpar_class(name) == "theta"
    }
    if predicted:
        logits: list[LetExprNode] = []
        for index in range(1, count + 1):
            predictor = predicted.get(f"theta{index}")
            if predictor is None:
                logits.append(LetExprLiteral(value=0.0))
                continue
            namespace = _Namespace(location="theta", index=index, shared=shared)
            emission = _sub_emission(
                lens,
                families["gaussian"],
                # The proportion predictor reads no response categories.
                predictor.with_(
                    response_layout=predictor.response_layout.with_(category_levels=())
                ),
                namespace,
                distributional={},
                fixed_parameters={"sigma": 1.0},
                binomial_trials=None,
                supplied_sites=drawn(),
                supplied=local({f"dpar_theta{index}_": ""}),
            )
            absorb(emission)
            steps.extend(emission.steps)
            logits.append(_var(f"dpar_theta{index}_eta"))
        _row_stack("mixture_logits", logits, steps)
        steps.append(_let(WEIGHTS, _call("softmax", _var("mixture_logits"))))
    elif fixed:
        total = sum(float(value) for value in fixed.values())
        steps.append(
            _let(
                WEIGHTS,
                _factor(
                    (("component", COMPONENT_AXIS),),
                    cases=tuple(
                        (
                            index - 1,
                            LetExprLiteral(value=float(fixed[f"theta{index}"]) / total),
                        )
                        for index in range(1, count + 1)
                    ),
                ),
            )
        )
    else:
        concentration = ", ".join("1.0" for _ in range(count))
        steps.append(
            lens._prior_draw("theta", f"Dirichlet(concentration=[{concentration}])")
        )
        steps.append(_let(WEIGHTS, _var("theta")))
        targets.append(PriorTarget(site="theta", class_="theta", event_rank=1))
    assert lens._family.mixture is not None
    native = {family.mixture_law for family in lens._family.mixture.components}
    mixture_law = native.pop() if len(native) == 1 else None
    site = _qvr_name(formula.response_name)
    if mixture_law is not None:
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
            kernel=mixture_law,
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
    for modifier in response_modifiers(formula.modifiers, None):
        mixed = modifier.observe(mixed)
    lens._observation = mixed
    steps.append(mixed.step())
    lens._names = ProgramNames(inputs=inputs, bindings=bindings)
    check_names(laws)
    for parameter in declare_inputs(law_inputs(laws)):
        parameters.setdefault(parameter.name, parameter)
    statements.append(
        ProgramDecl(
            name="model",
            params=None,
            value_parameters=tuple(parameters.values())
            + _row_inputs(lens, formula, steps, parameters),
            domain=TypeName(name="Resp"),
            codomain=TypeName(name="Resp"),
            draws=tuple(steps),
            return_vars=(_qvr_name(formula.response_name),),
        )
    )
    statements.append(ExportDecl(expr=ExprIdent(name="model")))
    return MixtureEmission(
        module=Module(statements=tuple(statements)),
        prior_targets=tuple(targets),
        group_laws=tuple(laws),
        families=reduce(FamilyCatalog.with_catalog, term_laws) if term_laws else None,
        names=lens._names,
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
    "MixtureEmission",
    "emit_mixture",
    "expand_mixture",
    "mixture",
    "mixture_component_count",
    "mixture_distributional",
    "validate_mixture_options",
]
