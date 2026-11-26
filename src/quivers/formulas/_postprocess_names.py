"""brms parameter names for single-response formula fits.

brms names its parameters by class: population coefficients ``b_<coef>``
(``b_<dpar>_<coef>`` for a distributional or nonlinear parameter), ordinal
thresholds ``b_Intercept[k]``, unregressed auxiliary parameters by their
name (``sigma``), group standard deviations ``sd_<group>__<coef>``, group
correlations ``cor_<group>__<coef>__<coef>``, and per-level group
coefficients ``r_<group>[<level>,<coef>]``. Coefficient labels pass through
brms's character renaming, so the design column ``f[B]`` is ``fB``. Native
sites without a brms counterpart keep their QVR site names.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING

import torch

from quivers.dsl.ast_nodes import Module
from quivers.formulas._brms_priors import derived_site_draws
from quivers.formulas._formula_options import display_label
from quivers.formulas._priors import PriorTarget
from quivers.formulas._smooth import smooth_states
from quivers.formulas._threshold_groups import LOCATION_SITE
from quivers.formulas.formula import Formula
from quivers.formulas._group_law import GroupMemberLaw
from quivers.formulas._name_scope import (
    coefficient_names,
    NameScope,
    level_coefficients,
    predictor_variables,
)
from quivers.formulas._variable_catalogue import (
    brms_order,
    brms_rename,
    checked_variables,
    class_variables,
    native_variables,
    scalar_coordinates,
    shrinkage_blocks,
    shrinkage_scale_site,
    shrinkage_variables,
    threshold_variables,
    CoefficientName,
    time_label,
    group_variables,
)
from quivers.inference._posterior import draw_count

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit

_INDEXED = re.compile(r"^(?P<stem>.*\()(?P<args>.*)\)\[(?P<index>[0-9]+)\]$")


def _r_label(label: str, *, categoric: bool = False) -> str:
    """Rewrite a formulae column label to R's ``model.matrix`` label.

    Parameters
    ----------
    label : str
        formulae label such as ``f[q]``, ``x:f[q]`` or ``poly(z, 2)[0]``.
    categoric : bool
        Whether the label belongs to a categorical term, whose bracket holds
        a level, as in ``factor(k)[2]``, and not a column position.

    Returns
    -------
    str
        R's label before brms renaming: factor levels follow the term
        (``fq``, ``factor(k)2``), the columns of a matrix-valued call are
        numbered from one (``poly(z, 2)1``), and a power inside ``I()``,
        which the formula layer evaluates as ``**``, is written ``^`` as in
        the R formula.
    """
    # R's ^ in I(): brms labels I(x^2) as IxE2.
    label = label.replace(" ** ", "^").replace("**", "^")
    match = _INDEXED.match(label)
    if match is not None and not categoric:
        return f"{match['stem']}{match['args']}){int(match['index']) + 1}"
    return label


def coefficient_labels(fit: BayesianFit) -> dict[tuple[str, str], str]:
    """Map every fixed design column to its brms coefficient label.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(parameter, column name)`` to the renamed R column label, using
        the evaluated formulae labels of each term; columns that formulae
        did not evaluate keep their renamed column name.
    """
    plan = fit.prediction_plan
    result: dict[tuple[str, str], str] = {}
    for parameter, _, predictor in fit._predictor_items():
        design = plan.designs.get(parameter) if plan is not None else None
        terms = (
            design.common.terms
            if design is not None and design.common is not None
            else {}
        )
        seen: dict[str, int] = {}
        for column in predictor.fixed_columns:
            position = seen.get(column.term, 0)
            seen[column.term] = position + 1
            term = terms.get(column.term)
            labels = getattr(term, "labels", None)
            label = (
                _r_label(
                    labels[position],
                    categoric=getattr(term, "kind", None) == "categoric",
                )
                if labels is not None and position < len(labels)
                else column.name
            )
            # brms's 0 + Intercept column is the coefficient "Intercept".
            result[(parameter, column.name)] = brms_rename(display_label(label))
    return result


def _infix(parameter: str) -> str:
    """Return the name infix of a predicted parameter.

    Parameters
    ----------
    parameter : str
        ``mu``, a distributional parameter, or a nonlinear parameter.

    Returns
    -------
    str
        Empty for ``mu``, otherwise ``"<parameter>_"``.
    """
    return "" if parameter == "mu" else f"{parameter}_"


def _ordinal_thresholds(fit: BayesianFit) -> bool:
    """Report whether thresholds have brms's ``b_Intercept[k]`` form.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model.

    Returns
    -------
    bool
        True for a cumulative family with one threshold vector shared by all
        rows: no grouped thresholds and no category-specific effects.
    """
    if fit.family.ordinal is None:
        return False
    # brms thres(n, gr): every ordinal family reports its group thresholds.
    if fit.formula.modifiers.threshold_groups is not None:
        return True
    if fit.family.ordinal.kind != "cumulative" or fit.thresholds_by is not None:
        return False
    return not fit.formula.category_specific


def coordinate_labels(fit: BayesianFit, target: PriorTarget) -> tuple[str, ...]:
    """Return the coordinate labels of a correlation-class target.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model, or the response view of a joint fit.
    target : PriorTarget
        A target whose site is a Cholesky factor.

    Returns
    -------
    tuple[str, ...]
        The retained component labels of the target, the free response
        categories of a logistic-normal correlation, or for ``unstr()`` the
        time levels of the autocorrelation term; empty when none is retained.
    """
    if target.component_labels:
        return tuple(str(label) for label in target.component_labels)
    if target.class_ == "rescor":
        # A logistic-normal response correlates its free categories.
        return tuple(brms_rename(str(label)) for label in fit.coefficient_labels)
    if fit.residual is None:
        return ()
    return tuple(time_label(level) for level in fit.residual.time_levels)


def shrinkage_scales(
    module: Module, draws: Mapping[str, torch.Tensor], names: Mapping[str, str]
) -> dict[str, dict[str, torch.Tensor]]:
    """Evaluate the prior scale of every coefficient under a shrinkage prior.

    A horseshoe or R2D2 block binds each member coefficient's prior standard
    deviation as a deterministic expression of the block's sites, the
    quantity brms reports as ``sdb``. The expressions are evaluated draw by
    draw from the emitted program, at cost O(S k) for S draws and k members.

    Parameters
    ----------
    module : Module
        The fitted module.
    draws : Mapping[str, torch.Tensor]
        Constrained draws.
    names : Mapping[str, str]
        Coefficient site to its brms name without the class prefix.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Per shrinkage block, coefficient name to scale draws.
    """
    result: dict[str, dict[str, torch.Tensor]] = {}
    for block in shrinkage_blocks(draws):
        bindings = {
            shrinkage_scale_site(block, site): name for site, name in names.items()
        }
        derived = derived_site_draws(module, draws, list(bindings))
        result[block] = {
            name: derived[binding]
            for binding, name in bindings.items()
            if binding in derived and binding not in draws
        }
    return result


_MIXTURE_LOCATION = re.compile(r"^mu([0-9]+)$")


def distributional_classes(fit: BayesianFit) -> tuple[str, ...]:
    """Return the family's distributional parameter names in brms's order.

    A mixture lists the parameters of each component in turn and then the
    mixing proportions. Any other family lists the location and the
    auxiliary parameters, followed for a family with per-category
    parameters by each of them for every category.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model.

    Returns
    -------
    tuple[str, ...]
        Parameter names as brms's ``valid_dpars`` lists them.
    """
    family = fit.family
    if family.mixture is not None:
        components = family.mixture.components
        return (
            *(
                f"{name}{index}"
                for index, component in enumerate(components, start=1)
                for name in ("mu", *(aux.name for aux in component.aux_params))
            ),
            *(f"theta{index}" for index in range(1, len(components) + 1)),
        )
    base = ("mu", *(aux.name for aux in family.aux_params))
    categories = tuple(brms_rename(str(label)) for label in fit.coefficient_labels)
    return (*base, *(f"{name}{category}" for name in base for category in categories))


def _location_first(fit: BayesianFit) -> list[tuple[str, str, Formula]]:
    """List the fit's predictors with the location predictors first.

    brms moves the location predictors (``mu``, one per mixture component,
    or one per response category) to the front in the family's order and
    keeps the other predictors in their authored order.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model.

    Returns
    -------
    list[tuple[str, str, Formula]]
        Parameter labels, native prefixes and predictors.
    """
    categories = [f"mu{brms_rename(str(label))}" for label in fit.coefficient_labels]

    def rank(item: tuple[str, str, Formula]) -> tuple[int, int]:
        parameter = item[0]
        if parameter == "mu":
            return (0, 0)
        component = _MIXTURE_LOCATION.match(parameter)
        if fit.family.mixture is not None and component is not None:
            return (0, int(component.group(1)))
        if parameter in categories:
            return (0, categories.index(parameter))
        return (1, 0)

    return sorted(fit._predictor_items(), key=rank)


def name_scope(
    fit: BayesianFit,
    parameter: str,
    prefix: str,
    predictor: Formula,
    labels: Mapping[tuple[str, str], str],
    *,
    response: str = "",
    special: Mapping[str, str] | None = None,
) -> NameScope:
    """Build the names of one predictor of a fit.

    Parameters
    ----------
    fit : BayesianFit
        Fitted model, or the response view of a joint fit.
    parameter : str
        Predicted parameter.
    prefix : str
        Its native namespace.
    predictor : Formula
        Evaluated predictor.
    labels : Mapping[tuple[str, str], str]
        Design labels of the fit (`coefficient_labels`).
    response : str
        Response part of the names; empty for a single response.
    special : Mapping[str, str] or None
        Labels of special-effects columns, by column name.

    Returns
    -------
    NameScope
        The names of the predictor.
    """
    plan = fit.prediction_plan
    return NameScope(
        parameter=parameter,
        role="nlpar" if prefix.startswith("nlpar_") else "dpar",
        response=response,
        formula=predictor,
        labels={
            column: label
            for (owner, column), label in labels.items()
            if owner == parameter
        },
        states=smooth_states(plan.designs.get(parameter) if plan is not None else None),
        special=dict(special or {}),
    )


def single_variables(
    fit: BayesianFit, *, num_samples: int | None, seed: int
) -> dict[str, torch.Tensor]:
    """Return a single-response fit's draws under brms parameter names.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula model with its retained emission metadata.
    num_samples : int or None
        Draws per chain under the convention of `BayesianFit.draws`.
    seed : int
        Seed for variational draws.

    Returns
    -------
    dict[str, torch.Tensor]
        brms-named draws, each one scalar on chain and draw axes, in the
        order of brms's ``variables()`` (`brms_order`), followed by native
        sites without a brms counterpart.

    Raises
    ------
    ValueError
        If the fit retains no emission metadata.

    Notes
    -----
    Cost is one pass over the retained draws, plus one replay per draw
    for ordinal thresholds.
    """
    plan = fit.prediction_plan
    if plan is None:
        raise ValueError("brms parameter names require retained emission metadata")
    draws = fit.draws(num_samples=num_samples, seed=seed)
    shape = draw_count(fit.posterior, num_samples)
    targets: tuple[PriorTarget, ...] = plan.prior_targets
    consumed: set[str] = set()
    population: dict[str, torch.Tensor] = {}
    intercepts: set[str] = set()
    thresholds = _ordinal_thresholds(fit)
    labels = coefficient_labels(fit)
    special: dict[str, torch.Tensor] = {}
    latent: dict[str, torch.Tensor] = {}
    shrunk: dict[str, str] = {}
    items = _location_first(fit)
    planned = {predictor.prefix: predictor for predictor in plan.predictors}
    scopes = {
        prefix: name_scope(fit, parameter, prefix, predictor, labels)
        for parameter, prefix, predictor in items
    }
    for parameter, prefix, _ in items:
        named = predictor_variables(
            planned[prefix].terms,
            scopes[prefix].with_(intercept=not (thresholds and parameter == "mu")),
            draws,
        )
        population |= named.population
        special |= named.special
        latent |= named.latent
        intercepts |= named.intercepts
        shrunk |= named.sites
        consumed |= named.read
    # brms lists every intercept before the other population coefficients.
    population = {
        name: value for name, value in population.items() if name in intercepts
    } | {name: value for name, value in population.items() if name not in intercepts}
    if thresholds:
        intercept = draws.get("intercept")
        by_row = fit.posterior_thresholds(
            re_formula="~0", num_samples=num_samples, seed=seed
        )
        if intercept is not None:
            by_row = by_row - intercept[..., None, None]
        # brms thres(n, gr): each group's own thresholds.
        population = (
            threshold_variables(by_row, groups=fit.formula.modifiers.threshold_groups)
            | population
        )
        consumed.update(
            target.site
            for target in targets
            if target.class_.startswith("cutpoint")
            or target.class_ in {"Intercept", LOCATION_SITE}
        )

    def member_names(member: GroupMemberLaw) -> tuple[CoefficientName, ...]:
        """Name the components of one group-level coefficient."""
        if member.prefix not in scopes:
            return ()
        return coefficient_names(
            scopes[member.prefix],
            planned[member.prefix].output,
            member.slope,
            member.width if member.axis is not None else None,
        )

    auxiliary: dict[str, torch.Tensor] = {}
    block_correlations = {
        site for law in plan.group_laws for site in law.correlation.sites
    }
    for target in targets:
        if target.site not in draws or target.site in consumed:
            continue
        if target.whole is not None:
            # A component drawn on its own is part of its site's variables.
            consumed.update(target.sites)
            continue
        parameter = target.nlpar or target.dpar or "mu"
        value = draws[target.site]
        if target.class_ in {"b", "Intercept"}:
            # The reported coefficients of transformed coordinates.
            consumed.add(target.site)
        elif target.class_ == "sd" and target.group is not None:
            consumed.add(target.site)
        elif target.class_ == "cor" and target.site in block_correlations:
            continue
        elif (
            target.group is None
            and target.coef is None
            and target.dpar == target.class_
            and target.site == target.class_
        ):
            if value.ndim == 3 and value.shape[-1] == len(fit.coefficient_labels):
                # brms names a per-category family parameter <dpar><category>.
                for index, label in enumerate(fit.coefficient_labels):
                    auxiliary[f"{target.class_}{brms_rename(str(label))}"] = value[
                        ..., index
                    ]
            else:
                auxiliary |= scalar_coordinates(target.class_, value)
            consumed.add(target.site)
        else:
            special |= class_variables(
                target,
                value,
                infix=_infix(parameter),
                labels=coordinate_labels(fit, target),
            )
            consumed.add(target.site)
    order = {target.site: index for index, target in enumerate(targets)}
    scales, correlations = group_variables(
        sorted(
            plan.group_laws,
            key=lambda law: min(
                order.get(member.scale.site, len(order)) for member in law.members
            ),
        ),
        member_names,
        draws,
    )
    for law in plan.group_laws:
        consumed.update(
            site
            for member in law.members
            for site in (member.raw_site, member.scale.site, member.mean_site)
            if site is not None
        )
        consumed.update(law.correlation.sites)
    named, read = shrinkage_variables(
        draws, consumed, shrinkage_scales(plan.module, draws, shrunk)
    )
    special |= named
    consumed |= read
    group_coefficients: dict[str, torch.Tensor] = {}
    group_levels = fit.group_levels
    for parameter, prefix, predictor in items:
        if not predictor.random_terms:
            continue
        varying = fit._random_effects(draws, shape, predictor, prefix)
        for item in level_coefficients(varying, scopes[prefix], planned[prefix].output):
            for index, level in enumerate(group_levels[item.group]):
                group_coefficients[item.variable(level)] = item.values[:, :, index]
    return checked_variables(
        brms_order(
            population
            | auxiliary
            | scales
            | correlations
            | special
            | group_coefficients
            | latent
            | native_variables(draws, consumed),
            distributional_classes(fit),
        ),
        shape,
    )


def named_draw_classes(names: Mapping[str, torch.Tensor]) -> dict[str, tuple[str, ...]]:
    """Group brms parameter names into the sections of brms's ``summary``.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws.

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``"fixed"``, ``"random:<group>"``, ``"splines"`` (smoothing
        standard deviations ``sds_``), ``"rescor"``, ``"levels"`` and
        ``"other"`` name lists in their input order.
    """
    sections: dict[str, list[str]] = {}
    for name in names:
        if name.startswith(
            ("b_", "bs_", "bcs_", "bsp_", "bmo_", "bme_", "bmi_", "bm_")
        ):
            key = "fixed"
        elif name.startswith(("sd_", "cor_")) and "__" in name:
            key = "random:" + name.split("_", 1)[1].split("__", 1)[0]
        elif name.startswith("rescor__"):
            key = "rescor"
        elif name.startswith("sds_"):
            key = "splines"
        elif name.startswith("r_") and name.endswith("]"):
            key = "levels"
        else:
            key = "other"
        sections.setdefault(key, []).append(name)
    return {key: tuple(value) for key, value in sections.items()}


__all__ = [
    "brms_rename",
    "coefficient_labels",
    "coordinate_labels",
    "distributional_classes",
    "named_draw_classes",
    "shrinkage_scales",
    "single_variables",
]
