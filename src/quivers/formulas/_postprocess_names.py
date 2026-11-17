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
from quivers.formulas._autocor import autocor_binding
from quivers.formulas._brms_priors import derived_site_draws
from quivers.formulas._formula_options import display_label
from quivers.formulas._priors import PriorTarget
from quivers.formulas._smooth import smooth_states
from quivers.formulas._threshold_groups import LOCATION_SITE
from quivers.formulas.formula import Formula
from quivers.formulas._variable_catalogue import (
    brms_order,
    brms_rename,
    checked_variables,
    class_variables,
    correlation_pairs,
    design_term_variables,
    native_variables,
    scalar_coordinates,
    shrinkage_blocks,
    shrinkage_scale_site,
    shrinkage_variables,
    SmoothNames,
    smooth_names,
    threshold_variables,
)
from quivers.inference._posterior import draw_count

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit

_INDEXED = re.compile(r"^(?P<stem>.*\()(?P<args>.*)\)\[(?P<index>[0-9]+)\]$")


def _r_label(label: str) -> str:
    """Rewrite a formulae column label to R's ``model.matrix`` label.

    Parameters
    ----------
    label : str
        formulae label such as ``f[q]``, ``x:f[q]`` or ``poly(z, 2)[0]``.

    Returns
    -------
    str
        R's label before brms renaming: factor levels follow the term
        (``fq``), the columns of a matrix-valued call are numbered from
        one (``poly(z, 2)1``), and a power inside ``I()``, which the formula
        layer evaluates as ``**``, is written ``^`` as in the R formula.
    """
    # R's ^ in I(): brms labels I(x^2) as IxE2.
    label = label.replace(" ** ", "^").replace("**", "^")
    match = _INDEXED.match(label)
    if match is not None:
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
            labels = getattr(terms.get(column.term), "labels", None)
            label = (
                _r_label(labels[position])
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
    return not any(
        column.category_specific for column in fit.formula.fixed_columns
    ) and not any(
        term.column is not None and term.column.category_specific
        for term in fit.formula.random_terms
    )


def _time_label(level: str) -> str:
    """Spell a retained time level as R prints the time value.

    Parameters
    ----------
    level : str
        Canonical time level: ``repr`` of a float for numbers, text otherwise.

    Returns
    -------
    str
        An integer-valued number without its fraction (``"1.0"`` is ``"1"``),
        other values with brms's character renaming.
    """
    try:
        number = float(level)
    except ValueError:
        return brms_rename(level)
    return str(int(number)) if number.is_integer() else brms_rename(level)


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
    binding = autocor_binding(fit.family)
    if binding is None:
        return ()
    return tuple(_time_label(level) for level in binding.time_levels)


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
    prefixes: dict[str, str] = {}
    special: dict[str, torch.Tensor] = {}
    shrunk: dict[str, str] = {}
    smooths: dict[str, SmoothNames] = {}
    scale_names: dict[str, str] = {}
    items = _location_first(fit)
    for parameter, prefix, predictor in items:
        prefixes[parameter] = prefix
        smooths[parameter] = smooth_names(
            predictor,
            smooth_states(plan.designs.get(parameter)),
            infix=_infix(parameter),
        )
        scale_names |= {
            prefix + site: name for site, name in smooths[parameter].scales.items()
        }
        for column in predictor.fixed_columns:
            site = prefix + (
                "intercept" if column.is_intercept else f"beta_{column.qvr_name}"
            )
            if site not in draws:
                continue
            consumed.add(site)
            if thresholds and parameter == "mu" and column.is_intercept:
                continue
            value = draws[site]
            coef = labels[(parameter, column.name)]
            name = f"b_{_infix(parameter)}{coef}"
            shrunk[site] = f"{_infix(parameter)}{coef}"
            if column.is_intercept:
                intercepts.add(name)
            if column.name in smooths[parameter].population:
                # brms names the unpenalized columns of a smooth bs_<label>_<k>.
                population |= scalar_coordinates(
                    smooths[parameter].population[column.name], value
                )
            elif column.category_specific:
                # brms names category-specific coefficients bcs_<coef>[k].
                population |= scalar_coordinates(
                    f"bcs_{_infix(parameter)}"
                    f"{brms_rename(column.term.removeprefix('cs(').removesuffix(')'))}"
                    if column.term.startswith("cs(")
                    else f"bcs_{_infix(parameter)}{coef}",
                    value,
                )
            elif value.ndim > len(shape) and fit.coefficient_labels:
                for index, label in enumerate(fit.coefficient_labels):
                    population |= scalar_coordinates(
                        f"b_mu{brms_rename(str(label))}_{coef}"
                        if parameter == "mu"
                        else f"{name}[{index + 1}]",
                        value[..., index],
                    )
            else:
                population |= scalar_coordinates(name, value)
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
    structures = fit._group_structures()
    auxiliary: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    terms: dict[str, str] = {}
    by_scales: dict[str, list[list[str]]] = {}
    by_correlations = {
        site
        for structure in structures.values()
        for sites in structure.by_correlation_sites.values()
        for site in sites
    }
    for target in targets:
        if target.site not in draws or target.site in consumed:
            continue
        parameter = target.nlpar or target.dpar or "mu"
        value = draws[target.site]
        if target.class_ in {"b", "Intercept"}:
            local = target.site.removeprefix(prefixes.get(parameter, ""))
            if local.startswith("bsp_") and target.coef is not None:
                # brms names special-term coefficients bsp_<term>.
                population |= scalar_coordinates(
                    f"bsp_{_infix(parameter)}{brms_rename(target.coef)}", value
                )
            consumed.add(target.site)
        elif target.class_ == "sd" and target.group is not None:
            coef = target.coef or "Intercept"
            term = (
                f"{_infix(parameter)}{labels.get((parameter, coef), brms_rename(coef))}"
            )
            terms[target.site] = term
            structure = structures.get(target.group)
            if structure is not None and structure.by_levels and value.ndim == 3:
                # brms: one standard deviation per level of the by variable.
                by_names = [
                    f"sd_{target.group}__{term}:"
                    f"{brms_rename(str(structure.by) + str(level))}"
                    for level in structure.by_levels
                ]
                for index, name in enumerate(by_names):
                    scales[name] = value[..., index]
                by_scales.setdefault(target.group, []).append(by_names)
            else:
                scales |= scalar_coordinates(f"sd_{target.group}__{term}", value)
            consumed.add(target.site)
        elif target.class_ == "cor" and (
            target.site in by_correlations
            or any(
                layout.correlation_site == target.site for layout in plan.group_layouts
            )
        ):
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
                scales=scale_names,
                labels=coordinate_labels(fit, target),
            )
            consumed.add(target.site)
    # brms lists the standard deviations of a by-group level by level.
    for coefficient_names in by_scales.values():
        by_level = iter(
            [name for level in zip(*coefficient_names, strict=True) for name in level]
        )
        grouped = {name for names in coefficient_names for name in names}
        reordered: dict[str, torch.Tensor] = {}
        for name in scales:
            key = next(by_level) if name in grouped else name
            reordered[key] = scales[key]
        scales = reordered
    correlations: dict[str, torch.Tensor] = {}
    for layout in plan.group_layouts:
        consumed.update(
            site
            for member in layout.members
            for site in (member.raw_site, member.scale_site, member.mean_site)
            if site is not None
        )
        coordinates: list[str] = []
        for member in layout.members:
            parameter = (
                member.prefix.removeprefix("dpar_").removeprefix("nlpar_").rstrip("_")
                if member.prefix
                else "mu"
            )
            term = terms.get(
                member.scale_site or "",
                f"{_infix(parameter)}"
                f"{labels.get((parameter, member.slope), brms_rename(member.slope))}",
            )
            coordinates.extend(
                [term]
                if member.width == 1
                else [f"{term}[{index + 1}]" for index in range(member.width)]
            )
        structure = structures.get(layout.group)
        by_sites = (
            structure.by_correlation_sites.get(layout.identity, ())
            if structure is not None
            else ()
        )
        if structure is not None and by_sites:
            # brms: one correlation matrix per level of the by variable.
            for level, site in zip(structure.by_levels, by_sites, strict=True):
                if site not in draws:
                    continue
                consumed.add(site)
                suffix = ":" + brms_rename(str(structure.by) + str(level))
                correlations |= correlation_pairs(
                    f"cor_{layout.group}",
                    [coordinate + suffix for coordinate in coordinates],
                    draws[site],
                )
        if layout.correlation_site is None or layout.correlation_site not in draws:
            continue
        consumed.add(layout.correlation_site)
        correlations |= correlation_pairs(
            f"cor_{layout.group}", coordinates, draws[layout.correlation_site]
        )
    for parameter, prefix, predictor in items:
        named, read = design_term_variables(
            predictor,
            prefix,
            draws,
            infix=_infix(parameter),
            smooths=smooths[parameter],
        )
        special |= named
        consumed |= read
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
        suffix = "" if parameter == "mu" else f"__{parameter}"
        varying = fit._random_effects(draws, shape, predictor, prefix)
        for group, coefficients in varying.items():
            for coef, value in coefficients.items():
                label = labels.get((parameter, coef), brms_rename(coef))
                for index, level in enumerate(group_levels[group]):
                    group_coefficients |= scalar_coordinates(
                        f"r_{group}{suffix}[{level},{label}]", value[:, :, index]
                    )
    return checked_variables(
        brms_order(
            population
            | auxiliary
            | scales
            | correlations
            | special
            | group_coefficients
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
