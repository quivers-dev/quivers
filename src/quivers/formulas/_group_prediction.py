"""Select varying effects and remap only the grouping levels used in prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping
import math

import didactic.api as dx
import formulae as fo
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprFactor,
    LetExprLiteral,
    LetFactorBinder,
    TypeName,
    LetStep,
    Module,
    ProgramDecl,
)
from quivers.formulas.formula import (
    Formula,
    _normalize_cs_intercept,
    _qvr_name,
    _random_blocks,
)

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._prediction import PredictionContext
    from quivers.inference.registry import LatentRegistry


type NewLevelPolicy = Literal[
    "uncertainty",
    "old_levels",
    "gaussian",
    "feasible_uncertainty",
    "feasible_gaussian",
]


class GroupRemap(ValueModel):
    """One compact prediction axis and its positions in the fitted axis.

    Attributes
    ----------
    group : str
        Original grouping-factor name.
    fitted_size : int
        Number of training levels, including donors absent from fresh rows.
    fitted_indices : tuple[int, ...]
        Fitted index for each compact level; minus one denotes a new identity.
    row_indices : torch.Tensor
        Compact group code for each prediction row.
    sites : Mapping[str, int]
        Raw latent names and the number of values per fitted level.
    """

    group: str
    fitted_size: int
    fitted_indices: tuple[int, ...]
    row_indices: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sites: dict[str, int]
    standard_normal_sites: frozenset[str] = frozenset()


def selected_effects(
    formulas: Mapping[str, Formula], re_formula: str | None
) -> frozenset[tuple[str, str]]:
    """Resolve a random-effects selector against fitted expanded coefficients.

    Parameters
    ----------
    formulas : Mapping[str, Formula]
        Fitted location and auxiliary formulas.
    re_formula : str or None
        None includes every term; ``~0`` excludes every term. Otherwise a
        one-sided group formula selects fitted groups and slope terms.

    Returns
    -------
    frozenset[tuple[str, str]]
        Selected grouping-factor and expanded-slope pairs.

    Raises
    ------
    ValueError
        If the selector contains fixed effects or an unfitted varying term.
    TypeError
        If the selector is not a string or None.
    """
    terms = tuple(
        term for formula in formulas.values() for term in formula.random_terms
    )
    if re_formula is None:
        return frozenset((term.group, term.slope) for term in terms)
    if not isinstance(re_formula, str):
        raise TypeError("re_formula must be a one-sided formula string or None")
    if not re_formula.lstrip().startswith("~"):
        raise ValueError("re_formula must be a one-sided formula beginning with '~'")
    expression = re_formula.lstrip()[1:].strip()
    if expression in {"0", "-1"}:
        return frozenset()
    normalized, _ = _random_blocks(_normalize_cs_intercept(expression))
    model = fo.model_description(normalized)
    if any(term.name != "Intercept" for term in model.common_terms):
        raise ValueError("re_formula can select only fitted group-level terms")
    if not model.group_terms:
        raise ValueError("re_formula must contain group terms or be '~0'")
    selected = set()
    for requested in model.group_terms:
        matches = {
            (term.group, term.slope)
            for term in terms
            if term.group == requested.factor.name
            and (term.column.term if term.column is not None else term.slope)
            == requested.expr.name
        }
        if not matches:
            raise ValueError(
                f"re_formula requests unfitted term {requested.expr.name!r} "
                f"for group {requested.factor.name!r}"
            )
        selected.update(matches)
    return frozenset(selected)


def compact_groups(
    fit: BayesianFit,
    frame: pd.DataFrame,
    selected: frozenset[tuple[str, str]],
    allow_new_levels: bool,
    registry: LatentRegistry,
) -> dict[str, GroupRemap]:
    """Encode fresh identities once and retain only levels used in these rows.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formulas and reparameterization policy.
    frame : pandas.DataFrame
        Prediction rows.
    selected : frozenset[tuple[str, str]]
        Active grouping-factor and expanded-slope pairs.
    allow_new_levels : bool
        Permit unseen and missing grouping identities.
    registry : LatentRegistry
        Original fitted raw-site layout.

    Returns
    -------
    dict[str, GroupRemap]
        One mapping per grouping factor, shared across auxiliary formulas.

    Raises
    ------
    ValueError
        If new identities are forbidden or fitted shared encodings disagree.
    TypeError
        If an inferred raw group site has an unsupported layout.
    """
    predictors = {"mu": fit.formula, **fit.distributional}
    fitted: dict[str, tuple[str, ...]] = {}
    for formula in predictors.values():
        for group, levels in formula.group_levels.items():
            levels = tuple(levels)
            if group in fitted and fitted[group] != levels:
                raise ValueError(
                    f"inconsistent fitted coding for shared group {group!r}"
                )
            fitted[group] = levels
    result = {}
    for group, levels in fitted.items():
        active = group == fit.thresholds_by or any(g == group for g, _ in selected)
        positions = {label: index for index, label in enumerate(levels)}
        keys: list[str | None] = []
        row_codes = []
        key_codes: dict[str | None, int] = {}
        if active and group not in frame and not allow_new_levels:
            raise ValueError(f"new_data is missing predictors {[group]}")
        values = frame[group] if group in frame else [None] * len(frame)
        for value in values:
            key = None if bool(pd.isna(value)) else str(value)
            if not active:
                key = levels[0]
            elif (key is None or key not in positions) and not allow_new_levels:
                raise ValueError(
                    f"new_data contains unknown groups or missing group identities for {group!r}; "
                    "set allow_new_levels=True to sample new group effects"
                )
            if key not in key_codes:
                key_codes[key] = len(keys)
                keys.append(key)
            row_codes.append(key_codes[key])
        sites = {}
        standard_normal_sites: set[str] = set()
        for parameter, formula in predictors.items():
            prefix = "" if parameter == "mu" else f"dpar_{parameter}_"
            for term in formula.random_terms:
                if term.group != group:
                    continue
                qgroup, qslope = _qvr_name(group), _qvr_name(term.slope)
                candidates = (
                    f"{prefix}z_{qgroup}_{qslope}",
                    f"{prefix}alpha_{qgroup}"
                    if term.slope == "Intercept"
                    else f"{prefix}beta_{qgroup}_{qslope}",
                )
                name = next(
                    (name for name in candidates if name in registry.names), None
                )
                if name is None:
                    raise TypeError(f"cannot identify fitted group site for {term!r}")
                count = 1
                for dimension in registry[name].constrained_shape:
                    count *= dimension
                if count % len(levels):
                    raise TypeError(f"unsupported group layout at {name!r}")
                sites[name] = count // len(levels)
                if name == candidates[0]:
                    standard_normal_sites.add(name)
        if (
            group == fit.thresholds_by
            and "cutpoint_log_spacing_noise" in registry.names
        ):
            name = "cutpoint_log_spacing_noise"
            count = math.prod(registry[name].constrained_shape)
            if count % len(levels):
                raise TypeError("unsupported grouped threshold-noise layout")
            sites[name] = count // len(levels)
            standard_normal_sites.add(name)
        result[group] = GroupRemap(
            group=group,
            fitted_size=len(levels),
            fitted_indices=tuple(
                -1 if key is None else positions.get(key, -1) for key in keys
            ),
            row_indices=torch.tensor(row_codes, dtype=torch.long),
            sites=sites,
            standard_normal_sites=frozenset(standard_normal_sites),
        )
    return result


def specialized_formula(formula: Formula, groups: Mapping[str, GroupRemap]) -> Formula:
    """Replace group cardinalities while preserving fitted transforms and priors.

    Parameters
    ----------
    formula : Formula
        Fitted evaluated metadata.
    groups : Mapping[str, GroupRemap]
        Compact prediction axes.

    Returns
    -------
    Formula
        Metadata using synthetic labels for the compact group axes.
    """
    return formula.with_(
        group_levels={
            group: tuple(
                f"prediction_{index}"
                for index in range(len(groups[group].fitted_indices))
            )
            for group in formula.group_levels
        }
    )


def select_contributions(
    module: Module,
    formulas: Mapping[str, Formula],
    selected: frozenset[tuple[str, str]],
) -> Module:
    """Zero excluded row contributions without changing correlation blocks.

    Parameters
    ----------
    module : Module
        Formula-generated prediction module.
    formulas : Mapping[str, Formula]
        Fitted predictors.
    selected : frozenset[tuple[str, str]]
        Active group and slope pairs.

    Returns
    -------
    Module
        A module retaining the fitted latent prior but excluding selected-out
        effects from its response predictor.
    """
    from quivers.formulas.compile import _factor

    excluded = set()
    for parameter, formula in formulas.items():
        prefix = "" if parameter == "mu" else f"dpar_{parameter}_"
        for term in formula.random_terms:
            if (term.group, term.slope) in selected:
                continue
            qgroup, qslope = _qvr_name(term.group), _qvr_name(term.slope)
            latent = (
                f"alpha_{qgroup}"
                if term.slope == "Intercept"
                else f"beta_{qgroup}_{qslope}"
            )
            excluded.update(
                (
                    f"{prefix}{latent}_per_row",
                    f"{prefix}z_{qgroup}_{qslope}_contribution",
                )
            )
    statements = []
    for statement in module.statements:
        if isinstance(statement, ProgramDecl):
            steps = []
            for step in statement.draws:
                if isinstance(step, LetStep) and step.name in excluded:
                    zero = LetExprLiteral(value=0.0)
                    if isinstance(step.value, LetExprFactor):
                        binders = step.value.binders
                        if not any(
                            isinstance(binder.index, TypeName)
                            and binder.index.name == "Resp"
                            for binder in binders
                        ):
                            binders = (
                                *binders,
                                LetFactorBinder(
                                    var="prediction_row", index=TypeName(name="Resp")
                                ),
                            )
                        value = step.value.with_(binders=binders, body=zero, cases=())
                    else:
                        value = _factor((("prediction_row", "Resp"),), body=zero)
                    step = step.with_(value=value)
                steps.append(step)
            statement = statement.with_(draws=tuple(steps))
        statements.append(statement)
    return module.with_(statements=tuple(statements))


def remap_draw(
    context: PredictionContext,
    latents: dict[str, torch.Tensor],
    shapes: Mapping[str, tuple[int, ...]],
    generator: torch.Generator,
    fixed_donors: dict[str, torch.Tensor],
    donors: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Gather compact known levels and sample coherent effects for new identities.

    Parameters
    ----------
    context : PredictionContext
        Compact native program, data and new-level policy.
    latents : dict[str, torch.Tensor]
        One original constrained posterior draw.
    shapes : Mapping[str, tuple[int, ...]]
        Specialized raw latent shapes.
    generator : torch.Generator
        Request-owned CPU stream, independent of response sampling.
    fixed_donors : dict[str, torch.Tensor]
        Lazily retained per-identity donors for the old_levels policy.
    donors : dict[str, torch.Tensor] or None
        Optional receiver for joint donor indices, used to retain complete
        design-constrained threshold profiles alongside their raw effects.

    Returns
    -------
    dict[str, torch.Tensor]
        Constrained bindings on the compact prediction group axes.
    """
    from quivers.inference.trace import trace

    if not context.groups:
        return latents
    group_names = {name for group in context.groups.values() for name in group.sites}
    policy = context.new_level_policy.removeprefix("feasible_")
    need_gaussian = policy == "gaussian" and any(
        -1 in group.fitted_indices for group in context.groups.values()
    )
    direct_gaussian = need_gaussian and all(
        set(group.sites) <= group.standard_normal_sites
        for group in context.groups.values()
    )
    result = dict(latents)
    # Exit the fork before yielding a trace to its consumer. Otherwise response
    # sampling would advance the next draw's group stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(generator.get_state())
        sampled = None
        if need_gaussian and not direct_gaussian:
            sampled = trace(
                context.program,
                torch.zeros(1, 1, dtype=torch.long),
                dict(context.observations)
                | {
                    name: value
                    for name, value in latents.items()
                    if name not in group_names
                },
            )
        for group_name, group in context.groups.items():
            indices = torch.tensor(group.fitted_indices, dtype=torch.long)
            new = indices < 0
            count = int(new.sum())
            if count and policy != "gaussian":
                if policy == "old_levels":
                    if group_name not in fixed_donors:
                        fixed_donors[group_name] = torch.randint(
                            group.fitted_size, (count,)
                        )
                    indices[new] = fixed_donors[group_name]
                else:
                    indices[new] = torch.randint(group.fitted_size, (count,))
            if donors is not None and policy != "gaussian":
                donors[group_name] = indices.clone()
            for name, width in group.sites.items():
                source = latents[name].reshape(group.fitted_size, width)
                value = source[indices.clamp_min(0)].clone()
                if count and direct_gaussian:
                    value[new] = torch.randn(
                        (count, width), dtype=value.dtype, device=value.device
                    )
                elif count and sampled is not None:
                    value[new] = sampled.sites[name].value.reshape(len(indices), width)[
                        new
                    ]
                result[name] = value.reshape(shapes[name])
        generator.set_state(torch.random.get_rng_state())
    return result
