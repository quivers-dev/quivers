"""Prepare joint (multivariate) prediction rows through the response preparations.

Each response's fresh rows use the ordinary response binders for transforms,
contrasts, trials, response roles, and group coding. The joint program is
emitted once over the specialized response formulas, with one compact axis
per shared grouping factor, and compiled once per request.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import TYPE_CHECKING

import didactic.api as dx
import narwhals as nw
import numpy as np
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl.ast_nodes import ProgramDecl
from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._group_law import KnownCovariance
from quivers.formulas._group_prediction import (
    CovarianceRemap,
    GroupRemap,
    NewLevelPolicy,
    selected_effects,
    specialized_formula,
)
from quivers.formulas._joint_compile import emit_joint
from quivers.formulas._fit_core import compile_model
from quivers.formulas._joint_missing import (
    apply_missing,
    fitted_measurement_prior,
    fresh_values,
    missing_multipliers,
    missing_site_exclusions,
    split_measurement_priors,
)
from quivers.formulas._row_source import RowSource
from quivers.formulas._group_law import law_inputs
from quivers.formulas._term import bind_inputs
from quivers.formulas._response import response_tensor
from quivers.formulas._likelihood import prepare_likelihood_rows
from quivers.formulas._prediction import (
    JointResponsePreparation,
    OrdinalReplay,
    PredictionContext,
    PredictionRows,
    prepare_prediction_rows,
    residual_at,
)
from quivers.formulas._response_modifiers import response_input_name
from quivers.formulas.formula import Formula, RowSelection, _qvr_name
from quivers.inference._prior_projection import project_prior_sites
from quivers.inference.registry import LatentRegistry

if TYPE_CHECKING:
    from quivers.formulas._joint_fit import MultivariateFit

#: Name of the stacked observation of a residual-correlation event.
JOINT_RESPONSE = "formula_joint_response"


class JointPredictionContext(ValueModel):
    """One prepared joint program and the row count of each response.

    Parameters
    ----------
    context : PredictionContext
        Shared replay context of the joint program, including the compact
        group axes and the new-level policy.
    rows : dict[str, int]
        Prediction rows of each response.
    """

    context: PredictionContext = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    rows: dict[str, int]


def _response_selector(
    formulas: Mapping[str, Formula], selected: frozenset[tuple[str, str]]
) -> str | None:
    """Restrict a joint varying-effect selection to one response.

    Parameters
    ----------
    formulas : Mapping[str, Formula]
        The response's additive predictors.
    selected : frozenset[tuple[str, str]]
        Grouping-factor and slope pairs selected across all responses.

    Returns
    -------
    str or None
        None for every fitted term of the response, ``"~0"`` for none, and
        otherwise a one-sided selector over the response's selected terms.
    """
    terms = {
        (term.group, term.slope, term.column.term if term.column is not None else None)
        for formula in formulas.values()
        for term in formula.random_terms
    }
    chosen = {item for item in terms if (item[0], item[1]) in selected}
    if chosen == terms:
        return None
    if not chosen:
        return "~0"
    parts = sorted(
        "(1|" + group + ")"
        if slope == "Intercept"
        else "(0+" + (expression or slope) + "|" + group + ")"
        for group, slope, expression in chosen
    )
    return "~" + "+".join(parts)


def _joint_selection(
    fit: MultivariateFit, re_formula: str | None
) -> dict[str, frozenset[tuple[str, str]]]:
    """Resolve one varying-effect selector against every response.

    Parameters
    ----------
    fit : MultivariateFit
        Joint fit.
    re_formula : str or None
        One-sided selector, None for every term, or ``"~0"`` for none.

    Returns
    -------
    dict[str, frozenset[tuple[str, str]]]
        Selected grouping-factor and slope pairs of each response.

    Raises
    ------
    ValueError
        If a requested term occurs in no response.
    """
    formulas = {
        name + ":" + parameter: formula
        for name in fit.response_names
        for parameter, _, formula in fit.emission.responses[name].lens._predictor_items(
            fit.emission.responses[name].formula
        )
    }
    selected = selected_effects(formulas, re_formula)
    return {
        name: frozenset(
            (term.group, term.slope)
            for key, formula in formulas.items()
            if key.split(":", 1)[0] == name
            for term in formula.random_terms
            if (term.group, term.slope) in selected
        )
        for name in fit.response_names
    }


def _response_rows(
    context: PredictionRows,
    groups: Mapping[str, GroupRemap],
    selected: frozenset[tuple[str, str]],
) -> RowSource:
    """Read one response's rows on the joint levels of the grouping factors.

    Parameters
    ----------
    context : PredictionRows
        Prepared rows of the response alone.
    groups : Mapping[str, GroupRemap]
        Compact levels shared by the responses.
    selected : frozenset[tuple[str, str]]
        Group-level coefficients the request includes.

    Returns
    -------
    RowSource
        The same rows on the shared levels.
    """
    assert context.source is not None
    return context.source.regrouped(groups, selected)


def _merge_groups(
    fit: MultivariateFit,
    contexts: Mapping[str, PredictionRows],
    active: Mapping[str, frozenset[str]],
) -> dict[str, GroupRemap]:
    """Merge compact response axes into one axis per grouping factor.

    Parameters
    ----------
    fit : MultivariateFit
        Joint fit with namespaced group layouts.
    contexts : Mapping[str, PredictionRows]
        Response-local prepared row bindings.
    active : Mapping[str, frozenset[str]]
        Grouping factors whose identities each response evaluates.

    Returns
    -------
    dict[str, GroupRemap]
        One remap per grouping factor over joint raw site names.

    Raises
    ------
    ValueError
        If two responses encode one grouping factor differently, as happens
        when their included rows differ.
    """
    chosen: dict[str, GroupRemap] = {}
    owned: set[str] = set()
    for name, context in contexts.items():
        for group, remap in context.groups.items():
            if group not in active[name]:
                chosen.setdefault(group, remap)
            elif group not in owned:
                chosen[group] = remap
                owned.add(group)
            elif (
                chosen[group].fitted_indices != remap.fitted_indices
                or chosen[group].levels != remap.levels
                or not torch.equal(chosen[group].row_indices, remap.row_indices)
            ):
                raise ValueError(
                    f"responses encode grouping factor {group!r} differently on "
                    "new_data; joint prediction requires every response to "
                    "evaluate the same rows of each shared grouping factor"
                )
    merged = {}
    for group, remap in chosen.items():
        # The responses declaring by partitions code them on the compact levels.
        partitions = {
            partition.axis: partition
            for context in contexts.values()
            if (other := context.groups.get(group)) is not None
            and other.fitted_indices == remap.fitted_indices
            for partition in other.partitions
        }
        # One covariance of the factor may have blocks in several responses.
        covariances: dict[str, CovarianceRemap] = {}
        for context in contexts.values():
            other = context.groups.get(group)
            for known in other.covariances if other is not None else ():
                covariances.setdefault(known.name, known)
        merged[group] = remap.with_(
            level_sites=tuple(
                site for site in fit.emission.level_sites if site.group == group
            ),
            partitions=tuple(partitions.values()),
            covariances=tuple(
                known.with_(
                    laws=tuple(
                        law
                        for law in fit.emission.group_laws
                        if law.group == group
                        and isinstance(law.level.covariance, KnownCovariance)
                        and law.level.covariance.name == name
                    )
                )
                for name, known in covariances.items()
            ),
        )
    return merged


def _active_groups(
    response: JointResponsePreparation, selected: frozenset[tuple[str, str]]
) -> frozenset[str]:
    """Return the grouping factors whose identities a response evaluates.

    Parameters
    ----------
    response : JointResponsePreparation
        Prepared response metadata.
    selected : frozenset[tuple[str, str]]
        The response's selected grouping-factor and slope pairs.

    Returns
    -------
    frozenset[str]
        Selected grouping factors, threshold groups, and baseline groups.
    """
    groups = {group for group, _ in selected}
    if response.thresholds_by is not None:
        groups.add(response.thresholds_by)
    if response.family.baseline is not None and response.family.baseline.by:
        groups.add(response.family.baseline.by)
    return frozenset(groups)


def prepare_joint_prediction(
    fit: MultivariateFit,
    new_data: IntoDataFrame | None,
    *,
    recorded: frozenset[str] = frozenset(),
    re_formula: str | None = None,
    allow_new_levels: bool = False,
    sample_new_levels: NewLevelPolicy = "uncertainty",
    seed: int = 0,
    require_trials: bool = True,
    scoring: bool = False,
    data2: Mapping[str, AutocorMatrix] | None = None,
    incl_autocor: bool = True,
) -> JointPredictionContext:
    """Prepare one joint program and its bindings for a prediction request.

    Cost: one response preparation per response and one compilation of the
    joint program per request; replay then costs one native execution per
    retained draw, O(rows) in each response's design width.

    Parameters
    ----------
    fit : MultivariateFit
        Joint fit.
    new_data : IntoDataFrame or None
        Fresh rows holding every response's predictors, or None for the
        training rows.
    recorded : frozenset[str]
        Responses whose recorded outcomes are read from new_data for
        likelihood scoring. Other responses bind internal placeholders.
    re_formula : str or None
        Varying-effect selector applied to every response.
    allow_new_levels : bool
        Permit unseen grouping identities.
    sample_new_levels : NewLevelPolicy
        Existing donor or Gaussian policy for new identities, applied once
        to each shared grouping axis, so new effects stay jointly drawn
        across responses.
    seed : int
        Seed of the separate group-effect stream.
    require_trials : bool
        Require trial totals for fresh vector count responses.
    scoring : bool
        Prepare training rows for likelihood scoring, which uses the fitted
        program itself. Prediction on training rows of a model with ``mi()``
        responses instead keeps each ``mi()`` response's law on every row,
        with its missing values replayed from their posterior draws.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Matrices of ``sar`` and ``fcor`` terms over the rows of ``new_data``
        (brms ``newdata2``).
    incl_autocor : bool
        Include the dependence of residual laws and latent residual terms
        (brms ``incl_autocor``).

    Returns
    -------
    JointPredictionContext
        The joint replay context and each response's row count.

    Raises
    ------
    ValueError
        If rows, predictors, levels, or outcomes are invalid, a requested
        varying effect occurs in no response, or a cumulative response with
        category-specific effects needs retained threshold replay.
    TypeError
        If the specialized joint program changes the fitted latent layout.
    """
    rows = {
        name: len(fit.emission.responses[name].formula.response_values)
        for name in fit.response_names
    }
    # A joint residual event with mi() responses is fitted by a log-density
    # factor over completed vectors, so its event law is read from the
    # prediction program.
    factored = fit.specification.rescor and bool(fit.missing.responses)
    if (
        new_data is None
        and re_formula is None
        and incl_autocor
        and (fit.missing.empty or (scoring and not factored))
    ):
        return JointPredictionContext(
            context=PredictionContext(
                program=fit.program,
                observations=fit.observations,
                rows=max(rows.values()),
            ),
            rows=rows,
        )
    selection = _joint_selection(fit, re_formula)
    observations: dict[str, torch.Tensor]
    groups: dict[str, GroupRemap] = {}
    contexts: dict[str, PredictionRows] = {}
    order_groups: set[str] = set()
    frame = (
        nw.from_native(new_data, eager_only=True).to_pandas()
        if new_data is not None
        else None
    )
    fresh = (
        fresh_values(fit.missing, frame)
        if frame is not None and not fit.missing.empty
        else {}
    )
    if frame is None:
        observations = dict(fit.observations)
    else:
        active: dict[str, frozenset[str]] = {}
        for name in fit.response_names:
            response = fit._response_preparation(name)
            formulas = {parameter: f for parameter, _, f in response._predictor_items()}
            selector = _response_selector(formulas, selection[name])
            if name in recorded:
                context, bindings = prepare_likelihood_rows(
                    response,
                    frame,
                    None,
                    re_formula=selector,
                    allow_new_levels=allow_new_levels,
                    sample_new_levels=sample_new_levels,
                    seed=seed,
                    incl_autocor=incl_autocor,
                    data2=data2,
                )
                contexts[name] = context.with_(observations=bindings)
            else:
                contexts[name] = prepare_prediction_rows(
                    response,
                    frame,
                    None,
                    re_formula=selector,
                    allow_new_levels=allow_new_levels,
                    sample_new_levels=sample_new_levels,
                    seed=seed,
                    require_trials=require_trials,
                    incl_autocor=incl_autocor,
                    data2=data2,
                )
            rows[name] = contexts[name].rows
            if contexts[name].retain_ordinal_cutpoints:
                order_groups.update(contexts[name].ordinal_order_groups)
            active[name] = _active_groups(response, selection[name])
        groups = _merge_groups(fit, contexts, active)
        observations = {}
    components = {}
    for name in fit.response_names:
        response = fit.emission.responses[name]
        lens = copy(response.lens)
        # Joint prior rules are resolved again on the joint catalogue.
        lens._set_prior_configuration(None)
        # A residual law is rebuilt over the requested rows.
        if lens._residual is not None and (frame is not None or not incl_autocor):
            lens._residual = residual_at(
                lens._residual,
                fit._response_data(name),
                frame,
                include=incl_autocor,
                data2=data2,
            )
        formula = response.formula
        if groups:

            def specialize(value: Formula) -> Formula:
                """Retain fitted encodings on the compact prediction axes.

                Parameters
                ----------
                value : Formula
                    One fitted predictor.

                Returns
                -------
                Formula
                    The predictor over compact group labels and codes, with
                    its ``gr()`` options coded on the compact levels.
                """
                assert frame is not None
                value = specialized_formula(value, groups, frame)
                return value.with_(
                    group_indices={
                        group: tuple(groups[group].row_indices.tolist())
                        for group in value.group_levels
                    },
                )

            lens._distributional = {
                key: specialize(value) for key, value in lens._distributional.items()
            }
            if lens._predictor_graph is not None:
                lens._predictor_graph = lens._predictor_graph.with_(
                    leaves={
                        key: specialize(value)
                        for key, value in lens._predictor_graph.leaves.items()
                    }
                )
            formula = specialize(formula)
        lens._emitted_groups = []
        components[name] = (lens, formula)

    sources = {
        name: RowSource(selected=selection[name], include=incl_autocor)
        if frame is None
        else _response_rows(contexts[name], groups, selection[name])
        for name in fit.response_names
    }
    emission = emit_joint(
        components,
        # me() hyperparameter rules apply to the me() rewrite below.
        priors=split_measurement_priors(fit.priors)[0],
        rescor=fit.specification.rescor,
        rows=sources,
    )
    terms = {
        name: lens._terms(formula, sources[name])
        for name, (lens, formula) in components.items()
    }
    fresh_sites = tuple(
        emission.responses[name].names.get(site.name, site.name)
        for name, response_terms in terms.items()
        for term in response_terms
        for site in term.sites()
        if site.fresh
    )
    missing_inputs: dict[str, torch.Tensor] = {}
    if not fit.missing.empty:
        if frame is None:
            if fit.preparation_data is None:
                raise ValueError("mi() prediction requires the retained training rows")
            original = nw.from_native(fit.preparation_data, eager_only=True).to_pandas()
            emission, missing_inputs, _ = apply_missing(
                emission,
                fit.missing,
                fit.missing_specs,
                original,
                keep_observation=True,
                prior=fitted_measurement_prior(fit.emission.module),
                consumer_rows={
                    name: fit.emission.responses[name].formula.modifiers.subset
                    or RowSelection(
                        size=len(original), positions=tuple(range(len(original)))
                    )
                    for name in fit.response_names
                },
            )
            if factored:
                observations = {
                    key: value
                    for key, value in observations.items()
                    if not key.startswith("mi_joint__")
                }
                observations[JOINT_RESPONSE] = emission.fixed_inputs[JOINT_RESPONSE]
            else:
                for label in fit.missing.responses:
                    response = fit.emission.responses[label]
                    observations[
                        response.names[_qvr_name(response.formula.response_name)]
                    ] = response_tensor(response.formula, fit.prepared[label].family)
        else:
            emission, missing_inputs, _ = apply_missing(
                emission,
                fit.missing,
                fit.missing_specs,
                frame,
                fresh=fresh,
                multipliers=missing_multipliers(
                    fit.missing,
                    {name: part.designs for name, part in fit.prepared.items()},
                    frame,
                ),
            )
    compiled = compile_model(emission.module, emission.families)
    program = compiled.morphism
    if not isinstance(program, MonadicProgram):
        raise TypeError("joint prediction requires a native monadic program")
    parameters = {
        parameter.name
        for statement in emission.module.statements
        if isinstance(statement, ProgramDecl)
        for parameter in statement.value_parameters
    }
    if new_data is not None:
        observations = {
            key: value
            for key, value in emission.fixed_inputs.items()
            if key != JOINT_RESPONSE
        }
    for name, response in emission.responses.items():
        if (context := contexts.get(name)) is not None:
            for key, value in context.observations.items():
                target = response.names.get(key)
                if target is not None:
                    observations[target] = value
                elif key in parameters:
                    observations[key] = value
        # Even retained rows can have different term inputs, such as when
        # a request excludes latent residuals.
        lens, _ = components[name]
        observations.update(
            {
                response.names.get(key, key): value
                for key, value in bind_inputs(
                    (
                        *law_inputs(lens.group_laws),
                        *(item for term in terms[name] for item in term.inputs()),
                    )
                ).items()
                if key in response.names or key in parameters
            }
        )
    if new_data is not None:
        observations.update(
            {
                f"{_qvr_name(group)}_idx": remap.row_indices
                for group, remap in groups.items()
            }
        )
        if fit.specification.rescor:
            observations[JOINT_RESPONSE] = torch.stack(
                tuple(
                    contexts[name]
                    .observations[response_input_name(emission.responses[name].formula)]
                    .to(torch.get_default_dtype())
                    for name in fit.response_names
                ),
                dim=-1,
            )
    observations |= missing_inputs
    if new_data is not None:
        if missing := parameters - set(observations):
            raise ValueError(
                f"joint prediction left program inputs unbound: {sorted(missing)!r}"
            )
        _check_layout(fit, program, observations, groups, fresh_sites)
    prior_program = None
    if sample_new_levels.removeprefix("feasible_") == "gaussian" and any(
        -1 in group.fitted_indices
        and not set(group.sites) <= group.standard_normal_sites
        for group in groups.values()
    ):
        prior_program = project_prior_sites(
            program,
            {
                site
                for group in groups.values()
                for site in (*group.sites, *group.support_sites)
            },
        )
    # Cumulative responses with category-specific effects replay their fitted
    # thresholds, exactly as a single-response prediction does.
    replays: list[OrdinalReplay] = []
    for name, response in fit.emission.responses.items():
        if (
            response.formula.category_specific
            and fit._response_readout(name).constrains_cutpoints
        ):
            group = fit.specification.responses[name].thresholds_by
            replays.append(
                OrdinalReplay(
                    prefix=response.prefix,
                    thresholds_by=group,
                    fitted_levels=len(fit.group_levels[group])
                    if group is not None
                    else 0,
                    rows=rows[name],
                )
            )
    return JointPredictionContext(
        context=PredictionContext(
            program=program,
            observations=observations,
            rows=max(rows.values()),
            groups=groups,
            prior_program=prior_program,
            new_level_policy=sample_new_levels,
            seed=seed,
            retain_ordinal_cutpoints=bool(replays),
            ordinal_order_groups=tuple(sorted(order_groups)),
            ordinal_replays=tuple(replays),
            fresh_sites=fresh_sites,
        ),
        rows=rows,
    )


def _check_layout(
    fit: MultivariateFit,
    program: MonadicProgram,
    observations: Mapping[str, torch.Tensor],
    groups: Mapping[str, GroupRemap],
    fresh_sites: tuple[str, ...],
) -> None:
    """Require the specialized program to keep the fitted latent layout.

    Parameters
    ----------
    fit : MultivariateFit
        Joint fit and its training program.
    program : MonadicProgram
        Specialized prediction program.
    observations : Mapping[str, torch.Tensor]
        Prediction bindings.
    groups : Mapping[str, GroupRemap]
        Compact group axes with their raw sites.
    fresh_sites : tuple[str, ...]
        Sites drawn for the fresh rows.

    Raises
    ------
    TypeError
        If latent names change, a non-group site changes shape, or a group
        site's compact size disagrees with its remap.
    """
    original = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    fresh = LatentRegistry.from_model(program, observed_names=set(observations))
    covariance = (
        {site for group in groups.values() for site in group.support_sites}
        | missing_site_exclusions(fit.missing, original.names)
        | set(fresh_sites)
    )
    if set(original.names) - covariance != set(fresh.names) - covariance:
        raise TypeError("joint new_data specialization changed the fitted latent names")
    widths = {
        site: (group, width)
        for group in groups.values()
        for site, width in group.sites.items()
    }
    for name in original.names:
        if name in covariance or name not in fresh.names:
            continue
        if name in widths:
            group, width = widths[name]
            if (
                int(np.prod(fresh[name].constrained_shape))
                != len(group.fitted_indices) * width
            ):
                raise TypeError(f"joint new_data changed group layout at {name!r}")
        elif original[name].constrained_shape != fresh[name].constrained_shape:
            raise TypeError(f"joint new_data changed latent layout at {name!r}")


__all__: list[str] = []
