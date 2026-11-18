"""Select varying effects and remap only the grouping levels used in prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping
import math

import didactic.api as dx
import formulae as fo
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._group_covariance import (
    GroupCovariance,
    GroupCovarianceBasis,
    _tolerance,
)
from quivers.formulas._grouping import rewrite_grouping
from quivers.formulas._group_law import (
    GroupBlockLaw,
    encode_support,
    support_gaussian,
)
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

    Parameters
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
    standard_normal_sites : frozenset[str]
        Sites admitting direct standard-Normal new-group draws.
    levels : tuple[str, ...]
        Compact labels; supplied covariance retains authored identities.
    covariance : GroupCovariance or None
        Complete labeled covariance, including requested future identities.
    covariance_laws : tuple[GroupBlockLaw, ...]
        Fitted Gaussian support and coefficient slices.
    covariance_target, covariance_conditional : GroupCovarianceBasis or None
        Once-per-request compact and Schur-conditional support factors.
    covariance_mean_map : tuple[tuple[float, ...], ...]
        Once-per-request conditional mean map from all fitted levels.
    member_indices : tuple[torch.Tensor, ...]
        Compact codes of every member of a brms ``mm()`` factor, in member
        order; empty for an ordinary factor, whose codes are ``row_indices``.
    """

    group: str
    fitted_size: int
    fitted_indices: tuple[int, ...]
    row_indices: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sites: dict[str, int]
    standard_normal_sites: frozenset[str] = frozenset()
    levels: tuple[str, ...] = ()
    covariance: GroupCovariance | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariance_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariance_target: GroupCovarianceBasis | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariance_conditional: GroupCovarianceBasis | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariance_mean_map: tuple[tuple[float, ...], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    member_indices: tuple[torch.Tensor, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )


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
    # gr(), mm(), g1:g2, and g1/g2 select the factors they name in the fit.
    normalized, _ = _random_blocks(
        rewrite_grouping(_normalize_cs_intercept(expression)).formula
    )
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
    predictors = {name: formula for name, _, formula in fit._predictor_items()}
    prefixes = {name: prefix for name, prefix, _ in fit._predictor_items()}
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
        baseline_group = (
            fit.family.baseline.by if fit.family.baseline is not None else None
        )
        active = group in (fit.thresholds_by, baseline_group) or any(
            g == group for g, _ in selected
        )
        positions = {label: index for index, label in enumerate(levels)}
        keys: list[str | None] = []
        row_codes = []
        key_codes: dict[str | None, int] = {}
        # A brms mm() factor codes every member column on one compact axis.
        membership = next(
            (
                formula.grouping.membership(group)
                for formula in predictors.values()
                if formula.grouping.membership(group) is not None
            ),
            None,
        )
        sources = membership.spec.sources if membership is not None else (group,)
        missing = [source for source in sources if source not in frame]
        if active and missing and not allow_new_levels:
            raise ValueError(f"new_data is missing predictors {missing}")
        member_rows = []
        for source in sources:
            row_codes = []
            values = frame[source] if source in frame else [None] * len(frame)
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
            member_rows.append(row_codes)
        row_codes = member_rows[0]
        sites = {}
        standard_normal_sites: set[str] = set()
        layouts = (
            fit.prediction_plan.group_layouts if fit.prediction_plan is not None else ()
        )
        planned = set()
        for layout in layouts:
            if layout.group != group:
                continue
            sites.update(layout.raw_sites)
            planned.update((member.prefix, member.slope) for member in layout.members)
            if (
                layout.parameterization == "noncentered"
                and layout.level_covariance is None
            ):
                standard_normal_sites.update(layout.raw_sites)
        for parameter, formula in predictors.items():
            prefix = prefixes[parameter]
            for term in formula.random_terms:
                if term.group != group or (prefix, term.slope) in planned:
                    continue
                raise TypeError(
                    f"fitted group term {term.slope!r} for {group!r} has no "
                    "compiled coordinate layout"
                )
        # Each level of a Student-t factor also owns its mixing scale.
        if fit.prediction_plan is not None:
            for law in fit.prediction_plan.group_laws:
                if law.group == group and law.level.mixing is not None:
                    sites[law.level.mixing.mixing_site] = 1
        threshold_raw = next(
            (
                name
                for name in ("cutpoint_log_spacing_noise", "cutpoint_log_spacing_value")
                if name in registry.names
            ),
            None,
        )
        if group == fit.thresholds_by and threshold_raw is not None:
            count = math.prod(registry[threshold_raw].constrained_shape)
            if count % len(levels):
                raise TypeError("unsupported grouped threshold layout")
            sites[threshold_raw] = count // len(levels)
            if threshold_raw == "cutpoint_log_spacing_noise":
                standard_normal_sites.add(threshold_raw)
        if group == baseline_group:
            count = math.prod(registry["sbhaz"].constrained_shape)
            if count % len(levels):
                raise TypeError("unsupported grouped baseline layout")
            sites["sbhaz"] = count // len(levels)
        covariance = fit.group_covariances.get(group)
        covariance_laws = tuple(
            law
            for law in (
                fit.prediction_plan.group_laws
                if fit.prediction_plan is not None
                else ()
            )
            if law.group == group and law.level.basis is not None
        )
        if covariance is not None:
            if any(key is None for key in keys):
                raise ValueError(
                    f"supplied covariance requires explicit labeled identities for group {group!r}"
                )
            covariance.subset(tuple(key for key in keys if key is not None))
            for law in covariance_laws:
                basis = law.level.basis
                assert basis is not None
                fitted_covariance = covariance.subset(basis.levels).tensor()
                factor = basis.tensors()[2]
                if not torch.allclose(
                    fitted_covariance,
                    factor @ factor.T,
                    rtol=256 * torch.finfo(torch.float64).eps,
                    atol=8 * _tolerance(fitted_covariance),
                ):
                    raise ValueError(
                        f"prediction covariance for group {group!r} changes the fitted principal submatrix"
                    )
        target_basis, conditional_basis, mean_map = None, None, ()
        if covariance is not None:
            requested = tuple(key for key in keys if key is not None)
            target_basis = covariance.basis(requested)
            new_labels = tuple(label for label in requested if label not in positions)
            if new_labels:
                mean_tensor, conditional_basis = covariance.conditional(
                    levels, new_labels
                )
                mean_map = tuple(tuple(row) for row in mean_tensor.tolist())
        result[group] = GroupRemap(
            group=group,
            fitted_size=len(levels),
            fitted_indices=tuple(
                -1 if key is None else positions.get(key, -1) for key in keys
            ),
            row_indices=torch.tensor(row_codes, dtype=torch.long),
            sites=sites,
            standard_normal_sites=frozenset(standard_normal_sites),
            levels=tuple(
                key
                if covariance is not None and key is not None
                else f"prediction_{index}"
                for index, key in enumerate(keys)
            ),
            covariance=covariance,
            covariance_laws=covariance_laws,
            covariance_target=target_basis,
            covariance_conditional=conditional_basis,
            covariance_mean_map=mean_map,
            member_indices=tuple(
                torch.tensor(codes, dtype=torch.long) for codes in member_rows
            )
            if membership is not None
            else (),
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
        group_levels={group: groups[group].levels for group in formula.group_levels}
    )


def select_contributions(
    module: Module,
    formulas: Mapping[str, Formula],
    selected: frozenset[tuple[str, str]],
    *,
    prefixes: Mapping[str, str] | None = None,
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
    prefixes : Mapping[str, str] or None
        Retained coefficient namespaces for every additive graph leaf.

    Returns
    -------
    Module
        A module retaining the fitted latent prior but excluding selected-out
        effects from its response predictor.
    """
    from quivers.formulas._ast import _factor

    excluded = set()
    for parameter, formula in formulas.items():
        prefix = (
            prefixes[parameter]
            if prefixes is not None
            else ""
            if parameter == "mu"
            else f"dpar_{parameter}_"
        )
        for term in formula.random_terms:
            if (term.group, term.slope) in selected:
                continue
            qgroup, qslope = _qvr_name(term.group), _qvr_name(term.slope)
            latent = (
                f"alpha_{qgroup}"
                if term.slope == "Intercept"
                else f"beta_{qgroup}_{qslope}"
            )
            excluded.add(f"{prefix}{latent}_per_row")
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
        -1 in group.fitted_indices and group.covariance is None
        for group in context.groups.values()
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
            from quivers.inference._prior_projection import project_prior_sites

            sampled = trace(
                context.prior_program
                if context.prior_program is not None
                else project_prior_sites(context.program, group_names),
                torch.zeros(1, 1, dtype=torch.long),
                dict(context.observations)
                | {
                    name: value
                    for name, value in latents.items()
                    if name not in group_names
                },
            )
        for group_name, group in context.groups.items():
            if group.covariance is not None:
                result.update(
                    _remap_covariance_group(
                        group,
                        latents,
                        shapes,
                        policy,
                        fixed_donors,
                        context.observations,
                    )
                )
                for name in group.sites:
                    if name not in shapes:
                        result.pop(name, None)
                continue
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


def _remap_covariance_group(
    group: GroupRemap,
    latents: dict[str, torch.Tensor],
    shapes: Mapping[str, tuple[int, ...]],
    policy: str,
    fixed_donors: dict[str, torch.Tensor],
    observations: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Condition new levels on the fitted ones, then encode the compact support.

    Each block is read before its column factor, as rows ``G`` with the
    known covariance between levels and independent standard coordinates.
    The rows of new levels are drawn from their Gaussian conditional on the
    fitted rows, and the compact rows are encoded in the support of the
    prediction levels, where the block's law applies the column factor of
    each level.

    Parameters
    ----------
    group : GroupRemap
        Labeled covariance and the fitted laws of its blocks.
    latents : dict[str, torch.Tensor]
        One posterior draw in original support coordinates.
    shapes : Mapping[str, tuple[int, ...]]
        Specialized native site geometry.
    policy : str
        Gaussian innovations or empirical retained support-coordinate donors.
    fixed_donors : dict[str, torch.Tensor]
        Request-owned indices for the old_levels innovation policy.
    observations : Mapping[str, torch.Tensor]
        Prediction inputs, holding the ``by`` level of each compact level.

    Returns
    -------
    dict[str, torch.Tensor]
        Compact raw bindings preserving all known physical coefficients.

    Raises
    ------
    ValueError
        If an empirical policy has no stochastic fitted support to resample.
    """
    covariance = group.covariance
    assert covariance is not None
    indices = torch.tensor(group.fitted_indices, dtype=torch.long)
    new = indices < 0
    new_labels = tuple(
        label for label, fresh in zip(group.levels, new.tolist(), strict=True) if fresh
    )
    target_basis = group.covariance_target
    assert target_basis is not None
    conditional = group.covariance_conditional
    empirical_donors = None
    if (
        new_labels
        and conditional is not None
        and conditional.rank
        and policy != "gaussian"
    ):
        fitted_basis = group.covariance_laws[0].level.basis
        assert fitted_basis is not None
        fitted_rank = fitted_basis.rank
        if not fitted_rank:
            raise ValueError(
                "empirical new-group prediction has no fitted support innovations; use sample_new_levels='gaussian'"
            )
        key = f"{group.group}:support"
        if policy == "old_levels":
            if key not in fixed_donors:
                fixed_donors[key] = torch.randint(fitted_rank, (conditional.rank,))
            empirical_donors = fixed_donors[key]
        else:
            empirical_donors = torch.randint(fitted_rank, (conditional.rank,))
    output = {}
    for law in group.covariance_laws:
        basis = law.level.basis
        assert basis is not None
        by = law.level.by
        # The column factor of a compact level is that of its by level.
        codes = observations[by.input] if by is not None else None
        fitted = support_gaussian(latents, law, sample_ndims=0)
        compact = fitted[indices.clamp_min(0)].clone()
        if new_labels:
            assert conditional is not None
            mean_map = torch.tensor(
                group.covariance_mean_map, dtype=fitted.dtype, device=fitted.device
            )
            expectation = mean_map @ fitted
            if conditional.rank:
                _, _, factor = conditional.tensors(like=fitted)
                if policy == "gaussian":
                    noise = torch.randn(
                        (conditional.rank, law.width),
                        dtype=fitted.dtype,
                        device=fitted.device,
                    )
                else:
                    assert empirical_donors is not None
                    innovations = encode_support(
                        fitted, basis, latents, law, parameterization="noncentered"
                    )
                    noise = innovations[empirical_donors.to(innovations.device)]
                expectation = expectation + factor @ noise
            compact[new.to(compact.device)] = expectation
        encoded = encode_support(
            compact,
            target_basis,
            latents,
            law,
            parameterization=law.parameterization,
            codes=codes,
        )
        site = law.members[0].raw_site
        # A zero-rank fitting block can gain supported future dimensions.
        if target_basis.rank:
            site = site or f"group_covariance_{_qvr_name(law.identity)}"
            output[site] = encoded.reshape(shapes[site])
    return output
