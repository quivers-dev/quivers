"""Select varying effects and remap only the grouping levels used in prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping

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
from quivers.formulas._group_plan import block_covariance
from quivers.formulas._grouping_emission import by_axis, compact_option
from quivers.formulas._group_law import (
    KnownCovariance,
    GroupBlockLaw,
    LevelSite,
    encode_support,
    block_gaussian,
)
from quivers.formulas.formula import (
    Formula,
    _normalize_cs_intercept,
    _random_blocks,
)

if TYPE_CHECKING:
    from quivers.formulas._prediction import PredictionContext, PredictionDesign


type NewLevelPolicy = Literal[
    "uncertainty",
    "old_levels",
    "gaussian",
    "feasible_uncertainty",
    "feasible_gaussian",
]


class LevelPartition(ValueModel):
    """The levels of one ``by`` variable on the fitted and prediction axes.

    Parameters
    ----------
    axis : str
        Axis over the levels of the variable.
    codes : tuple[int, ...]
        The ``by`` level of each fitted level.
    compact : tuple[int, ...]
        The ``by`` level of each compact level.
    """

    axis: str
    codes: tuple[int, ...]
    compact: tuple[int, ...]


class CovarianceRemap(ValueModel):
    """One known level covariance on the prediction levels.

    Parameters
    ----------
    name : str
        Name of the covariance among those of the factor.
    covariance : GroupCovariance
        Complete labeled covariance, including requested future identities.
    laws : tuple[GroupBlockLaw, ...]
        Fitted blocks with the covariance.
    target : GroupCovarianceBasis
        Support factors of the compact levels.
    conditional : GroupCovarianceBasis or None
        Support factors of the new levels given the fitted ones.
    mean_map : tuple[tuple[float, ...], ...]
        Conditional mean map from all fitted levels.
    """

    name: str
    covariance: GroupCovariance = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    laws: tuple[GroupBlockLaw, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    target: GroupCovarianceBasis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditional: GroupCovarianceBasis | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    mean_map: tuple[tuple[float, ...], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


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
    level_sites : tuple[LevelSite, ...]
        Sample sites holding values for every level of the factor.
    levels : tuple[str, ...]
        Compact labels; supplied covariance retains authored identities.
    partitions : tuple[LevelPartition, ...]
        Partitions of the levels by ``by`` variables.
    covariances : tuple[CovarianceRemap, ...]
        Known covariances between the levels.
    member_indices : tuple[torch.Tensor, ...]
        Compact codes of every member of a brms ``mm()`` factor, in member
        order; empty for an ordinary factor, whose codes are ``row_indices``.
    """

    group: str
    fitted_size: int
    fitted_indices: tuple[int, ...]
    row_indices: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    level_sites: tuple[LevelSite, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    levels: tuple[str, ...] = ()
    partitions: tuple[LevelPartition, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariances: tuple[CovarianceRemap, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    member_indices: tuple[torch.Tensor, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def sites(self) -> dict[str, int]:
        """Return the number of values per level of each level-indexed site.

        Returns
        -------
        dict[str, int]
            Widths by site name.
        """
        return {site.site: site.width for site in self.level_sites}

    @property
    def standard_normal_sites(self) -> frozenset[str]:
        """Return the sites whose values are standard-normal innovations.

        Returns
        -------
        frozenset[str]
            Sites admitting direct standard-normal draws for new levels.
        """
        return frozenset(site.site for site in self.level_sites if site.standard_normal)

    @property
    def support_sites(self) -> frozenset[str]:
        """Return the sites in support coordinates of a known covariance.

        Returns
        -------
        frozenset[str]
            Raw sites whose size is the rank of the covariance in use.
        """
        return frozenset(
            law.support_site
            for known in self.covariances
            for law in known.laws
            if law.support_site is not None
        )

    def donor_indices(self) -> torch.Tensor:
        """Draw a fitted level for each new compact level.

        A donor shares the ``by`` levels of the new level, whose standard
        deviations and correlations its coefficients have, as in brms.

        Returns
        -------
        torch.Tensor
            Fitted level of each compact level, with a uniformly drawn donor
            in the place of each new level.

        Raises
        ------
        ValueError
            If no fitted level has the ``by`` levels of a new one.
        """
        indices = torch.tensor(self.fitted_indices, dtype=torch.long)
        new = indices < 0
        fitted = (
            list(zip(*(partition.codes for partition in self.partitions), strict=True))
            or [()] * self.fitted_size
        )
        compact = list(
            zip(*(partition.compact for partition in self.partitions), strict=True)
        ) or [()] * len(self.fitted_indices)
        chosen = [code for code, fresh in zip(compact, new.tolist()) if fresh]
        for code in sorted(set(chosen)):
            candidates = torch.tensor(
                [index for index, value in enumerate(fitted) if value == code],
                dtype=torch.long,
            )
            if not len(candidates):
                raise ValueError(
                    f"no fitted level of {self.group!r} has the by levels of a new "
                    "level; use sample_new_levels='gaussian'"
                )
            positions = torch.tensor(
                [
                    index
                    for index, value in enumerate(compact)
                    if value == code and bool(new[index])
                ],
                dtype=torch.long,
            )
            indices[positions] = candidates[
                torch.randint(len(candidates), (len(positions),))
            ]
        return indices


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
    fit: PredictionDesign,
    frame: pd.DataFrame,
    selected: frozenset[tuple[str, str]],
    allow_new_levels: bool,
) -> dict[str, GroupRemap]:
    """Encode fresh identities once and retain only levels used in these rows.

    Parameters
    ----------
    fit : PredictionDesign
        Fitted formulas and reparameterization policy.
    frame : pandas.DataFrame
        Prediction rows.
    selected : frozenset[tuple[str, str]]
        Active grouping-factor and expanded-slope pairs.
    allow_new_levels : bool
        Permit unseen and missing grouping identities.

    Returns
    -------
    dict[str, GroupRemap]
        One mapping per grouping factor, shared across auxiliary formulas.

    Raises
    ------
    ValueError
        If new identities are forbidden or fitted shared encodings disagree.
    TypeError
        If a fitted group term has no compiled law.
    """
    predictors = {name: formula for name, _, formula in fit._predictor_items()}
    plan = fit.prediction_plan
    plan_laws = plan.group_laws if plan is not None else ()
    level_sites = plan.level_sites if plan is not None else ()
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
        laws = tuple(law for law in plan_laws if law.group == group)
        planned = {
            (member.prefix, member.slope) for law in laws for member in law.members
        }
        for name, prefix, formula in fit._predictor_items():
            for term in formula.random_terms:
                if term.group == group and (prefix, term.slope) not in planned:
                    raise TypeError(
                        f"fitted group term {term.slope!r} of {name!r} for "
                        f"{group!r} has no compiled law"
                    )
        known: dict[str, GroupCovariance] = {}
        for formula in predictors.values():
            for term in formula.random_terms:
                if term.group == group:
                    name, covariance = block_covariance(
                        group, (term,), fit.group_covariances
                    )
                    if covariance is not None:
                        known[name] = covariance
        if group in fit.group_covariances:
            known.setdefault("", fit.group_covariances[group])
        blocks: dict[str, list[GroupBlockLaw]] = {}
        for law in laws:
            if isinstance(law.level.covariance, KnownCovariance):
                blocks.setdefault(law.level.covariance.name, []).append(law)
        requested = tuple(key for key in keys if key is not None)
        new_labels = tuple(label for label in requested if label not in positions)
        covariances = []
        for name, covariance in known.items():
            covariance_laws = blocks.get(name, [])
            if any(key is None for key in keys):
                raise ValueError(
                    f"supplied covariance requires explicit labeled identities for group {group!r}"
                )
            covariance.subset(requested)
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
            conditional_basis, mean_map = None, ()
            if new_labels:
                mean_tensor, conditional_basis = covariance.conditional(
                    levels, new_labels
                )
                mean_map = tuple(tuple(row) for row in mean_tensor.tolist())
            covariances.append(
                CovarianceRemap(
                    name=name,
                    covariance=covariance,
                    laws=tuple(covariance_laws),
                    target=covariance.basis(requested),
                    conditional=conditional_basis,
                    mean_map=mean_map,
                )
            )
        remap = GroupRemap(
            group=group,
            fitted_size=len(levels),
            fitted_indices=tuple(
                -1 if key is None else positions.get(key, -1) for key in keys
            ),
            row_indices=torch.tensor(row_codes, dtype=torch.long),
            level_sites=tuple(site for site in level_sites if site.group == group),
            levels=tuple(
                key if known and key is not None else f"prediction_{index}"
                for index, key in enumerate(keys)
            ),
            covariances=tuple(covariances),
            member_indices=tuple(
                torch.tensor(codes, dtype=torch.long) for codes in member_rows
            )
            if membership is not None
            else (),
        )
        remap = remap.with_(
            partitions=tuple(
                {
                    option.options.by: LevelPartition(
                        axis=by_axis(group, option.options.by),
                        codes=option.by_codes,
                        compact=compact_option(option, frame, remap).by_codes,
                    )
                    for formula in predictors.values()
                    for option in formula.grouping.options
                    if option.group == group
                    and option.by_levels
                    and option.options.by is not None
                }.values()
            )
        )
        result[group] = remap
    return result


def specialized_formula(
    formula: Formula, groups: Mapping[str, GroupRemap], frame: pd.DataFrame
) -> Formula:
    """Replace group cardinalities while preserving fitted transforms and priors.

    Parameters
    ----------
    formula : Formula
        Fitted evaluated metadata.
    groups : Mapping[str, GroupRemap]
        Compact prediction axes.
    frame : pandas.DataFrame
        Fresh rows, holding the ``by`` level of every new grouping level.

    Returns
    -------
    Formula
        Metadata using synthetic labels for the compact group axes, with the
        ``gr()`` options coded on them.
    """
    return formula.with_(
        group_levels={group: groups[group].levels for group in formula.group_levels},
        grouping=formula.grouping.with_(
            options=tuple(
                compact_option(option, frame, groups[option.group])
                if option.group in groups
                else option
                for option in formula.grouping.options
            )
        ),
    )


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
    group_names = {
        name
        for group in context.groups.values()
        for name in (*group.sites, *group.support_sites)
    }
    policy = context.new_level_policy.removeprefix("feasible_")
    need_gaussian = policy == "gaussian" and any(
        -1 in group.fitted_indices and group.sites for group in context.groups.values()
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
            for name in group.support_sites:
                result.pop(name, None)
            for position, known in enumerate(group.covariances):
                result.update(
                    _remap_covariance_group(
                        group, known, position, latents, shapes, policy, fixed_donors
                    )
                )
            if not group.sites:
                continue
            indices = torch.tensor(group.fitted_indices, dtype=torch.long)
            new = indices < 0
            count = int(new.sum())
            if count and policy != "gaussian":
                if policy != "old_levels":
                    indices = group.donor_indices()
                else:
                    if group_name not in fixed_donors:
                        fixed_donors[group_name] = group.donor_indices()[new]
                    indices[new] = fixed_donors[group_name]
            if donors is not None and policy != "gaussian" and not group.covariances:
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
    known: CovarianceRemap,
    position: int,
    latents: dict[str, torch.Tensor],
    shapes: Mapping[str, tuple[int, ...]],
    policy: str,
    fixed_donors: dict[str, torch.Tensor],
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
        Compact prediction axis.
    known : CovarianceRemap
        Labeled covariance and the fitted laws of its blocks.
    position : int
        Position of the covariance among those of the factor.
    latents : dict[str, torch.Tensor]
        One posterior draw in original support coordinates.
    shapes : Mapping[str, tuple[int, ...]]
        Specialized native site geometry.
    policy : str
        Gaussian innovations or empirical retained support-coordinate donors.
    fixed_donors : dict[str, torch.Tensor]
        Request-owned indices for the old_levels innovation policy.

    Returns
    -------
    dict[str, torch.Tensor]
        Compact raw bindings preserving all known physical coefficients.

    Raises
    ------
    ValueError
        If an empirical policy has no stochastic fitted support to resample.
    """
    indices = torch.tensor(group.fitted_indices, dtype=torch.long)
    new = indices < 0
    new_labels = tuple(
        label for label, fresh in zip(group.levels, new.tolist(), strict=True) if fresh
    )
    target_basis = known.target
    conditional = known.conditional
    empirical_donors = None
    if (
        new_labels
        and conditional is not None
        and conditional.rank
        and policy != "gaussian"
    ):
        fitted_basis = known.laws[0].level.basis
        assert fitted_basis is not None
        fitted_rank = fitted_basis.rank
        if not fitted_rank:
            raise ValueError(
                "empirical new-group prediction has no fitted support innovations; use sample_new_levels='gaussian'"
            )
        key = f"{group.group}:support" + (f":{position}" if position else "")
        if policy == "old_levels":
            if key not in fixed_donors:
                fixed_donors[key] = torch.randint(fitted_rank, (conditional.rank,))
            empirical_donors = fixed_donors[key]
        else:
            empirical_donors = torch.randint(fitted_rank, (conditional.rank,))
    output = {}
    for law in known.laws:
        basis = law.level.basis
        assert basis is not None
        # The column factor of a compact level is that of its by level.
        by = law.level.by
        codes = (
            torch.tensor(
                next(
                    partition.compact
                    for partition in group.partitions
                    if partition.axis == by.axis
                ),
                dtype=torch.long,
            )
            if by is not None
            else None
        )
        fitted = block_gaussian(latents, law, sample_ndims=0)
        compact = fitted[indices.clamp_min(0)].clone()
        if new_labels:
            assert conditional is not None
            mean_map = torch.tensor(
                known.mean_map, dtype=fitted.dtype, device=fitted.device
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
        # A zero-rank fitting block can gain supported future dimensions.
        if target_basis.rank and law.support_site is not None:
            output[law.support_site] = encoded.reshape(shapes[law.support_site])
    return output
