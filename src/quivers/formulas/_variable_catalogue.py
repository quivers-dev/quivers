"""One catalogue of brms variable names for the sites of every model term.

brms reports every parameter as a scalar variable: a vector parameter is
``name[i]``, a matrix ``name[i,j]``, a correlation matrix one variable per
pair, and a simplex of mixing proportions one variable per component. Each
term of a formula model registers its overrideable sites as `PriorTarget`
values. This module names those registrations, so a single-response fit and a
joint fit list the same variables for the same term and no variable keeps an
event axis:

- `class_variables` names a hyperparameter target from its prior class
  (``ar[1]``, ``theta1``, ``cortime__a__b``, ``df_g``). The terms of a
  predictor name their own draws (`quivers.formulas._name_scope`).
- `shrinkage_variables` names the auxiliary sites of horseshoe and R2D2
  priors (``hs_global``, ``hs_local[k]``, ``R2D2_R2``, ``R2D2_phi[k]``).
- `native_variables` lists every remaining site under its QVR name, one
  variable per coordinate.
- `brms_order` sorts the named variables into brms's class order.

Cost is one pass over the retained draws; no model replay occurs.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
import itertools
import re

import didactic.api as dx
import torch

from quivers._values import ValueModel
from quivers.formulas._group_law import GroupBlockLaw, GroupMemberLaw
from quivers.formulas._priors import PriorTarget
from quivers.formulas._smooth import SmoothState
from quivers.formulas.formula import Formula, ThresholdGroups

#: brms's default character substitutions in coefficient names (R/misc.R ``rename``).
_RENAMES: tuple[tuple[str, str], ...] = (
    (" ", ""),
    ("(", ""),
    (")", ""),
    ("[", ""),
    ("]", ""),
    (",", ""),
    ('"', ""),
    ("'", ""),
    ("?", ""),
    ("+", "P"),
    ("-", "M"),
    ("*", "MU"),
    ("/", "D"),
    ("^", "E"),
    ("=", "EQ"),
    ("$", "USD"),
)

#: Prior classes whose site is the Cholesky factor of a correlation matrix,
#: mapped to the brms class of the reported correlations: the time
#: correlations of ``unstr()`` and the category correlations of a
#: logistic-normal response.
CHOLESKY_CLASSES: dict[str, str] = {"Lcortime": "cortime", "rescor": "lncor"}

#: Prior classes brms reports as ``<class>[lag]``, one site per lag.
_LAGGED_CLASSES = frozenset({"ar", "ma"})

#: brms names of the auxiliary sites of a shrinkage prior block, by the
#: suffix of the site after its block prefix.
_SHRINKAGE_NAMES: dict[str, str] = {
    "_global": "hs_global",
    "_slab": "hs_slab",
    "_local": "hs_local",
    "_R2": "R2D2_R2",
    "_allocation": "R2D2_phi",
}

_SHRINKAGE_SITE = re.compile(
    r"^prior_(?P<block>[A-Za-z0-9_]+?)(?P<suffix>_global|_slab|_local|_R2|_allocation)$"
)


def brms_rename(label: str) -> str:
    """Apply brms's default renaming of special characters in a coefficient.

    Parameters
    ----------
    label : str
        Design column or coefficient label.

    Returns
    -------
    str
        The label with brms's substitutions applied in brms's order.
    """
    for pattern, replacement in _RENAMES:
        label = label.replace(pattern, replacement)
    return label


def smooth_label(term: str) -> str:
    """Return brms's label of a smooth term.

    brms labels a smooth by its function name followed by its covariates and
    its ``by`` variable (``frame_sm``), so ``s(x, k = 5)`` is ``sx``,
    ``t2(x, z)`` is ``t2xz`` and ``s(x, g, bs = "fs")`` is ``sxg``. Other
    arguments do not enter the label.

    Parameters
    ----------
    term : str
        Authored smooth term, a call of ``s`` or ``t2``.

    Returns
    -------
    str
        The label with brms's character renaming.

    Raises
    ------
    ValueError
        If the term is not a call with a plain function name.
    """
    try:
        call = ast.parse(term.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"smooth term {term!r} is not a function call") from error
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"smooth term {term!r} is not a function call")
    variables = [ast.unparse(argument) for argument in call.args]
    variables.extend(
        ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg == "by" and isinstance(keyword.value, ast.Name)
    )
    return call.func.id + brms_rename("".join(variables))


def make_unique(names: Sequence[str]) -> list[str]:
    """Repair duplicated names as R's ``make.unique(names, "__")`` does.

    brms repairs the parameter names of a fit this way, so the second smooth
    of one covariate is ``sds_sx_1__1`` and its coefficients are
    ``s_sx_1[1]__1``.

    Parameters
    ----------
    names : Sequence[str]
        Names in brms's parameter order.

    Returns
    -------
    list[str]
        The names with every later duplicate suffixed by ``__<count>``, the
        count rising from one until the name is unused.
    """
    taken = set(names)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
            continue
        count = counts.get(name, 0) + 1
        while f"{name}__{count}" in taken:
            count += 1
        counts[name] = count
        taken.add(f"{name}__{count}")
        result.append(f"{name}__{count}")
    return result


class SmoothNames(ValueModel):
    """brms variable names of the smooth terms of one predictor.

    Attributes
    ----------
    population : dict[str, str]
        Name of each unpenalized smooth column (``bs_sx_1``, or
        ``bs_sx:ga_1`` for a level of a factor ``by``), by column label.
    scales : dict[str, str]
        Name of each smoothing standard deviation (``sds_sx_1``,
        ``sds_sxga_1``), by the local name of its scale site.
    coefficients : dict[str, tuple[str, ...]]
        Names of each penalized block's coefficients (``s_sx_1[k]``), by
        the block's design identifier.
    """

    population: dict[str, str] = dx.field(default_factory=dict)
    scales: dict[str, str] = dx.field(default_factory=dict)
    coefficients: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)


def smooth_names(
    predictor: Formula, states: Mapping[str, SmoothState], *, infix: str
) -> SmoothNames:
    """Name the smooth columns, scales and coefficients of one predictor.

    brms labels a smooth by its function and variables (`smooth_label`) and
    appends the level to the label of each block of a factor ``by``
    (``tidy_smef``). Penalties are counted within a block. An unpenalized
    column takes mgcv's block label, which keeps the colon before the ``by``
    part. Names that coincide, as for two smooths of one covariate, are
    repaired with `make_unique` in brms's parameter order: unpenalized
    columns, then scales, then coefficients.

    Parameters
    ----------
    predictor : Formula
        One evaluated predictor.
    states : Mapping[str, SmoothState]
        Retained smooth states of the predictor's design, by term label.
    infix : str
        Parameter and response part of the names.

    Returns
    -------
    SmoothNames
        The names of every smooth term of the predictor.

    Raises
    ------
    ValueError
        If the predictor's smooth columns or blocks do not match the
        retained states.
    """
    population: list[tuple[str, str]] = []
    scales: list[tuple[str, str]] = []
    coefficients: list[tuple[str, str]] = []
    for term, state in states.items():
        base = smooth_label(term)
        fixed = [
            (brms_rename(block.label), position + 1)
            for block in state.blocks
            for position in range(block.fixed_width)
        ]
        columns = [column for column in predictor.fixed_columns if column.term == term]
        groups = [
            (
                base + ("" if block.level is None else brms_rename(block.level)),
                penalty + 1,
            )
            for block in state.blocks
            for penalty in range(len(block.random_widths))
        ]
        blocks = [
            block
            for block in predictor.design_terms
            if block.term == term and block.gp is None
        ]
        if len(columns) != len(fixed) or len(blocks) != len(groups):
            raise ValueError(
                f"smooth term {term!r} retains {len(fixed)} unpenalized columns "
                f"and {len(groups)} penalized blocks, but its predictor has "
                f"{len(columns)} and {len(blocks)}"
            )
        population.extend(
            (column.name, f"bs_{infix}{label}_{position}")
            for column, (label, position) in zip(columns, fixed, strict=True)
        )
        for block, (label, penalty) in zip(blocks, groups, strict=True):
            site = (block.scale_group or block.qvr_name) + "_scale"
            if all(site != known for known, _ in scales):
                scales.append((site, f"sds_{infix}{label}_{penalty}"))
            coefficients.extend(
                (block.qvr_name, f"s_{infix}{label}_{penalty}[{index + 1}]")
                for index in range(block.data.shape[1])
            )
    names = make_unique(
        [name for _, name in population]
        + [name for _, name in scales]
        + [name for _, name in coefficients]
    )
    by_block: dict[str, list[str]] = {}
    for (block_name, _), name in zip(
        coefficients, names[len(population) + len(scales) :], strict=True
    ):
        by_block.setdefault(block_name, []).append(name)
    return SmoothNames(
        population={
            column: name
            for (column, _), name in zip(
                population, names[: len(population)], strict=True
            )
        },
        scales={
            site: name
            for (site, _), name in zip(
                scales,
                names[len(population) : len(population) + len(scales)],
                strict=True,
            )
        },
        coefficients={block: tuple(found) for block, found in by_block.items()},
    )


def scalar_coordinates(stem: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split draws of one site into brms's scalar variables.

    Parameters
    ----------
    stem : str
        Variable name without an index.
    value : torch.Tensor
        Draws on ``(chain, draw, *event)`` axes.

    Returns
    -------
    dict[str, torch.Tensor]
        ``stem`` for a scalar site; otherwise one ``stem[i]`` or
        ``stem[i,j,...]`` entry per coordinate, with brms's indices starting
        at one, in row-major order. Every value has shape ``(chain, draw)``.
    """
    event = tuple(value.shape[2:])
    if not event:
        return {stem: value}
    return {
        f"{stem}[{','.join(str(i + 1) for i in index)}]": value[(..., *index)]
        for index in itertools.product(*(range(size) for size in event))
    }


def correlation_pairs(
    prefix: str, labels: Sequence[str], factor: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Name the correlations of a Cholesky factor as brms names pairs.

    Parameters
    ----------
    prefix : str
        Class and grouping part of the name, such as ``cor_g`` or ``rescor``.
    labels : Sequence[str]
        Coordinate labels in factor order.
    factor : torch.Tensor
        Lower Cholesky factors on ``(chain, draw, K, K)`` axes.

    Returns
    -------
    dict[str, torch.Tensor]
        ``<prefix>__<a>__<b>`` for every pair ``a`` before ``b``, ordered by
        the later coordinate as brms orders them.

    Raises
    ------
    ValueError
        If the labels do not match the factor's dimension.
    """
    if factor.ndim != 4 or factor.shape[-1] != len(labels):
        raise ValueError(
            f"correlation factor of shape {tuple(factor.shape)} does not match "
            f"the {len(labels)} coordinates of {prefix!r}"
        )
    matrix = factor @ factor.transpose(-1, -2)
    return {
        f"{prefix}__{labels[j]}__{labels[k]}": matrix[..., j, k]
        for k in range(len(labels))
        for j in range(k)
    }


def by_suffixes(law: GroupBlockLaw) -> tuple[str, ...]:
    """Name the ``by`` levels of a block as brms's coefficient suffixes.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.

    Returns
    -------
    tuple[str, ...]
        ``:<variable><level>`` per ``by`` level; one empty suffix for a block
        without a partition.
    """
    by = law.level.by
    if by is None:
        return ("",)
    return tuple(":" + brms_rename(by.variable + str(level)) for level in by.levels)


def scale_variables(
    law: GroupBlockLaw, member: GroupMemberLaw, stem: str, value: torch.Tensor
) -> tuple[dict[str, torch.Tensor], ...]:
    """Name the standard deviations of one group-level coefficient.

    Parameters
    ----------
    law : GroupBlockLaw
        Block of the coefficient.
    member : GroupMemberLaw
        The coefficient.
    stem : str
        ``sd_<group>__<coefficient>``.
    value : torch.Tensor
        Draws of its scale on ``(chain, draw)`` axes and the scale's axis.

    Returns
    -------
    tuple[dict[str, torch.Tensor], ...]
        Named scalars of each ``by`` level, or of the one shared scale; a
        coefficient over components adds its index.
    """
    if member.scale.by_axis is None:
        return (scalar_coordinates(stem, value),)
    suffixes = by_suffixes(law)
    levels = value.reshape(*value.shape[:2], len(suffixes), member.width)
    return tuple(
        scalar_coordinates(
            stem + suffix,
            levels[:, :, index, 0] if member.axis is None else levels[:, :, index],
        )
        for index, suffix in enumerate(suffixes)
    )


def correlation_variables(
    law: GroupBlockLaw, terms: Sequence[str], draws: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Name the correlations between the coefficients of one block.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.
    terms : Sequence[str]
        brms coefficient name of each member.
    draws : Mapping[str, torch.Tensor]
        Draws holding the block's Cholesky factors.

    Returns
    -------
    dict[str, torch.Tensor]
        One correlation per pair of coordinates and, with one correlation
        per ``by`` level, per level; empty for independent coordinates.
    """
    coordinates = [
        term if member.width == 1 else f"{term}[{index + 1}]"
        for term, member in zip(terms, law.members, strict=True)
        for index in range(member.width)
    ]
    sites = law.correlation.sites
    suffixes = by_suffixes(law) if len(sites) > 1 else ("",) * len(sites)
    result: dict[str, torch.Tensor] = {}
    for site, suffix in zip(sites, suffixes, strict=True):
        if site in draws:
            result |= correlation_pairs(
                f"cor_{law.group}",
                [coordinate + suffix for coordinate in coordinates],
                draws[site],
            )
    return result


def class_variables(
    target: PriorTarget,
    value: torch.Tensor,
    *,
    infix: str,
    labels: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    """Name the draws of one hyperparameter target from its prior class.

    Parameters
    ----------
    target : PriorTarget
        A registered site that is not a population coefficient, a group
        standard deviation, or a group correlation.
    value : torch.Tensor
        Its draws on ``(chain, draw, *event)`` axes.
    infix : str
        Parameter and response part of the name, such as ``""``,
        ``"sigma_"`` or ``"y_"``, ending in an underscore when nonempty.
    labels : Sequence[str]
        Coordinate labels of a correlation class (time levels of ``unstr``).

    Returns
    -------
    dict[str, torch.Tensor]
        brms-named scalar variables. A class without a coefficient label
        appends the response of a joint model, and a class without a brms
        counterpart keeps its QVR site name, one variable per coordinate.

    Raises
    ------
    ValueError
        If a correlation class has no coordinate labels.
    """
    kind = target.class_
    # brms appends the response of a joint model to a class without a
    # coefficient label (``sderr_y``, ``ar_y[1]``).
    owner = "_" + infix.rstrip("_") if infix else ""
    if kind in CHOLESKY_CLASSES:
        if not labels:
            raise ValueError(
                f"correlation site {target.site!r} retains no coordinate labels"
            )
        return correlation_pairs(CHOLESKY_CLASSES[kind] + owner, labels, value)
    if kind == "df" and target.group is not None:
        stem = f"df_{target.group}"
    elif kind in _LAGGED_CLASSES and re.fullmatch(rf"(.*_)?{kind}_[0-9]+", target.site):
        return {f"{kind}{owner}[{target.site.rsplit('_', 1)[1]}]": value}
    elif kind == "theta" and value.ndim == 3:
        return {f"theta{k + 1}{owner}": value[..., k] for k in range(value.shape[-1])}
    else:
        stem = target.site + owner
    return scalar_coordinates(stem, value)


def shrinkage_blocks(draws: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    """Return the labels of the shrinkage prior blocks among the draws.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained draws.

    Returns
    -------
    tuple[str, ...]
        Block labels in site order.
    """
    blocks: list[str] = []
    for site in draws:
        match = _SHRINKAGE_SITE.match(site)
        if match is not None and match["block"] not in blocks:
            blocks.append(match["block"])
    return tuple(blocks)


def shrinkage_scale_site(block: str, site: str) -> str:
    """Return the binding that holds a coefficient's shrinkage prior scale.

    Parameters
    ----------
    block : str
        Shrinkage block label.
    site : str
        Site of a coefficient in the block.

    Returns
    -------
    str
        Name of the deterministic scale binding of the emitted program.
    """
    return f"prior_{block}_scale_{site}"


def shrinkage_variables(
    draws: Mapping[str, torch.Tensor],
    consumed: Iterable[str],
    scales: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Name the auxiliary sites of horseshoe and R2D2 prior blocks.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained draws.
    consumed : Iterable[str]
        Sites already named.
    scales : Mapping[str, Mapping[str, torch.Tensor]] or None
        Per block, the prior standard deviation of each shrunk coefficient
        keyed by the coefficient's brms name without its class prefix.

    Returns
    -------
    tuple[dict[str, torch.Tensor], set[str]]
        The variables and the sites they read: ``hs_global``, ``hs_slab``
        and ``hs_local[k]`` of a horseshoe block, ``R2D2_R2`` and
        ``R2D2_phi[k]`` of an R2D2 block, and for either the coefficient
        scales ``sdb_<coef>`` that brms reports. An R2D2 block adds
        ``R2D2_tau2``, the sum of its squared coefficient scales, since the
        allocations ``R2D2_phi`` sum to one. The first block has brms's bare
        names; a further block appends its label (``hs_global_<block>``).
    """
    taken = set(consumed)
    result: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    blocks: list[str] = []
    for site, value in draws.items():
        match = _SHRINKAGE_SITE.match(site)
        if match is None or site in taken:
            continue
        if match["block"] not in blocks:
            blocks.append(match["block"])
        suffix = "" if blocks[0] == match["block"] else "_" + match["block"]
        result |= scalar_coordinates(_SHRINKAGE_NAMES[match["suffix"]] + suffix, value)
        used.add(site)
    for block, members in (scales or {}).items():
        if block not in blocks or not members:
            continue
        suffix = "" if blocks[0] == block else "_" + block
        if f"prior_{block}_R2" in draws:
            result[f"R2D2_tau2{suffix}"] = torch.stack(
                [
                    value.reshape(*value.shape[:2], -1).square().sum(-1)
                    for value in members.values()
                ]
            ).sum(0)
        for name, value in members.items():
            result |= scalar_coordinates(f"sdb_{name}{suffix}", value)
    return result, used


def threshold_variables(
    thresholds: torch.Tensor,
    *,
    infix: str = "",
    groups: ThresholdGroups | None = None,
) -> dict[str, torch.Tensor]:
    """Name ordinal thresholds as brms does.

    Parameters
    ----------
    thresholds : torch.Tensor
        brms-scale thresholds of every training row, shaped
        ``(chain, draw, row, threshold)``.
    infix : str
        Response and parameter part of the name, such as ``y_`` in a
        multivariate model or ``mu1_`` for a mixture component.
    groups : ThresholdGroups or None
        Resolved threshold groups of the training rows (``thres(n, gr)``),
        or None for one threshold vector shared by all rows.

    Returns
    -------
    dict[str, torch.Tensor]
        ``b_<infix>Intercept[<k>]`` read from the first row, or with groups
        ``b_<infix>Intercept[<group>,<k>]`` read from each group's first row
        without the padded positions, on chain and draw axes.

    Raises
    ------
    ValueError
        If a fitted group has no training row.
    """
    if groups is None:
        return {
            f"b_{infix}Intercept[{position + 1}]": thresholds[:, :, 0, position]
            for position in range(thresholds.shape[-1])
        }
    codes = torch.as_tensor(groups.column.data.copy())
    named: dict[str, torch.Tensor] = {}
    for index, (label, count) in enumerate(
        zip(groups.levels, groups.counts, strict=True)
    ):
        rows = torch.nonzero(codes == index).reshape(-1)
        if not len(rows):
            raise ValueError(f"threshold group {label!r} has no training row")
        for position in range(count):
            named[f"b_{infix}Intercept[{label},{position + 1}]"] = thresholds[
                :, :, int(rows[0]), position
            ]
    return named


def native_variables(
    draws: Mapping[str, torch.Tensor], consumed: Iterable[str]
) -> dict[str, torch.Tensor]:
    """List every site not yet named under its QVR name.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained draws.
    consumed : Iterable[str]
        Sites already named.

    Returns
    -------
    dict[str, torch.Tensor]
        One scalar variable per coordinate of each remaining site.
    """
    taken = set(consumed)
    result: dict[str, torch.Tensor] = {}
    for site, value in draws.items():
        if site not in taken:
            result |= scalar_coordinates(site, value)
    return result


#: brms's parameter classes in reporting order, before and after the
#: family's distributional parameters (R/brmsfit-helpers.R ``reorder_pars``).
_CLASSES_BEFORE_DPARS: tuple[str, ...] = (
    "b",
    "bs",
    "bsp",
    "bcs",
    "ar",
    "ma",
    "sderr",
    "lagsar",
    "errorsar",
    "car",
    "rhocar",
    "sdcar",
    "cosy",
    "cortime",
    "sd",
    "cor",
    "df",
    "sds",
    "sdgp",
    "lscale",
)
_CLASSES_AFTER_DPARS: tuple[str, ...] = (
    "hs",
    "R2D2",
    "sdb",
    "sdbsp",
    "sdbs",
    "sdar",
    "sdma",
    "lncor",
    "Intercept",
    "tmp",
    "rescor",
    "delta",
    "simo",
    "r",
    "s",
    "zgp",
    "rcar",
    "sbhaz",
    "Ymi",
    "Yl",
    "meanme",
    "sdme",
    "corme",
    "Xme",
    "prior",
    "lprior",
    "lp",
)
_INTERCEPT_PARAMETER = re.compile(r"_Intercept(_[0-9]+)?$")


def brms_order(
    names: Mapping[str, torch.Tensor], dpars: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Order variables as brms's ``reorder_pars`` orders a fit's parameters.

    The class of a variable is the text of its parameter name before the
    first underscore. brms sorts parameters by class, in a fixed class order
    that holds the family's distributional parameters between ``lscale``
    and ``hs``, and within a class lists the parameters whose name ends in
    ``_Intercept`` first. The sort is stable, so parameters of one class
    otherwise keep their given order, and a class brms does not list comes
    last. Costs O(n log n) for ``n`` variables.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        Named draws, with each class in the order of brms's Stan program.
    dpars : Sequence[str]
        The family's distributional parameter names in brms's order.

    Returns
    -------
    dict[str, torch.Tensor]
        The same variables in brms's order.
    """
    classes: dict[str, int] = {}
    for name in (*_CLASSES_BEFORE_DPARS, *dpars, *_CLASSES_AFTER_DPARS):
        classes.setdefault(name, len(classes))

    def rank(name: str) -> tuple[int, bool]:
        parameter = name.split("[", 1)[0]
        return (
            classes.get(parameter.split("_", 1)[0], len(classes)),
            _INTERCEPT_PARAMETER.search(parameter) is None,
        )

    return {name: names[name] for name in sorted(names, key=rank)}


def checked_variables(
    names: Mapping[str, torch.Tensor], shape: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Check the catalogue's contract: every variable is one scalar per draw.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        Named draws.
    shape : tuple[int, int]
        Chain and draw counts.

    Returns
    -------
    dict[str, torch.Tensor]
        The same variables.

    Raises
    ------
    ValueError
        If a variable keeps an event axis or has other sample axes.
    """
    for name, value in names.items():
        if tuple(value.shape) != shape:
            raise ValueError(
                f"variable {name!r} has shape {tuple(value.shape)}; every "
                f"variable is one scalar on the (chain, draw) axes {shape}"
            )
    return dict(names)


__all__: list[str] = []
