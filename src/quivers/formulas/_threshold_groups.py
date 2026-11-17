"""Group-specific ordinal thresholds of brms ``thres(n, gr)``.

brms gives each level of ``gr`` its own threshold vector, of the group's own
length, with no pooling between groups: ``ordered[nthres[g]] Intercept_g``
for the cumulative family and an unconstrained vector for the sequential and
adjacent-category families. A row is scored with the vector of its group.

The emission keeps the formula compiler's identified threshold construction
in every group. A group's thresholds are a shape with mean zero (cumulative
positive spacings for the cumulative family, free offsets from the first
threshold otherwise) plus the group's location. With a formula intercept the
first group's location is zero, so the intercept carries it and the other
locations are contrasts. Without one every group has a free location. The
map from brms's vectors to these coordinates is a bijection.

Groups with fewer thresholds than the largest group are padded with finite
increasing thresholds, so every row has a full threshold vector, and the
category log masses are then reduced to the group's own categories. For the
cumulative and sequential families the padded categories partition the
group's last category, whose log mass is thus the log-sum-exp of the padded
tail. For the adjacent-category family a category's mass is proportional to
a prefix sum that involves only its own thresholds, so the padded categories
are removed before normalization. Both reductions are exact, and absent
categories have probability zero. Cost: O(groups * thresholds) per draw for
the threshold matrix and O(rows * categories) for the reduction.

The hurdle cumulative family has one category before the ordinal ones (code
zero). Its mass is never padded, and the reduction applies to the ordinal
categories after it. Every component of an ordinal mixture is emitted with
its own group thresholds over one shared group axis and one set of category
selectors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    SampleStep,
    Statement,
    TypeName,
)
from quivers.formulas._ast import (
    _add,
    _call,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
)
from quivers.formulas.family import Family
from quivers.formulas.formula import Formula, ThresholdGroups

__all__: list[str] = []

#: Nominal axis of the threshold groups.
GROUP_AXIS = "FormulaThresholdGroup"
#: Site of the group locations and its brms-style prior class.
LOCATION_SITE = "threshold_location"
#: Inputs over group and category selecting a group's categories below its
#: last one, its last category, its padded tail, and its absent categories.
BELOW_INPUT = "formula_threshold_below"
LAST_INPUT = "formula_threshold_last"
TAIL_INPUT = "formula_threshold_tail"
ABSENT_INPUT = "formula_threshold_absent"
#: Names every component of an ordinal mixture shares: the group axis and
#: the selector inputs.
THRESHOLD_GROUP_NAMES = frozenset(
    {GROUP_AXIS, BELOW_INPUT, LAST_INPUT, TAIL_INPUT, ABSENT_INPUT}
)
#: Deterministic site holding the group-by-threshold matrix.
MATRIX_SITE = "cutpoint_matrix"
#: Deterministic site holding each row's threshold vector.
ROW_SITE = "row_cutpoints"


def _integer(value: int) -> LetExprLiteral:
    """Spell an integer index literal.

    Parameters
    ----------
    value : int
        Zero-based position.

    Returns
    -------
    LetExprLiteral
        Integer literal.
    """
    return LetExprLiteral(value=value, integral=True)


def has_location_reference(formula: Formula) -> bool:
    """Report whether a formula intercept carries the first group's location.

    Parameters
    ----------
    formula : Formula
        The location predictor.

    Returns
    -------
    bool
        True when the population-level design holds an intercept column.
    """
    return any(column.is_intercept for column in formula.fixed_columns)


def threshold_family(family: Family, formula: Formula) -> Family:
    """Score a response with threshold groups on category log masses.

    Parameters
    ----------
    family : Family
        Response family.
    formula : Formula
        Evaluated response formula.

    Returns
    -------
    Family
        The family, observed through category log masses when the formula
        declares threshold groups.

    Raises
    ------
    ValueError
        If threshold groups are declared for a family without thresholds, or
        for a mixture whose components mix hurdle and other ordinal families.
    """
    if formula.modifiers.threshold_groups is None:
        return family
    if family.mixture is not None and all(
        component.ordinal is not None for component in family.mixture.components
    ):
        hurdles = {
            component.ordinal is not None
            and component.ordinal.kind == "hurdle_cumulative"
            for component in family.mixture.components
        }
        if len(hurdles) > 1:
            raise ValueError(
                "thres(gr = ...) in a mixture requires components that all have "
                "the hurdle category or all lack it, since a group's threshold "
                "count is read from one coding of the response"
            )
        # brms gives every ordinal component its own group thresholds.
        return family.with_(
            mixture=family.mixture.with_(
                components=tuple(
                    component.with_(observe_family="Categorical")
                    for component in family.mixture.components
                )
            )
        )
    if family.ordinal is None:
        raise ValueError(
            "thres(gr = ...) requires an ordinal response family (cumulative, "
            "sratio, cratio, acat, or hurdle_cumulative) or a mixture of them"
        )
    return family.with_(observe_family="Categorical")


def threshold_kinds(family: Family) -> tuple[str, ...]:
    """List the ordinal kinds a family scores its response with.

    Parameters
    ----------
    family : Family
        Response family.

    Returns
    -------
    tuple[str, ...]
        The kind of an ordinal family, the kinds of the components of an
        ordinal mixture, and nothing otherwise.
    """
    if family.ordinal is not None:
        return (family.ordinal.kind,)
    if family.mixture is None:
        return ()
    return tuple(
        component.ordinal.kind
        for component in family.mixture.components
        if component.ordinal is not None
    )


def threshold_inputs(
    groups: ThresholdGroups, kinds: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Build the category selectors of every threshold group.

    Parameters
    ----------
    groups : ThresholdGroups
        Resolved threshold groups.
    kinds : Sequence[str]
        Ordinal family kinds scoring the response: one, or one per mixture
        component.

    Returns
    -------
    dict[str, torch.Tensor]
        ``(groups, categories)`` tensors: ``-inf`` at a group's absent
        categories and zero elsewhere, and, for families whose padded
        categories partition the last category, indicators of the categories
        below the last one and of the last one, and zero on the padded tail
        with ``-inf`` elsewhere.
    """
    # The hurdle category precedes the ordinal categories and is never padded.
    extra = int(groups.extra_category)
    categories = max(groups.counts) + 1 + extra
    shape = (len(groups.counts), categories)
    absent = torch.zeros(shape, dtype=torch.float64)
    below = torch.zeros(shape, dtype=torch.float64)
    last = torch.zeros(shape, dtype=torch.float64)
    tail = torch.full(shape, float("-inf"), dtype=torch.float64)
    for index, count in enumerate(groups.counts):
        absent[index, count + extra + 1 :] = float("-inf")
        below[index, : count + extra] = 1.0
        last[index, count + extra] = 1.0
        tail[index, count + extra :] = 0.0
    if set(kinds) == {"acat"}:
        return {ABSENT_INPUT: absent}
    return {
        ABSENT_INPUT: absent,
        BELOW_INPUT: below,
        LAST_INPUT: last,
        TAIL_INPUT: tail,
    }


def threshold_selectors(
    observations: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Select the fitted category selectors among a fit's inputs.

    Parameters
    ----------
    observations : Mapping[str, torch.Tensor]
        Inputs of the fitted program.

    Returns
    -------
    dict[str, torch.Tensor]
        The selector inputs of the threshold groups, empty without groups.
    """
    return {
        name: observations[name]
        for name in (ABSENT_INPUT, BELOW_INPUT, LAST_INPUT, TAIL_INPUT)
        if name in observations
    }


def absent_thresholds(
    thresholds: torch.Tensor, groups: ThresholdGroups, codes: torch.Tensor
) -> torch.Tensor:
    """Mark the padded thresholds of each row as missing.

    Parameters
    ----------
    thresholds : torch.Tensor
        Row-by-threshold values, padded to the longest group vector.
    groups : ThresholdGroups
        Resolved threshold groups.
    codes : torch.Tensor
        Threshold group code of each row.

    Returns
    -------
    torch.Tensor
        The thresholds with NaN at positions beyond a row's group count.
    """
    counts = torch.as_tensor(groups.counts, device=thresholds.device)[
        codes.to(thresholds.device).long()
    ]
    positions = torch.arange(thresholds.shape[-1], device=thresholds.device)
    return thresholds.masked_fill(positions >= counts.unsqueeze(-1), float("nan"))


def append_group_log_masses(
    groups: ThresholdGroups,
    kind: str,
    padded: str,
    result: str,
    *,
    row_axis: str,
    category_axis: str,
    steps: list[ProgramStep],
) -> None:
    """Reduce padded category log masses to each row's own categories.

    Parameters
    ----------
    groups : ThresholdGroups
        Resolved threshold groups.
    kind : str
        Ordinal family kind.
    padded : str
        Binder of the row-by-category log masses under padded thresholds.
    result : str
        Binder of the reduced log masses.
    row_axis : str
        Response row axis.
    category_axis : str
        Response category axis.
    steps : list[ProgramStep]
        Program steps, extended with the reduction.
    """
    row, category = _var("ordinal_row"), _var("ordinal_category")
    group = _index(groups.column.qvr_name, row)

    def selector(name: str) -> LetExprNode:
        """Read one selector at the row's group and the scoped category.

        Parameters
        ----------
        name : str
            Selector input.

        Returns
        -------
        LetExprNode
            The indexed selector.
        """
        return _index(name, group, category)

    mass = _index(padded, row, category)
    if kind == "acat":
        body = _add(mass, selector(ABSENT_INPUT))
    else:
        steps.append(
            _let(
                "threshold_tail_log_mass",
                _factor(
                    (("ordinal_row", row_axis),),
                    body=_call(
                        "logsumexp",
                        _factor(
                            (("ordinal_category", category_axis),),
                            body=_add(mass, selector(TAIL_INPUT)),
                        ),
                    ),
                ),
            )
        )
        body = _add(
            _mul(selector(BELOW_INPUT), mass),
            _mul(selector(LAST_INPUT), _index("threshold_tail_log_mass", row)),
            selector(ABSENT_INPUT),
        )
    steps.append(
        _let(
            result,
            _factor(
                (("ordinal_row", row_axis), ("ordinal_category", category_axis)),
                body=body,
            ),
        )
    )


def append_group_thresholds(
    formula: Formula,
    groups: ThresholdGroups,
    *,
    ordered: bool,
    cutpoint_axis: str,
    statements: list[Statement],
    steps: list[ProgramStep],
    declare: Callable[[list[Statement], str, int], None],
    draw: Callable[[str, str, TypeName], SampleStep],
    location_prior: str,
) -> str:
    """Emit one unpooled threshold vector per group and gather it by row.

    Parameters
    ----------
    formula : Formula
        The location predictor, whose intercept decides the reference group.
    groups : ThresholdGroups
        Resolved groups with one threshold count per label.
    ordered : bool
        Whether thresholds are ordered (the cumulative family). Ordered
        shapes are cumulative sums of positive spacings; unordered shapes
        are free offsets from the first threshold.
    cutpoint_axis : str
        Declared threshold axis, as long as the largest group's count.
    statements : list[Statement]
        Module declarations, extended with the group and coordinate axes.
    steps : list[ProgramStep]
        Program steps, extended with priors and threshold transformations.
    declare : Callable[[list[Statement], str, int], None]
        Declares one finite axis.
    draw : Callable[[str, str, TypeName], SampleStep]
        Emits the prior draw of a site from its default prior text over an
        axis, honoring a rule selected for the site.
    location_prior : str
        Default prior of a group location, that of a population-level
        coefficient.

    Returns
    -------
    str
        Binder of the row-by-threshold matrix.
    """
    counts = groups.counts
    width = max(counts)
    declare(statements, GROUP_AXIS, len(counts))
    reference = has_location_reference(formula)
    free_locations = len(counts) - int(reference)
    if free_locations:
        axis = "FormulaThresholdLocation"
        declare(statements, axis, free_locations)
        steps.append(draw(LOCATION_SITE, location_prior, TypeName(name=axis)))
    shape_site = "cutpoint_log_spacing" if ordered else "cutpoint_offset"
    free_shape = sum(count - 1 for count in counts)
    if free_shape:
        axis = "FormulaThresholdShape"
        declare(statements, axis, free_shape)
        steps.append(draw(shape_site, "Normal(0.0, 1.0)", TypeName(name=axis)))
    by_position: list[list[LetExprNode]] = [[] for _ in range(width)]
    offset = 0
    for group, count in enumerate(counts):
        # Uncentered shape: zero, then cumulative spacings or free offsets.
        uncentered: list[LetExprNode] = [LetExprLiteral(value=0.0)]
        for position in range(1, count):
            coordinate: LetExprNode = _index(
                shape_site, _integer(offset + position - 1)
            )
            if ordered:
                coordinate = _add(uncentered[-1], _call("exp", coordinate))
            name = f"threshold_shape_{group}_{position}"
            steps.append(_let(name, coordinate))
            uncentered.append(_var(name))
        offset += count - 1
        center = f"threshold_center_{group}"
        steps.append(
            _let(
                center,
                _call(
                    "mean",
                    _factor(
                        (("threshold_position", f"FormulaThresholdCount_{count}"),),
                        cases=tuple(enumerate(uncentered)),
                    ),
                ),
            )
        )
        declare(statements, f"FormulaThresholdCount_{count}", count)
        location_index = group - int(reference)
        for position in range(width):
            # Padded thresholds continue the group's last one in unit steps.
            value = _sub(uncentered[min(position, count - 1)], _var(center))
            if location_index >= 0:
                value = _add(value, _index(LOCATION_SITE, _integer(location_index)))
            if position >= count:
                value = _add(value, LetExprLiteral(value=float(position - count + 1)))
            by_position[position].append(value)
    steps.append(
        _let(
            "threshold_values_by_position",
            _factor(
                (("cutpoint", cutpoint_axis),),
                cases=tuple(
                    (
                        position,
                        _factor(
                            (("threshold_group", GROUP_AXIS),),
                            cases=tuple(enumerate(values)),
                        ),
                    )
                    for position, values in enumerate(by_position)
                ),
            ),
        )
    )
    steps.append(
        _let(
            MATRIX_SITE,
            _factor(
                (("threshold_group", GROUP_AXIS), ("cutpoint", cutpoint_axis)),
                body=_index(
                    "threshold_values_by_position",
                    _var("cutpoint"),
                    _var("threshold_group"),
                ),
            ),
        )
    )
    steps.append(_let(ROW_SITE, _index(MATRIX_SITE, _var(groups.column.qvr_name))))
    return ROW_SITE
