"""Emit one block of group-level coefficients from its law.

A block is emitted in two phases, driven by the factors its `GroupBlockLaw`
declares:

1. `draw_group_block` declares the block's axes, draws the hyperparameters of
   its grouping levels when it is the first block of those levels, draws its
   correlation and scale hyperparameters, and draws its raw sites from the
   law of the factors they carry.
2. `bind_group_block` applies the factors the raw sites do not carry, in the
   order of the law

       b_j = mean + dfm_j A_j g_j,    A_j = diag(sd[by_j]) L[by_j],

   binds each member's coefficients over the grouping levels, scores the prior
   weights of the innovations, and returns each member's coefficient gathered
   at the response rows' levels.

An absent factor is an absent step: an independent scalar coefficient is one
elementwise product. `bind_rows` then applies a member's row map to the gather
and binds the member's contribution to the response rows.

Every coefficient binding is indexed by grouping level and then, for a
member with a component axis, by component; a flat raw site over levels and
components holds each level's components consecutively, and a flat scale
over ``by`` levels and components each ``by`` level's components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import (
    AxisSpec,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    LetExprBinOp,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    OptionEntry,
    OptionList,
    OptionName,
    ProgramStep,
    SampleStep,
    ScoreStep,
    Statement,
    TypeName,
)
from quivers.formulas._ast import (
    _add,
    _call,
    _div,
    _draw,
    _factor,
    _index,
    _let,
    _mul,
    _over,
    _sub,
    _var,
)
from quivers.formulas.formula import _qvr_name
from quivers.formulas._group_law import (
    GroupBlockLaw,
    GroupLevelLaw,
    GroupMemberLaw,
    IndependentCoordinates,
    LKJCorrelation,
    MembershipRows,
    RowMap,
    SingleLevelRows,
)

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule

__all__: list[str] = []

_LOG_ROOT_TWO_PI = 0.9189385332046727
# Variables ranging over the grouping levels, a member's components, the
# block's coordinates, and the flat cells of a site over two axes.
LEVEL, COMPONENT = "group_level", "group_component"
COORDINATE, CELL = "group_coordinate", "group_cell"

type GroupValues = dict[tuple[str, str, str], LetExprNode]
# A block's level-by-coordinate array, read for a member at a component (None
# for a scalar member) and a grouping level.
type Reader = Callable[[GroupMemberLaw, LetExprNode | None, LetExprNode], LetExprNode]


def _integer(value: int) -> LetExprLiteral:
    """Build an integer literal.

    Parameters
    ----------
    value : int
        Literal value.

    Returns
    -------
    LetExprLiteral
        Integral literal expression.
    """
    return LetExprLiteral(value=value, integral=True)


def _stem(law: GroupBlockLaw) -> str:
    """Name the prefix of a block's intermediate bindings.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.

    Returns
    -------
    str
        Prefix of binding names.
    """
    return f"__formula_group_{_qvr_name(law.identity)}"


def _flat(major: LetExprNode, width: int, minor: LetExprNode | None) -> LetExprNode:
    """Locate a cell of a flat axis whose second coordinate varies fastest.

    Parameters
    ----------
    major : LetExprNode
        First coordinate.
    width : int
        Extent of the second coordinate.
    minor : LetExprNode or None
        Second coordinate; None for an axis without one.

    Returns
    -------
    LetExprNode
        Integer position on the flat axis.
    """
    if minor is None:
        return major
    return _add(_mul(major, _integer(width)), minor)


def scale_value(
    level: GroupLevelLaw,
    member: GroupMemberLaw,
    at: LetExprNode,
    component: LetExprNode | None,
) -> LetExprNode:
    """Read one member's standard deviation at one level and component.

    Parameters
    ----------
    level : GroupLevelLaw
        Level factors of the block.
    member : GroupMemberLaw
        Member whose scale is read.
    at : LetExprNode
        Grouping level.
    component : LetExprNode or None
        Component of a member with a component axis; None otherwise.

    Returns
    -------
    LetExprNode
        The scale indexed by the axes it declares: the level's ``by`` level,
        the component, both on the flat axis, or neither.
    """
    scale = member.scale
    minor = component if scale.component_axis is not None else None
    if scale.by_axis is not None and level.by is not None:
        return _index(
            scale.site, _flat(_index(level.by.input, at), member.width, minor)
        )
    return _var(scale.site) if minor is None else _index(scale.site, minor)


def _mean_value(
    member: GroupMemberLaw, component: LetExprNode | None
) -> LetExprNode | None:
    """Read one member's affine mean at one component.

    Parameters
    ----------
    member : GroupMemberLaw
        Member whose mean is read.
    component : LetExprNode or None
        Component of a member with a component axis; None otherwise.

    Returns
    -------
    LetExprNode or None
        The mean, or None for a zero mean.
    """
    if member.mean is None:
        return None
    if component is None:
        return _var(member.mean.site)
    return _index(member.mean.site, component)


def _level_factor(
    law: GroupBlockLaw,
    member: GroupMemberLaw,
    at: LetExprNode,
    component: LetExprNode | None,
) -> LetExprNode:
    """Read the factor scaling one coordinate of one level's coefficients.

    Parameters
    ----------
    law : GroupBlockLaw
        Block of the member.
    member : GroupMemberLaw
        Member whose coordinate is read.
    at : LetExprNode
        Grouping level.
    component : LetExprNode or None
        Component of a member with a component axis; None otherwise.

    Returns
    -------
    LetExprNode
        The standard deviation, times the mixing multiplier of a block whose
        raw sites carry the mixing.
    """
    scale = scale_value(law.level, member, at, component)
    mixing = law.level.mixing
    if mixing is None or "mixing" not in law.carried:
        return scale
    return _mul(_index(mixing.multiplier, at), scale)


def _scale_draw(
    lens: FormulaToQVRModule,
    level: GroupLevelLaw,
    member: GroupMemberLaw,
    statements: list[Statement],
) -> ProgramStep:
    """Draw one member's standard deviation over its declared axes.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    level : GroupLevelLaw
        Level factors of the block.
    member : GroupMemberLaw
        Member whose scale is drawn.
    statements : list[Statement]
        Module declarations receiving the flat axis of a scale with a ``by``
        and a component axis.

    Returns
    -------
    ProgramStep
        Prior draw indexed by the scale's axis.
    """
    scale = member.scale
    if scale.cell_axis is not None and level.by is not None:
        lens._append_finite_object(
            statements, scale.cell_axis, len(level.by.levels) * member.width
        )
    return lens._prior_draw(
        scale.site,
        scale.prior or lens._random_scale_prior,
        index=TypeName(name=scale.axis) if scale.axis is not None else None,
        lower=scale.lower if scale.site in lens._user_priors else None,
    )


def _mean_draw(lens: FormulaToQVRModule, member: GroupMemberLaw) -> ProgramStep:
    """Draw one member's affine mean over its component axis.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    member : GroupMemberLaw
        Member with an affine mean.

    Returns
    -------
    ProgramStep
        Prior draw of the mean.

    Raises
    ------
    ValueError
        If the member has no affine mean.
    """
    if member.mean is None:
        raise ValueError(f"group coefficient {member.coefficient!r} has no mean")
    return lens._prior_draw(
        member.mean.site,
        member.mean.prior,
        index=TypeName(name=member.axis) if member.axis is not None else None,
    )


def _correlation_draw(
    lens: FormulaToQVRModule, site: str, dimension: str
) -> ProgramStep:
    """Draw one Cholesky factor of a coefficient correlation.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    site : str
        Cholesky-factor site.
    dimension : str
        Coordinate axis of the block.

    Returns
    -------
    ProgramStep
        The correlation prior draw.

    Raises
    ------
    ValueError
        If the correlation prior is not an LKJ Cholesky distribution.
    """
    draw = lens._prior_draw(
        site, lens._correlation_prior, index=TypeName(name=dimension)
    )
    if draw.morphism != "LKJCholesky":
        raise ValueError("correlation priors must use LKJCholesky")
    return draw


def draw_level(
    lens: FormulaToQVRModule,
    level: GroupLevelLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Declare and draw the hyperparameters of a block's grouping levels.

    A ``by`` partition declares the axis over its levels. Student-t mixing
    draws the degrees of freedom ``df``, restricted to its lower bound, one
    ``udf_j ~ inv_chi_square(df)`` per level, and binds the multiplier
    ``dfm_j = sqrt(df * udf_j)`` of every Gaussian coefficient of level
    ``j``, a scale mixture whose marginal law is the multivariate Student-t
    with ``df`` degrees of freedom. The mixing of a grouping factor is drawn
    once, by the first block that declares it, unless another program draws
    it.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration and the emitted laws.
    level : GroupLevelLaw
        Level factors of the block being drawn.
    statements : list[Statement]
        Module declarations receiving the ``by`` axis.
    steps : list[ProgramStep]
        Program steps receiving the mixing draws.
    """
    if level.by is not None:
        lens._append_finite_object(statements, level.by.axis, len(level.by.levels))
    mixing = level.mixing
    if mixing is None or mixing.df_site in lens._drawn_sites:
        return
    lens._drawn_sites = lens._drawn_sites | {mixing.df_site}
    half = mixing.df_site + "_half"
    steps.append(lens._prior_draw(mixing.df_site, mixing.prior, lower=mixing.lower))
    steps.append(_let(half, _mul(_var(mixing.df_site), LetExprLiteral(value=0.5))))
    steps.append(
        SampleStep(
            vars=(mixing.mixing_site,),
            morphism="InverseGamma",
            args=(
                DrawArgNamed(parameter="concentration", value=DrawArgName(text=half)),
                DrawArgNamed(parameter="rate", value=DrawArgScalar(value=0.5)),
            ),
            index=TypeName(name=level.axis),
        )
    )
    steps.append(
        _let(
            mixing.multiplier,
            _over(
                LEVEL,
                level.axis,
                _call(
                    "sqrt",
                    _mul(_var(mixing.df_site), _index(mixing.mixing_site, _var(LEVEL))),
                ),
            ),
        )
    )


def _owns_coordinate_axis(law: GroupBlockLaw) -> bool:
    """Decide whether a block declares its own coordinate axis.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.

    Returns
    -------
    bool
        True when the coordinate axis is not the component axis of a member,
        which its predictor declares.
    """
    return law.coordinate_axis is not None and all(
        member.axis != law.coordinate_axis for member in law.members
    )


def _dimension(law: GroupBlockLaw) -> str:
    """Read the coordinate axis of a block.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.

    Returns
    -------
    str
        Finite axis over the block's coordinates.

    Raises
    ------
    ValueError
        If the block has no coordinate axis.
    """
    if law.coordinate_axis is None:
        raise ValueError(f"group block {law.identity!r} has no coordinate axis")
    return law.coordinate_axis


def _coordinates(
    law: GroupBlockLaw,
) -> tuple[tuple[GroupMemberLaw, LetExprNode | None], ...]:
    """List the member and literal component of every coordinate.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.

    Returns
    -------
    tuple[tuple[GroupMemberLaw, LetExprNode or None], ...]
        One entry per coordinate, in order; the component is None for a
        scalar member.
    """
    return tuple(
        (member, _integer(component) if member.axis is not None else None)
        for member in law.members
        for component in range(member.width)
    )


def _by_coordinate(
    law: GroupBlockLaw,
    value: Callable[[GroupMemberLaw, LetExprNode | None], LetExprNode],
) -> LetExprNode:
    """Build an array over a block's coordinates from its members.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.
    value : Callable
        Value of a member's component.

    Returns
    -------
    LetExprNode
        One case per coordinate.
    """
    return _factor(
        ((COORDINATE, _dimension(law)),),
        cases=tuple(
            (position, value(member, component))
            for position, (member, component) in enumerate(_coordinates(law))
        ),
    )


def _quotient(value: LetExprNode, count: int) -> LetExprNode:
    """Divide a coordinate by a component count.

    Parameters
    ----------
    value : LetExprNode
        Coordinate.
    count : int
        Number of components.

    Returns
    -------
    LetExprNode
        Integer quotient, evaluated for a literal coordinate.
    """
    if count == 1:
        return value
    if isinstance(value, LetExprLiteral):
        return _integer(int(value.value) // count)
    return _div(value, _integer(count))


def _remainder(value: LetExprNode, count: int) -> LetExprNode | None:
    """Read the component of a coordinate.

    Parameters
    ----------
    value : LetExprNode
        Coordinate.
    count : int
        Number of components.

    Returns
    -------
    LetExprNode or None
        Integer remainder, evaluated for a literal coordinate; None for a
        single component.
    """
    if count == 1:
        return None
    if isinstance(value, LetExprLiteral):
        return _integer(int(value.value) % count)
    return LetExprBinOp(op="%", left=value, right=_integer(count))


def _cell(
    law: GroupBlockLaw, at: LetExprNode, component: LetExprNode | None
) -> LetExprNode | None:
    """Locate the correlation factor of one level and component.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with correlated coordinates.
    at : LetExprNode
        Grouping level.
    component : LetExprNode or None
        Component of the coordinate; None for a single component.

    Returns
    -------
    LetExprNode or None
        Position on the axis of the factors; None for one factor.
    """
    by, correlation = law.level.by, law.correlation
    assert isinstance(correlation, LKJCorrelation)
    if by is None:
        return component
    return _flat(_index(by.input, at), correlation.components, component)


def _by_cell(
    law: GroupBlockLaw, value: Callable[[int, str], LetExprNode]
) -> LetExprNode:
    """Build an array over a block's correlation factors.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with several correlation factors.
    value : Callable
        Value of each factor, from its position and site.

    Returns
    -------
    LetExprNode
        One case per factor.
    """
    correlation = law.correlation
    assert isinstance(correlation, LKJCorrelation)
    assert correlation.cell_axis is not None
    return _factor(
        (("by", correlation.cell_axis),),
        cases=tuple(
            (index, value(index, site))
            for index, site in enumerate(correlation.factors)
        ),
    )


def _bind_factors(law: GroupBlockLaw, steps: list[ProgramStep]) -> None:
    """Bind the Cholesky factors of a block as one array.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose raw sites carry its correlation.
    steps : list[ProgramStep]
        Program steps receiving the array of several factors.
    """
    if len(law.correlation.sites) > 1:
        steps.append(
            _let(
                _stem(law) + "_by_factors",
                _by_cell(law, lambda index, site: _var(site)),
            )
        )


def _correlation_entry(
    law: GroupBlockLaw, at: LetExprNode, row: LetExprNode, column: LetExprNode
) -> LetExprNode | None:
    """Read one entry of the correlation factor of one grouping level.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose factors, if several, are bound.
    at : LetExprNode
        Grouping level.
    row, column : LetExprNode
        Coordinates.

    Returns
    -------
    LetExprNode or None
        The entry, zero between components, or None for independent
        coordinates.
    """
    correlation = law.correlation
    if not isinstance(correlation, LKJCorrelation):
        return None
    count = correlation.components
    of_row, of_column = _remainder(row, count), _remainder(column, count)
    first, second = _quotient(row, count), _quotient(column, count)
    cell = _cell(law, at, of_row)
    if cell is None:
        (site,) = correlation.factors
        return _index(site, first, second)
    entry = _index(_stem(law) + "_by_factors", cell, first, second)
    if of_row is None or of_column is None:
        return entry
    return _call(
        "select",
        LetExprBinOp(op="==", left=of_row, right=of_column),
        entry,
        LetExprLiteral(value=0.0),
    )


def _identity_entry(row: LetExprNode, column: LetExprNode) -> LetExprNode:
    """Read one entry of the identity matrix.

    Parameters
    ----------
    row, column : LetExprNode
        Coordinates.

    Returns
    -------
    LetExprNode
        One on the diagonal and zero elsewhere.
    """
    return _call(
        "select",
        LetExprBinOp(op="==", left=row, right=column),
        LetExprLiteral(value=1.0),
        LetExprLiteral(value=0.0),
    )


def draw_group_block(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Declare a block's axes and draw its hyperparameters and raw sites.

    The hyperparameters of the grouping levels come first, then every
    correlation factor, then each member's mean and scale. The raw sites
    follow the factors they carry: support coordinates for a known level
    covariance, one Gaussian vector per level for a carried correlation, and
    otherwise one site per member.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration and the emitted laws.
    law : GroupBlockLaw
        Block to draw.
    statements : list[Statement]
        Module declarations receiving the block's axes.
    steps : list[ProgramStep]
        Program steps receiving the draws.
    """
    level = law.level
    draw_level(lens, level, statements, steps)
    if _owns_coordinate_axis(law):
        lens._append_finite_object(statements, _dimension(law), law.width)
    correlation = law.correlation
    if isinstance(correlation, LKJCorrelation):
        lens._append_finite_object(statements, correlation.axis, correlation.size)
        if correlation.cell_axis is not None:
            lens._append_finite_object(
                statements, correlation.cell_axis, len(correlation.factors)
            )
        for site in correlation.factors:
            steps.append(_correlation_draw(lens, site, correlation.axis))
    lens._emitted_groups.append(law)
    for member in law.members:
        if member.mean is not None:
            steps.append(_mean_draw(lens, member))
        steps.append(_scale_draw(lens, level, member, statements))
        if not law.joint_site:
            _draw_member(lens, law, member, statements, steps)
    if level.basis is not None:
        _draw_support(lens, law, statements, steps)
    elif law.joint_site:
        _draw_levels(law, steps)


def _named(name: str, value: LetExprNode, steps: list[ProgramStep]) -> str:
    """Name a value, binding it unless it is a bound name.

    Parameters
    ----------
    name : str
        Binding of a composite value.
    value : LetExprNode
        Value to name.
    steps : list[ProgramStep]
        Program steps receiving the binding.

    Returns
    -------
    str
        Name of the value.
    """
    if isinstance(value, LetExprVar):
        return value.name
    steps.append(_let(name, value))
    return name


def _draw_member(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    member: GroupMemberLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Draw the raw site of one member with independent grouping levels.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler declaring the cell axis.
    law : GroupBlockLaw
        Block of the member.
    member : GroupMemberLaw
        Member owning its raw site.
    statements : list[Statement]
        Module declarations receiving the flat cell axis.
    steps : list[ProgramStep]
        Program steps receiving the draw.

    Raises
    ------
    ValueError
        If the member has no raw site.
    """
    level, raw = law.level, member.raw_site
    if raw is None:
        raise ValueError(f"group block {law.identity!r} has no raw site")
    plate, at = level.axis, _var(LEVEL)
    binder, component = LEVEL, None
    if member.cell_axis is not None:
        plate = member.cell_axis
        lens._append_finite_object(statements, plate, level.size * member.width)
        cell, width = _var(CELL), _integer(member.width)
        at, binder = _div(cell, width), CELL
        component = LetExprBinOp(op="%", left=cell, right=width)
    if "scale" not in law.carried:
        steps.append(_draw(raw, "Normal", (0.0, 1.0), index=TypeName(name=plate)))
        return

    def over(value: LetExprNode) -> LetExprNode:
        """Build a value over the plate, keeping a scalar as it is."""
        return value if isinstance(value, LetExprVar) else _over(binder, plate, value)

    stem = f"{raw}_cell" if member.cell_axis is not None else f"{raw}_level"
    mean = _mean_value(member, component)
    location: str | float = (
        0.0 if mean is None else _named(f"{stem}_mean", over(mean), steps)
    )
    scale = _named(
        f"{stem}_scale", over(_level_factor(law, member, at, component)), steps
    )
    steps.append(_draw(raw, "Normal", (location, scale), index=TypeName(name=plate)))


def _draw_levels(law: GroupBlockLaw, steps: list[ProgramStep]) -> None:
    """Draw every level's coefficients from one multivariate Gaussian.

    The scale factor of level ``j`` is ``dfm_j diag(sd[by_j]) L[by_j]``, one
    factor for every level of a block without level factors.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose raw site carries a coefficient correlation.
    steps : list[ProgramStep]
        Program steps receiving the factor and the draw.

    Raises
    ------
    ValueError
        If the block has no raw site.
    """
    level, dimension, raw = law.level, _dimension(law), law.members[0].raw_site
    if raw is None:
        raise ValueError(f"group block {law.identity!r} has no raw site")
    stem = _stem(law)
    scales, factor, zero = stem + "_scales", stem + "_factor", stem + "_zero"
    at, row, column = _var(LEVEL), _var("group_row"), _var("group_column")
    varies = level.by is not None or level.mixing is not None
    _bind_factors(law, steps)
    entry = _correlation_entry(law, at, row, column)
    assert entry is not None
    square = (("group_row", dimension), ("group_column", dimension))
    steps.extend(
        (
            _let(
                scales,
                _by_coordinate(
                    law,
                    lambda member, component: (
                        _over(
                            LEVEL,
                            level.axis,
                            _level_factor(law, member, at, component),
                        )
                        if varies
                        else _level_factor(law, member, at, component)
                    ),
                ),
            ),
            _let(
                zero,
                _by_coordinate(
                    law,
                    lambda member, component: (
                        _mean_value(member, component) or LetExprLiteral(value=0.0)
                    ),
                )
                if any(member.mean is not None for member in law.members)
                else _factor(
                    ((COORDINATE, dimension),), body=LetExprLiteral(value=0.0)
                ),
            ),
            _let(
                factor,
                _factor(
                    ((LEVEL, level.axis), *square) if varies else square,
                    body=_mul(
                        _index(scales, row, at) if varies else _index(scales, row),
                        entry,
                    ),
                ),
            ),
        )
    )
    draw = _draw(
        raw, "MultivariateNormal", (zero, factor), index=TypeName(name=level.axis)
    )
    steps.append(
        draw.with_(
            axes=AxisSpec(over=(dimension,), iid_over=(level.axis,)),
            options=(
                OptionEntry(key="over", value=OptionName(value=dimension)),
                OptionEntry(key="iid_over", value=OptionName(value=level.axis)),
            ),
        )
    )


def _draw_support(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Draw one Gaussian law in the support coordinates of a level covariance.

    Noncentered coordinates are standardized innovations. Centered ones carry
    the column factor ``A_j = diag(sd[by_j]) L[by_j]``: one matrix Gaussian
    with the support covariance between rows when every level has one column
    factor, and otherwise, for a covariance of full rank, the vector of
    coefficients with block factor ``L_V[j, l] A_j``. A covariance of rank
    zero has no draw.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler declaring the support axis.
    law : GroupBlockLaw
        Block with a known level covariance.
    statements : list[Statement]
        Module declarations receiving the support axis.
    steps : list[ProgramStep]
        Program steps receiving the draw.

    Raises
    ------
    ValueError
        If the block has no known covariance, or a covariance of positive
        rank has no raw site.
    """
    level, dimension = law.level, _dimension(law)
    basis, raw = level.basis, law.members[0].raw_site
    if basis is None:
        raise ValueError("a support-coordinate block needs a known covariance")
    if not basis.rank:
        return
    if raw is None:
        raise ValueError(f"group block {law.identity!r} has no raw site")
    support, stem = level.support_axis, _stem(law)
    lens._append_finite_object(statements, support, basis.rank)
    row_factor, zero = level.basis_input("row_factor"), stem + "_zero"
    row, column = _var("matrix_row"), _var("matrix_column")
    at, scales = _var(LEVEL), stem + "_scales"
    carried, varies = "scale" in law.carried, law.level_column_factors
    if carried:
        _bind_factors(law, steps)
        steps.append(
            _let(
                scales,
                _by_coordinate(
                    law,
                    lambda member, component: (
                        _over(
                            LEVEL, level.axis, scale_value(level, member, at, component)
                        )
                        if varies
                        else scale_value(level, member, at, component)
                    ),
                ),
            )
        )
    if varies:
        cells = f"FormulaSupportCell_{_qvr_name(law.identity)}"
        lens._append_finite_object(statements, cells, basis.rank * law.width)
        width = _integer(law.width)
        row_level, row_coordinate = (
            _div(row, width),
            LetExprBinOp(op="%", left=row, right=width),
        )
        column_level, column_coordinate = (
            _div(column, width),
            LetExprBinOp(op="%", left=column, right=width),
        )
        entry = _correlation_entry(law, row_level, row_coordinate, column_coordinate)
        factor = stem + "_cell_factor"
        steps.extend(
            (
                _let(
                    zero,
                    _factor(((CELL, cells),), body=LetExprLiteral(value=0.0)),
                ),
                _let(
                    factor,
                    _factor(
                        (("matrix_row", cells), ("matrix_column", cells)),
                        body=_mul(
                            _index(row_factor, row_level, column_level),
                            _mul(
                                _index(scales, row_coordinate, row_level),
                                entry
                                if entry is not None
                                else _identity_entry(row_coordinate, column_coordinate),
                            ),
                        ),
                    ),
                ),
            )
        )
        draw = _draw(raw, "MultivariateNormal", (zero, factor))
        steps.append(
            draw.with_(
                axes=AxisSpec(over=(cells,)),
                options=(OptionEntry(key="over", value=OptionName(value=cells)),),
            )
        )
        return
    entry = _correlation_entry(law, at, row, column) or _identity_entry(row, column)
    column_factor = stem + "_column_factor"
    steps.extend(
        (
            _let(
                zero,
                _factor(
                    (("support", support), (COORDINATE, dimension)),
                    body=LetExprLiteral(value=0.0),
                ),
            ),
            _let(
                column_factor,
                _factor(
                    (("matrix_row", dimension), ("matrix_column", dimension)),
                    body=_mul(_index(scales, row), entry)
                    if carried
                    else _identity_entry(row, column),
                ),
            ),
        )
    )
    draw = _draw(
        raw,
        "MatrixNormal",
        (zero, row_factor, column_factor),
        parameterization=("loc", "row_scale_tril", "col_scale_tril"),
    )
    steps.append(
        draw.with_(
            axes=AxisSpec(over=(support, dimension)),
            options=(
                OptionEntry(
                    key="over",
                    value=OptionList(
                        items=(OptionName(value=support), OptionName(value=dimension))
                    ),
                ),
            ),
        )
    )


def _position(member: GroupMemberLaw, component: LetExprNode | None) -> LetExprNode:
    """Locate a member's coordinate within its block.

    Parameters
    ----------
    member : GroupMemberLaw
        Member of the block.
    component : LetExprNode or None
        Component of a member with a component axis; None otherwise.

    Returns
    -------
    LetExprNode
        The member's offset plus the component.
    """
    offset = _integer(member.offset)
    if component is None:
        return offset
    if isinstance(component, LetExprLiteral):
        return _integer(member.offset + int(component.value))
    return _add(offset, component)


def _raw_reader(law: GroupBlockLaw) -> Reader:
    """Read a block's raw sites by row, component and member.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose raw sites have positive rank.

    Returns
    -------
    Reader
        Reader of the raw sites, indexed by grouping level, or by support
        coordinate for a known level covariance.
    """
    # A support block whose levels have their own column factors is one vector.
    vector = law.level.basis is not None and law.level_column_factors

    def read(
        member: GroupMemberLaw, component: LetExprNode | None, at: LetExprNode
    ) -> LetExprNode:
        site, position = member.raw_site, _position(member, component)
        if site is None:
            raise ValueError(
                f"group coefficient {member.coefficient!r} has no raw site"
            )
        if vector:
            return _index(site, _flat(at, law.width, position))
        if law.joint_site:
            return _index(site, at, position)
        return _index(site, _flat(at, member.width, component))

    return read


def _bind_levels(law: GroupBlockLaw, raw: Reader, steps: list[ProgramStep]) -> Reader:
    """Map support coordinates to the grouping levels.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with a known level covariance of positive rank.
    raw : Reader
        Reader of the support coordinates.
    steps : list[ProgramStep]
        Program steps receiving the map.

    Returns
    -------
    Reader
        Reader of the mapped coordinates at the grouping levels.
    """
    level = law.level
    if law.level_column_factors:
        # The covariance has full rank, so its support is the levels.
        return raw
    site = law.members[0].raw_site
    assert site is not None
    mapped = _stem(law) + "_mapped"
    matrix = "projection" if "scale" in law.carried else "factor"
    steps.append(
        _let(mapped, _call("matmul", _var(level.basis_input(matrix)), _var(site)))
    )
    return lambda member, component, at: _index(
        mapped, at, _position(member, component)
    )


def _bind_correlation(
    law: GroupBlockLaw, values: Reader, steps: list[ProgramStep]
) -> Reader:
    """Correlate each level's coordinates with the factor of its ``by`` level.

    Costs O(by levels x levels x width^2).

    Parameters
    ----------
    law : GroupBlockLaw
        Block with a coefficient correlation.
    values : Reader
        Reader of the uncorrelated coordinates.
    steps : list[ProgramStep]
        Program steps receiving the products.

    Returns
    -------
    Reader
        Reader of the correlated coordinates.
    """
    level, dimension, correlation = law.level, _dimension(law), law.correlation
    assert isinstance(correlation, LKJCorrelation)
    stem, at, coordinate = _stem(law), _var(LEVEL), _var(COORDINATE)
    innovations, transformed = stem + "_innovations", stem + "_transformed"
    steps.append(
        _let(
            innovations,
            _by_coordinate(
                law,
                lambda member, component: _over(
                    LEVEL, level.axis, values(member, component, at)
                ),
            ),
        )
    )
    count = correlation.components
    blocks = (innovations,)
    if count > 1:
        blocks = tuple(f"{innovations}_{part}" for part in range(count))
        steps.extend(
            _let(
                block,
                _factor(
                    (("group_row", correlation.axis), (LEVEL, level.axis)),
                    body=_index(
                        innovations,
                        _flat(_var("group_row"), count, _integer(part)),
                        at,
                    ),
                ),
            )
            for part, block in enumerate(blocks)
        )
    cell = _cell(law, at, _remainder(coordinate, count))
    if cell is None:
        (site,) = correlation.factors
        product: LetExprNode = _call("matmul", _var(site), _var(innovations))
    else:
        products = stem + "_by_products"
        steps.append(
            _let(
                products,
                _by_cell(
                    law,
                    lambda index, site: _call(
                        "matmul", _var(site), _var(blocks[index % count])
                    ),
                ),
            )
        )
        product = _factor(
            ((COORDINATE, dimension), (LEVEL, level.axis)),
            body=_index(products, cell, _quotient(coordinate, count), at),
        )
    steps.append(_let(transformed, product))
    return lambda member, component, position: _index(
        transformed, _position(member, component), position
    )


def _innovations(
    law: GroupBlockLaw, raw: Reader, steps: list[ProgramStep]
) -> tuple[LetExprNode, ...]:
    """Recover the standard-normal innovation of every coordinate and level.

    Noncentered raw sites are the innovations. Centered ones are standardized
    by inverting the factors they carry, in reverse order: the mean, the
    mixing and scale, the correlation by forward substitution, and the level
    covariance by the inverse of its Cholesky factor. Costs
    O(levels x width^2), plus O(levels^2 x width) for a known covariance.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose support coordinates, if any, are the grouping levels.
    raw : Reader
        Reader of the raw sites.
    steps : list[ProgramStep]
        Program steps receiving the standardized coordinates.

    Returns
    -------
    tuple[LetExprNode, ...]
        Innovation of each coordinate at the level variable.
    """
    level, at = law.level, _var(LEVEL)
    coordinates = _coordinates(law)
    if "scale" not in law.carried:
        return tuple(raw(member, component, at) for member, component in coordinates)
    stem = _stem(law)
    standardized: list[str] = []
    for position, (member, component) in enumerate(coordinates):
        value = raw(member, component, at)
        mean = _mean_value(member, component)
        if mean is not None and "mean" in law.carried:
            value = _sub(value, mean)
        value = _div(value, _level_factor(law, member, at, component))
        row = _integer(position)
        for column, name in enumerate(standardized):
            entry = _correlation_entry(law, at, row, _integer(column))
            if entry is not None:
                value = _sub(value, _mul(entry, _index(name, at)))
        diagonal = _correlation_entry(law, at, row, row)
        if diagonal is not None:
            value = _div(value, diagonal)
        name = f"{stem}_standardized_{position}"
        steps.append(_let(name, _over(LEVEL, level.axis, value)))
        standardized.append(name)
    if level.basis is not None:
        whitened = []
        for name in standardized:
            steps.append(
                _let(
                    name + "_whitened",
                    _call(
                        "matmul",
                        _var(level.basis_input("whitening")),
                        _var(name),
                    ),
                )
            )
            whitened.append(name + "_whitened")
        standardized = whitened
    return tuple(_index(name, at) for name in standardized)


def _score_prior_weights(
    law: GroupBlockLaw, raw: Reader, steps: list[ProgramStep]
) -> None:
    """Weight the standard-normal prior of each level's innovations.

    The raw sites keep the density of unit weights, so a weight ``w_j`` on
    level ``j`` adds ``(w_j - 1) * std_normal_lpdf(z_j)``, a density with
    respect to the innovations in either parameterization.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with prior weights.
    raw : Reader
        Reader of the raw sites.
    steps : list[ProgramStep]
        Program steps receiving the log-density factor.
    """
    level = law.level
    assert level.prior_weight is not None
    weight, at = level.prior_weight.input, _var(LEVEL)
    density = _add(
        *(
            _sub(
                _mul(LetExprLiteral(value=-0.5), _mul(value, value)),
                LetExprLiteral(value=_LOG_ROOT_TWO_PI),
            )
            for value in _innovations(law, raw, steps)
        )
    )
    steps.append(
        ScoreStep(
            name=_stem(law) + "_prior_weight",
            value=_call(
                "sum",
                _over(
                    LEVEL,
                    level.axis,
                    _mul(_sub(_index(weight, at), LetExprLiteral(value=1.0)), density),
                ),
            ),
        )
    )


def _gather(member: GroupMemberLaw) -> LetExprNode:
    """Read a member's coefficient at each response row's level.

    Parameters
    ----------
    member : GroupMemberLaw
        Bound member.

    Returns
    -------
    LetExprNode
        The coefficient indexed by the row index of its row map and, for a
        member with a component axis, the bound component.

    Raises
    ------
    ValueError
        If the row map is of an unknown kind.
    """
    rows = member.rows
    if not isinstance(rows, SingleLevelRows | MembershipRows):
        raise ValueError(f"unknown row map {type(rows).__name__}")
    if member.axis is None:
        return _index(member.coefficient, _var(rows.index))
    return _index(member.coefficient, _var(rows.index), _var(COMPONENT))


def bind_group_block(law: GroupBlockLaw, steps: list[ProgramStep]) -> GroupValues:
    """Apply the factors a drawn block's raw sites do not carry.

    Parameters
    ----------
    law : GroupBlockLaw
        Block whose raw sites are drawn.
    steps : list[ProgramStep]
        Program steps receiving the composition and the member bindings.

    Returns
    -------
    dict[tuple[str, str, str], LetExprNode]
        Each member's coefficient gathered at the response rows' levels,
        keyed by predictor namespace, grouping factor, and coefficient label.
    """
    level, at = law.level, _var(LEVEL)
    raw = _raw_reader(law)
    values: Reader | None = raw
    if level.basis is not None:
        # A covariance of rank zero gives every level a zero deviation.
        values = _bind_levels(law, raw, steps) if level.basis.rank else None
    if (
        values is not None
        and "correlation" not in law.carried
        and not isinstance(law.correlation, IndependentCoordinates)
    ):
        values = _bind_correlation(law, values, steps)
    for member in law.members:
        component = _var(COMPONENT) if member.axis is not None else None
        value: LetExprNode = LetExprLiteral(value=0.0)
        if values is not None:
            value = values(member, component, at)
            if "scale" not in law.carried:
                value = _mul(scale_value(level, member, at, component), value)
            if "mixing" not in law.carried and level.mixing is not None:
                value = _mul(_index(level.mixing.multiplier, at), value)
        mean = _mean_value(member, component)
        if mean is not None and "mean" not in law.carried:
            value = _add(mean, value)
        if member.axis is not None:
            bound: LetExprNode = _factor(
                ((LEVEL, level.axis), (COMPONENT, member.axis)), body=value
            )
        else:
            bound = _over(LEVEL, level.axis, value)
        if bound != _var(member.coefficient):
            steps.append(_let(member.coefficient, bound))
    if level.prior_weight is not None and values is not None:
        _score_prior_weights(law, raw, steps)
    return {
        (member.prefix, law.group, member.slope): _gather(member)
        for member in law.members
    }


def emit_group_block(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> GroupValues:
    """Draw and bind one block of group-level coefficients.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration and the emitted laws.
    law : GroupBlockLaw
        Block to emit.
    statements : list[Statement]
        Module declarations receiving the block's axes.
    steps : list[ProgramStep]
        Program steps receiving the draws and bindings.

    Returns
    -------
    dict[tuple[str, str, str], LetExprNode]
        Each member's coefficient gathered at the response rows' levels.
    """
    draw_group_block(lens, law, statements, steps)
    return bind_group_block(law, steps)


def row_contribution(rows: RowMap, gathered: LetExprNode) -> LetExprNode:
    """Carry a coefficient gathered at the rows' levels to a row contribution.

    Parameters
    ----------
    rows : RowMap
        Row map of the coefficient.
    gathered : LetExprNode
        The coefficient indexed first by the single-level row index.

    Returns
    -------
    LetExprNode
        The gather, or for a multi-membership factor the sum over members of
        the member's gather times its weighted covariate, times the row
        covariate.

    Raises
    ------
    ValueError
        If a multi-membership coefficient is not gathered by the factor's
        row index, or the row map is of an unknown kind.
    """
    if isinstance(rows, MembershipRows):
        index = _var(rows.index)
        if not isinstance(gathered, LetExprIndex) or gathered.indices[:1] != (index,):
            raise ValueError(
                f"multi-membership rows of {rows.index!r} require a level-indexed "
                "coefficient"
            )
        gathered = _add(
            *(
                _mul(
                    gathered.with_(indices=(_var(member), *gathered.indices[1:])),
                    _var(covariate),
                )
                for member, covariate in zip(rows.indices, rows.covariates, strict=True)
            )
        )
    elif not isinstance(rows, SingleLevelRows):
        raise ValueError(f"unknown row map {type(rows).__name__}")
    if rows.covariate is None:
        return gathered
    return _mul(
        gathered,
        _index(rows.covariate, _var(COMPONENT))
        if rows.by_component
        else _var(rows.covariate),
    )


def bind_rows(
    name: str,
    rows: RowMap,
    gathered: LetExprNode,
    axis: str | None,
    steps: list[ProgramStep],
) -> None:
    """Bind one coefficient's contribution to the response rows.

    Parameters
    ----------
    name : str
        Binding of the contribution.
    rows : RowMap
        Row map of the coefficient.
    gathered : LetExprNode
        The coefficient gathered at the rows' levels.
    axis : str or None
        Component axis of the coefficient; None for a scalar coefficient.
    steps : list[ProgramStep]
        Program steps receiving the binding: a row vector for a scalar
        coefficient, otherwise an array over the components and the rows.
    """
    value = row_contribution(rows, gathered)
    if axis is not None:
        value = _factor(((COMPONENT, axis),), body=value)
    steps.append(_let(name, value))
