"""Emit one block of group-level coefficients from its law.

Every block is emitted by the same two phases, driven by the factors its
`GroupBlockLaw` declares:

1. `draw_group_block` declares the block's axes, draws its correlation and
   scale hyperparameters, and draws its carrier: standard-normal innovations
   for noncentered coordinates, the coefficients themselves for centered
   coordinates, and support coordinates for a known level covariance.
2. `bind_group_block` composes the coefficient matrix

       B = diag(dfm) . L_V Z . (diag(sd[by]) L_R[by])^T + mean,

   scores the prior weights of the innovations, binds each member's slice of
   ``B`` over the grouping levels, and returns each member's coefficient
   gathered at the response rows' levels.

`row_contribution` then applies a member's row map to that gather.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import (
    AxisSpec,
    LetExprBinOp,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    OptionEntry,
    OptionList,
    OptionName,
    ProgramStep,
    ScoreStep,
    Statement,
    TypeName,
)
from quivers.formulas._ast import (
    _add,
    _call,
    _draw,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
)
from quivers.formulas.formula import _qvr_name
from quivers.formulas._group_law import (
    ByLKJCorrelation,
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

GroupValues = dict[tuple[str, str, str], LetExprNode]


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


def _scale_draw(lens: FormulaToQVRModule, member: GroupMemberLaw) -> ProgramStep:
    """Draw one member's standard deviation over its declared axes.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    member : GroupMemberLaw
        Member whose scale is drawn.

    Returns
    -------
    ProgramStep
        Prior draw indexed by the scale's component or ``by`` axis.

    Raises
    ------
    ValueError
        If the scale has both a component axis and a ``by`` axis.
    """
    scale = member.scale
    if scale.component_axis is not None and scale.by_axis is not None:
        raise ValueError(
            f"group scale {scale.site!r} cannot be indexed by both a component "
            "axis and a by axis"
        )
    axis = scale.component_axis or scale.by_axis
    return lens._prior_draw(
        scale.site,
        scale.prior or lens._random_scale_prior,
        index=TypeName(name=axis) if axis is not None else None,
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


def _scale_at(
    level: GroupLevelLaw, member: GroupMemberLaw, at: LetExprNode
) -> LetExprNode:
    """Read a scalar member's standard deviation at one grouping level.

    Parameters
    ----------
    level : GroupLevelLaw
        Level factors of the block.
    member : GroupMemberLaw
        Scalar member.
    at : LetExprNode
        Grouping level.

    Returns
    -------
    LetExprNode
        The scale, read at the level's ``by`` level when it has a ``by`` axis.
    """
    if member.scale.by_axis is None or level.by is None:
        return _var(member.scale.site)
    return LetExprIndex(
        array=_var(member.scale.site), indices=(_index(level.by.input, at),)
    )


def _mixed(level: GroupLevelLaw, value: LetExprNode, at: LetExprNode) -> LetExprNode:
    """Multiply a coefficient by its level's Student-t mixing multiplier.

    Parameters
    ----------
    level : GroupLevelLaw
        Level factors of the block.
    value : LetExprNode
        Gaussian coefficient at the level.
    at : LetExprNode
        Grouping level.

    Returns
    -------
    LetExprNode
        ``dfm[level] * value`` under mixing, ``value`` otherwise.
    """
    if level.mixing is None:
        return value
    return _mul(_index(level.mixing.multiplier, at), value)


def _prior_weight_score(
    level: GroupLevelLaw, name: str, innovations: tuple[str, ...]
) -> ScoreStep | None:
    """Weight the standard-normal prior of each level's innovations.

    The innovations keep their unit-weight standard-normal draws, so a
    weight ``w_j`` on level ``j`` adds ``(w_j - 1) * std_normal_lpdf(z_j)``.

    Parameters
    ----------
    level : GroupLevelLaw
        Level factors of the block.
    name : str
        Binder of the score.
    innovations : tuple[str, ...]
        Standard-normal level-indexed sites of the block.

    Returns
    -------
    ScoreStep or None
        The log-density factor, or None without prior weights.
    """
    if level.prior_weight is None or not innovations:
        return None
    at = _var("weighted_level")
    density = _add(
        *(
            _sub(
                _mul(
                    LetExprLiteral(value=-0.5),
                    _mul(_index(site, at), _index(site, at)),
                ),
                LetExprLiteral(value=_LOG_ROOT_TWO_PI),
            )
            for site in innovations
        )
    )
    return ScoreStep(
        name=name,
        value=_call(
            "sum",
            _factor(
                (("weighted_level", level.axis),),
                body=_mul(
                    _sub(
                        _index(level.prior_weight.input, at), LetExprLiteral(value=1.0)
                    ),
                    density,
                ),
            ),
        ),
    )


def _by_correlated(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    correlation: ByLKJCorrelation,
    innovations: str,
    steps: list[ProgramStep],
) -> LetExprNode:
    """Correlate each level's innovations with the factor of its ``by`` level.

    Each ``by`` level has its own Cholesky site with the configured
    correlation prior; the product costs O(levels x by levels x width^2).

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    law : GroupBlockLaw
        Block with a ``by`` partition.
    correlation : ByLKJCorrelation
        One Cholesky site per ``by`` level.
    innovations : str
        Binding of the coordinate-by-level innovations.
    steps : list[ProgramStep]
        Program steps receiving the correlation draws.

    Returns
    -------
    LetExprNode
        The correlated innovations, coordinate by level.

    Raises
    ------
    ValueError
        If the block declares no ``by`` partition or coordinate axis.
    """
    level, dimension = law.level, law.coordinate_axis
    if level.by is None or dimension is None:
        raise ValueError("per-by-level correlations require a by partition")
    at, row, column = _var("random_group"), _var("group_coordinate"), _var("k")
    terms = []
    for index, site in enumerate(correlation.by_sites):
        steps.append(_correlation_draw(lens, site, dimension))
        product = _call(
            "sum",
            _factor(
                (("k", dimension),),
                body=_mul(_index(site, row, column), _index(innovations, column, at)),
            ),
        )
        terms.append(
            _call(
                "select",
                LetExprBinOp(
                    op="==", left=_index(level.by.input, at), right=_integer(index)
                ),
                product,
                LetExprLiteral(value=0.0),
            )
        )
    return _factor(
        (("group_coordinate", dimension), ("random_group", level.axis)),
        body=_add(*terms),
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


def draw_group_block(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Declare a block's axes and draw its hyperparameters and carrier.

    The carrier follows the law's factors: support coordinates for a known
    level covariance, one joint Gaussian for centered correlated
    coefficients, and otherwise one site per member, holding standard-normal
    innovations or, for centered coordinates, the coefficients.

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
    correlation = law.correlation
    joint = law.parameterization == "centered" and not isinstance(
        correlation, IndependentCoordinates
    )
    dimension = law.coordinate_axis
    if _owns_coordinate_axis(law):
        assert dimension is not None
        lens._append_finite_object(statements, dimension, law.width)
    if isinstance(correlation, LKJCorrelation):
        assert dimension is not None
        steps.append(_correlation_draw(lens, correlation.site, dimension))
    lens._emitted_groups.append(law)
    for member in law.members:
        if member.mean is not None:
            steps.append(_mean_draw(lens, member))
        steps.append(_scale_draw(lens, member))
        if level.basis is None and not joint:
            _draw_member_carrier(lens, law, member, statements, steps)
    if level.basis is not None:
        _draw_support_coefficients(lens, law, statements, steps)
    elif joint:
        _draw_joint_coefficients(law, steps)


def _draw_member_carrier(
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
        If the member has no raw site, or a centered member with a component
        axis declares no binding for its per-cell scale.
    """
    level, raw, names = law.level, member.raw_site, member.binders
    if raw is None:
        raise ValueError(f"group block {law.identity!r} has no raw site")
    plate = level.axis
    if member.cell_axis is not None:
        plate = member.cell_axis
        lens._append_finite_object(statements, plate, level.size * member.width)
    if law.parameterization == "noncentered":
        steps.append(_draw(raw, "Normal", (0.0, 1.0), index=TypeName(name=plate)))
        return
    location: str | float = 0.0
    scale = member.scale.site
    if member.axis is not None:
        if names.cell_scale is None:
            raise ValueError(
                f"centered group coefficient {member.coefficient!r} needs a "
                "per-cell scale binding"
            )
        cells = [(names.cell_scale, scale)]
        scale = names.cell_scale
        if member.mean is not None and names.cell_mean is not None:
            cells.insert(0, (names.cell_mean, member.mean.site))
            location = names.cell_mean
        for name, site in cells:
            steps.append(
                _let(
                    name,
                    _factor(
                        ((names.cell, plate),),
                        body=_index(
                            site,
                            LetExprBinOp(
                                op="%",
                                left=_var(names.cell),
                                right=_integer(member.width),
                            ),
                        ),
                    ),
                )
            )
    steps.append(_draw(raw, "Normal", (location, scale), index=TypeName(name=plate)))


def _support_binding(law: GroupBlockLaw) -> str:
    """Name the level-by-coordinate coefficient matrix of a support block.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with a known level covariance.

    Returns
    -------
    str
        Binding of the physical coefficients.
    """
    return f"__formula_level_{_qvr_name(law.identity)}_coefficients"


def _draw_support_coefficients(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Draw one Gaussian law in the support coordinates of its level covariance.

    Centered coordinates carry Gaussian deviations with the block's column
    factor; noncentered coordinates are standardized and mapped through the
    support factor and the column factor. A covariance of rank zero has no
    draw and every level equals the mean.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler declaring the support axis.
    law : GroupBlockLaw
        Block with a known level covariance.
    statements : list[Statement]
        Module declarations receiving the support axis.
    steps : list[ProgramStep]
        Program steps receiving the draw and the matrix transformations.

    Raises
    ------
    ValueError
        If the block has no known covariance or coordinate axis, or one
        correlation factor per ``by`` level.
    """
    level, dimension = law.level, law.coordinate_axis
    basis = level.basis
    if basis is None or dimension is None:
        raise ValueError("a support-coordinate block needs a known covariance")
    if isinstance(law.correlation, ByLKJCorrelation):
        raise ValueError(
            f"group block {law.identity!r}: a known level covariance has one "
            "coefficient correlation for every level"
        )
    correlation = (
        law.correlation.site if isinstance(law.correlation, LKJCorrelation) else None
    )
    stem = f"__formula_level_{_qvr_name(law.identity)}"
    support = f"FormulaGroupSupport_{level.axis}"
    basis_stem = f"__formula_level_basis_{level.axis}"
    raw = law.members[0].raw_site
    scale_cases, mean_cases = [], []
    for member in law.members:
        for component in range(member.width):
            index = _integer(component)
            scale = (
                _index(member.scale.site, index)
                if member.axis is not None
                else _var(member.scale.site)
            )
            mean: LetExprNode = LetExprLiteral(value=0.0)
            if member.mean is not None:
                mean = (
                    _index(member.mean.site, index)
                    if member.axis is not None
                    else _var(member.mean.site)
                )
            scale_cases.append((member.offset + component, scale))
            mean_cases.append((member.offset + component, mean))
    scale_vector, mean_vector = stem + "_scales", stem + "_means"
    steps.extend(
        (
            _let(
                scale_vector,
                _factor((("coefficient", dimension),), cases=tuple(scale_cases)),
            ),
            _let(
                mean_vector,
                _factor((("coefficient", dimension),), cases=tuple(mean_cases)),
            ),
        )
    )
    physical = _support_binding(law)
    if not basis.rank:
        steps.append(
            _let(
                physical,
                _factor(
                    (("level", level.axis), ("coefficient", dimension)),
                    body=_index(mean_vector, _var("coefficient")),
                ),
            )
        )
        return
    if raw is None:
        raise ValueError(f"group block {law.identity!r} has no raw site")
    lens._append_finite_object(statements, support, basis.rank)
    identity_value = _call(
        "select",
        LetExprBinOp(op="==", left=_var("matrix_row"), right=_var("matrix_column")),
        LetExprLiteral(value=1.0),
        LetExprLiteral(value=0.0),
    )
    zero, column_factor = stem + "_zero", stem + "_column_factor"
    square = (("matrix_row", dimension), ("matrix_column", dimension))
    steps.append(
        _let(
            zero,
            _factor(
                (("support", support), ("coefficient", dimension)),
                body=LetExprLiteral(value=0.0),
            ),
        )
    )
    steps.append(
        _let(
            column_factor,
            _factor(
                square,
                body=_mul(
                    _index(scale_vector, _var("matrix_row")),
                    _index(correlation, _var("matrix_row"), _var("matrix_column"))
                    if correlation is not None
                    else identity_value,
                ),
            ),
        )
    )
    centered = law.parameterization == "centered"
    if centered:
        draw_column_factor = column_factor
        mapping = basis_stem + "_projection"
    else:
        draw_column_factor = stem + "_column_covariance"
        steps.append(_let(draw_column_factor, _factor(square, body=identity_value)))
        mapping = basis_stem + "_factor"
    draw = _draw(
        raw,
        "MatrixNormal",
        (zero, basis_stem + "_row_factor", draw_column_factor),
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
    mapped = stem + "_mapped"
    steps.append(_let(mapped, _call("matmul", _var(mapping), _var(raw))))
    if not centered:
        transpose = _factor(
            square,
            body=_index(column_factor, _var("matrix_column"), _var("matrix_row")),
        )
        transformed = stem + "_transformed"
        steps.append(_let(transformed, _call("matmul", _var(mapped), transpose)))
        mapped = transformed
    steps.append(
        _let(
            physical,
            _factor(
                (("level", level.axis), ("coefficient", dimension)),
                body=_add(
                    _index(mapped, _var("level"), _var("coefficient")),
                    _index(mean_vector, _var("coefficient")),
                ),
            ),
        )
    )


def _draw_joint_coefficients(law: GroupBlockLaw, steps: list[ProgramStep]) -> None:
    """Draw correlated coefficients of every level from one Gaussian law.

    Parameters
    ----------
    law : GroupBlockLaw
        Centered block with one shared correlation.
    steps : list[ProgramStep]
        Program steps receiving the scale factor and the draw.

    Raises
    ------
    ValueError
        If the block has no single correlation factor, coordinate axis, or
        shared raw site.
    """
    correlation, dimension = law.correlation, law.coordinate_axis
    raw = law.members[0].raw_site
    if not isinstance(correlation, LKJCorrelation) or dimension is None or raw is None:
        raise ValueError(
            f"centered group block {law.identity!r} needs one correlation factor "
            "over its coordinate axis"
        )
    stem = f"__formula_group_{_qvr_name(law.identity)}"
    cases = []
    for member in law.members:
        for component in range(member.width):
            cases.append(
                (
                    member.offset + component,
                    _index(member.scale.site, _integer(component))
                    if member.axis is not None
                    else _var(member.scale.site),
                )
            )
    scales, factor, zero = stem + "_scales", stem + "_factor", stem + "_zero"
    steps.extend(
        (
            _let(
                scales, _factor((("group_coordinate", dimension),), cases=tuple(cases))
            ),
            _let(
                zero,
                _factor(
                    (("group_coordinate", dimension),), body=LetExprLiteral(value=0.0)
                ),
            ),
            _let(
                factor,
                _factor(
                    (("group_row", dimension), ("group_column", dimension)),
                    body=_mul(
                        _index(scales, _var("group_row")),
                        _index(
                            correlation.site, _var("group_row"), _var("group_column")
                        ),
                    ),
                ),
            ),
        )
    )
    draw = _draw(
        raw, "MultivariateNormal", (zero, factor), index=TypeName(name=law.level.axis)
    )
    steps.append(
        draw.with_(
            axes=AxisSpec(over=(dimension,), iid_over=(law.level.axis,)),
            options=(
                OptionEntry(key="over", value=OptionName(value=dimension)),
                OptionEntry(key="iid_over", value=OptionName(value=law.level.axis)),
            ),
        )
    )


def _member_axes(
    law: GroupBlockLaw, member: GroupMemberLaw
) -> tuple[tuple[str, str], ...]:
    """Order the binders of one member's coefficient binding.

    Parameters
    ----------
    law : GroupBlockLaw
        Block of the member.
    member : GroupMemberLaw
        Member to bind.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Variable and axis of the component, if any, and of the level, in
        the member's declared order.
    """
    names = member.binders
    level = (names.level, law.level.axis)
    if member.axis is None:
        return (level,)
    component = (names.component, member.axis)
    return (level, component) if names.level_major else (component, level)


def _ordered(
    member: GroupMemberLaw, component: LetExprNode, level: LetExprNode
) -> tuple[LetExprNode, ...]:
    """Order a component index and a level index as a member binds them.

    Parameters
    ----------
    member : GroupMemberLaw
        Bound member.
    component : LetExprNode
        Component index.
    level : LetExprNode
        Level index.

    Returns
    -------
    tuple[LetExprNode, ...]
        The level alone for a scalar member, otherwise both indices in the
        member's declared order.
    """
    if member.axis is None:
        return (level,)
    return (level, component) if member.binders.level_major else (component, level)


def _gather(member: GroupMemberLaw) -> LetExprNode:
    """Read a member's coefficient at each response row's level.

    Parameters
    ----------
    member : GroupMemberLaw
        Bound member.

    Returns
    -------
    LetExprNode
        The coefficient indexed by its component, if any, and the row index
        of its row map.

    Raises
    ------
    ValueError
        If the row map is of an unknown kind.
    """
    rows = member.rows
    if not isinstance(rows, SingleLevelRows | MembershipRows):
        raise ValueError(f"unknown row map {type(rows).__name__}")
    return _index(
        member.coefficient,
        *_ordered(member, _var(member.binders.component), _var(rows.index)),
    )


def bind_group_block(
    lens: FormulaToQVRModule,
    law: GroupBlockLaw,
    steps: list[ProgramStep],
) -> GroupValues:
    """Compose a drawn block's coefficients and bind each member's slice.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    law : GroupBlockLaw
        Block whose carrier is drawn.
    steps : list[ProgramStep]
        Program steps receiving the composition and the member bindings.

    Returns
    -------
    dict[tuple[str, str, str], LetExprNode]
        Each member's coefficient gathered at the response rows' levels,
        keyed by predictor namespace, grouping factor, and coefficient label.
    """
    if law.level.basis is not None:
        _bind_support(law, steps)
    elif isinstance(law.correlation, IndependentCoordinates):
        _bind_independent(law, steps)
    else:
        _bind_correlated(lens, law, steps)
    return {
        (member.prefix, law.group, member.slope): _gather(member)
        for member in law.members
    }


def _bind_support(law: GroupBlockLaw, steps: list[ProgramStep]) -> None:
    """Bind each member's slice of a support block's coefficient matrix.

    Parameters
    ----------
    law : GroupBlockLaw
        Block with a known level covariance.
    steps : list[ProgramStep]
        Program steps receiving the member bindings.
    """
    level = law.level
    physical = _support_binding(law)
    for member in law.members:
        names = member.binders
        if names.level_major and member.width == law.width and level.mixing is None:
            # The member is the whole matrix in the matrix's own order.
            steps.append(_let(member.coefficient, _var(physical)))
            continue
        at = _var(names.level)
        coordinate: LetExprNode = _integer(member.offset)
        if member.axis is not None:
            coordinate = _add(coordinate, _var(names.component))
        steps.append(
            _let(
                member.coefficient,
                _factor(
                    _member_axes(law, member),
                    body=_mixed(level, _index(physical, at, coordinate), at),
                ),
            )
        )


def _bind_independent(law: GroupBlockLaw, steps: list[ProgramStep]) -> None:
    """Bind the members of a block with uncorrelated coordinates.

    Parameters
    ----------
    law : GroupBlockLaw
        Block without a coefficient correlation.
    steps : list[ProgramStep]
        Program steps receiving the member bindings and prior-weight score.

    Raises
    ------
    ValueError
        If a member with a component axis declares no binding for its raw
        site read over levels and components.
    """
    level = law.level
    centered = law.parameterization == "centered"
    for member in law.members:
        raw, names = member.raw_site, member.binders
        assert raw is not None
        if member.axis is not None:
            if names.matrix is None:
                raise ValueError(
                    f"group coefficient {member.coefficient!r} needs a binding "
                    "for its raw site over levels and components"
                )
            at, component = _var(names.level), _var(names.component)
            axes = _member_axes(law, member)
            flat = _add(
                _mul(at, LetExprLiteral(value=float(member.width), integral=True)),
                component,
            )
            steps.append(_let(names.matrix, _factor(axes, body=_index(raw, flat))))
            value: LetExprNode = _index(names.matrix, *_ordered(member, component, at))
            if not centered:
                value = _mul(_index(member.scale.site, component), value)
                if member.mean is not None:
                    value = _add(_index(member.mean.site, component), value)
            steps.append(_let(member.coefficient, _factor(axes, body=value)))
            continue
        if centered:
            continue
        if member.scale.by_axis is None and level.mixing is None:
            steps.append(
                _let(member.coefficient, _mul(_var(member.scale.site), _var(raw)))
            )
        else:
            at = _var("group_level")
            steps.append(
                _let(
                    member.coefficient,
                    _factor(
                        (("group_level", level.axis),),
                        body=_mixed(
                            level,
                            _mul(_scale_at(level, member, at), _index(raw, at)),
                            at,
                        ),
                    ),
                )
            )
        weighted = _prior_weight_score(level, f"{raw}_prior_weight", (raw,))
        if weighted is not None:
            steps.append(weighted)


def _bind_correlated(
    lens: FormulaToQVRModule, law: GroupBlockLaw, steps: list[ProgramStep]
) -> None:
    """Bind the members of a block with correlated coordinates.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    law : GroupBlockLaw
        Block with a coefficient correlation.
    steps : list[ProgramStep]
        Program steps receiving the correlated innovations, the prior-weight
        score and the member bindings.
    """
    level, dimension = law.level, law.coordinate_axis
    assert dimension is not None
    centered = law.parameterization == "centered"
    correlation = law.correlation
    stem = f"__formula_group_{_qvr_name(law.identity)}"
    transformed = stem + "_transformed"
    at = _var("random_group")
    if not centered:
        cases: list[tuple[int, LetExprNode]] = []
        for member in law.members:
            assert member.raw_site is not None
            for component in range(member.width):
                cell: LetExprNode = at
                if member.axis is not None:
                    cell = _add(_mul(cell, _integer(member.width)), _integer(component))
                cases.append(
                    (
                        member.offset + component,
                        _factor(
                            (("random_group", level.axis),),
                            body=_index(member.raw_site, cell),
                        ),
                    )
                )
        innovations = stem + "_innovations"
        steps.append(
            _let(
                innovations,
                _factor((("group_coordinate", dimension),), cases=tuple(cases)),
            )
        )
        if isinstance(correlation, ByLKJCorrelation):
            product = _by_correlated(lens, law, correlation, innovations, steps)
        else:
            assert isinstance(correlation, LKJCorrelation)
            product = _call("matmul", _var(correlation.site), _var(innovations))
        steps.append(_let(transformed, product))
        weighted = _prior_weight_score(
            level,
            stem + "_prior_weight",
            tuple(
                member.raw_site for member in law.members if member.raw_site is not None
            ),
        )
        if weighted is not None:
            steps.append(weighted)
    for member in law.members:
        coordinate: LetExprNode = _integer(member.offset)
        binders: tuple[tuple[str, str], ...] = (("random_group", level.axis),)
        if member.axis is not None:
            coordinate = _add(coordinate, _var("category_logit"))
            scale: LetExprNode = _index(member.scale.site, _var("category_logit"))
            binders = (("category_logit", member.axis), *binders)
        else:
            scale = _scale_at(level, member, at)
        if centered:
            assert member.raw_site is not None
            value: LetExprNode = _index(member.raw_site, at, coordinate)
        else:
            value = _mul(scale, _index(transformed, coordinate, at))
        steps.append(
            _let(member.coefficient, _factor(binders, body=_mixed(level, value, at)))
        )


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
    return bind_group_block(lens, law, steps)


def row_contribution(rows: RowMap, gathered: LetExprNode) -> LetExprNode:
    """Carry a coefficient gathered at the rows' levels to a row contribution.

    Parameters
    ----------
    rows : RowMap
        Row map of the coefficient.
    gathered : LetExprNode
        The coefficient indexed last by the single-level row index.

    Returns
    -------
    LetExprNode
        The gather times the row covariate, or for a multi-membership factor
        the sum over members of the member's gather times its weighted
        covariate.

    Raises
    ------
    ValueError
        If a multi-membership coefficient is not gathered by the factor's
        row index, or the row map is of an unknown kind.
    """
    if isinstance(rows, MembershipRows):
        index = _var(rows.index)
        if not isinstance(gathered, LetExprIndex) or gathered.indices[-1:] != (index,):
            raise ValueError(
                f"multi-membership rows of {rows.index!r} require a level-indexed "
                "coefficient"
            )
        return _add(
            *(
                _mul(
                    gathered.with_(indices=(*gathered.indices[:-1], _var(member))),
                    _var(covariate),
                )
                for member, covariate in zip(rows.indices, rows.covariates, strict=True)
            )
        )
    if isinstance(rows, SingleLevelRows):
        if rows.covariate is None:
            return gathered
        return _mul(gathered, _var(rows.covariate))
    raise ValueError(f"unknown row map {type(rows).__name__}")
