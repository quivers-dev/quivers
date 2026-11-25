"""Stateful ``s()`` and ``t2()`` formula terms with retained mgcv bases.

Each term is a formulae stateful transform. Its first call constructs the
mgcv basis on the training rows, absorbs the identifiability constraint,
and splits the result into brms's unpenalized and penalized coordinates
(`quivers.formulas._mgcv_smooth`). Later calls evaluate fresh rows through
the retained basis and coefficient maps, never through a refitted basis.

The evaluated columns are ordered with every unpenalized column first,
followed by each penalized group, so the unpenalized columns become ordinary
population coefficients and each penalized group one Gaussian design block.
"""

from __future__ import annotations

from collections.abc import Mapping

from formulae.matrices import DesignMatrices
from formulae.terms.terms import Term
import numpy as np
import numpy.typing as npt
import pandas as pd

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.family_catalog import FamilyCatalog
import didactic.api as dx

from quivers.formulas._data_options import (
    Data2Entry,
    dropping_unused_levels,
    smooth_knots,
)
from quivers.formulas._mgcv_smooth import (
    FactorSmoothBasis,
    MarkovRandomFieldBasis,
    RandomEffectBasis,
    RawSmooth,
    SmoothBlock,
    SmoothBasis,
    b_spline,
    cubic_regression,
    cyclic_cubic,
    factor_smooth,
    duchon_spline,
    gp_smooth,
    markov_random_field,
    mixed_model_blocks,
    neighbourhood_penalty,
    p_spline,
    random_effect,
    sphere_order,
    sphere_spline,
    tensor_t2,
    thin_plate,
)
from quivers.formulas._term import (
    AxisDecl,
    InputDecl,
    OutputAxis,
    SiteDecl,
    design_declarations,
)

type ColumnInput = pd.Series | npt.NDArray[np.generic]
type Order = int | float | tuple[int | float, ...] | None
type ExtraValue = str | int | float | bool | Data2Entry | None
type Extra = str | Mapping[str, ExtraValue] | None
"""mgcv's ``xt`` argument: a basis name, or the entries of an R ``list()``."""

_UNIVARIATE = frozenset({"cr", "cs", "cc", "ps", "bs"})
#: Bases that read an ``xt`` entry.
_XT_BASES = frozenset({"fs", "mrf"})
_SMOOTH_BASES = frozenset(
    {"tp", "ts", "cr", "cs", "cc", "ps", "bs", "re", "fs", "gp", "mrf", "sos", "ds"}
)
#: Bases whose knots are rows of several covariates.
_JOINT_KNOT_BASES = frozenset({"tp", "ts", "ds"})
_MARGIN_BASES = _SMOOTH_BASES - {"re", "fs", "gp", "mrf", "sos"}
#: Bases a factor-smooth interaction can copy per level: one penalty with a
#: polynomial null space.
_FS_BASES = frozenset({"tp", "cr", "cc", "ps", "bs", "gp"})


def r_list(*positional: ExtraValue, **named: ExtraValue) -> dict[str, ExtraValue]:
    """Evaluate R's ``list()`` for a smooth's ``xt`` argument.

    Parameters
    ----------
    *positional : str, int, float, bool, Data2Entry, or None
        Unnamed entries, which are refused.
    **named : str, int, float, bool, Data2Entry, or None
        Named entries: literals, or ``data2`` entries named in the formula.

    Returns
    -------
    dict[str, ExtraValue]
        The named entries in order.

    Raises
    ------
    ValueError
        If an entry is unnamed.
    """
    if positional:
        raise ValueError("list() in a formula takes named entries, as in list(bs='cr')")
    return dict(named)


def _xt_text(xt: Extra) -> str:
    """Describe ``xt`` for comparing smooths that share an ``id``.

    Parameters
    ----------
    xt : str, Mapping, or None
        Authored ``xt``.

    Returns
    -------
    str
        The entries, with a matrix written by its shape and contents.
    """
    if not isinstance(xt, Mapping):
        return repr(xt)
    return repr(
        {
            name: (
                (value.value.shape, np.asarray(value.value).tobytes())
                if isinstance(value, Data2Entry)
                else value
            )
            for name, value in xt.items()
        }
    )


def _fs_base(xt: Extra) -> str:
    """Read the base basis of a factor-smooth interaction from ``xt``.

    mgcv reads ``xt = "cr"`` and ``xt = list(bs = "cr")`` alike and defaults
    to a thin-plate base.

    Parameters
    ----------
    xt : str, Mapping, or None
        Authored ``xt``.

    Returns
    -------
    str
        Base basis name.

    Raises
    ------
    ValueError
        If ``xt`` has other entries, or names a base mgcv cannot use.
    """
    if xt is None:
        return "tp"
    if isinstance(xt, str):
        base = xt
    else:
        if set(xt) - {"bs"}:
            raise ValueError(
                f"bs='fs' reads only xt = list(bs = ...); found {sorted(xt)!r}"
            )
        base = xt.get("bs", "tp")
    if not isinstance(base, str):
        raise ValueError("the xt base of a bs='fs' smooth must be a basis name")
    if base in {"ts", "cs"}:
        raise ValueError(
            f"bs='fs' cannot copy the shrinkage basis {base!r}: its penalty has "
            "full rank, and mgcv's construction (and so brms) fails without a "
            "penalty null space"
        )
    if base not in _FS_BASES:
        raise ValueError(
            f"unsupported xt base {base!r} for bs='fs'; supported bases are "
            f"{sorted(_FS_BASES)}"
        )
    return base


class ByVariable(ValueModel):
    """Retained ``by`` variable of a smooth.

    Attributes
    ----------
    name : str
        Variable name used in mgcv block labels.
    levels : tuple[str, ...]
        Factor levels producing one block each, empty for a numeric ``by``.
    known : tuple[str, ...]
        Every training factor level, including an ordered factor's first
        level, which selects no block.
    """

    name: str
    levels: tuple[str, ...] = ()
    known: tuple[str, ...] = ()


class SmoothLayout(ValueModel):
    """Positions of one block's coordinates among a term's evaluated columns.

    Attributes
    ----------
    label : str
        mgcv block label.
    fixed : tuple[int, ...]
        Unpenalized columns, carried as population coefficients.
    random : tuple[tuple[int, ...], ...]
        Penalized groups, each a Gaussian design block with its own scale.
    """

    label: str
    fixed: tuple[int, ...]
    random: tuple[tuple[int, ...], ...]


class SmoothState(ValueModel):
    """Everything a fitted smooth retains for evaluation on fresh rows.

    Attributes
    ----------
    label : str
        mgcv term label.
    covariates : tuple[str, ...]
        Covariate names, in authored order.
    basis : SmoothBasis
        Retained raw basis.
    by : ByVariable or None
        Retained ``by`` variable.
    blocks : tuple[SmoothBlock, ...]
        Mixed-model blocks with their coefficient maps.
    constraint : tuple[float, ...]
        Training raw-basis means defining the sum-to-zero constraint.
    identity : str or None
        mgcv ``id`` shared by smooths with one smoothing scale.
    spec : tuple[str, ...]
        Normalized basis specification compared across shared ``id`` terms.
    data2 : tuple[str, ...]
        Names of the ``data2`` entries the term's ``xt`` argument read.
    """

    label: str
    covariates: tuple[str, ...]
    basis: SmoothBasis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    by: ByVariable | None = None
    blocks: tuple[SmoothBlock, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    constraint: tuple[float, ...] = ()
    identity: str | None = None
    spec: tuple[str, ...] = ()
    data2: tuple[str, ...] = ()

    @property
    def layout(self) -> tuple[SmoothLayout, ...]:
        """Column positions of each block, all unpenalized columns first.

        Returns
        -------
        tuple[SmoothLayout, ...]
            One layout per block.
        """
        fixed_total = sum(block.fixed_width for block in self.blocks)
        fixed_start, random_start = 0, fixed_total
        layouts = []
        for block in self.blocks:
            fixed = tuple(range(fixed_start, fixed_start + block.fixed_width))
            fixed_start += block.fixed_width
            groups = []
            for width in block.random_widths:
                groups.append(tuple(range(random_start, random_start + width)))
                random_start += width
            layouts.append(
                SmoothLayout(label=block.label, fixed=fixed, random=tuple(groups))
            )
        return tuple(layouts)


def _name(value: ColumnInput, position: int) -> str:
    """Return a covariate's name, or a positional placeholder.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated covariate.
    position : int
        Position among the smooth's covariates.

    Returns
    -------
    str
        Series name when present.
    """
    name = getattr(value, "name", None)
    return str(name) if name is not None else f"x{position + 1}"


def _numeric(value: ColumnInput, role: str) -> npt.NDArray[np.float64]:
    """Validate a finite real covariate column.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated covariate.
    role : str
        Description used in errors.

    Returns
    -------
    numpy.ndarray
        Double values.

    Raises
    ------
    ValueError
        If the column is not a finite real vector.
    """
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{role} must be a finite real row vector")
    return array.astype(np.float64)


def _factor_levels(value: ColumnInput) -> tuple[tuple[str, ...], bool] | None:
    """Return R factor levels for a categorical column, or None if numeric.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated covariate.

    Returns
    -------
    tuple[tuple[str, ...], bool] or None
        Levels in R's order and whether the factor is ordered.

    Raises
    ------
    ValueError
        If the column is Boolean or contains missing labels.
    """
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        if pd.isna(value).any():
            raise ValueError("smooth factor columns cannot contain missing values")
        return tuple(str(level) for level in dtype.categories), bool(dtype.ordered)
    array = np.asarray(value)
    if array.dtype.kind == "b":
        raise ValueError(
            "Boolean smooth covariates are ambiguous; convert them to a factor or number"
        )
    if array.dtype.kind in "iuf":
        return None
    if pd.isna(pd.Series(array)).any():
        raise ValueError("smooth factor columns cannot contain missing values")
    return tuple(sorted({str(label) for label in array.tolist()})), False


def _labels(value: ColumnInput) -> npt.NDArray[np.str_]:
    """Return a categorical column's labels as strings.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated factor.

    Returns
    -------
    numpy.ndarray
        One label per row.
    """
    return np.asarray(pd.Series(value).astype(str))


def _mrf_structure(xt: Extra, levels: tuple[str, ...]) -> npt.NDArray[np.float64]:
    """Read a Markov random field penalty for the regions from ``xt``.

    Parameters
    ----------
    xt : str, Mapping, or None
        Authored ``xt``: ``list(penalty = P)`` with a penalty matrix, or
        ``list(nb = A)`` with a neighbourhood matrix whose row ``i`` is
        nonzero at the neighbours of region ``i``. Either is a ``data2``
        entry: a data frame labeled by region on both axes, or an unlabeled
        array in the order of the factor levels.
    levels : tuple[str, ...]
        Regions in factor-level order.

    Returns
    -------
    numpy.ndarray
        Penalty matrix in level order.

    Raises
    ------
    ValueError
        If ``xt`` does not hold exactly one of ``penalty`` and ``nb``, the
        matrix is not square, or its labels are not the regions.
    """
    if not isinstance(xt, Mapping) or len(set(xt) & {"penalty", "nb"}) != 1:
        raise ValueError(
            "bs='mrf' requires xt = list(penalty = P) or xt = list(nb = A), "
            "with P or A a matrix given in data2"
        )
    if set(xt) - {"penalty", "nb"}:
        raise ValueError(
            "bs='mrf' reads xt = list(penalty = ...) or xt = list(nb = ...); "
            f"found {sorted(xt)!r} (boundary polygons are not supported)"
        )
    role = "penalty" if "penalty" in xt else "nb"
    entry = xt[role]
    if not isinstance(entry, Data2Entry):
        raise ValueError(
            f"the Markov random field {role} must be a matrix given in data2 and "
            f"named in xt, as in fit(..., data2={{'P': P}}) with xt = list({role} = P)"
        )
    value = entry.value
    if isinstance(value, pd.DataFrame):
        rows = [str(label) for label in value.index]
        columns = [str(label) for label in value.columns]
        if rows != columns or len(set(rows)) != len(rows):
            raise ValueError(
                f"the Markov random field {role} matrix must carry the same "
                "distinct region labels on its rows and columns"
            )
        if set(rows) != set(levels):
            raise ValueError(
                f"the {role} matrix labels do not match the regions of the data: "
                f"only in the matrix {sorted(set(rows) - set(levels))!r}, only in "
                f"the data {sorted(set(levels) - set(rows))!r}"
            )
        order = [rows.index(level) for level in levels]
        matrix = np.asarray(value.to_numpy(), dtype=np.float64)[np.ix_(order, order)]
    else:
        if value.dtype.kind not in "iufb":
            raise ValueError(f"the Markov random field {role} matrix must be numeric")
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(levels), len(levels)):
        raise ValueError(
            f"the Markov random field {role} matrix has shape {matrix.shape}; "
            f"the {len(levels)} regions need a {len(levels)} by {len(levels)} matrix"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"the Markov random field {role} matrix must be finite")
    return neighbourhood_penalty(matrix) if role == "nb" else matrix


def _orders(m: Order, bs: str) -> tuple[int, int] | int:
    """Normalize mgcv's ``m`` argument for one basis.

    Parameters
    ----------
    m : int, tuple of int, or None
        Authored penalty order.
    bs : str
        Basis name.

    Returns
    -------
    tuple[int, int] or int
        B-spline order pair, or thin-plate order (negative for a dropped
        null space encoded as ``-(m + 1)``).

    Raises
    ------
    ValueError
        If the order is malformed or not accepted by the basis.
    """
    if m is not None and (
        isinstance(m, bool)
        or not (
            isinstance(m, int)
            or (isinstance(m, tuple) and m and all(type(v) is int for v in m))
        )
    ):
        raise ValueError("smooth m must be an integer or c() of integers")
    values: tuple[int, ...] = () if m is None else ((m,) if isinstance(m, int) else m)
    if any(v < 0 for v in values):
        raise ValueError("smooth penalty orders must be nonnegative")
    if bs == "ps":
        if len(values) > 2:
            raise ValueError("bs='ps' takes m or c(m1, m2)")
        return (
            (2, 2)
            if not values
            else (values[0], values[0] if len(values) == 1 else values[1])
        )
    if bs == "bs":
        if len(values) > 2:
            raise ValueError(
                "bs='bs' with several penalties has several smoothing scales; brms supports one"
            )
        if not values:
            return (3, 2)
        return (
            (values[0], max(0, values[0] - 1))
            if len(values) == 1
            else (values[0], values[1])
        )
    if bs in {"tp", "ts"}:
        if len(values) > 2:
            raise ValueError("thin-plate smooths take m or c(m, 0)")
        order = values[0] if values else 0
        if len(values) == 2:
            if values[1] != 0:
                raise ValueError(
                    "thin-plate m=c(m, k) only accepts k=0, which drops the null space"
                )
            return -(order + 1)
        return order
    if values:
        raise ValueError(f"bs={bs!r} has a fixed penalty order; m is not accepted")
    return 0


def _construct(
    bs: str,
    columns: list[ColumnInput],
    k: int,
    m: Order,
    knots: npt.NDArray[np.float64] | None = None,
    xt: Extra = None,
) -> RawSmooth:
    """Construct one raw mgcv smooth on training columns.

    Parameters
    ----------
    bs : str
        Basis name.
    columns : list
        Covariate columns.
    k : int
        Basis dimension, negative for the default.
    m : int, tuple of int, or None
        Authored penalty order.
    knots : numpy.ndarray or None
        brms ``knots`` of the covariate, passed to the ``cr``, ``cs``,
        ``cc``, ``ps`` and ``bs`` constructors as mgcv passes them, or the
        knot rows (knots by covariates) of a ``tp``, ``ts``, ``ds`` or
        ``sos`` basis; a random-effect basis reads none, as in mgcv.
    xt : str, Mapping, or None
        mgcv's ``xt``: the base of a ``bs = "fs"`` smooth. Other bases
        read no ``xt`` entry and refuse one.

    Returns
    -------
    RawSmooth
        Constructed smooth.

    Raises
    ------
    ValueError
        If the basis, dimension, covariates, or knots are invalid.
    """
    if bs in {"ad", "so", "sf", "sw"}:
        raise ValueError(
            f"smooth basis bs={bs!r} has several overlapping penalties, which mgcv "
            "cannot write as random effects; brms refuses it as well"
        )
    if bs not in _SMOOTH_BASES:
        raise ValueError(
            f"unsupported smooth basis bs={bs!r}; supported bases are {sorted(_SMOOTH_BASES)}"
        )
    if xt is not None and bs == "sos":
        raise ValueError(
            "xt for a bs='sos' smooth is not supported: mgcv reads from it the "
            "size and seed of a knot subsample drawn with R's random number "
            "generator"
        )
    if xt is not None and bs not in _XT_BASES:
        raise ValueError(
            f"bs={bs!r} reads no xt entry; xt is accepted for {sorted(_XT_BASES)}"
        )
    if bs == "re":
        if m is not None:
            raise ValueError("bs='re' has a fixed penalty; m is not accepted")
        levels = []
        for column in columns:
            found = _factor_levels(column)
            levels.append(() if found is None else found[0])
            if found is None:
                _numeric(column, "random-effect smooth covariate")
        return random_effect(columns, tuple(levels))
    if bs == "mrf":
        if m is not None:
            raise ValueError("bs='mrf' takes its penalty from xt; m is not accepted")
        if knots is not None:
            raise ValueError("knots for a bs='mrf' smooth are not supported")
        found = _factor_levels(columns[0]) if len(columns) == 1 else None
        if found is None:
            raise ValueError(
                "bs='mrf' requires one factor of regions, as in "
                "s(region, bs='mrf', xt = list(penalty = P))"
            )
        labels = _labels(columns[0])
        present = set(labels.tolist())
        levels = tuple(
            level
            for level in found[0]
            if level in present or not dropping_unused_levels()
        )
        return markov_random_field(labels, levels, _mrf_structure(xt, levels), k)
    if bs == "fs":
        factors = [
            (position, found[0])
            for position, column in enumerate(columns)
            if (found := _factor_levels(column)) is not None
        ]
        if len(factors) != 1 or len(columns) < 2:
            raise ValueError(
                "bs='fs' requires exactly one factor and at least one numeric "
                "covariate, as in s(x, g, bs='fs')"
            )
        if knots is not None:
            raise ValueError("knots for a bs='fs' smooth are not supported")
        for position, column in enumerate(columns):
            if position != factors[0][0]:
                _numeric(column, "smooth covariate")
        base = _fs_base(xt)
        if base == "tp":
            order = _orders(m, "tp")
            if not isinstance(order, int) or order < 0:
                raise ValueError("bs='fs' takes one penalty order m")
        position, levels_found = factors[0]
        present = set(_labels(columns[position]).tolist())
        return factor_smooth(
            _construct(
                base,
                [column for index, column in enumerate(columns) if index != position],
                k,
                m,
            ),
            [
                _labels(column)
                if index == position
                else _numeric(column, "smooth covariate")
                for index, column in enumerate(columns)
            ],
            position,
            tuple(level for level in levels_found if level in present),
        )
    if bs == "sos" and len(columns) != 2:
        raise ValueError(
            "bs='sos' takes exactly two covariates, latitude and longitude in "
            "degrees, as in s(lat, lon, bs='sos')"
        )
    x = np.column_stack([_numeric(column, "smooth covariate") for column in columns])
    if bs == "sos":
        if isinstance(m, (tuple, bool)) or not isinstance(m, (int, float, type(None))):
            raise ValueError("bs='sos' takes one m from -2 to 4")
        return sphere_spline(
            x, k, 0 if m is None else sphere_order(float(m)), knots=knots
        )
    if bs == "gp":
        if knots is not None:
            raise ValueError("knots for a bs='gp' smooth are not supported")
        definition = () if m is None else (m,) if not isinstance(m, tuple) else m
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in definition
        ):
            raise ValueError("bs='gp' takes numeric m = c(type, range, power)")
        return gp_smooth(x, k, tuple(float(value) for value in definition))
    if bs == "ds":
        definition = () if m is None else (m,) if not isinstance(m, tuple) else m
        if len(definition) > 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in definition
        ):
            raise ValueError("bs='ds' takes numeric m = c(m, s)")
        if definition and definition[0] != round(definition[0]):
            raise ValueError("bs='ds' takes an integer penalty order m[1]")
        return duchon_spline(
            x,
            k,
            round(definition[0]) if definition else 2,
            float(definition[1]) if len(definition) > 1 else 0.0,
            knots=None if knots is None else knots.reshape(-1, x.shape[1]),
        )
    order = _orders(m, bs)
    if bs in _UNIVARIATE and x.shape[1] != 1:
        raise ValueError(
            f"bs={bs!r} is a one-dimensional basis; use bs='tp' or t2() for several covariates"
        )
    if bs in {"tp", "ts"}:
        assert isinstance(order, int)
        drop = order < 0
        return thin_plate(
            x,
            k,
            -order - 1 if drop else order,
            shrink=0.1 if bs == "ts" else None,
            drop_null=drop,
            knots=None if knots is None else knots.reshape(-1, x.shape[1]),
        )
    if bs in {"cr", "cs"}:
        return cubic_regression(x, k, shrink=0.1 if bs == "cs" else None, knots=knots)
    if bs == "cc":
        return cyclic_cubic(x, k, knots=knots)
    assert isinstance(order, tuple)
    return (
        p_spline(x, k, order, knots=knots)
        if bs == "ps"
        else b_spline(x, k, order, knots=knots)
    )


def _check_k(k: int | tuple[int, ...] | None, role: str) -> None:
    """Validate an integer basis dimension argument.

    Parameters
    ----------
    k : int, tuple of int, or None
        Authored dimension.
    role : str
        Description used in errors.

    Raises
    ------
    ValueError
        If any dimension is not an integer.
    """
    values = () if k is None else ((k,) if isinstance(k, int) else k)
    if (
        isinstance(k, bool)
        or not isinstance(values, tuple)
        or any(type(v) is not int for v in values)
    ):
        raise ValueError(f"{role} k must be an integer or c() of integers")


def _identity(value: int | str | None) -> str | None:
    """Normalize mgcv's ``id`` argument.

    Parameters
    ----------
    value : int, str, or None
        Authored identifier.

    Returns
    -------
    str or None
        Identifier text.

    Raises
    ------
    ValueError
        If the identifier is not an integer or string.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("smooth id must be an integer or string")
    return str(value)


class SmoothTransform:
    """Shared stateful evaluation of a retained mgcv smooth."""

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without retained training state."""
        self.state: SmoothState | None = None

    def _fit(
        self,
        raw: RawSmooth,
        *,
        label: str,
        covariates: tuple[str, ...],
        by: ColumnInput | None,
        identity: str | None,
        spec: tuple[str, ...],
        data2: tuple[str, ...] = (),
    ) -> None:
        """Retain the constructed basis and its mixed-model blocks.

        Parameters
        ----------
        raw : RawSmooth
            Constructed smooth.
        label : str
            mgcv term label.
        covariates : tuple[str, ...]
            Covariate names.
        by : pandas.Series, numpy.ndarray, or None
            Training ``by`` column.
        identity : str or None
            Normalized ``id``.
        spec : tuple[str, ...]
            Normalized specification for ``id`` checks.
        data2 : tuple[str, ...]
            Names of the ``data2`` entries the term read.

        Raises
        ------
        ValueError
            If the ``by`` variable is invalid.
        """
        by_state = None
        by_values = None
        by_labels = None
        levels: tuple[str, ...] = ()
        if by is not None:
            name = _name(by, 0)
            found = _factor_levels(by)
            if found is None:
                by_values = _numeric(by, "numeric smooth by variable")
                by_state = ByVariable(name=name)
            else:
                known, ordered = found
                levels = known[1:] if ordered and len(known) > 1 else known
                by_labels = _labels(by)
                by_state = ByVariable(name=name, levels=levels, known=known)
        constraint, blocks = mixed_model_blocks(
            raw,
            label=label,
            by_values=by_values,
            by_levels=levels,
            by_labels=by_labels,
            by_name=by_state.name if by_state is not None else "",
        )
        self.state = SmoothState(
            label=label,
            covariates=covariates,
            basis=raw.basis,
            by=by_state,
            blocks=blocks,
            constraint=tuple(constraint.tolist()) if raw.constraint == "mean" else (),
            identity=identity,
            spec=spec,
            data2=data2,
        )

    def _evaluate(
        self, raw: npt.NDArray[np.float64], by: ColumnInput | None
    ) -> npt.NDArray[np.float64]:
        """Map raw basis rows to the retained mixed-model columns.

        Costs O(rows times raw width times evaluated width).

        Parameters
        ----------
        raw : numpy.ndarray
            Rows evaluated in the retained raw basis.
        by : pandas.Series, numpy.ndarray, or None
            ``by`` column of the evaluated rows.

        Returns
        -------
        numpy.ndarray
            Unpenalized columns of every block, then each penalized group.

        Raises
        ------
        ValueError
            If the ``by`` variable disagrees with its training role.
        """
        state = self.state
        assert state is not None
        if (by is None) != (state.by is None):
            raise ValueError(
                "smooth by variable changed between training and evaluation"
            )
        mapped = []
        for block in state.blocks:
            values = raw @ block.coefficient_map
            if state.by is not None and by is not None:
                if state.by.known:
                    if _factor_levels(by) is None:
                        raise ValueError("smooth factor by variable became numeric")
                    labels = _labels(by)
                    unknown = set(labels.tolist()) - set(state.by.known)
                    if unknown:
                        raise ValueError(
                            f"smooth by variable has unseen levels {sorted(unknown)!r}"
                        )
                    values = (labels == block.level).astype(float)[:, None] * values
                else:
                    values = (
                        _numeric(by, "numeric smooth by variable")[:, None] * values
                    )
            mapped.append(values)
        fixed = [
            values[:, : block.fixed_width]
            for values, block in zip(mapped, state.blocks, strict=True)
        ]
        random = []
        for values, block in zip(mapped, state.blocks, strict=True):
            start = block.fixed_width
            for width in block.random_widths:
                random.append(values[:, start : start + width])
                start += width
        return np.column_stack(fixed + random)


def _check_sp(sp: float | tuple[float, ...] | None, term: str) -> None:
    """Check mgcv's ``sp`` argument, which brms accepts and does not use.

    Parameters
    ----------
    sp : float, tuple of float, or None
        Fixed smoothing parameters; a negative value asks mgcv to estimate.
    term : str
        Term named in the error.

    Raises
    ------
    ValueError
        If a value is not a finite real number.
    """
    if sp is None:
        return
    for value in sp if isinstance(sp, tuple) else (sp,):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{term} sp must hold finite numbers")


def _sphere_knots(names: tuple[str, ...]) -> npt.NDArray[np.float64] | None:
    """Collect the brms ``knots`` of a spline on the sphere.

    Parameters
    ----------
    names : tuple[str, ...]
        Covariate names, latitude then longitude.

    Returns
    -------
    numpy.ndarray or None
        Knot latitude and longitude, shape knots by two, or None when no
        knots were given for either covariate.

    Raises
    ------
    ValueError
        If knots are given for one covariate alone, which mgcv ignores, or
        the two knot vectors differ in length.
    """
    found = [smooth_knots(name) for name in names]
    given = [values for values in found if values is not None]
    if not given:
        return None
    if len(given) != len(names):
        raise ValueError(
            "knots for a bs='sos' smooth must be given for both covariates; "
            "mgcv ignores knots for one of them"
        )
    if len({values.size for values in given}) != 1:
        raise ValueError(
            "components of knots relating to a single smooth must be of same length"
        )
    return np.column_stack(given)


def _joint_knots(names: tuple[str, ...], bs: str) -> npt.NDArray[np.float64] | None:
    """Collect the brms ``knots`` of a smooth of several covariates.

    Parameters
    ----------
    names : tuple[str, ...]
        Covariate names of one smooth or one tensor margin.
    bs : str
        Basis name used in errors.

    Returns
    -------
    numpy.ndarray or None
        Knot rows, shape knots by covariates, or None when no knots were
        given for any covariate.

    Raises
    ------
    ValueError
        If knots are given for some covariates alone, which mgcv ignores,
        or the knot vectors differ in length.
    """
    found = [smooth_knots(name) for name in names]
    given = [values for values in found if values is not None]
    if not given:
        return None
    if len(given) != len(names):
        raise ValueError(
            f"knots for a bs={bs!r} smooth of {list(names)} must be given for "
            "every covariate; mgcv ignores knots for some of them"
        )
    if len({values.size for values in given}) != 1:
        raise ValueError(
            "components of knots relating to a single smooth must be of same length"
        )
    return np.column_stack(given)


class Smooth(SmoothTransform):
    """The brms and mgcv ``s()`` term, retaining its training basis."""

    def __call__(
        self,
        *covariates: ColumnInput,
        k: int = -1,
        bs: str = "tp",
        m: Order = None,
        by: ColumnInput | None = None,
        id: int | str | None = None,  # noqa: A002
        fx: bool = False,
        sp: float | tuple[float, ...] | None = None,
        xt: Extra = None,
        **unsupported: str | int | float | bool | None,
    ) -> npt.NDArray[np.float64]:
        """Construct on the first call and evaluate the retained basis after.

        Parameters
        ----------
        *covariates : pandas.Series or numpy.ndarray
            Smooth covariates, one column each.
        k : int
            Basis dimension, negative for mgcv's default.
        bs : str
            Basis: ``tp`` (default), ``ts``, ``cr``, ``cs``, ``cc``, ``ps``,
            ``bs``, ``re``, ``gp``, ``ds`` (Duchon spline), ``fs`` (one
            factor among the covariates), ``mrf`` (one factor of regions),
            or ``sos`` (latitude and longitude in degrees).
        m : int, tuple of int, or None
            Penalty order, as mgcv's ``m``.
        by : pandas.Series, numpy.ndarray, or None
            Numeric multiplier or factor producing one smooth per level.
        id : int, str, or None
            Smooths sharing an ``id`` share one smoothing scale.
        fx : bool
            Fit an unpenalized regression spline.
        sp : float, tuple of float, or None
            mgcv's fixed smoothing parameters. brms estimates the smoothing
            standard deviations and builds the same design with or without
            ``sp``, so the checked value changes nothing here either.
        xt : str, Mapping, or None
            mgcv's extra basis information: for ``bs = "fs"`` the base
            copied per factor level, as ``xt = "cr"`` or
            ``xt = list(bs = "cr")``; for ``bs = "mrf"`` the penalty or
            neighbourhood matrix, as ``xt = list(penalty = P)`` or
            ``xt = list(nb = A)`` with ``P`` or ``A`` an entry of ``data2``.
        **unsupported : str or int or float or bool or None
            Rejected mgcv arguments.

        Returns
        -------
        numpy.ndarray
            Unpenalized columns, then penalized groups.

        Raises
        ------
        ValueError
            If an argument, basis, or covariate is invalid.
        """
        if unsupported:
            raise ValueError(
                f"s() arguments {sorted(unsupported)!r} are not supported; "
                "supported arguments are k, bs, m, by, id, fx, sp, and xt"
            )
        _check_sp(sp, "s()")
        if not covariates:
            raise ValueError("s() requires at least one covariate")
        if not isinstance(bs, str):
            raise ValueError("s() bs must be a basis name string")
        if type(fx) is not bool:
            raise ValueError("s() fx must be TRUE or FALSE")
        _check_k(k, "s()")
        if not isinstance(k, int):
            raise ValueError("s() k must be one integer")
        if fx and id is not None:
            raise ValueError(
                "an unpenalized smooth (fx=TRUE) has no smoothing scale to share by id"
            )
        if bs == "re" and id is not None:
            raise ValueError("random-effect smooths (bs='re') cannot share an id")
        if bs == "fs" and fx:
            raise ValueError(
                "an unpenalized factor-smooth interaction (bs='fs', fx=TRUE) is "
                "not supported: mgcv cannot write it as random effects, and "
                "brms fails on it"
            )
        if self.state is None:
            names = tuple(_name(column, i) for i, column in enumerate(covariates))
            if len(set(names)) != len(names):
                raise ValueError(
                    "repeated variables as arguments of a smooth are not permitted"
                )
            # brms knots of the covariates.
            given = [name for name in names if smooth_knots(name) is not None]
            if given and len(names) > 1 and bs != "sos" and bs not in _JOINT_KNOT_BASES:
                raise ValueError(
                    f"knots for {given} of a multivariate bs={bs!r} smooth are "
                    "not supported"
                )
            raw = _construct(
                bs,
                list(covariates),
                k,
                m,
                _sphere_knots(names)
                if bs == "sos"
                else _joint_knots(names, bs)
                if len(names) > 1
                else smooth_knots(given[0])
                if given
                else None,
                xt,
            )
            if fx:
                raw = raw.with_(fixed=True)
            self._fit(
                raw,
                label=f"s({','.join(names)})",
                covariates=names,
                by=by,
                identity=_identity(id),
                spec=("s", bs, str(k), repr(m), str(len(names)), _xt_text(xt)),
                data2=tuple(
                    entry.name
                    for entry in (xt.values() if isinstance(xt, Mapping) else ())
                    if isinstance(entry, Data2Entry)
                ),
            )
            return self._evaluate(raw.design, by)
        basis = self.state.basis
        if isinstance(basis, RandomEffectBasis):
            return self._evaluate(basis.evaluate(list(covariates)), by)
        if isinstance(basis, MarkovRandomFieldBasis):
            if len(covariates) != 1 or _factor_levels(covariates[0]) is None:
                raise ValueError("a Markov random field smooth takes one factor")
            return self._evaluate(basis.evaluate([_labels(covariates[0])]), by)
        if isinstance(basis, FactorSmoothBasis):
            return self._evaluate(
                basis.evaluate(
                    [
                        _labels(column)
                        if position == basis.factor
                        else _numeric(column, "smooth covariate")
                        for position, column in enumerate(covariates)
                    ]
                ),
                by,
            )
        x = np.column_stack(
            [_numeric(column, "smooth covariate") for column in covariates]
        )
        return self._evaluate(basis.evaluate(x), by)


class TensorSmooth(SmoothTransform):
    """The brms and mgcv ``t2()`` tensor product with separate penalties."""

    def __call__(
        self,
        *covariates: ColumnInput,
        k: int | tuple[int, ...] | None = None,
        bs: str | tuple[str, ...] = "cr",
        m: Order = None,
        d: int | tuple[int, ...] | None = None,
        by: ColumnInput | None = None,
        id: int | str | None = None,  # noqa: A002
        full: bool = False,
        ord: int | tuple[int, ...] | None = None,  # noqa: A002
        sp: float | tuple[float, ...] | None = None,
        **unsupported: str | int | float | bool | None,
    ) -> npt.NDArray[np.float64]:
        """Construct on the first call and evaluate the retained basis after.

        Parameters
        ----------
        *covariates : pandas.Series or numpy.ndarray
            Covariates, grouped into margins by ``d``.
        k : int, tuple of int, or None
            Marginal basis dimensions, default five per covariate.
        bs : str or tuple of str
            Marginal bases, default ``cr``.
        m : int, tuple of int, or None
            One penalty order per margin.
        d : int, tuple of int, or None
            Covariates per margin, default one each.
        by : pandas.Series, numpy.ndarray, or None
            Numeric multiplier or factor producing one smooth per level.
        id : int, str, or None
            Tensor smooths sharing an ``id`` share their smoothing scales.
        full : bool
            mgcv's ``full`` option, penalizing null-space products separately.
        ord : int, tuple of int, or None
            Retained interaction orders.
        sp : float, tuple of float, or None
            mgcv's fixed smoothing parameters, checked and without effect, as
            in brms, which estimates the smoothing standard deviations.
        **unsupported : str or int or float or bool or None
            Rejected mgcv arguments.

        Returns
        -------
        numpy.ndarray
            Unpenalized columns, then penalized groups.

        Raises
        ------
        ValueError
            If an argument, basis, or covariate is invalid.
        """
        if unsupported:
            raise ValueError(
                f"t2() arguments {sorted(unsupported)!r} are not supported; "
                "supported arguments are k, bs, m, d, by, id, full, ord, and sp"
            )
        _check_sp(sp, "t2()")
        if len(covariates) < 1:
            raise ValueError("t2() requires covariates")
        if self.state is None:
            names = tuple(_name(column, i) for i, column in enumerate(covariates))
            if len(set(names)) != len(names):
                raise ValueError(
                    "repeated variables as arguments of a smooth are not permitted"
                )
            dims = self._dims(d, len(covariates))
            n_margins = len(dims)
            _check_k(k, "t2()")
            ks = self._per_margin(
                k, n_margins, "k", default=tuple(5**dim for dim in dims)
            )
            bss = self._per_margin(bs, n_margins, "bs", default=("cr",) * n_margins)
            ms = self._per_margin(m, n_margins, "m", default=(None,) * n_margins)
            if any(not isinstance(b, str) or b not in _MARGIN_BASES for b in bss):
                raise ValueError(f"t2() margins must use bases {sorted(_MARGIN_BASES)}")
            if any(not isinstance(v, int) or v < 3 for v in ks):
                raise ValueError(
                    "t2() marginal basis dimensions must be integers of at least 3"
                )
            if type(full) is not bool:
                raise ValueError("t2() full must be TRUE or FALSE")
            orders = () if ord is None else ((ord,) if isinstance(ord, int) else ord)
            if any(type(o) is not int or not 0 <= o <= n_margins for o in orders):
                raise ValueError(
                    "t2() ord must list interaction orders between 0 and the margin count"
                )
            margins = []
            start = 0
            for dim, basis, width, order in zip(dims, bss, ks, ms, strict=True):
                if dim > 1 and basis in {"cr", "cs", "ps"}:
                    basis = "tp"
                if (
                    (order is not None and type(order) is not int)
                    or not isinstance(basis, str)
                    or type(width) is not int
                ):
                    raise ValueError("t2() m must give one integer order per margin")
                # mgcv hands each margin the knots of its own covariates.
                margin_names = names[start : start + dim]
                margins.append(
                    _construct(
                        basis,
                        list(covariates[start : start + dim]),
                        width,
                        order,
                        _joint_knots(margin_names, basis)
                        if dim > 1
                        else smooth_knots(margin_names[0]),
                    )
                )
                start += dim
            raw = tensor_t2(margins, dims, full=full, orders=orders)
            self._fit(
                raw,
                label=f"t2({','.join(names)})",
                covariates=names,
                by=by,
                identity=_identity(id),
                spec=(
                    "t2",
                    repr(bss),
                    repr(ks),
                    repr(ms),
                    repr(dims),
                    str(full),
                    repr(orders),
                ),
            )
            return self._evaluate(raw.design, by)
        basis = self.state.basis
        if isinstance(basis, RandomEffectBasis):
            raise ValueError("t2() retained a random-effect basis")
        x = np.column_stack(
            [_numeric(column, "smooth covariate") for column in covariates]
        )
        return self._evaluate(basis.evaluate(x), by)

    @staticmethod
    def _dims(d: int | tuple[int, ...] | None, count: int) -> tuple[int, ...]:
        """Normalize mgcv's ``d`` grouping of covariates into margins.

        Parameters
        ----------
        d : int, tuple of int, or None
            Authored grouping.
        count : int
            Number of covariates.

        Returns
        -------
        tuple[int, ...]
            Covariates per margin.

        Raises
        ------
        ValueError
            If the grouping does not partition the covariates.
        """
        if d is None:
            return (1,) * count
        dims = (d,) if isinstance(d, int) else d
        if any(type(v) is not int or v < 1 for v in dims) or sum(dims) != count:
            raise ValueError(
                "t2() d must be positive integers summing to the covariate count"
            )
        return dims

    @staticmethod
    def _per_margin(
        value: int | str | tuple[int, ...] | tuple[str, ...] | None,
        count: int,
        role: str,
        *,
        default: tuple[int, ...] | tuple[str, ...] | tuple[None, ...],
    ) -> tuple[int | str | None, ...]:
        """Broadcast a per-margin argument.

        Parameters
        ----------
        value : int, str, tuple, or None
            Authored argument.
        count : int
            Number of margins.
        role : str
            Argument name used in errors.
        default : tuple
            Value when the argument is omitted.

        Returns
        -------
        tuple
            One value per margin.

        Raises
        ------
        ValueError
            If a tuple argument has the wrong length.
        """
        if value is None:
            return default
        if isinstance(value, tuple):
            if len(value) != count:
                raise ValueError(f"t2() {role} must have one entry per margin")
            return value
        return (value,) * count


class UnsupportedTensor:
    """Refuse ``te()`` and ``ti()``, which brms does not implement."""

    __stateful_transform__ = True

    def __call__(
        self, *covariates: ColumnInput, **options: str | int | float | bool | None
    ) -> npt.NDArray[np.float64]:
        """Raise the brms refusal naming ``t2``.

        Parameters
        ----------
        *covariates : pandas.Series or numpy.ndarray
            Ignored covariates.
        **options : str or int or float or bool or None
            Ignored options.

        Returns
        -------
        numpy.ndarray
            Never returns.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            "tensor product smooths 'te' and 'ti' are not implemented in brms; use 't2' instead"
        )


def term_constants(term: Term) -> frozenset[str]:
    """Return the names in a term's arguments that are not data columns.

    formulae lists every name inside a call among the term's variables. R's
    logical literals and the ``data2`` entries named by a smooth's ``xt``
    argument are resolved outside the data, so fresh rows need no column
    for them.

    Parameters
    ----------
    term : formulae.terms.terms.Term
        A term of a retained design.

    Returns
    -------
    frozenset[str]
        ``TRUE``, ``FALSE``, ``NA``, and the ``data2`` names the term read.
    """
    names = {"TRUE", "FALSE", "NA"}
    for component in getattr(term, "components", ()):
        transform = getattr(
            getattr(component, "call", None), "stateful_transform", None
        )
        if isinstance(transform, SmoothTransform) and transform.state is not None:
            names.update(transform.state.data2)
    return frozenset(names)


def smooth_states(design: DesignMatrices | None) -> dict[str, SmoothState]:
    """Return the retained state of every smooth term of an evaluated design.

    Parameters
    ----------
    design : formulae.matrices.DesignMatrices or None
        Retained design of one predictor, or None for a predictor without a
        formula design.

    Returns
    -------
    dict[str, SmoothState]
        Fitted ``s()`` and ``t2()`` states by term label, in term order.
    """
    common = getattr(design, "common", None)
    states: dict[str, SmoothState] = {}
    for label, term in getattr(common, "terms", {}).items():
        components = getattr(term, "components", ())
        if len(components) != 1:
            continue
        transform = getattr(
            getattr(components[0], "call", None), "stateful_transform", None
        )
        if isinstance(transform, SmoothTransform) and transform.state is not None:
            states[str(label)] = transform.state
    return states


def smooth_namespace() -> Mapping[str, type]:
    """Return the formula callables that construct smooth terms.

    Returns
    -------
    Mapping[str, type]
        ``s``, ``t2``, and the refused ``te`` and ``ti``.
    """
    return {
        "s": Smooth,
        "t2": TensorSmooth,
        "te": UnsupportedTensor,
        "ti": UnsupportedTensor,
    }


class BasisTerm(ValueModel):
    """One penalized basis block with independent Gaussian coordinates.

    The coefficients are ``scale * z`` for standard-normal ``z``, with one
    scale per coordinate of the output axis. Blocks sharing an mgcv ``id``
    share one scale.

    Parameters
    ----------
    label : str
        Authored term.
    name : str
        Prefixed design identifier.
    matrix : np.ndarray
        Rows by coordinates; read-only.
    scale : str
        Scale site.
    owns_scale : bool
        Whether this block draws the scale.
    prior : str
        Default prior of the scale.
    output : OutputAxis or None
        Axis of the coefficient vectors.
    """

    label: str
    name: str
    matrix: np.ndarray = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    scale: str
    owns_scale: bool
    prior: str
    output: OutputAxis | None = None

    @property
    def rowwise(self) -> bool:
        """Report whether the contribution varies over the rows.

        Returns
        -------
        bool
            True.
        """
        return True

    @property
    def deferred(self) -> bool:
        """Report whether the term is drawn after the auxiliary parameters.

        Returns
        -------
        bool
            False.
        """
        return False

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            The coordinate axis and, for coefficient vectors, the flat axis
            of the innovations.
        """
        axis, _ = design_declarations(self.name, self.matrix)
        if self.output is None:
            return (axis,)
        return (axis, self._innovations(self.output))

    def _innovations(self, output: OutputAxis) -> AxisDecl:
        """Declare the flat axis of the innovations of coefficient vectors.

        Parameters
        ----------
        output : OutputAxis
            Axis of the coefficient vectors.

        Returns
        -------
        AxisDecl
            One element per coordinate of the basis and of the output axis.
        """
        return AxisDecl(
            name=self.name + "_InnovationCoordinate",
            size=self.matrix.shape[1] * output.size,
            section="inline",
        )

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            The basis matrix.
        """
        return (design_declarations(self.name, self.matrix)[1],)

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            The scale, when this block draws it, and the innovations.
        """
        scale = SiteDecl(
            name=self.scale,
            class_="sds",
            section="design",
            default=self.prior,
            coef=self.label,
            axes=(self.output.name,) if self.output is not None else (),
            lower=0.0,
        )
        innovations = SiteDecl(
            name=self.name + "_z",
            class_="z",
            section="design",
            axes=(
                self.name + "_Coordinate"
                if self.output is None
                else self._innovations(self.output).name,
            ),
            overridable=False,
        )
        return (scale, innovations) if self.owns_scale else (innovations,)

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            None.
        """
        return None


__all__: list[str] = []
