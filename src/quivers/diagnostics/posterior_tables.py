"""brms-style summary tables over retained draws.

The functions here reproduce the table layouts of brms's
``posterior_summary``, ``posterior_interval``, ``predictive_interval``,
and ``posterior_table`` [@burkner2017brms]. Every input keeps its chain
and draw axes, ``(chain, draw, ...)``, and every function pools them, as
brms pools the draws of all chains. Missing values (NaN) are removed per
coordinate before summarizing, as R's ``na.rm = TRUE`` does; infinite
draws are retained. Quantiles use linear interpolation between order
statistics (R's type 7), and the robust dispersion is R's ``mad``, the
median absolute deviation scaled by the constant 1.4826.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from itertools import product

import didactic.api as dx
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._numeric import probabilities, real_samples

MAD_CONSTANT = 1.4826
"""The consistency constant of R's ``mad``, which scales the median absolute
deviation to the standard deviation of a normal sample."""

type RowLabel = str | int
"""A table row identity: a variable name or an original observation label."""


def _percent(value: float) -> str:
    """Format a percentage the way R's ``as.character`` prints it.

    Parameters
    ----------
    value : float
        Percentage value, such as 2.5 or 97.5.

    Returns
    -------
    str
        At most 15 significant digits, without trailing zeros.
    """
    return format(value, ".15g")


class SummaryTable(ValueModel):
    """A labeled summary matrix or three-way array of retained draws.

    Two-way tables have shape ``(row, column)``; three-way tables, which
    brms returns for vector-valued predictions and per-level coefficients,
    have shape ``(row, column, component)``.

    Parameters
    ----------
    rows : tuple[str or int, ...]
        Row labels: variable names or observation identities.
    columns : tuple[str, ...]
        Statistic labels, for instance ``("Estimate", "Est.Error", "Q2.5",
        "Q97.5")``.
    components : tuple[str or int, ...]
        Labels of the third axis, empty for a two-way table.
    values : torch.Tensor
        Float64 statistics in row, column, and component order.
    """

    rows: tuple[RowLabel, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    columns: tuple[str, ...]
    components: tuple[RowLabel, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    values: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    @dx.model_validator()
    def _validate(self) -> None:
        """Check that labels describe the value tensor.

        Raises
        ------
        ValueError
            If the value shape disagrees with the row, column, and component
            labels.
        """
        expected = (len(self.rows), len(self.columns)) + (
            (len(self.components),) if self.components else ()
        )
        if tuple(self.values.shape) != expected:
            raise ValueError(
                f"summary values have shape {tuple(self.values.shape)}; labels "
                f"describe {expected}"
            )

    def value(
        self, row: RowLabel, column: str, component: RowLabel | None = None
    ) -> float:
        """Read one statistic by its labels.

        Parameters
        ----------
        row : str or int
            Row label.
        column : str
            Statistic label.
        component : str or int or None
            Component label of a three-way table; None for a two-way table.

        Returns
        -------
        float
            The selected statistic.

        Raises
        ------
        KeyError
            If a label is unknown, or a component is required or not allowed.
        """
        if (component is None) != (not self.components):
            raise KeyError("select a component exactly for a three-way table")
        try:
            index = (self.rows.index(row), self.columns.index(column))
            if component is not None:
                index += (self.components.index(component),)
        except ValueError as error:
            raise KeyError(f"unknown summary label: {error}") from error
        return float(self.values[index].item())

    def to_frame(self, component: RowLabel | None = None) -> pd.DataFrame:
        """Return a two-way table, or one component slice, as a DataFrame.

        Parameters
        ----------
        component : str or int or None
            Component of a three-way table; None for a two-way table.

        Returns
        -------
        pandas.DataFrame
            Rows indexed by `rows`, columns named by `columns`.

        Raises
        ------
        KeyError
            If a component is required, not allowed, or unknown.
        """
        if (component is None) != (not self.components):
            raise KeyError("select a component exactly for a three-way table")
        values = self.values
        if component is not None:
            if component not in self.components:
                raise KeyError(f"unknown component {component!r}")
            values = values[..., self.components.index(component)]
        return pd.DataFrame(
            values.cpu().numpy(), index=list(self.rows), columns=list(self.columns)
        )


def _pooled(values: torch.Tensor, label: str) -> torch.Tensor:
    """Pool chain and draw axes of real retained draws.

    Parameters
    ----------
    values : torch.Tensor
        Draws with shape ``(chain, draw, ...)``.
    label : str
        Diagnostic name for errors.

    Returns
    -------
    torch.Tensor
        Float64 values with one leading sample axis.

    Raises
    ------
    ValueError
        If the chain or draw axis is missing or empty.
    TypeError
        If the draws are not a real tensor.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{label} requires a tensor of draws")
    if values.ndim < 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{label} requires nonempty chain and draw axes")
    return real_samples(values, label=label, finite=False).flatten(0, 1)


def _statistics(
    pooled: torch.Tensor, probs: tuple[float, ...], robust: bool
) -> torch.Tensor:
    """Compute brms's summary statistics per coordinate, removing NaN draws.

    Parameters
    ----------
    pooled : torch.Tensor
        Float64 draws with a leading sample axis.
    probs : tuple[float, ...]
        Quantile probabilities.
    robust : bool
        Use the median and R's ``mad`` instead of the mean and sample SD.

    Returns
    -------
    torch.Tensor
        Statistics stacked on a new second-to-front axis: estimate, error,
        then one quantile per probability, shape ``(2 + len(probs), ...)``.
    """
    present = ~torch.isnan(pooled)
    count = present.sum(0)
    if robust:
        estimate = torch.nanquantile(pooled, 0.5, dim=0)
        error = MAD_CONSTANT * torch.nanquantile((pooled - estimate).abs(), 0.5, dim=0)
    else:
        estimate = torch.nanmean(pooled, dim=0)
        squares = torch.where(present, (pooled - estimate) ** 2, 0.0).sum(0)
        error = torch.where(
            count > 1, (squares / (count - 1).clamp(min=1)).sqrt(), torch.nan
        )
    quantiles = [torch.nanquantile(pooled, p, dim=0) for p in probs]
    return torch.stack([estimate, error, *quantiles])


def _flatten_named(
    draws: Mapping[str, torch.Tensor], label: str
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Flatten named draws to brms's one-column-per-coordinate matrix.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Variables with shape ``(chain, draw, *event)``.
    label : str
        Diagnostic name for errors.

    Returns
    -------
    tuple[tuple[str, ...], torch.Tensor]
        Column names, with 1-based ``name[i,j]`` indices for event
        coordinates as in Stan and brms, and pooled draws ``(sample, column)``.

    Raises
    ------
    ValueError
        If no variable is given or sample axes disagree.
    """
    if not draws:
        raise ValueError(f"{label} requires at least one variable")
    names: list[str] = []
    columns: list[torch.Tensor] = []
    shape: tuple[int, int] | None = None
    for name, values in draws.items():
        pooled = _pooled(values, label)
        if shape is None:
            shape = (values.shape[0], values.shape[1])
        elif (values.shape[0], values.shape[1]) != shape:
            raise ValueError(f"{label} variables have different chain/draw axes")
        event = tuple(values.shape[2:])
        if not event:
            names.append(name)
        else:
            names.extend(
                f"{name}[{','.join(str(i + 1) for i in index)}]"
                for index in product(*(range(size) for size in event))
            )
        columns.append(pooled.reshape(pooled.shape[0], -1))
    return tuple(names), torch.cat(columns, dim=1)


def _matrix(
    draws: Mapping[str, torch.Tensor] | torch.Tensor,
    labels: Sequence[RowLabel] | None,
    label: str,
) -> tuple[tuple[RowLabel, ...], torch.Tensor]:
    """Normalize named or positional draws to rows and pooled draws.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor] or torch.Tensor
        Named variables, or a tensor ``(chain, draw, row[, component])``.
    labels : Sequence[str or int] or None
        Row labels of a positional tensor; None uses 0-based positions.
    label : str
        Diagnostic name for errors.

    Returns
    -------
    tuple[tuple[str or int, ...], torch.Tensor]
        Row labels and pooled draws ``(sample, row[, component])``.

    Raises
    ------
    ValueError
        If labels are given for named draws, or do not match the rows.
    """
    if isinstance(draws, Mapping):
        if labels is not None:
            raise ValueError("row labels apply only to a positional draw tensor")
        return _flatten_named(draws, label)
    pooled = _pooled(draws, label)
    if pooled.ndim not in (2, 3):
        raise ValueError(
            f"{label} requires a (chain, draw, row) or (chain, draw, row, "
            "component) tensor"
        )
    rows = tuple(range(pooled.shape[1])) if labels is None else tuple(labels)
    if len(rows) != pooled.shape[1] or len(set(rows)) != len(rows):
        raise ValueError(f"{label} requires one distinct label per row")
    return rows, pooled


def posterior_summary(
    draws: Mapping[str, torch.Tensor] | torch.Tensor,
    *,
    probs: Sequence[float] = (0.025, 0.975),
    robust: bool = False,
    labels: Sequence[RowLabel] | None = None,
    components: Sequence[RowLabel] | None = None,
) -> SummaryTable:
    """Summarize draws with brms's ``posterior_summary`` layout.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor] or torch.Tensor
        Named variables ``(chain, draw, *event)``, or positional draws
        ``(chain, draw, row)`` or ``(chain, draw, row, component)``.
    probs : Sequence[float]
        Quantile probabilities strictly between zero and one.
    robust : bool
        Report the median and R's ``mad`` instead of the mean and SD.
    labels : Sequence[str or int] or None
        Row labels of a positional tensor; 0-based positions by default.
    components : Sequence[str or int] or None
        Labels of a positional tensor's component axis; 0-based by default.

    Returns
    -------
    SummaryTable
        Columns ``Estimate``, ``Est.Error`` and ``Q<100 p>``. A four-axis
        tensor gives brms's ``N x E x C`` array with a component axis.

    Raises
    ------
    ValueError
        If probabilities, labels, or draw axes are invalid.
    TypeError
        If draws are not real tensors.

    Notes
    -----
    Cost is O(S M log S) for S pooled draws and M summarized coordinates.
    """
    levels = probabilities(probs)
    if type(robust) is not bool:
        raise TypeError("robust must be Boolean")
    rows, pooled = _matrix(draws, labels, "posterior_summary")
    statistics = _statistics(pooled, levels, robust).transpose(0, 1)
    columns = ("Estimate", "Est.Error", *(f"Q{_percent(100 * p)}" for p in levels))
    if pooled.ndim == 3:
        names = (
            tuple(range(pooled.shape[2])) if components is None else tuple(components)
        )
        if len(names) != pooled.shape[2]:
            raise ValueError("posterior_summary requires one label per component")
        return SummaryTable(
            rows=rows, columns=columns, components=names, values=statistics
        )
    if components is not None:
        raise ValueError("component labels require a component axis")
    return SummaryTable(rows=rows, columns=columns, values=statistics)


def _interval(
    draws: Mapping[str, torch.Tensor] | torch.Tensor,
    prob: float,
    labels: Sequence[RowLabel] | None,
    label: str,
) -> SummaryTable:
    """Compute central intervals with rstantools's column labels.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor] or torch.Tensor
        Named or positional draws.
    prob : float
        Central probability mass in the open unit interval.
    labels : Sequence[str or int] or None
        Row labels of a positional tensor.
    label : str
        Diagnostic name for errors.

    Returns
    -------
    SummaryTable
        Lower and upper bounds labeled ``<100 a>%`` and ``<100 (1-a)>%``.

    Raises
    ------
    ValueError
        If the probability is outside the open unit interval or the draws
        have a component axis.
    """
    if isinstance(prob, bool) or not math.isfinite(prob) or not 0 < prob < 1:
        raise ValueError(f"{label} requires a probability strictly inside (0, 1)")
    rows, pooled = _matrix(draws, labels, label)
    if pooled.ndim != 2:
        raise ValueError(f"{label} requires one value per row and draw")
    alpha = (1 - prob) / 2
    levels = (alpha, 1 - alpha)
    values = torch.stack([torch.nanquantile(pooled, p, dim=0) for p in levels], dim=1)
    return SummaryTable(
        rows=rows,
        columns=tuple(f"{_percent(100 * p)}%" for p in levels),
        values=values,
    )


def posterior_interval(
    draws: Mapping[str, torch.Tensor] | torch.Tensor,
    *,
    prob: float = 0.95,
    labels: Sequence[RowLabel] | None = None,
) -> SummaryTable:
    """Compute central posterior intervals, as brms ``posterior_interval``.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor] or torch.Tensor
        Named variables ``(chain, draw, *event)`` or positional draws
        ``(chain, draw, row)``.
    prob : float
        Central probability mass, 0.95 by default as in brms.
    labels : Sequence[str or int] or None
        Row labels of a positional tensor.

    Returns
    -------
    SummaryTable
        Columns ``2.5%`` and ``97.5%`` for the default mass.
    """
    return _interval(draws, prob, labels, "posterior_interval")


def predictive_interval(
    draws: torch.Tensor,
    *,
    prob: float = 0.9,
    labels: Sequence[RowLabel] | None = None,
) -> SummaryTable:
    """Compute central predictive intervals, as brms ``predictive_interval``.

    Parameters
    ----------
    draws : torch.Tensor
        Predictive draws ``(chain, draw, row)``.
    prob : float
        Central probability mass, 0.9 by default as in brms.
    labels : Sequence[str or int] or None
        Row labels; 0-based positions by default.

    Returns
    -------
    SummaryTable
        Columns ``5%`` and ``95%`` for the default mass.
    """
    return _interval(draws, prob, labels, "predictive_interval")


def posterior_table(
    draws: torch.Tensor,
    *,
    levels: Sequence[float | int | str] | None = None,
    labels: Sequence[RowLabel] | None = None,
) -> SummaryTable:
    """Tabulate relative frequencies of discrete draws, as brms ``posterior_table``.

    Parameters
    ----------
    draws : torch.Tensor
        Discrete draws ``(chain, draw, row)``, such as ordinal predictions.
    levels : Sequence or None
        Possible values in column order; None uses every distinct non-missing
        value in ascending order. A string sequence labels the columns of
        integer codes ``0..K-1``.
    labels : Sequence[str or int] or None
        Row labels; 0-based positions by default.

    Returns
    -------
    SummaryTable
        Columns ``P(Y = <level>)`` holding each row's frequencies among its
        non-missing draws.

    Raises
    ------
    ValueError
        If draws contain a value outside the given levels.
    """
    rows, pooled = _matrix(draws, labels, "posterior_table")
    if pooled.ndim != 2:
        raise ValueError("posterior_table requires one value per row and draw")
    present = ~torch.isnan(pooled)
    if levels is None:
        values = torch.unique(pooled[present]).tolist()
        names = [
            _percent(v) if not float(v).is_integer() else str(int(v)) for v in values
        ]
    else:
        given = tuple(levels)
        if all(isinstance(level, str) for level in given):
            values = [float(code) for code in range(len(given))]
            names = [str(level) for level in given]
        else:
            values = [float(level) for level in given]
            names = [
                _percent(float(v)) if not float(v).is_integer() else str(int(v))
                for v in given
            ]
    counts = torch.stack([(pooled == value).sum(0) for value in values], dim=1)
    total = present.sum(0)
    if bool((counts.sum(1) != total).any()):
        raise ValueError("posterior_table draws contain values outside the levels")
    return SummaryTable(
        rows=rows,
        columns=tuple(f"P(Y = {name})" for name in names),
        values=counts.to(torch.float64) / total.unsqueeze(1),
    )


__all__ = [
    "MAD_CONSTANT",
    "SummaryTable",
    "posterior_interval",
    "posterior_summary",
    "posterior_table",
    "predictive_interval",
]
