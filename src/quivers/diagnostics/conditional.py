"""Bounded prediction grids and draw-wise conditional or empirical effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._numeric import real_samples
from quivers.diagnostics.posterior import PosteriorAnalysis, PosteriorSummary

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
else:
    # A frame adapter is optional; source grids and retained diagnostics need
    # only Torch. Runtime signature inspection follows the existing core rule.
    IntoDataFrame = object


type GridValue = str | int | float | bool | None
"""A scalar predictor coordinate, preserving category and integer carriers."""


class PredictionGrid(ValueModel):
    """Explicit predictor rows with a bounded Cartesian effect layout.

    Parameters
    ----------
    columns : tuple[str, ...]
        Predictor columns in generated-row order.
    rows : tuple[tuple[GridValue, ...], ...]
        Immutable scalar rows, ordered by effect point then conditioning row.
    axes : tuple[str, ...]
        Varied predictor names in Cartesian-product order.
    points : tuple[tuple[GridValue, ...], ...]
        The varied predictor coordinates at each effect point.
    num_conditions : int
        Explicit conditioning or empirical-reference row count per point.
    marginalize : bool
        Reduce reference rows within each joint posterior draw when true.
        False retains a separate effect for every conditioning row.
    weights : tuple[float, ...]
        Normalized empirical-reference weights; empty for conditional effects.

    Notes
    -----
    Construct with `prediction_grid`, which checks the allocation bound before
    generating the Cartesian product. It never substitutes means or reference
    categories implicitly. Prediction retains the fitted model's own checks.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[GridValue, ...], ...]
    axes: tuple[str, ...]
    points: tuple[tuple[GridValue, ...], ...]
    num_conditions: int
    marginalize: bool = False
    weights: tuple[float, ...] = ()

    @dx.model_validator()
    def _validate(self) -> None:
        """Check that rows and reduction metadata describe one complete grid.

        Raises
        ------
        ValueError
            If coordinates, row widths, weights, or layout counts disagree.
        """
        if (
            self.num_conditions < 1
            or not self.points
            or not self.axes
            or len(set(self.columns)) != len(self.columns)
            or len(set(self.axes)) != len(self.axes)
            or not set(self.axes) <= set(self.columns)
            or any(len(point) != len(self.axes) for point in self.points)
            or len(self.rows) != len(self.points) * self.num_conditions
            or any(len(row) != len(self.columns) for row in self.rows)
        ):
            raise ValueError(
                "prediction grid rows and coordinates have inconsistent layout"
            )
        if self.marginalize:
            if (
                len(self.weights) != self.num_conditions
                or any(not math.isfinite(w) or w < 0 for w in self.weights)
                or not math.isclose(sum(self.weights), 1.0)
            ):
                raise ValueError(
                    "empirical grid weights must be nonnegative and sum to one"
                )
        elif self.weights:
            raise ValueError("reference weights require empirical marginalization")

    def to_frame(self) -> IntoDataFrame:
        """Build an independent dataframe for the existing prediction interface.

        Returns
        -------
        IntoDataFrame
            Pandas rows in the checked effect-point/conditioning order.

        Notes
        -----
        This optional adapter requires pandas, as provided by the formula
        frontend's dependencies. Core source workflows use `to_mapping`.
        """
        import pandas as pd

        return pd.DataFrame.from_records(self.rows, columns=self.columns)

    def to_mapping(self) -> dict[str, tuple[GridValue, ...]]:
        """Expose predictor columns without a dataframe dependency.

        Returns
        -------
        dict[str, tuple[GridValue, ...]]
            Independent column tuples in the checked generated-row order.
        """
        return {
            name: tuple(row[index] for row in self.rows)
            for index, name in enumerate(self.columns)
        }

    def result(
        self,
        predictions: torch.Tensor,
        *,
        kind: Literal["mcmc", "variational", "prior"],
        method: Literal["epred", "linpred", "predict"] = "epred",
        labels: tuple[int | str, ...] = (),
    ) -> ConditionalEffects:
        """Retain joint predictions, reducing reference rows before summaries.

        Parameters
        ----------
        predictions : torch.Tensor
            One joint prediction request with chain, draw, row, and event axes.
        kind : {"mcmc", "variational", "prior"}
            Actual draw provenance; a variational layout axis is not a chain.
        method : {"epred", "linpred", "predict"}
            Expected responses, linear predictors, or sampled responses.
        labels : tuple[int or str, ...]
            Optional final response-component labels, in native output order.

        Returns
        -------
        ConditionalEffects
            Draws on point/condition/event axes, or point/event axes after
            empirical marginalization. Cross-point dependence is retained.

        Raises
        ------
        ValueError, TypeError
            If prediction rows or sample axes disagree with this grid, or a
            numeric empirical reduction contains invalid values.
        """
        if (
            not isinstance(predictions, torch.Tensor)
            or predictions.ndim < 3
            or not all(predictions.shape[:2])
            or predictions.shape[2] != len(self.rows)
        ):
            raise ValueError(
                "grid predictions require matching chain, draw, and row axes"
            )
        values = predictions.reshape(
            *predictions.shape[:2],
            len(self.points),
            self.num_conditions,
            *predictions.shape[3:],
        )
        if self.marginalize:
            values = real_samples(values, label="empirical conditional effects")
            weights = values.new_tensor(self.weights).reshape(
                1, 1, 1, self.num_conditions, *((1,) * (values.ndim - 4))
            )
            values = (values * weights).sum(3)
        return ConditionalEffects(
            grid=self, draws=values, kind=kind, method=method, labels=labels
        )


class ConditionalEffects(ValueModel):
    """Joint posterior predictions over an explicit conditioning experiment.

    Parameters
    ----------
    grid : PredictionGrid
        Point labels, explicit reference rows, and reduction policy.
    draws : torch.Tensor
        Chain/draw/point/condition/event axes, with the condition axis removed
        only for explicit empirical marginalization. Storage is held by identity.
    kind : {"mcmc", "variational", "prior"}
        Actual draw provenance, forwarded to existing posterior diagnostics.
    method : {"epred", "linpred", "predict"}
        Response expectations, linear predictors, or response replications.
    labels : tuple[int or str, ...]
        Optional native response labels; category probabilities retain all labels.
    """

    grid: PredictionGrid = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    draws: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    kind: Literal["mcmc", "variational", "prior"]
    method: Literal["epred", "linpred", "predict"] = "epred"
    labels: tuple[int | str, ...] = ()

    def analysis(self) -> PosteriorAnalysis:
        """Reuse joint uncertainty, covariance, and checked contrast diagnostics.

        Returns
        -------
        PosteriorAnalysis
            Existing diagnostics over the retained effect tensor, under the
            site name ``effect``. No model replay or resampling occurs.

        Raises
        ------
        ValueError, TypeError
            If draws cannot be represented as real empirical samples.
        """
        return PosteriorAnalysis(
            samples={
                "effect": real_samples(
                    self.draws, label="conditional effects", finite=False
                )
            },
            log_likelihood={},
            kind=self.kind,
        )

    def summary(
        self, *, probs: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> PosteriorSummary:
        """Summarize retained effects through the common posterior diagnostics.

        Parameters
        ----------
        probs : Sequence[float]
            Interior quantile probabilities, retaining their requested order.

        Returns
        -------
        PosteriorSummary
            Draw-wise uncertainty on point/condition/event coordinates.

        Raises
        ------
        ValueError, TypeError
            If sample axes, carriers, or requested probabilities are invalid.
        """
        return self.analysis().summary(probs=probs)["effect"]


def prediction_grid(
    axes: Mapping[str, Sequence[GridValue]],
    *,
    conditions: Mapping[str, Sequence[GridValue]] | IntoDataFrame,
    marginalize: bool = False,
    weights: Sequence[float] | None = None,
    max_rows: int = 10000,
) -> PredictionGrid:
    """Cross explicit effect coordinates with bounded conditioning rows.

    Parameters
    ----------
    axes : Mapping[str, Sequence[GridValue]]
        Nonempty varied predictor values; insertion order defines point order.
    conditions : Mapping[str, Sequence[GridValue]] or IntoDataFrame
        Explicit conditioning rows, or an empirical reference population.
        Varied columns are replaced at every effect point. Other columns,
        including groups, offsets, trial counts, and auxiliaries, are retained.
        A mapping needs only core dependencies; an empty mapping specifies
        one empty conditioning row. Dataframes use the optional data adapter
        while retaining their selected backend, without requiring pandas.
    marginalize : bool
        Average reference-row predictions within each posterior draw. False
        conditions separately on each supplied row and never averages them.
    weights : Sequence[float] or None
        Optional finite nonnegative reference-population weights, normalized
        only for empirical averaging. These are not likelihood weights.
    max_rows : int
        Positive upper bound on points times reference rows, checked before
        Cartesian allocation. Larger requests require an explicit larger budget.

    Returns
    -------
    PredictionGrid
        Immutable scalar rows and their exact reduction/layout contract.

    Raises
    ------
    ValueError, TypeError
        If names, values, weights, dimensions, or allocation budgets are invalid.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if not isinstance(marginalize, bool):
        raise TypeError("marginalize must be a bool")
    if not axes or any(not isinstance(name, str) or not name for name in axes):
        raise ValueError("grid axes require nonempty predictor names")
    if any(isinstance(values, (str, bytes)) for values in axes.values()):
        raise TypeError("each grid axis requires a sequence of scalar values")
    sizes = tuple(len(values) for values in axes.values())
    if any(size < 1 for size in sizes):
        raise ValueError("each grid axis requires at least one value")
    if isinstance(conditions, Mapping):
        if any(isinstance(values, (str, bytes)) for values in conditions.values()):
            raise TypeError("conditioning columns require sequences of scalar values")
        condition_columns = tuple(conditions)
        counts = {len(values) for values in conditions.values()}
        if len(counts) > 1:
            raise ValueError("conditioning columns must have equal row counts")
        num_conditions = next(iter(counts), 1)
        frame = None
    else:
        import narwhals as nw

        frame = nw.from_native(conditions, eager_only=True)
        condition_columns = tuple(frame.columns)
        num_conditions = len(frame)
    if num_conditions < 1 or len(set(condition_columns)) != len(condition_columns):
        raise ValueError("conditions require nonempty rows and unique column names")
    count = math.prod(sizes) * num_conditions
    if count > max_rows:
        raise ValueError(
            f"prediction grid requires {count} rows, exceeding max_rows={max_rows}"
        )

    def scalar(value: object, *, allow_missing: bool = False) -> GridValue:
        """Retain a checked scalar coordinate without inventing an encoding.

        Parameters
        ----------
        value : object
            Native scalar from an explicit axis or reference frame.
        allow_missing : bool
            Retain a missing reference coordinate as None; the fitted input
            contract decides whether it can denote a new grouping identity.

        Returns
        -------
        GridValue
            Python scalar, preserving integer and Boolean carriers.

        Raises
        ------
        TypeError, ValueError
            If the coordinate is structured, unsupported, or nonfinite.
        """
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            numeric = float(value)
            if allow_missing and math.isnan(numeric):
                return None
            if math.isfinite(numeric):
                return numeric
        raise ValueError(
            "grid coordinates must be finite scalar numbers, strings, bools, or None"
        )

    names = tuple(axes)
    columns = condition_columns + tuple(
        name for name in names if name not in condition_columns
    )
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("condition columns require nonempty string names")
    normalized = ()
    if weights is not None and not marginalize:
        raise ValueError("reference weights require empirical marginalization")
    if marginalize:
        supplied = tuple(weights) if weights is not None else (1.0,) * num_conditions
        if (
            len(supplied) != num_conditions
            or any(
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not math.isfinite(w)
                or w < 0
                for w in supplied
            )
            or sum(supplied) <= 0
        ):
            raise ValueError(
                "reference weights require one finite nonnegative value per row and positive total"
            )
        scale = max(supplied)
        total = sum(w / scale for w in supplied)
        normalized = tuple((w / scale) / total for w in supplied)
    points = tuple(
        product(*(tuple(scalar(v) for v in values) for values in axes.values()))
    )
    if isinstance(conditions, Mapping):
        raw_rows = zip(*(conditions[name] for name in condition_columns), strict=True)
    else:
        assert frame is not None
        import narwhals as nw

        # Backend-native null masks avoid importing pandas or numpy to
        # recognize extension scalars such as pd.NA.
        raw_rows = (
            tuple(
                None if missing else value
                for value, missing in zip(row, mask, strict=True)
            )
            for row, mask in zip(
                frame.rows(), frame.select(nw.all().is_null()).rows(), strict=True
            )
        )
    reference = (
        tuple(
            tuple(
                None if name in axes else scalar(v, allow_missing=True)
                for name, v in zip(condition_columns, row, strict=True)
            )
            for row in raw_rows
        )
        if condition_columns
        else ((),) * num_conditions
    )
    rows = []
    for point in points:
        replacements = dict(zip(names, point, strict=True))
        for row in reference:
            base = dict(zip(condition_columns, row, strict=True))
            base.update(replacements)
            rows.append(tuple(base[name] for name in columns))
    return PredictionGrid(
        columns=columns,
        rows=tuple(rows),
        axes=names,
        points=points,
        num_conditions=num_conditions,
        marginalize=marginalize,
        weights=normalized,
    )


__all__ = ["GridValue", "PredictionGrid", "ConditionalEffects", "prediction_grid"]
