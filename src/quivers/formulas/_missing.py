"""Typed row plans for latent responses and known noisy measurements.

These plans retain observed values and latent coordinates separately. Emission
uses the existing tensor concatenation and gather operations; no data column is
filled with an estimated mean or an independently sampled replacement.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import didactic.api as dx
import narwhals as nw
import numpy as np
import pandas as pd

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas.formula import FixedColumn, RowSelection, _group_coding, _qvr_name

__all__: list[str] = []


class LatentResponseSpec(ValueModel):
    """A latent response axis and its actual recorded measurement events.

    Parameters
    ----------
    response : str
        Original modeled response column, also used by latent predictor references.
    rows : RowSelection
        Included original row identities, before latent-coordinate compaction.
    observed : FixedColumn
        Actual finite measurements, never placeholder or completed values.
    observed_positions : tuple[int, ...]
        Measurement positions on the compact truth axis.
    observed_rows : tuple[int, ...]
        Original row identity of each recorded measurement event.
    latent_positions : tuple[int, ...]
        Truth positions requiring continuous latent draws. Without measurement
        error these are exactly the missing values; with error every truth is latent.
    full_to_combined : tuple[int, ...]
        Gather from observed-then-latent coordinates without error, or directly
        from all latent truths with measurement error, into included row order.
    standard_error : FixedColumn or None
        Strictly positive finite known errors for actual measurement events.
        Infinite error denotes no recorded event and is omitted from this column.
    levels : tuple[str, ...]
        Existing canonical group labels for a shared noisy truth, empty for rowwise truths.
    group : str or None
        Group identifying repeated measurements of the same supplied noisy value.
    """

    response: str
    rows: RowSelection = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    observed: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    observed_positions: tuple[int, ...]
    observed_rows: tuple[int, ...]
    latent_positions: tuple[int, ...]
    full_to_combined: tuple[int, ...]
    standard_error: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    levels: tuple[str, ...] = ()
    group: str | None = None

    @dx.model_validator()
    def _validate(self) -> None:
        """Require exact observation, truth, and original-row coordinate agreement.

        Raises
        ------
        ValueError
            If supplied coordinates, measurements, errors, or group identities
            cannot describe the declared latent response law.
        """
        _validate_rows(self.rows)
        if not self.response or (self.group is None and self.levels):
            raise ValueError(
                "latent responses require a name and explicit group identity"
            )
        if self.group is not None and (
            not self.group
            or not self.levels
            or any(not isinstance(level, str) or not level for level in self.levels)
            or len(set(self.levels)) != len(self.levels)
        ):
            raise ValueError("shared truths require distinct canonical group levels")
        size = self.truth_size
        for positions in (self.observed_positions, self.latent_positions):
            if len(set(positions)) != len(positions) or any(
                type(position) is not int or not 0 <= position < size
                for position in positions
            ):
                raise ValueError(
                    "latent response coordinates must be distinct valid truth positions"
                )
        observed = self.observed.data
        if (
            observed.shape != (len(self.observed_positions),)
            or not np.isfinite(observed).all()
        ):
            raise ValueError(
                "observed events must contain only actual finite measurements"
            )
        if len(self.full_to_combined) != len(self.rows.positions):
            raise ValueError("completed values must cover the exact included row axis")
        if self.standard_error is None:
            if self.group is not None or set(
                self.observed_positions + self.latent_positions
            ) != set(range(size)):
                raise ValueError(
                    "missing-only truths partition observed and latent coordinates"
                )
            order = self.observed_positions + self.latent_positions
            expected = tuple(order.index(position) for position in range(size))
            representatives = tuple(range(size))
        else:
            errors = self.standard_error.data
            if (
                errors.shape != observed.shape
                or not np.isfinite(errors).all()
                or np.any(errors <= 0)
            ):
                raise ValueError(
                    "recorded measurement errors must be finite and positive"
                )
            if self.latent_positions != tuple(range(size)):
                raise ValueError("every noisy truth must retain its latent coordinate")
            expected = self.full_to_combined
            if any(
                type(position) is not int or not 0 <= position < size
                for position in expected
            ) or set(expected) != set(range(size)):
                raise ValueError("completed noisy values must map every declared truth")
            if self.group is None and expected != tuple(range(size)):
                raise ValueError("rowwise noisy truths retain their declared row order")
            representatives = tuple(
                expected.index(position) for position in range(size)
            )
        if self.full_to_combined != expected or self.observed_rows != tuple(
            self.rows.positions[representatives[position]]
            for position in self.observed_positions
        ):
            raise ValueError(
                "measurement units must retain their original row identities"
            )

    @property
    def truth_size(self) -> int:
        """Number of modeled truths, independent of repeated consumer rows.

        Returns
        -------
        int
            Number of group identities or original rowwise truths.
        """
        return len(self.levels) if self.group is not None else len(self.rows.positions)


class LatentInputSpec(ValueModel):
    """A nominal response reference and an explicit row gather into its values.

    Parameters
    ----------
    response : str
        Modeled latent response whose completed values the predictor consumes.
    positions : tuple[int, ...]
        Positions in that response's included row order, not its latent-only axis.
    source_size : int
        Number of completed source rows, retaining explicit gather bounds.
    rows : RowSelection
        Consumer's included original rows.
    index : str or None
        Authored lookup column, where an explicit cross-row mapping is used.
    """

    response: str
    positions: tuple[int, ...]
    source_size: int
    rows: RowSelection = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    index: str | None = None

    @dx.model_validator()
    def _validate(self) -> None:
        """Retain an exact gather into the explicitly named source row domain.

        Raises
        ------
        ValueError
            If the response, source extent, consumer rows, or gather is invalid.
        """
        _validate_rows(self.rows)
        if (
            not self.response
            or type(self.source_size) is not int
            or self.source_size < 0
        ):
            raise ValueError("latent inputs require a named source and its row extent")
        if len(self.positions) != len(self.rows.positions) or any(
            type(position) is not int or not 0 <= position < self.source_size
            for position in self.positions
        ):
            raise ValueError(
                "latent input gather must retain each consumer row within the source domain"
            )
        if self.index is not None and not self.index:
            raise ValueError("latent input index must name its explicit lookup column")


def _validate_rows(rows: RowSelection) -> None:
    """Require distinct included positions within one original row scope.

    Parameters
    ----------
    rows : RowSelection
        Original cardinality and retained row positions.

    Raises
    ------
    ValueError
        If the scope or its selected positions violate the shared row contract.
    """
    if (
        type(rows.size) is not int
        or rows.size < 0
        or len(set(rows.positions)) != len(rows.positions)
        or any(
            type(position) is not int or not 0 <= position < rows.size
            for position in rows.positions
        )
    ):
        raise ValueError("latent rows must name distinct valid original positions")


def _numeric(values: object, *, label: str, allow_missing: bool) -> np.ndarray:
    """Validate original continuous values before choosing their floating carrier.

    Parameters
    ----------
    values : object
        Scalar or one-dimensional numeric inputs, including explicit missing values.
    label : str
        Role used by diagnostics.
    allow_missing : bool
        Permit None, pandas.NA, and floating NaN as missing values.

    Returns
    -------
    numpy.ndarray
        Float64 values; missing entries remain NaN until the observation plan is built.

    Raises
    ------
    ValueError
        If values are Boolean, complex, nonnumeric, incorrectly ranked, or missing
        where a complete numeric input is required.
    """
    array = np.asarray(values, dtype=object)
    if array.ndim > 1:
        raise ValueError(f"{label} must be scalar or one-dimensional")
    flat = array.reshape(-1)
    result = np.empty(flat.size, dtype=np.float64)
    for position, value in enumerate(flat):
        missing = (
            value is None
            or value is pd.NA
            or (isinstance(value, Real) and value != value)
        )
        if missing and allow_missing:
            result[position] = np.nan
        elif (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or missing
        ):
            raise ValueError(
                f"{label} requires real numeric values, excluding Boolean values"
            )
        else:
            try:
                result[position] = value
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{label} requires representable real numeric values"
                ) from error
    return result


def _column(response: str, role: str, values: np.ndarray) -> FixedColumn:
    """Reuse immutable checked columns for one observed measurement role.

    Parameters
    ----------
    response : str
        Original response name.
    role : str
        Observed values or known measurement errors.
    values : numpy.ndarray
        Actual event values on their compact measurement axis.

    Returns
    -------
    FixedColumn
        Existing immutable column contract with a deterministic source binding.
    """
    name = f"mi_{_qvr_name(response)}_{role}"
    return FixedColumn(term=response, name=name, qvr_name=name, data=values)


def bind_latent_response(
    response: str,
    values: object,
    *,
    standard_error: object | None = None,
    rows: RowSelection | None = None,
    group: str | None = None,
    frame: pd.DataFrame | None = None,
) -> LatentResponseSpec:
    """Separate actual observations from the coordinates needing latent draws.

    Parameters
    ----------
    response : str
        Modeled response column.
    values : object
        Continuous responses with optional missing entries.
    standard_error : object or None
        Known strictly positive measurement SD, scalar or one per row. Positive
        infinity denotes no observation. Missing SD is allowed only on missing responses.
    rows : RowSelection or None
        Original row identities. None uses the complete supplied row order.
    group : str or None
        Shared noisy truth identity, using existing canonical group coding.
        Repeated group values and errors must agree, as in released me().
    frame : pandas.DataFrame or None
        Included rows containing the grouping factor when group is supplied.

    Returns
    -------
    LatentResponseSpec
        Exact observation and latent coordinate plan with no filled data values.

    Raises
    ------
    ValueError
        If rows, observations, errors, or repeated identities violate their contracts.
    """
    if not isinstance(response, str) or not response:
        raise ValueError("a latent response requires its modeled column name")
    data = _numeric(values, label=response, allow_missing=True)
    if np.isinf(data).any():
        raise ValueError("latent response values must be finite or explicitly missing")
    size = len(data)
    rows = (
        RowSelection(size=size, positions=tuple(range(size))) if rows is None else rows
    )
    _validate_rows(rows)
    if len(rows.positions) != size:
        raise ValueError("latent response rows must align with included values")
    noise = None
    if standard_error is not None:
        noise = _numeric(standard_error, label="measurement error", allow_missing=True)
        if len(noise) == 1:
            noise = np.broadcast_to(noise, (size,)).copy()
        if len(noise) != size:
            raise ValueError("measurement errors must match included response rows")
        if (
            np.any(noise <= 0)
            or np.any(np.isneginf(noise))
            or np.any(np.isnan(noise) & ~np.isnan(data))
        ):
            raise ValueError(
                "known measurement errors must be positive; only unmeasured rows may have missing errors"
            )
    levels: tuple[str, ...] = ()
    codes = tuple(range(size))
    representatives = tuple(range(size))
    if group is not None:
        if noise is None or frame is None or len(frame) != size:
            raise ValueError(
                "shared noisy truths require known errors and an aligned grouping frame"
            )
        levels, codes = _group_coding(nw.from_native(frame, eager_only=True), group)
        representatives = tuple(codes.index(level) for level in range(len(levels)))
        for row, code in enumerate(codes):
            first = representatives[code]
            same_value = data[row] == data[first] or (
                np.isnan(data[row]) and np.isnan(data[first])
            )
            same_noise = noise[row] == noise[first] or (
                np.isnan(noise[row]) and np.isnan(noise[first])
            )
            if not same_value or not same_noise:
                raise ValueError(
                    "repeated noisy-truth identities must have identical measured values and errors"
                )
    truth_data = data[list(representatives)]
    truth_noise = None if noise is None else noise[list(representatives)]
    measured = ~np.isnan(truth_data)
    if truth_noise is not None:
        measured &= np.isfinite(truth_noise)
    observed_positions = tuple(int(position) for position in np.flatnonzero(measured))
    latent_positions = (
        tuple(range(len(truth_data)))
        if noise is not None
        else tuple(int(position) for position in np.flatnonzero(~measured))
    )
    if noise is None:
        concatenated = observed_positions + latent_positions
        gather = tuple(concatenated.index(code) for code in codes)
    else:
        gather = codes
    return LatentResponseSpec(
        response=response,
        rows=rows,
        observed=_column(response, "observed", truth_data[measured]),
        observed_positions=observed_positions,
        observed_rows=tuple(
            rows.positions[representatives[position]] for position in observed_positions
        ),
        latent_positions=latent_positions,
        full_to_combined=gather,
        standard_error=None
        if truth_noise is None
        else _column(response, "sd", truth_noise[measured]),
        levels=levels,
        group=group,
    )


def bind_latent_input(
    source: LatentResponseSpec,
    rows: RowSelection,
    *,
    source_index: Sequence[str | int] | None = None,
    target_index: Sequence[str | int] | None = None,
    index: str | None = None,
) -> LatentInputSpec:
    """Join a latent predictor by original row identity or an explicit index map.

    Parameters
    ----------
    source : LatentResponseSpec
        Nominal modeled response, never selected by equal extent.
    rows : RowSelection
        Included consumer rows in the same original data scope.
    source_index, target_index : Sequence[str or int] or None
        Explicit unique source IDs and one target ID per consumer row.
    index : str or None
        Authored target index column, required for explicit index joins.

    Returns
    -------
    LatentInputSpec
        Checked gather into the completed source response values.

    Raises
    ------
    ValueError
        If original row identities do not align or explicit IDs are absent,
        ambiguous, or unknown.
    """
    _validate_rows(rows)
    lookup: dict[str | int, int]
    targets: Sequence[str | int]
    if source_index is None and target_index is None and index is None:
        if rows.size != source.rows.size:
            raise ValueError(
                "latent predictor rows require the same original scope or explicit indices"
            )
        lookup = {
            identity: position
            for position, identity in enumerate(source.rows.positions)
        }
        targets = rows.positions
    else:
        if source_index is None or target_index is None or not index:
            raise ValueError(
                "a latent predictor index join requires both index vectors and its column name"
            )
        if len(source_index) != len(source.rows.positions) or len(target_index) != len(
            rows.positions
        ):
            raise ValueError(
                "latent predictor index vectors must align with their declared rows"
            )
        identities = (*source_index, *target_index)
        if any(type(value) not in {int, str} for value in identities):
            raise ValueError(
                "latent predictor index identities must be exact strings or integers"
            )
        if len(set(source_index)) != len(source_index):
            raise ValueError("latent response index identities must be unique")
        lookup = {identity: position for position, identity in enumerate(source_index)}
        targets = target_index
    if any(identity not in lookup for identity in targets):
        raise ValueError("latent predictor index names no modeled source response row")
    return LatentInputSpec(
        response=source.response,
        positions=tuple(lookup[identity] for identity in targets),
        source_size=len(source.rows.positions),
        rows=rows,
        index=index,
    )
