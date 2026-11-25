"""Rows a fitted predictor is evaluated on.

A `RowSource` describes the rows of one request: the training rows, or the rows
of a fresh data frame. It owns everything that depends on the rows alone: the
retained formula evaluators, the compact levels of the grouping factors, the
selected group-level coefficients, and whether structured fields are included.
Each term reads it to return itself on those rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from formulae.categorical import CategoricalBox
from formulae.matrices import DesignMatrices
from formulae.terms.call import Call
from formulae.terms.variable import Variable
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from quivers.formulas._smooth import term_constants
from quivers.formulas.formula import FixedColumn, _CS_ONES

if TYPE_CHECKING:
    from quivers.formulas._group_prediction import GroupRemap
    from quivers.formulas._grouping import Membership

__all__: list[str] = []


def validate_categories(
    design: DesignMatrices,
    frame: pd.DataFrame,
    selected: frozenset[tuple[str, str]] | None = None,
) -> None:
    """Check required variables and categorical levels without global settings.

    Parameters
    ----------
    design : DesignMatrices
        Fitted expression and contrast state.
    frame : pandas.DataFrame
        Fresh predictor rows.
    selected : frozenset[tuple[str, str]] or None
        Active grouping-factor and expanded-slope pairs. None validates all
        fitted slopes; grouping identities are checked separately.

    Raises
    ------
    ValueError
        If predictors are absent, missing, or contain unknown categories.
    TypeError
        If a retained categorical evaluator is unsupported.
    """
    terms = []
    if design.common is not None:
        terms.extend(design.common.terms.values())
    if design.group is not None:
        for term in design.group.terms.values():
            if selected is None or any(
                group == term.factor.name
                and (
                    slope == term.expr.name.replace(_CS_ONES, "1")
                    or slope.startswith(term.expr.name.replace(_CS_ONES, "1") + "_")
                )
                for group, slope in selected
            ):
                terms.append(term.expr)
    seen: set[int] = set()
    for term in terms:
        # formulae's CallVarsExtractor represents literal arguments by "".
        missing = (set(term.var_names) - {""} - term_constants(term)) - set(
            frame.columns
        )
        if missing:
            raise ValueError(f"new_data is missing predictors {sorted(missing)}")
        for component in getattr(term, "components", ()):
            if id(component) in seen or component.kind != "categoric":
                continue
            seen.add(id(component))
            if isinstance(component, Variable):
                values = frame[component.name]
            elif isinstance(component, Call):
                values = component.call.eval(frame, component.env)
                if isinstance(values, CategoricalBox):
                    values = values.data
            else:
                raise TypeError("unsupported retained categorical evaluator")
            values = np.asarray(values)
            if pd.isna(values).any():
                raise ValueError(
                    "new_data categorical predictors cannot contain missing values"
                )
            levels = component.levels
            assert levels is not None
            unknown = set(values.reshape(-1)) - set(levels)
            if unknown:
                raise ValueError(
                    f"new_data contains unknown levels {sorted(map(str, unknown))} "
                    f"for {component.name!r}; only fitted levels are supported"
                )


def numeric_matrix(values: object, rows: int, name: str) -> np.ndarray:
    """Validate and normalize one evaluated predictor term on its row axis.

    Parameters
    ----------
    values : object
        Evaluated numeric term.
    rows : int
        Required row count.
    name : str
        Term name for diagnostics.

    Returns
    -------
    numpy.ndarray
        Finite numeric values with row and column axes.

    Raises
    ------
    ValueError
        If the term has an invalid shape or nonfinite values.
    """
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise ValueError(f"new_data term {name!r} must retain exactly {rows} rows")
    if matrix.dtype.kind not in "biuf" or not np.isfinite(matrix).all():
        raise ValueError(f"new_data term {name!r} must produce finite numeric values")
    return matrix


class PredictorRows:
    """The rows of one request, read through one predictor's evaluators."""

    def __init__(self, source: RowSource, design: DesignMatrices | None) -> None:
        """Read one predictor's evaluators.

        Parameters
        ----------
        source : RowSource
            Rows of the request.
        design : DesignMatrices or None
            Retained evaluators; None on the training rows.
        """
        self.source = source
        self._design = design
        self._common: np.ndarray | None = None

    @property
    def frame(self) -> pd.DataFrame | None:
        """Return the fresh rows.

        Returns
        -------
        pandas.DataFrame or None
            The frame, or None on the training rows.
        """
        return self.source.frame

    def selects(self, group: str, slope: str) -> bool:
        """Report whether a group-level coefficient is selected.

        Parameters
        ----------
        group, slope : str
            Grouping factor and coefficient label.

        Returns
        -------
        bool
            Whether the request includes the coefficient.
        """
        selected = self.source.selected
        return selected is None or (group, slope) in selected

    def matrix(self, term: str) -> np.ndarray:
        """Evaluate one population-level term.

        Parameters
        ----------
        term : str
            Term name.

        Returns
        -------
        numpy.ndarray
            Finite values by row and design column.

        Raises
        ------
        ValueError
            If the predictor has no such term, or the fresh design has
            another width.
        """
        design, frame = self._design, self.frame
        if design is None or design.common is None or frame is None:
            raise ValueError(f"term {term!r} has no retained evaluator")
        if self._common is None:
            evaluated = design.common.evaluate_new_data(frame)
            assert evaluated.design_matrix is not None
            self._common = np.asarray(evaluated.design_matrix)
        columns = design.common.slices[term]
        matrix = numeric_matrix(self._common[:, columns], len(frame), term)
        if matrix.shape[1] != columns.stop - columns.start:
            raise ValueError(f"new_data changed the design width of {term!r}")
        return matrix

    def components(self, term: str) -> Sequence[Variable | Call]:
        """Return the retained components of one population-level term.

        Parameters
        ----------
        term : str
            Term name.

        Returns
        -------
        Sequence[Variable or Call]
            Evaluators of the term's factors.
        """
        design = self._design
        assert design is not None and design.common is not None
        return design.common.terms[term].components

    def slopes(self, term: str) -> np.ndarray:
        """Evaluate one group-level slope term.

        Parameters
        ----------
        term : str
            Name of the slope expression.

        Returns
        -------
        numpy.ndarray
            Finite values by row and design column.

        Raises
        ------
        ValueError
            If the predictor has no such slope.
        """
        design, frame = self._design, self.frame
        if design is not None and design.group is not None and frame is not None:
            for item in design.group.terms.values():
                if item.expr.name == term:
                    return numeric_matrix(
                        item.expr.eval_new_data(frame), len(frame), term
                    )
        raise ValueError(f"slope {term!r} has no retained evaluator")

    def member_codes(self, membership: Membership) -> tuple[tuple[int, ...], ...]:
        """Code the members of a multi-membership factor.

        Parameters
        ----------
        membership : Membership
            Fitted factor.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            One compact code per row for each member.

        Raises
        ------
        ValueError
            If the request has no levels for the factor.
        """
        groups = self.source.groups
        if groups is None or membership.group not in groups:
            raise ValueError(f"the rows have no levels of {membership.group!r}")
        return tuple(
            tuple(int(code) for code in indices.tolist())
            for indices in groups[membership.group].member_indices
        )

    def column(self, term: str, position: int, width: int) -> npt.NDArray[np.float64]:
        """Read one column of a population-level term.

        Parameters
        ----------
        term : str
            Term name.
        position, width : int
            Position of the column among the term's columns.

        Returns
        -------
        numpy.ndarray
            The column.

        Raises
        ------
        ValueError
            If the fresh design has another width.
        """
        matrix = self.matrix(term)
        if position >= matrix.shape[1] or matrix.shape[1] < width:
            raise ValueError(f"new_data changed the design width of {term!r}")
        return matrix[:, position]


class RowSource:
    """The rows of one request and how the fitted terms read them."""

    def __init__(
        self,
        *,
        frame: pd.DataFrame | None = None,
        designs: Mapping[str, DesignMatrices] | None = None,
        groups: Mapping[str, GroupRemap] | None = None,
        selected: frozenset[tuple[str, str]] | None = None,
        include: bool = True,
        namespace: Callable[[str], str] | None = None,
    ) -> None:
        """Describe the rows of one request.

        Parameters
        ----------
        frame : pandas.DataFrame or None
            Fresh rows; None for the training rows.
        designs : Mapping[str, DesignMatrices] or None
            Retained evaluators by predictor namespace.
        groups : Mapping[str, GroupRemap] or None
            Compact levels of each grouping factor on fresh rows.
        selected : frozenset[tuple[str, str]] or None
            Group-level coefficients to include; None includes all.
        include : bool
            Whether structured fields contribute (brms ``incl_autocor``).
        namespace : Callable[[str], str] or None
            Namespace of a predictor in ``designs``, for a program whose
            predictors have names of their own.
        """
        self.frame = frame
        self.groups = groups
        self.selected = selected
        self.include = include
        self._designs = designs or {}
        self._namespace = namespace
        self._read: dict[str, PredictorRows] = {}

    @property
    def size(self) -> int | None:
        """Return the number of fresh rows.

        Returns
        -------
        int or None
            The count, or None on the training rows.
        """
        return None if self.frame is None else len(self.frame)

    def predictor(self, prefix: str) -> PredictorRows:
        """Read the rows through one predictor's evaluators.

        Parameters
        ----------
        prefix : str
            Predictor namespace.

        Returns
        -------
        PredictorRows
            The rows, with the evaluators of the predictor.
        """
        key = self._namespace(prefix) if self._namespace is not None else prefix
        if key not in self._read:
            design = self._designs.get(key)
            if self.frame is not None and design is not None:
                validate_categories(design, self.frame, self.selected)
            self._read[key] = PredictorRows(self, design)
        return self._read[key]

    def regrouped(
        self,
        groups: Mapping[str, GroupRemap],
        selected: frozenset[tuple[str, str]] | None,
    ) -> RowSource:
        """Read the same rows on other compact levels.

        Parameters
        ----------
        groups : Mapping[str, GroupRemap]
            Compact levels of each grouping factor.
        selected : frozenset[tuple[str, str]] or None
            Group-level coefficients to include.

        Returns
        -------
        RowSource
            The rows on the levels.
        """
        return RowSource(
            frame=self.frame,
            designs=self._designs,
            groups=groups,
            selected=selected,
            include=self.include,
            namespace=self._namespace,
        )

    def within(self, namespace: Callable[[str], str]) -> RowSource:
        """Read the same rows from a program nested in a namespace.

        Parameters
        ----------
        namespace : Callable[[str], str]
            Name of a nested predictor's namespace in this source.

        Returns
        -------
        RowSource
            The rows under the nested names.
        """
        outer = self._namespace
        return RowSource(
            frame=self.frame,
            designs=self._designs,
            groups=self.groups,
            selected=self.selected,
            include=self.include,
            namespace=namespace
            if outer is None
            else lambda prefix: outer(namespace(prefix)),
        )


def finite_inputs(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Check that every input of fresh rows is finite.

    Parameters
    ----------
    values : Mapping[str, torch.Tensor]
        Inputs by name.

    Returns
    -------
    dict[str, torch.Tensor]
        The inputs.

    Raises
    ------
    ValueError
        If a floating-point input holds a nonfinite value.
    """
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in values.values()
    ):
        raise ValueError(
            "new_data predictor values must fit finite native floating-point inputs"
        )
    return dict(values)


def fresh_columns(
    columns: Sequence[FixedColumn], read: Callable[[str, int, int], np.ndarray]
) -> tuple[FixedColumn, ...]:
    """Evaluate exploded design columns on fresh rows.

    Parameters
    ----------
    columns : Sequence[FixedColumn]
        Fitted columns, in design order.
    read : Callable[[str, int, int], numpy.ndarray]
        Reads one column from its term, its position among the term's
        columns, and their number.

    Returns
    -------
    tuple[FixedColumn, ...]
        The columns with their fresh values.
    """
    widths: dict[str, int] = {}
    positions: list[int] = []
    for column in columns:
        positions.append(widths.get(column.term, 0))
        widths[column.term] = positions[-1] + 1
    return tuple(
        column.with_(data=read(column.term, position, widths[column.term]))
        for column, position in zip(columns, positions, strict=True)
    )
