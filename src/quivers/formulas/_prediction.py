"""Evaluate new formula rows using the fitted encodings and native program."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal, Never, SupportsIndex

import didactic.api as dx
from formulae.categorical import CategoricalBox
from formulae.matrices import DesignMatrices
from formulae.terms.call import Call
from formulae.terms.variable import Variable
import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
    Module,
    ObjectDecl,
    TypeFromExpr,
)
from quivers.formulas.formula import FixedColumn, Formula, _qvr_name
from quivers.inference.registry import LatentRegistry

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


class FormulaPredictionPlan(ValueModel):
    """Fit-owned evaluator state with an explicit nonportable boundary.

    Parameters
    ----------
    module : Module
        Training program, including its priors and latent dimensions.
    designs : Mapping[str, DesignMatrices]
        Evaluated location and auxiliary formulas, retaining transform state.
    version : str
        Runtime-plan schema identifier. This plan contains formulae objects
        and is deliberately not a portable serialized model representation.
    """

    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    designs: Mapping[str, DesignMatrices] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    version: Literal["qvr-formula-prediction/v1"] = "qvr-formula-prediction/v1"

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        """Refuse to serialize mutable, environment-dependent evaluator state.

        Parameters
        ----------
        protocol : SupportsIndex
            Requested pickle protocol.

        Raises
        ------
        TypeError
            Runtime prediction plans are not a durable serialization format.
        """
        raise TypeError(
            f"{self.version} contains live formulae evaluators and cannot be serialized; "
            "retain the fitted object in this process for new_data prediction"
        )


class PredictionContext(ValueModel):
    """Prepared row bindings shared by every draw of one prediction request.

    Parameters
    ----------
    program : MonadicProgram
        Native training program or a row-specialized copy.
    observations : Mapping[str, torch.Tensor]
        Validated predictor bindings and internal response placeholders.
    rows : int
        Number of prediction rows.
    """

    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    rows: int


def _validate_categories(design: DesignMatrices, frame: pd.DataFrame) -> None:
    """Check required variables and categorical levels without global settings.

    Parameters
    ----------
    design : DesignMatrices
        Fitted expression and contrast state.
    frame : pandas.DataFrame
        Fresh predictor rows.

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
            terms.extend((term.expr, term.factor))
    seen: set[int] = set()
    for term in terms:
        # formulae's CallVarsExtractor represents literal arguments by "".
        missing = (set(term.var_names) - {""}) - set(frame.columns)
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


def _numeric_matrix(values: object, rows: int, name: str) -> np.ndarray:
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


def _column_values(
    columns: tuple[FixedColumn, ...], matrices: Mapping[str, np.ndarray]
) -> Iterator[tuple[FixedColumn, np.ndarray]]:
    """Match fitted exploded-column labels to fresh term-matrix positions.

    Parameters
    ----------
    columns : tuple[FixedColumn, ...]
        Fitted labels in design-column order.
    matrices : Mapping[str, numpy.ndarray]
        Fresh matrices keyed by the original term name.

    Yields
    ------
    tuple[FixedColumn, numpy.ndarray]
        Each fitted label and its fresh numeric values.

    Raises
    ------
    ValueError
        If an evaluated term changes its fitted design width.
    """
    positions: dict[str, int] = {}
    for column in columns:
        index = positions.get(column.term, 0)
        positions[column.term] = index + 1
        matrix = matrices[column.term]
        if index >= matrix.shape[1]:
            raise ValueError(f"new_data changed the design width of {column.term!r}")
        yield column, matrix[:, index]
    for name, count in positions.items():
        if matrices[name].shape[1] != count:
            raise ValueError(f"new_data changed the design width of {name!r}")


def _predictor_bindings(
    formula: Formula, design: DesignMatrices, frame: pd.DataFrame, prefix: str
) -> dict[str, torch.Tensor]:
    """Evaluate retained fixed columns, offsets, and random slopes once.

    Parameters
    ----------
    formula : Formula
        Fitted column labels and group coding.
    design : DesignMatrices
        Fitted evaluator state.
    frame : pandas.DataFrame
        Fresh predictor rows.
    prefix : str
        Auxiliary-parameter input namespace, or the empty location prefix.

    Returns
    -------
    dict[str, torch.Tensor]
        Validated native covariate and group-index bindings.

    Raises
    ------
    ValueError
        If evaluated columns or grouping levels are invalid.
    """
    rows = len(frame)
    _validate_categories(design, frame)
    matrices: dict[str, np.ndarray] = {}
    if design.common is not None:
        evaluated = design.common.evaluate_new_data(frame)
        assert evaluated.design_matrix is not None
        for name, slice_ in design.common.slices.items():
            matrices[name] = _numeric_matrix(
                evaluated.design_matrix[:, slice_], rows, name
            )
    result = {}
    for column, values in _column_values(formula.fixed_columns, matrices):
        if not column.is_intercept:
            result[prefix + column.qvr_name] = torch.as_tensor(
                values.copy(), dtype=torch.float32
            )
    for index, (column, values) in enumerate(_column_values(formula.offsets, matrices)):
        result[f"{prefix}formula_offset_{index}"] = torch.as_tensor(
            values.copy(), dtype=torch.float32
        )

    if design.group is not None:
        slope_matrices = {}
        for term in design.group.terms.values():
            if term.expr.name not in slope_matrices:
                slope_matrices[term.expr.name] = _numeric_matrix(
                    term.expr.eval_new_data(frame), rows, term.expr.name
                )
        # A slope shared by several grouping factors has one evaluated input.
        columns = tuple(
            {
                term.column.name: term.column
                for term in formula.random_terms
                if term.column is not None
            }.values()
        )
        for column, values in _column_values(columns, slope_matrices):
            if not column.is_intercept:
                result[f"{prefix}random_{column.qvr_name}"] = torch.as_tensor(
                    values.copy(), dtype=torch.float32
                )

    for group, levels in formula.group_levels.items():
        mapping = {level: index for index, level in enumerate(levels)}
        values = frame[group]
        if np.asarray(pd.isna(values)).any():
            raise ValueError(
                f"new_data grouping factor {group!r} contains missing values"
            )
        unknown = set(map(str, values)) - mapping.keys()
        if unknown:
            raise ValueError(
                f"new_data contains unknown groups {sorted(unknown)} for {group!r}; "
                "only fitted levels are supported"
            )
        result[f"{_qvr_name(group)}_idx"] = torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in result.values()
    ):
        raise ValueError(
            "new_data predictor values must fit finite native floating-point inputs"
        )
    return result


@torch.random.fork_rng(devices=[])
def prepare_prediction(
    fit: BayesianFit,
    new_data: IntoDataFrame | None,
    predictor_data: torch.Tensor | None,
) -> PredictionContext:
    """Prepare and validate prediction rows before sampling posterior draws.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula, program, evaluator plan, and posterior.
    new_data : IntoDataFrame or None
        Fresh predictor rows, or None for the retained training rows.
    predictor_data : torch.Tensor or None
        Explicit neural predictor input for fresh rows.

    Returns
    -------
    PredictionContext
        One native program and validated row bindings for the entire request.

    Raises
    ------
    ValueError
        If rows, predictors, levels, offsets, trials, or neural inputs are invalid.
    TypeError
        If the compiled prediction program changes its latent layout.
    """
    if new_data is None:
        if predictor_data is not None:
            raise ValueError("predictor_data requires new_data")
        return PredictionContext(
            program=fit.program,
            observations=fit.observations,
            rows=len(fit.formula.response_values),
        )
    plan = fit.prediction_plan
    if plan is None:
        raise ValueError("this fit has no retained new_data prediction plan")
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    if len(frame) < 1 or not frame.columns.is_unique:
        raise ValueError("new_data requires nonempty rows and unique column names")
    if fit.predictor is not None and predictor_data is None:
        raise ValueError("new_data for a neural fit requires explicit predictor_data")
    if fit.predictor is None and predictor_data is not None:
        raise ValueError("predictor_data requires a fitted neural predictor")

    observations: dict[str, torch.Tensor] = {}
    for parameter, formula in (("mu", fit.formula), *fit.distributional.items()):
        bindings = _predictor_bindings(
            formula,
            plan.designs[parameter],
            frame,
            "" if parameter == "mu" else f"dpar_{parameter}_",
        )
        for name, value in bindings.items():
            if name in observations and not torch.equal(value, observations[name]):
                raise ValueError(f"new_data has inconsistent shared input {name!r}")
            observations[name] = value

    from quivers.formulas._fit import _binomial_trial_tensor, _predictor_output

    rows = len(frame)
    response = _qvr_name(fit.formula.response_name)
    # Responses are internal replay placeholders, never read from new_data.
    # Reusing a fitted support point avoids family-specific prediction algebra.
    observations[response] = torch.full(
        (rows,), float(fit.formula.response_values[0]), dtype=torch.float32
    )
    if fit.formula.response_weights is not None:
        observations["formula_weights"] = torch.ones(rows, dtype=torch.float64)
    if fit.family.name in {"binomial", "beta_binomial"}:
        trials = _binomial_trial_tensor(frame, fit.binomial_trials, n_obs=rows)
        if isinstance(fit.binomial_trials, str):
            observations[_qvr_name(fit.binomial_trials)] = trials
        observations[response] = torch.zeros(rows)
    if fit.predictor is not None:
        assert predictor_data is not None and fit.predictor_name is not None
        categories = (
            len(fit.category_labels) if fit.family.name == "categorical" else None
        )
        modes = tuple((module, module.training) for module in fit.predictor.modules())
        try:
            fit.predictor.eval()
            with torch.no_grad():
                neural = _predictor_output(
                    fit.predictor, predictor_data, n_obs=rows, n_categories=categories
                ).detach()
        finally:
            for module, training in modes:
                module.training = training
        if not torch.isfinite(neural).all():
            raise ValueError("new_data neural predictor must produce finite values")
        observations[fit.predictor_name] = neural

    statements = []
    for statement in plan.module.statements:
        if isinstance(statement, ObjectDecl) and statement.names == ("Resp",):
            statement = statement.with_(
                init=TypeFromExpr(
                    expr=DiscreteConstructor(constructor="FinSet", args=(str(rows),))
                )
            )
        statements.append(statement)
    program = (
        Compiler(plan.module.with_(statements=tuple(statements))).compile().morphism
    )
    if not isinstance(program, MonadicProgram):
        raise TypeError("prediction requires a native monadic program")
    original = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    specialized = LatentRegistry.from_model(program, observed_names=set(observations))
    if original.names != specialized.names or any(
        original[name].constrained_shape != specialized[name].constrained_shape
        for name in original.names
    ):
        raise TypeError("new_data specialization changed the fitted latent layout")
    return PredictionContext(program=program, observations=observations, rows=rows)
