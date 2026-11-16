"""Evaluate new formula rows using the fitted encodings and native program."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Mapping, Never, SupportsIndex

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
from quivers.formulas.formula import FixedColumn, Formula, _CS_ONES, _qvr_name
from quivers.inference.registry import LatentRegistry
from quivers.formulas._group_prediction import (
    GroupRemap,
    NewLevelPolicy,
    compact_groups,
    selected_effects,
    select_contributions,
    specialized_formula,
)

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
    groups : dict[str, GroupRemap]
        Compact grouping axes and fitted raw-site mappings.
    new_level_policy : NewLevelPolicy
        Joint donor or conditional Gaussian policy for new identities.
    seed : int
        Request seed for the separate group-effect random stream.
    retain_ordinal_cutpoints : bool
        Replay fitted, design-constrained thresholds before applying prediction
        row bindings or varying-effect exclusions.
    ordinal_order_groups : tuple of str
        Groups whose fresh effects or threshold profiles can affect order.
    """

    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    rows: int
    groups: dict[str, GroupRemap] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    new_level_policy: NewLevelPolicy = "uncertainty"
    seed: int = 0
    retain_ordinal_cutpoints: bool = False
    ordinal_order_groups: tuple[str, ...] = ()


def _validate_categories(
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
    formula: Formula,
    design: DesignMatrices,
    frame: pd.DataFrame,
    prefix: str,
    groups: Mapping[str, GroupRemap] | None = None,
    selected: frozenset[tuple[str, str]] | None = None,
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
    groups : Mapping[str, GroupRemap] or None
        Compact group indices, or None to require original fitted levels.
    selected : frozenset[tuple[str, str]] or None
        Active grouping-factor and slope pairs; excluded slope inputs use zero.

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
    _validate_categories(design, frame, selected)
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
                active = selected is None or any(
                    group == term.factor.name
                    and (
                        slope == term.expr.name.replace(_CS_ONES, "1")
                        or slope.startswith(term.expr.name.replace(_CS_ONES, "1") + "_")
                    )
                    for group, slope in selected
                )
                if active:
                    slope_matrices[term.expr.name] = _numeric_matrix(
                        term.expr.eval_new_data(frame), rows, term.expr.name
                    )
                else:
                    width = len(
                        {
                            term_.column.name
                            for term_ in formula.random_terms
                            if term_.column is not None
                            and term_.column.term == term.expr.name
                        }
                    )
                    slope_matrices[term.expr.name] = np.zeros((rows, max(1, width)))
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
        if groups is not None:
            result[f"{_qvr_name(group)}_idx"] = groups[group].row_indices
            continue
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


def _set_trial_rows(
    fit: BayesianFit,
    observations: dict[str, torch.Tensor],
    counts: torch.Tensor,
    *,
    specification: int | str | None = None,
) -> None:
    """Bind checked row trials and a matching support point for native replay.

    Parameters
    ----------
    fit : BayesianFit
        Fitted response metadata.
    observations : dict[str, torch.Tensor]
        Request-local bindings to update.
    counts : torch.Tensor
        Exact integer trial counts per requested row.
    specification : int or str or None
        Authored constant or column override.

    Raises
    ------
    ValueError
        If a literal scalar trial count cannot be overridden at runtime.
    """
    response = _qvr_name(fit.formula.response_name)
    event = fit.formula.response_layout.event_shape
    if event:
        observations["formula_trials"] = counts
        point = torch.zeros(
            (len(counts), *event), dtype=torch.long, device=counts.device
        )
        point[:, 0] = counts
        observations[response] = point
    else:
        name = (
            fit.binomial_trials
            if isinstance(fit.binomial_trials, str)
            else specification
        )
        if isinstance(name, str):
            observations[_qvr_name(name)] = counts
        elif specification is not None and specification != fit.binomial_trials:
            raise ValueError(
                "constant scalar trial overrides require fitting with a trial column"
            )
        observations[response] = torch.zeros(len(counts), device=counts.device)


@torch.random.fork_rng(devices=[])
def prepare_prediction(
    fit: BayesianFit,
    new_data: IntoDataFrame | None,
    predictor_data: torch.Tensor | None,
    *,
    re_formula: str | None = None,
    allow_new_levels: bool = False,
    sample_new_levels: NewLevelPolicy = "uncertainty",
    seed: int = 0,
    trials: int | str | None = None,
    require_trials: bool = True,
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
    re_formula : str or None
        One-sided varying-effect selector; None includes all and ``~0`` none.
    allow_new_levels : bool
        Permit unseen and missing grouping identities.
    sample_new_levels : NewLevelPolicy
        Joint donor policy or conditional Gaussian law for new identities.
    seed : int
        Seed for an independent group-effect random stream.

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
    if not isinstance(allow_new_levels, bool):
        raise TypeError("allow_new_levels must be a bool")
    if sample_new_levels not in {
        "uncertainty",
        "old_levels",
        "gaussian",
        "feasible_uncertainty",
        "feasible_gaussian",
    }:
        raise ValueError(
            "sample_new_levels must be uncertainty, old_levels, gaussian, "
            "feasible_uncertainty, or feasible_gaussian"
        )
    formulas = {"mu": fit.formula, **fit.distributional}
    constrained_cs = (
        fit.family.ordinal is not None
        and fit.family.ordinal.kind in {"cumulative", "hurdle_cumulative"}
        and (
            any(column.category_specific for column in fit.formula.fixed_columns)
            or any(
                term.column is not None and term.column.category_specific
                for term in fit.formula.random_terms
            )
        )
    )
    selected = selected_effects(formulas, re_formula)
    retained_observations = fit.observations
    if trials is not None:
        if fit.family.trials_parameter is None:
            raise ValueError(
                "trial overrides require a count family with a trial parameter"
            )
        if new_data is None:
            from quivers.formulas._fit import _binomial_trial_tensor

            source = (
                fit.prediction_plan.designs["mu"].data
                if fit.prediction_plan is not None
                else None
            )
            if isinstance(trials, str) and source is None:
                raise ValueError(
                    "a trial column requires a retained training dataframe"
                )
            counts = _binomial_trial_tensor(
                source, trials, n_obs=len(fit.formula.response_values)
            )
            retained_observations = dict(fit.observations)
            _set_trial_rows(fit, retained_observations, counts, specification=trials)
    if new_data is None and re_formula is None:
        if predictor_data is not None:
            raise ValueError("predictor_data requires new_data")
        return PredictionContext(
            program=fit.program,
            observations=retained_observations,
            rows=len(fit.formula.response_values),
        )
    plan = fit.prediction_plan
    if new_data is None:
        if predictor_data is not None:
            raise ValueError("predictor_data requires new_data")
        if plan is None:
            raise ValueError("this fit has no retained formula prediction plan")
        module = select_contributions(plan.module, formulas, selected)
        program = Compiler(module).compile().morphism
        if not isinstance(program, MonadicProgram):
            raise TypeError("prediction requires a native monadic program")
        return PredictionContext(
            program=program,
            observations=retained_observations,
            rows=len(fit.formula.response_values),
            seed=seed,
            retain_ordinal_cutpoints=constrained_cs,
        )
    if plan is None:
        raise ValueError("this fit has no retained new_data prediction plan")
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    needs_cs_intercept = any(
        _CS_ONES in column.term
        for formula in formulas.values()
        for column in (
            *formula.fixed_columns,
            *(term.column for term in formula.random_terms if term.column is not None),
        )
    )
    if needs_cs_intercept:
        if _CS_ONES in frame.columns:
            raise ValueError(
                f"{_CS_ONES!r} is reserved for category-specific intercepts"
            )
        frame = frame.copy()
        frame[_CS_ONES] = 1.0
    if len(frame) < 1 or not frame.columns.is_unique:
        raise ValueError("new_data requires nonempty rows and unique column names")
    if fit.predictor is not None and predictor_data is None:
        raise ValueError("new_data for a neural fit requires explicit predictor_data")
    if fit.predictor is None and predictor_data is not None:
        raise ValueError("predictor_data requires a fitted neural predictor")

    original = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    groups = compact_groups(fit, frame, selected, allow_new_levels, original)
    order_groups = {
        term.group
        for term in fit.formula.random_terms
        if term.column is not None and term.column.category_specific
    }
    if fit.thresholds_by is not None:
        order_groups.add(fit.thresholds_by)
    if (
        constrained_cs
        and any(
            group in order_groups and -1 in mapping.fitted_indices
            for group, mapping in groups.items()
        )
        and not sample_new_levels.startswith("feasible_")
    ):
        raise ValueError(
            "new cumulative category-specific group effects require an explicit "
            "sample_new_levels='feasible_uncertainty' or 'feasible_gaussian' "
            "policy, conditioned on ordered thresholds for the requested rows"
        )
    observations: dict[str, torch.Tensor] = {}
    for parameter, formula in (("mu", fit.formula), *fit.distributional.items()):
        bindings = _predictor_bindings(
            formula,
            plan.designs[parameter],
            frame,
            "" if parameter == "mu" else f"dpar_{parameter}_",
            groups,
            selected,
        )
        for name, value in bindings.items():
            if name in observations and not torch.equal(value, observations[name]):
                raise ValueError(f"new_data has inconsistent shared input {name!r}")
            observations[name] = value

    from quivers.formulas._fit import _binomial_trial_tensor, _predictor_output

    rows = len(frame)
    # Responses are internal replay placeholders, never read from new_data.
    # Reusing a fitted support point avoids family-specific prediction algebra.
    point = torch.as_tensor(
        fit.formula.response_values[0].copy()
        if fit.formula.response_layout.event_shape
        else float(fit.formula.response_values[0]),
        dtype=torch.float64
        if fit.formula.response_layout.event_shape
        else torch.float32,
    )
    from quivers.formulas._response_modifiers import response_input_name

    observations[response_input_name(fit.formula)] = point.expand(
        rows, *fit.formula.response_layout.event_shape
    )
    if fit.formula.modifiers.weights is not None:
        observations["formula_weights"] = torch.ones(rows, dtype=torch.float64)
    if fit.formula.modifiers.decision is not None:
        observations["formula_decision"] = torch.ones(rows, dtype=torch.bool)
    if fit.family.trials_parameter is not None:
        specification = trials if trials is not None else fit.binomial_trials
        if specification is None:
            if require_trials:
                raise ValueError(
                    "new_data vector count predictions require trials=<constant or column name>"
                )
            specification = 0
        counts = _binomial_trial_tensor(frame, specification, n_obs=rows)
        _set_trial_rows(fit, observations, counts, specification=specification)
    if fit.predictor is not None:
        assert predictor_data is not None and fit.predictor_name is not None
        categories = (
            len(fit.category_labels) if fit.family.response is not None else None
        )
        reference = (
            fit.family.response.reference_index(
                tuple(str(label) for label in fit.category_labels)
            )
            if fit.family.response is not None
            else 0
        )
        modes = tuple((module, module.training) for module in fit.predictor.modules())
        try:
            fit.predictor.eval()
            with torch.no_grad():
                neural = _predictor_output(
                    fit.predictor,
                    predictor_data,
                    n_obs=rows,
                    n_categories=categories,
                    reference=reference,
                ).detach()
        finally:
            for module, training in modes:
                module.training = training
        if not torch.isfinite(neural).all():
            raise ValueError("new_data neural predictor must produce finite values")
        observations[fit.predictor_name] = neural

    module = plan.module
    if groups or re_formula is not None:
        specialized = {
            name: specialized_formula(formula, groups)
            for name, formula in formulas.items()
        }
        module, _ = fit._formula_lens(
            distributional={
                name: formula for name, formula in specialized.items() if name != "mu"
            }
        ).forward(specialized["mu"])
        module = select_contributions(module, formulas, selected)
    statements = []
    for statement in module.statements:
        if isinstance(statement, ObjectDecl) and statement.names == ("Resp",):
            statement = statement.with_(
                init=TypeFromExpr(
                    expr=DiscreteConstructor(constructor="FinSet", args=(str(rows),))
                )
            )
        statements.append(statement)
    program = Compiler(module.with_(statements=tuple(statements))).compile().morphism
    if not isinstance(program, MonadicProgram):
        raise TypeError("prediction requires a native monadic program")
    specialized_registry = LatentRegistry.from_model(
        program, observed_names=set(observations)
    )
    group_sites = {
        name: (group, width)
        for group in groups.values()
        for name, width in group.sites.items()
    }
    if original.names != specialized_registry.names:
        raise TypeError("new_data specialization changed the fitted latent names")
    for name in original.names:
        original_shape = original[name].constrained_shape
        fresh_shape = specialized_registry[name].constrained_shape
        if name not in group_sites:
            if original_shape != fresh_shape:
                raise TypeError(
                    f"new_data specialization changed latent layout at {name!r}"
                )
        else:
            group, width = group_sites[name]
            if int(np.prod(fresh_shape)) != len(group.fitted_indices) * width:
                raise TypeError(
                    f"new_data specialization changed group layout at {name!r}"
                )
    return PredictionContext(
        program=program,
        observations=observations,
        rows=rows,
        groups=groups,
        new_level_policy=sample_new_levels,
        seed=seed,
        retain_ordinal_cutpoints=constrained_cs,
        ordinal_order_groups=tuple(sorted(order_groups)) if constrained_cs else (),
    )
