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

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
    Module,
    ObjectDecl,
    TypeFromExpr,
)
from quivers.formulas._smooth import term_constants
from quivers.formulas.formula import (
    FixedColumn,
    Formula,
    _CS_ONES,
    _qvr_name,
)
from quivers.formulas._gp import (
    gp_conditional_catalog,
    gp_observations,
    gp_prediction,
)
from quivers.formulas._autocor import (
    AutocorMatrix,
    autocor_binding,
    autocor_catalog,
)
from quivers.formulas._grouping import add_grouping_columns
from quivers.formulas._grouping_emission import (
    membership_prediction_bindings,
    option_prediction_bindings,
)
from quivers.formulas._monotonic_emission import monotonic_prediction_bindings
from quivers.formulas._latent_effects import formula_families
from quivers.formulas._autocor import latent_prediction
from quivers.formulas._group_law import GroupBlockLaw
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._priors import PriorTarget

# Improper flat priors and the brms Intercept column.
from quivers.formulas._brms_priors import with_flat_catalog
from quivers.formulas._formula_options import BRMS_INTERCEPT
from quivers.inference.registry import LatentRegistry
from quivers.formulas._group_prediction import (
    GroupRemap,
    NewLevelPolicy,
    compact_groups,
    selected_effects,
    select_contributions,
    specialized_formula,
)
from quivers.formulas._threshold_groups import threshold_selectors

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
    preparation_data : IntoDataFrame or None
        One frozen original-row snapshot for durable design reconstruction.
        Clone-owned recipes share this snapshot internally; reference-owned
        recipes retain a separate training snapshot before caller mutations.
    group_laws : tuple[GroupBlockLaw, ...]
        Laws of the emitted group blocks, with their raw sites and
        coefficient-coordinate slices.
    prior_targets : tuple[PriorTarget, ...]
        Actual emitted parameter roles, authored labels, and nominal axes.
    version : str
        Runtime-plan schema identifier. This plan contains formulae objects
        and is deliberately not a portable serialized model representation.
    """

    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    designs: Mapping[str, DesignMatrices] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    preparation_data: IntoDataFrame | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    version: Literal["qvr-formula-prediction/v1"] = "qvr-formula-prediction/v1"

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Return the coordinate layout of each group block.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Raw sites and coefficient slices derived from the block laws.
        """
        return tuple(law.layout for law in self.group_laws)

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


class OrdinalReplay(ValueModel):
    """Locate one ordinal response whose fitted thresholds are replayed.

    Parameters
    ----------
    prefix : str
        Site-name prefix of the response: empty for a single-response
        program and ``response_<resp>__`` in a joint program.
    thresholds_by : str or None
        Grouping factor of grouped threshold profiles, or None.
    fitted_levels : int
        Number of fitted levels of that grouping factor, zero without one.
    rows : int
        Prediction rows of the response.
    """

    prefix: str
    thresholds_by: str | None = None
    fitted_levels: int = 0
    rows: int = 0


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
    prior_program : MonadicProgram or None
        One dependency projection for conditional new-group draws, prepared
        once per request rather than rebuilt for every posterior draw.
    new_level_policy : NewLevelPolicy
        Joint donor or conditional Gaussian policy for new identities.
    seed : int
        Request seed for the separate group-effect random stream.
    retain_ordinal_cutpoints : bool
        Replay fitted, design-constrained thresholds before applying prediction
        row bindings or varying-effect exclusions.
    ordinal_order_groups : tuple of str
        Groups whose fresh effects or threshold profiles can affect order.
    fresh_sites : tuple of str
        Latent sites drawn afresh from their law for the prediction rows
        (brms's new latent residuals) rather than replayed from the posterior.
    ordinal_replays : tuple of OrdinalReplay
        Ordinal responses with retained thresholds. Empty means the single
        response of the fit, named without a prefix.
    """

    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    rows: int
    groups: dict[str, GroupRemap] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    prior_program: MonadicProgram | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    new_level_policy: NewLevelPolicy = "uncertainty"
    seed: int = 0
    retain_ordinal_cutpoints: bool = False
    ordinal_order_groups: tuple[str, ...] = ()
    fresh_sites: tuple[str, ...] = ()
    ordinal_replays: tuple[OrdinalReplay, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


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
    columns: tuple[FixedColumn, ...],
    matrices: Mapping[str, np.ndarray],
    *,
    structured_widths: Mapping[str, int] | None = None,
) -> Iterator[tuple[FixedColumn, np.ndarray]]:
    """Match fitted exploded-column labels to fresh term-matrix positions.

    Parameters
    ----------
    columns : tuple[FixedColumn, ...]
        Fitted labels in design-column order.
    matrices : Mapping[str, numpy.ndarray]
        Fresh matrices keyed by the original term name.
    structured_widths : Mapping[str, int] or None
        Full retained width for terms whose remaining columns belong to
        explicitly identified coupled Gaussian designs.

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
        if matrices[name].shape[1] != (structured_widths or {}).get(name, count):
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
    result = monotonic_prediction_bindings(formula, design, frame, prefix)
    for term in formula.design_terms:
        values = matrices[term.term][:, term.column_indices]
        result[prefix + term.qvr_name] = torch.as_tensor(
            values.copy(), dtype=torch.float64
        )
        result.update(gp_observations(prefix + term.qvr_name, term.gp))
    structured_widths = {
        term.term: max(term.column_indices) + 1 for term in formula.design_terms
    }
    for column, values in _column_values(
        tuple(
            column
            for column in formula.fixed_columns
            if isinstance(column, FixedColumn)
        ),
        matrices,
        structured_widths=structured_widths,
    ):
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
                if isinstance(term.column, FixedColumn)
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
    if design.group is not None:
        result.update(
            membership_prediction_bindings(
                formula,
                {term.expr.name: term.expr for term in design.group.terms.values()},
                frame,
                prefix,
                groups,
            )
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
        from quivers.formulas._response import response_dtype

        observations[response] = torch.zeros(
            len(counts),
            device=counts.device,
            dtype=response_dtype(fit.formula, fit.family),
        )


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
    incl_autocor: bool = True,
    data2: Mapping[str, AutocorMatrix] | None = None,
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
    trials : int or str or None
        Trial count or column for count families.
    require_trials : bool
        Require trials for fresh vector-count rows.
    incl_autocor : bool
        Include an autocorrelation term's structure (brms ``incl_autocor``);
        False predicts independent rows from the same residual scale.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Matrices of ``sar`` and ``fcor`` terms over the rows of ``new_data``
        (brms ``newdata2``).

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
    formulas = {name: formula for name, _, formula in fit._predictor_items()}
    prefixes = {name: prefix for name, prefix, _ in fit._predictor_items()}
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
    # Autocorrelation terms rebuild their residual law for the requested rows.
    autocor_families = (
        autocor_catalog(
            fit,
            None
            if new_data is None
            else nw.from_native(new_data, eager_only=True).to_pandas(),
            include=incl_autocor,
            data2=data2,
        )
        if autocor_binding(fit.family) is not None
        else None
    )
    if new_data is None and re_formula is None and incl_autocor:
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
        module = select_contributions(
            plan.module, formulas, selected, prefixes=prefixes
        )
        program = (
            Compiler(
                module,
                # Improper flat priors are compilation-local native laws.
                families=with_flat_catalog(
                    formula_families(
                        autocor_families
                        if autocor_families is not None
                        else fit.family.custom.catalog
                        if fit.family.custom is not None
                        else None,
                        fit.formula,
                    ),
                    module,
                ),
            )
            .compile()
            .morphism
        )
        if not isinstance(program, MonadicProgram):
            raise TypeError("prediction requires a native monadic program")
        return PredictionContext(
            program=program,
            observations=dict(retained_observations)
            | latent_prediction(fit, None, include=incl_autocor)[1],
            rows=len(fit.formula.response_values),
            seed=seed,
            retain_ordinal_cutpoints=constrained_cs,
        )
    if plan is None:
        raise ValueError("this fit has no retained new_data prediction plan")
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    # brms g1:g2 and mm() factors are derived from their source columns.
    frame = add_grouping_columns(
        frame,
        tuple(
            {
                spec.name: spec
                for formula in formulas.values()
                for spec in formula.grouping.specs
            }.values()
        ),
        require=False,
    )
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
    # brms 0 + Intercept reads a reserved constant column.
    if any(
        BRMS_INTERCEPT in column.term
        for formula in formulas.values()
        for column in formula.fixed_columns
    ):
        if BRMS_INTERCEPT in frame.columns:
            raise ValueError(f"{BRMS_INTERCEPT!r} is reserved for the Intercept term")
        frame = frame.copy()
        frame[BRMS_INTERCEPT] = 1.0
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
    for parameter, prefix, formula in fit._predictor_items():
        bindings = _predictor_bindings(
            formula,
            plan.designs[parameter],
            frame,
            prefix,
            groups,
            selected,
        )
        for name, value in bindings.items():
            if name in observations and not torch.equal(value, observations[name]):
                raise ValueError(f"new_data has inconsistent shared input {name!r}")
            observations[name] = value

    if fit.predictor_graph is not None:
        from quivers.formulas._predictor_graph import graph_input_bindings

        observations.update(graph_input_bindings(fit.predictor_graph, frame))
    # brms gr(by =, pw =): per-level inputs on the compact prediction levels.
    observations.update(
        option_prediction_bindings(fit._predictor_items(), frame, groups)
    )

    from quivers.formulas._fit import _binomial_trial_tensor, _predictor_output

    rows = len(frame)
    # Responses are internal replay placeholders, never read from new_data.
    # Reusing a fitted support point avoids family-specific prediction algebra.
    from quivers.formulas._response import response_point

    point = response_point(fit.formula, fit.family)
    from quivers.formulas._response_modifiers import response_input_name

    observations[response_input_name(fit.formula)] = point.expand(
        rows, *fit.formula.response_layout.event_shape
    )
    if fit.formula.modifiers.weights is not None:
        observations["formula_weights"] = torch.ones(rows, dtype=torch.float64)
    if fit.formula.modifiers.decision is not None:
        observations["formula_decision"] = torch.ones(rows, dtype=torch.bool)
    from quivers.formulas._response_roles import prediction_role_bindings

    observations.update(prediction_role_bindings(fit.formula.modifiers, frame))
    # brms thres(n, gr): the fitted category selectors of every group.
    observations.update(threshold_selectors(fit.observations))
    # Latent terms (brms car, latent residuals) cover the prediction rows.
    latent_formula, latent_values, fresh_sites = latent_prediction(
        fit, frame, include=incl_autocor
    )
    observations.update(latent_values)
    # gp-exact: exact gp() blocks draw their latent function at the new inputs.
    gp_formulas, gp_values, gp_fresh = gp_prediction(
        fit._predictor_items(), plan.designs, frame
    )
    observations.update(gp_values)
    gp_catalog = gp_conditional_catalog(
        (prefixes[name] + term.qvr_name, term.gp)
        for name, formula in gp_formulas.items()
        for term in formula.design_terms
        if term.gp is not None
    )
    if fit.family.trials_parameter is not None:
        retained_source = dict(fit.formula.modifiers.sources).get("trials")
        specification = (
            trials
            if trials is not None
            else (
                retained_source
                if isinstance(retained_source, (str, int))
                and not isinstance(retained_source, bool)
                else fit.binomial_trials
            )
        )
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
    if groups or re_formula is not None or gp_fresh:
        specialized = {
            name: specialized_formula(gp_formulas[name], groups) for name in formulas
        }
        specialized_lens = fit._formula_lens(
            distributional={
                name: formula
                for name, formula in specialized.items()
                if name in fit.distributional
            },
            nonlinear_leaves={
                name: formula
                for name, formula in specialized.items()
                if fit.predictor_graph is not None
                and name in fit.predictor_graph.leaves
            },
        )
        module, _ = specialized_lens.forward(specialized["mu"])
        observations.update(
            {
                name: value
                for name, value in specialized_lens.fixed_column_observations(
                    specialized["mu"]
                ).items()
                if name.startswith("__formula_level_")
            }
        )
        module = select_contributions(module, formulas, selected, prefixes=prefixes)
    statements = []
    fresh_axes = {
        term.axis for term in latent_formula.latent_terms if term.name in fresh_sites
    }
    for statement in module.statements:
        if isinstance(statement, ObjectDecl) and (
            statement.names == ("Resp",) or statement.names[0] in fresh_axes
        ):
            statement = statement.with_(
                init=TypeFromExpr(
                    expr=DiscreteConstructor(constructor="FinSet", args=(str(rows),))
                )
            )
        statements.append(statement)
    families = formula_families(
        autocor_families
        if autocor_families is not None
        else fit.family.custom.catalog
        if fit.family.custom is not None
        else None,
        latent_formula,
    )
    if gp_catalog is not None:
        families = gp_catalog if families is None else families.with_catalog(gp_catalog)
    # Improper flat priors are compilation-local native laws.
    families = with_flat_catalog(families, module)
    program = (
        Compiler(module.with_(statements=tuple(statements)), families=families)
        .compile()
        .morphism
    )
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
    covariance_sites = {
        f"group_covariance_{_qvr_name(law.identity)}"
        for group in groups.values()
        for law in group.covariance_laws
    }
    # Fresh latent residuals are drawn for the new rows, not replayed.
    covariance_sites = covariance_sites | set(fresh_sites) | set(gp_fresh)
    if (
        set(original.names) - covariance_sites
        != set(specialized_registry.names) - covariance_sites
    ):
        raise TypeError("new_data specialization changed the fitted latent names")
    for name in original.names:
        if name in covariance_sites:
            continue
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
    prior_program = None
    if sample_new_levels.removeprefix("feasible_") == "gaussian" and any(
        -1 in group.fitted_indices
        and group.covariance is None
        and not set(group.sites) <= group.standard_normal_sites
        for group in groups.values()
    ):
        from quivers.inference._prior_projection import project_prior_sites

        prior_program = project_prior_sites(program, group_sites)
    return PredictionContext(
        program=program,
        observations=observations,
        rows=rows,
        groups=groups,
        prior_program=prior_program,
        new_level_policy=sample_new_levels,
        seed=seed,
        retain_ordinal_cutpoints=constrained_cs,
        ordinal_order_groups=tuple(sorted(order_groups)) if constrained_cs else (),
        fresh_sites=(*fresh_sites, *gp_fresh),
    )
