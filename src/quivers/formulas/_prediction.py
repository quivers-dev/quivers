"""Evaluate new formula rows using the fitted encodings and native program."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Never, SupportsIndex

import didactic.api as dx
from formulae.matrices import DesignMatrices
import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl.ast_nodes import (
    Module,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas.formula import (
    Formula,
    RowSelection,
    _CS_ONES,
    _qvr_name,
)
from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._blocks import ResponseData
from quivers.formulas._residual_law import ResidualLaw, recorded_history
from quivers.formulas._grouping import add_grouping_columns
from quivers.formulas._group_law import GroupBlockLaw, LevelSite, law_inputs
from quivers.formulas._joint_compile import place_single_response
from quivers.formulas._joint_missing import (
    SingleMissing,
    apply_missing,
    fitted_measurement_prior,
    fresh_values,
    missing_multipliers,
    missing_site_exclusions,
)
from quivers.formulas._predictor import Predictor
from quivers.formulas._row_source import RowSource, finite_inputs
from quivers.formulas._term import bind_inputs
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._priors import PriorTarget

# Improper flat priors and the brms Intercept column.
from quivers.formulas._fit_core import compile_model
from quivers.formulas._formula_options import BRMS_INTERCEPT
from quivers.inference.registry import LatentRegistry
from quivers.formulas._group_prediction import (
    GroupRemap,
    NewLevelPolicy,
    compact_groups,
    selected_effects,
    specialized_formula,
)
from quivers.formulas._threshold_groups import threshold_selectors

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._group_covariance import GroupCovariance
    from quivers.formulas._predictor_graph import PredictorGraph
    from quivers.formulas._response import ResponseReadout
    from quivers.formulas.compile import FormulaToQVRModule
    from quivers.formulas.family import Family


class FormulaPredictionPlan(ValueModel):
    """Fit-owned evaluator state with an explicit nonportable boundary.

    Parameters
    ----------
    module : Module
        Training program, including its priors and latent dimensions.
    designs : Mapping[str, DesignMatrices]
        Evaluated location and auxiliary formulas, retaining transform state.
    response : ResponseData or None
        Retained outcome evaluator, coordinates and response-side inputs.
    preparation_data : IntoDataFrame or None
        One frozen original-row snapshot for durable design reconstruction.
        Clone-owned recipes share this snapshot internally; reference-owned
        recipes retain a separate training snapshot before caller mutations.
    group_laws : tuple[GroupBlockLaw, ...]
        Laws of the emitted group blocks, with their raw sites and
        coefficient-coordinate slices.
    level_sites : tuple[LevelSite, ...]
        Emitted sites holding values for every level of a grouping factor.
    prior_targets : tuple[PriorTarget, ...]
        Actual emitted parameter roles, authored labels, and nominal axes.
    predictors : tuple[Predictor, ...]
        Planned predictors of the training program.
    missing : SingleMissing or None
        Retained missing-value plans and fitted design evaluators.
    version : str
        Runtime-plan schema identifier. This plan contains formulae objects
        and is deliberately not a portable serialized model representation.
    """

    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    designs: Mapping[str, DesignMatrices] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    response: ResponseData | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    preparation_data: IntoDataFrame | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    predictors: tuple[Predictor, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    level_sites: tuple[LevelSite, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    missing: SingleMissing | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
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


class PredictionRows(ValueModel):
    """Row bindings shared by formula prediction and likelihood requests.

    Parameters
    ----------
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
    fresh_sites : tuple of str
        Latent sites drawn afresh from their law for the prediction rows
        (brms's new latent residuals) rather than replayed from the posterior.
    ordinal_replays : tuple of OrdinalReplay
        Ordinal responses with retained thresholds. Empty means the single
        response of the fit, named without a prefix.
    source : RowSource or None
        Rows of the request; None when the fitted program serves it.
    """

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
    fresh_sites: tuple[str, ...] = ()
    source: RowSource | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    ordinal_replays: tuple[OrdinalReplay, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class PredictionContext(PredictionRows):
    """Compiled prediction over the bindings shared by all draws.

    Parameters
    ----------
    program : MonadicProgram
        Native training program or a row-specialized copy.
    prior_program : MonadicProgram or None
        One dependency projection for conditional new-group draws.
    """

    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prior_program: MonadicProgram | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class PreparedPredictionRows(PredictionRows):
    """Prepared rows with their local AST and family catalog for compilation."""

    module: Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    families: FamilyCatalog | None = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


@dataclass(frozen=True, slots=True)
class JointResponsePreparation:
    """Retained response metadata used to bind rows of a joint prediction."""

    formula: Formula
    family: Family
    lens: FormulaToQVRModule
    observations: Mapping[str, torch.Tensor]
    prediction_plan: FormulaPredictionPlan
    binomial_trials: int | str | None
    thresholds_by: str | None
    _response_data: ResponseData
    _response_readout: ResponseReadout
    predictor: None = None
    predictor_name: None = None

    @property
    def residual(self) -> ResidualLaw | None:
        return self.lens._residual

    @property
    def distributional(self) -> Mapping[str, Formula]:
        return self.lens._distributional

    @property
    def predictor_graph(self) -> PredictorGraph | None:
        return self.lens._predictor_graph

    @property
    def group_covariances(self) -> dict[str, GroupCovariance]:
        return self.lens.group_covariances

    @property
    def category_labels(self) -> tuple[int | str, ...]:
        from quivers.formulas._response import response_category_labels

        return response_category_labels(self.formula, self.family)

    def _predictor_items(self):
        return self.lens._predictor_items(self.formula)

    def _formula_lens(
        self,
        *,
        distributional: Mapping[str, Formula],
        nonlinear_leaves: Mapping[str, Formula],
        residual: ResidualLaw | None,
    ) -> FormulaToQVRModule:
        lens = copy(self.lens)
        lens._set_prior_configuration(None)
        lens._distributional = dict(distributional)
        if self.predictor_graph is not None:
            lens._predictor_graph = self.predictor_graph.with_(
                leaves=dict(nonlinear_leaves)
            )
        lens._residual = residual
        return lens


type PredictionDesign = BayesianFit | JointResponsePreparation


def _set_trial_rows(
    fit: PredictionDesign,
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
    data = fit._response_data
    response = _qvr_name(data.name)
    event = data.layout.event_shape
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
def residual_at(
    law: ResidualLaw,
    data: ResponseData,
    frame: pd.DataFrame | None,
    *,
    include: bool,
    data2: Mapping[str, AutocorMatrix] | None,
) -> ResidualLaw:
    """Rebuild a fit's residual law over prediction rows.

    Parameters
    ----------
    law : ResidualLaw
        The fitted residual law.
    data : ResponseData
        Fitted outcomes and evaluator for recorded history.
    frame : pandas.DataFrame or None
        Prediction rows, or None for the training rows.
    include : bool
        Include the dependence (brms ``incl_autocor``).
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices over the prediction rows.

    Returns
    -------
    ResidualLaw
        The law over the requested rows, with the recorded responses of new
        rows as recursion history.
    """
    history = None
    if frame is not None and law.conditional:
        evaluator = data.evaluator
        assert evaluator is not None
        history = recorded_history(
            frame.reset_index(drop=True),
            evaluator.term.term.eval_new_data,
            float(np.nanmean(np.asarray(data.values))),
            law.response,
        )
    return law.at(frame, include=include, data2=data2, history=history)


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
    retain_law: bool = True,
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
    retain_law : bool
        Expose the law on every retained row of a missing response so replay
        can read its moments or sample it. Training likelihood uses recorded events.

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
    if new_data is not None:
        return compile_prediction_rows(
            fit,
            prepare_prediction_rows(
                fit,
                new_data,
                predictor_data,
                re_formula=re_formula,
                allow_new_levels=allow_new_levels,
                sample_new_levels=sample_new_levels,
                seed=seed,
                trials=trials,
                require_trials=require_trials,
                incl_autocor=incl_autocor,
                data2=data2,
            ),
        )
    _validate_prediction_policy(allow_new_levels, sample_new_levels)
    data = fit._response_data
    readout = fit._response_readout
    formulas, _, response_formula, constrained_cs = _prediction_formulas(fit, readout)
    selected = selected_effects(formulas, re_formula)
    retained_observations = fit.observations
    if trials is not None:
        if readout.trials_parameter is None:
            raise ValueError(
                "trial overrides require a count family with a trial parameter"
            )
        from quivers.formulas._fit import _binomial_trial_tensor

        source = data.training_data
        if isinstance(trials, str) and source is None:
            raise ValueError("a trial column requires a retained training dataframe")
        counts = _binomial_trial_tensor(source, trials, n_obs=len(data.values))
        retained_observations = dict(fit.observations)
        _set_trial_rows(fit, retained_observations, counts, specification=trials)
    # A residual law is rebuilt over the requested rows.
    residual = (
        residual_at(
            fit.residual,
            data,
            None,
            include=incl_autocor,
            data2=data2,
        )
        if fit.residual is not None
        else None
    )
    plan = fit.prediction_plan
    missing = plan.missing if plan is not None else None
    restore_law = retain_law and missing is not None and missing.score_site is not None
    if re_formula is None and incl_autocor and not restore_law:
        if predictor_data is not None:
            raise ValueError("predictor_data requires new_data")
        return PredictionContext(
            program=fit.program,
            observations=retained_observations,
            rows=len(data.values),
        )
    if predictor_data is not None:
        raise ValueError("predictor_data requires new_data")
    if plan is None:
        raise ValueError("this fit has no retained formula prediction plan")
    source = RowSource(selected=selected, include=incl_autocor)
    lens = fit._formula_lens(residual=residual)
    module, complement = lens._forward(response_formula, rows=source)
    families = lens.families()
    observations = dict(retained_observations) | bind_inputs(
        item for term in lens._terms(response_formula, source) for item in term.inputs()
    )
    if missing is not None:
        if plan.preparation_data is None:
            raise ValueError("mi() prediction requires the retained training rows")
        original = nw.from_native(plan.preparation_data, eager_only=True).to_pandas()
        selection = response_formula.modifiers.subset or RowSelection(
            size=len(original), positions=tuple(range(len(original)))
        )
        emission, extra, removed = apply_missing(
            place_single_response(
                missing.label, lens, response_formula, module, complement
            ),
            missing.plan,
            missing.specs,
            original,
            keep_observation=restore_law,
            consumer_rows={missing.label: selection},
            prior=fitted_measurement_prior(plan.module),
        )
        module, families = emission.module, emission.families
        observations = {
            name: value for name, value in observations.items() if name not in removed
        } | extra
        if restore_law:
            from quivers.formulas._response import response_point
            from quivers.formulas._response_modifiers import response_input_name

            observations[response_input_name(response_formula)] = response_point(
                response_formula, fit.family
            ).expand(len(data.values), *data.layout.event_shape)
    program = compile_model(module, families).morphism
    if not isinstance(program, MonadicProgram):
        raise TypeError("prediction requires a native monadic program")
    return PredictionContext(
        program=program,
        observations=observations,
        rows=len(data.values),
        source=source,
        seed=seed,
        retain_ordinal_cutpoints=constrained_cs,
    )


def _validate_prediction_policy(
    allow_new_levels: bool, sample_new_levels: NewLevelPolicy
) -> None:
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


def _prediction_formulas(
    fit: PredictionDesign, readout: ResponseReadout
) -> tuple[dict[str, Formula], dict[str, str], Formula, bool]:
    location = readout.location
    predictors = tuple(
        (name, prefix, formula)
        for name, prefix, formula in fit._predictor_items()
        if prefix or name == location
    )
    formulas = {name: formula for name, _, formula in predictors}
    prefixes = {name: prefix for name, prefix, _ in predictors}
    response_formula = fit.formula if location is None else formulas[location]
    constrained_cs = (
        readout.constrains_cutpoints
        and location is not None
        and response_formula.category_specific
    )
    return formulas, prefixes, response_formula, constrained_cs


def prepare_prediction_rows(
    fit: PredictionDesign,
    new_data: IntoDataFrame,
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
) -> PreparedPredictionRows:
    """Bind fresh rows and retain their local emission without compiling them."""
    _validate_prediction_policy(allow_new_levels, sample_new_levels)
    data = fit._response_data
    readout = fit._response_readout
    location = readout.location
    formulas, prefixes, response_formula, constrained_cs = _prediction_formulas(
        fit, readout
    )
    selected = selected_effects(formulas, re_formula)
    if trials is not None and readout.trials_parameter is None:
        raise ValueError(
            "trial overrides require a count family with a trial parameter"
        )
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    residual = (
        residual_at(fit.residual, data, frame, include=incl_autocor, data2=data2)
        if fit.residual is not None
        else None
    )
    plan = fit.prediction_plan
    if plan is None:
        raise ValueError("this fit has no retained new_data prediction plan")
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

    groups = compact_groups(fit, frame, selected, allow_new_levels)
    order_groups = {
        term.group for term in response_formula.random_terms if term.category_specific
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
    source = RowSource(
        frame=frame,
        designs={prefixes[name]: plan.designs[name] for name in formulas},
        groups=groups,
        selected=selected,
        include=incl_autocor,
    )
    specialized = {
        name: specialized_formula(formula, groups, frame)
        for name, formula in formulas.items()
    }
    specialized_response = (
        specialized_formula(response_formula, groups, frame)
        if location is None
        else specialized[location]
    )
    specialized_lens = fit._formula_lens(
        distributional={
            name: formula
            for name, formula in specialized.items()
            if name in fit.distributional
        },
        nonlinear_leaves={
            name: formula
            for name, formula in specialized.items()
            if fit.predictor_graph is not None and name in fit.predictor_graph.leaves
        },
        residual=residual,
    )
    module, complement = specialized_lens._forward(specialized_response, rows=source)
    families = specialized_lens.families()
    terms = specialized_lens._terms(specialized_response, source)
    observations = finite_inputs(
        bind_inputs(item for term in terms for item in term.inputs())
    )
    observations.update(
        {
            f"{_qvr_name(group)}_idx": remap.row_indices
            for group, remap in groups.items()
        }
    )
    fresh_sites = tuple(
        site.name for term in terms for site in term.sites() if site.fresh
    )

    if fit.predictor_graph is not None:
        from quivers.formulas._predictor_graph import graph_input_bindings

        observations.update(graph_input_bindings(fit.predictor_graph, frame))
    # The response kernel's own data: row values from the fresh rows, the
    # others as fitted unless data2 gives them anew.
    observations.update(
        fit.family.kernel_bindings(frame, {**fit.observations, **(data2 or {})})
    )

    from quivers.formulas._fit import _binomial_trial_tensor, _predictor_output

    rows = len(frame)
    # Responses are internal replay placeholders, never read from new_data.
    # Reusing a fitted support point avoids family-specific prediction algebra.
    from quivers.formulas._response import response_point

    point = response_point(fit.formula, fit.family)
    from quivers.formulas._response_modifiers import response_input_name

    observations[response_input_name(fit.formula)] = point.expand(
        rows, *data.layout.event_shape
    )
    if data.modifiers.weights is not None:
        observations["formula_weights"] = torch.ones(rows, dtype=torch.float64)
    if data.modifiers.decision is not None:
        observations["formula_decision"] = torch.ones(rows, dtype=torch.bool)
    from quivers.formulas._response_roles import prediction_role_bindings

    observations.update(prediction_role_bindings(data.modifiers, frame))
    # brms thres(n, gr): the fitted category selectors of every group.
    observations.update(threshold_selectors(fit.observations))
    if readout.trials_parameter is not None:
        retained_source = dict(data.modifiers.sources).get("trials")
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
        modes = tuple((layer, layer.training) for layer in fit.predictor.modules())
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
            for layer, training in modes:
                layer.training = training
        if not torch.isfinite(neural).all():
            raise ValueError("new_data neural predictor must produce finite values")
        observations[fit.predictor_name] = neural

    # The level inputs of every block on the compact prediction levels.
    observations.update(bind_inputs(law_inputs(specialized_lens.group_laws)))
    if (missing := plan.missing) is not None:
        emission, extra, _ = apply_missing(
            place_single_response(
                missing.label,
                specialized_lens,
                specialized_response,
                module,
                complement,
            ),
            missing.plan,
            missing.specs,
            frame,
            fresh=fresh_values(missing.plan, frame, consumed_only=True),
            multipliers=missing_multipliers(missing.plan, missing.designs, frame),
        )
        module, families = emission.module, emission.families
        observations.update(extra)
    return PreparedPredictionRows(
        module=module,
        families=families,
        observations=observations,
        rows=rows,
        groups=groups,
        new_level_policy=sample_new_levels,
        seed=seed,
        retain_ordinal_cutpoints=constrained_cs,
        ordinal_order_groups=tuple(sorted(order_groups)) if constrained_cs else (),
        fresh_sites=fresh_sites,
        source=source,
    )


def compile_prediction_rows(
    fit: BayesianFit, prepared: PreparedPredictionRows
) -> PredictionContext:
    """Compile the fresh program and check it against the retained latent layout."""
    observations = prepared.observations
    groups = prepared.groups
    fresh_sites = prepared.fresh_sites
    sample_new_levels = prepared.new_level_policy
    original = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    program = compile_model(prepared.module, prepared.families).morphism
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
        site for group in groups.values() for site in group.support_sites
    }
    # Sites drawn for the new rows are not replayed.
    covariance_sites = covariance_sites | set(fresh_sites)
    plan = fit.prediction_plan
    if plan is not None and plan.missing is not None:
        covariance_sites |= missing_site_exclusions(plan.missing.plan, original.names)
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
        and not set(group.sites) <= group.standard_normal_sites
        for group in groups.values()
    ):
        from quivers.inference._prior_projection import project_prior_sites

        prior_program = project_prior_sites(
            program,
            {
                site
                for group in groups.values()
                for site in (*group.sites, *group.support_sites)
            },
        )
    return PredictionContext(
        program=program,
        prior_program=prior_program,
        observations=observations,
        rows=prepared.rows,
        groups=groups,
        new_level_policy=sample_new_levels,
        seed=prepared.seed,
        retain_ordinal_cutpoints=prepared.retain_ordinal_cutpoints,
        ordinal_order_groups=prepared.ordinal_order_groups,
        fresh_sites=prepared.fresh_sites,
        source=prepared.source,
    )
