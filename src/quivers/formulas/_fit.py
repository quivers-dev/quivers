"""User-facing `fit` entry point and the
`BayesianFit` result wrapper.

The compilation path is fully AST-driven: the formula lens emits a
[`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module], the existing
[`quivers.dsl.compiler.Compiler`][quivers.dsl.compiler.Compiler] consumes it directly (no
source-string round-trip), and inference runs on the resulting
[`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram].  Source text
is generated only when the user requests it via
`formula_to_qvr` or `BayesianFit.dump_qvr`, in which
case [`quivers.dsl.emit.module_to_source`][quivers.dsl.emit.module_to_source] walks the same AST
to produce canonical ``.qvr`` source.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
import numpy as np
from quivers._count_capacity import _MAX_EXACT_COUNT
from quivers.formulas._count_data import checked_integer_data
from pathlib import Path
from collections.abc import Generator
from typing import Callable, Literal, Mapping, Sequence, Unpack, cast

import didactic.api as dx
from formulae.matrices import DesignMatrices
import narwhals as nw
import torch
from torch import nn
from torch.distributions import Distribution
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.cross_validation import FoldPrediction, FoldSplit
from quivers.dsl.emit import module_to_source
from quivers.dsl.ast_nodes import Module
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas.family import Family, families
from quivers.formulas._mixture import mixture_component_count, mixture_distributional
from quivers.formulas._mixture_predict import pp_mixture
from quivers.formulas._category_formulas import (
    category_distributional,
    check_scale_formulas,
    is_category_parameter,
    is_scale_parameter,
    main_without_terms,
)
from quivers.formulas.formula import (
    Formula,
    FormulaData,
    RowSelection,
    _formula_and_design,
    _qvr_name,
)
from quivers.formulas._row_retention import (
    complete_rows,
    compose_positions,
    model_columns,
    model_texts,
)
from quivers.formulas._predictor_graph import (
    NonlinearExpression,
    PredictorGraph,
    build_predictor_graph,
    nonlinear_response_source,
    predictor_items,
)

from quivers.formulas._prediction import FormulaPredictionPlan, prepare_prediction
from quivers.formulas._r_squared import formula_r2
from quivers.formulas._autocor import (
    AutocorMatrix,
    fields_and_design,
    split_autocor_terms,
)
from quivers.formulas._residual_law import (
    ResidualLaw,
    bind_autocor_terms,
    linear_predictor,
    pointwise_scores,
)

# Joint (multivariate) formulas: typed overload contracts and dispatch target.
from typing import overload
from quivers.formulas._joint_options import FormulaFitOptions, JointFitOptions
from quivers.formulas._program_kernel import program_arguments, program_source
from quivers.formulas._multivariate import (
    MultivariateFormula,
    ResponseModel,
    is_mvbind,
    uses_missing_syntax,
)

# brms brm(bf(y ~ x, sigma ~ z)): a single-response bf as the formula.
from quivers.formulas._single_response import single_response
from quivers.formulas._ordinal_coding import CategoryCoding, family_category_coding
from quivers.formulas._brms_hypothesis import (
    HypothesisInput,
    HypothesisTable,
    fit_hypothesis,
    is_brms_hypothesis,
)
from quivers.formulas._threshold_groups import (
    absent_thresholds,
    check_threshold_groups,
    threshold_kinds,
)

# brms inits(): the chains' initial values, read at the driver.
from quivers.formulas._initial_values import report_initial_values

# gr(by = ) and gr(dist = "student"): per-level structure of group coefficients.
from quivers.formulas._group_law import group_coefficients

from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._response import (
    count_totals,
    expected_response,
    response_tensor,
)
from quivers.formulas._priors import PriorInput, PriorSet, normalize_priors

# brms sampler and fitting controls.
from quivers.formulas._fit_core import compile_model
from quivers.formulas._fit_controls import (
    Algorithm,
    FileRefit,
    InferenceControls,
    InitSpec,
    SamplerControl,
    chain_initialization,
    mcmc_kernel,
    needs_refit,
    resolve_algorithm,
    sampler_control,
)

# Pathfinder variational inference.
from quivers.formulas._fit_controls import (
    FitMethod,
    PathfinderOptions,
    checked_cores,
    pathfinder_driver,
    pathfinder_options,
)
import quivers.modeling.artifact as artifact_store

from quivers.formulas._formula_options import checked_unused, sparse_center

# brms prior policy, improper flat laws, and derived coefficient sites.
from quivers.formulas._brms_priors import (
    DefaultPriorPolicy,
    PriorTable,
    prior_table,
    require_proper,
    user_selection,
)
from quivers.formulas._response_modifiers import (
    ResponseDraws,
    ExpectationOptions,
    PredictionOptions,
    response_input_name,
)
from quivers.inference import (
    ELBO,
    MCMC,
    SVI,
    AutoNormalGuide,
    PriorDraws,
    PriorMode,
    SMC,
    SMCResult,
)
from quivers.modeling.refit import RefitRecipe, SnapshotPolicy
from quivers.formulas._posterior import (
    LinpredBinding,
    collect_draws,
    posterior_draws,
    replay,
    response_distribution,
    response_rng,
    row_binding,
)
from quivers.formulas._predictor import Predictor, response_axis
from quivers.inference._posterior import draw_count
from quivers.inference.trace import Trace
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


from quivers.diagnostics.posterior import PosteriorAnalysis
from quivers.diagnostics.conditional import ConditionalEffects, PredictionGrid
from quivers.diagnostics.contrast import ContrastExpression
from quivers.diagnostics.hypothesis import HypothesisSummary
from quivers.diagnostics.density import PointNullSpec

# brms post-processing methods (summaries, accessors, LOO family) and stored criteria.
from quivers.diagnostics.elpd import NO_CRITERIA, ELPDEstimate
from quivers.formulas._postprocess_methods import SinglePostprocess

# brms evidence, effects and predictive checks.
from quivers.formulas._evidence_effects import EvidenceEffectsMethods
from quivers.formulas._brms_effects import (
    BrmsConditionalEffects,
    ConditionalEffectsOptions,
    EffectMethod,
    automatic_conditional_effects,
    grid_options,
)

# Pathfinder draws report variational provenance.
from quivers.inference.pathfinder import draw_provenance

# brms knots and drop_unused_levels.
from quivers.formulas._data_options import Knots, formula_data_options

# brms cores: chains in worker processes.
from quivers.inference.mcmc.parallel import parallel_chains, run_chains
from quivers.formulas._parallel_chains import (
    ChainsRan,
    FormulaChainWorker,
    assigned_chains,
)


class BayesianFit(SinglePostprocess, EvidenceEffectsMethods, ValueModel):
    """A fitted Bayesian regression: the compiled program, the
    parsed formula, the family, the user-supplied data, and the
    posterior samples.

    Parameters
    ----------
    formula : Formula
        Parsed formula IR.
    family : Family
        Response family used at compile time.
    residual : ResidualLaw or None
        Residual law of the response: natural residuals correlated across
        rows by an autocorrelation term; None for independent rows.
    program : MonadicProgram
        The compiled program.
    posterior : MCMCResult or PriorDraws or Guide
        [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult] from NUTS / HMC, or
        a [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] from SVI.
        A prior view instead retains independently generated PriorDraws.
    sample_prior : {"no", "yes", "only"}
        Ordinary fitting, additionally retained independent priors, or
        observation-free inference with explicit score potentials retained.
    prior : PriorDraws or MCMCResult or None
        Independent prior samples, or explicitly factor-conditioned chains.
    losses : tuple[float, ...]
        Finite optimization losses for SVI; empty for recorded MCMC chains.
    observations : Mapping[str, torch.Tensor]
        Inference-time observations dict (response + per-column
        covariates + per-group plate indices).
    group_covariances : dict[str, GroupCovariance]
        Known labeled covariance between grouping levels, including any
        explicitly supplied future labels. Coefficient correlations remain separate.
    fixed_parameters : dict[str, float]
        Auxiliary parameters bound to constants rather than sampled or regressed.
    distributional : Mapping[str, Formula]
        Evaluated auxiliary-parameter design matrices, retained so source
        inspection emits the fitted model without re-evaluating transforms.
    predictor_graph : PredictorGraph or None
        Named nonlinear designs, typed expressions, and numeric input contracts.
    refit_recipe : RefitRecipe or None
        Retained original data and canonical fitting configuration, held by identity.
        Clone ownership is the default; explicit reference ownership observes mutations.
    prediction_plan : FormulaPredictionPlan or None
        Live fitted evaluators for new-data prediction. The versioned runtime
        plan is held by identity and explicitly refuses durable serialization.
    criteria : Mapping[str, ELPDEstimate or torch.Tensor]
        Criteria stored by `add_criterion`: ELPD estimates, or R-squared draws.
    default_priors : {"quivers", "brms"}
        Default prior policy the model was compiled with.
    center : bool or None
        brms intercept centering option.
    decomp : {"none", "QR"}
        brms population-coefficient decomposition.

    Notes
    -----
    Fits compare by value over the formula, family, and settings, and
    by identity over the live objects: the program, the posterior, the
    observations mapping, the predictor, its data, the prediction plan, and the
    refit recipe. Two fits are
    thus equal exactly when they share those objects and agree on
    everything else.
    """

    formula: Formula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: Family = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    residual: ResidualLaw | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | PriorDraws | Guide = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_prior: PriorMode = "no"
    prior: PriorDraws | MCMCResult | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    losses: tuple[float, ...] = ()
    observations: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    reparameterize: Literal["centered", "noncentered"] = "noncentered"
    binomial_trials: int | str | None = None
    thresholds_by: str | None = None
    mixture_components: int = 2
    fixed_prior: str = "Normal(0.0, 5.0)"
    random_scale_prior: str = "HalfNormal(1.0)"
    correlation_prior: str = "LKJCholesky(1.0)"
    group_covariances: dict[str, GroupCovariance] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    priors: PriorSet = dx.field(
        default_factory=PriorSet, opaque=True, extras=OPAQUE_BY_VALUE
    )
    predictor: nn.Module | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    predictor_data: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    predictor_name: str | None = None
    fixed_parameters: dict[str, float] = dx.field(default_factory=dict)
    distributional: Mapping[str, Formula] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    predictor_graph: PredictorGraph | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prediction_plan: FormulaPredictionPlan | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    refit_recipe: (
        RefitRecipe[IntoDataFrame, Mapping[str, object], BayesianFit] | None
    ) = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    # Criteria stored by add_criterion (brms ``x$criteria``), kept in artifacts.
    criteria: Mapping[str, ELPDEstimate | torch.Tensor] = dx.field(
        default=NO_CRITERIA, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    # brms prior policy and population-coordinate options.
    default_priors: DefaultPriorPolicy = "quivers"
    center: bool | None = None
    decomp: Literal["none", "QR"] = "none"

    def update(
        self, data: IntoDataFrame | None = None, **overrides: object
    ) -> BayesianFit:
        """Fit retained or replacement rows through the original fitting recipe.

        Parameters
        ----------
        data : IntoDataFrame or None
            Original-frame snapshot by default, or replacement unprepared rows.
        **overrides : object
            Explicit overrides of the canonical fit arguments, including formula,
            family, priors, inference method, budgets, or random seed. Changing
            methods resets inactive budgets unless explicitly overridden.

        Returns
        -------
        BayesianFit
            A fresh fitted result; the retained posterior and predictor are unchanged.

        Raises
        ------
        ValueError, TypeError
            If no recipe was retained or an override fails canonical fit validation.
        """
        from quivers.formulas._refit import update_fit

        return update_fit(self, data, overrides)

    def refit_callback(
        self,
        *,
        method_factory: Callable[[Mapping[str, object], int], Mapping[str, object]],
        partition: Callable[[IntoDataFrame, tuple[int, ...]], IntoDataFrame]
        | None = None,
        num_samples: int | None = None,
        **prediction: object,
    ) -> Callable[[FoldSplit], FoldPrediction]:
        """Adapt retained included rows and neural inputs to explicit refits.

        Parameters
        ----------
        method_factory : Callable[[Mapping[str, object], int], Mapping[str, object]]
            Fold-specific canonical fit settings and seed. Retained row
            attachments are partitioned automatically and cannot be replaced.
        partition : Callable[[IntoDataFrame, tuple[int, ...]], IntoDataFrame] or None
            Optional nonmutating original-frame row selection. By default,
            Narwhals preserves the dataframe backend and authored columns.
        num_samples : int or None
            Per-chain draw budget for raw held-out likelihood scoring.
        **prediction : object
            Existing log_likelihood options, including re_formula and explicit
            new-group policies. The adapter owns rows, neural inputs, and seeds.

        Returns
        -------
        Callable[[FoldSplit], FoldPrediction]
            Callback for cross_validate. Fold indices address fitted included
            rows; subset positions map them to the original raw frame once.
            Neural predictor_data follows the same train/test selection.

        Raises
        ------
        ValueError, TypeError
            If retained rows or attachments do not align, a partition has the
            wrong included-row count, or options fail canonical validation.
        """
        from quivers.formulas._cross_validation import formula_refitter

        return formula_refitter(
            self,
            method_factory=method_factory,
            partition=partition,
            num_samples=num_samples,
            prediction=prediction,
        )

    def evidence(self, sampler: SMC) -> SMCResult:
        """Estimate the retained formula target's normalizing constant with SMC.

        Parameters
        ----------
        sampler : SMC
            Configured likelihood-tempered driver, including its particle
            budget, rejuvenation kernel, and random stream.

        Returns
        -------
        SMCResult
            Independent particles and the estimated log normalizing constant.
            Ordinary data evidence requires a normalized prior and likelihood;
            response weights define a powered target's normalizer instead.

        Raises
        ------
        ValueError
            If SMC cannot partition the target, normalize its prior, or use
            an exact likelihood without observation subsampling.
        RuntimeError
            If no temperature step meets the driver's ESS requirement.

        Notes
        -----
        The compiled program, retained designs, fixed parameters, and observed
        tensors are reused without recompilation. The fitted posterior is
        neither sampled nor modified. Seeded drivers restore the caller's RNG.
        """
        # An improper flat prior has no normalizing constant.
        self._require_proper("SMC evidence")
        return sampler.run(self.program, torch.zeros(1, 1), dict(self.observations))

    def _require_proper(self, purpose: str) -> None:
        """Refuse an operation needing a normalized prior on every site.

        Parameters
        ----------
        purpose : str
            Operation named in the error.

        Raises
        ------
        ValueError
            If a site carries the improper flat prior.
        """
        if self.prediction_plan is not None:
            require_proper(self.prediction_plan.module, purpose)

    def prior_summary(self) -> PriorTable:
        """Tabulate the effective prior of every parameter, as brms ``prior_summary``.

        Returns
        -------
        PriorTable
            brms's columns, with ``source`` ``user`` for rules given to `fit`.

        Raises
        ------
        ValueError
            If the fit retains no emission.
        """
        plan = self.prediction_plan
        if plan is None:
            raise ValueError("prior_summary requires the fit's retained emission")
        return prior_table(
            plan.module,
            plan.prior_targets,
            user_selection(self.priors, plan.prior_targets),
            policy=self.default_priors,
        )

    @overload
    def conditional_effects(
        self,
        grid: PredictionGrid,
        *,
        method: EffectMethod = ...,
        parameter: str = ...,
        transform: bool = ...,
        **options: Unpack[ConditionalEffectsOptions],
    ) -> ConditionalEffects: ...

    @overload
    def conditional_effects(
        self,
        grid: str | Sequence[str] | None = ...,
        *,
        method: EffectMethod = ...,
        parameter: str = ...,
        transform: bool = ...,
        **options: Unpack[ConditionalEffectsOptions],
    ) -> BrmsConditionalEffects: ...

    def conditional_effects(
        self,
        grid: PredictionGrid | str | Sequence[str] | None = None,
        *,
        method: EffectMethod = "epred",
        parameter: str = "mu",
        transform: bool = False,
        **options: Unpack[ConditionalEffectsOptions],
    ) -> ConditionalEffects | BrmsConditionalEffects:
        """Predict one joint grid and retain conditional or empirical effects.

        Without a grid, or with brms ``effects`` such as ``"x"`` or
        ``"x:z"``, this computes brms's automatic conditional effects
        (`quivers.formulas._brms_effects.brms_conditional_effects`): default
        effects and conditions, grids, and labeled summary frames.

        Parameters
        ----------
        grid : PredictionGrid or str or Sequence[str] or None
            Explicit effect coordinates and conditioning/reference rows,
            constructed with `quivers.diagnostics.prediction_grid`; or brms
            ``effects`` (None for brms's default effects).
        method : str
            ``epred``, ``linpred``, ``predict``, or brms's names
            ``posterior_epred``, ``posterior_linpred``, ``posterior_predict``.
            Expectations average response laws, not sampled replications.
        parameter : str
            Predictor selected only for linpred; other methods require mu.
        transform : bool
            Apply the inverse link only for linpred.
        **options : ConditionalEffectsOptions
            Existing prediction keywords, including varying-effect selectors,
            new-level policies, trial/neural inputs, draw budget, and seed.
            The grid supplies new_data; mc_samples applies only to epred.
            Without a grid, brms's ``conditional_effects`` arguments
            (``conditions``, ``int_conditions``, ``re_formula`` defaulting to
            ``"~0"``, ``prob``, ``robust``, ``spaghetti``, ``surface``,
            ``categorical``, ``ordinal``, ``resolution``, ``select_points``,
            ``too_far``, ``dpar``, ``nlpar``, ``ndraws``, ``draw_ids``, ``seed``).

        Returns
        -------
        ConditionalEffects or BrmsConditionalEffects
            Joint chain/draw/point/condition/event predictions, with reference
            rows averaged within each draw only for empirical grids. Retained
            draws support paired contrasts, covariance, and common summaries.
            Without a grid, brms's labeled effect frames.

        Raises
        ------
        ValueError, TypeError
            If the method/options or grid are invalid, or the existing fitted
            prediction contract rejects a predictor, group, or response law.
        """
        # brms automatic conditional effects.
        if not isinstance(grid, PredictionGrid):
            return automatic_conditional_effects(
                self,
                grid,
                method=method,
                parameter=parameter,
                transform=transform,
                resp=None,
                options=options,
            )
        expectation = grid_options(options)
        short = method.removeprefix("posterior_")
        if "new_data" in options:
            raise ValueError("conditional_effects takes new_data from its grid")
        if short not in {"epred", "linpred", "predict"}:
            raise ValueError("method must be epred, linpred, or predict")
        method = cast(Literal["epred", "linpred", "predict"], short)
        if method != "linpred" and (parameter != "mu" or transform):
            raise ValueError("parameter and transform apply only to linpred effects")
        if method != "epred" and "mc_samples" in options:
            raise ValueError("mc_samples applies only to epred effects")
        request = expectation.copy()
        request["new_data"] = grid.to_frame()
        if method == "epred":
            predictions = self.posterior_epred(**request)
        else:
            # The guard above excludes the sole expectation-only key.
            prediction_options = cast(PredictionOptions, request)
            if method == "linpred":
                predictions = self.posterior_linpred(
                    parameter=parameter,
                    transform=transform,
                    **prediction_options,
                )
            else:
                predictions = self.posterior_predict(**prediction_options)
        return grid.result(
            predictions,
            kind=draw_provenance(self.posterior),
            method=method,
            labels=self.predictor_labels(parameter)
            if method == "linpred"
            else self.category_labels,
        )

    def prior_draws(
        self,
        *,
        num_samples: int = 100,
        seed: int = 0,
        method: Literal["ancestral"] | MCMC = "ancestral",
    ) -> PriorDraws | MCMCResult:
        """Draw conditional priors using retained designs and data-dependent priors.

        Parameters
        ----------
        num_samples : int
            Positive number of independent ancestral draws.
        seed : int
            Isolated CPU seed; the caller's stream is restored.
        method : {"ancestral"} or MCMC
            Independent conditional sampling, or an explicit configured driver
            for global score potentials. Its own budgets control MCMC.

        Returns
        -------
        PriorDraws or MCMCResult
            Constrained samples with independent provenance or genuine chains.
        """
        from quivers.inference.prior import prior_draws

        self._require_proper("prior_draws")
        return prior_draws(
            self.program,
            torch.zeros(1, 1),
            self.observations,
            num_samples=num_samples,
            seed=seed,
            method=method,
        )

    def prior_view(
        self,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        method: Literal["ancestral"] | MCMC = "ancestral",
    ) -> BayesianFit:
        """Reuse prediction, likelihood, and analysis methods with prior draws.

        Parameters
        ----------
        num_samples : int or None
            Fresh independent draw count. None reuses retained prior samples,
            or draws 100 if none were retained.
        seed : int
            Isolated CPU seed for fresh ancestral draws.
        method : {"ancestral"} or MCMC
            Conditional sampling or explicitly factor-conditioned inference.

        Returns
        -------
        BayesianFit
            A new view sharing the fitted model, encodings, and data.
        """
        prior = (
            self.prior
            if self.prior is not None and num_samples is None and method == "ancestral"
            else self.prior_draws(
                num_samples=100 if num_samples is None else num_samples,
                seed=seed,
                method=method,
            )
        )
        return self.with_(posterior=prior, prior=prior, sample_prior="only", losses=())

    def _formula_lens(
        self,
        *,
        distributional: Mapping[str, Formula] | None = None,
        nonlinear_leaves: Mapping[str, Formula] | None = None,
        residual: ResidualLaw | None = None,
    ) -> FormulaToQVRModule:
        """Reuse fitted compiler settings with optional specialized predictors.

        Parameters
        ----------
        distributional : Mapping[str, Formula] or None
            Specialized auxiliary metadata, or the fitted metadata.
        nonlinear_leaves : Mapping[str, Formula] or None
            Group-specialized designs for the same nonlinear graph.
        residual : ResidualLaw or None
            The residual law over other rows, or None for the fitted one.

        Returns
        -------
        FormulaToQVRModule
            A fresh lens retaining every fitted prior and compiler option.
        """
        return residual_lens(
            self.residual if residual is None else residual,
            self.fixed_parameters,
            lambda fixed: FormulaToQVRModule(
                self.family,
                fixed_parameters=fixed,
                fixed_prior=self.fixed_prior,
                random_scale_prior=self.random_scale_prior,
                correlation_prior=self.correlation_prior,
                group_covariances=self.group_covariances,
                user_priors=self.priors,
                reparameterize=self.reparameterize,
                binomial_trials=self.binomial_trials,
                thresholds_by=self.thresholds_by,
                mixture_components=self.mixture_components,
                predictor_name=self.predictor_name,
                distributional=self.distributional
                if distributional is None
                else distributional,
                predictor_graph=self.predictor_graph
                if nonlinear_leaves is None or self.predictor_graph is None
                else self.predictor_graph.with_(leaves=dict(nonlinear_leaves)),
                default_priors=self.default_priors,
                center=self.center,
                decomp=self.decomp,
            ),
        )

    @property
    def qvr_source(self) -> str:
        """Lazily emit the AST-equivalent ``.qvr`` source for display.

        Returns
        -------
        str
            Canonical source emitted from the retained formula AST.
        """
        module, _ = self._formula_lens().forward(self.formula)
        return module_to_source(module)

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the AST-equivalent ``.qvr`` source to ``path`` and
        return the resulting `Path`.

        Parameters
        ----------
        path : str or Path
            Destination for canonical QVR source.

        Returns
        -------
        Path
            Written source path.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out

    @property
    def included_rows(self) -> tuple[int, ...]:
        """Positions of the input data rows the model was fitted to.

        Rows excluded by ``subset`` or by a missing value in any model
        variable are absent. Pointwise likelihoods and predictions on the
        training rows follow this order.

        Returns
        -------
        tuple[int, ...]
            Zero-based positions in the data passed to ``fit``.
        """
        selection = self.formula.modifiers.subset
        if selection is not None:
            return selection.positions
        return tuple(range(len(self.formula.response_values)))

    @property
    def category_labels(self) -> tuple[int | str, ...]:
        """Response-category labels in posterior probability-array order.

        Returns
        -------
        tuple[int or str, ...]
            Factor levels, the integer values ``1..K`` of an ordinal response
            coded as brms codes it, or the codes ``0..K-1``; empty for other
            families. Predicted categories are positions in this tuple.
            Categorical coefficient arrays omit the first category.
        """
        if self.formula.response_layout.component_names:
            return self.formula.response_layout.component_names
        # brms integer categories keep their integer values as labels.
        if self.formula.response_layout.category_values:
            return self.formula.response_layout.category_values
        if self.formula.response_layout.category_levels:
            return self.formula.response_layout.category_levels
        if self.family.ordinal is None and not self.family.unordered_categories:
            return ()
        return tuple(
            range(
                FormulaToQVRModule._category_count(
                    self.formula, family=self.family.name
                )
            )
        )

    @property
    def threshold_labels(self) -> tuple[tuple[int | str, int | str], ...]:
        """Adjacent category pairs identifying the ordinal transition axis.

        Returns
        -------
        tuple of pair of int
            Adjacent category labels, excluding the structural-zero boundary
            in a hurdle cumulative model; empty for nonordinal families.
        """
        if self.family.ordinal is None:
            return ()
        labels = self.category_labels
        if self.family.ordinal.kind == "hurdle_cumulative":
            labels = labels[1:]
        return tuple(zip(labels[:-1], labels[1:], strict=True))

    @property
    def group_levels(self) -> dict[str, tuple[str, ...]]:
        """Retained group labels in each coefficient array's level-axis order.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Group names and the ordered labels used in coefficient arrays.
        """
        return {
            group: levels
            for _, _, predictor in self._predictor_items()
            for group, levels in predictor.group_levels.items()
        }

    def draws(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Return constrained posterior draws with separate chain and draw axes.

        Parameters
        ----------
        num_samples : int or None
            Maximum draws per MCMC chain, or fresh guide draws. Omission
            keeps all MCMC draws and draws 100 variational samples.
        seed : int
            Seed for variational draws; the caller's CPU RNG is preserved.

        Returns
        -------
        dict[str, torch.Tensor]
            Site values with shape ``(chain, draw, *site_shape)``. Guides
            produce one pseudo-chain, which supplies no MCMC diagnostics.

        Raises
        ------
        ValueError
            If the count is invalid or posterior sites mismatch the model.
        """
        return posterior_draws(self, num_samples, seed)

    def fixed_effects(
        self, *, parameter: str = "mu", num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Extract the population coefficients of one predictor.

        Parameters
        ----------
        parameter : str
            ``mu`` or a distributionally regressed auxiliary parameter.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for variational draws, restored on exit.

        Returns
        -------
        dict[str, torch.Tensor]
            Original column labels and coefficient arrays, shape
            ``(chain, draw)`` or ``(chain, draw, K-1)`` for categorical
            reference-category coefficients.
        """
        _, prefix = self._formula_predictor(parameter)
        draws = self.draws(num_samples=num_samples, seed=seed)
        return {
            name: self._at_parameter(parameter, values)
            for name, values in self._fixed_effects(draws, prefix).items()
        }

    def random_effects(
        self, *, parameter: str = "mu", num_samples: int | None = None, seed: int = 0
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Extract native group adjustments, including correlations and scales.

        Parameters
        ----------
        parameter : str
            ``mu`` or a distributionally regressed auxiliary parameter.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for variational draws, restored on exit.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Group name, then original coefficient label, mapped to arrays
            ``(chain, draw, level)`` or ``(chain, draw, level, K-1)``.
            Level order follows `group_levels`; categorical coefficients
            follow ``category_labels[1:]``.
        """

        predictor, prefix = self._formula_predictor(parameter)
        shape = draw_count(self.posterior, num_samples)
        if not predictor.random_terms:
            return {}
        draws = self.draws(num_samples=num_samples, seed=seed)
        return {
            group: {
                name: self._at_parameter(parameter, values)
                for name, values in coefficients.items()
            }
            for group, coefficients in self._random_effects(
                draws, shape, predictor, prefix
            ).items()
        }

    def coefficients(
        self, *, parameter: str = "mu", num_samples: int | None = None, seed: int = 0
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Combine population coefficients and group adjustments by level.

        Parameters
        ----------
        parameter : str
            ``mu`` or a distributionally regressed auxiliary parameter.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for variational draws, restored on exit.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Arrays with the structure of `random_effects`, now including
            population coefficients. A group-only slope has population
            coefficient zero; fixed-only coefficients repeat over levels.
        """

        predictor, prefix = self._formula_predictor(parameter)
        shape = draw_count(self.posterior, num_samples)
        if not predictor.random_terms:
            return {}
        draws = self.draws(num_samples=num_samples, seed=seed)
        fixed = self._fixed_effects(draws, prefix)
        varying = self._random_effects(draws, shape, predictor, prefix)
        for group, coefficients in varying.items():
            for name, coefficient in fixed.items():
                population = coefficient.unsqueeze(2)
                coefficients[name] = (
                    coefficients[name] + population
                    if name in coefficients
                    else population.expand(
                        *population.shape[:2],
                        len(predictor.group_levels[group]),
                        *population.shape[3:],
                    )
                )
        return {
            group: {
                name: self._at_parameter(parameter, values)
                for name, values in coefficients.items()
            }
            for group, coefficients in varying.items()
        }

    def posterior_thresholds(
        self,
        *,
        new_data: IntoDataFrame | None = None,
        predictor_data: torch.Tensor | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Replay complete ordinal threshold profiles at each requested row.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Fresh predictors evaluated with fitted encodings, or training rows.
        predictor_data : torch.Tensor or None
            Explicit neural input for fresh prediction rows when required.
        re_formula : str or None
            Varying-effect selector, with the same meaning as `posterior_epred`.
        allow_new_levels : bool
            Permit unseen grouping identities under the selected policy.
        sample_new_levels : str
            Joint donor or Gaussian policy. Explicit ``feasible_`` policies
            condition new category-specific effects on ordered row support.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for separate posterior and group-effect random streams.

        Returns
        -------
        torch.Tensor
            Effective thresholds with shape ``(chain, draw, row, threshold)``
            and axis labels `threshold_labels`. They include centered group
            shapes, selected varying location/slope shifts, and category-specific
            shifts. The proportional population predictor, its offsets, neural
            contribution, and discrimination are excluded. With threshold
            groups (``thres(n, gr)``) a row holds its group's thresholds and
            NaN at the positions its group lacks. Cumulative profiles
            are strictly ordered at every row.

        Raises
        ------
        ValueError
            If the family is not ordinal or a requested profile crosses.
        """
        from quivers.formulas._prediction import prepare_prediction

        if self.family.ordinal is None:
            raise ValueError("posterior_thresholds requires an ordinal response family")
        context = prepare_prediction(
            self,
            new_data,
            predictor_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        width = len(self.threshold_labels)
        plan = self.prediction_plan
        shifts = tuple(
            contribution.name
            for predictor in (plan.predictors if plan is not None else ())
            if not predictor.prefix
            for term in predictor.part(None)
            for contribution in term.contributions()
            if contribution.varying
        )

        def values():
            """Yield retained row values through the shared posterior replay.

            Yields
            ------
            torch.Tensor
                One checked row/event value per retained chain and draw.
            """
            for recorded in replay(self, draws, shape, context=context):
                if width == 0:
                    yield torch.empty(context.rows, 0)
                    continue
                cuts = row_binding(
                    recorded.sites[
                        "row_cutpoints"
                        if self.thresholds_by
                        or self.formula.modifiers.threshold_groups is not None
                        else "cutpoints"
                    ].value,
                    context.rows,
                    width,
                )
                threshold_groups = self.formula.modifiers.threshold_groups
                if threshold_groups is not None:
                    # brms thres(n, gr): a row has only its group's thresholds.
                    cuts = absent_thresholds(
                        cuts,
                        threshold_groups,
                        context.observations[threshold_groups.column.qvr_name],
                    )
                if "eta_nonbaseline" in recorded.sites:
                    cuts = cuts - row_binding(
                        recorded.sites["eta_nonbaseline"].value, context.rows, width
                    )
                for name in shifts:
                    cuts = cuts - row_binding(
                        recorded.sites[name].value, context.rows
                    ).unsqueeze(-1)
                yield cuts

        return collect_draws(values(), shape)

    def posterior_linpred(
        self,
        *,
        parameter: str = "mu",
        transform: bool = False,
        new_data: IntoDataFrame | None = None,
        predictor_data: torch.Tensor | None = None,
        trials: int | str | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        incl_autocor: bool = True,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> torch.Tensor:
        """Replay a fitted linear predictor on training or fresh predictor rows.

        Parameters
        ----------
        parameter : str
            ``mu`` or an auxiliary parameter of the family. An auxiliary
            parameter without a formula is constant over the rows: its
            draws are repeated for every row on the parameter's own scale
            whatever ``transform`` is, as brms does.
        transform : bool
            Return the native inverse-link binding instead of its linear
            predictor. For binomial data this is a probability, not a count.
            Ordinal probabilities are supplied by `posterior_epred`.
        new_data : IntoDataFrame or None
            Fresh predictor rows evaluated with fitted transforms, contrasts,
            and known group levels. None reuses the retained training rows.
            Response and likelihood-weight columns are not required.
        predictor_data : torch.Tensor or None
            Explicit input for a neural predictor on new rows. Required for
            neural fits with new_data; otherwise must be None.
        trials : int or str or None
            Response trial count or column for prediction rows. Vector count
            fits infer training totals; fresh rows require explicit totals
            unless a retained trial column supplies them. Linear predictors
            do not require totals.
        re_formula : str or None
            None includes all fitted varying effects; ``"~0"`` excludes
            them; a one-sided group formula selects fitted slopes. This
            sets excluded contributions to zero; it does not marginalize them.
        allow_new_levels : bool
            Permit unseen or missing grouping identities in new_data.
            Repeated missing values share one new identity per group.
            Ordinary categorical predictors still require fitted levels.
        sample_new_levels : str
            For each new identity, uncertainty chooses one fitted donor
            jointly across slopes per posterior draw; old_levels retains
            one donor across all chains and draws; gaussian samples the
            fitted conditional zero-mean group law. Existing groups retain
            their posterior effects. Group and response noise use distinct
            random streams, and the caller's CPU RNG is restored. Explicit
            feasible_uncertainty and feasible_gaussian policies condition joint
            new-group proposals on cumulative row-order support; at most 1024
            attempts are made per posterior draw, with an error on failure.
            Fitted draws and cutpoints are retained.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for variational draws, restored on exit.
        incl_autocor : bool
            Include an autocorrelation term (brms ``incl_autocor``). False
            treats rows as independent with the fitted residual scale.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        torch.Tensor
            Shape ``(chain, draw, row)`` or ``(chain, draw, row, category)``
            for categorical logits/probabilities, including reference zero.

        Raises
        ------
        ValueError
            If a parameter has no fitted predictor or an ordinal inverse
            link is requested without its thresholds.
        """
        from quivers.formulas._prediction import prepare_prediction

        binding = self._linpred_binding(parameter, transform)
        constant = self._constant_parameter(parameter)
        context = prepare_prediction(
            self,
            new_data,
            predictor_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            trials=trials,
            require_trials=False,
            incl_autocor=incl_autocor,
            data2=data2,
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows

        def values():
            """Yield retained row values through the shared posterior replay.

            Yields
            ------
            torch.Tensor
                One checked row/event value per retained chain and draw.
            """
            for recorded in replay(self, draws, shape, context=context):
                if constant:
                    fixed = recorded.sites[parameter].value
                    if fixed.numel() != 1:
                        raise ValueError(
                            f"parameter {parameter!r} has no fitted predictor "
                            "and is not one scalar per draw"
                        )
                    yield fixed.reshape(()).expand(rows)
                    continue

                yield binding.read(
                    self._row_reader(parameter, binding, recorded, lambda name: name),
                    rows,
                )

        return collect_draws(values(), shape)

    def posterior_epred(
        self,
        *,
        new_data: IntoDataFrame | None = None,
        predictor_data: torch.Tensor | None = None,
        trials: int | str | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        num_samples: int | None = None,
        mc_samples: int | None = None,
        component: Literal["response", "decision"] = "response",
        seed: int = 0,
        incl_autocor: bool = True,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> torch.Tensor:
        """Evaluate native expected responses on training or fresh rows.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Fresh predictor rows using fitted encodings and known levels,
            or None for training rows. No response or weight column is needed.
        predictor_data : torch.Tensor or None
            Explicit neural input for new_data, required for neural fits.
        trials : int or str or None
            Response trial count or column for prediction rows. Vector count
            fits infer training totals; fresh rows require explicit totals
            unless a retained trial column supplies them. Linear predictors
            do not require totals.
        re_formula : str or None
            None includes all fitted varying effects; ``"~0"`` excludes
            them; a one-sided group formula selects fitted slopes. This
            sets excluded contributions to zero; it does not marginalize them.
        allow_new_levels : bool
            Permit unseen or missing grouping identities in new_data.
            Repeated missing values share one new identity per group.
            Ordinary categorical predictors still require fitted levels.
        sample_new_levels : str
            For each new identity, uncertainty chooses one fitted donor
            jointly across slopes per posterior draw; old_levels retains
            one donor across all chains and draws; gaussian samples the
            fitted conditional zero-mean group law. Existing groups retain
            their posterior effects. Group and response noise use distinct
            random streams, and the caller's CPU RNG is restored. Explicit
            feasible_uncertainty and feasible_gaussian policies condition joint
            new-group proposals on cumulative row-order support; at most 1024
            attempts are made per posterior draw, with an error on failure.
            Fitted draws and cutpoints are retained.
        num_samples : int or None
            Draw count under the convention of `draws`.
        mc_samples : int or None
            Explicit Monte Carlo budget for bounded or certified finite-mean
            response laws without a closed mean. Sampling uses chunks of at most
            64; None requires an analytic expectation. Logistic-normal and Cox
            response means require a budget.
        component : {"response", "decision"}
            Wiener response means are positive reaction times, including delay.
            Decision returns lower/upper probabilities for a decision response.
        seed : int
            Seed for variational draws, restored on exit.
        incl_autocor : bool
            Include an autocorrelation term (brms ``incl_autocor``). False
            treats rows as independent with the fitted residual scale.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        torch.Tensor
            Native response means, shape ``(chain, draw, row)``. Binomial
            results are expected counts; mixtures include their weights;
            asymmetric Laplace results are means, not modeled quantiles.
            Ordinal and categorical responses instead return probabilities,
            ``(chain, draw, row, category)``, in `category_labels` order.
            Decision probabilities use a final axis ordered lower, upper.

        Raises
        ------
        ValueError
            If a draw has an undefined or nonfinite response mean.
        TypeError
            If the native family exposes no expectation, or a circular
            response would require a specified angular convention.
        """
        from quivers.formulas._prediction import prepare_prediction

        if component not in {"response", "decision"}:
            raise ValueError("expected response component must be response or decision")
        if component == "decision" and self.formula.modifiers.decision is None:
            raise ValueError(
                "decision probabilities require a decision response modifier"
            )
        if self.family.circular:
            raise TypeError(
                "a circular response expectation requires an angular convention"
            )
        context = prepare_prediction(
            self,
            new_data,
            predictor_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            trials=trials,
            incl_autocor=incl_autocor,
            data2=data2,
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows
        response = _qvr_name(self.formula.response_name)
        categories = len(self.category_labels) if self.family.category_response else 0
        if component == "decision":
            categories = 2

        def values():
            """Yield retained row values through the shared posterior replay.

            Yields
            ------
            torch.Tensor
                One checked row/event value per retained chain and draw.

            Raises
            ------
            TypeError
                If the selected response has no requested native operation.
            """
            for recorded in replay(self, draws, shape, context=context):
                distribution = response_distribution(recorded, response)
                value = expected_response(
                    distribution,
                    self.family,
                    categories=categories,
                    decision=component == "decision",
                    mc_samples=mc_samples,
                )
                yield row_binding(
                    value, rows, self.formula.response_layout.event_shape or categories
                )

        with response_rng(seed + 2, (*draws.values(), *context.observations.values())):
            try:
                return collect_draws(values(), shape)
            except TypeError as error:
                if self.family.baseline is not None:
                    # brms: "Cannot compute expected values of the posterior
                    # predictive distribution for family 'cox'."
                    error.add_note(
                        "brms refuses posterior_epred for family "
                        f"{self.family.name!r} as well: "
                        "the baseline hazard is a spline on the observed time "
                        "range, which fixes no mean survival time. Use "
                        "posterior_linpred for the log hazard ratio."
                    )
                raise

    def bayes_r2(
        self,
        *,
        method: Literal["auto", "model", "residual"] = "auto",
        observed: torch.Tensor | None = None,
        **options: Unpack[ExpectationOptions],
    ) -> torch.Tensor:
        """Compute per-draw explained variance on the response scale.

        Parameters
        ----------
        method : {"auto", "model", "residual"}
            Model-variance ratios use native conditional means and variances;
            residual ratios use actual minus expected responses. Auto selects
            model variances for ordinary Gaussian and Bernoulli responses,
            and residual variances otherwise. Explicit observations select
            residual variances under auto.
        observed : torch.Tensor or None
            Explicit actual responses on row/event axes for the residual
            method. Otherwise use retained training responses or evaluate
            their fitted transform in new_data. Censoring bounds cannot stand
            in for actual responses.
        **options
            Shared posterior expectation options, including new_data, group
            policies, trials, draw counts, and seed. Model variances require
            analytic native moments; mc_samples applies to residual means.

        Returns
        -------
        torch.Tensor
            Ratios on chain, draw, and event axes. Ordinal residual ratios
            use ordered integer scores. Unordered categories require a
            separately specified response metric. Zero total variance is NaN.
        """

        def laws() -> tuple[Generator[Distribution, None, None], tuple[int, int], int]:
            context = prepare_prediction(
                self,
                options.get("new_data"),
                options.get("predictor_data"),
                re_formula=options.get("re_formula"),
                allow_new_levels=options.get("allow_new_levels", False),
                sample_new_levels=options.get("sample_new_levels", "uncertainty"),
                seed=options.get("seed", 0),
                trials=options.get("trials"),
            )
            shape = draw_count(self.posterior, options.get("num_samples"))
            draws = self.draws(
                num_samples=options.get("num_samples"), seed=options.get("seed", 0)
            )
            response = _qvr_name(self.formula.response_name)
            return (
                (
                    response_distribution(recorded, response)
                    for recorded in replay(self, draws, shape, context=context)
                ),
                shape,
                context.rows,
            )

        return formula_r2(
            self,
            method,
            observed,
            new_data=options.get("new_data"),
            mc_samples=options.get("mc_samples"),
            expected=lambda: self.posterior_epred(**options),
            laws=laws,
        )

    def posterior_choice_probs(
        self, **options: Unpack[ExpectationOptions]
    ) -> torch.Tensor:
        """Return joint-model lower/upper decision probabilities.

        Parameters
        ----------
        **options
            Shared posterior expectation options, including new_data, varying
            effect selection, draw count, and seed.

        Returns
        -------
        torch.Tensor
            Probabilities shaped chain x draw x row x (lower, upper).
        """
        return self.posterior_epred(component="decision", **options)

    # mixtures: brms pp_mixture.
    def pp_mixture(
        self,
        *,
        new_data: IntoDataFrame | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        log: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Posterior probabilities of mixture component membership, brms ``pp_mixture``.

        ``P(K = k | y_i) = theta_k p_k(y_i) / sum_j theta_j p_j(y_i)`` at each
        posterior draw, scoring censored rows by their recorded event. Cost:
        O(S n K) for S draws, n rows and K components.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Rows with recorded responses and any censoring columns; None
            scores the training rows.
        re_formula : str or None
            Varying-effect selector, as for prediction.
        allow_new_levels : bool
            Permit unseen grouping levels in ``new_data``.
        sample_new_levels : str
            Policy for unseen grouping levels.
        log : bool
            Return log probabilities.
        num_samples : int or None
            Draw count under the convention of `draws`.
        seed : int
            Seed for variational draws and new-level policies.

        Returns
        -------
        torch.Tensor
            Shape ``(chain, draw, row, component)``.

        Raises
        ------
        ValueError
            If the fit is not a mixture model.
        """
        return pp_mixture(
            self,
            new_data=new_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            log=log,
            num_samples=num_samples,
            seed=seed,
        )

    def posterior_response(self, **options: Unpack[PredictionOptions]) -> ResponseDraws:
        """Draw responses in original measurement coordinates in one replay.

        Parameters
        ----------
        **options
            Shared posterior_predict options, including new_data, varying
            effect selection, draw count, and seed.

        Returns
        -------
        ResponseDraws
            Positive reaction time and its joint binary decision for Wiener;
            ordinary response values with no decision for other families.
        """
        from quivers.formulas._response_modifiers import decode_response_draws

        return decode_response_draws(self.formula, self.posterior_predict(**options))

    def hypothesis(
        self,
        expression: str | ContrastExpression | Sequence[str] | Mapping[str, str],
        *,
        class_: str | None = "b",
        parameter: str = "mu",
        group: str | None = None,
        scope: Literal["standard", "ranef", "coef"] = "standard",
        prior: bool = False,
        num_samples: int | None = None,
        num_prior_samples: int | None = None,
        seed: int = 0,
        null: float | int | torch.Tensor = 0.0,
        probs: Sequence[float] = (0.025, 0.975),
        robust: bool = False,
        point: PointNullSpec | None = None,
        name: str = "contrast",
        alpha: float = 0.05,
    ) -> HypothesisSummary | HypothesisTable:
        """Test a brms hypothesis string, or evaluate a checked contrast.

        A string of the form ``"left (= | < | >) right"``, or several such
        strings, is a brms hypothesis: every name is prefixed with the
        class (``"x > 0"`` tests ``b_x``, and with ``class_="sd"`` and
        ``group="g"`` the name ``Intercept`` is ``sd_g__Intercept``), the
        sides are R expressions (``^``, ``exp``), and the result is brms's
        table. ``class_=None`` reads names as `variables` reports them.

        Parameters
        ----------
        expression : str or ContrastExpression or Sequence[str] or Mapping[str, str]
            brms hypothesis strings (a mapping names them), or a QVR value
            expression over authored labels, with optional aliases.
        class_ : str or None
            Parameter role such as b, Intercept, sd, cor, or an auxiliary name.
            The b role includes population intercepts. None selects exact
            constrained latent site names.
        parameter : str
            Distributional or nonlinear predictor for b, Intercept, and sd.
        group : str or None
            Grouping factor for scale/correlation filtering or per-level scopes.
        scope : {"standard", "ranef", "coef"}
            Named parameters, per-level random adjustments, or per-level
            population plus random coefficients. The latter two require b/group.
        prior : bool
            Also evaluate the same contrast on the corresponding joint prior
            stream, yielding prior-corrected region evidence.
        num_samples, num_prior_samples : int or None
            Posterior and prior counts. None follows retained-draw conventions;
            retained prior samples are reused when available.
        seed : int
            Isolated seed for guide or fresh prior draws.
        null : float, int, or torch.Tensor
            Finite comparison value on event axes only.
        probs : Sequence[float]
            Quantile probabilities, in requested order.
        robust : bool
            Return median/scaled MAD rather than mean/sample SD.
        point : PointNullSpec or None
            Optional marginal point-density comparison using these same joint
            contrast streams. Requires prior=True and a checked Real result.
        name : str
            Output contrast label.
        alpha : float
            brms ``alpha`` of a hypothesis string: the interval is
            ``alpha`` to ``1 - alpha`` for a one-sided hypothesis and
            ``alpha / 2`` to ``1 - alpha / 2`` for a two-sided one.

        Returns
        -------
        HypothesisSummary or HypothesisTable
            For a brms hypothesis string, brms's table (``Estimate``,
            ``Est.Error``, ``CI.Lower``, ``CI.Upper``, ``Evid.Ratio``,
            ``Post.Prob``, ``Star``); its evidence ratio of a point
            hypothesis needs prior draws (``sample_prior`` or
            ``prior=True``). For a contrast expression,
            chain/draw/event-aware uncertainty and strict-region evidence.
            Posterior odds and prior-corrected region Bayes factors are distinct;
            equality frequencies do not estimate a point-null Bayes factor.
            Optional point_evidence uses the shared density estimator and retains
            its bandwidth and explicit assumption declarations.

        Raises
        ------
        ValueError
            If a hypothesis string is malformed or names a parameter that
            is not in the model (the message lists the class's parameters),
            or the contrast selectors match nothing.
        """
        from quivers.formulas._hypothesis import _formula_hypothesis

        if is_brms_hypothesis(expression):
            return fit_hypothesis(
                self,
                cast("HypothesisInput", expression),
                class_=class_,
                group=group,
                scope=scope,
                alpha=alpha,
                robust=robust,
                prior=prior,
                num_samples=num_samples,
                num_prior_samples=num_prior_samples,
                seed=seed,
            )
        return _formula_hypothesis(
            self,
            expression,
            class_=class_,
            parameter=parameter,
            group=group,
            scope=scope,
            prior=prior,
            num_samples=num_samples,
            num_prior_samples=num_prior_samples,
            seed=seed,
            null=null,
            probs=probs,
            robust=robust,
            point=point,
            name=name,
        )

    def analysis(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> PosteriorAnalysis:
        """Prepare reusable posterior summaries and predictive comparisons.

        Parameters
        ----------
        num_samples : int or None
            Maximum MCMC draws per chain or fresh variational draw count.
        seed : int
            Variational seed; the caller's CPU RNG is preserved.

        Returns
        -------
        quivers.diagnostics.PosteriorAnalysis
            Retained draws, raw training-row likelihoods, response data,
            and actual sampler statistics. Later analysis reuses these tensors.
        """
        from quivers.diagnostics.posterior import PosteriorAnalysis, _sample_stats

        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        response = _qvr_name(self.formula.response_name)
        with torch.no_grad():
            likelihood = pointwise_scores(
                replay(self, draws, shape, score_observations=True),
                {response},
                {response} if self.residual is not None else set(),
                shape,
            )
        return PosteriorAnalysis(
            samples=draws,
            log_likelihood=likelihood,
            observed_data={response: response_tensor(self.formula, self.family)},
            sample_stats=_sample_stats(self.posterior, shape[1]),
            kind=draw_provenance(self.posterior),
        )

    def log_likelihood(
        self,
        *,
        new_data: IntoDataFrame | None = None,
        predictor_data: torch.Tensor | None = None,
        trials: int | str | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        incl_autocor: bool = True,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate raw response log likelihoods on fitted or recorded fresh rows.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Fresh predictors, outcomes, and authored observation-event columns.
            None scores the fitted rows. Fitting weight columns are not required.
        predictor_data : torch.Tensor or None
            Neural inputs aligned with fresh rows before any subset selection.
        trials : int or str or None
            Existing count-trial override. Fresh vector counts otherwise infer
            their totals from their recorded components.
        re_formula : str or None
            Existing varying-effect selector; ``~0`` excludes varying effects.
        allow_new_levels : bool
            Permit fresh grouping identities under the selected prediction policy.
        sample_new_levels : str
            Existing joint donor or conditional Gaussian group-effect policy.
        num_samples : int or None
            Maximum retained draws per chain, or fresh variational draws.
        seed : int
            Seed for variational draws; the caller's CPU RNG is preserved.
        incl_autocor : bool
            Include an autocorrelation term (brms ``incl_autocor``). False
            treats rows as independent with the fitted residual scale.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        dict[str, torch.Tensor]
            Response name to raw scores with shape ``(chain, draw, row)``.
            Likelihood weights are excluded; event coordinates are reduced.

        Raises
        ------
        TypeError
            If a hidden response needs a marginal-likelihood observer.
        """

        context = None
        if new_data is not None:
            from quivers.formulas._likelihood import prepare_likelihood

            context = prepare_likelihood(
                self,
                new_data,
                predictor_data,
                trials=trials,
                re_formula=re_formula,
                allow_new_levels=allow_new_levels,
                sample_new_levels=sample_new_levels,
                seed=seed,
                incl_autocor=incl_autocor,
                data2=data2,
            )
        else:
            from quivers.formulas._prediction import prepare_prediction

            if trials is not None:
                if self.family.trials_parameter is None:
                    raise ValueError(
                        "trial overrides require a count family with a trial parameter"
                    )
                _response_trials(
                    self.formula,
                    self.prediction_plan.designs["mu"].data
                    if self.prediction_plan is not None
                    else None,
                    trials,
                    caller="log_likelihood",
                )
            context = prepare_prediction(
                self,
                None,
                predictor_data,
                trials=trials,
                re_formula=re_formula,
                allow_new_levels=allow_new_levels,
                sample_new_levels=sample_new_levels,
                seed=seed,
                incl_autocor=incl_autocor,
                data2=data2,
            )
            if trials is not None:
                name = response_input_name(self.formula)
                context = context.with_(
                    observations=dict(context.observations)
                    | {
                        name: self.observations[name],
                    }
                )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        response = _qvr_name(self.formula.response_name)
        with torch.no_grad():
            # Correlated residuals score brms's leave-one-out conditionals.
            return pointwise_scores(
                replay(self, draws, shape, context=context, score_observations=True),
                {response},
                {response} if self.residual is not None else set(),
                shape,
            )

    def posterior_predict(
        self,
        *,
        new_data: IntoDataFrame | None = None,
        predictor_data: torch.Tensor | None = None,
        trials: int | str | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: Literal[
            "uncertainty",
            "old_levels",
            "gaussian",
            "feasible_uncertainty",
            "feasible_gaussian",
        ] = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        incl_autocor: bool = True,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> torch.Tensor:
        """Sample native response distributions on training or fresh rows.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Fresh predictor rows using fitted encodings and known levels,
            or None for training rows. No response or weight column is needed.
        predictor_data : torch.Tensor or None
            Explicit neural input for new_data, required for neural fits.
        trials : int or str or None
            Response trial count or column for prediction rows. Vector count
            fits infer training totals; fresh rows require explicit totals
            unless a retained trial column supplies them. Linear predictors
            do not require totals.
        re_formula : str or None
            None includes all fitted varying effects; ``"~0"`` excludes
            them; a one-sided group formula selects fitted slopes. This
            sets excluded contributions to zero; it does not marginalize them.
        allow_new_levels : bool
            Permit unseen or missing grouping identities in new_data.
            Repeated missing values share one new identity per group.
            Ordinary categorical predictors still require fitted levels.
        sample_new_levels : str
            For each new identity, uncertainty chooses one fitted donor
            jointly across slopes per posterior draw; old_levels retains
            one donor across all chains and draws; gaussian samples the
            fitted conditional zero-mean group law. Existing groups retain
            their posterior effects. Group and response noise use distinct
            random streams, and the caller's CPU RNG is restored. Explicit
            feasible_uncertainty and feasible_gaussian policies condition joint
            new-group proposals on cumulative row-order support; at most 1024
            attempts are made per posterior draw, with an error on failure.
            Fitted draws and cutpoints are retained.
        num_samples : int or None
            Maximum draws per MCMC chain, or fresh guide draws, following
            `draws`. Each posterior draw supplies one response per row.
        seed : int
            Seed for separate latent and response random streams. The
            caller's CPU RNG state is restored on return or failure.
        incl_autocor : bool
            Include an autocorrelation term (brms ``incl_autocor``). False
            treats rows as independent with the fitted residual scale.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        torch.Tensor
            Predictive responses, shape ``(chain, draw, row, *event)``. Categorical
            and ordinal responses contain integer codes in `category_labels`;
            binomial responses contain counts. Likelihood weights do not
            alter the response sampling law at a fixed posterior draw.
            Wiener draws retain their joint signed representation: positive
            upper-boundary and negative lower-boundary reaction times. Use
            `posterior_response` for positive times and their joint decisions.

        Raises
        ------
        ValueError
            If the requested count or retained posterior layout is invalid.
        TypeError
            If the response has no native sampling operation.
        """
        from quivers.formulas._prediction import prepare_prediction
        from quivers.inference._response import response_sample

        context = prepare_prediction(
            self,
            new_data,
            predictor_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            trials=trials,
            incl_autocor=incl_autocor,
            data2=data2,
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows
        response = _qvr_name(self.formula.response_name)

        def values():
            """Yield retained row values through the shared posterior replay.

            Yields
            ------
            torch.Tensor
                One checked row/event value per retained chain and draw.

            Raises
            ------
            TypeError
                If the selected response has no requested native operation.
            """
            for recorded in replay(self, draws, shape, context=context):
                yield row_binding(
                    response_sample(recorded, response),
                    rows,
                    self.formula.response_layout.event_shape,
                )

        # Guide sampling uses its own seed and restores the caller's RNG.
        # A distinct stream avoids coupling response noise to guide noise.
        with (
            response_rng(seed + 1, (*draws.values(), *context.observations.values())),
            torch.no_grad(),
        ):
            return collect_draws(values(), shape)

    def predictor_labels(self, parameter: str = "mu") -> tuple[int | str, ...]:
        """Label the last axis returned by posterior_linpred for one graph output.

        Parameters
        ----------
        parameter : str
            Family parameter or named nonlinear intermediate.

        Returns
        -------
        tuple[int or str, ...]
            Output coordinate order; empty for scalar-row values. Family
            coordinates use fitted response order, with the ALR reference
            omitted. Intermediate vectors retain authored label order.

        Raises
        ------
        ValueError
            If the parameter has no retained predictor or graph binding.
        """
        from quivers.formulas._predictor_graph import ComponentExpression

        graph = self.predictor_graph
        if graph is not None and parameter in graph.expressions:
            value = graph.expressions[parameter]
            return value.components if isinstance(value, ComponentExpression) else ()
        if graph is not None and parameter in graph.leaves:
            return ()
        response = self.family.response
        # mixtures: mu stacks the component locations mu1, ..., muK.
        if parameter == "mu" and self.family.mixture is not None:
            return tuple(
                f"mu{index}"
                for index in range(1, mixture_component_count(self.family) + 1)
            )
        if parameter == "mu":
            if response is None:
                return ()
            labels = self.category_labels
            if response.kind == "logistic_normal":
                return tuple(
                    labels[index]
                    for index in response.coordinate_indices(
                        tuple(str(label) for label in labels)
                    )
                )
            return labels
        if graph is not None and parameter in graph.roots:
            value = graph.roots[parameter]
            if not isinstance(value, ComponentExpression):
                return ()
            assert response is not None
            labels = self.category_labels
            return tuple(
                labels[index]
                for index in response.coordinate_indices(
                    tuple(str(label) for label in labels)
                )
            )
        if self._constant_parameter(parameter):
            return ()
        self._formula_predictor(parameter)
        return (
            self.coefficient_labels
            if self.family.component_scale is not None
            and parameter == self.family.component_scale.name
            else ()
        )

    def _predictor_width(self, parameter: str) -> int:
        """Read the declared coordinate width of a fitted predictor.

        Parameters
        ----------
        parameter : str
            Location or auxiliary parameter name.

        Returns
        -------
        int
            Component width, or zero for a scalar predictor.
        """
        if self.predictor_graph is not None:
            from quivers.formulas._predictor_graph import ComponentExpression

            value = self.predictor_graph.expressions.get(parameter)
            if value is None:
                value = self.predictor_graph.roots.get(parameter)
            if parameter != "mu" and parameter in (
                self.predictor_graph.leaves.keys()
                | self.predictor_graph.expressions.keys()
                | self.predictor_graph.roots.keys()
            ):
                return (
                    len(value.components)
                    if isinstance(value, ComponentExpression)
                    else 0
                )
        response = self.family.response
        # mixtures: mu stacks the component locations mu1, ..., muK.
        if parameter == "mu" and self.family.mixture is not None:
            return mixture_component_count(self.family)
        if response is None:
            return 0
        if parameter == "mu":
            return len(self.category_labels) - int(response.kind == "logistic_normal")
        return (
            len(self.category_labels) - 1
            if self.family.component_scale is not None
            and parameter == self.family.component_scale.name
            else 0
        )

    @property
    def coefficient_labels(self) -> tuple[int | str, ...]:
        """Label free component coefficients in their retained coordinate order.

        Returns
        -------
        tuple[int | str, ...]
            Nonreference component labels, or all labels without a reference.
        """
        response = self.family.response
        if response is None:
            return ()
        labels = self.category_labels
        return tuple(
            labels[index]
            for index in response.coordinate_indices(tuple(str(v) for v in labels))
        )

    def _predictor_items(self):
        """Enumerate retained additive designs and their native coefficient prefixes.

        Returns
        -------
        tuple
            Public parameter labels, stable namespaces, and fitted designs.
        """
        return predictor_items(self.formula, self.distributional, self.predictor_graph)

    def _constant_parameter(self, parameter: str) -> bool:
        """Report whether an auxiliary parameter has no predictor of its own.

        Parameters
        ----------
        parameter : str
            Parameter name.

        Returns
        -------
        bool
            True for an auxiliary parameter of the family that neither a
            distributional formula nor the predictor graph predicts, so its
            value is constant over the rows.
        """
        graph = self.predictor_graph
        return (
            parameter != "mu"
            and parameter not in self.distributional
            and not (
                graph is not None
                and parameter
                in (graph.leaves.keys() | graph.expressions.keys() | graph.roots.keys())
            )
            and any(aux.name == parameter for aux in self.family.aux_params)
        )

    def _vector_coordinate(self, parameter: str) -> tuple[Predictor, int] | None:
        """Locate a parameter that is one coordinate of a vector predictor.

        Parameters
        ----------
        parameter : str
            Parameter name, for instance ``muB``.

        Returns
        -------
        tuple[Predictor, int] or None
            The predictor and the coordinate; None for any other parameter.
        """
        plan = self.prediction_plan
        for predictor in plan.predictors if plan is not None else ():
            output = predictor.output
            if (
                output is not None
                and not predictor.nonlinear
                and parameter in output.parameters
            ):
                return predictor, output.parameters.index(parameter)
        return None

    @property
    def _separable_rows(self) -> bool:
        """Whether each training row scores as a new row of the same data.

        A row's pointwise score on new data equals its training score when
        its law is independent of the other rows given the parameters and
        every term's value at the row is fixed by the parameters and the
        row's data. A residual law correlates rows or conditions on earlier
        recorded responses, and a latent drawn per training row is drawn
        again for new rows, so neither holds there.

        Returns
        -------
        bool
            True when no residual law and only separable terms enter.
        """
        plan = self.prediction_plan
        return (
            self.residual is None
            and plan is not None
            and all(
                term.separable
                for predictor in plan.predictors
                for term in predictor.terms
            )
        )

    def _row_reader(
        self,
        parameter: str,
        binding: LinpredBinding,
        recorded: Trace,
        site: Callable[[str], str],
        location: Callable[[], torch.Tensor] | None = None,
    ) -> Callable[[str], torch.Tensor]:
        """Read the row bindings of a linear predictor from one replay.

        brms's ``mu`` includes the conditional ARMA recursion of a residual
        law, which moves the location by lagged recorded residuals; the
        replayed residual law states that location.

        Parameters
        ----------
        parameter : str
            Predictor name.
        binding : LinpredBinding
            Binding holding the predictor.
        recorded : Trace
            One replayed execution.
        site : Callable[[str], str]
            Program name of a binding of this response: the identity for a
            single fit, the response namespace in a joint program.
        location : Callable[[], torch.Tensor] or None
            The response law's location, recursion included; None reads the
            law of the response's own observation.

        Returns
        -------
        Callable[[str], torch.Tensor]
            The value of a named binding of this response.
        """
        response = site(_qvr_name(self.formula.response_name))

        def read(name: str) -> torch.Tensor:
            if name == binding.site and parameter == "mu" and self.residual is not None:
                if location is not None:
                    return location()
                return linear_predictor(recorded, response)
            return recorded.sites[site(name)].value

        return read

    def _linpred_binding(self, parameter: str, transform: bool) -> LinpredBinding:
        """Choose the row binding holding a linear predictor.

        Parameters
        ----------
        parameter : str
            Predictor name.
        transform : bool
            Request the inverse-link binding.

        Returns
        -------
        LinpredBinding
            The binding, with how its values are read.

        Raises
        ------
        ValueError
            If the parameter has no predictor or an ordinal inverse link is
            requested.
        """
        if parameter == "mu" and not self.family.reads_location:
            raise ValueError(
                f"the response kernel {self.family.name!r} has no location 'mu'; "
                "name one of its parameters with a formula: "
                f"{[aux.name for aux in self.family.aux_params]}"
            )
        coordinate = self._vector_coordinate(parameter)
        if coordinate is not None and coordinate[0].prefix:
            predictor, index = coordinate
            assert predictor.output is not None
            return LinpredBinding(
                site=predictor.parameter
                if transform
                else f"{predictor.prefix}eta_nonbaseline",
                width=predictor.output.size,
                coordinate=index,
            )
        axis = (
            response_axis(
                self.family, tuple(str(label) for label in self.category_labels)
            )
            if self.family.response is not None
            else None
        )
        if axis is not None and parameter in axis.parameters:
            return LinpredBinding(
                site="eta_nonbaseline",
                width=axis.size,
                coordinate=axis.parameters.index(parameter),
            )
        graph = self.predictor_graph
        graph_parameter = graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys() | graph.roots.keys()
        )
        # brms: an auxiliary parameter without a formula is constant over the
        # rows, and its draws are returned on the parameter's own scale.
        if self._constant_parameter(parameter):
            return LinpredBinding(site=parameter)
        # mixtures: parameter="mu" returns every component location.
        if not graph_parameter and not (
            parameter == "mu" and self.family.mixture is not None
        ):
            self._formula_predictor(parameter)
        ordinal = self.family.ordinal
        if transform and parameter == "mu" and ordinal is not None:
            raise ValueError(
                "ordinal probabilities depend on thresholds; use posterior_epred"
            )
        retained_log_rate = self.family.retains_log_location and parameter == "mu"
        site = (
            ("mu" if transform and not retained_log_rate else "eta")
            if parameter == "mu"
            else (parameter if transform else f"dpar_{parameter}_eta")
        )
        if graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys()
        ):
            site = f"nlpar_{parameter}"
        specific = parameter == "mu" and (self.formula.category_specific)
        width = self._predictor_width(parameter)
        if specific:
            width = (
                len(self.category_labels)
                - 1
                - int(ordinal is not None and ordinal.kind == "hurdle_cumulative")
            )
        return LinpredBinding(
            site=site,
            width=width,
            exponentiate=retained_log_rate and transform,
            specific="eta_nonbaseline" if specific else None,
        )

    def _formula_predictor(self, parameter: str) -> tuple[Formula, str]:
        """Read retained predictor metadata without re-evaluating its formula.

        Parameters
        ----------
        parameter : str
            The location parameter mu or an auxiliary with a retained formula.

        Returns
        -------
        tuple[Formula, str]
            The evaluated predictor and its native coefficient-name prefix.

        Raises
        ------
        ValueError
            If the parameter has no retained distributional formula.
        """
        if (
            self.predictor_graph is not None
            and parameter in self.predictor_graph.leaves
        ):
            return self.predictor_graph.leaves[parameter], f"nlpar_{parameter}_"
        # mixtures: coefficients belong to the component locations mu<k>.
        if parameter == "mu" and self.family.mixture is not None:
            raise ValueError(
                "a mixture has no single location predictor; select a component "
                f"location {self.predictor_labels('mu')}"
            )
        if parameter == "mu":
            return self.formula, ""
        coordinate = self._vector_coordinate(parameter)
        if coordinate is not None:
            return self._formula_predictor(coordinate[0].parameter)
        if parameter not in self.distributional:
            raise ValueError(
                f"parameter {parameter!r} has no fitted distributional predictor"
            )
        return self.distributional[parameter], f"dpar_{parameter}_"

    def _at_parameter(self, parameter: str, values: torch.Tensor) -> torch.Tensor:
        """Select a parameter's coordinate of coefficients over a vector.

        Parameters
        ----------
        parameter : str
            Parameter name.
        values : torch.Tensor
            Coefficients of its predictor.

        Returns
        -------
        torch.Tensor
            The coefficients of the coordinate, or every coefficient for a
            parameter that is no coordinate.
        """
        coordinate = self._vector_coordinate(parameter)
        return values if coordinate is None else values[..., coordinate[1]]

    def _fixed_effects(
        self, draws: Mapping[str, torch.Tensor], prefix: str, namespace: str = ""
    ) -> dict[str, torch.Tensor]:
        """Label the population coefficients of one predictor.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Native site draws on chain and draw axes.
        prefix : str
            Namespace of the predictor in its own model.
        namespace : str
            Namespace of the model in the draws.

        Returns
        -------
        dict[str, torch.Tensor]
            Design-column labels mapped to coefficient draws.

        Raises
        ------
        ValueError
            If retained term metadata is unavailable.
        """
        if self.prediction_plan is None:
            raise ValueError(
                "coefficient extraction requires retained emission metadata"
            )
        return {
            coefficient.label: draws[namespace + coefficient.site]
            for predictor in self.prediction_plan.predictors
            if predictor.prefix == prefix
            for term in predictor.terms
            for coefficient in term.population()
        }

    def _random_effects(
        self,
        draws: dict[str, torch.Tensor],
        shape: tuple[int, int],
        predictor: Formula,
        prefix: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Read retained native per-level coefficient bindings at each draw.

        Parameters
        ----------
        draws : dict[str, torch.Tensor]
            Native site draws on chain and draw axes.
        shape : tuple[int, int]
            The chain count and draw count.
        predictor : Formula
            The retained group-column metadata and level coding.
        prefix : str
            Native coefficient-name prefix for the selected parameter.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Group names and coefficient labels mapped to per-level draws.

        Raises
        ------
        ValueError
            If retained group-layout metadata is unavailable.
        """
        if self.prediction_plan is None:
            raise ValueError(
                "group coefficient extraction requires retained emission metadata"
            )
        coefficients = group_coefficients(
            draws,
            tuple(
                law
                for law in self.prediction_plan.group_laws
                if any(member.prefix == prefix for member in law.members)
            ),
            self.group_levels,
        )
        result: dict[str, dict[str, torch.Tensor]] = {}
        for term in predictor.random_terms:
            value = coefficients[(prefix, term.group, term.slope)]
            if value.shape[:2] != shape:
                raise ValueError("group coefficients have incompatible sample axes")
            result.setdefault(term.group, {})[term.slope] = value
        return result


@overload
def fit(
    formula: MultivariateFormula,
    *,
    data: IntoDataFrame,
    **options: Unpack[JointFitOptions],
) -> _joint_fit.MultivariateFit: ...


@overload
def fit(
    formula: str | ResponseModel,
    *,
    data: IntoDataFrame,
    **options: Unpack[FormulaFitOptions],
) -> BayesianFit: ...


@overload
def fit(
    formula: None = None,
    *,
    data: IntoDataFrame,
    program: str | tuple[str | Path, str],
    response: str,
    entry: str | None = None,
    static: Mapping[str, int] | None = None,
    **options: Unpack[FormulaFitOptions],
) -> BayesianFit: ...


def fit(
    formula: str | ResponseModel | MultivariateFormula | None = None,
    *,
    data: IntoDataFrame,
    program: str | tuple[str | Path, str] | None = None,
    entry: str | None = None,
    static: Mapping[str, int] | None = None,
    parameters: Mapping[str, str | NonlinearExpression] | None = None,
    response: str | None = None,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    data2: Mapping[str, AutocorMatrix] | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    method: FitMethod = "nuts",
    num_warmup: int | None = None,
    num_samples: int | None = None,
    num_chains: int | None = None,
    num_steps: int | None = None,
    learning_rate: float = 1e-2,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    guide: type | None = None,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    seed: int = 0,
    sample_prior: PriorMode = "no",
    num_prior_samples: int = 100,
    prior_seed: int | None = None,
    snapshot_policy: SnapshotPolicy = "clone",
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
    control: SamplerControl | Mapping[str, float | int | str] | None = None,
    thin: int = 1,
    init: InitSpec | None = None,
    init_r: float = 2.0,
    algorithm: Algorithm | None = None,
    file: str | Path | None = None,
    file_refit: FileRefit = "never",
    pathfinder: PathfinderOptions | Mapping[str, float | int | bool] | None = None,
    cores: int = 1,
    knots: Knots | None = None,
    drop_unused_levels: bool = True,
) -> BayesianFit | _joint_fit.MultivariateFit:
    """Compile a brms-style formula, fit it, and return the result.

    See [`quivers.formulas`][quivers.formulas] for surface details.  This entry
    point composes `formula_from_data`, `FormulaToQVRModule`,
    `Compiler`, and the inference layer in one call.

    ``family="categorical"`` uses the sorted distinct integer labels or the
    declared factor levels and takes the first category as the default
    reference. An attached categorical predictor may return
    ``(N, K-1)`` reference logits or ``(N, K)`` full logits. For
    ``family="mixture"``, ``mixture_components`` selects any finite component
    count of at least two; the likelihood integrates assignments exactly.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula or None
        brms/lme4-style response formula, or a joint formula built with
        `bf` (``bf("mvbind(y1, y2) ~ x")`` or ``bf(...) + bf(...)``), which
        returns a `MultivariateFit`. Response-level keywords then belong
        to each `bf` call. A single-response ``bf("y ~ x", "sigma ~ z")``
        is fitted as the string formula with the response keywords it
        carries; a keyword given both there and here with different values
        raises. None for a program, given by ``program=``.
    data : IntoDataFrame
        Pandas, Polars, or another Narwhals-compatible dataframe.
    program : str or tuple[str or Path, str] or None
        QVR source, or a source file path and a program name. With source
        text alone, the one program or the one exported program is selected.
        The program's observation of ``response`` is the likelihood, and its
        domain is the rows of the data. Every sample site and input that ``parameters``
        names is replaced by a regression on the link scale its support
        requires; every other site keeps the program's own law. Data the
        program reads at each row are data columns of the same name; data on
        another axis and scalars come from ``data2``.
    entry : str or None
        The program name when it is given separately from the source text.
    static : Mapping[str, int] or None
        Finite extents of the program, by binder name.
    parameters : Mapping[str, str or NonlinearExpression] or None
        Formulas of the program's parameters, as ``{"noise": "~ 1 + (1 | g)"}``.
        A `NonlinearExpression` gives a link-scale expression and can read
        additional parameter formulas in the mapping. Priors of the program's
        parameter formulas use ``dpar=<parameter>``.
    response : str or None
        The site the program observes, a data column.
    family : str or Family
        Registered response family or a family value. ``Family.with_links``
        selects location or auxiliary inverse links.
    weights : str or None
        Numeric column of finite nonnegative likelihood weights. Values
        multiply each response's log density, retain their original scale,
        and do not weight priors or posterior predictions.
    decision : str or None
        Wiener boundary-choice column: numeric zero/lower or one/upper, or
        the labels lower and upper. Equivalent to response-side dec(column).
    nonlinear : Mapping[str, str or NonlinearExpression] or None
        Named additive parameter designs or deterministic QVR expressions.
        Supplying this mapping interprets the response RHS as a value
        expression. Defaults retain the same proper coefficient priors as
        ordinary QVR formulas; get_prior reports their exact site names.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Named matrices referenced by autocorrelation terms, as in brms:
        the spatial weights of ``sar(M)`` and the fixed covariance of
        ``fcor(M)``, each over the included response rows. A group-level
        ``gr(g, cov = M)`` reads a labeled ``DataFrame`` covariance between
        the levels of ``g``, and ``s(g, bs = "mrf", xt = list(penalty = M))``
        a penalty or neighbourhood matrix over them.
    distributional : Mapping[str, str or NonlinearExpression] or None
        Auxiliary-parameter formulas, for instance
        ``{"sigma": "~ z + (1 | g)"}``. The optional left-hand side must
        match the parameter name. Coefficient prior overrides use names
        such as ``dpar_sigma_intercept`` and ``dpar_sigma_beta_z``.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters held constant on their natural scale, for instance
        ``{"quantile": 0.9}``. A fixed parameter has no latent site or prior.
    method : {"nuts", "hmc", "svi", "fixed_param", "pathfinder"}
        Inference algorithm.
    num_warmup : int or None
        MCMC warmup iterations per chain (NUTS and HMC only); 500 when
        omitted.
    num_samples : int or None
        MCMC draws kept per chain after warmup (NUTS and HMC only); 1000
        when omitted.
    num_chains : int or None
        MCMC chains (NUTS and HMC only), or Pathfinder paths as brms's
        ``chains``; 4 when omitted.
    num_steps : int or None
        SVI optimization steps (SVI only); 1000 when omitted.
    learning_rate : float
        The Adam learning rate of the SVI optimizer (SVI only).
    fixed_prior : str
        Prior of every fixed-effect coefficient.
    random_scale_prior : str
        Prior of every random-effect standard deviation.
    correlation_prior : str
        Default prior for correlated group blocks, in LKJ Cholesky form.
        Explicit shared IDs join coefficients across distributional formulas.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known covariance between labeled levels, keyed by grouping factor.
        The same declaration serves compilation, inspection, and prediction.
    priors : PriorInput or None
        Named overrides or typed selectors, including sampled shrinkage
        hierarchies. Unknown, unused, or ambiguous rules raise ValueError.
    guide : type or None
        The SVI guide class; `AutoNormalGuide` when omitted.
    initial_values : Mapping[str, torch.Tensor or float] or None
        Natural-scale starting values keyed by latent site, including
        coefficient names. Scalars broadcast to a site's declared shape.
        Values must lie strictly inside the prior support. MCMC applies
        these after its seeded initial strategy; SVI passes them to the
        guide constructor. A custom guide must accept this keyword when
        values are supplied. For a regressed shifted-lognormal or Wiener ``ndt``
        with the log link and an intercept, default coefficient centers put the shift below
        the minimum observed response; explicit values override those centers.
        An unregressed learned Wiener delay starts at one quarter of the
        minimum reaction time, before explicit overrides.
    reparameterize : {"centered", "noncentered"}
        Parameterization of the random effects.
    binomial_trials : int or str
        Trials per row for binomial or beta-binomial responses: a constant or a column.
    thresholds_by : str or None
        Column grouping the cutpoints of an ordinal family.
    mixture_components : int
        Number of Gaussian components for the mixture family.
    predictor : nn.Module or None
        Optional differentiable contribution on the linear-predictor scale.
    predictor_data : torch.Tensor or None
        Input passed to ``predictor`` on each SVI step.
    seed : int
        Seed for compilation and inference. The caller's CPU random state
        is restored on return or failure.
    sample_prior : {"no", "yes", "only"}
        Fit the posterior, additionally retain independent prior draws, or
        disable observation likelihoods during inference. Priors, their data
        dependencies, support constraints, and explicit potentials remain.
    num_prior_samples : int
        Positive independent draw budget for mode "yes".
    prior_seed : int or None
        Isolated prior stream. None uses seed + 1 modulo 2**64. Retaining
        prior draws does not change fitted posterior draws or losses.

    snapshot_policy : {"clone", "reference"}
        Retain one independent original-data snapshot for update/refit by default.
        Reference mode shares caller data explicitly. Predictor initial states are
        copied before each refit, so fits do not share trained mutable parameters.
    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    control : SamplerControl or Mapping or None
        brms ``control``: ``adapt_delta`` (NUTS target acceptance),
        ``max_treedepth``, initial ``stepsize``, and ``metric`` (``diag_e``,
        ``dense_e``, ``unit_e``). MCMC only.
    thin : int
        Keep every ``thin``-th post-warmup transition; ``num_samples`` stays
        the retained draw count per chain. MCMC only.
    init : str or float or Mapping or Sequence or Callable or None
        brms ``init``: ``"random"`` draws unconstrained starts uniformly from
        ``(-init_r, init_r)``, ``0`` or ``"0"`` starts at zero, a mapping
        gives named natural-scale starts, a list gives one mapping per chain,
        and a function of the one-based chain index returns one. None keeps
        the driver's seeded jitter. Formula starts and ``initial_values``
        apply under every choice.
    init_r : float
        Half-width of ``init = "random"``.
    algorithm : {"sampling", "meanfield", "fullrank", "laplace", "fixed_param", "pathfinder"} or None
        brms ``algorithm``: NUTS, SVI with a mean-field, full-rank, or Laplace
        guide, the fixed-parameter sampler, which keeps every draw at the
        initial values, or multi-path Pathfinder with ``num_chains`` paths.
    file : str or Path or None
        brms ``file``: load the fit saved there, or fit and save it there.
    file_refit : {"never", "always", "on_change"}
        brms ``file_refit``: reuse a saved fit, always refit, or refit when the
        formula, data, priors, or settings differ from the saved recipe.
    pathfinder : PathfinderOptions or Mapping or None
        cmdstanr ``pathfinder`` arguments for ``algorithm="pathfinder"``
        (``draws``, ``single_path_draws``, ``history_size``,
        ``max_lbfgs_iters``, ``num_elbo_draws``, ``psis_resample``,
        ``calculate_lp``, ``save_single_paths``, ``init_alpha`` and the
        ``tol_*`` tolerances), which brms passes through its ``...``.
    cores : int
        brms ``cores``: run the MCMC chains in up to this many worker
        processes. Each worker rebuilds the fit from the retained arguments
        and data and runs its share of the chains; the draws equal those of
        the serial run with the same seed. Every argument must be picklable.
    knots : Mapping[str, Sequence[float]] or None
        brms ``knots``: knot values keyed by smooth covariate, passed to the
        ``cr``, ``cs``, ``cc``, ``ps`` and ``bs`` constructors of ``s()`` as
        mgcv's ``smoothCon(knots = )`` does.
    drop_unused_levels : bool
        brms ``drop_unused_levels``: with False, factor levels that no row
        uses keep their population columns (of zeros) and grouping-factor
        levels their group effects, as brms keeps them.

    Returns
    -------
    BayesianFit or MultivariateFit
        Parsed formula, compiled program, fitted posterior, and observations;
        a `MultivariateFit` for a joint formula.

    Raises
    ------
    ValueError
        If the family is unknown, a predictor and its data are not given
        together, a trainable predictor is used without SVI, or a count
        is given to a method it does not apply to (``num_steps`` to NUTS
        or HMC; ``num_warmup``, ``num_samples``, or ``num_chains`` to
        SVI).
    """
    from quivers.formulas._refit import capture_recipe

    # A single-response bf carries the response keywords of this call.
    if isinstance(formula, ResponseModel):
        authored = single_response(
            formula,
            family=family,
            distributional=distributional,
            nonlinear=nonlinear,
            fixed_parameters=fixed_parameters,
            weights=weights,
            decision=decision,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            center=center,
            decomp=decomp,
            cmc=cmc,
            sparse=sparse,
            unused=unused,
            caller="fit",
        )
        formula, family = authored["formula"], authored["family"]
        distributional = authored["distributional"]
        nonlinear = authored["nonlinear"]
        fixed_parameters = authored["fixed_parameters"]
        weights, decision = authored["weights"], authored["decision"]
        binomial_trials = authored["binomial_trials"]
        thresholds_by = authored["thresholds_by"]
        mixture_components = authored["mixture_components"]
        center, decomp = authored["center"], authored["decomp"]
        cmc, sparse = authored["cmc"], authored["sparse"]
        unused = authored["unused"]
    # An existing artifact with file_refit="never" needs no program file.
    if file is not None and file_refit == "never" and Path(file).exists():
        return cast(
            "BayesianFit | _joint_fit.MultivariateFit", artifact_store.load_fit(file)
        )
    if program is not None:
        program, entry = program_source(program, entry)
    # brms file / file_refit: reuse or refresh a fit in the artifact store.
    if file is not None:
        arguments = {
            name: value
            for name, value in locals().items()
            if name in _FIT_ARGUMENTS and name not in {"file", "file_refit"}
        }
        if not needs_refit(Path(file), file_refit, arguments):
            return cast(
                "BayesianFit | _joint_fit.MultivariateFit",
                artifact_store.load_fit(file),
            )
        result = fit(**arguments)
        artifact_store.save_fit(result, file)
        return result
    # brms algorithm selects the method and guide.
    method, guide = resolve_algorithm(algorithm, method, guide)
    controls = InferenceControls(
        control=sampler_control(control),
        thin=thin,
        init=init,
        init_r=init_r,
        pathfinder=pathfinder_options(pathfinder),
    )
    refit_recipe = capture_recipe(
        fit, data=data, arguments=locals(), snapshot_policy=snapshot_policy
    )
    # brms cores: workers rebuild the fit from the recipe.
    if checked_cores(cores) != 1:
        controls = controls.with_(
            cores=cores,
            chain_worker=FormulaChainWorker(
                arguments=dict(refit_recipe.method), data=refit_recipe._data
            ),
        )
    from quivers.modeling.refit import _snapshot

    preparation_data = (
        refit_recipe._data if snapshot_policy == "clone" else _snapshot(data)
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        # Joint (multivariate) formulas share the recipe and inference seam.
        if isinstance(formula, MultivariateFormula):
            return _joint_fit.fit_entry(
                formula, preparation_data, refit_recipe.method, refit_recipe
            )
        formula, family, distributional = program_arguments(
            formula,
            program=program,
            entry=entry,
            static=static,
            parameters=parameters,
            response=response,
            family=family,
            distributional=distributional,
            nonlinear=nonlinear,
            fixed_parameters=fixed_parameters,
            caller="fit",
        )
        _refuse_joint_syntax(formula, "fit")
        if method == "svi":
            given = [
                name
                for name, value in (
                    ("num_warmup", num_warmup),
                    ("num_samples", num_samples),
                    ("num_chains", num_chains),
                )
                if value is not None
            ]
            if given:
                raise ValueError(
                    f"fit: {', '.join(given)} count MCMC iterations, and "
                    "method='svi' runs none; give the optimization steps as "
                    "num_steps"
                )
        elif method != "pathfinder" and num_steps is not None:
            raise ValueError(
                f"fit: num_steps counts SVI optimization steps, and method={method!r} "
                "runs MCMC; give num_warmup, num_samples, and num_chains"
            )
        predictor_name = "neural_eta" if predictor is not None else None
        prepared = _prepare_formula(
            formula,
            data=preparation_data,
            family=family,
            weights=weights,
            decision=decision,
            data2=data2,
            distributional=distributional,
            nonlinear=nonlinear,
            fixed_parameters=fixed_parameters,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            priors=priors,
            reparameterize=reparameterize,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            predictor_name=predictor_name,
            caller="fit",
            default_priors=default_priors,
            center=center,
            decomp=decomp,
            cmc=cmc,
            sparse=sparse,
            unused=unused,
            knots=knots,
            drop_unused_levels=drop_unused_levels,
        )
        native = _prepare_fit(
            prepared,
            predictor=predictor,
            predictor_data=predictor_data,
            method=method,
            preparation_data=preparation_data,
        )
        # brms refuses prior sampling with improper flat priors.
        if sample_prior != "no":
            require_proper(native.module, f"sample_prior={sample_prior!r}")
        result = _run_inference(
            native.program,
            native.observations,
            method=method,
            num_steps=num_steps,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            learning_rate=learning_rate,
            seed=seed,
            guide=guide,
            sample_prior=sample_prior,
            num_prior_samples=num_prior_samples,
            prior_seed=prior_seed,
            predictor=native.predictor,
            predictor_data=native.predictor_data,
            predictor_name=native.predictor_name,
            predictor_categories=native.predictor_categories,
            predictor_reference=native.predictor_reference,
            initial_values=_initial_values(prepared, fixed_parameters, initial_values),
            controls=controls,
        )
        return _assemble_fit(
            native,
            result,
            refit_recipe=refit_recipe,
            reparameterize=reparameterize,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            priors=priors,
            fixed_parameters=fixed_parameters,
            default_priors=default_priors,
            center=center,
            decomp=decomp,
            cmc=cmc,
            sparse=sparse,
            unused=unused,
        )


_FIT_ARGUMENTS = frozenset(inspect.signature(fit).parameters)
"""The canonical `fit` arguments a cached fit is compared on."""


def formula_to_qvr(
    formula: str | ResponseModel | MultivariateFormula | None = None,
    *,
    data: IntoDataFrame,
    program: str | tuple[str | Path, str] | None = None,
    entry: str | None = None,
    static: Mapping[str, int] | None = None,
    parameters: Mapping[str, str | NonlinearExpression] | None = None,
    response: str | None = None,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    data2: Mapping[str, AutocorMatrix] | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
    path: str | Path | None = None,
    knots: Knots | None = None,
    drop_unused_levels: bool = True,
) -> str:
    """Emit ``.qvr`` source for a brms-style formula without fitting.

    Builds the formula AST → QVR Module via `FormulaToQVRModule`,
    then serialises the module via [`quivers.dsl.emit.module_to_source`][quivers.dsl.emit.module_to_source].
    Optionally writes the result to ``path``. ``mixture_components`` selects
    any integer component count of at least two for ``family="mixture"``.
    ``distributional`` maps auxiliary names to formulas, for instance
    ``{"sigma": "~ z"}``; the corresponding linked predictors are emitted
    in the same QVR program as the location formula. ``fixed_parameters``
    instead binds auxiliary parameters to constants on their natural scale.
    ``weights="column"`` retains nonnegative likelihood weights and emits
    them only on the response observation; priors and predictions are unweighted.
    ``decision="column"`` or ``rt | dec(column) ~ predictors`` retains a Wiener
    boundary choice alongside positive reaction times. Emitted source visibly
    adapts raw times and Boolean choices to a lossless signed response.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula or None
        Authored formula over original dataframe rows, a single-response
        `bf` carrying its response keywords, or a joint formula whose one
        composed program is emitted; None for a program.
    data : IntoDataFrame
        Original training columns, including response modifiers.
    program : str or tuple[str or Path, str] or None
        QVR source, or a source file path and a program name, as in `fit`.
    entry : str or None
        The program name when it is given separately from the source text.
    static : Mapping[str, int] or None
        Finite extents of the program, by binder name.
    parameters : Mapping[str, str or NonlinearExpression] or None
        Formulas of the program's parameters, by name.
    response : str or None
        The site the program observes.
    family : str or Family
        Checked response law and retained link, threshold, and event choices.
    weights, decision : str or None
        Optional existing response-role columns.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Named matrices referenced by autocorrelation terms, as in brms:
        the spatial weights of ``sar(M)`` and the fixed covariance of
        ``fcor(M)``, each over the included response rows. A group-level
        ``gr(g, cov = M)`` reads a labeled ``DataFrame`` covariance between
        the levels of ``g``.
    distributional, nonlinear : Mapping or None
        Auxiliary designs and checked derived predictor graph.
    fixed_parameters : Mapping[str, float] or None
        Bound auxiliary constants.
    fixed_prior, random_scale_prior, correlation_prior : str
        Defaults consumed by the existing typed prior catalog.
    group_covariances : Mapping or None
        Known group covariance structures.
    priors : PriorInput or None
        Typed selectors or existing exact-name overrides.
    reparameterize : {"centered", "noncentered"}
        Shared group coordinate policy.
    binomial_trials : int or str or None
        Finite-trial count or retained column.
    thresholds_by : str or None
        Group with an identified full threshold profile.
    mixture_components : int
        Finite-mixture component count.
    predictor_name : str or None
        Checked external neural binding.

    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    knots : Mapping[str, Sequence[float]] or None
        brms ``knots``: knot values keyed by smooth covariate, passed to the
        ``cr``, ``cs``, ``cc``, ``ps`` and ``bs`` constructors of ``s()`` as
        mgcv's ``smoothCon(knots = )`` does.
    drop_unused_levels : bool
        brms ``drop_unused_levels``: with False, factor levels that no row
        uses keep their population columns (of zeros) and grouping-factor
        levels their group effects, as brms keeps them.
    Returns
    -------
    str
        Canonical checked source, without fitting a posterior.
    """
    # Joint (multivariate) formulas emit one composed program.
    if isinstance(formula, MultivariateFormula):
        with formula_data_options(knots, drop_unused_levels):
            module, _ = _joint_fit.joint_module(
                formula.with_default_priors(default_priors),
                data,
                fixed_prior=fixed_prior,
                random_scale_prior=random_scale_prior,
                correlation_prior=correlation_prior,
                group_covariances=group_covariances,
                priors=priors,
                reparameterize=reparameterize,
                data2=data2,
            )
        source = module_to_source(module)
        if path is not None:
            Path(path).write_text(source)
        return source
    formula, family, distributional = program_arguments(
        formula,
        program=program,
        entry=entry,
        static=static,
        parameters=parameters,
        response=response,
        family=family,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        caller="formula_to_qvr",
    )
    _refuse_joint_syntax(formula, "formula_to_qvr")
    lens, parsed = _formula_lens(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        data2=data2,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances=group_covariances,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
        default_priors=default_priors,
        center=center,
        decomp=decomp,
        cmc=cmc,
        sparse=sparse,
        unused=unused,
        knots=knots,
        drop_unused_levels=drop_unused_levels,
    )
    module, _ = lens.forward(parsed)
    source = module_to_source(module)
    if path is not None:
        Path(path).write_text(source)
    return source


def get_prior(
    formula: str | ResponseModel | MultivariateFormula | None = None,
    *,
    data: IntoDataFrame,
    program: str | tuple[str | Path, str] | None = None,
    entry: str | None = None,
    static: Mapping[str, int] | None = None,
    parameters: Mapping[str, str | NonlinearExpression] | None = None,
    response: str | None = None,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    data2: Mapping[str, AutocorMatrix] | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
    knots: Knots | None = None,
    drop_unused_levels: bool = True,
) -> dict[str, str]:
    """Return effective named priors without running inference.

    The arguments match `formula_to_qvr`, except for its output path.
    Priors reflect evaluated design columns, selected links, distributional
    formulas, group correlations, and explicit overrides. Default coefficient
    scales include column autoscaling; explicit overrides are emitted as given.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula or None
        Response formula in brms/lme4 notation, a single-response `bf`
        carrying its response keywords, or a joint formula whose
        namespaced sites (``response_<resp>__...``, ``rescor``) are listed;
        None for a program.
    data : IntoDataFrame
        Rows used to evaluate design matrices and autoscale default priors.
    program : str or tuple[str or Path, str] or None
        QVR source, or a source file path and a program name, as in `fit`.
    entry : str or None
        The program name when it is given separately from the source text.
    static : Mapping[str, int] or None
        Finite extents of the program, by binder name.
    parameters : Mapping[str, str or NonlinearExpression] or None
        Formulas of the program's parameters, by name.
    response : str or None
        The site the program observes.
    family : str or Family
        Registered response family or a family value with selected links.
    weights : str or None
        Numeric column of finite nonnegative likelihood weights. Values
        multiply each response's log density, retain their original scale,
        and do not weight priors or posterior predictions.
    decision : str or None
        Wiener boundary-choice column, equivalent to response-side dec(column).
        Numeric codes must be exactly zero/lower or one/upper.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Named matrices referenced by autocorrelation terms, as in brms:
        the spatial weights of ``sar(M)`` and the fixed covariance of
        ``fcor(M)``, each over the included response rows. A group-level
        ``gr(g, cov = M)`` reads a labeled ``DataFrame`` covariance between
        the levels of ``g``.
    distributional : Mapping[str, str] or None
        Separate formulas for auxiliary parameters.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters held constant; these have no prior entries.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default coefficient, group-scale, and LKJ Cholesky priors.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known covariance between labeled levels, keyed by grouping factor.
        The same declaration serves compilation, inspection, and prediction.
    priors : PriorInput or None
        Named overrides or typed selectors, including sampled shrinkage
        hierarchies. Unknown, unused, or ambiguous rules raise ValueError.
    reparameterize : {"centered", "noncentered"}
        Group parameterization; its conditional draws are not overrideable.
    binomial_trials : int or str
        Fixed trial count or a column of counts.
    thresholds_by : str or None
        Grouping column for ordinal cutpoints.
    mixture_components : int
        Number of Gaussian mixture components.
    predictor_name : str or None
        External contribution on the linear-predictor scale.

    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    knots : Mapping[str, Sequence[float]] or None
        brms ``knots``: knot values keyed by smooth covariate, passed to the
        ``cr``, ``cs``, ``cc``, ``ps`` and ``bs`` constructors of ``s()`` as
        mgcv's ``smoothCon(knots = )`` does.
    drop_unused_levels : bool
        brms ``drop_unused_levels``: with False, factor levels that no row
        uses keep their population columns (of zeros) and grouping-factor
        levels their group effects, as brms keeps them.
    Returns
    -------
    dict[str, str]
        Actual overrideable site names and canonical prior calls.

    Raises
    ------
    ValueError
        If a formula or family setting is invalid, or a prior is unused.
    """
    # Joint (multivariate) formulas read priors from the composed program.
    if isinstance(formula, MultivariateFormula):
        with formula_data_options(knots, drop_unused_levels):
            module, targets = _joint_fit.joint_module(
                formula.with_default_priors(default_priors),
                data,
                fixed_prior=fixed_prior,
                random_scale_prior=random_scale_prior,
                correlation_prior=correlation_prior,
                group_covariances=group_covariances,
                priors=priors,
                reparameterize=reparameterize,
                data2=data2,
                caller="get_prior",
            )
        return _joint_fit.joint_prior(module, targets)
    formula, family, distributional = program_arguments(
        formula,
        program=program,
        entry=entry,
        static=static,
        parameters=parameters,
        response=response,
        family=family,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        caller="get_prior",
    )
    _refuse_joint_syntax(formula, "get_prior")
    lens, parsed = _formula_lens(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        data2=data2,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances=group_covariances,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
        default_priors=default_priors,
        center=center,
        decomp=decomp,
        cmc=cmc,
        sparse=sparse,
        unused=unused,
        knots=knots,
        drop_unused_levels=drop_unused_levels,
    )
    return lens.get_prior(parsed)


def residual_lens(
    law: ResidualLaw | None,
    fixed_parameters: Mapping[str, float],
    build: Callable[[dict[str, float]], FormulaToQVRModule],
) -> FormulaToQVRModule:
    """Build a formula lens that observes the response through a residual law.

    Parameters
    ----------
    law : ResidualLaw or None
        Residual law of the response, or None for the family's own law.
    fixed_parameters : Mapping[str, float]
        Constant values of family parameters and of the law's structural
        sites.
    build : Callable[[dict[str, float]], FormulaToQVRModule]
        The lens of the family from its own fixed values.

    Returns
    -------
    FormulaToQVRModule
        The lens, with the law bound.
    """
    structural = set(law.coefficients) if law is not None else set()
    lens = build(
        {
            name: value
            for name, value in fixed_parameters.items()
            if name not in structural
        }
    )
    lens._bind_residual(
        law,
        {name: value for name, value in fixed_parameters.items() if name in structural},
    )
    return lens


@dataclass(frozen=True, slots=True)
class _PreparedFormula:
    """Ephemeral shared preparation, including live fitted design evaluators.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Existing typed formula compiler.
    formula : Formula
        Evaluated response and predictor metadata.
    family : Family
        Checked response construction.
    binomial_trials : int or str or None
        Resolved trial configuration.
    designs : dict[str, DesignMatrices]
        Retained evaluators for each additive predictor.
    distributional : dict[str, Formula]
        Auxiliary additive designs.
    graph : PredictorGraph or None
        Checked nonlinear dependency graph.
    trials : torch.Tensor or None
        Validated finite-trial observations.
    inputs : dict[str, torch.Tensor]
        Data the response kernel reads by name.
    """

    lens: FormulaToQVRModule
    formula: Formula
    family: Family
    binomial_trials: int | str | None
    designs: dict[str, DesignMatrices]
    distributional: dict[str, Formula]
    graph: PredictorGraph | None
    trials: torch.Tensor | None
    inputs: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _PreparedFit:
    """Compiled native binding shared by fitting and durable reconstruction.

    Parameters
    ----------
    formula : _PreparedFormula
        Retained formula preparation.
    module : Module
        Canonical generated AST.
    complement : FormulaData
        Shared group layouts and prior targets.
    program : MonadicProgram
        Compiled native program.
    observations : dict[str, torch.Tensor]
        Checked original runtime bindings.
    predictor : nn.Module or None
        External neural predictor.
    predictor_data : torch.Tensor or None
        Included training inputs.
    predictor_name : str or None
        Neural observation binding.
    predictor_categories : int or None
        Component width for vector heads.
    predictor_reference : int or None
        Retained reference component.
    preparation_data : IntoDataFrame or None
        One immutable original-row snapshot used to reconstruct designs.
    """

    formula: _PreparedFormula
    module: Module
    complement: FormulaData
    program: MonadicProgram
    observations: dict[str, torch.Tensor]
    predictor: nn.Module | None
    predictor_data: torch.Tensor | None
    predictor_name: str | None
    predictor_categories: int | None
    predictor_reference: int | None
    preparation_data: IntoDataFrame | None


@dataclass(frozen=True, slots=True)
class _InferenceResult:
    """Retained outputs of the existing inference drivers, independent of formula layout.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        Existing retained posterior representation.
    losses : tuple[float, ...]
        SVI losses, empty for MCMC.
    prior : PriorDraws or MCMCResult or None
        Independently retained or explicitly conditioned prior result.
    sample_prior : PriorMode
        Canonical prior/likelihood scoring policy.
    """

    posterior: MCMCResult | PriorDraws | Guide
    losses: tuple[float, ...]
    prior: PriorDraws | MCMCResult | None
    sample_prior: PriorMode


def _prepare_fit(
    prepared: _PreparedFormula,
    *,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    method: FitMethod = "nuts",
    predictor_value: torch.Tensor | None = None,
    preparation_data: IntoDataFrame | None = None,
) -> _PreparedFit:
    """Compile and bind a prepared formula without invoking an inference driver.

    Parameters
    ----------
    prepared : _PreparedFormula
        Retained designs and checked compiler configuration.
    predictor : nn.Module or None
        Optional external predictor with its current deterministic state.
    predictor_data : torch.Tensor or None
        Original input rows, before any retained subset selection.
    method : {"nuts", "hmc", "svi"}
        Validate trainable external predictors against the requested driver.
    predictor_value : torch.Tensor or None
        Saved checked predictor output for restoration without executing its module.

    preparation_data : IntoDataFrame or None
        One fit-owned original-row snapshot for exact durable reconstruction.

    Returns
    -------
    _PreparedFit
        Native program, original observations, and shared prediction metadata.

    Raises
    ------
    TypeError, ValueError
        If compilation or the predictor binding violates its declared interface.
    """
    lens, parsed = prepared.lens, prepared.formula
    predictor_name = "neural_eta" if predictor is not None else None
    family_obj = prepared.family
    selection = parsed.modifiers.subset
    if selection is not None and predictor_data is not None:
        if predictor_data.shape[0] != selection.size:
            raise ValueError(
                "predictor_data must align with the original subset dataframe rows"
            )
        predictor_data = predictor_data.index_select(
            0, torch.tensor(selection.positions, device=predictor_data.device)
        )
    if predictor is not None and predictor_data is None:
        raise ValueError("fit: predictor_data is required when predictor is supplied")
    if predictor is None and predictor_data is not None:
        raise ValueError("fit: predictor_data requires predictor")
    if (
        predictor is not None
        and method != "svi"
        and any(parameter.requires_grad for parameter in predictor.parameters())
    ):
        raise ValueError(
            "fit: a trainable external predictor requires method='svi'; "
            "freeze its parameters before using NUTS or HMC"
        )
    module, complement = lens.forward(parsed)
    program_runtime = compile_model(module, lens.families())
    morphism = program_runtime.morphism
    if not isinstance(morphism, MonadicProgram):
        raise TypeError(
            f"fit: compiled morphism has type "
            f"{type(morphism).__name__}, expected MonadicProgram"
        )
    program = morphism

    observations = _formula_observations(prepared)
    if predictor is not None:
        assert predictor_data is not None
        assert predictor_name is not None
        predictor_categories = (
            len(FormulaToQVRModule._component_labels(parsed))
            if family_obj.response is not None
            else None
        )
        predictor_reference = (
            family_obj.response.reference_index(
                FormulaToQVRModule._component_labels(parsed)
            )
            if family_obj.response is not None
            else 0
        )
        observations[predictor_name] = (
            predictor_value
            if predictor_value is not None
            else _predictor_output(
                predictor,
                predictor_data,
                n_obs=parsed.response_values.shape[0],
                n_categories=predictor_categories,
                reference=predictor_reference,
            ).detach()
        )
    else:
        predictor_categories = None
        predictor_reference = 0

    return _PreparedFit(
        prepared,
        module,
        complement,
        program,
        observations,
        predictor,
        predictor_data,
        predictor_name,
        predictor_categories,
        predictor_reference,
        preparation_data,
    )


def _formula_observations(
    prepared: _PreparedFormula,
    *,
    formula: Formula | None = None,
    lens: FormulaToQVRModule | None = None,
) -> dict[str, torch.Tensor]:
    """Bind the existing response/design contracts independently of compilation.

    Parameters
    ----------
    prepared : _PreparedFormula
        Canonical family, trial and response preparation.
    formula : Formula or None
        Optional same-response formula with reconciled joint grouping codes.
    lens : FormulaToQVRModule or None
        Optional same-response lens with reconciled auxiliary and graph codes.

    Returns
    -------
    dict[str, torch.Tensor]
        Actual response, modifier, predictor and grouping input carriers.
    """
    parsed = prepared.formula if formula is None else formula
    lens = prepared.lens if lens is None else lens
    family_obj = prepared.family
    observations: dict[str, torch.Tensor] = dict(prepared.inputs)
    observations.update(lens.fixed_column_observations(parsed))

    observations[response_input_name(parsed)] = response_tensor(parsed, family_obj)
    for _, _, predictor_formula in lens._predictor_items(parsed):
        for group, codes in predictor_formula.group_indices.items():
            observations[f"{_qvr_name(group)}_idx"] = torch.as_tensor(
                list(codes), dtype=torch.long
            )
    if family_obj.trials_parameter is not None:
        trials = prepared.trials
        assert trials is not None
        if parsed.response_layout.event_shape:
            observations["formula_trials"] = trials
        if (
            isinstance(prepared.binomial_trials, str)
            and not parsed.response_layout.event_shape
        ):
            observations[_qvr_name(prepared.binomial_trials)] = trials
    return observations


def _initial_values(
    prepared: _PreparedFormula,
    fixed_parameters: Mapping[str, float] | None,
    initial_values: Mapping[str, torch.Tensor | float] | None,
) -> dict[str, torch.Tensor | float]:
    """Combine support-aware formula starts with explicit natural-scale overrides.

    Parameters
    ----------
    prepared : _PreparedFormula
        Bound response and auxiliary designs.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary values excluded from the latent layout.
    initial_values : Mapping[str, torch.Tensor or float] or None
        Explicit starts, which take precedence.

    Returns
    -------
    dict[str, torch.Tensor or float]
        Named initial values consumed by the canonical inference drivers.
    """
    parsed, family_obj = prepared.formula, prepared.family
    parameter_formulas = prepared.distributional
    starts: dict[str, torch.Tensor | float] = {}
    for shift in family_obj.aux_params:
        if not shift.below_response:
            continue
        # These are initial coordinates, never a model offset.
        start = float(np.min(parsed.response_values)) / 4
        shift_formula = parameter_formulas.get(shift.name)
        if shift_formula is None:
            if not shift.bounded_by_response and shift.name not in (
                fixed_parameters or {}
            ):
                starts[shift.name] = start
            continue
        if not shift.link.is_log or not any(
            column.is_intercept for column in shift_formula.fixed_columns
        ):
            continue
        for column in shift_formula.fixed_columns:
            name = f"dpar_{shift.name}_" + (
                "intercept" if column.is_intercept else f"beta_{column.qvr_name}"
            )
            starts[name] = math.log(start) if column.is_intercept else 0.0
    starts.update(initial_values or {})
    return starts


def _run_inference(
    program: MonadicProgram,
    observations: dict[str, torch.Tensor],
    *,
    method: FitMethod,
    num_steps: int | None = None,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    num_chains: int | None = None,
    learning_rate: float = 0.01,
    seed: int = 0,
    guide: type | None = None,
    sample_prior: PriorMode = "no",
    num_prior_samples: int = 100,
    prior_seed: int | None = None,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    predictor_name: str | None = None,
    predictor_categories: int | None = None,
    predictor_reference: int | None = 0,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
    controls: InferenceControls | None = None,
) -> _InferenceResult:
    """Run the canonical inference and prior collectors for one native program.

    Parameters
    ----------
    program : MonadicProgram
        Single-response or joint compiled program.
    observations : dict[str, torch.Tensor]
        Checked runtime inputs and observations; SVI refreshes neural outputs here.
    method : {"nuts", "hmc", "svi"}
        Existing driver to invoke.
    num_steps, num_warmup, num_samples, num_chains : int or None
        Driver budgets, using the public fit defaults when omitted.
    learning_rate : float
        SVI optimizer step size.
    seed : int
        MCMC seed and default source of the independent prior stream.
    guide : type or None
        Guide constructor, defaulting to AutoNormalGuide.
    sample_prior : {"no", "yes", "only"}
        Canonical likelihood-scoring and independent-prior retention policy.
    num_prior_samples : int
        Independent prior draw count when requested.
    prior_seed : int or None
        Explicit independent prior seed.
    predictor : nn.Module or None
        Optional trainable external predictor.
    predictor_data : torch.Tensor or None
        Selected training input rows.
    predictor_name : str or None
        Checked neural observation binding.
    predictor_categories : int or None
        Response component count for vector-valued heads.
    predictor_reference : int or None
        Reference component, or None for all-component heads.
    initial_values : Mapping[str, torch.Tensor or float] or None
        Named natural-scale starts.
    controls : InferenceControls or None
        brms ``control``, ``thin``, ``init`` and ``init_r``, and the
        Pathfinder arguments.

    Returns
    -------
    _InferenceResult
        Posterior, losses, and separately retained priors, with no result-view policy.

    Raises
    ------
    ValueError
        If a control, budget, or initialization does not apply to the method.
    """
    from quivers.effects.observation_scoring import observation_scoring
    from quivers.inference.prior import _ancestral_program, _prior_mode, prior_draws

    mode = _prior_mode(sample_prior)
    if mode == "yes":
        from quivers.inference._posterior import sample_count

        sample_count(num_prior_samples)
        _ancestral_program(program)
    losses: tuple[float, ...] = ()
    controls = InferenceControls() if controls is None else controls
    if method == "svi" and (
        controls.control is not None
        or controls.thin != 1
        or (controls.init is not None and not isinstance(controls.init, Mapping))
    ):
        raise ValueError(
            "control, thin, and init strategies configure MCMC; method='svi' "
            "takes named initial values only"
        )
    if method == "svi" and isinstance(controls.init, Mapping):
        initial_values = {**dict(initial_values or {}), **controls.init}
    # Pathfinder: num_chains counts paths, as brms's chains.
    if controls.pathfinder is not None and method != "pathfinder":
        raise ValueError("pathfinder arguments apply to algorithm='pathfinder'")
    if method == "pathfinder":
        given = [
            name
            for name, value in (
                ("num_warmup", num_warmup),
                ("num_samples", num_samples),
                ("num_steps", num_steps),
                ("control", controls.control),
                ("thin", None if controls.thin == 1 else controls.thin),
            )
            if value is not None
        ]
        if given:
            raise ValueError(
                f"fit: {', '.join(given)} configure MCMC or SVI and do not apply "
                "to algorithm='pathfinder'; give num_chains (the number of paths) "
                "and pathfinder arguments such as draws and history_size"
            )
    with observation_scoring(mode != "only"):
        if method == "pathfinder":
            driver = pathfinder_driver(
                controls.pathfinder,
                num_paths=4 if num_chains is None else num_chains,
                seed=seed,
                init=controls.init,
                init_r=controls.init_r,
                starts=dict(initial_values or {}),
            )
            posterior = driver.run(
                program, torch.zeros(1, 1, dtype=torch.long), observations
            )
        elif method == "svi":
            posterior, losses = _fit_svi(
                program,
                observations,
                1000 if num_steps is None else num_steps,
                learning_rate=learning_rate,
                guide_cls=guide,
                predictor=predictor,
                predictor_data=predictor_data,
                predictor_name=predictor_name,
                predictor_categories=predictor_categories,
                predictor_reference=predictor_reference,
                initial_values=initial_values,
            )
        else:
            posterior = _fit_mcmc(
                program,
                observations,
                sampler=method,
                num_warmup=0
                if method == "fixed_param"
                else 500
                if num_warmup is None
                else num_warmup,
                num_samples=1000 if num_samples is None else num_samples,
                num_chains=4 if num_chains is None else num_chains,
                seed=seed,
                initial_values=initial_values,
                controls=controls,
            )

    retained_prior = (
        prior_draws(
            program,
            torch.zeros(1, 1),
            observations,
            num_samples=num_prior_samples,
            seed=(seed + 1) % (2**64) if prior_seed is None else prior_seed,
        )
        if mode == "yes"
        else None
    )

    return _InferenceResult(posterior, losses, retained_prior, mode)


def _assemble_fit(
    prepared: _PreparedFit,
    result: _InferenceResult,
    *,
    refit_recipe: RefitRecipe | None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
) -> BayesianFit:
    """Assemble the same retained formula view after fitting or state restoration.

    Parameters
    ----------
    prepared : _PreparedFit
        Checked program, observations, and retained design evaluators.
    result : _InferenceResult
        Existing posterior and independent prior objects.
    refit_recipe : RefitRecipe or None
        Original-data refit configuration, captured before training and subsetting.
    reparameterize : {"centered", "noncentered"}
        Authored group coordinate policy.
    thresholds_by : str or None
        Group with a full retained threshold profile.
    mixture_components : int
        Fitted mixture component count.
    fixed_prior, random_scale_prior, correlation_prior : str
        Original default-prior specifications.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known covariance structures retained for prediction.
    priors : PriorInput or None
        Typed or exact-name prior overrides.
    fixed_parameters : Mapping[str, float] or None
        Fixed auxiliary values.

    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    Returns
    -------
    BayesianFit
        Retained result with a rebuilt live prediction plan.
    """
    return BayesianFit(
        refit_recipe=refit_recipe,
        formula=prepared.formula.formula,
        family=prepared.formula.family,
        residual=prepared.formula.lens._residual,
        program=prepared.program,
        posterior=result.posterior,
        sample_prior=result.sample_prior,
        prior=result.prior,
        losses=result.losses,
        observations=prepared.observations,
        reparameterize=reparameterize,
        binomial_trials=prepared.formula.binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        # Includes gr(cov = M) covariances resolved during preparation.
        group_covariances={
            **dict(group_covariances or {}),
            **prepared.formula.lens.group_covariances,
        },
        priors=normalize_priors(priors),
        predictor=prepared.predictor,
        predictor_data=prepared.predictor_data,
        predictor_name=prepared.predictor_name,
        distributional=prepared.formula.distributional,
        predictor_graph=prepared.formula.graph,
        fixed_parameters=dict(fixed_parameters or {}),
        default_priors=default_priors,
        center=sparse_center(center, decomp, sparse=sparse),
        decomp=decomp,
        prediction_plan=FormulaPredictionPlan(
            module=prepared.module,
            preparation_data=prepared.preparation_data,
            designs=prepared.formula.designs,
            group_laws=prepared.complement.group_laws,
            level_sites=prepared.complement.level_sites,
            prior_targets=prepared.complement.prior_targets,
            predictors=prepared.formula.lens._planned(prepared.formula.formula),
        ),
    )


def _refuse_joint_syntax(formula: str | ResponseModel, caller: str) -> None:
    """Refuse joint-program terms written in a plain formula string.

    Parameters
    ----------
    formula : str or ResponseModel
        Authored formula. Only a string is inspected.
    caller : str
        Public entry point named in the diagnostic.

    Raises
    ------
    ValueError
        If a formula string uses ``mvbind()``, ``mi()``, ``me()``, or
        ``index()``, which compile through the joint program reached by
        `bf`.
    """
    if isinstance(formula, str) and (
        is_mvbind(formula) or uses_missing_syntax(formula)
    ):
        raise ValueError(
            f"{caller}: mvbind(), mi(), me(), and index() terms compile through "
            f"the joint program; pass bf({formula!r})"
        )


def _formula_lens(
    formula: str | ResponseModel,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    data2: Mapping[str, AutocorMatrix] | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
    knots: Knots | None = None,
    drop_unused_levels: bool = True,
) -> tuple[FormulaToQVRModule, Formula]:
    """Use common preparation for source emission and prior inspection.

    Parameters
    ----------
    formula : str or ResponseModel
        Authored formula over original dataframe rows, or a single-response
        `bf` whose response keywords merge with this call's.
    data : IntoDataFrame
        Original training columns, including response modifiers.
    family : str or Family
        Checked response law and retained link, threshold, and event choices.
    weights, decision : str or None
        Optional existing response-role columns.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Named matrices referenced by autocorrelation terms, as in brms:
        the spatial weights of ``sar(M)`` and the fixed covariance of
        ``fcor(M)``, each over the included response rows. A group-level
        ``gr(g, cov = M)`` reads a labeled ``DataFrame`` covariance between
        the levels of ``g``.
    distributional, nonlinear : Mapping or None
        Auxiliary designs and checked derived predictor graph.
    fixed_parameters : Mapping[str, float] or None
        Bound auxiliary constants.
    fixed_prior, random_scale_prior, correlation_prior : str
        Defaults consumed by the existing typed prior catalog.
    group_covariances : Mapping or None
        Known group covariance structures.
    priors : PriorInput or None
        Typed selectors or existing exact-name overrides.
    reparameterize : {"centered", "noncentered"}
        Shared group coordinate policy.
    binomial_trials : int or str or None
        Finite-trial count or retained column.
    thresholds_by : str or None
        Group with an identified full threshold profile.
    mixture_components : int
        Finite-mixture component count.
    predictor_name : str or None
        Checked external neural binding.

    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    knots : Mapping[str, Sequence[float]] or None
        brms ``knots``: knot values keyed by smooth covariate, passed to the
        ``cr``, ``cs``, ``cc``, ``ps`` and ``bs`` constructors of ``s()`` as
        mgcv's ``smoothCon(knots = )`` does.
    drop_unused_levels : bool
        brms ``drop_unused_levels``: with False, factor levels that no row
        uses keep their population columns (of zeros) and grouping-factor
        levels their group effects, as brms keeps them.
    Returns
    -------
    tuple[FormulaToQVRModule, Formula]
        Existing compiler lens and its evaluated input.
    """
    # A single-response bf carries the response keywords of this call.
    if isinstance(formula, ResponseModel):
        authored = single_response(
            formula,
            family=family,
            distributional=distributional,
            nonlinear=nonlinear,
            fixed_parameters=fixed_parameters,
            weights=weights,
            decision=decision,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            center=center,
            decomp=decomp,
            cmc=cmc,
            sparse=sparse,
            unused=unused,
            caller="a formula entry point",
        )
        formula, family = authored["formula"], authored["family"]
        distributional = authored["distributional"]
        nonlinear = authored["nonlinear"]
        fixed_parameters = authored["fixed_parameters"]
        weights, decision = authored["weights"], authored["decision"]
        binomial_trials = authored["binomial_trials"]
        thresholds_by = authored["thresholds_by"]
        mixture_components = authored["mixture_components"]
        center, decomp = authored["center"], authored["decomp"]
        cmc, sparse = authored["cmc"], authored["sparse"]
        unused = authored["unused"]
    prepared = _prepare_formula(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        data2=data2,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances=group_covariances,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
        default_priors=default_priors,
        center=center,
        decomp=decomp,
        cmc=cmc,
        sparse=sparse,
        unused=unused,
        knots=knots,
        drop_unused_levels=drop_unused_levels,
    )
    return prepared.lens, prepared.formula


def _prepare_formula(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    data2: Mapping[str, AutocorMatrix] | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    caller: str = "formula_to_qvr",
    default_priors: DefaultPriorPolicy = "quivers",
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: Sequence[str] = (),
    rows: tuple[int, ...] | None = None,
    knots: Knots | None = None,
    drop_unused_levels: bool | None = None,
) -> _PreparedFormula:
    """Prepare one retained design and compiler configuration for every entry.

    Parameters
    ----------
    formula : str
        Authored formula over original dataframe rows.
    data : IntoDataFrame
        Original training columns, including response modifiers.
    family : str or Family
        Checked response law and retained link, threshold, and event choices.
    weights, decision : str or None
        Optional existing response-role columns.
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Named matrices referenced by autocorrelation terms, as in brms:
        the spatial weights of ``sar(M)`` and the fixed covariance of
        ``fcor(M)``, each over the included response rows. A group-level
        ``gr(g, cov = M)`` reads a labeled ``DataFrame`` covariance between
        the levels of ``g``.
    distributional, nonlinear : Mapping or None
        Auxiliary designs and checked derived predictor graph.
    fixed_parameters : Mapping[str, float] or None
        Bound auxiliary constants.
    fixed_prior, random_scale_prior, correlation_prior : str
        Defaults consumed by the existing typed prior catalog.
    group_covariances : Mapping or None
        Known group covariance structures.
    priors : PriorInput or None
        Typed selectors or existing exact-name overrides.
    reparameterize : {"centered", "noncentered"}
        Shared group coordinate policy.
    binomial_trials : int or str or None
        Finite-trial count or retained column.
    thresholds_by : str or None
        Group with an identified full threshold profile.
    mixture_components : int
        Finite-mixture component count.
    predictor_name : str or None
        Checked external neural binding.
    caller : str
        Entry-point name used in located configuration errors.
    rows : tuple[int, ...] or None
        Rows decided for a whole multivariate model, replacing this
        response's own decision; None decides from this response's variables.

    default_priors : {"quivers", "brms"}
        Default prior policy: the proper, autoscaled quivers defaults, or
        ``brms::default_prior`` (flat ``b``; Student-t intercept, ``sd`` and
        ``sigma`` priors scaled by the response). Explicit priors win either way.
    center : bool or None
        brms ``bf(center = )``: sample the intercept of the column-centered
        design and report ``b_Intercept = Intercept_c - mean(X) b``. None
        centers under the brms policy only.
    decomp : {"none", "QR"}
        brms ``bf(decomp = "QR")``: sample the location coefficients in scaled
        QR coordinates; class ``b`` priors act on those coordinates.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept keeps treatment
        coding (no cell-mean coding of its first factor).
    sparse : bool
        brms ``sparse``: the same model with centering disabled, as brms
        disables it for sparse designs; incompatible with ``decomp="QR"``.
    unused : Sequence[str]
        brms ``unused``: further columns kept with the model rows, so rows
        missing them are excluded as for model variables.
    knots : Mapping[str, Sequence[float]] or None
        brms ``knots`` of the smooth constructors. None keeps the knots of an
        enclosing `formula_data_options` scope, which joint preparation sets
        once for every response.
    drop_unused_levels : bool or None
        brms ``drop_unused_levels``; None keeps the enclosing scope's (True
        outside every scope).
    Returns
    -------
    _PreparedFormula
        One set of retained designs, typed family metadata, and compiler options.

    Raises
    ------
    ValueError
        If a family, response role, predictor, or trial contract is invalid.
    """
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"{caller}: unknown family {family!r}; choices are {sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family
    from quivers.formulas._custom_family import custom_distributional

    distributional = custom_distributional(
        family_obj, distributional, fixed_parameters=fixed_parameters
    )
    # mixtures: expand family="mixture" and copy the main formula to mu<k>.
    family_obj, distributional = mixture_distributional(
        family_obj, formula, distributional, mixture_components
    )
    # brms na_omit: one decision over every variable of every model part.
    original = nw.from_native(data, eager_only=True).to_pandas()
    # brms sparse disables centering; unused columns join row retention.
    center = sparse_center(center, decomp, sparse=sparse)
    unused = checked_unused(unused, original, caller=caller)
    retained = rows
    if rows is None:
        retained = complete_rows(
            original,
            model_texts(formula, (distributional, nonlinear)),
            model_columns(
                weights,
                decision,
                thresholds_by,
                binomial_trials,
                family_obj.baseline.by if family_obj.baseline is not None else None,
                *unused,
                *family_obj.kernel_columns,
            ),
        )
    if retained is not None:
        # A column holding missing values recovers its observed type.
        data = original.iloc[list(retained)].infer_objects()
    # Autocorrelation terms (brms ar/ma/arma/cosy/unstr/sar/fcor) leave the
    # predictor and bind the response law; see quivers.formulas._autocor.
    authored = formula
    formula, autocor_terms = split_autocor_terms(formula)
    # brms knots and drop_unused_levels reach every evaluation below; None
    # keeps the enclosing scope's.
    with formula_data_options(knots, drop_unused_levels, data2):
        parsed, design = _threshold_formula_and_design(
            nonlinear_response_source(formula, nonlinear),
            data,
            thresholds_by=thresholds_by,
            weights=weights,
            decision=decision,
            cmc=cmc,
            groups=(family_obj.baseline.by,)
            if family_obj.baseline is not None and family_obj.baseline.by is not None
            else (),
            # brms counts the thresholds of a hurdle family above the hurdle.
            extra_category="hurdle_cumulative" in threshold_kinds(family_obj),
            # brms integer categories: ordinal codes 1, ..., max(y).
            category_coding=family_category_coding(family_obj),
        )
    from quivers.formulas._response_roles import included_data, retained_trials

    # brms thres(n, gr): group thresholds are scored on category log masses.
    check_threshold_groups(family_obj, parsed)
    if retained is not None:
        # Retained rows keep their positions in the caller's frame.
        selection = parsed.modifiers.subset
        positions = compose_positions(
            retained, selection.positions if selection is not None else None, len(data)
        )
        assert positions is not None
        parsed = parsed.with_(
            modifiers=parsed.modifiers.with_(
                subset=RowSelection(size=len(original), positions=positions)
            )
        )
        data = original
    data = included_data(parsed, data)
    # Data the response kernel reads by name, on the included rows.
    kernel_inputs = family_obj.kernel_bindings(
        nw.from_native(data, eager_only=True).to_pandas(), data2
    )
    binomial_trials = retained_trials(parsed, family_obj, data, binomial_trials)
    residual = None
    if autocor_terms:
        residual, parsed = bind_autocor_terms(
            autocor_terms,
            family_obj,
            parsed,
            data=data,
            data2=data2,
            distributional=distributional,
            nonlinear=bool(nonlinear),
            caller=caller,
        )
    parsed = parsed.with_(formula=authored)
    # brms mu<c>: complete category-specific formulas with the main formula.
    if family_obj.response is not None:
        distributional = category_distributional(
            family_obj,
            FormulaToQVRModule._component_labels(parsed),
            authored,
            distributional,
            nonlinear=nonlinear is not None,
        )
        # brms sigma<c>: per-category logistic-normal scale formulas.
        check_scale_formulas(
            family_obj,
            FormulaToQVRModule._component_labels(parsed),
            (distributional or {}).keys() | (fixed_parameters or {}).keys(),
        )
    # Mixture components and per-category formulas own every predictor term;
    # the main formula keeps the response, modifiers, and group coding.
    if family_obj.mixture is not None or any(
        is_category_parameter(family_obj, name) for name in distributional or {}
    ):
        parsed = main_without_terms(parsed)
    parameter_designs: dict[str, DesignMatrices] = {}

    def evaluate(text: str) -> tuple[Formula, DesignMatrices]:
        """Evaluate one auxiliary formula on the included rows."""
        return fields_and_design(text, data=data, data2=data2, caller=caller)

    family_parameters = frozenset(
        {
            "mu",
            *(aux.name for aux in family_obj.aux_params),
            *(
                name
                for name in distributional or {}
                if is_category_parameter(family_obj, name)
                or is_scale_parameter(family_obj, name)
            ),
        }
    )
    with formula_data_options(knots, drop_unused_levels, data2):
        graph = build_predictor_graph(
            formula,
            nonlinear,
            distributional,
            data,
            parsed.response_name,
            family_parameters=family_parameters,
            evaluate=evaluate,
            designs=parameter_designs,
        )
        # A formula of a name that is no family parameter is a nonlinear
        # parameter of the graph.
        parameter_formulas = _parse_distributional(
            {
                name: value
                for name, value in (distributional or {}).items()
                if isinstance(value, str) and name in family_parameters
            },
            parsed.response_name,
            evaluate=evaluate,
            designs=parameter_designs,
        )
    trials = (
        _response_trials(parsed, data, binomial_trials, caller=caller)
        if family_obj.trials_parameter is not None
        else None
    )
    lens = residual_lens(
        residual,
        fixed_parameters or {},
        lambda fixed: FormulaToQVRModule(
            family_obj,
            fixed_parameters=fixed,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            user_priors=priors,
            reparameterize=reparameterize,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            predictor_name=predictor_name,
            distributional=parameter_formulas,
            predictor_graph=graph,
            default_priors=default_priors,
            center=center,
            decomp=decomp,
        ),
    )
    return _PreparedFormula(
        lens,
        parsed,
        family_obj,
        binomial_trials,
        {"mu": design, **parameter_designs},
        parameter_formulas,
        graph,
        trials,
        kernel_inputs,
    )


def _threshold_formula_and_design(
    formula: str,
    data: IntoDataFrame,
    *,
    thresholds_by: str | None,
    weights: str | None,
    decision: str | None = None,
    groups: tuple[str, ...] = (),
    cmc: bool = True,
    extra_category: bool = False,
    category_coding: CategoryCoding | None = None,
) -> tuple[Formula, DesignMatrices]:
    """Evaluate a full threshold profile with one shared group location.

    Parameters
    ----------
    formula : str
        Authored formula, retained for presentation.
    data : IntoDataFrame
        Training predictors and response.
    thresholds_by : str or None
        Group whose centered threshold shape also needs a location effect.
    weights : str or None
        Optional response-weight column.
    decision : str or None
        Optional lower/upper decision column.
    groups : tuple[str, ...]
        Additional structural groups without implied varying coefficients.
    cmc : bool
        brms cell-mean coding of a formula without intercept.
    extra_category : bool
        Whether response code zero is the hurdle category, outside the
        thresholds that ``thres(n)`` counts.
    category_coding : {"ordinal", "hurdle", "nominal"} or None
        How the family reads an integer response as categories.

    Returns
    -------
    tuple of Formula and DesignMatrices
        Evaluated metadata and retained transforms, with exactly one ordinary
        varying intercept for the threshold group.
    """
    parsed, design = _formula_and_design(
        formula,
        data,
        weights=weights,
        decision=decision,
        groups=groups,
        cmc=cmc,
        extra_category=extra_category,
        category_coding=category_coding,
    )
    if thresholds_by is None or any(
        term.group == thresholds_by and term.slope == "Intercept"
        for term in parsed.random_terms
    ):
        return parsed, design
    expanded = f"{formula} + (1 | {thresholds_by})"
    parsed, design = _formula_and_design(
        expanded,
        data,
        weights=weights,
        decision=decision,
        groups=groups,
        cmc=cmc,
        extra_category=extra_category,
        category_coding=category_coding,
    )
    return parsed.with_(formula=formula), design


def _parse_distributional(
    formulas: Mapping[str, str] | None,
    response: str,
    *,
    evaluate: Callable[[str], tuple[Formula, DesignMatrices]],
    designs: dict[str, DesignMatrices],
) -> dict[str, Formula]:
    """Evaluate auxiliary design matrices on the same response rows.

    Parameters
    ----------
    formulas : Mapping[str, str] or None
        Auxiliary parameter names and their formulas.
    response : str
        Primary response name used to evaluate design matrices.
    evaluate : Callable
        Evaluates one formula.
    designs : dict[str, DesignMatrices]
        Receives each fitted evaluator.

    Returns
    -------
    dict[str, Formula]
        One evaluated predictor per requested auxiliary parameter.

    Raises
    ------
    ValueError
        If a formula lacks a tilde, names a different parameter, or holds a
        measurement-error ``me()`` term.
    """
    parsed: dict[str, Formula] = {}
    for parameter, formula in (formulas or {}).items():
        if "~" not in formula:
            raise ValueError(f"distributional formula for {parameter!r} requires '~'")
        lhs, rhs = formula.split("~", 1)
        if lhs.strip() not in ("", parameter):
            raise ValueError(
                f"distributional formula for {parameter!r} has mismatched left-hand side {lhs.strip()!r}"
            )
        if re.search(r"(?<![A-Za-z0-9_.])me\(", rhs):
            raise ValueError(
                f"distributional formula for {parameter!r}: me() terms are "
                "supported in the main formula of a joint model only"
            )
        parsed[parameter], designs[parameter] = evaluate(f"{response} ~ {rhs}")
    return parsed


def _fit_mcmc(
    program: MonadicProgram,
    observations: dict[str, torch.Tensor],
    *,
    sampler: Literal["nuts", "hmc", "fixed_param"],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
    controls: InferenceControls | None = None,
) -> MCMCResult:
    """Run seeded NUTS or HMC on the compiled program.

    Parameters
    ----------
    program
        Compiled formula program.
    observations
        Data indexed by observed site name.
    sampler
        Hamiltonian transition kernel.
    num_warmup
        Warmup transitions per chain.
    num_samples
        Retained transitions per chain.
    num_chains
        Independent chains.
    seed
        Seed from which the driver derives each chain's stream.
    initial_values
        Optional natural-scale values for named latent sites.
    controls
        brms kernel control, thinning, and initialization.

    Returns
    -------
    MCMCResult
        Retained samples and sampler statistics.
    """
    controls = InferenceControls() if controls is None else controls
    strategy, starts = chain_initialization(
        controls.init, dict(initial_values or {}), num_chains
    )
    mcmc = MCMC(
        kernel=mcmc_kernel(sampler, controls.control),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        init_strategy=strategy,
        seed=seed,
        initial_values=starts,
        thin=controls.thin,
        init_radius=controls.init_r,
    )
    # One run of the program: its observation rows come from the data
    # through the formula's plate, not from the input's rows.
    x = torch.zeros(1, 1, dtype=torch.long)
    # brms inits(): a rerun of the fit's arguments stops here with the
    # chains' initial values.
    report_initial_values(mcmc, program, x, observations)
    # brms cores: a worker runs its assigned chains only; the
    # parent deals the chains to workers that rebuild this fit.
    assigned = assigned_chains()
    if assigned is not None:
        raise ChainsRan(run_chains(mcmc, program, x, observations, assigned))
    if controls.cores > 1 and num_chains > 1:
        if controls.chain_worker is None:
            raise ValueError("cores > 1 needs the fit's recipe to rebuild it")
        return parallel_chains(
            mcmc, program, x, observations, controls.chain_worker, controls.cores
        )
    return mcmc.run(program, x, observations)


def _fit_svi(
    program,
    observations,
    num_steps,
    *,
    learning_rate: float,
    guide_cls=None,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    predictor_name: str | None = None,
    predictor_categories: int | None = None,
    predictor_reference: int | None = 0,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
):
    """Run an SVI fit + ELBO.

    Default guide is `AutoNormalGuide`: a mean-field
    diagonal-Normal that scales well across model shapes. Mean-field
    is known to underestimate posterior variance components in
    hierarchical / mixed-effects models; for serious analysis of
    those models, use ``method="nuts"``. Users can swap in any other
    ``Guide`` class via ``fit(..., guide=SomeGuide)``; the class is
    constructed with ``(program, observed_names=...)``.

    Parameters
    ----------
    program : MonadicProgram
        Existing compiled native model.
    observations : dict[str, torch.Tensor]
        Mutable checked training bindings.
    num_steps : int
        Optimizer iteration budget.
    learning_rate : float
        Shared optimizer step size.
    guide_cls : type or None
        Existing guide constructor, defaulting to AutoNormalGuide.
    predictor : nn.Module or None
        External trainable module.
    predictor_data : torch.Tensor or None
        Included predictor rows.
    predictor_name : str or None
        Runtime neural binding.
    predictor_categories : int or None
        Component count for a vector response.
    predictor_reference : int or None
        Reference coordinate, or None for all components.
    initial_values : Mapping or None
        Named natural-scale guide starts.

    Returns
    -------
    tuple[Guide, tuple[float, ...]]
        Trained guide and retained finite losses.
    """
    if guide_cls is None:
        guide_cls = AutoNormalGuide
    guide = (
        guide_cls(
            program, observed_names=set(observations), initial_values=initial_values
        )
        if initial_values
        else guide_cls(program, observed_names=set(observations))
    )
    from quivers.inference.svi import _prepare_optimizer

    optimizer = _prepare_optimizer(
        program.parameters(),
        guide.parameters(),
        () if predictor is None else predictor.parameters(),
        learning_rate=learning_rate,
    )
    svi = SVI(program, guide, optimizer, ELBO())
    n_obs = int(observations[next(iter(observations))].shape[0])
    # One run of the program: its observation rows come from the data
    # through the formula's plate, not from the input's rows.
    x = torch.zeros(1, 1, dtype=torch.long)

    def step_bindings(index: int) -> dict[str, torch.Tensor]:
        """Refresh neural bindings for one existing SVI iteration.

        Parameters
        ----------
        index : int
            Iteration supplied by the shared driver.

        Returns
        -------
        dict[str, torch.Tensor]
            Checked observations with the current differentiable head output.
        """
        step_observations = observations
        if predictor is not None:
            assert predictor_data is not None and predictor_name is not None
            step_observations = dict(observations)
            step_observations[predictor_name] = _predictor_output(
                predictor,
                predictor_data,
                n_obs=n_obs,
                n_categories=predictor_categories,
                reference=predictor_reference,
            )
        return step_observations

    losses = svi.run(x, step_bindings, num_steps=num_steps)
    if predictor is not None:
        assert predictor_data is not None and predictor_name is not None
        observations[predictor_name] = _predictor_output(
            predictor,
            predictor_data,
            n_obs=n_obs,
            n_categories=predictor_categories,
            reference=predictor_reference,
        ).detach()
    return guide, losses


def _response_trials(
    formula: Formula,
    data: IntoDataFrame | None,
    specification: int | str | None,
    *,
    caller: str,
) -> torch.Tensor:
    """Validate shared trial inputs against scalar or vector response events.

    Parameters
    ----------
    formula : Formula
        Evaluated response with explicit row/event metadata.
    data : IntoDataFrame or None
        Source rows containing any specified trial column.
    specification : int or str or None
        Constant or trial column; None infers vector row totals.
    caller : str
        User-facing workflow name for diagnostics.

    Returns
    -------
    torch.Tensor
        Exact validated Int64 row totals.

    Raises
    ------
    ValueError
        If explicit trials disagree with a vector event's component sum.
    """
    retained = formula.modifiers.trials
    supplied = (
        torch.as_tensor(retained.data.copy(), dtype=torch.long)
        if retained is not None and specification == "formula_trials"
        else None
    )
    if formula.response_layout.event_shape:
        totals = count_totals(formula)
        if specification is None:
            return totals
        trials = (
            supplied
            if supplied is not None
            else _binomial_trial_tensor(data, specification, n_obs=len(totals))
        )
        if not torch.equal(totals, trials):
            raise ValueError(
                f"{caller}: vector response row totals disagree with supplied trials"
            )
        return trials
    trials = (
        supplied
        if supplied is not None
        else _binomial_trial_tensor(
            data,
            1 if specification is None else specification,
            n_obs=formula.response_values.shape[0],
        )
    )
    _validate_binomial_response(response_tensor(formula), trials, caller=caller)
    return trials


def _binomial_trial_tensor(
    data: IntoDataFrame | None,
    trials: int | str,
    *,
    n_obs: int,
) -> torch.Tensor:
    """Validate nonnegative binomial trials exactly representable in float64.

    Parameters
    ----------
    data : IntoDataFrame or None
        Rows containing a trial-count column, required for named trials.
    trials : int or str
        Shared nonnegative count or trial-count column name.
    n_obs : int
        Required number of rows.

    Returns
    -------
    torch.Tensor
        One signed-64-bit count per row, including zero-trial rows.

    Raises
    ------
    ValueError
        If counts are missing, negative, Boolean, fractional, nonfinite,
        nonnumeric, or greater than ``2**53``, the exact integer limit of
        the native float64 count distributions.
    """
    if isinstance(trials, bool) or not isinstance(trials, (int, str)):
        raise ValueError(
            "binomial_trials must be a nonnegative integer or a column name"
        )
    if isinstance(trials, int):
        if trials < 0 or trials > _MAX_EXACT_COUNT:
            raise ValueError(
                "binomial_trials must be a nonnegative integer at most 2**53"
            )
        return torch.full((n_obs,), trials, dtype=torch.long)
    if data is None:
        raise ValueError("a trial column requires a retained dataframe")
    frame = nw.from_native(data, eager_only=True)
    if trials not in frame.columns:
        raise ValueError(f"binomial_trials column {trials!r} is not in the data")
    column = frame[trials].to_numpy()
    checked_integer_data(
        column,
        error="binomial_trials must contain nonnegative finite integers at most 2**53",
    )
    if column.shape != (n_obs,):
        raise ValueError(
            f"binomial_trials column has {column.size} rows, expected {n_obs}"
        )
    return torch.as_tensor(column.astype(np.int64, copy=True))


def _predictor_output(
    predictor: nn.Module,
    data: torch.Tensor,
    *,
    n_obs: int,
    n_categories: int | None = None,
    reference: int | None = 0,
) -> torch.Tensor:
    """Validate neural predictors and identify their component coordinates.

    Parameters
    ----------
    predictor : nn.Module
        Trainable neural predictor on the linear-predictor scale.
    data : torch.Tensor
        Input rows supplied by the caller.
    n_obs : int
        Retained observation row count.
    n_categories : int or None
        Total component extent, or None for a scalar predictor.
    reference : int or None
        Identified zero component, or None for all-component predictors.

    Returns
    -------
    torch.Tensor
        Scalar row values or flattened ordered free component values.

    Raises
    ------
    TypeError
        If the predictor does not return a tensor.
    ValueError
        If row or component axes do not match the retained design.
    """
    value = predictor(data)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "fit: predictor must return a torch.Tensor on the linear-predictor scale"
        )
    if n_categories is not None:
        width = n_categories - int(reference is not None)
        if value.shape == (n_obs,) and width == 1:
            return value
        if value.shape == (n_obs, width):
            return value.reshape(-1)
        if reference is not None and value.shape == (n_obs, n_categories):
            coordinates = [index for index in range(n_categories) if index != reference]
            return (value[:, coordinates] - value[:, reference, None]).reshape(-1)
        raise ValueError(
            "fit: categorical predictor returned shape "
            f"{tuple(value.shape)}, expected ({n_obs}, {width}) "
            f"or ({n_obs}, {n_categories})"
        )
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape != (n_obs,):
        raise ValueError(
            f"fit: predictor returned shape {tuple(value.shape)}, expected ({n_obs},)"
        )
    return value


def _validate_binomial_response(
    response: torch.Tensor,
    trials: torch.Tensor,
    *,
    caller: str,
) -> None:
    """Validate observed counts against their checked finite-trial tensor.

    Parameters
    ----------
    response : torch.Tensor
        Original count values.
    trials : torch.Tensor
        Per-row finite trials.
    caller : str
        Entry-point diagnostic label.

    Raises
    ------
    ValueError
        If counts are not finite integers within their trial bounds.
    """
    if not torch.isfinite(response).all() or not torch.equal(
        response, response.round()
    ):
        raise ValueError(f"{caller}: binomial responses must be finite integers")
    if torch.any(response < 0) or torch.any(response > trials):
        raise ValueError(
            f"{caller}: binomial responses must lie between zero and "
            "binomial_trials for every row"
        )


# Joint (multivariate) fitting imports this module's shared preparation and
# inference seam, so its module is bound after every name above is defined.
from quivers.formulas import _joint_fit  # noqa: E402
