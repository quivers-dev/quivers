"""Joint (multivariate) formula fits over the shared inference and replay engines.

A `MultivariateFit` holds one compiled joint program and its posterior.
Response-scoped results replay that program once per retained draw and read
each response's native sites, so correlated residuals and shared group
effects stay jointly drawn. Parameter names follow brms (``b_<resp>_<coef>``,
``sd_<group>__<resp>_<coef>``, ``rescor__<a>__<b>``).
"""

from __future__ import annotations

import inspect
from dataclasses import replace
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import copy
from pathlib import Path
from typing import Literal, TypedDict, Unpack, cast, overload

import didactic.api as dx
import numpy as np
import narwhals as nw
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous._multivariate_student_t import MultivariateStudentT
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.conditional import ConditionalEffects, PredictionGrid
from quivers.diagnostics.contrast import ContrastExpression, _contrast_hypothesis
from quivers.diagnostics.cross_validation import FoldPrediction, FoldSplit
from quivers.diagnostics.density import PointNullSpec
from quivers.diagnostics.hypothesis import HypothesisSummary
from quivers.diagnostics.posterior import PosteriorAnalysis, _sample_stats
from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import Module, ObserveStep, ProgramDecl, SampleStep
from quivers.dsl.emit import draw_call_to_source, module_to_source
from quivers.formulas._fit import (
    BayesianFit,
    _PreparedFormula,
    _formula_observations,
    _initial_values,
    _prepare_formula,
    _run_inference,
    fit,
)
from quivers.formulas._autocor import (
    AutocorMatrix,
    autocor_joint,
    autocor_log_likelihood,
)

# brms fitting controls, flat priors, and derived sites.
from quivers.formulas._brms_priors import (
    FLAT_CATALOG,
    FLAT_FAMILY,
    flat_prior_step,
    improper_sites,
    derived_site_draws,
    require_proper,
    with_flat_catalog,
)
from quivers.formulas._fit_controls import (
    InferenceControls,
    checked_cores,
    pathfinder_options,
    resolve_algorithm,
    sampler_control,
)
from quivers.formulas._population_coordinates import reported_coefficients
from quivers.formulas._row_retention import (
    joint_complete_rows,
    model_columns,
    model_texts,
)
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._group_law import group_coefficients
from quivers.formulas._group_prediction import NewLevelPolicy
from quivers.formulas._joint_compile import _JointEmission, emit_joint
from quivers.formulas._joint_options import JointFitOptions
from quivers.formulas._joint_prediction import (
    JOINT_RESPONSE,
    JointPredictionContext,
    prepare_joint_prediction,
)
from quivers.formulas._brms_hypothesis import (
    HypothesisInput,
    HypothesisTable,
    fit_hypothesis,
    is_brms_hypothesis,
)
from quivers.formulas._multivariate import MultivariateFormula
from quivers.formulas._missing import LatentResponseSpec
from quivers.formulas._joint_missing import (
    MissingPlan,
    apply_missing,
    latent_specs,
    missing_frame,
    missing_specification,
    symbolize,
)
from quivers.formulas.formula import RowSelection
from quivers.formulas._posterior import replay, row_binding
from quivers.formulas._prediction import FormulaPredictionPlan
from quivers.formulas._refit import update_fit
from quivers.formulas._prior_emission import ordinary_prior_step
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas._priors import PriorSpec, resolve_priors
from quivers.formulas._priors import PriorInput, PriorSet, PriorTarget, normalize_priors
from quivers.formulas._response import expected_response, response_tensor
from quivers.formulas.formula import Formula, _qvr_name
from quivers.formulas._response_modifiers import response_input_name
from quivers.inference._likelihood import collect_log_likelihood
from quivers.inference._posterior import (
    collect_site_draws,
    constrained_draws,
    draw_count,
)
from quivers.inference._response import (
    response_distribution,
    response_mean,
    response_rng,
    response_sample,
)
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.prior import PriorDraws, PriorMode, prior_draws
from quivers.inference.pathfinder import draw_provenance
from quivers.formulas._parallel_chains import FormulaChainWorker
from quivers.formulas._data_options import formula_data_options
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace
from quivers.modeling._row_attachments import original_positions, validate_split_rows
from quivers.modeling.refit import RefitRecipe, refit_callback

# brms post-processing methods (summaries, accessors, LOO family) and stored criteria.
from quivers.diagnostics.elpd import NO_CRITERIA, ELPDEstimate
from quivers.formulas._postprocess_methods import JointPostprocess

# brms coefficient renaming shared with single fits.
from quivers.formulas._postprocess_names import (
    _ordinal_thresholds,
    coefficient_labels,
    name_scope,
    coordinate_labels,
    distributional_classes,
    shrinkage_scales,
)

# One catalogue of brms variable names shared with single-response fits.
from quivers.formulas._term import OutputAxis
from quivers.formulas._group_law import GroupMemberLaw
from quivers.formulas._name_scope import (
    NameScope,
    coefficient_names,
    level_coefficients,
    predictor_variables,
)
from quivers.formulas._variable_catalogue import (
    brms_order,
    checked_variables,
    class_variables,
    correlation_pairs,
    native_variables,
    scalar_coordinates,
    shrinkage_variables,
    threshold_variables,
    CoefficientName,
    group_variables,
)
from quivers.formulas._threshold_groups import LOCATION_SITE

# brms evidence, effects and predictive checks.
from quivers.formulas._evidence_effects import EvidenceEffectsMethods
from quivers.formulas._brms_effects import (
    BrmsConditionalEffects,
    EffectMethod,
    JointConditionalEffectsOptions,
    automatic_conditional_effects,
    joint_grid_options,
)

#: Response selection: one label, several labels, or None for every response.
type ResponseSelection = str | Sequence[str] | None


class JointEffectOptions(TypedDict, total=False):
    """Optional keywords shared by the joint prediction methods.

    Each key has the meaning of the keyword of the same name on
    `MultivariateFit.posterior_predict`.
    """

    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy
    num_samples: int | None
    seed: int


class JointScoreOptions(TypedDict, total=False):
    """Held-out scoring keywords a joint refit callback forwards.

    Each key has the meaning of the keyword of the same name on
    `MultivariateFit.log_likelihood`.
    """

    resp: ResponseSelection
    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy


def _kind(
    posterior: MCMCResult | PriorDraws | Guide,
) -> Literal["mcmc", "variational", "prior"]:
    """Name the provenance of retained draws.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        Retained posterior.

    Returns
    -------
    {"mcmc", "variational", "prior"}
        Draw provenance.
    """
    # Pathfinder draws report variational provenance.
    return draw_provenance(posterior)


class MultivariateFit(JointPostprocess, EvidenceEffectsMethods, ValueModel):
    """A fitted joint (multivariate) formula model.

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    program : MonadicProgram
        The single compiled joint program.
    posterior : MCMCResult or PriorDraws or Guide
        Retained posterior representation.
    observations : Mapping[str, torch.Tensor]
        Runtime inputs and observations bound to the program.
    emission : _JointEmission
        The composed emission with its response namespaces.
    prepared : Mapping[str, _PreparedFormula]
        Ordinary preparation of each response, retaining design evaluators.
    losses : tuple[float, ...]
        SVI losses, empty for MCMC.
    prior : PriorDraws or MCMCResult or None
        Retained prior result, if one was requested.
    sample_prior : {"no", "yes", "only"}
        Prior/likelihood scoring policy of the fit.
    priors : PriorSet
        Joint prior rules, including response selectors.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default priors of every response preparation.
    group_covariances : dict[str, GroupCovariance]
        Known group covariances, keyed by grouping factor.
    reparameterize : {"centered", "noncentered"}
        Group coordinate policy.
    preparation_data : IntoDataFrame or None
        Original-row snapshot used to rebuild designs.
    refit_recipe : RefitRecipe or None
        Retained original data and canonical fitting configuration.
    response_fits : dict[str, BayesianFit]
        Lazily built response views used by row preparation.
    missing : MissingPlan
        ``mi()`` declarations of the joint formula.
    missing_specs : dict[str, LatentResponseSpec]
        Recorded and latent coordinates of each ``mi()`` response.
    data2 : dict[str, AutocorMatrix]
        Named matrices given to `fit` as ``data2``.
    criteria : Mapping[str, ELPDEstimate or torch.Tensor]
        Criteria stored by `add_criterion`.

    Notes
    -----
    Equality follows `BayesianFit`: by value over the specification and
    settings, by identity over live objects.
    """

    specification: MultivariateFormula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | PriorDraws | Guide = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observations: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    emission: _JointEmission = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prepared: Mapping[str, _PreparedFormula] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    losses: tuple[float, ...] = ()
    prior: PriorDraws | MCMCResult | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_prior: PriorMode = "no"
    priors: PriorSet = dx.field(
        default_factory=PriorSet, opaque=True, extras=OPAQUE_BY_VALUE
    )
    fixed_prior: str = "Normal(0.0, 5.0)"
    random_scale_prior: str = "HalfNormal(1.0)"
    correlation_prior: str = "LKJCholesky(1.0)"
    group_covariances: dict[str, GroupCovariance] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    reparameterize: Literal["centered", "noncentered"] = "noncentered"
    preparation_data: IntoDataFrame | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    refit_recipe: (
        RefitRecipe[IntoDataFrame, Mapping[str, object], MultivariateFit] | None
    ) = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    response_fits: dict[str, BayesianFit] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    missing: MissingPlan = dx.field(
        default_factory=MissingPlan, opaque=True, extras=OPAQUE_BY_VALUE
    )
    missing_specs: dict[str, LatentResponseSpec] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    data2: dict[str, AutocorMatrix] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    # Criteria stored by add_criterion (brms ``x$criteria``), kept in artifacts.
    criteria: Mapping[str, ELPDEstimate | torch.Tensor] = dx.field(
        default=NO_CRITERIA, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def response_names(self) -> tuple[str, ...]:
        """Return the response labels in authored order.

        Returns
        -------
        tuple[str, ...]
            Response labels.
        """
        return self.specification.response_names

    @property
    def rescor(self) -> bool:
        """Report whether residual correlations were estimated.

        Returns
        -------
        bool
            The fitted residual-correlation setting.
        """
        return self.specification.rescor

    @property
    def qvr_source(self) -> str:
        """Return the canonical QVR source of the fitted joint program.

        Returns
        -------
        str
            Source emitted from the retained joint module.
        """
        return module_to_source(self.emission.module)

    def _selection(self, resp: ResponseSelection) -> tuple[str, ...]:
        """Validate a response selection.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            One label, several labels, or None for every response.

        Returns
        -------
        tuple[str, ...]
            Selected labels in authored order.

        Raises
        ------
        ValueError
            If a label is unknown, repeated, or the selection is empty.
        """
        if resp is None:
            return self.response_names
        names = (resp,) if isinstance(resp, str) else tuple(resp)
        if not names or len(set(names)) != len(names):
            raise ValueError("resp must name one or more distinct responses")
        unknown = [name for name in names if name not in self.response_names]
        if unknown:
            raise ValueError(
                f"unknown responses {unknown!r}; expected {self.response_names!r}"
            )
        return tuple(name for name in self.response_names if name in names)

    def _response_fit(self, name: str) -> BayesianFit:
        """Build the response view used by the ordinary row preparation.

        The view retains the response's fitted designs, joint grouping codes,
        and its own compiled program, so `prepare_prediction` applies the
        single-response row rules unchanged. Its posterior is never sampled.

        Parameters
        ----------
        name : str
            Response label.

        Returns
        -------
        BayesianFit
            The response view, built once per fit.

        Raises
        ------
        TypeError
            If the response does not compile to one MonadicProgram.
        """
        if name in self.response_fits:
            return self.response_fits[name]
        response = self.emission.responses[name]
        authored = self.specification.responses[name]
        part = self.prepared[name]
        lens = copy(response.lens)
        lens._emitted_groups = []
        module, complement = lens.forward(response.formula)
        family = part.family
        program = (
            Compiler(
                module,
                families=with_flat_catalog(lens.families(), module),
            )
            .compile()
            .morphism
        )
        if not isinstance(program, MonadicProgram):
            raise TypeError("a joint response must compile to one MonadicProgram")
        view = BayesianFit(
            formula=response.formula,
            family=family,
            program=program,
            posterior=self.posterior,
            observations=_formula_observations(
                part, formula=response.formula, lens=response.lens
            ),
            reparameterize=self.reparameterize,
            binomial_trials=part.binomial_trials,
            thresholds_by=authored.thresholds_by,
            mixture_components=authored.mixture_components,
            fixed_prior=self.fixed_prior,
            random_scale_prior=self.random_scale_prior,
            correlation_prior=self.correlation_prior,
            group_covariances=response.lens.group_covariances,
            distributional=dict(response.lens._distributional),
            predictor_graph=response.lens._predictor_graph,
            fixed_parameters=dict(authored.fixed_parameters or {}),
            prediction_plan=FormulaPredictionPlan(
                module=module,
                designs=part.designs,
                preparation_data=self.preparation_data,
                group_laws=complement.group_laws,
                level_sites=complement.level_sites,
                prior_targets=complement.prior_targets,
                predictors=lens.planned(response.formula),
            ),
        )
        self.response_fits[name] = view
        return view

    def draws(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Return constrained draws of the native joint sites.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain, or None for every retained draw (100 for a guide).
        seed : int
            Seed for draws from a variational guide.

        Returns
        -------
        dict[str, torch.Tensor]
            Constrained draws keyed by namespaced site, on chain and draw axes.
        """
        registry = LatentRegistry.from_model(
            self.program, observed_names=set(self.observations)
        )
        draws = constrained_draws(
            self.posterior,
            torch.zeros(1, 1, dtype=torch.long),
            num_samples=num_samples,
            seed=seed,
            registry=registry,
        )
        # Constants, centered intercepts, and QR coefficients are derived sites.
        reported = [
            response.names.get(local, local)
            for response in self.emission.responses.values()
            for local in reported_coefficients(
                response.lens._predictor_items(response.formula)
            )
        ]
        # mixtures: parameters equated to a sampled one keep their own name.
        equated = [
            response.names.get(target, target)
            for response in self.emission.responses.values()
            for target, source in (
                response.lens._family.mixture.equal
                if response.lens._family.mixture is not None
                else ()
            )
            if response.names.get(source, source) in draws
        ]
        return derived_site_draws(
            self.emission.module,
            draws,
            reported
            + [site for target in self.emission.prior_targets for site in target.sites]
            + equated,
        )

    def variables(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Return draws under brms parameter names.

        Population coefficients are ``b_<resp>_<coef>`` (location),
        ``b_<dpar>_<resp>_<coef>`` (distributional), and
        ``b_<resp>_<nlpar>_<coef>`` (nonlinear). Unregressed auxiliary
        parameters are ``<dpar>_<resp>``; group standard deviations are
        ``sd_<group>__<resp>_<coef>``; group correlations are
        ``cor_<group>__<term>__<term>``; residual correlations are
        ``rescor__<a>__<b>``; a shared Student degree of freedom is ``nu``.
        Per-level group coefficients are ``r_<group>__<resp>[<level>,<coef>]``
        (``r_<group>__<dpar>_<resp>[...]`` for distributional predictors).
        Vector coefficients of categorical responses keep a trailing
        component axis. The thresholds of a cumulative response with one
        threshold vector are ``b_<resp>_Intercept[<k>]``, and those of a
        response with threshold groups are
        ``b_<resp>_Intercept[<group>,<k>]``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            brms-named draws on chain and draw axes.
        """
        return self._named(
            self.draws(num_samples=num_samples, seed=seed),
            self._threshold_draws(num_samples=num_samples, seed=seed),
        )

    def _threshold_draws(
        self, *, num_samples: int | None, seed: int
    ) -> dict[str, torch.Tensor]:
        """Replay the brms-scale thresholds of the ordinal responses.

        One replay of the fitted program over the retained draws reads every
        response's threshold site, at the cost of one prediction pass.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            For each response whose thresholds brms reports as
            ``b_<resp>_Intercept[...]``, the thresholds of every training row
            minus the location intercept, shaped
            ``(chain, draw, row, threshold)``. Empty without such a response.
        """
        names = [
            name
            for name in self.response_names
            if _ordinal_thresholds(self._response_fit(name))
        ]
        if not names:
            return {}
        prepared = self._prepare(
            None,
            re_formula=None,
            allow_new_levels=False,
            sample_new_levels="uncertainty",
            seed=seed,
            require_trials=False,
        )

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Read every response's row thresholds from one trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Row-by-threshold values keyed by response.
            """
            result = {}
            for name in names:
                view = self._response_fit(name)
                site = (
                    "row_cutpoints"
                    if view.formula.modifiers.threshold_groups is not None
                    else "cutpoints"
                )
                result[name] = row_binding(
                    recorded.sites[self.emission.responses[name].names[site]].value,
                    prepared.rows[name],
                    len(view.threshold_labels),
                )
            return result

        thresholds = self._replay(
            prepared, operation, num_samples=num_samples, seed=seed, stream=2
        )
        draws = self.draws(num_samples=num_samples, seed=seed)
        for name in names:
            site = self.emission.responses[name].names.get("intercept")
            if site in draws:
                thresholds[name] = thresholds[name] - draws[site][..., None, None]
        return thresholds

    def _named(
        self,
        draws: Mapping[str, torch.Tensor],
        thresholds: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Rename native draws to brms parameter names.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Native constrained draws.
        thresholds : Mapping[str, torch.Tensor]
            brms-scale row thresholds of the ordinal responses reported as
            ``b_<resp>_Intercept[...]``, from `_threshold_draws`.

        Returns
        -------
        dict[str, torch.Tensor]
            brms-named draws.
        """
        result: dict[str, torch.Tensor] = {}
        # brms omits the response from names of a one-response model.
        joint = len(self.response_names) > 1
        # Coefficient labels as single fits name them:
        # brms's renaming of each response's design labels (f[q] -> fq).
        renamed = {
            resp: coefficient_labels(self._response_fit(resp))
            for resp in self.response_names
        }

        special: dict[tuple[str, str, str], str] = {}

        def tag(resp: str) -> str:
            """Return the response part of a brms name.

            Parameters
            ----------
            resp : str
                Response label.

            Returns
            -------
            str
                ``<resp>_`` in a multivariate model, empty otherwise.
            """
            return resp + "_" if joint else ""

        # brms names mi() and me() design coefficients bsp_<resp>_<term>,
        # with mi(x) spelled mix and me(x, sdx) spelled mexsdx.
        for item in self.missing.inputs:
            replacements = {}
            for source in item.sources:
                if source in self.missing.measurements:
                    term = self.missing.measurements[source]
                    replacements[term.variable] = term.label
                else:
                    variable = self.missing.responses[source].variable
                    replacements[variable] = "mi" + _brms_name(variable)
            pattern = re.compile(
                r"\b(" + "|".join(map(re.escape, replacements)) + r")\b"
            )
            special[(item.consumer, item.parameter, item.column.name)] = pattern.sub(
                lambda match: replacements[match.group(1)], item.column.name
            )
        consumed: set[str] = set()
        shrunk: dict[str, str] = {}
        scopes: dict[str, NameScope] = {}
        outputs: dict[str, OutputAxis | None] = {}
        for resp, response in self.emission.responses.items():
            plan = self._response_fit(resp).prediction_plan
            assert plan is not None
            local = {
                site.removeprefix(response.prefix): value
                for site, value in draws.items()
                if site.startswith(response.prefix)
            }
            planned = {predictor.prefix: predictor for predictor in plan.predictors}
            outputs |= {
                response.prefix + key: item.output for key, item in planned.items()
            }
            if resp in thresholds:
                # brms reports the thresholds in place of the intercept and
                # the native threshold coordinates.
                result |= threshold_variables(
                    thresholds[resp],
                    infix=tag(resp),
                    groups=self._response_fit(resp).formula.modifiers.threshold_groups,
                )
            for parameter, prefix, predictor in response.lens._predictor_items(
                response.formula
            ):
                scope = name_scope(
                    self._response_fit(resp),
                    parameter,
                    prefix,
                    predictor,
                    renamed[resp],
                    response=resp if joint else "",
                    special={
                        column: label
                        for (owner, name, column), label in special.items()
                        if (owner, name) == (resp, parameter)
                    },
                )
                scopes[response.prefix + prefix] = scope
                named = predictor_variables(
                    planned[prefix].terms,
                    scope.with_(
                        intercept=not (resp in thresholds and parameter == "mu")
                    ),
                    local,
                )
                result |= {**named.population, **named.special}
                shrunk |= {
                    response.prefix + site: name for site, name in named.sites.items()
                }
                consumed |= {response.prefix + site for site in named.read}

        def member_names(member: GroupMemberLaw) -> tuple[CoefficientName, ...]:
            """Name the components of one group-level coefficient."""
            return coefficient_names(
                scopes[member.prefix],
                outputs[member.prefix],
                member.slope,
                member.width if member.axis is not None else None,
            )

        block_laws = self.emission.group_laws
        grouped = {site for law in block_laws for site in law.correlation.sites}
        order = {
            target.site: index
            for index, target in enumerate(self.emission.prior_targets)
        }
        scales, correlations = group_variables(
            sorted(
                block_laws,
                key=lambda law: min(
                    order.get(member.scale.site, len(order)) for member in law.members
                ),
            ),
            member_names,
            draws,
        )
        result |= scales | correlations
        for target in self.emission.prior_targets:
            if target.site not in draws:
                continue
            value = draws[target.site]
            resp = target.response
            if target.site in consumed:
                continue
            # A component drawn on its own is part of its site's variables.
            consumed.update(target.sites)
            if target.whole is not None:
                continue
            if target.class_ == "cor" and target.site in grouped:
                continue
            if (
                resp is not None
                and resp in thresholds
                and (
                    target.class_.startswith("cutpoint")
                    or target.class_ == LOCATION_SITE
                    or (
                        target.class_ == "Intercept"
                        and target.nlpar is None
                        and target.dpar in (None, "mu")
                    )
                )
            ):
                continue
            if target.class_ in {"b", "Intercept"} and resp is not None:
                # The reported coefficients of transformed coordinates.
                continue
            elif target.class_ == "sd" and resp is not None:
                # The standard deviations of the group blocks.
                continue
            elif target.class_ == "cor" and target.site == "rescor":
                result |= correlation_pairs(
                    "rescor", self.response_names, value @ value.transpose(-1, -2)
                )
            elif target.dpar == target.site and resp is None:
                # The degrees of freedom that correlated responses share.
                result[target.site] = value
            elif target.class_ == "corme" and resp is not None:
                result |= correlation_pairs(
                    f"corme_{resp}" if joint else "corme",
                    target.component_labels,
                    value @ value.transpose(-1, -2),
                )
            elif target.class_ in {"meanme", "sdme"} and resp is not None:
                if target.coef is not None:
                    result |= scalar_coordinates(
                        f"{target.class_}_{tag(resp)}{target.coef}", value
                    )
            elif (
                target.group is None
                and resp is not None
                and target.coef is None
                and target.dpar == target.class_
            ):
                result |= scalar_coordinates(
                    f"{target.class_}_{resp}" if joint else target.class_, value
                )
            else:
                # Hyperparameters of special terms share one class catalogue
                # with single-response fits.
                result |= class_variables(
                    target.with_(
                        site=target.site.removeprefix(
                            self.emission.responses[resp].prefix
                        )
                    )
                    if resp is not None
                    else target,
                    value,
                    infix=self._infix(target),
                    labels=coordinate_labels(self._response_fit(resp), target)
                    if resp is not None
                    else tuple(str(label) for label in target.component_labels),
                )
        for label, spec in self.missing_specs.items():
            if spec.standard_error is not None and f"mi_{label}__truth" in draws:
                values = draws[f"mi_{label}__truth"]
                for index, row in enumerate(spec.rows.positions):
                    result[f"Yl_{label}[{row + 1}]"] = values[:, :, index]
            elif f"mi_{label}__missing" in draws:
                values = draws[f"mi_{label}__missing"]
                for index, position in enumerate(spec.latent_positions):
                    row = spec.rows.positions[position]
                    result[f"Ymi_{label}[{row + 1}]"] = values[:, :, index]
        levels = self.group_levels
        for resp, response in self.emission.responses.items():
            for parameter, prefix, predictor in response.lens._predictor_items(
                response.formula
            ):
                if not predictor.random_terms:
                    continue
                for item in level_coefficients(
                    self._random(draws, predictor, response.prefix + prefix),
                    scopes[response.prefix + prefix],
                    outputs[response.prefix + prefix],
                ):
                    for index, level in enumerate(levels[item.group]):
                        result[item.variable(level)] = item.values[:, :, index]
        for law in block_laws:
            consumed.update(law.correlation.sites)
            consumed.update(
                site
                for member in law.members
                for site in (member.raw_site, member.scale.site, member.mean_site)
                if site is not None
            )
        for label, spec in self.missing_specs.items():
            consumed.update((f"mi_{label}__truth", f"mi_{label}__missing"))
        # brms omits the latent values of me() terms unless
        # save_pars(latent = TRUE) is set; draws() keeps their innovations.
        consumed.update(
            f"me_{term.key}__z" for term in self.missing.measurements.values()
        )
        named, read = shrinkage_variables(
            draws, consumed, shrinkage_scales(self.emission.module, draws, shrunk)
        )
        result |= named
        consumed |= read
        chains, count = next(iter(draws.values())).shape[:2] if draws else (0, 0)
        return checked_variables(
            brms_order(
                result | native_variables(draws, consumed),
                [
                    name
                    for resp in self.response_names
                    for name in distributional_classes(self._response_fit(resp))
                ],
            ),
            (chains, count),
        )

    def _infix(self, target: PriorTarget) -> str:
        """Return the parameter and response part of a target's brms name.

        Parameters
        ----------
        target : PriorTarget
            A registered site of one response.

        Returns
        -------
        str
            ``<resp>_``, ``<dpar>_<resp>_`` or ``<resp>_<nlpar>_``; the
            response is omitted in a one-response model.
        """
        resp = (
            f"{target.response}_"
            if target.response is not None and len(self.response_names) > 1
            else ""
        )
        if target.nlpar is not None:
            return f"{resp}{target.nlpar}_"
        if target.dpar not in (None, "mu", target.class_):
            return f"{target.dpar}_{resp}"
        return resp

    @staticmethod
    def _local_prefix(target: PriorTarget) -> str:
        """Return the predictor prefix of a target's site within its response.

        Parameters
        ----------
        target : PriorTarget
            A registered site of one response.

        Returns
        -------
        str
            ``nlpar_<name>_``, ``dpar_<name>_``, or empty for the location.
        """
        if target.nlpar is not None:
            return f"nlpar_{target.nlpar}_"
        if target.dpar not in (None, "mu"):
            return f"dpar_{target.dpar}_"
        return ""

    def _predictor(self, resp: str, parameter: str) -> tuple[Formula, str]:
        """Locate one response predictor and its namespaced coefficient prefix.

        Parameters
        ----------
        resp : str
            Response label.
        parameter : str
            ``mu``, a distributional parameter, or a nonlinear parameter.

        Returns
        -------
        tuple[Formula, str]
            The retained predictor and its namespaced prefix.
        """
        predictor, prefix = self._response_fit(resp)._formula_predictor(parameter)
        return predictor, self.emission.responses[resp].prefix + prefix

    def fixed_effects(
        self,
        *,
        resp: str,
        parameter: str = "mu",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Extract one response's population coefficients by column label.

        Parameters
        ----------
        resp : str
            Response label.
        parameter : str
            ``mu`` or a regressed distributional or nonlinear parameter.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            Column labels mapped to coefficient draws on chain and draw axes.
        """
        self._selection(resp)
        view = self._response_fit(resp)
        return view._fixed_effects(
            self.draws(num_samples=num_samples, seed=seed),
            view._formula_predictor(parameter)[1],
            self.emission.responses[resp].prefix,
        )

    def random_effects(
        self,
        *,
        resp: str,
        parameter: str = "mu",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Extract one response's group adjustments by group and coefficient.

        Parameters
        ----------
        resp : str
            Response label.
        parameter : str
            ``mu`` or a regressed distributional or nonlinear parameter.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Group, then coefficient label, mapped to ``(chain, draw, level)``
            arrays in `group_levels` order. Shared blocks are read from the
            joint Gaussian block.
        """
        self._selection(resp)
        predictor, prefix = self._predictor(resp, parameter)
        draws = self.draws(num_samples=num_samples, seed=seed)
        return self._random(draws, predictor, prefix)

    def _random(
        self, draws: Mapping[str, torch.Tensor], predictor: Formula, prefix: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Read group coefficients from the joint layouts.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Native draws.
        predictor : Formula
            The response predictor.
        prefix : str
            Its namespaced coefficient prefix.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Group and coefficient label mapped to per-level draws.
        """
        coefficients = group_coefficients(
            draws,
            tuple(
                law
                for law in self.emission.group_laws
                if any(member.prefix == prefix for member in law.members)
            ),
            self.group_levels,
        )
        result: dict[str, dict[str, torch.Tensor]] = {}
        for term in predictor.random_terms:
            result.setdefault(term.group, {})[term.slope] = coefficients[
                (prefix, term.group, term.slope)
            ]
        return result

    def coefficients(
        self,
        *,
        resp: str,
        parameter: str = "mu",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Combine one response's population coefficients and group adjustments.

        Parameters
        ----------
        resp : str
            Response label.
        parameter : str
            ``mu`` or a regressed distributional or nonlinear parameter.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Arrays with the structure of `random_effects`, including
            population coefficients.
        """
        self._selection(resp)
        predictor, prefix = self._predictor(resp, parameter)
        draws = self.draws(num_samples=num_samples, seed=seed)
        view = self._response_fit(resp)
        fixed = view._fixed_effects(
            draws,
            view._formula_predictor(parameter)[1],
            self.emission.responses[resp].prefix,
        )
        varying = self._random(draws, predictor, prefix)
        for group, values in varying.items():
            for name, coefficient in fixed.items():
                population = coefficient.unsqueeze(2)
                values[name] = (
                    values[name] + population
                    if name in values
                    else population.expand(
                        *population.shape[:2],
                        len(self.group_levels[group]),
                        *population.shape[3:],
                    )
                )
        return varying

    @property
    def group_levels(self) -> dict[str, tuple[str, ...]]:
        """Return the joint level order of every grouping factor.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Group names and their joint level labels.
        """
        return {
            group: levels
            for response in self.emission.responses.values()
            for _, _, predictor in response.lens._predictor_items(response.formula)
            for group, levels in predictor.group_levels.items()
        }

    def _replay(
        self,
        prepared: JointPredictionContext,
        operation: Callable[[Trace], Mapping[str, torch.Tensor]],
        *,
        num_samples: int | None,
        seed: int,
        stream: int,
        score: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Replay each retained draw once and collect named values.

        Parameters
        ----------
        prepared : JointPredictionContext
            Prepared joint program and bindings.
        operation : Callable[[Trace], Mapping[str, torch.Tensor]]
            Reads named values from one replayed trace.
        num_samples : int or None
            Draws per chain.
        seed : int
            Posterior draw seed.
        stream : int
            Offset of the response random stream from ``seed``.
        score : bool
            Score observed responses during replay.

        Returns
        -------
        dict[str, torch.Tensor]
            Named values on chain and draw axes.
        """
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        context = prepared.context

        def values() -> Iterator[Mapping[str, torch.Tensor]]:
            """Apply the operation to each replayed trace in draw order.

            Yields
            ------
            Mapping[str, torch.Tensor]
                Named values of one chain/draw position.
            """
            for recorded in replay(
                cast(BayesianFit, self),
                draws,
                shape,
                context=context,
                score_observations=score,
            ):
                yield operation(recorded)

        with (
            response_rng(
                seed + stream, (*draws.values(), *context.observations.values())
            ),
            torch.no_grad(),
        ):
            return collect_site_draws(values(), shape)

    def _prepare(
        self,
        new_data: IntoDataFrame | None,
        *,
        recorded: frozenset[str] = frozenset(),
        re_formula: str | None,
        allow_new_levels: bool,
        sample_new_levels: NewLevelPolicy,
        seed: int,
        require_trials: bool = True,
        scoring: bool = False,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> JointPredictionContext:
        """Prepare training or fresh rows for one joint request.

        Parameters
        ----------
        new_data : IntoDataFrame or None
            Fresh rows, or None for the training rows.
        recorded : frozenset[str]
            Responses whose outcomes are read from new_data.
        re_formula : str or None
            Varying-effect selector.
        allow_new_levels : bool
            Permit new grouping identities.
        sample_new_levels : NewLevelPolicy
            New-level policy.
        seed : int
            Group-effect stream seed.
        require_trials : bool
            Require trial totals for fresh vector counts.
        scoring : bool
            Score training rows with the fitted program itself.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        JointPredictionContext
            Joint replay context.
        """
        return prepare_joint_prediction(
            self,
            new_data,
            recorded=recorded,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            require_trials=require_trials,
            scoring=scoring,
            data2=data2,
        )

    def posterior_linpred(
        self,
        *,
        resp: ResponseSelection = None,
        parameter: str = "mu",
        transform: bool = False,
        new_data: IntoDataFrame | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: NewLevelPolicy = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Replay linear predictors of the selected responses.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        parameter : str
            ``mu`` or a regressed distributional or nonlinear parameter of
            every selected response.
        transform : bool
            Return the inverse-link binding instead of the linear predictor.
        new_data : IntoDataFrame or None
            Fresh rows holding every response's predictors, or None.
        re_formula : str or None
            Varying-effect selector applied to every response.
        allow_new_levels : bool
            Permit new grouping identities.
        sample_new_levels : str
            Existing new-level policy, drawn once per shared grouping axis.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for posterior and group-effect streams.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        dict[str, torch.Tensor]
            Response label mapped to ``(chain, draw, row[, component])``.

        Raises
        ------
        ValueError
            If a selection, predictor, or row is invalid.
        """
        names = self._selection(resp)
        bindings = {
            name: self._response_fit(name)._linpred_binding(parameter, transform)
            for name in names
        }
        prepared = self._prepare(
            new_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            require_trials=False,
            data2=data2,
        )

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Read every selected linear predictor from one trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Linear predictors keyed by response.
            """
            result = {}
            for name in names:
                response = self.emission.responses[name]
                rows = prepared.rows[name]
                result[name] = bindings[name].read(
                    lambda binding, names=response.names: (
                        recorded.sites[names[binding]].value
                    ),
                    rows,
                )
            return result

        return self._replay(
            prepared, operation, num_samples=num_samples, seed=seed, stream=2
        )

    def posterior_epred(
        self,
        *,
        resp: ResponseSelection = None,
        new_data: IntoDataFrame | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: NewLevelPolicy = "uncertainty",
        num_samples: int | None = None,
        mc_samples: int | None = None,
        seed: int = 0,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return expected responses of the selected responses.

        Under residual correlation the expectation is the mean vector of the
        joint residual law, whose components are the response means.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        new_data : IntoDataFrame or None
            Fresh rows holding every response's predictors, or None.
        re_formula : str or None
            Varying-effect selector applied to every response.
        allow_new_levels : bool
            Permit new grouping identities.
        sample_new_levels : str
            Existing new-level policy, drawn once per shared grouping axis.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        mc_samples : int or None
            Monte Carlo budget for responses without a closed mean.
        seed : int
            Seed for posterior, group-effect, and Monte Carlo streams.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        dict[str, torch.Tensor]
            Response label mapped to ``(chain, draw, row[, event])``; ordinal
            and categorical responses give category probabilities.
        """
        names = self._selection(resp)
        prepared = self._prepare(
            new_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            data2=data2,
        )

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Read every selected expected response from one trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Expected responses keyed by response.
            """
            if self.rescor:
                value = response_mean(
                    response_distribution(recorded, JOINT_RESPONSE), mc_samples
                )
                rows = prepared.rows[names[0]]
                value = row_binding(value, rows, (len(self.response_names),))
                return {
                    name: value[..., self.response_names.index(name)] for name in names
                }
            result: dict[str, torch.Tensor] = {}
            for name in names:
                response = self.emission.responses[name]
                view = self._response_fit(name)
                family = view.family
                categories = (
                    len(view.category_labels) if family.category_response else 0
                )
                site = response.names[_qvr_name(response.formula.response_name)]
                value = expected_response(
                    response_distribution(recorded, site),
                    family,
                    categories=categories,
                    mc_samples=mc_samples,
                )
                result[name] = row_binding(
                    value,
                    prepared.rows[name],
                    response.formula.response_layout.event_shape or categories,
                )
            return result

        return self._replay(
            prepared, operation, num_samples=num_samples, seed=seed, stream=2
        )

    def posterior_predict(
        self,
        *,
        resp: ResponseSelection = None,
        new_data: IntoDataFrame | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: NewLevelPolicy = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw responses, keeping the residual correlation of joint events.

        Every response is drawn in the same replay; under residual
        correlation one vector event is drawn per row and the selected
        components are returned, so selected responses keep their joint law.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        new_data : IntoDataFrame or None
            Fresh rows holding every response's predictors, or None.
        re_formula : str or None
            Varying-effect selector applied to every response.
        allow_new_levels : bool
            Permit new grouping identities.
        sample_new_levels : str
            Existing new-level policy, drawn once per shared grouping axis.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for posterior, group-effect, and response streams.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        dict[str, torch.Tensor]
            Response label mapped to ``(chain, draw, row[, event])``.
        """
        names = self._selection(resp)
        prepared = self._prepare(
            new_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            data2=data2,
        )

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Sample every response from one trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Predicted responses keyed by response.
            """
            if self.rescor:
                value = row_binding(
                    response_sample(recorded, JOINT_RESPONSE),
                    prepared.rows[names[0]],
                    (len(self.response_names),),
                )
                return {
                    name: value[..., self.response_names.index(name)] for name in names
                }
            result: dict[str, torch.Tensor] = {}
            for name in self.response_names:
                response = self.emission.responses[name]
                site = response.names[_qvr_name(response.formula.response_name)]
                value = row_binding(
                    response_sample(recorded, site),
                    prepared.rows[name],
                    response.formula.response_layout.event_shape,
                )
                if name in names:
                    result[name] = value
            return result

        return self._replay(
            prepared, operation, num_samples=num_samples, seed=seed, stream=1
        )

    def log_likelihood(
        self,
        *,
        resp: ResponseSelection = None,
        new_data: IntoDataFrame | None = None,
        re_formula: str | None = None,
        allow_new_levels: bool = False,
        sample_new_levels: NewLevelPolicy = "uncertainty",
        num_samples: int | None = None,
        seed: int = 0,
        data2: Mapping[str, AutocorMatrix] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return pointwise log-likelihoods with brms observation units.

        Under residual correlation the unit is one row's joint vector event;
        a strict subset of responses is scored by the exact marginal of that
        event (a multivariate Normal or Student law over the selected
        components). Without residual correlation a selection of one
        response gives its own pointwise scores, and a selection of several
        sums their scores per row, as brms ``log_lik`` does.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        new_data : IntoDataFrame or None
            Fresh predictors and recorded outcomes of the selected responses,
            or None for the training rows.
        re_formula : str or None
            Varying-effect selector applied to every response.
        allow_new_levels : bool
            Permit new grouping identities.
        sample_new_levels : str
            Existing new-level policy, drawn once per shared grouping axis.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for posterior and group-effect streams.
        data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
            Matrices of ``sar`` and ``fcor`` terms over the rows of
            ``new_data`` (brms ``newdata2``).

        Returns
        -------
        dict[str, torch.Tensor]
            One entry keyed by the selected labels joined by ``"__"``, shaped
            ``(chain, draw, row)``. Likelihood weights are excluded.

        Raises
        ------
        ValueError
            If summed responses have different row counts.
        """
        names = self._selection(resp)
        key = "__".join(names)
        prepared = self._prepare(
            new_data,
            recorded=frozenset(names),
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            require_trials=False,
            scoring=True,
            data2=data2,
        )

        def recorded_rows(scores: torch.Tensor) -> torch.Tensor:
            """Report missing scores where a selected ``mi()`` value is missing.

            brms scores a training row of a joint residual event as NA when
            a selected response has no recorded value there.

            Parameters
            ----------
            scores : torch.Tensor
                Pointwise scores on ``(chain, draw, row)``.

            Returns
            -------
            torch.Tensor
                The scores with NaN at rows lacking a recorded value.
            """
            if new_data is not None:
                return scores
            for name in names:
                spec = self.missing_specs.get(name)
                if spec is None:
                    continue
                absent = torch.ones(scores.shape[-1], dtype=torch.bool)
                absent[list(spec.observed_positions)] = False
                scores = scores.masked_fill(absent.to(scores.device), float("nan"))
            return scores

        if self.rescor and names == self.response_names:
            shape = draw_count(self.posterior, num_samples)
            draws = self.draws(num_samples=num_samples, seed=seed)
            with torch.no_grad():
                joint = collect_log_likelihood(
                    replay(
                        cast(BayesianFit, self),
                        draws,
                        shape,
                        context=prepared.context,
                        score_observations=True,
                    ),
                    {JOINT_RESPONSE},
                    shape,
                )
            return {key: recorded_rows(joint[JOINT_RESPONSE])}
        if self.rescor:
            indices = [self.response_names.index(name) for name in names]
            outcome = prepared.context.observations[JOINT_RESPONSE][..., indices]

            def marginal(recorded: Trace) -> dict[str, torch.Tensor]:
                """Score the selected components by their exact marginal law.

                Parameters
                ----------
                recorded : Trace
                    One replayed joint trace.

                Returns
                -------
                dict[str, torch.Tensor]
                    Marginal pointwise scores.
                """
                distribution = response_distribution(recorded, JOINT_RESPONSE)
                if not isinstance(
                    distribution,
                    torch.distributions.MultivariateNormal | MultivariateStudentT,
                ):
                    raise TypeError("a residual-correlation event must be elliptical")
                selector = torch.tensor(indices, device=outcome.device)
                loc = distribution.loc.index_select(-1, selector)
                factor = distribution.scale_tril
                # The marginal of a Normal or Student vector over selected
                # coordinates keeps its location and scale submatrix.
                shape = (
                    (factor @ factor.transpose(-1, -2))
                    .index_select(-1, selector)
                    .index_select(-2, selector)
                )
                law: torch.distributions.Distribution = (
                    MultivariateStudentT(
                        distribution.df, loc, scale_tril=torch.linalg.cholesky(shape)
                    )
                    if isinstance(distribution, MultivariateStudentT)
                    else torch.distributions.MultivariateNormal(
                        loc, scale_tril=torch.linalg.cholesky(shape)
                    )
                )
                return {
                    key: row_binding(
                        law.log_prob(outcome.to(loc.dtype)), prepared.rows[names[0]]
                    )
                }

            scores = self._replay(
                prepared, marginal, num_samples=num_samples, seed=seed, stream=2
            )
            return {key: recorded_rows(scores[key])}
        training = new_data is None and re_formula is None
        # On training rows an mi() response is scored on its recorded events;
        # brms reports missing scores (NA) at its missing rows.
        events = {
            name: self.missing_specs[name]
            for name in names
            if training
            and name in self.missing_specs
            and (
                self.missing_specs[name].standard_error is not None
                or self.missing_specs[name].latent_positions
            )
        }
        sites = {
            name: f"mi_{name}__observed"
            if name in events
            else self.emission.responses[name].names[
                _qvr_name(self.emission.responses[name].formula.response_name)
            ]
            for name in names
        }
        counts = {prepared.rows[name] for name in names}
        if len(counts) != 1:
            raise ValueError(
                "summing log-likelihoods across responses requires equal row counts; "
                "select one response with resp"
            )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        with torch.no_grad():
            scores = collect_log_likelihood(
                replay(
                    cast(BayesianFit, self),
                    draws,
                    shape,
                    context=prepared.context,
                    score_observations=True,
                ),
                set(sites.values()),
                shape,
            )
        for name in names:
            # A non-factorized residual structure scores brms's leave-one-out
            # conditionals, as the single-response fit does.
            if name in events or not autocor_joint(
                self.emission.responses[name].lens._family
            ):
                continue
            with torch.no_grad():
                scores |= autocor_log_likelihood(
                    replay(
                        cast(BayesianFit, self),
                        draws,
                        shape,
                        context=prepared.context,
                        score_observations=True,
                    ),
                    sites[name],
                    shape,
                )
        for name, spec in events.items():
            recorded = scores[sites[name]]
            full = recorded.new_full(
                (*recorded.shape[:2], prepared.rows[name]), float("nan")
            )
            positions = torch.as_tensor(spec.observed_rows, dtype=torch.long)
            included = {row: index for index, row in enumerate(spec.rows.positions)}
            full[..., [included[int(row)] for row in positions]] = recorded
            scores[sites[name]] = full
        return {key: sum((scores[site] for site in sites.values()), torch.zeros(()))}

    def observed(self, *, resp: ResponseSelection = None) -> dict[str, torch.Tensor]:
        """Return the training outcomes of the selected responses.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.

        Returns
        -------
        dict[str, torch.Tensor]
            Native response carriers keyed by response.
        """
        return {
            name: response_tensor(
                self.emission.responses[name].formula, self.prepared[name].family
            )
            for name in self._selection(resp)
        }

    def analysis(
        self,
        *,
        resp: ResponseSelection = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> PosteriorAnalysis:
        """Prepare reusable summaries and predictive comparisons.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Responses whose pointwise scores enter the analysis, with the
            units of `log_likelihood`.
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        PosteriorAnalysis
            Native draws, pointwise scores, outcomes, and sampler statistics.
        """
        shape = draw_count(self.posterior, num_samples)
        return PosteriorAnalysis(
            samples=self.draws(num_samples=num_samples, seed=seed),
            log_likelihood=self.log_likelihood(
                resp=resp, num_samples=num_samples, seed=seed
            ),
            observed_data=self.observed(resp=resp),
            sample_stats=_sample_stats(self.posterior, shape[1]),
            kind=_kind(self.posterior),
        )

    @overload
    def conditional_effects(
        self,
        grid: PredictionGrid,
        *,
        resp: ResponseSelection = ...,
        method: EffectMethod = ...,
        mc_samples: int | None = ...,
        **options: Unpack[JointConditionalEffectsOptions],
    ) -> dict[str, ConditionalEffects]: ...

    @overload
    def conditional_effects(
        self,
        grid: str | Sequence[str] | None = ...,
        *,
        resp: ResponseSelection = ...,
        method: EffectMethod = ...,
        mc_samples: int | None = ...,
        **options: Unpack[JointConditionalEffectsOptions],
    ) -> BrmsConditionalEffects: ...

    def conditional_effects(
        self,
        grid: PredictionGrid | str | Sequence[str] | None = None,
        *,
        resp: ResponseSelection = None,
        method: EffectMethod = "epred",
        mc_samples: int | None = None,
        **options: Unpack[JointConditionalEffectsOptions],
    ) -> dict[str, ConditionalEffects] | BrmsConditionalEffects:
        """Predict one grid jointly and retain each response's effects.

        Without a grid, or with brms ``effects``, this computes brms's
        automatic conditional effects of every selected response
        (`quivers.formulas._brms_effects.brms_conditional_effects`).

        Parameters
        ----------
        grid : PredictionGrid or str or Sequence[str] or None
            Effect coordinates and reference rows from
            `quivers.diagnostics.prediction_grid`, holding every response's
            predictors; or brms ``effects`` (None for brms's defaults).
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        method : str
            ``epred``, ``linpred``, ``predict``, or brms's
            ``posterior_*`` names.
        mc_samples : int or None
            Monte Carlo budget of expectations without a closed mean.
        **options : JointConditionalEffectsOptions
            Varying-effect selection, new-level policy, draw count, and seed;
            without a grid, brms's ``conditional_effects`` arguments.

        Returns
        -------
        dict[str, ConditionalEffects] or BrmsConditionalEffects
            Effects keyed by response, or brms's labeled effect frames.

        Raises
        ------
        ValueError
            If the method is unknown or mc_samples is given outside epred.
        """
        # brms automatic conditional effects.
        if not isinstance(grid, PredictionGrid):
            if mc_samples is not None:
                raise ValueError("mc_samples applies only to explicit grids")
            return automatic_conditional_effects(
                self,
                grid,
                method=method,
                parameter="mu",
                transform=False,
                resp=resp,
                options=options,
            )
        prediction = joint_grid_options(options)
        short = method.removeprefix("posterior_")
        if short not in {"epred", "linpred", "predict"}:
            raise ValueError("method must be epred, linpred, or predict")
        method = cast(Literal["epred", "linpred", "predict"], short)
        if method != "epred" and mc_samples is not None:
            raise ValueError("mc_samples applies only to epred effects")
        frame = grid.to_frame()
        request = cast(JointEffectOptions, prediction)
        if method == "epred":
            predictions = self.posterior_epred(
                resp=resp, new_data=frame, mc_samples=mc_samples, **request
            )
        elif method == "linpred":
            predictions = self.posterior_linpred(resp=resp, new_data=frame, **request)
        else:
            predictions = self.posterior_predict(resp=resp, new_data=frame, **request)
        return {
            name: grid.result(
                value,
                kind=_kind(self.posterior),
                method=method,
                labels=self._response_fit(name).predictor_labels("mu")
                if method == "linpred"
                else self._response_fit(name).category_labels,
            )
            for name, value in predictions.items()
        }

    def prior_draws(
        self, *, num_samples: int = 100, seed: int = 0
    ) -> PriorDraws | MCMCResult:
        """Draw the joint prior with the retained designs.

        Parameters
        ----------
        num_samples : int
            Independent ancestral draws.
        seed : int
            Isolated seed.

        Returns
        -------
        PriorDraws or MCMCResult
            Constrained prior samples.
        """
        return prior_draws(
            self.program,
            torch.zeros(1, 1),
            self.observations,
            num_samples=num_samples,
            seed=seed,
        )

    def prior_view(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> MultivariateFit:
        """Reuse every result method with joint prior draws.

        Parameters
        ----------
        num_samples : int or None
            Fresh draw count, or None to reuse retained prior samples.
        seed : int
            Isolated seed of fresh draws.

        Returns
        -------
        MultivariateFit
            A view sharing the fitted model with the prior as its posterior.
        """
        prior = (
            self.prior
            if self.prior is not None and num_samples is None
            else self.prior_draws(
                num_samples=100 if num_samples is None else num_samples, seed=seed
            )
        )
        return self.with_(
            posterior=prior,
            prior=prior,
            sample_prior="only",
            losses=(),
            response_fits=dict(self.response_fits),
        )

    def hypothesis(
        self,
        expression: str | ContrastExpression | Sequence[str] | Mapping[str, str],
        *,
        prior: bool = False,
        num_samples: int | None = None,
        num_prior_samples: int | None = None,
        seed: int = 0,
        null: float | int | torch.Tensor = 0.0,
        probs: Sequence[float] = (0.025, 0.975),
        robust: bool = False,
        point: PointNullSpec | None = None,
        name: str = "contrast",
        class_: str | None = "b",
        group: str | None = None,
        scope: Literal["standard", "ranef", "coef"] = "standard",
        alpha: float = 0.05,
    ) -> HypothesisSummary | HypothesisTable:
        """Test a brms hypothesis string, or a contrast over brms names.

        Parameters
        ----------
        expression : str or ContrastExpression or Sequence[str] or Mapping[str, str]
            brms hypothesis strings ``"left (= | < | >) right"``, whose
            names take the class prefix (``"y1_x > y2_x"`` tests
            ``b_y1_x - b_y2_x``); or a QVR expression over the names of
            `variables`, for instance ``"b_y1_x - b_y2_x"`` or
            ``"rescor__y1__y2"``.
        prior : bool
            Also evaluate the contrast on joint prior draws.
        num_samples, num_prior_samples : int or None
            Posterior and prior draw counts.
        seed : int
            Isolated seed.
        null : float or int or torch.Tensor
            Comparison value.
        probs : Sequence[float]
            Quantile probabilities.
        robust : bool
            Use median and scaled MAD.
        point : PointNullSpec or None
            Optional point-density comparison; requires prior=True.
        name : str
            Output contrast label.
        class_ : str or None
            Class prefix of the names in a brms hypothesis string; None
            reads them as `variables` reports them.
        group : str or None
            Grouping factor of a brms hypothesis string.
        scope : {"standard", "ranef", "coef"}
            Scope of a brms hypothesis string.
        alpha : float
            brms ``alpha`` of a hypothesis string.

        Returns
        -------
        HypothesisSummary or HypothesisTable
            brms's table for a hypothesis string; posterior summaries and
            region evidence for a contrast expression.

        Raises
        ------
        ValueError
            If point evidence or a prior count is requested without prior,
            or a hypothesis string is malformed or names an unknown
            parameter.
        """
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
        if point is not None and not prior:
            raise ValueError("point-density evidence requires prior=True")
        if num_prior_samples is not None and not prior:
            raise ValueError("num_prior_samples requires prior=True")
        samples = self.variables(num_samples=num_samples, seed=seed)
        prior_samples = (
            self.prior_view(
                num_samples=num_prior_samples, seed=(seed + 1) % 2**64
            ).variables()
            if prior
            else None
        )
        return _contrast_hypothesis(
            samples,
            cast("str | ContrastExpression", expression),
            prior=prior_samples,
            null=null,
            probs=probs,
            robust=robust,
            point=point,
            name=name,
            kind=_kind(self.posterior),
        )

    def update(
        self, data: IntoDataFrame | None = None, **overrides: object
    ) -> MultivariateFit:
        """Refit retained or replacement rows through the original recipe.

        Parameters
        ----------
        data : IntoDataFrame or None
            Original-frame snapshot by default, or replacement rows.
        **overrides : object
            Overrides of the canonical `fit` arguments, including the joint
            formula, priors, method, budgets, or seed.

        Returns
        -------
        MultivariateFit
            A fresh fit.

        Raises
        ------
        ValueError
            If no recipe was retained.
        """
        return cast(
            MultivariateFit, update_fit(cast(BayesianFit, self), data, overrides)
        )

    def refit_callback(
        self,
        *,
        method_factory: Callable[[Mapping[str, object], int], Mapping[str, object]],
        partition: Callable[[IntoDataFrame, tuple[int, ...]], IntoDataFrame]
        | None = None,
        num_samples: int | None = None,
        **prediction: object,
    ) -> Callable[[FoldSplit], FoldPrediction]:
        """Adapt the retained rows to explicit refits for cross-validation.

        As in brms ``kfold``, each fold refits the joint model on every
        original row outside the held-out rows, so each response applies its
        own ``subset()`` within them, and scores the held-out rows with
        `log_likelihood`, whose units follow brms ``log_lik``. Fold indices
        address the rows included by the scored responses; with ``subset()``
        those are the included original rows, mapped back once.
        `PosteriorAnalysis.reloo` uses the same callback for exact
        leave-one-out refits.

        Parameters
        ----------
        method_factory : Callable
            Fold-specific canonical fit settings and seed.
        partition : Callable or None
            Optional nonmutating row selection.
        num_samples : int or None
            Draws per chain for held-out scoring.
        **prediction : object
            Further `log_likelihood` options such as ``resp`` or
            ``allow_new_levels``.

        Returns
        -------
        Callable[[FoldSplit], FoldPrediction]
            Callback for `cross_validate`.

        Raises
        ------
        ValueError
            If no recipe was retained, the adapter's own options are given,
            the original rows changed, or scored responses include different
            rows (brms ``log_lik`` sums only responses with equal rows).
        """
        recipe = self.refit_recipe
        if recipe is None:
            raise ValueError("this fit has no retained refit recipe")
        if forbidden := set(prediction) & {"new_data", "seed", "num_samples"}:
            raise ValueError(
                f"the refit adapter owns prediction arguments {sorted(forbidden)}"
            )
        inspect.signature(self.log_likelihood).bind(**prediction)
        scoring = cast(JointScoreOptions, dict(prediction))
        size = len(nw.from_native(recipe.data, eager_only=True))
        scored = self._selection(scoring.get("resp"))
        selections = {
            self.emission.responses[name].formula.modifiers.subset for name in scored
        }
        if len(selections) != 1:
            raise ValueError(
                "scored responses include different subset() rows; select one "
                "response with resp, as brms log_lik sums only equal rows"
            )
        (selection,) = selections
        if (selection.size if selection is not None else size) != size:
            raise ValueError(
                "retained original frame no longer has the fitted row layout"
            )
        included = selection.positions if selection is not None else tuple(range(size))
        # With mi() terms, held-out units are rows whose mi() values used by
        # the scored responses are recorded; brms scores other rows as NA.
        variables = {
            self.missing.responses[name].variable
            for name in scored
            if name in self.missing.responses
        } | {
            self.missing.responses[source].variable
            for item in self.missing.inputs
            if item.consumer in scored
            for source in item.sources
        }
        if variables:
            original = nw.from_native(recipe.data, eager_only=True).to_pandas()
            recorded = np.ones(size, dtype=bool)
            for variable in variables:
                recorded &= np.isfinite(np.asarray(original[variable], dtype=float))
            included = tuple(row for row in included if recorded[row])

        def select(data: IntoDataFrame, rows: tuple[int, ...]) -> IntoDataFrame:
            """Select original rows, preserving the dataframe backend.

            Parameters
            ----------
            data : IntoDataFrame
                Original rows.
            rows : tuple[int, ...]
                Row positions.

            Returns
            -------
            IntoDataFrame
                Selected rows.
            """
            return nw.from_native(data, eager_only=True)[list(rows), :].to_native()

        def score(
            result: MultivariateFit, rows: IntoDataFrame, seed: int
        ) -> Mapping[str, torch.Tensor]:
            """Score held-out rows with the fold's posterior.

            Parameters
            ----------
            result : MultivariateFit
                Fold fit.
            rows : IntoDataFrame
                Held-out rows.
            seed : int
                Fold seed.

            Returns
            -------
            Mapping[str, torch.Tensor]
                Pointwise held-out scores.
            """
            return result.log_likelihood(
                new_data=rows,
                seed=seed,
                num_samples=num_samples,
                **scoring,
            )

        callback = refit_callback(
            recipe,
            select if partition is None else partition,
            score,
            method_factory=method_factory,
        )

        def invoke(split: FoldSplit) -> FoldPrediction:
            """Map included-row folds to original rows and refit once.

            Parameters
            ----------
            split : FoldSplit
                Fold over the scored responses' included rows.

            Returns
            -------
            FoldPrediction
                Refitted result and held-out pointwise scores.
            """
            validate_split_rows(split, len(included))
            omitted = original_positions(split.omitted, included)
            removed = set(omitted)
            return callback(
                split.with_(
                    train=tuple(row for row in range(size) if row not in removed),
                    test=original_positions(split.test, included),
                    omitted=omitted,
                )
            )

        return invoke

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the canonical joint QVR source.

        Parameters
        ----------
        path : str or Path
            Destination.

        Returns
        -------
        Path
            Written path.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out


def _brms_name(text: str) -> str:
    """Remove the characters brms drops from variable names.

    Parameters
    ----------
    text : str
        Variable name.

    Returns
    -------
    str
        Letters and digits only.
    """
    return "".join(character for character in text if character.isalnum())


def _joint_observations(
    emission: _JointEmission, prepared: Mapping[str, _PreparedFormula]
) -> dict[str, torch.Tensor]:
    """Namespace the same canonical response binder used by ordinary fitting.

    Parameters
    ----------
    emission : _JointEmission
        The composed emission with its response namespaces.
    prepared : Mapping[str, _PreparedFormula]
        Ordinary preparation of each response.

    Returns
    -------
    dict[str, torch.Tensor]
        Joint inputs and every namespaced response observation and input.
    """
    observed = {
        step.vars[0]
        for statement in emission.module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, ObserveStep)
    }
    values = dict(emission.fixed_inputs)
    for name, response in emission.responses.items():
        local = _formula_observations(
            prepared[name], formula=response.formula, lens=response.lens
        )
        if JOINT_RESPONSE in observed:
            # The residual-correlation event replaces each scalar observation.
            local.pop(response_input_name(response.formula), None)
        values.update(
            {response.names.get(key, key): value for key, value in local.items()}
        )
    return values


class JointComposition(ValueModel):
    """One composed joint program before compilation.

    Parameters
    ----------
    prepared : dict[str, _PreparedFormula]
        Ordinary preparation of each response, with ``mi()`` syntax removed.
    emission : _JointEmission
        The composed emission, after the ``mi()`` rewrite.
    observations : dict[str, torch.Tensor]
        Every runtime input and observation of the program.
    missing : MissingPlan
        ``mi()`` declarations.
    specs : dict[str, LatentResponseSpec]
        Recorded and latent coordinates of each ``mi()`` response.
    """

    prepared: dict[str, _PreparedFormula] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    emission: _JointEmission = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: dict[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    missing: MissingPlan = dx.field(
        default_factory=MissingPlan, opaque=True, extras=OPAQUE_BY_VALUE
    )
    specs: dict[str, LatentResponseSpec] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )


def compose_joint(
    specification: MultivariateFormula,
    data: IntoDataFrame,
    *,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    data2: Mapping[str, AutocorMatrix] | None = None,
    caller: str = "fit",
) -> JointComposition:
    """Prepare every response and compose one joint program.

    ``mi()`` syntax is removed before the ordinary preparation, the design
    columns of ``mi()`` predictors become symbolic columns, and the composed
    program is rewritten once (`apply_missing`).

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    data : IntoDataFrame
        Data holding every response and predictor column.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default priors of each ordinary response preparation.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known group covariances, keyed by grouping factor.
    priors : PriorInput or None
        One joint catalogue of prior rules.
    reparameterize : {"centered", "noncentered"}
        Group-effect parameterization.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices available to every response, as brms ``data2``.
    caller : str
        Entry point named in located errors.

    Returns
    -------
    JointComposition
        Preparations, emission, bindings, and ``mi()`` plans.
    """
    plain, declared, uses, measurements, sources = missing_specification(specification)
    original = nw.from_native(data, eager_only=True).to_pandas()
    frame = missing_frame(original, declared) if declared else data
    # me() hyperparameter rules are resolved on the me() targets.
    rules = normalize_priors(priors).rules
    measurement_rules = PriorSet(
        rules=tuple(rule for rule in rules if rule.class_ in MEASUREMENT_CLASSES)
    )
    priors = PriorSet(
        rules=tuple(rule for rule in rules if rule.class_ not in MEASUREMENT_CLASSES)
    )
    # brms na_omit: one row decision for every response together.
    rows = joint_complete_rows(
        nw.from_native(frame, eager_only=True).to_pandas(),
        tuple(
            (
                model_texts(
                    response.formula, (response.distributional, response.nonlinear)
                ),
                model_columns(
                    response.weights,
                    response.decision,
                    response.binomial_trials,
                    response.thresholds_by,
                    # brms bf(unused = ): kept columns join row retention.
                    *response.unused,
                ),
            )
            for response in (plain.responses[name] for name in plain.response_names)
        ),
    )
    prepared: dict[str, _PreparedFormula] = {}
    for name in plain.response_names:
        response = plain.responses[name]
        prepared[name] = _prepare_formula(
            response.formula,
            data=frame,
            rows=rows,
            family=response.resolved_family,
            distributional=response.distributional,
            nonlinear=response.nonlinear,
            fixed_parameters=response.fixed_parameters,
            weights=response.weights,
            decision=response.decision,
            binomial_trials=response.binomial_trials,
            thresholds_by=response.thresholds_by,
            mixture_components=response.mixture_components,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            reparameterize=reparameterize,
            data2=data2,
            caller=caller,
            # brms bf(center, decomp) and the fit's default prior policy.
            default_priors=response.default_priors,
            center=response.center,
            decomp=response.decomp,
            cmc=response.cmc,
            sparse=response.sparse,
            unused=response.unused,
        )
        # brms scales default priors on the recorded values of an mi()
        # response, not on its placeholders.
        if name in declared:
            subset = prepared[name].formula.modifiers.subset
            positions = (
                np.asarray(subset.positions)
                if subset is not None
                else np.arange(len(original))
            )
            recorded = original[declared[name].variable].to_numpy(dtype=float)
            prepared[name].lens._recorded_rows = np.isfinite(recorded[positions])
    components = {name: (part.lens, part.formula) for name, part in prepared.items()}
    inputs = []
    for name, used in uses.items():
        lens, formula, columns = symbolize(
            name,
            prepared[name],
            nw.from_native(frame, eager_only=True).to_pandas(),
            used,
            sources[name],
        )
        components[name] = (lens, formula)
        inputs.extend(columns)
    emission = emit_joint(
        components, priors=normalize_priors(priors), rescor=specification.rescor
    )
    plan = MissingPlan(
        responses=declared,
        inputs=tuple(inputs),
        measurements=measurements,
        mecor={
            name: specification.responses[name].mecor for name in plain.response_names
        },
    )
    specs = latent_specs(plan, original, prepared) if not plan.empty else {}
    observations = _joint_observations(emission, prepared)
    if not plan.empty:
        lens = next(iter(prepared.values())).lens
        defaults = {
            "meanme": fixed_prior,
            "sdme": random_scale_prior,
            "corme": correlation_prior,
        }
        targets: dict[str, PriorTarget] = {}

        def prior(site: str, kind: str, coef: str | None) -> SampleStep:
            """Draw one me() hyperparameter from its rule or default.

            Parameters
            ----------
            site : str
                Latent site.
            kind : str
                brms class.
            coef : str or None
                brms coefficient label.

            Returns
            -------
            SampleStep
                The prior draw.
            """
            target = PriorTarget(
                site=site,
                class_=kind,
                coef=coef,
                lower=0.0 if kind == "sdme" else None,
                event_rank=2 if kind == "corme" else 0,
            )
            targets[site] = target
            # brms gives meanme and sdme flat priors.
            if lens._default_priors == "brms" and kind in {"meanme", "sdme"}:
                return flat_prior_step(site, lower=target.lower, upper=None, index=None)
            step = lens._parse_prior(site, defaults[kind])
            return step

        emission, extra, removed = apply_missing(
            emission,
            plan,
            specs,
            original,
            consumer_rows={
                name: prepared[name].formula.modifiers.subset
                or RowSelection(
                    size=len(original), positions=tuple(range(len(original)))
                )
                for name in plain.response_names
            },
            prior=prior,
        )
        if measurement_rules.rules:
            catalogue = [
                target for target in emission.prior_targets if target.site in targets
            ]
            selected = resolve_priors(measurement_rules, catalogue)
            emission = _replace_priors(emission, catalogue, selected, lens)
        observations = {
            key: value for key, value in observations.items() if key not in removed
        } | extra
        # Flat me() hyperparameters may be the module's first flat sites.
        if improper_sites(emission.module) and FLAT_FAMILY not in emission.families:
            emission = replace(
                emission, families=emission.families.with_catalog(FLAT_CATALOG)
            )
    return JointComposition(
        prepared=prepared,
        emission=emission,
        observations=observations,
        missing=plan,
        specs=specs,
    )


#: brms prior classes of me() hyperparameters.
MEASUREMENT_CLASSES: frozenset[str | None] = frozenset({"meanme", "sdme", "corme"})


def _replace_priors(
    emission: _JointEmission,
    catalogue: Sequence[PriorTarget],
    selected: Mapping[str, PriorSpec],
    lens: FormulaToQVRModule,
) -> _JointEmission:
    """Replace me() hyperparameter priors by their selected rules.

    Parameters
    ----------
    emission : _JointEmission
        Emission with default me() priors.
    catalogue : Sequence[PriorTarget]
        The me() targets.
    selected : Mapping[str, PriorSpec]
        Resolved rules by site.
    lens : FormulaToQVRModule
        Lens whose prior parser reads the rules.

    Returns
    -------
    _JointEmission
        The emission with replaced prior draws, nominal axes retained.
    """
    by_site = {target.site: target for target in catalogue}
    replacements = {
        site: ordinary_prior_step(by_site[site], rule, draw=lens._parse_prior)
        for site, rule in selected.items()
    }
    module = emission.module.with_(
        statements=tuple(
            statement.with_(
                draws=tuple(
                    replacements[step.vars[0]].with_(index=step.index)
                    if isinstance(step, SampleStep) and step.vars[0] in replacements
                    else step
                    for step in statement.draws
                )
            )
            if isinstance(statement, ProgramDecl)
            else statement
            for statement in emission.module.statements
        )
    )
    return replace(emission, module=module)


def _prepare_joint(
    specification: MultivariateFormula,
    data: IntoDataFrame,
    *,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    data2: Mapping[str, AutocorMatrix] | None = None,
    caller: str = "fit",
) -> tuple[JointComposition, MonadicProgram]:
    """Compose and compile exactly one joint program.

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    data : IntoDataFrame
        Data holding every response and predictor column.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default priors of each ordinary response preparation.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known group covariances, keyed by grouping factor.
    priors : PriorInput or None
        One joint catalogue of prior rules.
    reparameterize : {"centered", "noncentered"}
        Group-effect parameterization.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices available to every response, as brms ``data2``.
    caller : str
        Entry point named in located errors.

    Returns
    -------
    tuple[JointComposition, MonadicProgram]
        The composition and its compiled program.

    Raises
    ------
    TypeError
        If the joint emission does not compile to one MonadicProgram.
    """
    composition = compose_joint(
        specification,
        data,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances=group_covariances,
        priors=priors,
        reparameterize=reparameterize,
        data2=data2,
        caller=caller,
    )
    emission = composition.emission
    program = Compiler(emission.module, families=emission.families).compile().morphism
    if not isinstance(program, MonadicProgram):
        raise TypeError("joint formula emission must compile to one MonadicProgram")
    return composition, program


def joint_module(
    specification: MultivariateFormula,
    data: IntoDataFrame,
    *,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    data2: Mapping[str, AutocorMatrix] | None = None,
    caller: str = "formula_to_qvr",
) -> tuple[Module, tuple[PriorTarget, ...]]:
    """Emit the joint module without compiling or fitting it.

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    data : IntoDataFrame
        Training rows.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default priors.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known group covariances.
    priors : PriorInput or None
        Joint prior rules.
    reparameterize : {"centered", "noncentered"}
        Group coordinate policy.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices available to every response.
    caller : str
        Entry point named in located errors.

    Returns
    -------
    tuple[Module, tuple[PriorTarget, ...]]
        The joint module and its prior targets.
    """
    composition = compose_joint(
        specification,
        data,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances=group_covariances,
        priors=priors,
        reparameterize=reparameterize,
        data2=data2,
        caller=caller,
    )
    return composition.emission.module, composition.emission.prior_targets


def joint_prior(module: Module, targets: Sequence[PriorTarget]) -> dict[str, str]:
    """Read effective joint priors from the emitted module.

    Parameters
    ----------
    module : Module
        Joint module.
    targets : Sequence[PriorTarget]
        Overrideable sites.

    Returns
    -------
    dict[str, str]
        Namespaced site names and canonical prior calls.
    """
    names = {target.site for target in targets}
    calls: dict[str, str] = {}
    for statement in module.statements:
        if not isinstance(statement, ProgramDecl):
            continue
        for step in statement.draws:
            if isinstance(step, SampleStep) and step.vars[0] in names:
                calls[step.vars[0]] = draw_call_to_source(
                    step.morphism, step.args, static_args=step.static_args
                )
    return calls


def fit_joint(
    specification: MultivariateFormula,
    data: IntoDataFrame,
    *,
    refit_recipe: RefitRecipe[IntoDataFrame, Mapping[str, object], MultivariateFit]
    | None = None,
    **options: Unpack[JointFitOptions],
) -> MultivariateFit:
    """Run one joint program through the shared inference seam of `fit`.

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    data : IntoDataFrame
        Data holding every response and predictor column.
    refit_recipe : RefitRecipe or None
        Recipe captured by `fit`.
    **options : JointFitOptions
        The `fit` keywords that apply to a joint formula, with their `fit`
        defaults. Initial values name namespaced sites and take precedence
        over the formula starts. ``snapshot_policy`` is consumed by the recipe.

    Returns
    -------
    MultivariateFit
        The retained joint posterior.
    """
    fixed_prior = options.get("fixed_prior", "Normal(0.0, 5.0)")
    random_scale_prior = options.get("random_scale_prior", "HalfNormal(1.0)")
    correlation_prior = options.get("correlation_prior", "LKJCholesky(1.0)")
    group_covariances = options.get("group_covariances")
    priors = options.get("priors")
    reparameterize = options.get("reparameterize", "noncentered")
    # brms algorithm, control, thin, init, and default priors.
    method, guide = resolve_algorithm(
        options.get("algorithm"), options.get("method", "nuts"), options.get("guide")
    )
    controls = InferenceControls(
        control=sampler_control(options.get("control")),
        thin=options.get("thin", 1),
        init=options.get("init"),
        init_r=options.get("init_r", 2.0),
        # Pathfinder arguments and brms cores.
        pathfinder=pathfinder_options(options.get("pathfinder")),
        cores=checked_cores(options.get("cores", 1)),
        chain_worker=None
        if refit_recipe is None
        else FormulaChainWorker(
            arguments=dict(refit_recipe.method), data=refit_recipe._data
        ),
    )
    if options.get("default_priors", "quivers") != "quivers":
        specification = specification.with_default_priors(
            options.get("default_priors", "quivers")
        )
    if method == "svi" and any(
        options.get(name) is not None
        for name in ("num_warmup", "num_samples", "num_chains")
    ):
        raise ValueError(
            "fit: num_warmup, num_samples, and num_chains count MCMC iterations, "
            "and method='svi' runs none; give the optimization steps as num_steps"
        )
    if method not in ("svi", "pathfinder") and options.get("num_steps") is not None:
        raise ValueError(
            f"fit: num_steps counts SVI optimization steps, and method={method!r} "
            "runs MCMC; give num_warmup, num_samples, and num_chains"
        )
    # brms knots and drop_unused_levels reach every response.
    with formula_data_options(
        options.get("knots"), options.get("drop_unused_levels", True)
    ):
        composition, program = _prepare_joint(
            specification,
            data,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            priors=priors,
            reparameterize=reparameterize,
            data2=options.get("data2"),
        )
    prepared, emission = composition.prepared, composition.emission
    observations = composition.observations
    if options.get("sample_prior", "no") != "no":
        require_proper(emission.module, f"sample_prior={options.get('sample_prior')!r}")
    starts: dict[str, torch.Tensor | float] = {}
    for name, part in prepared.items():
        response = emission.responses[name]
        starts.update(
            {
                response.names.get(key, key): value
                for key, value in _initial_values(
                    part, part.lens._fixed_parameters, None
                ).items()
            }
        )
    starts.update(options.get("initial_values") or {})
    result = _run_inference(
        program,
        observations,
        method=method,
        seed=options.get("seed", 0),
        initial_values=starts,
        num_steps=options.get("num_steps"),
        num_warmup=options.get("num_warmup"),
        num_samples=options.get("num_samples"),
        num_chains=options.get("num_chains"),
        learning_rate=options.get("learning_rate", 1e-2),
        guide=guide,
        sample_prior=options.get("sample_prior", "no"),
        num_prior_samples=options.get("num_prior_samples", 100),
        prior_seed=options.get("prior_seed"),
        controls=controls,
    )
    return MultivariateFit(
        specification=specification,
        program=program,
        posterior=result.posterior,
        observations=observations,
        emission=emission,
        prepared=prepared,
        losses=result.losses,
        prior=result.prior,
        sample_prior=result.sample_prior,
        priors=normalize_priors(priors),
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        group_covariances={
            group: covariance
            for part in prepared.values()
            for group, covariance in part.lens.group_covariances.items()
        },
        reparameterize=reparameterize,
        preparation_data=data,
        refit_recipe=refit_recipe,
        missing=composition.missing,
        missing_specs=composition.specs,
        data2=dict(options.get("data2") or {}),
    )


def fit_entry(
    formula: MultivariateFormula,
    data: IntoDataFrame,
    options: Mapping[str, object],
    refit_recipe: RefitRecipe[IntoDataFrame, Mapping[str, object], BayesianFit],
) -> MultivariateFit:
    """Fit a joint formula from the canonical arguments captured by `fit`.

    Parameters
    ----------
    formula : MultivariateFormula
        The joint specification.
    data : IntoDataFrame
        The fit-owned original-row snapshot.
    options : Mapping[str, object]
        Every canonical `fit` argument other than the data.
    refit_recipe : RefitRecipe
        The recipe captured by `fit`.

    Returns
    -------
    MultivariateFit
        The fitted joint model.

    Raises
    ------
    ValueError
        If a response-level option is given to `fit` instead of `bf`.
    """
    keys = JointFitOptions.__annotations__
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(fit).parameters.items()
        if name not in keys and name not in {"formula", "data"}
    }
    misplaced = sorted(
        name
        for name, default in defaults.items()
        if name in options
        and not (
            options[name] is default
            or (isinstance(default, str) and options[name] == default)
        )
    )
    if misplaced:
        raise ValueError(
            f"fit: {misplaced!r} configure one response; give them to that "
            "response's bf() call in a joint formula"
        )
    return fit_joint(
        formula,
        data,
        # The recipe reruns `fit`, which returns a MultivariateFit for this formula.
        refit_recipe=cast(
            RefitRecipe[IntoDataFrame, Mapping[str, object], MultivariateFit],
            refit_recipe,
        ),
        **cast(
            JointFitOptions,
            {name: value for name, value in options.items() if name in keys},
        ),
    )


__all__: list[str] = []
