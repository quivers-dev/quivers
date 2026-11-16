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

import math
from dataclasses import dataclass
import numpy as np
from quivers._count_capacity import _MAX_EXACT_COUNT
from quivers.formulas._count_data import checked_integer_data
from pathlib import Path
from typing import Literal, Mapping, Unpack

import didactic.api as dx
from formulae.matrices import DesignMatrices
import narwhals as nw
import torch
from torch import nn
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.emit import module_to_source
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas.family import Family, families
from quivers.formulas.formula import (
    Formula,
    _formula_and_design,
    _qvr_name,
    formula_from_data,
)
from quivers.formulas._predictor_graph import (
    NonlinearExpression,
    PredictorGraph,
    build_predictor_graph,
    nonlinear_response_source,
    predictor_items,
)

from quivers.formulas._prediction import FormulaPredictionPlan
from quivers.formulas._response import count_totals, response_tensor
from quivers.formulas._response_modifiers import (
    ResponseDraws,
    ExpectationOptions,
    PredictionOptions,
)
from quivers.inference import (
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
    AutoNormalGuide,
    PriorDraws,
    PriorMode,
)
from quivers.inference._posterior import draw_count
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


from quivers.diagnostics.posterior import PosteriorAnalysis


class BayesianFit(ValueModel):
    """A fitted Bayesian regression: the compiled program, the
    parsed formula, the family, the user-supplied data, and the
    posterior samples.

    Attributes
    ----------
    formula : Formula
        Parsed formula IR.
    family : Family
        Response family used at compile time.
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
    fixed_parameters : dict[str, float]
        Auxiliary parameters bound to constants rather than sampled or regressed.
    distributional : Mapping[str, Formula]
        Evaluated auxiliary-parameter design matrices, retained so source
        inspection emits the fitted model without re-evaluating transforms.
    predictor_graph : PredictorGraph or None
        Named nonlinear designs, typed expressions, and numeric input contracts.
    prediction_plan : FormulaPredictionPlan or None
        Live fitted evaluators for new-data prediction. The versioned runtime
        plan is held by identity and explicitly refuses durable serialization.

    Fits compare by value over the formula, family, and settings, and
    by identity over the live objects: the program, the posterior, the
    observations mapping, the predictor, its data, and the prediction plan. Two fits are
    thus equal exactly when they share those objects and agree on
    everything else.
    """

    formula: Formula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: Family
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
    priors: dict[str, str] = dx.field(default_factory=dict)
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
    ) -> FormulaToQVRModule:
        """Reuse fitted compiler settings with optional specialized predictors.

        Parameters
        ----------
        distributional : Mapping[str, Formula] or None
            Specialized auxiliary metadata, or the fitted metadata.
        nonlinear_leaves : Mapping[str, Formula] or None
            Group-specialized designs for the same nonlinear graph.

        Returns
        -------
        FormulaToQVRModule
            A fresh lens retaining every fitted prior and compiler option.
        """
        return FormulaToQVRModule(
            self.family,
            fixed_prior=self.fixed_prior,
            random_scale_prior=self.random_scale_prior,
            correlation_prior=self.correlation_prior,
            user_priors=self.priors,
            reparameterize=self.reparameterize,
            binomial_trials=self.binomial_trials,
            thresholds_by=self.thresholds_by,
            mixture_components=self.mixture_components,
            predictor_name=self.predictor_name,
            distributional=self.distributional
            if distributional is None
            else distributional,
            fixed_parameters=self.fixed_parameters,
            predictor_graph=self.predictor_graph
            if nonlinear_leaves is None or self.predictor_graph is None
            else self.predictor_graph.with_(leaves=dict(nonlinear_leaves)),
        )

    @property
    def qvr_source(self) -> str:
        """Lazily emit the AST-equivalent ``.qvr`` source for display."""
        module, _ = self._formula_lens().forward(self.formula)
        return module_to_source(module)

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the AST-equivalent ``.qvr`` source to ``path`` and
        return the resulting `Path`.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out

    @property
    def category_labels(self) -> tuple[int | str, ...]:
        """Response-category labels in posterior probability-array order.

        Returns
        -------
        tuple[int, ...]
            Codes ``0..K-1`` for categorical and ordinal responses; empty
            for other families. Categorical coefficient arrays omit code zero.
        """
        if self.formula.response_layout.component_names:
            return self.formula.response_layout.component_names
        if self.formula.response_layout.category_levels:
            return self.formula.response_layout.category_levels
        if self.family.observe_family not in {
            "Categorical",
            "OrderedLogistic",
            "OrderedProbit",
        }:
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
        from quivers.formulas._posterior import posterior_draws

        return posterior_draws(self, num_samples, seed)

    def fixed_effects(
        self, *, parameter: str = "mu", num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Extract coefficients on the evaluated fixed-design columns.

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
        predictor, prefix = self._formula_predictor(parameter)
        return self._fixed_effects(
            self.draws(num_samples=num_samples, seed=seed), predictor, prefix
        )

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
        return self._random_effects(
            self.draws(num_samples=num_samples, seed=seed), shape, predictor, prefix
        )

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
        fixed = self._fixed_effects(draws, predictor, prefix)
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
        return varying

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
            contribution, and discrimination are excluded. Cumulative profiles
            are strictly ordered at every row.

        Raises
        ------
        ValueError
            If the family is not ordinal or a requested profile crosses.
        """
        from quivers.formulas._posterior import collect_draws, replay, row_binding
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

        def values():
            for recorded in replay(self, draws, shape, context=context):
                if width == 0:
                    yield torch.empty(context.rows, 0)
                    continue
                cuts = row_binding(
                    recorded.sites[
                        "row_cutpoints" if self.thresholds_by else "cutpoints"
                    ].value,
                    context.rows,
                    width,
                )
                if "eta_nonbaseline" in recorded.sites:
                    cuts = cuts - row_binding(
                        recorded.sites["eta_nonbaseline"].value, context.rows, width
                    )
                for term in self.formula.random_terms:
                    if term.column is not None and term.column.category_specific:
                        continue
                    group = _qvr_name(term.group)
                    latent = (
                        f"alpha_{group}"
                        if term.slope == "Intercept"
                        else f"beta_{group}_{_qvr_name(term.slope)}"
                    )
                    name = f"{latent}_per_row"
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
    ) -> torch.Tensor:
        """Replay a fitted linear predictor on training or fresh predictor rows.

        Parameters
        ----------
        parameter : str
            ``mu`` or a distributionally regressed auxiliary parameter.
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
        from quivers.formulas._posterior import (
            collect_draws,
            replay,
            row_binding,
        )
        from quivers.formulas._prediction import prepare_prediction

        graph = self.predictor_graph
        graph_parameter = graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys() | graph.roots.keys()
        )
        if not graph_parameter:
            self._formula_predictor(parameter)
        ordinal = self.family.ordinal is not None
        if transform and parameter == "mu" and ordinal:
            raise ValueError(
                "ordinal probabilities depend on thresholds; use posterior_epred"
            )
        retained_log_rate = (
            (self.family.name == "com_poisson" or self.family.baseline is not None)
            and self.family.location_link.name == "log"
            and parameter == "mu"
        )
        binding = (
            ("mu" if transform and not retained_log_rate else "eta")
            if parameter == "mu"
            else (parameter if transform else f"dpar_{parameter}_eta")
        )
        if graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys()
        ):
            binding = f"nlpar_{parameter}"
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
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows
        category_specific = parameter == "mu" and (
            any(column.category_specific for column in self.formula.fixed_columns)
            or any(
                term.column is not None and term.column.category_specific
                for term in self.formula.random_terms
            )
        )
        categories = self._predictor_width(parameter)
        if category_specific:
            categories = (
                len(self.category_labels)
                - 1
                - int(
                    self.family.ordinal is not None
                    and self.family.ordinal.kind == "hurdle_cumulative"
                )
            )

        def values():
            for recorded in replay(self, draws, shape, context=context):
                value = recorded.sites[binding].value
                if retained_log_rate and transform:
                    value = value.exp()
                if category_specific:
                    proportional = row_binding(value, rows).unsqueeze(-1)
                    specific = row_binding(
                        recorded.sites["eta_nonbaseline"].value, rows, categories
                    )
                    value = proportional + specific
                yield row_binding(value, rows, categories)

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
        from quivers.formulas._posterior import (
            collect_draws,
            replay,
            response_distribution,
            response_mean,
            response_rng,
            row_binding,
        )
        from quivers.formulas._prediction import prepare_prediction

        if component not in {"response", "decision"}:
            raise ValueError("expected response component must be response or decision")
        if component == "decision" and self.formula.modifiers.decision is None:
            raise ValueError(
                "decision probabilities require a decision response modifier"
            )
        if self.family.observe_family == "VonMises":
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
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows
        response = _qvr_name(self.formula.response_name)
        categories = (
            len(self.category_labels)
            if self.family.observe_family == "Categorical"
            or self.family.ordinal is not None
            else 0
        )
        if component == "decision":
            categories = 2

        def values():
            for recorded in replay(self, draws, shape, context=context):
                distribution = response_distribution(recorded, response)
                if component == "decision":
                    from quivers.continuous._wiener import Wiener

                    if not isinstance(distribution, Wiener):
                        raise TypeError(
                            "the response has no binary decision probabilities"
                        )
                    value = distribution.choice_probs
                elif self.family.name == "wiener":
                    from quivers.continuous._wiener import Wiener

                    assert isinstance(distribution, Wiener)
                    value = distribution.mean_response_time
                elif self.family.observe_family == "Categorical":
                    if not isinstance(distribution, torch.distributions.Categorical):
                        raise TypeError(
                            "the categorical response has no native class probabilities"
                        )
                    value = distribution.probs
                elif self.family.ordinal is not None:
                    from quivers.continuous._ordinal import OrdinalDistribution

                    if isinstance(distribution, OrdinalDistribution):
                        yield row_binding(
                            distribution.category_probs(), rows, categories
                        )
                        continue
                    codes = torch.arange(
                        categories, device=recorded.sites[response].value.device
                    )
                    value = (
                        distribution.log_prob(
                            codes.reshape(
                                categories, *([1] * len(distribution.batch_shape))
                            )
                        )
                        .exp()
                        .movedim(0, -1)
                    )
                else:
                    value = response_mean(distribution, mc_samples)
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"{self.family.observe_family} has an undefined or nonfinite response mean at a posterior draw"
                    )
                yield row_binding(
                    value, rows, self.formula.response_layout.event_shape or categories
                )

        with response_rng(seed + 2, (*draws.values(), *context.observations.values())):
            return collect_draws(values(), shape)

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
        from quivers.formulas._posterior import replay
        from quivers.inference._likelihood import collect_log_likelihood

        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        response = _qvr_name(self.formula.response_name)
        with torch.no_grad():
            likelihood = collect_log_likelihood(
                replay(self, draws, shape, score_observations=True), {response}, shape
            )
        return PosteriorAnalysis(
            samples=draws,
            log_likelihood=likelihood,
            observed_data={response: response_tensor(self.formula, self.family)},
            sample_stats=_sample_stats(self.posterior, shape[1]),
            kind="mcmc"
            if isinstance(self.posterior, MCMCResult)
            else "prior"
            if isinstance(self.posterior, PriorDraws)
            else "variational",
        )

    def log_likelihood(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Evaluate raw response log likelihoods on the fitted rows.

        Parameters
        ----------
        num_samples : int or None
            Maximum retained draws per chain, or fresh variational draws.
        seed : int
            Seed for variational draws; the caller's CPU RNG is preserved.

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
        from quivers.formulas._posterior import replay
        from quivers.inference._likelihood import collect_log_likelihood

        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        response = _qvr_name(self.formula.response_name)
        with torch.no_grad():
            return collect_log_likelihood(
                replay(self, draws, shape, score_observations=True), {response}, shape
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
        from quivers.formulas._posterior import (
            collect_draws,
            replay,
            row_binding,
            response_rng,
        )
        from quivers.formulas._prediction import prepare_prediction

        context = prepare_prediction(
            self,
            new_data,
            predictor_data,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            trials=trials,
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = context.rows
        response = _qvr_name(self.formula.response_name)

        def values():
            for recorded in replay(self, draws, shape, context=context):
                sampleable = recorded.sites[response].sampleable
                if sampleable is None:
                    raise TypeError("the response has no native sampling operation")
                # The native observation expands shared parameters before
                # sampling, so different rows receive independent draws.
                yield row_binding(
                    sampleable.raw_morphism.rsample(sampleable.input),
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
        self._formula_predictor(parameter)
        return (
            self.coefficient_labels
            if response is not None
            and response.kind == "logistic_normal"
            and parameter == "sigma"
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
        if response is None:
            return 0
        if parameter == "mu":
            return len(self.category_labels) - int(response.kind == "logistic_normal")
        return (
            len(self.category_labels) - 1
            if response.kind == "logistic_normal" and parameter == "sigma"
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
        if parameter == "mu":
            return self.formula, ""
        if parameter not in self.distributional:
            raise ValueError(
                f"parameter {parameter!r} has no fitted distributional predictor"
            )
        return self.distributional[parameter], f"dpar_{parameter}_"

    @staticmethod
    def _fixed_effects(
        draws: dict[str, torch.Tensor], predictor: Formula, prefix: str
    ) -> dict[str, torch.Tensor]:
        """Label native fixed-site draws with original design-column names.

        Parameters
        ----------
        draws : dict[str, torch.Tensor]
            Native site draws on chain and draw axes.
        predictor : Formula
            The evaluated design-column metadata.
        prefix : str
            Native coefficient-name prefix for the selected parameter.

        Returns
        -------
        dict[str, torch.Tensor]
            Original design-column labels mapped to native coefficient draws.
        """
        return {
            column.name: draws[
                prefix
                + ("intercept" if column.is_intercept else f"beta_{column.qvr_name}")
            ]
            for column in predictor.fixed_columns
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
        """
        from quivers.formulas._posterior import replay

        collected: dict[tuple[str, str], list[torch.Tensor]] = {
            (term.group, term.slope): [] for term in predictor.random_terms
        }
        response = self.family.response
        categorical = response is not None and (
            not prefix or response.kind == "logistic_normal" and prefix == "dpar_sigma_"
        )
        if categorical:
            assert response is not None
            categories = len(
                response.coordinate_indices(tuple(str(v) for v in self.category_labels))
            )
        else:
            categories = len(self.category_labels) - 1
        if (
            self.family.ordinal is not None
            and self.family.ordinal.kind == "hurdle_cumulative"
        ):
            categories -= 1
        specifications = [
            (
                term.group,
                term.slope,
                prefix
                + (
                    f"alpha_{_qvr_name(term.group)}"
                    if term.slope == "Intercept"
                    else f"beta_{_qvr_name(term.group)}_{_qvr_name(term.slope)}"
                ),
                len(predictor.group_levels[term.group]),
                categories
                if categorical
                or (term.column is not None and term.column.category_specific)
                else 0,
            )
            for term in predictor.random_terms
        ]
        for recorded in replay(self, draws, shape):
            for group, slope, name, levels, width in specifications:
                value = recorded.sites[name].value
                value = (
                    value.reshape(width, levels).T if width else value.reshape(levels)
                )
                collected[(group, slope)].append(value)
        result: dict[str, dict[str, torch.Tensor]] = {}
        for (group, slope), values in collected.items():
            stacked = torch.stack(values)
            result.setdefault(group, {})[slope] = stacked.reshape(
                *shape, *stacked.shape[1:]
            )
        return result


def fit(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    method: Literal["nuts", "hmc", "svi"] = "nuts",
    num_warmup: int | None = None,
    num_samples: int | None = None,
    num_chains: int | None = None,
    num_steps: int | None = None,
    learning_rate: float = 1e-2,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
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
) -> BayesianFit:
    """Compile a brms-style formula, fit it, and return the result.

    See [`quivers.formulas`][quivers.formulas] for surface details.  This entry
    point composes `formula_from_data`, `FormulaToQVRModule`,
    `Compiler`, and the inference layer in one call.

    ``family="categorical"`` infers contiguous labels ``0..K-1`` and uses
    label zero as the reference. An attached categorical predictor may return
    ``(N, K-1)`` reference logits or ``(N, K)`` full logits. For
    ``family="mixture"``, ``mixture_components`` selects any finite component
    count of at least two; the likelihood integrates assignments exactly.

    Parameters
    ----------
    formula : str
        brms/lme4-style response formula.
    data : IntoDataFrame
        Pandas, Polars, or another Narwhals-compatible dataframe.
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
    distributional : Mapping[str, str or NonlinearExpression] or None
        Auxiliary-parameter formulas, for instance
        ``{"sigma": "~ z + (1 | g)"}``. The optional left-hand side must
        match the parameter name. Coefficient prior overrides use names
        such as ``dpar_sigma_intercept`` and ``dpar_sigma_beta_z``.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters held constant on their natural scale, for instance
        ``{"quantile": 0.9}``. A fixed parameter has no latent site or prior.
    method : {"nuts", "hmc", "svi"}
        Inference algorithm.
    num_warmup : int or None
        MCMC warmup iterations per chain (NUTS and HMC only); 500 when
        omitted.
    num_samples : int or None
        MCMC draws kept per chain after warmup (NUTS and HMC only); 1000
        when omitted.
    num_chains : int or None
        MCMC chains (NUTS and HMC only); 4 when omitted.
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
    priors : Mapping[str, str] or None
        Per-parameter priors that override the defaults.
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

    Returns
    -------
    BayesianFit
        Parsed formula, compiled program, fitted posterior, and observations.

    Raises
    ------
    ValueError
        If the family is unknown, a predictor and its data are not given
        together, a trainable predictor is used without SVI, or a count
        is given to a method it does not apply to (``num_steps`` to NUTS
        or HMC; ``num_warmup``, ``num_samples``, or ``num_chains`` to
        SVI).
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
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
        elif num_steps is not None:
            raise ValueError(
                f"fit: num_steps counts SVI optimization steps, and method={method!r} "
                "runs MCMC; give num_warmup, num_samples, and num_chains"
            )
        predictor_name = "neural_eta" if predictor is not None else None
        prepared = _prepare_formula(
            formula,
            data=data,
            family=family,
            weights=weights,
            decision=decision,
            distributional=distributional,
            nonlinear=nonlinear,
            fixed_parameters=fixed_parameters,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            priors=priors,
            reparameterize=reparameterize,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            predictor_name=predictor_name,
            caller="fit",
        )
        lens, parsed = prepared.lens, prepared.formula
        family_obj = prepared.family
        binomial_trials = prepared.binomial_trials
        parameter_formulas, graph = prepared.distributional, prepared.graph
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
            raise ValueError(
                "fit: predictor_data is required when predictor is supplied"
            )
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
        compiler = Compiler(module)
        program_runtime = compiler.compile()
        morphism = program_runtime.morphism
        if not isinstance(morphism, MonadicProgram):
            raise TypeError(
                f"fit: compiled morphism has type "
                f"{type(morphism).__name__}, expected MonadicProgram"
            )
        program = morphism

        observations: dict[str, torch.Tensor] = {}
        observations.update(lens.fixed_column_observations(parsed))
        from quivers.formulas._response_modifiers import response_input_name

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
                isinstance(binomial_trials, str)
                and not parsed.response_layout.event_shape
            ):
                observations[_qvr_name(binomial_trials)] = trials
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
            observations[predictor_name] = _predictor_output(
                predictor,
                predictor_data,
                n_obs=parsed.response_values.shape[0],
                n_categories=predictor_categories,
                reference=predictor_reference,
            ).detach()
        else:
            predictor_categories = None
            predictor_reference = 0

        starts: dict[str, torch.Tensor | float] = {}
        if (
            family_obj.name in {"shifted_lognormal", "wiener"}
            and "ndt" in parameter_formulas
            and next(
                aux for aux in family_obj.aux_params if aux.name == "ndt"
            ).link.name
            == "log"
        ):
            shift_formula = parameter_formulas["ndt"]
            if any(column.is_intercept for column in shift_formula.fixed_columns):
                # These are initial coordinates, never a model offset.
                for column in shift_formula.fixed_columns:
                    name = "dpar_ndt_" + (
                        "intercept"
                        if column.is_intercept
                        else f"beta_{column.qvr_name}"
                    )
                    starts[name] = (
                        math.log(float(np.min(parsed.response_values)) / 4)
                        if column.is_intercept
                        else 0.0
                    )
        if (
            family_obj.name == "wiener"
            and "ndt" not in parameter_formulas
            and "ndt" not in (fixed_parameters or {})
        ):
            starts["ndt"] = float(np.min(parsed.response_values)) / 4
        starts.update(initial_values or {})
        from quivers.effects.observation_scoring import observation_scoring
        from quivers.inference.prior import _ancestral_program, _prior_mode, prior_draws

        mode = _prior_mode(sample_prior)
        if mode == "yes":
            from quivers.inference._posterior import sample_count

            sample_count(num_prior_samples)
            _ancestral_program(program)
        losses: tuple[float, ...] = ()
        with observation_scoring(mode != "only"):
            if method == "svi":
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
                    initial_values=starts,
                )
            else:
                posterior = _fit_mcmc(
                    program,
                    observations,
                    sampler=method,
                    num_warmup=500 if num_warmup is None else num_warmup,
                    num_samples=1000 if num_samples is None else num_samples,
                    num_chains=4 if num_chains is None else num_chains,
                    seed=seed,
                    initial_values=starts,
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

        return BayesianFit(
            formula=parsed,
            family=family_obj,
            program=program,
            posterior=posterior,
            sample_prior=mode,
            prior=retained_prior,
            losses=losses,
            observations=observations,
            reparameterize=reparameterize,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            priors=dict(priors or {}),
            predictor=predictor,
            predictor_data=predictor_data,
            predictor_name=predictor_name,
            distributional=parameter_formulas,
            predictor_graph=graph,
            fixed_parameters=dict(fixed_parameters or {}),
            prediction_plan=FormulaPredictionPlan(
                module=module,
                designs=prepared.designs,
                group_layouts=complement.group_layouts,
            ),
        )


def formula_to_qvr(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    path: str | Path | None = None,
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
    """
    lens, parsed = _formula_lens(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
    )
    module, _ = lens.forward(parsed)
    source = module_to_source(module)
    if path is not None:
        Path(path).write_text(source)
    return source


def get_prior(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
) -> dict[str, str]:
    """Return effective named priors without running inference.

    The arguments match `formula_to_qvr`, except for its output path.
    Priors reflect evaluated design columns, selected links, distributional
    formulas, group correlations, and explicit overrides. Default coefficient
    scales include column autoscaling; explicit overrides are emitted as given.

    Parameters
    ----------
    formula : str
        Response formula in brms/lme4 notation.
    data : IntoDataFrame
        Rows used to evaluate design matrices and autoscale default priors.
    family : str or Family
        Registered response family or a family value with selected links.
    weights : str or None
        Numeric column of finite nonnegative likelihood weights. Values
        multiply each response's log density, retain their original scale,
        and do not weight priors or posterior predictions.
    decision : str or None
        Wiener boundary-choice column, equivalent to response-side dec(column).
        Numeric codes must be exactly zero/lower or one/upper.
    distributional : Mapping[str, str] or None
        Separate formulas for auxiliary parameters.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters held constant; these have no prior entries.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default coefficient, group-scale, and LKJ Cholesky priors.
    priors : Mapping[str, str] or None
        Named overrides. Unknown or unused names raise ``ValueError``.
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

    Returns
    -------
    dict[str, str]
        Actual overrideable site names and canonical prior calls.

    Raises
    ------
    ValueError
        If a formula or family setting is invalid, or a prior is unused.
    """
    lens, parsed = _formula_lens(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
    )
    return lens.get_prior(parsed)


@dataclass(frozen=True, slots=True)
class _PreparedFormula:
    """Ephemeral shared preparation, including live fitted design evaluators."""

    lens: FormulaToQVRModule
    formula: Formula
    family: Family
    binomial_trials: int | str | None
    designs: dict[str, DesignMatrices]
    distributional: dict[str, Formula]
    graph: PredictorGraph | None
    trials: torch.Tensor | None


def _formula_lens(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
) -> tuple[FormulaToQVRModule, Formula]:
    """Use common preparation for source emission and prior inspection."""
    prepared = _prepare_formula(
        formula,
        data=data,
        family=family,
        weights=weights,
        decision=decision,
        distributional=distributional,
        nonlinear=nonlinear,
        fixed_parameters=fixed_parameters,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
    )
    return prepared.lens, prepared.formula


def _prepare_formula(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    decision: str | None = None,
    distributional: Mapping[str, str | NonlinearExpression] | None = None,
    nonlinear: Mapping[str, str | NonlinearExpression] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    caller: str = "formula_to_qvr",
) -> _PreparedFormula:
    """Prepare one retained design and compiler configuration for every entry."""
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"{caller}: unknown family {family!r}; choices are {sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family
    parsed, design = _threshold_formula_and_design(
        nonlinear_response_source(formula, nonlinear),
        data,
        thresholds_by=thresholds_by,
        weights=weights,
        decision=decision,
        groups=(family_obj.baseline.by,)
        if family_obj.baseline is not None and family_obj.baseline.by is not None
        else (),
    )
    from quivers.formulas._response_roles import included_data, retained_trials

    data = included_data(parsed, data)
    binomial_trials = retained_trials(parsed, family_obj, data, binomial_trials)
    parsed = parsed.with_(formula=formula)
    parameter_designs: dict[str, DesignMatrices] = {}
    graph = build_predictor_graph(
        formula,
        nonlinear,
        distributional,
        data,
        parsed.response_name,
        family_parameters=frozenset(
            {"mu", *(aux.name for aux in family_obj.aux_params)}
        ),
        designs=parameter_designs,
    )
    parameter_formulas = _parse_distributional(
        {
            name: value
            for name, value in (distributional or {}).items()
            if isinstance(value, str)
        },
        data,
        parsed.response_name,
        designs=parameter_designs,
    )
    trials = (
        _response_trials(parsed, data, binomial_trials, caller=caller)
        if family_obj.trials_parameter is not None
        else None
    )
    lens = FormulaToQVRModule(
        family_obj,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        correlation_prior=correlation_prior,
        user_priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
        distributional=parameter_formulas,
        predictor_graph=graph,
        fixed_parameters=fixed_parameters,
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
    )


def _threshold_formula_and_design(
    formula: str,
    data: IntoDataFrame,
    *,
    thresholds_by: str | None,
    weights: str | None,
    decision: str | None = None,
    groups: tuple[str, ...] = (),
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

    Returns
    -------
    tuple of Formula and DesignMatrices
        Evaluated metadata and retained transforms, with exactly one ordinary
        varying intercept for the threshold group.
    """
    parsed, design = _formula_and_design(
        formula, data, weights=weights, decision=decision, groups=groups
    )
    if thresholds_by is None or any(
        term.group == thresholds_by and term.slope == "Intercept"
        for term in parsed.random_terms
    ):
        return parsed, design
    expanded = f"{formula} + (1 | {thresholds_by})"
    parsed, design = _formula_and_design(
        expanded, data, weights=weights, decision=decision, groups=groups
    )
    return parsed.with_(formula=formula), design


def _parse_distributional(
    formulas: Mapping[str, str] | None,
    data: IntoDataFrame,
    response: str,
    *,
    designs: dict[str, DesignMatrices] | None = None,
) -> dict[str, Formula]:
    """Evaluate auxiliary design matrices on the same response rows.

    Parameters
    ----------
    formulas : Mapping[str, str] or None
        Auxiliary parameter names and their formulas.
    data : IntoDataFrame
        Shared source rows.
    response : str
        Primary response name used to evaluate design matrices.
    designs : dict[str, DesignMatrices] or None
        Optional output mapping that retains each fitted evaluator.

    Returns
    -------
    dict[str, Formula]
        One evaluated predictor per requested auxiliary parameter.

    Raises
    ------
    ValueError
        If a formula lacks a tilde or names a different parameter.
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
        if designs is None:
            parsed[parameter] = formula_from_data(f"{response} ~ {rhs}", data)
        else:
            parsed[parameter], designs[parameter] = _formula_and_design(
                f"{response} ~ {rhs}", data
            )
    return parsed


def _fit_mcmc(
    program: MonadicProgram,
    observations: dict[str, torch.Tensor],
    *,
    sampler: Literal["nuts", "hmc"],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
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

    Returns
    -------
    MCMCResult
        Retained samples and sampler statistics.
    """
    kernel = NUTSKernel() if sampler == "nuts" else HMCKernel()
    mcmc = MCMC(
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        initial_values=initial_values,
    )
    # One run of the program: its observation rows come from the data
    # through the formula's plate, not from the input's rows.
    x = torch.zeros(1, 1, dtype=torch.long)
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
    from quivers.inference.svi import _unique_parameters

    parameters = _unique_parameters(
        program.parameters(),
        guide.parameters(),
        () if predictor is None else predictor.parameters(),
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    svi = SVI(program, guide, optimizer, ELBO())
    n_obs = int(observations[next(iter(observations))].shape[0])
    # One run of the program: its observation rows come from the data
    # through the formula's plate, not from the input's rows.
    x = torch.zeros(1, 1, dtype=torch.long)

    def step_bindings(index: int) -> dict[str, torch.Tensor]:
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
    data: IntoDataFrame,
    specification: int | str | None,
    *,
    caller: str,
) -> torch.Tensor:
    """Validate shared trial inputs against scalar or vector response events.

    Parameters
    ----------
    formula : Formula
        Evaluated response with explicit row/event metadata.
    data : IntoDataFrame
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
    if not torch.isfinite(response).all() or not torch.equal(
        response, response.round()
    ):
        raise ValueError(f"{caller}: binomial responses must be finite integers")
    if torch.any(response < 0) or torch.any(response > trials):
        raise ValueError(
            f"{caller}: binomial responses must lie between zero and "
            "binomial_trials for every row"
        )
