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

from pathlib import Path
from typing import Literal, Mapping

import didactic.api as dx
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
from quivers.formulas.formula import Formula, _qvr_name, formula_from_data
from quivers.inference import (
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
    AutoNormalGuide,
)
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


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
    posterior : MCMCResult or Guide
        [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult] from NUTS / HMC, or
        a [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] from SVI.
    observations : Mapping[str, torch.Tensor]
        Inference-time observations dict (response + per-column
        covariates + per-group plate indices).
    fixed_parameters : dict[str, float]
        Auxiliary parameters bound to constants rather than sampled or regressed.
    distributional : Mapping[str, Formula]
        Evaluated auxiliary-parameter design matrices, retained so source
        inspection emits the fitted model without re-evaluating transforms.

    Fits compare by value over the formula, family, and settings, and
    by identity over the live objects: the program, the posterior, the
    observations mapping, the predictor, and its data. Two fits are
    thus equal exactly when they share those objects and agree on
    everything else.
    """

    formula: Formula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: Family
    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | Guide = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    reparameterize: Literal["centered", "noncentered"] = "noncentered"
    binomial_trials: int | str = 1
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

    @property
    def qvr_source(self) -> str:
        """Lazily emit the AST-equivalent ``.qvr`` source for display."""
        lens = FormulaToQVRModule(
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
            distributional=self.distributional,
            fixed_parameters=self.fixed_parameters,
        )
        module, _ = lens.forward(self.formula)
        return module_to_source(module)

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the AST-equivalent ``.qvr`` source to ``path`` and
        return the resulting `Path`.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out

    @property
    def category_labels(self) -> tuple[int, ...]:
        """Response-category labels in posterior probability-array order.

        Returns
        -------
        tuple[int, ...]
            Codes ``0..K-1`` for categorical and ordinal responses; empty
            for other families. Categorical coefficient arrays omit code zero.
        """
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
    def group_levels(self) -> dict[str, tuple[str, ...]]:
        """Retained group labels in each coefficient array's level-axis order.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Group names and the ordered labels used in coefficient arrays.
        """
        return {
            group: levels
            for predictor in (self.formula, *self.distributional.values())
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
        from quivers.formulas._posterior import draw_count

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
        from quivers.formulas._posterior import draw_count

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

    def posterior_linpred(
        self,
        *,
        parameter: str = "mu",
        transform: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Replay a fitted linear predictor on its retained training rows.

        Parameters
        ----------
        parameter : str
            ``mu`` or a distributionally regressed auxiliary parameter.
        transform : bool
            Return the native inverse-link binding instead of its linear
            predictor. For binomial data this is a probability, not a count.
            Ordinal probabilities are supplied by `posterior_epred`.
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
        from quivers.formulas._posterior import draw_count, replay, row_binding

        self._formula_predictor(parameter)
        ordinal = self.family.observe_family in {"OrderedLogistic", "OrderedProbit"}
        if transform and parameter == "mu" and ordinal:
            raise ValueError(
                "ordinal probabilities depend on thresholds; use posterior_epred"
            )
        binding = (
            ("mu" if transform else "eta")
            if parameter == "mu"
            else (parameter if transform else f"dpar_{parameter}_eta")
        )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = self.formula.response_values.shape[0]
        categories = (
            len(self.category_labels)
            if self.family.observe_family == "Categorical"
            else 0
        )
        values = [
            row_binding(recorded.sites[binding].value, rows, categories)
            for recorded in replay(self, draws, shape)
        ]
        stacked = torch.stack(values)
        return stacked.reshape(*shape, *stacked.shape[1:])

    def posterior_epred(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> torch.Tensor:
        """Evaluate the native expected response on retained training rows.

        Parameters
        ----------
        num_samples : int or None
            Draw count under the convention of `draws`.
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

        Raises
        ------
        ValueError
            If a draw has an undefined or nonfinite response mean.
        TypeError
            If the native family exposes no expectation, or a circular
            response would require a specified angular convention.
        """
        from quivers.formulas._posterior import (
            draw_count,
            replay,
            response_distribution,
            row_binding,
        )

        if self.family.observe_family == "VonMises":
            raise TypeError(
                "a circular response expectation requires an angular convention"
            )
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = self.formula.response_values.shape[0]
        response = _qvr_name(self.formula.response_name)
        categories = len(self.category_labels)
        values: list[torch.Tensor] = []
        for recorded in replay(self, draws, shape):
            distribution = response_distribution(recorded, response)
            if self.family.observe_family == "Categorical":
                if not isinstance(distribution, torch.distributions.Categorical):
                    raise TypeError(
                        "the categorical response has no native class probabilities"
                    )
                value = distribution.probs
            elif categories:
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
                try:
                    value = distribution.mean
                except NotImplementedError as error:
                    raise TypeError(
                        f"{self.family.observe_family} exposes no native response expectation"
                    ) from error
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"{self.family.observe_family} has an undefined or nonfinite response mean at a posterior draw"
                )
            values.append(row_binding(value, rows, categories))
        stacked = torch.stack(values)
        return stacked.reshape(*shape, *stacked.shape[1:])

    def posterior_predict(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> torch.Tensor:
        """Sample native response distributions on retained training rows.

        Parameters
        ----------
        num_samples : int or None
            Maximum draws per MCMC chain, or fresh guide draws, following
            `draws`. Each posterior draw supplies one response per row.
        seed : int
            Seed for separate latent and response random streams. The
            caller's CPU RNG state is restored on return or failure.

        Returns
        -------
        torch.Tensor
            Predictive responses, shape ``(chain, draw, row)``. Categorical
            and ordinal responses contain integer codes in `category_labels`;
            binomial responses contain counts. Likelihood weights do not
            alter the response sampling law at a fixed posterior draw.

        Raises
        ------
        ValueError
            If the requested count or retained posterior layout is invalid.
        TypeError
            If the response has no native sampling operation.
        """
        from quivers.formulas._posterior import draw_count, replay, row_binding

        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        rows = self.formula.response_values.shape[0]
        response = _qvr_name(self.formula.response_name)
        result: torch.Tensor | None = None
        # Guide sampling uses its own seed and restores the caller's RNG.
        # A distinct stream avoids coupling response noise to guide noise.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed((seed + 1) % 2**64)
            for index, recorded in enumerate(replay(self, draws, shape)):
                sampleable = recorded.sites[response].sampleable
                if sampleable is None:
                    raise TypeError("the response has no native sampling operation")
                # The native observation expands shared parameters before
                # sampling, so different rows receive independent draws.
                value = row_binding(
                    sampleable.raw_morphism.rsample(sampleable.input), rows
                )
                if result is None:
                    result = value.new_empty((*shape, rows))
                result.reshape(-1, rows)[index].copy_(value)
        if result is None:
            raise ValueError("the posterior has no retained predictive draws")
        return result

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
        categorical = self.family.observe_family == "Categorical"
        categories = len(self.category_labels) - 1 if categorical else 0
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
            )
            for term in predictor.random_terms
        ]
        for recorded in replay(self, draws, shape):
            for group, slope, name, levels in specifications:
                value = recorded.sites[name].value
                value = (
                    value.reshape(categories, levels).T
                    if categorical
                    else value.reshape(levels)
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
    distributional: Mapping[str, str] | None = None,
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
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    seed: int = 0,
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
    distributional : Mapping[str, str] or None
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
        if isinstance(family, str):
            if family not in families:
                raise ValueError(
                    f"fit: unknown family {family!r}; choices are {sorted(families)}"
                )
            family_obj = families[family]
        else:
            family_obj = family

        parsed = formula_from_data(formula, data, weights=weights)
        parameter_formulas = _parse_distributional(
            distributional, data, parsed.response_name
        )
        predictor_name = "neural_eta" if predictor is not None else None
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
            fixed_parameters=dict(fixed_parameters or {}),
        )
        module, _ = lens.forward(parsed)
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
        response_name = parsed.response_name
        observations[response_name] = torch.as_tensor(
            parsed.response_values.copy(), dtype=torch.float32
        ).reshape(-1)
        for predictor_formula in (parsed, *parameter_formulas.values()):
            for group, codes in predictor_formula.group_indices.items():
                observations[f"{_qvr_name(group)}_idx"] = torch.as_tensor(
                    list(codes), dtype=torch.long
                )
        if family_obj.name in {"binomial", "beta_binomial"}:
            trials = _binomial_trial_tensor(
                data, binomial_trials, n_obs=parsed.response_values.shape[0]
            )
            response = observations[response_name]
            _validate_binomial_response(response, trials, caller="fit")
            if isinstance(binomial_trials, str):
                observations[_qvr_name(binomial_trials)] = trials
        if predictor is not None:
            assert predictor_data is not None
            assert predictor_name is not None
            predictor_categories = (
                FormulaToQVRModule._category_count(parsed, family="categorical")
                if family_obj.name == "categorical"
                else None
            )
            observations[predictor_name] = _predictor_output(
                predictor,
                predictor_data,
                n_obs=parsed.response_values.shape[0],
                n_categories=predictor_categories,
            ).detach()
        else:
            predictor_categories = None

        if method == "svi":
            posterior = _fit_svi(
                program,
                observations,
                1000 if num_steps is None else num_steps,
                learning_rate=learning_rate,
                guide_cls=guide,
                predictor=predictor,
                predictor_data=predictor_data,
                predictor_name=predictor_name,
                predictor_categories=predictor_categories,
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
            )

        return BayesianFit(
            formula=parsed,
            family=family_obj,
            program=program,
            posterior=posterior,
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
            fixed_parameters=dict(fixed_parameters or {}),
        )


def formula_to_qvr(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    distributional: Mapping[str, str] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
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
    """
    lens, parsed = _formula_lens(
        formula,
        data=data,
        family=family,
        weights=weights,
        distributional=distributional,
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
    distributional: Mapping[str, str] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
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
        distributional=distributional,
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


def _formula_lens(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    weights: str | None = None,
    distributional: Mapping[str, str] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
) -> tuple[FormulaToQVRModule, Formula]:
    """Evaluate and configure the shared source/inspection compilation path."""
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"formula_to_qvr: unknown family {family!r}; choices are "
                f"{sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family
    parsed = formula_from_data(formula, data, weights=weights)
    parameter_formulas = _parse_distributional(
        distributional, data, parsed.response_name
    )
    if family_obj.name in {"binomial", "beta_binomial"}:
        trials = _binomial_trial_tensor(
            data, binomial_trials, n_obs=parsed.response_values.shape[0]
        )
        response = torch.as_tensor(parsed.response_values.copy())
        _validate_binomial_response(response, trials, caller="formula_to_qvr")
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
        fixed_parameters=fixed_parameters,
    )
    return lens, parsed


def _parse_distributional(
    formulas: Mapping[str, str] | None,
    data: IntoDataFrame,
    response: str,
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
        parsed[parameter] = formula_from_data(f"{response} ~ {rhs}", data)
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
    guide = guide_cls(program, observed_names=set(observations.keys()))
    parameters = list(program.parameters()) + list(guide.parameters())
    if predictor is not None:
        parameters.extend(predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    svi = SVI(program, guide, optimizer, ELBO())
    n_obs = int(observations[next(iter(observations))].shape[0])
    # One run of the program: its observation rows come from the data
    # through the formula's plate, not from the input's rows.
    x = torch.zeros(1, 1, dtype=torch.long)
    for _ in range(num_steps):
        step_observations = observations
        if predictor is not None:
            assert predictor_data is not None and predictor_name is not None
            step_observations = dict(observations)
            step_observations[predictor_name] = _predictor_output(
                predictor,
                predictor_data,
                n_obs=n_obs,
                n_categories=predictor_categories,
            )
        svi.step(x, step_observations)
    if predictor is not None:
        assert predictor_data is not None and predictor_name is not None
        observations[predictor_name] = _predictor_output(
            predictor,
            predictor_data,
            n_obs=n_obs,
            n_categories=predictor_categories,
        ).detach()
    return guide


def _binomial_trial_tensor(
    data: IntoDataFrame,
    trials: int | str,
    *,
    n_obs: int,
) -> torch.Tensor:
    if isinstance(trials, bool):
        raise ValueError("binomial_trials must be a positive integer or a column name")
    if isinstance(trials, int):
        if trials < 1:
            raise ValueError("binomial_trials must be at least one")
        return torch.full((n_obs,), trials, dtype=torch.long)
    frame = nw.from_native(data, eager_only=True)
    if trials not in frame.columns:
        raise ValueError(f"binomial_trials column {trials!r} is not in the data")
    values = torch.as_tensor(frame[trials].to_numpy().copy()).reshape(-1)
    if values.shape[0] != n_obs:
        raise ValueError(
            f"binomial_trials column has {values.shape[0]} rows, expected {n_obs}"
        )
    if values.dtype == torch.bool or values.is_complex():
        raise ValueError("binomial_trials must contain positive finite integers")
    if values.is_floating_point():
        if not torch.isfinite(values).all() or torch.any(values < 1):
            raise ValueError("binomial_trials must contain positive finite integers")
        if not torch.equal(values, values.round()):
            raise ValueError("binomial_trials must contain integers")
        if torch.any(values >= 2**63):
            raise ValueError("binomial_trials must fit in a signed 64-bit integer")
    integer_values = values.to(dtype=torch.long)
    if torch.any(integer_values < 1):
        raise ValueError("binomial_trials must contain positive signed 64-bit integers")
    return integer_values


def _predictor_output(
    predictor: nn.Module,
    data: torch.Tensor,
    *,
    n_obs: int,
    n_categories: int | None = None,
) -> torch.Tensor:
    value = predictor(data)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "fit: predictor must return a torch.Tensor on the linear-predictor scale"
        )
    if n_categories is not None:
        if value.shape == (n_obs,) and n_categories == 2:
            return value
        if value.shape == (n_obs, n_categories - 1):
            return value.reshape(-1)
        if value.shape == (n_obs, n_categories):
            return (value[:, 1:] - value[:, :1]).reshape(-1)
        raise ValueError(
            "fit: categorical predictor returned shape "
            f"{tuple(value.shape)}, expected ({n_obs}, {n_categories - 1}) "
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
