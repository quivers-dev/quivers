"""Joint result views over the existing formula inference and replay engines."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack, cast

import torch
from narwhals.typing import IntoDataFrame

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import ObserveStep, ProgramDecl
from quivers.formulas._fit import (
    BayesianFit,
    _PreparedFormula,
    _formula_observations,
    _initial_values,
    _prepare_formula,
    _run_inference,
)
from quivers.formulas._joint_compile import _JointEmission, emit_joint
from quivers.formulas._multivariate import MultivariateFormula
from quivers.formulas._posterior import replay, row_binding
from quivers.formulas._response import expected_response
from quivers.formulas._priors import PriorInput, normalize_priors
from quivers.formulas._group_covariance import GroupCovariance
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
from quivers.inference.registry import LatentRegistry
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.prior import PriorDraws, PriorMode
from quivers.inference.trace import Trace


class _JointPredictionOptions(TypedDict, total=False):
    """Optional keywords of a joint posterior prediction.

    Each key follows `_JointFit.posterior_predict`: the draw count and the
    random seed. Omitted keys retain that method's defaults.
    """

    num_samples: int | None
    seed: int


class _JointExpectationOptions(_JointPredictionOptions, total=False):
    """Optional keywords of a joint posterior expectation.

    The additional mc_samples key follows `_JointFit.posterior_epred`.
    """

    mc_samples: int | None


class _JointInferenceOptions(TypedDict, total=False):
    """Optional inference keywords passed unchanged to the formula fit seam.

    Each key follows the keyword of the same name in `_run_inference`. Omitted
    keys retain its defaults.
    """

    num_steps: int | None
    num_warmup: int | None
    num_samples: int | None
    num_chains: int | None
    learning_rate: float
    guide: type[Guide] | None
    sample_prior: PriorMode
    num_prior_samples: int
    prior_seed: int | None
    predictor: torch.nn.Module | None
    predictor_data: torch.Tensor | None
    predictor_name: str | None
    predictor_categories: int | None
    predictor_reference: int | None


@dataclass(frozen=True)
class _JointFit:
    """One retained joint posterior, with explicit named response projections.

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
        Ordinary preparation of each response.
    losses : tuple[float, ...]
        SVI losses, empty for MCMC.
    prior : PriorDraws or MCMCResult or None
        Retained prior result, if one was requested.
    sample_prior : {"no", "yes", "only"}
        Prior/likelihood scoring policy of the fit.
    """

    specification: MultivariateFormula
    program: MonadicProgram
    posterior: MCMCResult | PriorDraws | Guide
    observations: Mapping[str, torch.Tensor]
    emission: _JointEmission
    prepared: Mapping[str, _PreparedFormula]
    losses: tuple[float, ...] = ()
    prior: PriorDraws | MCMCResult | None = None
    sample_prior: Literal["no", "yes", "only"] = "no"

    @property
    def response_names(self) -> tuple[str, ...]:
        """Retain authored response order independently of component coordinates.

        Returns
        -------
        tuple[str, ...]
            Response labels in authored order.
        """
        assert self.specification.response_names is not None
        return self.specification.response_names

    def draws(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Read canonical constrained draws without another inference or compiler.

        Parameters
        ----------
        num_samples : int or None
            Number of draws per chain, or None for every retained draw.
        seed : int
            Seed for draws from a variational guide.

        Returns
        -------
        dict[str, torch.Tensor]
            Constrained draws keyed by latent site, on chain and draw axes.
        """
        registry = LatentRegistry.from_model(
            self.program, observed_names=set(self.observations)
        )
        return constrained_draws(
            self.posterior,
            torch.zeros(1, 1, dtype=torch.long),
            num_samples=num_samples,
            seed=seed,
            registry=registry,
        )

    def _collect(
        self,
        operation: Callable[[Trace], Mapping[str, torch.Tensor]],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Replay each joint draw once and collect named heterogeneous events.

        Parameters
        ----------
        operation : Callable[[Trace], Mapping[str, torch.Tensor]]
            Reads named response values from one replayed trace.
        num_samples : int or None
            Number of draws per chain, or None for every retained draw.
        seed : int
            Seed for the posterior draws, offset for response randomness.

        Returns
        -------
        dict[str, torch.Tensor]
            Named values on chain and draw axes.
        """
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)

        def values() -> Iterator[Mapping[str, torch.Tensor]]:
            """Apply the operation to each replayed trace in draw order.

            Yields
            ------
            Mapping[str, torch.Tensor]
                Named values of one chain/draw position.
            """
            for recorded in replay(cast(BayesianFit, self), draws, shape):
                yield operation(recorded)

        with response_rng(seed + 2, (*draws.values(), *self.observations.values())):
            return collect_site_draws(values(), shape)

    def posterior_epred(
        self,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        mc_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return response means/probabilities from one joint conditional replay.

        Parameters
        ----------
        num_samples : int or None
            Number of draws per chain, or None for every retained draw.
        seed : int
            Seed for the posterior draws and any Monte Carlo moments.
        mc_samples : int or None
            Monte Carlo sample count for responses without a closed mean.

        Returns
        -------
        dict[str, torch.Tensor]
            Expected responses keyed by response label.
        """

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Read every expected response from one replayed trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Expected responses keyed by response label.
            """
            if self.specification.rescor:
                value = response_mean(
                    response_distribution(recorded, "formula_joint_response"),
                    mc_samples,
                )
                return {
                    name: value[..., index]
                    for index, name in enumerate(self.response_names)
                }
            result: dict[str, torch.Tensor] = {}
            for name, response in self.emission.responses.items():
                family = response.lens._family
                categories = (
                    response.lens._category_count(response.formula, family=family.name)
                    if family.ordinal is not None
                    or family.observe_family == "Categorical"
                    else 0
                )
                site = response.names[response.formula.response_name]
                value = expected_response(
                    response_distribution(recorded, site),
                    family,
                    categories=categories,
                    mc_samples=mc_samples,
                )
                result[name] = row_binding(
                    value,
                    len(response.formula.response_values),
                    response.formula.response_layout.event_shape or categories,
                )
            return result

        return self._collect(operation, num_samples=num_samples, seed=seed)

    def posterior_predict(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Draw the full correlated event before selecting named responses.

        Parameters
        ----------
        num_samples : int or None
            Number of draws per chain, or None for every retained draw.
        seed : int
            Seed for the posterior draws and the response noise.

        Returns
        -------
        dict[str, torch.Tensor]
            Predicted responses keyed by response label.
        """

        def operation(recorded: Trace) -> dict[str, torch.Tensor]:
            """Sample every response from one replayed trace.

            Parameters
            ----------
            recorded : Trace
                One replayed joint trace.

            Returns
            -------
            dict[str, torch.Tensor]
                Predicted responses keyed by response label.
            """
            if self.specification.rescor:
                value = row_binding(
                    response_sample(recorded, "formula_joint_response"),
                    len(
                        next(
                            iter(self.emission.responses.values())
                        ).formula.response_values
                    ),
                    (len(self.response_names),),
                )
                return {
                    name: value[..., index]
                    for index, name in enumerate(self.response_names)
                }
            result: dict[str, torch.Tensor] = {}
            for name, response in self.emission.responses.items():
                site = response.names[response.formula.response_name]
                result[name] = row_binding(
                    response_sample(recorded, site),
                    len(response.formula.response_values),
                    response.formula.response_layout.event_shape,
                )
            return result

        return self._collect(operation, num_samples=num_samples, seed=seed)

    def log_likelihood(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Retain actual joint scoring units, where correlated responses form one event.

        Parameters
        ----------
        num_samples : int or None
            Number of draws per chain, or None for every retained draw.
        seed : int
            Seed for draws from a variational guide.

        Returns
        -------
        dict[str, torch.Tensor]
            Pointwise log-likelihoods keyed by observed site.
        """
        shape = draw_count(self.posterior, num_samples)
        draws = self.draws(num_samples=num_samples, seed=seed)
        return collect_log_likelihood(
            replay(cast(BayesianFit, self), draws, shape, score_observations=True),
            {
                step.vars[0]
                for statement in self.emission.module.statements
                if isinstance(statement, ProgramDecl)
                for step in statement.draws
                if isinstance(step, ObserveStep)
            },
            shape,
        )

    def response(self, name: str) -> _ResponseView:
        """Select an authored response without constructing another posterior.

        Parameters
        ----------
        name : str
            Authored response label.

        Returns
        -------
        _ResponseView
            A view that projects joint results onto the response.

        Raises
        ------
        ValueError
            If the label names no response.
        """
        if name not in self.emission.responses:
            raise ValueError(
                f"unknown response {name!r}; expected {self.response_names!r}"
            )
        return _ResponseView(self, name)


@dataclass(frozen=True)
class _ResponseView:
    """A named projection that preserves its owning joint posterior.

    Parameters
    ----------
    fit : _JointFit
        The owning joint fit.
    name : str
        Authored response label.
    """

    fit: _JointFit
    name: str

    def posterior_epred(
        self, **options: Unpack[_JointExpectationOptions]
    ) -> torch.Tensor:
        """Project the joint expected responses onto this response.

        Parameters
        ----------
        **options : _JointExpectationOptions
            Keywords of `_JointFit.posterior_epred`.

        Returns
        -------
        torch.Tensor
            Expected values of this response.
        """
        return self.fit.posterior_epred(**options)[self.name]

    def posterior_predict(
        self, **options: Unpack[_JointPredictionOptions]
    ) -> torch.Tensor:
        """Project the joint predicted responses onto this response.

        Parameters
        ----------
        **options : _JointPredictionOptions
            Keywords of `_JointFit.posterior_predict`.

        Returns
        -------
        torch.Tensor
            Predicted values of this response.
        """
        return self.fit.posterior_predict(**options)[self.name]


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
        response_name = response.formula.response_name
        if response.names[response_name] not in observed:
            local.pop(response_name, None)
        values.update(
            {response.names.get(key, key): value for key, value in local.items()}
        )
    return values


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
) -> tuple[
    dict[str, _PreparedFormula], _JointEmission, MonadicProgram, dict[str, torch.Tensor]
]:
    """Reuse each response design and compile exactly one composed program.

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

    Returns
    -------
    tuple
        The prepared responses, the composed emission, the compiled program,
        and the joint observations.

    Raises
    ------
    TypeError
        If the joint emission does not compile to one MonadicProgram.
    """
    prepared: dict[str, _PreparedFormula] = {}
    for name in specification.response_names or ():
        response = specification.responses[name]
        prepared[name] = _prepare_formula(
            response.formula,
            data=data,
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
        )
    emission = emit_joint(
        {name: (part.lens, part.formula) for name, part in prepared.items()},
        priors=normalize_priors(priors),
        rescor=specification.rescor,
    )
    program = Compiler(emission.module, families=emission.families).compile().morphism
    if not isinstance(program, MonadicProgram):
        raise TypeError("joint formula emission must compile to one MonadicProgram")
    return prepared, emission, program, _joint_observations(emission, prepared)


def _fit_joint(
    specification: MultivariateFormula,
    data: IntoDataFrame,
    *,
    method: Literal["nuts", "hmc", "svi"] = "nuts",
    seed: int = 0,
    initial_values: Mapping[str, torch.Tensor | float] | None = None,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    correlation_prior: str = "LKJCholesky(1.0)",
    group_covariances: Mapping[str, GroupCovariance] | None = None,
    priors: PriorInput | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    **inference: Unpack[_JointInferenceOptions],
) -> _JointFit:
    """Delegate one joint program to the existing fit execution seam.

    Parameters
    ----------
    specification : MultivariateFormula
        The authored joint specification.
    data : IntoDataFrame
        Data holding every response and predictor column.
    method : {"nuts", "hmc", "svi"}
        Existing inference driver to invoke.
    seed : int
        Inference seed, also isolating preparation randomness.
    initial_values : Mapping[str, torch.Tensor or float] or None
        Explicit natural-scale starts over namespaced sites, which take
        precedence over the formula starts.
    fixed_prior, random_scale_prior, correlation_prior : str
        Default priors of each ordinary response preparation.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known group covariances, keyed by grouping factor.
    priors : PriorInput or None
        One joint catalogue of prior rules.
    reparameterize : {"centered", "noncentered"}
        Group-effect parameterization.
    **inference : _JointInferenceOptions
        Driver budgets and prior options of `_run_inference`.

    Returns
    -------
    _JointFit
        The retained joint posterior.

    Raises
    ------
    TypeError
        If the joint emission does not compile to one MonadicProgram.
    """
    with response_rng(seed, ()):
        prepared, emission, program, observations = _prepare_joint(
            specification,
            data,
            fixed_prior=fixed_prior,
            random_scale_prior=random_scale_prior,
            correlation_prior=correlation_prior,
            group_covariances=group_covariances,
            priors=priors,
            reparameterize=reparameterize,
        )
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
        starts.update(initial_values or {})
        result = _run_inference(
            program,
            observations,
            method=method,
            seed=seed,
            initial_values=starts,
            **inference,
        )
    return _JointFit(
        specification,
        program,
        result.posterior,
        observations,
        emission,
        prepared,
        result.losses,
        result.prior,
        result.sample_prior,
    )


__all__: list[str] = []
