"""Fit one formula to several imputed datasets and pool the posterior draws.

This follows brms ``brm_multiple`` and ``combine_models``: each dataset is
fitted separately through the ordinary `fit` entry point, and the draws of the
fitted models are concatenated chain by chain. The pooled posterior is the
equally weighted mixture of the per-dataset posteriors, the usual Bayesian
counterpart of multiple-imputation pooling. Convergence is assessed for each
dataset separately: chains fitted to different datasets need not overlap, so
a pooled split ``R-hat`` is not a convergence check, as the brms documentation
warns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MethodType
from typing import Concatenate, Literal, Unpack, cast, overload

import didactic.api as dx
from didactic.types._typing import FieldValue
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.diagnostics.convergence import rhat
from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._fit import BayesianFit, fit
from quivers.formulas._joint_fit import MultivariateFit
from quivers.formulas._joint_options import FormulaFitOptions, JointFitOptions
from quivers.formulas._multivariate import MultivariateFormula, ResponseModel
from quivers.inference.mcmc.driver import MCMCResult, StopReason
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry

__all__: list[str] = []

#: A fitted formula model whose recorded draws can be pooled.
type Pooled = BayesianFit | MultivariateFit

type PooledMember = MethodType | ValueModel | torch.Tensor | torch.nn.Module
"""A bound method of the pooled fit, or the value of one of its properties
that is not a plain field value."""


def _structure(fitted: Pooled) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the latent layout two pooled fits must share.

    Default prior scales may depend on each dataset, as brms's do, so the
    pooled models are compared by their latent sites and shapes.

    Parameters
    ----------
    fitted : BayesianFit or MultivariateFit
        One fit.

    Returns
    -------
    tuple
        Sorted latent site names and constrained shapes.
    """
    registry = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    )
    return tuple(
        sorted(
            (name, tuple(registry[name].constrained_shape)) for name in registry.names
        )
    )


def _concatenate(results: Sequence[MCMCResult | PriorDraws]) -> MCMCResult | PriorDraws:
    """Concatenate recorded draws along the chain axis.

    Parameters
    ----------
    results : Sequence[MCMCResult or PriorDraws]
        Recorded draws of fits of one model.

    Returns
    -------
    MCMCResult or PriorDraws
        One result whose chains are the chains of every input in order.

    Raises
    ------
    TypeError
        If the results mix kinds.
    ValueError
        If site names, draw counts, or warmup lengths differ.
    """
    first = results[0]
    if any(type(result) is not type(first) for result in results):
        raise TypeError("combine_models requires one kind of recorded draws")
    if any(set(result.samples) != set(first.samples) for result in results):
        raise ValueError("combined models must have the same parameters")
    samples = {
        name: torch.cat([result.samples[name] for result in results], dim=0)
        for name in first.samples
    }
    if isinstance(first, PriorDraws):
        if any(result.num_samples != first.num_samples for result in results):
            raise ValueError("combined prior draws must have equal draw counts")
        return first.with_(samples=samples)
    assert all(isinstance(result, MCMCResult) for result in results)
    chains = [result for result in results if isinstance(result, MCMCResult)]
    if any(
        result.num_samples != first.num_samples
        or result.num_warmup != chains[0].num_warmup
        for result in chains
    ):
        raise ValueError("combined models must have equal warmup and draw counts")
    updates: dict[
        str, torch.Tensor | dict[str, torch.Tensor] | tuple[StopReason, ...]
    ] = {"samples": samples}
    for name in type(first).__field_specs__:
        value = getattr(first, name)
        if name == "samples" or isinstance(value, int):
            continue
        if isinstance(value, torch.Tensor):
            updates[name] = torch.cat([getattr(result, name) for result in chains], 0)
        elif isinstance(value, Mapping):
            updates[name] = {
                key: torch.cat([getattr(result, name)[key] for result in chains], 0)
                for key in value
            }
        elif isinstance(value, tuple):
            updates[name] = tuple(
                item for result in chains for item in getattr(result, name)
            )
    return first.with_(**updates)


@overload
def combine_models(*fits: BayesianFit) -> BayesianFit: ...


@overload
def combine_models(*fits: MultivariateFit) -> MultivariateFit: ...


def combine_models(*fits: Pooled) -> Pooled:
    """Pool the draws of fits of one model, as brms ``combine_models`` does.

    Cost: one concatenation of every retained draw tensor.

    Parameters
    ----------
    *fits : BayesianFit or MultivariateFit
        At least one fit of the same formula and settings, possibly to
        different datasets, each with recorded draws (MCMC or prior draws).

    Returns
    -------
    BayesianFit or MultivariateFit
        The first fit with the chains of every fit. Its data, designs, and
        prediction plan are the first fit's, as in brms.

    Raises
    ------
    ValueError
        If no fit is given or the fits differ in structure.
    TypeError
        If fits mix kinds or a posterior is a variational guide.
    """
    if not fits:
        raise ValueError("combine_models requires at least one fit")
    first = fits[0]
    if any(type(fitted) is not type(first) for fitted in fits):
        raise TypeError("combine_models requires fits of one kind")
    posteriors = []
    for fitted in fits:
        if not isinstance(fitted.posterior, MCMCResult | PriorDraws):
            raise TypeError(
                "combine_models pools recorded draws; a variational guide has none"
            )
        posteriors.append(fitted.posterior)
    structure = _structure(first)
    if any(_structure(fitted) != structure for fitted in fits[1:]):
        raise ValueError(
            "combine_models requires fits of the same model; their latent sites differ"
        )
    return first.with_(posterior=_concatenate(posteriors))


def _pooled[**P, R](
    method: Callable[Concatenate[BayesianFit, P], R],
) -> Callable[Concatenate[MultipleFit, P], R]:
    """Build the combined fit's method that calls the pooled fit's method.

    Parameters
    ----------
    method : Callable
        A method of `BayesianFit`, whose name, signature and documentation
        the returned method takes.

    Returns
    -------
    Callable
        A method of `MultipleFit` that calls the method of that name on the
        pooled fit, a `BayesianFit` or a `MultivariateFit`, with the same
        arguments.
    """
    name = method.__name__

    def call(self: MultipleFit, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the pooled fit's method of the same name.

        Parameters
        ----------
        *args : P.args
            Positional arguments of the pooled fit's method.
        **kwargs : P.kwargs
            Keyword arguments of the pooled fit's method.

        Returns
        -------
        R
            The value the pooled fit's method returns.
        """
        return cast(R, getattr(self.pooled, name)(*args, **kwargs))

    call.__name__ = name
    call.__qualname__ = f"MultipleFit.{name}"
    call.__doc__ = method.__doc__
    return call


class MultipleFit(ValueModel):
    """Fits of one formula to several datasets and their pooled posterior.

    Parameters
    ----------
    pooled : BayesianFit or MultivariateFit
        Fit retaining every dataset's chains (`combine_models`).
    fits : tuple
        The per-dataset fits, in dataset order.

    Notes
    -----
    Every public method and property of ``pooled`` is also an attribute of
    this value, acting on the pooled draws: ``summary()``,
    ``posterior_predict()``, ``conditional_effects()``, ``loo()`` and the
    rest, as a brms ``brmsfit_multiple`` is an ordinary ``brmsfit``. The
    methods defined here (`rhats`, `update`) take precedence. Data-dependent
    quantities use the first dataset, as in brms.
    """

    pooled: BayesianFit | MultivariateFit = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fits: tuple[BayesianFit | MultivariateFit, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    # brms post-processing methods, each acting on the pooled draws.
    ndraws = _pooled(BayesianFit.ndraws)
    nchains = _pooled(BayesianFit.nchains)
    niterations = _pooled(BayesianFit.niterations)
    nwarmup = _pooled(BayesianFit.nwarmup)
    nvariables = _pooled(BayesianFit.nvariables)
    posterior_summary = _pooled(BayesianFit.posterior_summary)
    posterior_interval = _pooled(BayesianFit.posterior_interval)
    summary = _pooled(BayesianFit.summary)
    ngrps = _pooled(BayesianFit.ngrps)
    fixef = _pooled(BayesianFit.fixef)
    ranef = _pooled(BayesianFit.ranef)
    coef = _pooled(BayesianFit.coef)
    varcorr = _pooled(BayesianFit.varcorr)
    vcov = _pooled(BayesianFit.vcov)
    rhat = _pooled(BayesianFit.rhat)
    neff_ratio = _pooled(BayesianFit.neff_ratio)
    control_params = _pooled(BayesianFit.control_params)
    inits = _pooled(BayesianFit.inits)
    nuts_params = _pooled(BayesianFit.nuts_params)
    log_posterior = _pooled(BayesianFit.log_posterior)
    criterion = _pooled(BayesianFit.criterion)
    variables = _pooled(BayesianFit.variables)
    nobs = _pooled(BayesianFit.nobs)
    predict = _pooled(BayesianFit.predict)
    fitted = _pooled(BayesianFit.fitted)
    predictive_error = _pooled(BayesianFit.predictive_error)
    residuals = _pooled(BayesianFit.residuals)
    predictive_interval = _pooled(BayesianFit.predictive_interval)
    loo = _pooled(BayesianFit.loo)
    loo_moment_match = _pooled(BayesianFit.loo_moment_match)
    waic = _pooled(BayesianFit.waic)
    psis = _pooled(BayesianFit.psis)
    loo_subsample = _pooled(BayesianFit.loo_subsample)
    loo_predict = _pooled(BayesianFit.loo_predict)
    loo_epred = _pooled(BayesianFit.loo_epred)
    loo_linpred = _pooled(BayesianFit.loo_linpred)
    loo_predictive_interval = _pooled(BayesianFit.loo_predictive_interval)
    loo_r2 = _pooled(BayesianFit.loo_r2)
    kfold = _pooled(BayesianFit.kfold)
    add_criterion = _pooled(BayesianFit.add_criterion)
    conditional_smooths = _pooled(BayesianFit.conditional_smooths)
    pp_check = _pooled(BayesianFit.pp_check)
    mcmc_plot = _pooled(BayesianFit.mcmc_plot)
    plot = _pooled(BayesianFit.plot)
    pairs = _pooled(BayesianFit.pairs)
    prior_summary = _pooled(BayesianFit.prior_summary)
    conditional_effects = _pooled(BayesianFit.conditional_effects)
    prior_draws = _pooled(BayesianFit.prior_draws)
    prior_view = _pooled(BayesianFit.prior_view)
    draws = _pooled(BayesianFit.draws)
    posterior_linpred = _pooled(BayesianFit.posterior_linpred)
    posterior_epred = _pooled(BayesianFit.posterior_epred)
    posterior_predict = _pooled(BayesianFit.posterior_predict)
    log_likelihood = _pooled(BayesianFit.log_likelihood)
    bayes_r2 = _pooled(BayesianFit.bayes_r2)
    hypothesis = _pooled(BayesianFit.hypothesis)
    posterior_response = _pooled(BayesianFit.posterior_response)

    def __getattr__(self, name: str) -> FieldValue | PooledMember:
        """Read a field, or a method or property of the pooled fit.

        brms treats a ``brmsfit_multiple`` as an ordinary ``brmsfit``:
        ``summary``, the prediction methods, ``conditional_effects``,
        ``loo`` and every other method act on the pooled draws. A name that
        is not a field of this value is thus looked up on ``pooled``, so the
        whole method surface of the pooled `BayesianFit` or
        `MultivariateFit` is available here through the one shared
        post-processing path.

        Parameters
        ----------
        name : str
            Field of this value, or public method or property of the
            pooled fit.

        Returns
        -------
        FieldValue or PooledMember
            The field value, or the pooled fit's bound method or property
            value.

        Raises
        ------
        AttributeError
            If the name is private, or neither a field nor a public member
            of the pooled fit.
        """
        if name in type(self).__field_specs__ or name.startswith("_"):
            return super().__getattr__(name)
        pooled = super().__getattr__("pooled")
        if not hasattr(pooled, name):
            raise AttributeError(
                f"{type(self).__name__!r} has no field {name!r}, and its pooled "
                f"{type(pooled).__name__} has no method or property of that name"
            )
        return cast(PooledMember, getattr(pooled, name))

    def __dir__(self) -> list[str]:
        """List this value's names and the pooled fit's public names.

        Returns
        -------
        list[str]
            Sorted names reachable as attributes.
        """
        own = set(super().__dir__())
        public = {name for name in dir(self.pooled) if not name.startswith("_")}
        return sorted(own | public)

    def rhats(self) -> list[dict[str, torch.Tensor]]:
        """Return split R-hat of every parameter for each dataset.

        Convergence is assessed within each dataset, as brms reports in the
        ``rhats`` element of a ``brm_multiple`` fit.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            One mapping per dataset from native site to R-hat.
        """
        return [
            {name: rhat(value) for name, value in fitted.draws().items()}
            for fitted in self.fits
        ]

    def update(
        self,
        data: Sequence[IntoDataFrame] | None = None,
        *,
        data2: Sequence[Mapping[str, AutocorMatrix]] | None = None,
        **overrides: object,
    ) -> MultipleFit:
        """Refit every dataset through the first fit's retained recipe.

        This follows brms ``update`` of a ``brm_multiple`` fit: the first
        fit is the template, the datasets must be given again, and every
        other fitting argument is kept unless overridden. Each dataset is
        refitted by the template's `update`, so all submodels share the
        formula, priors, budgets, and seed, and their draws are pooled with
        `combine_models`. The number of datasets may differ from the
        original fit's.

        Cost: one ordinary refit per dataset.

        Parameters
        ----------
        data : Sequence[IntoDataFrame] or None
            Imputed datasets. Required, as brms requires ``newdata``.
        data2 : Sequence[Mapping[str, AutocorMatrix]] or None
            One mapping of named matrices per dataset; None keeps the
            template's matrices for every dataset.
        **overrides : object
            Overrides of the canonical `fit` arguments, as in the `update`
            method of a single fit.

        Returns
        -------
        MultipleFit
            Fresh per-dataset fits and their pooled fit.

        Raises
        ------
        ValueError
            If the datasets are missing or empty, ``data2`` does not match
            them, the datasets are given as an override, or the updated
            method records no draws.
        """
        if "newdata" in overrides:
            raise ValueError("Please use argument 'data' to update the data.")
        if data is None:
            raise ValueError("'data' is required when updating a 'MultipleFit' object.")
        if isinstance(data, Mapping) or not isinstance(data, Sequence):
            raise ValueError("'data' must be a sequence of dataframes.")
        datasets = list(data)
        if not datasets:
            raise ValueError("update requires at least one dataset")
        if data2 is not None:
            if isinstance(data2, Mapping) or not all(
                isinstance(item, Mapping) for item in data2
            ):
                raise ValueError("'data2' must be a sequence of named mappings.")
            if len(data2) != len(datasets):
                raise ValueError("'data2' must have the same length as 'data'.")
        if overrides.get("method") == "svi":
            raise ValueError(
                "fit_multiple pools recorded draws; use method='nuts' or 'hmc'"
            )
        template = self.fits[0]
        settings = [
            dict(overrides) if data2 is None else {**overrides, "data2": data2[index]}
            for index in range(len(datasets))
        ]
        if isinstance(template, MultivariateFit):
            joint = tuple(
                template.update(frame, **options)
                for frame, options in zip(datasets, settings, strict=True)
            )
            return MultipleFit(pooled=combine_models(*joint), fits=joint)
        single = tuple(
            template.update(frame, **options)
            for frame, options in zip(datasets, settings, strict=True)
        )
        return MultipleFit(pooled=combine_models(*single), fits=single)


@overload
def fit_multiple(
    formula: MultivariateFormula,
    *,
    data: Sequence[IntoDataFrame],
    combine: Literal[True] = True,
    **options: Unpack[JointFitOptions],
) -> MultipleFit: ...


@overload
def fit_multiple(
    formula: str | ResponseModel,
    *,
    data: Sequence[IntoDataFrame],
    combine: Literal[True] = True,
    **options: Unpack[FormulaFitOptions],
) -> MultipleFit: ...


@overload
def fit_multiple(
    formula: MultivariateFormula,
    *,
    data: Sequence[IntoDataFrame],
    combine: Literal[False],
    **options: Unpack[JointFitOptions],
) -> tuple[MultivariateFit, ...]: ...


@overload
def fit_multiple(
    formula: str | ResponseModel,
    *,
    data: Sequence[IntoDataFrame],
    combine: Literal[False],
    **options: Unpack[FormulaFitOptions],
) -> tuple[BayesianFit, ...]: ...


def fit_multiple(
    formula: str | ResponseModel | MultivariateFormula,
    *,
    data: Sequence[IntoDataFrame],
    combine: bool = True,
    **options: Unpack[FormulaFitOptions],
) -> MultipleFit | tuple[BayesianFit, ...] | tuple[MultivariateFit, ...]:
    """Fit one formula to each imputed dataset, as brms ``brm_multiple`` does.

    Every dataset is fitted by `fit` with the same options and seed, as
    brms passes one seed to every submodel, and the draws are pooled with
    `combine_models`.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula
        Formula fitted to every dataset.
    data : Sequence[IntoDataFrame]
        Imputed datasets, for instance from a multiple-imputation package.
    combine : bool
        brms ``combine``: pool the fits into one `MultipleFit`. False
        returns the per-dataset fits, which may then use any method.
    **options : FormulaFitOptions
        Every other `fit` keyword. A combined fit requires a method that
        records draws (NUTS or HMC).

    Returns
    -------
    MultipleFit or tuple
        Per-dataset fits and the pooled fit, or the per-dataset fits alone
        with ``combine=False``.

    Raises
    ------
    ValueError
        If no dataset is given or a combined fit is requested with SVI.
    TypeError
        If ``combine`` is not Boolean.
    """
    if type(combine) is not bool:
        raise TypeError("combine must be True or False")
    datasets = list(data)
    if not datasets:
        raise ValueError("fit_multiple requires at least one dataset")
    if combine and options.get("method", "nuts") == "svi":
        raise ValueError(
            "fit_multiple pools recorded draws; use method='nuts' or 'hmc'"
        )
    if isinstance(formula, MultivariateFormula):
        settings = cast(JointFitOptions, options)
        joint = tuple(fit(formula, data=frame, **settings) for frame in datasets)
        if not combine:
            return joint
        return MultipleFit(pooled=combine_models(*joint), fits=joint)
    single = tuple(fit(formula, data=frame, **options) for frame in datasets)
    if not combine:
        return single
    return MultipleFit(pooled=combine_models(*single), fits=single)
