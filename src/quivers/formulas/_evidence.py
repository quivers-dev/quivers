"""Marginal likelihoods, Bayes factors and posterior model probabilities of fits.

brms computes ``bridge_sampler``, ``bayes_factor`` and ``post_prob`` by
handing its Stan model to the R ``bridgesampling`` package
[@gronau2020bridgesampling]. Here the same estimator
(`quivers.diagnostics.bridge.bridge_sampler`) runs on a fit's retained
MCMC draws, mapped to the sampler's unconstrained coordinates, with the
fit's own Jacobian-corrected log joint (`PotentialFn`) as the log
unnormalized posterior. Every density of a compiled formula program keeps
its normalizing constant, so the requirement brms states as
``save_pars(all = TRUE)`` with a normalized Stan model holds by
construction. What cannot hold is refused with a located error: improper
priors, explicit score potentials of unknown mass, opaque joint
overrides, discrete Gibbs sites, and draws that are not posterior MCMC draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from quivers.diagnostics.bridge import (
    BridgeMethod,
    BridgeResult,
    bayes_factor_from_logml,
    bridge_sampler as bridge_estimate,
    posterior_model_probabilities,
)
from quivers.diagnostics.elpd import response_hash
from quivers.effects.joint_plan import _has_custom_joint
from quivers.formulas._brms_priors import require_proper
from quivers.formulas._postprocess_loo import criterion_outcomes, unconstrained_draws
from quivers.inference._prior_projection import _has_potential, project_prior_sites
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.mcmc.kernel import PotentialFn
from quivers._values import ValueModel
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type EvidenceInput = Fit | BridgeResult
"""A fit, or a bridge-sampling estimate computed from one."""


class BayesFactor(ValueModel):
    """A Bayes factor of two models, as a ``bf_bridge`` object.

    Parameters
    ----------
    bf : float
        :math:`p(y \\mid M_1) / p(y \\mid M_2)`, or its logarithm.
    log : bool
        Whether ``bf`` is on the log scale.
    numerator, denominator : str
        Names of the numerator and denominator models.
    """

    bf: float
    log: bool
    numerator: str = "x1"
    denominator: str = "x2"


def _joint(fit: Fit) -> bool:
    """Return whether a fit is a joint multivariate fit.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.

    Returns
    -------
    bool
        True for a `MultivariateFit`.
    """
    return hasattr(fit, "response_fits")


def require_normalized(fit: Fit, purpose: str) -> None:
    """Refuse a fit whose log joint is not a normalized posterior kernel.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    purpose : str
        Operation named in the error.

    Raises
    ------
    ValueError
        If a prior is improper, the program carries an explicit score
        potential of unknown mass or an opaque joint override, or the
        posterior is not retained MCMC draws.
    """
    if _joint(fit):
        require_proper(cast("MultivariateFit", fit).emission.module, purpose)
    else:
        cast("BayesianFit", fit)._require_proper(purpose)
    if _has_custom_joint(fit.program):
        raise ValueError(
            f"{purpose} requires the canonical log joint; an overridden "
            "log_joint has no verifiable normalizing constant"
        )
    if _has_potential(project_prior_sites(fit.program)):
        raise ValueError(
            f"{purpose} requires a normalized prior, but the program carries an "
            "explicit score potential whose mass is unknown"
        )
    if not isinstance(fit.posterior, MCMCResult):
        raise ValueError(
            f"{purpose} requires posterior MCMC draws (method 'nuts' or 'hmc'); "
            "variational and prior draws are not draws of the posterior"
        )


def log_posterior_function(fit: Fit) -> tuple[np.ndarray, PotentialFn]:
    """Return a fit's unconstrained draws and its log-posterior potential.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fit with retained MCMC draws.

    Returns
    -------
    tuple[numpy.ndarray, PotentialFn]
        Float64 draws ``(chain, draw, D)`` in the sampler's flat layout, and
        the potential whose ``log_density`` is the Jacobian-corrected log joint.

    Raises
    ------
    ValueError
        If a retained draw is not a site of the latent registry (a discrete
        Gibbs site, for instance) or a registry site has no draws.
    """
    draws = fit.draws()
    registry, upars = unconstrained_draws(fit, draws)
    if set(draws) != set(registry.names):
        raise ValueError(
            "bridge sampling requires every latent site in the unconstrained "
            "registry; sites "
            f"{sorted(set(draws) ^ set(registry.names))} are discrete or "
            "outside it"
        )
    potential = PotentialFn(
        fit.program,
        registry,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fit.observations),
    )
    return upars.numpy(), potential


def bridge_sampler(
    fit: Fit,
    *,
    method: BridgeMethod = "normal",
    repetitions: int = 1,
    use_neff: bool = True,
    maxiter: int = 1000,
    seed: int = 0,
) -> BridgeResult:
    """Estimate a fit's log marginal likelihood by bridge sampling.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fit with posterior MCMC draws and proper priors.
    method : {"normal", "warp3"}
        Proposal of the ``bridgesampling`` package.
    repetitions : int
        Independent proposal draws.
    use_neff : bool
        Use the effective size of the iteration draws.
    maxiter : int
        Iteration bound of the fixed-point scheme.
    seed : int
        Seed of the proposal draws.

    Returns
    -------
    BridgeResult
        The estimate, with the retained log densities.

    Raises
    ------
    ValueError
        If the fit is refused by `require_normalized` or its draws are
        invalid for the estimator.
    RuntimeError
        If the scheme does not converge.

    Notes
    -----
    Cost is ``S / 2 (1 + R)`` program evaluations for ``S`` draws and ``R``
    repetitions (twice that for ``"warp3"``), each O(C) for one log-joint
    evaluation of the fit's program.
    """
    require_normalized(fit, "bridge_sampler")
    upars, potential = log_posterior_function(fit)

    def log_posterior(points: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian-corrected log joint row-wise.

        Parameters
        ----------
        points : numpy.ndarray
            Unconstrained points ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            Log densities ``(N,)``, ``-inf`` outside the support.
        """
        values = torch.from_numpy(np.ascontiguousarray(points))
        with torch.no_grad():
            return np.array(
                [float(potential.log_density(row)) for row in values],
                dtype=np.float64,
            )

    return bridge_estimate(
        log_posterior,
        upars,
        method=method,
        repetitions=repetitions,
        use_neff=use_neff,
        maxiter=maxiter,
        seed=seed,
    )


class BridgeSettings(ValueModel):
    """`bridge_sampler` keywords applied to every fit of a comparison.

    Parameters
    ----------
    method : {"normal", "warp3"}
        Proposal of the ``bridgesampling`` package.
    repetitions : int
        Independent proposal draws.
    use_neff : bool
        Use the effective size of the iteration draws.
    maxiter : int
        Iteration bound of the fixed-point scheme.
    seed : int
        Seed of the proposal draws.
    """

    method: BridgeMethod = "normal"
    repetitions: int = 1
    use_neff: bool = True
    maxiter: int = 1000
    seed: int = 0


def _logml(model: EvidenceInput, settings: BridgeSettings) -> float:
    """Return a model's log marginal likelihood.

    Parameters
    ----------
    model : BayesianFit or MultivariateFit or BridgeResult
        A fit, estimated with `bridge_sampler`, or a stored estimate.
    settings : BridgeSettings
        `bridge_sampler` keywords for fits.

    Returns
    -------
    float
        The log marginal likelihood.
    """
    if isinstance(model, BridgeResult):
        return model.logml
    return bridge_sampler(
        model,
        method=settings.method,
        repetitions=settings.repetitions,
        use_neff=settings.use_neff,
        maxiter=settings.maxiter,
        seed=settings.seed,
    ).logml


def _response_hash(model: EvidenceInput) -> str | None:
    """Return the hash of a fit's observed responses.

    Parameters
    ----------
    model : BayesianFit or MultivariateFit or BridgeResult
        A fit or an estimate.

    Returns
    -------
    str or None
        The hash, or None for a stored estimate.
    """
    if isinstance(model, BridgeResult):
        return None
    joint = _joint(model)
    return response_hash(criterion_outcomes(model, joint=joint, resp=None))


def match_responses(models: Sequence[EvidenceInput]) -> None:
    """Require that fits model the same response data, as brms ``match_response``.

    Parameters
    ----------
    models : Sequence[BayesianFit or MultivariateFit or BridgeResult]
        Compared models; stored estimates are not checked.

    Raises
    ------
    ValueError
        If two fits score different observed responses.
    """
    hashes = {h for h in map(_response_hash, models) if h is not None}
    if len(hashes) > 1:
        raise ValueError(
            "marginal likelihoods of models with different response data are "
            "not comparable"
        )


def bayes_factor(
    x1: EvidenceInput,
    x2: EvidenceInput,
    *,
    log: bool = False,
    model_names: tuple[str, str] = ("x1", "x2"),
    method: BridgeMethod = "normal",
    repetitions: int = 1,
    use_neff: bool = True,
    maxiter: int = 1000,
    seed: int = 0,
) -> BayesFactor:
    """Return the Bayes factor of two models, as brms ``bayes_factor``.

    Parameters
    ----------
    x1, x2 : BayesianFit or MultivariateFit or BridgeResult
        Numerator and denominator models, or their bridge estimates.
    log : bool
        Return the log Bayes factor.
    model_names : tuple[str, str]
        Names of the two models.
    method, repetitions, use_neff, maxiter, seed
        `bridge_sampler` settings for fits.

    Returns
    -------
    BayesFactor
        :math:`p(y \\mid M_1) / p(y \\mid M_2)`.

    Raises
    ------
    ValueError
        If the fits model different response data or are refused by
        `bridge_sampler`.
    """
    match_responses((x1, x2))
    options = BridgeSettings(
        method=method,
        repetitions=repetitions,
        use_neff=use_neff,
        maxiter=maxiter,
        seed=seed,
    )
    value = bayes_factor_from_logml(_logml(x1, options), _logml(x2, options), log=log)
    return BayesFactor(
        bf=value, log=log, numerator=model_names[0], denominator=model_names[1]
    )


def post_prob(
    models: Mapping[str, EvidenceInput],
    *,
    prior_prob: Sequence[float] | None = None,
    method: BridgeMethod = "normal",
    repetitions: int = 1,
    use_neff: bool = True,
    maxiter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Return posterior model probabilities, as brms ``post_prob``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit or BridgeResult]
        At least two named models of the same response data.
    prior_prob : Sequence[float] or None
        Prior model probabilities in the order of ``models``; uniform when None.
    method, repetitions, use_neff, maxiter, seed
        `bridge_sampler` settings for fits.

    Returns
    -------
    dict[str, float]
        Posterior probabilities summing to one.

    Raises
    ------
    ValueError
        If fewer than two models are given, the fits model different data,
        or the prior probabilities are invalid.
    """
    match_responses(tuple(models.values()))
    options = BridgeSettings(
        method=method,
        repetitions=repetitions,
        use_neff=use_neff,
        maxiter=maxiter,
        seed=seed,
    )
    return posterior_model_probabilities(
        {name: _logml(model, options) for name, model in models.items()},
        prior_prob=prior_prob,
    )


class EvidenceMethods:
    """Marginal-likelihood methods shared by `BayesianFit` and `MultivariateFit`."""

    def bridge_sampler(
        self,
        *,
        method: BridgeMethod = "normal",
        repetitions: int = 1,
        use_neff: bool = True,
        maxiter: int = 1000,
        seed: int = 0,
    ) -> BridgeResult:
        """Estimate the log marginal likelihood, as brms ``bridge_sampler``.

        Parameters
        ----------
        method : {"normal", "warp3"}
            Proposal of the ``bridgesampling`` package.
        repetitions : int
            Independent proposal draws.
        use_neff : bool
            Use the effective size of the iteration draws.
        maxiter : int
            Iteration bound of the fixed-point scheme.
        seed : int
            Seed of the proposal draws.

        Returns
        -------
        BridgeResult
            The estimate.

        Raises
        ------
        ValueError
            If the fit has improper priors, unnormalized potentials, discrete
            latent sites, or no posterior MCMC draws.
        RuntimeError
            If the scheme does not converge.
        """
        return bridge_sampler(
            cast("Fit", self),
            method=method,
            repetitions=repetitions,
            use_neff=use_neff,
            maxiter=maxiter,
            seed=seed,
        )

    def bayes_factor(
        self,
        other: EvidenceInput,
        *,
        log: bool = False,
        method: BridgeMethod = "normal",
        repetitions: int = 1,
        use_neff: bool = True,
        maxiter: int = 1000,
        seed: int = 0,
    ) -> BayesFactor:
        """Return the Bayes factor of this model over another.

        Parameters
        ----------
        other : BayesianFit or MultivariateFit or BridgeResult
            Denominator model or its estimate.
        log : bool
            Return the log Bayes factor.
        method, repetitions, use_neff, maxiter, seed
            `bridge_sampler` settings.

        Returns
        -------
        BayesFactor
            The Bayes factor.

        Raises
        ------
        ValueError
            If the models have different response data or a fit is refused.
        """
        return bayes_factor(
            cast("Fit", self),
            other,
            log=log,
            method=method,
            repetitions=repetitions,
            use_neff=use_neff,
            maxiter=maxiter,
            seed=seed,
        )


__all__ = [
    "BayesFactor",
    "BridgeSettings",
    "EvidenceMethods",
    "bayes_factor",
    "bridge_sampler",
    "log_posterior_function",
    "match_responses",
    "post_prob",
    "require_normalized",
]
