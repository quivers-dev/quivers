"""Variational objectives.

An `Objective` is a torch.nn.Module-callable that, given
(model, guide, x, observations), returns a scalar loss whose
``backward()`` produces a gradient on the model and guide
parameters. The most common objective is `ELBO`; tighter
multi-sample bounds (`IWAEBound`, `RenyiBound`, `VRIWAEBound`,
`DReGsBound`) trade compute for bound-tightness. `ChiVI` gives
an upper bound on the marginal likelihood via the chi-squared
divergence, and `RWS` implements reweighted wake-sleep for
inference nets over discrete latents.

Every objective accepts a [`quivers.inference.estimators.GradientEstimator`][quivers.inference.estimators.GradientEstimator]
strategy that decides how the per-particle log-density tensors
are turned into a scalar loss whose gradient is the chosen
estimator. The default is [`quivers.inference.estimators.Reparameterized`][quivers.inference.estimators.Reparameterized].

Each particle is a call to `Guide.rsample_strata`, which returns
one or more weighted strata whose weighted average estimates
expectations under the density `Guide.log_prob` scores. An ordinary
guide returns one stratum; `AutoMixtureGuide` returns one per
component, so every objective here becomes its stratified
counterpart for a mixture guide (the stratified ELBO and SIWAE of
[Morningstar, Vikram, Ham, Gallagher and Dillon 2021](https://doi.org/10.48550/arXiv.2003.01687)),
an unbiased estimate of the same quantity for the mixture density,
with gradients to the mixture logits through the stratum weights.
Per-draw log-densities are stacked along a leading torch axis of
shape ``(K * S, batch)``, with a log-weight per draw, before the
estimator turns them into a scalar loss. The particle draws themselves run in a
Python ``for`` loop over ``num_particles``, since the model's
runtime path and the underlying torch.distributions calls are
not vectorized over a Monte Carlo dimension; the per-step cost
scales linearly in ``K``.

References
==========

- Standard ELBO: [Kingma and Welling 2013](https://doi.org/10.48550/arXiv.1312.6114).
- IWAE: [Burda, Grosse and Salakhutdinov 2016](https://doi.org/10.48550/arXiv.1509.00519).
- Rényi divergence VI: [Li and Turner 2016](https://doi.org/10.48550/arXiv.1602.02311).
- VR-IWAE: [Daudel, Douc and Roueff 2023](https://doi.org/10.48550/arXiv.2210.06226).
- CUBO / chi-square VI: [Dieng, Tran, Ranganath, Paisley and Blei 2017](https://doi.org/10.48550/arXiv.1611.00328).
- Reweighted wake-sleep: [Bornschein and Bengio 2015](https://doi.org/10.48550/arXiv.1406.2751).
- DReG surrogate: [Tucker, Lawson, Gu and Maddison 2019](https://doi.org/10.48550/arXiv.1810.04152).
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod

import didactic.api as dx
import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.estimators import (
    DoublyReparameterized,
    GradientEstimator,
    Reparameterized,
)
from quivers.inference.guides import Guide


class Objective(nn.Module, ABC):
    """Base class for variational objectives.

    Subclasses implement `forward` to return a scalar loss
    (negated objective). The ``estimator`` attribute is the
    gradient-estimation strategy applied to the per-particle
    log-densities.
    """

    estimator: GradientEstimator

    def __init__(self, estimator: GradientEstimator | None = None) -> None:
        super().__init__()
        self.estimator = estimator if estimator is not None else Reparameterized()

    @abstractmethod
    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Multi-particle bookkeeping
# ---------------------------------------------------------------------------


class _ParticleLogDensities(dx.Model):
    """Per-particle log-densities of a multi-particle objective.

    Parameters
    ----------
    log_p
        Model log-joint at each particle, shape ``(N, batch)``.
    log_q
        Guide log-density at each particle, shape ``(N, batch)``.
    log_q_detached
        Guide log-density with the variational parameters detached,
        shape ``(N, batch)``, or ``None`` when not requested.
    log_weights
        Log-weight of each particle, shape ``(N, batch)``, with
        ``logsumexp`` over the particle axis equal to zero. Uniform
        (``-log N``) for independent draws; for stratified draws the
        stratum log-weights (minus ``log K``), which keep their
        gradient to the variational parameters.
    stratified
        Whether the guide drew more than one stratum per particle.
    """

    log_p: torch.Tensor = dx.field(opaque=True)
    log_q: torch.Tensor = dx.field(opaque=True)
    log_q_detached: torch.Tensor | None = dx.field(opaque=True)
    log_weights: torch.Tensor = dx.field(opaque=True)
    stratified: bool

    @property
    def estimator_log_weights(self) -> torch.Tensor | None:
        """The ``log_weights`` argument for a `GradientEstimator`.

        Returns
        -------
        torch.Tensor or None
            The particle log-weights for stratified particles;
            ``None`` for independent draws from the guide, which the
            estimators weight uniformly.
        """
        return self.log_weights if self.stratified else None

    def log_mean_weight(self, power: float) -> torch.Tensor:
        """Estimate of :math:`\\log \\mathbb{E}_q[(p / q)^{a}]`.

        Computes :math:`\\log \\sum_n \\omega_n
        \\exp\\bigl(a (\\log p_n - \\log q_n)\\bigr)`, the log of an
        unbiased estimate of :math:`\\mathbb{E}_q[(p / q)^{a}]`
        (plain importance sampling for independent draws, stratified
        importance sampling for stratified ones).

        Parameters
        ----------
        power : float
            The exponent :math:`a`.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.
        """
        log_w = self.log_p - self.log_q
        return torch.logsumexp(self.log_weights + power * log_w, dim=0)


def _multi_particle_log_densities(
    model: MonadicProgram,
    guide: Guide,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    num_particles: int,
    *,
    need_detached: bool,
) -> _ParticleLogDensities:
    """Run ``num_particles`` independent stratified guide draws through
    the model's log-joint.

    Each particle calls `Guide.rsample_strata`, which returns ``S``
    strata (``S = 1`` for a guide whose draws are reparameterized as a
    whole; one per component for `AutoMixtureGuide`). The ``K * S``
    stratum draws are stacked along a leading particle axis of size
    ``N = K * S``, and each carries the log-weight
    :math:`\\log \\omega_s - \\log K`, so a weighted sum over the
    particle axis is an unbiased estimate of an expectation under the
    density that `Guide.log_prob` scores.

    For ``need_detached=True`` ``log q`` is also computed with the
    guide's parameters detached (needed by sticking-the-landing and
    DReG).

    The particle loop is a Python ``for`` over ``num_particles``,
    since torch.distributions and the model's runtime path are not
    vectorized over a Monte Carlo dimension.

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    guide : Guide
        The variational guide.
    x : torch.Tensor
        Program input.
    observations : dict[str, torch.Tensor]
        Observed-site values.
    num_particles : int
        Number of independent stratified draws ``K``.
    need_detached : bool
        Whether to compute the detached guide log-density.

    Returns
    -------
    _ParticleLogDensities
        The stacked per-particle log-densities and log-weights.

    Raises
    ------
    ValueError
        If the guide returns a different number of strata on
        different particles.
    """
    log_p_parts: list[torch.Tensor] = []
    log_q_parts: list[torch.Tensor] = []
    log_q_det_parts: list[torch.Tensor] = []
    log_weight_parts: list[torch.Tensor] = []
    detached_guide: Guide | None = None
    if need_detached:
        detached_guide = copy.deepcopy(guide)
        for p in detached_guide.parameters():
            p.requires_grad_(False)
    num_strata: int | None = None
    for _ in range(num_particles):
        stratum_log_weights, stratum_draws = guide.rsample_strata(x)
        if num_strata is None:
            num_strata = len(stratum_draws)
        elif len(stratum_draws) != num_strata:
            raise ValueError(
                f"{type(guide).__name__}.rsample_strata returned "
                f"{len(stratum_draws)} strata after returning {num_strata}; "
                f"the stratum count must not vary between particles"
            )
        for stratum_log_weight, latents in zip(stratum_log_weights, stratum_draws):
            all_sites = {**latents, **observations}
            log_p = model.log_joint(x, all_sites)
            log_q = guide.log_prob(x, latents)
            log_p_parts.append(log_p)
            log_q_parts.append(log_q)
            log_weight_parts.append(
                (stratum_log_weight.to(dtype=log_q.dtype) - math.log(num_particles))
                .reshape(1)
                .expand_as(log_q)
            )
            if detached_guide is not None:
                log_q_det_parts.append(detached_guide.log_prob(x, latents))
    return _ParticleLogDensities(
        log_p=torch.stack(log_p_parts, dim=0),
        log_q=torch.stack(log_q_parts, dim=0),
        log_q_detached=(
            torch.stack(log_q_det_parts, dim=0) if detached_guide is not None else None
        ),
        log_weights=torch.stack(log_weight_parts, dim=0),
        stratified=num_strata is not None and num_strata > 1,
    )


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------


class ELBO(Objective):
    """Evidence lower bound objective.

    .. math::

        \\mathcal{L}_{\\mathrm{ELBO}}
            = \\mathbb{E}_{q_\\phi(z)} \\bigl[ \\log p(z, y) - \\log q_\\phi(z) \\bigr].

    Returns the *negated* ELBO so `Objective.forward` can be
    plugged into a minimizer. ``num_particles`` averages independent
    Monte-Carlo estimates; ``num_particles == 1`` is the standard
    reparameterization-trick ELBO.

    Parameters
    ----------
    num_particles : int
        Number of independent guide samples per step. Default ``1``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.
    """

    def __init__(
        self,
        num_particles: int = 1,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(estimator=estimator)
        if num_particles < 1:
            raise ValueError(f"ELBO: num_particles must be >= 1, got {num_particles}")
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        need_detached = not isinstance(self.estimator, Reparameterized)
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=need_detached,
        )
        # ELBO loss = − E_q[log p − log q], estimated by the weighted
        # sum over particles; the estimator determines how the
        # gradient flows.
        return self.estimator.negative_objective(
            particles.log_p,
            particles.log_q,
            particles.log_q_detached,
            log_weights=particles.estimator_log_weights,
        )


# ---------------------------------------------------------------------------
# IWAE
# ---------------------------------------------------------------------------


class IWAEBound(Objective):
    """Importance-weighted bound (Burda-Grosse-Salakhutdinov 2016).

    .. math::

        \\mathcal{L}_{\\mathrm{IWAE}}
            = \\mathbb{E}\\Bigl[\\log \\frac{1}{K} \\sum_{k=1}^{K}
                \\frac{p(z_k, y)}{q_\\phi(z_k)}\\Bigr],

    a tighter lower bound on :math:`\\log p(y)` than the ELBO.
    Approaches the marginal likelihood as :math:`K \\to \\infty`.

    The default estimator is `DoublyReparameterized`
    because the naive reparameterized gradient's signal-to-noise
    ratio for the inference network collapses as :math:`K` grows
    (Tucker-Lawson-Gu-Maddison 2019).

    For a guide with several strata per draw (`AutoMixtureGuide`)
    the bound is the stratified IWAE bound
    :math:`\\log \\frac{1}{K} \\sum_k \\sum_s \\omega_s\\,
    p(z_{ks}, y) / q_\\phi(z_{ks})`, still a lower bound on
    :math:`\\log p(y)`. `DoublyReparameterized` does not apply to
    stratified draws and raises; pass ``estimator=Reparameterized()``.
    """

    def __init__(
        self,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        if estimator is None:
            estimator = DoublyReparameterized()
        super().__init__(estimator=estimator)
        if num_particles < 1:
            raise ValueError(
                f"IWAEBound: num_particles must be >= 1, got {num_particles}"
            )
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # IWAE always needs the detached log_q for the DReG path;
        # for plain reparameterized IWAE we still compute it so
        # the estimator interface is uniform.
        need_detached = True
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=need_detached,
        )
        # If the chosen estimator is DReG, hand the surrogate
        # construction off to it (it builds the importance-weight²
        # surrogate, and rejects stratified particles). For other
        # estimators the loss is the IWAE bound itself,
        # log Σ_n ω_n p_n / q_n (SIWAE for stratified particles).
        if isinstance(self.estimator, DoublyReparameterized):
            return self.estimator.negative_objective(
                particles.log_p,
                particles.log_q,
                particles.log_q_detached,
                log_weights=particles.estimator_log_weights,
            )
        bound = particles.log_mean_weight(1.0)
        # Return the negated bound averaged over the batch.
        return -(bound.mean())


# ---------------------------------------------------------------------------
# Rényi α-divergence
# ---------------------------------------------------------------------------


class RenyiBound(Objective):
    """Rényi α-divergence variational bound (Li-Turner 2016).

    .. math::

        \\mathcal{L}_\\alpha = \\frac{1}{1 - \\alpha}
            \\log \\mathbb{E}_q\\Bigl[ \\bigl(p(z, y) / q_\\phi(z)\\bigr)^{1-\\alpha}\\Bigr].

    Recovers the ELBO at :math:`\\alpha = 1` (in the limit) and
    the IWAE bound at :math:`\\alpha = 0`. The interesting regime
    is :math:`\\alpha < 0`, which gives an *upper* bound on
    :math:`\\log p(y)` and so a tighter posterior-mode estimate
    when the variational family is too narrow.

    Parameters
    ----------
    alpha : float
        Divergence order. ``alpha != 1``; values close to 1 may
        be numerically unstable.
    num_particles : int
        Number of guide samples per step.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(estimator=estimator)
        if alpha == 1.0:
            raise ValueError(
                "RenyiBound: alpha == 1.0 recovers the ELBO in the "
                "limit but is numerically singular here. Use the "
                "ELBO objective instead."
            )
        if num_particles < 1:
            raise ValueError(
                f"RenyiBound: num_particles must be >= 1, got {num_particles}"
            )
        self.alpha = alpha
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        need_detached = not isinstance(self.estimator, Reparameterized)
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=need_detached,
        )
        # log E_q[(p/q)^{1-α}] ≈ log Σ_n ω_n (p_n / q_n)^{1-α}
        a = 1.0 - self.alpha
        bound = particles.log_mean_weight(a) / a
        return -(bound.mean())


# ---------------------------------------------------------------------------
# VR-IWAE (Variational Rényi + IWAE unified)
# ---------------------------------------------------------------------------


class VRIWAEBound(Objective):
    """Variational Rényi-IWAE bound (Daudel-Douc-Roueff 2023).

    Unifies `ELBO`, `IWAEBound`, and
    `RenyiBound` into a single bound parameterized by
    ``alpha`` and ``num_particles``:

    .. math::

        \\mathcal{L}_{\\mathrm{VR\\text{-}IWAE}}
            = \\frac{1}{1 - \\alpha} \\,\\log\\,
              \\frac{1}{K} \\sum_{k=1}^{K} \\Bigl(\\frac{p}{q}\\Bigr)^{1-\\alpha}.

    Special cases:

    * ``alpha = 0, K > 1`` → IWAE bound.
    * ``alpha = 0, K = 1`` → ELBO.
    * ``alpha != 0, K = 1`` → Rényi α-VI.

    For intermediate ``alpha`` the bound interpolates between
    "cheap, biased" (high α) and "expensive, tight" (low α).
    """

    def __init__(
        self,
        alpha: float = 0.0,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(estimator=estimator)
        if alpha == 1.0:
            raise ValueError("VRIWAEBound: alpha == 1.0 is singular. Use ELBO instead.")
        if num_particles < 1:
            raise ValueError(
                f"VRIWAEBound: num_particles must be >= 1, got {num_particles}"
            )
        self.alpha = alpha
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        need_detached = not isinstance(self.estimator, Reparameterized)
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=need_detached,
        )
        a = 1.0 - self.alpha
        bound = particles.log_mean_weight(a) / a
        return -(bound.mean())


# ---------------------------------------------------------------------------
# CUBO / chi-square upper bound
# ---------------------------------------------------------------------------


class ChiVI(Objective):
    r"""Chi-squared variational upper bound (Dieng et al. 2017).

    Minimises the chi-squared divergence
    :math:`\chi^2(p \| q_\phi)` between the true posterior and the
    guide via the CUBO surrogate

    .. math::

        \mathcal{L}_{n\text{-CUBO}}(\phi)
            = \tfrac{1}{n} \log \mathbb{E}_{q_\phi}
              \Bigl[\bigl(p(z, y) / q_\phi(z)\bigr)^{n}\Bigr]
            \geq \log p(y).

    In contrast to the ELBO (a lower bound), this is an *upper*
    bound on :math:`\log p(y)`, so minimising it drives the guide
    to over-cover the true posterior. Useful for posterior
    calibration and for sandwich estimates when paired with ELBO.

    Reference:
    [Dieng, Tran, Ranganath, Paisley and Blei 2017](https://doi.org/10.48550/arXiv.1611.00328).

    Parameters
    ----------
    n : float
        Divergence order (``n = 2`` for standard chi-squared).
        Must be ``> 0``.
    num_particles : int
        Number of guide samples per step.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.
    """

    def __init__(
        self,
        n: float = 2.0,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(estimator=estimator)
        if n <= 0.0:
            raise ValueError(f"ChiVI: n must be > 0, got {n}")
        if num_particles < 1:
            raise ValueError(f"ChiVI: num_particles must be >= 1, got {num_particles}")
        self.n = float(n)
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        need_detached = not isinstance(self.estimator, Reparameterized)
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=need_detached,
        )
        # CUBO_n = (1/n) log E_q [ (p/q)^n ]
        #        ≈ (1/n) log Σ_n ω_n (p_n / q_n)^n
        cubo = particles.log_mean_weight(self.n) / self.n
        # Minimise CUBO directly (it's an upper bound; we want it
        # small). Loss is the batch mean without negation.
        return cubo.mean()


# ---------------------------------------------------------------------------
# Reweighted wake-sleep
# ---------------------------------------------------------------------------


class RWS(Objective):
    r"""Reweighted wake-sleep (Bornschein and Bengio 2015).

    Combines a wake-phase gradient on the model (importance-weighted
    log-likelihood) with a sleep-phase gradient on the guide (KL from
    the self-normalised importance-weighted posterior). RWS handles
    discrete latents where the reparameterization trick does not
    apply and where score-function estimators are high-variance.

    The loss returned combines both phases:

    .. math::

        \mathcal{L}_{\mathrm{RWS}}
            = -\mathbb{E}_{q}\bigl[\tilde w_k \log p(z_k, y)\bigr]
              -\mathbb{E}_{q}\bigl[\tilde w_k \log q_\phi(z_k)\bigr],

    where :math:`\tilde w_k` are self-normalised importance weights
    :math:`w_k / \sum_j w_j` with :math:`w_k = p(z_k, y) / q(z_k)`,
    detached from the gradient path so both phases are unbiased.

    Reference:
    [Bornschein and Bengio 2015](https://doi.org/10.48550/arXiv.1406.2751).

    Parameters
    ----------
    num_particles : int
        Number of guide samples per step.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`;
        the RWS surrogate is estimator-agnostic because the weights
        are detached.
    """

    def __init__(
        self,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(estimator=estimator)
        if num_particles < 1:
            raise ValueError(f"RWS: num_particles must be >= 1, got {num_particles}")
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # RWS needs detached log_q for the wake-guide phase because
        # the wake-guide surrogate is E_q[w * log q] with w computed
        # from the current (detached) guide but scored under the
        # differentiable guide.
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=True,
        )
        log_p = particles.log_p
        log_q = particles.log_q
        log_q_det = particles.log_q_detached
        assert log_q_det is not None
        # Self-normalised importance weights ω_n p_n / q_n from the
        # detached log-densities (no gradient through the weights).
        # The particle weights ω_n are uniform for independent draws
        # and the stratum weights for stratified ones.
        with torch.no_grad():
            log_w = particles.log_weights + log_p.detach() - log_q_det
            log_norm = torch.logsumexp(log_w, dim=0, keepdim=True)
            w_tilde = torch.exp(log_w - log_norm)
        # Wake-theta: -E_q[w_tilde * log p]. Guide is treated as the
        # proposal, model log-density carries the gradient.
        wake_theta = -(w_tilde * log_p).sum(dim=0)
        # Wake-phi: -E_q[w_tilde * log q]. Model log-density is
        # detached (already baked into w_tilde); guide log-density
        # carries the gradient.
        wake_phi = -(w_tilde * log_q).sum(dim=0)
        return (wake_theta + wake_phi).mean()


# ---------------------------------------------------------------------------
# Doubly reparameterised gradient bound
# ---------------------------------------------------------------------------


class DReGsBound(Objective):
    r"""Doubly reparameterised IWAE bound (Tucker et al. 2019).

    The bound is the IWAE bound; the estimator is the DReG
    surrogate that removes the score-function term whose signal-
    to-noise ratio collapses as :math:`K` grows in the naive
    reparameterised IWAE gradient. Equivalent to
    ``IWAEBound(num_particles=K, estimator=DoublyReparameterized())``
    but exposed as a first-class objective for callers that want
    to switch bound + estimator together.

    Reference:
    [Tucker, Lawson, Gu and Maddison 2019](https://doi.org/10.48550/arXiv.1810.04152).

    Note: `DReGsBound` here is a *bound* + *estimator* pair; the
    scalar gradient-estimator strategy also called "DReG" lives at
    [`quivers.inference.estimators.DoublyReparameterized`][quivers.inference.estimators.DoublyReparameterized]
    and can be attached to any objective through the ``estimator``
    argument.

    The DReG surrogate requires independent draws from the
    guide; with a stratified guide (`AutoMixtureGuide`) it raises,
    and `IWAEBound` with ``estimator=Reparameterized()`` gives the
    stratified bound instead.

    Parameters
    ----------
    num_particles : int
        Number of guide samples per step. DReG's variance-
        reduction benefit is largest for ``K >= 8``.
    """

    def __init__(self, num_particles: int = 8) -> None:
        super().__init__(estimator=DoublyReparameterized())
        if num_particles < 1:
            raise ValueError(
                f"DReGsBound: num_particles must be >= 1, got {num_particles}"
            )
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        particles = _multi_particle_log_densities(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            need_detached=True,
        )
        return self.estimator.negative_objective(
            particles.log_p,
            particles.log_q,
            particles.log_q_detached,
            log_weights=particles.estimator_log_weights,
        )


__all__ = [
    "Objective",
    "ELBO",
    "IWAEBound",
    "RenyiBound",
    "VRIWAEBound",
    "ChiVI",
    "RWS",
    "DReGsBound",
]
