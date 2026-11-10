r"""Variational objectives.

An `Objective` is a torch.nn.Module-callable that, given
(model, guide, x, observations), returns a scalar loss whose
``backward()`` produces a gradient on the model and guide
parameters.

`ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`, `ChiVI` and
`DReGsBound` are Monte Carlo power-mean bounds of the per-draw log
importance ratios :math:`\ell_n = \log p(z_n, y) - \log q_\phi(z_n)`:

.. math::

    B_a = \frac{1}{a} \log \sum_n \omega_n e^{a \ell_n}, \qquad
    B_0 = \sum_n \omega_n \ell_n,

with exponent ``a = 0`` (ELBO), ``1`` (IWAE), ``1 - alpha`` (Rényi and
VR-IWAE) or ``n`` (`ChiVI`), and particle weights :math:`\omega_n`.
Each takes a [`quivers.inference.estimators.GradientEstimator`][quivers.inference.estimators.GradientEstimator]
that estimates :math:`\nabla \mathbb{E}[B_a]` for the model and
variational parameters, and checks at construction and on each call
that the estimator is valid for the bound; a pair without a valid
estimator raises `ValueError` naming one that is. `RWS` is not a
bound and takes no estimator: it implements the wake-phase updates of
reweighted wake-sleep.

Each particle is a call to `Guide.rsample_strata`, which returns
one or more weighted strata whose weighted average estimates
expectations under the guide density. An ordinary guide returns one
stratum; `AutoMixtureGuide` returns one per component, so every bound
here becomes its stratified counterpart for a mixture guide (the
stratified ELBO and SIWAE of
[Morningstar, Vikram, Ham, Gallagher and Dillon 2021](https://doi.org/10.48550/arXiv.2003.01687)),
with gradients to the mixture logits through the stratum weights. An
estimator that needs fixed draws (`ScoreFunction`) gets independent
`Guide.rsample` draws instead, detached before they are scored.
Per-draw log-densities are stacked along a leading axis of shape
``(N, batch)``. The draws run in a Python ``for`` loop over
``num_particles``, since the model's runtime path is not vectorized
over a Monte Carlo dimension; the per-step cost scales linearly in the
number of draws.

References
==========

- Standard ELBO: [Kingma and Welling 2013](https://doi.org/10.48550/arXiv.1312.6114).
- IWAE: [Burda, Grosse and Salakhutdinov 2016](https://doi.org/10.48550/arXiv.1509.00519).
- Rényi divergence VI: [Li and Turner 2016](https://doi.org/10.48550/arXiv.1602.02311).
- VR-IWAE: [Daudel, Benton, Shi and Doucet 2023](https://doi.org/10.48550/arXiv.2210.06226).
- CUBO / chi-square VI: [Dieng, Tran, Ranganath, Paisley and Blei 2017](https://doi.org/10.48550/arXiv.1611.00328).
- Reweighted wake-sleep: [Bornschein and Bengio 2015](https://doi.org/10.48550/arXiv.1406.2751).
- DReG: [Tucker, Lawson, Gu and Maddison 2019](https://doi.org/10.48550/arXiv.1810.04152).
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.estimators import (
    DoublyReparameterized,
    GradientEstimator,
    ParticleLogDensities,
    Reparameterized,
)
from quivers.inference.guides import Guide, GuideDraw


class Objective(nn.Module, ABC):
    """Base class for variational objectives.

    Subclasses implement `forward` to return a scalar loss whose
    ``backward()`` produces the objective's gradient.
    """

    @abstractmethod
    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Scalar loss for one optimisation step.

        Parameters
        ----------
        model : MonadicProgram
            The generative model.
        guide : Guide
            The variational guide.
        x : torch.Tensor
            Program input, shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor]
            Observed-site values.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        ...


# ---------------------------------------------------------------------------
# Multi-particle bookkeeping
# ---------------------------------------------------------------------------


def _draw_particles(
    model: MonadicProgram,
    guide: Guide,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    num_particles: int,
    estimator: GradientEstimator,
    power: float,
) -> ParticleLogDensities:
    """Draw ``num_particles`` particles and score them for ``estimator``.

    With ``estimator.fixed_draws`` each particle is one independent
    `Guide.rsample` draw, detached before it is scored by
    `Guide.log_prob`, with weight ``1/K``. Otherwise each particle is a
    call to `Guide.rsample_strata`, whose ``S`` weighted draws enter
    with log-weight :math:`\\log \\omega_s - \\log K` and are scored
    by the density each `GuideDraw` carries. The detached-parameter
    guide density and the model log-joint at detached draws are
    computed only when the estimator reads them.

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
        Number of particles ``K``.
    estimator : GradientEstimator
        The estimator the particles are for.
    power : float
        The bound's exponent.

    Returns
    -------
    ParticleLogDensities
        The stacked per-draw log-densities, shape ``(K * S, batch)``.

    Raises
    ------
    ValueError
        If the guide returns a different number of strata on
        different particles.
    """
    log_p_parts: list[torch.Tensor] = []
    log_q_parts: list[torch.Tensor] = []
    log_q_det_parts: list[torch.Tensor] = []
    log_p_fixed_parts: list[torch.Tensor] = []
    log_weight_parts: list[torch.Tensor] = []
    detached_guide: Guide | None = None
    if estimator.needs_detached_parameters:
        detached_guide = copy.deepcopy(guide)
        for parameter in detached_guide.parameters():
            parameter.requires_grad_(False)
    needs_fixed_log_p = estimator.needs_fixed_log_p(power)
    num_strata: int | None = None
    for _ in range(num_particles):
        if estimator.fixed_draws:
            fixed = {name: value.detach() for name, value in guide.rsample(x).items()}
            draws: tuple[GuideDraw, ...] = (
                GuideDraw(
                    sites=fixed,
                    log_density=guide.log_prob(x, fixed),
                    log_weight=torch.zeros((), device=x.device),
                ),
            )
        else:
            draws = guide.rsample_strata(x)
        if num_strata is None:
            num_strata = len(draws)
        elif len(draws) != num_strata:
            raise ValueError(
                f"{type(guide).__name__}.rsample_strata returned "
                f"{len(draws)} strata after returning {num_strata}; "
                f"the stratum count must not vary between particles"
            )
        for draw in draws:
            latents = draw.sites
            stratum_log_weight = draw.log_weight
            log_p = model.log_joint(x, {**latents, **observations})
            log_q = draw.log_density
            log_p_parts.append(log_p)
            log_q_parts.append(log_q)
            log_weight_parts.append(
                (stratum_log_weight.to(dtype=log_q.dtype) - math.log(num_particles))
                .reshape(1)
                .expand_as(log_q)
            )
            if detached_guide is not None:
                log_q_det_parts.append(detached_guide.log_prob(x, latents))
            if needs_fixed_log_p:
                fixed = {name: value.detach() for name, value in latents.items()}
                log_p_fixed_parts.append(model.log_joint(x, {**fixed, **observations}))
    return ParticleLogDensities(
        log_p=torch.stack(log_p_parts, dim=0),
        log_q=torch.stack(log_q_parts, dim=0),
        log_weights=torch.stack(log_weight_parts, dim=0),
        log_q_detached=(
            torch.stack(log_q_det_parts, dim=0) if detached_guide is not None else None
        ),
        log_p_fixed=(
            torch.stack(log_p_fixed_parts, dim=0) if needs_fixed_log_p else None
        ),
        stratified=num_strata is not None and num_strata > 1,
    )


class _PowerMeanBound(Objective):
    """A Monte Carlo power-mean bound with a gradient estimator.

    Parameters
    ----------
    name : str
        Class name used in error messages.
    power : float
        The bound's exponent :math:`a`.
    num_particles : int
        Number of particles per step.
    estimator : GradientEstimator
        The gradient estimator.
    minimize : bool
        Whether the loss is the bound itself (an upper bound to
        minimise) rather than its negation.

    Raises
    ------
    ValueError
        If ``num_particles < 1`` or the estimator has no valid form for
        this bound with ``num_particles`` independent draws.
    """

    estimator: GradientEstimator

    def __init__(
        self,
        name: str,
        power: float,
        num_particles: int,
        estimator: GradientEstimator,
        *,
        minimize: bool = False,
    ) -> None:
        super().__init__()
        if num_particles < 1:
            raise ValueError(f"{name}: num_particles must be >= 1, got {num_particles}")
        self.num_particles = num_particles
        self.power = float(power)
        self.estimator = estimator
        self._minimize = minimize
        estimator.check(self.power, num_particles, stratified=False)

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        particles = _draw_particles(
            model,
            guide,
            x,
            observations,
            self.num_particles,
            self.estimator,
            self.power,
        )
        self.estimator.check(
            self.power, int(particles.log_p.shape[0]), particles.stratified
        )
        surrogate = self.estimator.surrogate(self.power, particles).mean()
        return surrogate if self._minimize else -surrogate


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------


class ELBO(_PowerMeanBound):
    """Evidence lower bound objective.

    .. math::

        \\mathcal{L}_{\\mathrm{ELBO}}
            = \\mathbb{E}_{q_\\phi(z)} \\bigl[ \\log p(z, y) - \\log q_\\phi(z) \\bigr]
            \\leq \\log p(y).

    Returns the *negated* estimate so it can be plugged into a
    minimizer. ``num_particles`` averages independent Monte Carlo
    estimates (the bound's exponent is ``a = 0``). Every estimator is
    valid: `DoublyReparameterized` coincides with `StickingTheLanding`
    here.

    Parameters
    ----------
    num_particles : int
        Number of independent guide samples per step. Default ``1``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.

    Raises
    ------
    ValueError
        If ``num_particles < 1``.
    """

    def __init__(
        self,
        num_particles: int = 1,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(
            "ELBO",
            0.0,
            num_particles,
            estimator if estimator is not None else Reparameterized(),
        )


# ---------------------------------------------------------------------------
# IWAE
# ---------------------------------------------------------------------------


class IWAEBound(_PowerMeanBound):
    """Importance-weighted bound (Burda-Grosse-Salakhutdinov 2016).

    .. math::

        \\mathcal{L}_{\\mathrm{IWAE}}
            = \\mathbb{E}\\Bigl[\\log \\frac{1}{K} \\sum_{k=1}^{K}
                \\frac{p(z_k, y)}{q_\\phi(z_k)}\\Bigr] \\leq \\log p(y),

    tighter than the ELBO for :math:`K > 1` and approaching
    :math:`\\log p(y)` as :math:`K \\to \\infty` (exponent ``a = 1``).

    The default estimator is `DoublyReparameterized`, because the
    reparameterized gradient's signal-to-noise ratio for the
    variational parameters collapses as :math:`K` grows
    (Tucker-Lawson-Gu-Maddison 2019). `StickingTheLanding` is biased
    for this bound and is rejected when ``num_particles > 1``.

    For a guide with several strata per draw (`AutoMixtureGuide`) the
    bound is the stratified IWAE bound
    :math:`\\log \\frac{1}{K} \\sum_k \\sum_s \\omega_s\\,
    p(z_{ks}, y) / q_\\phi(z_{ks})`, still a lower bound on
    :math:`\\log p(y)`. `DoublyReparameterized` does not apply to
    stratified draws and raises; pass ``estimator=Reparameterized()``.

    Parameters
    ----------
    num_particles : int
        Number of guide samples per step. Default ``8``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `DoublyReparameterized`.

    Raises
    ------
    ValueError
        If ``num_particles < 1`` or the estimator is
        `StickingTheLanding` with ``num_particles > 1``.
    """

    def __init__(
        self,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        super().__init__(
            "IWAEBound",
            1.0,
            num_particles,
            estimator if estimator is not None else DoublyReparameterized(),
        )


# ---------------------------------------------------------------------------
# Rényi α-divergence
# ---------------------------------------------------------------------------


class RenyiBound(_PowerMeanBound):
    """Rényi α-divergence variational bound (Li-Turner 2016).

    .. math::

        \\mathcal{L}_\\alpha = \\frac{1}{1 - \\alpha}
            \\log \\mathbb{E}_q\\Bigl[ \\bigl(p(z, y) / q_\\phi(z)\\bigr)^{1-\\alpha}\\Bigr],

    estimated with ``num_particles`` draws as
    :math:`\\frac{1}{1-\\alpha} \\log \\frac{1}{K} \\sum_k
    (p_k/q_k)^{1-\\alpha}` (exponent ``a = 1 - alpha``), the same Monte
    Carlo estimator as `VRIWAEBound`. Recovers the ELBO as
    :math:`\\alpha \\to 1` and the IWAE bound at :math:`\\alpha = 0`.
    For :math:`\\alpha \\in [0, 1)` the estimate's expectation is a
    lower bound on :math:`\\log p(y)`. For :math:`\\alpha < 0` the
    exact :math:`\\mathcal{L}_\\alpha` is an upper bound, but the
    :math:`K`-sample estimate is biased below it, so its expectation is
    not guaranteed to bound :math:`\\log p(y)` from either side.

    `StickingTheLanding` is biased here and is rejected when
    ``num_particles > 1``; `DoublyReparameterized` is valid
    (Daudel-Benton-Shi-Doucet 2023, Theorem 2).

    Parameters
    ----------
    alpha : float
        Divergence order. ``alpha != 1``; values close to 1 may
        be numerically unstable.
    num_particles : int
        Number of guide samples per step. Default ``8``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.

    Raises
    ------
    ValueError
        If ``alpha == 1``, ``num_particles < 1``, or the estimator is
        `StickingTheLanding` with ``num_particles > 1``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        if alpha == 1.0:
            raise ValueError(
                "RenyiBound: alpha == 1.0 recovers the ELBO in the "
                "limit but is numerically singular here. Use the "
                "ELBO objective instead."
            )
        super().__init__(
            "RenyiBound",
            1.0 - alpha,
            num_particles,
            estimator if estimator is not None else Reparameterized(),
        )
        self.alpha = alpha


# ---------------------------------------------------------------------------
# VR-IWAE (Variational Rényi + IWAE unified)
# ---------------------------------------------------------------------------


class VRIWAEBound(_PowerMeanBound):
    """Variational Rényi-IWAE bound (Daudel-Benton-Shi-Doucet 2023).

    Unifies `ELBO`, `IWAEBound`, and
    `RenyiBound` into a single bound parameterized by
    ``alpha`` and ``num_particles``:

    .. math::

        \\mathcal{L}_{\\mathrm{VR\\text{-}IWAE}}
            = \\mathbb{E}\\Bigl[\\frac{1}{1 - \\alpha} \\,\\log\\,
              \\frac{1}{K} \\sum_{k=1}^{K} \\Bigl(\\frac{p}{q}\\Bigr)^{1-\\alpha}\\Bigr].

    Special cases:

    * ``alpha = 0, K > 1`` → IWAE bound.
    * ``K = 1`` → ELBO, for every ``alpha``.

    For ``alpha`` in ``[0, 1)`` it is a lower bound on
    :math:`\\log p(y)` that interpolates between the IWAE bound and the
    ELBO. `DoublyReparameterized` is the generalized DReG estimator of
    Theorem 2 of Daudel, Benton, Shi and Doucet (2023);
    `StickingTheLanding` is biased here and is rejected when
    ``num_particles > 1``.

    Parameters
    ----------
    alpha : float
        Divergence order; ``alpha != 1``.
    num_particles : int
        Number of guide samples per step. Default ``8``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.

    Raises
    ------
    ValueError
        If ``alpha == 1``, ``num_particles < 1``, or the estimator is
        `StickingTheLanding` with ``num_particles > 1``.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        if alpha == 1.0:
            raise ValueError("VRIWAEBound: alpha == 1.0 is singular. Use ELBO instead.")
        super().__init__(
            "VRIWAEBound",
            1.0 - alpha,
            num_particles,
            estimator if estimator is not None else Reparameterized(),
        )
        self.alpha = alpha


# ---------------------------------------------------------------------------
# CUBO / chi-square upper bound
# ---------------------------------------------------------------------------


class ChiVI(_PowerMeanBound):
    r"""Chi-squared variational upper bound (Dieng et al. 2017).

    Minimises the chi-squared divergence
    :math:`\chi^2(p \| q_\phi)` between the true posterior and the
    guide via the CUBO

    .. math::

        \mathcal{L}_{n\text{-CUBO}}(\phi)
            = \tfrac{1}{n} \log \mathbb{E}_{q_\phi}
              \Bigl[\bigl(p(z, y) / q_\phi(z)\bigr)^{n}\Bigr]
            \geq \log p(y),

    estimated with ``num_particles`` draws as
    :math:`\tfrac{1}{n} \log \frac{1}{K} \sum_k (p_k / q_k)^n`
    (exponent ``a = n``) and returned without negation, so minimising
    the loss minimises the estimate. By Jensen's inequality the
    estimate's expectation lies below the CUBO, so it is not itself
    guaranteed to be an upper bound on :math:`\log p(y)`; it
    approaches the CUBO as :math:`K` grows. The estimators target the
    gradient of the estimate's expectation; `DoublyReparameterized`
    uses the same reparameterization identity as for the lower bounds,
    and `StickingTheLanding` is biased here and is rejected when
    ``num_particles > 1``.

    Reference:
    [Dieng, Tran, Ranganath, Paisley and Blei 2017](https://doi.org/10.48550/arXiv.1611.00328).

    Parameters
    ----------
    n : float
        Divergence order (``n = 2`` for standard chi-squared).
        Must be ``> 0``.
    num_particles : int
        Number of guide samples per step. Default ``8``.
    estimator : GradientEstimator, optional
        Gradient-estimator strategy. Default `Reparameterized`.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``num_particles < 1``, or the estimator is
        `StickingTheLanding` with ``num_particles > 1``.
    """

    def __init__(
        self,
        n: float = 2.0,
        num_particles: int = 8,
        estimator: GradientEstimator | None = None,
    ) -> None:
        if n <= 0.0:
            raise ValueError(f"ChiVI: n must be > 0, got {n}")
        super().__init__(
            "ChiVI",
            float(n),
            num_particles,
            estimator if estimator is not None else Reparameterized(),
            minimize=True,
        )
        self.n = float(n)


# ---------------------------------------------------------------------------
# Reweighted wake-sleep
# ---------------------------------------------------------------------------


class RWS(Objective):
    r"""Reweighted wake-sleep, wake phase (Bornschein and Bengio 2015).

    Draws ``num_particles`` samples :math:`z_k \sim q_\phi`, holds them
    fixed, and forms self-normalised importance weights
    :math:`\tilde w_k \propto \omega_k\, p(z_k, y) / q_\phi(z_k)`
    (detached). The loss

    .. math::

        \mathcal{L}_{\mathrm{RWS}}
            = -\sum_k \tilde w_k \log p_\theta(z_k, y)
              -\sum_k \tilde w_k \log q_\phi(z_k)

    gives the wake-phase model update
    :math:`\sum_k \tilde w_k \nabla_\theta \log p_\theta(z_k, y)`
    and the wake-phase guide update
    :math:`\sum_k \tilde w_k \nabla_\phi \log q_\phi(z_k)`, a
    self-normalised importance-sampling estimate of
    :math:`-\nabla_\phi \mathrm{KL}(p(z \mid y) \| q_\phi)`.
    Because the draws are fixed, neither term moves the guide through
    the draws, and the model term does not move the guide at all. With
    an `AutoMixtureGuide` the draws are its stratified draws and
    :math:`\omega_k` the stratum weights. The sleep phase, which trains
    an amortised guide on samples from the model, is not part of this
    objective. RWS is not a bound and takes no gradient estimator.

    Reference:
    [Bornschein and Bengio 2015](https://doi.org/10.48550/arXiv.1406.2751).

    Parameters
    ----------
    num_particles : int
        Number of guide samples per step. Default ``8``.

    Raises
    ------
    ValueError
        If ``num_particles < 1``.
    """

    def __init__(self, num_particles: int = 8) -> None:
        super().__init__()
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
        log_p_parts: list[torch.Tensor] = []
        log_q_parts: list[torch.Tensor] = []
        log_weight_parts: list[torch.Tensor] = []
        for _ in range(self.num_particles):
            for draw in guide.rsample_strata(x):
                stratum_log_weight = draw.log_weight
                fixed = {name: value.detach() for name, value in draw.sites.items()}
                log_p = model.log_joint(x, {**fixed, **observations})
                log_q = guide.log_prob(x, fixed)
                log_p_parts.append(log_p)
                log_q_parts.append(log_q)
                log_weight_parts.append(
                    stratum_log_weight.detach().to(dtype=log_q.dtype).expand_as(log_q)
                )
        log_p_all = torch.stack(log_p_parts, dim=0)
        log_q_all = torch.stack(log_q_parts, dim=0)
        with torch.no_grad():
            log_w = torch.stack(log_weight_parts, dim=0) + log_p_all - log_q_all
            w_tilde = torch.softmax(log_w, dim=0)
        wake_theta = -(w_tilde * log_p_all).sum(dim=0)
        wake_phi = -(w_tilde * log_q_all).sum(dim=0)
        return (wake_theta + wake_phi).mean()


# ---------------------------------------------------------------------------
# Doubly reparameterised gradient bound
# ---------------------------------------------------------------------------


class DReGsBound(_PowerMeanBound):
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

    The DReG surrogate requires independent draws from the
    guide; with a stratified guide (`AutoMixtureGuide`) it raises,
    and `IWAEBound` with ``estimator=Reparameterized()`` gives the
    stratified bound instead.

    Parameters
    ----------
    num_particles : int
        Number of guide samples per step. DReG's variance-
        reduction benefit is largest for ``K >= 8``.

    Raises
    ------
    ValueError
        If ``num_particles < 1``.
    """

    def __init__(self, num_particles: int = 8) -> None:
        super().__init__("DReGsBound", 1.0, num_particles, DoublyReparameterized())


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
