r"""Gradient estimators for variational objectives.

Every Monte Carlo objective in [`quivers.inference.objectives`][quivers.inference.objectives]
is a *power-mean bound* of the per-draw log importance ratios
:math:`\ell_n = \log p(z_n, y) - \log q_\phi(z_n)` with particle
weights :math:`\omega_n` (``1/K`` for independent draws, the stratum
weights for stratified ones):

.. math::

    B_a = \frac{1}{a} \log \sum_n \omega_n e^{a \ell_n}
    \quad (a \neq 0), \qquad
    B_0 = \sum_n \omega_n \ell_n.

``a = 0`` is the ELBO, ``a = 1`` the IWAE bound, ``a = 1 - \alpha``
the Rényi / VR-IWAE bound, and ``a = n`` the CUBO of `ChiVI`. A
`GradientEstimator` turns the per-draw log-densities into a surrogate
whose value is :math:`B_a` and whose gradient is an estimate of
:math:`\nabla \mathbb{E}[B_a]`, the gradient of the expected Monte
Carlo objective, for both the variational parameters :math:`\phi` and
the model parameters :math:`\theta`:

* `Reparameterized`: the pathwise gradient of :math:`B_a`. Unbiased
  for every ``a``.
* `StickingTheLanding`: drops the direct dependence of
  :math:`\log q_\phi` on :math:`\phi` and keeps the pathwise one
  (Roeder, Wu and Duvenaud 2017). The dropped term has mean zero only
  for the ELBO or a single draw, so other objectives with several
  draws reject it.
* `DoublyReparameterized`: rewrites the dropped score term with the
  reparameterization identity instead of discarding it (Tucker, Lawson,
  Gu and Maddison 2019 for IWAE; Daudel, Benton, Shi and Doucet 2023,
  Theorem 2, for VR-IWAE). Unbiased for independent draws; it
  coincides with sticking the landing at ``a = 0``.
* `ScoreFunction`: REINFORCE (Ranganath, Gerrish and Blei 2014) with
  the draws held fixed, so it applies to guides whose draws are not
  differentiable.

Estimators are stateless strategies. `Objective` implementations call
`GradientEstimator.check` with their bound and draw count, so an
(objective, estimator) pair without a valid estimator raises before
it is used, and `GradientEstimator.surrogate` on the particles they
draw.

References
==========

- Reparameterization: [Kingma and Welling 2013](https://doi.org/10.48550/arXiv.1312.6114).
- Sticking the landing: [Roeder, Wu and Duvenaud 2017](https://doi.org/10.48550/arXiv.1703.09194).
- DReG: [Tucker, Lawson, Gu and Maddison 2019](https://doi.org/10.48550/arXiv.1810.04152).
- Generalized DReG for VR-IWAE: [Daudel, Benton, Shi and Doucet 2023](https://doi.org/10.48550/arXiv.2210.06226).
- Score function: [Ranganath, Gerrish and Blei 2014](https://doi.org/10.48550/arXiv.1401.0118).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel


class ParticleLogDensities(ValueModel):
    """Per-draw log-densities that an objective hands its estimator.

    All tensors share the shape ``(N, batch)``, where ``N`` is the
    number of draws per batch row (particles times strata). The
    tensors compare by identity: two records are equal exactly when
    they hold the same tensor objects and agree on ``stratified``.

    Parameters
    ----------
    log_p
        Model log-joint :math:`\\log p_\\theta(z_n, y)` at each draw,
        differentiable in the model parameters and, unless the draws
        are fixed, in the variational parameters through the draw.
    log_q
        Guide log-density :math:`\\log q_\\phi(z_n)` at each draw.
    log_weights
        Log particle weights :math:`\\log \\omega_n`, with ``logsumexp``
        over the draw axis equal to zero: ``-log K`` for independent
        draws, the stratum log-weights (minus ``log K``) for stratified
        ones. Stratum weights keep their gradient to the variational
        parameters that set them.
    log_q_detached
        :math:`\\log q_{\\bar\\phi}(z_n)` with the variational
        parameters detached but the draw not, or ``None`` when the
        estimator does not need it.
    log_p_fixed
        :math:`\\log p_\\theta(\\bar z_n, y)` at the detached draw, or
        ``None`` when the estimator does not need it.
    stratified
        Whether the draws are stratified rather than independent draws
        from the guide.
    """

    log_p: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_q: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_q_detached: torch.Tensor | None = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_p_fixed: torch.Tensor | None = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    stratified: bool


def power_mean_bound(
    log_ratio: torch.Tensor, log_weights: torch.Tensor, power: float
) -> torch.Tensor:
    """The power-mean bound :math:`B_a` of weighted log ratios.

    Parameters
    ----------
    log_ratio : torch.Tensor
        Per-draw :math:`\\ell_n`, shape ``(N, batch)``.
    log_weights : torch.Tensor
        Per-draw :math:`\\log \\omega_n`, broadcastable to
        ``log_ratio``, normalized over the draw axis.
    power : float
        The exponent :math:`a`; ``0`` gives the weighted mean.

    Returns
    -------
    torch.Tensor
        :math:`\\frac{1}{a} \\log \\sum_n \\omega_n e^{a \\ell_n}`, or
        :math:`\\sum_n \\omega_n \\ell_n` at ``a = 0``; shape ``(batch,)``.
    """
    if power == 0.0:
        return (log_weights.exp() * log_ratio).sum(dim=0)
    return torch.logsumexp(log_weights + power * log_ratio, dim=0) / power


def _require_detached(estimator: str, particles: ParticleLogDensities) -> torch.Tensor:
    if particles.log_q_detached is None:
        raise RuntimeError(
            f"{estimator}: the particles carry no log_q_detached (the guide "
            f"log-density with detached variational parameters); the "
            f"objective must compute it when needs_detached_parameters is set"
        )
    return particles.log_q_detached


class GradientEstimator(ABC):
    """Strategy for estimating :math:`\\nabla \\mathbb{E}[B_a]`.

    Attributes
    ----------
    fixed_draws : bool
        Whether the estimator needs the guide's draws held fixed. When
        true, objectives draw independent samples with `Guide.rsample`,
        detach them, and score them, so ``log_q`` carries only its direct
        dependence on the variational parameters.
    needs_detached_parameters : bool
        Whether the estimator reads `ParticleLogDensities.log_q_detached`.
    """

    fixed_draws: bool = False
    needs_detached_parameters: bool = False

    def needs_fixed_log_p(self, power: float) -> bool:
        """Whether the estimator reads `ParticleLogDensities.log_p_fixed`.

        Parameters
        ----------
        power : float
            The bound's exponent :math:`a`.

        Returns
        -------
        bool
            ``False`` unless a subclass needs the model log-joint at
            the detached draws.
        """
        del power
        return False

    def check(self, power: float, num_draws: int, stratified: bool) -> None:
        """Raise if this estimator has no valid form for the bound.

        Objectives call this at construction with their particle count
        and again on each call with the drawn particles.

        Parameters
        ----------
        power : float
            The bound's exponent :math:`a`.
        num_draws : int
            Draws per batch row.
        stratified : bool
            Whether the draws are stratified.

        Raises
        ------
        ValueError
            If the estimator is biased or undefined for this bound.
        """
        del power, num_draws, stratified

    @abstractmethod
    def surrogate(self, power: float, particles: ParticleLogDensities) -> torch.Tensor:
        """Surrogate for the bound :math:`B_a`.

        Parameters
        ----------
        power : float
            The bound's exponent :math:`a`.
        particles : ParticleLogDensities
            The drawn particles' log-densities.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``. Its value is :math:`B_a` and its
            gradient is this estimator's estimate of
            :math:`\\nabla \\mathbb{E}[B_a]`.

        Raises
        ------
        RuntimeError
            If the particles lack a tensor the estimator needs.
        """
        ...


class Reparameterized(GradientEstimator):
    """Pathwise gradient of the bound (Kingma and Welling 2013).

    Differentiates :math:`B_a` through the reparameterized draws, the
    guide density, and the particle weights. Unbiased for every bound.
    """

    def surrogate(self, power: float, particles: ParticleLogDensities) -> torch.Tensor:
        return power_mean_bound(
            particles.log_p - particles.log_q, particles.log_weights, power
        )


class StickingTheLanding(GradientEstimator):
    """Sticking-the-landing gradient (Roeder, Wu and Duvenaud 2017).

    Replaces :math:`\\log q_\\phi(z)` with
    :math:`\\log q_{\\bar\\phi}(z)`, whose parameters are detached
    while the draw is not, so the gradient keeps only the pathwise
    dependence through :math:`z` (and through stratum weights). The
    dropped term is :math:`-\\sum_n \\bar w_n \\nabla_\\phi \\log
    q_\\phi(z_n)` with :math:`\\bar w = \\mathrm{softmax}(\\log \\omega
    + a \\ell)`. Its expectation is zero for the ELBO (``a = 0``, where
    :math:`\\bar w = \\omega`) and for a single draw, and nonzero
    otherwise, so `check` rejects other bounds with several draws; for
    those, `DoublyReparameterized` is the unbiased counterpart. As
    :math:`q \\to p` the gradient's variance vanishes.
    """

    needs_detached_parameters = True

    def check(self, power: float, num_draws: int, stratified: bool) -> None:
        if power == 0.0 or (num_draws == 1 and not stratified):
            return
        alternative = "Reparameterized()" if stratified else "DoublyReparameterized()"
        raise ValueError(
            "StickingTheLanding is unbiased only for the ELBO or a single "
            f"draw; this bound (exponent {power}) uses {num_draws} draws per "
            f"row, where dropping the score term biases the gradient. Use "
            f"estimator={alternative}."
        )

    def surrogate(self, power: float, particles: ParticleLogDensities) -> torch.Tensor:
        log_q_detached = _require_detached("StickingTheLanding", particles)
        return power_mean_bound(
            particles.log_p - log_q_detached, particles.log_weights, power
        )


class DoublyReparameterized(GradientEstimator):
    """Doubly reparameterized gradient (DReG).

    The gradient of :math:`B_a` contains the score term
    :math:`-\\sum_n \\bar w_n \\nabla_\\phi \\log q_\\phi(z_n)` with
    :math:`\\bar w = \\mathrm{softmax}(\\log \\omega + a \\ell)`. The
    reparameterization identity
    :math:`\\mathbb{E}_q[g(z) \\nabla_\\phi \\log q_\\phi(z)] =
    \\mathbb{E}_\\epsilon[\\nabla_z g(z)\\, \\partial z / \\partial \\phi]`
    rewrites it as a pathwise term, giving

    .. math::

        \\nabla_\\phi \\mathbb{E}[B_a] = \\mathbb{E}\\Bigl[\\sum_n
            \\bigl((1 - a)\\bar w_n + a \\bar w_n^2\\bigr)
            \\nabla_z \\bigl(\\log p(z_n) - \\log q_{\\bar\\phi}(z_n)\\bigr)
            \\frac{\\partial z_n}{\\partial \\phi}\\Bigr].

    At ``a = 1`` this is the IWAE estimator of Tucker, Lawson, Gu and
    Maddison (2019); for ``a = 1 - \\alpha`` it is Theorem 2 of Daudel,
    Benton, Shi and Doucet (2023); at ``a = 0`` it is sticking the
    landing. The derivation uses only the identity above, so it holds
    for every ``a``, including `ChiVI`. Model parameters get
    :math:`\\sum_n \\bar w_n \\nabla_\\theta \\log p_\\theta(z_n)`, the
    gradient of :math:`B_a` itself.

    The identity requires each draw to come from :math:`q_\\phi`, so
    stratified draws (an `AutoMixtureGuide` enumerating its components)
    are rejected unless ``a = 0``.
    """

    needs_detached_parameters = True

    def needs_fixed_log_p(self, power: float) -> bool:
        return power != 0.0

    def check(self, power: float, num_draws: int, stratified: bool) -> None:
        del num_draws
        if stratified and power != 0.0:
            raise ValueError(
                "DoublyReparameterized: the guide supplies stratified draws "
                "(for instance, an AutoMixtureGuide enumerates its components). "
                "The DReG identity rewrites the score of the density each draw "
                "comes from, and stratified draws come from the strata rather "
                "than from q, so the estimator would be biased. Use "
                "estimator=Reparameterized()."
            )

    def surrogate(self, power: float, particles: ParticleLogDensities) -> torch.Tensor:
        log_q_detached = _require_detached("DoublyReparameterized", particles)
        if power == 0.0:
            return power_mean_bound(
                particles.log_p - log_q_detached, particles.log_weights, power
            )
        if particles.stratified:
            self.check(power, particles.log_p.shape[0], stratified=True)
        log_p_fixed = particles.log_p_fixed
        if log_p_fixed is None:
            raise RuntimeError(
                "DoublyReparameterized: the particles carry no log_p_fixed (the "
                "model log-joint at the detached draws); the objective must "
                "compute it when needs_fixed_log_p is true"
            )
        log_weights = particles.log_weights.detach()
        log_ratio = (particles.log_p - particles.log_q).detach()
        normalized = torch.softmax(log_weights + power * log_ratio, dim=0)
        coefficients = (1.0 - power) * normalized + power * normalized**2
        pathwise = particles.log_p - log_q_detached
        value = power_mean_bound(log_ratio, log_weights, power)
        # The variational gradient flows through the pathwise term with
        # the DReG coefficients; the model gradient is moved from those
        # coefficients to the bound's own weights with log p at the
        # detached draws, which carries no variational gradient.
        return (
            value
            + (coefficients * (pathwise - pathwise.detach())).sum(dim=0)
            + ((normalized - coefficients) * (log_p_fixed - log_p_fixed.detach())).sum(
                dim=0
            )
        )


class ScoreFunction(GradientEstimator):
    """REINFORCE / black-box VI gradient (Ranganath, Gerrish and Blei 2014).

    Uses the log-derivative identity with the draws held fixed:

    .. math::

        \\nabla_\\phi \\mathbb{E}_{q_\\phi}[B_a]
            = \\mathbb{E}\\Bigl[\\nabla_\\phi B_a\\big|_{z}
              + B_a \\sum_n \\nabla_\\phi \\log q_\\phi(z_n)\\Bigr].

    For the ELBO (``a = 0``) the first term has mean zero and the
    second decouples across draws, so the estimate is
    :math:`\\sum_n \\omega_n \\ell_n \\nabla_\\phi \\log q_\\phi(z_n)`.
    The objectives draw independent samples and detach them
    (`GradientEstimator.fixed_draws`), so the estimator applies to
    guides whose ``rsample`` is not differentiable. Model parameters
    receive :math:`\\nabla_\\theta B_a` at the fixed draws. Variance is
    typically orders of magnitude higher than the reparameterized
    estimator's.
    """

    fixed_draws = True

    def check(self, power: float, num_draws: int, stratified: bool) -> None:
        del power, num_draws
        if stratified:
            raise ValueError(
                "ScoreFunction: needs independent draws from the guide, each "
                "scored with the density it was drawn from; stratified draws "
                "do not satisfy this. Objectives draw independent samples when "
                "this estimator is selected."
            )

    def surrogate(self, power: float, particles: ParticleLogDensities) -> torch.Tensor:
        self.check(power, particles.log_p.shape[0], particles.stratified)
        log_q = particles.log_q
        log_q_fixed = log_q.detach()
        score = log_q - log_q_fixed
        log_weights = particles.log_weights.detach()
        if power == 0.0:
            # Value: sum_n w_n (log p - log q). Gradient: w_n * l_n * score
            # for the guide and w_n * grad log p for the model.
            log_ratio = (particles.log_p - log_q).detach()
            per_draw = particles.log_p - log_q_fixed + log_ratio * score
            return (log_weights.exp() * per_draw).sum(dim=0)
        bound = power_mean_bound(particles.log_p - log_q, log_weights, power)
        return bound + bound.detach() * score.sum(dim=0)


__all__ = [
    "GradientEstimator",
    "ParticleLogDensities",
    "power_mean_bound",
    "Reparameterized",
    "StickingTheLanding",
    "DoublyReparameterized",
    "ScoreFunction",
]
