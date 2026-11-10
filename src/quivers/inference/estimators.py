r"""Gradient estimators for variational objectives.

The lower-bound objectives in [`quivers.inference.objectives`][quivers.inference.objectives]
are *power-mean bounds* of the per-draw log importance ratios
:math:`\ell_n = \log p(z_n, y) - \log q_\phi(z_n)` with particle
weights :math:`\omega_n` (``1/K`` for independent draws, the stratum
weights for stratified ones):

.. math::

    B_a = \frac{1}{a} \log \sum_n \omega_n e^{a \ell_n}
    \quad (a \neq 0), \qquad
    B_0 = \sum_n \omega_n \ell_n.

``a = 0`` is the ELBO, ``a = 1`` the IWAE bound, and ``a = 1 - \alpha``
the Rényi / VR-IWAE bound. A
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

`ChiVI` instead minimizes the exponentiated upper bound
:math:`L_n = \mathbb{E}_q[(p/q)^n] = \exp(n\, \mathrm{CUBO}_n)` of Dieng,
Tran, Ranganath, Paisley and Blei (2017), whose Monte Carlo estimate is
unbiased; it calls `GradientEstimator.check_moment` and
`GradientEstimator.moment_surrogate`, which `Reparameterized` (their
Eq. 3) and `ScoreFunction` (their Algorithm 1) implement and the other
estimators reject.

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
- CHIVI: [Dieng, Tran, Ranganath, Paisley and Blei 2017](https://doi.org/10.48550/arXiv.1611.00328).
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
    num_particles
        Number of independent particles ``K``; the ``N`` draws are
        ``K`` consecutive groups of ``N / K`` strata.
    """

    log_p: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_q: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_q_detached: torch.Tensor | None = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_p_fixed: torch.Tensor | None = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    stratified: bool
    num_particles: int


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


def leave_one_out_log_scale(
    scaled_log_ratio: torch.Tensor, num_particles: int
) -> torch.Tensor:
    """Per-draw stabilizing log-scale from the other particles' draws.

    For a draw in particle :math:`k` the scale is
    :math:`m_{-k} = \\max_{j \\notin k} n \\ell_j`, the largest scaled
    log ratio among the draws of the other particles. It is independent
    of particle :math:`k`'s own draws, so dividing particle :math:`k`'s
    term by :math:`e^{m_{-k}}` rescales its expected gradient by the
    positive constant :math:`\\mathbb{E}[e^{-m_{-k}}]` without biasing
    its direction.

    Parameters
    ----------
    scaled_log_ratio : torch.Tensor
        :math:`n \\ell`, shape ``(K * S, batch)`` in particle-major order.
    num_particles : int
        ``K``; at least 2.

    Returns
    -------
    torch.Tensor
        Detached, shape ``(K * S, batch)``.

    Raises
    ------
    ValueError
        If ``num_particles < 2``.
    """
    if num_particles < 2:
        raise ValueError(
            f"leave_one_out_log_scale: needs at least 2 particles, got {num_particles}"
        )
    values = scaled_log_ratio.detach()
    per_draw = values.shape[0] // num_particles
    grouped = values.reshape(num_particles, per_draw, *values.shape[1:])
    particle_max = grouped.amax(dim=1)  # (K, batch)
    top = particle_max.topk(2, dim=0)
    first, second = top.values[0], top.values[1]
    index = torch.arange(num_particles, device=values.device).reshape(
        num_particles, *([1] * (particle_max.dim() - 1))
    )
    other = torch.where(index == top.indices[0], second, first)  # (K, batch)
    return other.unsqueeze(1).expand_as(grouped).reshape(values.shape)


def _moment_terms(order: float, particles: ParticleLogDensities) -> torch.Tensor:
    """Detached per-draw coefficients ``omega_n (p_n/q_n)^n / e^{m_{-k}}``.

    Raises
    ------
    RuntimeError
        If a coefficient overflows, which happens only when one
        particle's scaled log weight exceeds every other particle's by
        more than the floating-point exponent range.
    """
    log_ratio = (particles.log_p - particles.log_q).detach()
    scaled = order * log_ratio
    shift = leave_one_out_log_scale(scaled, particles.num_particles)
    terms = torch.exp(particles.log_weights.detach() + scaled - shift)
    if not torch.isfinite(terms).all():
        raise RuntimeError(
            "ChiVI: a scaled importance weight overflowed, because one "
            "particle's weight exceeds all others' by more than the "
            "floating-point range; the Monte Carlo estimate of E[(p/q)^n] is "
            "degenerate here. Use more particles or a wider guide."
        )
    return terms


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

    def check_moment(self, order: float, num_particles: int, stratified: bool) -> None:
        """Raise unless this estimator applies to the exponentiated CUBO.

        `ChiVI` minimizes :math:`L_n = \\mathbb{E}_q[(p/q)^n]` and calls
        this at construction and on each call. The default rejects:
        an estimator supports :math:`L_n` only if it implements
        `moment_surrogate`.

        Parameters
        ----------
        order : float
            The order :math:`n`.
        num_particles : int
            Particles per batch row.
        stratified : bool
            Whether the draws are stratified.

        Raises
        ------
        ValueError
            Always, naming the estimators that apply.
        """
        del order, num_particles, stratified
        raise ValueError(
            f"{type(self).__name__} has no unbiased estimator for the "
            f"exponentiated CUBO exp(n * CUBO_n) that ChiVI minimizes. Use "
            f"estimator=Reparameterized() (Dieng et al. 2017, Eq. 3) or "
            f"estimator=ScoreFunction() (their Algorithm 1)."
        )

    def moment_surrogate(
        self, order: float, particles: ParticleLogDensities
    ) -> torch.Tensor:
        """Surrogate whose gradient estimates :math:`\\nabla L_n`, scaled.

        Parameters
        ----------
        order : float
            The order :math:`n`.
        particles : ParticleLogDensities
            The drawn particles' log-densities.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``. Its gradient's expectation is
            :math:`c\\, \\nabla L_n` for a positive constant :math:`c`
            (see `ChiVI`); its value is not meaningful.

        Raises
        ------
        ValueError
            For an estimator that does not support :math:`L_n`.
        """
        self.check_moment(order, particles.num_particles, particles.stratified)
        raise ValueError(f"{type(self).__name__} does not support ChiVI")

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

    def check_moment(self, order: float, num_particles: int, stratified: bool) -> None:
        del order, num_particles, stratified

    def moment_surrogate(
        self, order: float, particles: ParticleLogDensities
    ) -> torch.Tensor:
        # Dieng et al. Eq. 3: grad (p/q)^n = n (p/q)^n grad log(p/q), each
        # term divided by the other particles' largest weight.
        log_ratio = particles.log_p - particles.log_q
        terms = _moment_terms(order, particles)
        log_weights = particles.log_weights
        pathwise = terms * order * (log_ratio - log_ratio.detach())
        weight_path = terms * (log_weights - log_weights.detach())
        return (pathwise + weight_path).sum(dim=0)


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

    def check_moment(self, order: float, num_particles: int, stratified: bool) -> None:
        self.check(order, num_particles, stratified)

    def moment_surrogate(
        self, order: float, particles: ParticleLogDensities
    ) -> torch.Tensor:
        # Dieng et al. Algorithm 1 and Appendix C: at fixed draws,
        # grad_phi E[(p/q)^n] = (1 - n) E[(p/q)^n grad log q] and
        # grad_theta E[(p/q)^n] = n E[(p/q)^n grad log p].
        self.check_moment(order, particles.num_particles, particles.stratified)
        terms = _moment_terms(order, particles)
        log_p, log_q = particles.log_p, particles.log_q
        per_draw = order * (log_p - log_p.detach()) + (1.0 - order) * (
            log_q - log_q.detach()
        )
        return (terms * per_draw).sum(dim=0)


__all__ = [
    "GradientEstimator",
    "ParticleLogDensities",
    "power_mean_bound",
    "leave_one_out_log_scale",
    "Reparameterized",
    "StickingTheLanding",
    "DoublyReparameterized",
    "ScoreFunction",
]
