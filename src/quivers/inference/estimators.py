"""Gradient estimators for variational objectives.

A `GradientEstimator` is the strategy that takes (latent
samples, model log-density, guide log-density) and returns a
scalar loss whose ``backward()`` produces the chosen gradient
estimator. Different strategies trade variance against
applicability:

* `Reparameterized`: pathwise gradient (the standard SVI
  reparameterization trick). Lowest variance for reparameterizable
  families; requires ``rsample``.
* `StickingTheLanding`: detaches the variational-parameter
  dependence in :math:`\\log q_\\phi(z)` so the gradient variance
  asymptotically vanishes as :math:`q \\to p^*`
  (Roeder-Wu-Duvenaud 2017,
  `doi:10.48550/arXiv.1703.09194 <https://doi.org/10.48550/arXiv.1703.09194>`_).
* `DoublyReparameterized`: the DReG estimator for IWAE
  (Tucker-Lawson-Gu-Maddison 2019,
  `doi:10.48550/arXiv.1810.04152 <https://doi.org/10.48550/arXiv.1810.04152>`_).
  Removes the score-function term whose variance grows with the
  particle count :math:`K`.
* `ScoreFunction`: REINFORCE / black-box VI. The
  fallback for non-reparameterizable sites (discrete latents,
  reject-sampled families). Highest variance; pair with a
  baseline whenever possible.

Objectives pass per-particle log-weights when the guide draws
stratified particles (`Guide.rsample_strata`, as for
`AutoMixtureGuide`); the estimators then weight each particle's
term by its stratum weight. `DoublyReparameterized` rejects
stratified particles.

Estimators are *strategies* held by `Objective`
implementations; they don't store any state themselves and
operate on tensors only. The `Reparameterized` instance
is a singleton; every objective defaults to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GradientEstimator(ABC):
    """Strategy for computing :math:`\\nabla_\\phi \\mathcal{L}` from
    samples + densities.

    Subclasses implement `negative_objective`: given the
    per-particle ``log_p`` and ``log_q`` tensors (and any
    estimator-specific auxiliary data), return the *negated*
    objective whose ``backward()`` produces the desired gradient
    estimator.
    """

    @abstractmethod
    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scalar loss whose gradient is the chosen
        estimator.

        Parameters
        ----------
        log_p : torch.Tensor
            Model log-joint ``log p(z, y)`` at the sampled latents.
            Shape ``(K, batch)`` where ``K`` is the particle axis
            (``K = 1`` for plain ELBO) and ``batch`` is the
            program-input batch axis.
        log_q : torch.Tensor
            Guide log-density ``log q_phi(z)`` at the sampled
            latents. Same shape as ``log_p``. Gradients flow back
            to the variational parameters through this tensor.
        log_q_detached : torch.Tensor or None
            ``log q_{stop_grad(phi)}(z)``, the guide log-density
            with the variational parameters detached from the
            autograd graph. Required by sticking-the-landing and
            DReG; ignored by the basic estimators.
        log_weights : torch.Tensor or None
            Log-weights of the particles, broadcastable to
            ``log_p``'s shape, with ``logsumexp`` over the particle
            axis equal to zero. They are supplied when the particles
            are stratified draws (see `Guide.rsample_strata`): the
            weighted sum :math:`\\sum_n \\omega_n f(z_n)` is then the
            unbiased estimate of :math:`\\mathbb{E}_q[f]`, and the
            weights carry gradient to the variational parameters
            that set them (such as mixture logits). ``None`` means the
            particles are independent draws from :math:`q` with equal
            weight :math:`1/K`.

        Returns
        -------
        torch.Tensor
            Scalar loss.

        Raises
        ------
        RuntimeError
            If the estimator needs an input the caller did not
            supply, or cannot be applied to stratified particles.
        """
        ...


class Reparameterized(GradientEstimator):
    """Standard pathwise gradient.

    For the ELBO with ``num_particles = 1`` this is the textbook
    reparameterization trick (Kingma-Welling 2013,
    `doi:10.48550/arXiv.1312.6114 <https://doi.org/10.48550/arXiv.1312.6114>`_).
    For higher ``num_particles`` it's the importance-weighted
    score function with reparameterized samples, i.e. the IWAE
    bound under the naive gradient.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_q_detached
        # ELBO with K particles:
        #   L = Σ_k ω_k [ log p(z_k) − log q(z_k) ],  ω_k = 1/K for i.i.d. draws
        # Stratified draws carry their own ω_k, which stay in the
        # graph so the gradient reaches the parameters that set them.
        diff = log_p - log_q
        if log_weights is None:
            return -(diff.mean())
        return -((log_weights.exp() * diff).sum(dim=0).mean())


class StickingTheLanding(GradientEstimator):
    """Roeder-Wu-Duvenaud 2017 sticking-the-landing estimator.

    Replaces ``log q(z)`` in the loss with
    ``log q_{detach(phi)}(z)``: the score is evaluated at the
    same sample but the variational parameters are detached
    from the autograd graph. The total derivative loses its
    direct dependence on the variational parameters through
    ``log q``, leaving only the indirect dependence through the
    sampled ``z``. As :math:`q \\to p^*` the latter vanishes and
    so does the gradient variance.

    Use when training with a guide that's already close to the
    true posterior, typically after a warm-up phase. May
    *increase* variance early in training when ``q`` is far
    from ``p``.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if log_q_detached is None:
            raise RuntimeError(
                "StickingTheLanding: requires the caller to supply "
                "log_q_detached (log q evaluated with the variational "
                "parameters detached). The objective in use does not "
                "produce this; check that the Objective implementation "
                "supplies log_q_detached when this estimator is selected."
            )
        del log_q
        diff = log_p - log_q_detached
        if log_weights is None:
            return -(diff.mean())
        # The dropped direct term has zero expectation under the
        # stratified weights as well, since Σ_s ω_s E_{q_s}[g] = E_q[g];
        # the weights themselves keep their gradient.
        return -((log_weights.exp() * diff).sum(dim=0).mean())


class DoublyReparameterized(GradientEstimator):
    """Doubly-reparameterized IWAE gradient (Tucker-Lawson-Gu-
    Maddison 2019).

    Specialised for the IWAE bound at K particles. Reweights the
    per-particle terms so the variance no longer collapses as
    :math:`K \\to \\infty`. The objective itself is the standard
    IWAE bound; only the gradient is reweighted.

    The estimator's mathematical content is the *gradient*:

    .. math::

        \\nabla_\\phi \\mathcal{L}_{\\mathrm{IWAE}} \\;=\\;
        \\sum_k w_k^2 \\, \\nabla_\\phi
            \\bigl[\\log p(z_k) - \\log q_\\phi(z_k)\\bigr]

    where :math:`w_k = \\exp(\\log p_k - \\log q_k) /
    \\sum_j \\exp(\\log p_j - \\log q_j)`. Implementing this as a
    surrogate loss whose ``backward()`` yields the right gradient
    is the standard trick: detach the importance weights from the
    autograd graph and use them as a non-differentiable scaling
    on the per-particle reparameterized difference.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if log_q_detached is None:
            raise RuntimeError(
                "DoublyReparameterized: requires the caller to supply "
                "log_q_detached (log q evaluated with the variational "
                "parameters detached). The IWAE objective produces this "
                "natively; confirm the objective is IWAEBound."
            )
        if log_weights is not None:
            raise RuntimeError(
                "DoublyReparameterized: the guide supplies stratified "
                "particles (for instance, an AutoMixtureGuide enumerates its "
                "components). The DReG identity rewrites the score of the "
                "density each particle was drawn from, and stratified "
                "particles are drawn from the strata rather than from q, so "
                "the surrogate would be biased. Use estimator=Reparameterized() "
                "or estimator=StickingTheLanding() with this guide."
            )
        if log_p.dim() < 1:
            raise RuntimeError(
                "DoublyReparameterized: log_p must have a leading "
                f"particle axis (K, batch); got shape {tuple(log_p.shape)}"
            )
        # w_k = softmax_k (log p_k - log q_k) along the particle
        # axis (axis 0). Detach the weights so they don't
        # contribute to the gradient.
        log_w = log_p - log_q_detached
        weights = torch.softmax(log_w, dim=0).detach()
        # Surrogate loss whose gradient is the DReG estimator.
        surrogate = (weights**2) * (log_p - log_q)
        return -(surrogate.sum(dim=0).mean())


class ScoreFunction(GradientEstimator):
    """REINFORCE / black-box VI gradient
    (Ranganath-Gerrish-Blei 2014,
    `doi:10.48550/arXiv.1401.0118 <https://doi.org/10.48550/arXiv.1401.0118>`_).

    Uses the log-derivative identity instead of the
    reparameterization trick. Required when sampling is not
    differentiable (discrete latents, hard-truncated families,
    accept-reject samplers). Variance is typically orders of
    magnitude higher than reparameterized; combine with a
    control-variate baseline whenever possible.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_q_detached
        # Score-function estimator: ∇φ E_q[f(z)] = E_q[f(z) ∇φ log q(z)].
        # As a surrogate loss whose backward gives the right
        # gradient: detach f(z) so it contributes as a scalar
        # weight on the score ∇φ log q(z).
        f = (log_p - log_q).detach()
        if log_weights is None:
            return -((f * log_q).mean())
        # E_q[f ∇ log q] is itself an expectation under q, so the
        # stratified draws estimate it with detached weights.
        return -((log_weights.exp().detach() * f * log_q).sum(dim=0).mean())


__all__ = [
    "GradientEstimator",
    "Reparameterized",
    "StickingTheLanding",
    "DoublyReparameterized",
    "ScoreFunction",
]
