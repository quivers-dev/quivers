r"""Differentiable annealed importance sampling (DAIS) guide.

DAIS runs a short chain of uncorrected Hamiltonian transitions from a
base guide :math:`q_0` toward the posterior and uses the chain's
endpoint as the variational draw (Zhang, Hsu, Li, Finn and Grosse
2021; Geffner and Domke 2021). Transition :math:`k` draws a fresh
momentum :math:`v_{k-1} \sim \mathcal{N}(0, I)` and applies leapfrog
steps on the bridge :math:`\pi_k \propto q_0^{1-\beta_k} p^{\beta_k}`,
ending at :math:`(u_k, \hat v_k)`; there is no Metropolis correction.
The step sizes, the inverse-temperature schedule :math:`\beta_k`, and
the base guide's parameters are all variational parameters, and the
whole trajectory is differentiated through, including the bridge
gradients inside each leapfrog step.

Leapfrog is volume preserving, so the forward chain over
:math:`(u_0, v_0, \dots, v_{K-1})` and its time reversal define the
extended-space importance weight (Zhang et al. 2021, Eq. 15)

.. math::

    \log w = \log p(u_K, y) - \log q_0(u_0)
        + \sum_{k=1}^{K} \bigl[\log \mathcal{N}(\hat v_k)
                              - \log \mathcal{N}(v_{k-1})\bigr],

with densities taken in the unconstrained space where the chain runs
(each including its bijector's log-Jacobian). Its expectation under
the forward chain is :math:`\mathbb{E}[w] = p(y)`, so
:math:`\mathbb{E}[\log w] \le \log p(y)` is a valid lower bound
and the multi-sample objectives built from ``w`` are valid bounds as
well. The bridges and step sizes shape the chain, not the validity of
the bound; an uncorrected chain does not leave the bridges invariant,
so the AIS weight :math:`\sum_k (\beta_k - \beta_{k-1})
(\log p - \log q_0)` evaluated along it would not be a valid bound and
is not used. The bound is not guaranteed to improve on the base guide's
ELBO, and a fitted run need not recover every posterior mode.

`AutoDAIS.rsample_strata` returns the endpoint together with its
extended-space log-density
:math:`\log q_0(u_0) - \log|J(u_K)| + \sum_k [K(\hat v_k) - K(v_{k-1})]`
(:math:`K` the kinetic energy), so that ``log p - log q`` for the
draw is exactly :math:`\log w`. The endpoint has no closed-form
marginal density, so `AutoDAIS.log_prob` raises: the estimators and
objectives that need the guide density at a fixed point
(`StickingTheLanding`, `DoublyReparameterized`, `ScoreFunction`,
`RWS`) are not available with this guide; use the reparameterized
estimator.

References: Zhang, Hsu, Li, Finn and Grosse (2021),
[Differentiable Annealed Importance Sampling and the Perils of Gradient
Noise](https://doi.org/10.48550/arXiv.2107.10211); Geffner and Domke
(2021), [MCMC Variational Inference via Uncorrected Hamiltonian
Annealing](https://doi.org/10.48550/arXiv.2107.04150).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide, GuideDraw
from quivers.inference.registry import _require_one_run


class AutoDAIS(Guide):
    """Annealed-importance-sampling guide wrapping a base guide.

    Parameters
    ----------
    base : Guide
        Base variational guide (the chain's start). Must be built
        against the same model + observed-name set the DAIS guide
        is for, and must have a density at arbitrary points.
        `AutoNormalGuide` or `AutoMultivariateNormalGuide` are the
        canonical choices.
    model : MonadicProgram
        Generative model. Required because the bridges need the target
        log-density.
    observations : dict[str, torch.Tensor]
        Observed-site values that pin the model's likelihood in the
        bridges. They shape the dynamics only: the bound that an
        objective evaluates uses that objective's own input and
        observations, and is valid for any of them, though the chain
        is tuned for these.
    x : torch.Tensor
        Program input fed to the model's log-joint inside the bridges:
        one row, since the bridges target one run's posterior. Default
        ``torch.zeros(1, 1)``.
    num_steps : int
        Number of Hamiltonian transitions along the path; at least 1.
        Default ``8``.
    leapfrog_steps : int
        Leapfrog substeps per transition. Default ``1``.
    init_step_size : float
        Initial leapfrog step size, one learnable value per transition.
    init_temperature : float
        Initial first inverse temperature, in ``(0, 1)``. The schedule
        anneals from this value at the first transition to ``1.0`` at
        the last and is learnable.

    Raises
    ------
    ValueError
        If ``num_steps < 1``, ``leapfrog_steps < 1``,
        ``init_step_size <= 0``, ``init_temperature`` is outside
        ``(0, 1)``, the base registry has no continuous latents, or
        ``x`` has more than one row.
    """

    def __init__(
        self,
        base: Guide,
        model: MonadicProgram,
        observations: dict[str, torch.Tensor],
        x: torch.Tensor | None = None,
        num_steps: int = 8,
        leapfrog_steps: int = 1,
        init_step_size: float = 0.1,
        init_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError(f"AutoDAIS: num_steps must be >= 1, got {num_steps}")
        if leapfrog_steps < 1:
            raise ValueError(
                f"AutoDAIS: leapfrog_steps must be >= 1, got {leapfrog_steps}"
            )
        if init_step_size <= 0:
            raise ValueError(
                f"AutoDAIS: init_step_size must be > 0, got {init_step_size}"
            )
        if not 0.0 < init_temperature < 1.0:
            raise ValueError(
                f"AutoDAIS: init_temperature must be in (0, 1), got {init_temperature}"
            )
        self.base = base
        self._registry = base.registry
        self._model = model
        self._observations = observations
        self._x = x if x is not None else torch.zeros(1, 1)
        _require_one_run(self._x, "AutoDAIS")
        self._num_steps = num_steps
        self._leapfrog_steps = leapfrog_steps
        self._D = self._registry.total_unconstrained_dim
        if self._D == 0:
            raise ValueError(
                "AutoDAIS: base registry has zero total "
                "unconstrained dimension; no continuous latents "
                "to anneal over"
            )
        # Learnable trajectory params.
        log_step = torch.log(torch.tensor(float(init_step_size)))
        self.log_step_size = nn.Parameter(log_step.expand(num_steps).clone())
        # Inverse temperatures are the normalized cumulative sum of
        # softplus increments, so they increase monotonically and the
        # last transition anneals at beta = 1 (the full target).
        # Initial schedule: linear in inverse temperature.
        init_betas = torch.linspace(init_temperature, 1.0, num_steps + 1)[:-1]
        increments = torch.zeros(num_steps)
        increments[0] = init_betas[0]
        for i in range(1, num_steps):
            increments[i] = init_betas[i] - init_betas[i - 1]
        # Store log(increments) so cumsum of softplus(raw) is positive
        # and monotone.
        raw_increments = torch.log(torch.expm1(increments.clamp(min=1e-4)))
        self.beta_increments_raw = nn.Parameter(raw_increments)

    # ------------------------------------------------------------------
    # Unconstrained-space densities
    # ------------------------------------------------------------------

    def _betas(self) -> torch.Tensor:
        """Return the inverse-temperature schedule, monotone in
        ``(0, 1]`` with the final entry fixed at 1.0."""
        increments = F.softplus(self.beta_increments_raw)
        cumulative = torch.cumsum(increments, dim=0)
        return cumulative / cumulative[-1]

    def _trace_side_with_log_det(
        self, z_flat: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Map a flat unconstrained point to batch-of-one trace-side
        values and the summed log-Jacobian of the bijectors.

        Parameters
        ----------
        z_flat : torch.Tensor
            Flat unconstrained point, shape ``(D,)``.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            The values `LatentRegistry.trace_values` produces with a
            batch of one, and the scalar total
            ``log |det dT/dz|`` over every site.
        """
        per_site_unc = self._registry.unflatten_unconstrained(z_flat)
        constrained, log_abs_dets = self._registry.to_constrained(per_site_unc)
        log_det_total = torch.zeros((), device=z_flat.device, dtype=z_flat.dtype)
        for site in self._registry.sites.values():
            log_det_total = log_det_total + log_abs_dets[site.name].sum()
        return self._registry.trace_values(constrained, batch_size=1), log_det_total

    def _bridge_log_density(
        self, z_flat: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalised log-density of the bridge at ``beta`` in the
        unconstrained space.

        Parameters
        ----------
        z_flat : torch.Tensor
            Flat unconstrained point, shape ``(D,)``.
        beta : torch.Tensor
            Scalar inverse temperature.

        Returns
        -------
        torch.Tensor
            :math:`(1 - \\beta) \\log q_0(z) + \\beta \\log p(z, y)`, both
            densities in the unconstrained space (each with the
            bijectors' log-Jacobian).
        """
        trace_side, log_det = self._trace_side_with_log_det(z_flat)
        log_q0 = self.base.log_prob(self._x[:1], trace_side).sum()
        log_p = self._model.log_joint(
            self._x, {**trace_side, **self._observations}
        ).sum()
        return (1.0 - beta) * log_q0 + beta * log_p + log_det

    def _grad_bridge(
        self, z_flat: torch.Tensor, beta: torch.Tensor, create_graph: bool
    ) -> torch.Tensor:
        """Gradient of the bridge log-density at ``z_flat``.

        The gradient keeps its dependence on ``z_flat`` and on the
        variational parameters when ``create_graph`` is true, so the
        leapfrog map is differentiated exactly.

        Parameters
        ----------
        z_flat : torch.Tensor
            Flat unconstrained point, shape ``(D,)``.
        beta : torch.Tensor
            Scalar inverse temperature.
        create_graph : bool
            Whether to record the gradient's own graph.

        Returns
        -------
        torch.Tensor
            Shape ``(D,)``.
        """
        with torch.enable_grad():
            point = z_flat if z_flat.requires_grad else z_flat.detach().requires_grad_()
            log_density = self._bridge_log_density(point, beta)
            (gradient,) = torch.autograd.grad(
                log_density, point, create_graph=create_graph
            )
        return gradient

    # ------------------------------------------------------------------
    # The annealed trajectory
    # ------------------------------------------------------------------

    def _flatten_constrained_sample(
        self, base_sample: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert the base guide's batch-of-one trace-side sample
        into a single flat unconstrained vector.

        Parameters
        ----------
        base_sample : dict[str, torch.Tensor]
            The base guide's `Guide.rsample` output for a batch of one.

        Returns
        -------
        torch.Tensor
            The flat unconstrained point, shape ``(D,)``.
        """
        unconstrained, _ = self._registry.to_unconstrained(
            self._registry.trace_row(base_sample)
        )
        return self._registry.flatten_unconstrained(unconstrained)

    def _trajectory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one annealed trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Program input, shape ``(batch, ...)``; the base guide is
            sampled and scored on its first row, since the trajectory
            is a single point.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The flat unconstrained endpoint :math:`u_K`, shape ``(D,)``,
            and the endpoint's extended-space log-density in the
            constrained space,
            :math:`\\log q_0(u_0) - \\log|J(u_K)| + \\sum_k [K(\\hat v_k)
            - K(v_{k-1})]`, a scalar.
        """
        x_one = x[:1]
        base_sample = self.base.rsample(x_one)
        z = self._flatten_constrained_sample(base_sample)
        _, log_det_start = self._trace_side_with_log_det(z)
        log_q0_start = self.base.log_prob(x_one, base_sample).sum() + log_det_start
        create_graph = torch.is_grad_enabled()
        betas = self._betas()
        kinetic_change = torch.zeros((), device=z.device, dtype=z.dtype)
        for t in range(self._num_steps):
            beta_t = betas[t]
            eps = torch.exp(self.log_step_size[t])
            momentum = torch.randn_like(z)
            kinetic_start = 0.5 * (momentum**2).sum()
            for _ in range(self._leapfrog_steps):
                momentum = momentum + 0.5 * eps * self._grad_bridge(
                    z, beta_t, create_graph
                )
                z = z + eps * momentum
                momentum = momentum + 0.5 * eps * self._grad_bridge(
                    z, beta_t, create_graph
                )
            kinetic_change = kinetic_change + 0.5 * (momentum**2).sum() - kinetic_start
        _, log_det_end = self._trace_side_with_log_det(z)
        return z, log_q0_start - log_det_end + kinetic_change

    def _materialise_constrained(
        self, z_flat: torch.Tensor, batch_size: int = 1
    ) -> dict[str, torch.Tensor]:
        """Map a flat unconstrained point to trace-side site values.

        Parameters
        ----------
        z_flat : torch.Tensor
            Flat unconstrained point, shape ``(D,)``.
        batch_size : int
            Batch axis the non-plate sites are expanded across.

        Returns
        -------
        dict[str, torch.Tensor]
            The values `LatentRegistry.trace_values` produces.
        """
        constrained, _ = self._registry.to_constrained(
            self._registry.unflatten_unconstrained(z_flat)
        )
        return self._registry.trace_values(constrained, batch_size)

    # ------------------------------------------------------------------
    # Guide contract
    # ------------------------------------------------------------------

    def observe(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Forward the step's data to the base guide.

        Parameters
        ----------
        observations : Mapping[str, torch.Tensor]
            Observed values and host data.
        """
        self.base.observe(observations)

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Endpoint of one annealed trajectory, as trace-side values.

        One trajectory is run per call and shared by every row of
        ``x``'s batch axis. The endpoint has no closed-form density, so
        objectives draw through `rsample_strata`, which returns the
        trajectory's density with the draw.

        Parameters
        ----------
        x : torch.Tensor
            Program input, shape ``(batch, ...)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Non-plate sites at ``(batch, *constrained_shape)`` and
            plate sites at ``constrained_shape``.
        """
        z_final, _ = self._trajectory(x)
        return self._materialise_constrained(z_final, int(x.shape[0]))

    def rsample_strata(self, x: torch.Tensor) -> tuple[GuideDraw, ...]:
        """One trajectory's endpoint with its extended-space density.

        The draw's ``log_density`` is the extended-space log-density of
        the trajectory that produced it, so for a model and input whose
        log-joint at the endpoint is :math:`\\log p(z_K, y)`, the value
        ``log p - log_density`` is the trajectory's DAIS log-weight
        :math:`\\log w`, and ``ELBO`` estimates
        :math:`\\mathbb{E}[\\log w] \\le \\log p(y)`. Each call runs a
        fresh trajectory, so several particles are independent.

        Parameters
        ----------
        x : torch.Tensor
            Program input, shape ``(batch, ...)``.

        Returns
        -------
        tuple[GuideDraw, ...]
            A single draw with log-weight zero; its density is
            expanded to ``(batch,)``.
        """
        z_final, log_density = self._trajectory(x)
        batch = int(x.shape[0])
        return (
            GuideDraw(
                sites=self._materialise_constrained(z_final, batch),
                log_density=log_density.expand(batch),
                log_weight=torch.zeros((), device=x.device, dtype=log_density.dtype),
            ),
        )

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Refuse: the DAIS endpoint has no density at arbitrary sites.

        The endpoint's marginal density integrates over every momentum
        draw of the trajectory and has no closed form. Its
        extended-space density is defined only for the trajectory that
        produced a draw and is returned with the draw by
        `rsample_strata`, which the objectives use.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        sites : dict[str, torch.Tensor]
            Site values to score.

        Returns
        -------
        torch.Tensor
            Never returns.

        Raises
        ------
        ValueError
            Always.
        """
        del x, sites
        raise ValueError(
            "AutoDAIS has no density at arbitrary sites: the endpoint's "
            "density is defined only on the extended space of the trajectory "
            "that produced it, which rsample_strata returns with the draw. "
            "Objectives that need the guide density at a fixed point "
            "(StickingTheLanding, DoublyReparameterized, ScoreFunction, RWS) "
            "are not available with AutoDAIS; use "
            "estimator=Reparameterized()."
        )

    @property
    def latent_names(self) -> list[str]:
        """Names of the latent sites the guide covers.

        Returns
        -------
        list[str]
            The registry's site names, in order.
        """
        return list(self._registry.names)


__all__ = ["AutoDAIS"]
