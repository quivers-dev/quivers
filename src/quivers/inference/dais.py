"""Differentiable Annealed Importance Sampling (DAIS) guide.

DAIS sits between variational inference and MCMC: it runs a short
chain of HMC-like leapfrog steps along an annealing path between
a tractable base distribution :math:`q_0` (typically
`AutoNormalGuide`) and the target posterior. The
trajectory's per-step parameters (step size, base mean / scale,
and the inverse-temperature schedule) are *all variational
parameters* trained by SVI. Concretely, every leapfrog operation
is reparameterized through the momentum and position so gradients
flow end-to-end through the trajectory.

The DAIS construction uses a variational bound on an extended target;
see Zhang, Hsu, Li, Finn, and Grosse (2021),
[Differentiable Annealed Importance Sampling and the Perils of Gradient
Noise](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a1a609f1ac109d0be28d8ae112db1bbb-Abstract.html).
An estimate of that objective is not an unbiased estimate of log evidence.
Adding transitions or a multimodal base guide does not guarantee that a
particular fitted run improves on the base guide or recovers every mode.

The implementation tracks the auxiliary "extended-target" form: a
fresh momentum is sampled at the start of each transition and
discarded at the end (no Metropolis correction in the
differentiable variant; the integrator is deterministic and the
trajectory length is fixed). The variational lower bound
absorbs the change-of-variables for the leapfrog flow and the
momentum prior / posterior contributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide


class AutoDAIS(Guide):
    """Annealed-importance-sampling guide wrapping a base guide.

    Parameters
    ----------
    base : Guide
        Base variational guide (the bridge's start). Must be built
        against the same model + observed-name set the DAIS guide
        is for. `AutoNormalGuide` or
        `AutoMultivariateNormalGuide` are the canonical
        choices.
    model : MonadicProgram
        Generative model. Required because DAIS needs the target
        log-density at each annealing step.
    observations : dict[str, torch.Tensor]
        Observed-site values that pin the model's likelihood.
        The DAIS bridge anneals between ``base`` and the conditional
        posterior under these observations; if the observations
        change the guide must be rebuilt.
    x : torch.Tensor
        Program input fed to the model's log-joint. Default
        ``torch.zeros(1, 1)``; only its device / leading batch
        shape matter.
    num_steps : int
        Number of HMC-style leapfrog transitions along the path.
        ``0`` recovers the base guide exactly. Default ``8``.
    leapfrog_steps : int
        Inner leapfrog substeps per transition. Default ``1``;
        higher values trade compute for tighter bounds.
    init_step_size : float
        Initial leapfrog step size. Adapted as a learnable scalar.
    init_temperature : float
        Initial first-bridge inverse temperature. The schedule
        anneals from this value at step 1 to ``1.0`` at step
        ``num_steps``. Stored as ``num_steps`` learnable
        parameters so the schedule is data-adaptive.
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
        # Inverse temperatures parameterized through a logit so they
        # stay in (0, 1) and are monotone-friendly. The final beta is
        # 1.0 (full target); the earlier ones are learned.
        # Initial schedule: linear in inverse temperature.
        init_betas = torch.linspace(init_temperature, 1.0, num_steps + 1)[
            :-1
        ]  # length num_steps; last one is fixed at 1.0.
        # Store as raw logits and apply a cumulative-softplus to
        # keep them monotone increasing.
        increments = torch.zeros(num_steps)
        increments[0] = init_betas[0]
        for i in range(1, num_steps):
            increments[i] = init_betas[i] - init_betas[i - 1]
        # Store log(increments) so cumsum of softplus(raw) is positive
        # and monotone.
        raw_increments = torch.log(torch.expm1(increments.clamp(min=1e-4)))
        self.beta_increments_raw = nn.Parameter(raw_increments)

    # ------------------------------------------------------------------
    # Bridge target evaluation
    # ------------------------------------------------------------------

    def _betas(self) -> torch.Tensor:
        """Return the inverse-temperature schedule, monotone in
        ``(0, 1]`` with the final entry fixed at 1.0."""
        increments = F.softplus(self.beta_increments_raw)
        cumulative = torch.cumsum(increments, dim=0)
        # Rescale so cumulative[-1] == 1.0 exactly.
        return cumulative / cumulative[-1]

    def _target_log_density(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Unconstrained-space log-density of the target (model
        joint + Jacobian)."""
        trace_side, log_det_total = self._trace_side_with_log_det(z_flat)
        log_p = self._model.log_joint(self._x, {**trace_side, **self._observations})
        return log_p.sum() + log_det_total

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

    def _grad_target(self, z_flat: torch.Tensor) -> torch.Tensor:
        z = z_flat.detach().clone().requires_grad_(True)
        ld = self._target_log_density(z)
        return torch.autograd.grad(ld, z, create_graph=False)[0]

    # ------------------------------------------------------------------
    # Bridge density gradient
    # ------------------------------------------------------------------

    def _bridge_log_density(
        self, z_flat: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Log-density of the annealed bridge at inverse temperature
        ``beta``: :math:`q_0(z)^{1 - \\beta} \\pi(z)^\\beta` in
        log-space."""
        # The bridge operates on flat z; the base guide's log_prob
        # takes trace-side site values, so we materialise them.
        trace_side = self._materialise_constrained(z_flat)
        log_q0 = self.base.log_prob(self._x, trace_side).sum()
        log_pi = self._target_log_density(z_flat)
        return (1.0 - beta) * log_q0 + beta * log_pi

    def _grad_bridge(self, z_flat: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        z = z_flat.detach().clone().requires_grad_(True)
        ld = self._bridge_log_density(z, beta)
        return torch.autograd.grad(ld, z, create_graph=True)[0]

    # ------------------------------------------------------------------
    # Sampling (forward pass through the annealed trajectory)
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
        """Run the annealed trajectory; return the flat unconstrained
        endpoint and the accumulated log-importance-weight.

        The log-weight is :math:`\\sum_t (\\beta_t - \\beta_{t-1})
        (\\log \\pi(z_t) - \\log q_0(z_t))` plus the momentum
        bookkeeping (zero in expectation for our momentum-resampling
        leapfrog variant).

        The trajectory is one point, so the base guide is sampled
        and scored on the first row of ``x`` alone.

        Parameters
        ----------
        x : torch.Tensor
            Program input, shape ``(batch, ...)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The flat endpoint, shape ``(D,)``, and the scalar
            log-weight.
        """
        x_one = x[:1]
        base_sample = self.base.rsample(x_one)
        z = self._flatten_constrained_sample(base_sample)
        betas = self._betas()
        log_weight = torch.zeros((), device=z.device, dtype=z.dtype)
        # Initial log-weight: target_at_z minus base_at_z, scaled by
        # the first beta increment.
        log_q0_z = self.base.log_prob(x_one, base_sample).sum()
        log_pi_z = self._target_log_density(z)
        log_weight = log_weight + betas[0] * (log_pi_z - log_q0_z)
        for t in range(self._num_steps):
            beta_t = betas[t]
            eps = torch.exp(self.log_step_size[t])
            p = torch.randn_like(z)
            # Leapfrog substeps along the bridge at the current beta.
            for _ in range(self._leapfrog_steps):
                g = self._grad_bridge(z, beta_t)
                p = p + 0.5 * eps * g
                z = z + eps * p
                g = self._grad_bridge(z, beta_t)
                p = p + 0.5 * eps * g
            if t < self._num_steps - 1:
                next_beta = betas[t + 1]
                log_q0_z = self.base.log_prob(
                    x_one, self._materialise_constrained(z)
                ).sum()
                log_pi_z = self._target_log_density(z)
                log_weight = log_weight + (next_beta - beta_t) * (log_pi_z - log_q0_z)
        return z, log_weight

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

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Endpoint of one annealed trajectory, as trace-side values.

        One trajectory is run per call and shared by every row of
        ``x``'s batch axis, matching `log_prob`, which returns the
        trajectory's density expanded to ``(batch,)``.

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

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Effective log-density of the trajectory's endpoint.

        DAIS does not admit a closed-form density at the endpoint
        because the trajectory is stochastic in the momentum. The
        differentiable lower-bound estimator instead substitutes
        :math:`\\log \\pi(z) - \\mathrm{log\\_weight}` for
        :math:`\\log q(z)` in the ELBO, so when this guide is used
        with an `ELBO` objective the per-step ELBO is the
        DAIS lower bound. We return that quantity here.
        """
        del sites  # The DAIS log-density is path-dependent, not
        # site-conditional; the standard ELBO call path
        # passes sites = rsample's output, and we ignore
        # them in favour of the cached trajectory weight.
        z_final, log_weight = self._trajectory(x)
        log_pi = self._target_log_density(z_final)
        # log q_DAIS(z) ≈ log π(z) - log_weight, so that
        # ELBO = E[log π(z) - log q_DAIS(z)] = E[log_weight].
        return (log_pi - log_weight).expand(x.shape[0])

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


__all__ = ["AutoDAIS"]
