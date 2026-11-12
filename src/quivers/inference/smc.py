"""Adaptive likelihood-tempered sequential Monte Carlo.

`SMC` moves a population of particles from the prior to the posterior
through the tempered targets

.. math::

    \\pi_\\beta(z) \\propto p(z)\\, L(z)^\\beta, \\qquad
    0 = \\beta_0 < \\beta_1 < \\dots < \\beta_T = 1,

where :math:`p(z)` is the density of the model's latent draws (the
distribution ancestral sampling from the model produces) and
:math:`L(z)` everything else the model scores: its observations and its
score steps (Del Moral, Doucet, and Jasra 2006,
`doi:10.1111/j.1467-9868.2006.00553.x
<https://doi.org/10.1111/j.1467-9868.2006.00553.x>`_). Each step

1. chooses the next temperature by bisection, as the largest
   :math:`\\beta' \\le 1` at which the conditional effective sample size
   of the incremental weights :math:`L(z_i)^{\\beta' - \\beta}` is at
   least ``target_ess`` times the number of particles (Jasra et al. 2011,
   `doi:10.1111/j.1467-9469.2010.00723.x
   <https://doi.org/10.1111/j.1467-9469.2010.00723.x>`_; Zhou,
   Johansen, and Aston 2016,
   `doi:10.1080/10618600.2015.1060885
   <https://doi.org/10.1080/10618600.2015.1060885>`_);
2. reweights by those increments, adding the log of their weighted mean
   to the log-evidence estimate;
3. resamples by the chosen `ResamplingScheme`;
4. rejuvenates every particle by ``num_mcmc_steps`` transitions of the
   kernel at the new temperature, with the kernel's parameters fixed for
   the whole step: a pilot run on a particle, discarded, tunes them
   after the kernel has seen the population (`MCMCKernel.tune`), and
   every particle then moves under the frozen kernel, so each move
   leaves :math:`\\pi_{\\beta'}` invariant.

The product of the mean incremental weights is an unbiased estimate of
the evidence :math:`p(y) = \\int p(z) L(z)\\, dz`; its logarithm is
`SMCResult.log_evidence`.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram
from quivers.effects.joint_plan import JointPlan
from quivers.effects.trace_types import Trace
from quivers.inference.mcmc.driver import (
    MCMC,
    _check_gibbs_sites,
    _gibbs_draw,
)
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    _check_observed_support,
    _rejects_position,
    _with_generator,
)
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import trace


ResamplingScheme = Literal["multinomial", "systematic", "stratified", "residual"]
"""How `SMC` resamples its particles; each draws every particle's
number of copies with expectation ``N`` times its normalized weight
(Douc and Cappé 2005, `doi:10.1109/ISPA.2005.195385
<https://doi.org/10.1109/ISPA.2005.195385>`_):

* ``"multinomial"``: ``N`` independent draws from the weights;
* ``"stratified"``: one uniform draw in each of ``N`` equal strata of
  the cumulative weights;
* ``"systematic"``: one uniform offset shared by the ``N`` strata;
* ``"residual"``: ``floor(N w_i)`` copies of each particle, and the
  remaining copies by multinomial draws from the residual weights."""


def resample(
    weights: torch.Tensor,
    scheme: ResamplingScheme,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ancestor indices for a weighted population.

    Parameters
    ----------
    weights : torch.Tensor
        Non-negative weights, shape ``(N,)``, not all zero; they need not
        be normalized.
    scheme : ResamplingScheme
        The resampling scheme.
    generator : torch.Generator, optional
        The generator every uniform draw comes from.

    Returns
    -------
    torch.Tensor
        ``N`` ancestor indices, ``int64``, in increasing order for the
        stratified, systematic, and residual schemes.

    Raises
    ------
    ValueError
        If the weights are not a non-negative vector with a positive
        finite sum, or the scheme is unknown.
    """
    if weights.dim() != 1 or weights.numel() == 0:
        raise ValueError(
            f"resample: weights must be a non-empty vector, got {tuple(weights.shape)}"
        )
    w = weights.to(torch.float64)
    total = float(w.sum())
    if bool((w < 0).any()) or not math.isfinite(total) or total <= 0.0:
        raise ValueError("resample: weights must be non-negative with a positive sum")
    w = w / total
    n = w.numel()
    if scheme == "multinomial":
        return torch.multinomial(w, n, replacement=True, generator=generator)
    if scheme == "residual":
        copies = torch.floor(n * w).to(torch.int64)
        remaining = n - int(copies.sum())
        base = torch.repeat_interleave(torch.arange(n), copies)
        if remaining == 0:
            return base
        residual = n * w - copies.to(torch.float64)
        extra = torch.multinomial(
            residual / residual.sum(), remaining, replacement=True, generator=generator
        )
        return torch.sort(torch.cat([base, extra])).values
    if scheme == "stratified":
        offsets = torch.rand(n, generator=generator, dtype=torch.float64)
    elif scheme == "systematic":
        offsets = torch.rand((), generator=generator, dtype=torch.float64).expand(n)
    else:
        raise ValueError(f"resample: unknown scheme {scheme!r}")
    points = (torch.arange(n, dtype=torch.float64) + offsets) / n
    cumulative = torch.cumsum(w, 0)
    return torch.searchsorted(cumulative, points, right=True).clamp_max(n - 1)


def _conditional_ess(log_weights: torch.Tensor, increments: torch.Tensor) -> float:
    """The conditional effective sample size of incremental weights,
    as a fraction of the population.

    Parameters
    ----------
    log_weights : torch.Tensor
        Normalized log-weights of the current population, ``(N,)``.
    increments : torch.Tensor
        Log incremental weights, ``(N,)``.

    Returns
    -------
    float
        ``(sum_i W_i w_i)^2 / sum_i W_i w_i^2``, between ``0`` and ``1``;
        ``0`` when every increment is ``-inf``.
    """
    combined = log_weights + increments
    if not bool(torch.isfinite(combined).any()):
        return 0.0
    numerator = 2.0 * torch.logsumexp(combined, 0)
    denominator = torch.logsumexp(log_weights + 2.0 * increments, 0)
    return float(torch.exp(numerator - denominator))


class TemperedPotential(PotentialFn):
    """The potential of a tempered posterior :math:`p(z)\\, L(z)^\\beta`.

    :math:`\\log p(z)` is the log-density of the model's latent draws,
    the sites of the registry and the sites the potential is conditioned
    on, together with the bijectors' log-Jacobian; :math:`\\log L(z)` is
    the rest of the model's log joint, its observations and score steps.
    At temperature one the potential is the posterior's, at zero the
    prior's.

    Parameters
    ----------
    model : MonadicProgram
        The model.
    registry : LatentRegistry
        The latent registry.
    x : torch.Tensor
        The one-row input.
    observations : dict[str, torch.Tensor]
        The data.
    temperature : float
        The inverse temperature :math:`\\beta \\in [0, 1]`. Default
        ``1.0``.

    Raises
    ------
    ValueError
        If ``temperature`` is outside ``[0, 1]``, or the input or data
        are refused as by `PotentialFn`.
    """

    def __init__(
        self,
        model: MonadicProgram,
        registry: LatentRegistry,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model, registry, x, observations)
        self._temperature = _check_temperature(temperature)

    @property
    def temperature(self) -> float:
        """The inverse temperature :math:`\\beta`."""
        return self._temperature

    def at_temperature(self, temperature: float) -> "TemperedPotential":
        """The same potential at another temperature.

        Parameters
        ----------
        temperature : float
            The inverse temperature, in ``[0, 1]``.

        Returns
        -------
        TemperedPotential
            A copy sharing the model, data, conditioning, and evaluation
            counts.

        Raises
        ------
        ValueError
            If ``temperature`` is outside ``[0, 1]``.
        """
        tempered = copy.copy(self)
        tempered._temperature = _check_temperature(temperature)
        return tempered

    def log_prior_and_likelihood(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two parts of the log joint at a position.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            :math:`\\log p(z)` (with the log-Jacobian) and
            :math:`\\log L(z)`, scalars; both ``-inf`` for a position
            outside the model's support, and the likelihood ``-inf``
            where the data are impossible.

        Raises
        ------
        Exception
            Any error the program raises other than one marking the
            position as outside its support.
        """
        self._counts[0] += 1
        trace_side, log_det_total = self._trace_side(z)
        minus_inf = torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        try:
            densities, log_joint = self._densities({**trace_side, **self._observations})
        except (ValueError, torch.linalg.LinAlgError) as error:
            if not _rejects_position(error):
                raise
            return minus_inf, minus_inf
        latent = set(self._registry.names) | set(self._conditioned)
        log_prior = log_det_total
        for label, density in densities.items():
            if any(name in latent for name in label.split(",")):
                log_prior = log_prior + density.sum()
        log_likelihood = log_joint.sum() + log_det_total - log_prior
        if not torch.isfinite(log_prior):
            return minus_inf, minus_inf
        if torch.isnan(log_likelihood):
            return log_prior, minus_inf
        return log_prior, log_likelihood

    def _densities(
        self, values: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Each binding's density and the joint, as a trace records them.

        Values of a layout the potential has a prepared joint for are
        scored by it; any other values by tracing the program.

        Parameters
        ----------
        values : dict[str, torch.Tensor]
            The latent values and the data.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            The density of each site and binding by label, and the joint.
        """
        plan = self._plans.plan(values)
        if isinstance(plan, JointPlan):
            return plan.evaluate(self._x, values)
        recorded = trace(self._model, self._x, observations=values)
        assert recorded.log_joint is not None
        densities = {label: site.log_prob for label, site in recorded.sites.items()}
        return densities, recorded.log_joint

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        """The tempered log-density :math:`\\log p(z) + \\beta \\log L(z)`.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        torch.Tensor
            The scalar log-density, ``-inf`` outside the support.
        """
        log_prior, log_likelihood = self.log_prior_and_likelihood(z)
        if self._temperature == 0.0:
            return log_prior
        result = log_prior + self._temperature * log_likelihood
        if not torch.isfinite(result):
            return torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        return result


def _check_temperature(temperature: float) -> float:
    """Refuse an inverse temperature outside ``[0, 1]``.

    Parameters
    ----------
    temperature : float
        The inverse temperature.

    Returns
    -------
    float
        The temperature.

    Raises
    ------
    ValueError
        If it is outside ``[0, 1]``.
    """
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(
            f"TemperedPotential: temperature must be in [0, 1], got {temperature}"
        )
    return float(temperature)


class SMCResult(ValueModel):
    """The output of an `SMC` run.

    Attributes
    ----------
    samples : dict[str, torch.Tensor]
        The final, equally weighted particles on the constrained support,
        Gibbs sites included: each site at ``(num_particles,
        *constrained_shape)``.
    log_evidence : float
        The log of the evidence estimate
        :math:`\\prod_t \\sum_i W_i^{(t)} L(z_i)^{\\beta_t - \\beta_{t-1}}`,
        whose exponent is unbiased for :math:`p(y)`.
    temperatures : torch.Tensor
        The ladder :math:`\\beta_0 = 0 < \\dots < \\beta_T = 1`, shape
        ``(T + 1,)``.
    ess : torch.Tensor
        The effective sample size, as a fraction of the population, of
        the weights at each temperature before resampling, shape ``(T,)``.
    acceptance : torch.Tensor
        The mean acceptance of the rejuvenation moves at each
        temperature, shape ``(T,)``.
    log_evidence_increments : torch.Tensor
        The log mean incremental weight at each temperature, shape
        ``(T,)``; their sum is ``log_evidence``.
    log_likelihoods : torch.Tensor
        :math:`\\log L` at each final particle, shape ``(num_particles,)``.
    num_evaluations : int
        Potential evaluations over the run.
    """

    samples: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_evidence: float
    temperatures: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    ess: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    acceptance: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_evidence_increments: torch.Tensor = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_likelihoods: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    num_evaluations: int

    @property
    def num_particles(self) -> int:
        """The population size."""
        return int(self.log_likelihoods.shape[0])

    @property
    def num_temperatures(self) -> int:
        """The number of tempering steps ``T``."""
        return int(self.ess.shape[0])


class SMC:
    """Adaptive likelihood-tempered sequential Monte Carlo sampler.

    Parameters
    ----------
    kernel : MCMCKernel
        The rejuvenation kernel: any kernel, composites and Gibbs blocks
        included. Its configuration is shared by every particle; it is
        re-tuned at every temperature.
    num_particles : int
        Population size. Default ``1024``.
    num_mcmc_steps : int
        Rejuvenation transitions per particle per temperature. Default
        ``5``.
    target_ess : float
        The conditional effective sample size, as a fraction of the
        population, each temperature step is chosen to keep. Default
        ``0.5``.
    resampling : ResamplingScheme
        The resampling scheme. Default ``"systematic"``.
    num_pilot_steps : int
        Adaptation transitions of the discarded pilot run that tunes the
        kernel at each temperature; ``0`` uses the kernel as tuned to the
        population alone. Default ``50``.
    seed : int, optional
        Seed of the run's generator. A run with the same seed is
        reproducible and leaves Torch's global RNG alone.
    generator : torch.Generator, optional
        Caller-owned CPU generator from which each run draws its seed;
        mutually exclusive with ``seed``.

    Raises
    ------
    ValueError
        If a count is out of range, ``target_ess`` is outside ``(0, 1)``,
        ``resampling`` is unknown, or both ``seed`` and ``generator`` are
        given.
    """

    def __init__(
        self,
        kernel: MCMCKernel,
        num_particles: int = 1024,
        num_mcmc_steps: int = 5,
        target_ess: float = 0.5,
        resampling: ResamplingScheme = "systematic",
        num_pilot_steps: int = 50,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if num_particles < 2:
            raise ValueError(f"SMC: num_particles must be >= 2, got {num_particles}")
        if num_mcmc_steps < 1:
            raise ValueError(f"SMC: num_mcmc_steps must be >= 1, got {num_mcmc_steps}")
        if num_pilot_steps < 0:
            raise ValueError(
                f"SMC: num_pilot_steps must be >= 0, got {num_pilot_steps}"
            )
        if not 0.0 < target_ess < 1.0:
            raise ValueError(f"SMC: target_ess must be in (0, 1), got {target_ess}")
        if resampling not in ("multinomial", "systematic", "stratified", "residual"):
            raise ValueError(f"SMC: unknown resampling scheme {resampling!r}")
        if seed is not None and generator is not None:
            raise ValueError("SMC: seed and generator are mutually exclusive")
        self.kernel = kernel
        self.num_particles = num_particles
        self.num_mcmc_steps = num_mcmc_steps
        self.target_ess = target_ess
        self.resampling: ResamplingScheme = resampling
        self.num_pilot_steps = num_pilot_steps
        self.seed = seed
        self.generator = generator

    def _run_generator(self) -> torch.Generator:
        """The generator of one run.

        Returns
        -------
        torch.Generator
            Derived from ``seed``, from a draw of ``generator``, or from
            Torch's initial seed, without touching the global RNG.
        """
        if self.generator is not None:
            base = int(
                torch.randint(
                    0, 2**63 - 1, (), dtype=torch.int64, generator=self.generator
                ).item()
            )
        elif self.seed is not None:
            base = self.seed
        else:
            base = torch.initial_seed()
        return MCMC._chain_generator(base, 0)

    @staticmethod
    def _log_evidence_increment(
        log_weights: torch.Tensor, increments: torch.Tensor
    ) -> torch.Tensor:
        """The log of the weighted mean incremental weight.

        Parameters
        ----------
        log_weights : torch.Tensor
            Normalized log-weights, ``(N,)``.
        increments : torch.Tensor
            Log incremental weights, ``(N,)``.

        Returns
        -------
        torch.Tensor
            :math:`\\log \\sum_i W_i w_i`.
        """
        return torch.logsumexp(log_weights + increments, 0)

    def _next_temperature(
        self,
        temperature: float,
        log_weights: torch.Tensor,
        log_likelihoods: torch.Tensor,
    ) -> float:
        """Choose the next temperature by bisection on the conditional ESS.

        Parameters
        ----------
        temperature : float
            The current inverse temperature.
        log_weights : torch.Tensor
            Normalized log-weights, ``(N,)``.
        log_likelihoods : torch.Tensor
            Each particle's :math:`\\log L`, ``(N,)``.

        Returns
        -------
        float
            One if the step to one keeps the target; otherwise the
            largest temperature, to bisection precision, whose
            conditional ESS is at least ``target_ess``.

        Raises
        ------
        RuntimeError
            If no step, however small, keeps the target, because too
            few particles have a finite likelihood.
        """

        def cess(delta: float) -> float:
            return _conditional_ess(log_weights, _increments(log_likelihoods, delta))

        if cess(1.0 - temperature) >= self.target_ess:
            return 1.0
        low, high = 0.0, 1.0 - temperature
        for _ in range(64):
            middle = 0.5 * (low + high)
            if cess(middle) >= self.target_ess:
                low = middle
            else:
                high = middle
        if low <= 0.0 or temperature + low <= temperature:
            raise RuntimeError(
                "SMC: no temperature step keeps the target ESS; fewer than "
                f"{self.target_ess:.0%} of the particles have a finite likelihood"
            )
        return temperature + low

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> SMCResult:
        """Run the sampler from the prior to the posterior.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        x : torch.Tensor
            A one-row input; as for `MCMC.run`, an input of several rows
            is several independent runs, and the sampler targets one.
        observations : dict[str, torch.Tensor]
            Observed-site values and host data.

        Returns
        -------
        SMCResult
            The final particles, the evidence estimate, and the ladder.

        Raises
        ------
        ValueError
            If ``x`` has more than one row, the data are outside their
            support, or the kernel's Gibbs sites do not fit the model.
        RuntimeError
            If no temperature step keeps the target ESS.
        """
        _require_one_run(x, "SMC.run")
        _check_observed_support(model, observations, "SMC.run")
        with torch.random.fork_rng(devices=[]):
            recorded = trace(model, x, observations=observations)
        gibbs_sites = self.kernel.gibbs_sites
        gibbs_morphisms = _check_gibbs_sites(recorded, gibbs_sites)
        registry = LatentRegistry.from_trace(model, recorded, exclude=gibbs_sites)
        potential = TemperedPotential(model, registry, x, observations, 0.0)
        generator = self._run_generator()
        self.kernel._set_generator(generator)

        positions, gibbs = self._prior_particles(
            model, x, observations, registry, gibbs_sites, generator
        )
        log_likelihoods = self._log_likelihoods(potential, positions, gibbs)
        n = self.num_particles
        log_weights = torch.full((n,), -math.log(n), dtype=torch.float64)
        temperature = 0.0
        temperatures = [0.0]
        ess: list[float] = []
        acceptance: list[float] = []
        increments_log: list[float] = []
        while temperature < 1.0:
            following = self._next_temperature(
                temperature, log_weights, log_likelihoods
            )
            increments = _increments(log_likelihoods, following - temperature)
            increment = self._log_evidence_increment(log_weights, increments)
            increments_log.append(float(increment))
            log_weights = log_weights + increments
            log_weights = log_weights - torch.logsumexp(log_weights, 0)
            ess.append(float(torch.exp(-torch.logsumexp(2.0 * log_weights, 0))) / n)
            ancestors = resample(torch.exp(log_weights), self.resampling, generator)
            positions = positions[ancestors]
            gibbs = [gibbs[int(i)] for i in ancestors]
            log_weights = torch.full((n,), -math.log(n), dtype=torch.float64)
            temperature = following
            temperatures.append(temperature)
            tempered = potential.at_temperature(temperature)
            positions, gibbs, accepted = self._rejuvenate(
                model, x, observations, registry, tempered, positions, gibbs, generator
            )
            acceptance.append(accepted)
            log_likelihoods = self._log_likelihoods(potential, positions, gibbs)

        samples = self._samples(registry, positions, gibbs, gibbs_morphisms)
        return SMCResult(
            samples=samples,
            log_evidence=float(sum(increments_log)),
            temperatures=torch.tensor(temperatures, dtype=torch.float64),
            ess=torch.tensor(ess, dtype=torch.float64),
            acceptance=torch.tensor(acceptance, dtype=torch.float64),
            log_evidence_increments=torch.tensor(increments_log, dtype=torch.float64),
            log_likelihoods=log_likelihoods,
            num_evaluations=potential.num_evaluations,
        )

    def _prior_particles(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        registry: LatentRegistry,
        gibbs_sites: frozenset[str],
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Draw the initial population exactly from the prior.

        Each particle is one ancestral run of the model at the data, with
        its latent draws mapped into the registry's flat vector.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The input.
        observations : dict[str, torch.Tensor]
            The data.
        registry : LatentRegistry
            The registry.
        gibbs_sites : frozenset[str]
            Sites held outside the flat vector.
        generator : torch.Generator
            The run's generator.

        Returns
        -------
        tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
            The positions, ``(N, D)``, and each particle's Gibbs values.
        """
        positions: list[torch.Tensor] = []
        gibbs: list[dict[str, torch.Tensor]] = []
        for _ in range(self.num_particles):
            run = _with_generator(
                generator, lambda: trace(model, x, observations=observations)
            )
            draw = _registry_draw(registry, run)
            unconstrained, _ = registry.to_unconstrained(draw)
            positions.append(registry.flatten_unconstrained(unconstrained))
            gibbs.append({name: run.sites[name].value.detach() for name in gibbs_sites})
        return torch.stack(positions), gibbs

    @staticmethod
    def _log_likelihoods(
        potential: TemperedPotential,
        positions: torch.Tensor,
        gibbs: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Each particle's :math:`\\log L`.

        Parameters
        ----------
        potential : TemperedPotential
            The potential.
        positions : torch.Tensor
            The positions, ``(N, D)``.
        gibbs : list[dict[str, torch.Tensor]]
            Each particle's Gibbs values.

        Returns
        -------
        torch.Tensor
            The log-likelihoods, ``(N,)``, ``float64``.
        """
        out = torch.empty(positions.shape[0], dtype=torch.float64)
        with torch.no_grad():
            for i in range(positions.shape[0]):
                conditioned = potential.condition(gibbs[i]) if gibbs[i] else potential
                assert isinstance(conditioned, TemperedPotential)
                _, log_likelihood = conditioned.log_prior_and_likelihood(positions[i])
                out[i] = float(log_likelihood)
        return out

    def _rejuvenate(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        registry: LatentRegistry,
        potential: TemperedPotential,
        positions: torch.Tensor,
        gibbs: list[dict[str, torch.Tensor]],
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], float]:
        """Tune the kernel at a temperature and move every particle.

        A pilot chain from a randomly chosen particle runs
        ``num_pilot_steps`` adapting transitions after the kernel has
        been tuned to the population (`MCMCKernel.tune`); its moves are
        discarded and its frozen state is installed for every particle,
        which then makes ``num_mcmc_steps`` transitions.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The input.
        observations : dict[str, torch.Tensor]
            The data.
        registry : LatentRegistry
            The registry.
        potential : TemperedPotential
            The potential at the new temperature.
        positions : torch.Tensor
            The resampled positions, ``(N, D)``.
        gibbs : list[dict[str, torch.Tensor]]
            Each particle's Gibbs values.
        generator : torch.Generator
            The run's generator.

        Returns
        -------
        tuple[torch.Tensor, list[dict[str, torch.Tensor]], float]
            The moved positions, the moved Gibbs values, and the mean
            acceptance of the moves.
        """
        kernel = self.kernel
        pilot = int(torch.randint(0, positions.shape[0], (), generator=generator))
        state = kernel.init(registry, model, x, observations, positions[pilot])
        state.gibbs_values = dict(gibbs[pilot])
        kernel.tune(positions)
        if self.num_pilot_steps > 0:
            kernel.start_adaptation(self.num_pilot_steps)
            for _ in range(self.num_pilot_steps):
                state = kernel.step(state, _conditioned(potential, state))
            kernel.stop_adaptation()
        tuned = kernel.state_dict()

        moved = torch.empty_like(positions)
        moved_gibbs: list[dict[str, torch.Tensor]] = []
        accepted = 0.0
        for i in range(positions.shape[0]):
            state = kernel.init(registry, model, x, observations, positions[i])
            kernel.load_state_dict(tuned)
            state.gibbs_values = dict(gibbs[i])
            for _ in range(self.num_mcmc_steps):
                state = kernel.step(state, _conditioned(potential, state))
                accepted += state.stats.get("accepted", 0.0)
            moved[i] = state.position
            moved_gibbs.append(state.gibbs_values)
        return moved, moved_gibbs, accepted / (positions.shape[0] * self.num_mcmc_steps)

    @staticmethod
    def _samples(
        registry: LatentRegistry,
        positions: torch.Tensor,
        gibbs: list[dict[str, torch.Tensor]],
        gibbs_morphisms: dict[str, ContinuousMorphism],
    ) -> dict[str, torch.Tensor]:
        """The final particles on the constrained support.

        Parameters
        ----------
        registry : LatentRegistry
            The registry.
        positions : torch.Tensor
            The positions, ``(N, D)``.
        gibbs : list[dict[str, torch.Tensor]]
            Each particle's Gibbs values.
        gibbs_morphisms : dict[str, ContinuousMorphism]
            Each Gibbs site's distribution.

        Returns
        -------
        dict[str, torch.Tensor]
            Each site at ``(N, *constrained_shape)``.
        """
        constrained, _ = registry.to_constrained(
            registry.unflatten_unconstrained(positions)
        )
        samples = dict(constrained)
        for name, morphism in gibbs_morphisms.items():
            samples[name] = torch.stack(
                [_gibbs_draw(values[name], morphism) for values in gibbs]
            )
        return samples


def _registry_draw(registry: LatentRegistry, run: Trace) -> dict[str, torch.Tensor]:
    """One draw per registry site, at its constrained shape, from a run.

    A site a destructuring draw binds takes its share of the draw's
    value, split evenly along the last axis.

    Parameters
    ----------
    registry : LatentRegistry
        The registry.
    run : Trace
        A run of the model.

    Returns
    -------
    dict[str, torch.Tensor]
        Each registry site's value at `LatentSite.constrained_shape`.

    Raises
    ------
    ValueError
        If a registry site is missing from the run.
    """
    out: dict[str, torch.Tensor] = {}
    for name, site in registry.sites.items():
        record = run.sites.get(name)
        value: torch.Tensor | None = None
        if record is not None:
            value = record.value.detach()
        else:
            for label, candidate in run.sites.items():
                names = label.split(",")
                if name in names:
                    value = candidate.value.detach().chunk(len(names), dim=-1)[
                        names.index(name)
                    ]
                    break
        if value is None:
            raise ValueError(f"SMC: the run did not reach the latent site {name!r}")
        out[name] = value.reshape(site.constrained_shape)
    return out


def _increments(log_likelihoods: torch.Tensor, delta: float) -> torch.Tensor:
    """Log incremental weights ``delta * log L``, with ``-inf`` kept.

    Parameters
    ----------
    log_likelihoods : torch.Tensor
        Each particle's :math:`\\log L`, ``(N,)``.
    delta : float
        The temperature step.

    Returns
    -------
    torch.Tensor
        The log increments; a particle with an impossible likelihood
        gets ``-inf`` for any positive step.
    """
    out = delta * log_likelihoods
    return torch.where(torch.isneginf(log_likelihoods), float("-inf"), out)


def _conditioned(potential: TemperedPotential, state: KernelState) -> PotentialFn:
    """The potential conditioned on a state's Gibbs values.

    Parameters
    ----------
    potential : TemperedPotential
        The potential.
    state : KernelState
        The state.

    Returns
    -------
    PotentialFn
        The conditioned potential, or ``potential`` without Gibbs sites.
    """
    return potential.condition(state.gibbs_values) if state.gibbs_values else potential


__all__ = [
    "SMC",
    "SMCResult",
    "TemperedPotential",
    "ResamplingScheme",
    "resample",
]
