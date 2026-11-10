"""Sub-Giry measures and their primitive transformations.

A `Measure` is an unnormalized positive measure on a Borel space. Probability
measures have total mass one; restrictions and likelihood scores may produce
sub-probability measures. The runtime tracks total mass as a log normalizer and
normalizes at sample or observation boundaries.

The module provides point masses, restriction, pushforward through a bijector,
finite mixtures, event-dimension declarations with `Independent`, and
normalization of positive-mass sub-measures.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution

from quivers.continuous.bijectors import Bijector


_NEG_INF = float("-inf")


class Measure(Distribution, ABC):
    """Abstract base for sub-Giry monad values.

    The runtime tracks each `Measure` as a sub-distribution: it
    has a `log_prob` returning the unnormalised log-density and a
    `log_normalizer()` returning the log of the total mass. For
    probability measures (mass one), `log_normalizer()` returns
    zero by definition. For sub-measures from `Restrict` or
    `Mixture` with unnormalised weights, `log_normalizer()` carries
    the residual mass symbolically until the next
    [`Normalize`][quivers.continuous.measure.Normalize] or
    sample / observe boundary.

    `log_prob_normalized(value)` is a derived helper returning
    `log_prob(value) - log_normalizer()` so consumers that want the
    probability-measure log-density can ask for it explicitly.
    """

    @abstractmethod
    def log_normalizer(self) -> Tensor:
        """Log of the total mass of this measure. For probability
        measures, zero. For sub-measures, the symbolic normaliser
        the lazy algebra carries through composition.
        """

    def log_prob_normalized(self, value: Tensor) -> Tensor:
        """Log-density of the renormalised probability measure."""
        return self.log_prob(value) - self.log_normalizer()


class PointMass(Measure):
    """Dirac measure at a deterministic value.

    The unit $\\eta$ of the Giry monad: `PointMass(x)` is the
    measure that places mass one on the singleton $\\{x\\}$.
    `log_prob(y)` is zero when `y == value` and $-\\infty$
    otherwise.

    In a [`Mixture`][quivers.continuous.measure.Mixture] alongside
    a non-degenerate component, `PointMass` is the structural-zero
    spike that yields zero-inflation and hurdle factorisations.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        value: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.value = torch.as_tensor(value)
        super().__init__(
            batch_shape=self.value.shape,
            event_shape=torch.Size(()),
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        return _constraints.real

    @property
    def mean(self) -> Tensor:
        return self.value

    @property
    def variance(self) -> Tensor:
        return torch.zeros_like(self.value)

    def log_normalizer(self) -> Tensor:
        return torch.zeros(self.batch_shape) if self.batch_shape else torch.tensor(0.0)

    def log_prob(self, value: Tensor) -> Tensor:
        value_b = torch.as_tensor(value)
        match = value_b == self.value
        if match.dtype != torch.bool:
            match = match.bool()
        out = torch.full(
            torch.broadcast_shapes(value_b.shape, self.value.shape),
            _NEG_INF,
            dtype=torch.get_default_dtype(),
        )
        out = torch.where(match, torch.zeros_like(out), out)
        return out

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = torch.Size(sample_shape) + self.batch_shape
        return self.value.expand(shape) if shape else self.value


class Restrict(Measure):
    """Restriction of a base distribution to a sub-interval.

    Restricts `base` to the interval `[low, high]` (with `low` or
    `high` set to $\\pm\\infty$ for one-sided restrictions). The
    log-density is `base.log_prob(value)` inside the interval and
    $-\\infty$ outside; the symbolic log-normaliser is the log of
    the base measure's mass on `[low, high]`.

    The compiler does not renormalise eagerly; the sub-measure
    propagates through `Mixture`, `Pushforward`, and other
    operators until a [`Normalize`][quivers.continuous.measure.Normalize]
    or sample / observe boundary collapses it. This is the
    discipline that makes the algebra close, per
    [Di Lavore-Roman-Sobocinski 2025](https://arxiv.org/abs/2502.03477).

    Numerically, the log-normaliser is computed via the base's
    `cdf` (and the `1 - cdf` complement when only `low` is finite)
    with the `logsubexp` stable variant. Bases without a closed-form
    CDF raise at construction time.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        low: Tensor | float | None = None,
        high: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if low is None and high is None:
            raise ValueError(
                "Restrict: at least one of `low` or `high` must be "
                "supplied; an unrestricted base is just the base "
                "distribution itself."
            )
        self.base = base
        self.low = (
            torch.as_tensor(low, dtype=torch.get_default_dtype())
            if low is not None
            else None
        )
        self.high = (
            torch.as_tensor(high, dtype=torch.get_default_dtype())
            if high is not None
            else None
        )
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        if self.low is not None and self.high is not None:
            return _constraints.interval(self.low, self.high)
        if self.low is not None:
            return _constraints.greater_than(self.low)
        if self.high is not None:
            return _constraints.less_than(self.high)
        return self.base.support

    def log_normalizer(self) -> Tensor:
        # Try the base's closed-form CDF; fall back to a discrete-base
        # pmf-sum for bases that PyTorch ships without `cdf` (Poisson,
        # Geometric, NegativeBinomial).
        #
        # Continuous interpretation: `low` and `high` are inclusive
        # endpoints of an interval; the mass on `[low, high]` is
        # `cdf(high) - cdf(low)`, with one-sided cases using the
        # complement of the missing endpoint.
        #
        # The discrete fallback uses an INTEGER interpretation where
        # the support `{low, low+1, ..., high}` includes both
        # endpoints; the survival uses `1 - cdf(low - 1)` so the mass
        # at `low` is included.
        try:
            if self.low is None:
                cdf_high = self.base.cdf(self.high)
                return torch.log(cdf_high.clamp_min(torch.finfo(cdf_high.dtype).tiny))
            if self.high is None:
                cdf_low = self.base.cdf(self.low)
                surv = 1.0 - cdf_low
                return torch.log(surv.clamp_min(torch.finfo(surv.dtype).tiny))
            cdf_high = self.base.cdf(self.high)
            cdf_low = self.base.cdf(self.low)
            return torch.log(
                (cdf_high - cdf_low).clamp_min(
                    torch.finfo(cdf_high.dtype).tiny,
                )
            )
        except NotImplementedError:
            symmetric = _symmetric_fold_log_normalizer(
                self.base,
                self.low,
                self.high,
            )
            if symmetric is not None:
                return symmetric
            return _discrete_restriction_log_normalizer(
                self.base,
                self.low,
                self.high,
            )

    def log_prob(self, value: Tensor) -> Tensor:
        value_b = torch.as_tensor(value)
        base_lp = self.base.log_prob(value_b)
        out = base_lp.clone()
        if self.low is not None:
            below = value_b < self.low
            out = torch.where(below, torch.full_like(out, _NEG_INF), out)
        if self.high is not None:
            above = value_b > self.high
            out = torch.where(above, torch.full_like(out, _NEG_INF), out)
        return out

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        draws = self.base.sample(sample_shape)
        for _ in range(200):
            ok = torch.ones_like(draws, dtype=torch.bool)
            if self.low is not None:
                ok &= draws >= self.low
            if self.high is not None:
                ok &= draws <= self.high
            if ok.all():
                return draws
            draws = self.base.sample(sample_shape)
        return draws


class Pushforward(Measure):
    """Pushforward of a base distribution through a bijector.

    Implements $f_*\\mu$, the measure whose log-density at `y` is
    `base.log_prob(b.inverse(y)) + b.inverse_log_det_jacobian(y)`.
    Functorial in `b`: composing bijectors composes the pushforward.

    Does not commute with [`Restrict`][quivers.continuous.measure.Restrict]
    in general; the compiler's rewrite pass handles the cases that
    do (monotone bijectors with interval restrictions).
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        bijector: Bijector,
        validate_args: bool | None = None,
    ) -> None:
        self.base = base
        self.bijector = bijector
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        return _constraints.real

    def log_normalizer(self) -> Tensor:
        if isinstance(self.base, Measure):
            return self.base.log_normalizer()
        return torch.zeros(self.batch_shape) if self.batch_shape else torch.tensor(0.0)

    def log_prob(self, value: Tensor) -> Tensor:
        x = self.bijector.inverse(value)
        base_lp = self.base.log_prob(x)
        log_det = self.bijector.inverse_log_det_jacobian(value)
        return base_lp + log_det

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        x = self.base.sample(sample_shape)
        return self.bijector.forward(x)


class Mixture(Measure):
    """Finite mixture of arbitrary-family components.

    `weights` is a vector summing to one (or, in the sub-measure
    regime, any non-negative vector; the total mass is the sum).
    `components` is a list of `Distribution` instances, possibly of
    different families and possibly including
    [`PointMass`][quivers.continuous.measure.PointMass] for
    inflation / hurdle structure.

    The log-density is `logsumexp(log_weights + log_components)`
    over the component axis, broadcasting weights against the
    distribution batch shape. The symbolic log-normaliser is
    `log(sum_k w_k * Z_k)` where `Z_k` is the per-component
    normaliser (in particular when components are themselves
    sub-measures from `Restrict`).

    Associativity (the Giry monad's multiplication law) is
    canonicalised by `flatten()`, which the compiler's rewrite
    pass calls before lowering.

    [PyMC issue #5533](https://github.com/pymc-devs/pymc/issues/5533)
    documents how nested-mixture composition breaks in production
    PPLs; the operator algebra closes the gap.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        weights: Tensor,
        components: list[Distribution],
        validate_args: bool | None = None,
    ) -> None:
        if not components:
            raise ValueError("Mixture: components list must be non-empty")
        if weights.shape[-1] != len(components):
            raise ValueError(
                f"Mixture: weights last-dim {weights.shape[-1]} must "
                f"equal len(components) {len(components)}"
            )
        self.weights = weights
        self.components = list(components)
        self._num_components = len(components)
        batch_shape = weights.shape[:-1]
        for c in components:
            batch_shape = torch.broadcast_shapes(batch_shape, c.batch_shape)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=components[0].event_shape,
            validate_args=validate_args,
        )

    @property
    def num_components(self) -> int:
        return self._num_components

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        return _constraints.real

    def log_normalizer(self) -> Tensor:
        log_w = torch.log(
            self.weights.clamp_min(
                torch.finfo(self.weights.dtype).tiny,
            )
        )
        component_norms = []
        for c in self.components:
            if isinstance(c, Measure):
                component_norms.append(c.log_normalizer())
            else:
                component_norms.append(
                    torch.zeros(c.batch_shape) if c.batch_shape else torch.tensor(0.0)
                )
        stacked = torch.stack(
            [
                lw + zn
                for lw, zn in zip(
                    log_w.unbind(dim=-1),
                    component_norms,
                    strict=False,
                )
            ],
            dim=-1,
        )
        return torch.logsumexp(stacked, dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        log_w = torch.log(
            self.weights.clamp_min(
                torch.finfo(self.weights.dtype).tiny,
            )
        )
        component_lps = [c.log_prob(value) for c in self.components]
        stacked = torch.stack(
            [
                lw + lp
                for lw, lp in zip(
                    log_w.unbind(dim=-1),
                    component_lps,
                    strict=False,
                )
            ],
            dim=-1,
        )
        return torch.logsumexp(stacked, dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        cat = torch.distributions.Categorical(probs=self.weights)
        idx = cat.sample(sample_shape)
        # Components may carry smaller batch shapes than the mixture (a
        # `PointMass` spike has none), so every draw is broadcast to the
        # mixture's full sample shape before one is selected per entry.
        shape = torch.Size(sample_shape) + self.batch_shape + self.event_shape
        draws = [c.sample(sample_shape).expand(shape) for c in self.components]
        stacked = torch.stack(draws, dim=-1)
        return stacked.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    def cdf(self, value: Tensor) -> Tensor:
        """Cumulative distribution function: the weighted sum of
        component CDFs. Required for `Restrict(Mixture, ...)` to
        compute its log-normaliser without falling through to the
        discrete-base fallback.
        """
        per_component = []
        for c in self.components:
            per_component.append(c.cdf(value))
        stacked = torch.stack(per_component, dim=-1)
        return (self.weights * stacked).sum(dim=-1)

    def flatten(self) -> Mixture:
        """Apply the Giry-monad-multiplication associativity rule:
        any nested `Mixture` component is replaced by its components
        with weights scaled by the outer weight. Returns a flat
        `Mixture` whose components are not themselves `Mixture`
        instances.
        """
        flat_weights_parts = []
        flat_components: list[Distribution] = []
        for k, comp in enumerate(self.components):
            w_k = self.weights[..., k : k + 1]
            if isinstance(comp, Mixture):
                inner = comp.flatten()
                flat_weights_parts.append(w_k * inner.weights)
                flat_components.extend(inner.components)
            else:
                flat_weights_parts.append(w_k)
                flat_components.append(comp)
        flat_weights = torch.cat(flat_weights_parts, dim=-1)
        return Mixture(flat_weights, flat_components)

    def pushforward_inside(self, bijector) -> Mixture:
        """Push a single bijector inside every component:

        $$g_*\\left(\\sum_k \\pi_k\\,\\mu_k\\right)
          = \\sum_k \\pi_k\\,g_*\\mu_k.$$

        Functoriality of the Giry monad on measurable isomorphisms;
        the per-component pushforward is the canonical shape the
        renderer prefers.
        """
        new_components = [Pushforward(c, bijector) for c in self.components]
        return Mixture(self.weights, new_components)

    def restrict_to(self, low=None, high=None) -> Mixture:
        """Restrict every component to the same subset, with the
        per-component reweighting that the non-commutation of
        Restrict and Mixture requires:

        $$\\mathrm{Restrict}\\!\\left(\\sum_k \\pi_k\\,\\mu_k,\\; S\\right)
          = \\sum_k \\frac{\\pi_k\\,\\mu_k(S)}{\\sum_j \\pi_j\\,\\mu_j(S)}\\,
            \\mathrm{Restrict}(\\mu_k, S).$$

        Returns a `Mixture` whose components are each
        `Restrict(component, low, high)` and whose weights are the
        form weighted by each component's mass on the subset. This
        is the ordinary conditional probability rule: multiply each
        prior mixture weight by the probability of the subset under
        that component, then normalize the resulting weights. The
        total mass of the subset must be positive.
        """
        restricted_raw = [Restrict(c, low=low, high=high) for c in self.components]
        log_w = torch.log(
            self.weights.clamp_min(
                torch.finfo(self.weights.dtype).tiny,
            )
        )
        log_masses = [r.log_normalizer() for r in restricted_raw]
        stacked = torch.stack(
            [
                lw + lm
                for lw, lm in zip(
                    log_w.unbind(dim=-1),
                    log_masses,
                    strict=False,
                )
            ],
            dim=-1,
        )
        log_denom = torch.logsumexp(stacked, dim=-1, keepdim=True)
        new_log_weights = stacked - log_denom
        new_weights = torch.exp(new_log_weights)
        # Components are NORMALIZED truncations so the Mixture's
        # weighted sum of per-component densities equals the
        # conditional density of the original mixture restricted to
        # `S`. Using unnormalised `Restrict` here would double-count
        # the per-component truncation mass.
        normalized_components = [Normalize(r) for r in restricted_raw]
        return Mixture(new_weights, normalized_components)

    def lift_point_masses(self) -> Mixture:
        """Surface the `PointMass` components as a Bernoulli-style
        branch with the non-degenerate components in the
        complementary mixture. Returns a `Mixture` whose components
        are pairwise measure-disjoint: each `PointMass(x)` carries
        only its spike, and the non-degenerate components live in the
        complement.

        For a single `PointMass` plus a single non-degenerate
        component (the ZIP / hurdle canonical shape), the rewrite is
        a no-op: the current form is already the Bernoulli-branch
        factorisation the renderers expect. For Mixtures with
        multiple `PointMass` components at distinct values, returns a
        canonical form with the point masses ordered first.
        """
        point_masses = []
        rest = []
        point_mass_weights = []
        rest_weights = []
        for k, comp in enumerate(self.components):
            w_k = self.weights[..., k : k + 1]
            if isinstance(comp, PointMass):
                point_masses.append(comp)
                point_mass_weights.append(w_k)
            else:
                rest.append(comp)
                rest_weights.append(w_k)
        if not point_masses:
            return self
        new_components = point_masses + rest
        new_weights = torch.cat(point_mass_weights + rest_weights, dim=-1)
        return Mixture(new_weights, new_components)


class Independent(Measure):
    """Reinterpret the last `reinterpreted_batch_ndims` batch dims
    of `base` as event dims. Wraps PyTorch's
    [`Independent`][torch.distributions.independent.Independent].

    The log-density sums `base.log_prob(value)` over the
    reinterpreted dims; the log-normaliser sums in the same
    direction when `base` is itself a sub-measure.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        reinterpreted_batch_ndims: int,
        validate_args: bool | None = None,
    ) -> None:
        if reinterpreted_batch_ndims < 1:
            raise ValueError(
                "Independent: reinterpreted_batch_ndims must be >= 1, "
                f"got {reinterpreted_batch_ndims}"
            )
        self.base = base
        self.reinterpreted_batch_ndims = int(reinterpreted_batch_ndims)
        self._inner = torch.distributions.Independent(
            base,
            reinterpreted_batch_ndims,
        )
        super().__init__(
            batch_shape=self._inner.batch_shape,
            event_shape=self._inner.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        return self._inner.support

    def log_normalizer(self) -> Tensor:
        if isinstance(self.base, Measure):
            inner = self.base.log_normalizer()
            for _ in range(self.reinterpreted_batch_ndims):
                inner = inner.sum(dim=-1)
            return inner
        return torch.zeros(self.batch_shape) if self.batch_shape else torch.tensor(0.0)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._inner.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self._inner.sample(sample_shape)


class Normalize(Measure):
    """Rescale a sub-measure to a probability measure.

    `log_prob_normalized(value) = log_prob(value) - log_normalizer()`
    by construction; this wrapper makes the normalisation explicit
    so downstream code can rely on `log_normalizer() == 0`. Use at
    the observe / sample boundary or whenever a probability measure
    is required (e.g. before sampling).
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        self.base = base
        if isinstance(base, Measure):
            self._base_log_normalizer = base.log_normalizer()
        else:
            self._base_log_normalizer = (
                torch.zeros(base.batch_shape) if base.batch_shape else torch.tensor(0.0)
            )
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        return self.base.support

    def log_normalizer(self) -> Tensor:
        return torch.zeros(self.batch_shape) if self.batch_shape else torch.tensor(0.0)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.base.log_prob(value) - self._base_log_normalizer

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.base.sample(sample_shape)


def _symmetric_fold_log_normalizer(
    base: Distribution,
    low: Tensor | None,
    high: Tensor | None,
) -> Tensor | None:
    """Closed-form log-normaliser for a one-sided restriction that
    cuts a location-symmetric base at its own centre of symmetry.

    A base whose density is symmetric about its location `loc`
    (Normal, Cauchy, Laplace, StudentT, ...) splits exactly in half
    at `loc`: the mass above `loc` and the mass below `loc` are each
    `0.5`, whatever the tails. So a one-sided restriction to
    `[loc, +inf)` or `(-inf, loc]` has mass `0.5`, i.e.
    `log_normalizer = -log 2`, with no CDF required. This is the fold
    that turns `Restrict(StudentT(nu, 0, s), low=0)` into the standard
    half-Student-t whose density on the positive half is
    `base.log_prob(x) + log 2`.

    Returns the `-log 2` normaliser (shaped like `base.batch_shape`)
    when the restriction is one-sided, the base exposes a `loc`, the
    finite cut equals `loc` elementwise, and the base is numerically
    symmetric about `loc`; returns `None` otherwise so the caller can
    fall through to the discrete pmf-sum path.
    """
    one_sided = (low is None) != (high is None)
    if not one_sided:
        return None
    loc = getattr(base, "loc", None)
    if not isinstance(loc, Tensor):
        return None
    cut = low if high is None else high
    if cut is None:
        return None
    cut = cut.to(dtype=loc.dtype)
    if not bool(torch.isclose(loc, cut, atol=1e-7, rtol=1e-6).all()):
        return None
    scale = getattr(base, "scale", None)
    ref = scale if isinstance(scale, Tensor) else torch.ones_like(loc)
    for factor in (0.5, 1.0, 2.0):
        delta = factor * ref
        lp_above = base.log_prob(loc + delta)
        lp_below = base.log_prob(loc - delta)
        if not bool(torch.isclose(lp_above, lp_below, atol=1e-6, rtol=1e-5).all()):
            return None
    return torch.full_like(loc, -math.log(2.0))


def _discrete_restriction_log_normalizer(
    base: Distribution,
    low: Tensor | None,
    high: Tensor | None,
) -> Tensor:
    """Fallback log-normaliser for discrete bases without a closed-form
    `cdf` method (PyTorch's `Poisson`, `Geometric`, `NegativeBinomial`
    fall here). Sums the pmf over the inclusive integer range
    `[low, high]`; for one-sided cases, the complement uses
    `expm1` / closed forms when the base supplies them analytically.
    """
    if isinstance(base, torch.distributions.Poisson):
        rate = base.rate
        if low is not None and high is None:
            low_int = int(low.item()) if isinstance(low, Tensor) else int(low)
            if low_int <= 0:
                return torch.zeros_like(rate)
            if low_int == 1:
                surv = -torch.expm1(-rate)
                return torch.log(surv.clamp_min(torch.finfo(surv.dtype).tiny))
            cum = torch.zeros_like(rate)
            for k in range(low_int):
                cum = cum + torch.exp(
                    -rate
                    + k * torch.log(rate)
                    - torch.lgamma(
                        torch.tensor(k + 1, dtype=rate.dtype),
                    )
                )
            surv = (1.0 - cum).clamp_min(torch.finfo(rate.dtype).tiny)
            return torch.log(surv)
        if low is None and high is not None:
            high_int = int(high.item()) if isinstance(high, Tensor) else int(high)
            cum = torch.zeros_like(rate)
            for k in range(high_int + 1):
                cum = cum + torch.exp(
                    -rate
                    + k * torch.log(rate)
                    - torch.lgamma(
                        torch.tensor(k + 1, dtype=rate.dtype),
                    )
                )
            return torch.log(cum.clamp_min(torch.finfo(rate.dtype).tiny))
        low_int = int(low.item()) if isinstance(low, Tensor) else int(low)
        high_int = int(high.item()) if isinstance(high, Tensor) else int(high)
        cum = torch.zeros_like(rate)
        for k in range(low_int, high_int + 1):
            cum = cum + torch.exp(
                -rate
                + k * torch.log(rate)
                - torch.lgamma(
                    torch.tensor(k + 1, dtype=rate.dtype),
                )
            )
        return torch.log(cum.clamp_min(torch.finfo(rate.dtype).tiny))
    raise NotImplementedError(
        f"Restrict.log_normalizer: no closed-form fallback for "
        f"discrete base {type(base).__name__!r}; supply a base with "
        "a `cdf` method or extend "
        "`_discrete_restriction_log_normalizer` with the case."
    )


def normalize_at_boundary(d: Distribution) -> Distribution:
    """Lift `d` to a probability measure if it is a sub-measure
    with non-zero log-normaliser; otherwise return `d` unchanged.

    Called by the inline / observe path to enforce the
    "renormalise at the observe / sample boundary" invariant the
    sub-Giry algebra relies on.
    """
    if isinstance(d, Measure):
        norm = d.log_normalizer()
        if isinstance(norm, Tensor):
            if torch.any(norm.abs() > 1e-10):
                return Normalize(d)
        else:
            if abs(float(norm)) > 1e-10:
                return Normalize(d)
    return d


__all__ = [
    "Independent",
    "Measure",
    "Mixture",
    "Normalize",
    "PointMass",
    "Pushforward",
    "Restrict",
    "normalize_at_boundary",
]
