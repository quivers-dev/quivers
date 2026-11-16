"""Normalized positive Bernstein baseline hazards with an explicit tail."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints

from quivers.continuous._shape_families import _log1mexp_neg_exp


def _bernstein(coefficients: Tensor, u: Tensor) -> Tensor:
    """Evaluate Bernstein polynomials by de Casteljau's convex recurrence.

    Parameters
    ----------
    coefficients : Tensor
        Coefficients on the final polynomial axis.
    u : Tensor
        Evaluation coordinates on the unit interval.

    Returns
    -------
    Tensor
        Polynomial values after convex de Casteljau reductions.
    """
    for _ in range(coefficients.shape[-1] - 1):
        coefficients = (1 - u[..., None]) * coefficients[..., :-1] + u[
            ..., None
        ] * coefficients[..., 1:]
    return coefficients[..., 0]


class BernsteinHazard(Distribution):
    """A normalized positive baseline hazard and a proportional rate.

    Parameters
    ----------
    breaks : Tensor or Sequence[float]
        Strictly increasing finite boundaries with trailing shape ``(S+1,)``,
        starting at zero.
    controls : Tensor or Sequence[Sequence[float]]
        Positive Bernstein coefficients of trailing shape ``(S,K)``. The
        integrated baseline over the finite interval must equal one.
    rate : Tensor or float, optional
        Positive hazard multiplier, exclusive with log_rate.
    log_rate : Tensor or float, optional
        Finite log hazard multiplier.
    validate_args : bool or None
        Torch parameter-validation policy.

    Notes
    -----
    The final hazard value continues constantly beyond the upper boundary.
    This positive tail retains total probability one. Normalizing the finite
    baseline integral identifies its shape separately from the rate.
    The immutable finite_moment_orders certificate records the first two
    finite moments; an analytic formula for those moments is not provided.
    """

    arg_constraints = {
        "breaks": constraints.real_vector,
        "controls": constraints.independent(constraints.positive, 2),
        "rate": constraints.positive,
        "log_rate": constraints.real,
    }
    support = constraints.nonnegative
    has_rsample = True
    finite_moment_orders: frozenset[int] = frozenset({1, 2})

    def __init__(
        self, breaks, controls, rate=None, *, log_rate=None, validate_args=None
    ):
        if (rate is None) == (log_rate is None):
            raise ValueError("supply exactly one of rate and log_rate")
        device = next(
            (
                value.device
                for value in (controls, breaks, rate, log_rate)
                if isinstance(value, Tensor)
            ),
            None,
        )
        if isinstance(rate, bool) or isinstance(log_rate, bool):
            raise ValueError("Bernstein hazard rate must be real, not Boolean")
        raw_breaks = (
            torch.as_tensor(breaks, device=device)
            if isinstance(breaks, Tensor)
            else torch.as_tensor(breaks, dtype=torch.float64, device=device)
        )
        raw_controls = (
            torch.as_tensor(controls, device=device)
            if isinstance(controls, Tensor)
            else torch.as_tensor(controls, dtype=torch.float64, device=device)
        )
        if raw_breaks.dtype == torch.bool or raw_controls.dtype == torch.bool:
            raise ValueError(
                "Bernstein structural parameters must be real, not Boolean"
            )
        self.breaks = raw_breaks.to(torch.float64)
        self.controls = raw_controls.to(torch.float64)
        if self.breaks.ndim < 1 or self.controls.ndim < 2:
            raise ValueError(
                "Bernstein hazard needs a boundary vector and coefficient matrix"
            )
        segments, width = self.controls.shape[-2:]
        if segments < 1 or width < 1 or self.breaks.shape[-1] != segments + 1:
            raise ValueError(
                "Bernstein boundaries[S+1] must match controls[S,K] with S,K positive"
            )
        value = rate if rate is not None else log_rate
        raw = (
            torch.as_tensor(value, device=device)
            if isinstance(value, Tensor)
            else torch.as_tensor(value, dtype=torch.float64, device=device)
        )
        if raw.dtype == torch.bool:
            raise ValueError("Bernstein hazard rate must be real, not Boolean")
        raw = raw.to(torch.float64)
        if rate is not None:
            if validate_args is not False and not bool(
                torch.all(torch.isfinite(raw) & (raw > 0))
            ):
                raise ValueError("Bernstein hazard rate must be finite and positive")
            raw = raw.log()
        batch = torch.broadcast_shapes(
            self.breaks.shape[:-1], self.controls.shape[:-2], raw.shape
        )
        self.breaks = self.breaks.expand(*batch, segments + 1)
        self.controls = self.controls.expand(*batch, segments, width)
        self.log_rate = raw.expand(batch)
        intervals = self.breaks[..., 1:] - self.breaks[..., :-1]
        area = intervals * (self.controls / width).sum(-1)
        if validate_args is not False:
            if not bool(
                torch.isfinite(self.breaks).all()
                & (self.breaks[..., 0] == 0).all()
                & (intervals > 0).all()
                & torch.isfinite(intervals).all()
            ):
                raise ValueError(
                    "Bernstein boundaries must start zero and increase finitely"
                )
            if not bool(
                torch.isfinite(self.controls).all() & (self.controls > 0).all()
            ):
                raise ValueError(
                    "Bernstein controls must be finite and strictly positive"
                )
            if not bool(torch.isfinite(self.log_rate).all()):
                raise ValueError("Bernstein log rate must be finite")
            if not bool(
                torch.isclose(
                    area.sum(-1), area.new_tensor(1.0), rtol=1e-6, atol=1e-8
                ).all()
            ):
                raise ValueError("Bernstein baseline integral must equal one")
        local = (self.controls / width) * intervals[..., None]
        prefix = torch.cat(
            (torch.zeros_like(area[..., :1]), area.cumsum(-1)[..., :-1]), -1
        )
        self._integrals = prefix[..., None] + torch.cat(
            (torch.zeros_like(local[..., :1]), local.cumsum(-1)), -1
        )
        super().__init__(batch, validate_args=validate_args)

    @property
    def rate(self):
        """Return the proportional multiplier when representable.

        Returns
        -------
        Tensor
            Exponentiated log rate. Density and tail calculations retain
            log-rate arithmetic even when this view overflows.
        """
        return self.log_rate.exp()

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without rebuilding the integrated table.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested leading batch dimensions.
        _instance : BernsteinHazard or None
            Optional instance populated through the Torch expansion protocol.

        Returns
        -------
        BernsteinHazard
            Distribution retaining the same baseline and integrated table.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        new.breaks = self.breaks.expand(*shape, self.breaks.shape[-1])
        new.controls = self.controls.expand(*shape, *self.controls.shape[-2:])
        new.log_rate = self.log_rate.expand(shape)
        new._integrals = self._integrals.expand(*shape, *self._integrals.shape[-2:])
        super(BernsteinHazard, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _baseline(self, value, *, detached=False):
        """Evaluate the baseline hazard and its cumulative integral.

        Parameters
        ----------
        value : Tensor
            Event time.
        detached : bool
            Evaluate primal parameters without forward or reverse derivatives.

        Returns
        -------
        tuple[Tensor, Tensor]
            Baseline hazard and integrated baseline hazard.
        """
        shape = torch.broadcast_shapes(value.shape, self.batch_shape)
        value = value.expand(shape)
        breaks = self.breaks.detach() if detached else self.breaks
        controls = self.controls.detach() if detached else self.controls
        integrals = self._integrals.detach() if detached else self._integrals
        upper = breaks[..., -1]
        x = torch.where(value > upper, upper, value.clamp_min(0))
        bounds = breaks[..., 1:-1].expand(*shape, breaks.shape[-1] - 2)
        index = torch.searchsorted(
            bounds.contiguous(), x[..., None].contiguous(), right=True
        ).squeeze(-1)
        expanded = breaks.expand(*shape, breaks.shape[-1])
        a = expanded.gather(-1, index[..., None]).squeeze(-1)
        b = expanded.gather(-1, index[..., None] + 1).squeeze(-1)
        u = (x - a) / (b - a)
        results = []
        for table in (controls, integrals):
            segments, width = table.shape[-2:]
            piece = (
                table.expand(*shape, segments, width)
                .gather(-2, index[..., None, None].expand(*shape, 1, width))
                .squeeze(-2)
            )
            results.append(_bernstein(piece, u))
        hazard, integral = results
        tail_hazard = controls[..., -1, -1]
        integral = integral + tail_hazard * torch.where(
            value > upper, value - upper, 0.0
        )
        return torch.where(value > upper, tail_hazard, hazard), integral

    def _value(self, value):
        """Convert an observation to the parameter dtype and device.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Converted observation tensor.

        Raises
        ------
        ValueError
            If any entry is NaN.
        """
        value = torch.as_tensor(
            value, dtype=self.controls.dtype, device=self.controls.device
        )
        if bool(torch.isnan(value).any()):
            raise ValueError("Bernstein hazard observation must not be NaN")
        return value

    def _log_cumulative(self, value):
        """Evaluate the log cumulative proportional hazard.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Log cumulative hazard with exact support limits.
        """
        safe = torch.where(
            torch.isfinite(value) & (value > 0),
            value,
            self.breaks[..., -1],
        )
        upper = self.breaks[..., -1]
        _, baseline = self._baseline(torch.where(safe > upper, upper, safe))
        tail_hazard, tail_integral = self._baseline(upper)
        tail = safe > upper
        excess = torch.where(tail, safe - upper, 1.0)
        log_baseline = torch.where(
            tail,
            torch.logaddexp(tail_integral.log(), tail_hazard.log() + excess.log()),
            baseline.log(),
        )
        result = self.log_rate + log_baseline
        return torch.where(
            value <= 0, -torch.inf, torch.where(value == torch.inf, torch.inf, result)
        )

    def _cumulative(self, value):
        """Evaluate the cumulative proportional hazard.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Cumulative hazard with exact support limits.
        """
        return self._log_cumulative(value).exp()

    def log_prob(self, value):
        """Evaluate the normalized time density on the nonnegative support.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Normalized log density, negative infinity outside the support.

        Raises
        ------
        ValueError
            If the time is NaN.
        """
        value = self._value(value)
        safe = torch.where(
            torch.isfinite(value) & (value >= 0),
            value,
            self.breaks[..., -1],
        )
        hazard, _ = self._baseline(
            torch.where(safe > self.breaks[..., -1], self.breaks[..., -1], safe)
        )
        cumulative = self._cumulative(value)
        result = self.log_rate + hazard.log() - cumulative
        return torch.where(torch.isfinite(value) & (value >= 0), result, -torch.inf)

    def log_survival_function(self, value):
        """Evaluate log survival directly from the integrated hazard.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Log survival, evaluated directly from cumulative hazard.
        """
        value = self._value(value)
        return -self._cumulative(value)

    def cdf(self, value):
        """Evaluate cumulative event probability, including exact support tails.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Cumulative event probability.
        """
        return -torch.expm1(self.log_survival_function(value))

    def log_cdf(self, value):
        """Evaluate log CDF without subtracting small survival probabilities.

        Parameters
        ----------
        value : Tensor or float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        Tensor
            Log cumulative probability, retaining small positive probabilities.
        """
        value = self._value(value)
        return _log1mexp_neg_exp(self._log_cumulative(value))

    def _inverse_baseline(self, log_target):
        """Invert the cumulative baseline with an explicit solver budget.

        Parameters
        ----------
        log_target : Tensor
            Log positive target integrated hazard.

        Returns
        -------
        Tensor
            Event time, using the analytic constant-hazard tail when applicable.

        Raises
        ------
        FloatingPointError
            If a positive target or result exceeds floating-point capacity.
        RuntimeError
            If 128 safeguarded Newton iterations cannot resolve the root.
        """
        upper = self.breaks[..., -1]
        tail_hazard, tail_integral = self._baseline(upper)
        is_tail = log_target >= tail_integral.log()
        safe_target = torch.where(is_tail, (tail_integral / 2).log(), log_target).exp()
        if bool((safe_target <= 0).any()):
            raise FloatingPointError(
                "Bernstein hazard positive integrated quantile is below floating-point capacity"
            )
        with torch.no_grad():
            primal_target = safe_target.detach()
            low, high = (
                torch.zeros_like(primal_target),
                upper.detach().expand_as(primal_target),
            )
            root = upper.detach() * primal_target / tail_integral.detach()
            done = torch.zeros_like(root, dtype=torch.bool)
            for _ in range(128):
                hazard, integral = self._baseline(root, detached=True)
                residual = integral - primal_target
                done = (
                    done
                    | (
                        residual.abs()
                        <= 32 * torch.finfo(root.dtype).eps * primal_target
                    )
                    | (torch.nextafter(low, high) >= high)
                )
                if bool(done.all()):
                    break
                high = torch.where((residual > 0) & ~done, root, high)
                low = torch.where((residual <= 0) & ~done, root, low)
                candidate = root - residual / hazard
                candidate = torch.where(
                    (candidate > low) & (candidate < high),
                    candidate,
                    low / 2 + high / 2,
                )
                root = torch.where(done, root, candidate)
            else:
                raise RuntimeError(
                    "Bernstein hazard quantile solver exceeded 128 safeguarded Newton iterations"
                )
        # Two corrections reconstruct first and second implicit derivatives at
        # the converged root without retaining the iterative solver's graph.
        for _ in range(2):
            hazard, integral = self._baseline(root)
            root = root + (safe_target - integral) / hazard
        tail_hazard = torch.where(is_tail, tail_hazard, 1.0)
        tail_integral = torch.where(is_tail, tail_integral, 1.0)
        difference = torch.where(is_tail, log_target - tail_integral.log(), 1.0)
        small = difference < 0.6931471805599453
        low = torch.where(small, difference, 0.5)
        high = torch.where(small, 1.0, difference)
        log_scale = tail_integral.log() - tail_hazard.log()
        finite_scale = log_scale < 709.782712893384
        direct_scale = torch.where(small & finite_scale, log_scale, 0.0).exp()
        direct = low.expm1() * direct_scale
        rescue = small & ~finite_scale & (low > 0)
        rescued = (
            torch.where(rescue, log_scale, 0.0)
            + torch.where(rescue, low, 1.0).expm1().log()
        ).exp()
        direct = torch.where(finite_scale, direct, torch.where(low > 0, rescued, 0.0))
        log_large = (
            torch.where(~small, log_scale, 0.0) + high + torch.log1p(-(-high).exp())
        )
        tail = upper + torch.where(small, direct, log_large.exp())
        result = torch.where(is_tail, tail, root)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError(
                "Bernstein hazard quantile exceeds finite floating-point time capacity"
            )
        return result

    def icdf(self, value):
        """Invert the integrated hazard with a bounded, checked root solve.

        Parameters
        ----------
        value : Tensor
            Probability in the closed unit interval.

        Returns
        -------
        Tensor
            Event quantile; the endpoints return zero and positive infinity.

        Raises
        ------
        ValueError
            If a probability is NaN or outside the unit interval.
        FloatingPointError
            If the finite positive quantile is unrepresentable.
        RuntimeError
            If the bounded root solver does not converge.
        """
        value = torch.as_tensor(
            value, dtype=self.controls.dtype, device=self.controls.device
        )
        if bool(((value < 0) | (value > 1) | torch.isnan(value)).any()):
            raise ValueError("Bernstein hazard quantile probability must lie in [0, 1]")
        interior = (value > 0) & (value < 1)
        safe = torch.where(interior, value, value.new_tensor(0.5))
        log_target = (-torch.log1p(-safe)).log() - self.log_rate
        log_target = torch.where(
            interior, log_target, torch.full_like(log_target, -0.6931471805599453)
        )
        result = self._inverse_baseline(log_target)
        return torch.where(value == 0, 0.0, torch.where(value == 1, torch.inf, result))

    def rsample(self, sample_shape=torch.Size()):
        """Draw vectorized event times by inverse cumulative hazard.

        Parameters
        ----------
        sample_shape : torch.Size
            Independent sample dimensions preceding batch dimensions.

        Returns
        -------
        Tensor
            Pathwise event times from the full normalized law.
        """
        shape = self._extended_shape(sample_shape)
        exponential = torch.empty(
            shape, dtype=self.controls.dtype, device=self.controls.device
        ).exponential_()
        return self._inverse_baseline(exponential.log() - self.log_rate)
