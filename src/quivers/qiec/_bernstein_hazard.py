"""Standard-library reference for normalized Bernstein baseline hazards."""

from __future__ import annotations

import bisect
import math
import random

from quivers._splines import _bernstein_value


class BernsteinHazard:
    """A scalar normalized piecewise Bernstein proportional-hazards law.

    Parameters
    ----------
    breaks : Sequence[float]
        Finite increasing boundaries starting at zero.
    controls : Sequence[Sequence[float]]
        Positive rectangular coefficient table; its finite-interval baseline
        integral equals one within ordinary simplex tolerance.
    rate : float or None
        Positive hazard multiplier, exclusive with log_rate.
    log_rate : float or None
        Finite log multiplier.
    """

    def __init__(self, breaks, controls, rate=None, *, log_rate=None):
        if (rate is None) == (log_rate is None):
            raise ValueError("supply exactly one of rate and log_rate")
        self.breaks = tuple(map(float, breaks))
        self.controls = tuple(tuple(map(float, row)) for row in controls)
        segments = len(self.controls)
        width = len(self.controls[0]) if segments else 0
        if (
            segments < 1
            or width < 1
            or len(self.breaks) != segments + 1
            or any(len(row) != width for row in self.controls)
        ):
            raise ValueError(
                "Bernstein boundaries[S+1] must match controls[S,K] with S,K positive"
            )
        if (
            not all(math.isfinite(x) for x in self.breaks)
            or self.breaks[0] != 0
            or any(b <= a for a, b in zip(self.breaks, self.breaks[1:]))
        ):
            raise ValueError(
                "Bernstein boundaries must start zero and increase finitely"
            )
        if any(not math.isfinite(c) or c <= 0 for row in self.controls for c in row):
            raise ValueError("Bernstein controls must be finite and strictly positive")
        integral = 0.0
        pieces = []
        for a, b, row in zip(self.breaks, self.breaks[1:], self.controls):
            coefficients = [integral]
            for c in row:
                coefficients.append(coefficients[-1] + (b - a) * (c / width))
            integral = coefficients[-1]
            pieces.append(tuple(coefficients))
        if not math.isclose(integral, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Bernstein baseline integral must equal one")
        self._integrals = tuple(pieces)
        if rate is not None:
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Bernstein hazard rate must be finite and positive")
            self.log_rate = math.log(rate)
        else:
            assert log_rate is not None
            self.log_rate = float(log_rate)
        if not math.isfinite(self.log_rate):
            raise ValueError("Bernstein log rate must be finite")

    def _baseline(self, value):
        """Evaluate the baseline hazard and its cumulative integral.

        Parameters
        ----------
        value : float
            Event time.

        Returns
        -------
        tuple[float, float]
            Baseline hazard and integrated baseline hazard.
        """
        upper = self.breaks[-1]
        x = max(0.0, min(value, upper))
        index = min(bisect.bisect_right(self.breaks, x) - 1, len(self.controls) - 1)
        u = (x - self.breaks[index]) / (self.breaks[index + 1] - self.breaks[index])
        hazard = _bernstein_value(self.controls[index], u)
        cumulative = _bernstein_value(self._integrals[index], u)
        if value > upper:
            hazard = self.controls[-1][-1]
            cumulative += hazard * (value - upper)
        return hazard, cumulative

    def log_prob(self, value):
        """Evaluate the normalized scalar log density.

        Parameters
        ----------
        value : float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        float
            Normalized log density, negative infinity outside the support.

        Raises
        ------
        ValueError
            If the time is NaN.
        """
        if math.isnan(value):
            raise ValueError("Bernstein hazard observation must not be NaN")
        if value < 0 or math.isinf(value):
            return -math.inf
        hazard, _ = self._baseline(min(value, self.breaks[-1]))
        return self.log_rate + math.log(hazard) + self.log_survival_function(value)

    def _log_cumulative(self, value):
        """Evaluate the log cumulative proportional hazard.

        Parameters
        ----------
        value : float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        float
            Log cumulative hazard with exact support limits.

        Raises
        ------
        ValueError
            If the time is NaN.
        """
        if math.isnan(value):
            raise ValueError("Bernstein hazard observation must not be NaN")
        if value <= 0:
            return -math.inf
        upper = self.breaks[-1]
        if value == math.inf:
            return math.inf
        if value > upper:
            hazard, integral = self._baseline(upper)
            a, b = math.log(integral), math.log(hazard) + math.log(value - upper)
            large, small = max(a, b), min(a, b)
            return self.log_rate + large + math.log1p(math.exp(small - large))
        _, integral = self._baseline(value)
        return self.log_rate + math.log(integral)

    def log_survival_function(self, value):
        """Evaluate log survival, including the infinite tail.

        Parameters
        ----------
        value : float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        float
            Log survival, evaluated directly from cumulative hazard.
        """
        try:
            return -math.exp(self._log_cumulative(value))
        except OverflowError:
            return -math.inf

    def cdf(self, value):
        """Evaluate the cumulative event probability.

        Parameters
        ----------
        value : float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        float
            Cumulative event probability.
        """
        return -math.expm1(self.log_survival_function(value))

    def log_cdf(self, value):
        """Evaluate log cumulative probability without tail subtraction.

        Parameters
        ----------
        value : float
            Event time; negative and infinite values use the exact support limits.

        Returns
        -------
        float
            Log cumulative probability, retaining small positive probabilities.
        """
        exponent = self._log_cumulative(value)
        if exponent < -36:
            return exponent
        if exponent > math.log(750):
            return 0.0
        return math.log(-math.expm1(-math.exp(exponent)))

    def _inverse_baseline(self, log_target):
        """Invert the cumulative baseline with an explicit solver budget.

        Parameters
        ----------
        log_target : float
            Log positive target integrated hazard.

        Returns
        -------
        float
            Event time, using the analytic constant-hazard tail when applicable.

        Raises
        ------
        OverflowError
            If a positive target or result exceeds floating-point capacity.
        RuntimeError
            If 128 safeguarded Newton iterations cannot resolve the root.
        """
        upper = self.breaks[-1]
        tail_hazard, tail_integral = self._baseline(upper)
        if log_target >= math.log(tail_integral):
            difference = log_target - math.log(tail_integral)
            if difference == 0:
                return upper
            log_difference = (
                math.log(math.expm1(difference))
                if difference < math.log(2)
                else difference + math.log1p(-math.exp(-difference))
            )
            try:
                value = upper + math.exp(
                    math.log(tail_integral) - math.log(tail_hazard) + log_difference
                )
            except OverflowError as error:
                raise OverflowError(
                    "Bernstein hazard quantile exceeds finite floating-point time capacity"
                ) from error
            if not math.isfinite(value):
                raise OverflowError(
                    "Bernstein hazard quantile exceeds finite floating-point time capacity"
                )
            return value
        target = math.exp(log_target)
        if target == 0:
            raise OverflowError(
                "Bernstein hazard positive integrated quantile is below floating-point capacity"
            )
        low, high = 0.0, upper
        value = upper * target / tail_integral
        for _ in range(128):
            hazard, integral = self._baseline(value)
            residual = integral - target
            if (
                abs(residual) <= 32 * math.ulp(1.0) * target
                or math.nextafter(low, high) >= high
            ):
                return value
            if residual > 0:
                high = value
            else:
                low = value
            following = value - residual / hazard
            value = following if low < following < high else low / 2 + high / 2
        raise RuntimeError(
            "Bernstein hazard quantile solver exceeded 128 safeguarded Newton iterations"
        )

    def icdf(self, probability):
        """Invert the event CDF, with explicit floating-point capacity checks.

        Parameters
        ----------
        probability : float
            Probability in the closed unit interval.

        Returns
        -------
        float
            Event quantile; the endpoints return zero and positive infinity.

        Raises
        ------
        ValueError
            If a probability is NaN or outside the unit interval.
        OverflowError
            If the finite positive quantile is unrepresentable.
        RuntimeError
            If the bounded root solver does not converge.
        """
        if not 0 <= probability <= 1:
            raise ValueError("Bernstein hazard quantile probability must lie in [0, 1]")
        if probability == 0:
            return 0.0
        if probability == 1:
            return math.inf
        log_target = math.log(-math.log1p(-probability)) - self.log_rate
        return self._inverse_baseline(log_target)

    def sample(self, rng: random.Random):
        """Draw one event time from a caller-owned random generator.

        Parameters
        ----------
        rng : random.Random
            Caller-owned uniform random generator.

        Returns
        -------
        float
            One event time from the full normalized law.
        """
        return self.icdf(rng.random())


def distribution(arguments):
    """Build the checked scalar kernel from named reference arguments.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Boundary vector, coefficient matrix, and one rate representation.

    Returns
    -------
    BernsteinHazard
        Checked normalized baseline law.

    Raises
    ------
    DistributionError
        If structure, numeric values, normalization, or rate are invalid.
    """
    from collections.abc import Sequence
    from quivers.qiec.reference_support import DistributionError, real_parameter

    def vector(value):
        """Read one finite real structural row.

        Parameters
        ----------
        value : object
            Candidate numeric sequence.

        Returns
        -------
        tuple[float, ...]
            Real components before constructor validation.

        Raises
        ------
        DistributionError
            If any component is not a real number.
        """
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise DistributionError(
                "Bernstein hazard structural parameters must be sequences"
            )
        if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
            raise DistributionError(
                "Bernstein hazard structural values must be real numbers"
            )
        return tuple(map(float, value))

    try:
        breaks = vector(arguments["breaks"])
        raw = arguments["controls"]
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            raise DistributionError("Bernstein hazard controls must be a matrix")
        controls = tuple(vector(row) for row in raw)
        rates = {
            name: real_parameter(arguments, name, "BernsteinHazard")
            for name in ("rate", "log_rate")
            if name in arguments
        }
        return BernsteinHazard(breaks, controls, **rates)
    except (ValueError, TypeError, KeyError) as error:
        raise DistributionError(f"BernsteinHazard: {error}") from error


def density(arguments, value):
    """Evaluate one scalar event under the reference hazard law.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named family arguments.
    value : object
        Scalar event time.

    Returns
    -------
    float
        Normalized log density.

    Raises
    ------
    DistributionError
        If arguments or the observation are invalid.
    """
    from quivers.qiec.reference_support import DistributionError, finite_value

    try:
        return distribution(arguments).log_prob(finite_value(value, "BernsteinHazard"))
    except ValueError as error:
        raise DistributionError(f"BernsteinHazard: {error}") from error


def sample(arguments, rng):
    """Draw one scalar event through the bounded cumulative-hazard inverse.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named family arguments.
    rng : random.Random
        Caller-owned random generator.

    Returns
    -------
    float
        Event time from the full law, including its constant-hazard tail.

    Raises
    ------
    DistributionError
        If parameters are invalid or inversion exceeds numerical capacity.
    """
    from quivers.qiec.reference_support import DistributionError

    try:
        return distribution(arguments).sample(rng)
    except (ValueError, OverflowError, RuntimeError) as error:
        raise DistributionError(f"BernsteinHazard: {error}") from error
