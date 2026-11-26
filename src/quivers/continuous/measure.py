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
from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints, biject_to, transform_to
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform, SigmoidTransform
from quivers._positions import anywhere, demand, depends_on_position
from quivers._independent import independent_ndims

from quivers.continuous.bijectors import Bijector
from quivers.continuous._truncated_normal import TruncatedNormal
from quivers.continuous._measure_queries import log_mass as _native_log_mass
from quivers.continuous._interval import (
    _closed_flag,
    _endpoint_tails,
    _log_tails,
    _reference,
    log_interval_mass,
)
from quivers.continuous._recorded_event_tails import _log_complement, invert_log_cdf
from quivers.continuous._truncated_mean import truncated_mean


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

    def select_batch(self, active: Tensor, *, output_shape=None) -> Distribution:
        """Reconstruct selected rows of the compositional measure algebra.

        Parameters
        ----------
        active : Tensor
            Boolean selection over the measure's broadcast batch.
        output_shape : torch.Size or None
            Optional selected batch layout, retaining reinterpreted event axes.

        Returns
        -------
        Distribution
            The same measure construction on the selected parameter rows.

        Raises
        ------
        NotImplementedError
            If a custom measure supplies no batch reconstruction rule.
        """
        from quivers.continuous._batch_selection import select_batch, select_parameter

        def parameter(value, rank=0):
            """Select parameter batch axes while retaining intrinsic axes."""
            if not isinstance(value, Tensor):
                return value
            return select_parameter(
                value, self.batch_shape, rank, active, output_shape=output_shape
            )

        def base(distribution):
            """Broadcast a component before selecting the common batch rows."""
            expanded = distribution.expand(self.batch_shape)
            return select_batch(expanded, active, output_shape=output_shape)

        if isinstance(self, PointMass):
            return PointMass(parameter(self.value))
        if isinstance(self, Restrict):
            return Restrict(
                base(self.base),
                parameter(self.low),
                parameter(self.high),
                lower_closed=parameter(self.lower_closed),
                upper_closed=parameter(self.upper_closed),
            )
        if isinstance(self, Normalize):
            return Normalize(base(self.base))
        if isinstance(self, Pushforward):
            transform = self.bijector.select_batch(
                self.batch_shape,
                active,
                event_shape=self.base.event_shape,
                output_shape=output_shape,
            )
            return Pushforward(base(self.base), transform)
        if isinstance(self, Mixture):
            return Mixture(
                parameter(self.weights, 1), [base(c) for c in self.components]
            )
        if isinstance(self, Independent):
            selected = select_batch(self._inner, active, output_shape=output_shape)
            return Independent(selected.base_dist, self.reinterpreted_batch_ndims)
        raise NotImplementedError(
            f"{type(self).__name__} has no batch reconstruction rule"
        )

    def log_prob_normalized(self, value: Tensor) -> Tensor:
        """Log-density of the renormalised probability measure.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Log density after subtracting the raw total log mass."""
        return self.log_prob(value) - self.log_normalizer()

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Read a raw event mass supplied by the concrete measure algebra.

        Parameters
        ----------
        low, high : Tensor or float or None
            Optional scalar bounds.
        lower_closed, upper_closed : bool or Tensor
            Whether each endpoint belongs to the event.

        Returns
        -------
        Tensor
            Raw log event mass.

        Raises
        ------
        NotImplementedError
            If a custom measure has no interval integration capability.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interval event masses"
        )

    def log_cdf(self, value: Tensor) -> Tensor:
        """Read raw log mass on the closed lower half-line.

        Parameters
        ----------
        value : Tensor
            Scalar cutoffs, broadcast against the measure batch.

        Returns
        -------
        Tensor
            Log mass on ``(-inf,value]``; a normalized measure has an ordinary log CDF.
        """
        return self.log_interval_mass(None, value)

    def log_survival_function(self, value: Tensor) -> Tensor:
        """Read raw log mass on the open upper half-line.

        Parameters
        ----------
        value : Tensor
            Scalar cutoffs, broadcast against the measure batch.

        Returns
        -------
        Tensor
            Log mass on ``(value,inf)``.
        """
        return self.log_interval_mass(value, None, lower_closed=False)

    def log_cdf_left(self, value: Tensor) -> Tensor:
        """Read raw log mass strictly below a cutoff.

        Parameters
        ----------
        value : Tensor
            Scalar cutoffs, broadcast against the measure batch.

        Returns
        -------
        Tensor
            Log mass on ``(-inf,value)``.
        """
        return self.log_interval_mass(None, value, upper_closed=False)

    def log_survival_function_left(self, value: Tensor) -> Tensor:
        """Read raw log mass at or above a cutoff.

        Parameters
        ----------
        value : Tensor
            Scalar cutoffs, broadcast against the measure batch.

        Returns
        -------
        Tensor
            Log mass on ``[value,inf)``.
        """
        return self.log_interval_mass(value, None)

    def cdf(self, value: Tensor) -> Tensor:
        """Read raw lower-half-line mass from the same interval operation.

        Parameters
        ----------
        value : Tensor
            Scalar cutoffs, broadcast against the measure batch.

        Returns
        -------
        Tensor
            CDF for a normalized measure; raw cumulative mass otherwise.
        """
        return self.log_cdf(value).exp()


class PointMass(Measure):
    """Dirac measure at a deterministic value.

    The unit $\\eta$ of the Giry monad: `PointMass(x)` is the
    measure that places mass one on the singleton $\\{x\\}$.
    `log_prob(y)` is zero when `y == value` and $-\\infty$
    otherwise.

    In a [`Mixture`][quivers.continuous.measure.Mixture] alongside
    a non-degenerate component, `PointMass` is the structural-zero
    spike that yields zero-inflation and hurdle factorisations.

    Parameters
    ----------
    value : Tensor or float
        Retained location of the atom.
    validate_args : bool or None
        Optional Torch argument-validation policy."""

    arg_constraints: dict = {}
    has_rsample = False

    _atomicity = "discrete"

    def atom_reference(self, value: Tensor) -> Tensor:
        """Identify the retained Dirac carrier independently of scalar dtype.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Whether each query equals the retained atom.
        """
        point = torch.as_tensor(value, dtype=torch.float64, device=self.value.device)
        return point == self.value.double()

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Read singleton mass under the same Dirac carrier.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Zero at the atom and negative infinity elsewhere.
        """
        carrier = self.atom_reference(value)
        return torch.where(
            carrier, torch.zeros_like(carrier, dtype=_reference(self).dtype), -torch.inf
        )

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
        """Constraint describing valid response values.

        Returns
        -------
        _constraints.Constraint
            Constraint describing valid response values."""
        return _constraints.real

    @property
    def mean(self) -> Tensor:
        """Expected response under the probability law.

        Returns
        -------
        Tensor
            Expected response under the probability law."""
        return self.value

    @property
    def variance(self) -> Tensor:
        """Response variance under the probability law.

        Returns
        -------
        Tensor
            Response variance under the probability law."""
        return torch.zeros_like(self.value)

    def log_normalizer(self) -> Tensor:
        """Log of the raw total mass, one value per batch entry.

        Returns
        -------
        Tensor
            Log of the raw total mass, one value per batch entry."""
        return torch.zeros_like(_reference(self))

    def expand(self, batch_shape, _instance=None):
        """Broadcast the retained atom over a new batch shape.

        Parameters
        ----------
        batch_shape : object
            Requested broadcast batch shape.
        _instance : object
            Unused Torch expansion instance.

        Returns
        -------
        object
            Distribution with the requested batch shape and retained construction."""
        return PointMass(self.value.expand(batch_shape))

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Keep the atom exactly when both interval bounds admit its value.

        Parameters
        ----------
        low, high : Tensor or float or None
            Optional event bounds.
        lower_closed, upper_closed : bool or Tensor
            Endpoint membership.

        Returns
        -------
        Tensor
            Zero at included atoms and negative infinity elsewhere.

        Raises
        ------
        ValueError
            If an endpoint is NaN.
        """
        low = -torch.inf if low is None else low
        high = torch.inf if high is None else high
        low, high = torch.broadcast_tensors(
            torch.as_tensor(low, device=self.value.device, dtype=torch.float64),
            torch.as_tensor(high, device=self.value.device, dtype=torch.float64),
        )
        demand(
            ~(low.isnan() | high.isnan()),
            lambda: ValueError("point-mass interval endpoints must not be NaN"),
        )
        lower_closed = torch.as_tensor(
            _closed_flag(lower_closed, "lower_closed"), device=self.value.device
        )
        upper_closed = torch.as_tensor(
            _closed_flag(upper_closed, "upper_closed"), device=self.value.device
        )
        atom = self.value.double()
        inside = ((atom > low) | ((atom == low) & lower_closed)) & (
            (atom < high) | ((atom == high) & upper_closed)
        )
        return torch.where(
            inside, torch.zeros_like(inside, dtype=_reference(self).dtype), -torch.inf
        )

    def icdf(self, value: Tensor) -> Tensor:
        """Return the atom at every valid probability.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Response quantiles at the requested probabilities.

        Raises
        ------
        ValueError
            If a probability is NaN or lies outside [0,1]."""
        point = torch.as_tensor(value, device=self.value.device)
        if not bool(((point >= 0) & (point <= 1)).all()):
            raise ValueError("point-mass quantiles require probabilities in [0,1]")
        return self.value.expand(torch.broadcast_shapes(point.shape, self.batch_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        """The log mass at a point: zero at the atom, ``-inf`` elsewhere.

        The result takes the floating dtype of ``value`` (the default
        dtype for an integer or Boolean value), so a double-precision
        mixture is not rounded to single precision by its atom's term.

        Parameters
        ----------
        value : Tensor
            The points.

        Returns
        -------
        Tensor
            The log masses.
        """
        value_b = torch.as_tensor(value)
        match = value_b == self.value
        if match.dtype != torch.bool:
            match = match.bool()
        dtype = (
            value_b.dtype if value_b.is_floating_point() else torch.get_default_dtype()
        )
        out = torch.full(
            torch.broadcast_shapes(value_b.shape, self.value.shape),
            _NEG_INF,
            dtype=dtype,
            device=match.device,
        )
        out = torch.where(match, torch.zeros_like(out), out)
        return out

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draws from the normalized law with leading sample dimensions.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Draws from the normalized law with leading sample dimensions."""
        shape = torch.Size(sample_shape) + self.batch_shape
        return self.value.expand(shape) if shape else self.value


def _normal_interval(
    base: torch.distributions.Normal, low: Tensor | None, high: Tensor | None
) -> TruncatedNormal:
    """Construct the normal interval distribution for a restriction.

    Parameters
    ----------
    base : torch.distributions.Normal
        The untruncated distribution.
    low : Tensor or None
        Lower bound, or an unbounded left endpoint.
    high : Tensor or None
        Upper bound, or an unbounded right endpoint.

    Returns
    -------
    TruncatedNormal
        The normalized restriction with the base's dtype and device.
    """
    lower = base.loc.new_tensor(-torch.inf) if low is None else low.to(base.loc)
    upper = base.loc.new_tensor(torch.inf) if high is None else high.to(base.loc)
    return TruncatedNormal(base.loc, base.scale, lower, upper)


def _intersection_bound(first, second, first_closed, second_closed, *, lower):
    """Intersect one lower or upper endpoint, retaining atoms at equal bounds.

    Parameters
    ----------
    first : object
        First optional endpoint.
    second : object
        Second optional endpoint.
    first_closed : object
        Membership policy of the first endpoint.
    second_closed : object
        Membership policy of the second endpoint.
    lower : object
        Whether the intersection concerns lower bounds.

    Returns
    -------
    object
        Intersection endpoint and its membership policy."""
    if first is None:
        return second, second_closed
    if second is None:
        return first, first_closed
    first, second = torch.broadcast_tensors(first, second)
    left = torch.as_tensor(first_closed, device=first.device)
    right = torch.as_tensor(second_closed, device=first.device)
    select_first = first > second if lower else first < second
    value = torch.maximum(first, second) if lower else torch.minimum(first, second)
    closed = torch.where(
        first == second, left & right, torch.where(select_first, left, right)
    )
    return value, closed


class _RestrictedSupport(_constraints.Constraint):
    """Intersect a base support with exact interval endpoint membership.

    Parameters
    ----------
    restriction : Restrict
        Retained base law, bounds, and endpoint policies.
    """

    def __init__(self, restriction):
        self.restriction = restriction
        self.base = restriction.base.support
        self.is_discrete = self.base.is_discrete
        self.event_dim = self.base.event_dim
        lower = restriction.low
        upper = restriction.high
        base_lower = getattr(self.base, "lower_bound", None)
        base_upper = getattr(self.base, "upper_bound", None)
        if base_lower is not None:
            lower = (
                base_lower
                if lower is None
                else torch.maximum(
                    lower, torch.as_tensor(base_lower, device=lower.device)
                )
            )
        if base_upper is not None:
            upper = (
                base_upper
                if upper is None
                else torch.minimum(
                    upper, torch.as_tensor(base_upper, device=upper.device)
                )
            )
        self.lower_bound = -math.inf if lower is None else lower
        self.upper_bound = math.inf if upper is None else upper
        self.bounds = (
            _constraints.interval(lower, upper)
            if lower is not None and upper is not None
            else _constraints.greater_than(lower)
            if lower is not None
            else _constraints.less_than(upper)
            if upper is not None
            else self.base
        )

    def check(self, value: Tensor) -> Tensor:
        """Require both base support and the requested endpoint memberships.

        Parameters
        ----------
        value : Tensor
            Candidate scalar or event values.

        Returns
        -------
        Tensor
            Validity mask with intrinsic event dimensions reduced.
        """
        inside = self.restriction._inside(value)
        if self.event_dim:
            inside = inside.all(tuple(range(inside.ndim - self.event_dim, inside.ndim)))
        return self.base.check(value) & inside


class _ExtendedIntervalTransform(Transform):
    """Use the ordinary real, half-line, or interval map independently per row.

    Parameters
    ----------
    support : _RestrictedSupport
        Scalar support whose bounds may be finite or infinite in different rows.
    """

    domain = _constraints.real
    bijective = True

    def __init__(self, support):
        super().__init__()
        self.codomain = support
        self.low = support.lower_bound
        self.high = support.upper_bound

    def _parts(self, value):
        """Select finite bounds before any inactive interval arithmetic.

        Parameters
        ----------
        value : Tensor
            Input carrying the transform device and floating dtype.

        Returns
        -------
        tuple[Tensor, ...]
            Safe lower bound, upper bound, width, and finite-side masks.
        """
        low = torch.as_tensor(self.low, device=value.device)
        high = torch.as_tensor(self.high, device=value.device)
        left, right = torch.isfinite(low), torch.isfinite(high)
        a, b = torch.where(left, low, 0.0), torch.where(right, high, 1.0)
        width = torch.where(left & right, b - a, 1.0)
        return a, b, width, left, right

    @property
    def sign(self):
        """Retain the decreasing orientation of upper-only half-lines.

        Returns
        -------
        Tensor
            Per-row orientation, negative only for an upper-only bound.
        """
        low, high = torch.as_tensor(self.low), torch.as_tensor(self.high)
        return torch.where(~torch.isfinite(low) & torch.isfinite(high), -1, 1)

    def _call(self, value):
        """Map real coordinates into each row's open support interior.

        Parameters
        ----------
        value : Tensor
            Unconstrained coordinates.

        Returns
        -------
        Tensor
            Constrained coordinates under the row's ordinary interval map.
        """
        a, b, width, left, right = self._parts(value)
        bounded = a + width * SigmoidTransform()(value)
        exponential = torch.exp(torch.where(left ^ right, value, 0.0))
        return torch.where(
            left & right,
            bounded,
            torch.where(
                left, a + exponential, torch.where(right, b - exponential, value)
            ),
        )

    def _inverse(self, value):
        """Invert each ordinary map without evaluating inactive logarithms.

        Parameters
        ----------
        value : Tensor
            Constrained coordinates.

        Returns
        -------
        Tensor
            Corresponding unconstrained coordinates.
        """
        a, b, width, left, right = self._parts(value)
        fraction = torch.where(left & right, (value - a) / width, 0.5)
        bounded = SigmoidTransform().inv(fraction)
        lower = torch.where(left & ~right, value - a, 1.0).log()
        upper = torch.where(right & ~left, b - value, 1.0).log()
        return torch.where(
            left & right,
            bounded,
            torch.where(left, lower, torch.where(right, upper, value)),
        )

    def log_abs_det_jacobian(self, x, y):
        """Read the same interval and exponential Jacobians used by Torch.

        Parameters
        ----------
        x : Tensor
            Unconstrained coordinates.
        y : Tensor
            Corresponding constrained coordinates.

        Returns
        -------
        Tensor
            Log absolute derivative for each scalar coordinate.
        """
        _, _, width, left, right = self._parts(x)
        bounded = (
            width.log()
            - torch.nn.functional.softplus(-x)
            - torch.nn.functional.softplus(x)
        )
        return torch.where(left & right, bounded, torch.where(left | right, x, 0.0))


@biject_to.register(_RestrictedSupport)
def _biject_restricted(support):
    """Choose the ordinary interval map independently for each support row.

    Parameters
    ----------
    support : _RestrictedSupport
        Exact support with a scalar bounding interval.

    Returns
    -------
    Transform
        Identity, exponential, reflected exponential, or logistic interval map.

    Raises
    ------
    NotImplementedError
        If the support is discrete or has coupled event constraints.
    """
    if support.is_discrete or support.event_dim:
        raise NotImplementedError(
            "restricted support has no continuous scalar interval bijection"
        )
    return _ExtendedIntervalTransform(support)


@transform_to.register(_RestrictedSupport)
def _transform_restricted(support):
    """Use the same bounded coordinate map for optimizer transforms.

    Parameters
    ----------
    support : _RestrictedSupport
        Retained restricted support.

    Returns
    -------
    Transform
        The existing scalar interval transform.
    """
    return _biject_restricted(support)


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

    Normal interval masses use log-CDF differences with tail reflection.
    Other continuous bases use their CDF difference, and discrete bases
    use interval probabilities that include both endpoints.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    low, high : Tensor or float or None
        Optional interval endpoints; omitted sides are unbounded.
    validate_args : bool or None
        Optional Torch argument-validation policy.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership, closed by default; Boolean tensors may vary by batch row."""

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        low: Tensor | float | None = None,
        high: Tensor | float | None = None,
        validate_args: bool | None = None,
        *,
        lower_closed: bool | Tensor = True,
        upper_closed: bool | Tensor = True,
    ) -> None:
        if low is None and high is None:
            raise ValueError(
                "Restrict: at least one of `low` or `high` must be "
                "supplied; an unrestricted base is just the base "
                "distribution itself."
            )
        self.lower_closed = _closed_flag(lower_closed, "lower_closed")
        self.upper_closed = _closed_flag(upper_closed, "upper_closed")
        self.base = base
        try:
            reference = _reference(base)
        except NotImplementedError:
            reference = torch.tensor(0.0)
        self.low = (
            torch.as_tensor(low, dtype=torch.float64, device=reference.device)
            if low is not None
            else None
        )
        self.high = (
            torch.as_tensor(high, dtype=torch.float64, device=reference.device)
            if high is not None
            else None
        )
        if not base.event_shape:
            shape = torch.broadcast_shapes(
                base.batch_shape,
                *(
                    value.shape
                    for value in (
                        self.low,
                        self.high,
                        self.lower_closed,
                        self.upper_closed,
                    )
                    if isinstance(value, Tensor)
                ),
            )
            if shape != base.batch_shape:
                self.base = base.expand(shape)
        super().__init__(
            batch_shape=self.base.batch_shape,
            event_shape=self.base.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        """Intersect the base support with the exact interval event.

        Returns
        -------
        Constraint
            Includes closed endpoint atoms and excludes open endpoints.
        """
        return _RestrictedSupport(self)

    def log_normalizer(self) -> Tensor:
        # All base families share endpoint-aware event masses. Internal
        # tail arithmetic remains wide; the public normalizer retains
        # the base's dtype and device.
        """Log of the raw total mass, one value per batch entry.

        Returns
        -------
        Tensor
            Log of the raw total mass, one value per batch entry."""
        try:
            return log_interval_mass(
                self.base,
                self.low,
                self.high,
                lower_closed=self.lower_closed,
                upper_closed=self.upper_closed,
            ).to(_reference(self.base))
        except NotImplementedError:
            symmetric = _symmetric_fold_log_normalizer(
                self.base,
                self.low,
                self.high,
            )
            if symmetric is not None:
                return symmetric
            raise

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Query the raw mass of an intersection without normalizing twice.

        Parameters
        ----------
        low : object
            Lower event bound; None leaves that side unbounded.
        high : object
            Upper event bound; None leaves that side unbounded.
        lower_closed : object
            Whether the lower endpoint belongs to the event.
        upper_closed : object
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        object
            Raw log mass on the requested interval, retaining independent batch axes."""
        reference = _reference(self.base)
        low = (
            None
            if low is None
            else torch.as_tensor(low, dtype=torch.float64, device=reference.device)
        )
        high = (
            None
            if high is None
            else torch.as_tensor(high, dtype=torch.float64, device=reference.device)
        )
        low, lower_closed = _intersection_bound(
            self.low, low, self.lower_closed, lower_closed, lower=True
        )
        high, upper_closed = _intersection_bound(
            self.high, high, self.upper_closed, upper_closed, lower=False
        )
        return log_interval_mass(
            self.base, low, high, lower_closed=lower_closed, upper_closed=upper_closed
        )

    def expand(self, batch_shape, _instance=None):
        """Broadcast the retained restriction without changing endpoint semantics.

        Parameters
        ----------
        batch_shape : object
            Requested broadcast batch shape.
        _instance : object
            Unused Torch expansion instance.

        Returns
        -------
        object
            Distribution with the requested batch shape and retained construction."""
        return Restrict(
            self.base.expand(batch_shape),
            self.low,
            self.high,
            lower_closed=self.lower_closed,
            upper_closed=self.upper_closed,
        )

    @property
    def _atomicity(self) -> str:
        """Retain the base law's atomicity proof.

        Returns
        -------
        str
            Atomless, discrete, or mixed classification inherited from the component measures.
        """
        from quivers.continuous._atoms import atomicity

        return atomicity(self.base)

    def atom_reference(self, value: Tensor) -> Tensor:
        """Preserve the base carrier even when restriction removes its mass.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Membership in the fixed base atomic carrier.
        """
        from quivers.continuous._atoms import atom_reference

        return atom_reference(self.base, value)

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Restrict base singleton masses using the retained endpoint policy.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Raw singleton log mass inside the restriction.
        """
        from quivers.continuous._atoms import log_atom_mass

        return torch.where(
            self._inside(value), log_atom_mass(self.base, value), -torch.inf
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Raw log density or mass at the evaluation points.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Raw log density or mass at the evaluation points."""
        from quivers.continuous._atoms import _point, safe_log_prob

        point = _point(self.base, value, reject_nan=False)
        inside = self._inside(point)
        if self.event_shape:
            inside = inside.all(
                tuple(range(inside.ndim - len(self.event_shape), inside.ndim))
            )
        return safe_log_prob(self.base, point, active=inside)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw from the restricted, renormalised measure.

        A base with a closed-form CDF and inverse CDF is drawn by
        inversion: a uniform draw between the CDF at the two ends of
        the interval, mapped back through the inverse CDF, which is
        exactly the base conditioned on the interval. Any other base is
        drawn by rejection one entry at a time: each entry outside the
        interval is redrawn from the base until it falls inside, so
        every entry is an independent draw of the conditioned base.

        Parameters
        ----------
        sample_shape : torch.Size
            The leading sample shape.

        Returns
        -------
        Tensor
            Draws of shape ``sample_shape + batch_shape + event_shape``,
            each inside the interval.

        Raises
        ------
        ValueError
            If the restricted measure has zero or nonfinite total mass.
        RuntimeError
            If rejection leaves an entry outside the interval after
            `_REJECTION_ROUNDS` redraws, which happens only when the
            interval holds a vanishing share of the base's mass.
        """
        return self._sample_active(
            sample_shape,
            torch.ones(
                self.batch_shape, dtype=torch.bool, device=_reference(self.base).device
            ),
        )

    def _sample_active(self, sample_shape: torch.Size, active: Tensor) -> Tensor:
        """Sample retained batch entries without requiring unused entries to have mass.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.
        active : Tensor
            Boolean batch mask selecting entries whose draws are required.

        Returns
        -------
        Tensor
            Normalized draws; unselected batch entries carry unused placeholders.

        Raises
        ------
        ValueError
            If an active batch entry has zero or nonfinite mass."""
        with torch.no_grad():
            mass = self.log_normalizer()
            if not bool((~active | torch.isfinite(mass)).all()):
                raise ValueError(
                    "Restrict sampling requires finite positive mass on every active batch entry"
                )
            inverted = self._sample_by_inversion(sample_shape, active)
            if inverted is not None:
                return inverted
            return self._sample_by_rejection(sample_shape, active)

    def _inside(self, draws: Tensor) -> Tensor:
        """Which entries of ``draws`` lie in the interval.

        Parameters
        ----------
        draws : Tensor
            Candidate draws.

        Returns
        -------
        Tensor
            A boolean mask of the draws' shape.
        """
        ok = torch.ones_like(draws, dtype=torch.bool)
        if self.low is not None:
            above = (draws > self.low) | (
                (draws == self.low)
                & torch.as_tensor(self.lower_closed, device=draws.device)
            )
            ok = ok & above
        if self.high is not None:
            below = (draws < self.high) | (
                (draws == self.high)
                & torch.as_tensor(self.upper_closed, device=draws.device)
            )
            ok = ok & below
        return ok

    def _sample_by_inversion(
        self, sample_shape: torch.Size, active: Tensor
    ) -> Tensor | None:
        """Draw by inverting the base's CDF over the interval.

        Parameters
        ----------
        sample_shape : torch.Size
            The leading sample shape.

        Returns
        -------
        Tensor | None
            The draws, or ``None`` when the base has no CDF. A continuous
            base without an inverse CDF, and a discrete base, are inverted
            by bisection on the log CDF; a discrete draw is the smallest
            count whose CDF reaches the uniform variate.
        """
        if isinstance(self.base, torch.distributions.Normal):
            low = self.low if self.low is not None else -torch.inf
            high = self.high if self.high is not None else torch.inf
            return _normal_interval(
                self.base,
                torch.where(active, low, -torch.inf),
                torch.where(active, high, torch.inf),
            ).sample(sample_shape)
        discrete = bool(getattr(self.base.support, "is_discrete", False))
        # The CDF values and the inversion run in double precision: an
        # interval deep in a tail has CDF values within float32 rounding
        # of one another, which single precision would collapse.
        wide = torch.float64
        try:
            reference = _reference(self.base)
            zero = torch.zeros(self.batch_shape, dtype=wide, device=reference.device)
            # The draw is made in log CDF coordinates, so an interval whose
            # CDF values underflow (a far tail) still carries a uniform draw.
            log_lower = (
                zero - torch.inf
                if self.low is None
                else _endpoint_tails(
                    self.base,
                    self.low.to(device=reference.device, dtype=wide),
                    self.lower_closed,
                )[0]
            )
            log_upper = (
                zero
                if self.high is None
                else _endpoint_tails(
                    self.base,
                    self.high.to(device=reference.device, dtype=wide),
                    not self.upper_closed
                    if isinstance(self.upper_closed, bool)
                    else ~self.upper_closed,
                )[0]
            )
            log_lower = torch.where(active, log_lower, -torch.inf)
            log_upper = torch.where(active, log_upper, 0.0)
            if not bool((torch.isfinite(log_upper) & (log_upper > log_lower)).all()):
                return None
            # Bisection costs one log-CDF evaluation per halving. A continuous
            # base without an inverse CDF is left to rejection when every
            # active row's interval holds at least a twentieth of the mass;
            # lighter intervals are inverted by bisection. A discrete base is
            # always inverted, which needs about 64 halvings, so its draws
            # keep one carrier whatever the interval mass.
            light = bool(
                (
                    active
                    & (
                        log_upper + _log_complement(log_lower - log_upper)
                        < math.log(0.05)
                    )
                ).any()
            )
            shape = self._extended_shape(sample_shape)
            uniform = torch.rand(shape, dtype=wide, device=reference.device)
            log_probability = log_upper + torch.log(
                uniform + (1 - uniform) * (log_lower - log_upper).exp()
            )
            probability = log_probability.exp()
            base = self.base
            low = (
                torch.full_like(probability, -torch.inf)
                if self.low is None
                else self.low.to(probability).expand(shape)
            )
            high = (
                torch.full_like(probability, torch.inf)
                if self.high is None
                else self.high.to(probability).expand(shape)
            )
            if not light and not discrete and type(self.base).icdf is Distribution.icdf:
                return None
            if discrete:
                # The smallest count whose CDF reaches the uniform draw: the
                # bisection point lies within float64 resolution above it.
                draws = invert_log_cdf(
                    lambda point: _log_tails(base, point)[0],
                    log_probability,
                    low.floor() - 1,
                    high,
                    iterations=80,
                ).floor()
            else:
                # The closed-form inverse CDF needs representable probabilities.
                closed_form = None
                if bool(((probability > 0) & (probability < 1)).all()):
                    try:
                        closed_form = self.base.icdf(probability)
                    except NotImplementedError:
                        closed_form = None
                draws = (
                    closed_form
                    if closed_form is not None
                    else invert_log_cdf(
                        lambda point: _log_tails(base, point)[0],
                        log_probability,
                        low,
                        high,
                    )
                )
        except NotImplementedError:
            return None
        if not bool((~active | (torch.isfinite(draws) & self._inside(draws))).all()):
            return None
        # Counts stay in float64, which represents every count below 2**53.
        return draws if discrete else draws.to(reference)

    def _sample_by_rejection(self, sample_shape: torch.Size, active: Tensor) -> Tensor:
        """Draw by redrawing each entry outside the interval.

        Parameters
        ----------
        sample_shape : torch.Size
            The leading sample shape.

        Returns
        -------
        Tensor
            The draws, each inside the interval.

        Raises
        ------
        RuntimeError
            If an entry stays outside after `_REJECTION_ROUNDS` redraws.
        """
        return _bounded_rejection(
            lambda: _sample_active(self.base, sample_shape, active),
            lambda values: self._inside(values) | ~active,
            max_rounds=_REJECTION_ROUNDS,
            label="Restrict",
        )


#: How many times rejection sampling redraws an entry outside a
#: restriction's interval before it reports the interval as too small.
_REJECTION_ROUNDS = 10_000


def _bounded_rejection(
    draw: Callable[[], Tensor],
    accept: Callable[[Tensor], Tensor],
    *,
    max_rounds: int,
    label: str,
) -> Tensor:
    """Fill independent coordinates while preserving the supplied draw's AD policy.

    Parameters
    ----------
    draw : Callable
        Fresh draws with the same sample, batch, and event shape on every call.
    accept : Callable
        Boolean acceptance mask broadcastable to those draws.
    max_rounds : int
        Maximum replacement draws following the initial candidate.
    label : str
        Operator name included in exhaustion diagnostics.

    Returns
    -------
    Tensor
        Accepted coordinates in their original shape.

    Raises
    ------
    RuntimeError
        If the budget expires before every coordinate is accepted.
    """
    values = draw()
    accepted = accept(values)
    for _ in range(max_rounds):
        if bool(accepted.all()):
            return values
        values = torch.where(accepted, values, draw())
        accepted = accept(values)
    if bool(accepted.all()):
        return values
    raise RuntimeError(
        f"{label} rejection sampling failed to fill {int((~accepted).sum())} "
        f"coordinates after {max_rounds} redraws; the interval holds too little "
        "of the base's mass to sample by rejection"
    )


class Pushforward(Measure):
    """Pushforward of a base distribution through a bijector.

    Implements $f_*\\mu$, the measure whose log-density at `y` is
    `base.log_prob(b.inverse(y)) + b.inverse_log_det_jacobian(y)`.
    Functorial in `b`: composing bijectors composes the pushforward.

    Does not commute with [`Restrict`][quivers.continuous.measure.Restrict]
    in general; the compiler's rewrite pass handles the cases that
    do (monotone bijectors with interval restrictions).

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    bijector : Bijector
        Invertible measurable coordinate transformation.
    validate_args : bool or None
        Optional Torch argument-validation policy."""

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
            event_shape=bijector.forward_shape(base.event_shape),
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        """Constraint describing valid response values.

        Returns
        -------
        _constraints.Constraint
            Constraint describing valid response values."""
        return _constraints.real

    def log_normalizer(self) -> Tensor:
        """Preserve the base measure's raw mass under the bijection.

        Returns
        -------
        Tensor
            Log total mass on the retained batch axes.
        """
        return _native_log_mass(self.base, batch_axes=len(self.base.batch_shape))

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Pull an interval back through the bijector's scalar capability.

        Parameters
        ----------
        low : object
            Lower event bound; None leaves that side unbounded.
        high : object
            Upper event bound; None leaves that side unbounded.
        lower_closed : object
            Whether the lower endpoint belongs to the event.
        upper_closed : object
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        object
            Raw log mass on the requested interval, retaining independent batch axes."""
        reference = _reference(self.base)
        low = torch.as_tensor(
            -torch.inf if low is None else low,
            dtype=torch.float64,
            device=reference.device,
        )
        high = torch.as_tensor(
            torch.inf if high is None else high,
            dtype=torch.float64,
            device=reference.device,
        )
        low, high = self.bijector._interval_preimage(low, high)
        return log_interval_mass(
            self.base, low, high, lower_closed=lower_closed, upper_closed=upper_closed
        )

    @property
    def _atomicity(self) -> str:
        """Retain the base law's atomicity proof.

        Returns
        -------
        str
            Atomless, discrete, or mixed classification inherited from the component measures.
        """
        from quivers.continuous._atoms import atomicity

        return atomicity(self.base)

    def atom_reference(self, value: Tensor) -> Tensor:
        """Pull the atomic carrier back through the retained bijection.

        Parameters
        ----------
        value : Tensor
            Query points in the image space.

        Returns
        -------
        Tensor
            Membership in the image of the base carrier.
        """
        from quivers.continuous._atoms import atom_reference

        return atom_reference(self.base, self.bijector.inverse(value))

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Transport singleton masses without a density Jacobian.

        Parameters
        ----------
        value : Tensor
            Query points in the image space.

        Returns
        -------
        Tensor
            Raw singleton log mass in the image space.
        """
        from quivers.continuous._atoms import log_atom_mass

        return log_atom_mass(self.base, self.bijector.inverse(value))

    def log_prob(self, value: Tensor) -> Tensor:
        """Raw log density or mass at the evaluation points.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Raw log density or mass at the evaluation points.

        Raises
        ------
        NotImplementedError
            If a mixed vector law requires an undeclared stratified reference.
        """
        from quivers.continuous._atoms import (
            atom_reference,
            log_atom_mass,
            safe_log_prob,
        )

        x = self.bijector.inverse(value)
        if self.event_shape:
            if self._atomicity == "discrete":
                return self.base.log_prob(x)
            if self._atomicity == "mixed":
                raise NotImplementedError(
                    "mixed vector transport requires a stratified reference measure"
                )
            determinant = self.bijector.inverse_log_det_jacobian(value)
            dims = len(self.base.event_shape) - self.bijector.jacobian_event_dim
            if dims:
                determinant = determinant.sum(tuple(range(-dims, 0)))
            return self.base.log_prob(x) + determinant
        atomic = atom_reference(self.base, x)
        base_lp = safe_log_prob(self.base, x, active=~atomic)
        log_det = self.bijector.inverse_log_det_jacobian(value)
        return torch.where(atomic, log_atom_mass(self.base, x), base_lp + log_det)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draws from the normalized law with leading sample dimensions.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Draws from the normalized law with leading sample dimensions."""
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

    Parameters
    ----------
    weights : Tensor
        Nonnegative raw component weights on the last axis.
    components : list[Distribution]
        Laws with the same event shape and broadcastable batches.
    validate_args : bool or None
        Optional Torch argument-validation policy."""

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
        if not weights.ndim:
            raise ValueError(
                "Mixture weights must be finite nonnegative values with a component axis"
            )
        demand(
            torch.isfinite(weights) & (weights >= 0),
            lambda: ValueError(
                "Mixture weights must be finite nonnegative values with a component axis"
            ),
        )
        if any(
            component.event_shape != components[0].event_shape
            for component in components[1:]
        ):
            raise ValueError("Mixture components must have the same event shape")
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
        """Number of retained mixture components.

        Returns
        -------
        int
            Number of retained mixture components."""
        return self._num_components

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        """Constraint describing valid response values.

        Returns
        -------
        _constraints.Constraint
            Constraint describing valid response values."""
        return _constraints.real

    def _log_weights(self) -> Tensor:
        """The log of each mixing weight, ``-inf`` for a zero weight.

        A zero weight drops its component from the mixture exactly. The
        logarithm is taken of a weight replaced by one where it is zero
        and then masked, so the gradient through a zero weight is zero
        rather than the ``0 * inf`` of differentiating ``log`` there.

        Returns
        -------
        Tensor
            The log weights, shaped as the weights.
        """
        positive = self.weights > 0
        safe = torch.where(positive, self.weights, torch.ones_like(self.weights))
        return torch.where(
            positive, torch.log(safe), torch.full_like(self.weights, -torch.inf)
        )

    def _component_logits(self) -> Tensor:
        """Combine weights and raw component masses for scoring and selection.

        Returns
        -------
        Tensor
            Log weight times raw component mass, stacked along the component axis."""
        terms = [
            weight + _native_log_mass(component, batch_axes=len(component.batch_shape))
            for weight, component in zip(
                self._log_weights().unbind(-1), self.components, strict=True
            )
        ]
        return torch.stack(torch.broadcast_tensors(*terms), dim=-1)

    def _weighted_score(self, scores: list[Tensor]) -> Tensor:
        """Sum component scores while retaining zero-weight boundary derivatives.

        Parameters
        ----------
        scores : list[Tensor]
            Raw component log scores with broadcastable batch shapes.

        Returns
        -------
        Tensor
            Logarithm of the raw weighted sum.
        """
        from quivers.continuous._weighted_logsumexp import weighted_logsumexp

        return weighted_logsumexp(
            self.weights, torch.stack(torch.broadcast_tensors(*scores), -1)
        )

    def log_normalizer(self) -> Tensor:
        """Read raw total mass using the common weighted measure sum.

        Returns
        -------
        Tensor
            Log mass on each mixture batch entry.
        """
        return self._weighted_score(
            [
                _native_log_mass(component, batch_axes=len(component.batch_shape))
                for component in self.components
            ]
        )

    # mixtures: moments of the normalized mixture from component moments.
    def _selection_probabilities(self) -> Tensor:
        """Normalized probabilities of selecting each component.

        Returns
        -------
        Tensor
            ``w_k Z_k / sum_j w_j Z_j`` on the batch and component axes,
            where ``Z_k`` is the raw mass of component ``k``.
        """
        return torch.softmax(self._component_logits(), dim=-1)

    @property
    def component_probabilities(self) -> Tensor:
        """Normalized probabilities of the components.

        Returns
        -------
        Tensor
            ``w_k Z_k / sum_j w_j Z_j`` on the batch and component axes,
            where ``Z_k`` is the raw mass of component ``k``.
        """
        return self._selection_probabilities()

    @property
    def mean(self) -> Tensor:
        """Mean of the normalized mixture, ``sum_k p_k E_k[Y]``.

        Returns
        -------
        Tensor
            The mean on the batch and event axes.

        Raises
        ------
        NotImplementedError
            If a component has no analytic mean.
        """
        probabilities = self._selection_probabilities()
        means = torch.stack(
            torch.broadcast_tensors(*(component.mean for component in self.components)),
            dim=-1,
        )
        for _ in self.event_shape:
            probabilities = probabilities.unsqueeze(-2)
        return (probabilities * means).sum(-1)

    @property
    def variance(self) -> Tensor:
        """Variance of the normalized mixture by the law of total variance.

        Returns
        -------
        Tensor
            ``sum_k p_k (Var_k + E_k^2) - E^2`` on the batch and event axes.

        Raises
        ------
        NotImplementedError
            If a component has no analytic mean or variance.
        """
        probabilities = self._selection_probabilities()
        means = torch.stack(
            torch.broadcast_tensors(*(component.mean for component in self.components)),
            dim=-1,
        )
        variances = torch.stack(
            torch.broadcast_tensors(
                *(component.variance for component in self.components)
            ),
            dim=-1,
        )
        for _ in self.event_shape:
            probabilities = probabilities.unsqueeze(-2)
        second = (probabilities * (variances + means.square())).sum(-1)
        return second - self.mean.square()

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Add raw component event masses, including any point atoms.

        Parameters
        ----------
        low : object
            Lower event bound; None leaves that side unbounded.
        high : object
            Upper event bound; None leaves that side unbounded.
        lower_closed : object
            Whether the lower endpoint belongs to the event.
        upper_closed : object
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        object
            Raw log mass on the requested interval, retaining independent batch axes."""
        return self._weighted_score(
            [
                log_interval_mass(
                    component,
                    low,
                    high,
                    lower_closed=lower_closed,
                    upper_closed=upper_closed,
                )
                for component in self.components
            ]
        )

    def expand(self, batch_shape, _instance=None):
        """Broadcast the common mixture batch and its component weights.

        Parameters
        ----------
        batch_shape : object
            Requested broadcast batch shape.
        _instance : object
            Unused Torch expansion instance.

        Returns
        -------
        object
            Distribution with the requested batch shape and retained construction."""
        return Mixture(
            self.weights.expand(*batch_shape, self._num_components),
            [component.expand(batch_shape) for component in self.components],
        )

    @property
    def _atomicity(self) -> str:
        """Prove homogeneous component carriers or retain a mixed law.

        Returns
        -------
        str
            Atomless, discrete, or mixed classification inherited from the component measures.
        """
        from quivers.continuous._atoms import atomicity

        kinds = {atomicity(component) for component in self.components}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    def atom_reference(self, value: Tensor) -> Tensor:
        """Take the union of component carriers, including zero-weight components.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Membership in at least one component atomic carrier.
        """
        from quivers.continuous._atoms import atom_reference

        carriers = torch.broadcast_tensors(
            *(atom_reference(c, value) for c in self.components)
        )
        return torch.stack(carriers, -1).any(-1)

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Add weighted singleton masses under the common atomic carrier.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Weighted raw singleton log mass.
        """
        from quivers.continuous._atoms import log_atom_mass

        return self._weighted_score([log_atom_mass(c, value) for c in self.components])

    def log_prob(self, value: Tensor) -> Tensor:
        """Raw log density or mass at the evaluation points.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Raw log density or mass at the evaluation points.

        Raises
        ------
        NotImplementedError
            If mixed vector components lack a common stratified reference measure.
        """
        from quivers.continuous._atoms import safe_log_prob

        value = torch.as_tensor(value, device=self.weights.device)
        atomic = atom_score = None
        if self.event_shape:
            if self._atomicity == "mixed":
                raise NotImplementedError(
                    "mixed vector mixtures require a common stratified reference measure"
                )
            scores = [component.log_prob(value) for component in self.components]
        else:
            atomic = self.atom_reference(value)
            atom_score = self.log_atom_mass(value)
            scores = [
                safe_log_prob(component, value, active=~atomic)
                for component in self.components
            ]
        density = self._weighted_score(scores)
        if atomic is None:
            return density
        assert atom_score is not None
        return torch.where(atomic, atom_score, density)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draws from the normalized law with leading sample dimensions.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Draws from the normalized law with leading sample dimensions."""
        active = torch.ones(
            self.batch_shape, dtype=torch.bool, device=self.weights.device
        )
        return self._sample_active(sample_shape, active)

    def _sample_active(self, sample_shape: torch.Size, active: Tensor) -> Tensor:
        """Select components by their weighted masses before sampling active rows.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.
        active : Tensor
            Boolean batch mask selecting entries whose draws are required.

        Returns
        -------
        Tensor
            Normalized draws; unselected batch entries carry unused placeholders.

        Raises
        ------
        ValueError
            If an active batch entry has zero or nonfinite mass."""
        logits = self._component_logits()
        if not bool((~active | torch.isfinite(torch.logsumexp(logits, -1))).all()):
            raise ValueError(
                "Mixture sampling requires finite positive mass on every active batch entry"
            )
        logits = torch.where(active.unsqueeze(-1), logits, torch.zeros_like(logits))
        idx = torch.distributions.Categorical(logits=logits).sample(sample_shape)
        shape = torch.Size(sample_shape) + self.batch_shape + self.event_shape
        result = self.weights.new_zeros(shape)
        for index, component in enumerate(self.components):
            selected = (idx == index) & active
            selected_batch = (
                selected.any(tuple(range(len(sample_shape))))
                if sample_shape
                else selected
            )
            if not bool(selected_batch.any()):
                continue
            draws = _sample_active(
                component.expand(self.batch_shape)
                if component.batch_shape != self.batch_shape
                else component,
                sample_shape,
                selected_batch,
            )
            for _ in self.event_shape:
                selected = selected.unsqueeze(-1)
            result = torch.where(selected, draws, result)
        return result

    def flatten(self) -> Mixture:
        """Apply the Giry-monad-multiplication associativity rule:
        any nested `Mixture` component is replaced by its components
        with weights scaled by the outer weight. Returns a flat
        `Mixture` whose components are not themselves `Mixture`
        instances.

        Returns
        -------
        Mixture
            Equivalent mixture with nested component weights multiplied."""
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

        Parameters
        ----------
        bijector : object
            Measurable invertible coordinate transformation.

        Returns
        -------
        Mixture
            Mixture obtained by applying the bijector to each component."""
        new_components = [Pushforward(c, bijector) for c in self.components]
        return Mixture(self.weights, new_components)

    def restrict_to(
        self, low=None, high=None, *, lower_closed=True, upper_closed=True
    ) -> Mixture:
        """Condition the raw mixture on an endpoint-aware interval.

        Components retain their restricted raw measures. Dividing each
        original weight by the common event mass normalizes their sum;
        component selection then uses the same weighted masses as every
        other mixture. Zero-mass components need no separate normalization.

        Parameters
        ----------
        low, high : Tensor or float or None
            Optional event bounds.
        lower_closed, upper_closed : bool or Tensor
            Endpoint membership, closed by default.

        Returns
        -------
        Mixture
            A probability measure on the requested event. Its retained
            component weights multiply raw restricted measures.

        Raises
        ------
        ValueError
            If the event has zero or nonfinite mass on any batch entry.
        """
        components = [
            Restrict(
                component,
                low,
                high,
                lower_closed=lower_closed,
                upper_closed=upper_closed,
            )
            for component in self.components
        ]
        raw = Mixture(self.weights, components)
        mass = raw.log_normalizer()
        demand(
            torch.isfinite(mass),
            lambda: ValueError(
                "Mixture restriction requires finite positive event mass"
            ),
        )
        return Mixture(torch.exp(self._log_weights() - mass.unsqueeze(-1)), components)

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

        Returns
        -------
        Mixture
            Equivalent mixture with atomic components placed first."""
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

    Parameters
    ----------
    base : Distribution
        Independent coordinate laws or raw measures.
    reinterpreted_batch_ndims : int or Tensor
        Nonnegative number of trailing batch axes forming one event. Zero
        preserves the base law's batch and event axes. A tensor must be scalar
        and carry an integer value.
    validate_args : bool or None
        Optional Torch argument-validation policy."""

    arg_constraints: dict = {}

    def __init__(
        self,
        base: Distribution,
        reinterpreted_batch_ndims: int | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        count_value: object = reinterpreted_batch_ndims
        if isinstance(count_value, Tensor):
            if count_value.ndim != 0:
                raise TypeError(
                    "Independent: reinterpreted_batch_ndims must be a scalar integer"
                )
            count_value = count_value.item()
        count = independent_ndims(count_value, len(base.batch_shape))
        self.base = base
        self.reinterpreted_batch_ndims = count
        self._inner = torch.distributions.Independent(
            base,
            count,
        )
        super().__init__(
            batch_shape=self._inner.batch_shape,
            event_shape=self._inner.event_shape,
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:
        """Whether the base law supports a pathwise sample.

        Returns
        -------
        bool
            The base law's declared sampling capability.
        """
        return self.base.has_rsample

    @property
    def mean(self) -> Tensor:
        """Base means with the declared batch and event axes retained.

        Returns
        -------
        Tensor
            Coordinate means, when defined by the base law.
        """
        return self._inner.mean

    @property
    def variance(self) -> Tensor:
        """Base marginal variances with every coordinate axis retained.

        Returns
        -------
        Tensor
            Marginal coordinate variances, when defined by the base law.
        """
        return self._inner.variance

    def rsample(
        self, sample_shape: torch.Size | tuple[int, ...] | list[int] = torch.Size()
    ) -> Tensor:
        """Draw pathwise samples only when the base law supports them.

        Parameters
        ----------
        sample_shape : torch.Size or tuple of int or list of int
            Additional leading independent sample axes.

        Returns
        -------
        Tensor
            Base pathwise draws with the same coordinate layout.

        Raises
        ------
        NotImplementedError
            If the base law has no pathwise sampler.
        """
        if not self.has_rsample:
            raise NotImplementedError("Independent base law has no pathwise sampler")
        return self._inner.rsample(sample_shape)

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        """Constraint describing valid response values.

        Returns
        -------
        _constraints.Constraint
            Constraint describing valid response values."""
        return self._inner.support

    def log_normalizer(self) -> Tensor:
        """Reduce raw mass over the newly declared independent event axes.

        Returns
        -------
        Tensor
            Log total mass with only independent batch axes retained.
        """
        inner = _native_log_mass(self.base, batch_axes=len(self.base.batch_shape))
        if self.reinterpreted_batch_ndims == 0:
            return inner
        return inner.sum(
            tuple(range(inner.ndim - self.reinterpreted_batch_ndims, inner.ndim))
        )

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Multiply independent coordinate event masses by adding their logarithms.

        Parameters
        ----------
        low : object
            Lower event bound; None leaves that side unbounded.
        high : object
            Upper event bound; None leaves that side unbounded.
        lower_closed : object
            Whether the lower endpoint belongs to the event.
        upper_closed : object
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        object
            Raw log mass on the requested interval, retaining independent batch axes."""
        mass = log_interval_mass(
            self.base, low, high, lower_closed=lower_closed, upper_closed=upper_closed
        )
        if self.reinterpreted_batch_ndims == 0:
            return mass
        return mass.sum(
            tuple(range(mass.ndim - self.reinterpreted_batch_ndims, mass.ndim))
        )

    @property
    def _atomicity(self) -> str:
        """Retain the base law's atomicity proof.

        Returns
        -------
        str
            Atomless, discrete, or mixed classification inherited from the component measures.
        """
        from quivers.continuous._atoms import atomicity

        return atomicity(self.base)

    def atom_reference(self, value: Tensor) -> Tensor:
        """Identify singleton atoms jointly across reinterpreted event axes.

        Parameters
        ----------
        value : Tensor
            Queries with intrinsic event axes.

        Returns
        -------
        Tensor
            Whether every coordinate belongs to its base atomic carrier.
        """
        from quivers.continuous._atoms import atom_reference

        carrier = atom_reference(self.base, value)
        dims = self.reinterpreted_batch_ndims
        return carrier.all(tuple(range(-dims, 0))) if dims else carrier

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Multiply coordinate singleton masses across declared event axes.

        Parameters
        ----------
        value : Tensor
            Queries with intrinsic event axes.

        Returns
        -------
        Tensor
            Joint singleton log mass.
        """
        from quivers.continuous._atoms import log_atom_mass

        score = log_atom_mass(self.base, value)
        dims = self.reinterpreted_batch_ndims
        return score.sum(tuple(range(-dims, 0))) if dims else score

    def log_prob(self, value: Tensor) -> Tensor:
        """Raw log density or mass at the evaluation points.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Raw log density or mass at the evaluation points."""
        score = self.base.log_prob(value)
        dims = self.reinterpreted_batch_ndims
        return score.sum(tuple(range(-dims, 0))) if dims else score

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draws from the normalized law with leading sample dimensions.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Draws from the normalized law with leading sample dimensions."""
        return self._inner.sample(sample_shape)


class Normalize(Measure):
    """Rescale a sub-measure to a probability measure.

    `log_prob_normalized(value) = log_prob(value) - log_normalizer()`
    by construction; this wrapper makes the normalisation explicit
    so downstream code can rely on `log_normalizer() == 0`. Use at
    the observe / sample boundary or whenever a probability measure
    is required (e.g. before sampling).

    Parameters
    ----------
    base : Distribution
        Raw measure with finite positive total mass on every batch entry.
    validate_args : bool or None
        Optional Torch argument-validation policy.

    Raises
    ------
    ValueError
        If any base batch entry has zero or nonfinite total mass.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        self.base = base
        self._base_log_normalizer = _native_log_mass(
            base, batch_axes=len(base.batch_shape)
        )
        demand(
            torch.isfinite(self._base_log_normalizer),
            lambda: ValueError("Normalize requires a finite positive measure mass"),
        )
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    @_constraints.dependent_property
    def support(self) -> _constraints.Constraint:
        """Constraint describing valid response values.

        Returns
        -------
        _constraints.Constraint
            Constraint describing valid response values."""
        return self.base.support

    def log_normalizer(self) -> Tensor:
        """Log of the raw total mass, one value per batch entry.

        Returns
        -------
        Tensor
            Log of the raw total mass, one value per batch entry."""
        return torch.zeros_like(self._base_log_normalizer)

    @property
    def mean(self) -> Tensor:
        """Mean of a scalar law conditioned on an interval, without sampling.

        Returns
        -------
        Tensor
            The truncated mean from a closed form, a finite sum, or
            deterministic quadrature, in the dtype of the base law.

        Raises
        ------
        NotImplementedError
            If the base is not the restriction of a scalar law to an
            interval, or no deterministic rule covers that law there.
        """
        restricted = self.base
        if not isinstance(restricted, Restrict) or restricted.event_shape:
            raise NotImplementedError(
                "a normalized measure has a deterministic mean only as a scalar "
                "law restricted to an interval"
            )
        return truncated_mean(
            restricted.base,
            restricted.low,
            restricted.high,
            lower_closed=restricted.lower_closed,
            upper_closed=restricted.upper_closed,
        ).to(_reference(restricted.base))

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Normalize the same base event mass by the retained total mass.

        Parameters
        ----------
        low : object
            Lower event bound; None leaves that side unbounded.
        high : object
            Upper event bound; None leaves that side unbounded.
        lower_closed : object
            Whether the lower endpoint belongs to the event.
        upper_closed : object
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        object
            Raw log mass on the requested interval, retaining independent batch axes."""
        return (
            log_interval_mass(
                self.base,
                low,
                high,
                lower_closed=lower_closed,
                upper_closed=upper_closed,
            )
            - self._base_log_normalizer
        )

    def expand(self, batch_shape, _instance=None):
        """Broadcast a normalized measure through its underlying construction.

        Parameters
        ----------
        batch_shape : object
            Requested broadcast batch shape.
        _instance : object
            Unused Torch expansion instance.

        Returns
        -------
        object
            Distribution with the requested batch shape and retained construction."""
        return Normalize(self.base.expand(batch_shape))

    @property
    def _atomicity(self) -> str:
        """Retain the base law's atomicity proof.

        Returns
        -------
        str
            Atomless, discrete, or mixed classification inherited from the component measures.
        """
        from quivers.continuous._atoms import atomicity

        return atomicity(self.base)

    def atom_reference(self, value: Tensor) -> Tensor:
        """Preserve the base atomic carrier under normalization.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Membership in the base carrier.
        """
        from quivers.continuous._atoms import atom_reference

        return atom_reference(self.base, value)

    def log_atom_mass(self, value: Tensor) -> Tensor:
        """Normalize raw base singleton masses exactly once.

        Parameters
        ----------
        value : Tensor
            Query points.

        Returns
        -------
        Tensor
            Normalized singleton log mass.
        """
        from quivers.continuous._atoms import log_atom_mass

        return log_atom_mass(self.base, value) - self._base_log_normalizer

    def log_prob(self, value: Tensor) -> Tensor:
        """Raw log density or mass at the evaluation points.

        Parameters
        ----------
        value : Tensor
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Raw log density or mass at the evaluation points."""
        return self.base.log_prob(value) - self._base_log_normalizer

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draws from the normalized law with leading sample dimensions.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Draws from the normalized law with leading sample dimensions."""
        return self.base.sample(sample_shape)


def _sample_active(
    distribution: Distribution, sample_shape: torch.Size, active: Tensor
) -> Tensor:
    """Propagate active batch entries through measure compositions.

    Unselected mixture entries need no draw and may legitimately have
    zero mass. Native constructor parameters are selected before sampling;
    restrictions and mixtures propagate the same retained entry mask.

    Parameters
    ----------
    distribution : Distribution
        Native distribution or raw measure.
    sample_shape : torch.Size
        Leading independent sampling dimensions.
    active : Tensor
        Boolean batch mask selecting entries whose draws are required.

    Returns
    -------
    Tensor
        Normalized draws; unselected batch entries carry unused placeholders."""
    if isinstance(distribution, Normalize):
        return _sample_active(distribution.base, sample_shape, active)
    if isinstance(distribution, PointMass):
        return distribution.sample(sample_shape)
    if isinstance(distribution, (Restrict, Mixture)):
        return distribution._sample_active(sample_shape, active)
    if isinstance(distribution, Pushforward):
        return distribution.bijector.forward(
            _sample_active(distribution.base, sample_shape, active)
        )
    if isinstance(distribution, Independent):
        distribution = distribution._inner
    if isinstance(distribution, torch.distributions.Independent):
        base = distribution.base_dist
        inner = base.batch_shape[len(distribution.batch_shape) :]
        mask = active.reshape(active.shape + (1,) * len(inner)).expand(base.batch_shape)
        return _sample_active(base, sample_shape, mask)
    if bool(active.all()):
        return distribution.sample(sample_shape)
    from quivers.continuous._batch_selection import sample_active

    return sample_active(distribution, sample_shape, active)


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
    finite cut equals `loc` elementwise, and the base has a known
    symmetric location family; returns `None` otherwise so the caller can
    fall through to the discrete pmf-sum path.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    low : Tensor | None
        Lower event bound; None leaves that side unbounded.
    high : Tensor | None
        Upper event bound; None leaves that side unbounded.

    Returns
    -------
    Tensor | None
        Log half-mass when a fixed symmetry center certifies it, otherwise None."""
    one_sided = (low is None) != (high is None)
    if not one_sided:
        return None
    if not isinstance(
        base,
        (
            torch.distributions.Normal,
            torch.distributions.StudentT,
            torch.distributions.Cauchy,
            torch.distributions.Laplace,
        ),
    ):
        return None
    loc = getattr(base, "loc", None)
    if not isinstance(loc, Tensor):
        return None
    if loc.requires_grad or depends_on_position(loc):
        # A fixed cut that happens to equal a learnable center has a
        # nonzero location derivative. A constant half-mass would erase it.
        return None
    cut = low if high is None else high
    if cut is None:
        return None
    cut = cut.to(dtype=loc.dtype)
    if not bool((loc == cut).all()):
        return None
    return torch.full_like(loc, -math.log(2.0))


def truncated_law(
    base: Distribution, event: Mapping[str, Tensor | float | int | None] | None
) -> Distribution:
    """The sampling law of a response under its recording roles.

    Parameters
    ----------
    base : Distribution
        The response law.
    event : Mapping[str, Tensor | float | int | None] or None
        The bound recording roles. Only a truncation interval changes how
        a response is sampled; censoring says how it is recorded.

    Returns
    -------
    Distribution
        ``base``, normalized on its truncation interval when it has one.
    """
    roles: Mapping[str, Tensor | float | int | None] = {} if event is None else event
    lower, upper = roles.get("trunc_lower"), roles.get("trunc_upper")
    if lower is None and upper is None:
        return base
    truncated = getattr(base, "componentwise_truncation", None)
    if callable(truncated):
        law = truncated(lower, upper)
        assert isinstance(law, Distribution)
        return law
    return Normalize(Restrict(base, lower, upper))


def normalize_at_boundary(d: Distribution) -> Distribution:
    """Lift `d` to a probability measure if it is a sub-measure
    with non-zero log-normaliser; otherwise return `d` unchanged.

    Called by the inline / observe path to enforce the
    "renormalise at the observe / sample boundary" invariant the
    sub-Giry algebra relies on.

    Parameters
    ----------
    d : Distribution
        Native distribution or measure to normalize at an execution boundary.

    Returns
    -------
    Distribution
        Probability law, with explicit normalization when required."""
    if isinstance(d, Measure):
        norm = d.log_normalizer()
        if isinstance(norm, Tensor):
            if anywhere(norm.abs() > 1e-10):
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
    "truncated_law",
]
