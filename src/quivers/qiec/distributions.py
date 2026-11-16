"""Runtime distributions for the reference evaluator.

A :class:`DistributionValue` evaluates to a :class:`RuntimeDistribution`: the
family's name and its named arguments as host values, with no host library
behind it. Sampling and log densities go through a backend. The reference
backend here implements the scalar and simplex families in plain Python, so
the reference machine can run a model end to end and be compared against
an independent oracle; a host provider may install a backend that covers
every family through its own library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
from types import MappingProxyType
from typing import Literal, Protocol, overload

import didactic.api as dx

from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec._multinomial import (
    SAMPLERS as _COUNT_VECTOR_SAMPLERS,
    DENSITIES as _COUNT_VECTOR_DENSITIES,
)
from quivers.qiec._count_tails import (
    _discrete_weibull_parameters,
    _discrete_weibull_log_interval,
)
from quivers.qiec._sampling import (
    _binomial_sample as _sample_binomial_count,
    _dirichlet_weights,
    _trial_count,
)
from quivers.qiec._wiener import density as _wiener_density, sample as _wiener_sample
from quivers.qiec._extended_beta import (
    density as _extended_beta_density,
    sample as _extended_beta_sample,
)
from quivers.qiec._bernstein_hazard import (
    density as _bernstein_density,
    sample as _bernstein_sample,
)
from quivers.qiec._com_poisson import (
    density as _com_poisson_density,
    sample as _com_poisson_sample,
)
from quivers.qiec._count_mixtures import (
    SAMPLERS as _COUNT_MIXTURE_SAMPLERS,
    _binomial_log_mass,
    _poisson as _sample_poisson_count,
    _poisson_log_mass,
    DENSITIES as _COUNT_MIXTURE_DENSITIES,
)
from quivers.qiec.reference_families import (
    DENSITIES as _STRUCTURED_DENSITIES,
    LOG_MASSES as _LOG_MASSES,
    SAMPLERS as _STRUCTURED_SAMPLERS,
)
from quivers.qiec.reference_support import (
    Density,
    DistributionError,
    Sampler,
    finite_value as _finite_value,
    _integer_observation,
    normal_log_prob as _normal_log_prob,
    log_probabilities as _log_probabilities,
    probability as _probabilities,
    real_parameter as _real,
    simplex as _simplex,
    vector_parameter as _vector,
)


#: Families whose draw has a size no parameter fixes; the constructed type
#: supplies it, and the backend receives it as the ``dimension`` argument.
SHAPE_FROM_EVENT: frozenset[str] = frozenset({"LKJCholesky", "LKJCorrelationFactor"})


class DistributionBackend(Protocol):
    """What a host must supply to sample and score a family."""

    def sample(
        self, family: str, arguments: Mapping[str, object], rng: random.Random
    ) -> object:
        """Draw one value.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.
        rng : random.Random
            The generator to draw from.

        Returns
        -------
        object
            The drawn host value.
        """
        ...

    def log_prob(
        self, family: str, arguments: Mapping[str, object], value: object
    ) -> float:
        """Evaluate the log density at a value.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.
        value : object
            The point evaluated.

        Returns
        -------
        float
            The log density.
        """
        ...

    def log_mass(self, family: str, arguments: Mapping[str, object]) -> float:
        """The log of the total mass of a construction.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.

        Returns
        -------
        float
            Zero for a probability measure; the log mass of a restriction
            or of a mixture of restrictions, which ``Normalize`` divides
            by.
        """
        ...


def tensor_rank(value: object) -> int:
    """The nesting depth of a host tensor.

    Parameters
    ----------
    value : object
        A host value: a scalar, or a tuple of values of one rank.

    Returns
    -------
    int
        Zero for a scalar, one more than the first entry's rank for a
        tuple; an empty tuple has rank one.
    """
    if isinstance(value, tuple):
        return 1 + (tensor_rank(value[0]) if value else 0)
    return 0


def _slice_at(value: object, position: Sequence[int], rank: int) -> object:
    """Read an argument at one plate position, broadcasting shorter ranks.

    Parameters
    ----------
    value : object
        The argument, of tensor rank at most the parameter's rank plus
        the plate's.
    position : Sequence[int]
        The plate position, outermost axis first.
    rank : int
        The parameter's own rank.

    Returns
    -------
    object
        The argument's entry for the position: the argument itself when
        it carries no plate dimensions, else the slice its trailing plate
        dimensions select.
    """
    leading = tensor_rank(value) - rank
    if leading <= 0:
        return value
    for index in position[len(position) - leading :]:
        value = value[index]  # type: ignore[index]
    return value


def _positions(shape: Sequence[int]) -> list[tuple[int, ...]]:
    """Every position of a shape in row-major order.

    Parameters
    ----------
    shape : Sequence[int]
        The shape.

    Returns
    -------
    list[tuple[int, ...]]
        The positions; one empty position for the empty shape.
    """
    positions: list[tuple[int, ...]] = [()]
    for extent in shape:
        positions = [
            (*position, index) for position in positions for index in range(extent)
        ]
    return positions


def _nest(entries: Mapping[tuple[int, ...], object], shape: Sequence[int]) -> object:
    """Arrange per-position entries as nested tuples.

    Parameters
    ----------
    entries : Mapping[tuple[int, ...], object]
        One entry per position of ``shape``.
    shape : Sequence[int]
        The shape.

    Returns
    -------
    object
        The entry at the empty position for the empty shape, else nested
        tuples with the outermost axis first.
    """

    def build(prefix: tuple[int, ...], depth: int) -> object:
        """Build the tuple at one prefix of the shape.

        Parameters
        ----------
        prefix : tuple[int, ...]
            The position so far.
        depth : int
            How many axes the prefix covers.

        Returns
        -------
        object
            The entry when every axis is covered, else a tuple over the
            next axis.
        """
        if depth == len(shape):
            return entries[prefix]
        return tuple(
            build((*prefix, index), depth + 1) for index in range(shape[depth])
        )

    return build((), 0)


class RuntimeDistribution(QiecModel):
    """A constructed distribution, as host data.

    Parameter mappings are owned snapshots compared by identity. Their
    host values retain identity, including tensors and their gradients.
    Plate extents and parameter ranks compare by value.

    Parameters
    ----------
    family
        The family's source name.
    arguments
        The named parameters as host values; frozen on construction.
    batch
        The plate's batch extents, outermost first; each position is an
        independent draw scored on its own.
    event
        The plate's extended event extents, outermost first; draws along
        them join one event and are scored together.
    ranks
        Each parameter's own tensor rank, from the family registry, so
        an argument carrying plate dimensions can be told from one that
        does not.
    natural
        The family's own event extents, so a scalar given for a tensor
        parameter can be broadcast to the shape the family expects.
    """

    family: str
    arguments: Mapping[str, HeldValue] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    batch: tuple[int, ...] = ()
    event: tuple[int, ...] = ()
    ranks: Mapping[str, int] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    natural: tuple[int, ...] = ()

    @dx.model_validator()
    def _freeze_arguments(self) -> None:
        """Snapshot the parameter mapping while retaining its host values."""
        self._normalize_field("arguments", MappingProxyType(dict(self.arguments)))

    @property
    def plated(self) -> bool:
        """Whether the distribution is constructed over a plate.

        Returns
        -------
        bool
            ``True`` when it has batch or extended event axes.
        """
        return bool(self.batch or self.event)

    def _at(self, position: tuple[int, ...]) -> Mapping[str, object]:
        """The family's arguments at one plate position.

        Parameters
        ----------
        position : tuple[int, ...]
            A position of the batch and event axes together.

        Returns
        -------
        Mapping[str, object]
            The named parameters with their plate dimensions selected.
        """
        arguments = {
            name: self._broadcast(
                name, _slice_at(value, position, self.ranks.get(name, 0))
            )
            for name, value in self.arguments.items()
        }
        if self.family in SHAPE_FROM_EVENT and self.natural:
            # The family's parameters say nothing about the size of a
            # draw; the constructed type does, through the event extents.
            arguments["dimension"] = self.natural[-1]
        return arguments

    def _broadcast(self, name: str, value: object) -> object:
        """Expand a scalar given for a tensor parameter to its shape.

        Parameters
        ----------
        name : str
            The parameter.
        value : object
            Its value at one plate position.

        Returns
        -------
        object
            The value, or nested tuples of it over the parameter's own
            dimensions when it is a scalar and the parameter has rank:
            the family's event extents when their count is the rank,
            else the last event extent repeated.
        """
        rank = self.ranks.get(name, 0)
        if rank == 0 or isinstance(value, tuple) or not self.natural:
            return value
        dims = (
            self.natural
            if len(self.natural) == rank
            else tuple(self.natural[-1] for _ in range(rank))
        )
        result: object = value
        for extent in reversed(dims):
            result = tuple(result for _ in range(extent))
        return result

    def at(self, position: tuple[int, ...]) -> RuntimeDistribution:
        """The distribution at one position of the plate, unplated.

        Parameters
        ----------
        position : tuple[int, ...]
            A position of the leading plate axes, outermost first; fewer
            positions than axes leave the remaining axes in place.

        Returns
        -------
        RuntimeDistribution
            The family with its arguments selected at the position and
            the selected axes dropped from the plate.

        Raises
        ------
        DistributionError
            If the position has more entries than the plate has axes.
        """
        shape = (*self.batch, *self.event)
        if len(position) > len(shape):
            raise DistributionError(
                f"position {position!r} is deeper than a plate of rank {len(shape)}"
            )
        arguments = {
            name: self._broadcast(
                name, _slice_at(value, position, self.ranks.get(name, 0))
            )
            for name, value in self.arguments.items()
        }
        remaining = len(position)
        batch = self.batch[remaining:] if remaining < len(self.batch) else ()
        event = self.event[max(remaining - len(self.batch), 0) :]
        return RuntimeDistribution(
            family=self.family,
            arguments=arguments,
            batch=batch,
            event=event,
            ranks=self.ranks,
            natural=self.natural,
        )

    def sample(self, rng: random.Random | None = None) -> object:
        """Draw one value through the installed backend.

        Parameters
        ----------
        rng : random.Random | None
            The generator to draw from; the module generator by default,
            which :func:`seed_reference_rng` controls.

        Returns
        -------
        object
            The drawn host value: one draw of the family, or nested
            tuples of draws over the plate's batch and event axes.

        Raises
        ------
        DistributionError
            If the backend cannot sample the family.
        """
        generator = rng or _RNG
        if not self.plated:
            return _backend.sample(self.family, self._at(()), generator)
        shape = (*self.batch, *self.event)
        return _nest(
            {
                position: _backend.sample(self.family, self._at(position), generator)
                for position in _positions(shape)
            },
            shape,
        )

    @overload
    def log_mass(self, batch_axes: Literal[0] = 0) -> float:
        """Query the scalar total mass.

        Parameters
        ----------
        batch_axes : Literal[0]
            Reduce every independent axis.

        Returns
        -------
        float
            Total log mass.
        """

    @overload
    def log_mass(self, batch_axes: int) -> object:
        """Query mass with a retained batch prefix.

        Parameters
        ----------
        batch_axes : int
            Number of leading batch axes to retain.

        Returns
        -------
        object
            Nested log weights on the retained axes.
        """

    def log_mass(self, batch_axes: int = 0) -> object:
        """The log of the distribution's total mass through the installed backend.

        Parameters
        ----------
        batch_axes : int
            Number of leading independent batch axes retained in the result.
            The default totals the entire product measure.

        Returns
        -------
        float or nested tuple
            Zero for a probability measure; over a plate, the sum of the
            masses at every position, since the plate's draw is their
            product. Requested batch axes remain separate; event axes sum.

        Raises
        ------
        DistributionError
            If the retained rank is not an integer in the batch rank's range.
        """
        if (
            isinstance(batch_axes, bool)
            or not isinstance(batch_axes, int)
            or not 0 <= batch_axes <= len(self.batch)
        ):
            raise DistributionError(
                "log_mass retained axes must be an integer within the batch rank"
            )
        if not self.plated:
            return _backend.log_mass(self.family, self._at(()))
        shape = (*self.batch, *self.event)

        def total(prefix: tuple[int, ...]) -> float:
            """Sum independent log masses under one retained position.

            Parameters
            ----------
            prefix : tuple[int, ...]
                Position in retained batch axes.

            Returns
            -------
            float
                Log mass of the remaining independent product.
            """
            return math.fsum(
                _backend.log_mass(self.family, self._at((*prefix, *position)))
                for position in _positions(shape[batch_axes:])
            )

        if not batch_axes:
            return total(())
        kept = shape[:batch_axes]
        return _nest({position: total(position) for position in _positions(kept)}, kept)

    def log_prob(self, value: object, keep_batch: bool = False) -> object:
        """Evaluate the log density at a value through the installed backend.

        Parameters
        ----------
        value : object
            The point evaluated; over a plate, nested tuples of points
            shaped like the plate.
        keep_batch : bool
            Whether to keep one weight per batch position rather than
            total them.

        Returns
        -------
        object
            The log density as a float, or, with ``keep_batch`` over a
            batch, nested tuples of floats shaped like the batch axes.

        Raises
        ------
        DistributionError
            If the backend cannot score the family, or the point is not
            shaped like the plate.
        """
        if not self.plated:
            return _backend.log_prob(self.family, self._at(()), value)
        shape = (*self.batch, *self.event)
        if tensor_rank(value) < len(shape):
            raise DistributionError(
                f"a point scored under a plate of rank {len(shape)} has rank "
                f"{tensor_rank(value)}"
            )
        totals: dict[tuple[int, ...], float] = {
            position: 0.0 for position in _positions(self.batch)
        }
        for position in _positions(shape):
            point = value
            for index in position:
                point = point[index]  # type: ignore[index]
            totals[position[: len(self.batch)]] += _backend.log_prob(
                self.family, self._at(position), point
            )
        if keep_batch:
            return _nest(totals, self.batch)
        return math.fsum(totals.values())


def _normal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Normal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.gauss(_real(a, "loc", "Normal"), _real(a, "scale", "Normal"))


def _normal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Normal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    return _normal_log_prob(
        _finite_value(value, "Normal"),
        _real(a, "loc", "Normal"),
        _real(a, "scale", "Normal"),
    )


def _lognormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LogNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return math.exp(
        rng.gauss(_real(a, "loc", "LogNormal"), _real(a, "scale", "LogNormal"))
    )


def _lognormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LogNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "LogNormal")
    if point <= 0:
        return -math.inf
    return _normal_log_prob(
        math.log(point),
        _real(a, "loc", "LogNormal"),
        _real(a, "scale", "LogNormal"),
    ) - math.log(point)


def _halfnormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``HalfNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return abs(rng.gauss(0.0, _real(a, "scale", "HalfNormal")))


def _halfnormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``HalfNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "HalfNormal")
    if point < 0:
        return -math.inf
    return math.log(2.0) + _normal_log_prob(point, 0.0, _real(a, "scale", "HalfNormal"))


def _bernoulli_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Bernoulli``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.random() < _probabilities(a, "Bernoulli")


def _bernoulli_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Bernoulli`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    log_success, log_failure = _log_probabilities(a, "Bernoulli")
    point = _integer_observation(value, "Bernoulli")
    return log_success if point == 1 else log_failure if point == 0 else -math.inf


def _categorical_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Categorical``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    probs = _simplex(a, "Categorical")
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probs):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probs) - 1


def _categorical_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Categorical`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    probs = _simplex(a, "Categorical")
    index = _integer_observation(value, "Categorical")
    if index is None or not 0 <= index < len(probs) or probs[index] <= 0:
        return -math.inf
    return math.log(probs[index])


def _beta_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Beta``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.betavariate(
        _real(a, "concentration1", "Beta"), _real(a, "concentration0", "Beta")
    )


def _beta_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Beta`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Beta")
    alpha = _real(a, "concentration1", "Beta")
    beta = _real(a, "concentration0", "Beta")
    if not 0 < point < 1:
        return -math.inf
    return (
        (alpha - 1) * math.log(point)
        + (beta - 1) * math.log1p(-point)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )


def _gamma_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Gamma``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.gammavariate(
        _real(a, "concentration", "Gamma"), 1.0 / _real(a, "rate", "Gamma")
    )


def _gamma_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Gamma`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Gamma")
    shape = _real(a, "concentration", "Gamma")
    rate = _real(a, "rate", "Gamma")
    if point <= 0:
        return -math.inf
    return (
        shape * math.log(rate)
        + (shape - 1) * math.log(point)
        - rate * point
        - math.lgamma(shape)
    )


def _hurdle_probability(a: Mapping[str, object], family: str) -> float:
    """Validate the hurdle probability and positive-component parameters.

    Parameters
    ----------
    a : Mapping[str, object]
        Natural-scale family parameters.
    family : str
        HurdleGamma or HurdleLogNormal.

    Returns
    -------
    float
        Probability of the structural zero atom.

    Raises
    ------
    DistributionError
        If the probability or a component parameter is invalid.
    """
    probability = _real(a, "zero_prob", family)
    if not 0 <= probability <= 1:
        raise DistributionError(f"{family} needs zero_prob in [0, 1]")
    if family == "HurdleGamma":
        if _real(a, "concentration", family) <= 0 or _real(a, "rate", family) <= 0:
            raise DistributionError("HurdleGamma needs concentration > 0 and rate > 0")
    elif _real(a, "scale", family) <= 0:
        raise DistributionError("HurdleLogNormal needs scale > 0")
    else:
        _real(a, "loc", family)
    return probability


def _hurdle_density(
    a: Mapping[str, object], value: object, family: str, density: Density
) -> float:
    """Score an atom or positive density against their disjoint reference measures.

    Parameters
    ----------
    a : Mapping[str, object]
        Natural-scale family parameters.
    value : object
        Response being scored.
    family : str
        Name used for validation diagnostics.
    density : Density
        Positive-component density under its original parameter names.

    Returns
    -------
    float
        Log mass at zero, log density above zero, or negative infinity.
    """
    probability = _hurdle_probability(a, family)
    point = _finite_value(value, family)
    if point == 0:
        return math.log(probability) if probability > 0 else -math.inf
    if point < 0 or probability == 1:
        return -math.inf
    return math.log1p(-probability) + density(a, point)


def _hurdle_gamma_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw a zero atom or positive Gamma variate.

    Parameters
    ----------
    a : Mapping[str, object]
        Zero probability, concentration, and rate.
    rng : random.Random
        Random generator.

    Returns
    -------
    object
        Zero or a positive Gamma draw.
    """
    probability = _hurdle_probability(a, "HurdleGamma")
    return 0.0 if rng.random() < probability else _gamma_sample(a, rng)


def _hurdle_gamma_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate the zero-atom/Gamma density.

    Parameters
    ----------
    a : Mapping[str, object]
        Zero probability, concentration, and rate.
    value : object
        Response being scored.

    Returns
    -------
    float
        Log mass or positive-response log density.
    """
    return _hurdle_density(a, value, "HurdleGamma", _gamma_density)


def _hurdle_lognormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw a zero atom or positive LogNormal variate.

    Parameters
    ----------
    a : Mapping[str, object]
        Zero probability, log location, and log scale.
    rng : random.Random
        Random generator.

    Returns
    -------
    object
        Zero or a positive LogNormal draw.
    """
    probability = _hurdle_probability(a, "HurdleLogNormal")
    return 0.0 if rng.random() < probability else _lognormal_sample(a, rng)


def _hurdle_lognormal_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate the zero-atom/LogNormal density.

    Parameters
    ----------
    a : Mapping[str, object]
        Zero probability, log location, and log scale.
    value : object
        Response being scored.

    Returns
    -------
    float
        Log mass or positive-response log density.
    """
    return _hurdle_density(a, value, "HurdleLogNormal", _lognormal_density)


def _exponential_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Exponential``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.expovariate(_real(a, "rate", "Exponential"))


def _exponential_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Exponential`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Exponential")
    rate = _real(a, "rate", "Exponential")
    return math.log(rate) - rate * point if point >= 0 else -math.inf


def _uniform_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Uniform``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.uniform(_real(a, "low", "Uniform"), _real(a, "high", "Uniform"))


def _uniform_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Uniform`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Uniform")
    low = _real(a, "low", "Uniform")
    high = _real(a, "high", "Uniform")
    return -math.log(high - low) if low <= point < high else -math.inf


def _poisson_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Poisson``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return _sample_poisson_count(_real(a, "rate", "Poisson"), rng)


def _poisson_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Poisson`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support; at a zero rate
        the mass sits at zero alone.
    """
    count = _integer_observation(value, "Poisson", float64=True)
    rate = _real(a, "rate", "Poisson")
    if count is None:
        return -math.inf
    return _poisson_log_mass(count, rate)


def _binomial_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Binomial``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    trials = _trial_count(a.get("total_count"))
    probability = _probabilities(a, "Binomial")
    return _sample_binomial_count(trials, probability, rng)


def _binomial_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Binomial`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    count = _integer_observation(value, "Binomial", float64=True)
    if count is None:
        return -math.inf
    trials = _trial_count(a.get("total_count"))
    log_success, log_failure = _log_probabilities(a, "Binomial")
    if not 0 <= count <= trials:
        return -math.inf
    return _binomial_log_mass(float(trials), count, log_success, log_failure)


def _geometric_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Geometric``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.

    Raises
    ------
    DistributionError
        If probabilities are improper or a draw exceeds exact count capacity.
    """
    from quivers.qiec._count_tails import geometric_coordinates
    from quivers.qiec._count_mixtures import _require_sample_capacity

    _, log_failure = geometric_coordinates(a)
    if log_failure == 0:
        raise DistributionError(
            "Geometric sampling exceeds exact Float64 count capacity"
        )
    value = math.log1p(-rng.random()) / log_failure
    return int(math.floor(_require_sample_capacity(value)))


def _geometric_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Geometric`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    failures = _integer_observation(value, "Geometric", float64=True)
    from quivers.qiec._count_tails import geometric_coordinates

    log_success, log_failure = geometric_coordinates(a)
    if failures is None or failures < 0:
        return -math.inf
    return _times(failures, log_failure) + log_success


def _times(count: int, log_mass: float) -> float:
    """``count`` times a log mass, zero when the count is.

    Parameters
    ----------
    count : int
        How many times the outcome occurs.
    log_mass : float
        Its log probability, possibly ``-inf``.

    Returns
    -------
    float
        The product, with ``0 * -inf`` read as zero: an outcome that
        never occurs contributes nothing, whatever its mass.
    """
    return 0.0 if count == 0 else count * log_mass


def _log_erfcx_positive(value: float) -> float:
    """Log scaled complementary error function for a nonnegative argument.

    Parameters
    ----------
    value : float
        A nonnegative argument.

    Returns
    -------
    float
        Log of the scaled complementary error function, including its tail limit.
    """
    if value < 5:
        return value * value + math.log(math.erfc(value))
    fraction = 0.0
    for index in range(40, 0, -1):
        fraction = (index / 2) / (value + fraction)
    return -0.5 * math.log(math.pi) - math.log(value + fraction)


def _exgaussian_parameters(a: Mapping[str, object]) -> tuple[float, float, float]:
    """Read a Normal location/scale and a positive Exponential rate.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, scale, and Exponential rate.

    Returns
    -------
    tuple[float, float, float]
        Validated parameters in constructor order.

    Raises
    ------
    DistributionError
        If scale or rate is nonpositive, or any parameter is malformed.
    """
    loc = _real(a, "loc", "ExGaussian")
    scale = _real(a, "scale", "ExGaussian")
    rate = _real(a, "rate", "ExGaussian")
    if scale <= 0 or rate <= 0:
        raise DistributionError("ExGaussian needs scale > 0 and rate > 0")
    return loc, scale, rate


def _exgaussian_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw independent Normal and Exponential components and add them.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, scale, and Exponential rate.
    rng : random.Random
        Local generator used for both component draws.

    Returns
    -------
    object
        One draw from the named response distribution.
    """
    loc, scale, rate = _exgaussian_parameters(a)
    return rng.gauss(loc, scale) + rng.expovariate(rate)


def _exgaussian_density(a: Mapping[str, object], value: object) -> float:
    """Score an exponentially modified Gaussian without left-tail underflow.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, scale, and Exponential rate.
    value : object
        Real evaluation point.

    Returns
    -------
    float
        Log density at the point; negative infinity outside the support.
    """
    loc, scale, rate = _exgaussian_parameters(a)
    point = _finite_value(value, "ExGaussian")
    standardized = (point - loc) / scale
    rate_scale = rate * scale
    difference = rate_scale - standardized
    if difference > 0:
        return (
            math.log(rate)
            - math.log(2)
            - 0.5 * standardized**2
            + _log_erfcx_positive(difference / math.sqrt(2))
        )
    return (
        math.log(rate)
        + 0.5 * rate_scale**2
        - rate_scale * standardized
        + math.log(0.5 * math.erfc(difference / math.sqrt(2)))
    )


def _inverse_gaussian_parameters(a: Mapping[str, object]) -> tuple[float, float]:
    """Read the positive mean and shape of an inverse Gaussian.

    Parameters
    ----------
    a : Mapping[str, object]
        Named response mean and positive shape.

    Returns
    -------
    tuple[float, float]
        Validated parameters in constructor order.

    Raises
    ------
    DistributionError
        If mean or shape is nonpositive, or any parameter is malformed.
    """
    mu = _real(a, "mu", "InverseGaussian")
    shape = _real(a, "shape", "InverseGaussian")
    if mu <= 0 or shape <= 0:
        raise DistributionError("InverseGaussian needs mu > 0 and shape > 0")
    return mu, shape


def _inverse_gaussian_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw an inverse Gaussian using the exact quadratic transformation.

    Parameters
    ----------
    a : Mapping[str, object]
        Named response mean and positive shape.
    rng : random.Random
        Local generator used for the transformation and branch choice.

    Returns
    -------
    object
        One draw from the named response distribution.
    """
    mu, shape = _inverse_gaussian_parameters(a)
    t = mu * rng.gauss(0, 1) ** 2 / (2 * shape)
    smaller = mu / (1 + t + math.sqrt(t) * math.sqrt(t + 2))
    return smaller if rng.random() <= mu / (mu + smaller) else mu * (mu / smaller)


def _inverse_gaussian_density(a: Mapping[str, object], value: object) -> float:
    """Score the inverse Gaussian in mean/shape parameterization.

    Parameters
    ----------
    a : Mapping[str, object]
        Named response mean and positive shape.
    value : object
        Real evaluation point.

    Returns
    -------
    float
        Log density at the point; negative infinity outside the support.
    """
    mu, shape = _inverse_gaussian_parameters(a)
    point = _finite_value(value, "InverseGaussian")
    if point <= 0:
        return -math.inf
    return (
        0.5 * (math.log(shape) - math.log(2 * math.pi) - 3 * math.log(point))
        - 0.5 * shape / point * ((point - mu) / mu) ** 2
    )


def _asymmetric_laplace_parameters(
    a: Mapping[str, object],
) -> tuple[float, float, float]:
    """Read the location, positive scale, and strictly interior quantile.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, positive scale, and interior quantile.

    Returns
    -------
    tuple[float, float, float]
        Validated parameters in constructor order.

    Raises
    ------
    DistributionError
        If scale is nonpositive, quantile is not interior, or a parameter is malformed.
    """
    loc = _real(a, "loc", "AsymmetricLaplace")
    scale = _real(a, "scale", "AsymmetricLaplace")
    quantile = _real(a, "quantile", "AsymmetricLaplace")
    if scale <= 0 or not 0 < quantile < 1:
        raise DistributionError(
            "AsymmetricLaplace needs scale > 0 and 0 < quantile < 1"
        )
    return loc, scale, quantile


def _asymmetric_laplace_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw the difference of exponentials in the quantile parameterization.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, positive scale, and interior quantile.
    rng : random.Random
        Local generator used for both Exponential draws.

    Returns
    -------
    object
        One draw from the named response distribution.
    """
    loc, scale, quantile = _asymmetric_laplace_parameters(a)
    return loc + scale * (rng.expovariate(quantile) - rng.expovariate(1 - quantile))


def _asymmetric_laplace_density(a: Mapping[str, object], value: object) -> float:
    """Score the check-loss density in the quantile parameterization.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, positive scale, and interior quantile.
    value : object
        Real evaluation point.

    Returns
    -------
    float
        Log density at the point; negative infinity outside the support.
    """
    loc, scale, quantile = _asymmetric_laplace_parameters(a)
    residual = (_finite_value(value, "AsymmetricLaplace") - loc) / scale
    loss = residual * (quantile - int(residual < 0))
    return math.log(quantile) + math.log1p(-quantile) - math.log(scale) - loss


def _discrete_weibull_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw a count by a constant-work exponential transform.

    Parameters
    ----------
    a : Mapping[str, object]
        Survival probability and concentration.
    rng : random.Random
        Local random generator.

    Returns
    -------
    object
        Nonnegative integer draw.

    Raises
    ------
    DistributionError
        If the sampled count exceeds exact Float64 capacity.
    """
    q, power = _discrete_weibull_parameters(a)
    if q == 0:
        return 0
    exponential = rng.expovariate(1)
    if exponential == 0:
        return 0
    log_count = (math.log(exponential) - math.log(-math.log(q))) / power
    if log_count > math.log(2**53):
        raise DistributionError(
            "DiscreteWeibull sampling exceeds exact Float64 count capacity 2**53"
        )
    return int(math.floor(math.exp(log_count)))


def _discrete_weibull_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate a count mass through a stable adjacent-power increment.

    Parameters
    ----------
    a : Mapping[str, object]
        Survival probability and concentration.
    value : object
        Candidate integer response.

    Returns
    -------
    float
        Log mass, negative infinity outside the integer support.
    """
    q, power = _discrete_weibull_parameters(a)
    point = _integer_observation(value, "DiscreteWeibull", float64=True)
    if point is None or point < 0:
        return -math.inf
    return _discrete_weibull_log_interval(q, power, point, point)


def _zero_inflated_asymmetric_laplace_probability(a: Mapping[str, object]) -> float:
    """Validate the structural zero probability and continuous component.

    Parameters
    ----------
    a : Mapping[str, object]
        Structural probability and asymmetric Laplace parameters.

    Returns
    -------
    float
        Closed-unit-interval zero probability.

    Raises
    ------
    DistributionError
        If a mixture or component parameter is invalid.
    """
    _asymmetric_laplace_parameters(a)
    probability = _real(a, "zero_prob", "ZeroInflatedAsymmetricLaplace")
    if not 0 <= probability <= 1:
        raise DistributionError(
            "ZeroInflatedAsymmetricLaplace requires zero_prob in [0,1]"
        )
    return probability


def _zero_inflated_asymmetric_laplace_sample(
    a: Mapping[str, object], rng: random.Random
) -> object:
    """Select the structural atom before drawing its continuous component.

    Parameters
    ----------
    a : Mapping[str, object]
        Structural probability and asymmetric Laplace parameters.
    rng : random.Random
        Local random generator.

    Returns
    -------
    object
        Exact zero or a continuous asymmetric Laplace response.
    """
    probability = _zero_inflated_asymmetric_laplace_probability(a)
    return 0.0 if rng.random() < probability else _asymmetric_laplace_sample(a, rng)


def _zero_inflated_asymmetric_laplace_density(
    a: Mapping[str, object], value: object
) -> float:
    """Score an atomic zero or a continuous asymmetric Laplace density.

    Parameters
    ----------
    a : Mapping[str, object]
        Structural probability and asymmetric Laplace parameters.
    value : object
        Real response.

    Returns
    -------
    float
        Log atom mass at zero, weighted component log density elsewhere.
    """
    probability = _zero_inflated_asymmetric_laplace_probability(a)
    point = _finite_value(value, "ZeroInflatedAsymmetricLaplace")
    if point == 0:
        return math.log(probability) if probability > 0 else -math.inf
    if probability == 1:
        return -math.inf
    return math.log1p(-probability) + _asymmetric_laplace_density(a, point)


def _laplace_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Laplace``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    loc = _real(a, "loc", "Laplace")
    scale = _real(a, "scale", "Laplace")
    draw = rng.random() - 0.5
    return loc - scale * math.copysign(1.0, draw) * math.log1p(-2 * abs(draw))


def _laplace_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Laplace`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Laplace")
    scale = _real(a, "scale", "Laplace")
    return -abs(point - _real(a, "loc", "Laplace")) / scale - math.log(2 * scale)


def _cauchy_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Cauchy``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    loc = _real(a, "loc", "Cauchy")
    scale = _real(a, "scale", "Cauchy")
    return loc + scale * math.tan(math.pi * (rng.random() - 0.5))


def _cauchy_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Cauchy`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "Cauchy")
    loc = _real(a, "loc", "Cauchy")
    scale = _real(a, "scale", "Cauchy")
    return -math.log(math.pi * scale) - math.log1p(((point - loc) / scale) ** 2)


def _studentt_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``StudentT``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    df = _real(a, "df", "StudentT")
    loc = _real(a, "loc", "StudentT")
    scale = _real(a, "scale", "StudentT")
    draw = rng.gauss(0.0, 1.0)
    chi = rng.gammavariate(df / 2.0, 2.0)
    return loc + scale * draw / math.sqrt(chi / df)


def _studentt_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``StudentT`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.
    """
    point = _finite_value(value, "StudentT")
    df = _real(a, "df", "StudentT")
    loc = _real(a, "loc", "StudentT")
    scale = _real(a, "scale", "StudentT")
    standardized = (point - loc) / scale
    return (
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1) / 2 * math.log1p(standardized**2 / df)
    )


def _generalized_pareto_parameters(
    a: Mapping[str, object],
) -> tuple[float, float, float]:
    """Read finite location, positive scale, and real tail index.

    Parameters
    ----------
    a : Mapping[str, object]
        Named GeneralizedPareto parameters.

    Returns
    -------
    tuple[float, float, float]
        Location, scale, and concentration.

    Raises
    ------
    DistributionError
        If a parameter is nonfinite or scale is nonpositive.
    """
    loc, scale, concentration = (
        _real(a, name, "GeneralizedPareto")
        for name in ("loc", "scale", "concentration")
    )
    if (
        not all(math.isfinite(item) for item in (loc, scale, concentration))
        or scale <= 0
    ):
        raise DistributionError(
            "GeneralizedPareto needs finite parameters and positive scale"
        )
    return loc, scale, concentration


def _generalized_pareto_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw through the exponential quantile transformation.

    Parameters
    ----------
    a : Mapping[str, object]
        Named parameters.
    rng : random.Random
        Host random generator.

    Returns
    -------
    object
        A real observation on the concentration-dependent support.
    """
    loc, scale, concentration = _generalized_pareto_parameters(a)
    exponential = rng.expovariate(1.0)
    standardized = (
        exponential
        if concentration == 0
        else math.expm1(concentration * exponential) / concentration
    )
    return loc + scale * standardized


def _generalized_pareto_density(a: Mapping[str, object], value: object) -> float:
    """Score the generalized Pareto law, including its exponential limit.

    Parameters
    ----------
    a : Mapping[str, object]
        Named parameters.
    value : object
        Real observation to score.

    Returns
    -------
    float
        Log density, with negative infinity outside the support.
    """
    loc, scale, concentration = _generalized_pareto_parameters(a)
    standardized = (_finite_value(value, "GeneralizedPareto") - loc) / scale
    if standardized < 0:
        return -math.inf
    if concentration == 0:
        return -math.log(scale) - standardized
    endpoint = 1 + concentration * standardized
    if endpoint < 0:
        return -math.inf
    if endpoint == 0:
        if concentration == -1:
            return -math.log(scale)
        return math.inf if concentration < -1 else -math.inf
    return -math.log(scale) - (1 + 1 / concentration) * math.log1p(
        concentration * standardized
    )


def _response_shape_parameters(
    a: Mapping[str, object], family: str
) -> dict[str, float]:
    """Read finite parameters for skew, shifted, and extreme-value responses.

    Parameters
    ----------
    a : Mapping[str, object]
        Named scalar parameters.
    family : str
        Registered response family.

    Returns
    -------
    dict[str, float]
        Finite parameters satisfying their scale and shape domains.

    Raises
    ------
    DistributionError
        If a scale, shift, or shape violates its domain.
    """
    names = {
        "SkewNormal": ("loc", "scale", "skewness"),
        "ShiftedLogNormal": ("loc", "scale", "shift"),
        "Frechet": ("scale", "concentration"),
        "GeneralizedExtremeValue": ("loc", "scale", "concentration"),
    }[family]
    p = {name: _real(a, name, family) for name in names}
    valid = all(math.isfinite(v) for v in p.values()) and p["scale"] > 0
    if family == "ShiftedLogNormal":
        valid = valid and p["shift"] >= 0
    if family == "Frechet":
        valid = valid and p["concentration"] > 0
    if not valid:
        raise DistributionError(f"{family} parameters violate their response domains")
    return p


def _skew_normal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Sample a skew normal from two independent standard normals.

    Parameters
    ----------
    a : Mapping[str, object]
        Location, scale, and skewness.
    rng : random.Random
        Host random generator.

    Returns
    -------
    object
        A real skew-normal response.
    """
    p = _response_shape_parameters(a, "SkewNormal")
    norm = math.hypot(1, p["skewness"])
    return p["loc"] + p["scale"] * (
        (p["skewness"] / norm) * abs(rng.gauss(0, 1)) + rng.gauss(0, 1) / norm
    )


def _skew_normal_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate a skew-normal density with a stable Gaussian tail.

    Parameters
    ----------
    a : Mapping[str, object]
        Location, scale, and skewness.
    value : object
        Real response.

    Returns
    -------
    float
        Log density.
    """
    p = _response_shape_parameters(a, "SkewNormal")
    z = (_finite_value(value, "SkewNormal") - p["loc"]) / p["scale"]
    t = p["skewness"] * z
    log_phi = (
        math.log(0.5) + _log_erfcx_positive(-t / math.sqrt(2)) - t * t / 2
        if t < 0
        else math.log(0.5 * math.erfc(-t / math.sqrt(2)))
    )
    return (
        math.log(2)
        - math.log(p["scale"])
        - 0.5 * math.log(2 * math.pi)
        - z * z / 2
        + log_phi
    )


def _shifted_lognormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw a positive lognormal component plus its nonnegative shift.

    Parameters
    ----------
    a : Mapping[str, object]
        Log-location, log-scale, and shift.
    rng : random.Random
        Host random generator.

    Returns
    -------
    object
        A response above the shift.
    """
    p = _response_shape_parameters(a, "ShiftedLogNormal")
    return p["shift"] + rng.lognormvariate(p["loc"], p["scale"])


def _shifted_lognormal_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate the translated lognormal density.

    Parameters
    ----------
    a : Mapping[str, object]
        Log-location, log-scale, and shift.
    value : object
        Real response.

    Returns
    -------
    float
        Log density, negative infinity at or below the shift.
    """
    p = _response_shape_parameters(a, "ShiftedLogNormal")
    point = _finite_value(value, "ShiftedLogNormal") - p["shift"]
    return (
        _normal_log_prob(math.log(point), p["loc"], p["scale"]) - math.log(point)
        if point > 0
        else -math.inf
    )


def _frechet_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw a Frechet observation through an exponential transformation.

    Parameters
    ----------
    a : Mapping[str, object]
        Positive scale and concentration.
    rng : random.Random
        Host random generator.

    Returns
    -------
    object
        A positive real response.
    """
    p = _response_shape_parameters(a, "Frechet")
    exponential = max(rng.expovariate(1), math.nextafter(0, 1))
    return p["scale"] * math.exp(-math.log(exponential) / p["concentration"])


def _frechet_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate the Frechet density on its positive support.

    Parameters
    ----------
    a : Mapping[str, object]
        Scale and concentration.
    value : object
        Real response.

    Returns
    -------
    float
        Log density, negative infinity outside the support.
    """
    p = _response_shape_parameters(a, "Frechet")
    point = _finite_value(value, "Frechet")
    if point <= 0:
        return -math.inf
    ratio = math.log(point) - math.log(p["scale"])
    log_power = -p["concentration"] * ratio
    if log_power > 709:
        return -math.inf
    return (
        math.log(p["concentration"])
        - math.log(p["scale"])
        - (p["concentration"] + 1) * ratio
        - math.exp(log_power)
    )


def _generalized_extreme_value_sample(
    a: Mapping[str, object], rng: random.Random
) -> object:
    """Draw a generalized extreme-value response, including zero shape.

    Parameters
    ----------
    a : Mapping[str, object]
        Location, positive scale, and real concentration.
    rng : random.Random
        Host random generator.

    Returns
    -------
    object
        A real response in the concentration-dependent interval.
    """
    p = _response_shape_parameters(a, "GeneralizedExtremeValue")
    gumbel = -math.log(max(rng.expovariate(1), math.nextafter(0, 1)))
    c = p["concentration"]
    transformed = math.expm1(c * gumbel) / c if c else gumbel
    return p["loc"] + p["scale"] * transformed


def _generalized_extreme_value_density(a: Mapping[str, object], value: object) -> float:
    """Evaluate the GEV density and its support-endpoint limits.

    Parameters
    ----------
    a : Mapping[str, object]
        Location, positive scale, and real concentration.
    value : object
        Real response.

    Returns
    -------
    float
        Log density, negative infinity outside the shape-dependent support.
    """
    p = _response_shape_parameters(a, "GeneralizedExtremeValue")
    z = (_finite_value(value, "GeneralizedExtremeValue") - p["loc"]) / p["scale"]
    c = p["concentration"]
    t = 1 + c * z
    if t < 0:
        return -math.inf
    if t == 0:
        if c == -1:
            return -math.log(p["scale"])
        return math.inf if c < -1 else -math.inf
    r = math.log1p(c * z) / c if c else z
    if -r > 709:
        return -math.inf
    return -math.log(p["scale"]) - (1 + c) * r - math.exp(-r)


def _dirichlet_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Dirichlet``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    concentration = _vector(a, "concentration", "Dirichlet")
    return _dirichlet_weights(concentration, rng)


def _dirichlet_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Dirichlet`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the support.

    Raises
    ------
    DistributionError
        If the point is not a vector of the concentration's length.
    """
    concentration = _vector(a, "concentration", "Dirichlet")
    if not isinstance(value, Sequence) or len(value) != len(concentration):
        raise DistributionError("Dirichlet is scored at a non-vector value")
    point = tuple(float(item) for item in value)
    if any(item <= 0 for item in point):
        return -math.inf
    return (
        math.lgamma(sum(concentration))
        - sum(math.lgamma(item) for item in concentration)
        + sum(
            (alpha - 1) * math.log(item)
            for alpha, item in zip(concentration, point, strict=True)
        )
    )


#: Reference samplers by family name.
_SAMPLERS: Mapping[str, Sampler] = MappingProxyType(
    {
        "Normal": _normal_sample,
        "LogNormal": _lognormal_sample,
        "HalfNormal": _halfnormal_sample,
        "Bernoulli": _bernoulli_sample,
        "Categorical": _categorical_sample,
        "Beta": _beta_sample,
        "Gamma": _gamma_sample,
        "Exponential": _exponential_sample,
        "Uniform": _uniform_sample,
        "Poisson": _poisson_sample,
        "Binomial": _binomial_sample,
        "Geometric": _geometric_sample,
        "Laplace": _laplace_sample,
        "COMPoisson": _com_poisson_sample,
        "BernsteinHazard": _bernstein_sample,
        "ExtendedBeta": _extended_beta_sample,
        "Wiener": _wiener_sample,
        "DiscreteWeibull": _discrete_weibull_sample,
        "ZeroInflatedAsymmetricLaplace": _zero_inflated_asymmetric_laplace_sample,
        "AsymmetricLaplace": _asymmetric_laplace_sample,
        "ExGaussian": _exgaussian_sample,
        "InverseGaussian": _inverse_gaussian_sample,
        "HurdleGamma": _hurdle_gamma_sample,
        "HurdleLogNormal": _hurdle_lognormal_sample,
        "Cauchy": _cauchy_sample,
        "StudentT": _studentt_sample,
        "Dirichlet": _dirichlet_sample,
        "SkewNormal": _skew_normal_sample,
        "ShiftedLogNormal": _shifted_lognormal_sample,
        "Frechet": _frechet_sample,
        "GeneralizedExtremeValue": _generalized_extreme_value_sample,
        "GeneralizedPareto": _generalized_pareto_sample,
        **_STRUCTURED_SAMPLERS,
        **_COUNT_VECTOR_SAMPLERS,
        **_COUNT_MIXTURE_SAMPLERS,
    }
)

#: Reference log densities by family name.
_DENSITIES: Mapping[str, Density] = MappingProxyType(
    {
        "Normal": _normal_density,
        "LogNormal": _lognormal_density,
        "HalfNormal": _halfnormal_density,
        "Bernoulli": _bernoulli_density,
        "Categorical": _categorical_density,
        "Beta": _beta_density,
        "Gamma": _gamma_density,
        "Exponential": _exponential_density,
        "Uniform": _uniform_density,
        "Poisson": _poisson_density,
        "Binomial": _binomial_density,
        "Geometric": _geometric_density,
        "Laplace": _laplace_density,
        "COMPoisson": _com_poisson_density,
        "BernsteinHazard": _bernstein_density,
        "ExtendedBeta": _extended_beta_density,
        "Wiener": _wiener_density,
        "DiscreteWeibull": _discrete_weibull_density,
        "ZeroInflatedAsymmetricLaplace": _zero_inflated_asymmetric_laplace_density,
        "AsymmetricLaplace": _asymmetric_laplace_density,
        "ExGaussian": _exgaussian_density,
        "InverseGaussian": _inverse_gaussian_density,
        "HurdleGamma": _hurdle_gamma_density,
        "HurdleLogNormal": _hurdle_lognormal_density,
        "Cauchy": _cauchy_density,
        "StudentT": _studentt_density,
        "Dirichlet": _dirichlet_density,
        "SkewNormal": _skew_normal_density,
        "ShiftedLogNormal": _shifted_lognormal_density,
        "Frechet": _frechet_density,
        "GeneralizedExtremeValue": _generalized_extreme_value_density,
        "GeneralizedPareto": _generalized_pareto_density,
        **_STRUCTURED_DENSITIES,
        **_COUNT_VECTOR_DENSITIES,
        **_COUNT_MIXTURE_DENSITIES,
    }
)


class ReferenceBackend:
    """Plain-Python sampling and scoring for every family of the registry.

    The scalar core lives beside the backend and the structured and
    compositional families in :mod:`quivers.qiec.reference_families`; a
    family outside the tables is reported by name rather than approximated,
    so a host provider can be installed for it.
    """

    @property
    def families(self) -> frozenset[str]:
        """The families this backend implements.

        Returns
        -------
        frozenset[str]
            The family names.
        """
        return frozenset(_SAMPLERS)

    def sample(
        self, family: str, arguments: Mapping[str, object], rng: random.Random
    ) -> object:
        """Draw one value from an implemented family.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.
        rng : random.Random
            The generator to draw from.

        Returns
        -------
        object
            The drawn value: a float, an int, a bool, or a tuple of floats.

        Raises
        ------
        DistributionError
            If the family is not implemented here or a parameter is
            missing or malformed.
        """
        sampler = _SAMPLERS.get(family)
        if sampler is None:
            raise DistributionError(
                f"the reference backend cannot sample {family}; install a "
                "distribution backend for it"
            )
        return sampler(arguments, rng)

    def log_prob(
        self, family: str, arguments: Mapping[str, object], value: object
    ) -> float:
        """Evaluate the log density of an implemented family.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.
        value : object
            The point evaluated.

        Returns
        -------
        float
            The log density, ``-inf`` outside the support.

        Raises
        ------
        DistributionError
            If the family is not implemented here or a parameter or the
            point is malformed.
        """
        density = _DENSITIES.get(family)
        if density is None:
            raise DistributionError(
                f"the reference backend cannot score {family}; install a "
                "distribution backend for it"
            )
        return density(arguments, value)

    def log_mass(self, family: str, arguments: Mapping[str, object]) -> float:
        """The log of the total mass of a construction.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The named parameters as host values.

        Returns
        -------
        float
            The family's log mass, zero for every probability family.

        Raises
        ------
        DistributionError
            If the family is not implemented here.
        """
        if family not in _DENSITIES:
            raise DistributionError(
                f"the reference backend cannot weigh {family}; install a "
                "distribution backend for it"
            )
        mass = _LOG_MASSES.get(family)
        return 0.0 if mass is None else mass(arguments)


_RNG = random.Random(0)
_backend: DistributionBackend = ReferenceBackend()


def seed_reference_rng(seed: int) -> None:
    """Reseed the generator reference draws use by default.

    Parameters
    ----------
    seed : int
        The seed; two runs seeded alike draw alike.
    """
    _RNG.seed(seed)


def distribution_backend() -> DistributionBackend:
    """The backend distribution values currently sample and score through.

    Returns
    -------
    DistributionBackend
        The installed backend; the reference backend until replaced.
    """
    return _backend


def install_distribution_backend(backend: DistributionBackend) -> DistributionBackend:
    """Replace the backend distribution values sample and score through.

    Parameters
    ----------
    backend : DistributionBackend
        The backend to install, such as one over a host library that
        covers every family.

    Returns
    -------
    DistributionBackend
        The backend that was installed before, so a caller can restore it.
    """
    global _backend
    previous = _backend
    _backend = backend
    return previous


__all__ = [
    "DistributionBackend",
    "ReferenceBackend",
    "RuntimeDistribution",
    "distribution_backend",
    "install_distribution_backend",
    "seed_reference_rng",
]
