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
from dataclasses import dataclass
import math
import random
from types import MappingProxyType
from typing import Protocol

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
    log_choose as _log_choose,
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


@dataclass(frozen=True, slots=True)
class RuntimeDistribution:
    """A constructed distribution, as host data.

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
    arguments: Mapping[str, object]
    batch: tuple[int, ...] = ()
    event: tuple[int, ...] = ()
    ranks: Mapping[str, int] = MappingProxyType({})
    natural: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the arguments so a distribution value cannot drift."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "ranks", MappingProxyType(dict(self.ranks)))

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
            self.family, arguments, batch, event, self.ranks, self.natural
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

    def log_mass(self) -> float:
        """The log of the distribution's total mass through the installed backend.

        Returns
        -------
        float
            Zero for a probability measure; over a plate, the sum of the
            masses at every position, since the plate's draw is their
            product.
        """
        if not self.plated:
            return _backend.log_mass(self.family, self._at(()))
        shape = (*self.batch, *self.event)
        return math.fsum(
            _backend.log_mass(self.family, self._at(position))
            for position in _positions(shape)
        )

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
    return log_success if bool(_finite_value(value, "Bernoulli")) else log_failure


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
    index = int(_finite_value(value, "Categorical"))
    if not 0 <= index < len(probs) or probs[index] <= 0:
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
    rate = _real(a, "rate", "Poisson")
    # Knuth's method, exact for the moderate rates a reference run uses.
    limit = math.exp(-rate)
    count = 0
    product = rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


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
    count = int(_finite_value(value, "Poisson"))
    rate = _real(a, "rate", "Poisson")
    if count < 0:
        return -math.inf
    if rate == 0.0:
        return 0.0 if count == 0 else -math.inf
    return count * math.log(rate) - rate - math.lgamma(count + 1)


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
    trials = int(_real(a, "total_count", "Binomial"))
    probability = _probabilities(a, "Binomial")
    return sum(1 for _ in range(trials) if rng.random() < probability)


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
    count = int(_finite_value(value, "Binomial"))
    trials = int(_real(a, "total_count", "Binomial"))
    probability = _probabilities(a, "Binomial")
    if not 0 <= count <= trials:
        return -math.inf
    return (
        _log_choose(trials, count)
        + count * math.log(probability)
        + (trials - count) * math.log1p(-probability)
    )


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
    """
    probability = _probabilities(a, "Geometric")
    failures = 0
    while rng.random() >= probability:
        failures += 1
    return failures


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
    failures = int(_finite_value(value, "Geometric"))
    probability = _probabilities(a, "Geometric")
    if failures < 0:
        return -math.inf
    return failures * math.log1p(-probability) + math.log(probability)


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
    draws = [rng.gammavariate(item, 1.0) for item in concentration]
    total = sum(draws)
    return tuple(item / total for item in draws)


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
        "Cauchy": _cauchy_sample,
        "StudentT": _studentt_sample,
        "Dirichlet": _dirichlet_sample,
        **_STRUCTURED_SAMPLERS,
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
        "Cauchy": _cauchy_density,
        "StudentT": _studentt_density,
        "Dirichlet": _dirichlet_density,
        **_STRUCTURED_DENSITIES,
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
