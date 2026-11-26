"""The reference machine's distributions, sampled and scored by torch.

A `quivers.qiec.distributions.RuntimeDistribution` names a family, its
arguments as tensors, and the plate it is drawn over. `TensorBackend`
builds the family's torch distribution exactly as the compiled runtime
builds the value of a ``Distribution`` term, with the plate's axes as the
distribution's leading batch axes, so the machine and the compiled runtime
score a family by the same code. A density keeps the gradient of every
argument and of the point, and a draw of a reparameterizable family keeps
the gradient of the arguments; a draw of any other family is a sample
whose law depends on the arguments, which no derivative of the draw
itself accounts for, and has none.
"""

from __future__ import annotations

import math
import random

import torch
from torch.distributions import Distribution
from torch.distributions import constraints as _constraints

from quivers.continuous._measure_queries import log_mass as measure_log_mass
from quivers.continuous.inline import _raw_distribution_value
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.plate import PlateDraw
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.qiec.distributions import (
    SHAPE_FROM_EVENT,
    DistributionError,
    RuntimeDistribution,
)
from quivers.qiec.families import family as family_record
from quivers.qiec.numbers import NumberDomain, number_kind

type TorchArgument = torch.Tensor | Distribution | tuple[TorchArgument, ...]
"""An argument of a torch family: a tensor, a distribution, or a tuple."""


def _argument(value: object, rank: int, natural: tuple[int, ...]) -> TorchArgument:
    """One argument of a runtime distribution as the torch family takes it.

    Parameters
    ----------
    value : object
        A tensor, a runtime distribution, or a tuple of either.
    rank : int
        The parameter's own rank.
    natural : tuple[int, ...]
        The family's own event extents.

    Returns
    -------
    TorchArgument
        A tensor, expanded to the parameter's own shape when it is a
        scalar given for a parameter of positive rank; a torch
        distribution for a runtime distribution.

    Raises
    ------
    DistributionError
        If the argument is neither a tensor nor a distribution.
    """
    if isinstance(value, RuntimeDistribution):
        return torch_law(value)
    if isinstance(value, tuple):
        return tuple(_argument(item, rank, natural) for item in value)
    if not isinstance(value, torch.Tensor):
        raise DistributionError(
            f"a tensor backend argument is a tensor or a distribution, not "
            f"{type(value).__name__}"
        )
    if rank == 0 or value.dim() > 0 or not natural:
        return value
    dims = natural if len(natural) == rank else tuple(natural[-1] for _ in range(rank))
    return value.expand(dims)


def torch_law(distribution: RuntimeDistribution) -> Distribution:
    """The torch distribution of a runtime distribution.

    Parameters
    ----------
    distribution : RuntimeDistribution
        The distribution.

    Returns
    -------
    torch.distributions.Distribution
        The family's distribution, its batch axes the plate's batch and
        event axes, its event axes the family's own.

    Raises
    ------
    DistributionError
        If the family has no torch constructor for its arguments.
    """
    arguments: dict[str, object] = {
        name: _argument(value, distribution.ranks.get(name, 0), distribution.natural)
        for name, value in distribution.arguments.items()
    }
    if distribution.family in SHAPE_FROM_EVENT and distribution.natural:
        arguments["dimension"] = distribution.natural[-1]
    try:
        return _raw_distribution_value(
            distribution.family,
            arguments,
            distribution.natural,
            (*distribution.batch, *distribution.event),
        )
    except (TypeError, ValueError) as error:
        raise DistributionError(
            f"{distribution.family} has no torch distribution for its arguments: "
            f"{error}"
        ) from error


def _point(value: object, distribution: Distribution) -> torch.Tensor:
    """A point as a torch distribution scores it.

    Parameters
    ----------
    value : object
        The point, as the machine holds it.
    distribution : torch.distributions.Distribution
        The distribution.

    Returns
    -------
    torch.Tensor
        The point in the dtype the distribution's support is scored in: a
        Boolean or one-hot support scores floating values, an integer
        support integers, a continuous one reals.

    Raises
    ------
    DistributionError
        If the point is not a tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise DistributionError("a tensor backend scores a tensor")
    support = distribution.support
    if support is not None and support.is_discrete:
        if isinstance(
            support,
            (
                torch.distributions.constraints._Boolean,
                torch.distributions.constraints._OneHot,
            ),
        ):
            return value.to(_parameter_dtype(distribution))
        return value if not value.dtype.is_floating_point else value.long()
    return (
        value if value.is_floating_point() else value.to(_parameter_dtype(distribution))
    )


def _parameter_dtype(distribution: Distribution) -> torch.dtype:
    """The floating dtype a distribution computes in.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        The distribution.

    Returns
    -------
    torch.dtype
        The dtype of its mean, or the default floating dtype when it has
        none.
    """
    try:
        mean = distribution.mean
    except NotImplementedError:
        return torch.get_default_dtype()
    return mean.dtype if mean.is_floating_point() else torch.get_default_dtype()


def _draw(value: torch.Tensor, kind: str | None, numbers: NumberDomain) -> torch.Tensor:
    """A torch draw in the dtype of the family's element.

    Parameters
    ----------
    value : torch.Tensor
        The draw.
    kind : str | None
        The kind of number the family draws.
    numbers : NumberDomain
        The run's numbers.

    Returns
    -------
    torch.Tensor
        ``bool`` for truth values, ``int64`` for integers, the run's real
        dtype for reals, keeping a reparameterized draw's gradient.
    """
    if kind == "truth":
        return value != 0
    if kind == "integer":
        return value.to(torch.int64)
    return value.to(numbers.real_dtype)


class TensorBackend:
    """Sampling and scoring by torch, with gradients.

    Every density is the family's torch ``log_prob`` at once over the
    plate, with the plate's event axes summed; draws use torch's global
    generator.
    """

    @property
    def differentiable(self) -> bool:
        """Yes: densities and reparameterized draws keep their gradients.

        Returns
        -------
        bool
            ``True``.
        """
        return True

    def sample(self, distribution: RuntimeDistribution, rng: random.Random) -> object:
        """Draw at every position of the plate.

        Parameters
        ----------
        distribution : RuntimeDistribution
            The distribution.
        rng : random.Random
            Unused: torch draws from its own generator.

        Returns
        -------
        object
            The draw, reparameterized when the family allows it.

        Raises
        ------
        DistributionError
            If the family has no torch distribution for its arguments.
        """
        del rng
        native = torch_law(distribution)
        drawn = native.rsample() if native.has_rsample else native.sample()
        kind = number_kind(family_record(distribution.family).element)
        return _draw(drawn, kind, distribution.numbers)

    def log_prob(
        self, distribution: RuntimeDistribution, value: object, keep_batch: bool
    ) -> torch.Tensor:
        """Evaluate the log density at a point.

        Parameters
        ----------
        distribution : RuntimeDistribution
            The distribution.
        value : object
            The point.
        keep_batch : bool
            Whether to keep one density per batch position.

        Returns
        -------
        torch.Tensor
            The density in the run's real dtype, its plate event axes
            summed, its batch axes summed unless kept.

        Raises
        ------
        DistributionError
            If the family has no torch distribution for its arguments, or
            the point is not shaped like the plate.
        """
        native = torch_law(distribution)
        density = native.log_prob(_point(value, native))
        kept = len(distribution.batch) if keep_batch else 0
        if density.dim() < kept:
            raise DistributionError(
                f"a point of {distribution.family} scored under a plate of "
                f"{kept} batch axes has density shape {tuple(density.shape)}"
            )
        if density.dim() > kept:
            density = density.sum(dim=tuple(range(kept, density.dim())))
        return density.to(distribution.numbers.real_dtype)

    def log_mass(
        self, distribution: RuntimeDistribution, batch_axes: int
    ) -> torch.Tensor:
        """The log total mass of a construction.

        Parameters
        ----------
        distribution : RuntimeDistribution
            The distribution.
        batch_axes : int
            The leading batch axes kept.

        Returns
        -------
        torch.Tensor
            The log mass in the run's real dtype.

        Raises
        ------
        DistributionError
            If the family has no torch distribution for its arguments.
        """
        mass = measure_log_mass(torch_law(distribution), batch_axes=batch_axes)
        return torch.as_tensor(mass).to(distribution.numbers.real_dtype)


class DrawnLaw(ContinuousMorphism):
    """The law a site of a checked program was drawn from, as a morphism.

    A site of a program that runs on the reference machine draws from a
    `RuntimeDistribution` its run constructs. A trace records the site's
    law as the morphism its run drew from, so a latent registry reads its
    support and event shape, and a guide its bijector, as for a site of a
    compiled program. The morphism has no input: its law is the one the
    run constructed, and sampling and scoring it repeat that law.

    Parameters
    ----------
    distribution : RuntimeDistribution
        The distribution the site was drawn from, at one position of its
        plate.
    native : torch.distributions.Distribution
        Its torch distribution.
    """

    def __init__(self, distribution: RuntimeDistribution, native: Distribution) -> None:
        natural = tuple(int(size) for size in native.event_shape)
        codomain = Euclidean(
            name=distribution.family,
            dim=math.prod(natural),
            intrinsic_shape=natural,
        )
        super().__init__(Unit, codomain)
        self._native = native
        self._family = distribution.family

    @property
    def family_name(self) -> str:
        """The family's source name.

        Returns
        -------
        str
            The name.
        """
        return self._family

    @property
    def support(self) -> _constraints.Constraint:
        """The support of the law.

        Returns
        -------
        torch.distributions.constraints.Constraint
            The torch distribution's support.
        """
        support = self._native.support
        assert support is not None
        return support

    def rsample(  # type: ignore[override]
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw from the law.

        Parameters
        ----------
        x : torch.Tensor
            Unused: the law has no input.
        sample_shape : torch.Size
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Reparameterized draws when the family allows them.
        """
        del x
        if self._native.has_rsample:
            return self._native.rsample(sample_shape)
        return self._native.sample(sample_shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The log density of a value.

        Parameters
        ----------
        x : torch.Tensor
            Unused: the law has no input.
        y : torch.Tensor
            The value.

        Returns
        -------
        torch.Tensor
            The density at each position of the law's batch.
        """
        del x
        return self._native.log_prob(_point(y, self._native))


def site_morphism(distribution: RuntimeDistribution) -> ContinuousMorphism:
    """The morphism a trace records for a site drawn from a distribution.

    Parameters
    ----------
    distribution : RuntimeDistribution
        The distribution the site was drawn from.

    Returns
    -------
    ContinuousMorphism
        A `DrawnLaw` of one position, inside a `PlateDraw` over the
        distribution's plate when it has one, so a registry reads the
        plate's positions and each position's event as it does for a
        plated site of a compiled program.

    Raises
    ------
    DistributionError
        If the family has no torch distribution for its arguments.
    """
    native = torch_law(distribution)
    plate = (*distribution.batch, *distribution.event)
    if not plate:
        return DrawnLaw(distribution, native)
    return PlateDraw(
        math.prod(plate),
        DrawnLaw(distribution, native),
        index_shape=plate,
    )


__all__ = ["DrawnLaw", "TensorBackend", "site_morphism", "torch_law"]
