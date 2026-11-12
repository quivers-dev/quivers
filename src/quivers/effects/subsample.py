"""Subsampled plates: score a plate's sites at a subset of its rows, rescaled.

A program whose observations live in a plate scores every row of the plate
on every run. Stochastic variational inference
([Hoffman et al. 2013](https://jmlr.org/papers/v14/hoffman13a.html)) scores
a random subset of the rows instead and multiplies their contribution by
the plate's size over the subset's size, which keeps the objective and its
gradient unbiased.

`SubsampleHandler` does this for one plate. Within its block, every site
whose batch axes include the plate, a plate latent or an observation, is
scored at the subset's rows only, and its contribution is multiplied by
``N / B`` for a plate of ``N`` rows and a subset of ``B``. A site outside
the plate, a global latent among them, is scored in full and unscaled, and
so is a ``score`` step, which is a factor of the whole run. Handlers for
several plates compose: a site in two subsampled plates is scored at the
product of the two subsets and scaled by the product of the two factors,
and a site in one of two sibling plates by its own plate's factor alone.

Within a run a subsampled plate keeps its declared extent: its latents are
drawn at every row and the program's lets compute every row, so the rows
outside the subset keep well-defined values and only their scores are left
out. An observation may be given at the plate's extent or at the subset's
size, its rows then read in the order of the index; a plate latent is
given at the plate's extent.

The guides that factor over a plate's rows score a latent in a subsampled
plate at the same rows with the same factor
([`AutoNormalGuide`][quivers.inference.guides.AutoNormalGuide], and
[`AutoDeltaGuide`][quivers.inference.guides.AutoDeltaGuide], whose density
is constant); every other guide refuses such a latent, since its density
does not split over the rows. MCMC, the Laplace approximation, annealed
importance sampling, and the Bayesian lift target the exact posterior and
refuse to run inside the block.
"""

from __future__ import annotations

import torch

from quivers.continuous.program_steps import Draw, Observe
from quivers.continuous.programs import MonadicProgram
from quivers.effects.base import (
    EffectHandler,
    Installation,
    RunContext,
    _handler_stack,
)


class SubsampleHandler(EffectHandler):
    """Score one plate's sites at a subset of its rows, rescaled.

    Parameters
    ----------
    plate : str
        The plate's name: the object its sites range over, as the
        program declares it (``Item`` for ``observe y : Item <- ...``).
    index : torch.Tensor
        The rows scored: a nonempty one-dimensional integer tensor of
        distinct rows in increasing order.

    Raises
    ------
    ValueError
        If ``index`` is not a nonempty one-dimensional tensor of distinct
        nonnegative integers in increasing order.
    """

    def __init__(self, plate: str, index: torch.Tensor) -> None:
        rows = torch.as_tensor(index)
        if rows.dim() != 1 or rows.numel() == 0:
            raise ValueError(
                f"subsample({plate!r}): the index must be a nonempty "
                f"one-dimensional tensor of rows, got shape {tuple(rows.shape)}"
            )
        if (
            rows.dtype.is_floating_point
            or rows.dtype.is_complex
            or (rows.dtype == torch.bool)
        ):
            raise ValueError(
                f"subsample({plate!r}): the index must hold integer rows, "
                f"got dtype {rows.dtype}"
            )
        rows = rows.long()
        if int(rows.min()) < 0:
            raise ValueError(f"subsample({plate!r}): the index holds a negative row")
        if rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()):
            raise ValueError(
                f"subsample({plate!r}): the index must list distinct rows in "
                "increasing order, so that a value given at the subset's size "
                "reads its rows unambiguously"
            )
        self.plate = plate
        self.index = rows

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Register the plate's subset on the run.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            No lexical handler: the run scores the plate's sites at the
            subset as it builds them.

        Raises
        ------
        ValueError
            If another handler on the stack subsamples the same plate, no
            site of the program ranges over the plate, or the index
            leaves the plate.
        """
        if self.plate in run.subsamples:
            raise ValueError(
                f"subsample({self.plate!r}) is active twice; a plate takes one "
                "subset per run"
            )
        extents = plate_extents(run)
        if self.plate not in extents:
            known = ", ".join(repr(name) for name in sorted(extents)) or "none"
            raise ValueError(
                f"subsample({self.plate!r}): no site of the program ranges over "
                f"a plate of that name (its plates: {known}). A program compiled "
                "from QVR records each site's plates in `site_plates`; a "
                "program built in Python records them there by hand"
            )
        extent = extents[self.plate]
        if int(self.index.max()) >= extent:
            raise ValueError(
                f"subsample({self.plate!r}): the index holds row "
                f"{int(self.index.max())}, beyond the plate's {extent} rows"
            )
        run.subsamples[self.plate] = self.index
        return ()


def plate_extents(run: RunContext) -> dict[str, int]:
    """The extent of every plate a run's sites range over.

    Parameters
    ----------
    run : RunContext
        The run.

    Returns
    -------
    dict[str, int]
        Each plate's number of rows, by name, read from the
        `MonadicProgram.site_plates` of the programs the run executes.

    Raises
    ------
    ValueError
        If two sites give one plate different extents.
    """
    extents: dict[str, int] = {}
    for step in run.kernel.steps:
        if step.kind != "draw" or not isinstance(step.spec, Draw | Observe):
            continue
        for name, extent in site_plates(step.program, step.spec):
            if extent is None:
                continue
            if extents.setdefault(name, extent) != extent:
                raise ValueError(
                    f"plate {name!r} has {extents[name]} rows at one site and "
                    f"{extent} at another"
                )
    return extents


def site_plates(
    program: MonadicProgram, spec: Draw | Observe
) -> tuple[tuple[str, int | None], ...]:
    """The plates a draw or observe step's site ranges over.

    Parameters
    ----------
    program : MonadicProgram
        The program the step belongs to.
    spec : Draw | Observe
        The step.

    Returns
    -------
    tuple[tuple[str, int], ...]
        The site's plate axes, outermost first, as ``(name, extent)``
        pairs, the extent ``None`` where the program leaves it to its
        data; empty for a site in no plate.
    """
    return program.site_plates.get(spec.names[0], ())


def subsampled_rows(
    site: str,
    axes: tuple[tuple[str, int | None], ...],
    subsets: dict[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], float] | None:
    """The rows a site is scored at, and its factor, under subsampled plates.

    Parameters
    ----------
    site : str
        The site's label, for the diagnostic.
    axes : tuple[tuple[str, int | None], ...]
        The site's plate axes, outermost first, as `site_plates` gives
        them.
    subsets : dict[str, torch.Tensor]
        The rows kept of each subsampled plate, by name.

    Returns
    -------
    tuple[tuple[torch.Tensor, ...], float] | None
        One index per plate axis (the subset's rows along a subsampled
        axis, every row along another) and the product over the
        subsampled axes of the axis's extent over the subset's size;
        ``None`` for a site in no subsampled plate.

    Raises
    ------
    ValueError
        If a subsampled plate of the site has no fixed extent, or the
        subset leaves it.
    """
    if not any(name in subsets for name, _ in axes):
        return None
    rows: list[torch.Tensor] = []
    factor = 1.0
    for name, extent in axes:
        if extent is None:
            raise ValueError(
                f"site {site!r} ranges over plate {name!r}, whose extent the "
                "program leaves to its data; a subsampled plate needs a fixed "
                "number of rows"
            )
        index = subsets.get(name)
        if index is None:
            rows.append(torch.arange(extent))
            continue
        if int(index.max()) >= extent:
            raise ValueError(
                f"subsample({name!r}): the index holds row {int(index.max())}, "
                f"beyond the {extent} rows of the plate at site {site!r}"
            )
        rows.append(index)
        factor *= extent / int(index.numel())
    return tuple(rows), factor


def active_subsets() -> dict[str, torch.Tensor]:
    """The rows kept of each plate the current thread's stack subsamples.

    Returns
    -------
    dict[str, torch.Tensor]
        The index of each subsampled plate, by name.

    Raises
    ------
    ValueError
        If two active handlers subsample one plate.
    """
    subsets: dict[str, torch.Tensor] = {}
    for handler in active_subsamples():
        if handler.plate in subsets:
            raise ValueError(
                f"subsample({handler.plate!r}) is active twice; a plate takes "
                "one subset per run"
            )
        subsets[handler.plate] = handler.index
    return subsets


def subsample(plate: str, index: torch.Tensor) -> SubsampleHandler:
    """Return a `SubsampleHandler` scoring a plate at a subset of its rows.

    Parameters
    ----------
    plate : str
        The plate's name.
    index : torch.Tensor
        The rows scored, distinct and in increasing order.

    Returns
    -------
    SubsampleHandler
        The handler, used as a context manager around the calls that
        score the program (an `SVI.step`, a `trace`, a `log_joint`).
    """
    return SubsampleHandler(plate, index)


def active_subsamples() -> tuple[SubsampleHandler, ...]:
    """The subsample handlers on the current thread's stack, outer first.

    Returns
    -------
    tuple[SubsampleHandler, ...]
        The active handlers.
    """
    return tuple(
        handler for handler in _handler_stack() if isinstance(handler, SubsampleHandler)
    )


def refuse_subsampling(consumer: str) -> None:
    """Refuse a subsampled plate in a consumer of the exact posterior.

    Parameters
    ----------
    consumer : str
        The consumer's name, for the diagnostic.

    Raises
    ------
    ValueError
        If any subsample handler is active.
    """
    active = active_subsamples()
    if active:
        plates = ", ".join(repr(handler.plate) for handler in active)
        raise ValueError(
            f"{consumer} targets the exact posterior, but the plate(s) {plates} "
            "are subsampled: a subsampled plate gives an unbiased estimate of "
            "the log joint, which only a stochastic objective such as the ELBO "
            "under SVI can use. Run it outside the subsample block"
        )


__all__ = ["SubsampleHandler", "subsample"]
