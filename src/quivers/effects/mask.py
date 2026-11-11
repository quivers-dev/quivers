"""Mask handler: gate every density contribution elementwise.

`MaskHandler` is a transformer of the program's ``score`` instance: the
mask is broadcast against every contribution on its way to the run's
accumulator. A boolean mask switches entries; a float mask scales them.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import reweight_handler


class MaskHandler(EffectHandler):
    """Gate every density contribution elementwise.

    Parameters
    ----------
    mask_tensor : torch.Tensor
        The mask, broadcast against each contribution. For a call on an
        input of ``B > 1`` rows, a mask whose leading axis has length
        ``B`` gates each run by its own entry; any other mask applies to
        every run.
    """

    def __init__(self, mask_tensor: torch.Tensor) -> None:
        self.mask = mask_tensor

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a masking transformer of the ``score`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The transformer.
        """
        mask = self.mask
        if (
            not run.vectorized
            and run.runs > 1
            and mask.dim() >= 1
            and int(mask.shape[0]) == run.runs
        ):
            # A mask laid out one entry per run gates each run by its own.
            mask = mask[run.run_index : run.run_index + 1]

        def masked(weight: object) -> object:
            """Mask one contribution.

            Parameters
            ----------
            weight : object
                The contribution.

            Returns
            -------
            object
                The contribution times the mask.
            """
            return torch.as_tensor(weight) * mask.to(torch.as_tensor(weight).dtype)

        return (
            Installation(
                run.kernel.score,
                reweight_handler(
                    masked,
                    score_instance=run.kernel.score.entry.instance,
                    answer_type=run.result_type,
                    key=f"mask-{id(self):x}",
                ),
            ),
        )


def mask(mask_tensor: torch.Tensor) -> MaskHandler:
    """Return a `MaskHandler` with the given mask.

    Parameters
    ----------
    mask_tensor : torch.Tensor
        The mask.

    Returns
    -------
    MaskHandler
        The handler.
    """
    return MaskHandler(mask_tensor)


__all__ = ["MaskHandler", "mask"]
