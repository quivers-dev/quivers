"""Scale handler: multiply every density contribution by a factor.

`ScaleHandler` is a transformer of the program's ``score`` instance:
every contribution any site makes passes through it on the way to the
run's accumulator, multiplied by the factor. Stacking ``scale`` with
``mask`` composes the two elementwise products.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import reweight_handler


class ScaleHandler(EffectHandler):
    """Multiply every density contribution by a scalar.

    Parameters
    ----------
    factor : float
        The multiplier.
    """

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a scaling transformer of the ``score`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The transformer.
        """
        factor = self.factor

        def scaled(weight: object) -> object:
            """Scale one contribution.

            Parameters
            ----------
            weight : object
                The contribution.

            Returns
            -------
            object
                The contribution times the factor.
            """
            return torch.as_tensor(weight) * factor

        return (
            Installation(
                instance=run.kernel.score,
                runtime=reweight_handler(
                    scaled,
                    score_instance=run.kernel.score.entry.instance,
                    answer_type=run.result_type,
                    key=f"scale-{id(self):x}",
                ),
            ),
        )


def scale(factor: float) -> ScaleHandler:
    """Return a `ScaleHandler` with the given factor.

    Parameters
    ----------
    factor : float
        The multiplier.

    Returns
    -------
    ScaleHandler
        The handler.
    """
    return ScaleHandler(factor)


__all__ = ["ScaleHandler", "scale"]
