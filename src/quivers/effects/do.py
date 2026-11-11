"""Intervention handler: fix named sites without scoring them.

`DoHandler` implements Pearl's ``do`` operator on the effect stack. A
site whose name appears in its data is answered with the given value and
contributes nothing to the joint: under intervention its distribution
is replaced by a point mass. Downstream sites see the intervened value,
and a trace records the site as deterministic.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import intervene_handler
from quivers.qiec.evaluator import RuntimeRequest


class DoHandler(EffectHandler):
    """Intervene on named sites.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site name to the value the site is fixed to.
    """

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = dict(data)

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install an intervening handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared; each intervened site's address is
            recorded on it, so a trace marks the site deterministic.

        Returns
        -------
        tuple[Installation, ...]
            The intervening handler.
        """

        def mark(request: RuntimeRequest, value: object) -> None:
            """Record an intervened site on the run.

            Parameters
            ----------
            request : RuntimeRequest
                The site's request.
            value : object
                The value fixed, unused.
            """
            del value
            run.interventions.add(run.key_of(request))

        return (
            Installation(
                run.kernel.random,
                intervene_handler(
                    {
                        name: run.slice_value(name, value)
                        for name, value in self.data.items()
                    },
                    result_validator=run.validator,
                    on_intervene=mark,
                    answer_type=run.result_type,
                    key=f"do-{id(self):x}",
                ),
            ),
        )


def do(data: dict[str, torch.Tensor]) -> DoHandler:
    """Return a `DoHandler` intervening on the given site values.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site name to value.

    Returns
    -------
    DoHandler
        The handler.
    """
    return DoHandler(data)


__all__ = ["DoHandler", "do"]
