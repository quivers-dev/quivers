"""Lift handler: draw every learned parameter from a prior.

`LiftHandler` handles the program's ``param`` instance: each parameter a
morphism reads through ``Param.get`` is answered with a draw from a
``Normal(value, prior_scale)`` prior centred on its current value, one
draw per parameter per run, and the morphism runs under the drawn
values. The draws are kept in ``sampled_params`` keyed by the site and
the parameter's qualified name.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import param_handler
from quivers.qiec.evaluator import RuntimeRequest


class LiftHandler(EffectHandler):
    """Draw every reachable parameter from a Normal prior.

    Parameters
    ----------
    prior_scale : float
        Standard deviation of the Normal prior around each parameter's
        current value.
    """

    def __init__(self, prior_scale: float = 1.0) -> None:
        self.prior_scale = float(prior_scale)
        self.sampled_params: dict[str, torch.Tensor] = {}

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a prior-drawing handler of the ``param`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared, whose parameter store holds each
            parameter's current value.

        Returns
        -------
        tuple[Installation, ...]
            The handler, which answers each parameter's first read with
            a fresh draw and later reads with the same draw.
        """
        scale = self.prior_scale
        sampled = self.sampled_params

        def draw(name: str, request: RuntimeRequest) -> object:
            """Answer a parameter read with its prior draw.

            Parameters
            ----------
            name : str
                The parameter's qualified name.
            request : RuntimeRequest
                The read, unused.

            Returns
            -------
            object
                The draw made for the name on this run.
            """
            del request
            if name not in sampled:
                current = run.parameters[name]
                sampled[name] = current.detach() + scale * torch.randn_like(current)
            return sampled[name]

        return (
            Installation(
                instance=run.kernel.param,
                runtime=param_handler(
                    draw,
                    result_validator=run.validator,
                    total=False,
                    answer_type=run.result_type,
                    key=f"lift-{id(self):x}",
                ),
            ),
        )

    def reset(self) -> None:
        """Clear the draws so the next run makes fresh ones."""
        self.sampled_params.clear()


def lift(prior_scale: float = 1.0) -> LiftHandler:
    """Return a `LiftHandler` with the given prior scale.

    Parameters
    ----------
    prior_scale : float
        Standard deviation of the prior.

    Returns
    -------
    LiftHandler
        The handler.
    """
    return LiftHandler(prior_scale)


__all__ = ["LiftHandler", "lift"]
