"""Clamp handler: condition named sample sites on given values.

`ClampHandler` is the effect-stack analogue of Pyro's ``condition``. A
site whose name appears in its data is answered with the given value
and scored under its own distribution, so the value contributes the
site's density to the joint and the trace records the site as observed.
The name avoids collision with the top-level
[`quivers.inference.conditioning.condition`][quivers.inference.conditioning.condition]
factory, which returns a `Conditioned` model wrapper.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping
from typing import cast

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import (
    ExtraValuePolicy,
    MissingValuePolicy,
    condition_handler,
)


class ClampHandler(EffectHandler):
    """Condition named sample sites on given values.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site name to the value the site is clamped to. Other sites pass
        through untouched under the default missing-value policy.
    missing : MissingValuePolicy
        Whether an unprovided random request is forwarded or refused. ERROR
        also consults the run's declared observations, including computed and
        scanned values, and refuses a genuinely missing random value.
    """

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        *,
        missing: MissingValuePolicy = MissingValuePolicy.FORWARD,
    ) -> None:
        self.data = dict(data)
        self.missing = missing

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a conditioning handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The conditioning handler with the selected missing-value policy.
            Values at sites the program never samples are ignored.
        """
        kernel = run.kernel
        return (
            Installation(
                instance=kernel.random,
                runtime=condition_handler(
                    cast(
                        Mapping[object, object],
                        ChainMap(
                            {
                                name: run.slice_value(name, value)
                                for name, value in self.data.items()
                            },
                            run.observations
                            if self.missing is MissingValuePolicy.ERROR
                            else {},
                        ),
                    ),
                    score_instance=kernel.score.entry.instance,
                    result_validator=run.validator,
                    missing=self.missing,
                    extra=ExtraValuePolicy.IGNORE,
                    answer_type=run.result_type,
                    key=f"clamp-{id(self):x}",
                    at_run=run.select_run,
                ),
            ),
        )


def clamp(data: dict[str, torch.Tensor]) -> ClampHandler:
    """Return a `ClampHandler` conditioning on the given site values.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site name to value.

    Returns
    -------
    ClampHandler
        The handler.
    """
    return ClampHandler(data)


__all__ = ["ClampHandler", "clamp"]
