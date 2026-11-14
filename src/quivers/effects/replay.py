"""Replay handler: reinstall the values a trace recorded.

`ReplayHandler` answers each site the trace holds with the recorded
value and scores it under the site's current distribution, so replay
fixes the values and not the densities: the joint of the replayed run is
the model's density at the recorded values.
"""

from __future__ import annotations

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.effects.trace_types import Trace
from quivers.qiec.builtins import ExtraValuePolicy, ReplayPolicy, replay_handler


class ReplayHandler(EffectHandler):
    """Replay the values of a recorded trace.

    Parameters
    ----------
    trace : Trace
        The trace whose site values are reinstalled.
    """

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a scoring replay handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The replay handler, which forwards sites the trace lacks and
            ignores recorded sites the program never samples.
        """
        kernel = run.kernel
        values = {
            name: run.slice_value(name, site.value)
            for name, site in self._trace.sites.items()
            if not site.is_deterministic or site.morphism is not None
        }
        return (
            Installation(
                instance=kernel.random,
                runtime=replay_handler(
                    values,
                    result_validator=run.validator,
                    policy=ReplayPolicy.CLAMP_AND_SCORE,
                    score_instance=kernel.score.entry.instance,
                    extra=ExtraValuePolicy.IGNORE,
                    answer_type=run.result_type,
                    key=f"replay-{id(self):x}",
                ),
            ),
        )


def replay(trace: Trace) -> ReplayHandler:
    """Return a `ReplayHandler` replaying the given trace.

    Parameters
    ----------
    trace : Trace
        The trace.

    Returns
    -------
    ReplayHandler
        The handler.
    """
    return ReplayHandler(trace)


__all__ = ["ReplayHandler", "replay"]
