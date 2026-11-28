"""Execution trace for monadic programs.

A `Trace` records every sample site visited during program
execution, capturing the morphism, sampled or observed value, and
log-density at each site. This is the foundation for all inference
algorithms: SVI uses traces to compute the ELBO, and conditioning
operates by clamping trace sites to observed data.

The `trace(program, x, observations)` free function is a thin
wrapper on top of the effect-handler machinery: it pushes a
`quivers.effects.trace_handler.TraceHandler` onto the active
handler stack, runs the program forward once by its own ``rsample``
(a compiled program on its kernel encoding, a checked program on its
checked computation), and returns the accumulated `Trace`. Any other handlers active in
the enclosing scope compose with `TraceHandler` in the standard
outer-to-inner order.

The `SampleSite` and `Trace` data types are defined in
`quivers.effects.trace_types` and re-exported here.
"""

from __future__ import annotations

from typing import cast

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.effects.trace_handler import TraceHandler
from quivers.effects.trace_types import SampleSite, Trace


__all__ = ["SampleSite", "Trace", "trace"]


def trace(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, object] | None = None,
) -> Trace:
    """Execute a program and record all sample sites.

    A thin wrapper around the handler-aware interpreter: pushes a
    `quivers.effects.trace_handler.TraceHandler` onto the active
    handler stack, runs the program forward once by its own
    ``rsample``, and returns the
    accumulated `Trace` (with ``output`` and ``log_joint`` filled
    in). Any other handlers already on the stack compose with
    trace in the standard outer-to-inner order.

    Parameters
    ----------
    program : MonadicProgram
        The program to trace.
    x : torch.Tensor
        Program input, shape ``(B, ...)``: ``B`` independent runs of
        the program.
    observations : dict[str, object] or None
        Data and clamped values by name. Tensor values may have a leading
        run axis of length ``B``, or be shared across runs. Checked programs
        can accept typed host data, including a finite structured law as
        an argument and one of its points as a site value.

    Returns
    -------
    Trace
        Complete execution trace with densities and joint on the run
        axis. A tensor site lays its value out on that axis. A finite
        structured site holds its typed value directly and requires one run.
    """
    with TraceHandler() as handler:
        output = program.rsample(
            x, torch.Size(), cast(dict[str, torch.Tensor] | None, observations)
        )
    handler.trace = handler.trace.with_(
        output=output, log_joint=handler.total_log_joint(x.device)
    )
    return handler.trace
