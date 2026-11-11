"""Execution trace for monadic programs.

A `Trace` records every sample site visited during program
execution, capturing the morphism, sampled or observed value, and
log-density at each site. This is the foundation for all inference
algorithms: SVI uses traces to compute the ELBO, and conditioning
operates by clamping trace sites to observed data.

The `trace(program, x, observations)` free function is a thin
wrapper on top of the effect-handler machinery: it pushes a
`quivers.effects.trace_handler.TraceHandler` onto the active
handler stack, invokes `quivers.effects.interpreter.run_program`,
and returns the accumulated `Trace`. Any other handlers active in
the enclosing scope compose with `TraceHandler` in the standard
outer-to-inner order.

The `SampleSite` and `Trace` data types are defined in
`quivers.effects.trace_types` and re-exported here.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.effects.interpreter import run_program
from quivers.effects.trace_handler import TraceHandler
from quivers.effects.trace_types import SampleSite, Trace


__all__ = ["SampleSite", "Trace", "trace"]


def trace(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor] | None = None,
) -> Trace:
    """Execute a program and record all sample sites.

    A thin wrapper around the handler-aware interpreter: pushes a
    `quivers.effects.trace_handler.TraceHandler` onto the active
    handler stack, delegates to
    `quivers.effects.interpreter.run_program`, and returns the
    accumulated `Trace` (with ``output`` and ``log_joint`` filled
    in). Any other handlers already on the stack compose with
    trace in the standard outer-to-inner order.

    Parameters
    ----------
    program : MonadicProgram
        The program to trace.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    observations : dict[str, torch.Tensor] or None
        Values to clamp observed variables to. Keys are variable
        names, values are tensors of the appropriate shape.

    Returns
    -------
    Trace
        Complete execution trace with all sites, output, and
        log-joint.
    """
    with TraceHandler() as handler:
        output = run_program(program, x, observations)
    handler.trace.output = output
    handler.trace.log_joint = handler.total_log_joint(x.device)
    return handler.trace
