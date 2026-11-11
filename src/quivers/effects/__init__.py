"""Algebraic effect handlers for probabilistic programs.

Effect handlers in the Pyro ``poutine`` / NumPyro ``handlers`` shape,
grounded in the algebraic-effects calculus of
[Plotkin and Pretnar 2009](https://doi.org/10.1007/978-3-642-00590-9_7)
and its probabilistic application in
[Scibior et al. 2018](https://doi.org/10.1145/3236778) and
[Nguyen et al. 2023](https://doi.org/10.1145/3609026.3609729), and
executed as lexical handlers of the Quivers Indexed Effect Core: a
`MonadicProgram` runs on the reference machine as the kernel computation
`quivers.effects.program_module` encodes it to, and every handler here
is a handler of that computation's ``random``, ``score``, or ``param``
instance.

The `EffectHandler` ABC and the thread-local handler stack live in
`quivers.effects.base`. Concrete handlers (`TraceHandler`,
`ClampHandler`, `DoHandler`, `MaskHandler`, `ScaleHandler`,
`BlockHandler`, `ReplayHandler`, `LiftHandler`, `CollapseHandler`)
live in per-file modules and are re-exported here along with their
short-name factories. The `reparam` subpackage collects the
reparameterisation strategies.

The `clamp` handler is the effect-stack analogue of Pyro's
`condition`; the name avoids collision with the top-level
[`quivers.inference.conditioning.condition`][quivers.inference.conditioning.condition]
factory, which returns a
[`Conditioned`][quivers.inference.conditioning.Conditioned]
model wrapper. Both compose observations onto a model; pick
`clamp` when writing a handler stack, `condition` when writing a
top-level `Conditioned` object.

The handler-aware interpreter `run_program` lives in
`quivers.effects.interpreter`; the thin `quivers.inference.trace.trace`
wrapper stacks a `TraceHandler` on and returns the recorded trace.
"""

from __future__ import annotations

from quivers.effects import (
    base as _base,
    block as _block,
    checked_program as _checked_program,
    clamp as _clamp,
    collapse as _collapse,
    conjugate as _conjugate,
    do as _do,
    interpreter as _interpreter,
    lift as _lift,
    mask as _mask,
    program_module as _program_module,
    replay as _replay,
    scale as _scale,
    sites as _sites,
    subsample as _subsample,
    trace_handler as _trace_handler,
    trace_types as _trace_types,
)
from quivers.effects.base import *  # noqa: F403
from quivers.effects.block import *  # noqa: F403
from quivers.effects.checked_program import *  # noqa: F403
from quivers.effects.clamp import *  # noqa: F403
from quivers.effects.collapse import *  # noqa: F403
from quivers.effects.conjugate import *  # noqa: F403
from quivers.effects.do import *  # noqa: F403
from quivers.effects.interpreter import *  # noqa: F403
from quivers.effects.lift import *  # noqa: F403
from quivers.effects.mask import *  # noqa: F403
from quivers.effects.program_module import *  # noqa: F403
from quivers.effects.replay import *  # noqa: F403
from quivers.effects.scale import *  # noqa: F403
from quivers.effects.sites import *  # noqa: F403
from quivers.effects.subsample import *  # noqa: F403
from quivers.effects.trace_handler import *  # noqa: F403
from quivers.effects.trace_types import *  # noqa: F403
from quivers.effects.reparam import (
    ConjugateReparam,
    LocScaleReparam,
    NeuTraReparam,
    Reparam,
    ReparamOrchestrator,
    SiteRequest,
    TransformReparam,
    reparam,
)

__all__: list[str] = [
    "ConjugateReparam",
    "LocScaleReparam",
    "NeuTraReparam",
    "Reparam",
    "ReparamOrchestrator",
    "SiteRequest",
    "TransformReparam",
    "reparam",
]
__all__ += _base.__all__
__all__ += _block.__all__
__all__ += _checked_program.__all__
__all__ += _clamp.__all__
__all__ += _collapse.__all__
__all__ += _conjugate.__all__
__all__ += _do.__all__
__all__ += _interpreter.__all__
__all__ += _lift.__all__
__all__ += _mask.__all__
__all__ += _program_module.__all__
__all__ += _replay.__all__
__all__ += _scale.__all__
__all__ += _sites.__all__
__all__ += _subsample.__all__
__all__ += _trace_handler.__all__
__all__ += _trace_types.__all__
