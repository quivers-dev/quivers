"""The let calls each Python target spells as one call.

The Python renderers spell a math builtin through its native path when
it has one, and a reduction or ``softmax`` through the target's array
library; any other math builtin has no spelling there. The transpile
planner reads the same table to decide whether an inlined computation's
expressions can be stated on the target.
"""

from __future__ import annotations

from quivers.dsl.pure_builtins import EAGER_BUILTIN_NAMES
from quivers.transpile._builtin_spellings import native_path

#: A non-pointwise builtin's per-target lowering: the symbol path to
#: emit as a dotted attribute, plus the import it needs (``None`` when the
#: symbol lives on a namespace the target already has in scope). The
#: pointwise math builtins are spelled by
#: [`BUILTIN_SPELLINGS`][quivers.transpile._builtin_spellings.BUILTIN_SPELLINGS].
CallEntry = tuple[tuple[str, ...], "tuple[tuple[str, ...], str] | None"]


#: ``target -> {reduction name -> lowering}`` for the one-argument
#: reductions and ``softmax``. A one-argument ``max`` / ``min`` routes to
#: the reduction that returns a bare array (torch's ``amax`` / ``amin``
#: rather than the ``(values, indices)`` pair).
LET_CALL_SYMBOLS: dict[str, dict[str, CallEntry]] = {
    "pyro": {
        "sum": (("torch", "sum"), None),
        "mean": (("torch", "mean"), None),
        "prod": (("torch", "prod"), None),
        "max": (("torch", "amax"), None),
        "min": (("torch", "amin"), None),
        "softmax": (("torch", "softmax"), None),
    },
    "numpyro": {
        "sum": (("jnp", "sum"), None),
        "mean": (("jnp", "mean"), None),
        "prod": (("jnp", "prod"), None),
        "max": (("jnp", "max"), None),
        "min": (("jnp", "min"), None),
        "softmax": (("jnn", "softmax"), (("jax", "nn"), "jnn")),
    },
    "pymc": {
        "sum": (("pymc", "math", "sum"), None),
        "mean": (("pymc", "math", "mean"), None),
        "prod": (("pymc", "math", "prod"), None),
        "max": (("pymc", "math", "max"), None),
        "min": (("pymc", "math", "min"), None),
        "softmax": (("pymc", "math", "softmax"), None),
    },
    "edward2": {
        "sum": (("tf", "reduce_sum"), None),
        "mean": (("tf", "reduce_mean"), None),
        "prod": (("tf", "reduce_prod"), None),
        "max": (("tf", "reduce_max"), None),
        "min": (("tf", "reduce_min"), None),
        "softmax": (("tf", "nn", "softmax"), None),
    },
}


#: The targets the Python helpers render.
PYTHON_TARGETS: frozenset[str] = frozenset(LET_CALL_SYMBOLS)


def python_call_renderable(target: str, func: str, arity: int) -> bool:
    """Whether a Python target's renderer can spell a let call.

    Parameters
    ----------
    target : str
        The transpile target.
    func : str
        The called name.
    arity : int
        The number of arguments.

    Returns
    -------
    bool
        False exactly when the renderer refuses the call: a math builtin
        with neither a native spelling nor a symbol on the target.
    """
    if native_path(target, func, arity) is not None:
        return True
    if func in LET_CALL_SYMBOLS.get(target, {}):
        return True
    return func not in EAGER_BUILTIN_NAMES


__all__: list[str] = []
