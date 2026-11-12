# Builtins

The prelude of the Quivers Indexed Effect Core ([QIEC](index.md)) declares the
[built-in effect interfaces](../../reference/qvr/effects-and-handlers.md#prelude-interfaces)
(`Random`, `Score`, `State`, `Abort`, `Choose`, `Weight`, `Compute`, `Param`) and gives their
reference handlers. A draw request, for instance, needs a
[handler that produces a value](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler)
from the requested distribution. Keeping each interface apart from its handlers lets the checker
reason about requests before a run chooses their implementation.

::: quivers.qiec.builtins
