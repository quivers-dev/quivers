# Builtins

The prelude supplies established [effect interfaces](../../reference/qvr/effects-and-handlers.md#prelude-interfaces) and their reference interpretations.
A sampling request, for instance, needs a [handler that can produce a value](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler) from the
requested distribution. Keeping the interface separate from its handler lets the checker
reason about requests before execution chooses their implementation.

The canonical prelude effects and their reference handlers.

::: quivers.qiec.builtins
