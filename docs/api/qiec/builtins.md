# Builtins

The prelude supplies established effect interfaces and their reference interpretations.
A sampling request, for instance, needs a handler that can produce a value from the
requested distribution. Keeping the interface separate from its handler lets the checker
reason about requests before execution chooses their implementation.

See [the prelude effect interfaces](../../reference/qvr/effects-and-handlers.md#prelude-interfaces) and
[writing and running a handler](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler).

The canonical prelude effects and their reference handlers.

::: quivers.qiec.builtins
