# Evaluator

The evaluator applies one execution rule at a time. Each step exposes how a request is
handled, how a value is returned, and which continuation resumes. This provides a
reference execution for inspecting the behavior of a checked computation.

See [how a handler resumes a computation](../../reference/qvr/effects-and-handlers.md#handler-declarations) and
[writing and running a handler](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler).

The small-step reference evaluator.

::: quivers.qiec.evaluator
