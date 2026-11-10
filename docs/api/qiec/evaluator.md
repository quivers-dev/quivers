# Evaluator

The evaluator applies one execution rule at a time. Each step exposes how a request is
handled, how a value is returned, and [which continuation resumes](../../reference/qvr/effects-and-handlers.md#handler-declarations). This provides a
reference execution for inspecting the behavior of a [checked computation](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler).

The small-step reference evaluator.

::: quivers.qiec.evaluator
