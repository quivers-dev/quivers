# Traces

`TraceHandler` records each [sample site](../../guides/inference-foundations.md#trace-and-sample-sites)
a run reaches, with its value and log density, for
[inspection or later replay](../../tutorials/python/05-variational-inference.md#tracing-the-model).
A trace describes one run, so a different input or handler stack can change its values.

::: quivers.effects.trace_handler
