# Traces

The trace handler observes execution and records the [sites it encounters](../../guides/inference-foundations.md#trace-and-sample-sites). It provides
the values and log-density contributions needed for [inspection or later replay](../../tutorials/python/05-variational-inference.md#tracing-the-model). A trace
describes a particular execution, so changing the input or handler configuration can
change both its values and its interpretation.

The trace handler and the trace records.

::: quivers.effects.trace_handler
