# Traces

The trace handler observes execution and records the sites it encounters. It provides
the values and log-density contributions needed for inspection or later replay. A trace
describes a particular execution, so changing the input or handler configuration can
change both its values and its interpretation.

See [the information recorded in a trace](../../guides/inference-foundations.md#trace-and-sample-sites) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

The trace handler and the trace records.

::: quivers.effects.trace_handler
