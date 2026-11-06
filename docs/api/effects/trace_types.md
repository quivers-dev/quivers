# Trace types

Trace records store information about one execution, including sampled values and their
scores. Inference algorithms use these records to compare executions, compute
objectives, or replay selected choices. The algorithm determines how collected traces
represent its target distribution.

See [trace values and scores](../../guides/inference-foundations.md#trace-and-sample-sites) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

The site and trace records.

::: quivers.effects.trace_types
