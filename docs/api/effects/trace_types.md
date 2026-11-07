# Trace types

Trace records store information about one execution, including [sampled values and their
scores](../../guides/inference-foundations.md#trace-and-sample-sites). Inference algorithms use these records to [compare executions](../../tutorials/python/05-variational-inference.md#tracing-the-model), compute
objectives, or replay selected choices. The algorithm determines how collected traces
represent its target distribution.

The site and trace records.

::: quivers.effects.trace_types
