# Trace types

`SampleSite` and `Trace` record one run of a program: each site's
[value and log density](../../guides/inference-foundations.md#trace-and-sample-sites). Inference
algorithms use these records to [compare runs](../../tutorials/python/05-variational-inference.md#tracing-the-model),
compute objectives, or replay chosen sites.

::: quivers.effects.trace_types
