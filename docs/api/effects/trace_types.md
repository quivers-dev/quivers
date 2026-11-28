# Trace types

`SampleSite` and `Trace` record one run of a program: each site's
[value and log density](../../guides/inference-foundations.md#trace-and-sample-sites). Inference
algorithms use these records to [compare runs](../../tutorials/python/05-variational-inference.md#tracing-the-model),
compute objectives, or replay chosen sites.

A finite structured sample site sets `is_structured` and keeps its
checked point as `value`. The same trace record carries its address,
sampleable, and tensor log weight. The point keeps its type under
conditioning and replay. A trace with such a site describes one
independent run. Tensor sites retain their ordinary run-axis layout.

A structured latent has no unconstrained tensor coordinates. A
[continuous latent registry](../inference/guide.md#quivers.inference.registry.LatentRegistry) records a
structured observation as observed and requires an unobserved
structured site to be handled separately. Exact enumeration uses the
runtime's existing finite `Random` handler.

::: quivers.effects.trace_types
