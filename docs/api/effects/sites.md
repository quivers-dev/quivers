# Sampleables

A `MonadicProgram` step draws from a `ContinuousMorphism` at an input tensor, and the QIEC request
`Random.sample` takes one sampleable value. `TorchSampleable` wraps the pair as that value
([sample sites](../../guides/inference-foundations.md#trace-and-sample-sites)). It offers the sampling
and scoring interface the prelude handlers use, and keeps the morphism and the input for a
strategy or a [trace](../../tutorials/python/05-variational-inference.md#tracing-the-model) to read.

::: quivers.effects.sites
