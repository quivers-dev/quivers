# Distributions

`RuntimeDistribution` values sample and score
[distribution families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families)
on the [reference machine](../../developer/qiec.md#reference-evaluator), through the
`DistributionBackend` of the run. The `ReferenceBackend` computes in plain Python numbers, position
by position, through a `HostLaw`; the [tensor backend](../continuous/tensor_backend.md) computes
with torch and keeps every gradient. A transpile target may support different families or
[parameterizations](../../semantics/transpile-correctness/index.md#4-distribution-parameterization),
which its [support table](../../transpile-support.md) lists.

::: quivers.qiec.distributions
