# Reference families

The reference backend implements families beyond the simplest scalar distributions.
These routines supply concrete sampling and density calculations for the declared family
semantics. Consult event-shape and support restrictions when translating a model whose
family exists on several backends.

See [fixed and conditional distribution families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families) and
[preserving a distribution's parameterization across runtimes](../../semantics/transpile-correctness/index.md#4-distribution-parameterization).

The reference backend's samplers and densities beyond the scalar core.

::: quivers.qiec.reference_families
