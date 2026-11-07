# Sampleables

A [sample request](../../guides/inference-foundations.md#trace-and-sample-sites) pairs a conditional distribution with the input at which it is
evaluated. These adapters package that pair for the kernel. The active handler then
draws a value, supplies an observation, or reuses a [recorded value](../../tutorials/python/05-variational-inference.md#tracing-the-model).

A torch morphism at an input, as the kernel's sampleable.

::: quivers.effects.sites
