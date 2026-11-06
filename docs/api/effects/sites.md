# Sampleables

A sample request pairs a conditional distribution with the input at which it is
evaluated. These adapters package that pair for the kernel. The active handler then
draws a value, supplies an observation, or reuses a recorded value.

See [sample sites and their distributions](../../guides/inference-foundations.md#trace-and-sample-sites) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

A torch morphism at an input, as the kernel's sampleable.

::: quivers.effects.sites
