# Continuous Morphisms

A conditional distribution accepts an input and specifies possible outputs. Its [sampling
method](../../guides/continuous-spaces.md#continuousmorphism) generates a value, while `log_prob` evaluates the [mass or density](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample) at a supplied
value. For a composition, simulation draws an intermediate value and then an output;
evaluating the output density requires summing or integrating over the intermediate
value [@giry1982categorical; @fritz2020synthetic].

Operational kernels over discrete or continuous carriers. A
`ContinuousMorphism` evaluates log densities and draws samples rather
than materializing a finite morphism tensor.

::: quivers.continuous.morphisms
