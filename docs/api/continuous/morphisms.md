# Continuous Morphisms

A conditional distribution accepts an input and specifies possible outputs. Its sampling
method generates a value, while `log_prob` evaluates the mass or density at a supplied
value. For a composition, simulation draws an intermediate value and then an output;
evaluating the output density requires summing or integrating over the intermediate
value.

See [sampling and scoring continuous morphisms](../../guides/continuous-spaces.md#continuousmorphism) and
[the distinction between mass, density, and a sampled value](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample).

For probability kernels and their composition, see [@giry1982categorical; @fritz2020synthetic].

Operational kernels over discrete or continuous carriers. A
`ContinuousMorphism` evaluates log densities and draws samples rather
than materializing a finite morphism tensor.

::: quivers.continuous.morphisms
