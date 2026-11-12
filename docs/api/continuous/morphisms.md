# Continuous Morphisms

A `ContinuousMorphism` is a conditional distribution over discrete or continuous spaces that
evaluates log densities and draws samples rather than storing a finite tensor
([ContinuousMorphism](../../guides/continuous-spaces.md#continuousmorphism)). Its sampling method
draws an output for an input, and `log_prob` evaluates the mass or density of a given output. For a
composite, sampling draws an intermediate value and then an output, while the output density sums
or integrates over the intermediate value, as in the composition of
[Markov kernels](https://en.wikipedia.org/wiki/Markov_kernel) [@giry1982categorical;
@fritz2020synthetic].

::: quivers.continuous.morphisms
