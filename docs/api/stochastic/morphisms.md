# Stochastic Morphisms

A `StochasticMorphism` is a [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel) between
finite sets, stored as a [stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix) whose
rows are output distributions: nonnegative entries, each row summing to one
([FinStoch](../../guides/stochastic.md#the-finstoch-category)) [@fritz2020synthetic].
[Composition](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels)
averages the next transition over the intermediate values and gives the distribution of the output
given the initial input. `stochastic(domain, codomain)` builds a learnable one whose rows are a softmax of free parameters.

::: quivers.stochastic.morphisms

## The sum-product semiring {#the-markov-algebra}

`quivers.stochastic` re-exports `MarkovAlgebra` and its singleton `MARKOV`, the `Algebra` under
which stochastic morphisms compose (the sum-product semiring), from `quivers.core.algebras`.

::: quivers.stochastic
    options:
      members:
        - MARKOV
      show_root_heading: false
      show_root_toc_entry: false
