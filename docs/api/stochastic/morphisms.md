# Stochastic Morphisms

A [stochastic matrix](../../guides/stochastic.md#the-finstoch-category) contains one output distribution per input row: all entries are
nonnegative and each row sums to one [@fritz2020synthetic]. [Composition averages the next transition](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels) over the
possible intermediate values, producing the output distribution conditional on the
initial input.

Stochastic relations, kernels, and morphisms for probabilistic computation.

::: quivers.stochastic.morphisms

## The Markov algebra

`quivers.stochastic` re-exports `MarkovAlgebra` and its singleton `MARKOV`, the algebra
stochastic morphisms compose under, from `quivers.core.algebras`.

::: quivers.stochastic
    options:
      members:
        - MARKOV
      show_root_heading: false
      show_root_toc_entry: false
