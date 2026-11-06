# Stochastic Morphisms

A stochastic matrix contains one output distribution per input row: all entries are
nonnegative and each row sums to one. Composition averages the next transition over the
possible intermediate values, producing the output distribution conditional on the
initial input.

See [stochastic matrices and their composition](../../guides/stochastic.md#the-finstoch-category) and
[a Markov kernel calculation](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels).

For finite and general stochastic maps, see [@fritz2020synthetic].

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
