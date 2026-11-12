# Monads

A [monad](https://ncatlab.org/nlab/show/monad) has a unit that inserts ordinary values into a
computation and a multiplication that flattens one level of nested computation
[@moggi1991notions; @wadler1992essence]. For lists, the unit makes a singleton list and the
multiplication concatenates. For distributions, the multiplication averages inner distributions by
the outer probabilities
([monads and handlers](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue)).
The module gives `FuzzyPowersetMonad` and `KleisliCategory`. `FuzzyPowersetMonad` requires an
`IdempotentSemiring`: its Kleisli category is the category of relations valued in that semiring
[@hofmann2014monoidal], its unit and multiplication are identities on finite sets, and its Kleisli
composition is the matrix product ([fuzzy powerset monad](../../guides/monadic.md#fuzzy-powerset-monad)).

::: quivers.monadic.monads
