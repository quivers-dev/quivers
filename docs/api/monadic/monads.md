# Monads

A [monad](https://ncatlab.org/nlab/show/monad) has a unit that inserts ordinary values into a
computation and a multiplication that flattens one level of nested computation
[@moggi1991notions; @wadler1992essence]. For lists, the unit makes a singleton list and the
multiplication concatenates; for distributions, the multiplication averages inner distributions by
the outer probabilities
([monads and handlers](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue)).
The module gives `FuzzyPowersetMonad`, `FreeMonoidMonad`, and `KleisliCategory`. The unit and
multiplication of `FuzzyPowersetMonad` are identities on finite sets, and its Kleisli composition
is composition under its `Algebra`, which is associative only when that `Algebra` distributes
([monads](../../guides/monadic.md#monads)).

::: quivers.monadic.monads
