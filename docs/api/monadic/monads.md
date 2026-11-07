# Monads

A monad provides a way to [insert ordinary values into a computation and flatten one
level of nested computation](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue). For lists, insertion makes a singleton list and flattening
concatenates lists of alternatives [@moggi1991notions; @wadler1992essence]. For probabilities, flattening averages distributions
using outer probabilities. The finite and fuzzy implementations below require the [bounds
and algebraic qualifications](../../guides/monadic.md#monads) discussed in the guide.

Monad definitions with units, multiplications, and monad laws.

::: quivers.monadic.monads
