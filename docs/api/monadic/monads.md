# Monads

A monad provides a way to insert ordinary values into a computation and flatten one
level of nested computation. For lists, insertion makes a singleton list and flattening
concatenates lists of alternatives. For probabilities, flattening averages distributions
using outer probabilities. The finite and fuzzy implementations below require the bounds
and algebraic qualifications discussed in the guide.

Milewski's [programmer's definition of monads](https://bartoszmilewski.com/2016/11/21/monads-programmers-definition/) develops sequencing from Kleisli composition [@milewski2016monads].

See [unit, multiplication, and the monad laws](../../guides/monadic.md#monads) and
[an introduction to sequencing computations](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Monad definitions with units, multiplications, and monad laws.

::: quivers.monadic.monads
