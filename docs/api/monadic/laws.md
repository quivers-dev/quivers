# Typeclass Laws

These records express equations for monadic operations, including the identity and
associativity laws for sequencing. They provide specifications for implementations and
for numerical checks on chosen values. A proof of a law must cover the implementation’s
full domain.

Milewski derives the sequencing laws in [his programmer's definition of monads](https://bartoszmilewski.com/2016/11/21/monads-programmers-definition/) [@milewski2016monads].

See [the monad equations](../../guides/monadic.md#monads) and
[an introduction to sequencing computations](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Runtime-checkable typeclass-law scaffolding.

::: quivers.monadic.laws
