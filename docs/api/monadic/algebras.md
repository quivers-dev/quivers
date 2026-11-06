# Algebras and Coalgebras

An algebra for a monad interprets its structured values. For the list monad, a monoid
operation combines a list into one value; associativity makes this compatible with
flattening nested lists. A coalgebra for a comonad equips a value with structured
context. The equations below specify the required compatibility.

Milewski's [free monoids chapter](https://bartoszmilewski.com/2015/07/21/free-monoids/) develops the list-and-fold example behind this interpretation [@milewski2015free].

See [interpreting monadic structure](../../guides/monadic.md#algebras-and-coalgebras) and
[an introduction to sequencing computations](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Monad algebras, comonad coalgebras, and their structural properties.

::: quivers.monadic.algebras
