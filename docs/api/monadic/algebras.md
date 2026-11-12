# Algebras and Coalgebras

An [algebra of a monad](https://ncatlab.org/nlab/show/algebra+over+a+monad) evaluates its
structured values into a carrier, compatibly with the monad's unit and multiplication
[@maclane1998categories]. For the list monad, an associative operation with a unit folds a list
into one value, and associativity makes the fold agree with flattening nested lists. A coalgebra
of a [comonad](https://ncatlab.org/nlab/show/comonad) dually equips a value with structured
context. The module gives [monad algebras and comonad coalgebras](../../guides/monadic.md#algebras-and-coalgebras)
with their compatibility equations.

::: quivers.monadic.algebras
