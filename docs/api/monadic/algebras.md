# Algebras and Coalgebras

An algebra for a monad [interprets its structured values](../../guides/monadic.md#algebras-and-coalgebras). For the list monad, a monoid
operation combines a list into one value; associativity makes this compatible with
flattening nested lists [@moggi1991notions; @wadler1992essence]. A coalgebra for a comonad equips a value with structured
context. The equations below specify the required compatibility.

Monad algebras, comonad coalgebras, and their structural properties.

::: quivers.monadic.algebras
