# Typeclass Laws

These records express equations for monadic operations, including the [identity and
associativity laws for sequencing](../../guides/monadic.md#monads) [@moggi1991notions; @wadler1992essence]. They provide specifications for implementations and
for numerical checks on chosen values. A proof of a law must cover the implementation’s
full domain.

Runtime-checkable typeclass-law scaffolding.

::: quivers.monadic.laws
