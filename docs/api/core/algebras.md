# Algebras

An `Algebra` fixes how composition
[combines weights](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine):
a product $\otimes$ along a path and a reduction $\oplus$ across alternative paths. Each built-in
instance is a [semiring](https://en.wikipedia.org/wiki/Semiring) [@goodman1999semiring;
@mohri2002semiring], a [t-norm](https://en.wikipedia.org/wiki/T-norm) with its t-conorm
[@klement2000triangular], or a product with capped sum. Multiplication with summation gives the
ordinary matrix product, and addition with minimum gives shortest path costs. Composition is
associative only when $\otimes$ is associative and distributes over $\oplus$ [@kelly1982basic].
The default product t-norm with noisy-OR reduction does not distribute, so with it a change of
grouping can change the result.

::: quivers.core.algebras
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The algebra singletons, such as `PRODUCT_FUZZY`, are documented on the
[core overview](objects.md#package-constants).
