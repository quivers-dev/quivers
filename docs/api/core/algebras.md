# Algebras

An algebra specifies how to [combine weights along a path and across alternative paths](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine).
Multiplication and summation give ordinary matrix multiplication; addition and minimum
compute shortest path costs. Reassociating compositions requires distributivity as well
as associativity [@kelly1982basic]. With the default product/noisy-OR pair, changing the grouping can
change the result.

Ordered algebraic structures for morphism composition and enriched category theory.

::: quivers.core.algebras
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The algebra singletons, such as `PRODUCT_FUZZY`, are documented on the
[core overview](objects.md#package-constants).
