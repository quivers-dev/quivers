# Algebras

An algebra specifies how to combine weights along a path and across alternative paths.
Multiplication and summation give ordinary matrix multiplication; addition and minimum
compute shortest path costs. Reassociating compositions requires distributivity as well
as associativity. With the default product/noisy-OR pair, changing the grouping can
change the result.

Milewski's [enriched categories chapter](https://bartoszmilewski.com/2017/05/13/enriched-categories/) introduces the role of the base category through monoidal structure and preorders [@milewski2017enriched].

See [the available reduction algebras](../../guides/core.md#algebras-enrichment-algebras) and
[a numerical comparison of combination rules](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine).

For the enriched theory under its mathematical hypotheses, see [@kelly1982basic].

Ordered algebraic structures for morphism composition and enriched category theory.

::: quivers.core.algebras
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The algebra singletons, such as `PRODUCT_FUZZY`, are documented on the
[core overview](objects.md#package-constants).
