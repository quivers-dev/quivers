# Monoidal Structures

A monoidal structure lets us combine interfaces side by side. Cartesian products pair
values; coproducts select a tagged alternative. Associators regroup nested combinations
and braidings exchange their order. These operations concern how inputs and outputs are
organized; the selected morphism algebra separately determines how their weights
combine.

For pairs and tagged alternatives, begin with Milewski's [products and coproducts chapter](https://bartoszmilewski.com/2015/01/07/products-and-coproducts/) [@milewski2015products].

See [products, coproducts, and coherence maps](../../guides/categorical.md#monoidal-structures) and
[the table interpretation of objects and arrows](../../tutorials/qvr/07-categorical.md#begin-with-composable-tables).

For the categorical definitions and their laws, see [@maclane1998categories].

The `MonoidalStructure` interface with cartesian-product and
coproduct implementations.

::: quivers.categorical.monoidal
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The `EMPTY` singleton is documented on the
[categorical overview](functors.md#package-constants).
