# Monoidal Structures

A monoidal structure lets us combine interfaces side by side [@maclane1998categories]. Cartesian products pair
values; coproducts select a tagged alternative. [Associators regroup nested combinations
and braidings exchange their order](../../guides/categorical.md#monoidal-structures). These operations concern how inputs and outputs are
organized; the selected morphism algebra separately determines how their weights
combine.

The `MonoidalStructure` interface with cartesian-product and
coproduct implementations.

::: quivers.categorical.monoidal
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The `EMPTY` singleton is documented on the
[categorical overview](functors.md#package-constants).
