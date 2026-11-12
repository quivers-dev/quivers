# Monoidal Structures

A [monoidal structure](https://ncatlab.org/nlab/show/monoidal+category) combines objects and morphisms
side by side [@maclane1998categories]. The cartesian product pairs values, and the coproduct selects
a tagged alternative. [Associators regroup nested combinations and braidings exchange their order](../../guides/categorical.md#monoidal-structures).
These operations arrange inputs and outputs, and the chosen `Algebra` separately fixes how their
weights combine. The module gives the `MonoidalStructure` interface with the instances
`CartesianProductMonoidal` and `CoproductMonoidal`. On weighted relations the Cartesian product is
a symmetric monoidal product, not a categorical product.

::: quivers.categorical.monoidal
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The `EMPTY` singleton is documented on the
[categorical overview](functors.md#package-constants).
