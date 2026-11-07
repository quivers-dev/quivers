# Natural Transformations

A natural transformation converts one functor’s output into another’s compatibly with
source arrows [@maclane1998categories]. Its [naturality equation](../../guides/categorical.md#natural-transformations) compares two routes: convert and then process, or
process and then convert. A component supplies the conversion for one object, and the
equation must hold for every source arrow.

The `NaturalTransformation` interface and componentwise natural
transformations.

::: quivers.categorical.natural_transformations
