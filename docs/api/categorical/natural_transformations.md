# Natural Transformations

A [natural transformation](https://ncatlab.org/nlab/show/natural+transformation) from one functor to
another gives, for each object, a component that converts the first functor's output into the
second's [@maclane1998categories]. Its [naturality equation](../../guides/categorical.md#natural-transformations)
says that converting and then mapping an arrow agrees with mapping and then converting, for every
arrow of the source category. The module gives the `NaturalTransformation` interface and natural
transformations defined by their components.

::: quivers.categorical.natural_transformations
