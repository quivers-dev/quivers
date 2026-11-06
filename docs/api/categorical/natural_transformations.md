# Natural Transformations

A natural transformation converts one functor’s output into another’s compatibly with
source arrows. Its naturality equation compares two routes: convert and then process, or
process and then convert. A component supplies the conversion for one object, and the
equation must hold for every source arrow.

For the commuting-square intuition, see Milewski's [natural transformations chapter](https://bartoszmilewski.com/2015/04/07/natural-transformations/) [@milewski2015natural].

See [the singleton-list naturality example](../../guides/categorical.md#natural-transformations) and
[the table interpretation of objects and arrows](../../tutorials/qvr/07-categorical.md#begin-with-composable-tables).

For the categorical definitions and their laws, see [@maclane1998categories].

The `NaturalTransformation` interface and componentwise natural
transformations.

::: quivers.categorical.natural_transformations
