# Kan Extensions

`left_kan` and `right_kan` compute [Kan extensions](https://ncatlab.org/nlab/show/Kan+extension)
along a function between finite sets (an `ObjectMap`, such as a `Projection` or an `Inclusion`),
the special case of a Kan extension along a functor between discrete categories
[@maclane1998categories; @kelly1982basic]. The [left extension](../../guides/enriched.md#kan-extensions)
joins the weights over each fiber, the set of elements that map to one point, and the right
extension meets them. Along a projection that forgets a coordinate, the left extension is thus
the reduction over that coordinate.

::: quivers.enriched.kan_extensions
