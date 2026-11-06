# Arrow Typeclass Hierarchy

Arrows organize computations with explicit input and output interfaces. Sequential
composition connects compatible interfaces; `first` applies a computation to one part of
a pair while carrying the other part along. Choice and loop add further capabilities.
These interfaces are useful when the shape of a computation matters independently of the
values flowing through it.

See [the arrow hierarchy](../../guides/effects.md#typeclass-hierarchy) and
[the relationship to monadic sequencing](../monadic/bridges.md).

For the arrow interfaces and their motivation, see [@hughes2000arrows].

The Hughes-style arrow hierarchy: `Category_`, `Arrow`, `ArrowChoice`,
`ArrowApply`, `ArrowLoop`, `ArrowZero`, and `ArrowPlus`.

::: quivers.arrows.typeclasses

## Package constants

The arrow typeclass theories of `quivers.arrows`, each a `TypeclassTheory`.

::: quivers.arrows
    options:
      members:
        - ThCategory_
        - ThArrow
        - ThArrowChoice
        - ThArrowApply
        - ThArrowLoop
        - ThArrowZero
        - ThArrowPlus
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
