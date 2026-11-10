# Arrow Typeclass Hierarchy

Arrows organize computations with [explicit input and output interfaces](../monadic/bridges.md). [Sequential
composition](../../guides/effects.md#typeclass-hierarchy) connects compatible interfaces; `first` applies a computation to one part of
a pair while carrying the other part along. Choice and loop add further capabilities.
These interfaces are useful when the shape of a computation matters independently of the
values flowing through it [@hughes2000arrows].

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
