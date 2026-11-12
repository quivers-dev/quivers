# Arrow Typeclass Hierarchy

[Arrows](https://en.wikipedia.org/wiki/Arrow_(computer_science)) organize computations by their
input and output interfaces [@hughes2000arrows]. The hierarchy is `Category_`, `Arrow`,
`ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, and `ArrowPlus`.
[Sequential composition](../../guides/effects.md#typeclass-hierarchy) connects matching interfaces,
`first` applies a computation to one component of a pair and carries the other along, and the
choice and loop classes add branching and feedback. The [bridges](../monadic/bridges.md) relate
arrows to monads.

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
