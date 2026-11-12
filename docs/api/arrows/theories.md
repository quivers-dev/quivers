# Arrow Theories (panproto mirror)

Each `TypeclassTheory` records the operations and laws of one arrow class as a panproto theory
([type class hierarchy](../../guides/effects.md#typeclass-hierarchy)): `ThCategory_`, `ThArrow`,
`ThArrowChoice`, `ThArrowApply`, `ThArrowLoop`, `ThArrowZero`, and `ThArrowPlus`
[@hughes2000arrows]. The identity law, for instance, requires composition with an identity to leave
an arrow unchanged.

::: quivers.arrows.theories
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The theory constants are documented on the
[arrows overview](typeclasses.md#package-constants).
