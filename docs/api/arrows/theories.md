# Arrow Theories (panproto mirror)

These records describe an arrow interface through its operations and equations. An
identity law, for instance, requires composing an arrow with an identity to leave it
unchanged. Tooling can inspect this specification when working with an implementation.

See [the operations represented by an arrow theory](../../guides/effects.md#typeclass-hierarchy) and
[the relationship to monadic sequencing](../monadic/bridges.md).

For the arrow interfaces and their motivation, see [@hughes2000arrows].

`TypeclassTheory` records for the arrow typeclasses: `ThCategory_`,
`ThArrow`, `ThArrowChoice`, `ThArrowApply`, `ThArrowLoop`, `ThArrowZero`,
and `ThArrowPlus`.

::: quivers.arrows.theories
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The theory constants are documented on the
[arrows overview](typeclasses.md#package-constants).
