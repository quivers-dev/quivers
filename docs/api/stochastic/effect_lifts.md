# Effect Schema Lifting

[Lifting adapts a base rule](../../guides/effects.md#class-driven-schema-lifting) to values that carry an effect, such as alternatives or a
[context](../../guides/effects.md#worked-example-scope-taking-via-continuation). The available typeclass operations determine how the rule can use those values [@hughes2000arrows].
These functions construct the corresponding rule schemas; the generated rule still
depends on the laws and scope conventions of its effect instance.

`class_directed_lifts(base_schema, effect)` and
`lift_rule_set(base_schemas, effects)` emit lifted `SchemaDecl`s for
the typeclasses inhabited by an effect.

::: quivers.stochastic.effect_lifts
