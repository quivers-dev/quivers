# Effect Schema Lifting

Lifting adapts a base rule to values that carry an effect, such as alternatives or a
context. The available typeclass operations determine how the rule can use those values.
These functions construct the corresponding rule schemas; the generated rule still
depends on the laws and scope conventions of its effect instance.

See [lifting a schema through an effect](../../guides/effects.md#class-driven-schema-lifting) and
[a continuation-based scope example](../../guides/effects.md#worked-example-scope-taking-via-continuation).

For the arrow interface used by these effect constructions, see [@hughes2000arrows].

`class_directed_lifts(base_schema, effect)` and
`lift_rule_set(base_schemas, effects)` emit lifted `SchemaDecl`s for
the typeclasses inhabited by an effect.

::: quivers.stochastic.effect_lifts
