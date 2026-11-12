# Effect Schema Lifting

These functions lift a base rule schema to values that carry an effect, such as a set of
alternatives or a [continuation](../../guides/effects.md#worked-example-scope-taking-via-continuation)
([schema lifting](../../guides/effects.md#class-driven-schema-lifting)). Which lifted rules exist
depends on the [type classes](https://en.wikipedia.org/wiki/Type_class) the effect inhabits
(functor, applicative, monad, alternative) [@mcbride2008applicative; @wadler1992essence].
`class_directed_lifts(base_schema, effect)` and `lift_rule_set(base_schemas, effects)` emit the
lifted `SchemaDecl`s, and each generated rule depends on the laws and scope conventions of its
effect instance.

::: quivers.stochastic.effect_lifts
