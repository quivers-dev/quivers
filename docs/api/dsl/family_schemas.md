# Family schemas

These keyword schemas fix the parameter names and argument structure that a call of a distribution
family in QVR source must match
([fixed and conditional families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families)).
They catch an error such as a scale keyword given where the family expects a rate before any
backend distribution is built.

::: quivers.dsl.family_schemas
