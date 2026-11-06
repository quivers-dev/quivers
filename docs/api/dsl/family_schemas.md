# Family schemas

A distribution call must use recognized parameter names and the correct argument
structure. Family schemas describe those expectations for the source language. This
catches errors such as confusing a scale keyword with a rate keyword before a backend
distribution is constructed.

See [family call conventions](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families) and
[program elaboration in the type system](../../semantics/typing.md#8-program-elaboration).

The keyword schemas a distribution-family call in QVR source must match.

::: quivers.dsl.family_schemas
