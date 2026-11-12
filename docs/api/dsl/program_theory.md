# Program Theory

`QVR_PROGRAM_PROTOCOL` is the panproto protocol over which QVR programs are schemas, and
`extract_program_schema` walks a compiled environment and emits a panproto `Schema` recording the
program's [operations and equations](../../semantics/program-theory.md#3-what-a-schema-records)
([programs as panproto schemas](../../guides/dsl-overview.md#programs-as-panproto-schemas)). A claim
that a rewrite of such a schema preserves a distribution also needs the probabilistic meaning of
the operations and the rewrite's assumptions.

::: quivers.dsl.program_theory
