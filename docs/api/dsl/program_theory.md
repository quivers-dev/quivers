# Constraint Solver and Program Theory

Program theory records the [operations and equations](../../semantics/program-theory.md#3-what-a-schema-records) used to describe program structure.
These declarations support [inspection and transformation of programs](../../guides/dsl-overview.md#programs-as-panproto-schemas). Any claim that a
rewrite preserves a probability distribution also requires the operations’ probabilistic
interpretation and the assumptions of the rewrite.

`QVR_PROGRAM_PROTOCOL` is the panproto protocol over which QVR programs
are schemas. `extract_program_schema` walks a compiled environment and
emits a panproto `Schema`.

::: quivers.dsl.program_theory
