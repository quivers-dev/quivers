# Constraint Solver and Program Theory

Program theory records the operations and equations used to describe program structure.
These declarations support inspection and transformation of programs. Any claim that a
rewrite preserves a probability distribution also requires the operations’ probabilistic
interpretation and the assumptions of the rewrite.

See [what a program schema records](../../semantics/program-theory.md#3-what-a-schema-records) and
[extracting a program schema](../../guides/dsl-overview.md#programs-as-panproto-schemas).

`QVR_PROGRAM_PROTOCOL` is the panproto protocol over which QVR programs
are schemas. `extract_program_schema` walks a compiled environment and
emits a panproto `Schema`.

::: quivers.dsl.program_theory
