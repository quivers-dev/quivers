# Program elaboration

Program elaboration turns a parsed program body into executable steps, resolving its
names and dependencies along the way. Extend this implementation when adding syntax
whose meaning affects how draws, observations, or deterministic bindings execute.

See [the source statements elaboration translates](../../reference/qvr/probabilistic-programs.md#step-reference) and
[program elaboration in the type system](../../semantics/typing.md#8-program-elaboration).

Elaboration of `program` declarations into QIEC computations.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.program_elaboration
