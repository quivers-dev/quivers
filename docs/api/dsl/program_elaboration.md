# [Program elaboration](../../semantics/typing.md#8-program-elaboration)

Program elaboration turns a parsed program body into executable steps, resolving its
names and dependencies along the way. Extend this implementation when adding syntax
whose meaning affects how [draws, observations, or deterministic bindings](../../reference/qvr/probabilistic-programs.md#step-reference) execute.

Elaboration of `program` declarations into QIEC computations.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.program_elaboration
