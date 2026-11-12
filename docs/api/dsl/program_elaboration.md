# Program elaboration

These functions [elaborate](../../semantics/typing.md#8-program-elaboration) a `program` declaration
into a [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) computation, resolving its names and dependencies. A change to how
[draws, observations, or deterministic bindings](../../reference/qvr/probabilistic-programs.md#step-reference)
run is made here.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.program_elaboration
