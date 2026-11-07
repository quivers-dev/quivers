# Step resolution

A [program step](../../reference/qvr/probabilistic-programs.md#step-reference) may refer to a previously declared distribution or supply an inline
family call. [Resolution determines what distribution that reference denotes](../../semantics/typing.md#8-program-elaboration) and which
inputs it needs. This stage links source names to executable sampling and scoring
behavior.

Resolving the distribution references of program steps.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.step_resolution
