# Step resolution

A program step may refer to a previously declared distribution or supply an inline
family call. Resolution determines what distribution that reference denotes and which
inputs it needs. This stage links source names to executable sampling and scoring
behavior.

See [the supported kinds of program step](../../reference/qvr/probabilistic-programs.md#step-reference) and
[program elaboration in the type system](../../semantics/typing.md#8-program-elaboration).

Resolving the distribution references of program steps.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.step_resolution
