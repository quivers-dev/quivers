# Step resolution

These functions resolve the distribution a
[program step](../../reference/qvr/probabilistic-programs.md#step-reference) names, whether a
declared morphism or an inline family call, and the inputs it needs
([program elaboration](../../semantics/typing.md#8-program-elaboration)).

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.step_resolution
