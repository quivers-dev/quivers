# QIEC tooling

These functions expose analysis and symbol metadata of the checked
[Quivers Indexed Effect Core](../qiec/index.md) (QIEC) to
[diagnostics](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing) and
[editor features](../../reference/qvr/execution-and-tooling.md#language-server-features), linking
source locations, names, and types to the corresponding core declarations.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.qiec_tooling
