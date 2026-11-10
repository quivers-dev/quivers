# QIEC tooling

These adapters expose checked core information to [diagnostics](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing) and [editor features](../../reference/qvr/execution-and-tooling.md#language-server-features). Use
them to connect source locations, names, and types to the corresponding core
declarations when implementing inspection tools.

Shared QIEC analysis and symbol metadata for interactive tooling.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.qiec_tooling
