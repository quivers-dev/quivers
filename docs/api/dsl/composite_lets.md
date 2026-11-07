# Composite lets

Normalization rewrites supported composite expressions into forms the next compilation
stage expects. This [makes implicit dependencies explicit](../../guides/dsl-overview.md#one-checked-core) without asking downstream code
to handle every source spelling separately. These helpers are internal; ordinary model
authors should express the computation in QVR.

Normalizing program ASTs before lowering.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.composite_lets
