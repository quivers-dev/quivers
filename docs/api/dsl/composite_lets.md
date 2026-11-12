# Composite lets

These helpers normalize a program AST before lowering. They rewrite supported composite
expressions into the forms the next stage expects, which makes their dependencies explicit
([one checked core](../../guides/dsl-overview.md#one-checked-core)) without asking later stages to
handle every source spelling.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.composite_lets
