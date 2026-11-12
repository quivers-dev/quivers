# Draw arguments

This adapter reads the `DrawArg` AST of
[arguments at a distribution call](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families),
matching positional and named arguments to the family's parameters so that
[elaboration](../../semantics/typing.md#8-program-elaboration) can build the draw.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.draw_args
