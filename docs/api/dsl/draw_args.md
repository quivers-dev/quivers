# Draw arguments

These compiler helpers interpret [arguments at distribution call sites](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families). They associate
positional and named arguments with the family’s parameters so [elaboration](../../semantics/typing.md#8-program-elaboration) can construct
the requested draw. Model authors normally supply those arguments in QVR or through the
public family constructor.

The reading adapter over the `DrawArg` AST.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.draw_args
