# Draw arguments

These compiler helpers interpret arguments at distribution call sites. They associate
positional and named arguments with the family’s parameters so elaboration can construct
the requested draw. Model authors normally supply those arguments in QVR or through the
public family constructor.

See [arguments to fixed and conditional families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families) and
[program elaboration in the type system](../../semantics/typing.md#8-program-elaboration).

The reading adapter over the `DrawArg` AST.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.draw_args
