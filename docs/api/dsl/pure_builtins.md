# Pure builtins

A [pure builtin](../../reference/qvr/types-and-indexed-families.md#pure-expressions) computes from its
arguments without requesting an effect such as a draw. The [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) lowering resolves each builtin by
the types of its arguments so that the core can run or translate it. A pure builtin still returns
a random value when its arguments are random.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.pure_builtins
