# Pure builtins

A [pure builtin](../../reference/qvr/types-and-indexed-families.md#pure-expressions) computes from its arguments without requesting an effect such as
sampling. Lowering resolves the supported operation using its type so the core can
execute or translate it. A deterministic builtin can still return different values
across model runs when its arguments are random.

The pure-expression builtins the QIEC lowering resolves by type.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.pure_builtins
