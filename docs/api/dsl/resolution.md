# [Resolution](../../semantics/types-and-spaces.md#6-resolution-in-code)

Resolution turns a symbolic object expression into a runtime object or space. A [product
expression](../../guides/core.md#productset-and-coproductset) must retain the distinction between a finite index and a real-valued
component even when both appear in one signature. These adapters implement that
interpretation for the compiler.

The compiler mixin that resolves QVR `ObjectExpr` ASTs to runtime
`SetObject` and `ContinuousSpace` values. A single walk handles both
strata, so a mixed-domain product such as `Real 3 * Token` resolves to
one `ProductSpace`.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.compiler.resolution
