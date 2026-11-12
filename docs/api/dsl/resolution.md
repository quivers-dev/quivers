# Resolution

The compiler mixin that [resolves](../../semantics/types-and-spaces.md#6-resolution-in-code) a QVR
`ObjectExpr` to a runtime `SetObject` or `ContinuousSpace`. One walk handles both, so a
[product](../../guides/core.md#productset-and-coproductset) that mixes the two, such as
`Real 3 * Token`, resolves to one `ProductSpace` that keeps the finite index apart from the real
component.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.compiler.resolution
