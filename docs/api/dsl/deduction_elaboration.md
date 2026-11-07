# Deduction elaboration

A [deduction declaration](../../reference/qvr/generated-computations.md#deduction-declarations) names rules and how their results combine. Elaboration
represents that behavior through [explicit core families, handlers, and computations](../../tutorials/qvr/11-parsing-and-search.md#3-see-the-generated-effects).
This is the bridge between a compact rule declaration and the structures that the
checked execution path can inspect.

Elaboration of `deduction` declarations into QIEC families, handlers, and
computations.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.deduction_elaboration
