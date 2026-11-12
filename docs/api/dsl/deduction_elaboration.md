# Deduction elaboration

These functions elaborate a [deduction declaration](../../reference/qvr/generated-computations.md#deduction-declarations)
into QIEC families, handlers, and computations
([generated effects](../../tutorials/qvr/11-parsing-and-search.md#3-see-the-generated-effects)).
The [QIEC developer note](../../developer/qiec.md#qvr-surface) lists what a deduction elaborates
to.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.deduction_elaboration
