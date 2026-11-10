# QIEC lowering

Lowering translates parsed source into the [Quivers Indexed Effect Core (QIEC)](../../guides/dsl-overview.md#one-checked-core), the typed
representation shared by checking, execution, and transpilation. It expands source
syntax into [explicit operations](../../semantics/adequacy.md#2-direct-correspondences). An unsupported construct produces a diagnostic
identifying the part that could not be represented.

The exact QVR to QIEC lowering.

::: quivers.dsl.qiec_lowering
