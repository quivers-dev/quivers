# QIEC lowering

Lowering translates parsed source into the Quivers Indexed Effect Core (QIEC), the typed
representation shared by checking, execution, and transpilation. It expands source
syntax into explicit operations. An unsupported construct produces a diagnostic
identifying the part that could not be represented.

See [the common checked representation](../../guides/dsl-overview.md#one-checked-core) and
[the correspondence between syntax and runtime objects](../../semantics/adequacy.md#2-direct-correspondences).

The exact QVR to QIEC lowering.

::: quivers.dsl.qiec_lowering
