# QIEC lowering

`lower_qvr_to_qiec` translates a parsed module into the
[Quivers Indexed Effect Core](../../guides/dsl-overview.md#one-checked-core) (QIEC), the typed core
language that checking, execution, and transpilation share. It rewrites source syntax into
[explicit operations](../../semantics/adequacy.md#2-direct-correspondences), checks the result, and
raises a diagnostic naming any construct it cannot represent.

::: quivers.dsl.qiec_lowering
