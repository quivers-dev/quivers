# `quivers.qiec`

The Quivers Indexed Effect Core (QIEC) is the typed core language that every executable QVR
declaration elaborates to. It makes
[value types, index constraints, and effect requests](../../semantics/typing.md#4-qiec-kinds-and-types)
explicit, so that [execution and transpilation](../../reference/qvr/index.md#compilation-and-execution-map)
read the same well-typed computation. The package holds the data model, the checker, the
serialization codec, and the [reference machine](../../developer/qiec.md#reference-evaluator), and
re-exports the public names of its modules.

::: quivers.qiec
