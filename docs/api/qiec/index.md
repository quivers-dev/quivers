# Quivers Indexed Effect Core

The Quivers Indexed Effect Core (QIEC) is the common typed representation used after
source elaboration. It makes [value types, index constraints, and effect requests](../../semantics/typing.md#4-qiec-kinds-and-types)
explicit so [execution and transpilation](../../reference/qvr/index.md#compilation-and-execution-map) inspect the same computation. Most model authors
use QVR; these modules are for inspecting or extending the checked core.

The stable kernel every QVR module checks against and every transpiler reads. The package re-exports the modules below.

::: quivers.qiec
