# Canonical types

Canonical constructors give [shared core types](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders) one consistent representation. This
matters when the compiler, checker, and runtime must recognize the same type without
relying on a particular spelling in source. These helpers construct type descriptions
rather than sampling or transforming model values.

Canonical type constructors shared by the kernel, prelude, and compiler.

::: quivers.qiec.canonical
