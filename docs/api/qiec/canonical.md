# Canonical types

Canonical constructors give shared core types one consistent representation. This
matters when the compiler, checker, and runtime must recognize the same type without
relying on a particular spelling in source. These helpers construct type descriptions
rather than sampling or transforming model values.

See [static parameters and their kinds](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders) and
[using a vector length index in a fold](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check).

Canonical type constructors shared by the kernel, prelude, and compiler.

::: quivers.qiec.canonical
