# Kinds

A kind says what sort of static argument a binder expects, such as a type or an index. A
[telescope](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders) is an ordered list of
binders in which later ones may mention earlier ones [@debruijn1991telescopic]. For a
[vector indexed by length](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), these
keep the element type and the length apart from the elements. The module gives kinds, finite
enumerated index types, and declaration telescopes.

::: quivers.qiec.kinds
