# Types

Static terms are the types, indices, shapes, and effect references of the core
([family declarations](../../reference/qvr/types-and-indexed-families.md#family-declarations)). A
[vector's length index](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), for
instance, constrains how many elements a value has, and the checker tracks that constraint through
constructor applications and effectful computations. The module also renders static terms as text.

::: quivers.qiec.types
