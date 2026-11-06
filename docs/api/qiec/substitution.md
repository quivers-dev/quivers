# Substitution

Substitution replaces variables with terms while respecting binding. If a replacement
contains a free name, an inner binder must not accidentally capture it. Telescope
instantiation applies the same discipline to ordered dependent parameters, where
replacing one index can change the types of later arguments.

See [binding static parameters](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders) and
[using a vector length index in a fold](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check).

Capture-free substitution and telescope instantiation.

::: quivers.qiec.substitution
