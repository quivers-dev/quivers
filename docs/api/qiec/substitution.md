# Substitution

Substitution replaces variables with terms while respecting binding. If a replacement
contains a free name, an inner binder must not accidentally capture it. [Telescope
instantiation](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders) applies the same discipline to [ordered dependent parameters](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), where
replacing one index can change the types of later arguments.

Capture-free substitution and telescope instantiation.

::: quivers.qiec.substitution
