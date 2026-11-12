# Substitution

Substitution replaces variables with terms without capture: a free name in the replacement must
not be bound by an inner binder.
[Telescope instantiation](../../reference/qvr/types-and-indexed-families.md#kinds-and-binders)
applies the same rule to [dependent parameters](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check),
where replacing one index can change the types of later arguments.

::: quivers.qiec.substitution
