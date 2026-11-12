# Checking

The [QIEC](index.md) checker validates a module and infers the types and effects of its terms
([indexed case analysis](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis)).
A [constructor match can refine an index](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check),
and later operations in the branch may use the resulting equality. Runtime providers give the
implementations an accepted computation needs to run.

::: quivers.qiec.checking
