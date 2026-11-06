# Evidence

Equality evidence records an equality established during checking. Matching a
constructor can determine an index value; later checks use that equality to justify a
type conversion. The evidence carries this refinement through the computation.

See [the equalities learned from a constructor match](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis) and
[using a vector length index in a fold](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check).

Kernel-generated equality evidence.

::: quivers.qiec.evidence
