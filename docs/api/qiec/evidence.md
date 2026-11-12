# Evidence

Equality evidence records an equality between indices that the checker has established.
[Matching a constructor](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis) can
fix an [index](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), and later checks
use that equality to justify a type conversion. The checker generates this evidence and carries
the refinement through the computation.

::: quivers.qiec.evidence
