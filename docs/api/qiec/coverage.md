# Coverage

These functions refine indices in the branches of a case analysis over a
[GADT](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type) (generalized algebraic
data type) [@xi2003guarded] and check that the case covers every constructor. Matching an
[empty vector](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), for instance,
fixes its length at zero. [Coverage checking](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis)
uses such refinements to tell a missing case from a branch the indices make impossible.

::: quivers.qiec.coverage
