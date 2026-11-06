# Coverage

In an indexed family, matching a constructor can reveal facts about an index. Matching
an empty vector, for instance, determines that its length is zero. Coverage checking
uses such refinements to distinguish missing cases from branches that the
index makes impossible.

See [exhaustiveness for indexed cases](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis) and
[using a vector length index in a fold](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check).

GADT branch refinement and coverage checking.

::: quivers.qiec.coverage
