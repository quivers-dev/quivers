# Coverage

In an indexed family, matching a constructor can reveal facts about an index. Matching
an [empty vector](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), for instance, determines that its length is zero. [Coverage checking](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis)
uses such refinements to distinguish missing cases from branches that the
index makes impossible.

GADT branch refinement and coverage checking.

::: quivers.qiec.coverage
