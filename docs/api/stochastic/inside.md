# Inside Algorithm

The [inside algorithm](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm) computes, by
[dynamic programming](../../guides/deduction.md#charts-as-first-class-differentiable-values) over
CKY spans, the semiring sum over all derivations of a span of the product of their weights
[@goodman1999semiring]. This module computes these
[inside weights](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) in log space.
The weight of the whole sentence is its log probability only when the grammar is a normalized
PCFG.

::: quivers.stochastic.inside
