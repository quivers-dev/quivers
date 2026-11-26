# Inside Algorithm

The [inside algorithm](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm) computes, by
[dynamic programming](../../guides/deduction.md#charts-as-first-class-differentiable-values) over
CKY spans, the semiring sum over all derivations of a span of the product of their weights
[@goodman1999semiring]. This module computes these
[inside weights](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) in log space.
The weight of the whole sentence is its log probability only when the grammar is a normalized
PCFG.

A rule of probability zero has the log weight $-\infty$ exactly, so a sentence without a
derivation scores $-\infty$. The split points of each premise pair are summed before the rules
apply, as in the [span plans](planning.md#span-plans), so no product over categories and split
points together is kept. The derivative of $\log P$ in a probability $p$ is
$(\partial P / \partial p) / P$, and the backward pass computes $\partial P / \partial p$ by the
outside pass in the log semiring, as a sum of products of outside and inside weights with no
factor $p$. It is thus exact where $p$ is zero, where differentiating $\log p$ would multiply zero
by an infinite slope, and it keeps only the chart. A sentence without a derivation has derivative
zero.

::: quivers.stochastic.inside
