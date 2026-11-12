# Forward Sampling

`sample_corpus` draws yields of a fixed length $L$ from the chart's distribution conditioned on
length,
$p(s \mid \text{length} = L, \mathbf{w}) = Z(s; \mathbf{w}) / \sum_{s' \text{ of length } L} Z(s'; \mathbf{w})$,
where $Z$ is the [chart weight](../../../semantics/grammar.md#6-chart-denotation) of a yield. It
enumerates every sequence of length $L$ over the deduction's vocabulary $V$, computes each
$\log Z$ exactly by the chart, normalizes with a softmax, and draws from the resulting categorical
distribution. The procedure is exact, and its cost is the $|V|^L$ enumeration.

::: quivers.stochastic.deduction.sample
