# Forward Sampling

Sampling a sentence of a fixed length requires normalizing over the candidate yields of
that length. This routine enumerates that finite set and uses each yield’s chart weight.
The distribution is conditional on length, and enumeration can grow rapidly with
vocabulary size and sentence length.

See [the weights represented by a chart](../../../semantics/grammar.md#6-chart-denotation) and
[a probabilistic context-free grammar](../../../examples/pcfg.md).

`sample_corpus` draws length-fixed yields from the chart's
length-conditional distribution
$p(s \mid \text{length} = L, \mathbf{w}) =
Z(s; \mathbf{w}) / \sum_{s' \text{ of length } L} Z(s'; \mathbf{w})$.
The implementation enumerates every length-$L$ sequence over the
deduction's surface vocabulary, evaluates $\log Z$ exactly via
the chart, softmaxes the log-weights, and draws a multinomial.
The procedure is exact (no MCMC over derivations); the $|V|^L$
enumeration is the cost of this implementation.

::: quivers.stochastic.deduction.sample
