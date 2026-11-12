# Chart Semirings

One chart algorithm answers different questions under
[different semirings](../../guides/deduction.md#the-framework) [@goodman1999semiring]. The Boolean
semiring asks whether a derivation exists, the log semiring adds the probabilities of alternatives
in log space (the [inside weight](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores)),
the Viterbi semiring keeps the best derivation's weight, and the counting semiring counts
derivations. In log space, multiplication is addition and summation is
[log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp).

::: quivers.stochastic.semiring
    options:
      filters: ["!^_", "!^[A-Z][A-Z_]+$"]

## Semiring singletons

::: quivers.stochastic
    options:
      members:
        - LOG_PROB
        - VITERBI
        - BOOLEAN
        - COUNTING
      show_root_heading: false
      show_root_toc_entry: false
