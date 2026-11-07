# Chart Semirings

The same chart algorithm can answer different questions by [changing two operations](../../guides/deduction.md#the-framework).
Boolean operations ask whether a derivation exists; [sum-product adds the weights of
alternatives](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores); max-product selects the best weight. In log space, multiplication becomes
addition and summation becomes log-sum-exp [@goodman1999semiring]. Reusing the algorithm requires the relevant
semiring laws.

Semiring abstractions for parameterizing chart parsing
([Goodman 1999](https://aclanthology.org/J99-4004/)). Selecting a semiring
changes what the CKY skeleton computes: marginal log-probability, a Viterbi
best parse, boolean recognition, or a derivation count.

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
