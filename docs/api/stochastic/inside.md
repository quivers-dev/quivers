# Inside Algorithm

The [inside algorithm](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) combines the weights of every derivation for a span by [dynamic
programming](../../guides/deduction.md#charts-as-first-class-differentiable-values). It multiplies contributions within a derivation and adds across
alternatives, or performs the corresponding operations in log space [@goodman1999semiring]. The result is a
sentence probability only when the grammar’s weights and normalization justify that
interpretation.

The CKY inside algorithm computes inside log weights by dynamic programming
over the supported parse trees. Normalized PCFG weights give sentence log
probabilities.

::: quivers.stochastic.inside
