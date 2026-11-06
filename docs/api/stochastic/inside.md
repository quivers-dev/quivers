# Inside Algorithm

The inside algorithm combines the weights of every derivation for a span by dynamic
programming. It multiplies contributions within a derivation and adds across
alternatives, or performs the corresponding operations in log space. The result is a
sentence probability only when the grammar’s weights and normalization justify that
interpretation.

See [inspecting and differentiating a chart](../../guides/deduction.md#charts-as-first-class-differentiable-values) and
[comparing inside scores for a parser](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores).

For inside calculations in the semiring framework, see [@goodman1999semiring].

The CKY inside algorithm computes inside log weights by dynamic programming
over the supported parse trees. Normalized PCFG weights give sentence log
probabilities.

::: quivers.stochastic.inside
