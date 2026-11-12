# Families

A [finite distribution family](../../guides/stochastic.md#discretized-distribution-families) assigns
probabilities to finitely many outputs, as a function of its parameters. The module gives
discretized Normal, logit-Normal, Beta, and truncated Normal families. Each evaluates its density
at evenly spaced bin centers and normalizes, which approximates the probability that integrating
the density over each bin would give.

::: quivers.stochastic.families
