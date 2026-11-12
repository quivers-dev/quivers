# Queries

These functions compute probabilities, [marginals, and expectations](../../guides/stochastic.md#queries-probability-and-expectation)
from stochastic morphisms. `prob` reads the probability of an output given an input,
`marginal_prob` sums over the codomain coordinates being removed, and `expectation` weights values
by their probabilities [@bishop2006pattern].

::: quivers.stochastic.queries
