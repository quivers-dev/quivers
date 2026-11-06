# Queries

Queries compute expectations, marginals, and related summaries from distributions or
transition tables. A marginal sums over the variables being removed; an expectation
weights possible values by their probabilities. Identify the fixed input and the
variables being reduced when choosing a query.

See [probability and expectation queries](../../guides/stochastic.md#queries-probability-and-expectation) and
[a Markov kernel calculation](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels).

For conditional probability and expectations, see [@bishop2006pattern].

Probabilistic queries and inference computations on stochastic morphisms.

::: quivers.stochastic.queries
