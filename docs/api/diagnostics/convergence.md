# `quivers.diagnostics.convergence`

These estimators [assess whether Markov chains have mixed](../../guides/analysis-fitting-and-diagnostics.md#convergence-diagnostics-for-any-draws)
and how precisely their draws estimate posterior summaries. They take any tensor laid out
`(num_chains, num_draws, *event_shape)`, so a derived quantity or the output of a hand-written
sampling loop is diagnosed with the same estimators that
[`MCMCResult`](../inference/mcmc.md) reports. Rank-normalized split
[R-hat](https://mc-stan.org/docs/reference-manual/analysis.html#potential-scale-reduction),
bulk and tail [effective sample size](https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section),
and the Monte Carlo standard errors of the mean, standard deviation, and quantiles follow
Vehtari et al. [@vehtari2021rhat]. The module needs only PyTorch, and its tests check it against ArviZ.

::: quivers.diagnostics.convergence
