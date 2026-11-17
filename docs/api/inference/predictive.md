# Predictive

`Predictive` draws posterior predictive samples: it repeatedly draws latents from a fitted guide or
a Markov chain Monte Carlo (MCMC) chain and runs the model forward from them, recording every site
([predictive sampling](../../guides/inference-svi.md#predictive-sampling)). In a
[regression](../../tutorials/python/05-variational-inference.md#building-a-model), the draws carry
both the uncertainty in the coefficients and the variation of the response [@gelman2013bayesian],
and can be compared with observations or summarized as predictive intervals.

`posterior_expectations` instead reads each response's native conditional law
at the retained joint posterior draws. Recorded chains keep separate chain and
draw axes; categorical and ordinal responses return category probabilities.
Other responses use the shared analytic or explicitly budgeted finite-mean
policy. Upstream observed and lexical values remain conditioned, and observed
densities are disabled during this replay. Exact Scan occurrences and their
execution order follow the raw likelihood collector's addressing convention.
See [source conditional effects](../../guides/conditional-effects.md#expected-responses-from-qvr-source)
for the checked model adapter and row-axis contract.

::: quivers.inference.predictive
