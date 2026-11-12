# Predictive

`Predictive` draws posterior predictive samples: it repeatedly draws latents from a fitted guide or
a Markov chain Monte Carlo (MCMC) chain and runs the model forward from them, recording every site
([predictive sampling](../../guides/inference-svi.md#predictive-sampling)). In a
[regression](../../tutorials/python/05-variational-inference.md#building-a-model), the draws carry
both the uncertainty in the coefficients and the variation of the response [@gelman2013bayesian],
and can be compared with observations or summarized as predictive intervals.

::: quivers.inference.predictive
