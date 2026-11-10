# Predictive

[Posterior prediction](../../guides/inference-svi.md#predictive-sampling) draws latent parameters from a fitted approximation and then
simulates responses conditional on those parameters. In a [regression](../../tutorials/python/05-variational-inference.md#building-a-model), this includes
uncertainty in the coefficients and variation in the response distribution [@gelman2013bayesian]. The returned
simulations can be compared with observations or summarized as predictive intervals.

Predictive inference and posterior sampling.

::: quivers.inference.predictive
