# Predictive

Posterior prediction draws latent parameters from a fitted approximation and then
simulates responses conditional on those parameters. In a regression, this includes
uncertainty in the coefficients and variation in the response distribution. The returned
simulations can be compared with observations or summarized as predictive intervals.

See [posterior predictive sampling](../../guides/inference-svi.md#predictive-sampling) and
[a regression fitted with variational inference](../../tutorials/python/05-variational-inference.md#building-a-model).

For posterior predictive inference, see [@gelman2013bayesian].

Predictive inference and posterior sampling.

::: quivers.inference.predictive
