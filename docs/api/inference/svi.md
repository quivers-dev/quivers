# SVI

Stochastic variational inference (SVI) [updates a guide](../../guides/inference-svi.md#svi-stochastic-variational-inference) using sampled estimates of a
variational objective [@hoffman2013svi; @blei2017variational]. For a [coin-bias model](https://pyro.ai/examples/svi_part_i.html#A-simple-example),
the guide might be a Beta distribution: optimization adjusts its two shape
parameters to approximate the posterior over the coin’s bias [@pyroSVI].
[Posterior predictive simulations](predictive.md) then show what observations
the fitted approximation predicts, including features the model may fit poorly.

Stochastic variational inference algorithms.

::: quivers.inference.svi
