# SVI

[Stochastic variational inference](../../guides/inference-svi.md#svi-stochastic-variational-inference)
(SVI) fits a guide by gradient steps on a sampled estimate of a variational
[objective](elbo.md), such as the evidence lower bound (ELBO), IWAE, Rényi, or VR-IWAE bound [@hoffman2013svi; @blei2017variational; @pyroSVI]. For a [model of a coin's bias](https://pyro.ai/examples/svi_part_i.html#A-simple-example),
for instance, the guide may be a Beta distribution whose two shape parameters the optimization
adjusts. [Posterior predictive draws](predictive.md) then show which observations the fitted
approximation predicts.

::: quivers.inference.svi

A parameter-free model and guide may use `optim=None`. SVI still evaluates the
requested finite objectives and invokes callbacks, but has no gradient update to
perform. The source and formula fitting adapters select this policy when their
combined trainable parameter set is empty; frozen parameters do not create an
optimizer. Omitting the optimizer while trainable parameters exist is an error.
