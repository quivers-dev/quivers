# SVI

Stochastic variational inference (SVI) updates a guide using sampled estimates of a
variational objective. A falling loss shows progress on that objective. Assess the
fitted approximation with posterior predictive simulations, repeated initializations,
and comparisons with other inference methods where feasible.

Pyro's [coin-flip example](https://pyro.ai/examples/svi_part_i.html#A-simple-example) follows a guide through a complete variational fit [@pyroSVI].

See [the SVI optimization loop](../../guides/inference-svi.md#svi-stochastic-variational-inference) and
[a regression fitted with variational inference](../../tutorials/python/05-variational-inference.md#building-a-model).

For stochastic variational inference, see [@hoffman2013svi; @blei2017variational].

Stochastic variational inference algorithms.

::: quivers.inference.svi
