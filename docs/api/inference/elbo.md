# Objectives

A variational objective evaluates how well a guide approximates the target distribution.
The evidence lower bound (ELBO) averages the joint log density minus the guide’s log
density. Its gap from log evidence is a Kullback–Leibler divergence under the usual
support and integrability assumptions. Different objectives change the optimization
target; estimators determine how gradients are computed.

Pyro's [ELBO derivation](https://pyro.ai/examples/svi_part_i.html#ELBO) connects this objective to the discrepancy between the guide and the posterior [@pyroSVI].

See [the ELBO objective](../../guides/inference-svi.md#elbo-evidence-lower-bound) and
[a regression fitted with variational inference](../../tutorials/python/05-variational-inference.md#building-a-model).

For the ELBO derivation and variational optimization, see [@blei2017variational; @hoffman2013svi].

Variational objectives include `ELBO`, `IWAEBound`, `RenyiBound`,
`VRIWAEBound`, `ChiVI`, `RWS`, and `DReGsBound`. They pair with the
gradient-estimator strategies in [Estimators](estimators.md).

::: quivers.inference.objectives
