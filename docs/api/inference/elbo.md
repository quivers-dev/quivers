# Objectives

A [variational objective](../../guides/inference-svi.md#elbo-evidence-lower-bound) evaluates how well a guide approximates the target distribution.
The [evidence lower bound (ELBO)](https://pyro.ai/examples/svi_part_i.html#ELBO) [@pyroSVI] averages the joint log density minus the guide’s log
density. Its gap from log evidence is a Kullback–Leibler divergence under the usual
support and integrability assumptions [@blei2017variational; @hoffman2013svi]. Different objectives change the optimization
target; estimators determine how gradients are computed.

Variational objectives include `ELBO`, `IWAEBound`, `RenyiBound`,
`VRIWAEBound`, `ChiVI`, `RWS`, and `DReGsBound`. They pair with the
gradient-estimator strategies in [Estimators](estimators.md).

::: quivers.inference.objectives
