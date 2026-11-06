# Estimators

Gradient estimators approximate derivatives of objectives containing expectations. A
pathwise estimator differentiates through a transformation of fixed base noise. A score-function estimator uses derivatives of log probabilities, allowing discrete sample
values. The available estimators differ in their differentiability requirements and
sampling variance.

See [pathwise and score-function estimates](../../guides/inference-svi.md#objectives-and-gradient-estimators) and
[a regression fitted with variational inference](../../tutorials/python/05-variational-inference.md#building-a-model).

For stochastic variational optimization and score-function gradients, see
[@hoffman2013svi; @ranganath2014bbvi].

Gradient-estimator strategies for [Objectives](elbo.md):
`Reparameterized` supplies the default pathwise gradient;
`StickingTheLanding` reduces variance near convergence;
`DoublyReparameterized` supplies DReG for `IWAEBound` at large K; and
`ScoreFunction` supplies REINFORCE for non-reparameterizable sites.

::: quivers.inference.estimators
