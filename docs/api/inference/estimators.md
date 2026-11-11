# Estimators

[Gradient estimators](../../guides/inference-svi.md#objectives-and-gradient-estimators) approximate derivatives of objectives containing expectations. A
pathwise estimator differentiates through a transformation of fixed base noise. A score-function estimator uses derivatives of log probabilities, allowing discrete sample
values [@hoffman2013svi; @ranganath2014bbvi]. The available estimators differ in their differentiability requirements and
sampling variance.

Gradient-estimator strategies for [Objectives](elbo.md):
`Reparameterized` supplies the default pathwise gradient;
`StickingTheLanding` reduces variance near convergence;
`DoublyReparameterized` supplies DReG for `IWAEBound` at large K; and
`ScoreFunction` supplies REINFORCE with the draws held fixed, for guides
whose draws are not differentiable. `ChiVI` accepts `Reparameterized`
and `ScoreFunction`, which estimate the gradient of the exponentiated
CUBO through `GradientEstimator.moment_surrogate` [@dieng2017chi].

::: quivers.inference.estimators
