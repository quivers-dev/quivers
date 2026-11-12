# Estimators

A [gradient estimator](../../guides/inference-svi.md#objectives-and-gradient-estimators)
approximates the gradient of an objective that contains an expectation [@hoffman2013svi].
`Reparameterized`, the default, differentiates through a transformation of fixed noise
([reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick))
[@kingma2014autoencoding]. `StickingTheLanding` reduces its variance near convergence
[@roeder2017sticking]. `DoublyReparameterized` gives the DReG estimator for `IWAEBound` at large
$K$ [@tucker2019dreg]. `ScoreFunction` gives the score function (REINFORCE) estimator with the draws
held fixed, for guides whose draws are not differentiable [@ranganath2014bbvi]. `ChiVI` accepts
`Reparameterized` and `ScoreFunction`, which estimate the gradient of the exponentiated CUBO through
`GradientEstimator.moment_surrogate` [@dieng2017chi].

::: quivers.inference.estimators
