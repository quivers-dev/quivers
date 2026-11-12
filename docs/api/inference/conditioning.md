# Conditioning

`condition` pairs a `MonadicProgram` with observed values for named sites and returns a
`Conditioned` model, which threads those observations through tracing and log joint evaluation
([conditioning on observations](../../guides/inference-foundations.md#conditioning-on-observations)).
A forward run draws the remaining sites from their conditional priors and
[records the scores](../../tutorials/python/05-variational-inference.md#tracing-the-model), which an
inference algorithm uses to fit an approximation, reweight draws, or build a chain targeting the
posterior [@gelman2013bayesian].

::: quivers.inference.conditioning
