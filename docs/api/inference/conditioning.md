# Conditioning

Conditioning binds observed values to named sites and retains their likelihood
contributions. A forward execution samples the remaining variables from their prior
conditionals and records those scores. An inference algorithm uses the scores to update
an approximation, reweight draws, or construct a chain targeting the posterior.

See [supplying observed values](../../guides/inference-foundations.md#conditioning-on-observations) and
[inspecting a model execution](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For the distinction between a model and posterior inference, see [@gelman2013bayesian].

Conditioning on observations and evidence in probabilistic programs.

::: quivers.inference.conditioning
