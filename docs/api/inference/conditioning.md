# Conditioning

[Conditioning binds observed values to named sites](../../guides/inference-foundations.md#conditioning-on-observations) and retains their likelihood
contributions. A forward execution samples the remaining variables from their prior
conditionals and [records those scores](../../tutorials/python/05-variational-inference.md#tracing-the-model). An inference algorithm uses the scores to update
an approximation, reweight draws, or construct a chain targeting the posterior [@gelman2013bayesian].

Conditioning on observations and evidence in probabilistic programs.

::: quivers.inference.conditioning
