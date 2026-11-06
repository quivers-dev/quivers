# Collapse

Collapsing integrates out a latent variable when a supported conjugate relationship
gives an analytic marginal likelihood. For a Beta–Bernoulli model, this integrates over
the uncertain success probability and scores the observations using the resulting
marginal. Removing that sampled variable can reduce the dimension of the inference
problem.

See [the analytic updates supplied by conjugate families](conjugate.md) and
[a Beta–Binomial model with an analytically marginalized rate](../../examples/beta-binomial-ab-test.md).

For analytic marginalization in Bayesian inference, see [@bernardo1994bayesian].

Integrating a conjugate parent out of the joint.

::: quivers.effects.collapse
