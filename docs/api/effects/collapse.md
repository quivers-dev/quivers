# Collapse

Collapsing integrates out a latent variable when a [supported conjugate relationship](conjugate.md)
gives an analytic marginal likelihood. For a [Beta–Bernoulli model](../../examples/beta-binomial-ab-test.md), this integrates over
the uncertain success probability and scores the observations using the resulting
marginal. Removing that sampled variable can reduce the dimension of the inference
problem [@bernardo1994bayesian].

Integrating a conjugate parent out of the joint.

::: quivers.effects.collapse
