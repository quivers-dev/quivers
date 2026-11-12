# Collapse

`CollapseHandler` sums or integrates a conjugate parent out of the joint
[@bernardo1994bayesian]. The parent adds nothing to the joint, and its child is scored at the run's
observation under the closed-form marginal that the pair's [conjugacy](conjugate.md) gives. In a
[Beta and Bernoulli model](../../examples/beta-binomial-ab-test.md), for instance, the observations
are scored with the success probability integrated out, which removes that latent from the
inference problem.

::: quivers.effects.collapse
