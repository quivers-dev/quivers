# Observation scoring

`observation_scoring(False)` omits declared observation likelihoods and
compiler-generated marginal likelihood factors during execution. Latent priors,
their support constraints, and authored score potentials retain their scores.
The policy avoids evaluating the omitted density, including an inactive response
law whose parameters would be invalid for the recorded outcome.

The policy changes scoring only. Recorded values, raw pointwise likelihood
accessors, and the response-generating distribution retain their usual meaning.
Handlers compose by scope, with the innermost policy taking precedence, and
leaving a scope restores the enclosing policy even after an exception.

::: quivers.effects.observation_scoring
