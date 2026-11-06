# Point-Estimate Fitting

Fitting adjusts learnable rule weights to increase the aggregate scores of observed
sentences, optionally adding a prior penalty. A chart can exactly aggregate finite
derivations while the outer optimization remains numerical. Unless the grammar is
normalized as required, a sum of log chart weights is an objective on weights rather
than a normalized corpus log likelihood.

See [differentiating through chart scores](../../../guides/deduction.md#charts-as-first-class-differentiable-values) and
[placing a posterior over rule weights](bayes.md).

For the distinction between probabilistic normalization and weighted deduction, see [@goodman1999semiring].

`adam_fit_deduction` maximises the corpus log-marginal
$\sum_n \log Z(s_n; \mathbf{w})$ under an optional isotropic
Normal prior on the deduction's learnable log-weights. Each
$\log Z$ is computed exactly by the chart's LogProb-semiring
fixed point; autograd through the agenda's semiring operations
gives the exact gradient
$\nabla_\mathbf{w} \log Z(s; \mathbf{w})
= \mathbb{E}_{d \mid s}[\phi(d)]$ (the standard inside-outside
identity).

::: quivers.stochastic.deduction.fit
