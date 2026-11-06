# Bayesian Wrap

Bayesian lifting assigns priors to rule weights and exposes chart scores to an inference
program. The target combines the prior with those scores. A posterior exists when the
resulting measure has finite, positive total mass; this condition depends on the chosen
priors and scoring model.

See [Bayesian parameter lifting](../../../semantics/programs.md#7-bayesian-lifts) and
[choosing an approximation to the resulting posterior](../../../guides/inference-svi.md#guides-variational-families).

`nuts_program_from_deduction` lifts a deduction's learnable log weights
into a `MonadicProgram` with joint log density
$-\tfrac{1}{2\sigma^2}\lVert \mathbf{w} \rVert^2
+ \sum_n \log Z(s_n; \mathbf{w})$, ready for
[`MCMC`](../../inference/mcmc.md#quivers.inference.mcmc.MCMC).

The sampler targets this joint through a deterministic log density and
exact gradients. Whether that joint is the
Bayesian posterior $p(\mathbf{w} \mid S)$ depends on the
modeling interpretation (CRF / globally normalized vs. PCFG / locally
normalized). The module docstring states the required cancellation
condition.

::: quivers.stochastic.deduction.bayes
