# Bayesian Wrap

`nuts_program_from_deduction` places an isotropic Normal prior on a deduction's learnable log
weights $\mathbf{w}$ and returns a `MonadicProgram` with log joint density
$-\tfrac{1}{2\sigma^2}\lVert \mathbf{w} \rVert^2 + \sum_n \log Z(s_n; \mathbf{w})$, where
$Z(s_n; \mathbf{w})$ is the chart weight of sentence $s_n$, ready for
[`MCMC`](../../inference/mcmc.md#quivers.inference.mcmc.MCMC) or an
[SVI guide](../../../guides/inference-svi.md#guides-variational-families). This joint is the
posterior $p(\mathbf{w} \mid S)$ for a globally normalized (log-linear) model. For a locally
normalized PCFG it differs from the posterior by the sentence normalizer, as the module docstring
states. A posterior exists only when the joint has finite, positive total mass, which depends on
the prior and the scoring model.

::: quivers.stochastic.deduction.bayes
