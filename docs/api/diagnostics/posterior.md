# Shared posterior analysis

`ModelFit.analysis` and `BayesianFit.analysis` produce the same
`PosteriorAnalysis`: constrained posterior draws, raw observation likelihoods,
observed responses, and recorded sampler statistics. A request samples a
variational guide once; later methods reuse the retained tensors. Tensor mappings
are held by identity, so callers own their storage.

`summary` returns means, sample standard deviations, and requested quantiles on
each latent site's event coordinates. Recorded MCMC chains also supply a
`ConvergenceSummary`; variational draws supply `None`. Their one pseudo-chain
records array shape, without asserting MCMC convergence.

## Observation units and predictive scores

Log-likelihood tensors have shape `(chain, draw, *observation_shape)`, after
reducing event coordinates. They contain raw conditional densities, excluding
priors and likelihood weights. Multiple responses require an explicit `var_name`;
combining their scores requires a justified common predictive unit. An observation
hidden inside exact marginalization requires a corresponding likelihood observer.

`waic` needs only PyTorch and at least two pooled posterior draws with finite
scores. It uses unbiased sample log-score variances, with divisor `S - 1`,
matching the [loo implementation](https://github.com/stan-dev/loo/blob/master/R/waic.R).
`WAICResult` exposes `elpd_waic`, effective parameter estimate `p_waic`, and
`waic = -2 * elpd_waic`. `se` is the standard error on the ELPD scale;
`se_waic` is twice that value, and `se_p_waic` describes the effective parameter
estimate. These standard errors are undefined for one observation unit.
`elpd_i`, `variance_i`, and `waic_i` retain the observation axes.
A pointwise log-score variance above 0.4 sets `warning`, following the
[loo diagnostic](https://mc-stan.org/loo/reference/waic.html).

`to_datatree` and `loo` require the optional `diagnostics` extra. The DataTree
retains chain and draw axes and records `quivers_posterior_kind` in its attributes.
PSIS-LOO exposes ArviZ 1 results, including `elpd`, `elpd_i`, and `pareto_k`.
Variational draws use relative efficiency one; this does not correct approximation
error. Unstable importance weights may require exact refitting or a different
predictive unit [@vehtari2017loo].

The [formula workflow](../../guides/analysis-data-and-formulas.md#raw-likelihoods-and-retained-analysis)
gives an executable fit and analysis. Low-level samplers can construct
`PosteriorAnalysis` directly with explicit tensor mappings and provenance.

::: quivers.diagnostics.posterior
