# Shared posterior analysis

`ModelFit.analysis` and `BayesianFit.analysis` produce the same
`PosteriorAnalysis`: constrained posterior draws, raw observation likelihoods,
observed responses, and recorded sampler statistics. A request samples a
variational guide once; later methods reuse the retained tensors. Tensor mappings
are held by identity, so callers own their storage.

`summary` returns means, sample standard deviations, and requested quantiles on
each latent site's event coordinates. Recorded MCMC chains also supply a
`ConvergenceSummary`; variational draws supply `None`. Their one pseudo-chain
records array shape, without asserting MCMC convergence. Original integer
capacity is checked before Float64 conversion. A coordinate containing a
nonfinite draw has NaN mean, SD, and quantiles; draws are not deleted, and
valid coordinates remain independent of that failure. The same scaled
arithmetic and type-7 quantile kernel serve recorded-chain and retained
uncertainty summaries.

A source program with no latent sites can still supply observation likelihoods.
Its recorded posterior count determines the chain/draw layout; the checked
registry must establish that no unobserved variable is missing. Such an analysis
has an empty parameter summary, while raw likelihoods, WAIC, and LOO remain
available. No placeholder parameter or convergence statistic is introduced.
Covariance and contrast requests still require their selected parameter values.

## Joint covariance and correlation

`analysis.covariance(var_names=None, ddof=1)` returns a `PosteriorCovariance`
containing both empirical covariance and Pearson correlation. It pools chain
and draw axes by default; `chainwise=True` returns one matrix per retained
chain axis. The divisor is the reduced sample count minus `ddof`, which must be a
nonnegative integer smaller than that count. No convergence diagnostic is
implied by either matrix.

Selected variables retain their requested order. Each event flattens in
row-major order, while `coordinates` retains each exact `(site, index_tuple)`;
`variables`, `event_shapes`, and `sample_shape` retain the original layout.
Pass `dims` and `coords` as for `to_datatree`: the same validator checks named
axis extents and label lengths. `dimension_names` and `coordinate_labels`
identify each matrix column without parsing a rendered name. Chain and draw
labels remain available even when those axes were reduced.

These summaries use retained tensors and one joint Float64 matrix, without
resampling or likelihood evaluation. Boolean and integer prior draws are
numeric inputs; original integers outside the exact Float64 range are refused
before conversion. Coordinate labels retain their original values.
Missing and nonfinite draws are refused, without pairwise deletion. A constant
coordinate has zero covariance and NaN correlations with every coordinate,
including itself. Mathematical covariances below Float64 resolution may
underflow; correlations are computed from scaled coordinates independently
of that underflow. Unrepresentable covariance magnitudes cause an explicit
error. Tensor mappings and returned tensor storage remain caller-owned.

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
Pass `coords={"row": [...]}` and `dims={"y": ["row"]}` to label observation
axes, or assign event-axis labels to latent variables. The shared converter
uses the same sampler-statistic names and integer/Boolean dtypes for direct
MCMC results and retained analysis snapshots; conversion detaches tensors
without changing the retained draws.
PSIS-LOO exposes ArviZ 1 results, including `elpd`, `elpd_i`, and `pareto_k`.
Variational draws use relative efficiency one; this does not correct approximation
error. If every selected likelihood column is finite and constant across draws,
LOO uses exact uniform importance weights through ArviZ's
[precomputed-weight interface](https://python.arviz.org/projects/stats/en/v1.1.0/api/generated/arviz_stats.loo.html).
The pointwise score is unchanged; Pareto shapes are NaN because no tail was
fitted. This also applies when retained parameters vary but do not affect the
selected response likelihood. Varying-score analyses retain ordinary PSIS.
Unstable importance weights may require exact refitting or a different
predictive unit [@vehtari2017loo].

The [formula workflow](../../guides/analysis-data-and-formulas.md#raw-likelihoods-and-retained-analysis)
gives an executable fit and analysis. Low-level samplers can construct
`PosteriorAnalysis` directly with explicit tensor mappings and provenance.

::: quivers.diagnostics.posterior
