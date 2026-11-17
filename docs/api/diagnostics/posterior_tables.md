# `quivers.diagnostics.posterior_tables`

The table functions reproduce brms's `posterior_summary`, `posterior_interval`,
`predictive_interval`, and `posterior_table` [@burkner2017brms] on retained
`(chain, draw, ...)` draws. Chains are pooled. Missing values are removed per
coordinate, as R's `na.rm = TRUE` does. Quantiles interpolate linearly between
order statistics (R's type 7), and the robust dispersion is R's `mad`, the
median absolute deviation scaled by 1.4826. Every result is a `SummaryTable`
with labeled rows and statistic columns, and a component axis for brms's
three-way arrays. The [analysis guide](../../guides/analysis-fitting-and-diagnostics.md#brms-summaries-residuals-and-predictive-criteria)
shows the fit methods built on these tables.

::: quivers.diagnostics.posterior_tables
