# `quivers.data`

The data tools turn a data table into the tensors a model reads, with rows, category codes, and
observation names that match the model, and record a schema of the result
([data frames](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata)). They accept
pandas, polars, and the other backends of [Narwhals](https://narwhals-dev.github.io/narwhals/)
through the `IntoDataFrame` type. The schema fixes the order of categories and the policy for
missing values, which determine what an encoded integer means
([data inputs](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays)).

::: quivers.data
