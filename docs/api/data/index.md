# `quivers.data`

A data table must become tensors whose rows, category codes, and observation names match
the model. This package supplies that conversion and records a schema describing the
result. Inspect category ordering and missing-value policy before assuming an encoded
integer has the intended meaning.

See [the data conversion interface](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata) and
[aligning rows, covariates, and index arrays](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays).

Schema inference, observation packing, and DSL composition for tabular
data. The package accepts pandas, polars, and other
[Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible
backends via the `IntoDataFrame` shim.

::: quivers.data
