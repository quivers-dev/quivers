# `quivers.data`

A data table must become tensors whose rows, category codes, and observation names match
the model. This package supplies that [conversion](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata) and records a schema describing the
result. Inspect [category ordering and missing-value policy](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays) before assuming an encoded
integer has the intended meaning.

Schema inference, observation packing, and DSL composition for tabular
data. The package accepts pandas, polars, and other
[Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible
backends via the `IntoDataFrame` shim.

::: quivers.data
