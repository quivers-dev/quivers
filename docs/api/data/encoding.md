# `quivers.data.encoding`

[Column encoding](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata) converts a data-frame column into runtime values. A participant name may
become an integer index; a numeric response becomes a tensor value. Keep the [category
mapping and missing-value policy](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays) with the encoded data so those values can be
interpreted when fitting and inspecting predictions.

Column-level encoding utilities and the `ColumnRole` / `MissingPolicy` enums.

::: quivers.data.encoding
