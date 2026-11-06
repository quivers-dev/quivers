# `quivers.data.encoding`

Column encoding converts a data-frame column into runtime values. A participant name may
become an integer index; a numeric response becomes a tensor value. Keep the category
mapping and missing-value policy with the encoded data so those values can be
interpreted when fitting and inspecting predictions.

See [encoding a data frame](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata) and
[aligning rows, covariates, and index arrays](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays).

Column-level encoding utilities and the `ColumnRole` / `MissingPolicy` enums.

::: quivers.data.encoding
